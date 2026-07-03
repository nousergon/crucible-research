"""End-to-end daily run against moto S3 with a fake OpenAI-compatible backend.

Covers the P0 closes-when behaviors: intake → theses → ledger, theme
seed/update/reconcile lifecycle with churn discipline, events sweep →
event-driven thesis updates, manifests + month cost ledger, dry-run
writes nothing, budget breach refuses the run.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws

from thinktank import LEDGER_KEY
from thinktank.client import ThinktankClient
from thinktank.costs import BudgetExceededError
from thinktank.run import run_daily
from thinktank.settings import load_settings
from thinktank.storage import ThinktankStore

BUCKET = "alpha-engine-research"


# ── fake OpenAI-compatible backend ───────────────────────────────────────────


class _FakeBackend:
    """Dispatches on response_format schema name; scriptable per test."""

    def __init__(self):
        self.macro_material_change = False
        self.sweep_updates: dict[str, str] = {}  # ticker -> rationale
        self.calls: list[str] = []
        self.users: list[tuple[str, str]] = []  # (schema_name, user_content)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        user = kwargs["messages"][-1]["content"]
        self.calls.append(name)
        self.users.append((name, user))
        if name == "ThemeThesisLLM":
            body = {
                "narrative": "themes narrative",
                "stance": "neutral",
                "drivers": ["d1"],
                "watch_items": ["w1"],
                "material_change": self.macro_material_change,
                "change_summary": "changed" if self.macro_material_change else "",
            }
        elif name == "CompanyThesisRatedLLM":
            body = {
                "business_summary": "b", "moat": "m", "filings_review": "f",
                "news_sentiment": "n", "valuation": "v", "market_dynamics": "md",
                "risks": ["r1"], "catalysts": ["c1"], "stance": "attractive",
                "conviction": 70, "summary": "s",
                "rating": 72, "rating_rationale": "evidence-driven number",
            }
        elif name == "SweepBatchLLM":
            m = re.search(r"batch: (.+)", user)
            tickers = [t.strip() for t in m.group(1).splitlines()[0].split(",")]
            body = {
                "assessments": [
                    {
                        "ticker": t,
                        "action": "update_thesis" if t in self.sweep_updates else "none",
                        "severity": 80 if t in self.sweep_updates else 5,
                        "rationale": self.sweep_updates.get(t, "quiet"),
                    }
                    for t in tickers
                ],
                "macro_relevant": "surprise jobs print" if self.sweep_updates else "",
            }
        else:  # pragma: no cover
            raise AssertionError(f"unexpected schema {name}")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        )


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tt_config(tmp_path, monkeypatch):
    cfg = {
        "thinktank": {
            "bucket": BUCKET,
            "coverage": {"daily_new_names": 3, "rank_ceiling": 150, "sweep_chunk_size": 25},
            "budget": {"monthly_usd_default": 25.0, "ssm_param": "/thinktank/monthly_budget_usd"},
            "llm": {
                "providers": {
                    "fake": {"base_url": "http://fake", "key_secret": "OPENROUTER_API_KEY"}
                },
                "tiers": {
                    t: {
                        "provider": "fake", "model": f"fake/{t}", "max_tokens": 1000,
                        "price_in_per_m": 1.0, "price_out_per_m": 2.0,
                        "structured_outputs": True,
                    }
                    for t in ("sweep", "themes", "thesis")
                },
            },
        }
    }
    import yaml

    path = tmp_path / "thinktank.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("THINKTANK_CONFIG_PATH", str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("THINKTANK_MONTHLY_BUDGET_USD", "25.0")
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
    return path


def _seed_read_side(s3, *, signals_date="2026-06-28"):
    s3.create_bucket(Bucket=BUCKET)
    stocks = [
        {"ticker": f"T{i}", "sector": "Tech", "attractiveness_score": 100 - i}
        for i in range(8)
    ]
    s3.put_object(
        Bucket=BUCKET,
        Key="scanner/universe/latest.json",
        Body=json.dumps({"schema_version": 3, "stocks": stocks}),
    )
    s3.put_object(
        Bucket=BUCKET,
        Key="signals/latest.json",
        Body=json.dumps(
            {
                "date": signals_date,
                "market_regime": "neutral",
                "sector_ratings": {"Tech": {"rating": "market_weight"}},
                "signals": {},
            }
        ),
    )
    s3.put_object(
        Bucket=BUCKET, Key="archive/macro/macro_report.md", Body=b"# Macro\nSteady."
    )


def _run(settings_env, backend, s3, **kw):
    settings = load_settings()
    store = ThinktankStore(BUCKET, s3)
    client = ThinktankClient(
        settings=settings,
        run_id=f"run{len(backend.calls)}",
        client_factory=lambda p, k: backend,
    )
    return run_daily(settings, store=store, client=client, **kw), store


# ── tests ────────────────────────────────────────────────────────────────────


def test_daily_lifecycle_end_to_end(tt_config):
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)

        # RUN 1 — seed: themes (macro + Tech) + 3 initial theses, no sweep yet
        manifest, store = _run(tt_config, backend, s3)
        assert manifest.names_added == ["T0", "T1", "T2"]
        assert manifest.theses_written == 3
        assert manifest.sweep_tickers == 0
        assert manifest.theme_updates_written == 2  # macro seed + Tech seed
        ledger = store.get_json(LEDGER_KEY)
        assert set(ledger["entries"]) == {"T0", "T1", "T2"}
        t0 = store.get_json("thinktank/theses/T0/latest.json")
        assert t0["version"] == 1 and t0["update_reason"] == "initial"
        assert t0["macro_theme_version"] == 1
        macro = store.get_json("thinktank/themes/macro/macro/latest.json")
        assert macro["version"] == 1 and macro["update_reason"] == "seed"
        assert macro["weekly_anchor_date"] == "2026-06-28"

        # RUN 2 — next 3 names; sweep covers T0-T2; T1 flagged → event update;
        # macro note → churn-gated update WITH material change → macro v2
        backend.sweep_updates = {"T1": "guidance cut"}
        backend.macro_material_change = True
        manifest2, store = _run(tt_config, backend, s3)
        assert manifest2.names_added == ["T3", "T4", "T5"]
        assert manifest2.sweep_tickers == 3
        assert manifest2.events_flagged == 1
        assert manifest2.event_updates_written == 1
        t1 = store.get_json("thinktank/theses/T1/latest.json")
        assert t1["version"] == 2 and t1["update_reason"] == "event"
        assert t1["event_context"] == "guidance cut"
        macro = store.get_json("thinktank/themes/macro/macro/latest.json")
        assert macro["version"] == 2 and macro["update_reason"] == "event"
        events = store.get_text(f"thinktank/events/{manifest2.trading_day}.jsonl")
        assert events and '"update_thesis"' in events
        # month cost ledger accrues across runs
        month = store.get_json(f"thinktank/costs/{manifest2.calendar_date[:7]}.json")
        assert month["spent_usd"] > 0 and len(month["runs"]) == 2

        # RUN 3 — churn discipline: sweep quiet + no material change → no new
        # macro version; a NEW weekly signals date → reconcile bumps anchor
        backend.sweep_updates = {}
        backend.macro_material_change = False
        _seed_read_side(s3, signals_date="2026-07-05")
        manifest3, store = _run(tt_config, backend, s3)
        assert manifest3.themes_reconciled is True
        macro = store.get_json("thinktank/themes/macro/macro/latest.json")
        assert macro["update_reason"] == "reconcile"
        assert macro["weekly_anchor_date"] == "2026-07-05"


def test_dry_run_writes_nothing(tt_config):
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        settings = load_settings()
        store = ThinktankStore(BUCKET, s3)
        manifest = run_daily(settings, dry_run=True, store=store)
        assert manifest.mode == "dry_run"
        assert manifest.names_added == ["T0", "T1", "T2"]
        assert backend.calls == []
        assert store.get_json(LEDGER_KEY) is None
        listing = s3.list_objects_v2(Bucket=BUCKET, Prefix="thinktank/")
        assert listing.get("KeyCount", 0) == 0


def test_budget_breach_refuses_run(tt_config, monkeypatch):
    monkeypatch.setenv("THINKTANK_MONTHLY_BUDGET_USD", "0.0")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        settings = load_settings()
        store = ThinktankStore(BUCKET, s3)
        with pytest.raises(BudgetExceededError):
            run_daily(settings, store=store)


def test_missing_universe_board_aborts_loud(tt_config):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        settings = load_settings()
        store = ThinktankStore(BUCKET, s3)
        with pytest.raises(RuntimeError, match="universe board"):
            run_daily(settings, store=store)


def test_ratings_board_and_prompt_independence(tt_config):
    """The independent-rating contract (Brian, 2026-07-02):

    1. Every run writes the ratings board (dated + latest) with one row per
       covered name carrying the analyst's OWN 0-100 rating plus the scanner
       composite AS METADATA and the divergence column.
    2. The thesis prompt NEVER contains the scanner's opinion — no
       attractiveness/focus/tech composite, no pillar sub-scores. This is
       what makes the rating independent rather than an echo.
    """
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        manifest, store = _run(tt_config, backend, s3)

        board = store.get_json("thinktank/ratings/latest.json")
        assert set(board["rows"]) == {"T0", "T1", "T2"}
        assert manifest.ratings_rows == 3
        row = board["rows"]["T0"]
        assert row["rating"] == 72
        assert row["rating_rationale"] == "evidence-driven number"
        assert row["stance"] == "attractive" and row["conviction"] == 70
        # scanner composite rides as metadata; T0 seeded with score 100
        assert row["attractiveness_score"] == 100
        assert row["rating_minus_attractiveness"] == -28.0
        assert row["thesis_version"] == 1
        # dated partition written alongside latest
        dated = store.get_json(
            f"thinktank/ratings/{manifest.trading_day}.json"
        )
        assert dated["rows"].keys() == board["rows"].keys()

        # the anchoring pin: no scanner-opinion JSON keys in any thesis prompt
        thesis_prompts = [u for n, u in backend.users if n == "CompanyThesisRatedLLM"]
        assert thesis_prompts, "no thesis calls captured"
        for user in thesis_prompts:
            for banned in ('"attractiveness_score"', '"pillars"', '"focus_score"', '"tech_score"'):
                assert banned not in user, (
                    f"scanner opinion {banned} leaked into the thesis prompt — "
                    "the independent rating must not see the house composite "
                    "(analyst._facts_board_row)"
                )

        # stored thesis carries the rating (and old-artifact compat: the
        # storage schema tolerates absence — exercised by the operator-refresh
        # test below via a hand-written pre-rating artifact)
        t0 = store.get_json("thinktank/theses/T0/latest.json")
        assert t0["thesis"]["rating"] == 72


def test_operator_refresh_mode(tt_config):
    """{"refresh_tickers": [...]} re-underwrites ONLY covered names: no
    intake, no sweep, no theme work; the board row advances; an uncovered
    ticker fails loud. Also proves pre-rating artifacts still parse (the
    add-only storage contract) by rewinding T0's stored thesis to a
    rating-less shape before the refresh."""
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        _run(tt_config, backend, s3)  # seed 3 names

        # simulate a thesis written before the rating field existed
        store = ThinktankStore(BUCKET, s3)
        t0 = store.get_json("thinktank/theses/T0/latest.json")
        t0["thesis"].pop("rating")
        t0["thesis"].pop("rating_rationale")
        s3.put_object(
            Bucket=BUCKET,
            Key="thinktank/theses/T0/latest.json",
            Body=json.dumps(t0),
        )

        manifest, store = _run(
            tt_config, backend, s3, refresh_tickers=["T0"]
        )
        assert manifest.mode == "operator_refresh"
        assert manifest.names_added == []
        assert manifest.names_refreshed == ["T0"]
        assert manifest.sweep_tickers == 0
        assert manifest.theme_updates_written == 0
        assert manifest.theses_written == 1

        t0 = store.get_json("thinktank/theses/T0/latest.json")
        assert t0["version"] == 2 and t0["update_reason"] == "operator_refresh"
        assert t0["thesis"]["rating"] == 72
        board = store.get_json("thinktank/ratings/latest.json")
        assert board["rows"]["T0"]["thesis_version"] == 2
        assert set(board["rows"]) == {"T0", "T1", "T2"}  # untouched rows kept

        with pytest.raises(ValueError, match="not in coverage ledger"):
            _run(tt_config, backend, s3, refresh_tickers=["ZZZZ"])
