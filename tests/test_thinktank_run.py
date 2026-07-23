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
from thinktank.challenger_selection import CHALLENGER_TOP_N, write_challenger_selection
from thinktank.client import ThinktankClient
from thinktank.costs import BudgetExceededError
from thinktank.run import run_daily
from thinktank.schemas import CoverageLedger, LedgerEntry, RatingRow, RatingsBoard
from thinktank.settings import load_settings
from thinktank.storage import ThinktankStore

BUCKET = "alpha-engine-research"


# ── fake OpenAI-compatible backend ───────────────────────────────────────────


class _FakeBackend:
    """Dispatches on response_format schema name; scriptable per test."""

    def __init__(self):
        self.macro_material_change = False
        self.sweep_updates: dict[str, str] = {}  # ticker -> rationale
        self.ratings: dict[str, int] = {}  # ticker -> independent rating override (default 72)
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
            m_ticker = re.search(r'"ticker":\s*"(\w+)"', user)
            ticker = m_ticker.group(1) if m_ticker else None
            body = {
                "business_summary": "b", "moat": "m", "filings_review": "f",
                "news_sentiment": "n", "valuation": "v", "market_dynamics": "md",
                "risks": ["r1"], "catalysts": ["c1"], "stance": "attractive",
                "conviction": 70, "summary": "s",
                "rating": self.ratings.get(ticker, 72),
                "rating_rationale": "evidence-driven number",
            }
        elif name == "QualitativePillarAssessment":
            m_ticker = re.search(r'"ticker":\s*"(\w+)"', user)
            ticker = m_ticker.group(1) if m_ticker else None
            score = self.ratings.get(ticker, 72)

            def _sub(pillar: str) -> dict:
                return {
                    "pillar": pillar, "score": score, "confidence": "medium",
                    "evidence": ["e1"],
                }

            body = {
                "quality": _sub("quality"),
                "quality_moat": {
                    "primary_type": "none", "secondary_types": [], "width": "none",
                    "durability_years": 0, "trend": "stable", "evidence": [],
                },
                "value": _sub("value"),
                "momentum": _sub("momentum"),
                "growth": _sub("growth"),
                "stewardship": _sub("stewardship"),
                "defensiveness": _sub("defensiveness"),
                "catalyst_horizon_modulation": 0,
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
                    for t in ("sweep", "themes", "thesis", "pillar")
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


def test_gap_fill_mode_sizes_intake_to_measured_gap_only(tt_config):
    """The Saturday SF's gap-fill mode (2026-07-14 cadence design): sized to
    the EXACT measured coverage_gap, never daily_new_names, and never
    padded with stale-refill — a re-run with a zero gap adds nothing."""
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)

        # RUN 1 — normal daily mode covers T0-T2 (daily_new_names=3 in tt_config)
        manifest, store = _run(tt_config, backend, s3)
        assert manifest.mode == "daily"
        assert manifest.names_added == ["T0", "T1", "T2"]

        # RUN 2 — gap_fill_only: the board has 8 tickers, 3 already covered,
        # so the measured gap is 5 (T3-T7) — ALL of them get covered in one
        # pass, ignoring tt_config's daily_new_names=3 cap entirely.
        manifest2, store = _run(tt_config, backend, s3, gap_fill_only=True)
        assert manifest2.mode == "gap_fill"
        assert manifest2.names_added == ["T3", "T4", "T5", "T6", "T7"]
        assert manifest2.names_refreshed == []
        assert manifest2.coverage_gap["uncovered_count"] == 5

        # RUN 3 — fully covered now; gap_fill_only must add NOTHING, and must
        # NOT fall back to stale-refill even though covered names exist.
        manifest3, store = _run(tt_config, backend, s3, gap_fill_only=True)
        assert manifest3.mode == "gap_fill"
        assert manifest3.names_added == []
        assert manifest3.names_refreshed == []
        assert manifest3.coverage_gap["uncovered_count"] == 0


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
    3. (config#2678) The pillar-extraction prompt is held to the SAME
       scanner-blind standard, and the operative ``rating`` is the
       pillar-blended value while ``raw_llm_rating`` preserves the
       pre-blend one — the fake's pillar scores equal the raw rating, so
       the blend is a no-op here and the pre-2678 assertions still hold.
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
        assert row["raw_llm_rating"] == 72
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

        # the anchoring pin: no scanner-opinion JSON keys in any thesis OR
        # pillar-extraction prompt (config#2678 must not reverse this)
        opinion_prompts = [
            u for n, u in backend.users
            if n in ("CompanyThesisRatedLLM", "QualitativePillarAssessment")
        ]
        assert opinion_prompts, "no thesis/pillar calls captured"
        for user in opinion_prompts:
            for banned in ('"attractiveness_score"', '"pillars"', '"focus_score"', '"tech_score"'):
                assert banned not in user, (
                    f"scanner opinion {banned} leaked into the analyst prompt — "
                    "the independent rating must not see the house composite "
                    "(analyst._facts_board_row)"
                )

        # stored thesis carries the rating (and old-artifact compat: the
        # storage schema tolerates absence — exercised by the operator-refresh
        # test below via a hand-written pre-rating artifact)
        t0 = store.get_json("thinktank/theses/T0/latest.json")
        assert t0["thesis"]["rating"] == 72
        assert t0["pillar_assessment"]["quality"]["score"] == 72

        # config#2678 deliverable 1: moat profile gets a live producer again
        moat = store.get_json("thinktank/moat_profile/T0.json")
        assert moat and moat[-1]["run_date"] == manifest.trading_day
        assert moat[-1]["primary_type"] == "none"


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


# ── challenger selection (epic alpha-engine-config-I2515) ───────────────────


def test_challenger_selection_written_daily_then_gap_fill(tt_config):
    """The challenger-selection artifact (leaderboard submission for the
    Think-Tank challenger arm) is written at the tail of every non-dry run,
    and the conforming ``signals_shadow/`` view only once coverage is
    complete.

    NOTE on ``coverage_gap``/``coverage_complete`` timing: ``manifest.
    coverage_gap`` keeps its pre-run convention (computed before
    ``select_intake``), but this artifact's ``uncovered_count``/
    ``coverage_complete`` are recomputed at the call site AFTER this run's
    thesis writes — the run that fills the last gap (Saturday's gap_fill)
    must self-report complete, or the leaderboard shadow view slips to the
    NEXT run.

    - RUN 1 (daily): adds T0-T2 against an empty ledger → post-run gap is
      5 of 8, coverage_complete False; selections ranked by TT's OWN
      rating (not attractiveness); shadow signals NOT written (incomplete).
    - RUN 2 (gap_fill): adds T3-T7 (the whole remaining gap) → post-run
      gap is 0, coverage_complete True on the SAME run that completed
      coverage; conforming shadow signals ARE written.
    - RUN 3 (gap_fill, no-op): still fully covered → complete stays True,
      shadow refreshed.
    """
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        # stamp an as_of on the universe board so board_date propagates
        board = json.loads(
            s3.get_object(Bucket=BUCKET, Key="scanner/universe/latest.json")["Body"].read()
        )
        board["as_of"] = "2026-07-11"
        s3.put_object(Bucket=BUCKET, Key="scanner/universe/latest.json", Body=json.dumps(board))

        # RUN 1 — daily covers T0-T2 (daily_new_names=3); distinct ratings
        # so ranking-by-rating is distinguishable from ranking-by-attractiveness
        # (T0/T1/T2 are attractiveness-ranked 1/2/3, but rating order differs).
        backend.ratings = {"T0": 60, "T1": 90, "T2": 75}
        manifest, store = _run(tt_config, backend, s3)
        assert manifest.mode == "daily"
        assert manifest.challenger_selection_written is True

        sel = store.get_json("thinktank/challenger_selection/latest.json")
        assert sel["schema_version"] == 2  # config#2678 bump — see thinktank/__init__.py
        assert sel["arm"] == "thinktank_coverage"
        assert sel["mode"] == "daily"
        assert sel["run_id"] == manifest.run_id
        assert sel["trading_day"] == manifest.trading_day
        assert sel["board_date"] == "2026-07-11"
        # post-run: 3 of 8 covered by this run's intake → 5 uncovered
        assert sel["uncovered_count"] == 5
        assert sel["coverage_complete"] is False
        # ranked by RATING desc, not by attractiveness_score/rank
        assert [row["ticker"] for row in sel["selections"]] == ["T1", "T2", "T0"]
        assert [row["rating"] for row in sel["selections"]] == [90, 75, 60]
        # attractiveness_rank rides along as metadata only, unrelated to order
        by_ticker = {row["ticker"]: row for row in sel["selections"]}
        assert by_ticker["T0"]["attractiveness_rank"] == 1
        assert by_ticker["T1"]["attractiveness_rank"] == 2
        assert by_ticker["T0"]["stance"] == "attractive"
        assert by_ticker["T0"]["conviction"] == 70
        assert by_ticker["T0"]["thesis_version"] == 1

        dated = store.get_json(f"thinktank/challenger_selection/{manifest.trading_day}.json")
        assert dated == sel
        assert store.get_json(
            f"signals_shadow/thinktank_coverage/{manifest.trading_day}/signals.json"
        ) is None

        # RUN 2 — gap_fill_only shores up T3-T7 (the whole remaining gap)
        backend.ratings.update({"T3": 95, "T4": 50, "T5": 85, "T6": 40, "T7": 70})
        manifest2, store = _run(tt_config, backend, s3, gap_fill_only=True)
        assert manifest2.mode == "gap_fill"
        assert manifest2.names_added == ["T3", "T4", "T5", "T6", "T7"]
        assert manifest2.challenger_selection_written is True

        sel2 = store.get_json("thinktank/challenger_selection/latest.json")
        assert sel2["mode"] == "gap_fill"
        # post-run: this run filled the whole remaining gap → complete on
        # the SAME run that completed coverage (the Saturday-gap_fill case)
        assert sel2["uncovered_count"] == 0
        assert sel2["coverage_complete"] is True
        assert [row["ticker"] for row in sel2["selections"]] == [
            "T3", "T1", "T5", "T2", "T7", "T0", "T4", "T6",
        ]
        assert [row["rating"] for row in sel2["selections"]] == [
            95, 90, 85, 75, 70, 60, 50, 40,
        ]
        # conforming shadow view written on the completing run itself
        assert store.get_json(
            f"signals_shadow/thinktank_coverage/{manifest2.trading_day}/signals.json"
        ) is not None

        # RUN 3 — gap_fill_only no-op: still fully covered → complete stays
        # True, shadow refreshed.
        manifest3, store = _run(tt_config, backend, s3, gap_fill_only=True)
        assert manifest3.mode == "gap_fill"
        assert manifest3.names_added == []

        sel3 = store.get_json("thinktank/challenger_selection/latest.json")
        assert sel3["uncovered_count"] == 0
        assert sel3["coverage_complete"] is True
        assert [row["ticker"] for row in sel3["selections"]] == [
            "T3", "T1", "T5", "T2", "T7", "T0", "T4", "T6",
        ]

        shadow = store.get_json(
            f"signals_shadow/thinktank_coverage/{manifest3.trading_day}/signals.json"
        )
        assert shadow is not None
        assert shadow["date"] == manifest3.trading_day
        assert shadow["run_date"] == manifest3.calendar_date
        signals = shadow["signals"]
        assert set(signals) == {"T3", "T1", "T5", "T2", "T7", "T0", "T4", "T6"}
        for row in signals.values():
            assert row["signal"] == "ENTER"
            assert isinstance(row["score"], float)
        # scores strictly descending in the same rank order as selections
        ordered_scores = [signals[t]["score"] for t in
                           ["T3", "T1", "T5", "T2", "T7", "T0", "T4", "T6"]]
        assert ordered_scores == sorted(ordered_scores, reverse=True)
        assert ordered_scores == [95.0, 90.0, 85.0, 75.0, 70.0, 60.0, 50.0, 40.0]


def test_challenger_selection_truncates_to_top_n_by_rating():
    """Unit-level check of the producer's ranking/truncation logic directly
    (no LLM/moto round-trip needed): with more covered+rated names than
    CHALLENGER_TOP_N, only the top N by rating are selected, in descending
    rating order, and a None-rating row is excluded from the ranking pool."""
    n_extra = CHALLENGER_TOP_N + 5
    ledger = CoverageLedger(
        entries={
            f"X{i}": LedgerEntry(
                ticker=f"X{i}", covered_since="2026-07-01",
                thesis_version=1, thesis_updated_on="2026-07-01",
            )
            for i in range(n_extra)
        }
        | {
            "UNRATED": LedgerEntry(
                ticker="UNRATED", covered_since="2026-07-01",
                thesis_version=1, thesis_updated_on="2026-07-01",
            )
        }
    )
    board = RatingsBoard(
        trading_day="2026-07-14",
        rows={
            f"X{i}": RatingRow(
                ticker=f"X{i}", rating=i, stance="attractive",
                conviction=50, thesis_version=1, attractiveness_rank=n_extra - i,
            )
            for i in range(n_extra)
        }
        | {
            "UNRATED": RatingRow(
                ticker="UNRATED", rating=None, stance="neutral",
                conviction=None, thesis_version=1, attractiveness_rank=1,
            )
        },
    )

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        store = ThinktankStore(BUCKET, s3)
        selection = write_challenger_selection(
            store, ledger, board,
            run_id="run1", mode="daily",
            trading_day="2026-07-14", calendar_date="2026-07-14",
            board_date="2026-07-11",
            coverage_gap={"uncovered_count": 3},
        )

        assert len(selection.selections) == CHALLENGER_TOP_N
        ratings = [row.rating for row in selection.selections]
        assert ratings == sorted(ratings, reverse=True)
        # the top N by rating are the highest-numbered X{i} (rating == i)
        expected_tickers = [f"X{i}" for i in range(n_extra - 1, n_extra - 1 - CHALLENGER_TOP_N, -1)]
        assert [row.ticker for row in selection.selections] == expected_tickers
        assert "UNRATED" not in {row.ticker for row in selection.selections}

        # incomplete coverage (uncovered_count=3) → conforming shadow NOT written
        assert selection.coverage_complete is False
        assert store.get_json(
            "signals_shadow/thinktank_coverage/2026-07-14/signals.json"
        ) is None


def test_challenger_selection_shadow_signals_conforming_shape():
    """Direct check of the conforming ``signals_shadow/`` shape against what
    ``scoring/leaderboard_producers.py::_enter_ranked_and_scores`` actually
    reduces on: a top-level ``signals`` dict keyed by ticker, each entry
    carrying ``signal == "ENTER"`` and a numeric ``score`` — written only
    when ``coverage_complete``."""
    ledger = CoverageLedger(
        entries={
            t: LedgerEntry(
                ticker=t, covered_since="2026-07-01",
                thesis_version=1, thesis_updated_on="2026-07-01",
            )
            for t in ("A", "B", "C")
        }
    )
    board = RatingsBoard(
        trading_day="2026-07-14",
        rows={
            "A": RatingRow(ticker="A", rating=80, stance="attractive", conviction=60, thesis_version=1, attractiveness_rank=2),
            "B": RatingRow(ticker="B", rating=95, stance="attractive", conviction=90, thesis_version=2, attractiveness_rank=1),
            "C": RatingRow(ticker="C", rating=40, stance="avoid", conviction=30, thesis_version=1, attractiveness_rank=3),
        },
    )

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        store = ThinktankStore(BUCKET, s3)
        selection = write_challenger_selection(
            store, ledger, board,
            run_id="run1", mode="gap_fill",
            trading_day="2026-07-14", calendar_date="2026-07-14",
            board_date="2026-07-11",
            coverage_gap={"uncovered_count": 0},  # coverage complete
        )
        assert selection.coverage_complete is True

        shadow = store.get_json("signals_shadow/thinktank_coverage/2026-07-14/signals.json")
        assert shadow is not None
        assert shadow["date"] == "2026-07-14"
        assert shadow["run_date"] == "2026-07-14"
        signals = shadow["signals"]
        assert set(signals) == {"A", "B", "C"}
        for row in signals.values():
            assert row["signal"] == "ENTER"
            assert isinstance(row["score"], float)
        # B (95) > A (80) > C (40) — the reducer re-derives rank from score,
        # but this view is already in that order too.
        assert [signals[t]["score"] for t in ("B", "A", "C")] == [95.0, 80.0, 40.0]


def test_challenger_selection_raises_on_empty_board_with_nonempty_ledger():
    """Fleet rule: a missing/empty ratings board while the coverage ledger
    is non-empty is a producer desync — RAISE, never silently skip/empty."""
    ledger = CoverageLedger(
        entries={
            "A": LedgerEntry(
                ticker="A", covered_since="2026-07-01",
                thesis_version=1, thesis_updated_on="2026-07-01",
            )
        }
    )
    board = RatingsBoard()  # no rows — out of sync with the ledger

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        store = ThinktankStore(BUCKET, s3)
        with pytest.raises(RuntimeError, match="out of sync"):
            write_challenger_selection(
                store, ledger, board,
                run_id="run1", mode="daily",
                trading_day="2026-07-14", calendar_date="2026-07-14",
                board_date=None,
                coverage_gap={"uncovered_count": 1},
            )
