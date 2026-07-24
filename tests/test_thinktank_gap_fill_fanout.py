"""Bounded-parallel gap_fill fan-out (config#3072 component C):
plan -> build (N independent, possibly concurrent units) -> finalize.

Covers: PLAN sizes the ticker set to the measured gap; BUILD is idempotent
(a checkpoint short-circuits a repeat call, no duplicate LLM spend);
FINALIZE merges every BUILD checkpoint into the ledger + ratings board +
challenger selection exactly like the sequential ``run_daily`` path, and
concurrent BUILD calls for DIFFERENT tickers never clobber each other's
ledger entries (the race the per-ticker checkpoint keys are designed to
avoid).
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
from thinktank.gap_fill_fanout import build_gap_fill_unit, finalize_gap_fill, plan_gap_fill
from thinktank.run import run_daily
from thinktank.schemas import CoverageLedger
from thinktank.settings import load_settings
from thinktank.storage import ThinktankStore

BUCKET = "alpha-engine-research"


# ── fake OpenAI-compatible backend (mirrors test_thinktank_run.py's) ────────


class _FakeBackend:
    """Dispatches on response_format schema name; scriptable per test."""

    def __init__(self):
        self.macro_material_change = False
        self.sweep_updates: dict[str, str] = {}
        self.ratings: dict[str, int] = {}
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        user = kwargs["messages"][-1]["content"]
        self.calls.append(name)
        if name == "ThemeThesisLLM":
            body = {
                "narrative": "themes narrative", "stance": "neutral",
                "drivers": ["d1"], "watch_items": ["w1"],
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
                return {"pillar": pillar, "score": score, "confidence": "medium", "evidence": ["e1"]}

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
        Bucket=BUCKET, Key="scanner/universe/latest.json",
        Body=json.dumps({"schema_version": 3, "stocks": stocks}),
    )
    s3.put_object(
        Bucket=BUCKET, Key="signals/latest.json",
        Body=json.dumps({
            "date": signals_date, "market_regime": "neutral",
            "sector_ratings": {"Tech": {"rating": "market_weight"}}, "signals": {},
        }),
    )
    s3.put_object(Bucket=BUCKET, Key="archive/macro/macro_report.md", Body=b"# Macro\nSteady.")


def _client(backend, run_id="run") -> ThinktankClient:
    settings = load_settings()
    return ThinktankClient(settings=settings, run_id=run_id, client_factory=lambda p, k: backend)


def test_plan_sizes_to_measured_gap_and_settles_themes(tt_config):
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        settings = load_settings()
        store = ThinktankStore(BUCKET, s3)

        # daily covers T0-T2 first (daily_new_names=3 in tt_config)
        run_daily(settings, store=store, client=_client(backend, "d"))

        plan = plan_gap_fill(settings, store=store, client=_client(backend, "plan"), run_id="gf1")
        assert plan["tickers"] == ["T3", "T4", "T5", "T6", "T7"]
        assert plan["coverage_gap"]["uncovered_count"] == 5
        # themes were seeded as a side effect (BUILD workers read, never write)
        assert store.get_json("thinktank/themes/macro/macro/latest.json") is not None


def test_build_unit_is_idempotent_on_repeat_call(tt_config):
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        settings = load_settings()
        store = ThinktankStore(BUCKET, s3)
        plan_gap_fill(settings, store=store, client=_client(backend, "plan"), run_id="gf1")

        calls_before = len(backend.calls)
        cp1 = build_gap_fill_unit(
            settings, store=store, client=_client(backend, "u-T0"),
            run_id="gf1", trading_day="2026-07-18", calendar_date="2026-07-18",
            ticker="T0",
        )
        calls_after_first = len(backend.calls)
        assert calls_after_first > calls_before

        # repeat call for the SAME ticker/trading_day — idempotent, no new LLM calls
        cp2 = build_gap_fill_unit(
            settings, store=store, client=_client(backend, "u-T0-retry"),
            run_id="gf1", trading_day="2026-07-18", calendar_date="2026-07-18",
            ticker="T0",
        )
        assert len(backend.calls) == calls_after_first
        assert cp2 == cp1


def test_finalize_merges_concurrent_build_checkpoints_without_losing_any(tt_config):
    """The race checkpoint-per-ticker keys exist to avoid: if BUILD wrote
    straight to the shared ledger, concurrent load-mutate-save calls across
    tickers would lose updates. Simulate several BUILD units for DIFFERENT
    tickers (each its own store/client, standing in for separate concurrent
    Lambda invocations) then verify FINALIZE's merge drops none of them."""
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        settings = load_settings()
        store = ThinktankStore(BUCKET, s3)
        run_daily(settings, store=store, client=_client(backend, "d"))

        plan = plan_gap_fill(settings, store=store, client=_client(backend, "plan"), run_id="gf1")
        assert plan["tickers"] == ["T3", "T4", "T5", "T6", "T7"]

        # Each ticker built through its OWN ThinktankClient/store handle —
        # standing in for independent concurrent Lambda invocations. None
        # touch the coverage ledger directly (only their own checkpoint key).
        for ticker in plan["tickers"]:
            build_gap_fill_unit(
                settings, store=ThinktankStore(BUCKET, s3), client=_client(backend, f"u-{ticker}"),
                run_id="gf1", trading_day=plan["trading_day"],
                calendar_date=plan["calendar_date"], ticker=ticker,
            )

        # Ledger untouched by BUILD — still only the daily run's T0-T2.
        ledger = CoverageLedger.model_validate(store.get_json(LEDGER_KEY))
        assert set(ledger.entries) == {"T0", "T1", "T2"}

        manifest = finalize_gap_fill(
            settings, store=store, client=_client(backend, "fin"),
            run_id="gf1", trading_day=plan["trading_day"],
            calendar_date=plan["calendar_date"],
        )
        assert manifest.mode == "gap_fill"
        assert sorted(manifest.names_added) == ["T3", "T4", "T5", "T6", "T7"]
        assert manifest.theses_written == 5
        assert manifest.coverage_gap["uncovered_count"] == 0
        assert manifest.challenger_selection_written is True

        ledger = CoverageLedger.model_validate(store.get_json(LEDGER_KEY))
        assert set(ledger.entries) == {"T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"}

        # ratings board self-healed rows for every BUILD-checkpointed ticker
        board = store.get_json("thinktank/ratings/latest.json")
        assert set(board["rows"]) == {"T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"}

        # cost ledger accrues BOTH the plan's (themes) and each BUILD unit's
        # spend, rolled up once by finalize — never lost, never double-counted.
        month = store.get_json(f"thinktank/costs/{plan['calendar_date'][:7]}.json")
        assert month["spent_usd"] > 0
        run_ids = {r["run_id"] for r in month["runs"]}
        assert "gf1" in run_ids  # finalize's own rollup entry


def test_finalize_with_zero_checkpoints_is_a_safe_no_op(tt_config):
    backend = _FakeBackend()
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        _seed_read_side(s3)
        settings = load_settings()
        store = ThinktankStore(BUCKET, s3)
        run_daily(settings, store=store, client=_client(backend, "d"))

        manifest = finalize_gap_fill(
            settings, store=store, client=_client(backend, "fin"),
            run_id="gf-empty", trading_day="2026-07-18", calendar_date="2026-07-18",
        )
        assert manifest.names_added == []
        assert manifest.theses_written == 0
