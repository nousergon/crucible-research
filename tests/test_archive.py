"""Tests for archive manager (using in-memory SQLite, no S3 calls)."""

import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch

ArchiveManager = pytest.importorskip("archive.manager", reason="archive.manager requires gitignored config").ArchiveManager


@pytest.fixture
def archive_in_memory():
    """Create an ArchiveManager with an in-memory SQLite DB and mocked S3."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    manager = ArchiveManager(bucket="test-bucket", local_db_path=db_path)
    manager.s3 = MagicMock()
    manager.s3.get_object.side_effect = Exception("NoSuchKey")
    manager.s3.put_object = MagicMock()
    manager.s3.upload_file = MagicMock()
    manager.s3.download_file = MagicMock(side_effect=Exception("mock"))

    # Initialize with fresh schema
    manager.db_conn = sqlite3.connect(db_path)
    manager.db_conn.row_factory = sqlite3.Row
    manager._ensure_schema()

    yield manager

    manager.close()
    os.unlink(db_path)


class TestArchiveSchema:
    def test_schema_created(self, archive_in_memory):
        conn = archive_in_memory.db_conn
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        expected = {
            "investment_thesis", "agent_reports", "candidate_tenures",
            "active_candidates", "scanner_appearances", "technical_scores",
            "macro_snapshots", "score_performance", "news_article_hashes",
        }
        assert expected.issubset(set(tables))

    def test_score_performance_has_calibrator_v1_context_columns(self, archive_in_memory):
        """Regression for ROADMAP P0 line ~103: schema migration v12 adds
        the per-row context columns the calibrator-v1 GBM upgrade needs.
        Pin the column names here so a future migration that renames them
        breaks the test (and the producer wire-up at
        scoring/performance_tracker.py:record_new_buy_scores)."""
        conn = archive_in_memory.db_conn
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(score_performance)").fetchall()
        }
        for col in ("quant_score", "qual_score", "conviction",
                    "sector_modifier", "market_regime"):
            assert col in cols, (
                f"score_performance missing calibrator-v1 column '{col}' — "
                f"schema migration v12 must run on init"
            )

    def test_schema_version_is_recorded(self, archive_in_memory):
        """schema_version table should record the latest migration application
        on a fresh init. Catches the class of bug where MIGRATIONS gets
        a new entry but SCHEMA_VERSION constant isn't bumped (the test
        for the constant pin lives below; this covers the runtime
        application path)."""
        from archive.schema import SCHEMA_VERSION
        conn = archive_in_memory.db_conn
        applied = {
            row[0]
            for row in conn.execute(
                "SELECT version FROM schema_version"
            ).fetchall()
        }
        assert SCHEMA_VERSION in applied

    def test_predictor_outcomes_has_horizon_agnostic_columns(self, archive_in_memory):
        """Regression for predictor 21d canonical-alpha migration
        (alpha-engine-docs/private/predictor-21d-migration-260509.md PR B):
        schema migration v13 adds horizon-agnostic predictor outcome columns.
        Pin the column names here so a future migration that renames them
        breaks the test (and the producer wire-up at
        alpha-engine-data/collectors/signal_returns.py:_backfill_predictor_returns)."""
        conn = archive_in_memory.db_conn
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictor_outcomes)").fetchall()
        }
        for col in ("actual_log_alpha", "horizon_days", "correct"):
            assert col in cols, (
                f"predictor_outcomes missing horizon-agnostic column '{col}' — "
                f"schema migration v13 must run on init"
            )
        # Old columns retained for transition parity (4-week parallel-write
        # window); downstream COALESCEs over both. Pinning their presence
        # protects against an over-eager retirement that breaks the
        # backtester analytics fallback path.
        for col in ("actual_5d_return", "correct_5d"):
            assert col in cols, (
                f"predictor_outcomes lost legacy column '{col}' — "
                f"premature retirement breaks transition COALESCE in "
                f"alpha-engine-backtester analytics readers"
            )

    def test_schema_version_constant_matches_latest_migration(self):
        """SCHEMA_VERSION must equal the highest key in MIGRATIONS so a
        forgotten bump leaves migrations un-applied. Lockstep pin per
        the schema.py docstring discipline ("To add a new migration:
        add an entry + bump SCHEMA_VERSION")."""
        from archive.schema import SCHEMA_VERSION, MIGRATIONS
        assert SCHEMA_VERSION == max(MIGRATIONS.keys())

    def test_score_performance_has_stance_column(self, archive_in_memory):
        """Regression for the stance taxonomy arc: schema migration v16
        denormalizes the predictor's stance label onto the
        score_performance fact table (Kimball dimensional pattern).
        Without this, backtester per-stance attribution
        (alpha-engine-backtester#182) can only do compute-time joins
        with predictions.json archive — fragile + repeats the join
        every weekly run.

        Pin the column name + type here so a future migration that
        renames it breaks the test (and the producer wire-up in
        alpha-engine-data/collectors/signal_returns.py)."""
        conn = archive_in_memory.db_conn
        info = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(score_performance)").fetchall()
        }
        assert "stance" in info, (
            "score_performance missing 'stance' column — migration v16 "
            "must run on init"
        )
        assert info["stance"] == "TEXT", (
            f"stance column type drift: expected TEXT, got {info['stance']}"
        )


class TestInvestmentThesisWrite:
    def test_write_and_read_thesis(self, archive_in_memory):
        thesis = {
            "ticker": "NVDA",
            "date": "2026-03-04",
            "rating": "BUY",
            "final_score": 85.5,
            "technical_score": 88.0,
            "quant_score": 80.0,
            "qual_score": 82.0,
            "macro_modifier": 1.15,
            "thesis_summary": "NVDA rates BUY. AI demand is strong.",
            "prior_score": 80.0,
            "prior_rating": "BUY",
            "last_material_change_date": "2026-03-04",
            "stale_days": 0,
            "consistency_flag": 0,
        }
        archive_in_memory.write_investment_thesis(thesis, run_time="2026-03-04T06:20:00Z")

        row = archive_in_memory.db_conn.execute(
            "SELECT * FROM investment_thesis WHERE symbol = 'NVDA'"
        ).fetchone()
        assert row is not None
        assert row["rating"] == "BUY"
        assert abs(row["score"] - 85.5) < 0.01

    def test_load_prior_theses(self, archive_in_memory):
        thesis = {
            "ticker": "AAPL",
            "date": "2026-03-03",
            "rating": "HOLD",
            "final_score": 58.0,
            "technical_score": None,
            "quant_score": None,
            "qual_score": None,
            "macro_modifier": None,
            "thesis_summary": "AAPL rates HOLD.",
            "prior_score": None,
            "prior_rating": None,
            "last_material_change_date": None,
            "stale_days": 2,
            "consistency_flag": 0,
        }
        archive_in_memory.write_investment_thesis(thesis, run_time="2026-03-03T06:20:00Z")

        prior = archive_in_memory.load_prior_theses(["AAPL"])
        assert "AAPL" in prior
        assert prior["AAPL"]["rating"] == "HOLD"


class TestActiveCandidates:
    def test_save_and_load_candidates(self, archive_in_memory):
        candidates = [
            {"slot": 1, "symbol": "NVDA", "entry_date": "2026-02-15", "prior_tenures": 0, "score": 85, "consecutive_low_runs": 0},
            {"slot": 2, "symbol": "MSFT", "entry_date": "2026-02-20", "prior_tenures": 1, "score": 78, "consecutive_low_runs": 0},
            {"slot": 3, "symbol": "AMZN", "entry_date": "2026-03-01", "prior_tenures": 0, "score": 72, "consecutive_low_runs": 0},
        ]
        archive_in_memory.save_active_candidates(candidates)
        loaded = archive_in_memory.load_active_candidates()
        assert len(loaded) == 3
        symbols = {c["symbol"] for c in loaded}
        assert symbols == {"NVDA", "MSFT", "AMZN"}


class TestNewsHashes:
    def test_upsert_and_load_hashes(self, archive_in_memory):
        hashes = ["abc123", "def456"]
        archive_in_memory.upsert_news_hashes("AAPL", hashes, "2026-03-04")

        loaded = archive_in_memory.load_news_hashes("AAPL")
        assert "abc123" in loaded
        assert "def456" in loaded

    def test_mention_count_increments(self, archive_in_memory):
        archive_in_memory.upsert_news_hashes("AAPL", ["abc123"], "2026-03-03")
        archive_in_memory.upsert_news_hashes("AAPL", ["abc123"], "2026-03-04")

        row = archive_in_memory.db_conn.execute(
            "SELECT mention_count FROM news_article_hashes WHERE symbol='AAPL' AND article_hash='abc123'"
        ).fetchone()
        assert row["mention_count"] == 2


class TestTechnicalScoreWrite:
    def test_write_technical_score(self, archive_in_memory):
        data = {
            "rsi_14": 42.5,
            "macd_cross": 1.0,
            "price_vs_ma50": 3.2,
            "price_vs_ma200": -1.5,
            "momentum_20d": 4.0,
            "technical_score": 67.3,
        }
        archive_in_memory.write_technical_score("COST", "2026-03-04", data)

        row = archive_in_memory.db_conn.execute(
            "SELECT * FROM technical_scores WHERE symbol='COST'"
        ).fetchone()
        assert row is not None
        assert abs(row["rsi_14"] - 42.5) < 0.01


class TestWriteSignalsJson:
    """Regression tests for the signals/latest.json pointer write.

    The executor's signal_reader tries signals/latest.json first and falls
    back to date-scanning only if the pointer is missing. Without this
    pointer, every executor boot does multiple S3 GETs against dated
    signals files before finding one that exists. These tests pin both
    the dated write and the latest.json pointer so the behavior cannot
    silently regress.
    """

    def test_writes_both_dated_and_latest(self, archive_in_memory):
        """write_signals_json must put to both the dated key and latest.json."""
        signals = {
            "market_regime": "neutral",
            "universe": [{"ticker": "AAPL", "score": 75.0}],
            "buy_candidates": [],
        }
        archive_in_memory.write_signals_json(
            trading_date="2026-04-11",
            generated_at="00:15:00",
            signals=signals,
        )

        put_calls = archive_in_memory.s3.put_object.call_args_list
        put_keys = [call.kwargs.get("Key") for call in put_calls]

        assert "signals/2026-04-11/signals.json" in put_keys, (
            f"Dated signals.json not written. Keys: {put_keys}"
        )
        assert "signals/latest.json" in put_keys, (
            f"latest.json pointer not written. Keys: {put_keys}"
        )

    def test_latest_and_dated_have_same_content(self, archive_in_memory):
        """Both writes must contain the same JSON payload."""
        signals = {"market_regime": "bull", "universe": []}
        archive_in_memory.write_signals_json(
            trading_date="2026-04-11",
            generated_at="00:15:00",
            signals=signals,
        )

        put_calls = archive_in_memory.s3.put_object.call_args_list
        bodies_by_key = {
            call.kwargs["Key"]: call.kwargs["Body"].decode("utf-8")
            for call in put_calls
        }

        dated_body = bodies_by_key["signals/2026-04-11/signals.json"]
        latest_body = bodies_by_key["signals/latest.json"]
        assert dated_body == latest_body, (
            "Dated and latest.json bodies must match — the latter is a "
            "pointer copy, not a derived summary."
        )

    def test_payload_includes_run_metadata(self, archive_in_memory):
        """The payload must include date and run_time at the top level."""
        signals = {"market_regime": "neutral"}
        archive_in_memory.write_signals_json(
            trading_date="2026-04-11",
            generated_at="00:15:00",
            signals=signals,
        )

        put_calls = archive_in_memory.s3.put_object.call_args_list
        body_str = put_calls[-1].kwargs["Body"].decode("utf-8")
        payload = json.loads(body_str)

        assert payload["date"] == "2026-04-11"
        # `run_date` field carries the timestamp of when the Lambda fired
        # (semantically distinct from `date` which is the trading day the
        # signals are FOR). The internal SQL column is still `run_time`;
        # only the JSON output schema was renamed for consumer clarity.
        assert payload["run_date"] == "00:15:00"
        assert payload["market_regime"] == "neutral"


class TestSavePopulationMacroFields:
    """Regression for the 2026-05-11 "Market Regime: NEUTRAL" brief defect.

    Pre-fix `save_population` defaulted market_regime to "neutral" and
    omitted sector_modifiers entirely. The caller in research_graph.py
    didn't pass any macro args, so population/latest.json diverged from
    signals/latest.json on every Saturday run — population carried the
    pre-critic regime, signals carried the post-critic regime. The
    predictor's load_universe read population first and rendered the
    wrong regime on the weekday morning brief.

    These tests pin the post-fix schema + signature so future readers
    can't accidentally drop the macro fields again.
    """

    def _save_minimal(self, mgr, **macro_kwargs):
        """Helper: call save_population with a one-element population."""
        mgr.save_population(
            population=[{
                "ticker": "AAPL", "sector": "Technology",
                "long_term_score": 80.0, "long_term_rating": "BUY",
                "conviction": "rising", "price_target_upside": 0.15,
                "thesis_summary": "test", "entry_date": "2026-05-11",
                "tenure_weeks": 4,
            }],
            run_date="2026-05-11",
            **macro_kwargs,
        )
        # Find the population/latest.json put_object call
        for call in mgr.s3.put_object.call_args_list:
            if call.kwargs.get("Key") == "population/latest.json":
                return json.loads(call.kwargs["Body"].decode("utf-8"))
        raise AssertionError("population/latest.json was not written")

    def test_writes_market_regime_from_caller(self, archive_in_memory):
        """When caller passes market_regime="bull", the payload carries
        "bull" (not the default "neutral")."""
        payload = self._save_minimal(archive_in_memory, market_regime="bull")
        assert payload["market_regime"] == "bull"

    def test_writes_sector_modifiers_from_caller(self, archive_in_memory):
        """sector_modifiers is now a first-class parameter — the writer
        must persist the caller's dict, not silently omit it as the
        pre-fix signature did."""
        modifiers = {"Technology": 1.10, "Energy": 1.05, "Healthcare": 0.95}
        payload = self._save_minimal(
            archive_in_memory,
            sector_modifiers=modifiers,
        )
        assert payload["sector_modifiers"] == modifiers

    def test_writes_sector_ratings_from_caller(self, archive_in_memory):
        ratings = {"Technology": {"rating": "OVERWEIGHT", "rationale": "x"}}
        payload = self._save_minimal(
            archive_in_memory,
            sector_ratings=ratings,
        )
        assert payload["sector_ratings"] == ratings

    def test_writes_full_canonical_macro_surface(self, archive_in_memory):
        """End-to-end pin: all three macro fields land in the JSON with
        the values passed by the caller. This is the invariant that
        keeps population.json and signals.json from diverging."""
        payload = self._save_minimal(
            archive_in_memory,
            market_regime="bull",
            sector_ratings={"Tech": {"r": "OVERWEIGHT"}},
            sector_modifiers={"Tech": 1.1},
        )
        assert payload["market_regime"] == "bull"
        assert payload["sector_ratings"] == {"Tech": {"r": "OVERWEIGHT"}}
        assert payload["sector_modifiers"] == {"Tech": 1.1}

    def test_omitted_args_use_safe_defaults(self, archive_in_memory):
        """If a caller forgets to pass macro args, the payload should
        still be well-formed (no key absent). Defaults preserve the
        pre-fix behavior for any unmigrated callers."""
        payload = self._save_minimal(archive_in_memory)  # no macro kwargs
        assert payload["market_regime"] == "neutral"
        assert payload["sector_ratings"] == {}
        # sector_modifiers MUST be present even when omitted — predictor's
        # load_watchlist treats absence as "use signals overlay", which
        # rebuilds the divergence symptom in a different guise.
        assert "sector_modifiers" in payload
        assert payload["sector_modifiers"] == {}

    def test_dated_and_latest_have_identical_macro(self, archive_in_memory):
        """population/{date}.json and population/latest.json must carry
        identical macro fields — both are written from the same JSON
        blob in save_population. This is the pin against accidentally
        derivig latest from a different source than the dated copy."""
        archive_in_memory.save_population(
            population=[],
            run_date="2026-05-11",
            market_regime="bull",
            sector_ratings={"Tech": {"r": "OVERWEIGHT"}},
            sector_modifiers={"Tech": 1.1},
        )
        bodies = {
            call.kwargs["Key"]: call.kwargs["Body"].decode("utf-8")
            for call in archive_in_memory.s3.put_object.call_args_list
            if call.kwargs.get("Key", "").startswith("population/")
        }
        assert bodies["population/latest.json"] == bodies["population/2026-05-11.json"]


# ── Consolidated morning brief persistence ──────────────────────────────
#
# Regression coverage for the 2026-05-20 finding: archive_writer was
# building the consolidated_report state field, emailing it via
# email_sender, and then dropping it on the floor — save_consolidated_report
# existed but had no caller for ~2 months (last morning.md write
# 2026-03-16). The dashboard's Research Briefing Archive page was
# correctly reading what was in S3, which was nothing.


class TestConsolidatedReportPersistence:
    def test_save_consolidated_report_writes_morning_md_to_s3(
        self, archive_in_memory
    ):
        archive_in_memory.save_consolidated_report(
            "2026-05-20", "# Weekly research brief\n\nTop picks: ..."
        )
        calls = archive_in_memory.s3.put_object.call_args_list
        morning_calls = [
            c for c in calls
            if c.kwargs.get("Key", "").endswith("/morning.md")
        ]
        assert len(morning_calls) == 1
        c = morning_calls[0]
        assert c.kwargs["Key"] == "consolidated/2026-05-20/morning.md"
        body = c.kwargs["Body"]
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        assert "Weekly research brief" in body

    def test_archive_writer_wires_save_consolidated_report(self):
        # Structural regression: pin that archive_writer's source calls
        # save_consolidated_report. If the call is removed again, this
        # test fails at CI time instead of staling the archive page
        # silently for two months.
        import inspect
        rg = pytest.importorskip(
            "graph.research_graph",
            reason="graph.research_graph requires gitignored config",
        )
        src = inspect.getsource(rg.archive_writer)
        assert "save_consolidated_report" in src, (
            "archive_writer must persist consolidated_report — without "
            "this call the dashboard's Research Briefing Archive stales "
            "out (regression of 2026-03-16 silent drop, fixed 2026-05-20)"
        )
