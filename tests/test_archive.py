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
