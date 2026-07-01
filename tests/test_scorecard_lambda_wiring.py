"""Tests for the Saturday Research Lambda scorecard wire-up.

The handler body just calls `_maybe_emit_scorecard(archive, trading_date)`;
the failure-posture + flag-gating + S3 emission contract is in the
helper. Tests stub the archive + boto3.client to keep the surface
unit-level.
"""

from __future__ import annotations

import datetime
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_handler():
    """Import lambda/handler.py — the dir name collides with the `lambda`
    Python keyword, so the natural `from lambda import handler` is a
    SyntaxError. Load it via importlib instead.
    """
    import importlib.util
    handler_path = Path(__file__).parent.parent / "lambda" / "handler.py"
    spec = importlib.util.spec_from_file_location("research_handler_under_test", handler_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def handler_mod():
    return _import_handler()


@pytest.fixture
def stub_archive(tmp_path):
    """Mimic ArchiveManager — has a `.db_conn` attribute pointing at a
    minimal SQLite schema. build_scorecard reads this connection.
    """
    db_path = tmp_path / "research.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE score_performance (
            id INTEGER PRIMARY KEY, symbol TEXT, score_date TEXT,
            score REAL, beat_spy_10d INTEGER, beat_spy_30d INTEGER,
            return_10d REAL, spy_10d_return REAL, return_30d REAL,
            spy_30d_return REAL, price_on_date REAL, price_10d REAL,
            price_30d REAL, eval_date_10d TEXT, eval_date_30d TEXT,
            price_21d REAL, return_21d REAL, spy_21d_return REAL,
            beat_spy_21d INTEGER, eval_date_21d TEXT, log_alpha_21d REAL
        );
        CREATE TABLE predictor_outcomes (
            id INTEGER PRIMARY KEY, symbol TEXT, prediction_date TEXT,
            predicted_direction TEXT, prediction_confidence REAL,
            p_up REAL, p_flat REAL, p_down REAL,
            score_modifier_applied REAL DEFAULT 0.0,
            actual_5d_return REAL, correct_5d INTEGER,
            actual_log_alpha REAL, horizon_days INTEGER, correct INTEGER
        );
        CREATE TABLE population (
            id INTEGER PRIMARY KEY, symbol TEXT UNIQUE, sector TEXT,
            long_term_score REAL, long_term_rating TEXT, conviction TEXT,
            price_target_upside REAL, thesis_summary TEXT,
            entry_date TEXT, tenure_weeks INTEGER
        );
        CREATE TABLE macro_snapshots (id INTEGER PRIMARY KEY, date TEXT, regime TEXT);
        """
    )
    conn.commit()

    class _StubArchive:
        def __init__(self, c):
            self.db_conn = c

    yield _StubArchive(conn)
    conn.close()


class TestFlagGating:
    def test_flag_off_skips_emission(self, handler_mod, stub_archive, monkeypatch):
        # No env var set -> _scorecard_enabled() returns False
        monkeypatch.delenv("RESEARCH_SCORECARD_ENABLED", raising=False)
        with patch("evals.last_week_scorecard.emit_scorecard_to_s3") as mock_emit:
            handler_mod._maybe_emit_scorecard(stub_archive, datetime.date(2026, 5, 23))
            mock_emit.assert_not_called()

    def test_flag_explicit_false_skips_emission(self, handler_mod, stub_archive, monkeypatch):
        monkeypatch.setenv("RESEARCH_SCORECARD_ENABLED", "false")
        with patch("evals.last_week_scorecard.emit_scorecard_to_s3") as mock_emit:
            handler_mod._maybe_emit_scorecard(stub_archive, datetime.date(2026, 5, 23))
            mock_emit.assert_not_called()

    @pytest.mark.parametrize("flag", ["1", "true", "True", "TRUE", "yes", "YES"])
    def test_flag_on_invokes_emission(self, handler_mod, stub_archive, monkeypatch, flag):
        monkeypatch.setenv("RESEARCH_SCORECARD_ENABLED", flag)
        with patch("boto3.client") as mock_boto:
            mock_client = mock_boto.return_value
            handler_mod._maybe_emit_scorecard(stub_archive, datetime.date(2026, 5, 23))
            # Two put_object calls — dated + latest.
            assert mock_client.put_object.call_count == 2


class TestFailureIsNonFatal:
    def test_s3_failure_logs_warn_and_returns(self, handler_mod, stub_archive, monkeypatch, caplog):
        monkeypatch.setenv("RESEARCH_SCORECARD_ENABLED", "1")
        with patch("boto3.client") as mock_boto:
            mock_boto.return_value.put_object.side_effect = RuntimeError("simulated S3 outage")
            # Helper must NOT raise — shadow mode is non-fatal.
            handler_mod._maybe_emit_scorecard(stub_archive, datetime.date(2026, 5, 23))
        # WARN log surfaces the failure as the recording surface.
        assert any(
            "Scorecard emission failed" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )

    def test_build_failure_logs_warn_and_returns(self, handler_mod, monkeypatch, caplog):
        """A broken archive (no db_conn) should not blow up the handler."""
        monkeypatch.setenv("RESEARCH_SCORECARD_ENABLED", "1")

        class _BadArchive:
            db_conn = None  # build_scorecard will explode trying to .execute()
        handler_mod._maybe_emit_scorecard(_BadArchive(), datetime.date(2026, 5, 23))
        # Captured WARN with the failure message.
        assert any(
            "Scorecard emission failed" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        )


class TestScorecardEnabledHelper:
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("True", True), ("yes", True),
        ("0", False), ("false", False), ("False", False), ("no", False),
        ("", False), ("anything-else", False),
    ])
    def test_flag_parsing(self, handler_mod, monkeypatch, val, expected):
        monkeypatch.setenv("RESEARCH_SCORECARD_ENABLED", val)
        assert handler_mod._scorecard_enabled() is expected

    def test_flag_missing_defaults_off(self, handler_mod, monkeypatch):
        monkeypatch.delenv("RESEARCH_SCORECARD_ENABLED", raising=False)
        assert handler_mod._scorecard_enabled() is False
