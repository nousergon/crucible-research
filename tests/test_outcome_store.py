"""Unit tests for `evals.outcome_store` — this repo's single accessor over
the long-format `score_performance_outcomes` store (EPIC config#1483,
consumer cutover config#1530).
"""

from __future__ import annotations

import sqlite3

import pytest
from nousergon_lib.quant.horizons import DEFAULT_POLICY

from evals.outcome_store import load_primary_outcomes, store_exists


def _conn_without_table() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def _conn_with_table() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE score_performance_outcomes (
            id INTEGER PRIMARY KEY, signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL, score_date TEXT NOT NULL,
            horizon_days INTEGER NOT NULL, beat_spy INTEGER,
            stock_return REAL, spy_return REAL, log_alpha REAL,
            is_primary INTEGER NOT NULL, resolved_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE(signal_id, horizon_days)
        )"""
    )
    return conn


def _insert(conn, symbol, score_date, horizon_days, beat_spy, stock_return,
            spy_return, log_alpha, is_primary):
    conn.execute(
        "INSERT INTO score_performance_outcomes "
        "(signal_id, symbol, score_date, horizon_days, beat_spy, stock_return, "
        " spy_return, log_alpha, is_primary, resolved_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            f"{symbol}:{score_date}", symbol, score_date, horizon_days,
            beat_spy, stock_return, spy_return, log_alpha, int(is_primary),
            "2026-06-27T00:00:00+00:00",
        ),
    )
    conn.commit()


class TestStoreExists:
    def test_false_when_table_absent(self):
        assert store_exists(_conn_without_table()) is False

    def test_true_when_table_present(self):
        assert store_exists(_conn_with_table()) is True


class TestLoadPrimaryOutcomesGracefulEmpty:
    def test_table_absent_returns_empty_dict(self):
        assert load_primary_outcomes(_conn_without_table()) == {}

    def test_table_present_but_empty_returns_empty_dict(self):
        assert load_primary_outcomes(_conn_with_table()) == {}

    def test_only_diagnostic_horizon_present_returns_empty_not_an_error(self):
        """Rows exist only at the diagnostic (5d) horizon — the primary is
        legitimately not-yet-resolved for signals scored in the last 21
        days. This accessor is primary-only and single-horizon, so it
        cannot distinguish "not resolved yet" from "producer starving the
        primary" — it must NOT raise (see the module docstring: raising here
        would fire on every normal early-window call, not just genuine
        starvation)."""
        conn = _conn_with_table()
        _insert(conn, "AAPL", "2026-06-01", 5, 1, 0.02, 0.01, None, False)
        assert load_primary_outcomes(conn) == {}


class TestLoadPrimaryOutcomesPopulated:
    def test_keyed_by_symbol_score_date(self):
        conn = _conn_with_table()
        _insert(conn, "AAPL", "2026-05-01", 21, 1, 0.0432, 0.0201, 0.023, True)
        _insert(conn, "MSFT", "2026-05-01", 21, 0, -0.0311, 0.0201, -0.05, True)
        result = load_primary_outcomes(conn)
        assert set(result) == {("AAPL", "2026-05-01"), ("MSFT", "2026-05-01")}
        assert result[("AAPL", "2026-05-01")].beat_spy == 1
        assert result[("AAPL", "2026-05-01")].log_alpha == pytest.approx(0.023)
        assert result[("MSFT", "2026-05-01")].beat_spy == 0

    def test_diagnostic_horizon_rows_excluded(self):
        conn = _conn_with_table()
        _insert(conn, "AAPL", "2026-05-08", 5, 1, 0.02, 0.01, None, False)
        _insert(conn, "AAPL", "2026-05-08", 21, 1, 0.0432, 0.0201, 0.023, True)
        result = load_primary_outcomes(conn)
        assert len(result) == 1
        assert result[("AAPL", "2026-05-08")].stock_return == pytest.approx(0.0432)

    def test_returns_are_decimal_not_percent(self):
        """The canonical unit is decimal (0.0432 for +4.32%), NOT the legacy
        wide-column 2dp-percent convention (4.32). A caller that mistakenly
        multiplies by 100 again would produce a nonsensical 432%."""
        conn = _conn_with_table()
        _insert(conn, "AAPL", "2026-05-01", 21, 1, 0.0432, 0.0201, 0.023, True)
        outcome = load_primary_outcomes(conn)[("AAPL", "2026-05-01")]
        assert -1.0 < outcome.stock_return < 1.0
        assert -1.0 < outcome.spy_return < 1.0

    def test_window_filters_by_score_date(self):
        conn = _conn_with_table()
        _insert(conn, "AAPL", "2026-04-01", 21, 1, 0.01, 0.005, 0.005, True)
        _insert(conn, "MSFT", "2026-05-15", 21, 0, -0.02, 0.005, -0.025, True)
        result = load_primary_outcomes(conn, "2026-05-01", "2026-05-31")
        assert set(result) == {("MSFT", "2026-05-15")}

    def test_explicit_policy_honored(self):
        conn = _conn_with_table()
        _insert(conn, "AAPL", "2026-05-01", 21, 1, 0.0432, 0.0201, 0.023, True)
        result = load_primary_outcomes(conn, policy=DEFAULT_POLICY)
        assert len(result) == 1
