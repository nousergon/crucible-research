"""Tests for per-team override attribution (config#750, v23 migration).

scanner_evaluations gained ``override_team_id`` so the focus-list audit can
show WHICH sector team's quant agent reached outside its focus list (via
@tool get_factor_profile) — previously overrides were unioned across teams and
landed with focus_team_id=NULL, collapsing every team's overrides into one
anonymous dashboard row group.

Mirrors the in-memory ArchiveManager fixture in test_team_inputs_ledger.py.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from archive.manager import ArchiveManager
from archive.schema import SCHEMA_VERSION, ensure_schema


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    ensure_schema(c)
    yield c
    c.close()


def _make_am(conn):
    am = ArchiveManager.__new__(ArchiveManager)
    am.bucket = "test"
    am.s3 = MagicMock()
    am.local_db_path = ":memory:"
    am.db_conn = conn
    return am


def test_schema_version_at_least_23():
    assert SCHEMA_VERSION >= 23


def test_override_team_id_column_exists(conn):
    cols = {
        r[1]
        for r in conn.execute("PRAGMA table_info(scanner_evaluations)").fetchall()
    }
    assert "override_team_id" in cols


def test_writer_round_trips_override_team_id(conn):
    am = _make_am(conn)
    am.write_scanner_evaluations([
        # A focus-list member — no override attribution.
        {"ticker": "NVDA", "eval_date": "2026-07-11", "focus_team_id": "technology",
         "focus_list_passed": 1, "agent_override": 0, "override_team_id": None,
         "quant_filter_pass": 1},
        # An override reached by the technology team's agent.
        {"ticker": "TSLA", "eval_date": "2026-07-11", "focus_team_id": None,
         "focus_list_passed": 0, "agent_override": 1,
         "override_team_id": "technology", "quant_filter_pass": 0},
        # An override reached by the energy team's agent.
        {"ticker": "XOM", "eval_date": "2026-07-11", "focus_team_id": None,
         "focus_list_passed": 0, "agent_override": 1,
         "override_team_id": "energy", "quant_filter_pass": 1},
    ])
    conn.commit()
    rows = dict(conn.execute(
        "SELECT ticker, override_team_id FROM scanner_evaluations "
        "WHERE eval_date=? ORDER BY ticker",
        ("2026-07-11",),
    ).fetchall())
    assert rows["NVDA"] is None
    assert rows["TSLA"] == "technology"
    assert rows["XOM"] == "energy"


def test_writer_defaults_override_team_id_to_null_when_absent(conn):
    """A row dict without the key writes NULL (backward compat with any
    caller that hasn't been updated / pre-v23 evaluation dicts)."""
    am = _make_am(conn)
    am.write_scanner_evaluations([
        {"ticker": "AAPL", "eval_date": "2026-07-11", "agent_override": 0},
    ])
    conn.commit()
    val = conn.execute(
        "SELECT override_team_id FROM scanner_evaluations WHERE ticker=?",
        ("AAPL",),
    ).fetchone()[0]
    assert val is None


def test_per_team_override_counts_are_separable(conn):
    """The whole point of config#750: an aggregate GROUPed by override_team_id
    attributes overrides per team rather than collapsing to one NULL group."""
    am = _make_am(conn)
    am.write_scanner_evaluations([
        {"ticker": "TSLA", "eval_date": "2026-07-11", "agent_override": 1,
         "override_team_id": "technology", "quant_filter_pass": 0},
        {"ticker": "AMD", "eval_date": "2026-07-11", "agent_override": 1,
         "override_team_id": "technology", "quant_filter_pass": 1},
        {"ticker": "XOM", "eval_date": "2026-07-11", "agent_override": 1,
         "override_team_id": "energy", "quant_filter_pass": 0},
    ])
    conn.commit()
    counts = dict(conn.execute(
        "SELECT override_team_id, COUNT(*) FROM scanner_evaluations "
        "WHERE agent_override=1 GROUP BY override_team_id"
    ).fetchall())
    assert counts == {"technology": 2, "energy": 1}
