"""Tests for the team_inputs ledger (v19) — the scanner→team input-assignment
audit consumed by the decision-review console's per-sector-team page.

Mirrors the in-memory ArchiveManager fixture in test_team_candidates_sub_scores.py.
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


def test_schema_version_is_19():
    assert SCHEMA_VERSION >= 19


def test_team_inputs_table_exists(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(team_inputs)").fetchall()}
    assert {"ticker", "eval_date", "team_id", "source", "sector"} <= cols


def test_round_trip(conn):
    am = _make_am(conn)
    am.write_team_inputs([
        {"ticker": "NVDA", "eval_date": "2026-06-13", "team_id": "technology",
         "source": "scanner", "sector": "Information Technology"},
        {"ticker": "AAPL", "eval_date": "2026-06-13", "team_id": "technology",
         "source": "held_population", "sector": "Information Technology"},
    ])
    conn.commit()
    rows = conn.execute(
        "SELECT ticker, source, sector FROM team_inputs "
        "WHERE eval_date=? AND team_id=? ORDER BY ticker",
        ("2026-06-13", "technology"),
    ).fetchall()
    assert [r[0] for r in rows] == ["AAPL", "NVDA"]
    assert {r[0]: r[1] for r in rows} == {"AAPL": "held_population", "NVDA": "scanner"}


def test_idempotent_on_rerun(conn):
    am = _make_am(conn)
    rec = [{"ticker": "MSFT", "eval_date": "2026-06-13", "team_id": "technology",
            "source": "scanner", "sector": "Information Technology"}]
    am.write_team_inputs(rec)
    # Re-run with a corrected source — INSERT OR REPLACE updates in place.
    rec[0]["source"] = "held_population"
    am.write_team_inputs(rec)
    conn.commit()
    rows = conn.execute(
        "SELECT source FROM team_inputs WHERE ticker='MSFT' AND eval_date='2026-06-13' "
        "AND team_id='technology'"
    ).fetchall()
    assert len(rows) == 1 and rows[0][0] == "held_population"


def test_noop_on_empty(conn):
    am = _make_am(conn)
    am.write_team_inputs([])  # must not raise
    assert conn.execute("SELECT COUNT(*) FROM team_inputs").fetchone()[0] == 0


def test_noop_without_connection():
    am = ArchiveManager.__new__(ArchiveManager)
    am.db_conn = None
    am.write_team_inputs([{"ticker": "X", "eval_date": "d", "team_id": "t"}])  # no raise
