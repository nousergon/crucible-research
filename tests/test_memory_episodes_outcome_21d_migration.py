"""Tests for the memory_episodes.outcome_10d -> outcome_21d rename
(config#1480, folded into config#1530).

The column NAME was stale — memory/episodic.py has stored the canonical 21d
realized return in it since the config#1456 canonical-alpha cutover; only the
name lagged. Migration 22 (archive/schema.py) renames it in place; this
module pins both the fresh-DB shape and the legacy-DB migration path.
"""

from __future__ import annotations

import sqlite3

import pytest

from archive.schema import SCHEMA_VERSION, ensure_schema


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_schema_version_at_least_22():
    assert SCHEMA_VERSION >= 22


def test_fresh_db_has_outcome_21d_not_outcome_10d():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    cols = _columns(conn, "memory_episodes")
    assert "outcome_21d" in cols
    assert "outcome_10d" not in cols
    conn.close()


def test_legacy_db_with_outcome_10d_is_migrated_in_place():
    """Simulate a pre-migration-22 database (created before the rename
    landed): outcome_10d exists, holds real data. ensure_schema() must
    rename it to outcome_21d WITHOUT touching the data."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE memory_episodes (
            id              INTEGER PRIMARY KEY,
            ticker          TEXT NOT NULL,
            signal_date     TEXT NOT NULL,
            score           REAL,
            rating          TEXT,
            conviction      TEXT,
            thesis_summary  TEXT,
            outcome_10d     REAL,
            outcome_vs_spy  REAL,
            lesson          TEXT,
            sector          TEXT,
            pattern_tags    TEXT,
            created_date    TEXT NOT NULL,
            UNIQUE(ticker, signal_date)
        )"""
    )
    conn.execute(
        "INSERT INTO memory_episodes "
        "(ticker, signal_date, outcome_10d, created_date) VALUES (?, ?, ?, ?)",
        ("NVDA", "2026-03-10", -0.12, "2026-03-10"),
    )
    conn.commit()

    ensure_schema(conn)

    cols = _columns(conn, "memory_episodes")
    assert "outcome_21d" in cols
    assert "outcome_10d" not in cols
    row = conn.execute(
        "SELECT outcome_21d FROM memory_episodes WHERE ticker = 'NVDA'"
    ).fetchone()
    assert row[0] == pytest.approx(-0.12)
    conn.close()


def test_ensure_schema_idempotent_after_rename():
    """A second ensure_schema() call (e.g. next Lambda cold-start) on an
    already-migrated DB must not error or re-apply the rename."""
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_schema(conn)  # must not raise
    cols = _columns(conn, "memory_episodes")
    assert "outcome_21d" in cols
    conn.close()
