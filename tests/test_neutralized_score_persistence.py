"""Tests for LIVE #1142 neutralized-score DUAL-field persistence (config#1187).

The live score-neutralization cutover (2026-06-22, gated by
``NEUTRALIZATION_LIVE_ENABLED``) rewrites ONLY ``signals.json``'s per-ticker
score and was NEVER persisted to ``research.db`` — so every graded forward-IC
metric read the RAW composite (``cio_evaluations.final_score``) and could not
measure whether the live neutralization actually recovered forward selection
edge.

Migration #20 adds ``cio_evaluations.neutralized_final_score`` as a DUAL field
alongside the raw ``final_score``. archive_writer populates it at the exact
point the live neutralization is applied. These tests assert:
  (a) the column exists + the migration is wired into the version constant,
  (b) the dual field round-trips when a neutralized score is supplied
      (neutralization ON), keeping the raw final_score intact,
  (c) the field persists as NULL when no neutralized score is supplied
      (neutralization OFF / name absent / no exposures) — backward compatible,
      so the backtester reads raw==neutralized (identity) there.
"""

from __future__ import annotations

import sqlite3

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
    from unittest.mock import MagicMock

    am = ArchiveManager.__new__(ArchiveManager)
    am.bucket = "test"
    am.s3 = MagicMock()
    am.local_db_path = ":memory:"
    am.db_conn = conn
    return am


def _record(**overrides) -> dict:
    r = {
        "ticker": "NVDA",
        "eval_date": "2026-06-27",
        "team_id": "technology",
        "quant_score": 80,
        "qual_score": 65,
        "combined_score": 72.5,
        "macro_shift": 1.0,
        "final_score": 73.5,
        "cio_decision": "ADVANCE",
        "cio_conviction": 75,
        "cio_rank": 1,
        "rationale": "AI infra cycle",
    }
    r.update(overrides)
    return r


# ── Schema ──────────────────────────────────────────────────────────────────


def test_schema_version_includes_neutralized_migration():
    """Migration #20 adds neutralized_final_score; the version constant must
    reflect it so cold-starts on older DBs apply the ALTER."""
    assert SCHEMA_VERSION >= 20


def test_neutralized_column_exists(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cio_evaluations)")]
    assert "neutralized_final_score" in cols
    # The raw composite must remain — this is a DUAL field, not a replacement.
    assert "final_score" in cols


def test_fresh_db_runs_migration_20(conn):
    applied = {r[0] for r in conn.execute("SELECT version FROM schema_version")}
    assert 20 in applied


def test_idempotent_schema_init(conn):
    ensure_schema(conn)  # second run must not error or duplicate
    n = conn.execute(
        "SELECT COUNT(*) FROM schema_version WHERE version=20"
    ).fetchone()[0]
    assert n == 1


# ── Persistence: neutralization ON ────────────────────────────────────────────


def test_dual_field_written_when_neutralization_on(conn):
    """When the live gate rewrote this ticker's score, archive_writer passes a
    neutralized_final_score. BOTH the raw final_score AND the neutralized score
    persist, clearly distinct."""
    am = _make_am(conn)
    am.write_cio_evaluations(
        [_record(final_score=73.5, neutralized_final_score=68.1)]
    )
    row = conn.execute(
        "SELECT final_score, neutralized_final_score "
        "FROM cio_evaluations WHERE ticker='NVDA'"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(73.5)          # raw composite preserved
    assert row[1] == pytest.approx(68.1)          # live neutralized DUAL
    assert row[0] != row[1]                        # genuinely distinct fields


def test_dual_field_distinct_across_cross_section(conn):
    """Multiple names with the live neutralization on: each row carries its own
    raw + neutralized pair (the cross-section the forward-IC grades)."""
    am = _make_am(conn)
    am.write_cio_evaluations([
        _record(ticker="NVDA", final_score=73.5, neutralized_final_score=68.1),
        _record(ticker="AAPL", final_score=61.0, neutralized_final_score=64.4),
    ])
    rows = {
        t: (raw, neu)
        for t, raw, neu in conn.execute(
            "SELECT ticker, final_score, neutralized_final_score FROM cio_evaluations"
        )
    }
    assert rows["NVDA"] == (pytest.approx(73.5), pytest.approx(68.1))
    assert rows["AAPL"] == (pytest.approx(61.0), pytest.approx(64.4))


# ── Persistence: neutralization OFF / absent ─────────────────────────────────


def test_null_when_neutralization_off(conn):
    """Gate OFF (the default): archive_writer supplies no neutralized score, so
    the column persists NULL while the raw final_score is unchanged. Backward
    compatible — the backtester reads raw==neutralized (identity) here."""
    am = _make_am(conn)
    am.write_cio_evaluations([_record(final_score=73.5)])  # no neutralized key
    row = conn.execute(
        "SELECT final_score, neutralized_final_score "
        "FROM cio_evaluations WHERE ticker='NVDA'"
    ).fetchone()
    assert row[0] == pytest.approx(73.5)
    assert row[1] is None


def test_null_when_neutralized_explicitly_none(conn):
    """A name present in the cross-section but with no neutralized score (no
    factor exposures / below the min-names floor) comes through as explicit
    None — must persist as NULL, same as the gate-off case."""
    am = _make_am(conn)
    am.write_cio_evaluations([_record(neutralized_final_score=None)])
    row = conn.execute(
        "SELECT neutralized_final_score FROM cio_evaluations WHERE ticker='NVDA'"
    ).fetchone()
    assert row[0] is None
