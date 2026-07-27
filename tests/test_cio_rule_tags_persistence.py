"""Tests for CIO rule-tag attribution persistence (DB column + write path).

Pairs with alpha-engine-lib v0.7.0 (rule_tags field on CIORawDecision)
and ic_cio prompt v1.3.0 (closed-vocab tag instructions). The schema
migration must be backward-compatible — legacy rows from prompts <
v1.3.0 persist with NULL rule_tags.
"""

from __future__ import annotations

import json
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
    """Build an ArchiveManager wired to the in-memory DB with a stub
    S3 client. ``ArchiveManager.__init__`` constructs a real boto3
    client; we replace it post-init since these tests don't exercise
    S3 paths."""
    from unittest.mock import MagicMock
    am = ArchiveManager.__new__(ArchiveManager)
    am.bucket = "test"
    am.s3 = MagicMock()
    am.local_db_path = ":memory:"
    am.db_conn = conn
    return am


def _write_one(am: ArchiveManager, **overrides) -> None:
    record = {
        "ticker": "NVDA",
        "eval_date": "2026-05-16",
        "team_id": "technology",
        "quant_score": 80,
        "qual_score": 65,
        "combined_score": 72.5,
        "macro_shift": 1.0,
        "final_score": 73.5,
        "cio_decision": "ADVANCE",
        "cio_conviction": 75,
        "cio_rank": 1,
        "rationale": "Strong R/R on AI infra cycle",
    }
    record.update(overrides)
    am.write_cio_evaluations([record])


# ── Schema ──────────────────────────────────────────────────────────────────


def test_schema_version_includes_rule_tags_migration():
    """Migration #14 adds rule_tags. The version constant must reflect
    it so cold-starts on older DBs apply the ALTER."""
    assert SCHEMA_VERSION >= 14


def test_rule_tags_column_exists(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cio_evaluations)")]
    assert "rule_tags" in cols


def test_fresh_db_runs_migration_14(conn):
    applied = {
        r[0] for r in conn.execute("SELECT version FROM schema_version")
    }
    assert 14 in applied


def test_idempotent_schema_init(conn):
    """ensure_schema must be safe on every cold-start. Running it twice
    must not error or duplicate the migration record."""
    ensure_schema(conn)  # second run
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM schema_version WHERE version=14"
    ).fetchone()[0]
    assert n_rows == 1


# ── Persistence ─────────────────────────────────────────────────────────────


def test_writes_rule_tags_as_json_string(conn):
    am = _make_am(conn)
    _write_one(am, rule_tags=["rr_asymmetry", "catalyst_specificity"])

    row = conn.execute(
        "SELECT rule_tags FROM cio_evaluations WHERE ticker='NVDA'"
    ).fetchone()
    assert row is not None
    assert json.loads(row[0]) == ["rr_asymmetry", "catalyst_specificity"]


def test_writes_null_when_rule_tags_missing(conn):
    """Legacy decisions (prompt < v1.3.0) come through with no rule_tags
    key. Must persist as NULL so analytics can distinguish 'untagged
    legacy' from 'tagged but empty.'"""
    am = _make_am(conn)
    _write_one(am)  # no rule_tags override

    row = conn.execute(
        "SELECT rule_tags FROM cio_evaluations WHERE ticker='NVDA'"
    ).fetchone()
    assert row[0] is None


def test_writes_null_when_rule_tags_explicitly_none(conn):
    """An explicit None (e.g., post-processed ADVANCE_FORCED records,
    floor-fill, or LLM fallback paths) must persist as NULL — same
    as the missing-key case."""
    am = _make_am(conn)
    _write_one(am, rule_tags=None)

    row = conn.execute(
        "SELECT rule_tags FROM cio_evaluations WHERE ticker='NVDA'"
    ).fetchone()
    assert row[0] is None


def test_round_trips_multi_tag_reject(conn):
    """The common case: REJECTs cite multiple gates simultaneously
    (qual<50 AND sector underweight). Vocabulary-validated upstream
    by the lib schema; here we just test the round trip."""
    am = _make_am(conn)
    _write_one(
        am,
        ticker="MCD",
        cio_decision="REJECT",
        rationale="Qual<50; Consumer Discretionary underweight",
        rule_tags=["qual_veto", "macro_alignment"],
    )

    row = conn.execute(
        "SELECT rule_tags FROM cio_evaluations WHERE ticker='MCD'"
    ).fetchone()
    assert json.loads(row[0]) == ["qual_veto", "macro_alignment"]


def test_upsert_replaces_rule_tags(conn):
    """INSERT OR REPLACE on the (ticker, eval_date) UNIQUE constraint
    must overwrite rule_tags too — re-running CIO for the same date
    can't leave stale tags."""
    am = _make_am(conn)
    _write_one(am, rule_tags=["catalyst_specificity"])
    _write_one(am, rule_tags=["rr_asymmetry", "prior_continuity"])

    row = conn.execute(
        "SELECT rule_tags FROM cio_evaluations WHERE ticker='NVDA'"
    ).fetchone()
    assert json.loads(row[0]) == ["rr_asymmetry", "prior_continuity"]
