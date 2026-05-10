"""Tests for team_candidates per-sub-signal score persistence (PR-B of
the 2026-05-09 sector-team diagnostic arc).

The 2026-05-09 evaluator-email post-mortem found per-sector
corr(quant_rank, return_5d) at +0.33-0.36 in healthcare/industrials/
tech — anti-skill territory. The backtester's PR-A diagnostic surfaces
the rank inversion weekly; PR-B (this) persists the 5 sub-scores
(rsi/macd/ma50/ma200/momentum) so PR-C can run weight-ablation analysis
to identify per-sector calibration targets.

Locked behavior:

- Schema migration #15 adds 5 nullable REAL columns
- compute_technical_sub_scores returns the same per-signal values that
  feed compute_technical_score (regression: existing composite must be
  numerically unchanged)
- Persistence path: write_team_candidates JSON-NULL on missing inputs,
  number on present inputs
- Round-trip: full sub-score dict persists + reads back
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from archive.manager import ArchiveManager
from archive.schema import SCHEMA_VERSION, ensure_schema
from scoring.technical import (
    compute_technical_score,
    compute_technical_sub_scores,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    ensure_schema(c)
    yield c
    c.close()


def _make_am(conn):
    """Build an ArchiveManager wired to in-memory DB. Mirrors the fixture
    in test_cio_rule_tags_persistence.py."""
    am = ArchiveManager.__new__(ArchiveManager)
    am.bucket = "test"
    am.s3 = MagicMock()
    am.local_db_path = ":memory:"
    am.db_conn = conn
    return am


def _record(**overrides):
    base = {
        "ticker": "NVDA",
        "eval_date": "2026-05-16",
        "team_id": "technology",
        "quant_rank": 1,
        "quant_score": 80.0,
        "qual_score": 65.0,
        "team_recommended": 1,
    }
    base.update(overrides)
    return base


# ── Schema ──────────────────────────────────────────────────────────────────


def test_schema_version_includes_sub_score_migration():
    assert SCHEMA_VERSION >= 15


def test_sub_score_columns_exist(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(team_candidates)")}
    expected = {
        "rsi_sub_score", "macd_sub_score", "ma50_sub_score",
        "ma200_sub_score", "momentum_sub_score",
    }
    assert expected.issubset(cols)


def test_idempotent_schema_re_run(conn):
    """ensure_schema must be safe on every cold-start. Running it twice
    must not duplicate the migration record."""
    ensure_schema(conn)
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM schema_version WHERE version=15"
    ).fetchone()[0]
    assert n_rows == 1


# ── Helper: compute_technical_sub_scores ────────────────────────────────────


class TestComputeTechnicalSubScores:
    def test_returns_five_keys(self):
        indicators = {
            "rsi_14": 50.0, "macd_cross": 0.0, "macd_above_zero": True,
            "price_vs_ma50": 0.02, "price_vs_ma200": 0.05,
            "momentum_20d": 5.0,
        }
        sub = compute_technical_sub_scores(indicators)
        assert set(sub.keys()) == {"rsi", "macd", "ma50", "ma200", "momentum"}

    def test_all_in_zero_to_hundred_range(self):
        indicators = {
            "rsi_14": 30.0, "macd_cross": 1.0, "macd_above_zero": True,
            "price_vs_ma50": 0.10, "price_vs_ma200": 0.20,
            "momentum_20d": 15.0,
        }
        sub = compute_technical_sub_scores(indicators)
        for k, v in sub.items():
            assert 0.0 <= v <= 100.0, f"{k}={v} out of range"

    def test_regime_affects_rsi_only(self):
        """Bull regime raises overbought threshold for RSI; other
        sub-scores are regime-independent."""
        indicators = {
            "rsi_14": 75.0, "macd_cross": 0.0, "macd_above_zero": True,
            "price_vs_ma50": 0.0, "price_vs_ma200": 0.0,
            "momentum_20d": 0.0,
        }
        bull = compute_technical_sub_scores(indicators, market_regime="bull")
        neutral = compute_technical_sub_scores(indicators, market_regime="neutral")
        assert bull["rsi"] != neutral["rsi"]
        # Other sub-scores identical
        for k in ("macd", "ma50", "ma200", "momentum"):
            assert bull[k] == neutral[k]

    def test_consistent_with_compute_technical_score(self):
        """Regression: the new sub-score helper must produce the same
        per-signal values that the existing composite formula uses
        internally. Catches drift if either side gets refactored
        independently."""
        from config import TECHNICAL_CFG
        indicators = {
            "rsi_14": 45.0, "macd_cross": 1.0, "macd_above_zero": True,
            "price_vs_ma50": 0.03, "price_vs_ma200": 0.07,
            "momentum_20d": 8.0,
        }
        sub = compute_technical_sub_scores(indicators)
        weights = TECHNICAL_CFG.get("composite_weights", {})
        recomposed = (
            sub["rsi"] * weights["rsi"]
            + sub["macd"] * weights["macd"]
            + sub["ma50"] * weights["ma50"]
            + sub["ma200"] * weights["ma200"]
            + sub["momentum"] * weights["momentum"]
        )
        composite = compute_technical_score(indicators)
        assert abs(round(recomposed, 2) - composite) < 0.05


# ── Persistence ─────────────────────────────────────────────────────────────


def test_persists_full_sub_score_dict(conn):
    am = _make_am(conn)
    am.write_team_candidates([_record(
        rsi_sub_score=42.0, macd_sub_score=70.0, ma50_sub_score=80.0,
        ma200_sub_score=85.0, momentum_sub_score=66.7,
    )])
    row = conn.execute(
        "SELECT rsi_sub_score, macd_sub_score, ma50_sub_score, "
        "ma200_sub_score, momentum_sub_score "
        "FROM team_candidates WHERE ticker='NVDA'"
    ).fetchone()
    assert row == (42.0, 70.0, 80.0, 85.0, 66.7)


def test_writes_null_when_sub_scores_missing(conn):
    """Legacy producers (pre-v15 wire-up) come through with no
    sub-score keys. Must persist as NULL — not coerced to 0 — so the
    backtester ablation can distinguish 'not yet emitted' from
    'computed and zero.'"""
    am = _make_am(conn)
    am.write_team_candidates([_record()])  # no sub-score overrides
    row = conn.execute(
        "SELECT rsi_sub_score, macd_sub_score, ma50_sub_score, "
        "ma200_sub_score, momentum_sub_score "
        "FROM team_candidates WHERE ticker='NVDA'"
    ).fetchone()
    assert row == (None, None, None, None, None)


def test_writes_null_on_explicit_none(conn):
    am = _make_am(conn)
    am.write_team_candidates([_record(
        rsi_sub_score=None, macd_sub_score=None, ma50_sub_score=None,
        ma200_sub_score=None, momentum_sub_score=None,
    )])
    row = conn.execute(
        "SELECT rsi_sub_score, macd_sub_score, ma50_sub_score, "
        "ma200_sub_score, momentum_sub_score "
        "FROM team_candidates WHERE ticker='NVDA'"
    ).fetchone()
    assert row == (None, None, None, None, None)


def test_partial_sub_scores_round_trip(conn):
    """One sub-score missing (e.g. compute_technical_indicators couldn't
    derive momentum because of insufficient price history) — others
    must still persist correctly."""
    am = _make_am(conn)
    am.write_team_candidates([_record(
        rsi_sub_score=42.0, macd_sub_score=70.0, ma50_sub_score=80.0,
        ma200_sub_score=85.0, momentum_sub_score=None,
    )])
    row = conn.execute(
        "SELECT rsi_sub_score, momentum_sub_score "
        "FROM team_candidates WHERE ticker='NVDA'"
    ).fetchone()
    assert row[0] == 42.0
    assert row[1] is None


def test_upsert_overwrites_sub_scores(conn):
    """INSERT OR REPLACE on (ticker, eval_date, team_id) must overwrite
    all sub-score columns — re-running the producer for a date can't
    leave stale partial sub-scores."""
    am = _make_am(conn)
    am.write_team_candidates([_record(
        rsi_sub_score=10.0, macd_sub_score=20.0, ma50_sub_score=30.0,
        ma200_sub_score=40.0, momentum_sub_score=50.0,
    )])
    am.write_team_candidates([_record(
        rsi_sub_score=99.0, macd_sub_score=88.0, ma50_sub_score=77.0,
        ma200_sub_score=66.0, momentum_sub_score=55.0,
    )])
    row = conn.execute(
        "SELECT rsi_sub_score, macd_sub_score, ma50_sub_score, "
        "ma200_sub_score, momentum_sub_score "
        "FROM team_candidates WHERE ticker='NVDA'"
    ).fetchone()
    assert row == (99.0, 88.0, 77.0, 66.0, 55.0)
