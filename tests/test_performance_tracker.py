"""
Tests for scoring/performance_tracker.py.
Uses in-memory SQLite — no network, no S3.

The legacy 10d/30d evaluation loop (run_performance_checks,
get_trading_day_offset, _get_spy_price_on_date, _compute_accuracy_stats)
was retired here (config#1456 canonical-alpha cutover, alpha-engine-config#1479):
it had no non-test consumer, and the canonical 21d horizon it would
otherwise migrate to is produced out-of-repo (alpha-engine-data
signal_returns._backfill_score_returns), not by this module. Only
record_new_buy_scores (the BUY-signal recorder) remains.
"""

import sqlite3

import pytest

_pt = pytest.importorskip("scoring.performance_tracker", reason="scoring.performance_tracker is gitignored")
record_new_buy_scores = _pt.record_new_buy_scores


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE score_performance (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            score_date      TEXT NOT NULL,
            score           REAL NOT NULL,
            price_on_date   REAL,
            -- Canonical 21d columns (schema v18) — populated by the
            -- separate alpha-engine-data producer, not this module.
            price_21d       REAL,
            spy_21d_return  REAL,
            return_21d      REAL,
            beat_spy_21d    INTEGER,
            eval_date_21d   TEXT,
            log_alpha_21d   REAL,
            -- Calibrator-v1 context columns (schema v12)
            quant_score     REAL,
            qual_score      REAL,
            conviction      TEXT,
            sector_modifier REAL,
            market_regime   TEXT,
            UNIQUE(symbol, score_date)
        );
    """)
    return conn


@pytest.fixture
def db():
    conn = _make_db()
    yield conn
    conn.close()


# ── record_new_buy_scores ─────────────────────────────────────────────────────

class TestRecordNewBuyScores:
    def test_inserts_buy_threshold_ticker(self, db):
        theses = {"PLTR": {"final_score": 75.0}}
        prices = {"PLTR": 88.50}
        record_new_buy_scores(db, "2026-03-05", theses, prices)
        row = db.execute(
            "SELECT symbol, score, price_on_date FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row is not None
        assert row[1] == 75.0
        assert row[2] == 88.50

    def test_skips_below_threshold(self, db):
        theses = {"AAPL": {"final_score": 50.0}}
        prices = {"AAPL": 175.0}
        record_new_buy_scores(db, "2026-03-05", theses, prices)
        row = db.execute("SELECT * FROM score_performance WHERE symbol='AAPL'").fetchone()
        assert row is None

    def test_skips_missing_price(self, db):
        theses = {"NVDA": {"final_score": 80.0}}
        record_new_buy_scores(db, "2026-03-05", theses, {})
        row = db.execute("SELECT * FROM score_performance WHERE symbol='NVDA'").fetchone()
        assert row is None

    def test_idempotent_on_duplicate(self, db):
        theses = {"PLTR": {"final_score": 75.0}}
        prices = {"PLTR": 88.50}
        record_new_buy_scores(db, "2026-03-05", theses, prices)
        record_new_buy_scores(db, "2026-03-05", theses, prices)
        count = db.execute("SELECT COUNT(*) FROM score_performance").fetchone()[0]
        assert count == 1

    def test_multiple_tickers(self, db):
        theses = {
            "PLTR": {"final_score": 75.0},
            "RKLB": {"final_score": 80.0},
            "HOLD_ME": {"final_score": 45.0},
        }
        prices = {"PLTR": 88.0, "RKLB": 22.0, "HOLD_ME": 10.0}
        record_new_buy_scores(db, "2026-03-05", theses, prices)
        count = db.execute("SELECT COUNT(*) FROM score_performance").fetchone()[0]
        assert count == 2

    def test_writes_calibrator_v1_context_when_thesis_carries_it(self, db):
        """Regression for ROADMAP P0 line ~103: per-row context columns
        (quant_score, qual_score, conviction, sector_modifier, market_regime)
        are populated when the producer thesis dict + market_regime arg
        carry them. Missing fields write NULL — backward-compat with
        legacy callers."""
        theses = {
            "PLTR": {
                "final_score": 75.0,
                "quant_score": 70.5,
                "qual_score": 80.0,
                "conviction": "rising",
                "macro_modifier": 1.15,
            },
        }
        prices = {"PLTR": 88.50}
        record_new_buy_scores(
            db, "2026-03-05", theses, prices, market_regime="bull",
        )
        row = db.execute(
            "SELECT quant_score, qual_score, conviction, sector_modifier, market_regime "
            "FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row == (70.5, 80.0, "rising", 1.15, "bull")

    def test_legacy_thesis_without_context_writes_nulls(self, db):
        """A thesis dict missing the new context fields (older test-shape
        + any legacy in-flight caller that hasn't been updated yet) must
        still INSERT cleanly with NULLs in the new columns."""
        theses = {"PLTR": {"final_score": 75.0}}
        prices = {"PLTR": 88.50}
        record_new_buy_scores(db, "2026-03-05", theses, prices)
        row = db.execute(
            "SELECT quant_score, qual_score, conviction, sector_modifier, market_regime "
            "FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row == (None, None, None, None, None)

    def test_market_regime_arg_optional(self, db):
        """market_regime is sourced from the caller (lambda handler has
        state['market_regime'] available; tests can opt out). Omitted
        arg writes NULL, matching the calibrator's tolerance for older
        rows."""
        theses = {"PLTR": {"final_score": 75.0, "quant_score": 70.0}}
        prices = {"PLTR": 88.50}
        # No market_regime arg passed.
        record_new_buy_scores(db, "2026-03-05", theses, prices)
        row = db.execute(
            "SELECT quant_score, market_regime FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row == (70.0, None)

    def test_missing_final_score_raises(self, db):
        """final_score is a required key on the thesis dict — a missing
        key must hard-fail rather than silently skip tracking."""
        theses = {"PLTR": {}}
        prices = {"PLTR": 88.50}
        with pytest.raises(KeyError):
            record_new_buy_scores(db, "2026-03-05", theses, prices)
