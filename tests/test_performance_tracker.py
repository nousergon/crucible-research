"""
Tests for scoring/performance_tracker.py.
Uses in-memory SQLite — no network, no S3.

The yfinance fallback leg was replaced by the alpha-engine-data
daily_closes S3 reader (yfinance-centralization arc, 2026-05-16,
plan doc: alpha-engine-docs/private/yfinance-centralization-260516.md,
item R5 / PR 3). Polygon grouped-daily stays the PRIMARY path. The
fallback tests below fake ``feature_store_reader.read_latest_daily_closes``
via ``monkeypatch`` (NOT ``unittest.mock.patch`` — documented full-suite
bleed in this repo; mirrors tests/test_held_thesis_strict.py style).
"""

import sqlite3
import pytest

_pt = pytest.importorskip("scoring.performance_tracker", reason="scoring.performance_tracker is gitignored")
get_trading_day_offset = _pt.get_trading_day_offset
record_new_buy_scores = _pt.record_new_buy_scores
run_performance_checks = _pt.run_performance_checks
_get_spy_price_on_date = _pt._get_spy_price_on_date
_compute_accuracy_stats = _pt._compute_accuracy_stats


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE technical_scores (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            UNIQUE(symbol, date)
        );
        CREATE TABLE macro_snapshots (
            id          INTEGER PRIMARY KEY,
            date        TEXT NOT NULL UNIQUE,
            sp500_close REAL
        );
        CREATE TABLE score_performance (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            score_date      TEXT NOT NULL,
            score           REAL NOT NULL,
            price_on_date   REAL,
            price_10d       REAL,
            price_30d       REAL,
            spy_10d_return  REAL,
            spy_30d_return  REAL,
            return_10d      REAL,
            return_30d      REAL,
            beat_spy_10d    INTEGER,
            beat_spy_30d    INTEGER,
            eval_date_10d   TEXT,
            eval_date_30d   TEXT,
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


def _insert_tech_dates(conn, dates):
    for d in dates:
        conn.execute(
            "INSERT OR IGNORE INTO technical_scores(symbol, date) VALUES ('SPY', ?)", (d,)
        )
    conn.commit()


def _fake_daily_closes(monkeypatch, tickers_and_prices: dict | None):
    """Fake the daily_closes S3 fallback reader (no S3/network).

    ``read_latest_daily_closes`` is imported *inside* run_performance_checks
    from data.fetchers.feature_store_reader, so patch it there. Passing
    None simulates an unavailable feature store (reader returns None).
    """
    import data.fetchers.feature_store_reader as fsr

    monkeypatch.setattr(
        fsr,
        "read_latest_daily_closes",
        lambda: dict(tickers_and_prices) if tickers_and_prices else None,
    )


def _disable_polygon(monkeypatch):
    """Force the polygon grouped-daily PRIMARY path to yield nothing so the
    daily_closes fallback is exercised (the in-function
    `from polygon_client import polygon_client` then raises, caught)."""
    monkeypatch.setitem(__import__("sys").modules, "polygon_client", None)


# ── get_trading_day_offset ────────────────────────────────────────────────────

class TestGetTradingDayOffset:
    def test_returns_nth_date_when_enough_rows(self, db):
        dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        _insert_tech_dates(db, dates)
        result = get_trading_day_offset("2026-01-01", 3, db)
        assert result == "2026-01-06"

    def test_returns_calendar_fallback_when_insufficient_rows(self, db):
        _insert_tech_dates(db, ["2026-01-02", "2026-01-05"])
        result = get_trading_day_offset("2026-01-01", 10, db)
        # Falls back to calendar-based calculation when DB has insufficient rows
        assert result is not None

    def test_returns_calendar_fallback_on_empty_table(self, db):
        result = get_trading_day_offset("2026-01-01", 5, db)
        # Falls back to calendar-based calculation when DB is empty
        assert result is not None

    def test_offset_of_one(self, db):
        _insert_tech_dates(db, ["2026-01-02", "2026-01-05"])
        result = get_trading_day_offset("2026-01-01", 1, db)
        assert result == "2026-01-02"


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


# ── _get_spy_price_on_date ────────────────────────────────────────────────────

class TestGetSpyPriceOnDate:
    def test_returns_price_when_row_exists(self, db):
        db.execute("INSERT INTO macro_snapshots(date, sp500_close) VALUES ('2026-01-01', 500.0)")
        db.commit()
        result = _get_spy_price_on_date("2026-01-01", db)
        assert result == 500.0

    def test_returns_none_when_no_row(self, db):
        result = _get_spy_price_on_date("2026-01-01", db)
        assert result is None

    def test_returns_none_when_price_is_null(self, db):
        db.execute("INSERT INTO macro_snapshots(date, sp500_close) VALUES ('2026-01-01', NULL)")
        db.commit()
        result = _get_spy_price_on_date("2026-01-01", db)
        assert result is None


# ── _compute_accuracy_stats ───────────────────────────────────────────────────

class TestComputeAccuracyStats:
    def _insert_perf_row(self, db, symbol, score_date, beat_10d, beat_30d=None):
        db.execute(
            """INSERT INTO score_performance
               (symbol, score_date, score, price_on_date, beat_spy_10d, beat_spy_30d)
               VALUES (?, ?, 75.0, 100.0, ?, ?)""",
            (symbol, score_date, beat_10d, beat_30d),
        )
        db.commit()

    def test_empty_table_returns_none(self, db):
        result = _compute_accuracy_stats(db, "2026-03-05")
        assert result["accuracy_10d"] is None
        assert result["accuracy_30d"] is None
        assert result["recalibration_flag"] is False

    def test_all_beat_spy_no_recalibration(self, db):
        for i, sym in enumerate(["A", "B", "C"]):
            self._insert_perf_row(db, sym, f"2026-02-0{i+1}", beat_10d=1, beat_30d=1)
        result = _compute_accuracy_stats(db, "2026-03-05")
        assert result["accuracy_10d"] == 100.0
        assert result["recalibration_flag"] is False

    def test_low_accuracy_triggers_recalibration(self, db):
        # Only 1 out of 4 beats SPY → 25% < 55% threshold
        self._insert_perf_row(db, "A", "2026-02-01", beat_10d=1)
        for i, sym in enumerate(["B", "C", "D"]):
            self._insert_perf_row(db, sym, f"2026-02-0{i+2}", beat_10d=0)
        result = _compute_accuracy_stats(db, "2026-03-05")
        assert result["recalibration_flag"] is True
        assert result["accuracy_10d"] < 55.0

    def test_sample_size_reported(self, db):
        for i, sym in enumerate(["A", "B"]):
            self._insert_perf_row(db, sym, f"2026-02-0{i+1}", beat_10d=1)
        result = _compute_accuracy_stats(db, "2026-03-05")
        assert result["sample_size"] == 2

    def test_30d_accuracy_computed(self, db):
        self._insert_perf_row(db, "A", "2026-02-01", beat_10d=1, beat_30d=1)
        self._insert_perf_row(db, "B", "2026-02-02", beat_10d=1, beat_30d=0)
        result = _compute_accuracy_stats(db, "2026-03-05")
        assert result["accuracy_30d"] == 50.0


# ── run_performance_checks ────────────────────────────────────────────────────

class TestRunPerformanceChecks:
    def test_no_pending_rows_returns_stats(self, db):
        result = run_performance_checks(db, "2026-03-05")
        assert "accuracy_10d" in result
        assert "recalibration_flag" in result

    def test_module_is_yfinance_free(self):
        """Post-PR3 the module imports no yfinance and has no yf symbol."""
        import inspect

        src = inspect.getsource(_pt)
        assert "import yfinance" not in src
        assert "yf.download" not in src
        assert not hasattr(_pt, "yf")

    def test_degrades_when_fallback_unavailable(self, db, monkeypatch):
        """Polygon empty + daily_closes reader returns None → graceful
        degrade to accuracy-stats-only, never raises (replaces the old
        'skips when yfinance fails' contract)."""
        _disable_polygon(monkeypatch)
        _fake_daily_closes(monkeypatch, None)  # reader returns None
        db.execute(
            "INSERT INTO score_performance(symbol, score_date, score, price_on_date) VALUES ('PLTR', '2025-12-01', 75.0, 100.0)"
        )
        db.commit()
        result = run_performance_checks(db, "2026-03-05")
        assert "accuracy_10d" in result  # falls back gracefully, no raise

    def test_evaluates_10d_window_via_daily_closes_fallback(self, db, monkeypatch):
        score_date = "2025-12-01"
        today = "2026-03-05"

        # 10 trading days after score_date
        td_dates = [f"2025-12-{i:02d}" for i in range(2, 12)]
        _insert_tech_dates(db, td_dates)

        # SPY reference price
        db.execute(
            "INSERT INTO macro_snapshots(date, sp500_close) VALUES (?, ?)",
            (score_date, 500.0),
        )

        db.execute(
            "INSERT INTO score_performance(symbol, score_date, score, price_on_date) VALUES (?, ?, ?, ?)",
            ("PLTR", score_date, 75.0, 100.0),
        )
        db.commit()

        _disable_polygon(monkeypatch)
        _fake_daily_closes(monkeypatch, {"PLTR": 115.0, "SPY": 510.0})

        result = run_performance_checks(db, today)
        assert "accuracy_10d" in result

        row = db.execute(
            "SELECT price_10d, return_10d FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row[0] == 115.0
        assert abs(row[1] - 15.0) < 0.1  # (115/100 - 1) * 100 = 15%

    def test_beat_spy_flag_set_via_daily_closes_fallback(self, db, monkeypatch):
        score_date = "2025-12-01"
        today = "2026-03-05"

        td_dates = [f"2025-12-{i:02d}" for i in range(2, 12)]
        _insert_tech_dates(db, td_dates)
        db.execute(
            "INSERT INTO macro_snapshots(date, sp500_close) VALUES (?, ?)",
            (score_date, 500.0),
        )
        db.execute(
            "INSERT INTO score_performance(symbol, score_date, score, price_on_date) VALUES (?, ?, ?, ?)",
            ("PLTR", score_date, 75.0, 100.0),
        )
        db.commit()

        # PLTR +20%, SPY +2% → beats SPY
        _disable_polygon(monkeypatch)
        _fake_daily_closes(monkeypatch, {"PLTR": 120.0, "SPY": 510.0})
        run_performance_checks(db, today)

        row = db.execute(
            "SELECT beat_spy_10d FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row[0] == 1

    def test_missing_current_price_skips_row(self, db, monkeypatch):
        score_date = "2025-12-01"
        td_dates = [f"2025-12-{i:02d}" for i in range(2, 12)]
        _insert_tech_dates(db, td_dates)
        db.execute(
            "INSERT INTO score_performance(symbol, score_date, score, price_on_date) VALUES (?, ?, ?, ?)",
            ("PLTR", score_date, 75.0, 100.0),
        )
        db.commit()

        # PLTR absent from the fallback (only SPY present) → row skipped.
        _disable_polygon(monkeypatch)
        _fake_daily_closes(monkeypatch, {"SPY": 510.0})
        result = run_performance_checks(db, "2026-03-05")
        assert "accuracy_10d" in result
        row = db.execute(
            "SELECT price_10d FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row[0] is None  # no eval recorded — graceful skip

    def test_polygon_primary_path_unaffected(self, db, monkeypatch):
        """The yfinance→daily_closes swap is fallback-only: when polygon
        grouped-daily returns prices, the daily_closes reader is never
        called (primary path unchanged)."""
        score_date = "2025-12-01"
        today = "2026-03-05"
        td_dates = [f"2025-12-{i:02d}" for i in range(2, 12)]
        _insert_tech_dates(db, td_dates)
        db.execute(
            "INSERT INTO macro_snapshots(date, sp500_close) VALUES (?, ?)",
            (score_date, 500.0),
        )
        db.execute(
            "INSERT INTO score_performance(symbol, score_date, score, price_on_date) VALUES (?, ?, ?, ?)",
            ("PLTR", score_date, 75.0, 100.0),
        )
        db.commit()

        # Stub polygon_client so the PRIMARY path supplies all prices.
        import sys
        import types

        fake_mod = types.ModuleType("polygon_client")

        class _FakeClient:
            def get_grouped_daily(self, _today):
                return {
                    "PLTR": {"close": 130.0},
                    "SPY": {"close": 505.0},
                }

        fake_mod.polygon_client = lambda: _FakeClient()
        monkeypatch.setitem(sys.modules, "polygon_client", fake_mod)

        # Tripwire: the fallback reader must NOT be called when polygon
        # covers every needed ticker.
        import data.fetchers.feature_store_reader as fsr

        def _boom():
            raise AssertionError(
                "read_latest_daily_closes called despite polygon covering all tickers"
            )

        monkeypatch.setattr(fsr, "read_latest_daily_closes", _boom)

        result = run_performance_checks(db, today)
        assert "accuracy_10d" in result
        row = db.execute(
            "SELECT price_10d, return_10d FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row[0] == 130.0
        assert abs(row[1] - 30.0) < 0.1  # (130/100 - 1) * 100
