"""Tests for `evals.last_week_scorecard`.

Synthetic SQLite fixture mirrors the research.db schema. Each test
focuses on one invariant of the scorecard build path.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from evals.last_week_scorecard import (
    DEFAULT_SCORECARD_PREFIX,
    Scorecard,
    SectorRow,
    TickerOutcome,
    build_scorecard,
    emit_scorecard_to_s3,
    format_scorecard_text,
)


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE score_performance (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            score_date TEXT NOT NULL,
            score REAL NOT NULL,
            price_on_date REAL,
            price_10d REAL,
            price_30d REAL,
            spy_10d_return REAL,
            spy_30d_return REAL,
            return_10d REAL,
            return_30d REAL,
            beat_spy_10d INTEGER,
            beat_spy_30d INTEGER,
            eval_date_10d TEXT,
            eval_date_30d TEXT,
            UNIQUE(symbol, score_date)
        );
        CREATE TABLE predictor_outcomes (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT,
            prediction_confidence REAL,
            p_up REAL,
            p_flat REAL,
            p_down REAL,
            score_modifier_applied REAL DEFAULT 0.0,
            actual_5d_return REAL,
            correct_5d INTEGER,
            actual_log_alpha REAL,
            horizon_days INTEGER,
            correct INTEGER,
            UNIQUE(symbol, prediction_date)
        );
        CREATE TABLE population (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL UNIQUE,
            sector TEXT NOT NULL,
            long_term_score REAL,
            long_term_rating TEXT,
            conviction TEXT,
            price_target_upside REAL,
            thesis_summary TEXT,
            entry_date TEXT,
            tenure_weeks INTEGER
        );
        CREATE TABLE macro_snapshots (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            regime TEXT
        );
        """
    )
    conn.commit()
    return conn


def _seed_signal(conn, sym, score_date, beat_10d=None, beat_30d=None,
                 return_10d=None, spy_10d=None, score=70.0):
    conn.execute(
        "INSERT INTO score_performance (symbol, score_date, score, beat_spy_10d, "
        "beat_spy_30d, return_10d, spy_10d_return) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sym, score_date, score, beat_10d, beat_30d, return_10d, spy_10d),
    )


def _seed_population(conn, sym, sector):
    conn.execute(
        "INSERT INTO population (symbol, sector) VALUES (?, ?)",
        (sym, sector),
    )


def _seed_prediction(conn, sym, pred_date, direction, conf,
                     correct=None, log_alpha=None,
                     correct_5d=None, ret_5d=None):
    conn.execute(
        "INSERT INTO predictor_outcomes (symbol, prediction_date, predicted_direction, "
        "prediction_confidence, correct, actual_log_alpha, correct_5d, actual_5d_return) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sym, pred_date, direction, conf, correct, log_alpha, correct_5d, ret_5d),
    )


@pytest.fixture
def empty_db(tmp_path):
    conn = _make_db(tmp_path / "research.db")
    yield conn
    conn.close()


@pytest.fixture
def populated_db(tmp_path):
    conn = _make_db(tmp_path / "research.db")
    # Population — 2 sectors, 5 tickers.
    for sym, sector in (
        ("AAPL", "Tech"),
        ("MSFT", "Tech"),
        ("GOOG", "Tech"),
        ("JNJ", "Healthcare"),
        ("PFE", "Healthcare"),
    ):
        _seed_population(conn, sym, sector)

    # Signals across the lookback window (4 weeks ending 2026-05-22).
    # Tech: 3 signals, 2 beats at 10d -> 67% hit rate
    _seed_signal(conn, "AAPL", "2026-05-09", beat_10d=1, beat_30d=1,
                 return_10d=0.05, spy_10d=0.02)
    _seed_signal(conn, "MSFT", "2026-05-09", beat_10d=1, beat_30d=0,
                 return_10d=0.04, spy_10d=0.02)
    _seed_signal(conn, "GOOG", "2026-05-09", beat_10d=0, beat_30d=1,
                 return_10d=0.01, spy_10d=0.02)
    # Healthcare: 3 signals, 1 beat at 10d -> 33% hit rate
    _seed_signal(conn, "JNJ", "2026-05-09", beat_10d=1, beat_30d=1,
                 return_10d=0.03, spy_10d=0.02)
    _seed_signal(conn, "PFE", "2026-05-09", beat_10d=0, beat_30d=0,
                 return_10d=-0.01, spy_10d=0.02)
    _seed_signal(conn, "JNJ", "2026-05-16", beat_10d=0, beat_30d=None,
                 return_10d=0.0, spy_10d=0.01)
    # Predictions: 5 UP calls with varied realized alpha
    _seed_prediction(conn, "AAPL", "2026-05-09", "UP", 0.82,
                     correct=1, log_alpha=0.06)
    _seed_prediction(conn, "MSFT", "2026-05-09", "UP", 0.71,
                     correct=1, log_alpha=0.04)
    _seed_prediction(conn, "GOOG", "2026-05-09", "UP", 0.65,
                     correct=0, log_alpha=-0.08)  # surprise candidate
    _seed_prediction(conn, "JNJ", "2026-05-09", "UP", 0.69,
                     correct=1, log_alpha=0.02)
    _seed_prediction(conn, "PFE", "2026-05-09", "UP", 0.74,
                     correct=0, log_alpha=-0.06)  # surprise candidate

    # Legacy-column row to validate COALESCE: pre-cutover prediction
    # populates correct_5d / actual_5d_return only.
    _seed_prediction(conn, "PFE", "2026-05-02", "UP", 0.60,
                     correct=None, log_alpha=None,
                     correct_5d=1, ret_5d=0.03)

    conn.execute(
        "INSERT INTO macro_snapshots (date, regime) VALUES (?, ?)",
        ("2026-05-20", "neutral"),
    )
    conn.commit()
    yield conn
    conn.close()


class TestEmptyDB:
    def test_returns_scorecard_with_no_outcomes(self, empty_db):
        sc = build_scorecard(empty_db, as_of_date=date(2026, 5, 23))
        assert isinstance(sc, Scorecard)
        assert sc.n_resolved_predictions == 0
        assert sc.n_resolved_signals_10d == 0
        assert sc.overall_predictor_hit_rate is None
        assert sc.overall_signal_hit_rate_10d is None
        assert sc.per_sector == []
        assert sc.top_surprises == []
        assert sc.top_confirmations == []
        assert sc.market_regime is None


class TestPopulatedDB:
    def test_overall_predictor_hit_rate(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        # 6 predictor rows: 4 canonical-resolved (3 correct, 1 wrong) +
        # 1 canonical-resolved (PFE 5/9, wrong) + 1 legacy-resolved (PFE 5/2, correct).
        # Total resolved = 6, correct = 4 (AAPL/MSFT/JNJ canonical + PFE legacy).
        assert sc.n_resolved_predictions == 6
        assert sc.overall_predictor_hit_rate == pytest.approx(4 / 6)

    def test_overall_signal_hit_rates(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        # 6 signals, 5 with beat_spy_10d, 5 with beat_spy_30d
        # beat_spy_10d sum: 1+1+0+1+0+0 = 3 of 6 -> 50%
        assert sc.n_resolved_signals_10d == 6
        assert sc.overall_signal_hit_rate_10d == pytest.approx(3 / 6)
        # beat_spy_30d sum: 1+0+1+1+0 = 3 of 5 (JNJ 5/16 is None) -> 60%
        assert sc.n_resolved_signals_30d == 5
        assert sc.overall_signal_hit_rate_30d == pytest.approx(3 / 5)

    def test_per_sector_rows_only_above_min_n(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        sectors = {s.sector: s for s in sc.per_sector}
        # Both sectors have >= 3 signals in the window.
        assert "Tech" in sectors
        assert "Healthcare" in sectors
        # Tech: 2 of 3 beat at 10d
        assert sectors["Tech"].hit_rate_10d == pytest.approx(2 / 3)
        # Healthcare: 1 of 3 beat at 10d (JNJ 5/9 + JNJ 5/16 + PFE 5/9)
        assert sectors["Healthcare"].hit_rate_10d == pytest.approx(1 / 3)

    def test_surprises_predicted_up_realized_worst(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        # Top surprise should be GOOG or PFE (canonical) — the two
        # negative realized alphas. Cap at K=3 means we get both.
        surprise_symbols = [s.symbol for s in sc.top_surprises]
        assert "GOOG" in surprise_symbols
        assert "PFE" in surprise_symbols  # the canonical 5/9 row
        # Surprises sorted ascending by sigma — most negative first.
        sigmas = [s.surprise_sigma for s in sc.top_surprises]
        assert sigmas == sorted(sigmas)

    def test_confirmations_predicted_up_realized_best(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        confirm_symbols = [c.symbol for c in sc.top_confirmations]
        # AAPL is the largest positive realized log-α and should top.
        assert confirm_symbols[0] == "AAPL"
        # Confirmations sorted by sigma descending — most positive first.
        sigmas = [c.surprise_sigma for c in sc.top_confirmations]
        assert sigmas == sorted(sigmas, reverse=True)

    def test_market_regime_pulled_from_window(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        assert sc.market_regime == "neutral"

    def test_text_render_includes_overall_section(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        text = format_scorecard_text(sc)
        assert "Prior cycle's realized outcomes" in text
        assert "Predictor hit rate" in text
        assert "Per-sector hit rate" in text
        assert "Tech:" in text and "Healthcare:" in text
        assert "Surprises" in text and "Confirmations" in text
        assert "neutral" in text  # regime echoed


class TestCoalesceLegacyOutcomes:
    def test_legacy_only_prediction_counts_as_resolved(self, tmp_path):
        conn = _make_db(tmp_path / "research.db")
        _seed_population(conn, "AAPL", "Tech")
        # Legacy-only row: correct_5d set, canonical correct NULL.
        _seed_prediction(conn, "AAPL", "2026-05-02", "UP", 0.7,
                         correct=None, log_alpha=None,
                         correct_5d=1, ret_5d=0.03)
        conn.commit()
        sc = build_scorecard(conn, as_of_date=date(2026, 5, 23))
        assert sc.n_resolved_predictions == 1
        assert sc.overall_predictor_hit_rate == 1.0


class TestSurpriseListEdgeCases:
    def test_constant_realized_alpha_yields_empty_lists(self, tmp_path):
        # std=0 across the window -> sigma undefined -> empty lists.
        conn = _make_db(tmp_path / "research.db")
        _seed_population(conn, "A", "Tech")
        _seed_population(conn, "B", "Tech")
        _seed_prediction(conn, "A", "2026-05-09", "UP", 0.7,
                         correct=1, log_alpha=0.05)
        _seed_prediction(conn, "B", "2026-05-09", "UP", 0.7,
                         correct=1, log_alpha=0.05)
        conn.commit()
        sc = build_scorecard(conn, as_of_date=date(2026, 5, 23))
        assert sc.top_surprises == []
        assert sc.top_confirmations == []

    def test_single_resolved_prediction_yields_empty_lists(self, tmp_path):
        conn = _make_db(tmp_path / "research.db")
        _seed_population(conn, "A", "Tech")
        _seed_prediction(conn, "A", "2026-05-09", "UP", 0.7,
                         correct=1, log_alpha=0.05)
        conn.commit()
        sc = build_scorecard(conn, as_of_date=date(2026, 5, 23))
        assert sc.top_surprises == []
        assert sc.top_confirmations == []


class TestSerialization:
    def test_to_dict_round_trips_through_json(self, populated_db):
        import json
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        d = sc.to_dict()
        s = json.dumps(d)
        d2 = json.loads(s)
        # Top-level keys preserved
        assert set(d.keys()) == set(d2.keys())
        # Nested dataclasses become dicts under asdict
        assert isinstance(d["per_sector"], list)
        if d["per_sector"]:
            assert "sector" in d["per_sector"][0]
            assert "hit_rate_10d" in d["per_sector"][0]


class _StubS3Client:
    """Captures put_object calls for assertion."""

    def __init__(self, fail_after: int | None = None):
        self.calls: list[dict] = []
        self._fail_after = fail_after

    def put_object(self, **kwargs):
        if self._fail_after is not None and len(self.calls) >= self._fail_after:
            raise RuntimeError("simulated S3 outage")
        self.calls.append(kwargs)
        return {"ETag": '"deadbeef"'}


class TestEmitScorecardToS3:
    def test_writes_dated_and_latest_keys(self, populated_db):
        import json
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        result = emit_scorecard_to_s3(sc, s3_client=client, bucket="bkt")
        # Two put_object calls — dated + latest sidecar.
        assert len(client.calls) == 2
        keys = [c["Key"] for c in client.calls]
        assert any(k.endswith("/latest.json") for k in keys)
        # Dated key has the YYMMDDHHMM run_id, not "latest".
        dated = [k for k in keys if not k.endswith("/latest.json")][0]
        assert dated.startswith(DEFAULT_SCORECARD_PREFIX)
        # Result dict carries both keys back.
        assert result["latest_key"].endswith("/latest.json")
        assert result["dated_key"] == dated
        # Both payloads identical.
        bodies = [c["Body"] for c in client.calls]
        assert bodies[0] == bodies[1]
        # Round-trip the body back through json.
        payload = json.loads(bodies[0])
        assert payload["as_of_date"] == "2026-05-23"

    def test_custom_prefix_honored(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        emit_scorecard_to_s3(
            sc, s3_client=client, bucket="bkt", prefix="custom/path"
        )
        assert all(c["Key"].startswith("custom/path/") for c in client.calls)

    def test_explicit_run_id_used(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        result = emit_scorecard_to_s3(
            sc, s3_client=client, bucket="bkt", run_id="2605271400"
        )
        assert result["run_id"] == "2605271400"
        assert any("2605271400" in c["Key"] for c in client.calls)

    def test_empty_bucket_raises(self, populated_db):
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client()
        with pytest.raises(ValueError, match="non-empty bucket"):
            emit_scorecard_to_s3(sc, s3_client=client, bucket="")

    def test_s3_failure_propagates_per_no_silent_fails(self, populated_db):
        # Producer-side path: must raise, not swallow. Phase 2 consumers
        # depend on the artifact existing; silent failure here would
        # leave the next research cycle without scorecard data and the
        # operator with no signal that it's missing.
        sc = build_scorecard(populated_db, as_of_date=date(2026, 5, 23))
        client = _StubS3Client(fail_after=0)  # fail on first call
        with pytest.raises(RuntimeError, match="simulated S3 outage"):
            emit_scorecard_to_s3(sc, s3_client=client, bucket="bkt")
