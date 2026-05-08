"""Tests for scripts/backfill_calibrator_v1_context.py.

Three contract layers exercised:

1. ``extract_context_for_ticker`` pulls the right fields from real
   signals.json shape (top-level market_regime + sector_modifiers,
   per-ticker quant_score / qual_score / conviction / sector). Returns
   None for unknown tickers AND for "no useful context" rows so the
   UPDATE-WHERE-NULL caller can short-circuit cleanly.

2. ``backfill_one_row`` honors the WHERE-NULL semantics — does NOT
   overwrite already-populated columns. Repeated runs are idempotent
   (after one full pass, second pass updates 0 rows).

3. ``backfill_all`` groups fetches by date (1 signals.json per
   score_date), tolerates missing signals.json (older score_perf rows
   pre-archive), aggregates a clean summary dict.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from scripts.backfill_calibrator_v1_context import (
    backfill_all,
    backfill_one_row,
    extract_context_for_ticker,
    fetch_signals_payload,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_db() -> sqlite3.Connection:
    """Mirror the v12 score_performance schema for in-memory tests."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE score_performance (
            id              INTEGER PRIMARY KEY,
            symbol          TEXT NOT NULL,
            score_date      TEXT NOT NULL,
            score           REAL NOT NULL,
            price_on_date   REAL,
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
def db() -> sqlite3.Connection:
    conn = _make_db()
    yield conn
    conn.close()


def _signals_payload(
    *, market_regime: str = "neutral",
    sector_modifiers: dict | None = None,
    signals: dict | None = None,
) -> dict:
    """Build a signals.json-shaped dict matching the verified
    2026-04-24 production sample."""
    return {
        "date": "2026-04-24",
        "run_date": "2026-04-24",
        "market_regime": market_regime,
        "sector_modifiers": sector_modifiers or {},
        "signals": signals or {},
    }


# ── extract_context_for_ticker ──────────────────────────────────────────────


class TestExtractContext:
    def test_extracts_all_five_fields_from_real_shape(self) -> None:
        payload = _signals_payload(
            market_regime="bull",
            sector_modifiers={"Health Care": 1.12, "Technology": 1.18},
            signals={
                "PODD": {
                    "ticker": "PODD",
                    "quant_score": 72,
                    "qual_score": 62,
                    "conviction": "stable",
                    "sector": "Health Care",
                },
            },
        )
        ctx = extract_context_for_ticker(payload, "PODD")
        assert ctx == {
            "quant_score": 72,
            "qual_score": 62,
            "conviction": "stable",
            "sector_modifier": 1.12,
            "market_regime": "bull",
        }

    def test_unknown_ticker_returns_none(self) -> None:
        payload = _signals_payload(signals={"PODD": {"sector": "Health Care"}})
        assert extract_context_for_ticker(payload, "MISSING") is None

    def test_partial_fields_return_partial_context(self) -> None:
        """If a ticker carries some but not all fields, the partial
        context still returns — UPDATE-WHERE-NULL will only write the
        non-None columns."""
        payload = _signals_payload(
            market_regime="bear",
            sector_modifiers={},  # no modifier for the sector
            signals={"PLTR": {"quant_score": 80, "sector": "Technology"}},
        )
        ctx = extract_context_for_ticker(payload, "PLTR")
        assert ctx is not None
        assert ctx["quant_score"] == 80
        assert ctx["market_regime"] == "bear"
        assert ctx["sector_modifier"] is None  # sector not in modifiers

    def test_all_fields_none_returns_none(self) -> None:
        """Defensive: if a signal has nothing useful, return None so the
        caller short-circuits the UPDATE entirely (preserves
        idempotency on repeated runs)."""
        payload = {"signals": {"PLTR": {"sector": "Unknown"}}}
        assert extract_context_for_ticker(payload, "PLTR") is None

    def test_missing_signals_dict(self) -> None:
        """signals.json without a top-level signals dict — older format
        or corrupted file. Don't crash, return None."""
        assert extract_context_for_ticker({}, "PLTR") is None


# ── backfill_one_row (UPDATE-WHERE-NULL semantics) ──────────────────────────


class TestBackfillOneRow:
    def _seed(self, db, **kwargs):
        cols = {
            "symbol": "PLTR", "score_date": "2026-04-24", "score": 75.0,
            "quant_score": None, "qual_score": None, "conviction": None,
            "sector_modifier": None, "market_regime": None,
            **kwargs,
        }
        db.execute(
            """INSERT INTO score_performance
               (symbol, score_date, score, quant_score, qual_score,
                conviction, sector_modifier, market_regime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cols["symbol"], cols["score_date"], cols["score"],
             cols["quant_score"], cols["qual_score"], cols["conviction"],
             cols["sector_modifier"], cols["market_regime"]),
        )
        db.commit()

    def test_writes_when_all_columns_null(self, db):
        self._seed(db)
        ctx = {
            "quant_score": 70.0, "qual_score": 80.0, "conviction": "rising",
            "sector_modifier": 1.15, "market_regime": "bull",
        }
        wrote = backfill_one_row(
            db, "PLTR", "2026-04-24", ctx, dry_run=False,
        )
        assert wrote is True
        row = db.execute(
            "SELECT quant_score, qual_score, conviction, sector_modifier, market_regime "
            "FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row == (70.0, 80.0, "rising", 1.15, "bull")

    def test_skips_already_populated_columns(self, db):
        """Per UPDATE-WHERE-NULL contract: don't overwrite non-NULL
        values. Repeated runs after producer wire-up has populated rows
        are no-ops."""
        self._seed(db, quant_score=99.0, market_regime="existing")
        ctx = {
            "quant_score": 70.0, "qual_score": 80.0, "conviction": "rising",
            "sector_modifier": 1.15, "market_regime": "bull",
        }
        wrote = backfill_one_row(
            db, "PLTR", "2026-04-24", ctx, dry_run=False,
        )
        assert wrote is True
        row = db.execute(
            "SELECT quant_score, qual_score, conviction, sector_modifier, market_regime "
            "FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        # quant_score + market_regime preserved; the other 3 backfilled.
        assert row == (99.0, 80.0, "rising", 1.15, "existing")

    def test_returns_false_when_nothing_to_write(self, db):
        """Idempotency: a fully-populated row does not churn."""
        self._seed(
            db, quant_score=50.0, qual_score=60.0, conviction="stable",
            sector_modifier=1.0, market_regime="neutral",
        )
        ctx = {
            "quant_score": 70.0, "qual_score": 80.0, "conviction": "rising",
            "sector_modifier": 1.15, "market_regime": "bull",
        }
        wrote = backfill_one_row(
            db, "PLTR", "2026-04-24", ctx, dry_run=False,
        )
        assert wrote is False
        row = db.execute(
            "SELECT quant_score, qual_score, conviction, sector_modifier, market_regime "
            "FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        # Original values unchanged.
        assert row == (50.0, 60.0, "stable", 1.0, "neutral")

    def test_dry_run_does_not_write(self, db):
        self._seed(db)
        ctx = {
            "quant_score": 70.0, "qual_score": 80.0, "conviction": "rising",
            "sector_modifier": 1.15, "market_regime": "bull",
        }
        wrote = backfill_one_row(
            db, "PLTR", "2026-04-24", ctx, dry_run=True,
        )
        assert wrote is True  # would have written
        row = db.execute(
            "SELECT quant_score, qual_score, conviction, sector_modifier, market_regime "
            "FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        # No actual change.
        assert row == (None, None, None, None, None)

    def test_missing_row_returns_false(self, db):
        """Callers may pass (symbol, date) pairs that don't exist —
        return False, don't crash."""
        wrote = backfill_one_row(
            db, "GHOST", "2026-04-24",
            {"quant_score": 50, "qual_score": None, "conviction": None,
             "sector_modifier": None, "market_regime": None},
            dry_run=False,
        )
        assert wrote is False


# ── backfill_all (orchestration + summary) ──────────────────────────────────


def _put_payload(s3: MagicMock, bucket: str, date: str, payload: dict) -> None:
    """Configure the mock S3 to return ``payload`` for the canonical
    signals/{date}/signals.json key."""
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()

    def get_object(Bucket=None, Key=None):
        if Key == f"signals/{date}/signals.json":
            return {"Body": body}
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject",
        )

    s3.get_object.side_effect = get_object


class TestBackfillAll:
    def _seed_perf(self, db, rows):
        for symbol, score_date in rows:
            db.execute(
                "INSERT INTO score_performance (symbol, score_date, score) "
                "VALUES (?, ?, 75.0)",
                (symbol, score_date),
            )
        db.commit()

    def test_groups_fetches_by_date(self, db):
        self._seed_perf(db, [
            ("PLTR", "2026-04-24"), ("RKLB", "2026-04-24"),
        ])
        s3 = MagicMock()
        _put_payload(s3, "alpha-engine-research", "2026-04-24",
                     _signals_payload(
                         market_regime="bull",
                         sector_modifiers={"Technology": 1.18},
                         signals={
                             "PLTR": {"quant_score": 70, "qual_score": 60,
                                      "conviction": "rising",
                                      "sector": "Technology"},
                             "RKLB": {"quant_score": 80, "qual_score": 75,
                                      "conviction": "stable",
                                      "sector": "Technology"},
                         },
                     ))
        summary = backfill_all(db, s3, "alpha-engine-research", dry_run=False)
        assert summary["rows_eligible"] == 2
        assert summary["dates_processed"] == 1
        assert summary["rows_updated"] == 2
        # One get_object per distinct date — not per row.
        assert s3.get_object.call_count == 1

    def test_missing_signals_json_skips_date_cleanly(self, db):
        self._seed_perf(db, [("PLTR", "2026-01-01")])
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject",
        )
        summary = backfill_all(db, s3, "alpha-engine-research", dry_run=False)
        assert summary["dates_missing_signals"] == 1
        assert summary["rows_updated"] == 0
        assert summary["rows_skipped_no_match"] == 1
        # Row stays NULL.
        row = db.execute(
            "SELECT quant_score FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row[0] is None

    def test_ticker_not_in_signals_skips(self, db):
        """A score_performance row whose ticker isn't in that day's
        signals.json — older format / mismatched source — counts as
        skipped, not failed."""
        self._seed_perf(db, [("GHOST", "2026-04-24")])
        s3 = MagicMock()
        _put_payload(s3, "alpha-engine-research", "2026-04-24",
                     _signals_payload(
                         signals={"PLTR": {"quant_score": 70, "qual_score": 60,
                                           "conviction": "rising",
                                           "sector": "Technology"}}))
        summary = backfill_all(db, s3, "alpha-engine-research", dry_run=False)
        assert summary["rows_eligible"] == 1
        assert summary["dates_processed"] == 1
        assert summary["rows_updated"] == 0
        assert summary["rows_skipped_no_match"] == 1

    def test_dry_run_reports_but_does_not_write(self, db):
        self._seed_perf(db, [("PLTR", "2026-04-24")])
        s3 = MagicMock()
        _put_payload(s3, "alpha-engine-research", "2026-04-24",
                     _signals_payload(
                         market_regime="bull",
                         sector_modifiers={"Technology": 1.18},
                         signals={"PLTR": {"quant_score": 70, "qual_score": 60,
                                           "conviction": "rising",
                                           "sector": "Technology"}}))
        summary = backfill_all(db, s3, "alpha-engine-research", dry_run=True)
        assert summary["rows_updated"] == 1
        # No actual write.
        row = db.execute(
            "SELECT quant_score FROM score_performance WHERE symbol='PLTR'"
        ).fetchone()
        assert row[0] is None


# ── fetch_signals_payload (S3 layer) ───────────────────────────────────────


class TestFetchSignalsPayload:
    def test_returns_payload_on_success(self) -> None:
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps(
            _signals_payload(market_regime="bull")
        ).encode()
        s3.get_object.return_value = {"Body": body}
        payload = fetch_signals_payload(s3, "alpha-engine-research", "2026-04-24")
        assert payload is not None
        assert payload["market_regime"] == "bull"

    def test_returns_none_on_no_such_key(self) -> None:
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject",
        )
        assert fetch_signals_payload(s3, "alpha-engine-research", "2026-01-01") is None

    def test_other_s3_errors_propagate(self) -> None:
        """Access denied / 500s should NOT be silently swallowed —
        operator should know."""
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "x"}}, "GetObject",
        )
        with pytest.raises(ClientError):
            fetch_signals_payload(s3, "alpha-engine-research", "2026-04-24")

    def test_malformed_json_returns_none(self) -> None:
        """Older format / corrupted file — log and skip, don't crash
        the whole backfill."""
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = b"not json"
        s3.get_object.return_value = {"Body": body}
        assert fetch_signals_payload(s3, "alpha-engine-research", "2026-04-24") is None
