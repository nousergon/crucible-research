"""Unit tests for ``thinktank/outcome_ic.py`` — the Think Tank rating ->
realized 21d alpha IC validation (config#2467).

Mirrors ``tests/test_judge_outcome_ic.py``'s coverage shape: pure-compute
correctness on synthetic (rating, realized-alpha) pairs, the honest small-N
"insufficient" floor, and the S3-listing loader's dated-snapshot union +
latest.json exclusion — all with no AWS/network calls (injected fakes only).
"""

from __future__ import annotations

import json

from thinktank.outcome_ic import (
    MIN_EVAL_DATES,
    SCHEMA_VERSION,
    compute_think_tank_outcome_ic,
    load_rating_events,
    load_realized_alpha,
)


# ── compute_think_tank_outcome_ic ──────────────────────────────────────────


def test_insufficient_below_min_eval_dates():
    """A single rating date never clusters (MIN_EVAL_DATES floor) —> insufficient.
    date_clustered_stats still reports a mean for n=1 (matches
    evals/judge_outcome_ic's own n=1 semantics), but se/t/p stay undefined,
    so status can never tip to "ok" on a single date — never a fabricated
    significance claim."""
    ratings = {("AAA", "2026-07-02"): 80.0, ("BBB", "2026-07-02"): 20.0}
    realized = {("AAA", "2026-07-02"): 0.05, ("BBB", "2026-07-02"): -0.03}
    block = compute_think_tank_outcome_ic(ratings, realized)
    assert block["schema_version"] == SCHEMA_VERSION
    assert block["status"] == "insufficient"
    assert block["overall"]["date_ic_mean"] == 1.0
    assert block["overall"]["date_ic_t"] is None
    assert block["overall"]["date_ic_p"] is None
    assert block["overall"]["n_rating_dates"] == 1
    # Pooled IC is still a descriptive companion even when clustering can't run.
    assert block["overall"]["pooled_ic"] == 1.0
    assert block["n_ratings_total"] == 2
    assert block["n_unresolved"] == 0


def test_insufficient_when_nothing_resolved():
    """Every rating unresolved (realized alpha not yet matured) -> insufficient,
    n=0 everywhere, n_unresolved == n_ratings_total. This is the exact shape
    seen against live production data on 2026-07-14 (67 ratings, 0 resolved,
    earliest rating too recent for its 21-trading-day window to have closed)."""
    ratings = {("AAA", "2026-07-13"): 80.0, ("BBB", "2026-07-13"): 20.0}
    block = compute_think_tank_outcome_ic(ratings, realized={})
    assert block["status"] == "insufficient"
    assert block["overall"]["n"] == 0
    assert block["overall"]["pooled_ic"] is None
    assert block["n_ratings_total"] == 2
    assert block["n_unresolved"] == 2


def test_ok_status_with_enough_clustered_dates_and_ic_variance():
    """>= MIN_EVAL_DATES contributing dates, each with >= MIN_PAIRS_PER_DATE
    joined pairs, with some cross-date IC variance (not all-identical, so the
    clustered SE/t are defined, unlike the degenerate zero-variance all-1.0
    case) -> status ok, non-null clustered stats."""
    ratings = {}
    realized = {}
    # Two dates with perfect rank agreement, one date with perfect rank
    # DISAGREEMENT — gives the per-date IC series (1.0, 1.0, -1.0) nonzero
    # variance so date_clustered_stats's t_stat is defined.
    for date, direction in [
        ("2026-06-01", 1.0), ("2026-06-02", 1.0), ("2026-06-03", -1.0),
    ]:
        for j, ticker in enumerate(["AAA", "BBB", "CCC"]):
            ratings[(ticker, date)] = float(90 - j * 30)  # AAA=90, BBB=60, CCC=30
            realized[(ticker, date)] = direction * float(0.05 - j * 0.02)
    block = compute_think_tank_outcome_ic(ratings, realized)
    assert block["status"] == "ok"
    assert block["overall"]["n_rating_dates"] == 3
    assert block["overall"]["date_ic_mean"] == round(1 / 3, 6)
    assert block["overall"]["date_ic_t"] is not None
    assert block["overall"]["date_ic_p"] is not None
    assert block["overall"]["n"] == 9
    assert block["n_unresolved"] == 0


def test_repeat_rating_of_same_ticker_on_different_dates_both_count():
    """A ticker re-rated on a later trading day contributes a second,
    independent (ticker, date) pair — matches evals/judge_outcome_ic's
    per-eval-date treatment of repeat judge scores of the same agent."""
    ratings = {
        ("AAA", "2026-06-01"): 80.0, ("BBB", "2026-06-01"): 20.0,
        ("AAA", "2026-06-15"): 40.0, ("BBB", "2026-06-15"): 90.0,
    }
    realized = {
        ("AAA", "2026-06-01"): 0.05, ("BBB", "2026-06-01"): -0.03,
        ("AAA", "2026-06-15"): -0.02, ("BBB", "2026-06-15"): 0.04,
    }
    block = compute_think_tank_outcome_ic(ratings, realized)
    assert block["overall"]["n"] == 4
    assert block["overall"]["n_rating_dates"] == 2


# ── load_rating_events (fake S3, no network) ───────────────────────────────


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return self._pages


class _FakeS3:
    """Minimal boto3 S3-client double: a fixed key listing + in-memory bodies."""

    def __init__(self, keys, bodies):
        self._keys = keys
        self._bodies = bodies

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator([{"Contents": [{"Key": k} for k in self._keys]}])

    def get_object(self, Bucket, Key):
        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Body(self._bodies[Key])}


def test_load_rating_events_unions_dated_snapshots_and_skips_latest():
    board_day1 = {
        "rows": {
            "AAA": {"rating": 80, "thesis_trading_day": "2026-07-02"},
            "BBB": {"rating": 20, "thesis_trading_day": "2026-07-02"},
        }
    }
    # day2's board upserts AAA (re-rated) and keeps BBB's row unchanged (per
    # thinktank/ratings.update_ratings_board's upsert-in-place semantics) —
    # BBB's (BBB, 2026-07-02) pair must NOT be double-counted from this file.
    board_day2 = {
        "rows": {
            "AAA": {"rating": 60, "thesis_trading_day": "2026-07-06"},
            "BBB": {"rating": 20, "thesis_trading_day": "2026-07-02"},
        }
    }
    keys = [
        "thinktank/ratings/2026-07-02.json",
        "thinktank/ratings/2026-07-06.json",
        "thinktank/ratings/latest.json",  # must be skipped
    ]
    bodies = {
        "thinktank/ratings/2026-07-02.json": json.dumps(board_day1).encode(),
        "thinktank/ratings/2026-07-06.json": json.dumps(board_day2).encode(),
        "thinktank/ratings/latest.json": json.dumps(board_day2).encode(),
    }
    s3 = _FakeS3(keys, bodies)
    events = load_rating_events(s3, "fake-bucket")
    assert events == {
        ("AAA", "2026-07-02"): 80.0,
        ("BBB", "2026-07-02"): 20.0,
        ("AAA", "2026-07-06"): 60.0,
    }


def test_load_rating_events_excludes_none_rating_and_empty_date():
    board = {
        "rows": {
            "AAA": {"rating": None, "thesis_trading_day": "2026-07-02"},  # pre-rating legacy
            "BBB": {"rating": 50, "thesis_trading_day": ""},  # no as-of date
            "CCC": {"rating": 70, "thesis_trading_day": "2026-07-02"},
        }
    }
    keys = ["thinktank/ratings/2026-07-02.json"]
    bodies = {keys[0]: json.dumps(board).encode()}
    events = load_rating_events(_FakeS3(keys, bodies), "fake-bucket")
    assert events == {("CCC", "2026-07-02"): 70.0}


# ── load_realized_alpha (fake sqlite, in-memory) ───────────────────────────


def test_load_realized_alpha_excludes_missing_and_unmatured_rows():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.executemany(
        "INSERT INTO universe_returns VALUES (?, ?, ?, ?)",
        [
            ("AAA", "2026-06-01", 0.10, 0.03),   # resolved
            ("BBB", "2026-06-01", None, None),   # unmatured (21d window not closed)
            # CCC has no row at all for this eval_date
        ],
    )
    conn.commit()
    keys = [("AAA", "2026-06-01"), ("BBB", "2026-06-01"), ("CCC", "2026-06-01")]
    out = load_realized_alpha(conn, keys)
    assert out == {("AAA", "2026-06-01"): 0.10 - 0.03}
