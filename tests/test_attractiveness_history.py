"""Tests for scoring/attractiveness_history.py — the per-stock attractiveness
time-series store + backfill.

Locks:
  1. Recompute parity — backfill recompute == live-board attractiveness for the
     same factor profiles (both call compute_cross_sectional_attractiveness).
  2. Idempotent upsert — re-appending a date replaces (never duplicates); a new
     date grows the series.
  3. Backfill over factor-profile snapshots (in-memory S3) yields one row per
     (date, ticker) with recomputed attractiveness.
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.attractiveness_history import (  # noqa: E402
    HISTORY_KEY,
    append_history,
    backfill_from_factor_profiles,
    build_history_rows,
    extract_history_rows_from_board,
    read_history,
)
from scoring.universe_board import _PILLAR_ORDER, build_universe_board  # noqa: E402

_EQUAL = dict.fromkeys(_PILLAR_ORDER, 1 / 6)


# ── In-memory S3 ─────────────────────────────────────────────────────────────

class _FakeS3:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body, **kw):
        self.store[Key] = Body if isinstance(Body, (bytes, bytearray)) else bytes(Body)

    def get_object(self, *, Bucket, Key, **kw):
        if Key not in self.store:
            raise KeyError(Key)  # builder catches broadly → treated as absent
        return {"Body": io.BytesIO(self.store[Key])}

    def get_paginator(self, _name):
        store = self.store

        class _P:
            def paginate(self, *, Bucket, Prefix, Delimiter):
                seen = set()
                for k in store:
                    if not k.startswith(Prefix):
                        continue
                    rest = k[len(Prefix):]
                    if Delimiter in rest:
                        seen.add(Prefix + rest.split(Delimiter)[0] + Delimiter)
                yield {"CommonPrefixes": [{"Prefix": p} for p in sorted(seen)]}

        return _P()


def _profiles():
    return {
        "AAPL": {"sector": "Information Technology", "quality_score": 90.0, "value_score": 30.0,
                 "momentum_score": 85.0, "low_vol_score": 60.0, "growth_score": 80.0,
                 "stewardship_score": 70.0},
        "MSFT": {"sector": "Information Technology", "quality_score": 60.0, "value_score": 50.0,
                 "momentum_score": 55.0, "low_vol_score": 70.0, "growth_score": 65.0,
                 "stewardship_score": 60.0},
        "LIN": {"sector": "Materials", "quality_score": 50.0, "value_score": 40.0,
                "momentum_score": 30.0, "low_vol_score": 60.0},  # 4-of-6 coverage
    }


# ── 1. Recompute parity vs the live board ────────────────────────────────────

def test_recompute_matches_live_board():
    profiles = _profiles()
    evals = [{"ticker": t, "sector": p["sector"], "quant_filter_pass": 1,
              "filter_fail_reason": None} for t, p in profiles.items()]
    board = build_universe_board(
        "2026-06-26", evals, factor_profiles=profiles, classification={},
        technical_df=None, fundamental_df=None, pillar_weights=dict(_EQUAL), gate_config={},
    )
    board_attr = {s["ticker"]: (s["attractiveness_raw"], s["attractiveness_score"])
                  for s in board["stocks"]}
    rows = {r["ticker"]: (r["attractiveness_raw"], r["attractiveness_score"])
            for r in build_history_rows("2026-06-26", profiles, dict(_EQUAL))}
    assert rows == board_attr  # byte-identical numbers from the shared chokepoint


def test_build_rows_shape_and_coverage():
    rows = {r["ticker"]: r for r in build_history_rows("2026-06-26", _profiles(), dict(_EQUAL))}
    assert rows["AAPL"]["as_of"] == "2026-06-26"
    assert rows["AAPL"]["sector"] == "Information Technology"
    # LIN missing growth/stewardship pillars → None (never fabricated).
    assert rows["LIN"]["growth"] is None and rows["LIN"]["stewardship"] is None
    assert rows["LIN"]["attractiveness_raw"] is not None  # still scores on its 4


# ── 2. Idempotent upsert ─────────────────────────────────────────────────────

def test_append_idempotent_by_date():
    s3 = _FakeS3()
    rows = build_history_rows("2026-06-26", _profiles(), dict(_EQUAL))
    n1 = append_history(rows, s3_client=s3)
    n2 = append_history(rows, s3_client=s3)  # same date again → replace, not append
    assert n1 == n2 == len(rows)
    df = read_history(s3_client=s3)
    assert df["as_of"].nunique() == 1 and len(df) == len(rows)


def test_append_new_date_grows_series():
    s3 = _FakeS3()
    append_history(build_history_rows("2026-06-19", _profiles(), dict(_EQUAL)), s3_client=s3)
    append_history(build_history_rows("2026-06-26", _profiles(), dict(_EQUAL)), s3_client=s3)
    df = read_history(s3_client=s3)
    assert sorted(df["as_of"].unique()) == ["2026-06-19", "2026-06-26"]
    assert len(df) == 2 * len(_profiles())


# ── 3. Forward extract from a board ──────────────────────────────────────────

def test_extract_from_board():
    board = {"as_of": "2026-07-04", "stocks": [
        {"ticker": "AAPL", "attractiveness_raw": 0.5, "attractiveness_score": 88.0,
         "sector": "Information Technology", "industry": "Consumer Electronics",
         "pillars": dict.fromkeys(_PILLAR_ORDER, 70.0)},
    ]}
    rows = extract_history_rows_from_board(board)
    assert rows[0]["ticker"] == "AAPL" and rows[0]["attractiveness_score"] == 88.0
    assert rows[0]["quality"] == 70.0 and rows[0]["industry"] == "Consumer Electronics"


# ── 4. Backfill from factor-profile snapshots ────────────────────────────────

def test_backfill_from_factor_profiles():
    s3 = _FakeS3()
    for d in ("2026-06-19", "2026-06-26"):
        s3.put_object(Bucket="b", Key=f"factors/profiles/{d}/by_ticker.json",
                      Body=json.dumps(_profiles()).encode())
    summary = backfill_from_factor_profiles(s3_client=s3)
    assert summary["dates"] == ["2026-06-19", "2026-06-26"]
    df = read_history(s3_client=s3)
    assert len(df) == 2 * len(_profiles())
    assert HISTORY_KEY in s3.store
    # recomputed attractiveness present for every row
    assert df["attractiveness_score"].notna().sum() == len(df)


def test_backfill_no_profiles_raises():
    import pytest
    with pytest.raises(RuntimeError, match="no factors/profiles"):
        backfill_from_factor_profiles(s3_client=_FakeS3())
