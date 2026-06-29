"""Per-stock attractiveness time-series history + one-time backfill.

The dated universe boards (``scanner/universe/{date}/universe.json``) already
carry each cycle's attractiveness, but assembling a per-stock series means
reading N JSON boards. This module maintains a compact, query-optimized,
APPEND-ONLY parquet time-series —
``scanner/universe/history/attractiveness_history.parquet`` — keyed by
``(as_of, ticker)``, plus a one-time BACKFILL that seeds it by recomputing the
SOTA v2 attractiveness from the retained factor-profile history
(``factors/profiles/{date}/by_ticker.json``).

SINGLE SOURCE OF TRUTH: recompute calls
``scoring/universe_board.compute_cross_sectional_attractiveness`` — the exact
function the live board uses — so backfilled history == live-board numbers for
the same inputs.

Failure posture: SECONDARY observability (like the board). The forward append
(``compute_and_write_universe_board``) fail-SOFTs with a WARN; the backfill CLI
fails LOUD (an operator ran it deliberately and wants to know if it broke).
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Optional

from scoring.universe_board import (
    _bucket,
    _client,
    _load_pillar_weights,
    _num,
    _PILLAR_ORDER,
    compute_cross_sectional_attractiveness,
)

logger = logging.getLogger(__name__)

HISTORY_KEY = "scanner/universe/history/attractiveness_history.parquet"

# Stable column order for the parquet (as_of + ticker key, then scores + class).
_HISTORY_COLS = ["as_of", "ticker", "attractiveness_raw", "attractiveness_score",
                 *list(_PILLAR_ORDER), "sector", "industry"]


# ── Recompute (backfill) + extract (forward) ─────────────────────────────────

def _pillar_scores_from_profiles(profiles: dict) -> dict[str, dict]:
    """``factors/profiles`` ``{ticker: {*_score}}`` → ``{ticker: {pillar: 0-100|None}}``."""
    from scoring.composite import _PILLAR_TO_FACTOR_KEY
    return {
        ticker: {p: _num(prof.get(_PILLAR_TO_FACTOR_KEY[p])) for p in _PILLAR_ORDER}
        for ticker, prof in profiles.items()
    }


def build_history_rows(
    as_of: str,
    profiles: dict,
    pillar_weights: dict[str, float],
    classification: dict | None = None,
) -> list[dict]:
    """Recompute v2 attractiveness for one historical cycle's factor profiles →
    history rows. ``classification`` (near-static latest) supplies industry;
    sector falls back to the profile's own sector field."""
    classification = classification or {}
    pillar_scores_by_ticker = _pillar_scores_from_profiles(profiles)
    attr = compute_cross_sectional_attractiveness(pillar_scores_by_ticker, pillar_weights)
    rows = []
    for ticker, prof in profiles.items():
        a = attr.get(ticker, {})
        ps = pillar_scores_by_ticker[ticker]
        cls = classification.get(ticker, {})
        row = {
            "as_of": as_of,
            "ticker": ticker,
            "attractiveness_raw": a.get("attractiveness_raw"),
            "attractiveness_score": a.get("attractiveness_score"),
            "sector": prof.get("sector") or cls.get("sector"),
            "industry": cls.get("industry"),
        }
        for p in _PILLAR_ORDER:
            row[p] = ps[p]
        rows.append(row)
    return rows


def extract_history_rows_from_board(board: dict) -> list[dict]:
    """Forward path — pull the attractiveness slice straight off a v2 board dict
    (no recompute; the board already ran the shared chokepoint)."""
    as_of = board.get("as_of")
    rows = []
    for s in board.get("stocks", []) or []:
        ticker = s.get("ticker")
        if not ticker:
            continue
        pillars = s.get("pillars") or {}
        row = {
            "as_of": as_of,
            "ticker": ticker,
            "attractiveness_raw": s.get("attractiveness_raw"),
            "attractiveness_score": s.get("attractiveness_score"),
            "sector": s.get("sector"),
            "industry": s.get("industry"),
        }
        for p in _PILLAR_ORDER:
            row[p] = pillars.get(p)
        rows.append(row)
    return rows


# ── Parquet store (read / upsert) ────────────────────────────────────────────

def read_history(*, bucket: str | None = None, s3_client: Any = None):
    """Read the full history parquet → DataFrame, or None when absent."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return None
    s3 = _client(s3_client)
    try:
        obj = s3.get_object(Bucket=_bucket(bucket), Key=HISTORY_KEY)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
    except Exception:
        return None


def append_history(rows: list[dict], *, bucket: str | None = None, s3_client: Any = None) -> int:
    """Upsert ``rows`` into the history parquet, IDEMPOTENT by ``as_of`` (a
    re-run of a date replaces that date's full slice — never double-appends).
    Returns the total row count after the write. No-op on empty rows."""
    import pandas as pd

    if not rows:
        return 0
    new = pd.DataFrame(rows)
    new = new.reindex(columns=_HISTORY_COLS)
    existing = read_history(bucket=bucket, s3_client=s3_client)
    if existing is not None and not existing.empty:
        existing = existing.reindex(columns=_HISTORY_COLS)
        replaced_dates = set(new["as_of"].unique())
        keep = existing[~existing["as_of"].isin(replaced_dates)]
        combined = pd.concat([keep, new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values(["as_of", "ticker"]).reset_index(drop=True)
    _write_history(combined, bucket=bucket, s3_client=s3_client)
    logger.info("[attractiveness_history] wrote %d rows (%d dates) → s3://%s/%s",
                len(combined), combined["as_of"].nunique(), _bucket(bucket), HISTORY_KEY)
    return len(combined)


def _write_history(df, *, bucket: str | None = None, s3_client: Any = None) -> None:
    s3 = _client(s3_client)
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    s3.put_object(Bucket=_bucket(bucket), Key=HISTORY_KEY,
                  Body=buf.getvalue(), ContentType="application/octet-stream")


# ── Backfill from factor-profile history ─────────────────────────────────────

def _list_profile_dates(s3: Any, bucket: str) -> list[str]:
    """Dated subdirs under ``factors/profiles/`` (YYYY-MM-DD), chronological."""
    dates: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="factors/profiles/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            token = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if len(token) == 10 and token[4] == "-" and token[7] == "-":
                dates.append(token)
    return sorted(set(dates))


def _read_profiles(s3: Any, bucket: str, date: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"factors/profiles/{date}/by_ticker.json")
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def _read_classification(bucket: str | None, s3_client: Any) -> dict:
    s3 = _client(s3_client)
    try:
        obj = s3.get_object(Bucket=_bucket(bucket),
                            Key="market_data/universe_classification/latest.json")
        return json.loads(obj["Body"].read()).get("data", {})
    except Exception:
        return {}


def backfill_from_factor_profiles(
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    pillar_weights: dict[str, float] | None = None,
) -> dict:
    """One-time bootstrap: recompute v2 attractiveness for every retained
    ``factors/profiles/{date}`` and (upsert-)write the history parquet. Returns
    a summary ``{dates, rows}``. Fails LOUD on a broken store (operator-run)."""
    s3 = _client(s3_client)
    b = _bucket(bucket)
    if pillar_weights is None:
        pillar_weights = _load_pillar_weights(bucket, s3_client)
    classification = _read_classification(bucket, s3_client)
    dates = _list_profile_dates(s3, b)
    if not dates:
        raise RuntimeError(
            "attractiveness_history backfill: no factors/profiles/{date} dirs found "
            f"under s3://{b}/factors/profiles/ — nothing to backfill."
        )
    all_rows: list[dict] = []
    used: list[str] = []
    for d in dates:
        profiles = _read_profiles(s3, b, d)
        if not profiles:
            logger.warning("[attractiveness_history] no by_ticker.json for %s — skipped", d)
            continue
        all_rows.extend(build_history_rows(d, profiles, pillar_weights, classification))
        used.append(d)
    if not all_rows:
        raise RuntimeError("attractiveness_history backfill: profiles existed but yielded no rows.")
    total = append_history(all_rows, bucket=bucket, s3_client=s3_client)
    logger.info("[attractiveness_history] backfilled %d dates (%s … %s), %d total rows",
                len(used), used[0], used[-1], total)
    return {"dates": used, "rows": total}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Attractiveness history backfill")
    ap.add_argument("--backfill", action="store_true", help="rebuild history from factor-profile snapshots")
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    args = ap.parse_args()
    if args.backfill:
        summary = backfill_from_factor_profiles(bucket=args.bucket)
        print(json.dumps(summary, indent=2))
    else:
        ap.error("nothing to do — pass --backfill")
