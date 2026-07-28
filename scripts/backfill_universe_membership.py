#!/usr/bin/env python3
"""One-time backfill of ``universe_membership/{date}/membership.json`` for the
cycles that ran BEFORE the producer existed (alpha-engine-config-I4818/I4819).

Why. The membership artifact is the substrate for the console's churn view. A
churn view with one data point is useless, and the history it needs already
exists — scattered across two artifacts that every past cycle happened to write:

  scanner cut    ``candidates/{date}/candidates.json::scanner_tickers``
  attractiveness ``scanner/universe/history/attractiveness_history.parquet``
                 (per-``as_of`` slice — chosen over the dated boards because the
                 parquet reaches back to 2026-05-21 while only five dated boards
                 survive, and because the boards' gate block is untrustworthy
                 for 2026-07-02 (I4820) — this script never reads that block)

Only dates present in BOTH sources can be reconstructed; the rest are reported
and skipped, never silently interpolated. Every artifact this script writes
carries ``backfilled_from`` so a consumer can tell a reconstruction from a
first-class live write.

The ``latest.json`` sidecar is deliberately NOT written — a backfill must never
overwrite the pointer the predictor resolves its live universe from.

Usage::

    python scripts/backfill_universe_membership.py --dry-run    # report only
    python scripts/backfill_universe_membership.py              # write
    python scripts/backfill_universe_membership.py --overwrite   # re-do existing
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3  # noqa: E402

from scoring.universe_membership import (  # noqa: E402
    UniverseMembershipError,
    build_universe_membership,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_universe_membership")

_BUCKET = os.environ.get("S3_BUCKET", "alpha-engine-research")
_HISTORY_KEY = "scanner/universe/history/attractiveness_history.parquet"
_BACKFILL_SOURCE = (
    "candidates/{date}/candidates.json::scanner_tickers + scanner/universe/history/attractiveness_history.parquet"
)


def _candidate_dates(s3) -> list[str]:
    """Every date prefix under ``candidates/``."""
    dates: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_BUCKET, Prefix="candidates/", Delimiter="/"):
        for prefix in page.get("CommonPrefixes") or []:
            dates.append(prefix["Prefix"].split("/")[1])
    return sorted(dates)


def _scanner_tickers(s3, date: str) -> list[str] | None:
    try:
        obj = s3.get_object(Bucket=_BUCKET, Key=f"candidates/{date}/candidates.json")
        return json.loads(obj["Body"].read())["scanner_tickers"]
    except Exception as exc:  # noqa: BLE001 — reported, then skipped
        logger.warning("  %s: candidates.json unreadable (%s)", date, exc)
        return None


def _attractiveness_by_date(s3) -> dict[str, dict[str, float]]:
    """``{as_of: {ticker: attractiveness_score}}`` from the history parquet."""
    import pandas as pd

    obj = s3.get_object(Bucket=_BUCKET, Key=_HISTORY_KEY)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
    df = df[df["attractiveness_score"].notna()]
    return {
        str(as_of): dict(
            zip(
                slice_["ticker"].astype(str),
                slice_["attractiveness_score"].astype(float),
                strict=True,
            )
        )
        for as_of, slice_ in df.groupby("as_of")
    }


def _exists(s3, key: str) -> bool:
    """Whether ``key`` is already present. Absence is the normal case here, so
    it is a return value rather than an exception path."""
    try:
        s3.head_object(Bucket=_BUCKET, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001 — absent (404) is expected
        logger.debug("head_object(%s): %s", key, exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    parser.add_argument(
        "--overwrite", action="store_true", help="rewrite dates that already have a membership artifact"
    )
    args = parser.parse_args()

    s3 = boto3.client("s3")
    attractiveness = _attractiveness_by_date(s3)
    logger.info("attractiveness history: %d dated slices", len(attractiveness))

    candidate_dates = _candidate_dates(s3)
    logger.info("candidate artifacts: %d dates", len(candidate_dates))

    reconstructable = [d for d in candidate_dates if d in attractiveness]
    skipped = [d for d in candidate_dates if d not in attractiveness]
    logger.info(
        "reconstructable: %d — skipped (no attractiveness slice): %s",
        len(reconstructable),
        skipped or "none",
    )

    written, existing, failed = 0, 0, 0
    for date in reconstructable:
        key = f"universe_membership/{date}/membership.json"
        if not args.overwrite and _exists(s3, key):
            logger.info("  %s: already present — skipping", date)
            existing += 1
            continue

        tickers = _scanner_tickers(s3, date)
        if not tickers:
            failed += 1
            continue
        try:
            membership = build_universe_membership(
                date,
                tickers,
                attractiveness[date],
                backfilled_from=_BACKFILL_SOURCE.format(date=date),
            )
        except UniverseMembershipError as exc:
            logger.warning("  %s: not reconstructable (%s)", date, exc)
            failed += 1
            continue

        cuts = {n: c["size"] for n, c in membership["cuts"].items()}
        if args.dry_run:
            logger.info("  %s: WOULD write %s over %d ranked names", date, cuts, membership["universe_count"])
            continue

        # Dated key ONLY — never the latest.json sidecar (see module docstring).
        s3.put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=json.dumps(membership, separators=(",", ":"), default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("  %s: wrote %s over %d ranked names", date, cuts, membership["universe_count"])
        written += 1

    logger.info(
        "done — written=%d already-present=%d failed=%d skipped=%d%s",
        written,
        existing,
        failed,
        len(skipped),
        " (DRY RUN — nothing written)" if args.dry_run else "",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
