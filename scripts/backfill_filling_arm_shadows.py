#!/usr/bin/env python3
"""Backfill the filling arms' shadow signals (alpha-engine-config-I9307).

WHY A BACKFILL IS POSSIBLE AT ALL — and why that matters more than it sounds.
The issue's own working assumption was that the champion's cohort would have to
be rebuilt forward from zero, leaving the slot unable to promote for ~21
sessions. It does not: both filling arms rank over DURABLE, DATED inputs.

  * ``scanner_predictor_direct`` — ``predictor/research_free_backfill/
    predictor_outcomes_research_free.parquet`` is a HISTORY, not a latest-cohort
    snapshot. Measured 2026-08-29: 2080 rows across 35 distinct
    ``prediction_date`` values, covering every recent weekly cohort date.
  * ``scanner_top20_predictor`` — ``universe_membership/{date}`` and
    ``predictor/predictions/{date}.json`` are both dated and retained.

So the arms' picks for a past date are RECONSTRUCTED, not invented: the same
rule over the same inputs the executor read on that date. That is the whole
difference between a backfill and a fabrication, and it only holds because
neither input is mutated in place.

WHAT THIS IS NOT. It does not touch ``signals/``, and it writes only to
``signals_shadow/``, which is observe-only and never read by live trading.
Re-running is idempotent: the same inputs produce the same document.

Usage::

    AWS_PROFILE=ne-admin python3 scripts/backfill_filling_arm_shadows.py --dry-run
    AWS_PROFILE=ne-admin python3 scripts/backfill_filling_arm_shadows.py --write
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import boto3

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from producers.filling_arms import (  # noqa: E402
    POOL_LOADERS,
    FillingShadowError,
    build_filling_shadow,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("backfill_filling_arm_shadows")

BUCKET = "alpha-engine-research"
SHADOW_KEY = "signals_shadow/{arm}/{date}/signals.json"


class _Manager:
    """Minimal stand-in exposing the (s3, bucket) pair the builders need."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = boto3.client("s3")


def cohort_dates(s3, bucket: str) -> list[str]:
    """Every date any arm has already written a shadow for.

    The backfill targets the EXISTING cohort rather than every date its inputs
    could serve: an arm scoring dates no other arm covers widens the board
    without widening the intersection, and the intersection is the only basis a
    promotion is taken on (champion-challenger-policy.md §4).
    """
    dates: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="signals_shadow/"):
        for obj in page.get("Contents", []) or []:
            parts = obj["Key"].split("/")
            if len(parts) > 2 and parts[2]:
                dates.add(parts[2])
    return sorted(dates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="actually PUT the objects")
    ap.add_argument("--dry-run", action="store_true", help="report only (default)")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--arm", action="append", choices=sorted(POOL_LOADERS))
    ap.add_argument("--overwrite", action="store_true", help="rewrite an existing shadow")
    args = ap.parse_args()
    if not args.write:
        logger.info("DRY RUN — pass --write to persist. Nothing will be written.\n")

    mgr = _Manager(args.bucket)
    arms = args.arm or sorted(POOL_LOADERS)
    dates = cohort_dates(mgr.s3, args.bucket)
    logger.info("cohort dates already on the board: %d (%s .. %s)\n",
                len(dates), dates[0] if dates else "-", dates[-1] if dates else "-")

    wrote = skipped = failed = 0
    for arm in arms:
        for date in dates:
            key = SHADOW_KEY.format(arm=arm, date=date)
            if not args.overwrite:
                try:
                    mgr.s3.head_object(Bucket=args.bucket, Key=key)
                    logger.info("skip   %s %s (already present)", date, arm)
                    skipped += 1
                    continue
                except Exception:
                    pass
            try:
                payload = build_filling_shadow(arm, date, mgr)
            except FillingShadowError as exc:
                # NOT swallowed: a date whose inputs are genuinely absent is a
                # legitimate gap in this arm's record and must stay a gap. What
                # would be wrong is writing an empty document for it, which is
                # exactly the artifact the whole issue is about.
                logger.warning("gap    %s %s: %s", date, arm, str(exc)[:160])
                failed += 1
                continue
            payload["run_date"] = f"backfill:{date}"
            body = json.dumps(payload, indent=2, default=str)
            if args.write:
                mgr.s3.put_object(Bucket=args.bucket, Key=key, Body=body.encode())
            logger.info(
                "%s %s %s: %d ENTER from %s (pool %d)",
                "WROTE " if args.write else "would ", date, arm,
                len(payload["signals"]), payload["arm_pool"]["pool_source"],
                payload["arm_pool"]["pool_size"],
            )
            wrote += 1

    logger.info("\n%s=%d skipped=%d gaps=%d",
                "wrote" if args.write else "would-write", wrote, skipped, failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
