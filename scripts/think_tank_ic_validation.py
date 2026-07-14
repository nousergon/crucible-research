"""Operator script: run the Think Tank rating -> realized 21d alpha IC
validation (config#2467) and print/persist the result.

Usage::

    python scripts/think_tank_ic_validation.py
    python scripts/think_tank_ic_validation.py --s3-key thinktank/outcome_ic/2026-07-14.json

Pulls every distinct (ticker, thesis_trading_day) rating event from all
dated ``thinktank/ratings/{trading_day}.json`` snapshots in the
``alpha-engine-research`` bucket, joins each to its realized 21-trading-day
forward log-domain SPY-relative alpha via ``universe_returns`` (see
``thinktank/outcome_ic.py`` for why this bypasses ``evals.outcome_store``),
and reports the date-clustered Spearman IC.

OBSERVATION ONLY: this script computes a diagnostic block; it does not gate
or write to any config/production path. Optional ``--s3-key`` persists the
JSON result under the ``alpha-engine-research`` bucket (same pattern as
``scripts/run_judge_cross_validation.py --s3-key``) as a durable record.

This script is operator-driven — not on any cron.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Repo root on sys.path so ``from thinktank.outcome_ic import ...`` resolves
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thinktank.outcome_ic import (  # noqa: E402
    RATINGS_PREFIX,
    build_think_tank_outcome_ic_block,
)

BUCKET = "alpha-engine-research"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratings-prefix", type=str, default=RATINGS_PREFIX,
        help=f"S3 prefix for dated ratings-board snapshots (default: {RATINGS_PREFIX!r}).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to write the JSON result locally (default: stdout only).",
    )
    parser.add_argument(
        "--s3-key", type=str, default=None,
        help="Optional S3 key under the alpha-engine-research bucket to persist "
             "the result JSON (eg 'thinktank/outcome_ic/2026-07-14.json').",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable INFO logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import boto3

    s3 = boto3.client("s3")
    block = build_think_tank_outcome_ic_block(
        s3, BUCKET, ratings_prefix=args.ratings_prefix,
    )

    payload = json.dumps(block, indent=2, sort_keys=True)
    print(payload)

    if args.out:
        args.out.write_text(payload + "\n")
        print(f"Wrote result -> {args.out}", file=sys.stderr)

    if args.s3_key:
        s3.put_object(
            Bucket=BUCKET,
            Key=args.s3_key,
            Body=payload.encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Uploaded -> s3://{BUCKET}/{args.s3_key}", file=sys.stderr)

    if block["status"] == "insufficient":
        print(
            f"NOTE: status=insufficient (n_ratings_total="
            f"{block['n_ratings_total']}, n_unresolved={block['n_unresolved']}, "
            f"n_rating_dates={block['overall']['n_rating_dates']}) — no "
            "resolved 21d-forward outcomes yet for the rated cohort, or too "
            "few contributing dates. This is expected cohort maturation, not "
            "a bug; re-run once more ratings have matured past the 21-trading-"
            "day forward window.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
