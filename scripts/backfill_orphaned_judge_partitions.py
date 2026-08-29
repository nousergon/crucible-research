"""
backfill_orphaned_judge_partitions.py — re-judge capture partitions the
pre-fix weekly window never reached (alpha-engine-config-I9331).

Root cause: the weekly judge (`EvalJudgeSubmitWeekly` /
`EvalJudgeSubmitFirstSaturday`) built its capture window by expanding
`capture_lookback_days` straight off the raw execution anchor. Think
Tank writes day D's captures on **D+1 at ~14:35 UTC**; the weekly SF
enters the judge stage at ~05:11 UTC on its own anchor day — hours
before the newest trading day's captures land. Fixed in
`evals/orchestrator.py::compute_judge_window_dates`
(`CAPTURE_WRITE_SETTLE_DAYS`), which the Lambda now uses instead of
expanding off the raw anchor.

This script finds every capture partition since Think Tank's daily
cadence started that STILL has zero corresponding entries in
`decision_artifacts/_eval_by_capture/{date}/manifest.json` — i.e. was
never judged by any run, past or present — and re-submits exactly
those dates through the same `build_batch_plan` / `submit_batch` path
production uses. `load_already_judged_keys` makes this safely
idempotent: a partition judged since this script was written is
skipped automatically, so re-running it is harmless.

This is a manual one-off PRODUCTION write. Per fleet policy it MUST
run IN-REGION (an EC2 box in the bucket's region), never from a
laptop — the read/list/judge round-trip against the research bucket
dominates on cross-region latency, and a batch judge run makes live
Anthropic-billed calls. Verify locally with --dry-run only.

Usage:
    # local / laptop — read-only, no writes, no LLM calls, no cost
    python scripts/backfill_orphaned_judge_partitions.py --dry-run \\
        --since 2026-07-01

    # in-region (EC2, over SSM) — the real backfill
    python scripts/backfill_orphaned_judge_partitions.py --since 2026-07-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as _date
from datetime import timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import boto3  # noqa: E402
from krepis.trading_calendar import is_trading_day  # noqa: E402

from evals.orchestrator import (  # noqa: E402
    DEFAULT_HAIKU_MODEL,
    DEFAULT_SONNET_MODEL,
    _persist_client_side_skips,
    build_batch_plan,
    list_capture_keys,
    load_already_judged_keys,
    submit_batch,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"


def _trading_days_between(since: _date, until: _date) -> list[str]:
    out: list[str] = []
    cur = since
    while cur <= until:
        if is_trading_day(cur):
            out.append(str(cur))
        cur += timedelta(days=1)
    return out


def find_orphaned_dates(
    s3: Any, *, bucket: str, since: str, until: str,
) -> list[str]:
    """Trading days in [since, until] that have ≥1 capture and ZERO
    entries in that date's ``_eval_by_capture`` manifest — i.e. no
    run, past or present, has ever judged anything from that day.

    A day with captures but a PARTIAL manifest (some artifacts judged,
    some not) is not "orphaned" by this definition — it re-enters the
    normal weekly plan on its own via the unjudged-keys diff, which
    ``build_batch_plan`` already performs. This script targets the
    stronger, unambiguous case: a day the judge chain never touched.
    """
    y0, m0, d0 = (int(x) for x in since.split("-"))
    y1, m1, d1 = (int(x) for x in until.split("-"))
    candidates = _trading_days_between(_date(y0, m0, d0), _date(y1, m1, d1))

    orphaned: list[str] = []
    for d in candidates:
        keys = list_capture_keys(s3, date=d, bucket=bucket)
        if not keys:
            continue  # no captures that day — nothing to judge, not orphaned
        judged = load_already_judged_keys(s3, dates=[d], bucket=bucket)
        unjudged = [k for k in keys if k not in judged]
        if unjudged and len(unjudged) == len(keys):
            orphaned.append(d)
            logger.info(
                "[backfill] %s: %d captures, 0 judged — ORPHANED", d, len(keys),
            )
        elif unjudged:
            logger.info(
                "[backfill] %s: %d/%d captures unjudged — PARTIAL, not "
                "targeted by this script (normal weekly plan will pick it up)",
                d, len(unjudged), len(keys),
            )
    return orphaned


def backfill(
    *, bucket: str, since: str, until: str, dry_run: bool,
    haiku_model: str = DEFAULT_HAIKU_MODEL,
    sonnet_model: str = DEFAULT_SONNET_MODEL,
) -> dict:
    s3 = boto3.client("s3")
    orphaned = find_orphaned_dates(s3, bucket=bucket, since=since, until=until)

    if not orphaned:
        logger.info("[backfill] no orphaned partitions found in [%s, %s]", since, until)
        return {"orphaned_dates": [], "results": []}

    logger.info(
        "[backfill] %d orphaned date(s): %s%s",
        len(orphaned), orphaned, " (DRY RUN — no writes, no LLM calls)" if dry_run else "",
    )

    results: list[dict] = []
    for d in orphaned:
        plan = build_batch_plan(
            date=d, bucket=bucket,
            haiku_model=haiku_model, sonnet_model=sonnet_model,
            force_sonnet_pass=False, s3_client=s3,
        )
        entry = {
            "date": d,
            "capture_keys_total": plan["capture_keys_total"],
            "skipped_unmapped": plan["skipped_unmapped"],
            "request_count": len(plan["requests"]),
        }
        if dry_run:
            entry["action"] = "would_submit"
            results.append(entry)
            continue

        skip_count, degenerate_skip_count, _, skip_failed = _persist_client_side_skips(
            plan, s3=s3, bucket=bucket,
        )
        submit_result = submit_batch(plan, s3_client=s3)
        entry.update({
            "action": "submitted",
            "batch_id": submit_result["batch_id"],
            "processing_status": submit_result["processing_status"],
            "skipped_empty_input_persisted": skip_count,
            "skipped_degenerate_input_persisted": degenerate_skip_count,
            "skip_failed": skip_failed,
        })
        results.append(entry)
        logger.info("[backfill] %s: submitted batch_id=%s", d, submit_result["batch_id"])

    return {"orphaned_dates": orphaned, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--since", required=True,
        help="First calendar date (YYYY-MM-DD) to scan — Think Tank's "
             "daily-cadence start date, or an earlier bound if unsure "
             "(non-trading days and days with no captures are no-ops).",
    )
    parser.add_argument(
        "--until", default=str(_date.today() - timedelta(days=1)),
        help="Last calendar date (YYYY-MM-DD) to scan, default yesterday.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List orphaned dates and the plan each would submit — no S3 "
             "writes, no Anthropic/router calls.",
    )
    args = parser.parse_args()

    summary = backfill(
        bucket=args.bucket, since=args.since, until=args.until,
        dry_run=args.dry_run,
    )
    logger.info("[backfill] done: %s", summary)


if __name__ == "__main__":
    main()
