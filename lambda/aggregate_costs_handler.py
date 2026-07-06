"""Lambda entry point — daily cost aggregation.

Triggered by Saturday SF after Research completes. Reads per-call JSONL
files from ``decision_artifacts/_cost_raw/{date}/``, writes the daily
parquet at ``decision_artifacts/_cost/{date}/cost.parquet``, and emits
per-agent_id CloudWatch metrics.

Per ROADMAP L1146 (SF-wire ``aggregate_costs.py`` CLI). The script was
manual-trigger-only since PR #74 shipped 2026-05-01; this handler is the
institutional path that closes the manual surface. Shared image with
``handler.py`` + the eval-judge / rationale-clustering Lambdas — CMD
override sets entry point at deploy time.

Event shape (all fields optional except ``date``):

    {
      "date": "2026-05-25",            # ISO YYYY-MM-DD (required)
      "bucket": "alpha-engine-research", # default RESEARCH_BUCKET env / fallback
      "dry_run_llm": true,              # shell-run dry path — early return
    }

Returns one of:

    {"status": "OK", "summary": {...}}                — aggregated + parquet written
    {"status": "SKIPPED", "reason": "no_cost_raw_for_date", "date": "..."}
                                                       — no JSONL partitions for the date
    {"status": "ERROR", "error": "<msg>"}             — exception caught hard

The ``SKIPPED`` status mirrors data #295's pattern (deploy.sh canary
accepts both ``OK`` and ``SKIPPED``) and the L3277 audit's contract —
legitimate upstream no-op MUST NOT trigger rollback.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date as date_type

# Repo root on sys.path so ``from scripts.aggregate_costs import ...``
# resolves under Lambda's task layout (mirrors rationale_clustering /
# eval_rolling_mean handlers).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch
_install_ls_patch()

from nousergon_lib.logging import monitor_handler, setup_logging
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "aggregate_costs",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors the other
    shared-image handlers — Lambda init phase 10s ceiling."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


@monitor_handler
def handler(event, context):
    """Aggregate per-call JSONL cost files into the daily parquet."""
    _ensure_init()

    import boto3
    from evals.lambda_dry import is_dry
    from scripts.aggregate_costs import aggregate_day

    # Shell-run dry path — boot + imports above already exercised the
    # bootstrap smoke. Return BEFORE aggregate_day (which reads S3 +
    # writes parquet + emits CW). dry_run_llm short-circuits everything
    # for the Friday-Preflight shell run that doesn't actually need to
    # produce a parquet.
    if is_dry(event):
        logger.info(
            "[aggregate_costs_handler] dry_run_llm=True: shell-run "
            "no-op (no S3 read/write, no CW emit)",
        )
        return {"status": "OK", "dry_run": True}

    date_str = event.get("date")
    if not date_str:
        logger.error(
            "[aggregate_costs_handler] event missing required 'date' field"
        )
        return {
            "status": "ERROR",
            "error": "event missing required 'date' field (ISO YYYY-MM-DD)",
        }

    try:
        target_date = date_type.fromisoformat(date_str)
    except ValueError as exc:
        logger.error(
            "[aggregate_costs_handler] invalid date %r: %s", date_str, exc,
        )
        return {
            "status": "ERROR",
            "error": f"invalid date {date_str!r}: {exc}",
        }

    bucket = event.get("bucket", _DEFAULT_BUCKET)

    logger.info(
        "[aggregate_costs_handler] start date=%s bucket=%s",
        date_str, bucket,
    )

    s3_client = boto3.client("s3")
    cw_client = boto3.client("cloudwatch")

    try:
        summary = aggregate_day(
            s3_client=s3_client,
            bucket=bucket,
            target_date=target_date,
            cw_client=cw_client,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[aggregate_costs_handler] aggregation failed hard"
        )
        return {"status": "ERROR", "error": str(exc)}

    if summary is None:
        # Legitimate upstream no-op — no JSONL partitions emitted for
        # this date (e.g. Saturday SF ran with cost-telemetry kill
        # switch on, or a recovery SF that bypassed Research). Per
        # [[feedback_no_silent_fails]] the no-op is loudly visible
        # (WARN-log + named SKIPPED status), but does NOT raise — the
        # consumer / canary must accept SKIPPED as pass.
        logger.info(
            "[aggregate_costs_handler] no _cost_raw partitions for %s — "
            "skipping parquet write (no error)",
            date_str,
        )
        return {
            "status": "SKIPPED",
            "reason": "no_cost_raw_for_date",
            "date": date_str,
        }

    logger.info(
        "[aggregate_costs_handler] done rows_in=%d total_usd=%.4f output_key=%s",
        summary.get("rows_in", 0),
        summary.get("total_cost_usd", 0.0),
        summary.get("output_key", ""),
    )
    return {"status": "OK", "summary": summary, "date": date_str}
