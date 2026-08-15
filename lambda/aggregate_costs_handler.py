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
      "date": "2026-05-25",            # ISO YYYY-MM-DD (required) — the END
                                       #   of the aggregation window
      "lookback_days": 8,              # optional; default DEFAULT_LOOKBACK_DAYS
      "bucket": "alpha-engine-research", # default RESEARCH_BUCKET env / fallback
      "dry_run_llm": true,              # shell-run dry path — early return
    }

``date`` names the window END, not the only date aggregated (config-I7407).
Capture is daily and this Lambda is invoked weekly with the SF's ``run_date``,
so a single-date aggregation left every intervening day's rows unaggregated —
measured 2026-08-15, two days of real cost rows and a DEGRADED weekly run.

Returns one of:

    {"status": "OK", "summary": {...}}                — aggregated + parquet written
    {"status": "SKIPPED", "reason": "no_cost_raw_in_window", "date": "..."}
                                                       — no JSONL partitions anywhere in the window
    {"status": "ERROR", "error": "<msg>"}             — exception caught hard

The ``SKIPPED`` status mirrors data #295's pattern (deploy.sh canary
accepts both ``OK`` and ``SKIPPED``) and the L3277 audit's contract —
legitimate upstream no-op MUST NOT trigger rollback.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import UTC, datetime
from datetime import date as date_type

# Repo root on sys.path so ``from scripts.aggregate_costs import ...``
# resolves under Lambda's task layout (mirrors rationale_clustering /
# eval_rolling_mean handlers).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch

_install_ls_patch()

# Imported after the sys.path.insert above — this Lambda entrypoint isn't
# on sys.path until that line runs (mirrors lambda/handler.py's pattern).
from nousergon_lib.logging import monitor_handler, setup_logging  # noqa: E402

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
    flow_name="research-aggregate-costs",
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
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    _init_done = True


def _attach_stage_coverage(result: dict, *, run_date: str, window_start) -> None:
    """Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    §2.3a rescope): the assertion lives in the stage's own handler,
    immediately before it returns, rather than a separate end-of-run SF
    state. OBSERVE MODE ONLY — never enables enforcement, never raises.
    Shared by both this handler's terminal returns (OK and the legitimate
    SKIPPED no-op) since both are real completions, not failures."""
    try:
        from krepis.stage_coverage import assert_stage_coverage

        result["stage_coverage"] = assert_stage_coverage(
            "AggregateCosts", run_date=run_date, window_start=window_start,
        )
    except ImportError as exc:
        # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
        # the handler's own outcome is unchanged (config-I7214).
        logger.error("stage-coverage assertion unavailable: %s", exc)


@monitor_handler
def handler(event, context):
    """Aggregate per-call JSONL cost files into the daily parquet."""
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.now(UTC)
    _ensure_init()

    import boto3

    from evals.lambda_dry import is_dry
    from scripts.aggregate_costs import (
        DEFAULT_LOOKBACK_DAYS,
        aggregate_window,
    )

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

    # config-I7407: aggregate the WINDOW ending at `date`, not that one date.
    # Capture is daily (krepis.cost_sink writes under the wall-clock date of
    # the call, so the Think Tank's daily runs land on dates no SF names)
    # while this Lambda was invoked with a single `$.run_date`. Measured
    # 2026-08-15: raw existed for 08-10/08-11/08-12, parquet stopped at 08-10,
    # and the Saturday run asked for 08-14 -- which had no raw at all, so the
    # stage produced nothing and reported SKIPPED. The `cost_telemetry`
    # transparency row then failed its 8-day freshness check, which is the
    # single [FAIL] behind that run's DEGRADED terminal.
    #
    # The window default matches that row's `max_age_days`, so the producer
    # covers exactly what the consumer asserts. Overridable per invocation for
    # a wider backfill.
    lookback = int(event.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
    try:
        result = aggregate_window(
            s3_client=s3_client,
            bucket=bucket,
            end_date=target_date,
            lookback_days=lookback,
            cw_client=cw_client,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[aggregate_costs_handler] aggregation failed hard"
        )
        return {"status": "ERROR", "error": str(exc)}

    # A window with nothing to do is the legitimate quiet case; a window that
    # aggregated at least one date is a success naming what it covered.
    summary = None if not result["aggregated"] else {
        "dates_aggregated": result["aggregated"],
        "dates_quiet": result["skipped"],
        "lookback_days": lookback,
    }

    if summary is None:
        # Legitimate upstream no-op — no JSONL partitions emitted ANYWHERE
        # in the window (e.g. cost-telemetry kill switch on, or a genuinely
        # quiet stretch). config-I7407 narrowed what this means: it used to
        # fire whenever the ONE named date was empty, which is the common
        # case rather than the exceptional one. Per
        # [[feedback_no_silent_fails]] the no-op is loudly visible
        # (WARN-log + named SKIPPED status), but does NOT raise — the
        # consumer / canary must accept SKIPPED as pass.
        logger.info(
            "[aggregate_costs_handler] no _cost_raw partitions in the %d-day "
            "window ending %s — skipping parquet write (no error)",
            lookback, date_str,
        )
        skipped_result = {
            "status": "SKIPPED",
            "reason": "no_cost_raw_in_window",
            "date": date_str,
            "lookback_days": lookback,
        }
        _attach_stage_coverage(skipped_result, run_date=date_str, window_start=_started)
        return skipped_result

    logger.info(
        "[aggregate_costs_handler] done dates_aggregated=%s quiet=%d "
        "lookback_days=%d",
        ",".join(summary["dates_aggregated"]),
        len(summary["dates_quiet"]),
        lookback,
    )
    result = {"status": "OK", "summary": summary, "date": date_str}
    _attach_stage_coverage(result, run_date=date_str, window_start=_started)
    return result
