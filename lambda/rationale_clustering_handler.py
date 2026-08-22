"""Lambda entry point — cross-week rationale clustering.

Triggered weekly after the Saturday SF eval pipeline completes. Reads
captured decision artifacts from the trailing 8 weeks, clusters
rationales per agent_id, persists per-agent analysis JSON, and emits
the ``agent_rationale_template_concentration`` CloudWatch metric.

Per ROADMAP P0 "Cross-week rationale clustering for agent-justification".

Event shape (all fields optional):

    {
      "end_time_iso": "2026-05-09T00:00:00Z",  # default = now UTC
      "window_days": 56,                        # default 8 weeks
      "dry_run": false                          # if true, skip emit + persist
    }

Returns:

    {
      "status": "OK" | "PARTIAL" | "ERROR",
      "summary": <rationale_clustering.compute_and_emit result>
    }
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import UTC, datetime

# Repo root on sys.path so ``from evals.rationale_clustering import ...``
# resolves under Lambda's task layout.
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
    "rationale_clustering",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-rationale-clustering",
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors the eval-judge
    + rolling-mean handlers — Lambda init phase 10s ceiling."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    _init_done = True


@monitor_handler
def handler(event, context):
    """Entry point. Runs the handler, then flushes cost telemetry.

    The `finally` is the whole point (alpha-engine-config-I7423).
    `krepis.cost_sink.S3JsonlCostSink` buffers to 200 records per
    `(date, callsite_id)` group and otherwise relies on an `atexit` hook —
    and **an AWS Lambda container is FROZEN between invocations, not exited,
    so `atexit` never runs.** A handler finishing below the threshold writes
    nothing at all, and the container may be reclaimed hours later without
    ever reaching interpreter shutdown.

    Measured 2026-08-15 on weekly-SF execution `watch-rerun-2026-08-15-2`:
    `AggregateCosts` reported `single-agent-quant` among `2 stage(s) ran and
    emitted no cost record ... Observed producers: (none)`. The env wiring was
    correct, the sink was constructed, the records were priced and accepted,
    and every one of them died in memory.

    Applied to EVERY handler in this directory rather than to the ones known
    to call an LLM today: `flush_default_sink` returns 0 when no sink is
    configured and never raises, so the uniform rule costs nothing and leaves
    no per-handler judgment call for the next producer to get wrong.
    """
    try:
        return _run(event, context)
    finally:
        try:
            from krepis.cost_sink import flush_default_sink
            _n = flush_default_sink()
            if _n:
                logger.info("cost sink flushed: %d object(s)", _n)
        except ImportError as exc:
            # Loud, not silent: the image's krepis pin predates the function
            # (floor is >=0.59.8). Cost records for this invocation are lost,
            # and AggregateCosts' fan-in coverage check will name this stage.
            logger.error("cost-sink flush unavailable — records lost: %s", exc)


def _run(event, context):
    """Compute + emit per-agent rationale-template concentration."""
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.now(UTC)
    _ensure_init()

    from evals.lambda_dry import dry_clustering_result, is_dry
    from evals.rationale_clustering import (
        DEFAULT_WINDOW_DAYS,
        compute_and_emit,
    )

    # ── Shell-run dry path ───────────────────────────────────────────
    # Boot + the evals.rationale_clustering import (above) ran for real
    # — that's the keystone's bootstrap smoke. Return BEFORE
    # compute_and_emit, which reads decision_artifacts/, clusters, and
    # (the documented gap) S3-persists _analysis/ JSON via
    # _persist_analysis regardless of the existing `dry_run` flag — that
    # flag only suppresses the CW metric. dry_run_llm short-circuits the
    # entire read+cluster+persist, no Anthropic call.
    if is_dry(event):
        logger.info(
            "[rationale_clustering_handler] dry_run_llm=True: shell-run "
            "no-op (no S3 read/persist, no CW emit)",
        )
        return dry_clustering_result()

    end_time_iso = event.get("end_time_iso")
    end_time = (
        datetime.fromisoformat(end_time_iso.replace("Z", "+00:00"))
        if end_time_iso else None
    )
    window_days = int(event.get("window_days", DEFAULT_WINDOW_DAYS))
    dry_run = bool(event.get("dry_run", False))
    # This handler's own event carries no `run_date` field — only
    # end_time_iso ($$.Execution.StartTime, SF-threaded verbatim). Its date
    # portion IS the execution's own un-normalized run_date (verified
    # against nousergon-data/infrastructure/step_function.json —
    # `RationaleClustering`'s Payload is `end_time_iso.$: "$$.Execution.
    # StartTime"`, the same source InitializeInput derives $.run_date from),
    # not a proxy for it. alpha-engine-config-I8155: `_started` (this
    # handler's OWN invocation time, via `datetime.now(UTC)`) must never
    # substitute for a genuinely-absent end_time_iso — that was exactly the
    # forbidden fabrication class. When end_time_iso is absent there is no
    # execution identity to attribute to; the assertion is skipped below.
    _execution_run_date = end_time.date().isoformat() if end_time else None

    logger.info(
        "[rationale_clustering_handler] start end_time_iso=%s "
        "window_days=%d dry_run=%s",
        end_time_iso or "(now UTC)", window_days, dry_run,
    )

    try:
        summary = compute_and_emit(
            end_time=end_time,
            window_days=window_days,
            emit_metrics=not dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[rationale_clustering_handler] computation failed hard")
        return {"status": "ERROR", "error": str(exc)}

    has_failures = bool(summary["load_failures"]) or bool(summary["cluster_failures"])
    status = "PARTIAL" if has_failures else "OK"

    logger.info(
        "[rationale_clustering_handler] done status=%s agents=%d "
        "skipped_thin=%d load_failures=%d cluster_failures=%d",
        status,
        summary["agents_analyzed"],
        len(summary["agents_skipped_thin_sample"]),
        len(summary["load_failures"]),
        len(summary["cluster_failures"]),
    )
    result = {"status": status, "summary": summary}

    # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    # §2.3a rescope): the assertion lives in the stage's own handler,
    # immediately before it returns, rather than a separate end-of-run SF
    # state. OBSERVE MODE ONLY — never enables enforcement, never raises.
    if not _execution_run_date:
        # alpha-engine-config-I8155: never fabricate a date to satisfy the
        # (now-required) run_date argument.
        logger.error(
            "[rationale_clustering_handler] stage-coverage assertion "
            "SKIPPED for RationaleClustering: no end_time_iso on this "
            "event (execution identity absent)",
        )
        result["stage_coverage"] = {
            "stage": "RationaleClustering",
            "status": "UNMEASURED",
            "reason": "execution run_date absent from event (no end_time_iso)",
        }
    else:
        try:
            from krepis.stage_coverage import assert_stage_coverage

            result["stage_coverage"] = assert_stage_coverage(
                "RationaleClustering", run_date=_execution_run_date,
                window_start=_started,
            )
        except ImportError as exc:
            # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
            # the handler's own outcome is unchanged (config-I7214).
            logger.error("stage-coverage assertion unavailable: %s", exc)
            result["stage_coverage"] = {
                "stage": "RationaleClustering",
                "status": "UNMEASURED",
                "reason": f"assertion unavailable: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 — never let the observer kill the stage it observes
            # alpha-engine-config-I8155: the krepis landing this arc makes
            # run_date a required, contract-enforced kwarg (TypeError on
            # omission, StageCoverageContractError on blank/None). Log
            # loudly and degrade to UNMEASURED rather than raising out of
            # the handler.
            logger.error(
                "[rationale_clustering_handler] stage-coverage assertion "
                "raised for RationaleClustering: %s: %s",
                type(exc).__name__, exc,
            )
            result["stage_coverage"] = {
                "stage": "RationaleClustering",
                "status": "UNMEASURED",
                "reason": f"assertion raised: {type(exc).__name__}: {exc}",
            }

    return result
