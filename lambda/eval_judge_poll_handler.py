"""Lambda entry point — LLM-as-judge batch POLL phase.

Second of the three-Lambda chain (Submit → Poll → Process). Invoked
on each turn of the SF Wait→Choice loop to retrieve the batch's
current ``processing_status`` from Anthropic.

Event shape:

    {
      "batch_id": "msgbatch_..." | "empty-{date}",
      "submit_iso": "2026-05-09T22:30:00Z",   # for elapsed-time check
      "max_wait_seconds": 21600                # 6h cap; SF defaults to 21600
    }

Returns:

    {
      "batch_id": "...",
      "processing_status": "in_progress" | "ended" | "ended_empty",
      "request_counts": {...},
      "elapsed_seconds": int,
      "exceeded_max_wait": bool
    }

The SF Choice state branches on ``processing_status``:
- ``ended`` / ``ended_empty`` → route to EvalJudgeProcess
- ``exceeded_max_wait=true`` → fail-soft route to EvalRollingMean
- else → loop back to EvalJudgePollWait

Eval is observability — Anthropic API errors during poll are logged
and surfaced to the SF Catch for routing to EvalRollingMean. The
batch result remains retrievable for 29 days, so a transient poll
failure doesn't lose the run; it just loses the in-this-execution
processing.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.logging import monitor_handler, setup_logging

_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "eval_judge_poll",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=[],
    flow_name="research-eval-judge-poll",
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    global _init_done
    if _init_done:
        return
    _init_done = True


_DEFAULT_MAX_WAIT_SECONDS = 21600  # 6h


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
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.datetime.now(datetime.UTC)
    _ensure_init()

    from evals.lambda_dry import dry_poll_result, is_dry
    from evals.orchestrator import poll_batch

    batch_id = event.get("batch_id")

    # ── Shell-run dry path ───────────────────────────────────────────
    # Boot + import ran for real. Detect the dry sentinel threaded from
    # Submit (or the raw flag) and return a terminal `ended` WITHOUT any
    # anthropic.messages.batches.retrieve call. (Under the keystone the
    # SF skips Poll entirely via status=EMPTY → Process; this branch is
    # the defensive belt-and-braces if Poll is reached.)
    if is_dry(event):
        logger.info(
            "[eval_judge_poll_handler] dry_run_llm sentinel: shell-run "
            "no-op (no Anthropic poll) batch_id=%s", batch_id,
        )
        return dry_poll_result(batch_id)

    if not batch_id:
        return {
            "processing_status": "error",
            "error": "missing batch_id in event payload",
        }

    submit_iso = event.get("submit_iso")
    max_wait = int(event.get("max_wait_seconds", _DEFAULT_MAX_WAIT_SECONDS))
    elapsed = 0
    if submit_iso:
        try:
            submit_dt = datetime.datetime.fromisoformat(
                submit_iso.replace("Z", "+00:00")
            )
            elapsed = int(
                (datetime.datetime.now(datetime.UTC) - submit_dt)
                .total_seconds()
            )
        except (ValueError, TypeError):
            logger.warning(
                "[eval_judge_poll_handler] could not parse submit_iso=%r",
                submit_iso,
            )

    try:
        # alpha-engine-config-I9263: no provider SDK client at the call site.
        # A `sync-` or `empty-` batch id is terminal by construction and needs
        # no client at all; a real provider batch id would need a router-
        # resolved batch client, which `submit_batch` is the only thing that
        # can produce today.
        poll_result = poll_batch(batch_id=batch_id)
    except Exception:  # noqa: BLE001
        logger.exception(
            "[eval_judge_poll_handler] poll API failed for batch_id=%s",
            batch_id,
        )
        # Re-raise so SF Catch routes to EvalRollingMean rather than
        # silently looping with a bogus status.
        raise

    exceeded_max_wait = elapsed > max_wait

    logger.info(
        "[eval_judge_poll_handler] batch_id=%s status=%s elapsed=%ds "
        "max_wait=%ds exceeded=%s",
        batch_id, poll_result["processing_status"], elapsed, max_wait,
        exceeded_max_wait,
    )

    result = {
        "batch_id": batch_id,
        "processing_status": poll_result["processing_status"],
        "request_counts": poll_result["request_counts"],
        "ended_at": poll_result.get("ended_at"),
        "elapsed_seconds": elapsed,
        "exceeded_max_wait": exceeded_max_wait,
    }

    # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    # §2.3a rescope). EvalJudgePoll's own SF Payload carries no `run_date`
    # field (only batch_id/submit_iso/max_wait_seconds), but `submit_iso` IS
    # the execution's own identity: ComputeEvalCadence sets it directly from
    # `$$.Execution.StartTime` (alpha-engine-config-I8155 — verified against
    # nousergon-data/infrastructure/step_function.json), never normalized
    # against a trading-day calendar. `submit_iso[:10]` is therefore already
    # the un-normalized execution run_date, not a proxy for it.
    if submit_iso:
        _execution_run_date = submit_iso[:10]
        try:
            from krepis.stage_coverage import assert_stage_coverage

            result["stage_coverage"] = assert_stage_coverage(
                "EvalJudgePoll", run_date=_execution_run_date,
                window_start=_started,
            )
        except ImportError as exc:
            # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe
            # mode — the handler's own outcome is unchanged (config-I7214).
            logger.error("stage-coverage assertion unavailable: %s", exc)
            result["stage_coverage"] = {
                "stage": "EvalJudgePoll",
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
                "[eval_judge_poll_handler] stage-coverage assertion raised "
                "for EvalJudgePoll: %s: %s", type(exc).__name__, exc,
            )
            result["stage_coverage"] = {
                "stage": "EvalJudgePoll",
                "status": "UNMEASURED",
                "reason": f"assertion raised: {type(exc).__name__}: {exc}",
            }
    else:
        # alpha-engine-config-I8155: never fabricate a date to satisfy the
        # (now-required) run_date argument — no execution identity is
        # recoverable from this event at all.
        logger.error(
            "[eval_judge_poll_handler] stage-coverage assertion SKIPPED "
            "for EvalJudgePoll: no submit_iso on this event (execution "
            "identity absent)",
        )
        result["stage_coverage"] = {
            "stage": "EvalJudgePoll",
            "status": "UNMEASURED",
            "reason": "execution run_date absent from event (no submit_iso)",
        }

    return result
