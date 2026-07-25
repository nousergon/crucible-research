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
    _ensure_init()

    import anthropic

    from config import ANTHROPIC_API_KEY
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
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        poll_result = poll_batch(batch_id=batch_id, anthropic_client=client)
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

    return {
        "batch_id": batch_id,
        "processing_status": poll_result["processing_status"],
        "request_counts": poll_result["request_counts"],
        "ended_at": poll_result.get("ended_at"),
        "elapsed_seconds": elapsed,
        "exceeded_max_wait": exceeded_max_wait,
    }
