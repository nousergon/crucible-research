"""Lambda entry point — LLM-as-judge batch PROCESS phase.

Third of the three-Lambda chain (Submit → Poll → Process). Invoked
once the SF Choice loop confirms ``processing_status='ended'``.
Streams the completed batch's results, parses each into the existing
``RubricEvalArtifact`` schema, persists to S3, emits CW metrics, and
runs the small synchronous Sonnet escalation tail for any Haiku
result that flagged a borderline dimension.

Event shape:

    {
      "batch_id": "msgbatch_..." | "empty-{date}",
      "plan_s3_key": "decision_artifacts/_eval_batch_plans/{date}/{batch_id}.json"
    }

Returns:

    {
      "status": "OK" | "PARTIAL" | "ERROR",
      "summary": <process_batch_results return dict>
    }

``OK`` = no failures and full coverage. ``PARTIAL`` = at least one batch
result failed, OR a phase stopped on the deadline budget and the pass
covered less of the corpus than it was asked to. ``ERROR`` = the run
itself blew up. Mirrors the legacy single-Lambda contract so the Saturday
SF + dashboard result inspectors keep working unchanged.

**alpha-engine-config-I6920 class (deadline discipline).** This function
was killed at its 900s wall in 9 of 28 observed real invocations,
every one of them inside ``process_batch_results``'s parse-retry tail.
Killed at the wall means no manifest, no summary, no cause — the SF sees
``States.Timeout`` and routes to ``MarkEvalJudgeDegraded`` knowing
nothing about what was covered. The orchestrator now consumes a stated
fraction of the invocation's own clock, stops each loop when the next
item no longer fits, and returns a PARTIAL envelope carrying
``complete`` / ``budget_stopped`` / ``n_skipped_for_budget`` — hoisted to
the TOP level of the return so a Step Functions ``Choice`` can read it
without digging into ``$.summary``.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import UTC
from datetime import datetime as _utcnow_cls

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch

_install_ls_patch()

# Imported after the sys.path.insert above — this Lambda entrypoint isn't
# on sys.path until that line runs (mirrors lambda/handler.py's pattern).
from nousergon_lib.logging import monitor_handler, setup_logging  # noqa: E402

_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "eval_judge_process",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=[],
    flow_name="research-eval-judge-process",
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    _init_done = True


def _remaining_seconds(context):
    """Zero-arg callable giving the seconds left in this invocation.

    ``None`` when there is no Lambda context (local runs, tests), which
    the orchestrator treats as "no deadline" — identical behaviour to
    before alpha-engine-config-I6920.

    NOT decorated with ``@monitor_handler``: that decorator belongs on the
    entry point, and putting it on a helper both leaves the real handler
    unwrapped and misreports the helper's frame as the handler's. (The
    concordance Lambda in crucible-backtester has exactly that defect
    today — see the PR body.)
    """
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(getter):
        return None
    return lambda: getter() / 1000.0


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
    _started = _utcnow_cls.now(UTC)
    _ensure_init()

    from evals.lambda_dry import dry_process_result, is_dry
    from evals.orchestrator import process_batch_results

    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    batch_id = event.get("batch_id")
    plan_s3_key = event.get("plan_s3_key")

    # ── Shell-run dry path ───────────────────────────────────────────
    # Boot + import ran for real. Submit threaded the dry sentinel
    # batch_id; return BEFORE process_batch_results (S3 plan get_object,
    # Anthropic results stream, per-artifact S3 persist, CW emit).
    if is_dry(event):
        logger.info(
            "[eval_judge_process_handler] dry_run_llm sentinel: shell-run "
            "no-op (no S3 plan read, no Anthropic stream, no persist) "
            "batch_id=%s", batch_id,
        )
        return dry_process_result(batch_id)

    if not batch_id or not plan_s3_key:
        return {
            "status": "ERROR",
            "error": (
                f"missing batch_id={batch_id!r} or plan_s3_key={plan_s3_key!r}"
            ),
        }

    logger.info(
        "[eval_judge_process_handler] start batch_id=%s plan_key=%s",
        batch_id, plan_s3_key,
    )

    _process_error: str | None = None
    try:
        summary = process_batch_results(
            batch_id=batch_id,
            plan_s3_key=plan_s3_key,
            bucket=bucket,
            # alpha-engine-config-I9263: no provider SDK client at the call
            # site. On the `sync-` rung `process_batch_results` judges every
            # plan entry through the router-addressed `evaluate_artifact` and
            # never touches a batch stream.
            batch_client=None,
            remaining_s=_remaining_seconds(context),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[eval_judge_process_handler] process failed hard")
        _process_error = str(exc)
        summary = None

    # Build the _eval_by_capture index AFTER every eval persist, regardless
    # of whether process_batch_results succeeded or raised partway through.
    # config#4776: the previous arrangement placed build_manifests AFTER the
    # try/except — meaning a failure in process_batch_results (e.g. the
    # Sonnet-escalation tail's OpenRouter transport raising RuntimeError
    # after the Haiku batch results were already persisted) skipped manifest
    # maintenance entirely. The manifests went stale, dedup no-oped on
    # every retry, and the same 295-artifact corpus was re-judged ~10 times
    # (~28M input tokens wasted).
    #
    # build_manifests scans what was ACTUALLY persisted on S3, so it
    # correctly indexes whatever evals made it to disk before the failure.
    # Idempotent: any later run self-heals the index for dates it touches.
    # config#1579 P2: a failure here is WARN (the dedup then no-ops for the
    # affected dates — duplicate evals possible, never a silent skip).
    manifest_dates: list[str] = []
    try:
        from datetime import datetime

        from evals.eval_manifest import build_manifests

        written = build_manifests(
            s3_client=__import__("boto3").client("s3"),
            bucket=bucket,
            judge_run_dates=[datetime.now(UTC).date().isoformat()],
        )
        manifest_dates = sorted(written)
    except Exception:  # noqa: BLE001 — index maintenance; recorded below
        logger.warning(
            "[eval_judge_process_handler] _eval_by_capture manifest build "
            "failed — dedup will no-op for this batch's capture dates "
            "(duplicate evals possible, never skips)", exc_info=True,
        )

    if _process_error is not None:
        return {
            "status": "ERROR",
            "batch_id": batch_id,
            "error": _process_error,
            "manifest_capture_dates": manifest_dates,
            # An ERROR covered an unknown fraction of the corpus. Saying
            # anything else here would let a crashed pass read as complete
            # on the same field a truncated one reports honestly.
            "complete": False,
            "budget_stopped": False,
        }

    summary["manifest_capture_dates"] = manifest_dates

    # I6920: a pass that stopped on budget covered less of the corpus than
    # it was asked to. That is partial signal, and reporting OK would make
    # a truncated sweep read as a full one.
    budget_stopped = bool(summary.get("budget_stopped"))
    if budget_stopped:
        logger.warning(
            "[eval_judge_process_handler] stopped on budget in %s: %s "
            "items not processed",
            summary.get("budget_stopped_phases"),
            summary.get("n_skipped_for_budget"),
        )
    status = "PARTIAL" if (summary["failed"] or budget_stopped) else "OK"
    logger.info(
        "[eval_judge_process_handler] done status=%s haiku=%d sonnet=%d "
        "skipped_unmapped=%d skipped_empty_input=%d failed=%d complete=%s",
        status,
        summary["haiku_evaluated"],
        summary["sonnet_evaluated"],
        summary["skipped_unmapped"],
        summary["skipped_empty_input"],
        len(summary["failed"]),
        summary.get("complete", True),
    )
    result = {
        "status": status,
        # Hoisted out of `summary` so a Step Functions Choice can branch on
        # it — sf-pipeline-policy.md §2.3a requires the verdict to reach the
        # machine, not only the operator reading the log.
        "complete": bool(summary.get("complete", True)),
        "budget_stopped": budget_stopped,
        "summary": summary,
    }

    # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    # §2.3a rescope). This handler's own event carries no `run_date` field —
    # only batch_id/plan_s3_key — so the execution run_date is recovered
    # from plan_s3_key's own {date} path segment
    # ("decision_artifacts/_eval_batch_plans/{date}/{batch_id}.json",
    # written verbatim by eval_judge_submit_handler from the SAME execution's
    # `date` value — alpha-engine-config-I8155). OBSERVE MODE ONLY — never
    # enables enforcement, never raises out of this handler.
    _execution_run_date = None
    if plan_s3_key:
        _plan_key_parts = plan_s3_key.split("/")
        _execution_run_date = (
            _plan_key_parts[2] if len(_plan_key_parts) > 2 else None
        )

    if not _execution_run_date:
        # alpha-engine-config-I8155: never fabricate a date to satisfy the
        # (now-required) run_date argument.
        logger.error(
            "[eval_judge_process_handler] stage-coverage assertion SKIPPED "
            "for EvalJudgeProcess: could not recover execution run_date "
            "from plan_s3_key=%r (execution identity absent)", plan_s3_key,
        )
        result["stage_coverage"] = {
            "stage": "EvalJudgeProcess",
            "status": "UNMEASURED",
            "reason": f"execution run_date absent from plan_s3_key={plan_s3_key!r}",
        }
    else:
        try:
            from krepis.stage_coverage import assert_stage_coverage

            result["stage_coverage"] = assert_stage_coverage(
                "EvalJudgeProcess", run_date=_execution_run_date,
                window_start=_started,
            )
        except ImportError as exc:
            # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
            # the handler's own outcome is unchanged (config-I7214).
            logger.error("stage-coverage assertion unavailable: %s", exc)
            result["stage_coverage"] = {
                "stage": "EvalJudgeProcess",
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
                "[eval_judge_process_handler] stage-coverage assertion "
                "raised for EvalJudgeProcess: %s: %s",
                type(exc).__name__, exc,
            )
            result["stage_coverage"] = {
                "stage": "EvalJudgeProcess",
                "status": "UNMEASURED",
                "reason": f"assertion raised: {type(exc).__name__}: {exc}",
            }

    return result
