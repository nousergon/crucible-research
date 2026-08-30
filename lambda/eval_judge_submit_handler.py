"""Lambda entry point — LLM-as-judge batch SUBMIT phase.

First of the three-Lambda chain (Submit → Poll → Process) introduced
on 2026-05-07 to migrate the eval-judge pipeline to the Anthropic
Message Batches API per ROADMAP P1 §1642. Replaces the legacy
single-Lambda ``eval_judge_handler.handler`` for the Saturday SF
path; the legacy handler remains in place for ad-hoc invocations,
``dry_run`` smoke, and the ``judge_only`` test track.

Event shape (all fields optional):

    {
      "date": "2026-05-09",          # YYYY-MM-DD; default = today UTC
      "force_sonnet_pass": false,    # SF passes True every 4th run
      "haiku_model": "claude-haiku-4-5",
      "sonnet_model": "claude-sonnet-4-6",
      "judge_only": false,           # isolated test-track outputs
      "extra_dates": ["2026-06-29"], # optional: additional capture-date
                                     # partitions to enumerate (daily
                                     # producers, e.g. thinktank)
      "agent_id_prefixes": ["thinktank_"],  # optional: only judge agent_ids
                                     # with these prefixes (family selection)
      "capture_lookback_days": 6     # optional: expand extra_dates to the N
                                     # days before date (weekly SF passes 6)
    }

Returns:

    {
      "status": "OK" | "EMPTY" | "ERROR",
      "batch_id": "msgbatch_..." | "empty-{date}",
      "plan_s3_key": "decision_artifacts/_eval_batch_plans/...",
      "request_count": N,
      "submit_summary": {capture_keys_total, skipped_unmapped, ...}
    }

The SF Choice state inspects ``status`` + ``processing_status``
returned in the next Poll Lambda call; ``EMPTY`` short-circuits the
poll loop and routes directly to Process so the empty-batch case
still emits a clean SF result + downstream metric inputs.

Eval is observability per ROADMAP §1635 — submission failures must
NOT halt the Saturday pipeline. The SF state has its own Catch
that routes to EvalRollingMean on any error.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import tempfile

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
    "eval_judge_submit",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-eval-judge-submit",
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """Mirror lambda/handler.py's deferred-init pattern."""
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
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.datetime.now(datetime.UTC)
    _ensure_init()

    from evals.lambda_dry import dry_submit_result, is_dry
    from evals.orchestrator import (
        DEFAULT_HAIKU_MODEL,
        DEFAULT_SONNET_MODEL,
        _persist_client_side_skips,
        build_batch_plan,
        submit_batch,
    )

    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    # alpha-engine-config-I8155: `$.eval_cadence.eval_date` (delivered here as
    # `date`) is States.ArrayGetItem(States.StringSplit($$.Execution.StartTime,
    # 'T'), 0) — the date part of the SF execution's own start time, computed
    # independently of (but identical in value to) $.run_date. Captured here,
    # BEFORE the business-logic fallback below, so stage_coverage never
    # receives a fabricated `datetime.date.today()` substitute — that
    # fallback exists for `build_batch_plan`'s own needs, not for coverage
    # attribution.
    execution_run_date = event.get("date") or None
    date = event.get("date") or str(datetime.date.today())

    # ── Shell-run dry path ───────────────────────────────────────────
    # Boot + import (above) ran for real — that's the keystone's whole
    # point. Return BEFORE build_batch_plan / _persist_client_side_skips
    # (S3 put_object) / submit_batch (Anthropic Message Batches create).
    # The sentinel batch_id + status=EMPTY makes the SF Choice skip the
    # poll loop and route straight to Process, which also short-circuits.
    if is_dry(event):
        logger.info(
            "[eval_judge_submit_handler] dry_run_llm=True: shell-run "
            "no-op (no Anthropic batch, no S3 plan persist) date=%s", date,
        )
        return dry_submit_result(date)
    force_sonnet_pass = bool(event.get("force_sonnet_pass", False))
    # This one Lambda backs BOTH EvalJudgeSubmitFirstSaturday
    # (force_sonnet_pass=true) and EvalJudgeSubmitWeekly (false) — the
    # stage name MUST be derived from the invocation, never hardcoded, or
    # one stage's assertion is filed under the other's name and a real
    # miss is attributed to a stage that was working (config-I7214).
    _stage_name = (
        "EvalJudgeSubmitFirstSaturday" if force_sonnet_pass
        else "EvalJudgeSubmitWeekly"
    )
    haiku_model = event.get("haiku_model", DEFAULT_HAIKU_MODEL)
    sonnet_model = event.get("sonnet_model", DEFAULT_SONNET_MODEL)
    judge_only = bool(event.get("judge_only", False))
    # Optional family-selection params (config#1579 P2): judge daily
    # producers (thinktank) whose artifacts land in weekday partitions.
    extra_dates = event.get("extra_dates") or None
    agent_id_prefixes = event.get("agent_id_prefixes") or None
    # capture_lookback_days=N expands to extra_dates covering the N
    # write-settled trading days before `date` — the Saturday SF passes 6
    # so the week's thinktank captures ride the SAME weekly batch
    # (unmapped agent_ids in those partitions are skipped cleanly).
    #
    # alpha-engine-config-I9331: this does NOT expand off the raw `date`.
    # Think Tank writes day D's captures on D+1 ~14:35 UTC, and this
    # Lambda's own weekly invocation enters at ~05:11 UTC on `date` — so
    # `date`'s own trading day (and, depending on where `date` falls in
    # the week, the trading day immediately before it too) is frequently
    # NOT YET WRITTEN. `compute_judge_window_dates` shifts the window's
    # newest boundary back by the measured write-settle lag before
    # expanding the lookback, so every date it returns is guaranteed
    # already written. The day this defers is not lost — it is squarely
    # inside next week's unchanged 6-day lookback (see that function's
    # docstring), and `load_already_judged_keys` dedups the overlap.
    lookback = int(event.get("capture_lookback_days") or 0)
    if lookback > 0:
        from evals.orchestrator import compute_judge_window_dates

        computed = compute_judge_window_dates(date, lookback)
        extra_dates = sorted(set(computed) | set(extra_dates or []), reverse=True)

    logger.info(
        "[eval_judge_submit_handler] start date=%s force_sonnet=%s "
        "haiku=%s sonnet=%s judge_only=%s",
        date, force_sonnet_pass, haiku_model, sonnet_model, judge_only,
    )

    try:
        plan = build_batch_plan(
            date=date, bucket=bucket,
            haiku_model=haiku_model, sonnet_model=sonnet_model,
            force_sonnet_pass=force_sonnet_pass, judge_only=judge_only,
            extra_dates=extra_dates, agent_id_prefixes=agent_id_prefixes,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[eval_judge_submit_handler] plan build failed")
        return {"status": "ERROR", "stage": "plan_build", "error": str(exc)}

    import boto3
    s3 = boto3.client("s3")
    skip_count, degenerate_skip_count, _, skip_failed = _persist_client_side_skips(
        plan, s3=s3, bucket=bucket,
    )

    # alpha-engine-config-I9263 (Brian ruling 2026-08-29: "I will not fund the
    # anthropic account, at this point we shouldn't be using the anthropic api
    # at all"). No provider SDK client is built here any more. `submit_batch`
    # asks the router whether the judge's model group can serve the `batches`
    # CAPABILITY from this Lambda and, when it cannot, takes the synchronous
    # rung and records the degradation durably — see
    # `evals/judge_batch_transport.py` for the ladder.
    try:
        submit_result = submit_batch(plan, s3_client=s3)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[eval_judge_submit_handler] batch submission failed")
        return {"status": "ERROR", "stage": "submit", "error": str(exc)}

    is_empty = submit_result["processing_status"] == "ended_empty"
    status = "EMPTY" if is_empty else "OK"

    logger.info(
        "[eval_judge_submit_handler] done status=%s batch_id=%s "
        "request_count=%d empty_input_skips=%d degenerate_input_skips=%d "
        "skip_failed=%d",
        status, submit_result["batch_id"], submit_result["request_count"],
        skip_count, degenerate_skip_count, len(skip_failed),
    )

    result = {
        "status": status,
        "batch_id": submit_result["batch_id"],
        "plan_s3_key": submit_result["plan_s3_key"],
        "request_count": submit_result["request_count"],
        "processing_status": submit_result["processing_status"],
        "submit_summary": {
            "date": date,
            "capture_keys_total": plan["capture_keys_total"],
            "skipped_unmapped": plan["skipped_unmapped"],
            "skipped_empty_input_persisted": skip_count,
            "skipped_degenerate_input_persisted": degenerate_skip_count,
            "skip_failed": skip_failed,
            "force_sonnet_pass": force_sonnet_pass,
            "judge_only": judge_only,
            # I9331 deliverable 3: an expected-but-empty capture partition
            # is a named finding, not an absence. Non-empty here does NOT
            # halt the run (eval is observability) but MUST be visible on
            # every surface reading this result — never collapse it into
            # a clean-looking `capture_keys_total`.
            "capture_partition_counts": plan["capture_partition_counts"],
            "empty_trading_day_partitions": plan["empty_trading_day_partitions"],
        },
    }

    # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    # §2.3a rescope): the assertion lives in the stage's own handler,
    # immediately before it returns, rather than a separate end-of-run SF
    # state. OBSERVE MODE ONLY — never enables enforcement, never raises.
    if not execution_run_date:
        # alpha-engine-config-I8155: never fabricate a date to satisfy the
        # (now-required) run_date argument.
        logger.error(
            "[eval_judge_submit_handler] stage-coverage assertion SKIPPED "
            "for %s: no execution run_date on this event (execution "
            "identity absent)", _stage_name,
        )
        result["stage_coverage"] = {
            "stage": _stage_name,
            "status": "UNMEASURED",
            "reason": "execution run_date absent from event",
        }
    else:
        try:
            from krepis.stage_coverage import assert_stage_coverage

            result["stage_coverage"] = assert_stage_coverage(
                _stage_name, run_date=execution_run_date, window_start=_started,
            )
        except ImportError as exc:
            # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
            # the handler's own outcome is unchanged (config-I7214).
            logger.error("stage-coverage assertion unavailable: %s", exc)
            result["stage_coverage"] = {
                "stage": _stage_name,
                "status": "UNMEASURED",
                "reason": f"assertion unavailable: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 — never let the observer kill the stage it observes
            # alpha-engine-config-I8155: the krepis landing this arc makes
            # run_date a required, contract-enforced kwarg (TypeError on
            # omission, StageCoverageContractError on blank/None). Both are
            # library-internal contract failures, not stage failures — log
            # loudly and degrade to UNMEASURED rather than raising out of
            # the handler.
            logger.error(
                "[eval_judge_submit_handler] stage-coverage assertion "
                "raised for %s: %s: %s", _stage_name, type(exc).__name__, exc,
            )
            result["stage_coverage"] = {
                "stage": _stage_name,
                "status": "UNMEASURED",
                "reason": f"assertion raised: {type(exc).__name__}: {exc}",
            }

    return result
