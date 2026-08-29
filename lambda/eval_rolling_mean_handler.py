"""Lambda entry point — rolling-4-week-mean derived metric (PR 4b).

Triggered weekly (PR 4c will wire the EventBridge rule). Computes
the rolling 4-week average of ``AlphaEngine/Eval/agent_quality_score``
per (judged_agent_id, criterion, judge_model) combo and emits the
derived metric ``AlphaEngine/Eval/agent_quality_score_4w_mean`` —
the surface a CloudWatch alarm fires against per ROADMAP §1634.

Event shape (all fields optional):

    {
      "end_time_iso": "2026-05-09T00:00:00Z"   # default = now UTC
    }

Returns:

    {
      "status": "OK" | "PARTIAL" | "ERROR",
      "summary": <rolling_mean.compute_and_emit_4w_mean result>
    }
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import UTC, date, datetime

# Repo root on sys.path so ``from evals.rolling_mean import ...``
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
    "eval_rolling_mean",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-eval-rolling-mean",
)

logger = logging.getLogger(__name__)

_init_done = False


def _emit_producer_failure_alert(*, producer: str, artifact: str, exc: BaseException) -> None:
    """Put a secondary-producer failure on the machine-readable alert bus.

    This handler carries four best-effort aggregations whose failure must not
    sink the rolling mean (the primary deliverable). That carve-out is
    legitimate; recording the failure ONLY in a WARN log is not — a human-only
    signal is invisible (``observability-policy.md`` §7.3), and it is exactly
    what let the ``agent_quality`` date-type bug run for two months with eight
    report-card components reading N/A (alpha-engine-config-I8177).

    Best-effort by construction: alerting must never itself sink the handler,
    so a failure to publish is logged and swallowed here — that is the ONE
    place a swallow is correct, because the condition it would hide is already
    on the ERROR log the caller wrote before calling in.

    ``dedup_key`` is stable per producer per ISO week: one alert per weekly
    cycle no matter how many times the Lambda is retried, and it re-fires the
    following week if the producer is still broken (never latches shut).
    """
    try:
        from krepis import alerts

        week = datetime.now(UTC).strftime("%G-W%V")
        alerts.publish(
            f"{producer} producer failed — {artifact} will not be written this "
            f"cycle and its report-card components grade N/A-MISSING-INPUT. "
            f"{type(exc).__name__}: {exc}",
            severity="error",
            source="crucible-research/eval_rolling_mean_handler",
            dedup_key=f"eval-rolling-mean-producer-failure-{producer}-{week}",
            dedup_window_min=None,
        )
    except Exception:  # noqa: BLE001 — see docstring: alerting never sinks the handler
        logger.exception(
            "[eval_rolling_mean_handler] could not publish the %s producer-failure "
            "alert; the ERROR log above is the surviving record", producer,
        )


# --------------------------------------------------------------------------- #
# Bounding the secondary aggregations (alpha-engine-config#9102)
# --------------------------------------------------------------------------- #
#
# This handler's PRIMARY deliverable — the rolling mean — completes in ~1.5s.
# Four best-effort aggregations hang off it, and two of them (agent_quality,
# producer_leaderboard) scan an S3 corpus that GROWS every cycle, through a bare
# ``boto3.client("s3")`` carrying botocore's defaults: connect_timeout=60,
# read_timeout=60, legacy retries up to 5. Neither logs anything between
# entering its try block and finishing, so a slow scan is indistinguishable from
# a hang while it is happening.
#
# Measured on 2026-08-28: this state's `REPORT` durations ran 22-37s for seven
# invocations, then 120s, then 65s, then hit the 300s ceiling — the shape of a
# scan outgrowing its budget, not a network blip. When it hit the ceiling the SF
# recorded `States.Timeout`, `MarkEvalRollingMeanDegraded` fired, and the whole
# research/predictor branch fail-opened, so a run that HAD produced its primary
# deliverable terminated DEGRADED.
#
# The carve-out that lets these fail without sinking the rolling mean is
# legitimate. What was missing is that the carve-out only covered EXCEPTIONS: a
# block that never returns is not an exception, so it consumed the budget of the
# thing it was not allowed to sink. A secondary aggregation now cannot spend more
# than its own deadline, and blowing that deadline is recorded on exactly the
# surface an exception would have used — never swallowed.
_SECONDARY_DEADLINE_S = 45.0


def _bounded_client(service: str):
    """An AWS client that cannot hang open-endedly.

    Worst case per call is ~(connect + read) * attempts, so a wedged endpoint
    costs seconds rather than minutes. Mirrors the bound flow-doctor applies to
    its own notifier clients (flow-doctor#93, 0.16.2).
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        service,
        config=Config(
            connect_timeout=5,
            read_timeout=15,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def _run_bounded(name: str, artifact: str, fn, deadline_s: float = _SECONDARY_DEADLINE_S) -> dict:
    """Run a secondary aggregation under a wall-clock deadline.

    Returns the block's own result dict, or a TIMEOUT/ERROR record. The worker is
    a daemon thread so a blown deadline cannot hold the invocation open either —
    abandoning it is the whole point, and a non-daemon worker would reintroduce
    the defect one layer down.

    A blown deadline is NOT a silent degrade: it logs at ERROR and goes on the
    same machine-readable alert bus an exception would, because a producer that
    quietly stops producing is the failure mode this handler has already been
    bitten by twice (the two-month `agent_quality` date-type bug, #8177).
    """
    import threading

    box: dict = {}

    def _work() -> None:
        try:
            box["ok"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's side
            box["exc"] = exc

    logger.info("[eval_rolling_mean_handler] %s start (deadline %.0fs)", name, deadline_s)
    worker = threading.Thread(target=_work, name=f"secondary-{name}", daemon=True)
    started = datetime.now(UTC)
    worker.start()
    worker.join(deadline_s)
    elapsed = (datetime.now(UTC) - started).total_seconds()

    if worker.is_alive():
        logger.error(
            "[eval_rolling_mean_handler] %s EXCEEDED its %.0fs deadline (%.1fs elapsed) "
            "and was abandoned — %s will not be written this cycle. The rolling mean "
            "is unaffected; this is recorded so a growing scan cannot silently eat "
            "the stage budget again (alpha-engine-config#9102).",
            name, deadline_s, elapsed, artifact,
        )
        _emit_producer_failure_alert(
            producer=name,
            artifact=artifact,
            exc=TimeoutError(f"{name} exceeded its {deadline_s:.0f}s deadline"),
        )
        return {"status": "TIMEOUT", "deadline_s": deadline_s, "elapsed_s": round(elapsed, 1)}

    if "exc" in box:
        raise box["exc"]

    logger.info("[eval_rolling_mean_handler] %s done in %.1fs", name, elapsed)
    return box["ok"]


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors lambda/handler.py
    + lambda/eval_judge_handler.py — Lambda init phase 10s ceiling."""
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
    """Compute + emit the rolling 4-week mean derived metric."""
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.now(UTC)
    _ensure_init()

    # Imports deferred until after _ensure_init in case the rolling-mean
    # module ever pulls config that depends on SSM-loaded secrets.
    from evals.rolling_mean import compute_and_emit_4w_mean

    end_time_iso = event.get("end_time_iso")
    end_time = (
        datetime.fromisoformat(end_time_iso.replace("Z", "+00:00"))
        if end_time_iso else None
    )
    # This handler's own event carries no `run_date` field — only
    # end_time_iso ($$.Execution.StartTime, SF-threaded verbatim). Its date
    # portion IS the execution's own un-normalized run_date (verified
    # against nousergon-data/infrastructure/step_function.json —
    # `EvalRollingMean`'s Payload is `end_time_iso.$: "$$.Execution.
    # StartTime"`, the same source InitializeInput derives $.run_date from),
    # not a proxy for it. alpha-engine-config-I8155: `_started` (this
    # handler's OWN invocation time, via `datetime.now(UTC)`) must never
    # substitute for a genuinely-absent end_time_iso — that was exactly the
    # forbidden fabrication class. When end_time_iso is absent there is no
    # execution identity to attribute to; the assertion is skipped below.
    _execution_run_date = end_time.date().isoformat() if end_time else None

    logger.info(
        "[eval_rolling_mean_handler] start end_time_iso=%s",
        end_time_iso or "(now UTC)",
    )

    try:
        summary = compute_and_emit_4w_mean(end_time=end_time)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[eval_rolling_mean_handler] computation failed hard")
        return {"status": "ERROR", "error": str(exc)}

    status = "PARTIAL" if summary["failed"] else "OK"
    logger.info(
        "[eval_rolling_mean_handler] done status=%s emitted=%d skipped=%d failed=%d",
        status,
        summary["datapoints_emitted"],
        summary["combos_skipped_no_data"],
        len(summary["failed"]),
    )

    # Judge-calibration κ report (ROADMAP L480). Secondary observability
    # hung off this weekly eval Lambda — it reads the operator calibration
    # corpus and writes a κ report to S3 for the backtester evaluator
    # email to surface. A failure here MUST NOT sink the rolling-mean
    # metric (the primary deliverable), but per [[feedback_no_silent_fails]]
    # it is recorded: WARN log + a `calibration` field in the return value.
    calibration: dict
    try:
        from evals.calibration_kappa import emit_calibration_report

        report = emit_calibration_report()
        calibration = {
            "status": report["status"],
            "n_cells": report["n_cells"],
            "n_cells_sufficient": report["n_cells_sufficient"],
            "n_paired_reviews": report["n_paired_reviews"],
        }
        logger.info(
            "[eval_rolling_mean_handler] calibration κ status=%s cells=%d sufficient=%d",
            calibration["status"],
            calibration["n_cells"],
            calibration["n_cells_sufficient"],
        )
    except Exception as exc:  # noqa: BLE001 — secondary path, see comment above
        logger.error(
            "[eval_rolling_mean_handler] calibration κ report FAILED: %s",
            exc, exc_info=True,
        )
        calibration = {"status": "ERROR", "error": str(exc)}
        _emit_producer_failure_alert(
            producer="calibration_kappa",
            artifact="the eval-judge calibration κ report",
            exc=exc,
        )

    # Statistical control bands on the judge-score series (L4578(e)).
    # Reads the rolling-mean series this run just emitted and runs the
    # Shewhart + CUSUM charts. Secondary observability hung off the
    # primary rolling-mean deliverable: a failure here MUST NOT sink the
    # mean (already emitted above), but per [[feedback_no_silent_fails]]
    # it is recorded — WARN log + a `control_bands` field in the return.
    control_bands: dict
    try:
        from evals.control_bands import compute_and_emit_control_bands

        cb = compute_and_emit_control_bands(end_time=end_time)
        control_bands = {
            "status": "OK" if not cb["failed"] else "PARTIAL",
            "combos_discovered": cb["combos_discovered"],
            "combos_insufficient_history": cb["combos_insufficient_history"],
            "breach_count": cb["breach_count"],
            "breach_emits": cb["breach_emits"],
        }
        logger.info(
            "[eval_rolling_mean_handler] control bands status=%s combos=%d "
            "insufficient=%d breaches=%d",
            control_bands["status"],
            control_bands["combos_discovered"],
            control_bands["combos_insufficient_history"],
            control_bands["breach_count"],
        )
    except Exception as exc:  # noqa: BLE001 — secondary path, see comment above
        logger.error(
            "[eval_rolling_mean_handler] control bands FAILED: %s",
            exc, exc_info=True,
        )
        control_bands = {"status": "ERROR", "error": str(exc)}
        _emit_producer_failure_alert(
            producer="control_bands",
            artifact="the per-combo control-band breach emits",
            exc=exc,
        )

    # Agent-quality report-card artifact (config#1149 Batch A). THIRD secondary
    # observability aggregation hung off this weekly eval Lambda — same trigger
    # point (post-eval-judge convergence) + S3 access as the κ report and control
    # bands above, and the exact moment both finalized signals AND eval-judge
    # results exist. The producer (scripts/build_agent_quality) was complete but
    # wired into NOTHING, so backtest/{date}/agent_quality.json never landed and
    # its report-card components (signal_volume_adequacy / cost_per_signal /
    # judge_rubric_pass_rate / judge_rubric_distribution) read N/A. This is the
    # missing invocation. Dual-date per DATE_CONVENTIONS: target_date = trading
    # day (signals + output key), run_date = calendar day (cost + eval partitions).
    # Best-effort: a failure MUST NOT sink the rolling mean (primary deliverable);
    # recorded as an `agent_quality` field per [[feedback_no_silent_fails]].
    #
    # alpha-engine-config-I8177: this block previously passed
    # `dual.trading_day` / `dual.calendar_date` — which `now_dual()` returns as
    # ISO **strings** — into a producer annotated `date`. Annotations are
    # unenforced at runtime, so every Saturday from 2026-06-23 (the wiring
    # merge, #304) to 2026-08-22 the producer died on `'str' object has no
    # attribute 'isoformat'` and `agent_quality.json` never landed on ANY date.
    # Eight report-card components (agent_validation_failure_rate,
    # cost_per_signal, retry_storm_count, agent_latency_p95,
    # judge_rubric_distribution, judge_rubric_pass_rate, signal_volume_adequacy,
    # judge_outcome_ic) read N/A-MISSING-INPUT for two months as a result.
    # `build_agent_quality` now normalizes both carriers at its own boundary;
    # the explicit `date.fromisoformat` here states the contract at the call
    # site too, so a reader does not have to trust the coercion to see it.
    agent_quality: dict
    try:
        # boto3 is imported by _bounded_client, not here — every client in this
        # handler must carry an explicit timeout Config (alpha-engine-config#9102).
        from nousergon_lib.dates import now_dual

        from scripts.build_agent_quality import build_agent_quality, write_agent_quality

        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        dual = now_dual()
        s3c = _bounded_client("s3")

        def _build_agent_quality() -> dict:
            artifact = build_agent_quality(
                s3c, bucket,
                date.fromisoformat(dual.trading_day),
                run_date=date.fromisoformat(dual.calendar_date),
            )
            key = write_agent_quality(s3c, bucket, artifact)
            graded = sorted(
                k for k, v in artifact.items() if isinstance(v, dict) and "value" in v
            )
            logger.info(
                "[eval_rolling_mean_handler] agent_quality wrote %s (%d graded: %s)",
                key, len(graded), ",".join(graded) or "(none — no signals/evals this run)",
            )
            return {"status": "OK", "key": key, "graded_components": graded}

        agent_quality = _run_bounded(
            "agent_quality", "backtest/{date}/agent_quality.json", _build_agent_quality,
        )
    except Exception as exc:  # noqa: BLE001 — secondary path, see comment above
        # The primary deliverable (the rolling mean) still survives a failure
        # here — but the failure MUST NOT be invisible. Recording it only in a
        # WARN log and a return-value field nothing consumes is what let the
        # date-type bug above run unnoticed for two months
        # (observability-policy.md §7.3: a human-only signal is invisible;
        # the fleet rule forbids graceful-degrade on a PRODUCER/writer).
        # ERROR severity + a weekly-stable dedup key: one alert per cycle, not
        # one per retry, and it re-fires every week the producer stays broken.
        logger.error(
            "[eval_rolling_mean_handler] agent_quality build FAILED — "
            "backtest/{date}/agent_quality.json will not exist this cycle and "
            "its 8 report-card components grade N/A: %s", exc, exc_info=True,
        )
        agent_quality = {"status": "ERROR", "error": str(exc)}
        _emit_producer_failure_alert(
            producer="agent_quality",
            artifact="backtest/{date}/agent_quality.json",
            exc=exc,
        )

    # Research producer champion/challenger leaderboard (config#1223 B4 / #1221
    # shared scorer). FOURTH secondary observability aggregation hung off this
    # weekly eval Lambda — same trigger point + S3 access as agent_quality above,
    # and the moment realized forward (21d) outcomes for prior cohorts have
    # matured in alpha-engine-data's daily_closes. The shared scorer
    # (scoring/leaderboard_producers.build_producer_leaderboard) reads every
    # signals_shadow/ cohort + the live signals/ champion, joins to realized 21d
    # returns, scores each producer vs the champion on realized rank-IC +
    # long-only top-N alpha (date-clustered), and writes
    # research/producer_leaderboard/{date}.json. OBSERVE-ONLY + fail-soft: the
    # function never raises (returns a status dict); the extra try/except is
    # belt-and-suspenders so the rolling mean (primary deliverable) is never sunk.
    # Recorded as a `producer_leaderboard` field per [[feedback_no_silent_fails]].
    # Cohort-gated: ships n_dates=0 + null metrics until forward cohorts mature
    # (full closure of #1221/#1223 = the OBSERVATION_REGISTRY cohort gate).
    producer_leaderboard: dict
    try:
        # boto3 is imported by _bounded_client, not here — every client in this
        # handler must carry an explicit timeout Config (alpha-engine-config#9102).
        from nousergon_lib.dates import now_dual

        from scoring.leaderboard_producers import build_producer_leaderboard

        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        dual = now_dual()
        s3c = _bounded_client("s3")

        def _build_producer_leaderboard() -> dict:
            lb = build_producer_leaderboard(s3c, bucket, dual.trading_day)
            block = {
                "status": lb.get("status"),
                "key": lb.get("key"),
                "n_dates": (lb.get("leaderboard") or {}).get("n_dates"),
            }
            logger.info(
                "[eval_rolling_mean_handler] producer_leaderboard status=%s key=%s n_dates=%s",
                block["status"], block["key"], block["n_dates"],
            )
            return block

        producer_leaderboard = _run_bounded(
            "producer_leaderboard",
            "the research producer champion/challenger leaderboard",
            _build_producer_leaderboard,
        )
    except Exception as exc:  # noqa: BLE001 — secondary path, see comment above
        logger.error(
            "[eval_rolling_mean_handler] producer_leaderboard build FAILED: %s",
            exc, exc_info=True,
        )
        producer_leaderboard = {"status": "ERROR", "error": str(exc)}
        _emit_producer_failure_alert(
            producer="producer_leaderboard",
            artifact="the research producer champion/challenger leaderboard",
            exc=exc,
        )

    result = {
        "status": status,
        "summary": summary,
        "calibration": calibration,
        "control_bands": control_bands,
        "agent_quality": agent_quality,
        "producer_leaderboard": producer_leaderboard,
    }

    # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    # §2.3a rescope): the assertion lives in the stage's own handler,
    # immediately before it returns, rather than a separate end-of-run SF
    # state. OBSERVE MODE ONLY — never enables enforcement, never raises.
    if not _execution_run_date:
        # alpha-engine-config-I8155: never fabricate a date to satisfy the
        # (now-required) run_date argument.
        logger.error(
            "[eval_rolling_mean_handler] stage-coverage assertion SKIPPED "
            "for EvalRollingMean: no end_time_iso on this event (execution "
            "identity absent)",
        )
        result["stage_coverage"] = {
            "stage": "EvalRollingMean",
            "status": "UNMEASURED",
            "reason": "execution run_date absent from event (no end_time_iso)",
        }
    else:
        try:
            from krepis.stage_coverage import assert_stage_coverage

            result["stage_coverage"] = assert_stage_coverage(
                "EvalRollingMean", run_date=_execution_run_date,
                window_start=_started,
            )
        except ImportError as exc:
            # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
            # the handler's own outcome is unchanged (config-I7214).
            logger.error("stage-coverage assertion unavailable: %s", exc)
            result["stage_coverage"] = {
                "stage": "EvalRollingMean",
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
                "[eval_rolling_mean_handler] stage-coverage assertion "
                "raised for EvalRollingMean: %s: %s",
                type(exc).__name__, exc,
            )
            result["stage_coverage"] = {
                "stage": "EvalRollingMean",
                "status": "UNMEASURED",
                "reason": f"assertion raised: {type(exc).__name__}: {exc}",
            }

    return result
