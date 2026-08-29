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

from invocation_budget import (  # noqa: E402
    BlockTimeout,
    InvocationBudget,
    NoBudget,
    run_bounded,
)

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

# Per-block wall-clock ceilings for the FOUR secondary aggregations bolted onto
# this stage (alpha-engine-config-I9102). The primary deliverable — the rolling
# 4-week mean — runs first and unbounded; it took 1.6s on 2026-08-28 and 0.6s
# the week before, and it is the only thing this stage owes the pipeline.
#
# Every number below is a BUDGET, not an accommodation (sf-pipeline-policy.md
# §4). Each is the block's observed duration with headroom, measured from this
# function's own CloudWatch REPORT/log timeline:
#
#   calibration_kappa      0.6s   (2026-08-21, -22, -28: 0.59s / 0.60s / 0.59s)
#   control_bands          0.4-1.4s (0.39s / 0.92s / 1.41s over the same three)
#   agent_quality          NEVER COMPLETED. Its predecessor died in 0.2s on the
#                          I8177 date-type bug every week from 2026-06-23 to
#                          2026-08-22; the first run after that fix (2026-08-28)
#                          entered it and was still inside it 298s later, because
#                          evals.judge_outcome_ic.open_research_db downloads the
#                          356 MB s3://alpha-engine-research/research.db snapshot
#                          into a 512 MB function. 240s is a deliberate first
#                          budget for a block with no completed observation:
#                          generous enough that the download plausibly finishes
#                          at the raised memory/CPU, bounded enough that it can
#                          never again spend the stage's whole invocation.
#                          Re-size from the first COMPLETED run, not from this.
#   producer_leaderboard   60-95s (2026-08-21: 96.9s, -22: 61.4s — an ArcticDB
#                          load_universe_ohlcv walk over ~905 symbols). 240s is
#                          ~2.5x the worst observation, which is where a block
#                          that grows with the universe needs to sit.
#
# The ceilings sum to 600s. With the primary deliverable (~1.6s) and the 20s
# reserve that fits inside the function's 900s ceiling with ~280s spare, so the
# worst case is every block spending its whole ceiling and the stage STILL
# returning normally — never the SF state being the thing that stops it.
_CEILING_CALIBRATION_S = 60.0
_CEILING_CONTROL_BANDS_S = 60.0
_CEILING_AGENT_QUALITY_S = 240.0
_CEILING_PRODUCER_LEADERBOARD_S = 240.0


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

    # Everything below this line is SECONDARY (alpha-engine-config-I9102). The
    # primary deliverable is emitted; from here on the invocation's remaining
    # time is a resource that must be rationed, because on 2026-08-28 one
    # secondary block spent all 298 remaining seconds and the Step Functions
    # state raised States.Timeout on a stage that had already succeeded — which
    # fail-opened the research/predictor branch and terminated the weekly run
    # FAILED. Unbounded outside Lambda (tests, local runs), so behaviour there
    # is unchanged.
    _budget = InvocationBudget.from_context(context)

    def _shortfall(producer: str, artifact: str, exc: BaseException) -> dict:
        """Record a block the invocation could not afford, on a machine surface.

        Not a silent skip: ERROR log + the same producer-failure alert a hard
        failure raises + a status the consumer can distinguish from OK. The
        stage keeps its primary deliverable and returns.
        """
        logger.error(
            "[eval_rolling_mean_handler] %s NOT COMPLETED within this "
            "invocation's budget: %s", producer, exc,
        )
        _emit_producer_failure_alert(producer=producer, artifact=artifact, exc=exc)
        return {
            "status": "TIMEOUT" if isinstance(exc, BlockTimeout) else "SKIPPED_NO_BUDGET",
            "error": str(exc),
        }
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

        report = run_bounded(
            emit_calibration_report,
            name="calibration_kappa",
            ceiling_s=_CEILING_CALIBRATION_S,
            budget=_budget,
        )
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
    except (BlockTimeout, NoBudget) as exc:
        calibration = _shortfall(
            "calibration_kappa", "the eval-judge calibration κ report", exc,
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

        cb = run_bounded(
            lambda: compute_and_emit_control_bands(end_time=end_time),
            name="control_bands",
            ceiling_s=_CEILING_CONTROL_BANDS_S,
            budget=_budget,
        )
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
    except (BlockTimeout, NoBudget) as exc:
        control_bands = _shortfall(
            "control_bands", "the per-combo control-band breach emits", exc,
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
        import boto3
        from nousergon_lib.dates import now_dual

        from scripts.build_agent_quality import build_agent_quality, write_agent_quality

        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        dual = now_dual()
        s3c = boto3.client("s3")
        def _build_and_write_agent_quality() -> tuple[dict, str]:
            built = build_agent_quality(
                s3c, bucket,
                date.fromisoformat(dual.trading_day),
                run_date=date.fromisoformat(dual.calendar_date),
            )
            return built, write_agent_quality(s3c, bucket, built)

        artifact, key = run_bounded(
            _build_and_write_agent_quality,
            name="agent_quality",
            ceiling_s=_CEILING_AGENT_QUALITY_S,
            budget=_budget,
        )
        graded = sorted(k for k, v in artifact.items() if isinstance(v, dict) and "value" in v)
        agent_quality = {"status": "OK", "key": key, "graded_components": graded}
        logger.info(
            "[eval_rolling_mean_handler] agent_quality wrote %s (%d graded: %s)",
            key, len(graded), ",".join(graded) or "(none — no signals/evals this run)",
        )
    except (BlockTimeout, NoBudget) as exc:
        agent_quality = _shortfall(
            "agent_quality", "backtest/{date}/agent_quality.json", exc,
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
        import boto3
        from nousergon_lib.dates import now_dual

        from scoring.leaderboard_producers import build_producer_leaderboard

        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        dual = now_dual()
        s3c = boto3.client("s3")
        lb = run_bounded(
            lambda: build_producer_leaderboard(s3c, bucket, dual.trading_day),
            name="producer_leaderboard",
            ceiling_s=_CEILING_PRODUCER_LEADERBOARD_S,
            budget=_budget,
        )
        producer_leaderboard = {
            "status": lb.get("status"),
            "key": lb.get("key"),
            "n_dates": (lb.get("leaderboard") or {}).get("n_dates"),
        }
        logger.info(
            "[eval_rolling_mean_handler] producer_leaderboard status=%s key=%s n_dates=%s",
            producer_leaderboard["status"], producer_leaderboard["key"],
            producer_leaderboard["n_dates"],
        )
    except (BlockTimeout, NoBudget) as exc:
        producer_leaderboard = _shortfall(
            "producer_leaderboard",
            "the research producer champion/challenger leaderboard",
            exc,
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
