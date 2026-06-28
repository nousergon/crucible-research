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
from datetime import datetime

# Repo root on sys.path so ``from evals.rolling_mean import ...``
# resolves under Lambda's task layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch
_install_ls_patch()

from alpha_engine_lib.logging import monitor_handler, setup_logging
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
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors lambda/handler.py
    + lambda/eval_judge_handler.py — Lambda init phase 10s ceiling."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


@monitor_handler
def handler(event, context):
    """Compute + emit the rolling 4-week mean derived metric."""
    _ensure_init()

    # Imports deferred until after _ensure_init in case the rolling-mean
    # module ever pulls config that depends on SSM-loaded secrets.
    from evals.rolling_mean import compute_and_emit_4w_mean

    end_time_iso = event.get("end_time_iso")
    end_time = (
        datetime.fromisoformat(end_time_iso.replace("Z", "+00:00"))
        if end_time_iso else None
    )

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
        logger.warning(
            "[eval_rolling_mean_handler] calibration κ report failed (non-fatal): %s",
            exc,
        )
        calibration = {"status": "ERROR", "error": str(exc)}

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
        logger.warning(
            "[eval_rolling_mean_handler] control bands failed (non-fatal): %s",
            exc,
        )
        control_bands = {"status": "ERROR", "error": str(exc)}

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
    agent_quality: dict
    try:
        import boto3

        from alpha_engine_lib.dates import now_dual
        from scripts.build_agent_quality import build_agent_quality, write_agent_quality

        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        dual = now_dual()
        s3c = boto3.client("s3")
        artifact = build_agent_quality(
            s3c, bucket, dual.trading_day, run_date=dual.calendar_date,
        )
        key = write_agent_quality(s3c, bucket, artifact)
        graded = sorted(k for k, v in artifact.items() if isinstance(v, dict) and "value" in v)
        agent_quality = {"status": "OK", "key": key, "graded_components": graded}
        logger.info(
            "[eval_rolling_mean_handler] agent_quality wrote %s (%d graded: %s)",
            key, len(graded), ",".join(graded) or "(none — no signals/evals this run)",
        )
    except Exception as exc:  # noqa: BLE001 — secondary path, see comment above
        logger.warning(
            "[eval_rolling_mean_handler] agent_quality build failed (non-fatal): %s", exc,
        )
        agent_quality = {"status": "ERROR", "error": str(exc)}

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

        from alpha_engine_lib.dates import now_dual
        from scoring.leaderboard_producers import build_producer_leaderboard

        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        dual = now_dual()
        s3c = boto3.client("s3")
        lb = build_producer_leaderboard(s3c, bucket, dual.trading_day)
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
    except Exception as exc:  # noqa: BLE001 — secondary path, see comment above
        logger.warning(
            "[eval_rolling_mean_handler] producer_leaderboard build failed (non-fatal): %s",
            exc,
        )
        producer_leaderboard = {"status": "ERROR", "error": str(exc)}

    return {
        "status": status,
        "summary": summary,
        "calibration": calibration,
        "control_bands": control_bands,
        "agent_quality": agent_quality,
        "producer_leaderboard": producer_leaderboard,
    }
