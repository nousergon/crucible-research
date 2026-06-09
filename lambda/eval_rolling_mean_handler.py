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

    return {"status": status, "summary": summary, "calibration": calibration}
