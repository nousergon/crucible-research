"""Lambda entry point — weekly judge-sensitivity scorecard (Phase B, config#752).

Runs the synthetic-perturbation battery (``evals/perturbation.py``) on a weekly
cadence and writes the sensitivity scorecard to
``decision_artifacts/_perturbation/_report/{date}/sensitivity.{json,md}`` plus
``latest/`` pointers, for the backtester evaluator email to surface.

Phase A already gates per-PR judge regressions via a paths-filtered live smoke
(``tests/live_smoke/judge_perturbation_smoke.py``). Phase B catches the
BETWEEN-PR failure mode — silent model/API drift that no PR touches. The battery
needs live Anthropic access, so it runs as a CMD-override handler on the
eval-judge image (which carries the key) rather than the no-LLM rolling-mean
Lambda that already hosts the calibration-κ report. A new Saturday Step Functions
state invokes this handler with ``{"weekly_run": true}``.

Event shape (all fields optional):

    {
      "run_date":    "2026-07-11",   # default = today UTC; the report_date key
      "dry_run_llm": false           # true → skip the live battery + S3 write
    }

Returns:

    {
      "status": "OK" | "SKIPPED_DRY_RUN" | "ERROR",
      "n": <int>, "n_caught": <int>, "caught_rate": <float>,
      "judge_model": <str>, "report_keys": [<s3 key>, ...]
    }
"""

from __future__ import annotations

import logging
import os
import sys

# Repo root on sys.path so ``from evals.perturbation import ...`` resolves
# under Lambda's task layout (mirrors lambda/eval_rolling_mean_handler.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch

_install_ls_patch()

from nousergon_lib.logging import monitor_handler, setup_logging

_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "perturbation_battery",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-perturbation-battery",
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors the other eval-judge
    handlers — Lambda init phase 10s ceiling."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
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
    """Run the weekly perturbation battery and emit the sensitivity scorecard."""
    _ensure_init()

    event = event or {}
    run_date = event.get("run_date")
    dry_run_llm = bool(event.get("dry_run_llm", False))

    logger.info(
        "[perturbation_battery_handler] start run_date=%s dry_run_llm=%s",
        run_date or "(today UTC)", dry_run_llm,
    )

    # The battery is entirely live-LLM (judge the reference + each corrupted
    # variant). A dry run has nothing meaningful to compute, so skip it wholly
    # rather than write a hollow S3 artifact that would mask real drift.
    if dry_run_llm:
        logger.info("[perturbation_battery_handler] dry_run_llm — skipping live battery")
        return {"status": "SKIPPED_DRY_RUN"}

    # Deferred until after _ensure_init in case the battery pulls SSM-loaded
    # config (the Anthropic key).
    from evals.perturbation import emit_perturbation_report

    try:
        report = emit_perturbation_report(report_date=run_date)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[perturbation_battery_handler] battery failed hard")
        return {"status": "ERROR", "error": str(exc)}

    logger.info(
        "[perturbation_battery_handler] done caught=%d/%d model=%s keys=%d",
        report["n_caught"], report["n"], report["judge_model"],
        len(report.get("report_keys", [])),
    )
    return {
        "status": "OK",
        "n": report["n"],
        "n_caught": report["n_caught"],
        "caught_rate": report["caught_rate"],
        "judge_model": report["judge_model"],
        "report_keys": report.get("report_keys", []),
    }
