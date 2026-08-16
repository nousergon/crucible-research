"""Lambda entry point — LLM-as-judge eval pipeline.

Triggered by the Saturday Step Function after Research completes
(captured artifacts under ``decision_artifacts/{Y}/{M}/{D}/`` need to
be in S3 before this Lambda fans out the rubric scoring).

Event shape (all fields optional):

    {
      "date": "2026-05-09",          # YYYY-MM-DD; default = today UTC
      "force_sonnet_pass": false,    # SF passes True every 4th run
      "haiku_model": "claude-haiku-4-5",
      "sonnet_model": "claude-sonnet-4-6",
      "haiku_escalate_threshold": 3,

      # PR 4e test-track flags — both default false:
      "dry_run": false,              # list + render only, no LLM calls,
                                     # no persists, no metrics. $0.
      "judge_only": false            # real LLM calls but isolated outputs:
                                     # writes under decision_artifacts/
                                     #   _eval_judge_only/ and CW namespace
                                     # AlphaEngine/EvalJudgeOnly. Lets us
                                     # validate against captured production
                                     # artifacts without polluting the
                                     # rolling-mean window or dashboard.
    }

Returns:

    {
      "status": "OK" | "PARTIAL" | "ERROR",
      "summary": <orchestrator result dict>,
      ...
    }

``OK`` = no failures. ``PARTIAL`` = at least one artifact failed but
the run completed. ``ERROR`` = the run itself blew up (e.g. listing
failed). The Saturday SF state alarms on PARTIAL/ERROR; eval is
observability so we don't gate the rest of the pipeline on it.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import tempfile

# Repo root on sys.path so ``from evals.orchestrator import ...``
# resolves under Lambda's invocation layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Reuse the same LangSmith pandas-DataFrame patch the main research
# Lambda installs — the eval LLM calls run through the same callback
# tracer surface and would hit the same orjson-on-Timestamp crash
# without it.
from graph.langsmith_pandas_patch import install as _install_ls_patch

_install_ls_patch()

# Structured logging + flow-doctor singleton from alpha-engine-lib,
# matching the main research handler. exclude_patterns kept empty —
# the canonical lib pattern (mirrors lambda/handler.py:58) forces an
# explicit decision once real ERROR-level noise is observed.
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
    "eval_judge",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-eval-judge",
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation.

    Mirrors lambda/handler.py — the Lambda init phase has a 10-second
    hard ceiling; module-top work has bitten us before (cold-start
    INIT_REPORT timeout 2026-04-11). Idempotent."""
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
    """AWS Lambda handler — fans the judge module over captured
    artifacts for ``event["date"]`` and persists the eval results."""
    _ensure_init()

    # Imports deferred until after _ensure_init so any module that
    # reads ANTHROPIC_API_KEY at import time sees the SSM-loaded value.
    from evals.orchestrator import (
        DEFAULT_HAIKU_ESCALATE_THRESHOLD,
        DEFAULT_HAIKU_MODEL,
        DEFAULT_SONNET_MODEL,
        evaluate_corpus,
    )

    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    date = event.get("date") or str(datetime.date.today())
    force_sonnet_pass = bool(event.get("force_sonnet_pass", False))
    haiku_model = event.get("haiku_model", DEFAULT_HAIKU_MODEL)
    sonnet_model = event.get("sonnet_model", DEFAULT_SONNET_MODEL)
    haiku_escalate_threshold = int(
        event.get("haiku_escalate_threshold", DEFAULT_HAIKU_ESCALATE_THRESHOLD)
    )
    dry_run = bool(event.get("dry_run", False))
    judge_only = bool(event.get("judge_only", False))

    logger.info(
        "[eval_judge_handler] start date=%s force_sonnet_pass=%s "
        "haiku_model=%s sonnet_model=%s threshold=%d "
        "dry_run=%s judge_only=%s",
        date, force_sonnet_pass, haiku_model, sonnet_model,
        haiku_escalate_threshold, dry_run, judge_only,
    )

    try:
        summary = evaluate_corpus(
            date=date,
            bucket=bucket,
            haiku_model=haiku_model,
            sonnet_model=sonnet_model,
            force_sonnet_pass=force_sonnet_pass,
            haiku_escalate_threshold=haiku_escalate_threshold,
            dry_run=dry_run,
            judge_only=judge_only,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[eval_judge_handler] orchestrator failed hard")
        return {
            "status": "ERROR",
            "date": date,
            "error": str(exc),
        }

    status = "PARTIAL" if summary["failed"] else "OK"
    logger.info(
        "[eval_judge_handler] done status=%s haiku=%d sonnet=%d "
        "skipped=%d skipped_empty=%d failed=%d",
        status,
        summary["haiku_evaluated"],
        summary["sonnet_evaluated"],
        summary["skipped_unmapped"],
        summary.get("skipped_empty_input", 0),
        len(summary["failed"]),
    )
    return {"status": status, "summary": summary}
