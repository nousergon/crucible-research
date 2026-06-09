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

``OK`` = no failures. ``PARTIAL`` = at least one batch result failed
but the run completed. ``ERROR`` = the run itself blew up. Mirrors
the legacy single-Lambda contract so the Saturday SF + dashboard
result inspectors keep working unchanged.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch
_install_ls_patch()

from alpha_engine_lib.logging import monitor_handler, setup_logging
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
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


@monitor_handler
def handler(event, context):
    _ensure_init()

    import anthropic

    from config import ANTHROPIC_API_KEY
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

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        summary = process_batch_results(
            batch_id=batch_id,
            plan_s3_key=plan_s3_key,
            bucket=bucket,
            anthropic_client=client,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[eval_judge_process_handler] process failed hard")
        return {"status": "ERROR", "batch_id": batch_id, "error": str(exc)}

    status = "PARTIAL" if summary["failed"] else "OK"
    logger.info(
        "[eval_judge_process_handler] done status=%s haiku=%d sonnet=%d "
        "skipped_unmapped=%d skipped_empty_input=%d failed=%d",
        status,
        summary["haiku_evaluated"],
        summary["sonnet_evaluated"],
        summary["skipped_unmapped"],
        summary["skipped_empty_input"],
        len(summary["failed"]),
    )
    return {"status": status, "summary": summary}
