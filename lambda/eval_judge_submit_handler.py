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
    "eval_judge_submit",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """Mirror lambda/handler.py's deferred-init pattern."""
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
    from evals.lambda_dry import dry_submit_result, is_dry
    from evals.orchestrator import (
        DEFAULT_HAIKU_MODEL,
        DEFAULT_SONNET_MODEL,
        build_batch_plan,
        submit_batch,
        _persist_client_side_skips,
    )

    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
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
    haiku_model = event.get("haiku_model", DEFAULT_HAIKU_MODEL)
    sonnet_model = event.get("sonnet_model", DEFAULT_SONNET_MODEL)
    judge_only = bool(event.get("judge_only", False))
    # Optional family-selection params (config#1579 P2): judge daily
    # producers (thinktank) whose artifacts land in weekday partitions.
    extra_dates = event.get("extra_dates") or None
    agent_id_prefixes = event.get("agent_id_prefixes") or None
    # capture_lookback_days=N expands to extra_dates covering the N days
    # before `date` — the Saturday SF passes 6 so the week's thinktank
    # captures ride the SAME weekly batch (unmapped agent_ids in those
    # partitions are skipped cleanly; no prior Saturday falls in range).
    lookback = int(event.get("capture_lookback_days") or 0)
    if lookback > 0:
        from evals.orchestrator import expand_lookback_dates

        computed = expand_lookback_dates(date, lookback)
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

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        submit_result = submit_batch(plan, anthropic_client=client, s3_client=s3)
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

    return {
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
        },
    }
