"""Shared shell-run dry-path helper for the eval-judge chain +
rationale-clustering Lambdas.

The Saturday Step Function (alpha-engine-data step_function.json) has a
Friday-PM "shell run" keystone that boots + dry-executes every workload
state to catch import / bootstrap / lib-pin / transport breakage WITHOUT
paying for Anthropic tokens or polluting prod S3 / CloudWatch.

Research's main handler already routes dry via the ``dry_run_llm`` event
flag (``dry_run_llm.$: $.research_dry``). The eval-judge chain
(Submit → Poll → Process) and the rationale-clustering Lambda had NO
clean no-write dry path, so the keystone had to hard-SKIP them
(``skip_eval_judge`` / ``skip_rationale_clustering``) — meaning their
import/bootstrap paths were NOT exercised on the Friday smoke.

This module is the single canonical dry substrate for those four
handlers (no per-handler bespoke logic, no copy-paste). It deliberately
does NOT pull in ``dry_run.py`` (that module installs LangGraph agent
stubs — irrelevant here; these handlers never build the research graph).

Contract:
- ``DRY_FLAG`` is the EXACT event key the SF keystone passes. It is the
  SAME key Research uses (``dry_run_llm``); the SF rewire will pass
  ``"dry_run_llm.$": "$.research_dry"`` to these states.
- ``is_dry(event)`` — truthy iff the keystone requested a dry pass.
- The submit handler returns ``dry_submit_result(date)`` — a sentinel
  whose ``batch_id`` is ``DRY_SENTINEL_BATCH_ID`` and whose
  ``status`` is ``"EMPTY"`` + ``processing_status`` ``"ended_empty"``.
  The existing ``EvalJudgePollChoice`` SF Choice already routes
  ``status == "EMPTY"`` straight to ``EvalJudgeProcess`` (skipping the
  poll loop entirely), so the sentinel threads submit → process with no
  Anthropic poll in between.
- Poll + Process detect the sentinel (or the raw event flag) and
  short-circuit to a benign success WITHOUT any Anthropic call or S3
  read/write.
- Rationale clustering boots + imports, then returns
  ``dry_clustering_result()`` BEFORE any read/cluster/persist.

Hard invariant for every helper here: zero Anthropic/LLM calls, zero
S3/CloudWatch writes, returns a status the SF treats as OK.
"""

from __future__ import annotations

from typing import Any

# ── Canonical event flag ─────────────────────────────────────────────
# MUST match the SF keystone's chosen Lambda dry flag verbatim. Research
# (the only pre-existing clean-dry Lambda state) uses ``dry_run_llm``
# (step_function.json: ``"dry_run_llm.$": "$.research_dry"``). The
# eval-judge chain + rationale clustering are LLM-call-bearing, so they
# adopt the SAME ``dry_run_llm`` key — the SF rewire follow-on will pass
# ``"dry_run_llm.$": "$.research_dry"`` to these five states.
DRY_FLAG = "dry_run_llm"

# Sentinel batch_id that threads submit → (poll) → process. Chosen so it
# is NOT a real Anthropic ``msgbatch_`` id and NOT the existing
# ``empty-{date}`` empty-plan sentinel, so the two cases stay distinct
# in logs / dashboards.
DRY_SENTINEL_BATCH_ID = "dry-run-no-batch"

# Sentinel plan key. NOT ``None`` — see ``dry_submit_result``. A null here is
# a States.Runtime waiting for the Friday shell run.
DRY_SENTINEL_PLAN_S3_KEY = "dry-run-no-plan"


def is_dry(event: Any) -> bool:
    """True iff the SF keystone requested a shell-run dry pass.

    Accepts the canonical ``dry_run_llm`` flag. Defensively also honors
    a sentinel batch_id already on the event (so Poll/Process treat a
    sentinel threaded from Submit as dry even if the SF only set the
    flag on the Submit state).
    """
    if not isinstance(event, dict):
        return False
    if bool(event.get(DRY_FLAG, False)):
        return True
    if event.get("batch_id") == DRY_SENTINEL_BATCH_ID:
        return True
    return False


def dry_submit_result(date: str) -> dict[str, Any]:
    """eval_judge_submit dry return — NO Anthropic batch, NO S3 plan
    persist. Shaped so the existing SF Choice (`EvalJudgePollChoice`,
    ``status == "EMPTY"`` → `EvalJudgeProcess`) skips the poll loop and
    Process sees the sentinel batch_id.

    **Nothing here may be ``None`` (alpha-engine-config-I9329).**
    ``plan_s3_key`` used to be null. Once ``EvalJudgeProcess`` runs on a spot
    box, the Friday shell-run path threads ``plan_s3_key`` into a
    ``States.Format`` intrinsic on the new ``ssm:sendCommand`` stage, and
    ``States.Format`` raises ``States.Runtime`` on a null argument. That is a
    definition which works on Saturday and dies on Friday — the exact class
    ``dry_process_result``'s existing comment already names for its own hoisted
    keys. ``DRY_SENTINEL_PLAN_S3_KEY`` is a non-null string that is obviously
    not a real key, so the intrinsic resolves and a reader of the rendered
    command sees immediately that it was the dry path.
    """
    return {
        "status": "EMPTY",
        "batch_id": DRY_SENTINEL_BATCH_ID,
        "plan_s3_key": DRY_SENTINEL_PLAN_S3_KEY,
        "request_count": 0,
        "processing_status": "ended_empty",
        "dry_run": True,
        "submit_summary": {
            "date": date,
            "capture_keys_total": 0,
            "skipped_unmapped": 0,
            "skipped_empty_input_persisted": 0,
            "skipped_degenerate_input_persisted": 0,
            "skip_failed": [],
            "force_sonnet_pass": False,
            "judge_only": False,
            "dry_run": True,
        },
    }


def dry_poll_result(batch_id: str | None = None) -> dict[str, Any]:
    """eval_judge_poll dry return — NO Anthropic retrieve. Terminal
    ``ended`` so the SF Poll Choice routes forward to Process.
    """
    return {
        "batch_id": batch_id or DRY_SENTINEL_BATCH_ID,
        "processing_status": "ended",
        "request_counts": {
            "processing": 0, "succeeded": 0, "errored": 0,
            "canceled": 0, "expired": 0,
        },
        "ended_at": None,
        "elapsed_seconds": 0,
        "exceeded_max_wait": False,
        "dry_run": True,
    }


def dry_process_result(batch_id: str | None = None) -> dict[str, Any]:
    """eval_judge_process dry return — NO Anthropic results stream, NO
    S3 plan read, NO per-artifact persist, NO CW emit. ``status: OK``
    mirrors the legacy single-Lambda contract the SF treats as success.
    """
    return {
        "status": "OK",
        "dry_run": True,
        # I6920: the real path hoists these to the top level so a Step
        # Functions Choice can branch on coverage. The dry path must carry
        # the same keys or that Choice's JSONPath is unresolvable on the
        # Friday preflight run — a definition that works on Saturday and
        # breaks on Friday. `dry_run: True` alongside is what tells a
        # reader this completeness is the keystone's, not a corpus's.
        "complete": True,
        "budget_stopped": False,
        "summary": {
            "batch_id": batch_id or DRY_SENTINEL_BATCH_ID,
            "haiku_evaluated": 0,
            "sonnet_evaluated": 0,
            "skipped_unmapped": 0,
            "skipped_empty_input": 0,
            "failed": [],
            "persisted_keys": [],
            "dry_run": True,
        },
    }


def dry_clustering_result() -> dict[str, Any]:
    """rationale_clustering dry return — boot/import done; returns
    BEFORE ``compute_and_emit``, which is now itself a no-op RETIRED
    stub (alpha-engine-config-I8173: every decision_artifacts/{date}/{agent}
    family it used to read lost its producer to I7827/PR685). ``status:
    OK`` is what the SF treats as success.
    """
    return {
        "status": "OK",
        "dry_run": True,
        "summary": {
            "agents_analyzed": 0,
            "agents_skipped_thin_sample": [],
            "load_failures": [],
            "cluster_failures": [],
            "per_agent": [],
            "dry_run": True,
        },
    }
