"""Lambda entry point — daily research think-tank run (config#1579 P1).

Shares the main runner's ECR image with a CMD override to
``thinktank_handler.handler`` (the established image-share pattern —
eval_judge / scanner / rationale_clustering). Invoked by the EventBridge
rule ``alpha-research-thinktank-daily`` (14:30 UTC, 7 days/week —
after the weekday SF's RunDailyNews state lands the day's news
aggregates, and after the Saturday SF's fresh weekly artifacts so the
themes layer reconciles the same day). Weekend/holiday runs are
by-design: captures + events partition to the last trading day
(thinktank/capture.py), so they accrue into Friday's partition.

Failure contract — RAISE, never return an ERROR dict. The SF-invoked
handlers in this repo return ``{"status": "ERROR"}`` for their SF Catch
states; this Lambda has no SF above it. For an EventBridge async invoke
an ERROR-dict return is a *successful* invocation — the AWS/Lambda
Errors metric stays flat, no retry fires, and the failure is silent
(exactly the no-silent-fails failure mode). Raising instead (a) drives
the Errors metric that ``infrastructure/setup-thinktank-schedule.sh``'s
alarm watches, and (b) engages EventBridge's two built-in async retries,
so a transient provider blip self-heals. A retried run re-selects intake
against the ledger written so far; the worst case is a duplicate thesis
version (never a silent skip), and the SSM budget guard caps spend.

Event shape (all fields optional):

    {
      "dry_run_llm": true,        # shell-run smoke: boot + imports only, no S3
      "dry_run": true,            # plan-only: intake selection, no LLM/writes
      "refresh_tickers": ["X"]    # operator refresh of covered names only
    }

Returns ``{"status": "OK", "manifest": {...}}`` on success (or the
dry-path variants); raises on any failure.
"""

from __future__ import annotations

import logging
import os
import sys

# Repo root on sys.path so ``from thinktank.run import ...`` resolves under
# Lambda's task layout. Mirrors the existing shared-image handlers
# (scanner, rationale_clustering, eval_rolling_mean, aggregate_costs).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from alpha_engine_lib.logging import monitor_handler, setup_logging

_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging("thinktank", flow_doctor_yaml=_FLOW_DOCTOR_YAML)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """One-time cold-start hydration.

    The lib RAG layer (``nousergon_lib.rag``) reads ``RAG_DATABASE_URL``
    and ``VOYAGE_API_KEY`` from ``os.environ`` directly, so hydrate both
    from SSM via the ``get_secret`` chokepoint rather than baking them
    into the function's env-var config (the post-.env-retirement posture;
    the main runner's function-level env vars predate it). BOTH are
    required together: the availability probe only checks the DB, so a
    present DB URL with a missing Voyage key passes the probe and then
    fails every per-ticker retrieve — exactly the 2026-07-02 first-run
    gotcha this hydration closes.

    Decision capture is unconditionally ON for this producer: judge
    coverage of thinktank theses/themes (crucible-research#358) depends
    on every run emitting DecisionArtifacts — a run without capture is
    invisible to the eval layer.
    """
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    from alpha_engine_lib.secrets import get_secret

    for name in ("RAG_DATABASE_URL", "VOYAGE_API_KEY"):
        if not os.environ.get(name):
            os.environ[name] = get_secret(name)

    os.environ.setdefault("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    _init_done = True


@monitor_handler
def handler(event, context):
    """Run the daily think-tank cycle. Raises on failure (see module doc)."""
    from evals.lambda_dry import is_dry

    # Shell-run dry path — boot + imports above already exercised the
    # bootstrap smoke. Return BEFORE secrets hydration / any S3 access.
    if is_dry(event):
        logger.info(
            "[thinktank_handler] dry_run_llm=True: shell-run no-op "
            "(no secrets fetch, no S3 read/write, no LLM calls)",
        )
        return {"status": "OK", "dry_run": True}

    _ensure_init()

    from thinktank.run import run_daily

    plan_only = bool(event.get("dry_run")) if isinstance(event, dict) else False
    # Operator refresh: {"refresh_tickers": ["MNST", ...]} re-underwrites
    # ONLY those covered names (no intake/sweep/themes) — the ad-hoc
    # re-underwrite / rating-backfill knob. Absent on scheduled events.
    refresh = event.get("refresh_tickers") if isinstance(event, dict) else None
    manifest = run_daily(dry_run=plan_only, refresh_tickers=refresh)

    logger.info(
        "[thinktank_handler] done run_id=%s mode=%s trading_day=%s "
        "theses=%d sweep=%d theme_updates=%d cost=$%.4f month=$%.2f/$%.2f",
        manifest.run_id,
        manifest.mode,
        manifest.trading_day,
        manifest.theses_written,
        manifest.sweep_tickers,
        manifest.theme_updates_written,
        manifest.total_cost_usd,
        manifest.budget_month_spent_usd,
        manifest.budget_month_limit_usd,
    )
    return {"status": "OK", "manifest": manifest.model_dump()}
