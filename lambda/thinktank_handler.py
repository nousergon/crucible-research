"""Lambda entry point — research think-tank run (config#1579 P1).

Shares the main runner's ECR image with a CMD override to
``thinktank_handler.handler`` (the established image-share pattern —
eval_judge / scanner / rationale_clustering).

Invocation, consolidated 2026-07-14 (alpha-engine-config-I2487 incident +
SOTA follow-up): invoked ONLY by the Saturday weekly SF's
``ThinkTankCoverage`` state, ``mode=sf_cover``. The prior standalone
EventBridge rule (``alpha-research-thinktank-daily``, 7 days/week, then
briefly ``alpha-research-thinktank-maintenance`` at Mon/Wed/Fri) is
retired entirely — it was pure duplicated work: ``sf_cover`` mode's
``run_daily()`` already performs theme reconciliation + a full events
sweep over every covered name unconditionally (the mode only overrides
intake sizing — ``sf_cover_target``/``sf_cover_ceiling``, now
``rank_ceiling=150``, the full universe), and the universe board this
Lambda's intake ranks against (``scanner/universe/latest.json``) is
itself only produced on Saturday (Scanner runs immediately before this
state in the same SF branch) — a weekday invocation never had new
ranking data to act on.

Failure contract — RAISE, never return an ERROR dict. Unlike this repo's
other SF-invoked handlers (which return ``{"status": "ERROR"}`` for
their SF Catch states to inspect via ``ResultPath``), the SF's
``arn:aws:states:::lambda:invoke`` Task integration only triggers its
``Catch`` on an actual Lambda function error (a raised/unhandled
exception) — a normal return value, even one shaped like an error dict,
is treated as a *successful* Task completion and would NOT route through
``ThinkTankCoverage``'s Catch (silent failure, worse than before: the
non-blocking Catch exists precisely to keep the SF branch clean on a
think-tank failure). Raising instead drives the AWS/Lambda Errors metric
(watched by ``infrastructure/setup-thinktank-alarm.sh``'s alarm) and lets
the SF state's own Retry (mirroring the sibling ``Research`` state's
bridge — States.Timeout/Lambda.Unknown, 1 retry) self-heal a transient
blip. A retried run re-selects intake against the ledger written so far;
the worst case is a duplicate thesis version (never a silent skip,
since the coverage ledger is only persisted once at the very end of a
run — a mid-run timeout does not partially commit), and the SSM budget
guard caps spend.

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

from nousergon_lib.logging import monitor_handler, setup_logging

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

    from nousergon_lib.secrets import get_secret

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

    from dataclasses import replace as _replace_settings

    from thinktank.run import run_daily
    from thinktank.settings import load_settings

    plan_only = bool(event.get("dry_run")) if isinstance(event, dict) else False
    # Operator refresh: {"refresh_tickers": ["MNST", ...]} re-underwrites
    # ONLY those covered names (no intake/sweep/themes) — the ad-hoc
    # re-underwrite / rating-backfill knob. Absent on scheduled events.
    refresh = event.get("refresh_tickers") if isinstance(event, dict) else None

    # Saturday SF coverage mode — the ONLY scheduled invocation path
    # (2026-07-14 consolidation): overrides intake to fill ALL uncovered
    # top-N names (ignoring research/thinktank.yaml's base
    # daily_new_names — that base value is a fallback for ad-hoc manual
    # invokes without mode=sf_cover only). Runs observe-only — writes to
    # thinktank/ S3 prefix for validation tracking; does NOT gate the
    # Predictor.
    if isinstance(event, dict) and event.get("mode") == "sf_cover":
        settings = load_settings()
        sf_settings = _replace_settings(
            settings,
            daily_new_names=event.get("sf_cover_target", 150),
            rank_ceiling=event.get("sf_cover_ceiling", 150),
        )
        manifest = run_daily(settings=sf_settings, dry_run=plan_only)
    else:
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
