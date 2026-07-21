"""Lambda entry point — research think-tank run (config#1579 P1).

Shares the main runner's ECR image with a CMD override to
``thinktank_handler.handler`` (the established image-share pattern —
eval_judge / scanner / rationale_clustering). Two invocation sources
(2026-07-14 cadence design):

1. EventBridge rule ``alpha-research-thinktank-daily`` (14:30 UTC, 7
   days/week — after the weekday SF's RunDailyNews state lands the
   day's news aggregates, and after the Saturday SF's fresh weekly
   artifacts so the themes layer reconciles the same day). Plain event
   (no ``mode``) — grows coverage toward the full ``rank_ceiling=150``
   universe at ``daily_new_names``/day and handles staleness refresh
   gradually. Weekend/holiday runs are by-design: captures + events
   partition to the last trading day (thinktank/capture.py), so they
   accrue into Friday's partition.
2. The Saturday weekly SF's ``ThinkTankCoverage`` state, ``mode=gap_fill``
   — a narrow, reactive top-up: once the fresh weekly scan lands, shore
   up whatever of the CURRENT top-60 the daily cadence hasn't caught up
   to yet. Sized to the exact measured gap (see ``thinktank/run.py``'s
   ``GAP_FILL_TOP_N``/``gap_fill_only``), never a fixed constant, never
   padded with stale-refill — kept small and bounded regardless of how
   large the full-universe backlog gets.

Failure contract — RAISE, never return an ERROR dict, for BOTH
invocation sources. For the EventBridge async invoke, an ERROR-dict
return is a *successful* invocation — the AWS/Lambda Errors metric stays
flat, no retry fires, and the failure is silent (exactly the
no-silent-fails failure mode); raising instead drives the Errors metric
that ``infrastructure/setup-thinktank-schedule.sh``'s alarm watches and
engages EventBridge's two built-in async retries. For the SF's
``arn:aws:states:::lambda:invoke`` Task, the Catch only triggers on an
actual raised Lambda error — a normal return value (even an error-shaped
dict) is a *successful* Task completion and would never route through
the non-blocking Catch. Either way, a retried run re-selects intake
against the ledger written so far; the worst case is a duplicate thesis
version (never a silent skip, since the coverage ledger only persists
once at the end of a run — a mid-run failure never partially commits),
and the SSM budget guard caps spend.

Event shape (all fields optional):

    {
      "dry_run_llm": true,        # shell-run smoke: boot + imports only, no S3
      "dry_run": true,            # plan-only: intake selection, no LLM/writes
      "refresh_tickers": ["X"],   # operator refresh of covered names only
      "mode": "gap_fill"          # Saturday SF: top-60 gap-fill (see above)
    }

Returns ``{"status": "OK", "manifest": {...}}`` on success (or the
dry-path variants); raises on any failure.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

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
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

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

    from thinktank.run import run_daily

    plan_only = bool(event.get("dry_run")) if isinstance(event, dict) else False
    # Operator refresh: {"refresh_tickers": ["MNST", ...]} re-underwrites
    # ONLY those covered names (no intake/sweep/themes) — the ad-hoc
    # re-underwrite / rating-backfill knob. Absent on scheduled events.
    refresh = event.get("refresh_tickers") if isinstance(event, dict) else None

    # Saturday SF gap-fill mode: shores up whatever of the CURRENT top-60
    # the daily cadence hasn't caught up to yet, sized to the exact
    # measured gap (thinktank/run.py's GAP_FILL_TOP_N/gap_fill_only) —
    # never a fixed constant, never padded with stale-refill (that's the
    # daily job's role). Runs observe-only — writes to thinktank/ S3
    # prefix for validation tracking; does NOT gate the Predictor.
    gap_fill_only = isinstance(event, dict) and event.get("mode") == "gap_fill"
    manifest = run_daily(
        dry_run=plan_only, refresh_tickers=refresh, gap_fill_only=gap_fill_only
    )

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
