"""Lambda entry point — standalone scanner producing ``candidates.json``.

ROADMAP L1995 Phase 1 (plan doc:
``~/Development/alpha-engine-docs/private/scanner-rag-resequence-260524.md``).

Splits the quant scanner out of the Research Lambda into a dedicated
Saturday-SF state, writing ``s3://alpha-engine-research/candidates/
{run_date}/candidates.json`` so RAGIngestion (Phase 4) can ingest fresh
context for *every* stock the agents will evaluate — including the
~10-20 new picks each week the prior-signals.json-keyed RAG ingest
misses today.

Phase 1 posture: this Lambda exists + can be invoked (Phase 2 inserts
the SF state, gated default-off; operator flips the flag for Phase 3
soak). The Research Lambda still runs its own internal scanner — the
new artifact is written in parallel-observe mode, no consumer reads
it yet. Phase 5 will cut Research over to read this artifact + retire
the internal scanner.

Event shape (all fields optional except ``run_date``):

    {
      "run_date": "2026-05-30",          # ISO YYYY-MM-DD (required)
      "bucket": "alpha-engine-research", # default RESEARCH_BUCKET env
      "market_regime": "neutral",        # default "neutral"
      "dry_run_llm": false,              # shell-run dry path
    }

Returns one of:

    {"status": "OK", "summary": {...}}                — artifact written
    {"status": "ERROR", "error": "<msg>"}             — hard failure caught

Cost contract mirrors data #295 + L3277 audit: the scanner Lambda
never returns SKIPPED today (constituents.json + feature store are
hard preconditions; their absence raises). The status allowlist
nonetheless includes SKIPPED for forward-compat if Phase 5 ever adds
legitimate no-op paths (e.g. holiday short cycles).
"""

from __future__ import annotations

import logging
import os
import sys

# Repo root on sys.path so ``from data.scanner_orchestrator import ...``
# resolves under Lambda's task layout. Mirrors the existing handlers
# (rationale_clustering, eval_rolling_mean, aggregate_costs).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch
_install_ls_patch()

from alpha_engine_lib.logging import setup_logging
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "scanner",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors the other
    shared-image handlers — Lambda init phase 10s ceiling."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


def handler(event, context):
    """Produce the candidates.json artifact for ``event['run_date']``."""
    _ensure_init()

    import boto3
    from evals.lambda_dry import is_dry
    from data.scanner_orchestrator import (
        build_candidates_artifact,
        write_candidates_artifact,
        ScannerOrchestratorError,
    )

    # Shell-run dry path — boot + imports above already exercised the
    # bootstrap smoke. Return BEFORE the orchestrator (which reads
    # constituents + feature store + writes S3). dry_run_llm short-
    # circuits everything for Friday-Preflight shell runs.
    if is_dry(event):
        logger.info(
            "[scanner_handler] dry_run_llm=True: shell-run no-op "
            "(no S3 read/write, no scanner pass)",
        )
        return {"status": "OK", "dry_run": True}

    run_date = event.get("run_date")
    if not run_date:
        logger.error(
            "[scanner_handler] event missing required 'run_date' field"
        )
        return {
            "status": "ERROR",
            "error": "event missing required 'run_date' field (ISO YYYY-MM-DD)",
        }

    # Lenient parse — accept anything the orchestrator's S3 key
    # template accepts (the orchestrator doesn't itself parse run_date
    # as a date, just slots it into the partition prefix).
    if not isinstance(run_date, str) or len(run_date) < 10:
        logger.error(
            "[scanner_handler] invalid run_date %r — expected ISO YYYY-MM-DD",
            run_date,
        )
        return {
            "status": "ERROR",
            "error": f"invalid run_date {run_date!r}: expected ISO YYYY-MM-DD",
        }

    # ── Trading-day normalization (DATE_CONVENTIONS) ─────────────────────────
    # Every trade artifact in the system keys by the TRADING DAY, not the
    # calendar date: signals.json, sector_team_runs, scanner_evaluations, and
    # the Research run itself all key off most_recent_trading_day(today). The
    # Saturday SF passes a CALENDAR run_date (date(Execution.StartTime)) — e.g.
    # 2026-05-30 (Sat) — while Research keys off Friday 2026-05-29. candidates.json
    # MUST land on the same trading-day key or the Phase-5 consumer (Research
    # fetch_data → load_candidates_json) can't find it. The 2026-05-30 L4464
    # recovery failed exactly here: Scanner wrote candidates/2026-05-30/, Research
    # read candidates/2026-05-29/. Normalize at the producer to the canonical
    # trading-day axis (lib chokepoint), preserving on-or-before semantics so an
    # explicit operator backfill date is normalized too.
    import datetime as _dt
    from alpha_engine_lib import trading_calendar as _tc
    _cal = _dt.date.fromisoformat(run_date[:10])
    _td = _cal if _tc.is_trading_day(_cal) else _tc.previous_trading_day(_cal)
    _trading_day = _td.isoformat()
    if _trading_day != run_date[:10]:
        logger.info(
            "[scanner_handler] normalized run_date %s (calendar) → %s (trading "
            "day) per DATE_CONVENTIONS — candidates.json keys by trading day to "
            "match Research + signals.json",
            run_date, _trading_day,
        )
    run_date = _trading_day

    bucket = event.get("bucket", _DEFAULT_BUCKET)
    market_regime = event.get("market_regime", "neutral")

    logger.info(
        "[scanner_handler] start run_date=%s (trading day) bucket=%s market_regime=%s",
        run_date, bucket, market_regime,
    )

    s3_client = boto3.client("s3")

    try:
        artifact = build_candidates_artifact(
            run_date=run_date,
            s3_client=s3_client,
            bucket=bucket,
            market_regime=market_regime,
        )
    except ScannerOrchestratorError as exc:
        # Hard precondition failure (constituents missing, feature store
        # empty). The orchestrator already logged the cause; surface as
        # ERROR so the SF Catch handles it.
        logger.error("[scanner_handler] orchestrator precondition failed: %s", exc)
        return {"status": "ERROR", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("[scanner_handler] orchestrator failed hard")
        return {"status": "ERROR", "error": str(exc)}

    try:
        s3_key = write_candidates_artifact(
            artifact, s3_client=s3_client, bucket=bucket,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[scanner_handler] S3 write failed hard")
        return {"status": "ERROR", "error": f"S3 write failed: {exc}"}

    summary = {
        "s3_key": s3_key,
        "scanner_tickers": len(artifact["scanner_tickers"]),
        "population_tickers": len(artifact["population_tickers"]),
        "agent_input_set": len(artifact["agent_input_set"]),
        "new_vs_prior_cycle": len(artifact["stats"]["new_vs_prior_cycle"]),
        "dropped_vs_prior_cycle": len(
            artifact["stats"]["dropped_vs_prior_cycle"]
        ),
        "baseline_missing": artifact["stats"]["baseline_missing"],
    }

    logger.info(
        "[scanner_handler] done run_date=%s scanner_tickers=%d "
        "population=%d new=%d dropped=%d",
        run_date,
        summary["scanner_tickers"],
        summary["population_tickers"],
        summary["new_vs_prior_cycle"],
        summary["dropped_vs_prior_cycle"],
    )
    return {"status": "OK", "summary": summary, "date": run_date}
