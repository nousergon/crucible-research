"""Bounded-parallel gap_fill fan-out (config#3072, Brian's 2026-07-20 ruling
— component C of the ratified fix).

The Saturday SF's ``ThinkTankCoverage`` state used to be ONE sequential
Lambda invocation (``thinktank.run.run_daily(gap_fill_only=True)``) bounded
by the Lambda's 900s hard ceiling: ~2.5-3.5 min per thesis (pillar tier)
meant only ~4 theses fit per attempt, and 3 SF retries restarted the SAME
first names every time (fixed by the checkpointing in ``thinktank.run``,
but that alone still runs sequentially inside one Lambda).

This module splits gap_fill into three phases an SF Map state wires as:

    ThinkTankGapFillPlan (Task)
      -> ThinkTankGapFillBuild (Map, MaxConcurrency=N, one Lambda
         invocation per ticker)
      -> ThinkTankGapFillFinalize (Task)

Each Map iteration is its own Lambda invocation, still subject to the SAME
900s ceiling per UNIT, but the batch's overall wall-clock is no longer
sum(theses) — it collapses toward ~slowest-single-thesis-time *
ceil(gap / MaxConcurrency), with per-unit failure isolated by the Map
state's ``ToleratedFailurePercentage`` instead of one slow/failing ticker
restarting the whole batch.

Race safety: PLAN and FINALIZE are sequential (one Lambda invocation each)
and are the ONLY phases that touch the shared coverage ledger / monthly
cost ledger. BUILD workers never touch either — each writes its own,
uniquely-keyed checkpoint object (``GAP_FILL_CHECKPOINT_KEY_TMPL``), so
concurrent BUILD invocations have zero write contention (a shared
load-mutate-save on one S3 object across concurrent workers would lose
updates; per-ticker keys structurally can't collide). FINALIZE (run
strictly after the Map completes) reads every checkpoint for the run's
trading_day and merges them into the ledger + cost ledger sequentially —
the same pattern ``thinktank.run`` uses per-unit, just applied once, after
the fan-out, to the whole batch.

Idempotency: BUILD checks for an existing checkpoint before spending an
LLM call — a Map-iteration-level SF retry (transient Lambda error) never
re-bills or re-does an already-completed unit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from nousergon_lib.dates import now_dual

from thinktank import GAP_FILL_CHECKPOINT_KEY_TMPL
from thinktank.analyst import build_thesis, sweep
from thinktank.challenger_selection import write_challenger_selection
from thinktank.client import ThinktankClient
from thinktank.context import load_context
from thinktank.costs import BudgetGuard
from thinktank.ledger import (
    load_ledger,
    ranked_universe,
    record_sweep,
    record_thesis_write,
    save_ledger,
    select_intake,
)
from thinktank.ratings import update_ratings_board
from thinktank.run import GAP_FILL_TOP_N, _compute_coverage_gap
from thinktank.schemas import EventRecord, RunManifest
from thinktank.settings import ThinktankSettings, load_settings
from thinktank.storage import ThinktankStore
from thinktank.themes import ThemeKeeper

logger = logging.getLogger(__name__)


def _board_row_with_rank(board: dict, ticker: str) -> dict | None:
    for rank, row in enumerate(ranked_universe(board), start=1):
        if row.get("ticker") == ticker:
            row = dict(row)
            row["_attractiveness_rank"] = rank
            return row
    return None


def _checkpoint_key(trading_day: str, ticker: str) -> str:
    return GAP_FILL_CHECKPOINT_KEY_TMPL.format(trading_day=trading_day, ticker=ticker)


# ── phase 1: plan ────────────────────────────────────────────────────────────


def plan_gap_fill(
    settings: ThinktankSettings | None = None,
    *,
    store: ThinktankStore | None = None,
    client: ThinktankClient | None = None,
    ssm_client=None,
    run_id: str,
) -> dict:
    """Compute the gap_fill ticker set + settle themes ONCE before fan-out.

    Returns the plan the SF Map state iterates over (``tickers``) plus the
    run identity every BUILD/FINALIZE invocation must agree on.
    """
    settings = settings or load_settings()
    store = store or ThinktankStore(settings.bucket)
    dual = now_dual()
    trading_day, calendar_date = str(dual.trading_day), str(dual.calendar_date)

    guard = BudgetGuard(store, settings, ssm_client=ssm_client)
    guard.check(calendar_date)  # hard refusal at the monthly cap, unchanged

    ctx = load_context(store)
    if ctx.board is None:
        raise RuntimeError(
            "universe board (scanner/universe/latest.json) is missing — the "
            "think tank has no intake without it; aborting loudly."
        )
    ledger = load_ledger(store)
    gap = _compute_coverage_gap(ctx.board, ledger, top_n=GAP_FILL_TOP_N)
    gap_count = gap.get("uncovered_count", 0)
    new_rows, _ = select_intake(
        ledger, ctx.board, daily_new_names=gap_count,
        rank_ceiling=GAP_FILL_TOP_N, skip_stale_refill=True,
    )
    tickers = [r["ticker"] for r in new_rows]

    # Settle themes ONCE, sequentially, before fan-out — every BUILD worker
    # reads (never writes) macro/sector theme state via ThemeKeeper, so it
    # must already be current by the time the Map starts.
    client = client or ThinktankClient(settings=settings, run_id=run_id)
    themes = ThemeKeeper(store, client, ctx, trading_day=trading_day, calendar_date=calendar_date)
    themes.ensure_current()
    if client.total_cost_usd():
        guard.record_run(
            calendar_date, run_id=f"{run_id}-plan", trading_day=trading_day,
            cost_usd=client.total_cost_usd(),
        )

    return {
        "run_id": run_id,
        "trading_day": trading_day,
        "calendar_date": calendar_date,
        "tickers": tickers,
        "coverage_gap": gap,
    }


# ── phase 2: build (one Map iteration = one ticker) ─────────────────────────


def build_gap_fill_unit(
    settings: ThinktankSettings | None = None,
    *,
    store: ThinktankStore | None = None,
    client: ThinktankClient | None = None,
    run_id: str,
    trading_day: str,
    calendar_date: str,
    ticker: str,
) -> dict:
    """Build ONE ticker's gap_fill thesis and checkpoint it. Idempotent:
    a checkpoint already on file for (trading_day, ticker) short-circuits
    without another LLM call."""
    settings = settings or load_settings()
    store = store or ThinktankStore(settings.bucket)

    existing = store.get_json(_checkpoint_key(trading_day, ticker))
    if existing is not None:
        logger.info(
            "gap_fill unit %s already checkpointed for %s — skipping",
            ticker, trading_day,
        )
        return existing

    ctx = load_context(store)
    if ctx.board is None:
        raise RuntimeError(
            "universe board missing mid-fan-out — aborting loudly rather "
            "than checkpointing a thesis built without it."
        )
    board_row = _board_row_with_rank(ctx.board, ticker)

    client = client or ThinktankClient(settings=settings, run_id=run_id)
    themes = ThemeKeeper(store, client, ctx, trading_day=trading_day, calendar_date=calendar_date)

    thesis = build_thesis(
        store, client, ctx, themes,
        ticker=ticker,
        board_row=board_row,
        trading_day=trading_day,
        calendar_date=calendar_date,
        update_reason="initial",
    )
    checkpoint = {
        "ticker": ticker,
        "trading_day": trading_day,
        "thesis_version": thesis.version,
        "sector": thesis.sector,
        "attractiveness_rank": thesis.attractiveness_rank,
        "total_cost_usd": client.total_cost_usd(),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    store.put_json(_checkpoint_key(trading_day, ticker), checkpoint)
    client.flush_sft(store.s3, store.bucket, trading_day)
    return checkpoint


# ── phase 3: finalize (sequential — the only ledger/cost-ledger writer) ────


def _load_gap_fill_checkpoints(store: ThinktankStore, trading_day: str) -> list[dict]:
    prefix = f"thinktank/_gap_fill_checkpoints/{trading_day}/"
    checkpoints = []
    for key in store.list_keys(prefix):
        cp = store.get_json(key)
        if cp is not None:
            checkpoints.append(cp)
    return sorted(checkpoints, key=lambda c: c["ticker"])


def finalize_gap_fill(
    settings: ThinktankSettings | None = None,
    *,
    store: ThinktankStore | None = None,
    client: ThinktankClient | None = None,
    ssm_client=None,
    run_id: str,
    trading_day: str,
    calendar_date: str,
) -> RunManifest:
    """Merge every gap_fill checkpoint for ``trading_day`` into the coverage
    ledger, then run the same tail ``thinktank.run.run_daily`` runs after
    its intake loop: events sweep, churn-gated daily theme update, ratings
    board, challenger selection, persist, cost ledger, SFT flush."""
    settings = settings or load_settings()
    store = store or ThinktankStore(settings.bucket)
    manifest = RunManifest(
        run_id=run_id,
        mode="gap_fill",
        trading_day=trading_day,
        calendar_date=calendar_date,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    ledger = load_ledger(store)
    covered_before = sorted(ledger.covered())

    checkpoints = _load_gap_fill_checkpoints(store, trading_day)
    build_cost_usd = 0.0
    for cp in checkpoints:
        record_thesis_write(
            ledger,
            ticker=cp["ticker"],
            trading_day=cp["trading_day"],
            thesis_version=cp["thesis_version"],
            sector=cp.get("sector"),
            attractiveness_rank=cp.get("attractiveness_rank"),
        )
        build_cost_usd += cp.get("total_cost_usd", 0.0)
    manifest.names_added = [cp["ticker"] for cp in checkpoints]
    manifest.theses_written = len(checkpoints)
    save_ledger(store, ledger)  # ONE save — finalize is sequential, no race

    ctx = load_context(store)
    manifest.context_sources_present = dict(ctx.sources_present)
    manifest.coverage_gap = _compute_coverage_gap(ctx.board, ledger, top_n=GAP_FILL_TOP_N)

    client = client or ThinktankClient(settings=settings, run_id=run_id)
    themes = ThemeKeeper(store, client, ctx, trading_day=trading_day, calendar_date=calendar_date)
    ranked_rows = {s.get("ticker"): s for s in (ctx.board or {}).get("stocks", [])}

    theses_written = []
    event_rows: list[dict] = []
    if covered_before:
        assessments, macro_notes = sweep(
            client, ctx, covered=covered_before, chunk_size=settings.sweep_chunk_size
        )
        manifest.sweep_tickers = len(covered_before)
        record_sweep(ledger, covered_before, trading_day)
        for a in assessments:
            written_version: int | None = None
            if a.action == "update_thesis":
                manifest.events_flagged += 1
                thesis = build_thesis(
                    store, client, ctx, themes,
                    ticker=a.ticker,
                    board_row=ranked_rows.get(a.ticker),
                    trading_day=trading_day,
                    calendar_date=calendar_date,
                    update_reason="event",
                    event_context=a.rationale,
                )
                record_thesis_write(
                    ledger, ticker=a.ticker, trading_day=trading_day,
                    thesis_version=thesis.version,
                )
                save_ledger(store, ledger)
                manifest.event_updates_written += 1
                manifest.theses_written += 1
                theses_written.append(thesis)
                written_version = thesis.version
            event_rows.append(
                EventRecord(
                    ticker=a.ticker,
                    trading_day=trading_day,
                    action=a.action,
                    severity=a.severity,
                    rationale=a.rationale,
                    thesis_version_written=written_version,
                ).model_dump()
            )
        if macro_notes:
            themes.ensure_current(daily_developments=macro_notes)

    manifest.themes_reconciled = themes.reconciled
    manifest.theme_updates_written = themes.updates_written

    # Ratings board self-heals rows for covered tickers with no row yet by
    # reading their thesis latest.json (thinktank/ratings.py) — the BUILD
    # phase already wrote every checkpointed ticker's thesis artifact, so
    # passing only the event-driven theses here is sufficient; the board
    # still ends up complete.
    board = update_ratings_board(store, ledger, theses_written, trading_day=trading_day)
    manifest.ratings_rows = len(board.rows)
    write_challenger_selection(
        store, ledger, board,
        run_id=run_id, mode=manifest.mode,
        trading_day=trading_day, calendar_date=calendar_date,
        board_date=(ctx.board or {}).get("as_of"),
        coverage_gap=_compute_coverage_gap(ctx.board, ledger, top_n=GAP_FILL_TOP_N),
    )
    manifest.challenger_selection_written = True
    if event_rows:
        from thinktank import EVENTS_KEY_TMPL

        store.put_jsonl(EVENTS_KEY_TMPL.format(trading_day=trading_day), event_rows)

    manifest.usage_by_tier = client.usage_by_tier()
    manifest.total_cost_usd = round(build_cost_usd + client.total_cost_usd(), 6)
    guard = BudgetGuard(store, settings, ssm_client=ssm_client)
    cost_ledger = guard.record_run(
        calendar_date, run_id=run_id, trading_day=trading_day,
        cost_usd=manifest.total_cost_usd,
    )
    manifest.budget_month_spent_usd = cost_ledger.spent_usd
    manifest.budget_month_limit_usd = guard.limit_usd()
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    from thinktank import MANIFEST_KEY_TMPL

    store.put_json(
        MANIFEST_KEY_TMPL.format(trading_day=trading_day, run_id=run_id),
        manifest.model_dump(),
    )
    client.flush_sft(store.s3, store.bucket, trading_day)

    logger.info(
        "thinktank gap_fill fan-out %s finalized: +%d theses (%d event "
        "updates), swept %d, cost $%.4f (month $%.2f / $%.2f)",
        run_id, manifest.theses_written, manifest.event_updates_written,
        manifest.sweep_tickers, manifest.total_cost_usd,
        manifest.budget_month_spent_usd, manifest.budget_month_limit_usd,
    )
    return manifest
