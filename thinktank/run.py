"""Daily think-tank run — the skeleton-crew orchestrator.

Order of operations (one ``--daily`` invocation):
1. budget guard (hard refusal at the monthly cap)
2. load read-side context (board, signals, macro report, news, RAG probe)
3. themes: seed if absent / reconcile if a new weekly landed
4. intake: top-N uncovered by attractiveness (rank-bounded) + stalest refresh
   — UNLESS gap_fill_only (below), which skips the stalest-refresh half
5. thesis builds for the intake set
6. events sweep over all covered names → thesis updates where flagged
7. churn-gated daily macro-theme update from sweep-surfaced developments
8. persist ledger, events, manifest, month cost ledger; flush SFT rows

``--dry-run`` exercises 1–4 read-only and prints the plan (no LLM calls, no
writes) — the boot-validation mode.

``gap_fill_only`` (2026-07-14 cadence design): the Saturday SF's mode. The
daily EventBridge cadence (``research/thinktank.yaml``'s ``daily_new_names``,
still steps 1-8 above unmodified) grows coverage toward the full
``rank_ceiling=150`` universe day by day and handles staleness refresh
gradually. Saturday's job is narrower and reactive: once the fresh weekly
scan lands, shore up whatever of the CURRENT top-``GAP_FILL_TOP_N`` (60)
the daily cadence hasn't caught up to yet — sized to the exact measured
gap (``coverage_gap.uncovered_count``), never a fixed constant, and never
padded with stale-refill picks (that's the daily job's role). Keeps the
weekly SF's think-tank footprint small and bounded regardless of how big
the full-universe backlog gets.

Usage:
    python -m thinktank.run --daily
    python -m thinktank.run --daily --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone

from nousergon_lib.dates import now_dual

from thinktank import EVENTS_KEY_TMPL, MANIFEST_KEY_TMPL
from thinktank.analyst import build_thesis, sweep
from thinktank.client import ThinktankClient
from thinktank.context import load_context
from thinktank.costs import BudgetGuard
from thinktank.ledger import (
    load_ledger,
    record_sweep,
    record_thesis_write,
    save_ledger,
    select_intake,
)
from thinktank.ratings import update_ratings_board
from thinktank.schemas import CompanyThesis, EventRecord, RunManifest
from thinktank.settings import ThinktankSettings, load_settings
from thinktank.storage import ThinktankStore
from thinktank.themes import ThemeKeeper

logger = logging.getLogger(__name__)


GAP_FILL_TOP_N = 60
"""Rank window the Saturday SF's gap-fill mode shores up — the "current
scanner top-60" window, independent of the daily cadence's rank_ceiling
(150). Shared with ``_compute_coverage_gap``'s default so the reported
gap and the actual gap-fill selection are always computed over the same
window."""


def run_daily(
    settings: ThinktankSettings | None = None,
    *,
    dry_run: bool = False,
    refresh_tickers: list[str] | None = None,
    gap_fill_only: bool = False,
    store: ThinktankStore | None = None,
    client: ThinktankClient | None = None,
    ssm_client=None,
) -> RunManifest:
    settings = settings or load_settings()
    store = store or ThinktankStore(settings.bucket)
    run_id = uuid.uuid4().hex[:12]
    dual = now_dual()
    trading_day, calendar_date = str(dual.trading_day), str(dual.calendar_date)

    manifest = RunManifest(
        run_id=run_id,
        mode=(
            "dry_run" if dry_run
            else "operator_refresh" if refresh_tickers
            else "gap_fill" if gap_fill_only
            else "daily"
        ),
        trading_day=trading_day,
        calendar_date=calendar_date,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    guard = BudgetGuard(store, settings, ssm_client=ssm_client)
    spent, limit = guard.check(calendar_date)
    manifest.budget_month_spent_usd = spent
    manifest.budget_month_limit_usd = limit

    ctx = load_context(store)
    manifest.context_sources_present = dict(ctx.sources_present)
    if ctx.board is None:
        raise RuntimeError(
            "universe board (scanner/universe/latest.json) is missing — the "
            "think tank has no intake without it; aborting loudly."
        )

    ledger = load_ledger(store)
    manifest.coverage_gap = _compute_coverage_gap(ctx.board, ledger, top_n=GAP_FILL_TOP_N)
    if refresh_tickers is not None:
        # Operator-refresh mode ({"refresh_tickers": [...]} event / backfill):
        # re-underwrite ONLY the named covered tickers — no intake, no sweep,
        # no theme work. An uncovered name is a caller error: fail loud.
        uncovered = sorted(set(refresh_tickers) - ledger.covered())
        if uncovered:
            raise ValueError(
                f"refresh_tickers not in coverage ledger: {uncovered} — "
                "operator refresh only re-underwrites covered names."
            )
        new_rows, refresh = [], sorted(set(refresh_tickers))
    elif gap_fill_only:
        # Saturday SF gap-fill mode (2026-07-14 cadence design): the daily
        # cadence (settings.daily_new_names/day) already grows coverage
        # toward the full rank_ceiling=150 universe and handles staleness
        # refresh gradually — this mode's ONLY job is shoring up whatever
        # of the CURRENT scanner top-60 the daily cadence hasn't caught up
        # to yet by the time the fresh weekly board lands. Sized to the
        # EXACT current gap (manifest.coverage_gap, computed just above
        # against the same GAP_FILL_TOP_N window) rather than a fixed
        # constant — a small, data-driven weekly patch, not a bulk
        # re-cover pass. skip_stale_refill=True: staleness refresh is the
        # daily job's role, not this one's — padding this run's budget
        # with stale-refill picks would double-do daily's job and break
        # the "only what actually changed this week" sizing this mode
        # relies on to stay fast and small.
        gap = manifest.coverage_gap if isinstance(manifest.coverage_gap, dict) else {}
        gap_count = gap.get("uncovered_count", 0)
        new_rows, refresh = select_intake(
            ledger,
            ctx.board,
            daily_new_names=gap_count,
            rank_ceiling=GAP_FILL_TOP_N,
            skip_stale_refill=True,
        )
    else:
        new_rows, refresh = select_intake(
            ledger,
            ctx.board,
            daily_new_names=settings.daily_new_names,
            rank_ceiling=settings.rank_ceiling,
        )
    manifest.names_added = [r["ticker"] for r in new_rows]
    manifest.names_refreshed = refresh
    covered_before = [] if refresh_tickers is not None else sorted(ledger.covered())

    if dry_run:
        manifest.finished_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "DRY RUN — would add %s, refresh %s, sweep %d covered names; "
            "month spend $%.2f / cap $%.2f",
            manifest.names_added,
            refresh,
            len(covered_before),
            spent,
            limit,
        )
        return manifest

    client = client or ThinktankClient(settings=settings, run_id=run_id)
    themes = ThemeKeeper(
        store, client, ctx, trading_day=trading_day, calendar_date=calendar_date
    )
    if refresh_tickers is None:
        themes.ensure_current()

    theses_written: list[CompanyThesis] = []
    board_by_ticker = {r["ticker"]: r for r in new_rows}
    for ticker in manifest.names_added:
        thesis = build_thesis(
            store, client, ctx, themes,
            ticker=ticker,
            board_row=board_by_ticker[ticker],
            trading_day=trading_day,
            calendar_date=calendar_date,
            update_reason="initial",
        )
        record_thesis_write(
            ledger,
            ticker=ticker,
            trading_day=trading_day,
            thesis_version=thesis.version,
            sector=thesis.sector,
            attractiveness_rank=thesis.attractiveness_rank,
        )
        theses_written.append(thesis)
        manifest.theses_written += 1

    ranked_rows = {s.get("ticker"): s for s in (ctx.board or {}).get("stocks", [])}
    refresh_reason = "operator_refresh" if refresh_tickers is not None else "staleness_refresh"
    for ticker in refresh:
        thesis = build_thesis(
            store, client, ctx, themes,
            ticker=ticker,
            board_row=ranked_rows.get(ticker),
            trading_day=trading_day,
            calendar_date=calendar_date,
            update_reason=refresh_reason,
        )
        record_thesis_write(
            ledger, ticker=ticker, trading_day=trading_day, thesis_version=thesis.version
        )
        theses_written.append(thesis)
        manifest.theses_written += 1

    # ── events sweep over everything covered before today's additions ────────
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
                    ledger,
                    ticker=a.ticker,
                    trading_day=trading_day,
                    thesis_version=thesis.version,
                )
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

    # ── persist ──────────────────────────────────────────────────────────────
    save_ledger(store, ledger)
    board = update_ratings_board(
        store, ledger, theses_written, trading_day=trading_day
    )
    manifest.ratings_rows = len(board.rows)
    if event_rows:
        store.put_jsonl(EVENTS_KEY_TMPL.format(trading_day=trading_day), event_rows)

    manifest.usage_by_tier = client.usage_by_tier()
    manifest.total_cost_usd = client.total_cost_usd()
    cost_ledger = guard.record_run(
        calendar_date,
        run_id=run_id,
        trading_day=trading_day,
        cost_usd=manifest.total_cost_usd,
    )
    manifest.budget_month_spent_usd = cost_ledger.spent_usd
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    store.put_json(
        MANIFEST_KEY_TMPL.format(trading_day=trading_day, run_id=run_id),
        manifest.model_dump(),
    )
    client.flush_sft(store.s3, store.bucket, trading_day)

    logger.info(
        "thinktank run %s done: +%d theses (%d event updates), swept %d, "
        "themes written %d, cost $%.4f (month $%.2f / $%.2f)",
        run_id,
        manifest.theses_written,
        manifest.event_updates_written,
        manifest.sweep_tickers,
        manifest.theme_updates_written,
        manifest.total_cost_usd,
        manifest.budget_month_spent_usd,
        manifest.budget_month_limit_usd,
    )
    return manifest


def _compute_coverage_gap(
    board: dict | None,
    ledger: "CoverageLedger",
    *,
    top_n: int = 60,
) -> dict:
    """What % of scanner top-N have fresh Think Tank coverage?

    Emitted in every run manifest so downstream consumers (dashboard,
    report card) can track coverage-health trends without re-querying
    the board + ledger themselves.
    """
    if not board:
        return {"error": "universe_board_missing"}
    stocks = board.get("stocks", [])
    if not stocks:
        return {"top_n": top_n, "covered_pct": 0, "total_covered": len(ledger.entries), "uncovered_count": top_n}
    sorted_stocks = sorted(
        stocks,
        key=lambda s: s.get("attractiveness_score", 0) or 0,
        reverse=True,
    )
    top_tickers = {s["ticker"] for s in sorted_stocks[:top_n] if s.get("ticker")}
    covered = set(ledger.entries.keys())
    covered_in_top = covered & top_tickers
    pct = round(len(covered_in_top) / max(len(top_tickers), 1) * 100, 1)
    return {
        "top_n": top_n,
        "total_in_top": len(top_tickers),
        "covered_in_top": len(covered_in_top),
        "covered_pct": pct,
        "uncovered_count": len(top_tickers) - len(covered_in_top),
        "total_covered": len(ledger.entries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research think tank runner")
    parser.add_argument("--daily", action="store_true", help="run the daily cycle")
    parser.add_argument("--dry-run", action="store_true", help="plan only, no LLM/writes")
    parser.add_argument(
        "--refresh",
        nargs="+",
        metavar="TICKER",
        help="operator refresh: re-underwrite ONLY these covered tickers "
        "(no intake/sweep/themes) — e.g. a rating backfill",
    )
    args = parser.parse_args(argv)
    if not args.daily and not args.refresh:
        parser.error("one of --daily / --refresh TICKER... is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_daily(dry_run=args.dry_run, refresh_tickers=args.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
