"""Daily think-tank run — the skeleton-crew orchestrator.

Order of operations (one ``--daily`` invocation):
1. budget guard (hard refusal at the monthly cap)
2. load read-side context (board, signals, macro report, news, RAG probe)
3. themes: seed if absent / reconcile if a new weekly landed
4. intake: top-N uncovered by attractiveness (rank-bounded) + stalest refresh
5. thesis builds for the intake set
6. events sweep over all covered names → thesis updates where flagged
7. churn-gated daily macro-theme update from sweep-surfaced developments
8. persist ledger, events, manifest, month cost ledger; flush SFT rows

``--dry-run`` exercises 1–4 read-only and prints the plan (no LLM calls, no
writes) — the boot-validation mode.

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
from thinktank.schemas import EventRecord, RunManifest
from thinktank.settings import ThinktankSettings, load_settings
from thinktank.storage import ThinktankStore
from thinktank.themes import ThemeKeeper

logger = logging.getLogger(__name__)


def run_daily(
    settings: ThinktankSettings | None = None,
    *,
    dry_run: bool = False,
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
        mode="dry_run" if dry_run else "daily",
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
    new_rows, refresh = select_intake(
        ledger,
        ctx.board,
        daily_new_names=settings.daily_new_names,
        rank_ceiling=settings.rank_ceiling,
    )
    manifest.names_added = [r["ticker"] for r in new_rows]
    manifest.names_refreshed = refresh
    covered_before = sorted(ledger.covered())

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
    themes.ensure_current()

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
        manifest.theses_written += 1

    ranked_rows = {s.get("ticker"): s for s in (ctx.board or {}).get("stocks", [])}
    for ticker in refresh:
        thesis = build_thesis(
            store, client, ctx, themes,
            ticker=ticker,
            board_row=ranked_rows.get(ticker),
            trading_day=trading_day,
            calendar_date=calendar_date,
            update_reason="staleness_refresh",
        )
        record_thesis_write(
            ledger, ticker=ticker, trading_day=trading_day, thesis_version=thesis.version
        )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research think tank runner")
    parser.add_argument("--daily", action="store_true", help="run the daily cycle")
    parser.add_argument("--dry-run", action="store_true", help="plan only, no LLM/writes")
    args = parser.parse_args(argv)
    if not args.daily:
        parser.error("--daily is required (the only mode in P0)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_daily(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
