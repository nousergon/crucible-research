"""Daily think-tank run — the skeleton-crew orchestrator.

Order of operations (one ``--daily`` invocation):
1. budget guard (hard refusal at the monthly cap)
2. load read-side context (board, signals, macro report, news, RAG probe)
3. themes: seed if absent / reconcile if a new weekly landed
4. intake SELECTION: top-N uncovered by attractiveness (rank-bounded) +
   stalest refresh — UNLESS gap_fill_only (below), which skips the
   stalest-refresh half. The daily branch also enforces the staleness SLA
   (TT-2.1-staleness-sla-is-actionable, alpha-engine-config-I6478): any
   covered name past ``stale_after_days`` is force-refreshed regardless of
   whether the new-names slots were otherwise full — see ``thinktank.ledger
   .select_intake``.
5. events sweep over all covered names → thesis updates where flagged, then
   the churn-gated daily macro-theme update from the developments it surfaced
6. thesis builds for the intake set (new names, then refreshes)

   Steps 5 and 6 were the other way round until alpha-engine-config-I6650.
   DETECTION RUNS FIRST: it is the cheap tier that decides what the expensive
   tier is for, an abort in the thesis loop used to cost the whole day's
   detection (eight consecutive days of it, 2026-08-03..08), and under a
   deadline the truncation now lands on per-ticker refresh rather than on a
   single un-resumable fan-out.
8. persist ledger, ratings board, challenger selection, events, manifest,
   month cost ledger; flush SFT rows

New S3 keys (epic alpha-engine-config-I2515, champion/challenger
leaderboard — Think Tank is the CHALLENGER arm):
    ``thinktank/challenger_selection/{trading_day}.json`` + ``latest.json``
    — see ``thinktank.challenger_selection`` for the producer and
    ``thinktank.schemas.ChallengerSelection`` for the artifact contract.

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
from collections.abc import Callable
from datetime import UTC, datetime

from nousergon_lib.dates import now_dual

from thinktank import EVENTS_KEY_TMPL, MANIFEST_KEY_TMPL
from thinktank.analyst import build_thesis, sweep, triage
from thinktank.challenger_selection import write_challenger_selection
from thinktank.client import ThinktankClient
from thinktank.context import ContextBundle, load_context
from thinktank.costs import BudgetGuard
from thinktank.ledger import (
    load_ledger,
    record_sweep,
    record_thesis_write,
    save_ledger,
    select_intake,
)
from thinktank.ratings import update_ratings_board
from thinktank.schemas import CompanyThesis, CoverageLedger, EventRecord, RunManifest
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


def _checkpoint_thesis_write(
    store: ThinktankStore,
    ledger: CoverageLedger,
    *,
    ticker: str,
    trading_day: str,
    thesis_version: int,
    sector: str | None = None,
    attractiveness_rank: int | None = None,
) -> None:
    """Record a thesis write AND persist the ledger immediately (config#3072:
    each thesis is an idempotent checkpointed unit). A Lambda timeout or SF
    retry mid-run then re-selects intake against a ledger that already
    reflects every unit completed so far — ``select_intake`` skips covered
    tickers by construction, so a retry only rebuilds the remaining gap
    instead of restarting the whole batch from ticker 1."""
    record_thesis_write(
        ledger,
        ticker=ticker,
        trading_day=trading_day,
        thesis_version=thesis_version,
        sector=sector,
        attractiveness_rank=attractiveness_rank,
    )
    save_ledger(store, ledger)


# Wall-clock reserve, in seconds, held back for the terminal writes (ledger
# save, ratings board, challenger selection + its leaderboard shadow view,
# events jsonl, cost ledger, manifest, SFT flush). Those are S3 puts, not LLM
# calls, so this is generous by design — the cost of over-reserving is one
# fewer thesis, the cost of under-reserving is the entire run's terminal state.
_TERMINAL_WRITE_RESERVE_S = 120.0


def _out_of_time(
    seconds_remaining: Callable[[], float] | None,
    reserve: float = _TERMINAL_WRITE_RESERVE_S,
) -> bool:
    """True when too little wall-clock remains to start another LLM unit.

    alpha-engine-config-I5208. The Think Tank ran on a 900s Lambda with NO
    deadline awareness: every run since 2026-07-17 hit the ceiling mid-loop and
    died before its terminal writes, so ~15 theses of real work per day were
    completed and then thrown away — `thinktank/ratings/`,
    `thinktank/challenger_selection/` and the leaderboard shadow view all
    froze on 2026-07-17 while the logs looked busy and healthy.

    This is NOT a Lambda workaround. Any bounded runtime needs it — a spot box
    gets a 2-minute reclaim notice, which is squarely inside this reserve — so
    the guard belongs in the run loop regardless of where the run executes.

    ``None`` means "no deadline known" (local/operator invocation): never stop.
    A callable that raises is treated as no-deadline rather than as a stop,
    since a broken clock must not silently truncate a healthy run.
    """
    if seconds_remaining is None:
        return False
    try:
        return float(seconds_remaining()) <= reserve
    except Exception:  # noqa: BLE001 — a broken clock must not truncate a run
        logger.warning("[thinktank] seconds_remaining() raised — treating as no deadline")
        return False


def run_daily(
    settings: ThinktankSettings | None = None,
    *,
    dry_run: bool = False,
    refresh_tickers: list[str] | None = None,
    gap_fill_only: bool = False,
    store: ThinktankStore | None = None,
    client: ThinktankClient | None = None,
    ssm_client=None,
    seconds_remaining: Callable[[], float] | None = None,
) -> RunManifest:
    settings = settings or load_settings()
    store = store or ThinktankStore(settings.bucket)
    run_id = uuid.uuid4().hex[:12]
    dual = now_dual()
    trading_day, calendar_date = str(dual.trading_day), str(dual.calendar_date)

    manifest = RunManifest(
        run_id=run_id,
        mode=(
            "dry_run"
            if dry_run
            else "operator_refresh"
            if refresh_tickers
            else "gap_fill"
            if gap_fill_only
            else "daily"
        ),
        trading_day=trading_day,
        calendar_date=calendar_date,
        started_at=datetime.now(UTC).isoformat(),
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
        # stale_after_days/trading_day wired here (only): TT-2.1-staleness-
        # sla-is-actionable / alpha-engine-config-I6478 — the daily cadence
        # is the sole enforcer of the staleness SLA (see select_intake's
        # docstring); refresh_tickers and gap_fill_only intentionally omit
        # them.
        new_rows, refresh = select_intake(
            ledger,
            ctx.board,
            daily_new_names=settings.daily_new_names,
            rank_ceiling=settings.rank_ceiling,
            stale_after_days=settings.stale_after_days,
            trading_day=trading_day,
        )
    manifest.names_added = [r["ticker"] for r in new_rows]
    manifest.names_refreshed = refresh
    covered_before = [] if refresh_tickers is not None else sorted(ledger.covered())

    if dry_run:
        manifest.finished_at = datetime.now(UTC).isoformat()
        logger.info(
            "DRY RUN — would add %s, refresh %s, sweep %d covered names; month spend $%.2f / cap $%.2f",
            manifest.names_added,
            refresh,
            len(covered_before),
            spent,
            limit,
        )
        return manifest

    # ── I5223: wire the shared cost sink for per-call telemetry ────────────────
    from krepis.cost_sink import S3JsonlCostSink

    # prefix MUST be `_cost_raw` (not `_cost`). AggregateCosts reads
    # `_cost_raw/**/*.jsonl` and writes the dashboard's
    # `_cost/{date}/cost.parquet`. Writing JSONL into `_cost/` made
    # every Think Tank run's cost rows invisible to the aggregator —
    # the stream looked dead on the dashboard even though emission
    # was live (alpha-engine-config-I5206, verified 2026-08-01).
    cost_sink = S3JsonlCostSink(
        bucket=settings.bucket,
        prefix="decision_artifacts/_cost_raw",
        run_id=run_id,
        register_atexit=True,
    )
    client = client or ThinktankClient(
        settings=settings,
        run_id=run_id,
        cost_sink=cost_sink,
    )
    themes = ThemeKeeper(store, client, ctx, trading_day=trading_day, calendar_date=calendar_date)

    # Bound BEFORE the guarded section: an exception anywhere inside
    # ``_build_and_sweep`` must still leave the terminal-write block able to
    # persist whatever completed. See ``_terminal_writes``.
    theses_written: list[CompanyThesis] = []
    event_rows: list[dict] = []
    work_kwargs = {
        "settings": settings,
        "new_rows": new_rows,
        "refresh": refresh,
        "covered_before": covered_before,
        "refresh_tickers": refresh_tickers,
        "trading_day": trading_day,
        "calendar_date": calendar_date,
        "seconds_remaining": seconds_remaining,
        "theses_written": theses_written,
        "event_rows": event_rows,
    }
    terminal_kwargs = {
        "guard": guard,
        "run_id": run_id,
        "trading_day": trading_day,
        "calendar_date": calendar_date,
        "theses_written": theses_written,
        "event_rows": event_rows,
    }
    try:
        _build_and_sweep(store, ledger, client, themes, ctx, manifest, **work_kwargs)
    except Exception as exc:
        # A run killed mid-loop used to discard every terminal write — the same
        # loss the deadline guard was built to prevent (alpha-engine-config-I5208),
        # reached by a different cause. Persist what completed, record the cause
        # in the manifest, then RE-RAISE: the non-zero exit is the dispatcher's
        # and sf-watch's only honest failure signal, and this must not convert a
        # dead run into a reported success.
        manifest.aborted_by_error = f"{type(exc).__name__}: {exc}"
        manifest.errors.append(manifest.aborted_by_error)
        logger.error(
            "[thinktank] run %s ABORTED by %s after %d thesis write(s) — "
            "persisting terminal artifacts for the completed work, then "
            "re-raising. This run is PARTIAL: %s",
            run_id,
            type(exc).__name__,
            manifest.theses_written,
            exc,
        )
        try:
            _terminal_writes(store, ledger, client, themes, ctx, manifest, aborted=True, **terminal_kwargs)
        except Exception:
            # The abort path's own writes failed. Recorded loudly and swallowed
            # so the ORIGINAL cause is what propagates — chaining a secondary
            # S3 error over it would bury the diagnosis.
            logger.exception(
                "[thinktank] run %s: terminal writes ALSO failed on the abort "
                "path — this run persisted nothing terminal",
                run_id,
            )
        raise

    _terminal_writes(store, ledger, client, themes, ctx, manifest, **terminal_kwargs)

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


def _build_and_sweep(
    store: ThinktankStore,
    ledger: CoverageLedger,
    client: ThinktankClient,
    themes: ThemeKeeper,
    ctx: ContextBundle,
    manifest: RunManifest,
    *,
    settings: ThinktankSettings,
    new_rows: list[dict],
    refresh: list[str],
    covered_before: list[str],
    refresh_tickers: list[str] | None,
    trading_day: str,
    calendar_date: str,
    seconds_remaining: Callable[[], float] | None,
    theses_written: list[CompanyThesis],
    event_rows: list[dict],
) -> None:
    """Every LLM-bearing step of a run: themes, thesis builds, events sweep.

    Split out of :func:`run_daily` so the terminal-write block can be reached
    from BOTH the success path and the failure path. ``theses_written`` and
    ``event_rows`` are passed IN and appended to rather than returned, so work
    completed before an exception is still visible to the caller — a returned
    value would be lost with the frame.
    """
    if refresh_tickers is None:
        themes.ensure_current()

    # ── DETECTION FIRST (alpha-engine-config-I6650) ─────────────────────────
    # The events sweep used to run LAST, after intake and refresh. Three
    # consequences, all of them live:
    #
    # 1. Any abort in the thesis loop cost the whole day's DETECTION. Every
    #    run from 2026-08-03 aborted on a provider 402 partway through the
    #    `med`-group thesis writes, so `thinktank/events/` produced nothing
    #    for eight days while `sweep_tickers` sat at 0 and
    #    `deadline_skipped_sweep` stayed false — the sweep was never
    #    reached, not skipped.
    # 2. It inverted the escalation ladder products/thinktank.md §2.4
    #    describes. Detection is supposed to GATE the expensive tier;
    #    running the expensive tier first means a bad day at the write tier
    #    silently costs the cheap pass that decides what is worth writing.
    # 3. Under a deadline the truncation landed on the sweep — a single
    #    un-resumable fan-out — rather than on routine refresh, which is
    #    per-ticker and the most droppable work in the run.
    #
    # Running it first also makes two things strictly more correct rather
    # than merely earlier: `covered_before` genuinely means "covered before
    # today's additions" now that no addition has happened yet, and themes
    # absorb the day's macro developments BEFORE any thesis is written
    # against them.
    ranked_rows = {s.get("ticker"): s for s in (ctx.board or {}).get("stocks", [])}
    # ── events sweep over everything covered before today's additions ────────
    if covered_before and _out_of_time(seconds_remaining):
        # The sweep is a single LLM fan-out over every covered name — it cannot
        # be partially completed, so near the deadline it is skipped ENTIRELY
        # rather than started and lost. Recorded, never silent.
        manifest.deadline_truncated = True
        manifest.deadline_skipped_sweep = True
        logger.warning(
            "[thinktank] DEADLINE: skipping the events sweep over %d covered "
            "names — proceeding to terminal writes (alpha-engine-config-I5208)",
            len(covered_before),
        )
    elif covered_before:
        assessments, macro_notes = sweep(client, ctx, covered=covered_before, chunk_size=settings.sweep_chunk_size)
        manifest.sweep_tickers = len(covered_before)
        record_sweep(ledger, covered_before, trading_day)
        for a in assessments:
            written_version: int | None = None
            escalated: bool | None = None
            triage_reason: str | None = None
            if a.action == "update_thesis":
                manifest.events_flagged += 1
                # ── TRIAGE GATE (alpha-engine-config-I6649) ──────────────────
                # products/thinktank.md §2.4: the expensive write tier fires
                # only behind an escalation decision that is itself recorded.
                # The sweep is wide and low-precision by design, so without
                # this gate every false positive it produces is paid for at
                # the `med`-group thesis tier — the most expensive stage.
                #
                # FAIL-OPEN, deliberately, and it is the one place in this
                # module that does. A triage error must not silently suppress
                # a belief update: the gate exists to save cost, and the cost
                # of wrongly skipping a real re-underwrite is a stale belief
                # that no later run will revisit (the sweep only flags NEW
                # events). The error is counted in `triage_errors`, recorded
                # on the event row, and the escalation proceeds — so the
                # failure is visible on the manifest rather than absorbed.
                try:
                    decision = triage(
                        store, client, ctx, assessment=a, trading_day=trading_day
                    )
                    escalated = decision.escalate
                    triage_reason = decision.reason
                except Exception as exc:  # noqa: BLE001 — see fail-open note above
                    manifest.triage_errors += 1
                    escalated = True
                    triage_reason = f"triage failed, escalating: {exc}"
                    logger.warning(
                        "[thinktank] triage failed for %s — escalating rather "
                        "than silently skipping a belief update: %s",
                        a.ticker,
                        exc,
                    )
                if escalated:
                    manifest.triage_yes += 1
                else:
                    manifest.triage_no += 1
                    logger.info(
                        "[thinktank] triage held %s at the gate: %s",
                        a.ticker,
                        triage_reason,
                    )
            if a.action == "update_thesis" and escalated:
                thesis = build_thesis(
                    store,
                    client,
                    ctx,
                    themes,
                    ticker=a.ticker,
                    board_row=ranked_rows.get(a.ticker),
                    trading_day=trading_day,
                    calendar_date=calendar_date,
                    update_reason="event",
                    event_context=a.rationale,
                )
                _checkpoint_thesis_write(
                    store,
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
                    triage_escalated=escalated,
                    triage_reason=triage_reason,
                ).model_dump()
            )
        if macro_notes:
            themes.ensure_current(daily_developments=macro_notes)

    board_by_ticker = {r["ticker"]: r for r in new_rows}
    for idx, ticker in enumerate(manifest.names_added):
        if _out_of_time(seconds_remaining):
            skipped = manifest.names_added[idx:]
            manifest.deadline_truncated = True
            manifest.deadline_skipped_new = skipped
            logger.warning(
                "[thinktank] DEADLINE: stopping new-thesis intake with %d of %d "
                "remaining (%s) — proceeding to terminal writes so this run's "
                "completed work persists (alpha-engine-config-I5208)",
                len(skipped),
                len(manifest.names_added),
                ", ".join(skipped[:10]),
            )
            break
        thesis = build_thesis(
            store,
            client,
            ctx,
            themes,
            ticker=ticker,
            board_row=board_by_ticker[ticker],
            trading_day=trading_day,
            calendar_date=calendar_date,
            update_reason="initial",
        )
        _checkpoint_thesis_write(
            store,
            ledger,
            ticker=ticker,
            trading_day=trading_day,
            thesis_version=thesis.version,
            sector=thesis.sector,
            attractiveness_rank=thesis.attractiveness_rank,
        )
        theses_written.append(thesis)
        manifest.theses_written += 1

    refresh_reason = "operator_refresh" if refresh_tickers is not None else "staleness_refresh"
    for idx, ticker in enumerate(refresh):
        if _out_of_time(seconds_remaining):
            skipped = list(refresh[idx:])
            manifest.deadline_truncated = True
            manifest.deadline_skipped_refresh = skipped
            logger.warning(
                "[thinktank] DEADLINE: stopping refresh with %d of %d remaining "
                "— proceeding to terminal writes (alpha-engine-config-I5208)",
                len(skipped),
                len(refresh),
            )
            break
        thesis = build_thesis(
            store,
            client,
            ctx,
            themes,
            ticker=ticker,
            board_row=ranked_rows.get(ticker),
            trading_day=trading_day,
            calendar_date=calendar_date,
            update_reason=refresh_reason,
        )
        _checkpoint_thesis_write(store, ledger, ticker=ticker, trading_day=trading_day, thesis_version=thesis.version)
        theses_written.append(thesis)
        manifest.theses_written += 1

def _terminal_writes(
    store: ThinktankStore,
    ledger: CoverageLedger,
    client: ThinktankClient,
    themes: ThemeKeeper,
    ctx: ContextBundle,
    manifest: RunManifest,
    *,
    guard: BudgetGuard,
    run_id: str,
    trading_day: str,
    calendar_date: str,
    theses_written: list[CompanyThesis],
    event_rows: list[dict],
    aborted: bool = False,
) -> None:
    """Persist everything the run owes its consumers, then the manifest.

    Called on the success path AND from ``run_daily``'s abort handler. The
    deadline guard (alpha-engine-config-I5208) exists so a run that runs out of
    wall-clock still reaches this block; an exception mid-loop leaves the run in
    the IDENTICAL state — completed theses, unwritten terminal artifacts — so it
    reaches this block too. ``guard.record_run`` lives here, which is why
    skipping it also lost the run's spend from the monthly cost ledger.

    ``aborted=True`` HOLDS BACK THE CHALLENGER-SELECTION POINTER, and that
    exception is the whole reason this parameter exists. The Think Tank runs on
    a self-terminating spot box behind an async dispatcher that does not
    babysit the run (nousergon-data ``thinktank-spot-dispatcher``), so the
    dispatcher's alarm can only see "no box was ever launched". Per
    ARTIFACT_REGISTRY's ``thinktank_challenger_selection`` row, that key is
    therefore **the only end-to-end signal that this arm produced anything**,
    and the freshness monitor probes it with a HEAD — ``LastModified`` and
    ``ContentLength``, never content. Advancing the pointer from a dead run
    would make the failure read as a healthy run to the one detector that can
    see it. The dated key is still written (the partial cohort is real
    evidence); only the *latest* pointer is withheld, because that pointer is a
    claim this run produced a result, and it did not.

    Everything else persists exactly as on the success path: completed theses,
    the ratings board, events, spend, and the manifest that names the cause.
    """
    manifest.themes_reconciled = themes.reconciled
    manifest.theme_updates_written = themes.updates_written

    # ── persist ──────────────────────────────────────────────────────────────
    save_ledger(store, ledger)
    board = update_ratings_board(store, ledger, theses_written, trading_day=trading_day)
    manifest.ratings_rows = len(board.rows)
    # coverage_complete must reflect the ledger AFTER this run's thesis
    # writes, not the start-of-run manifest.coverage_gap — otherwise the
    # run that fills the last gap (Saturday's gap_fill) self-labels its
    # own selection incomplete and the leaderboard shadow view slips to
    # the NEXT run. manifest.coverage_gap keeps its pre-run convention.
    write_challenger_selection(
        store,
        ledger,
        board,
        run_id=run_id,
        mode=manifest.mode,
        trading_day=trading_day,
        calendar_date=calendar_date,
        board_date=(ctx.board or {}).get("as_of"),
        coverage_gap=_compute_coverage_gap(ctx.board, ledger, top_n=GAP_FILL_TOP_N),
        update_latest_pointer=not aborted,
    )
    manifest.challenger_selection_written = not aborted
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
    manifest.finished_at = datetime.now(UTC).isoformat()
    store.put_json(
        MANIFEST_KEY_TMPL.format(trading_day=trading_day, run_id=run_id),
        manifest.model_dump(),
    )
    client.flush_sft(store.s3, store.bucket, trading_day)


def _compute_coverage_gap(
    board: dict | None,
    ledger: CoverageLedger,
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
