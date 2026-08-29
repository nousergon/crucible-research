"""Coverage ledger — which names the think tank covers, and how fresh.

Intake policy (EPIC config#1579, skeleton-crew MVP):
- each daily run takes the top ``daily_new_names`` UNCOVERED names from the
  scanner's DECLARED ranking — ``thinktank.feed.FeedWindow``, resolved from
  ``universe_membership/latest.json`` through the live champion pointer, never
  re-derived here (alpha-engine-config-I7842) — bounded by ``rank_ceiling``
  (never initiate coverage on a name ranked below R);
- when fewer than ``daily_new_names`` eligible uncovered names exist, the
  remaining slots refresh the STALEST covered theses — coverage maintenance
  falls out of the intake rule for free (the DISCRETIONARY path);
- independent of the above, any covered name whose thesis is >=
  ``stale_after_days`` old is a MANDATORY refresh (TT-2.1-staleness-sla-is-
  actionable, alpha-engine-config-I6478) — added to ``refresh`` uncapped by
  ``daily_new_names`` and regardless of whether the discretionary slots were
  already spent on new names. Before this existed, a name past its staleness
  horizon produced no action and no signal on any day the new-names intake
  slot happened to be fully subscribed — the horizon was informational, not
  an SLA.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from thinktank import LEDGER_KEY
from thinktank.feed import FeedWindow, join_board_rows
from thinktank.schemas import CoverageLedger, LedgerEntry
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)


def _age_days(as_of: str, thesis_updated_on: str) -> int:
    """Whole days between ``thesis_updated_on`` and ``as_of`` (both ISO
    ``YYYY-MM-DD`` trading-day strings)."""
    return (date.fromisoformat(as_of) - date.fromisoformat(thesis_updated_on)).days


def load_ledger(store: ThinktankStore) -> CoverageLedger:
    raw = store.get_json(LEDGER_KEY)
    if raw is None:
        logger.info("no coverage ledger at %s — starting empty", LEDGER_KEY)
        return CoverageLedger()
    return CoverageLedger.model_validate(raw)


def save_ledger(store: ThinktankStore, ledger: CoverageLedger) -> None:
    ledger.updated_at = datetime.now(UTC).isoformat()
    store.put_json(LEDGER_KEY, ledger.model_dump())


def select_intake(
    ledger: CoverageLedger,
    board: dict,
    window: FeedWindow,
    *,
    daily_new_names: int,
    rank_ceiling: int,
    exit_rank: int | None = None,
    skip_stale_refill: bool = False,
    stale_after_days: int | None = None,
    trading_day: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Pick today's work: (new_names_with_board_rows, refresh_tickers).

    ``window`` is the scanner's declared contract (``thinktank.feed``): the
    coverage window's membership plus the full ranking in the SERVING arm's
    basis. Every rank in this function is that artifact's rank — nothing here
    sorts the board, and nothing here names a basis. Before
    alpha-engine-config-I7842 this module sorted ``board["stocks"]`` by
    ``attractiveness_score`` itself, which hardcoded the basis (so a promotion
    to a tech-basis champion would have left Think Tank covering the losing
    arm's names in silence) and used board POSITION as universe rank, which is
    off by one below rank 98 today (alpha-engine-config-I7844).

    ``new`` = top uncovered names with rank <= rank_ceiling (rank is the
    1-based rank the membership artifact publishes). ``refresh``
    fills any remaining slots with the stalest covered names (the
    DISCRETIONARY path) — UNLESS ``skip_stale_refill``, which suppresses
    that discretionary fill (used by the Saturday SF's gap-fill mode:
    shoring up newly-uncovered top-N names is a different job from
    staleness refresh, which the daily cadence already handles gradually;
    a weekly run padding its budget with stale-refill picks would do
    daily's job for it and lose the "only what actually changed this week"
    sizing gap_fill_only relies on).

    ``stale_after_days`` + ``trading_day`` (TT-2.1-staleness-sla-is-
    actionable, alpha-engine-config-I6478): independent of the
    discretionary path above, any covered entry whose thesis is
    >= ``stale_after_days`` old as of ``trading_day`` is a MANDATORY
    refresh — appended uncapped by ``daily_new_names`` and regardless of
    ``skip_stale_refill``. This is the SLA floor: without it, a name past
    its staleness horizon produced no action at all on a day the
    new-names intake slot happened to be fully subscribed by fresh
    coverage (the discretionary path never runs when slots_left <= 0).
    Omitting both arguments (the default) is a backward-compatible no-op —
    no SLA is enforced. The gap_fill caller intentionally omits them:
    staleness enforcement is the daily job's role, not gap_fill's, per
    ``thinktank/run.py``'s module docstring.
    """
    # Fails loud if a name in the declared window has no board row: the row is
    # what the analyst underwrites from, and the two artifacts come from the
    # same Scanner run.
    board_rows = join_board_rows(window, board)
    covered = ledger.covered()

    # ── HYSTERESIS EXIT (alpha-engine-config-I6648) ─────────────────────────
    # Until 2026-08-10 nothing in this module ever de-covered a name: the
    # `rank > rank_ceiling` break below gates ENTRY, and the next line skips
    # anything already covered, so an existing entry was never re-evaluated.
    # Coverage was a monotonic RATCHET — measured at 178 covered names against
    # rank_ceiling=150 on run b150c317eeef.
    #
    # `exit_rank` MUST be strictly wider than `rank_ceiling` (enforced at
    # config load, not here). That gap is the whole point: a name oscillating
    # between the two thresholds enters once and is never dropped, which is
    # what stops boundary churn from destroying the belief history it churns
    # through.
    #
    # A name ABSENT from today's ranking is NOT dropped. Absence means the
    # scanner did not rank it today — a universe change, a data gap, a halted
    # ticker — and inferring "it fell below the exit rank" from silence is the
    # failure this codebase has already shipped once. Only an explicit rank
    # worse than `exit_rank` de-covers.
    dropped: list[str] = []
    if exit_rank is not None and covered:
        for ticker in sorted(covered):
            rank = window.rank_of(ticker)
            if rank is not None and rank > exit_rank:
                entry = ledger.entries[ticker]
                entry.covered = False
                entry.dropped_on = trading_day or ""
                entry.attractiveness_rank_at_drop = rank
                dropped.append(ticker)
        if dropped:
            covered = ledger.covered()
            logger.info(
                "[thinktank] de-covered %d name(s) past exit_rank=%d: %s (thesis history retained — config-I6648)",
                len(dropped),
                exit_rank,
                ", ".join(dropped),
            )

    new_rows: list[dict] = []
    unjoinable: list[str] = []
    for rank, ticker in enumerate(window.ordered, start=1):
        if rank > rank_ceiling:
            break
        if ticker in covered:
            continue
        row = board_rows.get(ticker)
        if row is None:
            # Swallowed: a name the scanner RANKED inside the ceiling but for
            # which the board carries no row — there is nothing to underwrite
            # from, so it cannot be intake. Recorded, never silent: WARNed
            # below and returned to the caller, which puts the count in the run
            # manifest via ``thinktank.feed.unjoinable_ranked_names``. Names
            # INSIDE the declared window raise instead — ``join_board_rows``.
            unjoinable.append(ticker)
            continue
        row = dict(row)
        # Grandfathered field name (``thinktank.schemas.LedgerEntry``): this is
        # the rank in ``window.basis``, which is only "attractiveness" while
        # that arm holds the feed. The basis itself is carried in the run
        # manifest's window provenance, so the number is interpretable.
        row["_attractiveness_rank"] = rank
        new_rows.append(row)
        if len(new_rows) >= daily_new_names:
            break
    if unjoinable:
        logger.warning(
            "[thinktank] %d ranked name(s) inside rank_ceiling=%d have no "
            "universe-board row and were skipped for intake: %s "
            "(alpha-engine-config-I7844)",
            len(unjoinable),
            rank_ceiling,
            ", ".join(unjoinable),
        )

    graceful_refresh: list[str] = []
    if not skip_stale_refill:
        slots_left = daily_new_names - len(new_rows)
        if slots_left > 0 and ledger.entries:
            # `.covered` filter: a de-covered name must not consume a
            # refresh slot, or the ratchet returns through the back door
            # (config-I6648).
            stalest = sorted(
                (e for e in ledger.entries.values() if e.covered),
                key=lambda e: e.thesis_updated_on,
            )
            graceful_refresh = [e.ticker for e in stalest[:slots_left]]

    breached: list[str] = []
    if stale_after_days is not None and trading_day is not None and ledger.entries:
        breached_entries = [
            e
            for e in ledger.entries.values()
            if e.covered and _age_days(trading_day, e.thesis_updated_on) >= stale_after_days
        ]
        breached_entries.sort(key=lambda e: e.thesis_updated_on)  # stalest first
        breached = [e.ticker for e in breached_entries]
        if breached:
            logger.warning(
                "[thinktank] staleness SLA: %d covered name(s) >= %dd old as of "
                "%s — forcing mandatory refresh regardless of intake-slot fill "
                "state (TT-2.1-staleness-sla-is-actionable): %s",
                len(breached),
                stale_after_days,
                trading_day,
                breached,
            )

    # breach is the SLA floor: unconditional, uncapped, ordered ahead of the
    # discretionary graceful fill — stalest-first order is preserved overall
    # since a breached entry is by definition at least as stale as any
    # non-breached graceful pick.
    refresh = list(dict.fromkeys(breached + graceful_refresh))
    return new_rows, refresh


def record_thesis_write(
    ledger: CoverageLedger,
    *,
    ticker: str,
    trading_day: str,
    thesis_version: int,
    sector: str | None = None,
    attractiveness_rank: int | None = None,
) -> None:
    entry = ledger.entries.get(ticker)
    if entry is None:
        ledger.entries[ticker] = LedgerEntry(
            ticker=ticker,
            covered_since=trading_day,
            thesis_version=thesis_version,
            thesis_updated_on=trading_day,
            attractiveness_rank_at_entry=attractiveness_rank,
            sector=sector,
        )
    else:
        entry.thesis_version = thesis_version
        entry.thesis_updated_on = trading_day
        if sector and not entry.sector:
            entry.sector = sector
        # ── RE-COVER (alpha-engine-config-I9282) ────────────────────────────
        # Writing a thesis IS coverage. Until 2026-08-29 this branch advanced
        # the thesis fields and left `covered` alone — which was harmless
        # while `covered` could only ever be True, and became a permanent
        # deadlock the moment the hysteresis EXIT half shipped on 2026-08-10
        # (config-I6648) without this, its RE-ENTRY half:
        #
        #   name de-covered past exit_rank -> rank recovers, re-enters the
        #   top-60 window -> select_intake sees an UNCOVERED name and spends a
        #   full daily_new_names slot writing its thesis -> covered stays
        #   False -> the name is "uncovered" again tomorrow, forever.
        #
        # Measured on the live ledger 2026-08-29: HALO dropped 2026-08-10 and
        # carried thesis v11 written 2026-08-28 while still covered=False;
        # FTNT/WEX (dropped 08-17) carried v13 written 08-27. Those names were
        # re-underwritten daily to no effect, consuming most of the fixed
        # 5-slot intake budget, so `uncovered_count` could never reach 0 —
        # which is why `coverage_complete` has been False since 2026-08-14 and
        # the arm has written no shadow signal since, i.e. why
        # `thinktank_coverage` scores on a frozen 4-date cohort.
        #
        # The drop record is CLEARED, not kept: `dropped_on` is the audit
        # trail for a name that IS dropped, and leaving it set on a re-covered
        # name makes the ledger assert two contradictory things at once. The
        # immutable per-version thesis history under `thinktank/theses/` is
        # the durable record of the drop-and-return; this field is state.
        entry.covered = True
        entry.dropped_on = None
        entry.attractiveness_rank_at_drop = None


def record_sweep(ledger: CoverageLedger, tickers: list[str], trading_day: str) -> None:
    for t in tickers:
        if t in ledger.entries:
            ledger.entries[t].last_sweep_on = trading_day
