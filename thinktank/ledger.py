"""Coverage ledger — which names the think tank covers, and how fresh.

Intake policy (EPIC config#1579, skeleton-crew MVP):
- each daily run takes the top ``daily_new_names`` UNCOVERED names from the
  scanner attractiveness ranking, bounded by ``rank_ceiling`` (never initiate
  coverage on a name ranked below R);
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


def ranked_universe(board: dict) -> list[dict]:
    """Universe-board ``stocks[]`` sorted by attractiveness_score desc.

    Names with a null attractiveness_score sort last and are never eligible
    for intake (no basis to rank them).
    """
    stocks = board.get("stocks", [])
    return sorted(
        stocks,
        key=lambda s: (
            s.get("attractiveness_score") is None,
            -(s.get("attractiveness_score") or 0.0),
        ),
    )


def select_intake(
    ledger: CoverageLedger,
    board: dict,
    *,
    daily_new_names: int,
    rank_ceiling: int,
    exit_rank: int | None = None,
    skip_stale_refill: bool = False,
    stale_after_days: int | None = None,
    trading_day: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Pick today's work: (new_names_with_board_rows, refresh_tickers).

    ``new`` = top uncovered by attractiveness with rank <= rank_ceiling (rank
    is 1-based position in the attractiveness-sorted universe). ``refresh``
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
    ranked = ranked_universe(board)
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
        rank_by_ticker = {
            row.get("ticker"): rank
            for rank, row in enumerate(ranked, start=1)
            if row.get("ticker")
        }
        for ticker in sorted(covered):
            rank = rank_by_ticker.get(ticker)
            if rank is not None and rank > exit_rank:
                entry = ledger.entries[ticker]
                entry.covered = False
                entry.dropped_on = trading_day or ""
                entry.attractiveness_rank_at_drop = rank
                dropped.append(ticker)
        if dropped:
            covered = ledger.covered()
            logger.info(
                "[thinktank] de-covered %d name(s) past exit_rank=%d: %s "
                "(thesis history retained — config-I6648)",
                len(dropped), exit_rank, ", ".join(dropped),
            )

    new_rows: list[dict] = []
    for rank, row in enumerate(ranked, start=1):
        if rank > rank_ceiling:
            break
        ticker = row.get("ticker")
        if not ticker or ticker in covered:
            continue
        if row.get("attractiveness_score") is None:
            continue
        row = dict(row)
        row["_attractiveness_rank"] = rank
        new_rows.append(row)
        if len(new_rows) >= daily_new_names:
            break

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
            if e.covered
            and _age_days(trading_day, e.thesis_updated_on) >= stale_after_days
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


def record_sweep(ledger: CoverageLedger, tickers: list[str], trading_day: str) -> None:
    for t in tickers:
        if t in ledger.entries:
            ledger.entries[t].last_sweep_on = trading_day
