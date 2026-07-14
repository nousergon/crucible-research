"""Coverage ledger — which names the think tank covers, and how fresh.

Intake policy (EPIC config#1579, skeleton-crew MVP):
- each daily run takes the top ``daily_new_names`` UNCOVERED names from the
  scanner attractiveness ranking, bounded by ``rank_ceiling`` (never initiate
  coverage on a name ranked below R);
- when fewer than ``daily_new_names`` eligible uncovered names exist, the
  remaining slots refresh the STALEST covered theses — coverage maintenance
  falls out of the intake rule for free.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from thinktank import LEDGER_KEY
from thinktank.schemas import CoverageLedger, LedgerEntry
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)


def load_ledger(store: ThinktankStore) -> CoverageLedger:
    raw = store.get_json(LEDGER_KEY)
    if raw is None:
        logger.info("no coverage ledger at %s — starting empty", LEDGER_KEY)
        return CoverageLedger()
    return CoverageLedger.model_validate(raw)


def save_ledger(store: ThinktankStore, ledger: CoverageLedger) -> None:
    ledger.updated_at = datetime.now(timezone.utc).isoformat()
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
    skip_stale_refill: bool = False,
) -> tuple[list[dict], list[str]]:
    """Pick today's work: (new_names_with_board_rows, refresh_tickers).

    ``new`` = top uncovered by attractiveness with rank <= rank_ceiling (rank
    is 1-based position in the attractiveness-sorted universe). ``refresh``
    fills any remaining slots with the stalest covered names — UNLESS
    ``skip_stale_refill``, which returns new-name intake ONLY (used by the
    Saturday SF's gap-fill mode: shoring up newly-uncovered top-N names is a
    different job from staleness refresh, which the daily cadence already
    handles gradually; a weekly run padding its budget with stale-refill
    picks would do daily's job for it and lose the "only what actually
    changed this week" sizing gap_fill_only relies on).
    """
    ranked = ranked_universe(board)
    covered = ledger.covered()

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

    refresh: list[str] = []
    if not skip_stale_refill:
        slots_left = daily_new_names - len(new_rows)
        if slots_left > 0 and ledger.entries:
            stalest = sorted(
                ledger.entries.values(), key=lambda e: e.thesis_updated_on
            )
            refresh = [e.ticker for e in stalest[:slots_left]]
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
