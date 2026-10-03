"""Declared per-ticker holds: names the scanner must not evaluate this cycle.

A hold removes a ticker from the scanner universe in
``scanner_orchestrator.build_candidates_artifact``, before ``run_quant_filter``
runs. Every downstream set is derived from that universe: the eval log, the
universe board's rows, every attractiveness and tech_score rank table
(``universe_membership._restrict_to_scanned_universe``), and so every cut the
champion pointer can serve. A held name therefore cannot reach a pick through
any arm. The name stays in the S&P constituents and in every data store; only
its candidacy is suspended.

A hold is for a name whose INPUTS are known to be wrong, not for a view on the
name. The first entry is Corteva (CTVA), whose price history carries an
unadjusted 2026-10-01 spin-off break (77.65 -> 12.57) until the declared
adjustment in nousergon-data lands (alpha-engine-config-I11806). Until then its
return, momentum and ATR features read a false -84% day, and its P/E reads 1.75
because the post-spin price is divided by the pre-spin company's earnings.

Each hold carries an absolute ``valid_through`` date, so a forgotten entry
expires instead of excluding the name for ever, plus a ``why`` and the issue
that owns removing it. Entries change by PR only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TickerHold:
    valid_from: str  # YYYY-MM-DD, inclusive
    valid_through: str  # YYYY-MM-DD, inclusive; the hold is ignored after it
    why: str
    issue: str


DECLARED_HOLDS: dict[str, TickerHold] = {
    "CTVA": TickerHold(
        valid_from="2026-10-01",
        valid_through="2026-10-31",
        why=(
            "2026-10-01 spin-off of VYLR (1:1) left CTVA's stored price history "
            "unadjusted (77.65 -> 12.57, a false -84% day) and its P/E at 1.75; "
            "held until the declared CRSP adjustment (factor 0.1555) lands in "
            "nousergon-data and the features are rebuilt"
        ),
        issue="alpha-engine-config-I11806",
    ),
}


def active_holds(run_date: str, holds: dict[str, TickerHold] | None = None) -> dict[str, TickerHold]:
    """The holds in force on ``run_date`` (both bounds inclusive)."""
    d = date.fromisoformat(str(run_date)[:10])
    table = DECLARED_HOLDS if holds is None else holds
    return {
        ticker: hold
        for ticker, hold in table.items()
        if date.fromisoformat(hold.valid_from) <= d <= date.fromisoformat(hold.valid_through)
    }
