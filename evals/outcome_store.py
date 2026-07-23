"""evals.outcome_store — this repo's single accessor over the long-format
``score_performance_outcomes`` store (EPIC config#1483, consumer cutover
config#1530).

WHY THIS EXISTS
---------------
The eval horizon was historically encoded in wide, horizon-suffixed
``score_performance`` columns (a beat-SPY flag, a stock return, a SPY
return, and — for the canonical primary horizon only — a log-domain alpha,
each suffixed with the horizon in trading days). An incomplete horizon
rename silently starved consumers for months (config#1456 root cause). The
root-cause fix makes the horizon a PARAMETER: outcomes live in the
long-format ``score_performance_outcomes`` table (one row per signal x
horizon; DDL owned by this repo's ``archive/schema.py`` migration 21,
produced by alpha-engine-data ``collectors/signal_returns.py``), and
consumers filter ``WHERE horizon_days = :h`` with ``:h`` resolved from
``nousergon_lib.quant.horizons.HorizonPolicy`` instead of hardcoding
horizon-suffixed column-name literals.

This module is the ONE place in this repo's production code that reads that
store (M0 contract discipline). The three research consumers
(``evals.last_week_scorecard``, ``evals.team_accuracy``, ``memory.episodic``)
only ever need the CANONICAL PRIMARY horizon (21 trading days) joined onto
``score_performance``/other tables by ``(symbol, score_date)`` — none of them
consume the shorter diagnostic horizon — so this accessor exposes exactly
that: one row per resolved primary-horizon signal, decimal units, keyed for
a plain Python-side join (these callers do raw ``sqlite3`` queries, not
pandas).

UNITS — decimals are canonical
-------------------------------
The long store's ``stock_return``/``spy_return`` are DECIMALS (0.043). The
legacy wide return/SPY-return columns were 2dp-rounded PERCENT (4.30) — a
quirk of the wide producer (``round(x * 100, 2)`` in alpha-engine-data). All
three consumers here historically read the wide columns and did
decimal-domain math directly on them mislabeled as percent (e.g. episodic's
f-string ``{outcome_vs_spy:.1%}``) or passed the log-domain alpha column
straight through as an opaque float (it was never percent — always
log-domain decimal in both the wide and long stores, unlike its sibling
return columns). This accessor returns DECIMALS uniformly (the canonical
unit) for every field, including ``stock_return``/``spy_return`` — callers
that need the legacy percent convention for user-facing display must
multiply by 100 explicitly at the render boundary (see
``memory.episodic``'s prompt f-string, which already expected a decimal via
``:.1%`` formatting and therefore needs NO conversion).
"""

from __future__ import annotations

import sqlite3
from typing import NamedTuple

from nousergon_lib.quant.horizons import DEFAULT_POLICY, HorizonPolicy

_TABLE = "score_performance_outcomes"


class PrimaryOutcome(NamedTuple):
    """One resolved canonical-primary-horizon outcome, decimal units."""

    symbol: str
    score_date: str
    beat_spy: int | None  # 0/1, SQLite has no native bool
    stock_return: float | None
    spy_return: float | None
    log_alpha: float | None


def store_exists(conn: sqlite3.Connection) -> bool:
    """True iff the long-format store table exists in this research.db."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_TABLE,)
    ).fetchone()
    return row is not None


def load_primary_outcomes(
    conn: sqlite3.Connection,
    score_date_start: str | None = None,
    score_date_end: str | None = None,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> dict[tuple[str, str], PrimaryOutcome]:
    """Load canonical-primary-horizon outcomes keyed by ``(symbol, score_date)``.

    Args:
        conn: an open research.db connection.
        score_date_start / score_date_end: optional inclusive ISO-date bounds
            on ``score_date`` (mirrors the wide-column callers' existing
            ``BETWEEN ? AND ?`` window filters). ``None`` leaves that bound
            open.
        policy: the active ``HorizonPolicy`` (default: the ratified fleet
            policy, primary=21d).

    Returns ``{(symbol, score_date): PrimaryOutcome}``. Table-absent
    (pre-cutover DB) and zero-rows-in-window (nothing resolved yet for this
    window, e.g. every unit-test fixture, or the current cycle's own
    just-scored signals) both yield an empty dict — graceful-empty, exactly
    like an all-NULL wide-column read. This accessor is intentionally
    single-horizon (primary only): unlike a multi-horizon loader that can
    compare which of several REQUESTED horizons came back empty and flag a
    genuine starvation bug, a primary-only query returning zero rows is
    indistinguishable from "not resolved yet" and must not fail-loud here —
    doing so would raise on every early-window / fresh-DB call, which is the
    normal case for these three consumers, not an error state.
    """
    if not store_exists(conn):
        return {}

    h = policy.primary_horizon
    clauses = ["horizon_days = ?"]
    params: list[object] = [h]
    if score_date_start is not None:
        clauses.append("score_date >= ?")
        params.append(score_date_start)
    if score_date_end is not None:
        clauses.append("score_date <= ?")
        params.append(score_date_end)

    # `_TABLE` is a hardcoded module constant and `clauses` entries are hardcoded
    # column-comparison literals (never date/horizon-value-derived); bound values
    # travel via `params` below.
    sql = (
        "SELECT symbol, score_date, beat_spy, stock_return, spy_return, log_alpha "  # noqa: S608
        f"FROM {_TABLE} WHERE {' AND '.join(clauses)}"
    )
    rows = conn.execute(sql, params).fetchall()

    return {
        (r[0], r[1]): PrimaryOutcome(
            symbol=r[0],
            score_date=r[1],
            beat_spy=r[2],
            stock_return=r[3],
            spy_return=r[4],
            log_alpha=r[5],
        )
        for r in rows
    }
