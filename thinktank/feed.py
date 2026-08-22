"""Think Tank's coverage window, read from the scanner→consumer contract.

Think Tank does not decide which names it covers. The Scanner does, and it
says so in ``s3://alpha-engine-research/universe_membership/latest.json``:

    funnel.advances_to.thinktank_coverage_window -> "<cut name>"

and, when that cut is one of the promotable feed arms, the live champion
pointer (``config/scanner_cut_champion.json``) says which ARM is serving that
slot this week. This module is the only place in ``thinktank/`` that resolves
either, and it resolves both through
``scoring.universe_membership.resolve_funnel_cut`` — the same reader the sector
teams use, so there is exactly one implementation of "what does the scanner
say".

What this replaces (alpha-engine-config-I7842). Until this module existed,
``thinktank.ledger.ranked_universe`` sorted ``scanner/universe/latest.json``'s
``stocks[]`` by ``attractiveness_score`` in Python, and ``thinktank.run``
re-sorted the same rows a second time for the coverage-gap metric. Three
consequences, none of which raised anything:

  1. **The basis was hardcoded.** The moment the champion is a tech-basis cut,
     Think Tank keeps ranking on attractiveness and covers a different set —
     a cutover replacing the researched set with nothing raising, which is
     alpha-engine-config-I7808 verbatim.
  2. **Board position is not universe rank.** Measured 2026-08-20: the board
     carries 903 rows, ``ranks`` carries 906, and ``EQR`` (rank 98) is absent
     from the board — so every board position below 98 was off by one against
     the declared rank. At ``exit_rank: 200`` that is the difference between
     retaining and de-covering ``DT`` (board 200, declared 201).
     Filed as alpha-engine-config-I7844.
  3. **Two rankings, one truth.** The gap metric and the gap-fill selection
     each sorted independently, so a divergence between them would have been
     silent.

Division of labour, deliberately: the membership artifact supplies MEMBERSHIP
and ORDER; the universe board supplies per-name ROW CONTENT (sector, pillars,
scores) that the analyst prompt reads. They are joined here, and a cut member
with no board row is a hard error — the cut is the contract, and underwriting
a name with no row content is not something to do quietly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from scoring.universe_membership import (
    FUNNEL_CONSUMER_THINKTANK,
    UniverseMembershipError,
    resolve_funnel_cut,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedWindow:
    """The scanner's answer to "which names, in what order, by what rule".

    ``tickers`` is the coverage window — the cut the artifact declares for
    Think Tank, at the width the artifact declares. ``ordered`` is the full
    ranking in the serving arm's basis, rank 1 first; it is what a rank ceiling
    (``rank_ceiling``, ``exit_rank``) is measured against, and it is wider than
    the window by design (the window is 60, the daily intake ceiling is 150).
    """

    tickers: tuple[str, ...]
    ordered: tuple[str, ...]
    rank_by_ticker: dict[str, int]
    basis: str
    cut: str
    provenance: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        """The declared window width. Consumers use this instead of a constant:
        a consumer hardcoding 60 while the producer declares 60 is the same
        duplicated-truth defect this module exists to remove, one level down."""
        return len(self.tickers)

    def rank_of(self, ticker: str) -> int | None:
        """1-based rank in the serving basis, or ``None`` if unranked.

        ``None`` means the scanner did not rank this name today — a universe
        change, a data gap, a halted ticker. It does NOT mean "ranked worse
        than anything", and callers must not treat it as one (the hysteresis
        exit in ``thinktank.ledger`` turns only on an EXPLICIT rank)."""
        return self.rank_by_ticker.get(ticker)


def build_feed_window(
    tickers: list[str],
    ranks: dict[str, int],
    provenance: dict,
) -> FeedWindow:
    """Assemble + validate a :class:`FeedWindow`. Pure — unit-testable without S3.

    Asserts the cut is the HEAD of its own basis' ranking. That holds by
    construction in the producer (every cut is a ``_top_n`` of the table its
    ``basis`` names), which is exactly why it went unchecked there and is worth
    checking here: if it ever stops holding, a rank ceiling and a cut membership
    are two different opinions about the same question, and the consumer would
    quietly average them.
    """
    if not tickers:
        raise UniverseMembershipError(
            f"universe_membership names {provenance.get('cut')!r} as Think Tank's "
            "coverage window but it carries no tickers"
        )
    ordered = tuple(t for t, _ in sorted(ranks.items(), key=lambda kv: kv[1]))
    head = set(ordered[: len(tickers)])
    window = set(tickers)
    if head != window:
        raise UniverseMembershipError(
            f"universe_membership (run_date={provenance.get('run_date')}): cut "
            f"{provenance.get('cut')!r} is not the head of its own "
            f"{provenance.get('basis')!r} ranking — {len(window - head)} of "
            f"{len(window)} name(s) in the cut are outside the top {len(tickers)} "
            f"of the rank table ({sorted(window - head)}). The cut and the rank "
            "ceiling would then be two different orderings of the same universe, "
            "and Think Tank would be using both."
        )
    declared_size = provenance.get("declared_size")
    if declared_size is not None and int(declared_size) != len(tickers):
        raise UniverseMembershipError(
            f"universe_membership (run_date={provenance.get('run_date')}): cut "
            f"{provenance.get('cut')!r} declares size={declared_size} but carries "
            f"{len(tickers)} tickers"
        )
    return FeedWindow(
        tickers=tuple(tickers),
        ordered=ordered,
        rank_by_ticker=dict(ranks),
        basis=str(provenance.get("basis")),
        cut=str(provenance.get("cut")),
        provenance=dict(provenance),
    )


def load_feed_window(
    store: Any,
    *,
    minimum_rank_coverage: int | None = None,
) -> FeedWindow:
    """Read the live contract and return Think Tank's window.

    ``minimum_rank_coverage`` is the widest rank the caller will ask about —
    ``settings.exit_rank`` for the daily cadence. Passing it turns "the rank
    table is too short to answer my question" into a loud refusal instead of a
    silent de-coverage of every name past the table's end.

    Raises :class:`~scoring.universe_membership.UniverseMembershipError` on
    every ambiguity. There is no degraded mode and no board fallback: an empty
    or defaulted window is indistinguishable from "the scanner selected
    nobody", and re-deriving the ranking locally is precisely the defect this
    module removes.
    """
    tickers, ranks, provenance = resolve_funnel_cut(
        FUNNEL_CONSUMER_THINKTANK,
        bucket=store.bucket,
        s3_client=store.s3,
        minimum_rank_coverage=minimum_rank_coverage,
    )
    window = build_feed_window(tickers, ranks, provenance)
    logger.info(
        "[thinktank] coverage window: %d names from cut %r (declared %r, basis %r) "
        "of universe_membership run_date=%s cut_effective_date=%s; rank table "
        "covers %d names",
        window.size,
        window.cut,
        provenance.get("declared_cut"),
        window.basis,
        provenance.get("run_date"),
        provenance.get("cut_effective_date"),
        len(window.rank_by_ticker),
    )
    return window


def join_board_rows(window: FeedWindow, board: dict | None) -> dict[str, dict]:
    """``{ticker: board_row}`` for every name in the WINDOW, or raise.

    The board supplies the row content the analyst reads; the window supplies
    who is in it. A cut member with no board row is a hard error — the two
    artifacts are written by the same Scanner invocation, so a missing row is a
    producer-side divergence (alpha-engine-config-I7844), not a state Think
    Tank should paper over by underwriting a name it has no data for.
    """
    rows = {str(s.get("ticker")): s for s in ((board or {}).get("stocks") or []) if s.get("ticker")}
    missing = sorted(t for t in window.tickers if t not in rows)
    if missing:
        raise UniverseMembershipError(
            f"{len(missing)} of {window.size} ticker(s) in Think Tank's coverage "
            f"window {window.cut!r} have no row in scanner/universe/latest.json "
            f"({missing}). The window and the board come from the same Scanner "
            "run, so this is a producer divergence "
            "(alpha-engine-config-I7844) — refusing to underwrite a name with "
            "no row content."
        )
    return rows


def unjoinable_ranked_names(
    window: FeedWindow,
    board: dict | None,
    rank_ceiling: int,
) -> list[str]:
    """Names the scanner ranked inside ``rank_ceiling`` that the board omits.

    Always empty inside the declared window (:func:`join_board_rows` raises
    there); it is the band between the window and the intake ceiling this can
    be non-empty for. Emitted into the run manifest so the count has a durable
    surface rather than living only in a log line — a name silently dropped
    from intake and a name legitimately not selected must not look the same.

    Measured 2026-08-20: ``EQR`` (rank 98) is exactly this — ranked well inside
    ``rank_ceiling: 150`` and absent from the board's 903 rows
    (alpha-engine-config-I7844).
    """
    rows = {str(s.get("ticker")) for s in ((board or {}).get("stocks") or []) if s.get("ticker")}
    return [t for i, t in enumerate(window.ordered[:rank_ceiling], start=1) if t not in rows]
