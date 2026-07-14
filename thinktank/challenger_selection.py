"""Challenger-selection artifact — Think Tank's top-N covered names by
independent rating, for the champion/challenger leaderboard (epic
alpha-engine-config-I2515: champion = scanner→predictor direct, already
live; Think Tank is the CHALLENGER arm).

Written at the tail of every non-dry ``run_daily`` (after the ratings-board
upsert) to ``thinktank/challenger_selection/{trading_day}.json`` +
``latest.json`` (see ``thinktank.__init__`` for the key templates). ALWAYS
emitted for observability, but ``coverage_complete`` is the validity flag
downstream consumers (leaderboard, evaluator) must gate on — Brian's ruling
(config#1580): the selection only counts once the ENTIRE current-scan
top-``thinktank.run.GAP_FILL_TOP_N`` window is covered.

Names come from the ratings board — already independently rated, never the
scanner attractiveness ranking (independence is the point; the ranking
board's ``attractiveness_rank`` rides along on each row as metadata only,
see ``thinktank.ratings``).

Freshness (Brian, 2026-07-14, config#1580): the weekly SF must never rely
on week-old data, but the daily runs legitimately read Saturday's universe
board all week — this module does NOT hard-fail on a stale board. Instead
``board_date`` (the universe board's ``as_of`` at ranking time) is carried
so downstream consumers can verify same-day-ness themselves.

Leaderboard shadow view (epic alpha-engine-config-I2515 scope addendum):
the shared champion/challenger leaderboard scorer
(``scoring/leaderboard_producers.py::_load_producer_specs`` →
``_enter_ranked_and_scores``) can only join a challenger's picks from
``signals_shadow/{producer}/{trading_day}/signals.json`` in its conforming
shape (a top-level ``signals`` dict keyed by ticker, each entry carrying
``signal == "ENTER"`` and a numeric ``score``) — it cannot read this
module's native artifact directly. So, in addition to the native artifact,
this module ALSO writes that conforming shadow view — but ONLY when
``coverage_complete`` is True (Brian's ruling: an incomplete-coverage
selection must never enter the leaderboard as challenger evidence). The
native artifact is always written regardless, for observability.

Written via ``ThinktankStore.put_json`` directly rather than reusing
``archive.manager.ArchiveManager.write_shadow_signals_json``: that method
constructs its OWN ``boto3`` client against ``config.S3_BUCKET`` (an env-var
default), independent of ``settings.bucket`` / ``ThinktankStore.bucket``.
Think Tank's bucket is independently configurable (``thinktank.yaml``); a
divergence between the two would silently write the shadow cohort to the
wrong bucket relative to everything else this module reads/writes (the
ratings board, the universe board, the coverage ledger), breaking the join
without any error. Reusing ``ThinktankStore`` keeps this write on the exact
same bucket as every other thinktank artifact — this module mirrors that
method's payload SHAPE exactly (``{"date", "run_date", **signals}``)
without inheriting its client-construction risk.
"""

from __future__ import annotations

import logging

from thinktank import (
    CHALLENGER_SELECTION_KEY_TMPL,
    CHALLENGER_SELECTION_LATEST_KEY,
    CHALLENGER_SHADOW_SIGNALS_KEY_TMPL,
)
from thinktank.schemas import (
    ChallengerSelection,
    ChallengerSelectionRow,
    CoverageLedger,
    RatingsBoard,
)
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)


CHALLENGER_TOP_N = 20
"""Leaderboard submission size — the top covered/rated names by Think
Tank's own rating. Distinct from ``thinktank.run.GAP_FILL_TOP_N`` (60, the
coverage-window size that gates ``coverage_complete``); this is how many of
those covered names actually get submitted as the challenger arm's picks."""


def write_challenger_selection(
    store: ThinktankStore,
    ledger: CoverageLedger,
    ratings_board: RatingsBoard,
    *,
    run_id: str,
    mode: str,
    trading_day: str,
    calendar_date: str,
    board_date: str | None,
    coverage_gap: dict,
) -> ChallengerSelection:
    """Upsert this run's top-``CHALLENGER_TOP_N`` rated names and persist
    dated + latest. ``coverage_gap`` is the manifest's already-computed
    ``_compute_coverage_gap`` dict (same GAP_FILL_TOP_N window)."""
    covered = ledger.covered()
    if covered and not ratings_board.rows:
        # Fleet rule: a missing/empty ratings board when the ledger is
        # non-empty is a contract violation between the two producers
        # written moments apart in the same run — RAISE, never skip.
        raise RuntimeError(
            f"challenger selection: coverage ledger has {len(covered)} "
            "covered names but the ratings board has no rows — ratings "
            "board and coverage ledger are out of sync."
        )

    uncovered_count = coverage_gap.get("uncovered_count")
    if uncovered_count is None:
        raise RuntimeError(
            "challenger selection: coverage_gap is missing "
            f"'uncovered_count' ({coverage_gap!r}) — cannot determine "
            "coverage_complete."
        )

    # Rank by Think Tank's OWN rating only; a None rating (thesis predates
    # the rating field) has no basis to rank and is excluded from the pool.
    rated_rows = [r for r in ratings_board.rows.values() if r.rating is not None]
    rated_rows.sort(key=lambda r: r.rating, reverse=True)
    top_rows = rated_rows[:CHALLENGER_TOP_N]

    selection = ChallengerSelection(
        trading_day=trading_day,
        calendar_date=calendar_date,
        run_id=run_id,
        mode=mode,
        board_date=board_date,
        coverage_complete=(uncovered_count == 0),
        uncovered_count=uncovered_count,
        selections=[
            ChallengerSelectionRow(
                ticker=r.ticker,
                rating=r.rating,
                stance=r.stance,
                conviction=r.conviction,
                thesis_version=r.thesis_version,
                attractiveness_rank=r.attractiveness_rank,
            )
            for r in top_rows
        ],
    )
    payload = selection.model_dump()
    store.put_json(CHALLENGER_SELECTION_KEY_TMPL.format(trading_day=trading_day), payload)
    store.put_json(CHALLENGER_SELECTION_LATEST_KEY, payload)
    logger.info(
        "challenger selection written: %d/%d names, coverage_complete=%s "
        "(uncovered=%d) for %s",
        len(selection.selections),
        CHALLENGER_TOP_N,
        selection.coverage_complete,
        uncovered_count,
        trading_day,
    )

    shadow_key = _write_shadow_signals(store, selection)
    if shadow_key:
        logger.info("challenger shadow signals written: %s", shadow_key)

    return selection


def _write_shadow_signals(store: ThinktankStore, selection: ChallengerSelection) -> str | None:
    """Conforming ``signals_shadow/thinktank_coverage/{trading_day}/signals.json``
    for the shared leaderboard scorer — ONLY when ``coverage_complete``
    (see module docstring). Returns the S3 key written, or None when skipped."""
    if not selection.coverage_complete:
        logger.info(
            "challenger shadow signals skipped for %s — coverage incomplete "
            "(uncovered=%d), not valid leaderboard evidence",
            selection.trading_day,
            selection.uncovered_count,
        )
        return None

    # score = TT's own independent rating (0-100) — the ranking signal the
    # scorer's _enter_ranked_and_scores reduces on. Every selected name is,
    # by construction, an ENTER pick (this IS Think Tank's top-N submission).
    signals = {
        row.ticker: {
            "ticker": row.ticker,
            "signal": "ENTER",
            "score": float(row.rating),
            "stance": row.stance,
            "conviction": row.conviction,
            "thesis_version": row.thesis_version,
            "attractiveness_rank": row.attractiveness_rank,
        }
        for row in selection.selections
    }
    payload = {
        "date": selection.trading_day,
        "run_date": selection.calendar_date,
        "signals": signals,
    }
    key = CHALLENGER_SHADOW_SIGNALS_KEY_TMPL.format(trading_day=selection.trading_day)
    store.put_json(key, payload)
    return key
