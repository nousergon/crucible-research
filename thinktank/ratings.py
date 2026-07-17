"""Ratings board — the denormalized rollup of every covered name's view.

Written at the end of every run to ``thinktank/ratings/{trading_day}.json``
plus the ``latest.json`` pointer, upserted incrementally: rows for theses
written this run are rebuilt from the fresh artifacts; covered names the
run didn't touch keep their prior row (self-healed from the per-ticker
``theses/{ticker}/latest.json`` when the board has no row yet — e.g. the
first run after this module shipped); names dropped from the ledger are
pruned. One S3 read for consumers (console, eval joins) instead of N
per-ticker fetches.

The ``rating_minus_attractiveness`` divergence column is the point of the
board: the analyst's independent 0-100 call (which never saw the scanner
composite — see ``analyst._facts_board_row``) minus the scanner's
``attractiveness_score`` at thesis-write time. Large gaps in either
direction are the interesting cohort for the config#1580 restructure
evidence.

Since config#2678, ``rating`` is the OPERATIVE value — the raw LLM rating
blended with the pillar composite via ``thinktank.pillars.blend_rating``
(still fully scanner-independent; the pillar extraction is scanner-blind
too, see ``analyst.build_thesis``). ``raw_llm_rating`` preserves the
pre-blend value for audit/divergence display.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from thinktank import RATINGS_KEY_TMPL, RATINGS_LATEST_KEY
from thinktank.analyst import load_latest_thesis
from thinktank.pillars import blend_rating
from thinktank.schemas import CompanyThesis, CoverageLedger, RatingRow, RatingsBoard
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)


def _row_from_thesis(thesis: CompanyThesis) -> RatingRow:
    llm = thesis.thesis
    raw_rating = llm.rating
    rating = (
        blend_rating(raw_rating, thesis.pillar_assessment)
        if raw_rating is not None
        else None
    )
    delta: float | None = None
    if rating is not None and thesis.attractiveness_score is not None:
        delta = round(float(rating) - float(thesis.attractiveness_score), 2)
    return RatingRow(
        ticker=thesis.ticker,
        sector=thesis.sector,
        rating=rating,
        raw_llm_rating=raw_rating,
        rating_rationale=llm.rating_rationale,
        stance=llm.stance,
        conviction=llm.conviction,
        summary=llm.summary,
        thesis_version=thesis.version,
        thesis_trading_day=thesis.trading_day,
        update_reason=thesis.update_reason,
        attractiveness_score=thesis.attractiveness_score,
        attractiveness_rank=thesis.attractiveness_rank,
        rating_minus_attractiveness=delta,
    )


def update_ratings_board(
    store: ThinktankStore,
    ledger: CoverageLedger,
    theses_written: list[CompanyThesis],
    *,
    trading_day: str,
) -> RatingsBoard:
    """Upsert this run's theses into the board and persist dated + latest."""
    raw = store.get_json(RATINGS_LATEST_KEY)
    board = RatingsBoard.model_validate(raw) if raw is not None else RatingsBoard()

    for thesis in theses_written:
        board.rows[thesis.ticker] = _row_from_thesis(thesis)

    covered = ledger.covered()
    # Self-heal: covered names with no board row yet (theses that predate
    # this module). One-time per-ticker latest.json read, then the row
    # sticks. A missing thesis artifact for a covered name is a real
    # contract violation — fail loud, never a silent hole in the board.
    for ticker in sorted(covered - set(board.rows)):
        thesis = load_latest_thesis(store, ticker)
        if thesis is None:
            raise RuntimeError(
                f"ratings board: ledger covers {ticker} but "
                f"thinktank/theses/{ticker}/latest.json is missing — "
                "coverage ledger and thesis store are out of sync."
            )
        board.rows[ticker] = _row_from_thesis(thesis)

    # Prune names no longer covered so the board mirrors the ledger.
    for ticker in set(board.rows) - covered:
        del board.rows[ticker]

    board.trading_day = trading_day
    board.updated_at = datetime.now(timezone.utc).isoformat()
    payload = board.model_dump()
    store.put_json(RATINGS_KEY_TMPL.format(trading_day=trading_day), payload)
    store.put_json(RATINGS_LATEST_KEY, payload)
    rated = sum(1 for r in board.rows.values() if r.rating is not None)
    logger.info(
        "ratings board written: %d rows (%d rated) for %s",
        len(board.rows),
        rated,
        trading_day,
    )
    return board
