"""The five ``research``-slot arms (Brian's ruling 2026-09-22,
alpha-engine-config-I11393 / -I11398).

ONE SLOT, ONE PINNED PRE-FILTER, FIVE ARMS. The slot decides which names reach
the predictor, and it replaces the scanner_spec / universe_cut / producer split
— three independently-promoting pointers over successive stages of one
pipeline, which optimised each link separately and measured the composition
(the objective) nowhere.

    arm                  input                     ranks by            width
    ------------------   -----------------------   -----------------   -----
    attractiveness_60    pinned attractiveness_60   attractiveness      60
    attractiveness_20    pinned attractiveness_60   attractiveness      20
    tech_score_20        pinned attractiveness_60   tech_score          20
    predictor_from_60    pinned attractiveness_60   predicted_alpha     20
    thinktank_20         pinned attractiveness_60   TT thesis rating    20

``attractiveness_60`` is also the slot's BASELINE: the scanner's own opinion,
unmodified. Every other arm has to beat "just take the names the scanner
already ranked" to justify its cost, which closes §9.2's "we never checked" by
construction rather than by intention.

WHY THE WIDTHS DIFFER, AND WHY THAT IS NOT §4's CONFOUND.
``attractiveness_60`` against ``attractiveness_20`` is the SAME ranking at two
depths — it is the "what does depth cost?" experiment asked directly, rather
than inferred from two arms that also differ in ranking. §4's count-matching
rule binds when the RANKING is under test ("the selection-rule question — the
one being asked — becomes unanswerable"); here width IS the thing under test,
and the slot prices the resulting concentration through its
``information_ratio`` primary metric instead of forcing every arm to one width.
See ``scoring.leaderboard_scoring.information_ratio_stats`` for why a raw mean
cannot do that job: if a ranking carries any skill, mean alpha per name
declines with depth, so the narrowest arm wins on depth alone.

WHY THE PRE-FILTER IS PINNED. Every arm draws from
``producers.registry.PINNED_RESEARCH_PREFILTER``, resolved by
``scoring.universe_membership.resolve_pinned_cut`` — never through the live
champion pointer. Measured 2026-09-18: the universe_cut pointer moved between
two cuts sharing ZERO of 60 names, every pointer-resolving consumer had its
entire input population replaced, and no arm's spec hash changed. §3.1 makes an
arm an immutable RECIPE; "whatever cut won this week" is not one. A different
pre-filter is tested by REGISTERING AN ARM.

Arms in this slot therefore differ in exactly two declared things — the ranking
key and the width — and in nothing else. That is what makes a comparison
between any two of them mean something.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from data.fetchers.price_fetcher import fetch_sp500_sp400_with_sectors
from producers.filling_arms import (
    PREDICTIONS_RESEARCH_FREE_KEY,
    FillingShadowError,
    _get_json,
    _predictions_by_ticker,
    _s3,
    build_shadow_payload,
    rank_by_alpha,
)
from scoring.universe_membership import resolve_pinned_cut
from thinktank.context import UNIVERSE_BOARD_KEY

logger = logging.getLogger(__name__)


MIN_BOARD_JOIN_FRACTION = 0.9
"""Fraction of the pinned cut that must carry a usable ranking value.

The cut and the universe board are written by the SAME scanner invocation, so a
near-total join is the healthy state and anything well below it is a producer
divergence, not noise (alpha-engine-config-I7844 measured the shape: the board
carried 903 rows against 906 ranks, and one ranked name was absent entirely).

0.9 rather than 1.0 because a single halted or newly-listed ticker legitimately
lands without a score, and RAISING on one name would take the whole arm down
for a condition the arm can absorb. Below it, the arm's premise — that it ranks
THE PINNED CUT — is false, and it must say so rather than silently rank a
handful of survivors: that is exactly the defect alpha-engine-config-I11396
recorded on the sibling predictor arm, where a cut of 20 met an artifact of 25
thesis-covered names and the pool collapsed to ONE.
"""


def _pinned_cut(s3: Any, bucket: str) -> tuple[list[str], dict]:
    """The pinned pre-filter's tickers and provenance, or raise."""
    # Imported here rather than at module scope: registry imports this module's
    # builders, so a module-scope import of the registry would be circular.
    from producers.registry import PINNED_RESEARCH_PREFILTER

    tickers, _ranks, provenance = resolve_pinned_cut(
        PINNED_RESEARCH_PREFILTER, bucket=bucket, s3_client=s3,
    )
    return tickers, provenance


def _assert_join(
    arm: str, cut: list[str], rows: list[tuple[str, float]], *, source: str, artifact: str,
) -> None:
    """RAISE when the ranking artifact does not cover the pinned cut.

    A PRODUCER, so the fleet rule is raise — never a silent swallow, never a
    graceful degrade on a partial join. The message names both population sizes
    and the measured fraction so the fix is one hop away rather than a hunt.
    """
    if not cut:
        raise FillingShadowError(f"{arm}: the pinned cut resolved to zero names")
    fraction = len(rows) / len(cut)
    if fraction < MIN_BOARD_JOIN_FRACTION:
        raise FillingShadowError(
            f"{arm}: only {len(rows)} of {len(cut)} names in the pinned cut "
            f"({fraction:.1%}, floor {MIN_BOARD_JOIN_FRACTION:.0%}) carry a "
            f"usable ranking value in {artifact!r} (source={source!r}). This "
            "arm's premise is that it ranks THE PINNED CUT; ranking the "
            "survivors instead would report a healthy arm measuring something "
            "nobody chose (alpha-engine-config-I11396)."
        )


def load_board_ranked_pool(
    s3: Any, bucket: str, run_date: str, *, ranking_key: str, arm: str,
) -> tuple[list[tuple[str, float]], str]:
    """Rank the pinned cut by a per-name field of the scanner's universe board.

    ``ranking_key`` is the board field — ``attractiveness_score`` or
    ``tech_score``. The board supplies ROW CONTENT; the pinned cut supplies
    MEMBERSHIP. That division is deliberate and is the same one
    ``thinktank.feed.join_board_rows`` draws: re-deriving membership from the
    board's own ordering is how a consumer ends up with a second opinion about
    a ranking the producer already declared (alpha-engine-config-I7842).
    """
    cut, provenance = _pinned_cut(s3, bucket)
    board = _get_json(s3, bucket, UNIVERSE_BOARD_KEY)
    if not board:
        raise FillingShadowError(
            f"{arm}: no s3://{bucket}/{UNIVERSE_BOARD_KEY} — the pinned cut "
            "supplies membership but the board supplies the ranking value, so "
            "there is nothing to rank without it. Producer: the Scanner Lambda "
            "(scoring/universe_board.py)."
        )
    by_ticker = {
        str(row.get("ticker")): row
        for row in (board.get("stocks") or [])
        if row.get("ticker")
    }
    rows: list[tuple[str, float]] = []
    for ticker in cut:
        value = (by_ticker.get(ticker) or {}).get(ranking_key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN-safe
            continue
        rows.append((ticker, value))
    _assert_join(arm, cut, rows, source=ranking_key, artifact=UNIVERSE_BOARD_KEY)
    logger.info(
        "[research_arms] %s: ranked %d/%d of pinned cut %s by %s (board run_date=%s)",
        arm, len(rows), len(cut), provenance.get("cut"), ranking_key,
        provenance.get("run_date"),
    )
    return rank_by_alpha(rows), f"pinned_cut:{provenance.get('cut')}:{ranking_key}"


def load_pinned_cut_predictor_pool(
    s3: Any, bucket: str, run_date: str, *, arm: str,
) -> tuple[list[tuple[str, float]], str]:
    """Rank the pinned cut by the predictor's research-free ``predicted_alpha``.

    Reads ``predictor/predictions_research_free/{run_date}.json`` — the
    artifact whose population IS the scanner candidate pool — and NOT
    ``predictor/predictions/{date}.json``, whose ~25-name thesis-coverage
    universe is near-disjoint from any scanner cut. That distinction is not
    theoretical: it is alpha-engine-config-I11396, where the sibling arm joined
    a 20-name cut against the 25-name artifact, collapsed to a pool of ONE on
    two dates, and shared zero picks with its pair on 7 of 8 dates for weeks.
    """
    cut, provenance = _pinned_cut(s3, bucket)
    key = PREDICTIONS_RESEARCH_FREE_KEY.format(date=run_date)
    doc = _get_json(s3, bucket, key)
    if not doc:
        raise FillingShadowError(
            f"{arm}: no s3://{bucket}/{key} — this arm's ranking IS the "
            "predictor's research-free output. Producer: crucible-predictor "
            "inference/stages/research_free.py, daily PredictorInference Lambda."
        )
    alphas = _predictions_by_ticker(doc)
    in_cut = set(cut)
    rows = [
        (ticker, alpha) for ticker, alpha in alphas.items()
        if ticker in in_cut and alpha == alpha  # NaN-safe
    ]
    _assert_join(arm, cut, rows, source="predicted_alpha", artifact=key)
    logger.info(
        "[research_arms] %s: ranked %d/%d of pinned cut %s by predicted_alpha "
        "(artifact population %d)",
        arm, len(rows), len(cut), provenance.get("cut"), len(alphas),
    )
    return rank_by_alpha(rows), f"pinned_cut:{provenance.get('cut')}:predicted_alpha"


def build_research_shadow(arm: str, run_date: str, archive_manager, **_ctx) -> dict:
    """Build one research arm's shadow, at the arm's OWN declared width.

    The width comes from ``ProducerSpec.width`` rather than a module constant,
    because it is part of the arm's immutable recipe (§3.1) and the arms of
    this slot deliberately differ in it. Reading it from the register is also
    what makes ``_assert_emitted_width_matches_declared`` in
    ``producers.registry`` a real check rather than a restatement: the emitted
    count and the declared count come from one place, so they cannot drift
    apart the way ``no_agent_quant``'s did (25 -> 17 -> 40 across five weeks,
    compared by §4.1's ladder as one continuous series).
    """
    from producers.registry import RESEARCH_PRODUCERS

    spec = RESEARCH_PRODUCERS.get(arm)
    if spec is None or spec.width is None:
        raise FillingShadowError(
            f"research arm {arm!r} is not registered with a declared width — "
            "an arm emits its declared number of picks or it is not an arm "
            "(alpha-engine-config-I11393)."
        )
    loader = _LOADERS.get(arm)
    if loader is None:
        raise FillingShadowError(
            f"no pool loader registered for research arm {arm!r} — "
            f"known: {sorted(_LOADERS)}"
        )
    s3, bucket = _s3(archive_manager)
    ranked, pool_source = loader(s3, bucket, run_date, arm=arm)
    sector_map = _ctx.get("sector_map")
    if not isinstance(sector_map, Mapping):
        _, sector_map = fetch_sp500_sp400_with_sectors()
    payload = build_shadow_payload(
        arm, run_date, ranked,
        pool_size=len(ranked),
        pool_source=pool_source,
        top_n=spec.width,
        sector_map=sector_map,
    )
    emitted = len(payload["signals"])
    if emitted != spec.width:
        raise FillingShadowError(
            f"{arm}: emitted {emitted} picks but declares width={spec.width}. "
            "An arm's width is part of its immutable recipe; an arm that "
            "quietly emits a different count is a different arm, and §4.1's "
            "ladder would compare the two as one series "
            "(alpha-engine-config-I11393)."
        )
    logger.info(
        "[research_arms] %s %s: %d picks from pool=%s (%d ranked)",
        arm, run_date, emitted, pool_source, len(ranked),
    )
    return payload


def _board_loader(ranking_key: str):
    def _loader(s3, bucket, run_date, *, arm):
        return load_board_ranked_pool(
            s3, bucket, run_date, ranking_key=ranking_key, arm=arm,
        )
    return _loader


_LOADERS = {
    "attractiveness_60": _board_loader("attractiveness_score"),
    "attractiveness_20": _board_loader("attractiveness_score"),
    "tech_score_20": _board_loader("tech_score"),
    "predictor_from_60": load_pinned_cut_predictor_pool,
}
"""Pool loader per arm. ``thinktank_20`` is absent BY DESIGN — its shadow is
written by the Think Tank's own daily run (``thinktank.challenger_selection``),
not synthesised during the weekly producer pass, so its spec carries
``build=None`` and ``producers.runner`` skips it while the leaderboard scores
it like any other arm."""


def run_attractiveness_60(run_date: str, archive_manager, **ctx) -> dict:
    """``ProducerSpec.build`` for ``attractiveness_60`` — the slot's baseline."""
    return build_research_shadow("attractiveness_60", run_date, archive_manager, **ctx)


def run_attractiveness_20(run_date: str, archive_manager, **ctx) -> dict:
    """``ProducerSpec.build`` for ``attractiveness_20``."""
    return build_research_shadow("attractiveness_20", run_date, archive_manager, **ctx)


def run_tech_score_20(run_date: str, archive_manager, **ctx) -> dict:
    """``ProducerSpec.build`` for ``tech_score_20``."""
    return build_research_shadow("tech_score_20", run_date, archive_manager, **ctx)


def run_predictor_from_60(run_date: str, archive_manager, **ctx) -> dict:
    """``ProducerSpec.build`` for ``predictor_from_60``."""
    return build_research_shadow("predictor_from_60", run_date, archive_manager, **ctx)
