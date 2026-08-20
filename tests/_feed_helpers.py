"""Build a :class:`thinktank.feed.FeedWindow` from a test universe board.

Test fixtures used to hand ``select_intake`` a board and let it derive its own
ranking. That derivation is gone (alpha-engine-config-I7842), so a fixture now
has to supply the ranking the SCANNER would have published. This helper builds
the membership-artifact side of the contract from the same board the fixture
already describes, so a test still states its universe once.

Names with a null ``attractiveness_score`` are absent from the rank table
entirely — the producer filters unrankable names before ranking
(``build_universe_membership``), and absence is what an unrankable name looks
like in the artifact. That preserves the old "a null score is never eligible
for intake" behaviour without the consumer inspecting scores.
"""

from __future__ import annotations

from thinktank.feed import FeedWindow, build_feed_window

DEFAULT_WINDOW_SIZE = 60


def feed_window_for(
    board: dict,
    *,
    size: int | None = None,
    cut: str = "attractiveness_top_60",
    basis: str = "attractiveness_rank",
) -> FeedWindow:
    scored = [
        s for s in (board or {}).get("stocks") or [] if s.get("ticker") and s.get("attractiveness_score") is not None
    ]
    ordered = sorted(scored, key=lambda s: (-s["attractiveness_score"], s["ticker"]))
    ranks = {s["ticker"]: i + 1 for i, s in enumerate(ordered)}
    n = min(size if size is not None else DEFAULT_WINDOW_SIZE, len(ordered))
    tickers = sorted(t for t, r in ranks.items() if r <= n)
    return build_feed_window(
        tickers,
        ranks,
        {
            "consumer": "thinktank_coverage_window",
            "cut": cut,
            "declared_cut": cut,
            "basis": basis,
            "size": len(tickers),
            "declared_size": len(tickers),
            "run_date": board.get("as_of", "2026-08-20"),
            "cut_effective_date": board.get("as_of", "2026-08-20"),
        },
    )


def membership_for_board(
    board: dict,
    *,
    size: int | None = None,
    run_date: str = "2026-08-20",
    cut: str = "attractiveness_top_60",
) -> dict:
    """A ``universe_membership/latest.json`` body matching ``board``.

    Seeded alongside the board wherever a test exercises a Think Tank entry
    point: the window is READ from this artifact now, and a test that seeds only
    the board is testing a fleet state that fails loud by design.
    """
    window = feed_window_for(board, size=size, cut=cut)
    return {
        "schema_version": 1,
        "producer": "crucible-research/scoring/universe_membership.py",
        "run_date": run_date,
        "cut_effective_date": run_date,
        "cut_refresh_cadence": "daily",
        "universe_count": len(window.rank_by_ticker),
        "predictor_universe_cut": "attractiveness_top_20",
        "feed_cut": cut,
        "funnel": {
            "population": len(window.rank_by_ticker),
            "advances_to": {
                "predictor_universe": "attractiveness_top_20",
                "rag_corpus_scope": cut,
                "thinktank_coverage_window": cut,
            },
        },
        "cuts": {
            cut: {
                "basis": "attractiveness_rank",
                "size": window.size,
                "tickers": list(window.tickers),
                "source": f"scanner/universe/{run_date}/universe.json::attractiveness_score",
            }
        },
        "ranks": {
            t: {"attractiveness_rank": r, "attractiveness_score": float(1000 - r)}
            for t, r in window.rank_by_ticker.items()
        },
    }
