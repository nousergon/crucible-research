"""No-agent (pure-quant) research producer (config#1221 / M3 floor baseline).

Emits a conforming signals.json from the scanner candidate set using ONLY
quantitative scores — no LLM, no sector teams, no CIO, no macro agent. It is the
FLOOR baseline for the research champion/challenger substrate: if the agentic
layer adds ranking alpha, the no-agent producer should underperform it on
realized outcomes; if it doesn't, the floor is the honest answer.

Design (no shortcuts): the producer does NOT reimplement the contract-correct
signals.json assembly. It builds a quant-only ``state`` (theses scored by the
technical composite, a deterministic top-N ENTER gate replacing the CIO, neutral
macro defaults) and reuses the live ``_build_signals_payload`` — so the ENTER /
HOLD / EXIT / buy_candidates logic and every contract field are identical to the
champion's, and only the BELIEF (what gets a high score, what enters) differs.
"""

from __future__ import annotations

import logging

from scoring.composite import compute_composite_score

logger = logging.getLogger(__name__)

# Quant ENTER gate: a candidate must clear this composite score to be BUY-rated,
# and the top-N BUY-rated NEW names become entrants (the no-agent stand-in for
# the CIO entrant gate). Deliberately conservative defaults; tunable later via
# the same backtester config path the scanner params use.
DEFAULT_BUY_SCORE_THRESHOLD = 60.0
DEFAULT_MAX_NEW_ENTRANTS = 15


def _conviction_from_momentum(tech: dict) -> str:
    """Quant stand-in for the agent's conviction read: 20-day momentum sign."""
    m = tech.get("momentum_20d")
    if m is None:
        return "stable"
    if m > 0.02:
        return "rising"
    if m < -0.02:
        return "declining"
    return "stable"


def build_no_agent_signals(
    run_date: str,
    *,
    scanner_tickers: list[str],
    population: list[dict],
    prior_theses: dict[str, dict],
    technical_scores: dict[str, dict],
    sector_map: dict[str, str],
    market_regime: str = "neutral",
    run_time: str = "",
    buy_score_threshold: float = DEFAULT_BUY_SCORE_THRESHOLD,
    max_new_entrants: int = DEFAULT_MAX_NEW_ENTRANTS,
) -> dict:
    """Build a conforming signals.json payload from quant inputs alone.

    Pure function (no I/O) so it is unit-testable without S3/SQLite. Reuses the
    live ``_build_signals_payload`` for the actual assembly.
    """
    # Imported lazily: graph.research_graph pulls the LangGraph stack at import,
    # which the no-agent producer otherwise has no need for.
    from graph.research_graph import _build_signals_payload

    pop_tickers = {p["ticker"] for p in population}

    # 1. Quant-only theses for every scorable scanner candidate.
    theses: dict[str, dict] = {}
    for ticker in scanner_tickers:
        tech = technical_scores.get(ticker)
        if not tech:
            continue
        quant = tech.get("technical_score")
        comp = compute_composite_score(
            quant_score=quant,
            qual_score=None,            # no agent → no qualitative score
            sector_modifier=1.0,        # neutral; no macro agent
            macro_overlay_enabled=False,
        )
        final = comp.get("final_score")
        if final is None:               # unscorable → not a candidate
            continue
        rating = "BUY" if final >= buy_score_threshold else "HOLD"
        theses[ticker] = {
            "ticker": ticker,
            "rating": rating,
            "score": final,
            "final_score": final,
            "quant_score": quant,
            "qual_score": None,
            "conviction": _conviction_from_momentum(tech),
            "sector": sector_map.get(ticker, "Unknown"),
            "bull_case": "",
        }

    # 2. ENTER gate (no-agent stand-in for the CIO): top-N BUY-rated NEW names.
    new_buys = sorted(
        (t for t, th in theses.items() if th["rating"] == "BUY" and t not in pop_tickers),
        key=lambda t: theses[t]["final_score"],
        reverse=True,
    )
    advanced_tickers = new_buys[:max_new_entrants]

    # 3. Population = carryover all held names + the advanced new entrants.
    #    (No churn-out this increment — exits stay empty; the executor's risk
    #    layer governs exits, and a no-agent EXIT policy is a later refinement.)
    new_population = list(population) + [
        {
            "ticker": t,
            "sector": theses[t]["sector"],
            "long_term_rating": "BUY",
            "long_term_score": theses[t]["final_score"],
            "conviction": theses[t]["conviction"],
            "price_target_upside": None,
        }
        for t in advanced_tickers
    ]

    # 4. Neutral macro defaults (no macro agent): _build_signals_payload falls
    #    back to market_weight / neutral modifiers on empty dicts.
    state: dict = {
        "investment_theses": theses,
        "prior_theses": prior_theses,
        "new_population": new_population,
        "sector_map": sector_map,
        "sector_ratings": {},
        "sector_modifiers": {},
        "entry_theses": {},
        "advanced_tickers": advanced_tickers,
        "exits": [],
        "run_date": run_date,
        "run_time": run_time,
        "market_regime": market_regime,
    }

    payload = _build_signals_payload(state)
    logger.info(
        "[no_agent] run_date=%s scored=%d buy_candidates=%d population=%d "
        "new_entrants=%d",
        run_date, len(theses), len(payload.get("buy_candidates", [])),
        len(payload.get("population", [])), len(advanced_tickers),
    )
    return payload


def run_no_agent_producer(
    run_date: str,
    archive_manager,
    *,
    market_regime: str = "neutral",
    run_time: str = "",
    population: list[dict] | None = None,
) -> dict:
    """Integration entry: load the inputs the no-agent producer needs (the SAME
    scanner candidates the champion reads — scanner held constant) and build the
    signals payload. I/O lives here; the scoring/assembly logic is in the pure
    :func:`build_no_agent_signals`.

    ``population`` may OVERRIDE the SQLite read — the SF post-step passes the
    PRIOR population (snapshotted before the champion mutated it) so every
    producer's ENTER selections start from the same held book (a clean
    selection-only comparison)."""
    from data.fetchers.price_fetcher import fetch_sp500_sp400_with_sectors
    from data.scanner_orchestrator import _build_technical_scores_from_feature_store

    cand = archive_manager.load_candidates_json(run_date) or {}
    scanner_tickers = cand.get("scanner_tickers", [])
    if population is None:
        population = archive_manager.load_population()
    pop_tickers = [p["ticker"] for p in population]
    prior_theses = archive_manager.load_latest_theses(
        list(dict.fromkeys(scanner_tickers + pop_tickers))
    )
    constituents, sector_map = fetch_sp500_sp400_with_sectors()
    technical_scores, _ = _build_technical_scores_from_feature_store(constituents, sector_map)
    return build_no_agent_signals(
        run_date,
        scanner_tickers=scanner_tickers,
        population=population,
        prior_theses=prior_theses,
        technical_scores=technical_scores,
        sector_map=sector_map,
        market_regime=market_regime,
        run_time=run_time,
    )
