"""Signals payload builder — the assembly chokepoint every producer routes
its ``signals.json`` through (alpha-engine-config-I7827).

PURE MOTION: this is `_build_signals_payload`, moved verbatim out of the
retired ``graph/research_graph.py`` into a home the live producers
(``producers/no_agent.py``, ``producers/single_agent.py``) own directly. The
graph re-exported it for one PR so its own call site kept working
byte-identically; that graph was deleted 2026-08-20 and this is now the only
definition.

Zero behaviour change from the lift: same body, same config dependencies,
same logger name convention.

``state`` is a plain ``dict`` shaped like what the producers assemble — it was
annotated ``ResearchState`` (a ``TypedDict`` that lived in the deleted graph);
the function only ever uses ``.get()``, so the annotation is now the honest
``dict``.
"""

from __future__ import annotations

import logging

from config import (
    FACTOR_QUALITY_FLOOR_ENABLED,
    FACTOR_QUALITY_FLOOR_EXEMPT_SECTORS,
    FACTOR_QUALITY_FLOOR_MIN_PERCENTILE,
    SECTOR_COHERENCE_GATE_ENABLED,
    SECTOR_COHERENCE_UW_MIN_SCORE,
)
from scoring.composite import normalize_conviction

logger = logging.getLogger(__name__)


def _build_signals_payload(state: dict) -> dict:
    """Build backward-compatible signals.json payload.

    Includes both v2 keys (signals, population) and v1 keys (universe, buy_candidates)
    so the executor and predictor can read actionable signals.
    """
    theses = state.get("investment_theses", {})
    prior_theses = state.get("prior_theses", {})
    pop = state.get("new_population", [])
    pop_tickers = {p["ticker"] for p in pop}
    # Per-ticker quarantine (config#2247): tickers deterministically omitted
    # from this run. They must NOT be carried forward from prior_theses in the
    # population pass below — that would silently reinstate a stale thesis,
    # violating both the explicit-absence contract AND the no-stale-carry-
    # forward invariant. The quarantine set makes the omission LOUD instead.
    quarantined_records = state.get("quarantined", []) or []
    quarantined_tickers = {q.get("ticker") for q in quarantined_records}
    pop_lookup = {p["ticker"]: p for p in pop}
    sector_map = state.get("sector_map", {})
    sector_ratings = state.get("sector_ratings", {})
    entry_theses = state.get("entry_theses", {})

    # v2 signals dict (keyed by ticker)
    # Signal logic:
    #   ENTER  = BUY-rated AND in population (new entry or reaffirmed hold)
    #   HOLD   = held in population, not BUY-rated (maintain position)
    #   EXIT   = dropped from population (sell)
    #   Stocks not in population and not BUY-rated are excluded (irrelevant to executor)
    advanced_tickers = set(state.get("advanced_tickers", []))
    signals = {}

    # First: tickers with fresh theses from this run
    for ticker, thesis in theses.items():
        rating = thesis.get("rating", "HOLD")
        final_score = thesis.get("final_score")
        in_pop = ticker in pop_tickers

        # Safety gate: a BUY rating with no final_score is a broken thesis
        # (e.g., held-stock LLM update that dropped scoring fields). Downgrade
        # to HOLD so the executor does not attempt to ENTER on null score.
        # Root cause mitigation for the 2026-04-04 incident where
        # LNTH/KR/PR/HAL leaked through as ENTER with score=null.
        if rating == "BUY" and final_score is None:
            logger.warning(
                "[signals] %s has rating=BUY but final_score is None — "
                "downgrading to HOLD (broken thesis)",
                ticker,
            )
            rating = "HOLD"

        # Determine signal — CIO is the sole gate for new entrants.
        # Team recs that the CIO did not advance fall through and produce no
        # signal even if rated BUY. Reaffirmations of held BUY-rated names
        # remain unbounded.
        if rating == "BUY" and ticker in advanced_tickers:
            signal = "ENTER"  # CIO approved new entry
        elif rating == "BUY" and in_pop:
            signal = "ENTER"  # Reaffirm existing BUY position
        elif in_pop:
            signal = "HOLD"   # Held, not BUY-rated
        else:
            continue  # Not held, not CIO-advanced — drop

        # sector_map is authoritative (loaded from constituents.json with full
        # universe coverage). Held-stock thesis updates can leak sector="Unknown"
        # via the Pydantic default when the LLM omits the field (see
        # score_aggregator held-stock branch). Prefer sector_map; thesis sector
        # is the fallback. Mirrors the carry-over branch below.
        signals[ticker] = {
            "ticker": ticker,
            "score": thesis.get("final_score"),
            "rating": rating,
            "signal": signal,
            "conviction": normalize_conviction(thesis.get("conviction", "stable")),
            "thesis_summary": thesis.get("bull_case", ""),
            "sector": sector_map.get(ticker) or thesis.get("sector") or "Unknown",
            "team_id": thesis.get("team_id"),
            # Pick provenance (config#859 stance_source_provenance grader):
            # CIO-advanced new entrant vs reaffirmed held name. Derived from
            # already-available locals — cannot raise.
            "stance_source": "cio_entrant" if ticker in advanced_tickers else "reaffirmed_hold",
            "quant_score": thesis.get("quant_score"),
            "qual_score": thesis.get("qual_score"),
            "factor_subscore": thesis.get("factor_subscore"),
            "factor_weight_applied": thesis.get("factor_weight_applied", 0.0),
            "factor_blend_breakdown": thesis.get("factor_blend_breakdown"),
            "factor_quality_score": thesis.get("factor_quality_score"),
            "sub_scores": {
                "quant": thesis.get("quant_score"),
                "qual": thesis.get("qual_score"),
            },
        }

    # Second: population tickers without fresh theses — carry over from prior week
    for p in pop:
        ticker = p["ticker"]
        if ticker in quarantined_tickers:
            # Explicit absence (config#2247) — a quarantined held ticker is
            # deliberately dropped this run, NOT carried forward with a stale
            # prior thesis. The `quarantined` field below records it loudly.
            continue
        if ticker not in signals:
            prior = prior_theses.get(ticker, {})
            sector = sector_map.get(ticker, p.get("sector", "Unknown"))
            prior_rating = prior.get("rating") or p.get("long_term_rating", "HOLD")
            carried_score = prior.get("score") or p.get("long_term_score")
            # Only emit ENTER if we have a score — unscored holdovers stay HOLD
            if prior_rating == "BUY" and carried_score is not None:
                carried_signal = "ENTER"
            else:
                carried_signal = "HOLD"
            signals[ticker] = {
                "ticker": ticker,
                "score": carried_score,
                "rating": prior_rating,
                "signal": carried_signal,
                "conviction": normalize_conviction(prior.get("conviction") or p.get("conviction", "stable")),
                "thesis_summary": prior.get("thesis_summary", ""),
                "sector": sector,
                "team_id": prior.get("team_id"),
                # Pick provenance: population ticker with no fresh thesis this run.
                "stance_source": "carryover",
                "quant_score": prior.get("quant_score"),
                "qual_score": prior.get("qual_score"),
                "sub_scores": {
                    "quant": prior.get("quant_score"),
                    "qual": prior.get("qual_score"),
                },
            }

    # Third: exited stocks — explicit EXIT signal so executor knows to sell
    for e in state.get("exits", []):
        ticker = e.get("ticker_out", "")
        if ticker and ticker not in signals:
            sector = sector_map.get(ticker, "Unknown")
            signals[ticker] = {
                "ticker": ticker,
                "score": e.get("score_out", 0),
                "rating": "SELL",
                "signal": "EXIT",
                "conviction": "declining",
                "stance_source": "exit",
                "thesis_summary": e.get("reason", "Exited from population"),
                "sector": sector,
                "team_id": None,
                "quant_score": None,
                "qual_score": None,
                "sub_scores": {"quant": None, "qual": None},
            }

    # v1-compatible universe list (executor reads this)
    universe = []
    for ticker, sig in signals.items():
        sector = sig["sector"]
        sr = sector_ratings.get(sector, {})
        pop_entry = pop_lookup.get(ticker, {})
        universe.append({
            "ticker": ticker,
            "signal": sig["signal"],
            "score": sig["score"],
            "rating": sig["rating"],
            "conviction": sig["conviction"],
            "price_target_upside": pop_entry.get("price_target_upside"),
            "sector_rating": sr.get("rating", "market_weight"),
            "sector": sector,
            "thesis_summary": sig["thesis_summary"],
            # Propagate pick provenance to the v1 universe entries the
            # evaluator's stance_source_provenance grader reads (config#859).
            "stance_source": sig.get("stance_source"),
            "factor_quality_score": sig.get("factor_quality_score"),
            "sub_scores": sig.get("sub_scores"),
        })

    # v1-compatible buy_candidates list (ENTER signals with enriched theses).
    #
    # Two structural gates run in sequence on every ENTER signal — both gate
    # only NEW buys, never HOLDs / EXITs:
    #   1. Macro-sector coherence gate (260513): block new buys in UW sectors
    #      below SECTOR_COHERENCE_UW_MIN_SCORE — forces structural alignment
    #      between the macro call and per-pick action.
    #   2. Factor quality floor (Phase 4, 260513 plan): block new buys whose
    #      within-sector quality_score percentile is below the floor — drops
    #      bottom-decile-quality names regardless of agent sentiment.
    #      Replaces the dormant Piotroski-lite scanner-side quality_floor.
    buy_candidates = []
    blocked_by_coherence_gate: list[dict] = []
    blocked_by_quality_floor: list[dict] = []
    for entry in universe:
        if entry["signal"] != "ENTER":
            continue
        if (
            SECTOR_COHERENCE_GATE_ENABLED
            and entry.get("sector_rating") == "underweight"
            and (entry.get("score") or 0) < SECTOR_COHERENCE_UW_MIN_SCORE
        ):
            blocked_by_coherence_gate.append({
                "ticker": entry["ticker"],
                "sector": entry["sector"],
                "score": entry["score"],
                "uw_min_score": SECTOR_COHERENCE_UW_MIN_SCORE,
            })
            continue
        # Factor quality floor: skip when the ticker has no factor profile
        # (graceful degrade — same pattern as the rest of the factor blend)
        # or its sector is in the exempt list (Financial / Real Estate /
        # Utilities by default, where the quality factor metrics do not apply).
        if (
            FACTOR_QUALITY_FLOOR_ENABLED
            and entry.get("factor_quality_score") is not None
            and entry["sector"] not in FACTOR_QUALITY_FLOOR_EXEMPT_SECTORS
            and entry["factor_quality_score"] < FACTOR_QUALITY_FLOOR_MIN_PERCENTILE
        ):
            blocked_by_quality_floor.append({
                "ticker": entry["ticker"],
                "sector": entry["sector"],
                "quality_pct": entry["factor_quality_score"],
                "floor": FACTOR_QUALITY_FLOOR_MIN_PERCENTILE,
            })
            continue
        candidate = dict(entry)
        et = entry_theses.get(entry["ticker"], {})
        if et:
            candidate["thesis_summary"] = et.get("bull_case", candidate["thesis_summary"])
            candidate["catalysts"] = et.get("catalysts", [])
        buy_candidates.append(candidate)
    if blocked_by_coherence_gate:
        logger.info(
            "macro_sector_coherence_gate blocked %d ENTER signal(s) "
            "from UNDERWEIGHT sectors (uw_min_score=%.1f): %s",
            len(blocked_by_coherence_gate),
            SECTOR_COHERENCE_UW_MIN_SCORE,
            [f"{b['ticker']}({b['sector']},{b['score']:.1f})" for b in blocked_by_coherence_gate],
        )
    if blocked_by_quality_floor:
        logger.info(
            "factor_quality_floor blocked %d ENTER signal(s) "
            "with quality_score below %.1f-percentile: %s",
            len(blocked_by_quality_floor),
            FACTOR_QUALITY_FLOOR_MIN_PERCENTILE,
            [f"{b['ticker']}({b['sector']},quality={b['quality_pct']:.1f})" for b in blocked_by_quality_floor],
        )

    return {
        "date": state.get("run_date", ""),
        "time": state.get("run_time", ""),
        "market_regime": state.get("market_regime", "neutral"),
        "sector_modifiers": state.get("sector_modifiers", {}),
        "sector_ratings": sector_ratings,
        "signals": signals,
        "population": [p["ticker"] for p in pop],
        "universe": universe,
        "buy_candidates": buy_candidates,
        # Per-ticker quarantine (config#2247) — the explicit-absence contract.
        # Additive, backward-compatible: consumers that don't read it are
        # unaffected (they already must tolerate a ticker being absent per the
        # S3 signals contract). Consumers that DO read it get a loud, machine-
        # readable record of which tickers were deterministically omitted and
        # why, instead of a silent gap.
        "quarantined": [
            {
                "ticker": q.get("ticker"),
                "reason": q.get("reason"),
                "team_id": q.get("team_id"),
                "stage": q.get("stage"),
            }
            for q in quarantined_records
        ],
        "architecture_version": "sector_teams",
    }
