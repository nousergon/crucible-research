"""
data/population_selector.py — sector-balanced investment population management.

Replaces the old dual-branch architecture (20 static universe + 3 rotating
candidates) with a scanner-driven population of 20-25 stocks drawn entirely
from S&P 900.

Sector allocation is driven by the macro agent's sector_modifiers:
  - Overweight sectors (modifier >= 1.05): 3+ stocks each
  - Market-weight sectors (0.95 < modifier < 1.05): ~2 stocks each
  - Underweight sectors (modifier <= 0.95): at least 1 stock each

Rotation rules:
  - Stocks stay in population unless thesis degrades (long_term_score drops)
    or a same-sector challenger scores meaningfully higher.
  - Minimum tenure protection (2 weeks) prevents churn, with override for
    thesis collapse (score < 40 → immediate removal).
  - Min 10% rotation per run (keeps ideas fresh).
  - Max ~40% rotation per run (maintains continuity).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from nousergon_lib.universe import filter_to_universe
from graph.state_schemas import ADVANCE_DECISIONS
from scoring.composite import normalize_conviction

logger = logging.getLogger(__name__)

# 11 GICS sectors (display names matching our sector labels)
GICS_SECTORS = [
    "Technology",
    "Healthcare",
    "Financial",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
    "Consumer Staples",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]


def classify_sectors(
    sector_ratings: dict[str, dict],
    config: dict,
) -> dict[str, str]:
    """
    Classify each GICS sector as overweight / market_weight / underweight
    based on the macro agent's sector_modifier values.

    Args:
        sector_ratings: {sector_name: {modifier: float, rating: str, ...}}
        config: population config from universe.yaml

    Returns: {sector_name: "overweight" | "market_weight" | "underweight"}
    """
    sa_config = config.get("sector_allocation", {})
    ow_thresh = sa_config.get("overweight_modifier_threshold", 1.05)
    uw_thresh = sa_config.get("underweight_modifier_threshold", 0.95)

    result: dict[str, str] = {}
    for sector in GICS_SECTORS:
        data = sector_ratings.get(sector, {})
        modifier = data.get("modifier", 1.0)
        if modifier >= ow_thresh:
            result[sector] = "overweight"
        elif modifier <= uw_thresh:
            result[sector] = "underweight"
        else:
            result[sector] = "market_weight"

    return result


def compute_sector_slots(
    sector_classes: dict[str, str],
    config: dict,
) -> dict[str, int]:
    """
    Compute target slot count per sector based on classification.

    Starting allocation:
      overweight → overweight_min (default 3)
      market_weight → market_weight_target (default 2)
      underweight → underweight_min (default 1)

    Adjusts to hit target_size:
      If total < target: add extra slots to overweight sectors first
      If total > target: reduce market_weight sectors first

    Returns: {sector_name: slot_count}
    """
    sa_config = config.get("sector_allocation", {})
    ow_min = sa_config.get("overweight_min", 3)
    mw_target = sa_config.get("market_weight_target", 2)
    uw_min = sa_config.get("underweight_min", 1)
    target_size = config.get("target_size", 25)

    slots: dict[str, int] = {}
    for sector, classification in sector_classes.items():
        if classification == "overweight":
            slots[sector] = ow_min
        elif classification == "market_weight":
            slots[sector] = mw_target
        else:
            slots[sector] = uw_min

    total = sum(slots.values())

    # Adjust to hit target_size
    if total < target_size:
        # Add extra slots to overweight sectors (round-robin)
        ow_sectors = [s for s, c in sector_classes.items() if c == "overweight"]
        if not ow_sectors:
            ow_sectors = [s for s, c in sector_classes.items() if c == "market_weight"]
        idx = 0
        while total < target_size and ow_sectors:
            slots[ow_sectors[idx % len(ow_sectors)]] += 1
            total += 1
            idx += 1
    elif total > target_size:
        # Reduce market_weight sectors first (don't go below 1)
        mw_sectors = sorted(
            [s for s, c in sector_classes.items() if c == "market_weight"],
            key=lambda s: slots[s],
            reverse=True,
        )
        idx = 0
        while total > target_size and mw_sectors:
            s = mw_sectors[idx % len(mw_sectors)]
            if slots[s] > 1:
                slots[s] -= 1
                total -= 1
            idx += 1
            if idx >= len(mw_sectors) * 5:  # safety limit
                logger.warning("[population_selector] safety limit reached: total=%d target=%d — population may exceed target", total, target_size)
                break

    return slots


def select_population(
    scored_candidates: list[dict],
    current_population: list[dict],
    sector_ratings: dict[str, dict],
    config: dict,
    run_date: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Build the investment population from S&P 900 scanner results.

    Args:
        scored_candidates: list of scored stock dicts from scanner pipeline.
            Each must have: ticker, sector, long_term_score, long_term_rating,
            conviction, price_target_upside, thesis_summary, sub_scores, etc.
        current_population: existing population from prior run (may be empty on first run).
            Each must have: ticker, sector, long_term_score, entry_date, tenure_weeks.
        sector_ratings: from macro agent {sector: {modifier, rating, ...}}
        config: population config section from universe.yaml
        run_date: current run date (YYYY-MM-DD). Defaults to today.

    Returns:
        (new_population, rotation_events)
        - new_population: list of population dicts with all fields
        - rotation_events: list of {type, ticker, sector, reason, ...} dicts
    """
    if run_date is None:
        run_date = str(date.today())

    pop_config = config.get("population", config)
    rotation_config = pop_config.get("rotation", {})

    min_lt_score = rotation_config.get("min_long_term_score", 45)
    challenger_delta = rotation_config.get("challenger_score_delta", 5)
    max_rotations = rotation_config.get("max_rotations_per_run", 10)
    min_rotation_pct = rotation_config.get("min_rotation_pct", 0.10)
    min_tenure = rotation_config.get("min_tenure_weeks", 2)
    collapse_thresh = rotation_config.get("thesis_collapse_threshold", 40)

    # Classify sectors and compute slots
    sector_classes = classify_sectors(sector_ratings, pop_config)
    sector_slots = compute_sector_slots(sector_classes, pop_config)

    # Index current population by ticker
    current_by_ticker = {p["ticker"]: p for p in current_population}

    # Index scored candidates by ticker (includes re-scored incumbents)
    candidates_by_ticker = {c["ticker"]: c for c in scored_candidates}

    # Group all candidates by sector
    candidates_by_sector: dict[str, list[dict]] = {}
    for c in scored_candidates:
        sector = c.get("sector", "Unknown")
        candidates_by_sector.setdefault(sector, []).append(c)

    # Sort each sector's candidates by long_term_score descending
    for sector in candidates_by_sector:
        candidates_by_sector[sector].sort(
            key=lambda c: c.get("long_term_score", 0), reverse=True,
        )

    new_population: list[dict] = []
    rotation_events: list[dict] = []
    rotations_used = 0

    # ── Phase 1: Evaluate incumbents ──
    # Check each current population member — keep, remove, or replace
    incumbents_by_sector: dict[str, list[dict]] = {}
    removed_tickers: set[str] = set()

    for incumbent in current_population:
        ticker = incumbent["ticker"]
        sector = incumbent.get("sector", "Unknown")

        # Get refreshed score from scanner results (if available)
        refreshed = candidates_by_ticker.get(ticker)
        lt_score = refreshed.get("long_term_score", incumbent.get("long_term_score", 50.0)) if refreshed else incumbent.get("long_term_score", 50.0)

        # Compute tenure
        entry_date = incumbent.get("entry_date", run_date)
        try:
            tenure_weeks = (datetime.fromisoformat(run_date) - datetime.fromisoformat(entry_date)).days // 7
        except (ValueError, TypeError):
            tenure_weeks = incumbent.get("tenure_weeks", 0)

        # Check for thesis collapse (immediate removal regardless of tenure)
        if lt_score < collapse_thresh:
            rotation_events.append({
                "type": "REMOVE",
                "ticker": ticker,
                "sector": sector,
                "reason": f"thesis_collapse (lt_score={lt_score:.1f} < {collapse_thresh})",
                "long_term_score": lt_score,
            })
            removed_tickers.add(ticker)
            rotations_used += 1
            continue

        # Check for score below minimum (with tenure protection)
        if lt_score < min_lt_score and tenure_weeks >= min_tenure:
            if rotations_used < max_rotations:
                rotation_events.append({
                    "type": "REMOVE",
                    "ticker": ticker,
                    "sector": sector,
                    "reason": f"score_degraded (lt_score={lt_score:.1f} < {min_lt_score}, tenure={tenure_weeks}w)",
                    "long_term_score": lt_score,
                })
                removed_tickers.add(ticker)
                rotations_used += 1
                continue

        # Incumbent survives — track by sector
        kept = refreshed.copy() if refreshed else incumbent.copy()
        kept["entry_date"] = entry_date
        kept["tenure_weeks"] = tenure_weeks
        incumbents_by_sector.setdefault(sector, []).append(kept)

    # ── Phase 2: Fill sector slots ──
    for sector in GICS_SECTORS:
        target_slots = sector_slots.get(sector, 1)
        sector_incumbents = incumbents_by_sector.get(sector, [])
        sector_candidates = candidates_by_sector.get(sector, [])

        # Sort incumbents by score (keep best)
        sector_incumbents.sort(
            key=lambda x: x.get("long_term_score", 0), reverse=True,
        )

        # Trim incumbents if we have too many for this sector
        while len(sector_incumbents) > target_slots and rotations_used < max_rotations:
            weakest = sector_incumbents.pop()
            ticker = weakest["ticker"]
            tenure = weakest.get("tenure_weeks", 0)
            if tenure >= min_tenure:
                rotation_events.append({
                    "type": "REMOVE",
                    "ticker": ticker,
                    "sector": sector,
                    "reason": f"sector_rebalance (over_allocated, lt_score={weakest.get('long_term_score', 0):.1f})",
                    "long_term_score": weakest.get("long_term_score", 0),
                })
                removed_tickers.add(ticker)
                rotations_used += 1
            else:
                # Keep despite over-allocation (tenure too short)
                sector_incumbents.append(weakest)
                break

        # Check for superior challengers (replace weakest incumbent)
        if sector_incumbents and sector_candidates and rotations_used < max_rotations:
            weakest_incumbent = min(sector_incumbents, key=lambda x: x.get("long_term_score", 0))
            weakest_score = weakest_incumbent.get("long_term_score", 0)
            weakest_tenure = weakest_incumbent.get("tenure_weeks", 0)

            for challenger in sector_candidates:
                c_ticker = challenger["ticker"]
                if c_ticker in removed_tickers:
                    continue
                if any(inc["ticker"] == c_ticker for inc in sector_incumbents):
                    continue  # already an incumbent
                c_score = challenger.get("long_term_score", 0)

                if (
                    c_score > weakest_score + challenger_delta
                    and weakest_tenure >= min_tenure
                    and len(sector_incumbents) >= target_slots
                ):
                    # Replace weakest with challenger
                    rotation_events.append({
                        "type": "REPLACE",
                        "ticker_out": weakest_incumbent["ticker"],
                        "ticker_in": c_ticker,
                        "sector": sector,
                        "reason": f"superior_challenger ({c_score:.1f} > {weakest_score:.1f} + {challenger_delta})",
                        "score_out": weakest_score,
                        "score_in": c_score,
                    })
                    removed_tickers.add(weakest_incumbent["ticker"])
                    sector_incumbents.remove(weakest_incumbent)
                    challenger_entry = challenger.copy()
                    challenger_entry["entry_date"] = run_date
                    challenger_entry["tenure_weeks"] = 0
                    sector_incumbents.append(challenger_entry)
                    rotations_used += 1
                    break  # max 1 replacement per sector per run

        # Fill remaining slots with top candidates
        current_tickers = {inc["ticker"] for inc in sector_incumbents}
        slots_to_fill = target_slots - len(sector_incumbents)

        for challenger in sector_candidates:
            if slots_to_fill <= 0:
                break
            c_ticker = challenger["ticker"]
            if c_ticker in current_tickers or c_ticker in removed_tickers:
                continue
            # New addition — no rotation count needed for fills
            entry = challenger.copy()
            entry["entry_date"] = run_date
            entry["tenure_weeks"] = 0
            sector_incumbents.append(entry)
            current_tickers.add(c_ticker)
            slots_to_fill -= 1

            rotation_events.append({
                "type": "ADD",
                "ticker": c_ticker,
                "sector": sector,
                "reason": f"slot_fill (lt_score={entry.get('long_term_score', 0):.1f})",
                "long_term_score": entry.get("long_term_score", 0),
            })

        new_population.extend(sector_incumbents)

    # ── Phase 3: Enforce minimum rotation floor ──
    # If natural rotation fell below the min %, force-rotate lowest-scoring
    # tenure-eligible incumbents to keep the portfolio fresh.
    min_rotations = max(1, int(len(new_population) * min_rotation_pct))
    if rotations_used < min_rotations and current_population:
        # Build a set of all tickers that were newly added (not in prior population)
        prior_tickers = {p["ticker"] for p in current_population}

        # Identify tenure-eligible incumbents still in population, sorted by score asc
        eligible_for_forced_rotation = sorted(
            [p for p in new_population
             if p["ticker"] in prior_tickers
             and p["ticker"] not in removed_tickers
             and p.get("tenure_weeks", 0) >= min_tenure],
            key=lambda p: p.get("long_term_score", 0),
        )

        # All scored candidates not already in population, sorted by score desc
        population_tickers = {p["ticker"] for p in new_population}
        replacement_pool = sorted(
            [c for c in scored_candidates
             if c["ticker"] not in population_tickers
             and c["ticker"] not in removed_tickers
             and c.get("long_term_score", 0) >= min_lt_score],
            key=lambda c: c.get("long_term_score", 0),
            reverse=True,
        )

        replacement_idx = 0
        for incumbent in eligible_for_forced_rotation:
            if rotations_used >= min_rotations:
                break
            if rotations_used >= max_rotations:
                break
            if replacement_idx >= len(replacement_pool):
                break

            replacement = replacement_pool[replacement_idx]
            replacement_idx += 1

            # Remove incumbent, add replacement
            new_population = [p for p in new_population if p["ticker"] != incumbent["ticker"]]
            removed_tickers.add(incumbent["ticker"])

            entry = replacement.copy()
            entry["entry_date"] = run_date
            entry["tenure_weeks"] = 0
            new_population.append(entry)

            rotation_events.append({
                "type": "REPLACE",
                "ticker_out": incumbent["ticker"],
                "ticker_in": replacement["ticker"],
                "sector": incumbent.get("sector", "Unknown"),
                "reason": f"min_rotation_floor (forced: {incumbent.get('long_term_score', 0):.1f} → {replacement.get('long_term_score', 0):.1f})",
                "score_out": incumbent.get("long_term_score", 0),
                "score_in": replacement.get("long_term_score", 0),
            })
            rotations_used += 1

    # Sort final population by long_term_score descending
    new_population.sort(key=lambda p: p.get("long_term_score", 0), reverse=True)

    # Log summary
    sector_counts = {}
    for p in new_population:
        s = p.get("sector", "Unknown")
        sector_counts[s] = sector_counts.get(s, 0) + 1

    logger.info(
        "Population selection complete: %d stocks across %d sectors | "
        "%d rotations | sector breakdown: %s",
        len(new_population),
        len(sector_counts),
        rotations_used,
        ", ".join(f"{s}={n}" for s, n in sorted(sector_counts.items())),
    )

    return new_population, rotation_events


# ── Sector-Team Architecture Functions ───────────────────────────────────────

def compute_exits_and_open_slots(
    current_population: list[dict],
    investment_theses: dict[str, dict],
    config: dict,
    run_date: str | None = None,
    constituents: set[str] | frozenset[str] | None = None,
) -> tuple[list[dict], list[dict], int]:
    """
    Determine which stocks exit the population. Runs in parallel with sector teams
    (only needs prior week's theses, not this week's analysis).

    Args:
        current_population: Current held stocks.
        investment_theses: {ticker: thesis_dict} from prior week.
        config: Population config section.
        run_date: YYYY-MM-DD.
        constituents: current scanner universe (S&P 500 + 400) as a set. When
            provided, any incumbent whose ticker is NOT in the set is removed
            via a ``UNIVERSE_DROP`` exit event before score/tenure logic runs.
            Catches grandfathered tickers (ADRs, de-listed names, manual seeds)
            that predate the current constituent filter. Without this,
            ``compute_exits_and_open_slots`` had no way to rotate out tickers
            that weren't in the weekly S&P 900 scan — they kept their prior
            score and persisted indefinitely. Confirmed 2026-04-20: TSM + ASML
            persisted as incumbents with full thesis archives while absent
            from both constituents.json and the ArcticDB universe library,
            causing NoSuchVersionException downstream in the executor-sim
            replay. None = skip the check (backwards compatible for callers
            that don't have constituents at hand).

    Returns:
        (remaining_population, exit_events, open_slots)
    """
    if run_date is None:
        run_date = str(date.today())

    pop_config = config.get("population", config)
    rotation_config = pop_config.get("rotation", {})
    target_size = pop_config.get("target_size", 25)

    min_lt_score = rotation_config.get("min_long_term_score", 45)
    min_tenure = rotation_config.get("min_tenure_weeks", 2)
    collapse_thresh = rotation_config.get("thesis_collapse_threshold", 40)
    min_rotation_pct = rotation_config.get("min_rotation_pct", 0.10)
    max_rotations = rotation_config.get("max_rotations_per_run", 10)

    remaining: list[dict] = []
    exits: list[dict] = []
    rotations_used = 0

    # ── Universe guardrail ──
    # Drop incumbents that have fallen out of (or were never in) the current
    # constituent universe BEFORE score/tenure logic. A ticker not in the
    # scanner's S&P 900 has no refreshed score and no ArcticDB coverage —
    # both downstream prereqs. Drops here are separate from score-based
    # rotations and don't count toward ``max_rotations_per_run`` (they
    # aren't volitional trades, they're reconciliation).
    # Membership predicate is delegated to ``nousergon_lib.universe`` so
    # this Layer 1 filter and the executor's Layer 2 ``signal_reader`` filter
    # share one canonical code path (no silent divergence on universe drift).
    if constituents is not None:
        constituents_set = (
            constituents
            if isinstance(constituents, frozenset)
            else frozenset(constituents)
        )
        current_population, dropped_incumbents = filter_to_universe(
            current_population, constituents_set
        )
        for incumbent in dropped_incumbents:
            ticker = incumbent["ticker"]
            logger.warning(
                "[population_selector] dropping incumbent %s — not in current "
                "S&P 500+400 constituents. Sector=%s. Grandfathered outlier; "
                "executor will not be able to read ArcticDB universe for it.",
                ticker,
                incumbent.get("sector", "Unknown"),
            )
            exits.append({
                "type": "UNIVERSE_DROP",
                "ticker_out": ticker,
                "sector": incumbent.get("sector", "Unknown"),
                "reason": "not in current S&P 500+400 constituents",
                "score_out": incumbent.get("long_term_score", 0),
            })

    for incumbent in current_population:
        ticker = incumbent["ticker"]
        thesis = investment_theses.get(ticker, {})
        lt_score = thesis.get("long_term_score", incumbent.get("long_term_score", 50))

        # Compute tenure
        try:
            entry_date = incumbent.get("entry_date", run_date)
            tenure_weeks = (datetime.fromisoformat(run_date) - datetime.fromisoformat(entry_date)).days // 7
        except Exception:
            tenure_weeks = incumbent.get("tenure_weeks", 0)

        # Thesis collapse — immediate removal
        if lt_score < collapse_thresh:
            exits.append({
                "type": "REMOVE",
                "ticker_out": ticker,
                "sector": incumbent.get("sector", "Unknown"),
                "reason": f"thesis_collapse (score={lt_score:.1f} < {collapse_thresh})",
                "score_out": lt_score,
            })
            rotations_used += 1
            continue

        # Score degradation with tenure protection
        if lt_score < min_lt_score and tenure_weeks >= min_tenure and rotations_used < max_rotations:
            exits.append({
                "type": "REMOVE",
                "ticker_out": ticker,
                "sector": incumbent.get("sector", "Unknown"),
                "reason": f"score_degraded (score={lt_score:.1f} < {min_lt_score}, tenure={tenure_weeks}w)",
                "score_out": lt_score,
            })
            rotations_used += 1
            continue

        incumbent_copy = dict(incumbent)
        incumbent_copy["tenure_weeks"] = tenure_weeks
        incumbent_copy["long_term_score"] = lt_score
        remaining.append(incumbent_copy)

    # NOTE (L4534): the unconditional "minimum rotation floor" that used to
    # force-rotate the lowest-scoring ~10% out EVERY week was REMOVED here.
    # It was an asymmetric ratchet that destroyed alpha in a mature book:
    # it ejected held names scoring 52-61 ("to keep ideas fresh") while the
    # quality-gated entrant bar (~60) admitted nothing in saturated weeks, so
    # the population eroded (30 -> 27) and good names (e.g. KLAC@61.5) were
    # churned out for nothing. "Keep ideas fresh" is now REPLACEMENT-AWARE:
    # turnover happens only as a quality SWAP in ``apply_ic_entries`` — an
    # incumbent is rotated out only when a net-new entrant actually upgrades
    # its slot. Decay-based exits (thesis collapse / score degradation /
    # universe drop) above are unchanged. ``min_rotation_pct`` / ``max_rotations``
    # are retained in config but no longer drive unconditional ejection.
    open_slots = max(0, target_size - len(remaining))

    logger.info(
        "Exit evaluator: %d remain, %d exits, %d open slots",
        len(remaining), len(exits), open_slots,
    )

    return remaining, exits, open_slots


def apply_ic_entries(
    remaining_population: list[dict],
    ic_decisions: list[dict],
    entry_theses: dict[str, dict],
    sector_map: dict[str, str],
    run_date: str,
    max_size: int = 30,
    target_size: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Place IC ADVANCE decisions into the population, then apply replacement-aware
    rotation (L4534).

    Args:
        remaining_population: Population after exits.
        ic_decisions: CIO decisions, each with decision, ticker, rank.
        entry_theses: {ticker: thesis_dict} from CIO.
        sector_map: {ticker: sector} mapping.
        run_date: YYYY-MM-DD.
        max_size: Hard cap on population size (default 30).
        target_size: Target population size. When the book is over target after
            entries, the weakest incumbents are rotated out ONLY to the extent a
            net-new entrant upgrades the slot (incumbent score < the weakest
            admitted entrant). None disables the swap (size-neutral). This is the
            replacement-aware successor to the removed unconditional
            min_rotation_floor — in a saturated week (no entrants) nothing
            rotates, so the book no longer erodes by ejecting good names.

    Returns:
        (final_population, events) where ``events`` mixes IC_ADVANCE entry
        events (``ticker_in``) and FORCED_ROTATION swap exits (``ticker_out``);
        the caller routes the FORCED_ROTATION events into the exits channel so
        the executor gets EXIT signals for swapped-out names.
    """
    population = list(remaining_population)
    entry_events = []
    added_entries: list[dict] = []

    # Both ADVANCE and ADVANCE_FORCED admit a ticker — matching only "ADVANCE"
    # here silently dropped floor-forced entrants (the bug that hid 0-new-entrant
    # weeks). See ADVANCE_DECISIONS in graph/state_schemas.py.
    advanced = [d for d in ic_decisions if d.get("decision") in ADVANCE_DECISIONS]
    advanced.sort(key=lambda d: d.get("rank") if d.get("rank") is not None else 999)

    existing_tickers = {p["ticker"] for p in population}

    for decision in advanced:
        ticker = decision.get("ticker", "")
        if ticker in existing_tickers:
            continue  # already in population

        if len(population) >= max_size:
            logger.warning(
                "Population at max_size (%d) — skipping remaining %d IC advances",
                max_size,
                len([d for d in advanced if d.get("ticker") not in existing_tickers]) - len(entry_events),
            )
            break

        thesis = entry_theses.get(ticker, {})
        entry = {
            "ticker": ticker,
            "sector": sector_map.get(ticker, "Unknown"),
            "long_term_score": decision.get("conviction", thesis.get("score", 50)),
            "long_term_rating": "BUY",
            "conviction": normalize_conviction(decision.get("conviction", 50)),
            "entry_date": run_date,
            "tenure_weeks": 0,
            "thesis_summary": thesis.get("bull_case", decision.get("rationale", "")),
            "ic_conviction": decision.get("conviction"),
            "ic_rationale": decision.get("rationale"),
        }
        population.append(entry)
        added_entries.append(entry)
        existing_tickers.add(ticker)

        entry_events.append({
            "type": "IC_ADVANCE",
            "ticker_in": ticker,
            "sector": entry["sector"],
            "reason": decision.get("rationale", "CIO advanced"),
            "score_in": entry["long_term_score"],
            "ic_rank": decision.get("rank"),
        })

    # ── Replacement-aware rotation (L4534) ────────────────────────────────
    # When over target after entries, rotate out the weakest incumbents — but
    # ONLY those a net-new entrant actually upgrades (incumbent score < the
    # weakest admitted entrant). Never eject a name >= what came in; stop as
    # soon as the weakest remaining incumbent is no longer beaten. No entrants
    # (saturated week) → no rotation → book holds (no erosion).
    rotation_exits: list[dict] = []
    if target_size is not None and added_entries and len(population) > target_size:
        weakest_entrant_score = min(
            (e.get("long_term_score") or 0) for e in added_entries
        )
        added_tickers = {e["ticker"] for e in added_entries}
        incumbents = sorted(
            (p for p in population if p["ticker"] not in added_tickers),
            key=lambda p: p.get("long_term_score", 0),
        )
        for inc in incumbents:
            if len(population) <= target_size:
                break
            inc_score = inc.get("long_term_score", 0) or 0
            if inc_score >= weakest_entrant_score:
                break  # weakest remaining incumbent isn't beaten — no downgrade
            population = [p for p in population if p["ticker"] != inc["ticker"]]
            rotation_exits.append({
                "type": "FORCED_ROTATION",
                "ticker_out": inc["ticker"],
                "sector": inc.get("sector", "Unknown"),
                "reason": (
                    f"quality_swap: replaced by stronger net-new entrant "
                    f"(incumbent {inc_score:.1f} < entrant {weakest_entrant_score:.1f})"
                ),
                "score_out": inc_score,
            })

    logger.info(
        "IC entries: %d advanced into population, %d quality-swap rotations "
        "(total: %d, target: %s)",
        len(entry_events), len(rotation_exits), len(population), target_size,
    )

    return population, entry_events + rotation_exits
