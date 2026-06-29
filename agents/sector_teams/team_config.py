"""
Sector team configuration — GICS-to-team mapping, slot allocation, and team parameters.

6 permanent sector teams cover the S&P 900 (S&P 500 + S&P 400).
Each team has a Quant Analyst and Qualitative Analyst.
"""

from __future__ import annotations

import math
from typing import Optional

# ── GICS-to-Team Mapping ─────────────────────────────────────────────────────
# Maps each GICS sector name to one of 6 sector teams.
# Keys must match the sector names produced by price_fetcher.fetch_sp500_sp400_with_sectors().

SECTOR_TEAM_MAP: dict[str, str] = {
    # Wikipedia labels (primary — what fetch_sp500_sp400_with_sectors returns)
    "Technology": "technology",
    "Healthcare": "healthcare",
    "Financial": "financials",
    "Industrials": "industrials",
    "Materials": "industrials",
    "Consumer Discretionary": "consumer",
    "Consumer Staples": "consumer",
    "Communication Services": "consumer",
    "Energy": "defensives",
    "Utilities": "defensives",
    "Real Estate": "defensives",
    # Alternate labels (yfinance, GICS official, etc.)
    "Information Technology": "technology",
    "Health Care": "healthcare",
    "Financials": "financials",
    "Financial Services": "financials",
    "Basic Materials": "industrials",
    "Consumer Cyclical": "consumer",
    "Consumer Defensive": "consumer",
}

# Inverse mapping: team_id → list of sector names (include all known variants)
TEAM_SECTORS: dict[str, list[str]] = {
    "technology": ["Technology", "Information Technology"],
    "healthcare": ["Healthcare", "Health Care"],
    "financials": ["Financial", "Financials", "Financial Services"],
    "industrials": ["Industrials", "Materials", "Basic Materials"],
    "consumer": ["Consumer Discretionary", "Consumer Staples", "Communication Services",
                  "Consumer Cyclical", "Consumer Defensive"],
    "defensives": ["Energy", "Utilities", "Real Estate"],
}

ALL_TEAM_IDS = list(TEAM_SECTORS.keys())

# ── Agent Parameters ─────────────────────────────────────────────────────────
QUANT_TOP_N = 10       # number of picks the quant analyst screens for
MAX_TICKERS_IN_PROMPT = 200  # max tickers listed in quant analyst prompt

# ── Slot Allocation ──────────────────────────────────────────────────────────
# Teams always produce 2-3 picks. The slot allocation table determines how many
# open slots each team's picks compete for (used by CIO for context, not by teams).

_SLOT_TIERS = {
    # (min_open_slots, max_open_slots): base_picks
    (0, 0): 0,
    (1, 3): 1,
    (4, 7): 2,
    (8, 10): 3,
}


def _get_base_picks(open_slots: int) -> int:
    """Return base pick count for a given number of open slots."""
    if open_slots <= 0:
        return 0
    for (lo, hi), base in _SLOT_TIERS.items():
        if lo <= open_slots <= hi:
            return base
    # More than 10 open slots (extreme case)
    return 3


# Adaptive slot allocation (config#926): per-team accuracy nudge. A team whose
# historical recommendations have out-performed earns +1 eligible slot; a team
# that has under-performed loses 1. The nudge is intentionally small (±1, same
# scale as the sector-rating adjustment) and gated on a minimum observation
# count so a team can't be penalized/rewarded on noise. With no accuracy data
# the function is byte-identical to the static allocation (graceful degrade).
ADAPTIVE_SLOT_MIN_OBS = 8          # min scored recs before accuracy is trusted
ADAPTIVE_SLOT_TOP_PERCENTILE = 0.60   # accuracy >= this fraction → +1
ADAPTIVE_SLOT_BOTTOM_PERCENTILE = 0.40  # accuracy <= this fraction → -1


def compute_team_slots(
    open_slots: int,
    sector_ratings: dict[str, dict],
    team_sectors: dict[str, list[str]] | None = None,
    team_accuracy: dict[str, dict] | None = None,
) -> dict[str, int]:
    """
    Compute pick allocation per team based on open slots and macro sector ratings.

    Args:
        open_slots: Number of empty population slots after exits.
        sector_ratings: {sector: {rating: "overweight"|"market_weight"|"underweight", ...}}
        team_sectors: Optional override for TEAM_SECTORS.
        team_accuracy: Optional adaptive-allocation input (config#926):
            ``{team_id: {"accuracy": float in [0,1], "n_obs": int}}`` where
            ``accuracy`` is the team's historical hit rate (e.g. fraction of its
            recommendations that beat SPY). When provided, each team's slot count
            is nudged ±1 by ``_accuracy_adjustment``. ``None`` (or a team absent
            from the map, or below ``ADAPTIVE_SLOT_MIN_OBS``) → no nudge, so the
            allocation degrades gracefully to the static behavior.

    Returns:
        {team_id: allocated_slots} — how many picks from this team are eligible
        for CIO selection. Teams always produce 2-3 regardless.
    """
    ts = team_sectors or TEAM_SECTORS
    base = _get_base_picks(open_slots)

    allocation = {}
    for team_id, sectors in ts.items():
        # Team's sector rating = best rating among its sectors
        team_rating = _get_team_rating(sectors, sector_ratings)
        adj = _rating_adjustment(team_rating)
        acc_adj = _accuracy_adjustment(
            (team_accuracy or {}).get(team_id)
        )
        allocation[team_id] = max(0, base + adj + acc_adj)

    return allocation


def _accuracy_adjustment(team_acc: dict | None) -> int:
    """Slot nudge from a team's historical accuracy (config#926).

    Returns +1 for a top-percentile, well-sampled team, -1 for a
    bottom-percentile one, and 0 otherwise (including missing/under-sampled
    data — the graceful-degrade path).
    """
    if not team_acc:
        return 0
    n_obs = team_acc.get("n_obs", 0) or 0
    accuracy = team_acc.get("accuracy")
    if accuracy is None or n_obs < ADAPTIVE_SLOT_MIN_OBS:
        return 0
    try:
        accuracy = float(accuracy)
    except (TypeError, ValueError):
        return 0
    if accuracy >= ADAPTIVE_SLOT_TOP_PERCENTILE:
        return 1
    if accuracy <= ADAPTIVE_SLOT_BOTTOM_PERCENTILE:
        return -1
    return 0


def _get_team_rating(
    sectors: list[str],
    sector_ratings: dict[str, dict],
) -> str:
    """Return the best sector rating across a team's sectors."""
    ratings = []
    for sector in sectors:
        rating_data = sector_ratings.get(sector, {})
        ratings.append(rating_data.get("rating", "market_weight"))

    # Priority: overweight > market_weight > underweight
    if "overweight" in ratings:
        return "overweight"
    if "market_weight" in ratings:
        return "market_weight"
    return "underweight"


def _rating_adjustment(rating: str) -> int:
    """Slot adjustment based on sector rating."""
    return {"overweight": 1, "market_weight": 0, "underweight": -1}.get(rating, 0)


# ── Team Screening Parameters ────────────────────────────────────────────────
# Per-team overrides for quant analyst tools.
# These are hints — the ReAct agent decides its own screening strategy.

TEAM_SCREENING_PARAMS: dict[str, dict] = {
    "technology": {
        "focus_metrics": ["revenue_growth_yoy", "gross_margin", "rd_intensity"],
        "balance_sheet_exempt": False,
        "deep_value_enabled": False,  # tech rarely fits deep value criteria
    },
    "healthcare": {
        "focus_metrics": ["pipeline_stage", "revenue_growth_yoy", "gross_margin"],
        "balance_sheet_exempt": False,
        "deep_value_enabled": True,
    },
    "financials": {
        "focus_metrics": ["roe", "net_interest_margin", "capital_ratio"],
        "balance_sheet_exempt": True,  # D/E not meaningful for financials
        "deep_value_enabled": True,
    },
    "industrials": {
        "focus_metrics": ["capex_growth", "backlog", "operating_margin"],
        "balance_sheet_exempt": False,
        "deep_value_enabled": True,
    },
    "consumer": {
        "focus_metrics": ["same_store_sales", "brand_value", "consumer_sentiment"],
        "balance_sheet_exempt": False,
        "deep_value_enabled": True,
    },
    "defensives": {
        "focus_metrics": ["dividend_yield", "payout_ratio", "debt_to_ebitda"],
        "balance_sheet_exempt": True,  # Real estate has different capital structure
        "deep_value_enabled": True,
    },
}


def get_team_tickers(
    team_id: str,
    scanner_universe: list[str],
    sector_map: dict[str, str],
) -> list[str]:
    """Filter the scanner universe to tickers belonging to a team's sectors."""
    team_sectors_set = set()
    for gics, tid in SECTOR_TEAM_MAP.items():
        if tid == team_id:
            team_sectors_set.add(gics)

    return [t for t in scanner_universe if sector_map.get(t, "") in team_sectors_set]
