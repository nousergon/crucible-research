"""
Composite scoring for sector-team architecture.

Formula: quant_score × w_quant + qual_score × w_qual + macro_shift + boosts.

Weights are loaded from S3 config (auto-tuned by backtester) with YAML defaults.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Default weights — overridden by S3 config/scoring_weights.json
DEFAULT_W_QUANT = 0.50
DEFAULT_W_QUAL = 0.50

# Macro shift parameters — maps sector modifier range [0.70, 1.30] to point shift [-10, +10]
MACRO_MODIFIER_RANGE = 0.30      # distance from 1.0 to min/max (0.70 and 1.30)
MACRO_MAX_SHIFT_POINTS = 10.0    # max pts added/subtracted by macro shift


def compute_composite_score(
    quant_score: float | None,
    qual_score: float | None,
    sector_modifier: float,
    boosts: dict[str, float] | None = None,
    w_quant: float = DEFAULT_W_QUANT,
    w_qual: float = DEFAULT_W_QUAL,
    max_aggregate_boost: float = 10.0,
) -> dict:
    """
    Compute the composite attractiveness score.

    Args:
        quant_score: Quantitative score (0-100) from quant analyst. None if failed.
        qual_score: Qualitative score (0-100) from qual analyst. None if failed.
        sector_modifier: Macro sector modifier (0.70-1.30).
        boosts: {boost_name: points} from signal-boost enrichments.
        w_quant: Weight for quant score.
        w_qual: Weight for qual score.
        max_aggregate_boost: Cap on total boost points.

    Returns:
        {
            "final_score": float (0-100),
            "weighted_base": float,
            "macro_shift": float,
            "total_boost": float,
            "score_failed": bool,
        }
    """
    # Handle missing scores
    if quant_score is None and qual_score is None:
        return {
            "final_score": None,
            "weighted_base": None,
            "macro_shift": 0.0,
            "total_boost": 0.0,
            "score_failed": True,
        }

    # If one score is missing, use the other at full weight
    if quant_score is None:
        weighted_base = qual_score
    elif qual_score is None:
        weighted_base = quant_score
    else:
        weighted_base = quant_score * w_quant + qual_score * w_qual

    # Macro shift: (modifier - 1.0) / range × max_shift → [-10, +10]
    macro_shift = (sector_modifier - 1.0) / MACRO_MODIFIER_RANGE * MACRO_MAX_SHIFT_POINTS

    # Aggregate boosts with cap
    total_boost = 0.0
    if boosts:
        total_boost = sum(boosts.values())
        total_boost = max(-max_aggregate_boost, min(max_aggregate_boost, total_boost))

    final = weighted_base + macro_shift + total_boost
    final = max(0.0, min(100.0, final))

    return {
        "final_score": round(final, 1),
        "weighted_base": round(weighted_base, 1),
        "macro_shift": round(macro_shift, 1),
        "total_boost": round(total_boost, 1),
        "score_failed": False,
    }


_VALID_CONVICTIONS = {"rising", "stable", "declining"}


def normalize_conviction(raw_conviction) -> str:
    """Map agent + storage conviction formats to executor-compatible enum.

    Post-Option-A (2026-04-30) the agent format is uniformly int 0-100. The
    legacy ``"high"/"medium"/"low"`` agent-string branch is retired — every
    agent now emits int (qual_analyst_user.txt v1.1.0,
    sector_team_thesis_update.txt v1.1.0, ic_cio_evaluation.txt). Storage
    format remains the trend label ``rising/stable/declining`` and is kept
    here as passthrough so prior_theses loaded from existing rows in
    ``investment_thesis`` SQLite continue to round-trip cleanly.

    Accepts:
      - Storage format: "rising", "stable", "declining" -> pass through
      - Agent format (int 0-100): >= 70 -> "rising", 40-69 -> "stable",
        < 40 -> "declining"
      - Anything else (None, legacy string variants, sentences) -> "stable"
    """
    if isinstance(raw_conviction, str):
        lower = raw_conviction.strip().lower()
        if lower in _VALID_CONVICTIONS:
            return lower
        return "stable"

    if isinstance(raw_conviction, (int, float)):
        if raw_conviction >= 70:
            return "rising"
        if raw_conviction >= 40:
            return "stable"
        return "declining"

    return "stable"


def score_to_rating(score: float | None, buy_threshold: float = 70.0, sell_threshold: float = 40.0) -> str:
    """Convert a composite score to a rating."""
    if score is None:
        return "HOLD"
    if score >= buy_threshold:
        return "BUY"
    if score <= sell_threshold:
        return "SELL"
    return "HOLD"


def compute_narrative_regime_adjustment(
    thesis_text: str | None,
    market_regime: str | None,
    *,
    bull_defensive_markers: list[str],
    bull_growth_markers: list[str],
    bull_defensive_penalty: float,
    bull_growth_bonus: float,
    bear_defensive_bonus: float,
    bear_growth_penalty: float,
    max_marker_hits: int = 3,
) -> tuple[float, dict]:
    """Compute regime-conditional adjustment based on thesis text markers.

    Scans qual analyst's thesis for defensive vs growth narrative markers.
    In BULL regime, defensive narratives get penalized and growth bonused.
    Inverted in BEAR. NEUTRAL applies no adjustment. Marker hits per
    direction are capped at ``max_marker_hits`` to prevent over-penalizing
    richly worded theses (e.g. one that mentions "oversold" 5 times in a
    long bull_case shouldn't be 5x penalized).

    Returns (adjustment_pts, details) where adjustment is signed (negative
    = penalty, positive = bonus) and details has the marker hit counts
    for auditability.

    Pure text-match logic — no LLM call, deterministic, zero token cost.
    """
    if not thesis_text or not market_regime:
        return 0.0, {"reason": "no_thesis_or_regime"}

    regime = market_regime.strip().lower()
    if regime not in ("bull", "bear"):
        return 0.0, {"reason": f"regime_neutral_or_unknown ({regime})"}

    text_lower = thesis_text.lower()
    defensive_hits = sum(1 for m in bull_defensive_markers if m.lower() in text_lower)
    growth_hits = sum(1 for m in bull_growth_markers if m.lower() in text_lower)
    capped_defensive = min(defensive_hits, max_marker_hits)
    capped_growth = min(growth_hits, max_marker_hits)

    # Per-hit attenuation: full penalty/bonus on first hit; subsequent hits
    # contribute at decreasing weight to avoid double-counting near-synonyms.
    # Using simple linear scaling: hit_n contributes at 1/n of base.
    def _scale(hits_capped: int) -> float:
        return sum(1.0 / (i + 1) for i in range(hits_capped))

    if regime == "bull":
        adjustment = (
            -bull_defensive_penalty * _scale(capped_defensive)
            + bull_growth_bonus * _scale(capped_growth)
        )
    else:  # bear
        adjustment = (
            bear_defensive_bonus * _scale(capped_defensive)
            - bear_growth_penalty * _scale(capped_growth)
        )

    return round(adjustment, 2), {
        "regime": regime,
        "defensive_hits": defensive_hits,
        "growth_hits": growth_hits,
        "defensive_capped": capped_defensive,
        "growth_capped": capped_growth,
        "adjustment_pts": round(adjustment, 2),
    }
