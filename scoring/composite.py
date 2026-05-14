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
    factor_subscore: float | None = None,
    factor_weight: float = 0.0,
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
        factor_subscore: optional regime-conditional factor blend (0-100) from
            ``compute_factor_subscore``. Blended into ``weighted_base`` when
            ``factor_weight > 0``.
        factor_weight: weight applied to ``factor_subscore`` when blending into
            the quant+qual base (e.g. 0.30 → 70% quant_qual + 30% factor).
            Defaults to 0.0 so callers that do not pass a factor profile see
            backward-compatible behavior.

    Returns:
        {
            "final_score": float (0-100),
            "weighted_base": float,
            "macro_shift": float,
            "total_boost": float,
            "factor_subscore": float | None,
            "factor_weight_applied": float,
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
            "factor_subscore": factor_subscore,
            "factor_weight_applied": 0.0,
            "score_failed": True,
        }

    # If one score is missing, use the other at full weight
    if quant_score is None:
        quant_qual_base = qual_score
    elif qual_score is None:
        quant_qual_base = quant_score
    else:
        quant_qual_base = quant_score * w_quant + qual_score * w_qual

    # Factor blend: convex combination with quant_qual_base. Skipped when
    # subscore is unavailable for this ticker (e.g. no fundamentals → None)
    # or factor_weight is 0 — graceful degrade to pure quant_qual behavior.
    if factor_subscore is not None and factor_weight > 0.0:
        weighted_base = (1.0 - factor_weight) * quant_qual_base + factor_weight * factor_subscore
        factor_weight_applied = factor_weight
    else:
        weighted_base = quant_qual_base
        factor_weight_applied = 0.0

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
        "factor_subscore": round(factor_subscore, 1) if factor_subscore is not None else None,
        "factor_weight_applied": factor_weight_applied,
        "score_failed": False,
    }


# ── Factor subscore (Phase 3 of factor substrate, 260513 plan) ──────────────


_FACTOR_SCORE_KEYS = ("quality_score", "momentum_score", "value_score", "low_vol_score")


def compute_factor_subscore(
    factor_profile: dict | None,
    market_regime: str | None,
    regime_weights: dict[str, dict[str, float]] | None,
) -> tuple[float | None, dict]:
    """Compute a regime-conditional factor subscore from a ticker's factor profile.

    Linear combination of the 4 factor composites (quality / momentum / value /
    low_vol) with signed regime-conditional weights, clamped to ``[0, 100]``.
    Returns ``(None, details)`` whenever the blend cannot be computed —
    callers MUST handle ``None`` by falling back to the quant+qual-only path
    (``compute_composite_score`` does this when ``factor_subscore is None``).

    Args:
        factor_profile: per-ticker dict produced by
            ``scoring.factor_scoring.compute_factor_composites`` and persisted
            to ``factors/profiles/latest.json``. Expected to contain any subset
            of ``_FACTOR_SCORE_KEYS``; missing keys reallocate weight pro-rata
            across the keys that ARE present.
        market_regime: ``"bull" | "bear" | "neutral"`` (case-insensitive).
        regime_weights: ``{regime: {factor_score_key: weight}}`` config block
            (see ``alpha-engine-config/research/scoring.yaml``
            ``aggregator.factor_blend``). Weights are signed (negatives flip the
            sign of a factor's contribution — e.g. low_vol penalized in BULL).

    Returns:
        ``(subscore, details)`` where ``subscore`` is a clamped ``[0, 100]``
        float (or ``None`` when no components contributed) and ``details``
        carries the per-factor breakdown for observability and the rendered
        ``regime``.
    """
    if not factor_profile or not market_regime or not regime_weights:
        return None, {"reason": "no_profile_or_regime_or_config"}

    regime = market_regime.strip().lower()
    weights_for_regime = regime_weights.get(regime)
    if not weights_for_regime:
        return None, {"reason": f"no_blend_for_regime ({regime})"}

    # Normalize weights to absolute sum 1.0 across contributing factors. This
    # keeps the linear-combination output on the same ~0-100 scale as the raw
    # factor scores even when partial coverage drops a component. Without
    # renormalization a ticker missing one factor scores systematically lower
    # than its peers.
    breakdown: dict[str, float] = {}
    raw_sum = 0.0
    abs_weight_sum = 0.0
    n_contributing = 0
    for key in _FACTOR_SCORE_KEYS:
        weight = float(weights_for_regime.get(key, 0.0))
        if weight == 0.0:
            continue
        score = factor_profile.get(key)
        if score is None:
            continue
        contribution = weight * float(score)
        raw_sum += contribution
        abs_weight_sum += abs(weight)
        breakdown[key] = round(contribution, 2)
        n_contributing += 1

    if n_contributing == 0 or abs_weight_sum == 0.0:
        return None, {"reason": "no_factor_components_available", "regime": regime}

    # Re-normalize to keep output on the 0-100 scale regardless of partial
    # coverage. Then clamp (the signed weights in BULL drive low_vol's
    # contribution negative, which can push the un-clamped raw_sum slightly
    # below 0 when low_vol_score is near 100 while others are near 0).
    normalized = raw_sum / abs_weight_sum
    subscore = max(0.0, min(100.0, normalized))

    return round(subscore, 1), {
        "regime": regime,
        "raw_blend": round(raw_sum, 2),
        "abs_weight_sum": round(abs_weight_sum, 2),
        "normalized": round(normalized, 2),
        "clamped_subscore": round(subscore, 1),
        "breakdown": breakdown,
        "n_components": n_contributing,
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
