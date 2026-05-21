"""
Composite scoring for sector-team architecture.

Two paths:

  * ``compute_composite_score`` — legacy 0.5×quant + 0.5×qual + 0.3×factor blend
    with macro_shift + boosts. Used by the held-stock thesis recompute path
    (research_graph.py:1641) and pre-Phase-4 callers. Stays as the
    regression baseline for Phase 4.
  * ``compute_composite_breakdown`` — Phase 4 of attractiveness-pillars-260520
    arc (lib v0.23.0 ``CompositeBreakdown``). 7-term composite:
    6 pillar contributions (qual from ``QualitativePillarAssessment``, quant
    from factor profile) + a legacy quant/qual/factor blend. At Phase 4
    default weights (pillar_weights all 0, legacy_blend 0.35/0.35/0.30) the
    output ``final_score`` is IDENTICAL to ``compute_composite_score``'s by
    construction — the plan-doc ±0.5 acceptance criterion is satisfied
    structurally, not by fixture tuning. Phase 6 weight optimizer ramps
    pillar weights up + legacy weights down.

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

# ── Pillar composite (Phase 4) defaults ──────────────────────────────────
#
# 7-term composite default state: every pillar_weight is 0 → legacy_blend
# carries the entire composite, which makes ``compute_composite_breakdown``
# numerically identical to ``compute_composite_score`` at default. Phase 6
# weight optimizer ramps pillar weights up + legacy weights down.

DEFAULT_PILLAR_WEIGHTS: dict[str, float] = {
    "quality": 0.0,
    "value": 0.0,
    "momentum": 0.0,
    "growth": 0.0,
    "stewardship": 0.0,
    "defensiveness": 0.0,
}

DEFAULT_WITHIN_PILLAR_QUAL_WEIGHT: float = 0.5

DEFAULT_LEGACY_BLEND_WEIGHTS: dict[str, float] = {
    "w_legacy_quant": 0.35,
    "w_legacy_qual": 0.35,
    "w_factor": 0.30,
}

# Per-pillar mapping into factor profile field names. Defensiveness sources
# from low_vol_score (low-vol = defensive). Phase 3b added growth_score +
# stewardship_score to factor_scoring._COMPOSITE_DEFS.
_PILLAR_TO_FACTOR_KEY: dict[str, str] = {
    "quality": "quality_score",
    "value": "value_score",
    "momentum": "momentum_score",
    "growth": "growth_score",
    "stewardship": "stewardship_score",
    "defensiveness": "low_vol_score",
}


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


# ── compute_composite_breakdown — Phase 4 7-term composite ───────────────


def compute_composite_breakdown(
    *,
    quant_score: float | None,
    qual_score: float | None,
    factor_subscore: float | None,
    pillar_assessment: dict | None,
    factor_profile: dict | None,
    sector_modifier: float,
    boosts: dict[str, float] | None = None,
    pillar_weights: dict[str, float] | None = None,
    within_pillar_qual_weight: float = DEFAULT_WITHIN_PILLAR_QUAL_WEIGHT,
    legacy_blend_weights: dict[str, float] | None = None,
    max_aggregate_boost: float = 10.0,
    macro_max_shift_points: float = MACRO_MAX_SHIFT_POINTS,
    macro_modifier_range: float = MACRO_MODIFIER_RANGE,
):
    """Compute the Phase 4 7-term composite breakdown.

    Returns a ``CompositeBreakdown`` (from alpha_engine_lib.pillars) with:
      * 6 ``PillarContribution`` entries (or empty when both pillar_assessment
        and factor_profile are absent — pure legacy path)
      * 1 ``LegacyComponentBlend`` carrying the opaque legacy quant/qual/factor
        terms

    At default ``pillar_weights`` (all 0) + default ``legacy_blend_weights``
    (0.35 / 0.35 / 0.30), ``final_score`` equals what
    ``compute_composite_score`` would have returned for the same inputs —
    BY CONSTRUCTION, not by fixture tuning. The plan-doc ±0.5 regression
    test asserts this equivalence holds across a synthesized fixture grid.

    Graceful-degrade per missing input:
      * pillar_assessment None    → every pillar's qual_component = None,
                                     effective α = 0.0 for every pillar
                                     (blended uses pure quant_component when
                                     available).
      * factor_profile None       → every pillar's quant_component = None,
                                     effective α = 1.0 for every pillar
                                     (blended uses pure qual_component when
                                     available).
      * BOTH None                 → every pillar's blended = None,
                                     contribution = 0; legacy_blend carries
                                     the composite. Identical to pre-Phase-4
                                     behavior.
      * quant_score / qual_score / factor_subscore individually None
                                  → that term contributes 0 to the legacy
                                     blend (weight kept, value treated as 0).
      * ALL inputs None           → score_failed=True, final_score=None.
    """
    from alpha_engine_lib.pillars import (
        CompositeBreakdown,
        LegacyComponentBlend,
        PillarContribution,
        PILLARS,
    )

    pw = dict(DEFAULT_PILLAR_WEIGHTS)
    if pillar_weights:
        pw.update(pillar_weights)

    lbw = dict(DEFAULT_LEGACY_BLEND_WEIGHTS)
    if legacy_blend_weights:
        lbw.update(legacy_blend_weights)

    # Hard-fail unscoreable: every input None → score_failed
    if (
        quant_score is None
        and qual_score is None
        and factor_subscore is None
        and not pillar_assessment
        and not factor_profile
    ):
        return CompositeBreakdown(
            final_score=None,
            weighted_base=None,
            macro_shift=0.0,
            boosts_total=0.0,
            catalyst_modulation=0,
            pillar_contributions=[],
            legacy_blend=LegacyComponentBlend(
                quant_score=None,
                qual_score=None,
                factor_subscore=None,
                w_legacy_quant=lbw["w_legacy_quant"],
                w_legacy_qual=lbw["w_legacy_qual"],
                w_factor=lbw["w_factor"],
                contribution=0.0,
            ),
            score_failed=True,
        )

    # ── Per-pillar contributions ──────────────────────────────────────
    # Build all 6 entries when EITHER pillar_assessment OR factor_profile
    # is present. When BOTH absent, leave the list empty (pure legacy path,
    # no observability gain from emitting all-None pillar entries).
    pillar_contributions: list = []
    if pillar_assessment or factor_profile:
        for pillar in PILLARS:
            qual_comp: float | None = None
            if pillar_assessment:
                # QualitativePillarAssessment.{pillar} is a PillarSubscore;
                # may arrive as a model_dump dict (the usual case post-extraction)
                # or as a typed object — handle both.
                pillar_obj = pillar_assessment.get(pillar)
                if isinstance(pillar_obj, dict):
                    qual_comp = pillar_obj.get("score")
                elif pillar_obj is not None:
                    qual_comp = getattr(pillar_obj, "score", None)
                qual_comp = float(qual_comp) if qual_comp is not None else None

            quant_comp: float | None = None
            if factor_profile:
                factor_key = _PILLAR_TO_FACTOR_KEY[pillar]
                raw = factor_profile.get(factor_key)
                quant_comp = float(raw) if raw is not None else None

            # Effective α per the graceful-degrade contract
            if qual_comp is None and quant_comp is None:
                eff_alpha = within_pillar_qual_weight  # arbitrary, contribution will be 0
                blended = None
                contribution = 0.0
            elif qual_comp is None:
                eff_alpha = 0.0
                blended = quant_comp
                contribution = pw[pillar] * (quant_comp or 0.0)
            elif quant_comp is None:
                eff_alpha = 1.0
                blended = qual_comp
                contribution = pw[pillar] * (qual_comp or 0.0)
            else:
                eff_alpha = within_pillar_qual_weight
                blended = eff_alpha * qual_comp + (1.0 - eff_alpha) * quant_comp
                contribution = pw[pillar] * blended

            pillar_contributions.append(
                PillarContribution(
                    pillar=pillar,  # type: ignore[arg-type]
                    qual_component=qual_comp,
                    quant_component=quant_comp,
                    within_pillar_qual_weight=eff_alpha,
                    blended=blended,
                    pillar_weight=pw[pillar],
                    contribution=contribution,
                )
            )

    # ── Legacy blend ──────────────────────────────────────────────────
    # Match legacy ``compute_composite_score``'s partial-input handling so
    # the Phase 4 ±0.5 regression criterion holds across the full input
    # space, not just the all-components-present case. Specifically:
    #   * If BOTH quant_score AND qual_score are None: legacy treats the
    #     quant_qual_base as None (degenerate); we mirror by zero-treating
    #     both — combined with factor_subscore alone this matches legacy
    #     since legacy's "all None → return None" path was already caught
    #     above.
    #   * If ONLY quant_score is None: legacy uses qual_score at FULL
    #     w_legacy_quant+w_legacy_qual combined weight (its quant_qual_base
    #     becomes qual_score, then ×0.7 in weighted_base). We replicate by
    #     redistributing w_legacy_quant onto w_legacy_qual locally.
    #   * Symmetric when ONLY qual_score is None.
    #   * factor_subscore None: legacy skips it (factor_weight 0); we
    #     redistribute w_factor onto the quant_qual side proportionally,
    #     matching legacy's branch that drops factor_subscore from the
    #     weighted_base when it's None.
    eff_w_legacy_quant = lbw["w_legacy_quant"]
    eff_w_legacy_qual = lbw["w_legacy_qual"]
    eff_w_factor = lbw["w_factor"]

    qq_pool = eff_w_legacy_quant + eff_w_legacy_qual
    if quant_score is None and qual_score is not None:
        eff_w_legacy_qual = qq_pool
        eff_w_legacy_quant = 0.0
    elif qual_score is None and quant_score is not None:
        eff_w_legacy_quant = qq_pool
        eff_w_legacy_qual = 0.0

    if factor_subscore is None:
        # Redistribute w_factor proportionally onto the quant_qual side
        # (matches legacy's "factor blend disabled / no profile" branch
        # where weighted_base falls back to pure quant_qual_base = 0.5×quant
        # + 0.5×qual). With w_legacy_quant=w_legacy_qual=0.35 default, the
        # redistribution yields 0.5/0.5 — identical to legacy formula.
        if eff_w_legacy_quant + eff_w_legacy_qual > 0:
            scale = (eff_w_legacy_quant + eff_w_legacy_qual + eff_w_factor) / (
                eff_w_legacy_quant + eff_w_legacy_qual
            )
            eff_w_legacy_quant *= scale
            eff_w_legacy_qual *= scale
        eff_w_factor = 0.0

    legacy_contribution = (
        eff_w_legacy_quant * (quant_score or 0.0)
        + eff_w_legacy_qual * (qual_score or 0.0)
        + eff_w_factor * (factor_subscore or 0.0)
    )
    # Persist the CONFIGURED weights (not the effective re-weighted ones)
    # on the LegacyComponentBlend — that's the audit trail of what the
    # operator set. The effective re-weighting is a runtime detail.
    legacy_blend = LegacyComponentBlend(
        quant_score=quant_score,
        qual_score=qual_score,
        factor_subscore=factor_subscore,
        w_legacy_quant=lbw["w_legacy_quant"],
        w_legacy_qual=lbw["w_legacy_qual"],
        w_factor=lbw["w_factor"],
        contribution=legacy_contribution,
    )

    # ── Weighted base + macro + boosts + catalyst ─────────────────────
    weighted_base = sum(c.contribution for c in pillar_contributions) + legacy_contribution

    macro_shift = (sector_modifier - 1.0) / macro_modifier_range * macro_max_shift_points

    boosts_total = 0.0
    if boosts:
        boosts_total = sum(boosts.values())
        boosts_total = max(-max_aggregate_boost, min(max_aggregate_boost, boosts_total))

    catalyst_modulation = 0
    if pillar_assessment:
        catalyst_modulation = int(pillar_assessment.get("catalyst_horizon_modulation", 0) or 0)
        catalyst_modulation = max(-20, min(20, catalyst_modulation))

    final = weighted_base + macro_shift + boosts_total + catalyst_modulation
    final = max(0.0, min(100.0, final))

    return CompositeBreakdown(
        final_score=round(final, 1),
        weighted_base=round(weighted_base, 1),
        macro_shift=round(macro_shift, 1),
        boosts_total=round(boosts_total, 1),
        catalyst_modulation=catalyst_modulation,
        pillar_contributions=pillar_contributions,
        legacy_blend=legacy_blend,
        score_failed=False,
    )

