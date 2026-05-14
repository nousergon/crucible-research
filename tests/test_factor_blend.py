"""Tests for the Phase 3 factor blend — compute_factor_subscore + the
factor-aware path through compute_composite_score.

Plan: ~/Development/alpha-engine-docs/private/factor-substrate-260513.md (Phase 3).
Sibling unit: tests/test_factor_scoring.py (Phase 1c upstream module).
"""

import pytest

from scoring.composite import compute_composite_score, compute_factor_subscore


_FULL_PROFILE = {
    "sector": "Technology",
    "quality_score": 80.0,
    "momentum_score": 90.0,
    "value_score": 40.0,
    "low_vol_score": 20.0,
}

_REGIME_WEIGHTS = {
    "bull": {
        "momentum_score": 0.40,
        "quality_score": 0.30,
        "value_score": 0.20,
        "low_vol_score": -0.10,
    },
    "bear": {
        "low_vol_score": 0.40,
        "quality_score": 0.30,
        "momentum_score": -0.20,
        "value_score": 0.10,
    },
    "neutral": {
        "momentum_score": 0.25,
        "quality_score": 0.25,
        "value_score": 0.25,
        "low_vol_score": 0.25,
    },
}


class TestComputeFactorSubscore:
    def test_bull_blend_full_profile(self):
        subscore, details = compute_factor_subscore(
            _FULL_PROFILE, "bull", _REGIME_WEIGHTS,
        )
        # raw = 0.4×90 + 0.3×80 + 0.2×40 + (-0.1)×20 = 36 + 24 + 8 - 2 = 66
        # abs_weight_sum = 0.4 + 0.3 + 0.2 + 0.1 = 1.0  →  normalized = 66 / 1.0 = 66
        assert subscore == 66.0
        assert details["regime"] == "bull"
        assert details["n_components"] == 4
        assert details["raw_blend"] == 66.0
        assert "momentum_score" in details["breakdown"]
        assert details["breakdown"]["low_vol_score"] == -2.0

    def test_bear_blend_inverts_momentum_low_vol(self):
        subscore, details = compute_factor_subscore(
            _FULL_PROFILE, "bear", _REGIME_WEIGHTS,
        )
        # raw = 0.4×20 + 0.3×80 + (-0.2)×90 + 0.1×40 = 8 + 24 - 18 + 4 = 18
        # abs_weight_sum = 0.4 + 0.3 + 0.2 + 0.1 = 1.0  →  normalized = 18
        assert subscore == 18.0
        assert details["regime"] == "bear"

    def test_neutral_blend_balanced(self):
        subscore, details = compute_factor_subscore(
            _FULL_PROFILE, "neutral", _REGIME_WEIGHTS,
        )
        # raw = 0.25×(80+90+40+20) = 0.25 × 230 = 57.5
        # abs_weight_sum = 1.0  →  normalized = 57.5
        assert subscore == 57.5

    def test_partial_profile_reallocates_weight(self):
        # Missing low_vol_score — pure-positive BULL weights remain
        partial = {"quality_score": 80.0, "momentum_score": 90.0, "value_score": 40.0}
        subscore, details = compute_factor_subscore(partial, "bull", _REGIME_WEIGHTS)
        # raw = 0.4×90 + 0.3×80 + 0.2×40 = 68;  abs_weight_sum = 0.9
        # normalized = 68 / 0.9 ≈ 75.56
        assert subscore == 75.6
        assert details["n_components"] == 3
        assert "low_vol_score" not in details["breakdown"]

    def test_clamp_at_lower_bound(self):
        # Synthetic case forcing un-clamped raw_sum below 0
        weights = {"bull": {"low_vol_score": -1.0}}
        profile = {"low_vol_score": 100.0}
        subscore, details = compute_factor_subscore(profile, "bull", weights)
        # raw = -100, abs_weight_sum = 1.0, normalized = -100 → clamped to 0
        assert subscore == 0.0
        assert details["normalized"] == -100.0
        assert details["clamped_subscore"] == 0.0

    def test_clamp_at_upper_bound(self):
        # Synthetic case forcing un-clamped raw_sum above 100
        weights = {"bull": {"momentum_score": 0.4, "low_vol_score": -1.0}}
        profile = {"momentum_score": 100.0, "low_vol_score": 0.0}
        subscore, details = compute_factor_subscore(profile, "bull", weights)
        # raw = 40 + 0 = 40, abs_weight_sum = 1.4, normalized = 28.6 → in range
        assert 0.0 <= subscore <= 100.0

    def test_none_profile_returns_none(self):
        subscore, details = compute_factor_subscore(None, "bull", _REGIME_WEIGHTS)
        assert subscore is None
        assert "no_profile" in details["reason"]

    def test_empty_profile_returns_none(self):
        subscore, _ = compute_factor_subscore({}, "bull", _REGIME_WEIGHTS)
        assert subscore is None

    def test_none_regime_returns_none(self):
        subscore, _ = compute_factor_subscore(_FULL_PROFILE, None, _REGIME_WEIGHTS)
        assert subscore is None

    def test_unknown_regime_returns_none(self):
        subscore, details = compute_factor_subscore(
            _FULL_PROFILE, "transitional", _REGIME_WEIGHTS,
        )
        assert subscore is None
        assert "no_blend_for_regime" in details["reason"]

    def test_regime_case_insensitive(self):
        subscore_a, _ = compute_factor_subscore(_FULL_PROFILE, "BULL", _REGIME_WEIGHTS)
        subscore_b, _ = compute_factor_subscore(_FULL_PROFILE, "bull", _REGIME_WEIGHTS)
        assert subscore_a == subscore_b

    def test_no_components_available(self):
        # Profile contains keys the regime weights don't reference
        weird_profile = {"unrelated_factor": 99.0}
        subscore, details = compute_factor_subscore(
            weird_profile, "bull", _REGIME_WEIGHTS,
        )
        assert subscore is None
        assert "no_factor_components_available" in details["reason"]

    def test_empty_regime_weights_returns_none(self):
        subscore, _ = compute_factor_subscore(_FULL_PROFILE, "bull", None)
        assert subscore is None


class TestCompositeWithFactorSubscore:
    def test_factor_blend_applied_when_provided(self):
        # Without factor blend
        base = compute_composite_score(
            quant_score=60.0, qual_score=80.0, sector_modifier=1.0,
        )
        # With 30% factor weight, factor_subscore=20 (low)
        blended = compute_composite_score(
            quant_score=60.0, qual_score=80.0, sector_modifier=1.0,
            factor_subscore=20.0, factor_weight=0.30,
        )
        # base weighted_base = 0.5×60 + 0.5×80 = 70
        # blended weighted_base = 0.7×70 + 0.3×20 = 49 + 6 = 55
        assert base["weighted_base"] == 70.0
        assert blended["weighted_base"] == 55.0
        assert blended["factor_subscore"] == 20.0
        assert blended["factor_weight_applied"] == 0.30

    def test_factor_blend_skipped_when_subscore_none(self):
        # None subscore → degrade to pure quant_qual
        result = compute_composite_score(
            quant_score=60.0, qual_score=80.0, sector_modifier=1.0,
            factor_subscore=None, factor_weight=0.30,
        )
        assert result["weighted_base"] == 70.0
        assert result["factor_subscore"] is None
        assert result["factor_weight_applied"] == 0.0

    def test_factor_blend_skipped_when_weight_zero(self):
        # Weight=0 → degrade even if subscore is provided
        result = compute_composite_score(
            quant_score=60.0, qual_score=80.0, sector_modifier=1.0,
            factor_subscore=20.0, factor_weight=0.0,
        )
        assert result["weighted_base"] == 70.0
        assert result["factor_weight_applied"] == 0.0
        # But factor_subscore is still echoed for observability
        assert result["factor_subscore"] == 20.0

    def test_backward_compatible_default_args(self):
        # Callers that don't pass factor_subscore / factor_weight see
        # behavior identical to pre-Phase-3
        result = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.0,
        )
        assert result["weighted_base"] == 70.0
        assert result["final_score"] == 70.0
        assert result["factor_subscore"] is None
        assert result["factor_weight_applied"] == 0.0

    def test_factor_blend_with_single_score(self):
        # When qual_score is None, blend should still apply to the quant-only base
        result = compute_composite_score(
            quant_score=60.0, qual_score=None, sector_modifier=1.0,
            factor_subscore=100.0, factor_weight=0.30,
        )
        # quant_qual_base = 60, weighted_base = 0.7×60 + 0.3×100 = 42 + 30 = 72
        assert result["weighted_base"] == 72.0

    def test_factor_blend_propagates_through_macro_shift(self):
        # Macro shift still applies after the blended base
        result = compute_composite_score(
            quant_score=50.0, qual_score=50.0, sector_modifier=1.30,  # +10 shift
            factor_subscore=100.0, factor_weight=0.30,
        )
        # quant_qual_base = 50, weighted_base = 0.7×50 + 0.3×100 = 35 + 30 = 65
        # final = 65 + 10 = 75
        assert result["weighted_base"] == 65.0
        assert result["macro_shift"] == 10.0
        assert result["final_score"] == 75.0

    def test_factor_blend_clamped_at_100(self):
        result = compute_composite_score(
            quant_score=100.0, qual_score=100.0, sector_modifier=1.30,
            factor_subscore=100.0, factor_weight=0.30,
        )
        # weighted_base = 100, + 10 macro shift = 110 → clamped to 100
        assert result["final_score"] == 100.0

    def test_factor_subscore_zero_pulls_score_down(self):
        # Sanity: a 0 factor subscore at 30% weight should pull a 70/70 base down
        result = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.0,
            factor_subscore=0.0, factor_weight=0.30,
        )
        # weighted_base = 0.7×70 + 0.3×0 = 49
        assert result["weighted_base"] == 49.0

    def test_factor_subscore_echoed_in_failed_score_path(self):
        # Both quant + qual None → score_failed True; factor_subscore still echoed
        result = compute_composite_score(
            quant_score=None, qual_score=None, sector_modifier=1.0,
            factor_subscore=55.0, factor_weight=0.30,
        )
        assert result["score_failed"] is True
        assert result["final_score"] is None
        assert result["factor_subscore"] == 55.0
        assert result["factor_weight_applied"] == 0.0
