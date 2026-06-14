"""Tests for the technical scoring engine and aggregator."""

import pytest

_technical = pytest.importorskip("scoring.technical", reason="scoring.technical is gitignored")
_score_rsi = _technical._score_rsi
_score_macd = _technical._score_macd
_score_price_vs_ma = _technical._score_price_vs_ma
_score_momentum = _technical._score_momentum
compute_technical_score = _technical.compute_technical_score
compute_momentum_percentiles = _technical.compute_momentum_percentiles

from scoring.composite import (
    PillarCoverageError,
    compute_composite_breakdown,
    compute_composite_score,
    score_to_rating,
)


class TestRSIScoring:
    def test_neutral_regime_oversold(self):
        assert _score_rsi(25, "neutral") == 100.0

    def test_neutral_regime_overbought(self):
        assert _score_rsi(75, "neutral") == 0.0

    def test_neutral_regime_midpoint(self):
        score = _score_rsi(50, "neutral")
        assert 45 <= score <= 55  # roughly in middle

    def test_bull_overbought_matches_neutral_post_revert(self):
        # ROADMAP L1695 — the asymmetric bull-regime overbought relaxation
        # (overbought=80 vs neutral=70) was reverted 2026-05-15 because the
        # regime label driving it is unvalidated. Post-revert invariant: the
        # bull RSI curve is identical to neutral (no asymmetric relaxation).
        for rsi in (65, 70, 72, 75, 80, 85):
            assert _score_rsi(rsi, "bull") == _score_rsi(rsi, "neutral"), (
                f"bull/neutral RSI score diverged at rsi={rsi} — the bull "
                f"asymmetry was supposed to be reverted (L1695)"
            )

    def test_bear_regime_raises_oversold_threshold(self):
        # RSI=35 should not be as bullish in bear regime
        bear_score = _score_rsi(35, "bear")
        neutral_score = _score_rsi(35, "neutral")
        assert bear_score < neutral_score

    def test_score_in_valid_range(self):
        for rsi in [0, 15, 30, 50, 70, 85, 100]:
            for regime in ["bull", "neutral", "bear"]:
                score = _score_rsi(rsi, regime)
                assert 0.0 <= score <= 100.0


class TestMACDScoring:
    def test_bullish_cross_above_zero(self):
        assert _score_macd(1.0, True) == 100.0

    def test_bullish_cross_below_zero(self):
        assert _score_macd(1.0, False) == 70.0

    def test_bearish_cross_above_zero(self):
        assert _score_macd(-1.0, True) == 30.0

    def test_bearish_cross_below_zero(self):
        assert _score_macd(-1.0, False) == 0.0

    def test_no_cross_above_zero(self):
        assert _score_macd(0.0, True) == 60.0

    def test_no_cross_below_zero(self):
        assert _score_macd(0.0, False) == 40.0


class TestPriceVsMAScoring:
    def test_none_returns_50(self):
        assert _score_price_vs_ma(None) == 50.0

    def test_at_ma(self):
        assert _score_price_vs_ma(0.0) == 50.0

    def test_far_above(self):
        assert _score_price_vs_ma(10.0) > 80.0
        assert _score_price_vs_ma(10.0) <= 100.0

    def test_far_below(self):
        assert _score_price_vs_ma(-15.0) < 30.0
        assert _score_price_vs_ma(-15.0) >= 0.0

    def test_above_5pct(self):
        score = _score_price_vs_ma(5.0)
        assert score >= 80.0

    def test_valid_range(self):
        for pct in [-30, -10, -5, 0, 5, 10, 25]:
            score = _score_price_vs_ma(pct)
            assert 0.0 <= score <= 100.0


class TestCompositeScore:
    def test_strong_bull_indicators(self):
        indicators = {
            "rsi_14": 25.0,       # oversold = bullish
            "macd_cross": 1.0,    # bullish cross
            "macd_above_zero": True,
            "price_vs_ma50": 6.0,
            "price_vs_ma200": 8.0,
            "momentum_20d": 5.0,
        }
        score = compute_technical_score(indicators, market_regime="neutral")
        assert score > 70.0

    def test_strong_bear_indicators(self):
        indicators = {
            "rsi_14": 80.0,       # overbought = bearish
            "macd_cross": -1.0,   # bearish cross
            "macd_above_zero": False,
            "price_vs_ma50": -10.0,
            "price_vs_ma200": -20.0,
            "momentum_20d": -10.0,
        }
        score = compute_technical_score(indicators, market_regime="neutral")
        assert score < 30.0

    def test_score_clipped_to_range(self):
        indicators = {"rsi_14": 50, "macd_cross": 0, "macd_above_zero": True}
        score = compute_technical_score(indicators)
        assert 0.0 <= score <= 100.0


class TestCompositeWeightsPerSector:
    """Lock the precedence chain for per-sector composite_weights overrides.

    Override resolution order:
      1. team_id kwarg (canonical sector team — wins outright)
      2. sector kwarg → team_id via SECTOR_TEAM_MAP
      3. Fallback to top-level composite_weights when no entry / malformed

    Patches TECHNICAL_CFG so tests don't depend on live scoring.yaml shape.
    """

    INDICATORS_FLAT = {
        "rsi_14": 50.0,         # mid
        "macd_cross": 0.0,      # no cross above zero → 60
        "macd_above_zero": True,
        "price_vs_ma50": 0.0,   # 50
        "price_vs_ma200": 0.0,  # 50
        "momentum_20d": 0.0,    # 50 (no percentile)
    }

    def setup_method(self):
        # Stash and replace TECHNICAL_CFG composite_weights blocks
        self._orig_global = _technical.TECHNICAL_CFG.get("composite_weights", {})
        self._orig_overrides = _technical.TECHNICAL_CFG.get("composite_weights_per_sector")
        _technical.TECHNICAL_CFG["composite_weights"] = {
            "rsi": 0.25, "macd": 0.20, "ma50": 0.15, "ma200": 0.15, "momentum": 0.25,
        }
        # Clear the warn-once cache so each test sees fresh warnings
        _technical._warned_overrides.clear()

    def teardown_method(self):
        _technical.TECHNICAL_CFG["composite_weights"] = self._orig_global
        if self._orig_overrides is None:
            _technical.TECHNICAL_CFG.pop("composite_weights_per_sector", None)
        else:
            _technical.TECHNICAL_CFG["composite_weights_per_sector"] = self._orig_overrides
        _technical._warned_overrides.clear()

    def test_no_team_id_uses_global_weights(self):
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
        }
        score = compute_technical_score(self.INDICATORS_FLAT)
        # Global weights: 0.25×50 + 0.20×60 + 0.15×50 + 0.15×50 + 0.25×50
        # = 12.5 + 12.0 + 7.5 + 7.5 + 12.5 = 52.0
        assert score == pytest.approx(52.0, abs=0.01)

    def test_team_id_with_override_applies(self):
        # 100% RSI weight → score = 50 (mid RSI)
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
        }
        score = compute_technical_score(self.INDICATORS_FLAT, team_id="healthcare")
        assert score == pytest.approx(50.0, abs=0.01)

    def test_team_id_without_override_falls_back_to_global(self):
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
        }
        # No override for technology → global weights → 52.0
        score = compute_technical_score(self.INDICATORS_FLAT, team_id="technology")
        assert score == pytest.approx(52.0, abs=0.01)

    def test_sector_resolves_to_team_id_via_map(self):
        # "Healthcare" GICS → team_id="healthcare" → override applies
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
        }
        score = compute_technical_score(self.INDICATORS_FLAT, sector="Healthcare")
        assert score == pytest.approx(50.0, abs=0.01)

    def test_sector_alternate_label_resolves(self):
        # "Health Care" (alternate GICS label) → "healthcare"
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
        }
        score = compute_technical_score(self.INDICATORS_FLAT, sector="Health Care")
        assert score == pytest.approx(50.0, abs=0.01)

    def test_team_id_takes_precedence_over_sector(self):
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
            "technology": {"rsi": 0.0, "macd": 1.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
        }
        # Pass sector=Healthcare but team_id=technology — team_id wins
        score = compute_technical_score(
            self.INDICATORS_FLAT, sector="Healthcare", team_id="technology",
        )
        # 100% MACD weight → 60.0 (no_cross_above_zero)
        assert score == pytest.approx(60.0, abs=0.01)

    def test_unknown_sector_falls_back_to_global(self):
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0},
        }
        # "UnknownSector" not in SECTOR_TEAM_MAP → resolved team_id is None → global
        score = compute_technical_score(self.INDICATORS_FLAT, sector="UnknownSector")
        assert score == pytest.approx(52.0, abs=0.01)

    def test_malformed_override_missing_key_falls_back(self, caplog):
        # Missing "momentum" key — should warn + fall back to global
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 0.3, "macd": 0.3, "ma50": 0.2, "ma200": 0.2},
        }
        with caplog.at_level("WARNING"):
            score = compute_technical_score(self.INDICATORS_FLAT, team_id="healthcare")
        assert score == pytest.approx(52.0, abs=0.01)
        assert any("missing keys" in r.message for r in caplog.records)

    def test_malformed_override_bad_sum_falls_back(self, caplog):
        # Weights sum to 0.5, not 1.0 — should warn + fall back
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {
            "healthcare": {"rsi": 0.1, "macd": 0.1, "ma50": 0.1, "ma200": 0.1, "momentum": 0.1},
        }
        with caplog.at_level("WARNING"):
            score = compute_technical_score(self.INDICATORS_FLAT, team_id="healthcare")
        assert score == pytest.approx(52.0, abs=0.01)
        assert any("sums to" in r.message for r in caplog.records)

    def test_empty_overrides_dict_uses_global(self):
        _technical.TECHNICAL_CFG["composite_weights_per_sector"] = {}
        score = compute_technical_score(self.INDICATORS_FLAT, team_id="healthcare")
        assert score == pytest.approx(52.0, abs=0.01)

    def test_overrides_absent_uses_global(self):
        _technical.TECHNICAL_CFG.pop("composite_weights_per_sector", None)
        score = compute_technical_score(self.INDICATORS_FLAT, team_id="healthcare")
        assert score == pytest.approx(52.0, abs=0.01)


class TestMomentumPercentiles:
    def test_highest_momentum_gets_high_percentile(self):
        data = {"A": 10.0, "B": 5.0, "C": -5.0}
        result = compute_momentum_percentiles(data)
        assert result["A"] > result["B"] > result["C"]

    def test_handles_none(self):
        data = {"A": 5.0, "B": None}
        result = compute_momentum_percentiles(data)
        assert "A" in result
        assert "B" in result
        assert result["B"] == 50.0


class TestCompositeScoring:
    def test_score_formula(self):
        # With neutral modifier (1.0) the macro_shift is 0, so final == weighted_base
        result = compute_composite_score(
            quant_score=80.0, qual_score=60.0, sector_modifier=1.0,
        )
        expected_base = 80.0 * 0.50 + 60.0 * 0.50  # 70.0
        assert abs(result["weighted_base"] - expected_base) < 0.1
        assert abs(result["macro_shift"]) < 0.01
        assert abs(result["final_score"] - expected_base) < 0.1

    def test_macro_shift_additive(self):
        result_headwind = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=0.70,
        )
        result_tailwind = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.30,
        )
        assert abs(result_headwind["macro_shift"] - (-10.0)) < 0.1
        assert abs(result_tailwind["macro_shift"] - 10.0) < 0.1
        assert abs(result_tailwind["final_score"] - result_headwind["final_score"] - 20.0) < 0.2

    def test_macro_modifier_applied(self):
        result_neutral = compute_composite_score(
            quant_score=60.0, qual_score=60.0, sector_modifier=1.0,
        )
        result_boosted = compute_composite_score(
            quant_score=60.0, qual_score=60.0, sector_modifier=1.2,
        )
        assert result_boosted["final_score"] > result_neutral["final_score"]

    def test_score_clipped_at_100(self):
        result = compute_composite_score(
            quant_score=100, qual_score=100, sector_modifier=1.3,
        )
        assert result["final_score"] <= 100.0

    def test_score_clipped_at_0(self):
        result = compute_composite_score(
            quant_score=5, qual_score=5, sector_modifier=0.70,
        )
        assert result["final_score"] >= 0.0

    def test_missing_quant_uses_qual(self):
        result = compute_composite_score(
            quant_score=None, qual_score=70.0, sector_modifier=1.0,
        )
        assert result["final_score"] == 70.0
        assert result["score_failed"] is False

    def test_both_missing_returns_failed(self):
        result = compute_composite_score(
            quant_score=None, qual_score=None, sector_modifier=1.0,
        )
        assert result["score_failed"] is True

    def test_boosts_capped(self):
        result = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.0,
            boosts={"pead": 5, "revision": 3, "options": 4, "insider": 5},
        )
        assert result["total_boost"] == 10.0  # capped at 10

    def test_ratings(self):
        assert score_to_rating(75) == "BUY"
        assert score_to_rating(69) == "HOLD"
        assert score_to_rating(70) == "BUY"   # default buy threshold is 70
        assert score_to_rating(55) == "HOLD"
        assert score_to_rating(41) == "HOLD"
        assert score_to_rating(40) == "SELL"  # sell threshold is <= 40


# ── compute_composite_breakdown (Phase 4) ────────────────────────────────


# Fixture grid: every (quant, qual, factor, modifier, boosts) combination
# the score_aggregator can realistically pass. Default-weight regression:
# new ≡ legacy within ±0.5 across the full grid. At Phase 4 defaults
# (pillar_weights=0 each, legacy_blend 0.35/0.35/0.30), the actual delta is
# 0.0 by construction — the ±0.5 tolerance is the plan-doc contract
# accommodating future weight tuning, not an observed gap.
_REGRESSION_CASES = [
    # (label, quant, qual, factor, sector_modifier, boosts)
    ("all present, neutral macro",       70.0, 80.0, 60.0, 1.00, None),
    ("all present, OW macro",            70.0, 80.0, 60.0, 1.20, None),
    ("all present, UW macro",            70.0, 80.0, 60.0, 0.80, None),
    ("all present, extreme OW",          70.0, 80.0, 60.0, 1.30, None),
    ("all present, extreme UW",          70.0, 80.0, 60.0, 0.70, None),
    ("quant None — qual carries",        None, 80.0, 60.0, 1.00, None),
    ("qual None — quant carries",        70.0, None, 60.0, 1.00, None),
    ("factor None — quant+qual only",    70.0, 80.0, None, 1.00, None),
    ("quant+factor None — qual only",    None, 80.0, None, 1.00, None),
    ("qual+factor None — quant only",    70.0, None, None, 1.00, None),
    ("positive boost",                   70.0, 80.0, 60.0, 1.00, {"momentum": 3.0}),
    ("negative boost",                   70.0, 80.0, 60.0, 1.00, {"red_flag": -5.0}),
    ("boost cap pos",                    70.0, 80.0, 60.0, 1.00,
     {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0}),
    ("boost cap neg",                    70.0, 80.0, 60.0, 1.00,
     {"a": -5.0, "b": -4.0, "c": -3.0, "d": -2.0}),
    ("low scores, OW",                   30.0, 35.0, 25.0, 1.30, None),
    ("high scores, UW",                  90.0, 95.0, 85.0, 0.70, None),
    ("midpoint scores",                  50.0, 50.0, 50.0, 1.00, None),
    ("edge: all zeros",                   0.0,  0.0,  0.0, 1.00, None),
    ("edge: all hundreds",              100.0,100.0,100.0, 1.00, None),
    ("clamp upper — high + extreme OW", 100.0,100.0,100.0, 1.30, {"a": 10.0}),
    ("clamp lower — zeros + extreme UW",  0.0,  0.0,  0.0, 0.70, {"a": -10.0}),
]


class TestCompositeBreakdownLegacyRegression:
    """Phase 4 acceptance criterion: at default weights, new composite
    final_score equals legacy compute_composite_score within ±0.5 across
    a synthesized input grid (plan-doc — attractiveness-pillars-260520).

    By construction the delta is 0.0 — pillar_weights all 0 means the
    7-term composite reduces to the legacy 0.35×quant + 0.35×qual +
    0.30×factor formula. The ±0.5 tolerance accommodates Phase 6 weight
    tuning that may produce floating-point round drift below 0.5
    score-points."""

    @pytest.mark.parametrize("label,quant,qual,factor,modifier,boosts", _REGRESSION_CASES)
    def test_default_weights_match_legacy(self, label, quant, qual, factor, modifier, boosts):
        legacy = compute_composite_score(
            quant_score=quant, qual_score=qual, sector_modifier=modifier,
            factor_subscore=factor, factor_weight=0.30, boosts=boosts,
        )
        new = compute_composite_breakdown(
            quant_score=quant, qual_score=qual, factor_subscore=factor,
            pillar_assessment=None, factor_profile=None,
            sector_modifier=modifier, boosts=boosts,
        )
        # Both should agree on score_failed
        assert legacy["score_failed"] == new.score_failed, label
        if legacy["score_failed"]:
            assert legacy["final_score"] is None
            assert new.final_score is None
            return
        assert abs(legacy["final_score"] - new.final_score) <= 0.5, (
            f"{label}: legacy={legacy['final_score']} vs new={new.final_score}"
        )

    def test_both_missing_returns_failed(self):
        new = compute_composite_breakdown(
            quant_score=None, qual_score=None, factor_subscore=None,
            pillar_assessment=None, factor_profile=None,
            sector_modifier=1.0,
        )
        assert new.score_failed is True
        assert new.final_score is None

    def test_pillar_contributions_empty_when_no_pillar_inputs(self):
        """No pillar_assessment + no factor_profile → empty
        pillar_contributions list (pure legacy path)."""
        new = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=None, factor_profile=None,
            sector_modifier=1.0,
        )
        assert new.pillar_contributions == []

    def test_pillar_contributions_populated_when_either_input_present(self):
        """Either pillar_assessment OR factor_profile present → 6 pillar
        contributions emitted (even at pillar_weights=0)."""
        # Only factor_profile present
        new = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=None,
            factor_profile={
                "quality_score": 70.0, "value_score": 60.0,
                "momentum_score": 80.0, "growth_score": 65.0,
                "stewardship_score": 55.0, "low_vol_score": 50.0,
            },
            sector_modifier=1.0,
        )
        assert len(new.pillar_contributions) == 6
        # At pillar_weights=0, every contribution is 0
        assert all(c.contribution == 0.0 for c in new.pillar_contributions)
        # qual_components all None (no pillar_assessment); within_pillar_qual_weight
        # should reflect graceful-degrade α=0.0
        assert all(c.qual_component is None for c in new.pillar_contributions)
        assert all(c.within_pillar_qual_weight == 0.0 for c in new.pillar_contributions)

    def test_pillar_weights_non_zero_changes_score(self):
        """Phase 6 simulation: ramp pillar weights up + legacy down. At
        non-trivial pillar weights, the new composite SHOULD diverge from
        legacy — confirms the pillar contribution actually flows through."""
        pillar_assessment = {
            p: {"pillar": p, "score": 90, "confidence": "high"}
            for p in ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")
        }
        pillar_assessment["catalyst_horizon_modulation"] = 0

        # Half-and-half ramp: pillar_weights total 0.5, legacy total 0.5
        pillar_w = {p: 0.5 / 6 for p in (
            "quality", "value", "momentum", "growth", "stewardship", "defensiveness"
        )}
        legacy_w = {"w_legacy_quant": 0.175, "w_legacy_qual": 0.175, "w_factor": 0.150}

        new = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=pillar_assessment,
            factor_profile=None,
            sector_modifier=1.0,
            pillar_weights=pillar_w,
            legacy_blend_weights=legacy_w,
        )
        # Pillar side contributes 0.5 × 90 = 45; legacy side at half weight
        # contributes 0.175×70 + 0.175×80 + 0.150×60 = 12.25 + 14 + 9 = 35.25.
        # weighted_base = 45 + 35.25 = 80.25.
        assert new.weighted_base == pytest.approx(80.25, abs=0.1)
        # final_score with no macro/boosts ≈ 80.25 → 80.2 or 80.3 rounded
        assert new.final_score is not None
        assert abs(new.final_score - 80.25) < 0.1

    def test_catalyst_modulation_flows_through(self):
        """catalyst_horizon_modulation from pillar_assessment adjusts
        final_score by ±N points."""
        pillar_assessment_pos = {
            p: {"pillar": p, "score": 75, "confidence": "high"}
            for p in ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")
        }
        pillar_assessment_pos["catalyst_horizon_modulation"] = 15

        pillar_assessment_neg = dict(pillar_assessment_pos)
        pillar_assessment_neg["catalyst_horizon_modulation"] = -15

        baseline = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment={**pillar_assessment_pos, "catalyst_horizon_modulation": 0},
            factor_profile=None,
            sector_modifier=1.0,
        )
        with_pos = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=pillar_assessment_pos,
            factor_profile=None,
            sector_modifier=1.0,
        )
        with_neg = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=pillar_assessment_neg,
            factor_profile=None,
            sector_modifier=1.0,
        )
        # At Phase 4 default pillar_weights=0, the only difference between
        # these three breakdowns is catalyst_modulation — final_score
        # should differ by exactly 15 (or clamped at 0/100).
        assert with_pos.catalyst_modulation == 15
        assert with_neg.catalyst_modulation == -15
        assert with_pos.final_score - baseline.final_score == pytest.approx(15.0, abs=0.1)
        assert baseline.final_score - with_neg.final_score == pytest.approx(15.0, abs=0.1)

    def test_within_pillar_alpha_degrades_to_pure_qual_when_quant_absent(self):
        """factor_profile None for ticker → within_pillar_qual_weight forced
        to 1.0 for every pillar; blended equals pillar_assessment score."""
        pillar_assessment = {
            p: {"pillar": p, "score": 70, "confidence": "high"}
            for p in ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")
        }
        pillar_assessment["catalyst_horizon_modulation"] = 0

        new = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=pillar_assessment,
            factor_profile=None,
            sector_modifier=1.0,
        )
        for c in new.pillar_contributions:
            assert c.within_pillar_qual_weight == 1.0
            assert c.blended == 70.0
            assert c.quant_component is None

    def test_within_pillar_alpha_degrades_to_pure_quant_when_qual_absent(self):
        """pillar_assessment None for ticker (with factor_profile present)
        → within_pillar_qual_weight forced to 0.0; blended equals factor
        score."""
        new = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=None,
            factor_profile={
                "quality_score": 65.0, "value_score": 65.0,
                "momentum_score": 65.0, "growth_score": 65.0,
                "stewardship_score": 65.0, "low_vol_score": 65.0,
            },
            sector_modifier=1.0,
        )
        for c in new.pillar_contributions:
            assert c.within_pillar_qual_weight == 0.0
            assert c.blended == 65.0
            assert c.qual_component is None


class TestPillarCoverageGuard:
    """Hardening Item 1 (2026-05-21 AQR cutover incident): when config
    has Σ pillar_weights > 0 but a ticker has no pillar inputs, the
    consumer raises PillarCoverageError rather than silently producing
    a degenerate composite."""

    def test_raises_when_pillar_weights_nonzero_and_no_inputs(self):
        """AQR-cutover-equivalent config (pillar Σ=1.0, legacy Σ=0.0) +
        ticker with no pillar_assessment + no factor_profile → raise."""
        with pytest.raises(PillarCoverageError):
            compute_composite_breakdown(
                quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
                pillar_assessment=None, factor_profile=None,
                sector_modifier=1.0,
                pillar_weights={
                    "quality": 0.25, "value": 0.20, "momentum": 0.20,
                    "growth": 0.15, "defensiveness": 0.10,
                    "stewardship": 0.10,
                },
                legacy_blend_weights={
                    "w_legacy_quant": 0.0, "w_legacy_qual": 0.0,
                    "w_factor": 0.0,
                },
            )

    def test_no_raise_at_phase4_default_weights_when_no_inputs(self):
        """At Phase 4 defaults (pillar_weights all 0, legacy_blend
        0.35/0.35/0.30), no pillar inputs is fine — legacy_blend carries
        the composite by construction. This is the safe fallback the
        revert config restored."""
        breakdown = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=None, factor_profile=None,
            sector_modifier=1.0,
            # Default weights (omitted) — pillar all 0, legacy 0.35/0.35/0.30
        )
        assert breakdown.final_score is not None
        assert breakdown.score_failed is False
        assert breakdown.pillar_contributions == []

    def test_no_raise_when_pillar_assessment_present(self):
        """AQR weights + pillar_assessment present → no raise; pillar
        path takes over."""
        pillar_assessment = {
            p: {"pillar": p, "score": 70, "confidence": "high"}
            for p in ("quality", "value", "momentum", "growth",
                      "stewardship", "defensiveness")
        }
        pillar_assessment["catalyst_horizon_modulation"] = 0
        breakdown = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=pillar_assessment,
            factor_profile=None,
            sector_modifier=1.0,
            pillar_weights={
                "quality": 0.25, "value": 0.20, "momentum": 0.20,
                "growth": 0.15, "defensiveness": 0.10, "stewardship": 0.10,
            },
            legacy_blend_weights={
                "w_legacy_quant": 0.0, "w_legacy_qual": 0.0,
                "w_factor": 0.0,
            },
        )
        assert breakdown.final_score is not None
        assert len(breakdown.pillar_contributions) == 6

    def test_no_raise_when_factor_profile_present(self):
        """AQR weights + factor_profile present (but pillar_assessment
        None) → no raise; quant-only pillar path."""
        breakdown = compute_composite_breakdown(
            quant_score=70.0, qual_score=80.0, factor_subscore=60.0,
            pillar_assessment=None,
            factor_profile={
                "quality_score": 65.0, "value_score": 65.0,
                "momentum_score": 65.0, "growth_score": 65.0,
                "stewardship_score": 65.0, "low_vol_score": 65.0,
            },
            sector_modifier=1.0,
            pillar_weights={
                "quality": 0.25, "value": 0.20, "momentum": 0.20,
                "growth": 0.15, "defensiveness": 0.10, "stewardship": 0.10,
            },
            legacy_blend_weights={
                "w_legacy_quant": 0.0, "w_legacy_qual": 0.0,
                "w_factor": 0.0,
            },
        )
        assert breakdown.final_score is not None
        assert len(breakdown.pillar_contributions) == 6

    def test_PillarCoverageError_is_RuntimeError_subclass(self):
        """Per-ticker callers catching RuntimeError continue to work."""
        assert issubclass(PillarCoverageError, RuntimeError)
        assert score_to_rating(35) == "SELL"


class TestMacroOverlayKnob:
    """Macro-shift overlay enable knob (config#1060/#1061).

    The overlay costs ~+0.054 realized rank-IC and is structurally
    mis-specified; the disable is applied private-first via runtime config
    (``aggregator.macro_overlay.enabled: false``). These tests pin BOTH
    sides of the knob on both composite functions:
      * default (enabled) preserves CURRENT macro_shift behavior;
      * disabled forces macro_shift == 0.0 so the persisted final_score
        equals combined_score (weighted_base + boosts).
    """

    # ── module + config default preserves behavior ──────────────────────
    def test_public_default_is_enabled(self):
        from scoring.composite import MACRO_OVERLAY_ENABLED
        assert MACRO_OVERLAY_ENABLED is True, (
            "public-code default MUST preserve current behavior (overlay ON) "
            "per the divergence policy — the disable lives in runtime config"
        )

    def test_code_level_defaults_preserve_behavior(self):
        # The CODE default (composite.py constants) is the public reference
        # baseline — overlay ON at ±10 pts. The live runtime config may
        # override the points (the production scoring.yaml sets 25.0) and is
        # where the DISABLE lands; the code default must stay frozen.
        from scoring.composite import (
            MACRO_OVERLAY_ENABLED,
            MACRO_MAX_SHIFT_POINTS,
            MACRO_MODIFIER_RANGE,
        )
        assert MACRO_OVERLAY_ENABLED is True
        assert abs(MACRO_MAX_SHIFT_POINTS - 10.0) < 1e-9
        assert abs(MACRO_MODIFIER_RANGE - 0.30) < 1e-9

    def test_config_reads_runtime_overlay_knob(self):
        # config.py exposes the runtime knob; default (no macro_overlay block
        # in yaml) is enabled True. The points/range are read from yaml.
        import config
        assert config.MACRO_OVERLAY_ENABLED is True
        assert isinstance(config.MACRO_MAX_SHIFT_POINTS, float)
        assert isinstance(config.MACRO_MODIFIER_RANGE, float)
        assert config.MACRO_MAX_SHIFT_POINTS > 0.0
        assert config.MACRO_MODIFIER_RANGE > 0.0

    # ── compute_composite_score ─────────────────────────────────────────
    def test_default_preserves_macro_shift(self):
        # Tailwind sector → +10 macro_shift at default (enabled).
        on = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.30,
        )
        assert abs(on["macro_shift"] - 10.0) < 0.1
        assert abs(on["final_score"] - 80.0) < 0.2  # 70 base + 10 shift

    def test_disabled_zeroes_macro_shift_score(self):
        on = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.30,
        )
        off = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=1.30,
            macro_overlay_enabled=False,
        )
        assert abs(off["macro_shift"]) < 1e-9
        # final_score collapses to weighted_base (combined_score), no overlay.
        assert abs(off["final_score"] - off["weighted_base"]) < 1e-9
        assert off["final_score"] != on["final_score"]

    def test_disabled_zeroes_for_headwind_sector(self):
        # The drag removal must hold for BOTH signs of the overlay.
        off = compute_composite_score(
            quant_score=70.0, qual_score=70.0, sector_modifier=0.70,
            macro_overlay_enabled=False,
        )
        assert abs(off["macro_shift"]) < 1e-9
        assert abs(off["final_score"] - off["weighted_base"]) < 1e-9

    def test_disabled_keeps_boosts(self):
        # final_score == combined_score == weighted_base + boosts (overlay only is zeroed).
        off = compute_composite_score(
            quant_score=60.0, qual_score=60.0, sector_modifier=1.30,
            boosts={"pead": 5.0}, macro_overlay_enabled=False,
        )
        assert abs(off["macro_shift"]) < 1e-9
        assert abs(off["final_score"] - (off["weighted_base"] + off["total_boost"])) < 1e-9
        assert abs(off["total_boost"] - 5.0) < 1e-9

    # ── compute_composite_breakdown ─────────────────────────────────────
    def _legacy_only_breakdown(self, sector_modifier, macro_overlay_enabled=True, boosts=None):
        return compute_composite_breakdown(
            quant_score=70.0, qual_score=70.0, factor_subscore=None,
            pillar_assessment=None, factor_profile=None,
            sector_modifier=sector_modifier, boosts=boosts,
            macro_overlay_enabled=macro_overlay_enabled,
        )

    def test_breakdown_default_preserves_macro_shift(self):
        on = self._legacy_only_breakdown(1.30)
        assert abs(on.macro_shift - 10.0) < 0.1

    def test_breakdown_disabled_zeroes_macro_shift(self):
        on = self._legacy_only_breakdown(1.30)
        off = self._legacy_only_breakdown(1.30, macro_overlay_enabled=False)
        assert abs(off.macro_shift) < 1e-9
        # final_score == weighted_base + boosts_total + catalyst_modulation;
        # with no boosts / pillars / catalyst → final_score == weighted_base.
        assert abs(off.final_score - off.weighted_base) < 1e-9
        assert off.final_score != on.final_score

    def test_breakdown_disabled_combined_score_equals_final(self):
        # With boosts present, disabled final_score == combined_score
        # (weighted_base + boosts_total), i.e. the macro overlay term drops out.
        off = self._legacy_only_breakdown(0.70, macro_overlay_enabled=False, boosts={"pead": 4.0})
        assert abs(off.macro_shift) < 1e-9
        assert abs(off.final_score - (off.weighted_base + off.boosts_total)) < 1e-9
