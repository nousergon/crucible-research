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

    def test_bull_regime_raises_overbought_threshold(self):
        # RSI=75 should not be overbought in bull regime
        bull_score = _score_rsi(75, "bull")
        neutral_score = _score_rsi(75, "neutral")
        assert bull_score > neutral_score

    def test_bear_regime_raises_oversold_threshold(self):
        # RSI=35 should not be as bullish in bear regime
        bear_score = _score_rsi(35, "bear")
        neutral_score = _score_rsi(35, "neutral")
        assert bear_score < neutral_score

    def test_score_in_valid_range(self):
        for rsi in [0, 15, 30, 50, 70, 85, 100]:
            for regime in ["bull", "neutral", "caution", "bear"]:
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
        assert score_to_rating(35) == "SELL"
