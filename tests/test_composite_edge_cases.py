"""Edge case tests for composite scoring."""

from scoring.composite import compute_composite_score, normalize_conviction, score_to_rating


class TestCompositeEdgeCases:
    def test_extreme_bullish_modifier(self):
        result = compute_composite_score(quant_score=50, qual_score=50, sector_modifier=1.5)
        assert result["macro_shift"] > 10  # beyond normal range
        assert result["final_score"] <= 100

    def test_extreme_bearish_modifier(self):
        result = compute_composite_score(quant_score=50, qual_score=50, sector_modifier=0.5)
        assert result["macro_shift"] < -10
        assert result["final_score"] >= 0

    def test_all_negative_boosts(self):
        result = compute_composite_score(
            quant_score=70, qual_score=70, sector_modifier=1.0,
            boosts={"pead": -5, "revision": -3, "options": -4},
        )
        assert result["total_boost"] == -10.0  # capped at -10
        assert result["final_score"] == 60.0

    def test_single_score_with_boosts(self):
        result = compute_composite_score(
            quant_score=60, qual_score=None, sector_modifier=1.0,
            boosts={"pead": 5},
        )
        assert result["final_score"] == 65.0
        assert result["score_failed"] is False

    def test_zero_scores(self):
        result = compute_composite_score(quant_score=0, qual_score=0, sector_modifier=1.0)
        assert result["final_score"] == 0.0

    def test_perfect_scores(self):
        result = compute_composite_score(quant_score=100, qual_score=100, sector_modifier=1.0)
        assert result["final_score"] == 100.0


class TestNormalizeConviction:
    def test_valid_passthrough(self):
        assert normalize_conviction("rising") == "rising"
        assert normalize_conviction("stable") == "stable"
        assert normalize_conviction("declining") == "declining"

    def test_legacy_qual_strings_default_to_stable(self):
        # Post-Option-A 2026-04-30: agent format is uniformly int 0-100; the
        # legacy "high"/"medium"/"low" branch is retired. These strings now
        # fall through to the unknown-string default of "stable".
        assert normalize_conviction("high") == "stable"
        assert normalize_conviction("medium") == "stable"
        assert normalize_conviction("low") == "stable"

    def test_numeric_mapping(self):
        assert normalize_conviction(85) == "rising"
        assert normalize_conviction(55) == "stable"
        assert normalize_conviction(20) == "declining"

    def test_none_defaults_stable(self):
        assert normalize_conviction(None) == "stable"

    def test_sentence_defaults_stable(self):
        assert normalize_conviction("I am quite confident in this pick") == "stable"

    def test_case_insensitive(self):
        assert normalize_conviction("RISING") == "rising"
        assert normalize_conviction("Declining") == "declining"

    def test_numeric_boundary_values(self):
        """Boundary values at 70 and 40 thresholds."""
        assert normalize_conviction(70) == "rising"
        assert normalize_conviction(69.9) == "stable"
        assert normalize_conviction(40) == "stable"
        assert normalize_conviction(39.9) == "declining"
        assert normalize_conviction(70.0) == "rising"
        assert normalize_conviction(0) == "declining"
        assert normalize_conviction(100) == "rising"

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped on storage labels."""
        assert normalize_conviction("  rising  ") == "rising"
        assert normalize_conviction("\tstable\n") == "stable"

    def test_mixed_case_storage_labels(self):
        """Storage-format labels are case-insensitive."""
        assert normalize_conviction("Rising") == "rising"
        assert normalize_conviction("STABLE") == "stable"
        assert normalize_conviction("DECLINING") == "declining"

    def test_empty_string_defaults_stable(self):
        assert normalize_conviction("") == "stable"

    def test_cio_prose_defaults_stable(self):
        """Original bug: CIO returning prose instead of enum."""
        assert normalize_conviction("Best quant+qual in tech cohort") == "stable"
        assert normalize_conviction("Strong momentum with rising analyst consensus") == "stable"


class TestScoreToRating:
    def test_none_returns_hold(self):
        assert score_to_rating(None) == "HOLD"

    def test_custom_thresholds(self):
        assert score_to_rating(65, buy_threshold=60) == "BUY"
        assert score_to_rating(65, buy_threshold=70) == "HOLD"

    def test_boundary_values(self):
        assert score_to_rating(70.0) == "BUY"
        assert score_to_rating(69.9) == "HOLD"
        assert score_to_rating(40.0) == "SELL"
        assert score_to_rating(40.1) == "HOLD"
