"""Tests for the Rung 1 extraction agent (alpha-engine-config#3080).

Tests are unit-level and mock-free for the core logic (schema parsing,
grounding gate, cost estimation). The OpenRouter call path is covered by
integration tests that require OPENROUTER_API_KEY.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from agents.sector_teams.extraction_agent import (
    EXTRACTION_MODEL_ARMS,
    ExtractionFeatures,
    ExtractionRunRecord,
    _estimate_cost,
    check_grounding_threshold,
)


class TestExtractionFeatures:
    def test_valid_guidance_directions(self):
        """All four valid guidance directions should construct."""
        for direction in ("raised", "lowered", "maintained", "none"):
            features = ExtractionFeatures(
                guidance_direction=direction,
                risk_factor_count_delta_raw=0,
                management_tone_zscore=0.0,
            )
            assert features.guidance_direction == direction

    def test_invalid_guidance_direction_rejected(self):
        """An invalid guidance direction should be rejected by the model."""
        with pytest.raises(ValueError) as exc:
            ExtractionFeatures(
                guidance_direction="invalid_option",
                risk_factor_count_delta_raw=0,
                management_tone_zscore=0.0,
            )
        assert "guidance_direction" in str(exc.value).lower()

    def test_risk_delta_accepts_positive_negative_zero(self):
        for val in (5, -3, 0, 100, -100):
            f = ExtractionFeatures(
                guidance_direction="raised",
                risk_factor_count_delta_raw=val,
                management_tone_zscore=0.0,
            )
            assert f.risk_factor_count_delta_raw == val

    def test_tone_zscore_accepts_range(self):
        for val in (-3.0, -1.5, 0.0, 1.5, 3.0):
            f = ExtractionFeatures(
                guidance_direction="maintained",
                risk_factor_count_delta_raw=0,
                management_tone_zscore=val,
            )
            assert f.management_tone_zscore == val

    def test_json_schema_roundtrip(self):
        features = ExtractionFeatures(
            guidance_direction="raised",
            risk_factor_count_delta_raw=3,
            management_tone_zscore=1.25,
        )
        raw = json.loads(features.model_dump_json())
        restored = ExtractionFeatures(**raw)
        assert restored == features


class TestCheckGroundingThreshold:
    def test_passes_with_source_chunks(self):
        features = ExtractionFeatures(
            guidance_direction="raised",
            risk_factor_count_delta_raw=0,
            management_tone_zscore=0.0,
        )
        chunks = [{"source": "10-Q", "text": "Revenue guidance increased 15%"}]
        passed, score = check_grounding_threshold(features, chunks, threshold=0.7)
        assert passed
        assert score >= 0.7

    def test_fails_without_source_chunks(self):
        features = ExtractionFeatures(
            guidance_direction="none",
            risk_factor_count_delta_raw=0,
            management_tone_zscore=0.0,
        )
        passed, score = check_grounding_threshold(features, [], threshold=0.7)
        assert not passed
        assert score == 0.0

    def test_passes_with_custom_threshold(self):
        features = ExtractionFeatures(
            guidance_direction="lowered",
            risk_factor_count_delta_raw=0,
            management_tone_zscore=-1.0,
        )
        chunks = [{"source": "8-K", "text": "Guidance lowered"}]
        # Threshold of 0.0 — everything passes
        passed, score = check_grounding_threshold(features, chunks, threshold=0.0)
        assert passed


class TestExtractionModelArms:
    def test_all_arms_have_required_keys(self):
        for arm in EXTRACTION_MODEL_ARMS:
            assert "name" in arm
            assert "model" in arm
            assert "cost_per_1k_in" in arm
            assert "cost_per_1k_out" in arm
            assert arm["cost_per_1k_in"] > 0
            assert arm["cost_per_1k_out"] > 0

    def test_arm_names_are_unique(self):
        names = [a["name"] for a in EXTRACTION_MODEL_ARMS]
        assert len(names) == len(set(names))

    def test_arms_ordered_by_cost(self):
        """Arms should be listed in roughly increasing cost order."""
        costs = [a["cost_per_1k_in"] + a["cost_per_1k_out"] for a in EXTRACTION_MODEL_ARMS]
        # floor-stretch (gpt-oss-120b) is cheaper than floor (deepseek-v4-flash),
        # but both are cheaper than the mid and ceiling arms.
        assert costs[1] <= costs[2]  # floor-stretch <= mid
        assert costs[2] <= costs[3]  # mid <= ceiling-ref


class TestCostEstimation:
    def test_zero_tokens_zero_cost(self):
        cost = _estimate_cost(EXTRACTION_MODEL_ARMS[0], 0, 0)
        assert cost == 0.0

    def test_proportional_to_tokens(self):
        base = _estimate_cost(EXTRACTION_MODEL_ARMS[0], 1000, 500)
        doubled = _estimate_cost(EXTRACTION_MODEL_ARMS[0], 2000, 1000)
        assert abs(doubled - 2 * base) < 0.001

    def test_floor_cheaper_than_ceiling(self):
        floor_cost = _estimate_cost(EXTRACTION_MODEL_ARMS[0], 1000, 500)
        ceiling_cost = _estimate_cost(EXTRACTION_MODEL_ARMS[3], 1000, 500)
        assert floor_cost < ceiling_cost


class TestExtractionRunRecord:
    def test_record_constructs_with_minimal_fields(self):
        record = ExtractionRunRecord(
            ticker="AAPL",
            model_arm="floor",
            features=ExtractionFeatures(
                guidance_direction="raised",
                risk_factor_count_delta_raw=2,
                management_tone_zscore=1.5,
            ),
            grounding_score=0.85,
            grounding_passed=True,
            token_count_input=500,
            token_count_output=150,
            cost_usd=0.002,
            duration_ms=1200,
            triggered_by=["news_volume_spike"],
            run_date="2026-07-30",
        )
        assert record.ticker == "AAPL"
        assert record.features.guidance_direction == "raised"
        assert record.grounding_passed
