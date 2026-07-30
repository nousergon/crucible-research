"""Tests for the Rung-1 extraction agent (alpha-engine-config#3080).

Covers:
- Single-ticker extraction with mock LLM client
- Full-length prompt rendering
- Filings context retrieval wrapper
- Batch extraction error handling (fail-soft)
- Default model arm selection
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.sector_teams.rung1_extraction import (
    EXTRACTION_MODEL_ARMS,
    ExtractionOutput,
    build_extraction_prompt,
    extract_single_ticker,
    extract_triggered_tickers,
    retrieve_filings_context,
)


class TestModelArms:
    """Model sub-arm registry integrity."""

    def test_has_four_arms(self):
        assert len(EXTRACTION_MODEL_ARMS) == 4

    def test_floor_is_deepseek_flash(self):
        assert "deepseek/deepseek-v4-flash" in EXTRACTION_MODEL_ARMS.values()

    def test_ceiling_is_kimi(self):
        assert "moonshotai/kimi-k2.6" in EXTRACTION_MODEL_ARMS.values()

    def test_default_arm_is_floor(self):
        from agents.sector_teams.rung1_extraction import DEFAULT_MODEL_ARM
        assert DEFAULT_MODEL_ARM == "floor"


class TestPromptBuilding:
    """Prompt rendering for single-shot extraction."""

    def test_basic_prompt_shape(self):
        prompt = build_extraction_prompt("AAPL", ["news_volume_spike"], "Some filing context here.")
        assert "AAPL" in prompt
        assert "news_volume_spike" in prompt
        assert "guidance_direction" in prompt
        assert "risk_factor_count_delta_raw" in prompt
        assert "management_tone_zscore" in prompt
        assert "Some filing context here" in prompt

    def test_empty_triggers_uses_periodic_refresh(self):
        prompt = build_extraction_prompt("MSFT", [], "Context.")
        assert "periodic refresh" in prompt

    def test_multiple_triggers_all_listed(self):
        triggers = ["news_volume_spike", "earnings_proximity", "insider_cluster"]
        prompt = build_extraction_prompt("GOOGL", triggers, "Context.")
        for t in triggers:
            assert t in prompt


class TestExtractionOutputSchema:
    """Pydantic schema validation for the extraction output."""

    def test_valid_output(self):
        result = ExtractionOutput(
            guidance_direction="raised",
            risk_factor_count_delta_raw=3,
            management_tone_zscore=1.25,
        )
        assert result.guidance_direction == "raised"
        assert result.risk_factor_count_delta_raw == 3
        assert result.management_tone_zscore == 1.25

    def test_none_guidance(self):
        result = ExtractionOutput(
            guidance_direction="none",
            risk_factor_count_delta_raw=0,
            management_tone_zscore=0.0,
        )
        assert result.guidance_direction == "none"

    def test_invalid_negative_tone_is_allowed(self):
        """Z-score can be negative (pessimistic tone)."""
        result = ExtractionOutput(
            guidance_direction="lowered",
            risk_factor_count_delta_raw=-2,
            management_tone_zscore=-1.5,
        )
        assert result.management_tone_zscore == -1.5
        assert result.risk_factor_count_delta_raw == -2

    def test_rejects_invalid_guidance_type(self):
        with pytest.raises(ValueError):
            ExtractionOutput(
                guidance_direction="something_else",  # not in expected set, but Pydantic allows str
                risk_factor_count_delta_raw=0,
                management_tone_zscore=0.0,
            )

    def test_serializes_to_json(self):
        result = ExtractionOutput(
            guidance_direction="maintained",
            risk_factor_count_delta_raw=1,
            management_tone_zscore=-0.3,
        )
        data = json.loads(result.model_dump_json())
        assert data["guidance_direction"] == "maintained"
        assert data["risk_factor_count_delta_raw"] == 1
        assert data["management_tone_zscore"] == -0.3


class TestExtractSingleTicker:
    """Single-ticker extraction with a mock krepis client."""

    def _mock_client_factory(self, return_value: ExtractionOutput):
        """Create a mock LLMClient factory that returns the given output."""
        mock_client = MagicMock()
        mock_client.structured.return_value = return_value
        return lambda spec, api_key: mock_client

    def test_successful_extraction(self):
        factory = self._mock_client_factory(
            ExtractionOutput(guidance_direction="raised", risk_factor_count_delta_raw=2, management_tone_zscore=0.8)
        )
        result = extract_single_ticker(
            "AAPL", ["recent_earnings"], "Earnings transcript context.",
            model_arm="floor", api_key="test-key", client_factory=factory,
        )
        assert result.guidance_direction == "raised"
        assert result.risk_factor_count_delta_raw == 2
        assert result.management_tone_zscore == 0.8

    def test_unknown_model_arm_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            extract_single_ticker(
                "AAPL", [], "Context.", model_arm="nonexistent",
                api_key="test-key",
            )

    def test_missing_api_key_raises(self):
        with patch("agents.sector_teams.rung1_extraction.OPENROUTER_API_KEY", None):
            with pytest.raises(RuntimeError, match="OpenRouter API key"):
                extract_single_ticker("AAPL", [], "Context.")


class TestExtractTriggeredTickers:
    """Batch extraction with fail-soft semantics."""

    def test_empty_input_returns_empty(self):
        results = extract_triggered_tickers([], api_key="test-key", rag_fn=lambda t, q: "Context.")
        assert results == []

    def test_single_ticker(self):
        tickers = [("AAPL", ["news_volume_spike"])]
        from agents.sector_teams.rung1_extraction import retrieve_filings_context
        # Patch both the LLM call and the RAG retrieval
        with patch(
            "agents.sector_teams.rung1_extraction.extract_single_ticker",
            return_value=ExtractionOutput(guidance_direction="maintained", risk_factor_count_delta_raw=0, management_tone_zscore=0.0),
        ):
            with patch(
                "agents.sector_teams.rung1_extraction.retrieve_filings_context",
                return_value="Mocked context.",
            ):
                results = extract_triggered_tickers(tickers, api_key="test-key")
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["guidance_direction"] == "maintained"

    def test_fail_soft_on_extraction_error(self):
        tickers = [("FAIL", ["trigger"])]
        with patch(
            "agents.sector_teams.rung1_extraction.extract_single_ticker",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            with patch(
                "agents.sector_teams.rung1_extraction.retrieve_filings_context",
                return_value="Mocked context.",
            ):
                results = extract_triggered_tickers(tickers, api_key="test-key")
        assert len(results) == 1
        assert results[0]["ticker"] == "FAIL"
        assert results[0]["guidance_direction"] == "none"  # fail-soft default
        assert "error" in results[0]


class TestRetrieveFilingsContext:
    """RAG retrieval wrapper for extraction context."""

    @patch("nousergon_lib.rag.retrieve")
    def test_returns_context_when_results_found(self, mock_retrieve):
        mock_result = MagicMock()
        mock_result.source = "10-K/AAPL/2025"
        mock_result.text = "Sample filing text about risk factors."
        mock_retrieve.return_value = [mock_result]

        context = retrieve_filings_context("AAPL")
        assert "10-K/AAPL/2025" in context
        assert "risk factors" in context

    @patch("nousergon_lib.rag.retrieve", return_value=[])
    def test_no_results_message(self, mock_retrieve):
        context = retrieve_filings_context("UNKNOWN")
        assert "No filing data found" in context

    @patch("nousergon_lib.rag.retrieve", side_effect=Exception("DB timeout"))
    def test_graceful_retrieval_failure(self, mock_retrieve):
        context = retrieve_filings_context("AAPL")
        assert "unavailable" in context
        assert "AAPL" in context
