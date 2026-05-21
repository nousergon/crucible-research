"""Tests for the pillar-emit gating in qual_analyst — Phase 2 of the
attractiveness-pillars-260520 arc.

Coverage:
- Default-off behavior: with ``PILLAR_EMIT_ENABLED=False`` the second
  pillar extraction never fires and ``pillar_assessments`` is an empty
  dict; legacy QualAnalystOutput extraction path is unchanged.
- On-flag behavior: with ``PILLAR_EMIT_ENABLED=True`` a second extraction
  fires AND its parsed batch is returned keyed by ticker.
- Strict-mode parse failure raises.
- Lax-mode parse failure returns empty dict with a warning.
- Error-branch return shapes consistently include the new
  ``pillar_assessments`` key.
- Prompt-name selection picks ``qual_analyst_system_pillars`` under-flag.

We avoid hitting the live LLM by patching ``create_react_agent`` (for the
ReAct loop) + ``ChatAnthropic`` (for the structured-output extractions).
Mirrors the patching pattern in ``test_sector_team_recursion_partial``.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def reloaded_qual():
    """Force-reload the qual_analyst module so module-level constants
    (PILLAR_EMIT_ENABLED, schema classes) pick up the live config."""
    from agents.sector_teams import qual_analyst
    importlib.reload(qual_analyst)
    yield qual_analyst


def _qual_kwargs():
    return {
        "team_id": "technology",
        "quant_top5": [
            {"ticker": "NVDA", "quant_score": 88, "rationale": "AI tailwind"},
            {"ticker": "AAPL", "quant_score": 75, "rationale": "FCF strong"},
        ],
        "prior_theses": {},
        "market_regime": "neutral",
        "run_date": "2026-05-20",
        "api_key": "test-key",
    }


def _make_react_agent_returning(final_text: str) -> MagicMock:
    """Mock the create_react_agent return value — its invoke() returns a
    state dict containing an AIMessage in messages whose content is
    final_text."""
    from langchain_core.messages import AIMessage

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "messages": [AIMessage(content=final_text)],
    }
    return fake_agent


def _make_chat_anthropic_yielding(
    qual_parsed,
    pillar_parsed=None,
    qual_parse_error=None,
    pillar_parse_error=None,
) -> MagicMock:
    """Mock ChatAnthropic so its ``with_structured_output(...).invoke()``
    returns the supplied parsed payloads.

    First call → QualAnalystOutput-style payload (``qual_parsed``).
    Second call → _QualPillarBatch-style payload (``pillar_parsed``),
    only exercised when PILLAR_EMIT_ENABLED.
    """
    structured_qual = MagicMock()
    structured_qual.invoke.return_value = {
        "parsed": qual_parsed, "parsing_error": qual_parse_error,
    }
    structured_pillar = MagicMock()
    structured_pillar.invoke.return_value = {
        "parsed": pillar_parsed, "parsing_error": pillar_parse_error,
    }

    # with_structured_output is called twice in sequence under-flag, once
    # off-flag. side_effect cycles through.
    llm = MagicMock()
    llm.with_structured_output.side_effect = [structured_qual, structured_pillar]
    return llm


class TestDefaultOffBehavior:
    def test_flag_default_off_no_pillar_extraction(self, reloaded_qual):
        """With PILLAR_EMIT_ENABLED=False, only the legacy extraction runs
        and ``pillar_assessments`` returns as an empty dict."""
        from alpha_engine_lib.agent_schemas import QualAnalystOutput, QualAssessment

        qual_parsed = QualAnalystOutput(
            assessments=[
                QualAssessment(ticker="NVDA", qual_score=80, bull_case="strong"),
                QualAssessment(ticker="AAPL", qual_score=72, bull_case="ok"),
            ]
        )

        fake_agent = _make_react_agent_returning("some analyst reasoning text")
        fake_llm = _make_chat_anthropic_yielding(qual_parsed=qual_parsed)

        with patch.object(reloaded_qual, "PILLAR_EMIT_ENABLED", False), \
             patch.object(reloaded_qual, "create_react_agent", return_value=fake_agent), \
             patch.object(reloaded_qual, "ChatAnthropic", return_value=fake_llm):
            result = reloaded_qual.run_qual_analyst(**_qual_kwargs())

        assert result["error"] is None
        assert result["partial"] is False
        assert len(result["assessments"]) == 2
        assert result["pillar_assessments"] == {}
        # Only one with_structured_output call — the legacy one.
        assert fake_llm.with_structured_output.call_count == 1


class TestOnFlagBehavior:
    def test_flag_on_runs_second_extraction_and_returns_keyed_dict(
        self, reloaded_qual,
    ):
        """With PILLAR_EMIT_ENABLED=True a second extraction fires and
        its parsed batch is converted to a per-ticker dict."""
        from alpha_engine_lib.agent_schemas import QualAnalystOutput, QualAssessment
        from alpha_engine_lib.pillars import (
            MoatAssessment,
            PillarSubscore,
            QualitativePillarAssessment,
        )

        qual_parsed = QualAnalystOutput(
            assessments=[
                QualAssessment(ticker="NVDA", qual_score=80, bull_case="strong"),
            ]
        )

        def _stub_pillar(ticker):
            return QualitativePillarAssessment(
                quality=PillarSubscore(pillar="quality", score=80, confidence="high"),
                quality_moat=MoatAssessment(
                    primary_type="process_power",
                    width="wide",
                    durability_years=20,
                    trend="widening",
                ),
                value=PillarSubscore(pillar="value", score=50, confidence="medium"),
                momentum=PillarSubscore(pillar="momentum", score=75, confidence="medium"),
                growth=PillarSubscore(pillar="growth", score=85, confidence="high"),
                stewardship=PillarSubscore(pillar="stewardship", score=70, confidence="medium"),
                defensiveness=PillarSubscore(pillar="defensiveness", score=55, confidence="medium"),
                catalyst_horizon_modulation=5,
            )

        pillar_parsed = reloaded_qual._QualPillarBatch(
            items=[
                reloaded_qual._QualPillarItem(ticker="NVDA", pillar_assessment=_stub_pillar("NVDA")),
            ]
        )

        fake_agent = _make_react_agent_returning("analyst reasoning text")
        fake_llm = _make_chat_anthropic_yielding(
            qual_parsed=qual_parsed,
            pillar_parsed=pillar_parsed,
        )

        with patch.object(reloaded_qual, "PILLAR_EMIT_ENABLED", True), \
             patch.object(reloaded_qual, "create_react_agent", return_value=fake_agent), \
             patch.object(reloaded_qual, "ChatAnthropic", return_value=fake_llm), \
             patch.object(reloaded_qual, "load_prompt") as mock_load_prompt:
            # Mock the prompt loader to bypass needing the pillars prompt
            # file on disk for the test; we'll verify the *name* passed.
            mock_prompt = MagicMock()
            mock_prompt.format.return_value = "system prompt text"
            mock_prompt.version = "0.1.0"
            mock_prompt.hash = "abcdef123456"
            mock_load_prompt.return_value = mock_prompt

            result = reloaded_qual.run_qual_analyst(**_qual_kwargs())

        assert result["error"] is None
        assert "NVDA" in result["pillar_assessments"]
        nvda = result["pillar_assessments"]["NVDA"]
        # Result is the model_dump() of QualitativePillarAssessment.
        assert nvda["quality"]["score"] == 80
        assert nvda["quality_moat"]["primary_type"] == "process_power"
        assert nvda["catalyst_horizon_modulation"] == 5
        # Both extractions ran.
        assert fake_llm.with_structured_output.call_count == 2

    def test_flag_on_loads_pillar_prompt_name(self, reloaded_qual):
        """Verify ``_build_system_prompt`` picks the pillar prompt name
        when the flag is on. Asserts the load_prompt call args directly
        rather than constructing the full ReAct invocation."""
        with patch.object(reloaded_qual, "PILLAR_EMIT_ENABLED", True), \
             patch.object(reloaded_qual, "load_prompt") as mock_load_prompt:
            mock_prompt = MagicMock()
            mock_prompt.format.return_value = "rendered"
            mock_load_prompt.return_value = mock_prompt

            reloaded_qual._build_system_prompt("technology", "neutral", 5)

        mock_load_prompt.assert_called_once_with("qual_analyst_system_pillars")

    def test_flag_off_loads_legacy_prompt_name(self, reloaded_qual):
        with patch.object(reloaded_qual, "PILLAR_EMIT_ENABLED", False), \
             patch.object(reloaded_qual, "load_prompt") as mock_load_prompt:
            mock_prompt = MagicMock()
            mock_prompt.format.return_value = "rendered"
            mock_load_prompt.return_value = mock_prompt

            reloaded_qual._build_system_prompt("technology", "neutral", 5)

        mock_load_prompt.assert_called_once_with("qual_analyst_system")


class TestParseFailureModes:
    def test_pillar_parse_error_always_raises_regardless_of_lax_mode(
        self, reloaded_qual,
    ):
        """Hardening Item 2 (2026-05-21 AQR cutover incident): the
        pillar-extraction parse failure path now ALWAYS raises,
        REGARDLESS of the STRICT_VALIDATION flag. The prior lax-mode
        return-empty-dict path was the silent-fail class that produced
        the degenerate composite in the AQR cutover incident — see
        [[zero-legacy-weight-degenerates-on-pillar-emit-failure]].

        Run completes with error surfaced on result['error'] (caught by
        run_qual_analyst's outer except). Legacy assessments may still
        be present in result['assessments'] depending on where the
        exception lands relative to the legacy extraction (here: legacy
        succeeded first, pillar extraction then raises, outer except
        catches → assessments populated but error field set)."""
        from alpha_engine_lib.agent_schemas import QualAnalystOutput, QualAssessment

        qual_parsed = QualAnalystOutput(
            assessments=[QualAssessment(ticker="NVDA", qual_score=80)]
        )

        fake_agent = _make_react_agent_returning("analyst reasoning")
        fake_llm = _make_chat_anthropic_yielding(
            qual_parsed=qual_parsed,
            pillar_parse_error=ValueError("malformed JSON"),
        )

        # Even with strict=False (lax mode), pillar parse failure raises.
        with patch.object(reloaded_qual, "PILLAR_EMIT_ENABLED", True), \
             patch.object(reloaded_qual, "is_strict_validation_enabled", return_value=False), \
             patch.object(reloaded_qual, "create_react_agent", return_value=fake_agent), \
             patch.object(reloaded_qual, "ChatAnthropic", return_value=fake_llm), \
             patch.object(reloaded_qual, "load_prompt") as mock_load_prompt:
            mock_prompt = MagicMock()
            mock_prompt.format.return_value = "system prompt"
            mock_prompt.version = "0.1.0"
            mock_prompt.hash = "abcdef123456"
            mock_load_prompt.return_value = mock_prompt
            result = reloaded_qual.run_qual_analyst(**_qual_kwargs())

        # Error surfaced (not silently swallowed); pillar dict empty.
        assert result["error"] is not None
        assert "pillar-assessment parse failed" in result["error"]
        assert result["pillar_assessments"] == {}

    def test_strict_mode_raises_on_pillar_parse_error(self, reloaded_qual):
        """Strict-mode parse failure on the pillar extraction must raise —
        symmetric with the legacy QualAnalystOutput strict-mode contract."""
        from alpha_engine_lib.agent_schemas import QualAnalystOutput, QualAssessment

        qual_parsed = QualAnalystOutput(
            assessments=[QualAssessment(ticker="NVDA", qual_score=80)]
        )

        fake_agent = _make_react_agent_returning("analyst reasoning")
        fake_llm = _make_chat_anthropic_yielding(
            qual_parsed=qual_parsed,
            pillar_parse_error=ValueError("malformed JSON"),
        )

        with patch.object(reloaded_qual, "PILLAR_EMIT_ENABLED", True), \
             patch.object(reloaded_qual, "is_strict_validation_enabled", return_value=True), \
             patch.object(reloaded_qual, "create_react_agent", return_value=fake_agent), \
             patch.object(reloaded_qual, "ChatAnthropic", return_value=fake_llm), \
             patch.object(reloaded_qual, "load_prompt") as mock_load_prompt:
            mock_prompt = MagicMock()
            mock_prompt.format.return_value = "system prompt"
            mock_prompt.version = "0.1.0"
            mock_prompt.hash = "abcdef123456"
            mock_load_prompt.return_value = mock_prompt
            # Strict mode raises through run_qual_analyst's generic catch,
            # so the exception lands in result["error"] (not re-raised).
            # This matches the existing legacy-extraction strict-mode
            # behavior (RuntimeError from the QualAnalystOutput path also
            # lands in error after the generic except).
            result = reloaded_qual.run_qual_analyst(**_qual_kwargs())

        assert result["error"] is not None
        assert "pillar-assessment parse failed" in result["error"]
        assert result["pillar_assessments"] == {}


class TestErrorBranchShape:
    def test_recursion_branch_returns_empty_pillar_dict(self, reloaded_qual):
        """Even on GraphRecursionError the return shape includes
        ``pillar_assessments`` so downstream consumers can rely on the key."""
        from langgraph.errors import GraphRecursionError

        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = GraphRecursionError("limit")

        with patch.object(reloaded_qual, "create_react_agent", return_value=fake_agent), \
             patch.object(reloaded_qual, "load_prompt") as mock_load_prompt:
            mock_prompt = MagicMock()
            mock_prompt.format.return_value = "system prompt"
            mock_prompt.version = "0.1.0"
            mock_prompt.hash = "abcdef123456"
            mock_load_prompt.return_value = mock_prompt
            result = reloaded_qual.run_qual_analyst(**_qual_kwargs())

        assert result["partial"] is True
        assert result["pillar_assessments"] == {}

    def test_generic_exception_returns_empty_pillar_dict(self, reloaded_qual):
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = RuntimeError("kaboom")

        with patch.object(reloaded_qual, "create_react_agent", return_value=fake_agent), \
             patch.object(reloaded_qual, "load_prompt") as mock_load_prompt:
            mock_prompt = MagicMock()
            mock_prompt.format.return_value = "system prompt"
            mock_prompt.version = "0.1.0"
            mock_prompt.hash = "abcdef123456"
            mock_load_prompt.return_value = mock_prompt
            result = reloaded_qual.run_qual_analyst(**_qual_kwargs())

        assert result["error"] is not None
        assert result["pillar_assessments"] == {}
