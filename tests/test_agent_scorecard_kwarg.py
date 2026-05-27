"""Tests for Phase 2.A.2 — prior_cycle_scorecard kwarg passthrough.

This PR only changes the agent function signatures + `.format()` kwarg
passthrough. No graph wiring yet. The kwarg is silently unused by
`str.format` until Brian's gitignored prompt-template edit adds the
`{prior_cycle_scorecard}` placeholder. These tests verify:

1. The new kwarg is ACCEPTED by both agent entry points (signature change
   doesn't break callers).
2. The kwarg's value FLOWS THROUGH to the `.format()` call (when the
   template DOES carry the placeholder, the scorecard text lands in the
   rendered prompt).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCIOPromptSignature:
    def test_build_cio_prompt_accepts_prior_cycle_scorecard(self):
        from agents.investment_committee.ic_cio import _build_cio_prompt
        sig = inspect.signature(_build_cio_prompt)
        assert "prior_cycle_scorecard" in sig.parameters
        # Optional with default None — caller doesn't need to know about it.
        assert sig.parameters["prior_cycle_scorecard"].default is None

    def test_build_cio_prompt_threads_scorecard_into_format(self):
        from agents.investment_committee import ic_cio
        from agents.prompt_loader import LoadedPrompt

        # Stub prompt template carrying the placeholder so we can observe
        # the kwarg flowed all the way through .format().
        stub_text = "scorecard={prior_cycle_scorecard}|other"
        stub = LoadedPrompt(
            name="ic_cio_evaluation",
            version="0.0.0",
            hash="x",
            text=stub_text + "|{run_date}|{regime}|{open_slots}|{ratings_text}"
                 + "|{pop_text}|{exit_text}|{prior_decisions_block}|{candidates_text}",
            source_path=Path("."),
        )

        with patch("agents.investment_committee.ic_cio.load_prompt", return_value=stub):
            rendered = ic_cio._build_cio_prompt(
                candidates=[],
                macro_context={"market_regime": "neutral"},
                sector_ratings={},
                population=[],
                open_slots=3,
                exits=[],
                run_date="2026-05-23",
                prior_decisions=[],
                prior_cycle_scorecard="SCORECARD TEXT GOES HERE",
            )
        assert "scorecard=SCORECARD TEXT GOES HERE" in rendered

    def test_build_cio_prompt_defaults_to_empty_string(self):
        from agents.investment_committee import ic_cio
        from agents.prompt_loader import LoadedPrompt

        stub_text = "scorecard=[{prior_cycle_scorecard}]|x"
        stub = LoadedPrompt(
            name="ic_cio_evaluation",
            version="0.0.0",
            hash="x",
            text=stub_text + "|{run_date}|{regime}|{open_slots}|{ratings_text}"
                 + "|{pop_text}|{exit_text}|{prior_decisions_block}|{candidates_text}",
            source_path=Path("."),
        )
        with patch("agents.investment_committee.ic_cio.load_prompt", return_value=stub):
            rendered = ic_cio._build_cio_prompt(
                candidates=[],
                macro_context={"market_regime": "neutral"},
                sector_ratings={},
                population=[],
                open_slots=3,
                exits=[],
                run_date="2026-05-23",
                # prior_cycle_scorecard intentionally omitted — default None.
            )
        # `None or ""` → empty string in the rendered prompt.
        assert "scorecard=[]" in rendered


class TestMacroAgentSignature:
    def test_run_macro_agent_accepts_prior_cycle_scorecard(self):
        from agents.macro_agent import run_macro_agent
        sig = inspect.signature(run_macro_agent)
        assert "prior_cycle_scorecard" in sig.parameters
        assert sig.parameters["prior_cycle_scorecard"].default is None

    def test_run_macro_agent_with_reflection_accepts_prior_cycle_scorecard(self):
        from agents.macro_agent import run_macro_agent_with_reflection
        sig = inspect.signature(run_macro_agent_with_reflection)
        assert "prior_cycle_scorecard" in sig.parameters
        assert sig.parameters["prior_cycle_scorecard"].default is None

    def test_run_macro_agent_with_reflection_threads_kwarg_to_inner_call(self):
        """When the reflection wrapper is called with prior_cycle_scorecard,
        every inner run_macro_agent call must receive the same value.

        Verifies the wrapper doesn't drop the kwarg en route to the agent
        function — important since the wrapper is what the graph
        orchestrator actually calls.
        """
        from agents import macro_agent

        captured: list = []

        def _fake_run_macro_agent(**kwargs):
            captured.append(kwargs.get("prior_cycle_scorecard"))
            # Minimum return shape the wrapper expects so the loop exits cleanly.
            return {
                "report_md": "stub",
                "macro_json": {},
                "market_regime": "neutral",
                "sector_modifiers": {},
                "sector_ratings": {},
            }

        with patch.object(macro_agent, "run_macro_agent", side_effect=_fake_run_macro_agent):
            # max_iterations=1 keeps the loop tight (just the initial call).
            macro_agent.run_macro_agent_with_reflection(
                prior_report=None,
                prior_date="2026-05-23",
                macro_data={},
                max_iterations=1,
                prior_cycle_scorecard="SCORECARD TEXT GOES HERE",
            )

        # At least one inner call must have received the value verbatim.
        assert any(c == "SCORECARD TEXT GOES HERE" for c in captured), (
            f"prior_cycle_scorecard not threaded through to inner call. Captured: {captured}"
        )
