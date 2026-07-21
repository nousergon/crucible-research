"""
Unit tests for the OpenRouter shadow-judge tier (config#2575 items 2-3).

Mirrors ``tests/test_eval_judge.py``'s mocking style
(``TestEvaluateArtifact``) but mocks the ``openai.OpenAI`` client instead
of ``ChatAnthropic`` — ``evaluate_artifact_openrouter`` goes through
``krepis.llm``'s OpenAI-compatible transport, not LangChain.

The leak-guard's live-API validation (against a REAL OpenRouter call, not
a mock) is a separate, manually-run script
(``scripts/live_validate_openrouter_judge.py``) — this suite covers the
harness logic (retry bookkeeping, RubricEvalArtifact wrapping, skip
gates, distinct leak-guard logging) with mocks, same division of labor
the module docstring for ``krepis.judge`` describes for its own tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from evals.judge_models import OPENROUTER_SHADOW
from graph.state_schemas import RubricEvalArtifact

from tests.test_eval_judge import _make_artifact, _make_llm_output


def _openai_tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _openai_response(
    *, finish_reason: str, tool_calls=None, content=None, model="deepseek/deepseek-v4-flash",
    cost: float = 0.0001,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    usage = SimpleNamespace(cost=cost)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _valid_tool_args() -> dict:
    llm_output = _make_llm_output()
    return llm_output.model_dump()


class TestEvaluateArtifactOpenRouter:
    def test_unmapped_agent_raises(self):
        from evals.judge import evaluate_artifact_openrouter
        artifact = _make_artifact("totally_made_up_agent")
        with pytest.raises(ValueError, match="No rubric mapped"):
            evaluate_artifact_openrouter(artifact, api_key="sk-or-test")

    def test_full_pipeline_with_mocked_client(self, monkeypatch):
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="tool_calls",
            tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
        )

        with patch.object(judge_mod, "OpenAI", return_value=fake_client) as mock_openai_cls:
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact_openrouter(
                artifact,
                api_key="sk-or-test",
                judged_artifact_s3_key="decision_artifacts/2026/05/09/sector_quant:technology/r1.json",
            )

        assert isinstance(result, RubricEvalArtifact)
        assert result.judged_agent_id == "sector_quant:technology"
        assert result.judge_model == OPENROUTER_SHADOW.logical_key
        assert result.judge_request_model == OPENROUTER_SHADOW.request_model
        assert result.judge_resolved_model == "deepseek/deepseek-v4-flash"
        assert len(result.dimension_scores) == 6
        assert result.overall_reasoning == "Solid grounding; regime engagement weakest."
        # Only one call — first attempt succeeded.
        assert fake_client.chat.completions.create.call_count == 1
        # base_url must resolve to the OpenRouter endpoint.
        mock_openai_cls.assert_called_once()
        assert mock_openai_cls.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert mock_openai_cls.call_args.kwargs["api_key"] == "sk-or-test"
        # reasoning={"exclude": True} is forwarded per the documented
        # truncation-avoidance default.
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {"reasoning": {"exclude": True}}

    def test_leak_guard_trips_then_recovers_on_retry(self):
        """Reproduces the live-confirmed truncation shape on attempt 1
        (finish_reason=length, no tool_calls), then a clean structured
        call on attempt 2 — the guard must not fail the whole call, just
        consume a retry (same posture as ordinary schema non-conformance)."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _openai_response(finish_reason="length", tool_calls=None, content=None),
            _openai_response(
                finish_reason="tool_calls",
                tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
            ),
        ]

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact_openrouter(artifact, api_key="sk-or-test")

        assert isinstance(result, RubricEvalArtifact)
        assert fake_client.chat.completions.create.call_count == 2

    def test_leak_guard_exhaustion_raises_runtime_error(self):
        """Every attempt trips the leak guard — must fail loud (fail-loud
        contract mirrors evaluate_artifact's MAX_JUDGE_RETRIES exhaustion),
        not silently emit a garbage/empty eval."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="length", tool_calls=None, content=None,
        )

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError, match="attempts failed"):
                judge_mod.evaluate_artifact_openrouter(
                    artifact, api_key="sk-or-test", max_retries=3,
                )
        assert fake_client.chat.completions.create.call_count == 3

    def test_control_token_leak_logged_distinctly(self, caplog):
        """The control-token-leak class must be logged with the
        DISTINCT 'leak_guard_triggered' marker (config#2575 item 3's
        near-miss-visibility requirement) — not folded into a generic
        'attempt failed' line indistinguishable from ordinary schema
        non-conformance."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _openai_response(
                finish_reason="stop",
                content="<|tool_calls_section_begin|>garbage",
                tool_calls=None,
            ),
            _openai_response(
                finish_reason="tool_calls",
                tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
            ),
        ]

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            with caplog.at_level("WARNING"):
                judge_mod.evaluate_artifact_openrouter(artifact, api_key="sk-or-test")

        assert any("leak_guard_triggered" in rec.message for rec in caplog.records)
        assert any("control_token_leak" in rec.message for rec in caplog.records)

    def test_no_tool_call_at_all_retries_as_ordinary_failure(self):
        """finish_reason='stop' with no tool_calls and no leak-signature
        content is NOT a leak-guard case — it's an ordinary "model didn't
        call the tool" failure, retried through the normal path."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _openai_response(finish_reason="stop", tool_calls=None, content="I refuse."),
            _openai_response(
                finish_reason="tool_calls",
                tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
            ),
        ]
        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact_openrouter(artifact, api_key="sk-or-test")
        assert isinstance(result, RubricEvalArtifact)

    def test_missing_api_key_raises(self, monkeypatch):
        from evals import judge as judge_mod

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        artifact = _make_artifact("sector_quant:technology")
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            judge_mod.evaluate_artifact_openrouter(artifact, api_key=None)

    def test_empty_agent_output_short_circuits_without_api_call(self):
        from evals import judge as judge_mod

        fake_client = MagicMock()
        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            artifact.agent_output = {}
            result = judge_mod.evaluate_artifact_openrouter(artifact, api_key="sk-or-test")

        assert result.judge_skip_reason == "precluded_by_empty_upstream"
        fake_client.chat.completions.create.assert_not_called()

    def test_default_judge_model_is_openrouter_shadow_logical_key(self):
        import inspect
        from evals.judge import evaluate_artifact_openrouter

        sig = inspect.signature(evaluate_artifact_openrouter)
        assert sig.parameters["judge_model"].default == OPENROUTER_SHADOW.logical_key
