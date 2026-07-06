"""Unit tests for the Anthropic Message Batches API path (ROADMAP §1642).

Covers the helpers that translate captured artifacts into batch
requests + parse batch results back into the existing ``RubricEvalArtifact``
schema. The real Anthropic SDK is never called — these tests pin the
shape contracts the Submit/Process Lambdas will produce/consume.

The end-to-end orchestrator-level test (``build_batch_plan`` →
``submit_batch`` mock → ``process_batch_results``) lives in
``test_eval_orchestrator_batch.py``.
"""

from __future__ import annotations

import pytest

from nousergon_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)
from graph.state_schemas import RubricDimensionScore, RubricEvalLLMOutput


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_artifact(
    agent_id: str = "sector_quant:technology",
    *,
    run_id: str = "test-run-001",
    agent_output: dict | None = None,
) -> DecisionArtifact:
    return DecisionArtifact(
        run_id=run_id,
        timestamp="2026-05-09T22:30:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
        full_prompt_context=FullPromptContext(
            system_prompt="<see config/prompts>",
            user_prompt="<rendered at run time>",
        ),
        input_data_snapshot={"team_id": "technology"},
        input_data_summary="team_id=technology",
        agent_output=agent_output if agent_output is not None else {
            "ranked_picks": [{"ticker": "AAPL", "quant_score": 70}],
        },
    )


# ── Custom-id codec ───────────────────────────────────────────────────────


class TestCustomIdCodec:
    def test_round_trip_short_agent_id(self):
        from evals.judge import encode_custom_id, decode_custom_id

        cid = encode_custom_id(
            judged_agent_id="ic_cio",
            run_id="run-1",
            judge_model="claude-haiku-4-5",
        )
        agent, run, model = decode_custom_id(cid)
        assert run == "run-1"
        # ``-`` is the only round-trip-stable character; `:` was
        # mapped to `-` on encode (decode returns the sanitized form).
        assert agent == "ic_cio"
        assert model == "claude-haiku-4-5"

    def test_round_trip_sonnet_model(self):
        from evals.judge import encode_custom_id, decode_custom_id

        cid = encode_custom_id(
            judged_agent_id="macro_economist",
            run_id="run-1",
            judge_model="claude-sonnet-4-6",
        )
        _, _, model = decode_custom_id(cid)
        assert model == "claude-sonnet-4-6"

    def test_long_agent_id_truncates_under_64_char_cap(self):
        """Anthropic's custom_id regex enforces ``^[a-zA-Z0-9_-]{1,64}$``;
        the longest agent_id in the wild today is
        ``thesis_update:technology:NVDA`` (~30 chars) but a defensive
        truncation guard matters for future agent_ids that could overflow."""
        from evals.judge import encode_custom_id, _CUSTOM_ID_PATTERN

        long_agent = "thesis_update:technology:VERYLONGCOMPANYNAMEHERE_ABCDEFGHIJKL"
        cid = encode_custom_id(
            judged_agent_id=long_agent,
            run_id="run-with-medium-length-id",
            judge_model="claude-haiku-4-5",
        )
        assert _CUSTOM_ID_PATTERN.match(cid), (
            f"encoded custom_id {cid!r} violates Anthropic's "
            f"^[a-zA-Z0-9_-]{{1,64}}$ contract"
        )
        assert len(cid) <= 64

    def test_unique_per_judge_model_for_same_artifact(self):
        """First-Saturday submits two requests per artifact (Haiku +
        Sonnet) — their custom_ids must differ so the Process Lambda
        can join each result back to its tier."""
        from evals.judge import encode_custom_id

        cid_h = encode_custom_id(
            judged_agent_id="ic_cio", run_id="run-1",
            judge_model="claude-haiku-4-5",
        )
        cid_s = encode_custom_id(
            judged_agent_id="ic_cio", run_id="run-1",
            judge_model="claude-sonnet-4-6",
        )
        assert cid_h != cid_s

    def test_decode_rejects_malformed_custom_id(self):
        from evals.judge import decode_custom_id

        with pytest.raises(ValueError, match="three '__'-separated"):
            decode_custom_id("not_three_segments")


# ── build_batch_request ───────────────────────────────────────────────────


class TestBuildBatchRequest:
    def test_returns_anthropic_batch_request_shape(self):
        from evals.judge import build_batch_request, encode_custom_id

        artifact = _make_artifact("sector_quant:technology")
        cid = encode_custom_id(
            judged_agent_id=artifact.agent_id,
            run_id=artifact.run_id,
            judge_model="claude-haiku-4-5",
        )
        req = build_batch_request(
            artifact, judge_model="claude-haiku-4-5", custom_id=cid,
        )

        assert req["custom_id"] == cid
        params = req["params"]
        # The API request is pinned to the dated snapshot (L4578(a)); the
        # logical key 'claude-haiku-4-5' lives on the custom_id, not here.
        assert params["model"] == "claude-haiku-4-5-20251001"
        assert "max_tokens" in params
        # Tool use is set up so the LLM is forced to call the rubric
        # tool — same posture as ``with_structured_output`` on the sync
        # path.
        assert isinstance(params["tools"], list)
        assert len(params["tools"]) == 1
        tool = params["tools"][0]
        assert tool["name"] == "RubricEvalLLMOutput"
        assert "input_schema" in tool
        # tool_choice forces the rubric tool — prevents prose fallthrough.
        assert params["tool_choice"] == {
            "type": "tool", "name": "RubricEvalLLMOutput",
        }
        # User message carries the rendered rubric prompt.
        assert params["messages"][0]["role"] == "user"
        assert isinstance(params["messages"][0]["content"], str)

    def test_unmapped_agent_id_raises_value_error(self):
        from evals.judge import build_batch_request

        artifact = _make_artifact("totally_unknown_agent")
        with pytest.raises(ValueError, match="No rubric mapped"):
            build_batch_request(
                artifact,
                judge_model="claude-haiku-4-5",
                custom_id="ic_cio__test__h45",
            )

    def test_tool_input_schema_pinned_to_pydantic_model(self):
        """Schema bumps to RubricEvalLLMOutput must auto-flow into the
        batch tool spec — no second source of truth. Pinning by
        comparing the rendered tool schema against the model's own
        ``model_json_schema()`` proves the indirection is intact."""
        from evals.judge import build_batch_request, encode_custom_id

        artifact = _make_artifact("ic_cio")
        cid = encode_custom_id(
            judged_agent_id=artifact.agent_id,
            run_id=artifact.run_id,
            judge_model="claude-haiku-4-5",
        )
        req = build_batch_request(
            artifact, judge_model="claude-haiku-4-5", custom_id=cid,
        )
        # The Pydantic model is the contract — the tool spec must match.
        assert req["params"]["tools"][0]["input_schema"] == (
            RubricEvalLLMOutput.model_json_schema()
        )


# ── parse_batch_message ───────────────────────────────────────────────────


class TestParseBatchMessage:
    def test_parses_well_formed_tool_use_response(self):
        from evals.judge import parse_batch_message

        message = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "RubricEvalLLMOutput",
                    "input": {
                        "dimension_scores": [
                            {
                                "dimension": "numerical_grounding",
                                "score": 4,
                                "reasoning": "good",
                            },
                        ],
                        "overall_reasoning": "ok",
                    },
                },
            ],
        }
        out = parse_batch_message(message)
        assert isinstance(out, RubricEvalLLMOutput)
        assert len(out.dimension_scores) == 1
        assert out.dimension_scores[0].score == 4

    def test_parses_object_style_message(self):
        """Anthropic SDK returns Message objects with attribute access;
        accept either form so the parser is SDK-version agnostic."""
        from evals.judge import parse_batch_message

        class FakeBlock:
            type = "tool_use"
            name = "RubricEvalLLMOutput"
            input = {
                "dimension_scores": [
                    {"dimension": "d1", "score": 3, "reasoning": "r"},
                ],
                "overall_reasoning": "o",
            }

        class FakeMessage:
            content = [FakeBlock()]

        out = parse_batch_message(FakeMessage())
        assert out.dimension_scores[0].dimension == "d1"

    def test_missing_tool_use_block_raises_value_error(self):
        from evals.judge import parse_batch_message

        message = {
            "content": [
                {"type": "text", "text": "I refuse to call the tool"},
            ],
        }
        with pytest.raises(ValueError, match="No tool_use block named"):
            parse_batch_message(message)

    def test_wrong_tool_name_raises_value_error(self):
        """Defensive — the Process Lambda must not silently accept a
        tool_use block under a different name."""
        from evals.judge import parse_batch_message

        message = {
            "content": [
                {
                    "type": "tool_use",
                    "name": "SomeOtherTool",
                    "input": {"unrelated": "data"},
                },
            ],
        }
        with pytest.raises(ValueError, match="No tool_use block named"):
            parse_batch_message(message)
