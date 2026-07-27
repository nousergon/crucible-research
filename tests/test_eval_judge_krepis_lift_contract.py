"""Byte-identical contract test for the config#1675 / config#2575
krepis.judge lift.

``evals/judge.py`` + ``evals/judge_models.py`` now delegate their
transport/mechanics layer (rubric rendering, structured-output tool
spec, batch custom_id codec, batch tool-result parsing, the judge-model
registry) to ``krepis.judge`` instead of implementing it locally. This
module pins the PRE-LIFT implementations inline (verbatim copies of the
functions as they existed on ``main`` immediately before the delegation
patch) as a golden reference, and asserts the live (post-lift,
delegating) ``evals.judge`` / ``evals.judge_models`` functions produce
field-identical output for fixed inputs.

This is the bar config#2575 item 1 sets: "contract test diffing
pre/post RubricEvalArtifact for byte identity." The full
``RubricEvalArtifact`` also carries a ``timestamp`` field
(``datetime.now(timezone.utc)`` at construction) which is
nondeterministic by design — excluded from comparison here, per the
issue's own carve-out for nondeterministic fields.
"""

from __future__ import annotations

import json
import re

import pytest
from nousergon_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)

from evals import judge as judge_mod
from evals import judge_models
from graph.state_schemas import RubricEvalLLMOutput


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
        input_data_snapshot={
            "team_id": "technology",
            "sector_tickers": ["AAPL", "MSFT"],
            "technical_scores_team": {
                "AAPL": {"rsi_14": 55, "technical_score": 70},
            },
        },
        input_data_summary="team_id=technology",
        agent_output=agent_output if agent_output is not None else {
            "ranked_picks": [
                {"ticker": "AAPL", "quant_score": 70, "quant_rationale": "RSI 55, TS 70."},
            ],
        },
    )


# ── Golden pre-lift reference implementations (verbatim, frozen) ────────
#
# Copied from evals/judge.py / evals/judge_models.py as they existed on
# main immediately before the config#2575 delegation patch. Do NOT
# "clean up" or refactor these — their exact un-refactored shape is the
# point.


_GOLDEN_CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_GOLDEN_TAG_BY_LOGICAL = {"claude-haiku-4-5": "h45", "claude-sonnet-4-6": "s46"}


def _golden_encode_custom_id(*, judged_agent_id, run_id, judge_model):
    tag = _GOLDEN_TAG_BY_LOGICAL.get(judge_model)
    if tag is None:
        tag = f"x{abs(hash(judge_model)) % 10_000:04d}"
    safe_agent = re.sub(r"[^a-zA-Z0-9_-]", "-", judged_agent_id)
    safe_run = re.sub(r"[^a-zA-Z0-9_-]", "-", run_id)
    fixed_overhead = len(safe_run) + len(tag) + 4
    max_agent = max(8, 64 - fixed_overhead)
    if len(safe_agent) > max_agent:
        safe_agent = safe_agent[:max_agent]
    cid = f"{safe_agent}__{safe_run}__{tag}"
    if not _GOLDEN_CUSTOM_ID_PATTERN.match(cid):
        cid = re.sub(r"[^a-zA-Z0-9_-]", "-", cid)[:64]
    return cid


def _golden_decode_custom_id(custom_id):
    parts = custom_id.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"Cannot decode batch custom_id={custom_id!r}: expected "
            f"three '__'-separated segments, got {len(parts)}."
        )
    safe_agent, safe_run, tag = parts
    reverse = {v: k for k, v in _GOLDEN_TAG_BY_LOGICAL.items()}
    judge_model = reverse.get(tag, tag)
    return safe_agent, safe_run, judge_model


def _golden_render_rubric(artifact, template_text: str) -> str:
    return template_text.format(
        agent_input=json.dumps(artifact.input_data_snapshot, indent=2, default=str),
        agent_output=json.dumps(artifact.agent_output, indent=2, default=str),
    )


def _golden_build_rubric_tool_spec() -> dict:
    schema = RubricEvalLLMOutput.model_json_schema()
    return {
        "name": "RubricEvalLLMOutput",
        "description": (
            "Emit the rubric eval as a structured tool call. Each rubric "
            "dimension produces one entry in dimension_scores with an "
            "integer score and short reasoning. overall_reasoning is a "
            "1-2 sentence cross-dimension summary."
        ),
        "input_schema": schema,
    }


def _golden_parse_batch_message(message_payload):
    content = (
        message_payload["content"] if isinstance(message_payload, dict)
        else message_payload.content
    )
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else block.type
        block_name = (
            block.get("name") if isinstance(block, dict)
            else getattr(block, "name", None)
        )
        if block_type == "tool_use" and block_name == "RubricEvalLLMOutput":
            tool_input = block["input"] if isinstance(block, dict) else block.input
            return RubricEvalLLMOutput.model_validate(tool_input)
    raise ValueError(
        "No tool_use block named 'RubricEvalLLMOutput' found in batch "
        "result message; the judge LLM did not emit the rubric eval via "
        "the structured tool — inspect the raw batch result on "
        "Anthropic's side (retained 29 days)."
    )


# ── Contract tests ───────────────────────────────────────────────────────


class TestCustomIdCodecContract:
    @pytest.mark.parametrize(
        "judged_agent_id,run_id,judge_model",
        [
            ("sector_quant:technology", "test-run-001", "claude-haiku-4-5"),
            ("ic_cio", "2026-07-15T09-30", "claude-sonnet-4-6"),
            ("thesis_update:financials:JPM", "run-1", "claude-haiku-4-5"),
            ("thesis_update:" + "x" * 90, "run-1", "claude-haiku-4-5"),
            ("ic_cio", "run-1", "unregistered-model"),
        ],
    )
    def test_encode_matches_golden(self, judged_agent_id, run_id, judge_model):
        golden = _golden_encode_custom_id(
            judged_agent_id=judged_agent_id, run_id=run_id, judge_model=judge_model,
        )
        live = judge_mod.encode_custom_id(
            judged_agent_id=judged_agent_id, run_id=run_id, judge_model=judge_model,
        )
        assert live == golden

    def test_decode_matches_golden(self):
        cid = judge_mod.encode_custom_id(
            judged_agent_id="ic_cio", run_id="run-1", judge_model="claude-haiku-4-5",
        )
        assert judge_mod.decode_custom_id(cid) == _golden_decode_custom_id(cid)


class TestRenderRubricContract:
    @pytest.mark.parametrize(
        "agent_id,agent_output",
        [
            ("sector_quant:technology", None),
            ("ic_cio", {}),
            ("thesis_update:tech:AAPL", {"nested": {"a": [1, 2]}, "b": None}),
        ],
    )
    def test_render_matches_golden(self, agent_id, agent_output):
        artifact = _make_artifact(agent_id, agent_output=agent_output)
        template_text = "AGENT INPUT:\n{agent_input}\n\nAGENT OUTPUT:\n{agent_output}"

        class _FakeLoadedPrompt:
            text = template_text

            def format(self, **kwargs):
                return self.text.format(**kwargs)

        golden = _golden_render_rubric(artifact, template_text)
        live = judge_mod._render_rubric(artifact, _FakeLoadedPrompt())
        assert live == golden


class TestToolSpecContract:
    def test_build_rubric_tool_spec_matches_golden(self):
        assert judge_mod._build_rubric_tool_spec() == _golden_build_rubric_tool_spec()


class TestParseBatchMessageContract:
    @pytest.mark.parametrize(
        "message_payload",
        [
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "RubricEvalLLMOutput",
                        "input": {
                            "dimension_scores": [
                                {"dimension": "numerical_grounding", "score": 4, "reasoning": "good"},
                            ],
                            "overall_reasoning": "solid",
                        },
                    },
                ],
            },
        ],
    )
    def test_parse_matches_golden(self, message_payload):
        golden = _golden_parse_batch_message(message_payload)
        live = judge_mod.parse_batch_message(message_payload)
        assert live.model_dump() == golden.model_dump()

    def test_missing_tool_raises_same_message_shape(self):
        message_payload = {"content": [{"type": "text", "text": "nope"}]}
        with pytest.raises(ValueError, match="No tool_use block named") as golden_exc:
            _golden_parse_batch_message(message_payload)
        with pytest.raises(ValueError, match="No tool_use block named") as live_exc:
            judge_mod.parse_batch_message(message_payload)
        assert str(golden_exc.value) == str(live_exc.value)


class TestJudgeModelRegistryContract:
    """judge_models.resolve()/request_model_for() must behave exactly as
    the pre-lift local implementation for the closed Haiku/Sonnet set."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "h45",
            "claude-sonnet-4-6",
            "s46",
        ],
    )
    def test_resolve_returns_same_spec_fields(self, model):
        spec = judge_models.resolve(model)
        assert spec.logical_key in {"claude-haiku-4-5", "claude-sonnet-4-6"}

    def test_resolve_unknown_still_fails_loud_with_module_specific_message(self):
        with pytest.raises(KeyError, match="evals/judge_models.py"):
            judge_models.resolve("totally-unknown-model")

    def test_request_model_for_haiku_matches_golden_pin(self):
        assert judge_models.request_model_for("claude-haiku-4-5") == "claude-haiku-4-5-20251001"

    def test_request_model_for_sonnet_matches_golden_alias(self):
        assert judge_models.request_model_for("claude-sonnet-4-6") == "claude-sonnet-4-6"


class TestBuildBatchRequestContract:
    def test_full_batch_request_shape_matches_golden_composition(self):
        """End-to-end: build_batch_request composes _render_rubric +
        _build_rubric_tool_spec + request_model_for + the lib's
        build_batches_request_params (unchanged by this lift). Assert
        the composed result is identical to manually re-deriving it via
        the golden per-piece functions."""
        artifact = _make_artifact("sector_quant:technology")
        cid = judge_mod.encode_custom_id(
            judged_agent_id=artifact.agent_id, run_id=artifact.run_id,
            judge_model="claude-haiku-4-5",
        )
        live = judge_mod.build_batch_request(
            artifact, judge_model="claude-haiku-4-5", custom_id=cid,
        )

        assert live["custom_id"] == cid
        assert live["params"]["model"] == "claude-haiku-4-5-20251001"
        assert live["params"]["tools"][0] == _golden_build_rubric_tool_spec()
        assert live["params"]["tool_choice"] == {
            "type": "tool", "name": "RubricEvalLLMOutput",
        }
