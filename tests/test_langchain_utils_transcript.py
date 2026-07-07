"""Tests for serialize_transcript — the bounded ReAct-history serializer that
persists agent reasoning into decision artifacts for retrospective review
(L4567). Uses lightweight duck-typed message stand-ins (the serializer reads
.type / .content / .name / .tool_calls, mirroring LangChain messages)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agents.langchain_utils import (
    is_step_budget_exhausted_sentinel,
    serialize_transcript,
)


def _human(text):
    return SimpleNamespace(type="human", content=text)


def _ai(text, tool_calls=None):
    return SimpleNamespace(type="ai", content=text, tool_calls=tool_calls or [])


def _tool(text, name="get_factor_profile"):
    return SimpleNamespace(type="tool", content=text, name=name)


def test_basic_roles_and_tool_calls():
    msgs = [
        _human("Screen the technology sector."),
        _ai("Let me check factor profiles.",
            tool_calls=[{"name": "get_factor_profile", "args": {"ticker": "NVDA"}}]),
        _tool("NVDA momentum=88 quality=72"),
        _ai("NVDA ranks #1 on momentum; MSFT lags."),
    ]
    out = serialize_transcript(msgs)
    assert [e["role"] for e in out] == ["human", "ai", "tool", "ai"]
    assert out[1]["tool_calls"][0]["name"] == "get_factor_profile"
    assert "NVDA" in out[1]["tool_calls"][0]["args"]
    assert out[2]["tool"] == "get_factor_profile"
    assert "ranks #1" in out[3]["content"]


def test_tool_response_truncated_hard():
    big = "x" * 5000
    out = serialize_transcript([_tool(big)], max_tool_response_chars=300)
    assert out[0]["content"].endswith("…[truncated]")
    assert len(out[0]["content"]) <= 300 + len("…[truncated]")


def test_reasoning_message_truncated():
    big = "y" * 5000
    out = serialize_transcript([_ai(big)], max_msg_chars=800)
    assert out[0]["content"].endswith("…[truncated]")
    assert len(out[0]["content"]) <= 800 + len("…[truncated]")


def test_total_cap_drops_tail_with_marker():
    msgs = [_ai("z" * 700) for _ in range(20)]  # ~14k chars, cap 8k
    out = serialize_transcript(msgs, max_total_chars=8000, max_msg_chars=800)
    assert any("_truncated" in e for e in out)
    # truncation marker is the last element; some messages were dropped.
    assert "_truncated" in out[-1]
    assert len(out) < len(msgs) + 1


def test_ai_list_content_blocks_extracted():
    msg = SimpleNamespace(
        type="ai",
        content=[
            {"type": "text", "text": "Considering MSFT and AAPL."},
            {"type": "tool_use", "name": "x", "input": {}},  # ignored here
        ],
        tool_calls=[],
    )
    out = serialize_transcript([msg])
    assert "Considering MSFT and AAPL." in out[0]["content"]


def test_empty_and_json_serializable():
    assert serialize_transcript([]) == []
    out = serialize_transcript([_human("hi"), _ai("bye")])
    json.dumps(out)  # must be JSON-serializable for the S3 artifact


# ── is_step_budget_exhausted_sentinel (config#1822) ───────────────────────────
#
# langgraph.prebuilt.chat_agent_executor's internal remaining_steps guard
# swaps in this EXACT literal AIMessage content and returns normally
# (no GraphRecursionError) once its step budget is nearly exhausted. The
# quant/qual analysts must detect it exactly (not by substring) so
# genuine analyst prose that happens to mention running low on steps
# isn't misclassified, and must not miss the real sentinel via a stray
# whitespace/case difference from a future langgraph version bump.


def test_sentinel_exact_match_detected():
    assert is_step_budget_exhausted_sentinel(
        "Sorry, need more steps to process this request."
    ) is True


def test_sentinel_detected_with_surrounding_whitespace():
    assert is_step_budget_exhausted_sentinel(
        "  Sorry, need more steps to process this request.  \n"
    ) is True


def test_sentinel_not_detected_for_normal_analysis_text():
    assert is_step_budget_exhausted_sentinel(
        "Based on the data, my top pick is AAPL with a quant_score of 82."
    ) is False


def test_sentinel_not_detected_for_substring_containing_text():
    """Exact-match only — prose that happens to mention the phrase (but
    isn't literally just the langgraph bailout) must not be misclassified
    as a step-budget exhaustion."""
    assert is_step_budget_exhausted_sentinel(
        "Sorry, need more steps to process this request. Actually here "
        "are my final picks: AAPL, MSFT."
    ) is False


def test_sentinel_handles_none_and_empty():
    assert is_step_budget_exhausted_sentinel(None) is False
    assert is_step_budget_exhausted_sentinel("") is False
