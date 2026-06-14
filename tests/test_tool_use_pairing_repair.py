"""Tests for the pre-send tool_use/tool_result pairing repair (config#1065).

The 2026-06-13 Saturday run hard-failed 2 of 6 sector teams (consumer,
healthcare) with the Anthropic 400 ``messages.N: `tool_use` ids were found
without `tool_result` blocks immediately after: toolu_…`` — an INTERMITTENT
malformed-history artifact of the prebuilt ``create_react_agent`` loop (a
truncated / aborted tool_use emission that never received its ToolMessage
answer). ``invoke_react_with_recovery`` re-rolls on the 400 (the outer net);
this module pins the INNER structural fix — the ``pre_model_hook`` that
REPAIRS the LLM-input message view before every turn so an orphan
``tool_use`` can never be SENT.

Pins (config#1065 fix-plan 1+2+3):

  1. ``find_orphan_tool_use_ids`` — detects an unanswered assistant
     ``tool_use`` (the 400 signature); a clean history yields no orphans.
  2. ``validate_tool_use_pairing`` — raises on an orphan, passes on a
     clean / repaired list.
  3. ``repair_tool_use_pairing`` — drops the orphan-bearing assistant turn
     (and any now-dangling ToolMessage), never fabricates a tool_result,
     and is idempotent / a no-op on a clean list.
  4. ``make_tool_use_repair_hook`` — the create_react_agent pre_model_hook:
     returns ``{"llm_input_messages": repaired}`` (does NOT mutate state),
     WARNs on repair, no-ops on clean.
  5. Retry rebuilds clean state — a fresh ReAct invocation does not carry
     an orphan from a prior failed attempt (the hypothesised root cause).

All tests are pure message-array manipulation — NO Anthropic / network
calls.
"""
from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def _ai_tool_use(tid: str, name: str = "get_factor_profile"):
    """An assistant turn emitting a single tool_call ``tid``."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": tid}])


def _ai_multi(*tids: str):
    return AIMessage(
        content="",
        tool_calls=[{"name": "t", "args": {}, "id": t} for t in tids],
    )


def _clean_history() -> list:
    return [
        HumanMessage(content="screen the sector"),
        _ai_tool_use("toolu_a1"),
        ToolMessage(content="factor data", tool_call_id="toolu_a1"),
        AIMessage(content="here are my ranked picks"),
    ]


def _orphan_history() -> list:
    # AI emits a tool_use, then the NEXT message is another AI turn — the
    # ToolMessage answering toolu_a1 was never appended (truncated/aborted).
    return [
        HumanMessage(content="screen the sector"),
        _ai_tool_use("toolu_a1"),
        AIMessage(content="picks (but a1 was never answered)"),
    ]


class TestFindOrphans:
    def test_clean_history_has_no_orphans(self):
        from agents.langchain_utils import find_orphan_tool_use_ids

        assert find_orphan_tool_use_ids(_clean_history()) == []

    def test_unanswered_tool_use_is_orphan(self):
        from agents.langchain_utils import find_orphan_tool_use_ids

        assert find_orphan_tool_use_ids(_orphan_history()) == ["toolu_a1"]

    def test_partially_answered_multi_call_flags_the_unanswered_id(self):
        from agents.langchain_utils import find_orphan_tool_use_ids

        msgs = [
            HumanMessage(content="q"),
            _ai_multi("toolu_a1", "toolu_a2"),
            ToolMessage(content="r1", tool_call_id="toolu_a1"),
            # toolu_a2 never answered
        ]
        assert find_orphan_tool_use_ids(msgs) == ["toolu_a2"]

    def test_blank_tool_call_id_is_always_orphan(self):
        from agents.langchain_utils import find_orphan_tool_use_ids

        msgs = [HumanMessage(content="q"), _ai_tool_use("")]
        assert find_orphan_tool_use_ids(msgs) == ["<missing-id>"]

    def test_answer_anywhere_later_is_not_orphan(self):
        # The create_react_agent loop always appends the ToolMessage right
        # after, but "answered later" must not be flagged as an orphan.
        from agents.langchain_utils import find_orphan_tool_use_ids

        msgs = [
            HumanMessage(content="q"),
            _ai_tool_use("toolu_a1"),
            ToolMessage(content="r1", tool_call_id="toolu_a1"),
            _ai_tool_use("toolu_a2"),
            ToolMessage(content="r2", tool_call_id="toolu_a2"),
        ]
        assert find_orphan_tool_use_ids(msgs) == []


class TestValidate:
    def test_clean_history_passes(self):
        from agents.langchain_utils import validate_tool_use_pairing

        validate_tool_use_pairing(_clean_history())  # no raise

    def test_orphan_history_raises(self):
        from agents.langchain_utils import validate_tool_use_pairing

        with pytest.raises(ValueError, match="tool_use id"):
            validate_tool_use_pairing(_orphan_history())

    def test_repaired_history_passes_validation(self):
        from agents.langchain_utils import (
            repair_tool_use_pairing,
            validate_tool_use_pairing,
        )

        repaired, _ = repair_tool_use_pairing(_orphan_history())
        validate_tool_use_pairing(repaired)  # no raise — repair is sound


class TestRepair:
    def test_orphan_array_is_repaired_and_passes(self):
        from agents.langchain_utils import (
            find_orphan_tool_use_ids,
            repair_tool_use_pairing,
        )

        repaired, dropped = repair_tool_use_pairing(_orphan_history())

        assert dropped == ["toolu_a1"]
        # The orphan-bearing assistant turn is gone; the human + the final
        # assistant turn remain.
        assert [m.type for m in repaired] == ["human", "ai"]
        # And the repaired list is now clean.
        assert find_orphan_tool_use_ids(repaired) == []

    def test_clean_history_is_unchanged_noop(self):
        from agents.langchain_utils import repair_tool_use_pairing

        clean = _clean_history()
        repaired, dropped = repair_tool_use_pairing(clean)

        assert dropped == []
        assert [m.type for m in repaired] == [m.type for m in clean]

    def test_repair_is_idempotent(self):
        from agents.langchain_utils import repair_tool_use_pairing

        once, _ = repair_tool_use_pairing(_orphan_history())
        twice, dropped2 = repair_tool_use_pairing(once)

        assert dropped2 == []
        assert [m.type for m in twice] == [m.type for m in once]

    def test_dangling_tool_result_removed_when_its_turn_dropped(self):
        # A multi-call assistant turn with ONE unanswered id is dropped
        # whole; the ToolMessage that DID answer the other id is now
        # dangling (a tool_result with no preceding tool_use → also a 400)
        # and must be removed too.
        from agents.langchain_utils import (
            find_orphan_tool_use_ids,
            repair_tool_use_pairing,
        )

        msgs = [
            HumanMessage(content="q"),
            _ai_multi("toolu_a1", "toolu_a2"),  # a2 unanswered → whole turn orphan
            ToolMessage(content="r1", tool_call_id="toolu_a1"),
        ]
        repaired, dropped = repair_tool_use_pairing(msgs)

        assert sorted(dropped) == ["toolu_a1", "toolu_a2"]
        # Only the human message survives; the dangling ToolMessage is gone.
        assert [m.type for m in repaired] == ["human"]
        assert find_orphan_tool_use_ids(repaired) == []

    def test_repair_never_fabricates_a_tool_result(self):
        # Drop-not-fabricate: the repaired list must never contain a
        # synthesized ToolMessage for the dropped id (a fabricated
        # observation would feed the model a hallucinated tool result).
        from agents.langchain_utils import repair_tool_use_pairing

        repaired, _ = repair_tool_use_pairing(_orphan_history())
        tool_result_ids = [
            getattr(m, "tool_call_id", None)
            for m in repaired
            if getattr(m, "type", None) == "tool"
        ]
        assert "toolu_a1" not in tool_result_ids


class TestPreModelHook:
    def test_hook_returns_repaired_llm_input_messages(self, caplog):
        from agents.langchain_utils import make_tool_use_repair_hook

        hook = make_tool_use_repair_hook(label="quant:consumer")
        with caplog.at_level(logging.WARNING):
            out = hook({"messages": _orphan_history()})

        assert set(out.keys()) == {"llm_input_messages"}
        # Orphan dropped from the LLM-input view.
        assert [m.type for m in out["llm_input_messages"]] == ["human", "ai"]
        # Repair is surfaced (flow-doctor-detectable), not silent.
        assert any(
            "dropped 1 orphan tool_use" in r.message for r in caplog.records
        )
        assert any("config#1065" in r.message for r in caplog.records)

    def test_hook_does_not_mutate_input_state(self):
        # The hook returns a NEW list via llm_input_messages; the persisted
        # graph-state ``messages`` must be left intact.
        from agents.langchain_utils import make_tool_use_repair_hook

        state = {"messages": _orphan_history()}
        original_len = len(state["messages"])
        hook = make_tool_use_repair_hook(label="qual:healthcare")
        hook(state)

        assert len(state["messages"]) == original_len  # untouched

    def test_hook_noop_on_clean_history(self, caplog):
        from agents.langchain_utils import make_tool_use_repair_hook

        hook = make_tool_use_repair_hook(label="quant:tech")
        clean = _clean_history()
        with caplog.at_level(logging.WARNING):
            out = hook({"messages": clean})

        assert [m.type for m in out["llm_input_messages"]] == [
            m.type for m in clean
        ]
        # No repair WARN on a clean history.
        assert not any("orphan tool_use" in r.message for r in caplog.records)

    def test_hook_handles_empty_state(self):
        from agents.langchain_utils import make_tool_use_repair_hook

        hook = make_tool_use_repair_hook(label="quant:tech")
        assert hook({}) == {"llm_input_messages": []}
        assert hook({"messages": None}) == {"llm_input_messages": []}


class TestRetryRebuildsCleanState:
    """config#1065 fix-plan 3: the per-team retry wrapper must rebuild
    clean message history between attempts — a failed attempt's orphan
    ``tool_use`` must NOT carry into the re-roll.

    ``invoke_react_with_recovery`` re-invokes a thunk that constructs a
    FRESH ``agent.invoke({"messages": [user]})`` each time, so the graph
    state starts from a single user message every attempt — no carryover.
    We assert that property directly: each attempt sees a clean, single
    user-message input regardless of what the prior attempt's internal
    loop accumulated."""

    def test_each_reroll_starts_from_clean_single_user_message(self):
        from agents.langchain_utils import (
            find_orphan_tool_use_ids,
            invoke_react_with_recovery,
        )

        class _BadRequest400(Exception):
            def __init__(self, msg):
                super().__init__(msg)
                self.status_code = 400

        _DANGLING = (
            "Error code: 400 - invalid_request_error: messages.2: `tool_use` "
            "ids were found without `tool_result` blocks immediately after: "
            "toolu_x. Each `tool_use` block must have a corresponding "
            "`tool_result` block in the next message."
        )

        seen_inputs: list[list] = []
        attempts = {"n": 0}

        def fresh_invoke():
            # Mirror the real call sites: every attempt builds a brand-new
            # single-user-message input dict (clean graph state).
            inputs = [{"role": "user", "content": "screen the sector"}]
            seen_inputs.append(inputs)
            attempts["n"] += 1
            if attempts["n"] == 1:
                # Simulate the loop having internally accumulated an orphan
                # tool_use that produced the 400 — but it never leaks into
                # the NEXT attempt's input.
                raise _BadRequest400(_DANGLING)
            return {"messages": [AIMessage("recovered picks")]}

        result = invoke_react_with_recovery(
            fresh_invoke, label="quant:consumer:react",
        )

        assert result["messages"][0].content == "recovered picks"
        assert attempts["n"] == 2  # one clean re-roll
        # Both attempts started from an identical clean single-user input —
        # no orphan carried forward.
        assert len(seen_inputs) == 2
        for inp in seen_inputs:
            assert inp == [{"role": "user", "content": "screen the sector"}]
            assert find_orphan_tool_use_ids(inp) == []
