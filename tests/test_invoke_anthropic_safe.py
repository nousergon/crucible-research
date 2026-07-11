"""Tests for the single send-time tool_use/tool_result pairing chokepoint
``agents.langchain_utils.invoke_anthropic_safe`` (config#2255).

The "no orphan ``tool_use`` may reach the Anthropic Messages API" invariant bit
the fleet twice, ~1 month apart, at two different call sites (the ReAct loop —
config#1065 — and the structured-retry chokepoint — config#2245). Each fix was a
per-call-path belt. ``invoke_anthropic_safe`` lifts the invariant to a single
send-time chokepoint that EVERY Anthropic-backed multi-message ``.invoke()``
routes through, so a naive new call site that assembles an orphan ``tool_use``
is structurally incapable of reaching the API.

Pins:

  1. Clean history → forwarded to ``handle.invoke`` byte-for-byte (no-op repair).
  2. Orphan ``tool_use`` → dropped BEFORE the handle sees it; WARN names the id.
  3. Repair drops, never FABRICATES — the handle never receives a synthetic
     ``tool_result`` for the dropped turn.
  4. The "naive new call site" guard — an orphan-bearing history assembled by a
     caller that forgot the belt still cannot reach the handle.
  5. ``**invoke_kwargs`` (config=…) and ``deadline_seconds`` are forwarded.
  6. Return value passes through unchanged (bare AIMessage and include_raw dict).
  7. Fail-loud preserved — a non-429 error from the handle propagates unchanged.
"""
from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents import langchain_utils
from agents.langchain_utils import invoke_anthropic_safe


def _ai_tool_use(tid: str, name: str = "get_factor_profile") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": tid}])


def _clean_history() -> list:
    return [
        HumanMessage(content="screen the sector"),
        _ai_tool_use("toolu_a1"),
        ToolMessage(content="factor data", tool_call_id="toolu_a1"),
        AIMessage(content="here are my ranked picks"),
    ]


def _orphan_history() -> list:
    # AI emits a tool_use, then the NEXT message is another AI turn — the
    # ToolMessage answering toolu_a1 was never appended (the 400 signature).
    return [
        HumanMessage(content="screen the sector"),
        _ai_tool_use("toolu_orphan"),
        AIMessage(content="picks (but the tool_use was never answered)"),
    ]


class _RecordingHandle:
    """Fake LLM handle recording exactly what ``.invoke`` was sent."""

    def __init__(self, ret=None, exc: BaseException | None = None):
        self.received: list | None = None
        self.received_kwargs: dict | None = None
        self.call_count = 0
        self._ret = ret
        self._exc = exc

    def invoke(self, messages, **kwargs):
        self.received = messages
        self.received_kwargs = kwargs
        self.call_count += 1
        if self._exc is not None:
            raise self._exc
        return self._ret


class TestCleanHistoryPassthrough:
    def test_clean_history_forwarded_unchanged(self, caplog):
        handle = _RecordingHandle(ret="ok")
        msgs = _clean_history()
        with caplog.at_level(logging.WARNING):
            out = invoke_anthropic_safe(handle, msgs, label="t:clean")
        assert out == "ok"
        assert handle.call_count == 1
        # Same messages reach the handle (value-equal, nothing dropped).
        assert handle.received == msgs
        # No repair → no WARN.
        assert not any("dropped" in r.message.lower() for r in caplog.records)


class TestOrphanIsStrippedBeforeSend:
    def test_orphan_tool_use_never_reaches_handle(self, caplog):
        handle = _RecordingHandle(ret="ok")
        with caplog.at_level(logging.WARNING):
            invoke_anthropic_safe(handle, _orphan_history(), label="t:orphan")
        assert handle.call_count == 1
        received = handle.received
        # The orphan-bearing assistant turn is gone; no message the handle
        # receives carries the orphan tool_call id.
        for m in received:
            for tc in (getattr(m, "tool_calls", None) or []):
                assert tc.get("id") != "toolu_orphan"
        # Only the well-formed HumanMessage survives (the trailing AI turn is
        # kept — it had no tool_calls — and the orphan AI turn is dropped).
        assert HumanMessage(content="screen the sector") in received
        # WARN fired and NAMES the dropped id (flow-doctor visibility).
        warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("toolu_orphan" in w for w in warns)
        assert any("anthropic_safe:t:orphan" in w for w in warns)

    def test_repair_never_fabricates_a_tool_result(self):
        handle = _RecordingHandle(ret="ok")
        invoke_anthropic_safe(handle, _orphan_history(), label="t:nofab")
        # No ToolMessage answering the dropped tool_use was synthesized.
        for m in handle.received:
            if isinstance(m, ToolMessage):
                assert m.tool_call_id != "toolu_orphan"

    def test_naive_new_call_site_cannot_send_orphan(self):
        """The regression guard the issue asks for: a caller that assembles an
        orphan ``tool_use`` and routes it through the chokepoint (forgetting
        any per-site belt) STILL cannot get that orphan to the API."""
        handle = _RecordingHandle(ret="ok")
        # A naive caller appends a raw tool_use AIMessage and then a plain
        # correction HumanMessage — orphaning the tool_use (the config#2245
        # shape).
        naive = [
            HumanMessage(content="extract picks"),
            _ai_tool_use("toolu_naive"),
            HumanMessage(content="that failed, try again"),
        ]
        invoke_anthropic_safe(handle, naive, label="t:naive")
        ids = [
            tc.get("id")
            for m in handle.received
            for tc in (getattr(m, "tool_calls", None) or [])
        ]
        assert "toolu_naive" not in ids


class TestKwargForwarding:
    def test_config_and_kwargs_forwarded(self):
        handle = _RecordingHandle(ret="ok")
        cfg = {"metadata": {"ls": "x"}}
        invoke_anthropic_safe(
            handle, [HumanMessage(content="hi")], label="t:kw", config=cfg,
        )
        assert handle.received_kwargs == {"config": cfg}

    def test_deadline_seconds_forwarded_to_rate_limit_retry(self, monkeypatch):
        captured: dict = {}

        def _spy(fn, *, label, deadline_seconds=None):
            captured["label"] = label
            captured["deadline_seconds"] = deadline_seconds
            return fn()

        monkeypatch.setattr(langchain_utils, "invoke_with_rate_limit_retry", _spy)
        handle = _RecordingHandle(ret="ok")
        invoke_anthropic_safe(
            handle, [HumanMessage(content="hi")], label="t:dl",
            deadline_seconds=180.0,
        )
        assert captured["deadline_seconds"] == 180.0
        assert captured["label"] == "t:dl"


class TestReturnPassthrough:
    def test_include_raw_dict_passthrough(self):
        raw_dict = {"raw": object(), "parsed": object(), "parsing_error": None}
        handle = _RecordingHandle(ret=raw_dict)
        out = invoke_anthropic_safe(
            handle, [HumanMessage(content="hi")], label="t:dict",
        )
        assert out is raw_dict


class TestFailLoudPreserved:
    def test_non_429_error_propagates_unchanged(self):
        boom = ValueError("schema exploded")
        handle = _RecordingHandle(exc=boom)
        with pytest.raises(ValueError, match="schema exploded"):
            invoke_anthropic_safe(
                handle, [HumanMessage(content="hi")], label="t:boom",
            )
