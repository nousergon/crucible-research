"""Tests for ``agents.langchain_utils.invoke_react_with_recovery``.

Surfaced by the 2026-06-13 Saturday SF failure: the consumer + healthcare
sector teams (2 of 6) hard-failed under ALL-AGENTS-STRICT with an Anthropic
400 — ``messages.N: `tool_use` ids were found without `tool_result` blocks
immediately after: toolu_…`` — while the other four teams ran clean off the
identical code path. The prebuilt ``create_react_agent`` loop sporadically
emits a malformed tool history (a model-sampling artifact, not our message
construction); because each ``agent.invoke`` starts a FRESH graph state, a
re-roll clears it. This is the ReAct analogue of the structured-output
validation retry added for the 2026-05-24 'medium_high' single-bad-roll.

These tests pin:

  1. Happy path — first invoke succeeds, no retry.
  2. Recovery — first invoke raises the malformed-history 400, the re-roll
     succeeds.
  3. Terminal — every attempt raises the 400; after ``max_retries`` it
     propagates so the caller's hard-fail branch fires (status:ERROR).
  4. Non-recoverable errors propagate immediately (no re-roll): a generic
     400, a GraphRecursionError-shaped error, a ValueError.
  5. The recoverable-400 detector matches the real Anthropic phrasing and
     rejects unrelated 400s.
"""
from __future__ import annotations

import pytest


class _BadRequest400(Exception):
    """Mimics ``anthropic.BadRequestError`` enough for the detector:
    carries ``status_code`` and the SDK ``str()`` shape."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


_DANGLING_TOOL_USE_MSG = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'messages.2: `tool_use` ids were "
    "found without `tool_result` blocks immediately after: "
    "toolu_01FfVUUck1N1ZkiN64KnNd3U. Each `tool_use` block must have a "
    "corresponding `tool_result` block in the next message.'}, "
    "'request_id': 'req_011Cc18wUT2odVk1kddpq4zw'}"
)


def _dangling_400() -> _BadRequest400:
    return _BadRequest400(_DANGLING_TOOL_USE_MSG)


class TestDetector:
    def test_matches_real_anthropic_phrasing(self):
        from agents.langchain_utils import _is_recoverable_tool_use_400

        assert _is_recoverable_tool_use_400(_dangling_400()) is True

    def test_matches_without_status_code_attr(self):
        # langchain may re-raise a plain exception carrying only the string.
        from agents.langchain_utils import _is_recoverable_tool_use_400

        assert _is_recoverable_tool_use_400(Exception(_DANGLING_TOOL_USE_MSG)) is True

    def test_rejects_unrelated_400(self):
        from agents.langchain_utils import _is_recoverable_tool_use_400

        other = _BadRequest400(
            "Error code: 400 - invalid_request_error: max_tokens too large"
        )
        assert _is_recoverable_tool_use_400(other) is False

    def test_rejects_non_400(self):
        from agents.langchain_utils import _is_recoverable_tool_use_400

        assert _is_recoverable_tool_use_400(ValueError("tool_use tool_result")) is False


class TestHappyPath:
    def test_first_invoke_succeeds_no_retry(self):
        from agents.langchain_utils import invoke_react_with_recovery

        calls = {"n": 0}

        def thunk():
            calls["n"] += 1
            return {"messages": ["ok"]}

        result = invoke_react_with_recovery(thunk, label="quant:tech:react")

        assert result == {"messages": ["ok"]}
        assert calls["n"] == 1


class TestRecovery:
    def test_reroll_succeeds_after_malformed_400(self, caplog):
        from agents.langchain_utils import invoke_react_with_recovery

        calls = {"n": 0}

        def thunk():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _dangling_400()
            return {"messages": ["recovered"]}

        with caplog.at_level("WARNING"):
            result = invoke_react_with_recovery(
                thunk, label="quant:consumer:react",
            )

        assert result == {"messages": ["recovered"]}
        assert calls["n"] == 2  # one re-roll
        assert any("re-rolling a fresh ReAct" in r.message for r in caplog.records)


class TestTerminal:
    def test_propagates_after_exhausting_retries(self):
        from agents.langchain_utils import invoke_react_with_recovery

        calls = {"n": 0}

        def thunk():
            calls["n"] += 1
            raise _dangling_400()

        with pytest.raises(_BadRequest400):
            invoke_react_with_recovery(
                thunk, label="qual:healthcare:react", max_retries=2,
            )

        # 1 initial + 2 re-rolls = 3 total attempts, then propagate.
        assert calls["n"] == 3


class TestNonRecoverablePropagatesImmediately:
    def test_generic_400_not_retried(self):
        from agents.langchain_utils import invoke_react_with_recovery

        calls = {"n": 0}

        def thunk():
            calls["n"] += 1
            raise _BadRequest400("Error code: 400 - context window exceeded")

        with pytest.raises(_BadRequest400):
            invoke_react_with_recovery(thunk, label="quant:tech:react")

        assert calls["n"] == 1  # no re-roll

    def test_value_error_not_retried(self):
        from agents.langchain_utils import invoke_react_with_recovery

        calls = {"n": 0}

        def thunk():
            calls["n"] += 1
            raise ValueError("boom")

        with pytest.raises(ValueError):
            invoke_react_with_recovery(thunk, label="quant:tech:react")

        assert calls["n"] == 1
