"""Tests for ``agents.langchain_utils.invoke_structured_with_validation_retry``.

Surfaced by the 2026-05-24 Saturday SF healthcare-team failure: an LLM
returned ``'medium_high'`` for a Pydantic ``Literal['low','medium','high']``
field, the single-attempt extraction hard-failed under ALL-AGENTS-STRICT,
and the whole Research Lambda halted. SOTA tool-use pattern: feed the
``ValidationError`` back as correction context and retry up to N times.

These tests pin:

  1. Happy path — first attempt succeeds, no retry.
  2. Recovery — first attempt fails validation, second succeeds.
  3. Terminal failure — all retries exhaust, last response (with the
     populated ``parsing_error``) propagates so the caller's existing
     hard-fail branch fires as before.
  4. Correction-message shape — when a retry happens, the second invoke
     includes the prior failed AIMessage + a HumanMessage that names the
     specific ValidationError.
  5. Composes with rate-limit retry — 429s during a retry attempt route
     through ``invoke_with_rate_limit_retry``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _resp(parsed=None, parsing_error=None, raw=None):
    """Mimic the dict shape ``with_structured_output(include_raw=True)`` returns."""
    return {"parsed": parsed, "parsing_error": parsing_error, "raw": raw}


def _validation_error(message="mock validation failure"):
    """Build a plausible Pydantic-like exception for tests."""
    from pydantic import ValidationError  # type: ignore[import]
    try:
        # Real ValidationError needs a model; raise + catch the easy way
        from pydantic import BaseModel
        from typing import Literal

        class _M(BaseModel):
            confidence: Literal["low", "medium", "high"]

        _M(confidence="medium_high")  # type: ignore[arg-type]
    except ValidationError as e:
        return e
    raise AssertionError("could not synthesize ValidationError")


class TestHappyPath:
    def test_first_attempt_succeeds_no_retry(self):
        from agents.langchain_utils import invoke_structured_with_validation_retry

        parsed_obj = object()
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _resp(parsed=parsed_obj)

        resp = invoke_structured_with_validation_retry(
            structured_llm, ["msg"], label="test:happy",
        )

        assert resp["parsed"] is parsed_obj
        assert resp["parsing_error"] is None
        # No retry — invoke called exactly once
        assert structured_llm.invoke.call_count == 1


class TestRecovery:
    def test_second_attempt_succeeds_after_validation_error(self, caplog):
        import logging
        from agents.langchain_utils import invoke_structured_with_validation_retry

        parsed_obj = object()
        ve = _validation_error()
        raw_msg = MagicMock(name="raw-AIMessage-failed-attempt")
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = [
            _resp(parsed=None, parsing_error=ve, raw=raw_msg),
            _resp(parsed=parsed_obj),
        ]

        with caplog.at_level(logging.INFO, logger="agents.langchain_utils"):
            resp = invoke_structured_with_validation_retry(
                structured_llm, ["initial-msg"], label="test:recovery",
            )

        assert resp["parsed"] is parsed_obj
        assert resp["parsing_error"] is None
        assert structured_llm.invoke.call_count == 2
        # Second call should include the raw + correction message after the
        # original. Pull the call args and verify message-list grew.
        first_args, _ = structured_llm.invoke.call_args_list[0]
        second_args, _ = structured_llm.invoke.call_args_list[1]
        assert first_args[0] == ["initial-msg"]
        assert len(second_args[0]) == 3  # original + raw + correction
        assert second_args[0][0] == "initial-msg"
        assert second_args[0][1] is raw_msg
        # Correction message should name the schema violation
        correction = second_args[0][2]
        assert "schema validation" in correction.content.lower()
        assert "medium_high" in correction.content  # the bad value

        # Log confirms the recovery
        recovery_logs = [r for r in caplog.records if "succeeded after" in r.message]
        assert recovery_logs

    def test_correction_used_even_when_raw_is_None(self):
        """Defensive: if the structured-output handle doesn't include the raw
        AIMessage on failure, the retry still re-prompts with the correction
        appended to the original messages — model gets schema feedback even
        without the prior-attempt context."""
        from agents.langchain_utils import invoke_structured_with_validation_retry

        parsed_obj = object()
        ve = _validation_error()
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = [
            _resp(parsed=None, parsing_error=ve, raw=None),
            _resp(parsed=parsed_obj),
        ]

        resp = invoke_structured_with_validation_retry(
            structured_llm, ["msg"], label="test:no-raw",
        )

        assert resp["parsed"] is parsed_obj
        second_args, _ = structured_llm.invoke.call_args_list[1]
        # Original + correction only (no raw to interpose)
        assert len(second_args[0]) == 2
        assert second_args[0][0] == "msg"
        assert "schema validation" in second_args[0][1].content.lower()


class TestTerminalFailure:
    def test_all_retries_exhaust_returns_last_response_with_parsing_error(self, caplog):
        import logging
        from agents.langchain_utils import invoke_structured_with_validation_retry

        ve = _validation_error()
        structured_llm = MagicMock()
        # 3 calls all fail validation (default max_retries=2 → 3 total attempts)
        structured_llm.invoke.return_value = _resp(parsed=None, parsing_error=ve)

        with caplog.at_level(logging.WARNING, logger="agents.langchain_utils"):
            resp = invoke_structured_with_validation_retry(
                structured_llm, ["msg"], label="test:terminal",
            )

        assert resp["parsed"] is None
        assert resp["parsing_error"] is ve
        assert structured_llm.invoke.call_count == 3  # initial + 2 retries
        # Caller's existing branch on `parsing_error is not None` fires as before;
        # this helper does NOT raise.
        terminal_logs = [r for r in caplog.records if "failed after" in r.message]
        assert terminal_logs

    def test_max_retries_zero_does_one_attempt(self):
        from agents.langchain_utils import invoke_structured_with_validation_retry

        ve = _validation_error()
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _resp(parsed=None, parsing_error=ve)

        resp = invoke_structured_with_validation_retry(
            structured_llm, ["msg"], label="test:zero", max_retries=0,
        )

        assert resp["parsing_error"] is ve
        assert structured_llm.invoke.call_count == 1


class TestRateLimitComposition:
    def test_429_during_attempt_routes_through_rate_limit_retry(self, monkeypatch):
        """The outer validation-retry wrapper invokes through
        ``invoke_with_rate_limit_retry``, so a 429 during one attempt is
        backed-off-and-retried at the rate-limit layer BEFORE counting
        against the validation-retry budget. Verify the wrapper is called
        on every attempt (1 + max_retries times)."""
        from agents import langchain_utils

        calls: list[str] = []

        def _spy(fn, *, label, **kwargs):
            calls.append(label)
            return fn()

        monkeypatch.setattr(langchain_utils, "invoke_with_rate_limit_retry", _spy)

        parsed_obj = object()
        ve = _validation_error()
        structured_llm = MagicMock()
        structured_llm.invoke.side_effect = [
            _resp(parsed=None, parsing_error=ve),
            _resp(parsed=parsed_obj),
        ]

        resp = langchain_utils.invoke_structured_with_validation_retry(
            structured_llm, ["msg"], label="test:rate-limit",
        )

        assert resp["parsed"] is parsed_obj
        # Two attempts → two rate-limit-retry invocations
        assert len(calls) == 2
        assert all(c.startswith("test:rate-limit:attempt=") for c in calls)
