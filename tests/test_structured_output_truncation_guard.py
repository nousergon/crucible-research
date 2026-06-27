"""Tests for the runtime structured-output truncation guard (config#1294).

Before this guard NOTHING in the repo inspected ``stop_reason`` at runtime;
the only protection was a hand-estimated static budget table
(``tests/test_schema_max_tokens_audit.py``) which missed a real incident.
When an Anthropic tool-call truncates (``stop_reason == "max_tokens"``)
langchain captures a PARTIAL parameter block as a raw string and the failure
surfaces downstream as a confusing Pydantic shape error (e.g.
``catalysts: Input should be a valid list … input_type=str``) rather than at
the root cause.

These tests pin the SOTA fix: at the shared structured-output chokepoint
(``invoke_structured_with_validation_retry``) a ``max_tokens`` truncation is
detected post-call and a clear ``StructuredOutputTruncationError`` is raised
at the root cause — naming the call site / schema / token budget — instead of
being allowed to surface as a Pydantic ValidationError.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _raw(stop_reason="max_tokens", max_tokens=None):
    """A stand-in for the langchain-anthropic ``AIMessage`` carrying
    ``response_metadata`` — the truncation signal access path confirmed in
    this codebase (``evals/judge.py``, ``graph/llm_cost_tracker.py``)."""
    md = {"stop_reason": stop_reason, "model": "claude-haiku-4-5"}
    if max_tokens is not None:
        md["max_tokens"] = max_tokens
    msg = MagicMock()
    msg.response_metadata = md
    return msg


def _resp(parsed=None, parsing_error=None, raw=None):
    """Mimic the dict shape ``with_structured_output(include_raw=True)`` returns."""
    return {"parsed": parsed, "parsing_error": parsing_error, "raw": raw}


class TestRaiseIfTruncatedHelper:
    def test_raises_on_max_tokens_stop_reason(self):
        from agents.langchain_utils import (
            StructuredOutputTruncationError,
            raise_if_truncated,
        )

        resp = _resp(raw=_raw(stop_reason="max_tokens", max_tokens=512))
        with pytest.raises(StructuredOutputTruncationError) as exc_info:
            raise_if_truncated(resp, label="qual:tech:extract", schema_name="QualPillarBatch")

        msg = str(exc_info.value)
        # Root-cause diagnostics: call site, schema, budget all present.
        assert "qual:tech:extract" in msg
        assert "QualPillarBatch" in msg
        assert "512" in msg
        assert "max_tokens" in msg

    def test_noop_on_normal_stop_reason(self):
        from agents.langchain_utils import raise_if_truncated

        # end_turn / tool_use must NOT trip the guard.
        raise_if_truncated(_resp(raw=_raw(stop_reason="end_turn")), label="x")
        raise_if_truncated(_resp(raw=_raw(stop_reason="tool_use")), label="x")

    def test_noop_when_no_raw_or_metadata(self):
        from agents.langchain_utils import raise_if_truncated

        raise_if_truncated(_resp(raw=None), label="x")
        bare = MagicMock()
        bare.response_metadata = None
        raise_if_truncated(_resp(raw=bare), label="x")

    def test_length_alias_also_trips_guard(self):
        from agents.langchain_utils import (
            StructuredOutputTruncationError,
            raise_if_truncated,
        )

        # Defensive: an OpenAI-style ``length`` finish_reason must also fire.
        msg = MagicMock()
        msg.response_metadata = {"finish_reason": "length"}
        with pytest.raises(StructuredOutputTruncationError):
            raise_if_truncated(_resp(raw=msg), label="x")


class TestChokepointGuard:
    def test_truncated_response_raises_at_chokepoint(self):
        """A max_tokens truncation raises the clear error from the shared
        wrapper — NOT a confusing downstream Pydantic error."""
        from agents.langchain_utils import (
            StructuredOutputTruncationError,
            invoke_structured_with_validation_retry,
        )

        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _resp(
            parsed=None, parsing_error=None, raw=_raw(max_tokens=256),
        )

        with pytest.raises(StructuredOutputTruncationError):
            invoke_structured_with_validation_retry(
                structured_llm, ["msg"], label="qual:health:extract",
            )

    def test_truncation_raises_immediately_without_burning_retries(self):
        """Truncation is not re-promptable against the same budget, so the
        guard must raise on the FIRST attempt (one invoke), not exhaust the
        validation-retry budget."""
        from agents.langchain_utils import (
            StructuredOutputTruncationError,
            invoke_structured_with_validation_retry,
        )

        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _resp(raw=_raw(max_tokens=128))

        with pytest.raises(StructuredOutputTruncationError):
            invoke_structured_with_validation_retry(
                structured_llm, ["msg"], label="quant:tech:extract", max_retries=2,
            )
        # Exactly one LLM call — no validation-retry re-prompts.
        assert structured_llm.invoke.call_count == 1

    def test_truncation_takes_precedence_over_parsing_error(self):
        """Even when langchain ALSO populates a parsing_error (the confusing
        pydantic shape error), the truncation guard fires first so the
        operator sees the ROOT CAUSE, not the symptom."""
        from agents.langchain_utils import (
            StructuredOutputTruncationError,
            invoke_structured_with_validation_retry,
        )

        pydantic_like = ValueError(
            "catalysts: Input should be a valid list … input_type=str"
        )
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _resp(
            parsed=None, parsing_error=pydantic_like, raw=_raw(max_tokens=256),
        )

        with pytest.raises(StructuredOutputTruncationError) as exc_info:
            invoke_structured_with_validation_retry(
                structured_llm, ["msg"], label="qual:fin:extract",
            )
        # The raised error is the truncation root cause, not the pydantic symptom.
        assert "TRUNCATED" in str(exc_info.value)

    def test_clean_response_passes_through(self):
        """A non-truncated response is unaffected by the guard."""
        from agents.langchain_utils import invoke_structured_with_validation_retry

        parsed_obj = object()
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = _resp(
            parsed=parsed_obj, parsing_error=None, raw=_raw(stop_reason="tool_use"),
        )

        resp = invoke_structured_with_validation_retry(
            structured_llm, ["msg"], label="ok",
        )
        assert resp["parsed"] is parsed_obj
