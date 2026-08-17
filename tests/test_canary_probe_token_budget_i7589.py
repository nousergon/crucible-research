"""The canary's validation_retry probe must not truncate (alpha-engine-config-I7589).

`probe_validation_retry` failed on every canary run:

    StructuredOutputTruncationError: [canary_replay:validation_retry] structured-output
    call was TRUNCATED by the max_tokens ceiling (stop_reason='max_tokens')

The cause is the probe's own design, which is otherwise exactly right. Its
prompt's semantically-correct answer is deliberately NOT one of the enum
values, so the model argues its way to a non-conforming answer and `reasoning`
grows; `invoke_structured_with_validation_retry` then re-prompts WITH the
validation error appended, so each attempt carries more context and less
headroom than the one before. Against `MAX_TOKENS_STRATEGIC` — a budget tuned
for strategic agent calls, borrowed here — the later attempts had nowhere to
land.

Two changes, and the tests below pin that neither of them removes the coverage
the probe exists to provide: the deliberate validation failure must still
happen. A probe made to pass by being made trivial is worse than a red one,
because it reports PASS for a retry path nobody exercised.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

SRC = (
    Path(__file__).resolve().parents[1] / "agents" / "canary_replay.py"
).read_text(encoding="utf-8")


def _module():
    import agents.canary_replay as m

    return m


class TestTheProbeHasItsOwnBudget:
    def test_it_no_longer_borrows_the_strategic_budget(self):
        """`MAX_TOKENS_STRATEGIC` is tuned for strategic agent calls and moves
        for reasons that have nothing to do with this probe.

        Asserted on IMPORT and USE, not on the string: the constant is named in
        a comment explaining why it was dropped, and that comment is the record
        of the decision — a test that forbids the name would delete the reason
        along with the coupling.
        """
        code = "\n".join(
            line for line in SRC.splitlines() if not line.lstrip().startswith("#")
        )
        assert "MAX_TOKENS_STRATEGIC" not in code

    def test_the_budget_is_a_named_constant(self):
        assert _module()._CANARY_PROBE_MAX_TOKENS > 0

    def test_it_is_larger_than_the_budget_that_truncated(self):
        """4096 was the observed ceiling. A budget that is not strictly larger
        would leave the probe exactly where it was."""
        assert _module()._CANARY_PROBE_MAX_TOKENS > 4096

    def test_the_probe_actually_uses_it(self):
        src = inspect.getsource(_module().probe_validation_retry)
        assert "max_tokens=_CANARY_PROBE_MAX_TOKENS" in src


class TestReasoningIsBounded:
    def test_the_field_carries_a_length_bound(self):
        """On the SCHEMA, not only in the prompt: a prompt instruction is
        advice, a field constraint is a contract."""
        field = _module()._CanaryConfidenceProbe.model_fields["reasoning"]
        bounds = [
            getattr(m, "max_length", None)
            for m in getattr(field, "metadata", [])
            if getattr(m, "max_length", None) is not None
        ]
        assert bounds, "reasoning has no max_length — it can consume the budget"
        assert 0 < bounds[0] <= 2000

    def test_a_too_long_reasoning_is_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _module()._CanaryConfidenceProbe(confidence="low", reasoning="x" * 5000)

    def test_a_normal_reasoning_is_accepted(self):
        probe = _module()._CanaryConfidenceProbe(
            confidence="low", reasoning="Not enough evidence either way."
        )
        assert probe.confidence == "low"


class TestTheCoverageIsNotRemoved:
    """The reason this probe exists is that it forces a real validation failure
    against the live API. Making it pass by making it trivial would delete
    exactly that."""

    def test_the_enum_is_still_closed(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _module()._CanaryConfidenceProbe(confidence="uncertain", reasoning="…")

    def test_the_enum_still_has_the_three_original_values(self):
        import typing

        field = _module()._CanaryConfidenceProbe.model_fields["confidence"]
        assert set(typing.get_args(field.annotation)) == {"low", "medium", "high"}

    def test_it_still_goes_through_the_retry_chokepoint(self):
        src = inspect.getsource(_module().probe_validation_retry)
        assert "invoke_structured_with_validation_retry" in src

    def test_the_probe_still_reports_fail_on_a_terminal_validation_failure(self):
        src = inspect.getsource(_module().probe_validation_retry)
        assert "terminal validation failure after retries" in src
