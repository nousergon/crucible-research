"""A canary probe FAILs when its subject did not happen.

Both defects these tests pin were found the same way — by reading a marker
whose probes disagreed with their own logs (alpha-engine-config-I7459, I7463):

* ``qual_analyst`` reported ``PASS / 0 assessments returned`` while every call
  in the run was answered ``401 Authorization Required``. ``run_qual_analyst``
  swallows its own failures by design so the score aggregator can treat a dead
  team as degraded, and the probe only asked "did it raise".
* ``validation_retry`` reported PASS whether or not the injected validation
  failure ever tripped, so it could not tell "the recovery path works" from
  "the recovery path never ran" — and the probe exists to assert the former.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# The `TestQualAnalystProbeRejectsEmptyResults` class was REMOVED 2026-08-20
# (alpha-engine-config-I7816, I7817) together with `probe_qual_analyst`. It
# guarded a probe that delegated to `run_qual_analyst`, part of the multi-agent
# research path retired by Brian's 2026-07-27 ruling. Its assertions were good
# ones — they are what turned I7463's false PASS into a real FAIL — but a test
# whose subject is retired is not fresh, current or accurate, and keeping it
# made a TEST the sole evidence that a retired component was live (the I7011 §2
# reversal). Deleted with its subject rather than kept as coverage of nothing.


class TestThesisUpdateProbeRejectsUnratedTickers:
    @staticmethod
    def _probe(ratings: list):
        from agents.canary_replay import probe_thesis_update

        am = MagicMock()
        am.load_latest_theses.return_value = {}
        tickers = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]

        with patch(
            "agents.sector_teams.sector_team._update_thesis_for_held_stock",
            side_effect=[{"rating": r} for r in ratings],
        ):
            return probe_thesis_update(am, tickers)

    def test_none_ratings_are_a_fail(self):
        out = self._probe([None, None])
        assert out["status"] == "FAIL"
        assert "0/2" in out["detail"]

    def test_partial_ratings_are_a_fail(self):
        out = self._probe(["BUY", None])
        assert out["status"] == "FAIL"
        assert "1/2" in out["detail"]

    def test_all_rated_passes(self):
        out = self._probe(["BUY", "HOLD"])
        assert out["status"] == "PASS"


class TestValidationRetryProbeRequiresARetry:
    """The I7459 shape: a probe whose subject is the RECOVERY path must not
    pass on a run where nothing needed recovering."""

    @staticmethod
    def _probe(resp: dict):
        from agents.canary_replay import probe_validation_retry

        fake_prompt = MagicMock()
        fake_prompt.text = "mock canary validation-retry probe prompt"

        with (
            patch("agents.prompt_loader.load_prompt", return_value=fake_prompt),
            patch(
                "agents.langchain_utils.invoke_structured_with_validation_retry",
                return_value=resp,
            ),
        ):
            return probe_validation_retry(api_key="fake-key")

    def test_first_attempt_success_is_a_fail(self):
        parsed = MagicMock()
        parsed.confidence = "medium"
        out = self._probe(
            {"parsed": parsed, "parsing_error": None, "structured_output_attempts": 1}
        )
        assert out["status"] == "FAIL"
        assert "FIRST attempt" in out["detail"]

    @pytest.mark.parametrize("attempts", [2, 3])
    def test_recovery_passes_and_reports_the_attempt(self, attempts):
        parsed = MagicMock()
        parsed.confidence = "medium"
        out = self._probe(
            {
                "parsed": parsed,
                "parsing_error": None,
                "structured_output_attempts": attempts,
            }
        )
        assert out["status"] == "PASS"
        assert f"attempt {attempts}" in out["detail"]

    def test_terminal_failure_still_fails(self):
        out = self._probe(
            {"parsed": None, "parsing_error": "terminal mismatch", "structured_output_attempts": 3}
        )
        assert out["status"] == "FAIL"
        assert "terminal mismatch" in out["detail"]


class TestRetryHelperReportsAttempts:
    """The probe assertions above are only meaningful if the chokepoint
    actually reports the count."""

    def test_success_on_first_attempt_reports_one(self):
        from agents.langchain_utils import invoke_structured_with_validation_retry

        with patch(
            "agents.langchain_utils.invoke_anthropic_safe",
            return_value={"parsed": MagicMock(), "parsing_error": None, "raw": None},
        ):
            resp = invoke_structured_with_validation_retry(
                MagicMock(), [MagicMock()], label="t"
            )
        assert resp["structured_output_attempts"] == 1

    def test_recovery_reports_the_attempt_it_succeeded_on(self):
        from agents.langchain_utils import invoke_structured_with_validation_retry

        responses = [
            {"parsed": None, "parsing_error": ValueError("bad enum"), "raw": None},
            {"parsed": MagicMock(), "parsing_error": None, "raw": None},
        ]
        with patch(
            "agents.langchain_utils.invoke_anthropic_safe", side_effect=responses
        ):
            resp = invoke_structured_with_validation_retry(
                MagicMock(), [MagicMock()], label="t"
            )
        assert resp["structured_output_attempts"] == 2

    def test_terminal_failure_reports_every_attempt_spent(self):
        from agents.langchain_utils import invoke_structured_with_validation_retry

        bad = {"parsed": None, "parsing_error": ValueError("bad enum"), "raw": None}
        with patch(
            "agents.langchain_utils.invoke_anthropic_safe", side_effect=[bad, bad, bad]
        ):
            resp = invoke_structured_with_validation_retry(
                MagicMock(), [MagicMock()], label="t", max_retries=2
            )
        assert resp["structured_output_attempts"] == 3


class TestNoToolCallIsRetried:
    """`tool_choice="auto"` lets a model answer in prose. That must be a RETRY,
    not a terminal failure whose own error reads `None`.

    Measured on the canary box 2026-08-16 (marker
    ``pr-nousergon-crucible-research-638-472bfa7c04b4``): the validation-retry
    probe reported ``terminal validation failure after retries: None`` — the
    loop keyed entirely on ``parsing_error``, so a response with no tool call
    (``parsed=None, parsing_error=None``) returned on the first attempt with
    nothing to act on. Only reachable since structured output moved to
    ``tool_choice="auto"``; a forced tool choice cannot produce it, and a
    reasoning model rejects a forced tool choice.
    """

    def test_prose_answer_is_retried_and_can_recover(self):
        from agents.langchain_utils import invoke_structured_with_validation_retry

        parsed = MagicMock()
        responses = [
            # No tool call: nothing parsed, and no error to explain it.
            {"parsed": None, "parsing_error": None, "raw": None},
            {"parsed": parsed, "parsing_error": None, "raw": None},
        ]
        with patch(
            "agents.langchain_utils.invoke_anthropic_safe", side_effect=responses
        ):
            resp = invoke_structured_with_validation_retry(
                MagicMock(), [MagicMock()], label="t"
            )

        assert resp["parsed"] is parsed
        assert resp["structured_output_attempts"] == 2, (
            "a missing tool call must consume a retry — returning on attempt 1 "
            "is the defect this pins"
        )

    def test_persistent_prose_fails_with_an_actionable_error(self):
        from agents.langchain_utils import invoke_structured_with_validation_retry

        none_call = {"parsed": None, "parsing_error": None, "raw": None}
        with patch(
            "agents.langchain_utils.invoke_anthropic_safe",
            side_effect=[none_call, none_call, none_call],
        ):
            resp = invoke_structured_with_validation_retry(
                MagicMock(), [MagicMock()], label="t", max_retries=2
            )

        assert resp["structured_output_attempts"] == 3
        assert resp["parsing_error"] is not None, (
            "the terminal error must say what went wrong; `None` is what the "
            "caller used to report"
        )
        assert "no tool call" in str(resp["parsing_error"])


class TestReasoningHeadroom:
    """A thinking model spends the SAME max_tokens on its chain of thought, so
    every budget sized against Anthropic silently shrank when these agents
    moved onto the router (I7005). Measured: the qual-analyst pillar extraction
    died with StructuredOutputTruncationError at `stop_reason=max_tokens`,
    mid-tool-call."""

    @staticmethod
    def _spec(reasoning, max_tokens=4096):
        spec = MagicMock()
        spec.reasoning = reasoning
        spec.max_tokens = max_tokens
        spec.model = "stub"
        return spec

    def test_no_reasoning_gets_no_headroom(self):
        from agents.langchain_utils import _with_reasoning_headroom

        assert _with_reasoning_headroom(self._spec(None), model_class="low") == 4096

    def test_excluded_reasoning_gets_no_headroom(self):
        from agents.langchain_utils import _with_reasoning_headroom

        assert (
            _with_reasoning_headroom(self._spec({"exclude": True}), model_class="low")
            == 4096
        )

    def test_thinking_model_gets_headroom_on_top_of_the_callers_budget(self):
        from agents.langchain_utils import (
            REASONING_OUTPUT_HEADROOM_TOKENS,
            _with_reasoning_headroom,
        )

        got = _with_reasoning_headroom(
            self._spec({"effort": "max"}), model_class="high"
        )
        assert got == 4096 + REASONING_OUTPUT_HEADROOM_TOKENS, (
            "the caller's number must keep meaning 'answer tokens' — headroom is "
            "added, not substituted"
        )
