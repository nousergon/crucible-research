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


class TestQualAnalystProbeRejectsEmptyResults:
    """The I7463 shape: a swallowed failure must not read as a pass."""

    @staticmethod
    def _probe(result: dict):
        from agents.canary_replay import probe_qual_analyst

        am = MagicMock()
        am.load_latest_theses.return_value = {}
        tickers = [{"ticker": "AAPL", "long_term_score": 1}, {"ticker": "MSFT", "long_term_score": 2}]

        with patch(
            "agents.sector_teams.qual_analyst.run_qual_analyst", return_value=result
        ):
            return probe_qual_analyst(am, tickers)

    def test_swallowed_error_is_a_fail(self):
        """The exact 2026-08-16 payload: the ReAct loop died on a 401 and
        returned an empty, error-carrying dict rather than raising."""
        out = self._probe(
            {
                "assessments": [],
                "error": "AuthenticationError: 401 Authorization Required",
                "partial": False,
            }
        )
        assert out["status"] == "FAIL"
        assert "401" in out["detail"]

    def test_zero_assessments_is_a_fail_even_with_no_error(self):
        out = self._probe({"assessments": [], "error": None, "partial": False})
        assert out["status"] == "FAIL", (
            "zero assessments for real held tickers is a failure however it was "
            "produced — this is the 'plausible zero' the probe used to pass on"
        )

    def test_partial_is_a_fail(self):
        out = self._probe(
            {
                "assessments": [{"ticker": "AAPL"}],
                "error": None,
                "partial": True,
                "partial_reason": "recursion_limit_exhausted",
            }
        )
        assert out["status"] == "FAIL"
        assert "recursion_limit_exhausted" in out["detail"]

    def test_full_result_still_passes(self):
        out = self._probe(
            {
                "assessments": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
                "error": None,
                "partial": False,
            }
        )
        assert out["status"] == "PASS"


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
