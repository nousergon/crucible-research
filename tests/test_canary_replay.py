"""Tests for ``agents.canary_replay`` (alpha-engine-config#2246).

Mocks the LLM/archive boundary the same way
``tests/test_invoke_structured_with_validation_retry.py`` does — these
tests pin the canary's own orchestration logic (probe aggregation,
overall_status computation, fail-loud-on-empty-population, sentinel
trading-day validation), not the live API. A real spot-box run exercises
the live path; that isn't reproducible in a unit test.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestSentinelRunDate:
    def test_sentinel_is_a_real_trading_day(self):
        from agents.canary_replay import _assert_sentinel_is_trading_day

        # Must not raise — this pins the sentinel against calendar drift.
        _assert_sentinel_is_trading_day()

    def test_rejects_a_weekend_sentinel(self):
        from agents import canary_replay

        with patch.object(canary_replay, "CANARY_RUN_DATE", "2019-01-05"):  # Saturday
            with pytest.raises(RuntimeError, match="not a real NYSE trading day"):
                canary_replay._assert_sentinel_is_trading_day()


class TestProbeAggregation:
    def _fake_am(self, population):
        am = MagicMock()
        am.load_population.return_value = population
        am.load_latest_theses.return_value = {}
        return am

    def test_all_probes_pass_yields_overall_pass(self):
        from agents.canary_replay import run_canary

        population = [{"ticker": "AAPL", "long_term_score": 90, "sector": "Tech"}]
        am = self._fake_am(population)

        with (
            patch("agents.canary_replay._assert_sentinel_is_trading_day"),
            patch("archive.manager.ArchiveManager", return_value=am),
            patch(
                "agents.canary_replay.probe_thesis_update",
                return_value={"name": "thesis_update", "status": "PASS", "detail": "", "duration_s": 0.1},
            ),
            patch(
                "agents.canary_replay.probe_qual_analyst",
                return_value={"name": "qual_analyst", "status": "PASS", "detail": "", "duration_s": 0.1},
            ),
            patch(
                "agents.canary_replay.probe_validation_retry",
                return_value={"name": "validation_retry", "status": "PASS", "detail": "", "duration_s": 0.1},
            ),
        ):
            result = run_canary("test-run-id", n_tickers=5)

        assert result["overall_status"] == "PASS"
        assert result["held_tickers_probed"] == ["AAPL"]
        assert len(result["probes"]) == 3

    def test_one_probe_failing_yields_overall_fail(self):
        from agents.canary_replay import run_canary

        population = [{"ticker": "AAPL", "long_term_score": 90, "sector": "Tech"}]
        am = self._fake_am(population)

        with (
            patch("agents.canary_replay._assert_sentinel_is_trading_day"),
            patch("archive.manager.ArchiveManager", return_value=am),
            patch(
                "agents.canary_replay.probe_thesis_update",
                return_value={"name": "thesis_update", "status": "PASS", "detail": "", "duration_s": 0.1},
            ),
            patch(
                "agents.canary_replay.probe_qual_analyst",
                return_value={"name": "qual_analyst", "status": "FAIL", "detail": "boom", "duration_s": 0.1},
            ),
            patch(
                "agents.canary_replay.probe_validation_retry",
                return_value={"name": "validation_retry", "status": "PASS", "detail": "", "duration_s": 0.1},
            ),
        ):
            result = run_canary("test-run-id", n_tickers=5)

        assert result["overall_status"] == "FAIL"

    def test_empty_population_fails_loud_not_synthetic_fallback(self):
        from agents.canary_replay import run_canary

        am = self._fake_am([])

        with (
            patch("agents.canary_replay._assert_sentinel_is_trading_day"),
            patch("archive.manager.ArchiveManager", return_value=am),
        ):
            with pytest.raises(RuntimeError, match="population is empty"):
                run_canary("test-run-id", n_tickers=5)


class TestProbeThesisUpdate:
    def test_catches_and_reports_exception_as_fail(self):
        from agents.canary_replay import probe_thesis_update

        am = MagicMock()
        am.load_latest_theses.return_value = {}
        tickers = [{"ticker": "AAPL"}]

        with patch(
            "agents.sector_teams.sector_team._update_thesis_for_held_stock",
            side_effect=RuntimeError("simulated live failure"),
        ):
            result = probe_thesis_update(am, tickers)

        assert result["status"] == "FAIL"
        assert "simulated live failure" in result["detail"]


class TestProbeValidationRetry:
    def test_terminal_parsing_error_is_reported_as_fail(self):
        from agents.canary_replay import probe_validation_retry

        fake_prompt = MagicMock()
        fake_prompt.text = "mock canary validation-retry probe prompt"

        with (
            patch("langchain_anthropic.ChatAnthropic"),
            patch("agents.prompt_loader.load_prompt", return_value=fake_prompt),
            patch(
                "agents.langchain_utils.invoke_structured_with_validation_retry",
                return_value={"parsed": None, "parsing_error": "terminal mismatch", "raw": None},
            ),
        ):
            result = probe_validation_retry(api_key="fake-key")

        assert result["status"] == "FAIL"
        assert "terminal mismatch" in result["detail"]
