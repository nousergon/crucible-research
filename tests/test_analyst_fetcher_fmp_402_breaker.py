"""Tests for the FMP 402 circuit breaker in data/fetchers/analyst_fetcher.py.

config#1821: ALL FMP ``grades-consensus`` / ``price-target-consensus`` calls
return 402 Payment Required (current plan doesn't cover these endpoints) —
for every ticker, every run. Two failure modes fixed here:

1. Wasted runtime: without a breaker, every ticker re-attempts (and
   ``resp.raise_for_status()`` on a 402 is a real HTTP round-trip each
   time) — ~74s of serial 402s inside fetch_data alone.
2. Silent data hole: per-ticker ``logger.warning`` spam with no
   run-level signal that the endpoint is structurally dead this run.

These tests pin:
* A 402 trips the breaker on the FIRST call for that endpoint, logs
  exactly one WARN, and every subsequent call to the SAME endpoint
  short-circuits — zero additional HTTP requests, zero additional WARN
  logs (only DEBUG) — while incrementing a per-endpoint skip counter
  surfaced via ``fmp_402_skip_counts()``.
* 402 is NEVER retried/backed off (the gotcha from the issue — it's a
  deterministic per-plan rejection, not transient).
* 429 (quota exhausted) keeps its existing, separate semantics
  completely undisturbed — it must NOT trip the 402 breaker or be
  affected by it in any way.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from data.fetchers import analyst_fetcher as af


@pytest.fixture(autouse=True)
def _reset_breaker_and_budget():
    """Isolate module-level state across tests (mirrors the existing
    ``_fmp_daily_count`` module-global idiom already used in this file)."""
    af.reset_fmp_402_breaker()
    af._fmp_daily_count = 0
    af._fmp_last_call = 0.0
    yield
    af.reset_fmp_402_breaker()
    af._fmp_daily_count = 0
    af._fmp_last_call = 0.0


def _resp(status_code: int, payload=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.json.return_value = payload if payload is not None else []
    if status_code >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    else:
        r.raise_for_status.side_effect = None
    return r


@patch("data.fetchers.analyst_fetcher.get_secret", return_value="fake-key")
class TestFMP402Breaker:
    def test_first_402_trips_breaker_and_raises_plan_limited(self, _mock_secret):
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)) as mock_get:
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert mock_get.call_count == 1
        assert af._fmp_402_tripped["grades-consensus"] is True
        assert af.fmp_402_skip_counts() == {"grades-consensus": 1}

    def test_402_is_never_retried(self, _mock_secret):
        """The gotcha from the issue: 402 is deterministic per-plan, not
        transient. A single 402 must not trigger any retry/backoff —
        exactly one HTTP call for the whole ``_fmp_get`` invocation."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)) as mock_get, \
             patch("data.fetchers.analyst_fetcher.time.sleep") as mock_sleep:
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert mock_get.call_count == 1
        # No backoff sleep should ever be invoked for a 402.
        mock_sleep.assert_not_called()

    def test_subsequent_tickers_skip_with_no_http_call(self, _mock_secret):
        """After the breaker trips on ticker 1, ticker 2/3/... must not
        make any HTTP request at all — the call short-circuits."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)) as mock_get:
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})
            assert mock_get.call_count == 1

            for ticker in ("MSFT", "GOOG", "NVDA"):
                with pytest.raises(af.FMPPlanLimitedError):
                    af._fmp_get("grades-consensus", {"symbol": ticker})
            # No new HTTP calls made for the 3 subsequent tickers.
            assert mock_get.call_count == 1

        assert af.fmp_402_skip_counts() == {"grades-consensus": 4}

    def test_exactly_one_warn_log_for_repeated_402s(self, _mock_secret, caplog):
        """ONE summary WARN per endpoint, not per-ticker spam — the
        core ask from the issue."""
        import logging
        caplog.set_level(logging.WARNING, logger="data.fetchers.analyst_fetcher")

        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)):
            for ticker in ("AAPL", "MSFT", "GOOG"):
                with pytest.raises(af.FMPPlanLimitedError):
                    af._fmp_get("grades-consensus", {"symbol": ticker})

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) == 1
        assert "402" in warn_records[0].message

    def test_breaker_is_per_endpoint_not_global(self, _mock_secret):
        """Tripping grades-consensus must not affect price-target-consensus
        (or any other endpoint) — each endpoint gets its own breaker."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)):
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})

        assert af._fmp_402_tripped.get("price-target-consensus") is not True
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"targetConsensus": 100.0}])) as mock_get:
            data = af._fmp_get("price-target-consensus", {"symbol": "AAPL"})
        assert data == [{"targetConsensus": 100.0}]
        assert mock_get.call_count == 1

    def test_fetch_analyst_consensus_end_to_end_breaker_trip_then_skip(self, _mock_secret, caplog):
        """Full ``fetch_analyst_consensus`` call path: first ticker trips
        both endpoint breakers (one WARN each), subsequent tickers return
        the skeleton dict with no additional WARNs and no HTTP calls for
        the tripped endpoints."""
        import logging
        caplog.set_level(logging.WARNING, logger="data.fetchers.analyst_fetcher")

        call_log = []

        def _fake_get(url, params=None, timeout=None):
            call_log.append(url)
            if "earning_surprises" in url:
                return _resp(200, [])
            return _resp(402)

        with patch("data.fetchers.analyst_fetcher.requests.get", side_effect=_fake_get):
            for ticker in ("AAPL", "MSFT", "GOOG"):
                result = af.fetch_analyst_consensus(ticker)
                assert result["consensus_rating"] is None
                assert result["mean_target"] is None

        # 2 endpoints x 1 real HTTP 402 call each (breaker trips after AAPL),
        # plus 1 earnings-surprises call per ticker (unaffected endpoint).
        grades_calls = [u for u in call_log if "grades-consensus" in u]
        pt_calls = [u for u in call_log if "price-target-consensus" in u]
        assert len(grades_calls) == 1
        assert len(pt_calls) == 1

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) == 2  # one per endpoint, ever
        assert af.fmp_402_skip_counts() == {
            "grades-consensus": 3,
            "price-target-consensus": 3,
        }

    def test_reset_breaker_clears_state_for_new_run(self, _mock_secret):
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)):
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert af.fmp_402_skip_counts() == {"grades-consensus": 1}

        af.reset_fmp_402_breaker()
        assert af.fmp_402_skip_counts() == {}

        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"consensus": "buy"}])) as mock_get:
            data = af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert data == [{"consensus": "buy"}]
        assert mock_get.call_count == 1


@patch("data.fetchers.analyst_fetcher.get_secret", return_value="fake-key")
class TestExisting429And5xxSemanticsUndisturbed:
    """Prove the 402 breaker does not touch 429/5xx handling at all."""

    def test_429_still_raises_fmp_daily_limit_error_and_maxes_out_counter(self, _mock_secret):
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(429)):
            with pytest.raises(af.FMPDailyLimitError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        # 429 semantics unchanged: daily counter forced to the limit.
        assert af._fmp_daily_count == af._FMP_DAILY_LIMIT
        assert af.fmp_budget_exhausted() is True
        # The 402 breaker must NOT have been touched by a 429.
        assert af._fmp_402_tripped == {}
        assert af.fmp_402_skip_counts() == {}

    def test_429_does_not_trip_402_breaker_for_other_calls(self, _mock_secret):
        """A 429 on one endpoint must not cause a different (or the
        same) endpoint's calls to be treated as circuit-broken."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(429)):
            with pytest.raises(af.FMPDailyLimitError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})

        # Budget now exhausted (real, pre-existing 429 behavior) — reset it
        # to isolate just the breaker assertion.
        af._fmp_daily_count = 0
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"consensus": "hold"}])) as mock_get:
            data = af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert data == [{"consensus": "hold"}]
        assert mock_get.call_count == 1

    def test_5xx_still_raises_http_error_not_plan_limited(self, _mock_secret):
        """A plain 500 must surface as a normal HTTPError via
        ``raise_for_status`` — not treated as a 402/breaker case, and
        not silently swallowed."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(500)):
            with pytest.raises(requests.HTTPError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert af._fmp_402_tripped == {}
        assert af.fmp_402_skip_counts() == {}

    def test_5xx_retry_loop_semantics_unchanged(self, _mock_secret):
        """Confirm the pre-existing retry `for` loop structure around
        the HTTP call is untouched: a 500 followed by a 200 on a second
        manual invocation still succeeds via the same code path (loop
        body itself doesn't swallow-and-continue past raise_for_status,
        matching pre-change behavior — this test pins that we didn't
        change that pre-existing shape while adding the 402 branch)."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(500)) as mock_get:
            with pytest.raises(requests.HTTPError):
                af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert mock_get.call_count == 1

        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"consensus": "buy"}])) as mock_get2:
            data = af._fmp_get("grades-consensus", {"symbol": "AAPL"})
        assert data == [{"consensus": "buy"}]
        assert mock_get2.call_count == 1
