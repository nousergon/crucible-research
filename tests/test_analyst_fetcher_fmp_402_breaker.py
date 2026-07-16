"""Tests for the FMP 402 circuit breaker in data/fetchers/analyst_fetcher.py.

config#1821 (2026-05, part a): FMP's ``grades-consensus`` /
``price-target-consensus`` stable-API endpoints returned 402 Payment
Required (current plan doesn't cover these endpoints) — for every ticker,
every run. Two failure modes fixed here (#391):

1. Wasted runtime: without a breaker, every ticker re-attempts (and
   ``resp.raise_for_status()`` on a 402 is a real HTTP round-trip each
   time) — ~74s of serial 402s inside fetch_data alone.
2. Silent data hole: per-ticker ``logger.warning`` spam with no
   run-level signal that the endpoint is structurally dead this run.

The breaker itself is generic (keyed on the bare endpoint name), so these
tests exercise it directly via ``_fmp_get`` with placeholder endpoint
names rather than the two real endpoints above — config#1821 Option B
(2026-07-08, part b) subsequently removed the ``grades-consensus`` /
``price-target-consensus`` calls from ``fetch_analyst_consensus``
entirely (see ``test_fetch_analyst_consensus.py``), but the breaker
mechanism remains live infrastructure for any other FMP endpoint.

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
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
        assert mock_get.call_count == 1
        assert af._fmp_402_tripped["test-endpoint-a"] is True
        assert af.fmp_402_skip_counts() == {"test-endpoint-a": 1}

    def test_402_is_never_retried(self, _mock_secret):
        """The gotcha from the issue: 402 is deterministic per-plan, not
        transient. A single 402 must not trigger any retry/backoff —
        exactly one HTTP call for the whole ``_fmp_get`` invocation."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)) as mock_get, \
             patch("data.fetchers.analyst_fetcher.time.sleep") as mock_sleep:
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
        assert mock_get.call_count == 1
        # No backoff sleep should ever be invoked for a 402.
        mock_sleep.assert_not_called()

    def test_subsequent_tickers_skip_with_no_http_call(self, _mock_secret):
        """After the breaker trips on ticker 1, ticker 2/3/... must not
        make any HTTP request at all — the call short-circuits."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)) as mock_get:
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
            assert mock_get.call_count == 1

            for ticker in ("MSFT", "GOOG", "NVDA"):
                with pytest.raises(af.FMPPlanLimitedError):
                    af._fmp_get("test-endpoint-a", {"symbol": ticker})
            # No new HTTP calls made for the 3 subsequent tickers.
            assert mock_get.call_count == 1

        assert af.fmp_402_skip_counts() == {"test-endpoint-a": 4}

    def test_exactly_one_warn_log_for_repeated_402s(self, _mock_secret, caplog):
        """ONE summary WARN per endpoint, not per-ticker spam — the
        core ask from the issue."""
        import logging
        caplog.set_level(logging.WARNING, logger="data.fetchers.analyst_fetcher")

        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)):
            for ticker in ("AAPL", "MSFT", "GOOG"):
                with pytest.raises(af.FMPPlanLimitedError):
                    af._fmp_get("test-endpoint-a", {"symbol": ticker})

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) == 1
        assert "402" in warn_records[0].message

    def test_breaker_is_per_endpoint_not_global(self, _mock_secret):
        """Tripping test-endpoint-a must not affect test-endpoint-b
        (or any other endpoint) — each endpoint gets its own breaker."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)):
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})

        assert af._fmp_402_tripped.get("test-endpoint-b") is not True
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"targetConsensus": 100.0}])) as mock_get:
            data = af._fmp_get("test-endpoint-b", {"symbol": "AAPL"})
        assert data == [{"targetConsensus": 100.0}]
        assert mock_get.call_count == 1

    def test_fetch_analyst_consensus_never_calls_removed_endpoints(self, _mock_secret, caplog):
        """Regression guard for config#1821 Option B (2026-07-08):
        ``fetch_analyst_consensus`` must never call ``grades-consensus``
        or ``price-target-consensus`` again — those calls (and the
        fields they populated) were removed from the feature contract
        because the endpoints 402'd for every ticker on the current
        plan. Only the earnings-surprises v3 endpoint should be hit, and
        the breaker state for the two removed endpoints must never be
        touched."""
        import logging
        caplog.set_level(logging.WARNING, logger="data.fetchers.analyst_fetcher")

        call_log = []

        def _fake_get(url, params=None, timeout=None):
            call_log.append(url)
            return _resp(200, [])

        with patch("data.fetchers.analyst_fetcher.requests.get", side_effect=_fake_get):
            for ticker in ("AAPL", "MSFT", "GOOG"):
                result = af.fetch_analyst_consensus(ticker)
                assert set(result.keys()) == {"ticker", "current_price", "earnings_surprises"}

        assert all("earning_surprises" in u for u in call_log)
        assert len(call_log) == 3  # one earnings-surprises call per ticker, nothing else

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) == 0
        assert af.fmp_402_skip_counts() == {}

    def test_reset_breaker_clears_state_for_new_run(self, _mock_secret):
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(402)):
            with pytest.raises(af.FMPPlanLimitedError):
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
        assert af.fmp_402_skip_counts() == {"test-endpoint-a": 1}

        af.reset_fmp_402_breaker()
        assert af.fmp_402_skip_counts() == {}

        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"consensus": "buy"}])) as mock_get:
            data = af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
        assert data == [{"consensus": "buy"}]
        assert mock_get.call_count == 1


@patch("data.fetchers.analyst_fetcher.get_secret", return_value="fake-key")
class TestExisting429And5xxSemanticsUndisturbed:
    """Prove the 402 breaker does not touch 429/5xx handling at all."""

    def test_429_still_raises_fmp_daily_limit_error_and_maxes_out_counter(self, _mock_secret):
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(429)):
            with pytest.raises(af.FMPDailyLimitError):
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
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
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})

        # Budget now exhausted (real, pre-existing 429 behavior) — reset it
        # to isolate just the breaker assertion.
        af._fmp_daily_count = 0
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"consensus": "hold"}])) as mock_get:
            data = af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
        assert data == [{"consensus": "hold"}]
        assert mock_get.call_count == 1

    def test_5xx_still_raises_http_error_not_plan_limited(self, _mock_secret):
        """A plain 500 must surface as a normal HTTPError via
        ``raise_for_status`` — not treated as a 402/breaker case, and
        not silently swallowed."""
        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(500)):
            with pytest.raises(requests.HTTPError):
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
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
                af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
        assert mock_get.call_count == 1

        with patch("data.fetchers.analyst_fetcher.requests.get", return_value=_resp(200, [{"consensus": "buy"}])) as mock_get2:
            data = af._fmp_get("test-endpoint-a", {"symbol": "AAPL"})
        assert data == [{"consensus": "buy"}]
        assert mock_get2.call_count == 1
