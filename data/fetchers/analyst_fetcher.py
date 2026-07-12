"""
Analyst consensus fetcher — Financial Modeling Prep (FMP) stable + v3 API.
Free tier: 250 requests/day.

config#1821 Option B (operator ruling, 2026-07-08): the ``grades-consensus``
and ``price-target-consensus`` stable-API endpoints returned 402 Payment
Required for every ticker on the current FMP plan (not covered by the
plan's entitlements) and were removed from the research feature contract
rather than upgrading the plan — they were not being used anyway once the
per-endpoint 402 circuit breaker (#391) made them a permanent no-op skip.
``fetch_analyst_consensus`` no longer calls either endpoint; it now only
fetches earnings surprises from the v3 API for PEAD scoring (O10).
"""

from __future__ import annotations

import json as _json
import logging
import threading
import time
from datetime import date, datetime
from typing import Optional

import boto3
import requests

from nousergon_lib.secrets import get_secret

logger = logging.getLogger(__name__)

_FMP_STABLE = "https://financialmodelingprep.com/stable"
_FMP_V3 = "https://financialmodelingprep.com/api/v3"
_TIMEOUT = 10

# S3 persistence for FMP daily counter (survives Lambda cold starts)
_FMP_COUNTER_BUCKET = "alpha-engine-research"
_FMP_COUNTER_KEY = "health/fmp_daily_count.json"


def _load_fmp_counter() -> int:
    """Load today's FMP call count from S3. Returns 0 if not found or stale."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=_FMP_COUNTER_BUCKET, Key=_FMP_COUNTER_KEY)
        data = _json.loads(obj["Body"].read())
        if data.get("date") == str(date.today()):
            return data.get("count", 0)
    except Exception:
        pass
    return 0


def _save_fmp_counter(count: int) -> None:
    """Persist FMP call count to S3."""
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=_FMP_COUNTER_BUCKET,
            Key=_FMP_COUNTER_KEY,
            Body=_json.dumps({"date": str(date.today()), "count": count}).encode(),
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning("Failed to persist FMP counter: %s", e)


# Rate limiter: FMP free tier = 250 req/day.
# With 6 sector teams calling in parallel, we need a global lock and daily counter.
_fmp_lock = threading.Lock()
_fmp_last_call = 0.0
_fmp_daily_count = _load_fmp_counter()
_FMP_MIN_INTERVAL = 1.0  # 1s between calls — spreads 250 daily quota over ~4 min
_FMP_DAILY_LIMIT = 250  # FMP free tier hard limit; FMP returns 429 if exceeded
_FMP_MAX_RETRIES = 3
_FMP_RETRY_BACKOFF = 5.0  # seconds, doubles each retry


class FMPDailyLimitError(RuntimeError):
    """Raised when the FMP daily request budget is exhausted."""
    pass


class FMPPlanLimitedError(RuntimeError):
    """Raised when an FMP endpoint returns 402 (not covered by current plan).

    Distinct from ``FMPDailyLimitError``: a 402 is a deterministic,
    per-plan rejection (this endpoint is simply not entitled), not a
    transient/quota condition. It must never be retried — see the
    per-endpoint circuit breaker below.
    """
    pass


def fmp_budget_exhausted() -> bool:
    """Check if the FMP daily budget has been used up."""
    return _fmp_daily_count >= _FMP_DAILY_LIMIT


# ── 402 circuit breaker ───────────────────────────────────────────────────────
# FMP returns 402 Payment Required for endpoints the current plan doesn't
# cover. This is a deterministic per-plan rejection, not a transient
# failure — retrying it just burns the per-run wall clock (Research Lambda
# has a 900s ceiling) for a guaranteed repeat failure. Trip a breaker per
# endpoint on the first 402 seen in a run; every subsequent call to that
# endpoint short-circuits (no HTTP call, no log spam) and increments a skip
# counter that the run summary can report.
#
# Originally added (#391) for ``grades-consensus`` / ``price-target-consensus``,
# which 402'd for every ticker under the current plan; config#1821 Option B
# (2026-07-08) subsequently removed those two endpoints from the feature
# contract entirely (see module docstring) rather than upgrading the plan.
# The breaker itself is kept as generic infrastructure — it still protects
# any other FMP endpoint (e.g. ``revision_fetcher.py``'s shared ``_fmp_get``)
# that hits a plan-level 402 in the future.
#
# Module-level state mirrors the existing ``_fmp_daily_count`` idiom above —
# this codebase already tracks FMP run/day state at module scope rather than
# threading a context object through the fetch call chain, so the breaker
# follows the same pattern for consistency. ``reset_fmp_402_breaker()`` lets
# a fresh Lambda invocation (or a test) start from a clean slate.
_fmp_402_tripped: dict[str, bool] = {}
_fmp_402_skipped_count: dict[str, int] = {}


def reset_fmp_402_breaker() -> None:
    """Clear all per-endpoint 402 breaker state. Call at the start of a run."""
    _fmp_402_tripped.clear()
    _fmp_402_skipped_count.clear()


def fmp_402_skip_counts() -> dict[str, int]:
    """Return a copy of the per-endpoint 402-skip counters for the run summary.

    Keys are FMP endpoint names (e.g. ``analyst-estimates``); values are the
    number of calls short-circuited by the breaker because that endpoint had
    already tripped on a 402 earlier in the run.
    """
    return dict(_fmp_402_skipped_count)


def _fmp_get(endpoint: str, params: Optional[dict] = None, base: str = _FMP_STABLE) -> dict | list:
    global _fmp_last_call, _fmp_daily_count

    # 402 circuit breaker: if this endpoint already tripped earlier in the
    # run, skip the call entirely — no HTTP request, no per-call log line.
    # Checked before the API key / rate-limit bookkeeping so a tripped
    # endpoint costs nothing on repeat tickers. Keyed on the bare endpoint
    # name (e.g. "analyst-estimates"), not the full URL with params, since
    # the 402 is a plan-level rejection independent of the ticker.
    if _fmp_402_tripped.get(endpoint):
        _fmp_402_skipped_count[endpoint] = _fmp_402_skipped_count.get(endpoint, 0) + 1
        raise FMPPlanLimitedError(
            f"FMP {endpoint} circuit-broken after 402 earlier this run — skipping"
        )

    api_key = get_secret("FMP_API_KEY", required=False, default="")
    if not api_key:
        raise RuntimeError("FMP_API_KEY environment variable not set.")

    url = f"{base}/{endpoint}"
    p = {"apikey": api_key}
    if params:
        p.update(params)

    for attempt in range(_FMP_MAX_RETRIES):
        # Rate limit: enforce minimum interval and daily budget
        with _fmp_lock:
            if _fmp_daily_count >= _FMP_DAILY_LIMIT:
                raise FMPDailyLimitError(
                    f"FMP daily budget exhausted ({_fmp_daily_count}/{_FMP_DAILY_LIMIT})"
                )
            now = time.monotonic()
            wait = _FMP_MIN_INTERVAL - (now - _fmp_last_call)
            if wait > 0:
                time.sleep(wait)
            _fmp_last_call = time.monotonic()
            _fmp_daily_count += 1
            if _fmp_daily_count % 25 == 0:
                _save_fmp_counter(_fmp_daily_count)

        resp = requests.get(url, params=p, timeout=_TIMEOUT)

        if resp.status_code == 402:
            # 402 Payment Required: this endpoint is not covered by the
            # current FMP plan. Deterministic per-plan rejection, NOT
            # transient — do not retry/backoff. Trip the breaker so every
            # remaining ticker this run skips the call outright, log
            # exactly ONE summary WARN (not per-ticker spam), and count
            # this first occurrence as a skip too so the run-summary
            # counter reflects total calls avoided.
            _fmp_402_tripped[endpoint] = True
            _fmp_402_skipped_count[endpoint] = _fmp_402_skipped_count.get(endpoint, 0) + 1
            logger.warning(
                "FMP %s returned 402 (not covered by current plan) — "
                "circuit-breaking this endpoint for the rest of the run",
                endpoint,
            )
            raise FMPPlanLimitedError(f"FMP {endpoint} returned 402 — plan does not cover this endpoint")

        if resp.status_code == 429:
            # 429 means daily quota is exhausted — stop all FMP calls immediately
            with _fmp_lock:
                _fmp_daily_count = _FMP_DAILY_LIMIT
            _save_fmp_counter(_fmp_daily_count)
            logger.warning("FMP 429 for %s — daily quota exhausted, disabling FMP for remainder of run",
                           endpoint)
            raise FMPDailyLimitError(f"FMP 429 received — quota exhausted")

        resp.raise_for_status()
        return resp.json()

    # Final attempt failed
    resp.raise_for_status()
    return resp.json()


def fetch_analyst_consensus(ticker: str, current_price: Optional[float] = None) -> dict:
    """
    Fetch earnings surprise history for a ticker (PEAD scoring, O10).

    config#1821 Option B (2026-07-08): analyst grades consensus and price
    target consensus (formerly sourced from FMP's ``grades-consensus`` /
    ``price-target-consensus`` stable-API endpoints) were removed from the
    research feature contract — those endpoints 402'd for every ticker on
    the current plan and the circuit breaker (#391) had already made them
    a permanent no-op. ``consensus_rating`` / ``mean_target`` /
    ``num_analysts`` / ``upside_pct`` / ``rating_changes`` are no longer
    part of the returned shape.

    Returns empty result immediately if FMP daily budget is exhausted.

    Args:
        ticker: Stock symbol.
        current_price: Retained for call-site compatibility (was used for
                       the now-removed upside_pct calculation); passed
                       through into the result unchanged.

    Returns dict with keys: ticker, current_price, earnings_surprises.
    """
    result = {
        "ticker": ticker,
        "current_price": current_price,
        "earnings_surprises": [],
    }

    if fmp_budget_exhausted():
        logger.debug("FMP budget exhausted — skipping analyst data for %s", ticker)
        return result

    # O10: Earnings surprises (uses v3 API)
    try:
        data = _fmp_get(f"earning_surprises/{ticker}", base=_FMP_V3)
        if isinstance(data, list) and data:
            surprises = []
            for entry in data[:4]:  # last 4 quarters
                actual = entry.get("actualEarningResult")
                estimated = entry.get("estimatedEarning")
                surprise_pct = None
                if actual is not None and estimated is not None and estimated != 0:
                    surprise_pct = round((actual - estimated) / abs(estimated) * 100, 2)
                surprises.append({
                    "date": entry.get("date", ""),
                    "actual": actual,
                    "estimated": estimated,
                    "surprise_pct": surprise_pct,
                })
            result["earnings_surprises"] = surprises
    except Exception as e:
        logger.debug("FMP earnings surprises failed for %s: %s", ticker, e)

    return result
