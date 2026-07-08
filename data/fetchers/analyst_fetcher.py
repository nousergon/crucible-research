"""
Analyst consensus fetcher — Financial Modeling Prep (FMP) stable API.
Free tier: 250 requests/day. Daily usage ~73 — within free tier (§15.4).

Uses stable API endpoints for consensus data. Also fetches earnings
surprises from the v3 API for PEAD scoring (O10).
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
# cover (e.g. grades-consensus, price-target-consensus). This is a deterministic
# per-plan rejection, not a transient failure — retrying it just burns the
# per-run wall clock (Research Lambda has a 900s ceiling) for a guaranteed
# repeat failure. Trip a breaker per endpoint on the first 402 seen in a run;
# every subsequent call to that endpoint short-circuits (no HTTP call, no log
# spam) and increments a skip counter that the run summary can report.
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

    Keys are FMP endpoint names (e.g. ``grades-consensus``); values are the
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
    # name (e.g. "grades-consensus"), not the full URL with params, since
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
    Fetch analyst consensus rating, mean price target, and number of analysts.
    Returns empty result immediately if FMP daily budget is exhausted.

    Args:
        ticker: Stock symbol.
        current_price: If provided, used for upside_pct calculation instead of
                       making a separate FMP quote call. Pass from yfinance data.

    Returns dict with keys: consensus_rating, mean_target, num_analysts,
    current_price, upside_pct.
    """
    result = {
        "ticker": ticker,
        "consensus_rating": None,
        "mean_target": None,
        "num_analysts": None,
        "current_price": current_price,
        "upside_pct": None,
        "rating_changes": [],
        "earnings_surprises": [],
    }

    if fmp_budget_exhausted():
        logger.debug("FMP budget exhausted — skipping analyst data for %s", ticker)
        return result

    # Analyst grades consensus (strongBuy/buy/hold/sell counts + pre-computed consensus)
    try:
        data = _fmp_get("grades-consensus", {"symbol": ticker})
        if isinstance(data, list) and data:
            g = data[0]
            result["consensus_rating"] = g.get("consensus")
            total = sum(g.get(k, 0) or 0 for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
            result["num_analysts"] = total or None
    except FMPPlanLimitedError as e:
        # Breaker already logged its one-time WARN and counted the skip —
        # avoid per-ticker log spam on every subsequent call this run.
        logger.debug("FMP grades-consensus skipped for %s: %s", ticker, e)
    except Exception as e:
        logger.warning("FMP grades-consensus failed for %s: %s", ticker, e)

    # Price target consensus
    try:
        data = _fmp_get("price-target-consensus", {"symbol": ticker})
        if isinstance(data, list) and data:
            pt = data[0]
            result["mean_target"] = pt.get("targetConsensus")
    except FMPPlanLimitedError as e:
        logger.debug("FMP price-target-consensus skipped for %s: %s", ticker, e)
    except Exception as e:
        logger.warning("FMP price-target-consensus failed for %s: %s", ticker, e)

    # Compute upside from price already available (yfinance or passed in)
    if result["mean_target"] and result["current_price"]:
        result["upside_pct"] = round(
            (result["mean_target"] / result["current_price"] - 1) * 100, 1
        )

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
