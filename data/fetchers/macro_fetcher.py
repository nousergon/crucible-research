"""
Macro data reader — pulls macro + market breadth from alpha-engine-data's
weekly S3 output.

Phase 7c (2026-04-17): ripped out the live FRED + yfinance commodity/index
batch. The weekly research Lambda is now a pure consumer of alpha-engine-data;
the collector at ``alpha-engine-data/collectors/macro.py`` owns the FRED +
yfinance calls and writes the consolidated ``market_data/macro.json`` the
research Lambda reads here. Hard-fails on any S3 read miss — no live fallback.

``compute_market_breadth`` is a pure computation kept here (no API calls) for
callers that have a loaded ``price_data`` dict.
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class MacroFetchError(RuntimeError):
    """Raised when the macro S3 read fails to meet quality thresholds."""
    pass


_S3_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
_MARKET_DATA_PREFIX = "market_data/"


def compute_market_breadth(price_data: dict[str, pd.DataFrame]) -> dict:
    """
    Compute equity breadth metrics from ~900 S&P 500+400 stocks.

    Returns dict with:
      pct_above_50d_ma:  % of stocks trading above their 50-day MA
      pct_above_200d_ma: % of stocks trading above their 200-day MA
      advance_decline_ratio: advancers / decliners over last 5 trading days
      n_stocks: number of stocks with valid data
    """
    above_50d = 0
    total_50d = 0
    above_200d = 0
    total_200d = 0
    advancers = 0
    decliners = 0

    for _ticker, df in price_data.items():
        if df is None or df.empty or len(df) < 10:
            continue

        close = df["Close"]
        current = float(close.iloc[-1])

        # 50-day MA breadth
        if len(close) >= 50:
            ma50 = float(close.rolling(50).mean().iloc[-1])
            total_50d += 1
            if current > ma50:
                above_50d += 1

        # 200-day MA breadth
        if len(close) >= 200:
            ma200 = float(close.rolling(200).mean().iloc[-1])
            total_200d += 1
            if current > ma200:
                above_200d += 1

        # 5-day advance/decline
        if len(close) >= 6:
            five_day_return = current / float(close.iloc[-6]) - 1
            if five_day_return > 0:
                advancers += 1
            elif five_day_return < 0:
                decliners += 1

    result = {
        "pct_above_50d_ma": round(above_50d / total_50d * 100, 1) if total_50d > 0 else None,
        "pct_above_200d_ma": round(above_200d / total_200d * 100, 1) if total_200d > 0 else None,
        "advance_decline_ratio": round(advancers / max(decliners, 1), 2),
        "n_stocks": max(total_50d, total_200d),
    }
    logger.info(
        "[breadth] above_50dMA=%.1f%% above_200dMA=%.1f%% A/D=%.2f n=%d",
        result["pct_above_50d_ma"] or 0,
        result["pct_above_200d_ma"] or 0,
        result["advance_decline_ratio"],
        result["n_stocks"],
    )
    return result


def fetch_macro_data() -> dict:
    """
    Read macro data from alpha-engine-data's weekly S3 output.

    Hard-fails on any read error. No FRED / yfinance fallback — the collector
    at ``alpha-engine-data/collectors/macro.py`` is canonical, and its output
    lands at ``s3://<bucket>/market_data/<date>/macro.json`` with pointer
    ``market_data/latest_weekly.json``.

    Returns dict with:
      fed_funds_rate, treasury_2yr, treasury_10yr, yield_curve_slope,
      vix, unemployment, cpi_yoy,
      consumer_sentiment, initial_claims, hy_credit_spread_oas,
      sp500_close, sp500_30d_return, qqq_30d_return, iwm_30d_return,
      oil_wti, gold, copper, fetched_at
    """
    import boto3

    s3 = boto3.client("s3")
    try:
        ptr = s3.get_object(
            Bucket=_S3_BUCKET,
            Key=f"{_MARKET_DATA_PREFIX}latest_weekly.json",
        )
    except Exception as exc:
        raise MacroFetchError(
            f"s3://{_S3_BUCKET}/{_MARKET_DATA_PREFIX}latest_weekly.json unreadable: "
            f"{exc} — alpha-engine-data DataPhase1 did not run or the pointer is missing."
        ) from exc

    pointer = json.loads(ptr["Body"].read())
    prefix = pointer.get("s3_prefix", "")
    if not prefix:
        raise MacroFetchError(
            f"latest_weekly.json has no 's3_prefix' field: {pointer!r}"
        )

    try:
        obj = s3.get_object(Bucket=_S3_BUCKET, Key=f"{prefix}macro.json")
    except Exception as exc:
        raise MacroFetchError(
            f"s3://{_S3_BUCKET}/{prefix}macro.json unreadable: {exc}"
        ) from exc

    data = json.loads(obj["Body"].read())

    # Sanity gate: core FRED rate field must be populated. If the upstream
    # collector produced a malformed output we'd rather hard-fail than let
    # the downstream macro agent reason over a half-empty dict.
    if data.get("fed_funds_rate") is None:
        raise MacroFetchError(
            f"s3://{_S3_BUCKET}/{prefix}macro.json missing 'fed_funds_rate' — "
            f"upstream collector produced a malformed output."
        )

    logger.info(
        "[data_source=s3] Loaded macro data from %s (date=%s)",
        f"{prefix}macro.json", pointer.get("date"),
    )
    return data
