"""
Price fetcher — reads daily OHLCV from ArcticDB and constituents from S3.

Phase 7c (2026-04-17) rip-and-replaced the yfinance + slim-cache + Wikipedia
fallback chain. The weekly research Lambda is now a pure consumer of
alpha-engine-data outputs:

  * Prices → ArcticDB ``universe`` library (written by alpha-engine-data's
    daily DailyData step + weekly backfill).
  * Constituents → ``s3://<bucket>/market_data/constituents.json`` (written
    by alpha-engine-data's weekly DataPhase1).

Both sources hard-fail on miss — no yfinance / FRED / Wikipedia fallback.
Matches the predictor's Phase 7a pattern and the ``feedback_hard_fail_until_stable``
+ ``feedback_no_silent_fails`` conventions.

This module is yfinance-free: the orphaned ``fetch_short_interest`` yfinance
``Ticker.info`` path was removed (yfinance-centralization arc, 2026-05-16);
short-interest ingestion belongs to alpha-engine-data's ``collectors/short_interest.py``.
"""

from __future__ import annotations

import logging
import os

import arcticdb as _arcticdb  # noqa: F401  Hard dep: matches predictor's

# Phase 7a pattern. Imported here at module top
# (BEFORE pandas) to prime arcticdb's bundled
# aws-c-common allocator on macOS — the lib
# chokepoint uses lazy import which would
# otherwise let pyarrow's allocator load first
# and segfault on first get_library() call.
# If the Lambda image lacks arcticdb, fail loud at
# cold start rather than silently degrading.
import pandas as pd

logger = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
_MARKET_DATA_PREFIX = "market_data/"

# OHLCV columns in ArcticDB's universe library (title-case; matches the schema
# alpha-engine-data's builders/backfill.py + daily_append.py write).
_ARCTIC_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Max fraction of per-ticker read failures tolerated before we treat the read as
# a pipeline failure. Matches the predictor's Phase 7a threshold.
_MAX_ERR_RATE = 0.05


class PriceFetchError(RuntimeError):
    """Raised when ArcticDB reads fail to meet quality thresholds."""
    pass


# Sentinel recorded as the run-level ``data_snapshot_id`` when no versioned
# ArcticDB read produced a usable ``VersionedItem.version`` (e.g. every
# ticker missing, or the read object doesn't expose ``.version`` because a
# non-versioned backend / mock returned the frame). Provenance must never
# crash the run — an absent version records ``"unknown"`` instead (per
# ``feedback_no_silent_fails``: visible, not silent).
DATA_SNAPSHOT_ID_UNKNOWN = "unknown"


def _extract_arctic_version(res: object) -> int | None:
    """Best-effort read of ArcticDB ``VersionedItem.version``.

    Returns the integer snapshot version when the read returned a
    ``VersionedItem`` exposing a numeric ``.version`` (the monotonic
    per-symbol write generation), else ``None`` — for non-versioned reads,
    mocks, or any shape that doesn't carry the attribute. Never raises:
    provenance extraction is strictly best-effort and must not sink a read.
    """
    version = getattr(res, "version", None)
    if isinstance(version, bool):  # bool is an int subclass — reject it
        return None
    if isinstance(version, int):
        return version
    return None


def _summarize_snapshot_id(versions: list[int]) -> str:
    """Collapse the per-ticker ArcticDB versions seen this read into one
    run-level ``data_snapshot_id`` string.

    ArcticDB versions are per-symbol monotonic ints, so there is no single
    global snapshot number across a multi-ticker read. We record the MAX
    version observed (the newest write generation any read touched) as the
    run-level provenance stamp — sufficient to pin the as-of data state for
    the reproducibility contract, and stable for the common case where the
    universe is written in lockstep by alpha-engine-data's daily append.

    Returns ``DATA_SNAPSHOT_ID_UNKNOWN`` when no versioned read was seen.
    """
    if not versions:
        return DATA_SNAPSHOT_ID_UNKNOWN
    return str(max(versions))


def _connect_arctic() -> object:
    """Open the ArcticDB ``universe`` library. Hard-fail on unreachable.

    Thin wrapper over ``alpha_engine_lib.arcticdb.open_universe_lib`` —
    the lib chokepoint (L2771) centralizes S3 URI construction and
    ``get_library`` error wrapping. Local wrapper retained so
    ``PriceFetchError`` semantics (a research-specific exception type)
    are preserved on the failure path.
    """
    from nousergon_lib.arcticdb import open_universe_lib
    try:
        return open_universe_lib(_S3_BUCKET)
    except Exception as exc:
        raise PriceFetchError(str(exc)) from exc


def _period_to_lookback_days(period: str) -> int:
    """Map yfinance-style period strings to calendar-day lookback windows."""
    mapping = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
    }
    if period not in mapping:
        raise ValueError(
            f"Unsupported period {period!r}; expected one of {sorted(mapping)}"
        )
    return mapping[period]


def fetch_price_data(
    tickers: list[str],
    period: str = "1y",
    *,
    return_snapshot_id: bool = False,
):
    """
    Read daily OHLCV for a list of tickers from ArcticDB.

    Returns ``{ticker: DataFrame[Open, High, Low, Close, Volume]}`` with a
    ``DatetimeIndex``. Individual tickers missing from ArcticDB are dropped
    from the result and logged as warnings; per-ticker error rate above
    ``_MAX_ERR_RATE`` (5%) raises ``PriceFetchError``.

    When ``return_snapshot_id=True`` the return becomes a
    ``(result, data_snapshot_id)`` tuple, where ``data_snapshot_id`` is the
    run-level ArcticDB version stamp surfaced from each read's
    ``VersionedItem.version`` (the immutable price-data snapshot the read
    resolved to — the provenance anchor for the decision-capture
    reproducibility contract, L4567 sub-item 1b / issue #781). It is
    ``"unknown"`` when no read produced a usable version (every ticker
    missing, or a non-versioned / mocked backend). Default ``False`` keeps
    the legacy ``dict``-only contract for existing callers.

    Failure semantics (Phase 7c):
      * ArcticDB unreachable → ``PriceFetchError`` (hard fail).
      * Per-ticker error rate > 5% → ``PriceFetchError``.
      * Individual ticker missing/empty → logged WARNING, dropped from output.

    No yfinance / slim-cache fallback. Upstream ArcticDB is canonical; silent
    fallbacks masked data bugs for days at a time pre-Phase-7a.
    """
    if not tickers:
        return ({}, DATA_SNAPSHOT_ID_UNKNOWN) if return_snapshot_id else {}

    lookback_days = _period_to_lookback_days(period)
    end_ts = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    universe_lib = _connect_arctic()

    result: dict[str, pd.DataFrame] = {}
    versions: list[int] = []
    n_err = 0
    for ticker in tickers:
        try:
            res = universe_lib.read(
                ticker,
                date_range=(start_ts, end_ts),
                columns=_ARCTIC_OHLCV_COLS,
            )
            df = res.data
        except Exception as exc:
            logger.warning("ArcticDB read failed for %s: %s", ticker, exc)
            n_err += 1
            continue
        if df is None or df.empty:
            logger.warning("ArcticDB returned empty frame for %s", ticker)
            n_err += 1
            continue
        # Surface the immutable snapshot version this read resolved to (1b).
        # Best-effort: a read that doesn't expose ``.version`` (mock /
        # non-versioned backend) just doesn't contribute to the stamp.
        _v = _extract_arctic_version(res)
        if _v is not None:
            versions.append(_v)
        # Defensive dedup — matches predictor Phase 7a (removable after 1-2
        # clean Saturday cycles confirm the upstream write path is clean).
        df = df[~df.index.duplicated(keep="last")].sort_index()
        result[ticker] = df

    err_rate = n_err / max(len(tickers), 1)
    if err_rate > _MAX_ERR_RATE:
        raise PriceFetchError(
            f"ArcticDB per-ticker error rate {err_rate:.1%} exceeds "
            f"{_MAX_ERR_RATE:.0%} threshold ({n_err} failed of {len(tickers)})"
        )

    data_snapshot_id = _summarize_snapshot_id(versions)
    logger.info(
        "[data_source=arcticdb] Loaded %d/%d ticker prices (%d missing, "
        "window %s → %s, data_snapshot_id=%s)",
        len(result), len(tickers), n_err, start_ts.date(), end_ts.date(),
        data_snapshot_id,
    )
    if return_snapshot_id:
        return result, data_snapshot_id
    return result


# ── Constituents ─────────────────────────────────────────────────────────────

# Wikipedia GICS sector names → internal sector names used throughout the system.
# Retained for historical signal archives that may carry raw GICS labels; the
# fresh path below reads pre-mapped sectors from alpha-engine-data.
_GICS_SECTOR_MAP = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Financials": "Financial",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
}


def _load_constituents_from_s3() -> tuple[list[str], dict[str, str]]:
    """Load constituents + sectors from alpha-engine-data's weekly output.

    Hard-fails on any miss (no Wikipedia fallback). alpha-engine-data's
    Saturday DataPhase1 step writes ``market_data/latest_weekly.json`` +
    ``market_data/<date>/constituents.json``; a missing or stale pointer
    means upstream didn't run, which is a pipeline failure, not a prompt
    to go scrape Wikipedia.
    """
    import json

    import boto3

    s3 = boto3.client("s3")
    try:
        ptr = s3.get_object(
            Bucket=_S3_BUCKET,
            Key=f"{_MARKET_DATA_PREFIX}latest_weekly.json",
        )
    except Exception as exc:
        raise PriceFetchError(
            f"s3://{_S3_BUCKET}/{_MARKET_DATA_PREFIX}latest_weekly.json unreadable: "
            f"{exc} — alpha-engine-data DataPhase1 did not run or the pointer is missing."
        ) from exc

    pointer = json.loads(ptr["Body"].read())
    prefix = pointer.get("s3_prefix", "")
    if not prefix:
        raise PriceFetchError(
            f"latest_weekly.json has no 's3_prefix' field: {pointer!r}"
        )

    try:
        obj = s3.get_object(Bucket=_S3_BUCKET, Key=f"{prefix}constituents.json")
    except Exception as exc:
        raise PriceFetchError(
            f"s3://{_S3_BUCKET}/{prefix}constituents.json unreadable: {exc}"
        ) from exc

    data = json.loads(obj["Body"].read())
    tickers = data.get("tickers", [])
    sector_map = data.get("sector_map", {})
    if not tickers or len(tickers) < 800:
        raise PriceFetchError(
            f"constituents.json has {len(tickers)} tickers (expected >= 800 for "
            f"S&P 500+400) — upstream collector produced a malformed output."
        )
    logger.info(
        "[data_source=s3] Loaded %d constituents from %s (date=%s)",
        len(tickers), f"{prefix}constituents.json", pointer.get("date"),
    )
    return tickers, sector_map


def fetch_sp500_sp400_with_sectors() -> tuple[list[str], dict[str, str]]:
    """
    Fetch S&P 500 and S&P 400 constituents + GICS sectors from
    alpha-engine-data's weekly S3 output.

    Hard-fails on any read error — no Wikipedia fallback.

    Returns ``(tickers, sector_map)`` where sector_map is ``{ticker:
    internal_sector_name}`` for all tickers.

    Survivorship-bias note (unchanged): alpha-engine-data's constituents
    collector pulls the current index membership, same as the Wikipedia path
    it replaces; historical backtests still need a paid source (Compustat,
    Sharadar) for point-in-time constituent data.
    """
    return _load_constituents_from_s3()


def fetch_sp500_sp400_tickers() -> list[str]:
    """Return the deduplicated S&P 500 + S&P 400 ticker list from S3."""
    tickers, _ = fetch_sp500_sp400_with_sectors()
    return tickers


# ── Technical indicators (pure computation; no external calls) ───────────────

def compute_technical_indicators(df: pd.DataFrame) -> dict | None:
    """
    Compute RSI(14), MACD signal, price vs MA50, price vs MA200,
    20-day momentum, and 20-day average volume from a price DataFrame.
    Returns None if insufficient data.
    """
    if df.empty or len(df) < 30:
        return None

    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)

    # ── RSI 14 ──────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi.iloc[-1]) if not rsi.empty else 50.0

    # ── MACD ────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    macd_cross = 0.0  # no cross
    if len(macd_line) >= 2:
        prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
        curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        if prev_diff < 0 and curr_diff >= 0:
            macd_cross = 1.0   # bullish cross
        elif prev_diff > 0 and curr_diff <= 0:
            macd_cross = -1.0  # bearish cross
    macd_above_zero = bool(macd_line.iloc[-1] > 0)

    # ── Moving Averages ──────────────────────────────────────────────────────
    current_price = float(close.iloc[-1])

    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    price_vs_ma50 = ((current_price / ma50) - 1) * 100 if ma50 else None
    price_vs_ma200 = ((current_price / ma200) - 1) * 100 if ma200 else None

    # ── 20-day Momentum ──────────────────────────────────────────────────────
    momentum_20d = None
    if len(close) >= 21:
        momentum_20d = float(((close.iloc[-1] / close.iloc[-21]) - 1) * 100)

    # ── Average Volume ───────────────────────────────────────────────────────
    avg_volume_20d = None
    if not volume.empty and len(volume) >= 20:
        avg_volume_20d = float(volume.tail(20).mean())

    return {
        "rsi_14": rsi_14,
        "macd_cross": macd_cross,
        "macd_above_zero": macd_above_zero,
        "macd_line_last": float(macd_line.iloc[-1]),
        "signal_line_last": float(signal_line.iloc[-1]),
        "current_price": current_price,
        "ma50": ma50,
        "ma200": ma200,
        "price_vs_ma50": price_vs_ma50,
        "price_vs_ma200": price_vs_ma200,
        "momentum_20d": momentum_20d,
        "avg_volume_20d": avg_volume_20d,
    }
