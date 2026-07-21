"""
Feature store reader — reads pre-computed technical features from the predictor's
S3 feature store, providing richer indicators (MA200, 52-week high/low) than the
3-month local price history can compute.

Falls back gracefully (returns empty dict) if the feature store is unavailable.
"""

from __future__ import annotations

import io
import logging
import os

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("S3_BUCKET", "alpha-engine-research")
_FEATURE_PREFIX = "features/"


def read_latest_features() -> dict[str, dict] | None:
    """
    Read the most recent feature store snapshot from S3.

    Returns {ticker: {feature_name: value}} or None if unavailable.
    Only reads the 'technical' group (the features research needs).
    """
    try:
        import boto3
        import pandas as pd

        s3 = boto3.client("s3")

        # Find the latest date directory in features/
        response = s3.list_objects_v2(
            Bucket=_BUCKET, Prefix=_FEATURE_PREFIX, Delimiter="/"
        )
        prefixes = response.get("CommonPrefixes", [])
        dates = []
        for p in prefixes:
            part = p["Prefix"].rstrip("/").split("/")[-1]
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                dates.append(part)

        if not dates:
            logger.debug("No feature store snapshots found in s3://%s/%s", _BUCKET, _FEATURE_PREFIX)
            return None

        latest_date = sorted(dates)[-1]
        logger.info("Feature store: reading snapshot from %s", latest_date)

        # Read technical group
        key = f"{_FEATURE_PREFIX}{latest_date}/technical.parquet"
        obj = s3.get_object(Bucket=_BUCKET, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, engine="pyarrow")

        if df.empty or "ticker" not in df.columns:
            return None

        # Convert to {ticker: {feature: value}} dict
        result = {}
        for _, row in df.iterrows():
            ticker = row["ticker"]
            features = {col: float(row[col]) for col in df.columns
                        if col not in ("ticker", "date") and pd.notna(row[col])}
            # Reconstruct current_price from price_vs_ma50 if available
            # (feature store doesn't store raw price, but close is in OHLCV cache)
            result[ticker] = features

        logger.info("Feature store: loaded %d tickers from %s", len(result), latest_date)
        return result

    except Exception as e:
        logger.debug("Feature store read failed (non-blocking): %s", e)
        return None


def read_latest_factor_loadings(
    columns: tuple[str, ...] = (
        "momentum_20d_zscore",
        "return_60d_zscore",
        "beta_60d_zscore",
        "size_zscore",
    ),
) -> dict[str, dict[str, float]] | None:
    """Read the most recent Barra factor-loading snapshot from S3.

    The factor loadings (``*_zscore`` columns, group ``factor_loading`` in
    alpha-engine-data's feature registry) are written to a SEPARATE parquet
    ``features/{date}/factor_loading.parquet`` — NOT the ``technical.parquet``
    group that :func:`read_latest_features` reads. They are the per-name
    exposures the score-neutralization OBSERVE shadow residualizes the
    composite against (config#1142).

    Returns ``{ticker: {factor_name: exposure}}`` for the requested ``columns``
    (only finite values included), or ``None`` if unavailable / the group's
    parquet is missing. Fully fail-soft — research must never break because the
    loadings group hasn't shipped a snapshot yet.
    """
    try:
        import boto3
        import pandas as pd

        s3 = boto3.client("s3")

        # Find the latest date directory in features/ (same discovery as
        # read_latest_features — the two groups share the date partition).
        response = s3.list_objects_v2(
            Bucket=_BUCKET, Prefix=_FEATURE_PREFIX, Delimiter="/"
        )
        prefixes = response.get("CommonPrefixes", [])
        dates = []
        for p in prefixes:
            part = p["Prefix"].rstrip("/").split("/")[-1]
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                dates.append(part)

        if not dates:
            logger.debug("No feature store snapshots found in s3://%s/%s", _BUCKET, _FEATURE_PREFIX)
            return None

        latest_date = sorted(dates)[-1]
        key = f"{_FEATURE_PREFIX}{latest_date}/factor_loading.parquet"
        try:
            obj = s3.get_object(Bucket=_BUCKET, Key=key)
        except Exception as e:
            logger.debug("Factor-loading parquet not present at %s: %s", key, e)
            return None

        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, engine="pyarrow")

        if df.empty or "ticker" not in df.columns:
            return None

        present = [c for c in columns if c in df.columns]
        if not present:
            logger.debug(
                "Factor-loading parquet %s has none of the requested columns %s",
                key, columns,
            )
            return None

        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            ticker = row["ticker"]
            ex = {c: float(row[c]) for c in present if pd.notna(row[c])}
            if ex:
                result[ticker] = ex

        logger.info(
            "Factor loadings: loaded %d tickers from %s (%d/%d columns present)",
            len(result), latest_date, len(present), len(columns),
        )
        return result or None

    except Exception as e:
        logger.debug("Factor-loading read failed (non-blocking): %s", e)
        return None


def read_latest_daily_closes() -> dict[str, float] | None:
    """Read the most recent daily_closes parquet from S3.

    Returns {ticker: close_price} or None if unavailable.
    Much cheaper than yfinance batch fetch (~100KB single S3 read vs ~900 HTTP calls).

    Reads from ``staging/daily_closes/`` per the 2026-04-29 prefix migration
    in alpha-engine-data PR #112 (the parquet's role is intermediate state
    between API fetch and ArcticDB ingest, not authoritative storage).
    Hard-cutover with no fallback per ``feedback_no_silent_fails`` — if
    the staging prefix is empty/missing, the function returns ``None`` and
    callers must handle that explicitly (existing contract).
    """
    try:
        import boto3
        import pandas as pd

        s3 = boto3.client("s3")

        # Find the latest daily_closes file
        response = s3.list_objects_v2(
            Bucket=_BUCKET, Prefix="staging/daily_closes/", MaxKeys=100,
        )
        contents = response.get("Contents", [])
        if not contents:
            return None

        # Get the most recent parquet by key name (dates sort lexicographically)
        parquet_keys = [c["Key"] for c in contents if c["Key"].endswith(".parquet")]
        if not parquet_keys:
            return None

        latest_key = sorted(parquet_keys)[-1]
        logger.info("Daily closes: reading %s", latest_key)

        import io
        obj = s3.get_object(Bucket=_BUCKET, Key=latest_key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, engine="pyarrow")

        if df.empty:
            return None

        # Schema: index=ticker or column=ticker, with 'close' or 'adj_close' column
        result = {}
        close_col = "close" if "close" in df.columns else "Close"
        ticker_col = None
        if "ticker" in df.columns:
            ticker_col = "ticker"
        elif df.index.name == "ticker":
            df = df.reset_index()
            ticker_col = "ticker"

        if ticker_col and close_col in df.columns:
            for _, row in df.iterrows():
                t = row[ticker_col]
                c = row[close_col]
                if t and pd.notna(c) and c > 0:
                    result[t] = float(c)

        logger.info("Daily closes: %d tickers with prices", len(result))
        return result if result else None

    except Exception as e:
        logger.debug("Daily closes read failed (non-blocking): %s", e)
        return None
