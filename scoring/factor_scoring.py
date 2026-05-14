"""
Factor scoring — Phase 1c of the institutional factor substrate (260513 plan).

Combines raw factor data (already populated in the production feature store
at s3://alpha-engine-research/features/{date}/) into 4 composite factor
scores per ticker, each percentile-ranked WITHIN sector to avoid
cross-sector noise (an OW Tech sector shouldn't lose all its names just
because Tech mean-vol is high vs. Healthcare).

Composite factors (mirroring AQR Style Premia + GS QIS conventions):

- ``quality_score``     — Quality (QMJ-style): ROE + (1 - debt/equity) +
                          gross margin + current ratio
- ``momentum_score``    — Cross-sectional momentum: 12-1m + 6m + 1m
                          + distance-from-52w-high
- ``low_vol_score``     — Inverse realized vol: (1 - 20d realized vol
                          z-score) + (1 - vol_ratio_10_60 z-score)
- ``value_score``       — Inverse multiples: (1 - PE) + (1 - PB) + FCF yield

All composites returned on a 0-100 within-sector percentile scale so they
compose with the existing 0-100 quant/qual sub-scores in
scoring/composite.py.

Produced once per Saturday SF run (and on demand by ad-hoc backtester
runs). Cached to s3://alpha-engine-research/factors/profiles/{date}/by_ticker.json
so downstream consumers (composite scoring extension in Phase 3, quant
@tool in Phase 2, backtester attribution in Phase 5) read from a single
canonical artifact without re-deriving from raw factor parquets.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import date as _date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Composite definitions ───────────────────────────────────────────────────
# Each composite is a weighted sum of within-sector-percentile-ranked raw
# factors. Weights sum to 1.0 per composite. Higher composite score = more
# desirable on that factor axis (e.g. high quality_score = more profitable +
# lower leverage; high low_vol_score = LOWER realized vol).
#
# `invert=True` means the raw factor is INVERSELY desirable (e.g. higher PE
# = less desirable, so we invert the percentile rank before combining).
_COMPOSITE_DEFS: dict[str, list[tuple[str, float, bool]]] = {
    "quality_score": [
        # (raw_factor_column, weight, invert_rank)
        ("roe", 0.30, False),
        ("debt_to_equity", 0.25, True),    # less debt = better
        ("gross_margin", 0.25, False),
        ("current_ratio", 0.20, False),
    ],
    "momentum_score": [
        ("momentum_20d", 0.30, False),
        ("return_60d", 0.25, False),
        ("return_120d", 0.20, False),
        ("dist_from_52w_high", 0.15, False),  # closer to high (less negative) = better
        ("momentum_5d", 0.10, False),
    ],
    "low_vol_score": [
        ("realized_vol_20d", 0.50, True),  # lower vol = higher score
        ("vol_ratio_10_60", 0.30, True),   # vol stable / declining = higher score
        ("atr_14_pct", 0.20, True),
    ],
    "value_score": [
        ("pe_ratio", 0.40, True),
        ("pb_ratio", 0.30, True),
        ("fcf_yield", 0.30, False),
    ],
}


def _within_sector_pct_rank(
    df: pd.DataFrame,
    factor_col: str,
    sector_col: str,
    invert: bool = False,
) -> pd.Series:
    """Compute percentile rank (0-100) of `factor_col` within each sector.

    NaN inputs propagate (return NaN — composite weight reallocates to
    other available factors per ticker). If `invert=True`, percentile is
    inverted (e.g. for PE ratio: highest PE → lowest score).

    Pandas `rank(pct=True)` handles ties by averaging ranks (the standard
    Spearman-tie convention).
    """
    if factor_col not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    ranks = df.groupby(sector_col)[factor_col].rank(pct=True, na_option="keep")
    pct = ranks * 100.0
    if invert:
        pct = 100.0 - pct
    return pct


def compute_factor_composites(
    technical_df: pd.DataFrame,
    fundamental_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Compute the 4 factor composites per ticker.

    Args:
        technical_df: feature store technical.parquet (per-ticker price-derived).
            Must include `ticker` column + the raw factor columns referenced
            in _COMPOSITE_DEFS for momentum / low_vol composites.
        fundamental_df: feature store fundamental.parquet (per-ticker
            Finnhub-sourced). Must include `ticker` column + the raw factor
            columns referenced in _COMPOSITE_DEFS for quality / value.
        sector_map: {ticker: sector_name} mapping. Tickers without a sector
            mapping default to ``"Unknown"`` and are ranked together.

    Returns:
        DataFrame with columns:
            ticker, sector,
            quality_score, momentum_score, low_vol_score, value_score,
            quality_n, momentum_n, low_vol_n, value_n
            (the *_n columns count how many raw factors actually
            contributed per composite — see partial-data handling below.)
    """
    # Merge tech + fundamental on ticker — outer join so a ticker missing
    # from one source still produces the composites for which we DO have
    # data (partial coverage is normal: a name without fundamentals can
    # still get momentum + low_vol).
    merged = technical_df.merge(
        fundamental_df.drop(columns=[c for c in fundamental_df.columns if c == "date"], errors="ignore"),
        on="ticker", how="outer", suffixes=("", "_fund"),
    )

    merged["sector"] = merged["ticker"].map(lambda t: sector_map.get(t, "Unknown"))

    out_rows: list[dict] = []
    for composite, components in _COMPOSITE_DEFS.items():
        # Compute within-sector percentile rank for each component
        component_ranks: list[tuple[str, float, pd.Series]] = []
        for factor_col, weight, invert in components:
            ranks = _within_sector_pct_rank(merged, factor_col, "sector", invert=invert)
            component_ranks.append((factor_col, weight, ranks))

        # Per-ticker weighted average of available component ranks. If a
        # component is NaN for a ticker, its weight reallocates pro-rata
        # to the components that ARE available — partial-coverage tickers
        # get a defensible composite from whatever data they have.
        composite_vals: list[float] = []
        composite_n: list[int] = []
        for i in range(len(merged)):
            num = 0.0
            denom = 0.0
            count = 0
            for _, weight, ranks in component_ranks:
                v = ranks.iloc[i]
                if pd.notna(v):
                    num += weight * v
                    denom += weight
                    count += 1
            composite_vals.append(num / denom if denom > 0 else float("nan"))
            composite_n.append(count)

        merged[composite] = composite_vals
        merged[f"{composite[:-6]}_n"] = composite_n  # quality_n, momentum_n, etc.

    keep_cols = ["ticker", "sector"] + list(_COMPOSITE_DEFS.keys()) + [
        f"{c[:-6]}_n" for c in _COMPOSITE_DEFS.keys()
    ]
    return merged[keep_cols].copy()


def write_factor_profiles_to_s3(
    profiles_df: pd.DataFrame,
    run_date: str,
    bucket: str | None = None,
) -> str:
    """Write factor profiles to S3 as `{date}/by_ticker.json`.

    Schema: ``{ticker: {sector, quality_score, momentum_score,
    low_vol_score, value_score, *_n}}``. Consumers (composite scoring,
    quant @tool, backtester attribution) read this single canonical
    artifact rather than re-deriving from raw parquets.

    Returns the S3 key written.
    """
    import boto3

    bucket = bucket or os.environ.get("S3_BUCKET", "alpha-engine-research")
    key = f"factors/profiles/{run_date}/by_ticker.json"

    # Convert to {ticker: {field: value}} dict, dropping NaN scores
    payload: dict[str, dict] = {}
    for _, row in profiles_df.iterrows():
        ticker = row["ticker"]
        record = {"sector": row["sector"]}
        for col in profiles_df.columns:
            if col in ("ticker", "sector"):
                continue
            v = row[col]
            if pd.notna(v):
                record[col] = float(v) if isinstance(v, (int, float)) else int(v)
        payload[ticker] = record

    body = json.dumps(payload, indent=2)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    # Also write `latest.json` sidecar for cache-warm convenience
    latest_key = "factors/profiles/latest.json"
    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")

    logger.info(
        "Factor profiles written to s3://%s/%s (%d tickers, %d composite columns)",
        bucket, key, len(payload), len(_COMPOSITE_DEFS),
    )
    return key


def read_factor_profiles_from_s3(
    run_date: Optional[str] = None,
    bucket: str | None = None,
) -> dict[str, dict] | None:
    """Read factor profiles from S3.

    If `run_date` is None, reads `factors/profiles/latest.json` sidecar
    (cheap; no S3 list call). Returns None on any read failure (consumers
    should treat absence as "no factor data available, skip factor blend").
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket = bucket or os.environ.get("S3_BUCKET", "alpha-engine-research")
    key = (
        f"factors/profiles/{run_date}/by_ticker.json"
        if run_date else "factors/profiles/latest.json"
    )
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            logger.warning("Factor profile read error from s3://%s/%s: %s", bucket, key, e)
        return None
    except Exception as e:
        logger.warning("Unexpected factor profile read error: %s", e)
        return None


def compute_and_write_factor_profiles(
    run_date: str | _date,
    sector_map: dict[str, str],
    bucket: str | None = None,
) -> str:
    """Saturday SF entry point — read raw factor parquets, compute composites, write profiles.

    Reads:
      - s3://{bucket}/features/{run_date}/technical.parquet
      - s3://{bucket}/features/{run_date}/fundamental.parquet

    Writes:
      - s3://{bucket}/factors/profiles/{run_date}/by_ticker.json
      - s3://{bucket}/factors/profiles/latest.json (sidecar)

    Returns the dated S3 key written.
    """
    import boto3

    bucket = bucket or os.environ.get("S3_BUCKET", "alpha-engine-research")
    run_date_str = run_date.isoformat() if isinstance(run_date, _date) else run_date

    s3 = boto3.client("s3")

    def _read(parquet_name: str) -> pd.DataFrame:
        key = f"features/{run_date_str}/{parquet_name}.parquet"
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")

    technical_df = _read("technical")
    fundamental_df = _read("fundamental")

    profiles = compute_factor_composites(
        technical_df=technical_df,
        fundamental_df=fundamental_df,
        sector_map=sector_map,
    )

    return write_factor_profiles_to_s3(profiles, run_date_str, bucket=bucket)
