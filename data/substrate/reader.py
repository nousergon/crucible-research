"""Read producer-side substrate parquets and expose per-ticker accessors.

Wave 1 PR F of the institutional data-revamp arc (plan doc:
``~/Development/alpha-engine-docs/private/data-revamp-260513.md``).

Producer parquets (written by alpha-engine-data Wave 1 PRs):

  s3://alpha-engine-research/data/news_aggregates/{date}.parquet
    rows: ticker, aggregate_date, schema_version,
          n_articles, n_articles_trusted_weighted,
          n_articles_by_source_json,
          lm_sentiment_mean/max/min/trusted_mean,
          lm_positive/negative/uncertainty_words_total,
          event_count, event_severity_max/mean, event_categories,
          top_event_descriptions, entity_mentions_count

  s3://alpha-engine-research/data/insider_transactions/{date}.parquet
    rows per Form 4 transaction: ticker, issuer_cik, accession_number,
          filed_date, transaction_date, reporting_owner_name + flags
          (director/officer/10pct), security_title, transaction_code,
          shares, price_per_share, acquired_disposed_code,
          transaction_value_usd, shares_owned_after, etc.

  s3://alpha-engine-research/data/analyst_revisions/{date}.parquet
    rows per (ticker, as_of_date): mean_target_current/7d_ago/30d_ago,
          mean_target_delta_7d/30d, mean_target_pct_change_30d,
          num_analysts_current/30d_ago/delta_30d,
          consensus_rating_current/30d_ago, rating_changed_30d,
          n_snapshot_days_observed

Behavior:

- Missing parquet (data side hasn't produced it yet) returns empty
  DataFrame with the canonical schema — downstream consumers see
  None / 0 for missing fields rather than crashing.
- Reader is gated behind ``INSTITUTIONAL_SUBSTRATE_ENABLED=true`` at
  the fetch_data call site (parallel-observation cutover discipline);
  this module is import-safe + read-only regardless.
- Insider transactions need a rollup window (e.g. trailing 90 days)
  since a single date's parquet only holds that day's Form 4 filings.
  The reader exposes a ``window_days`` knob; the rollup walks back
  day-by-day with graceful tolerance for missing dates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_S3_BUCKET = "alpha-engine-research"

# Default key prefixes — must match the producer-side writers in
# alpha-engine-data. Pinning here as a single source of truth.
NEWS_AGGREGATES_PREFIX = "data/news_aggregates"
INSIDER_TRANSACTIONS_PREFIX = "data/insider_transactions"
ANALYST_REVISIONS_PREFIX = "data/analyst_revisions"


# ── Per-ticker snapshot shape ─────────────────────────────────────────


@dataclass(frozen=True)
class SubstrateSnapshot:
    """The institutional substrate's per-ticker view, ready to join
    onto research's existing ``input_data_snapshot``.

    All fields are optional (None / 0 / empty) when the corresponding
    parquet didn't exist or didn't have a row for this ticker. Agents
    must tolerate missing fields gracefully — same posture as PR #170's
    graceful degrade.
    """

    ticker: str
    as_of_date: Date

    # News (from news_aggregates/{date}.parquet)
    news_n_articles: int = 0
    news_n_articles_trusted_weighted: float = 0.0
    news_n_articles_by_source: dict[str, int] = field(default_factory=dict)
    news_lm_sentiment_mean: float | None = None
    news_lm_sentiment_trusted_mean: float | None = None
    news_lm_uncertainty_words_total: int = 0
    news_event_count: int = 0
    news_event_severity_max: float = 0.0
    news_event_categories: tuple[str, ...] = ()
    news_top_event_descriptions: str = ""

    # Insider (rolled up from insider_transactions/*.parquet window)
    insider_n_transactions_90d: int = 0
    insider_n_buys_90d: int = 0
    insider_n_sells_90d: int = 0
    insider_net_dollar_flow_90d: float = 0.0
    insider_distinct_insiders_90d: int = 0
    insider_max_single_transaction_usd: float = 0.0

    # Analyst revisions (from analyst_revisions/{date}.parquet)
    analyst_mean_target_current: float | None = None
    analyst_mean_target_delta_30d: float | None = None
    analyst_mean_target_pct_change_30d: float | None = None
    analyst_num_analysts_current: int | None = None
    analyst_num_analysts_delta_30d: int | None = None
    analyst_consensus_rating: str | None = None
    analyst_rating_changed_30d: bool = False

    @property
    def has_news_signal(self) -> bool:
        return self.news_n_articles > 0

    @property
    def has_insider_signal(self) -> bool:
        return self.insider_n_transactions_90d > 0

    @property
    def has_analyst_signal(self) -> bool:
        return self.analyst_mean_target_current is not None


# ── Parquet readers ───────────────────────────────────────────────────


def _read_parquet_safely(
    s3_client: Any, *, bucket: str, key: str,
) -> pd.DataFrame | None:
    """Read a parquet from S3 → DataFrame. Returns None on missing key
    or read failure (logged at INFO). Callers degrade to empty rows."""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        logger.info(
            "[substrate_reader] no parquet at s3://%s/%s (%s)",
            bucket, key, type(e).__name__,
        )
        return None
    try:
        return pd.read_parquet(BytesIO(obj["Body"].read()), engine="pyarrow")
    except Exception as e:
        logger.warning(
            "[substrate_reader] read failed for s3://%s/%s: %s",
            bucket, key, e,
        )
        return None


def _read_via_latest(
    s3_client: Any, *, bucket: str, prefix: str,
) -> pd.DataFrame | None:
    """Canonical-shape read: GET ``{prefix}/latest.json`` → resolve
    ``artifact_key`` → GET parquet body. Returns None if either step
    fails (logged at INFO so legacy-fallback callers can degrade).
    """
    import json as _json
    latest_key = f"{prefix}/latest.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=latest_key)
        sidecar = _json.loads(obj["Body"].read())
    except Exception as e:
        logger.info(
            "[substrate_reader] no canonical sidecar at s3://%s/%s (%s)",
            bucket, latest_key, type(e).__name__,
        )
        return None
    artifact_key = sidecar.get("artifact_key")
    if not artifact_key:
        return None
    return _read_parquet_safely(s3_client, bucket=bucket, key=artifact_key)


def read_news_aggregates(
    as_of_date: Date,
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = NEWS_AGGREGATES_PREFIX,
) -> pd.DataFrame:
    """Read news aggregates via canonical ``latest.json`` sidecar
    indirection. Falls back to legacy ``{as_of_date}.parquet`` shape
    during the transition window. Returns empty DataFrame if missing.

    ``as_of_date`` is used only for the legacy-shape fallback path —
    under the canonical shape, ``latest.json`` always points at the
    most recent run regardless of date; the parquet itself carries
    ``aggregate_date`` per row.
    """
    df = _read_via_latest(s3_client, bucket=bucket, prefix=prefix)
    if df is not None:
        return df
    # Legacy fallback
    legacy_key = f"{prefix}/{as_of_date.isoformat()}.parquet"
    df = _read_parquet_safely(s3_client, bucket=bucket, key=legacy_key)
    return df if df is not None else pd.DataFrame()


def read_analyst_revisions(
    as_of_date: Date,
    *,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = ANALYST_REVISIONS_PREFIX,
) -> pd.DataFrame:
    """Read analyst-revisions via canonical ``latest.json`` indirection
    + legacy fallback. Same shape as :func:`read_news_aggregates`."""
    df = _read_via_latest(s3_client, bucket=bucket, prefix=prefix)
    if df is not None:
        return df
    legacy_key = f"{prefix}/{as_of_date.isoformat()}.parquet"
    df = _read_parquet_safely(s3_client, bucket=bucket, key=legacy_key)
    return df if df is not None else pd.DataFrame()


def read_insider_transactions_window(
    as_of_date: Date,
    *,
    window_days: int = 90,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    prefix: str = INSIDER_TRANSACTIONS_PREFIX,
) -> pd.DataFrame:
    """Read insider transactions covering the trailing ``window_days``.

    Canonical shape: the producer writes one consolidated parquet per
    run (containing all Form 4 filings collected in that run, with
    ``filed_date`` as a row column). Read via ``latest.json`` → filter
    rows where ``filed_date`` falls within the window. Falls back to
    the legacy per-filed_date parquets if canonical sidecar missing.
    """
    df = _read_via_latest(s3_client, bucket=bucket, prefix=prefix)
    if df is not None and len(df) > 0 and "filed_date" in df.columns:
        cutoff = as_of_date - timedelta(days=window_days)
        # filed_date is stored as ISO date string OR pandas datetime;
        # convert defensively
        filed_dates = pd.to_datetime(df["filed_date"]).dt.date
        return df[
            (filed_dates >= cutoff) & (filed_dates <= as_of_date)
        ].reset_index(drop=True)

    # Legacy fallback: per-filed_date parquets
    frames: list[pd.DataFrame] = []
    for offset in range(window_days + 1):
        d = as_of_date - timedelta(days=offset)
        legacy_key = f"{prefix}/{d.isoformat()}.parquet"
        legacy_df = _read_parquet_safely(
            s3_client, bucket=bucket, key=legacy_key,
        )
        if legacy_df is not None and len(legacy_df) > 0:
            frames.append(legacy_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Per-ticker rollup ─────────────────────────────────────────────────


def _news_row_to_fields(row: pd.Series) -> dict:
    """Project one news_aggregates row → SubstrateSnapshot field subset."""
    try:
        source_counts = json.loads(row.get("n_articles_by_source_json") or "{}")
    except Exception:
        source_counts = {}
    categories_str = row.get("event_categories") or ""
    categories = tuple(
        c.strip() for c in str(categories_str).split(",") if c.strip()
    ) if categories_str else ()
    return {
        "news_n_articles": int(row.get("n_articles", 0) or 0),
        "news_n_articles_trusted_weighted": float(
            row.get("n_articles_trusted_weighted", 0.0) or 0.0
        ),
        "news_n_articles_by_source": source_counts,
        "news_lm_sentiment_mean": _opt_float(row.get("lm_sentiment_mean")),
        "news_lm_sentiment_trusted_mean": _opt_float(
            row.get("lm_sentiment_trusted_mean")
        ),
        "news_lm_uncertainty_words_total": int(
            row.get("lm_uncertainty_words_total", 0) or 0
        ),
        "news_event_count": int(row.get("event_count", 0) or 0),
        "news_event_severity_max": float(
            row.get("event_severity_max", 0.0) or 0.0
        ),
        "news_event_categories": categories,
        "news_top_event_descriptions": str(
            row.get("top_event_descriptions") or ""
        ),
    }


def _insider_rollup(rows: pd.DataFrame, *, ticker: str) -> dict:
    """Aggregate the ticker's Form 4 transactions over the window."""
    if rows is None or len(rows) == 0:
        return {}
    sub = rows[rows["ticker"] == ticker]
    if len(sub) == 0:
        return {}
    # Acquired ('A') = buy; Disposed ('D') = sell
    is_buy = sub["acquired_disposed_code"] == "A"
    is_sell = sub["acquired_disposed_code"] == "D"
    # Net dollar flow: buys positive, sells negative
    values = sub["transaction_value_usd"].fillna(0.0).astype(float)
    net_flow = float(values[is_buy].sum() - values[is_sell].sum())
    max_tx = float(values.abs().max()) if len(values) > 0 else 0.0
    distinct = sub["reporting_owner_name"].nunique() if "reporting_owner_name" in sub.columns else 0
    return {
        "insider_n_transactions_90d": int(len(sub)),
        "insider_n_buys_90d": int(is_buy.sum()),
        "insider_n_sells_90d": int(is_sell.sum()),
        "insider_net_dollar_flow_90d": net_flow,
        "insider_distinct_insiders_90d": int(distinct),
        "insider_max_single_transaction_usd": max_tx,
    }


def _analyst_row_to_fields(row: pd.Series) -> dict:
    return {
        "analyst_mean_target_current": _opt_float(
            row.get("mean_target_current")
        ),
        "analyst_mean_target_delta_30d": _opt_float(
            row.get("mean_target_delta_30d")
        ),
        "analyst_mean_target_pct_change_30d": _opt_float(
            row.get("mean_target_pct_change_30d")
        ),
        "analyst_num_analysts_current": _opt_int(
            row.get("num_analysts_current")
        ),
        "analyst_num_analysts_delta_30d": _opt_int(
            row.get("num_analysts_delta_30d")
        ),
        "analyst_consensus_rating": _opt_str(
            row.get("consensus_rating_current")
        ),
        "analyst_rating_changed_30d": bool(
            row.get("rating_changed_30d", False)
        ),
    }


def _opt_float(v) -> float | None:
    if v is None or (isinstance(v, float) and v != v):  # NaN check
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v) -> int | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    s = str(v).strip()
    return s or None


# ── Reader class ──────────────────────────────────────────────────────


class SubstrateReader:
    """Read producer-side substrate parquets, expose per-ticker
    :class:`SubstrateSnapshot` views.

    Caches parquet reads per (date, key) within one instance so a
    fetch_data call walking 25 held tickers does ~3 S3 GETs total
    instead of 75.
    """

    def __init__(
        self,
        s3_client: Any,
        *,
        bucket: str = DEFAULT_S3_BUCKET,
        insider_window_days: int = 90,
    ) -> None:
        self._s3 = s3_client
        self._bucket = bucket
        self._insider_window = insider_window_days

    def snapshot_for_ticker(
        self, ticker: str, *, as_of_date: Date,
        news_df: pd.DataFrame | None = None,
        insider_df: pd.DataFrame | None = None,
        analyst_df: pd.DataFrame | None = None,
    ) -> SubstrateSnapshot:
        """Build a per-ticker snapshot. DataFrames are passed in for
        caller-side caching; if any is None, the reader fetches it
        lazily.
        """
        if news_df is None:
            news_df = read_news_aggregates(
                as_of_date, s3_client=self._s3, bucket=self._bucket,
            )
        if insider_df is None:
            insider_df = read_insider_transactions_window(
                as_of_date,
                window_days=self._insider_window,
                s3_client=self._s3, bucket=self._bucket,
            )
        if analyst_df is None:
            analyst_df = read_analyst_revisions(
                as_of_date, s3_client=self._s3, bucket=self._bucket,
            )

        fields: dict = {"ticker": ticker, "as_of_date": as_of_date}

        # News
        if len(news_df) > 0 and "ticker" in news_df.columns:
            news_rows = news_df[news_df["ticker"] == ticker]
            if len(news_rows) > 0:
                fields.update(_news_row_to_fields(news_rows.iloc[0]))

        # Insider window
        fields.update(_insider_rollup(insider_df, ticker=ticker))

        # Analyst revisions
        if len(analyst_df) > 0 and "ticker" in analyst_df.columns:
            analyst_rows = analyst_df[analyst_df["ticker"] == ticker]
            if len(analyst_rows) > 0:
                fields.update(_analyst_row_to_fields(analyst_rows.iloc[0]))

        return SubstrateSnapshot(**fields)


# ── Convenience: read for a population in one pass ───────────────────


def read_substrate_for_population(
    tickers: list[str],
    *,
    as_of_date: Date,
    s3_client: Any,
    bucket: str = DEFAULT_S3_BUCKET,
    insider_window_days: int = 90,
) -> dict[str, SubstrateSnapshot]:
    """Read all 3 parquets once, then build per-ticker snapshots.

    Returns ``{ticker: SubstrateSnapshot}`` covering every input
    ticker. Tickers with no data get an empty snapshot (all fields
    default).
    """
    reader = SubstrateReader(
        s3_client, bucket=bucket,
        insider_window_days=insider_window_days,
    )
    news_df = read_news_aggregates(
        as_of_date, s3_client=s3_client, bucket=bucket,
    )
    insider_df = read_insider_transactions_window(
        as_of_date, window_days=insider_window_days,
        s3_client=s3_client, bucket=bucket,
    )
    analyst_df = read_analyst_revisions(
        as_of_date, s3_client=s3_client, bucket=bucket,
    )
    return {
        ticker: reader.snapshot_for_ticker(
            ticker, as_of_date=as_of_date,
            news_df=news_df, insider_df=insider_df,
            analyst_df=analyst_df,
        )
        for ticker in tickers
    }
