"""Tests for the producer-substrate reader (Wave 1 PR F).

Covers:
  - Reading news_aggregates / insider_transactions / analyst_revisions
    parquets (round-trip via in-memory mock)
  - Missing-parquet returns empty DataFrame
  - Per-ticker rollup: news fields / insider 90d aggregates / analyst deltas
  - Empty snapshot when no parquet data
  - SubstrateSnapshot convenience flags
  - read_substrate_for_population covers every input ticker
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO

import pandas as pd
import pytest

from data.substrate import (
    SubstrateReader,
    SubstrateSnapshot,
    read_substrate_for_population,
)
from data.substrate.reader import (
    NEWS_AGGREGATES_PREFIX,
    ANALYST_REVISIONS_PREFIX,
    INSIDER_TRANSACTIONS_PREFIX,
    read_analyst_revisions,
    read_insider_transactions_window,
    read_news_aggregates,
)


# ── In-memory S3 mock ──────────────────────────────────────────────────


class _InMemoryS3:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None, **kw):
        self.store[(Bucket, Key)] = Body
        return {"ETag": "stub"}

    def get_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise RuntimeError("NoSuchKey")
        return {"Body": BytesIO(self.store[(Bucket, Key)])}


def _put_parquet(s3, *, bucket, key, df):
    buf = BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# ── Per-parquet readers ────────────────────────────────────────────────


class TestReadNewsAggregates:
    def test_returns_dataframe_from_s3(self):
        s3 = _InMemoryS3()
        df_in = pd.DataFrame([{
            "ticker": "AAPL",
            "aggregate_date": date(2026, 5, 13),
            "schema_version": 1,
            "n_articles": 5,
            "n_articles_trusted_weighted": 4.2,
            "n_articles_by_source_json": '{"polygon": 3, "gdelt": 2}',
            "lm_sentiment_mean": 0.3,
            "lm_sentiment_trusted_mean": 0.4,
            "lm_uncertainty_words_total": 7,
            "event_count": 2,
            "event_severity_max": 0.8,
            "event_categories": "earnings_release,product_launch",
            "top_event_descriptions": "Q4 beat | New product unveiled",
        }])
        _put_parquet(
            s3,
            bucket="alpha-engine-research",
            key=f"{NEWS_AGGREGATES_PREFIX}/2026-05-13.parquet",
            df=df_in,
        )
        df_out = read_news_aggregates(date(2026, 5, 13), s3_client=s3)
        assert len(df_out) == 1
        assert df_out.iloc[0]["ticker"] == "AAPL"

    def test_missing_parquet_returns_empty_df(self):
        s3 = _InMemoryS3()
        df = read_news_aggregates(date(2026, 1, 1), s3_client=s3)
        assert len(df) == 0


class TestReadAnalystRevisions:
    def test_round_trip(self):
        s3 = _InMemoryS3()
        df_in = pd.DataFrame([{
            "ticker": "AAPL", "as_of_date": date(2026, 5, 13),
            "schema_version": 1, "primary_source": "yfinance",
            "mean_target_current": 260.0,
            "mean_target_7d_ago": 258.0,
            "mean_target_30d_ago": 250.0,
            "mean_target_delta_7d": 2.0,
            "mean_target_delta_30d": 10.0,
            "mean_target_pct_change_30d": 0.04,
            "num_analysts_current": 18, "num_analysts_30d_ago": 16,
            "num_analysts_delta_30d": 2,
            "consensus_rating_current": "buy",
            "consensus_rating_30d_ago": "hold",
            "rating_changed_30d": True,
            "n_snapshot_days_observed": 30,
        }])
        _put_parquet(
            s3, bucket="alpha-engine-research",
            key=f"{ANALYST_REVISIONS_PREFIX}/2026-05-13.parquet",
            df=df_in,
        )
        df = read_analyst_revisions(date(2026, 5, 13), s3_client=s3)
        assert df.iloc[0]["mean_target_delta_30d"] == 10.0


class TestReadInsiderTransactionsWindow:
    def test_concatenates_across_window(self):
        s3 = _InMemoryS3()
        # Two days of insider transactions, both within 90-day window
        for d, ticker in [
            (date(2026, 5, 10), "AAPL"),
            (date(2026, 5, 1), "MSFT"),
        ]:
            df = pd.DataFrame([{
                "ticker": ticker, "filed_date": d,
                "transaction_date": d, "acquired_disposed_code": "D",
                "transaction_value_usd": 1_000_000.0,
                "reporting_owner_name": "Insider",
            }])
            _put_parquet(
                s3, bucket="alpha-engine-research",
                key=f"{INSIDER_TRANSACTIONS_PREFIX}/{d.isoformat()}.parquet",
                df=df,
            )
        combined = read_insider_transactions_window(
            date(2026, 5, 13), window_days=30, s3_client=s3,
        )
        assert len(combined) == 2
        assert {combined.iloc[i]["ticker"] for i in range(2)} == {"AAPL", "MSFT"}

    def test_missing_dates_tolerated(self):
        """Most weekdays have no Form 4 filings for a given ticker —
        missing parquets just contribute nothing."""
        s3 = _InMemoryS3()
        combined = read_insider_transactions_window(
            date(2026, 5, 13), window_days=7, s3_client=s3,
        )
        assert len(combined) == 0


# ── Per-ticker rollup ─────────────────────────────────────────────────


class TestSubstrateReader:
    def test_full_snapshot_with_all_3_streams(self):
        s3 = _InMemoryS3()
        # News
        _put_parquet(
            s3, bucket="alpha-engine-research",
            key=f"{NEWS_AGGREGATES_PREFIX}/2026-05-13.parquet",
            df=pd.DataFrame([{
                "ticker": "AAPL", "aggregate_date": date(2026, 5, 13),
                "schema_version": 1, "n_articles": 8,
                "n_articles_trusted_weighted": 7.1,
                "n_articles_by_source_json": '{"polygon": 5, "gdelt": 3}',
                "lm_sentiment_mean": 0.25,
                "lm_sentiment_trusted_mean": 0.31,
                "lm_uncertainty_words_total": 4,
                "event_count": 3,
                "event_severity_max": 0.9,
                "event_categories": "earnings_release",
                "top_event_descriptions": "Q4 beat",
            }]),
        )
        # Insider — 2 buys + 1 sell within window
        _put_parquet(
            s3, bucket="alpha-engine-research",
            key=f"{INSIDER_TRANSACTIONS_PREFIX}/2026-05-10.parquet",
            df=pd.DataFrame([
                {"ticker": "AAPL", "filed_date": date(2026, 5, 10),
                 "acquired_disposed_code": "A",
                 "transaction_value_usd": 500_000.0,
                 "reporting_owner_name": "Cook"},
                {"ticker": "AAPL", "filed_date": date(2026, 5, 10),
                 "acquired_disposed_code": "A",
                 "transaction_value_usd": 300_000.0,
                 "reporting_owner_name": "Maestri"},
                {"ticker": "AAPL", "filed_date": date(2026, 5, 10),
                 "acquired_disposed_code": "D",
                 "transaction_value_usd": 200_000.0,
                 "reporting_owner_name": "Cook"},
            ]),
        )
        # Analyst revisions
        _put_parquet(
            s3, bucket="alpha-engine-research",
            key=f"{ANALYST_REVISIONS_PREFIX}/2026-05-13.parquet",
            df=pd.DataFrame([{
                "ticker": "AAPL", "as_of_date": date(2026, 5, 13),
                "schema_version": 1, "primary_source": "yfinance",
                "mean_target_current": 260.0,
                "mean_target_7d_ago": 258.0,
                "mean_target_30d_ago": 250.0,
                "mean_target_delta_7d": 2.0,
                "mean_target_delta_30d": 10.0,
                "mean_target_pct_change_30d": 0.04,
                "num_analysts_current": 18,
                "num_analysts_30d_ago": 16,
                "num_analysts_delta_30d": 2,
                "consensus_rating_current": "buy",
                "consensus_rating_30d_ago": "hold",
                "rating_changed_30d": True,
                "n_snapshot_days_observed": 30,
            }]),
        )

        reader = SubstrateReader(s3)
        snap = reader.snapshot_for_ticker(
            "AAPL", as_of_date=date(2026, 5, 13),
        )

        # News rollup
        assert snap.news_n_articles == 8
        assert snap.news_n_articles_trusted_weighted == 7.1
        assert snap.news_n_articles_by_source == {"polygon": 5, "gdelt": 3}
        assert snap.news_lm_sentiment_trusted_mean == 0.31
        assert snap.news_event_count == 3
        assert snap.news_event_categories == ("earnings_release",)

        # Insider rollup (90d window, AAPL only)
        assert snap.insider_n_transactions_90d == 3
        assert snap.insider_n_buys_90d == 2
        assert snap.insider_n_sells_90d == 1
        # Net = 500k + 300k (buys) - 200k (sells) = 600k
        assert snap.insider_net_dollar_flow_90d == 600_000.0
        assert snap.insider_distinct_insiders_90d == 2

        # Analyst revisions
        assert snap.analyst_mean_target_current == 260.0
        assert snap.analyst_mean_target_delta_30d == 10.0
        assert snap.analyst_consensus_rating == "buy"
        assert snap.analyst_rating_changed_30d is True

        # Convenience flags
        assert snap.has_news_signal is True
        assert snap.has_insider_signal is True
        assert snap.has_analyst_signal is True

    def test_empty_substrate_returns_default_snapshot(self):
        s3 = _InMemoryS3()
        reader = SubstrateReader(s3)
        snap = reader.snapshot_for_ticker(
            "AAPL", as_of_date=date(2026, 5, 13),
        )
        # Empty fields
        assert snap.news_n_articles == 0
        assert snap.insider_n_transactions_90d == 0
        assert snap.analyst_mean_target_current is None
        assert snap.has_news_signal is False
        assert snap.has_insider_signal is False
        assert snap.has_analyst_signal is False

    def test_ticker_not_in_parquet_yields_default_fields(self):
        """A ticker in the population that the producer hadn't seen
        yet shouldn't crash; it just gets empty fields."""
        s3 = _InMemoryS3()
        _put_parquet(
            s3, bucket="alpha-engine-research",
            key=f"{NEWS_AGGREGATES_PREFIX}/2026-05-13.parquet",
            df=pd.DataFrame([{
                "ticker": "AAPL", "aggregate_date": date(2026, 5, 13),
                "schema_version": 1, "n_articles": 5,
                "n_articles_trusted_weighted": 5.0,
                "n_articles_by_source_json": "{}",
                "lm_sentiment_mean": 0.1,
                "lm_sentiment_trusted_mean": 0.1,
                "lm_uncertainty_words_total": 0,
                "event_count": 0, "event_severity_max": 0.0,
                "event_categories": "", "top_event_descriptions": "",
            }]),
        )
        reader = SubstrateReader(s3)
        snap = reader.snapshot_for_ticker(
            "MSFT", as_of_date=date(2026, 5, 13),
        )
        assert snap.news_n_articles == 0
        assert snap.has_news_signal is False


# ── read_substrate_for_population ─────────────────────────────────────


class TestReadSubstrateForPopulation:
    def test_covers_every_input_ticker(self):
        s3 = _InMemoryS3()
        # Only AAPL has news data
        _put_parquet(
            s3, bucket="alpha-engine-research",
            key=f"{NEWS_AGGREGATES_PREFIX}/2026-05-13.parquet",
            df=pd.DataFrame([{
                "ticker": "AAPL", "aggregate_date": date(2026, 5, 13),
                "schema_version": 1, "n_articles": 5,
                "n_articles_trusted_weighted": 5.0,
                "n_articles_by_source_json": "{}",
                "lm_sentiment_mean": 0.1,
                "lm_sentiment_trusted_mean": 0.1,
                "lm_uncertainty_words_total": 0,
                "event_count": 0, "event_severity_max": 0.0,
                "event_categories": "", "top_event_descriptions": "",
            }]),
        )
        snapshots = read_substrate_for_population(
            ["AAPL", "MSFT", "GOOGL"],
            as_of_date=date(2026, 5, 13), s3_client=s3,
        )
        assert set(snapshots.keys()) == {"AAPL", "MSFT", "GOOGL"}
        assert snapshots["AAPL"].news_n_articles == 5
        assert snapshots["MSFT"].news_n_articles == 0
        assert snapshots["GOOGL"].news_n_articles == 0

    def test_only_one_s3_read_per_parquet_for_whole_population(self):
        """Performance pin: 25 tickers should result in O(1) reads per
        parquet type, not O(25). Counted via mock call_count."""
        from unittest.mock import MagicMock

        s3 = MagicMock()
        # Empty body for all reads
        from io import BytesIO as _BytesIO

        def get_object(*, Bucket, Key):
            raise RuntimeError("NoSuchKey")
        s3.get_object.side_effect = get_object

        snapshots = read_substrate_for_population(
            [f"T{i}" for i in range(25)],
            as_of_date=date(2026, 5, 13), s3_client=s3,
        )
        # Reads: 1 news + 1 analyst + (window_days+1) insider attempts.
        # With default window_days=90, max 92 get_object calls regardless
        # of ticker count.
        assert s3.get_object.call_count <= 95
        assert len(snapshots) == 25


# ── Substrate snapshot dataclass ──────────────────────────────────────


def test_substrate_snapshot_frozen():
    snap = SubstrateSnapshot(
        ticker="AAPL", as_of_date=date(2026, 5, 13),
    )
    with pytest.raises(Exception):
        snap.news_n_articles = 5  # type: ignore[misc]


def test_substrate_snapshot_default_fields():
    snap = SubstrateSnapshot(
        ticker="AAPL", as_of_date=date(2026, 5, 13),
    )
    assert snap.has_news_signal is False
    assert snap.has_insider_signal is False
    assert snap.has_analyst_signal is False
    assert snap.news_n_articles_by_source == {}
