"""Tests for ``graph.research_graph._pre_fetch_held_enrichment``.

ROADMAP P0 (2026-05-13, surfaced by L83 spot-check substrate): the
held-stock ``thesis_update`` path reads from ``ctx.news_data_by_ticker``
and ``ctx.analyst_data_by_ticker`` but those maps were silently empty
in production for two distinct reasons (wrong news function + analyst
never wired). These tests pin the contract so the bug can't regress.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graph.research_graph import _pre_fetch_held_enrichment


# ── News pre-fetch ──────────────────────────────────────────────────────


def test_news_batch_results_populate_articles_per_ticker():
    """Happy path: ``fetch_news_batch`` returns a per-ticker dict;
    the helper flattens yahoo + edgar_8k into a single ``articles`` list."""
    fake_batch = {
        "AAPL": {
            "yahoo": [{"headline": "Apple thing 1"}, {"headline": "Apple thing 2"}],
            "edgar_8k": [{"headline": "Apple 8-K"}],
        },
        "MSFT": {
            "yahoo": [{"headline": "MSFT thing"}],
            "edgar_8k": [],
        },
    }
    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
        return_value=fake_batch,
    ), patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
        return_value={},
    ):
        news, _analyst, _insider = _pre_fetch_held_enrichment(["AAPL", "MSFT"])

    assert news["AAPL"]["article_count"] == 3  # 2 yahoo + 1 edgar
    assert news["MSFT"]["article_count"] == 1
    assert len(news["AAPL"]["articles"]) == 3
    assert news["MSFT"]["articles"][0]["headline"] == "MSFT thing"


def test_news_batch_keyed_by_ticker_not_flat_dict():
    """Regression for the original bug: the helper must NOT treat the
    news_batch return as a flat ``{"yahoo": [...], "edgar_8k": [...]}``
    dict. If a buggy fetcher returned that shape, the helper should
    produce empty articles for each ticker (not crash, not return the
    same flat dict for every ticker)."""
    bug_shape = {"yahoo": [{"headline": "drift"}], "edgar_8k": []}
    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
        return_value=bug_shape,
    ), patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
        return_value={},
    ):
        news, _, _ = _pre_fetch_held_enrichment(["AAPL", "MSFT"])

    # Each ticker gets an empty articles list because bug_shape has no
    # "AAPL" / "MSFT" keys.
    assert news["AAPL"] == {"articles": [], "article_count": 0}
    assert news["MSFT"] == {"articles": [], "article_count": 0}


def test_news_batch_fetch_failure_degrades_to_empty():
    """When the batch fetcher raises, every ticker gets an empty
    articles list — not a crash mid-loop."""
    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
        side_effect=RuntimeError("yahoo down"),
    ), patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
        return_value={},
    ):
        news, _, _ = _pre_fetch_held_enrichment(["AAPL", "MSFT"])

    assert news == {
        "AAPL": {"articles": [], "article_count": 0},
        "MSFT": {"articles": [], "article_count": 0},
    }


def test_news_batch_handles_none_values_gracefully():
    """fetch_news_batch may return None for some tickers; helper
    treats that as empty without raising."""
    fake_batch = {"AAPL": None, "MSFT": {"yahoo": None, "edgar_8k": None}}
    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
        return_value=fake_batch,
    ), patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
        return_value={},
    ):
        news, _, _ = _pre_fetch_held_enrichment(["AAPL", "MSFT"])

    assert news["AAPL"]["article_count"] == 0
    assert news["MSFT"]["article_count"] == 0


# ── Analyst pre-fetch ───────────────────────────────────────────────────


def test_analyst_pre_fetch_populates_per_ticker():
    """The analyst pre-fetch must call fetch_analyst_consensus for every
    ticker — not just initialize the dict (the pre-2026-05-13 bug)."""
    captured: list[str] = []

    def fake_consensus(ticker, current_price=None):
        captured.append(ticker)
        return {"ticker": ticker, "consensus_rating": "buy", "num_analysts": 12}

    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
        return_value={},
    ), patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
        side_effect=fake_consensus,
    ):
        _, analyst, _ = _pre_fetch_held_enrichment(["AAPL", "MSFT", "GOOGL"])

    # Producer was called for every held ticker (the pre-fix bug never
    # called it at all).
    assert captured == ["AAPL", "MSFT", "GOOGL"]
    # Consumer-visible dict is populated for every ticker.
    assert set(analyst.keys()) == {"AAPL", "MSFT", "GOOGL"}
    assert analyst["AAPL"]["consensus_rating"] == "buy"
    assert analyst["AAPL"]["num_analysts"] == 12


def test_analyst_fetch_failure_skips_ticker_but_continues_batch():
    """When fetch_analyst_consensus raises for one ticker, the helper
    continues with the rest (graceful degrade per ticker)."""
    def fake_consensus(ticker, current_price=None):
        if ticker == "BROKEN":
            raise RuntimeError("FMP rate limit hit for one ticker only")
        return {"ticker": ticker, "consensus_rating": "hold"}

    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
        return_value={},
    ), patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
        side_effect=fake_consensus,
    ):
        _, analyst, _ = _pre_fetch_held_enrichment(["AAPL", "BROKEN", "MSFT"])

    assert "AAPL" in analyst
    assert "MSFT" in analyst
    assert "BROKEN" not in analyst  # skipped per the except branch


# ── Insider plumbing ────────────────────────────────────────────────────


def test_insider_data_returned_empty_until_wired():
    """Insider data is plumbed but not yet wired upstream — explicit
    follow-up. Helper should return an empty dict, not None or a
    half-populated structure."""
    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
        return_value={},
    ), patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
        return_value={},
    ):
        _, _, insider = _pre_fetch_held_enrichment(["AAPL"])

    assert insider == {}


# ── Empty population ────────────────────────────────────────────────────


def test_empty_population_returns_empty_maps_without_calling_fetchers():
    """Boundary: an empty population shouldn't trigger any fetcher calls."""
    with patch(
        "data.fetchers.news_fetcher.fetch_news_batch",
    ) as mock_news, patch(
        "data.fetchers.analyst_fetcher.fetch_analyst_consensus",
    ) as mock_analyst:
        mock_news.return_value = {}
        mock_analyst.return_value = {}
        news, analyst, insider = _pre_fetch_held_enrichment([])

    assert news == {}
    assert analyst == {}
    assert insider == {}
    # Empty list still triggers ONE call to fetch_news_batch (returns
    # nothing for free), but ZERO analyst calls.
    mock_news.assert_called_once_with([])
    assert mock_analyst.call_count == 0
