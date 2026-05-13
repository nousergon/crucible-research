"""Yahoo Finance RSS news adapter — fallback / cross-validation source.

Wraps the legacy ``data.fetchers.news_fetcher.fetch_yahoo_news`` (which
also feeds the post-#170 `_pre_fetch_held_enrichment` path) and
normalizes its output into the canonical ``NewsArticle`` shape.

Trust weight in config should be low (~0.5) — RSS is consumer-grade,
headlines-only, frequent dupes of wire stories Polygon already
indexed. Kept as a fallback / coverage-expansion source, not a primary.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from data.sources.protocols import NewsArticle

logger = logging.getLogger(__name__)


class YahooRssNewsAdapter:
    """Yahoo RSS adapter wrapping the legacy fetcher."""

    name = "yahoo_rss"

    def __init__(self, fetcher: Any = None) -> None:
        self._fetcher = fetcher

    def _get_fetcher(self) -> Any:
        if self._fetcher is None:
            from data.fetchers.news_fetcher import fetch_yahoo_news
            self._fetcher = fetch_yahoo_news
        return self._fetcher

    def fetch(
        self,
        tickers: list[str],
        *,
        hours: int = 48,
    ) -> list[NewsArticle]:
        articles: list[NewsArticle] = []
        for ticker in tickers:
            try:
                items = self._get_fetcher()(ticker, hours=hours)
            except Exception as e:
                logger.warning(
                    "[yahoo_rss] fetch failed for %s: %s", ticker, e
                )
                continue
            for item in items or []:
                article = _to_article(item, ticker=ticker)
                if article is not None:
                    articles.append(article)
        return articles


def _to_article(item: dict, *, ticker: str) -> NewsArticle | None:
    """Map one legacy yahoo-news entry to canonical ``NewsArticle``.

    Legacy shape (per ``data/fetchers/news_fetcher.py``):
      {headline, source, url, published_utc, article_excerpt, article_hash}
    """
    try:
        url = item.get("url") or ""
        if not url:
            return None
        published = item.get("published_utc")
        if isinstance(published, str):
            published_dt = datetime.fromisoformat(
                published.replace("Z", "+00:00")
            )
        elif isinstance(published, datetime):
            published_dt = published
        else:
            return None
        return NewsArticle(
            tickers=(ticker,),
            title=item.get("headline") or "",
            body_excerpt=item.get("article_excerpt") or "",
            url=url,
            published_at=published_dt,
            source="yahoo_rss",
            vendor_article_id=item.get("article_hash"),
            fetched_at=datetime.now(timezone.utc),
            headline_authors=None,
            tags=tuple(filter(None, [item.get("source")])),
        )
    except Exception as e:
        logger.warning("[yahoo_rss] schema drift on item: %s", e)
        return None
