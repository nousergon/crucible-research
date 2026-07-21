"""
News fetcher — Yahoo Finance RSS headlines + SEC EDGAR 8-K filings.
Fetches first ~500 chars of article body where available.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime, timedelta

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


_YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_EDGAR_8K_URL = (
    "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom"
    "&startdt={start_date}&enddt={end_date}&forms=8-K"
)
_REQUEST_HEADERS = {
    "User-Agent": "alpha-engine-research/1.0 research@yourdomain.com",
    "Accept-Encoding": "gzip",
}
_FETCH_TIMEOUT = 10  # seconds


def _article_hash(headline: str, source: str) -> str:
    """SHA-256 of headline + source domain for deduplication."""
    content = f"{headline.strip().lower()}|{source.strip().lower()}"
    return hashlib.sha256(content.encode()).hexdigest()


def _fetch_article_excerpt(url: str, max_chars: int = 500) -> str:
    """
    Attempt to fetch the first max_chars of article body text.
    Returns empty string on any failure (network, parsing, etc.).
    """
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove script and style tags
        for tag in soup(["script", "style"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        return text[:max_chars].strip()
    except Exception as e:
        logger.debug("Article excerpt fetch failed for %s: %s", url, e)
        return ""


def fetch_yahoo_news(
    ticker: str,
    hours: int = 48,
    max_articles: int = 10,
    fetch_body: bool = True,
) -> list[dict]:
    """
    Fetch recent news headlines for a ticker via Yahoo Finance RSS.

    Returns list of dicts with keys:
      headline, source, url, published_utc, article_excerpt, article_hash
    """
    url = _YAHOO_RSS_URL.format(ticker=ticker)
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning("Yahoo RSS parse failed for %s: %s", ticker, e)
        return []

    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    articles = []

    for entry in feed.entries[:max_articles]:
        try:
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub:
                pub_dt = datetime(*pub[:6], tzinfo=UTC)
            else:
                pub_dt = datetime.now(UTC)

            if pub_dt < cutoff:
                continue

            headline = entry.get("title", "").strip()
            source = entry.get("source", {}).get("title", "Yahoo Finance")
            link = entry.get("link", "")

            excerpt = ""
            if fetch_body and link:
                time.sleep(0.2)  # polite delay
                excerpt = _fetch_article_excerpt(link)

            articles.append({
                "headline": headline,
                "source": source,
                "url": link,
                "published_utc": pub_dt.isoformat(),
                "article_excerpt": excerpt,
                "article_hash": _article_hash(headline, source),
            })
        except Exception as e:
            logger.debug("Skipping news entry for %s: %s", ticker, e)
            continue

    return articles


def fetch_edgar_8k(ticker: str, days: int = 2) -> list[dict]:
    """
    Fetch recent 8-K filings from SEC EDGAR for a ticker.
    Returns list of dicts: {title, date, url, filing_type}
    """
    end_date = datetime.now(UTC).strftime("%Y-%m-%d")
    start_date = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")

    url = _EDGAR_8K_URL.format(ticker=ticker, start_date=start_date, end_date=end_date)
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("EDGAR 8-K fetch failed for %s: %s", ticker, e)
        return []

    filings = []
    for hit in data.get("hits", {}).get("hits", [])[:5]:
        src = hit.get("_source", {})
        filings.append({
            "title": src.get("display_names", [ticker])[0],
            "date": src.get("file_date", ""),
            "url": f"https://www.sec.gov/Archives/{src.get('file_path', '')}",
            "form_type": src.get("form_type", "8-K"),
            "description": src.get("period_of_report", ""),
        })

    return filings


def fetch_all_news(
    ticker: str,
    hours: int = 48,
) -> dict:
    """
    Convenience function: fetches Yahoo Finance news + EDGAR 8-Ks for a ticker.
    Returns {"yahoo": [...], "edgar_8k": [...]}
    """
    yahoo = fetch_yahoo_news(ticker, hours=hours)
    edgar = fetch_edgar_8k(ticker, days=max(2, hours // 24 + 1))
    if not yahoo and not edgar:
        logger.info("No news found for %s (Yahoo: 0, EDGAR: 0)", ticker)
    return {"yahoo": yahoo, "edgar_8k": edgar}


def fetch_news_batch(
    tickers: list[str],
    hours: int = 48,
) -> dict[str, dict]:
    """
    Fetch news for multiple tickers with aggregate failure logging.
    Returns {ticker: {"yahoo": [...], "edgar_8k": [...]}}
    """
    results = {}
    n_empty = 0
    for ticker in tickers:
        data = fetch_all_news(ticker, hours=hours)
        results[ticker] = data
        if not data["yahoo"] and not data["edgar_8k"]:
            n_empty += 1
    if n_empty == len(tickers) and len(tickers) > 0:
        logger.warning(
            "ALL %d tickers returned zero news articles — "
            "Yahoo RSS or EDGAR may be down",
            len(tickers),
        )
    elif n_empty > 0:
        logger.info("News fetch: %d/%d tickers had zero articles", n_empty, len(tickers))
    return results
