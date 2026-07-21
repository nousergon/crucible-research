"""
News article deduplication.

Uses SHA-256 hashes of (headline + source) to identify articles already
processed in a prior run. Tracks mention_count for recurring themes.

Hash state is persisted in the news_article_hashes SQLite table (see §7.2).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict


def article_hash(headline: str, source: str) -> str:
    """Canonical hash for a news article — matches hash computed in news_fetcher."""
    content = f"{headline.strip().lower()}|{source.strip().lower()}"
    return hashlib.sha256(content.encode()).hexdigest()


def deduplicate_articles(
    articles: list[dict],
    known_hashes: set[str],
) -> tuple[list[dict], list[str]]:
    """
    Filter incoming articles against the set of already-seen hashes.

    Args:
        articles: list of article dicts (must have 'article_hash' key)
        known_hashes: set of hash strings already processed in prior runs

    Returns:
        (novel_articles, new_hash_list)
        novel_articles: articles not seen before
        new_hash_list: hashes of novel articles (to be persisted after run)
    """
    novel = []
    new_hashes = []
    seen_this_run: set[str] = set()

    for article in articles:
        h = article.get("article_hash")
        if not h:
            h = article_hash(
                article.get("headline", ""),
                article.get("source", ""),
            )
            article["article_hash"] = h

        if h in known_hashes or h in seen_this_run:
            continue

        seen_this_run.add(h)
        novel.append(article)
        new_hashes.append(h)

    return novel, new_hashes


def compute_recurring_themes(
    articles: list[dict],
    min_mentions: int = 3,
) -> list[dict]:
    """
    Identify recurring themes across a set of articles by scanning headlines
    for common keywords/phrases. Returns themes with mention_count >= min_mentions.

    Returns list of {theme, mention_count, example_headline}.
    This is a lightweight keyword-frequency approach; LLM agents do deeper analysis.
    """
    from collections import Counter

    # Simple keyword extraction: lower-cased words from headlines
    word_freq: Counter = Counter()
    headline_by_word: dict[str, str] = {}

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "has", "have", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "its", "it", "this", "that", "these", "those", "says", "said",
        "stock", "shares", "market", "company", "report", "new",
    }

    for article in articles:
        headline = article.get("headline", "")
        words = [
            w.strip(".,!?\"'()[]").lower()
            for w in headline.split()
            if len(w) > 4
        ]
        unique_words = set(words) - stop_words
        for w in unique_words:
            word_freq[w] += 1
            if w not in headline_by_word:
                headline_by_word[w] = headline

    themes = []
    for word, count in word_freq.most_common(10):
        if count >= min_mentions:
            themes.append({
                "theme": word,
                "mention_count": count,
                "example_headline": headline_by_word.get(word, ""),
            })

    return themes


def build_known_hashes_from_db(rows: list[dict]) -> dict[str, set[str]]:
    """
    Build a ticker → set[hash] mapping from rows returned by the archive manager.
    Each row should have 'symbol' and 'article_hash' keys.
    """
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        result[row["symbol"]].add(row["article_hash"])
    return dict(result)
