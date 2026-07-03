"""Context assembly — everything the think tank READS (all pre-existing).

The think tank leverages every weekly-SF output it can (Brian, 2026-07-02):
- scanner attractiveness board  → ``scanner/universe/latest.json``
- weekly signals                → ``signals/latest.json`` (sector_ratings,
  market_regime, per-ticker stances)
- weekly macro report           → ``archive/macro/macro_report.md``
- daily news aggregates         → ``data/news_aggregates`` (substrate reader)
- filings corpus                → ``nousergon_lib.rag`` hybrid retrieval

Missing sources degrade the CONTEXT, never silently: each bundle records
which sources were present (surfaced in thesis ``sources_used`` and the run
manifest's ``context_sources_present``), and a WARN is logged per miss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

UNIVERSE_BOARD_KEY = "scanner/universe/latest.json"
SIGNALS_LATEST_KEY = "signals/latest.json"
MACRO_REPORT_KEY = "archive/macro/macro_report.md"

_NEWS_COLS = [
    "ticker",
    "aggregate_date",
    "n_articles",
    "lm_sentiment_mean",
    "lm_sentiment_trusted_mean",
    "event_count",
    "event_severity_max",
    "event_categories",
    "top_event_descriptions",
]


@dataclass
class ContextBundle:
    """Shared read-side state for one run."""

    board: dict | None = None
    signals: dict | None = None
    macro_report_md: str | None = None
    news_by_ticker: dict[str, dict] = field(default_factory=dict)
    rag_available: bool = False
    sources_present: dict[str, bool] = field(default_factory=dict)

    def weekly_signals_date(self) -> str | None:
        return (self.signals or {}).get("date")

    def sector_ratings(self) -> dict:
        return (self.signals or {}).get("sector_ratings", {}) or {}

    def market_regime(self) -> str:
        return (self.signals or {}).get("market_regime", "unknown")


def load_context(store: Any) -> ContextBundle:
    """Load the read-side artifacts. Each miss is a WARN + recorded absence."""
    bundle = ContextBundle()

    bundle.board = store.get_json(UNIVERSE_BOARD_KEY)
    bundle.signals = store.get_json(SIGNALS_LATEST_KEY)
    bundle.macro_report_md = store.get_text(MACRO_REPORT_KEY)

    for name, present in (
        ("universe_board", bundle.board is not None),
        ("signals", bundle.signals is not None),
        ("macro_report", bundle.macro_report_md is not None),
    ):
        bundle.sources_present[name] = present
        if not present:
            logger.warning("thinktank context: %s missing", name)

    bundle.news_by_ticker = _load_news(store)
    bundle.sources_present["news_aggregates"] = bool(bundle.news_by_ticker)

    try:
        from nousergon_lib.rag import is_available

        bundle.rag_available = bool(is_available())
    except Exception as exc:  # noqa: BLE001 — availability probe only
        logger.warning("thinktank context: rag availability probe failed: %s", exc)
        bundle.rag_available = False

    if bundle.rag_available:
        try:
            from nousergon_lib.secrets import get_secret

            if not get_secret("VOYAGE_API_KEY", required=False):
                logger.warning(
                    "thinktank context: rag_filings DB reachable but "
                    "VOYAGE_API_KEY unresolved — per-ticker retrieve() will "
                    "fail; recording rag_filings as absent"
                )
                bundle.rag_available = False
        except Exception as exc:  # noqa: BLE001 — probe only, never raises
            logger.warning(
                "thinktank context: VOYAGE_API_KEY probe failed: %s", exc
            )
            bundle.rag_available = False

    bundle.sources_present["rag_filings"] = bundle.rag_available

    return bundle


def _load_news(store: Any) -> dict[str, dict]:
    """Latest news-aggregate row per ticker (substrate reader, read-only)."""
    try:
        from data.substrate.reader import read_news_aggregates

        df = read_news_aggregates(s3_client=store.s3, bucket=store.bucket)
    except Exception as exc:  # noqa: BLE001 — context source, absence recorded
        logger.warning("thinktank context: news aggregates unreadable: %s", exc)
        return {}
    if df is None or df.empty:
        return {}
    df = df.sort_values("aggregate_date").groupby("ticker", as_index=False).last()
    cols = [c for c in _NEWS_COLS if c in df.columns]
    return {row["ticker"]: {c: row.get(c) for c in cols} for _, row in df.iterrows()}


def filings_excerpts(ticker: str, *, k: int = 6) -> list[str]:
    """Hybrid-retrieve filing chunks for one ticker (mirrors qual_tools)."""
    try:
        from datetime import date, timedelta

        from nousergon_lib.rag import retrieve

        hits = retrieve(
            query=(
                "business model, competitive position, guidance, risk factors, "
                "recent results"
            ),
            tickers=[ticker],
            doc_types=["10-K", "10-Q", "8-K", "earnings_transcript"],
            min_date=date.today() - timedelta(days=730),
            top_k=k,
            method="hybrid",
            vector_weight=0.7,
        )
    except Exception as exc:  # noqa: BLE001 — context source, absence recorded
        logger.warning("thinktank context: rag retrieve failed for %s: %s", ticker, exc)
        return []
    out: list[str] = []
    for r in hits or []:
        text = getattr(r, "content", None)
        if text:
            header = f"[{getattr(r, 'doc_type', '?')} | {getattr(r, 'filed_date', '?')}]"
            out.append(f"{header}\n{str(text)[:1500]}")
    return out
