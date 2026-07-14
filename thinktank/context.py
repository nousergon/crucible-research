"""Context assembly — everything the think tank READS (all pre-existing).

The think tank leverages every weekly-SF output it can (Brian, 2026-07-02)
plus institutional substrate feeds (Brian, 2026-07-13):
- scanner attractiveness board  → ``scanner/universe/latest.json``
- weekly signals                → ``signals/latest.json`` (sector_ratings,
  market_regime, per-ticker stances)
- weekly macro report           → ``archive/macro/macro_report.md``
- daily news aggregates         → ``data/news_aggregates`` (substrate reader)
- insider transactions          → ``data/insider_transactions`` (90d rollup)
- analyst revisions             → ``data/analyst_revisions`` (consensus deltas)
- institutional ownership (13F) → ``data/inst_ownership`` (QoQ deltas)
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

_INSIDER_ROLLUP_COLS = [
    "insider_n_transactions_90d",
    "insider_n_buys_90d",
    "insider_n_sells_90d",
    "insider_net_dollar_flow_90d",
    "insider_distinct_insiders_90d",
    "insider_max_single_transaction_usd",
]

_ANALYST_COLS = [
    "ticker",
    "mean_target_current",
    "mean_target_delta_30d",
    "mean_target_pct_change_30d",
    "num_analysts_current",
    "num_analysts_delta_30d",
    "consensus_rating_current",
    "rating_changed_30d",
]

_INST_OWNERSHIP_COLS = [
    "ticker",
    "quarter",
    "n_funds_holding",
    "total_shares_held",
    "total_value_usd",
    "shares_qoq_change",
    "value_qoq_change",
    "top5_concentration_pct",
    "n_funds_increasing",
    "n_funds_decreasing",
]


@dataclass
class ContextBundle:
    """Shared read-side state for one run."""

    board: dict | None = None
    signals: dict | None = None
    macro_report_md: str | None = None
    news_by_ticker: dict[str, dict] = field(default_factory=dict)
    insider_by_ticker: dict[str, dict] = field(default_factory=dict)
    analyst_by_ticker: dict[str, dict] = field(default_factory=dict)
    inst_ownership_by_ticker: dict[str, dict] = field(default_factory=dict)
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

    bundle.insider_by_ticker = _load_insider_transactions(store)
    bundle.sources_present["insider_transactions"] = bool(bundle.insider_by_ticker)

    bundle.analyst_by_ticker = _load_analyst_revisions(store)
    bundle.sources_present["analyst_revisions"] = bool(bundle.analyst_by_ticker)

    bundle.inst_ownership_by_ticker = _load_inst_ownership(store)
    bundle.sources_present["inst_ownership"] = bool(bundle.inst_ownership_by_ticker)

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


def _load_insider_transactions(store: Any, *, window_days: int = 90) -> dict[str, dict]:
    """Roll up Form 4 transactions per ticker (trailing window, read-only).

    Follows the same rollup logic as ``SubstrateReader`` / ``_insider_rollup``
    in ``data.substrate.reader``, returning per-ticker dicts matching
    ``_INSIDER_ROLLUP_COLS`` for direct prompt consumption.
    """
    from datetime import date, timedelta

    from data.substrate.reader import read_insider_transactions_window

    try:
        df = read_insider_transactions_window(
            date.today(),
            window_days=window_days,
            s3_client=store.s3,
            bucket=store.bucket,
        )
    except Exception as exc:  # noqa: BLE001 — context source, absence recorded
        logger.warning("thinktank context: insider transactions unreadable: %s", exc)
        return {}
    if df is None or len(df) == 0 or "ticker" not in df.columns:
        return {}
    result: dict[str, dict] = {}
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker]
        is_buy = sub["acquired_disposed_code"] == "A"
        is_sell = sub["acquired_disposed_code"] == "D"
        values = sub["transaction_value_usd"].fillna(0.0).astype(float)
        net_flow = float(values[is_buy].sum() - values[is_sell].sum())
        max_tx = float(values.abs().max()) if len(values) > 0 else 0.0
        distinct = (
            int(sub["reporting_owner_name"].nunique())
            if "reporting_owner_name" in sub.columns
            else 0
        )
        result[ticker] = {
            "insider_n_transactions_90d": int(len(sub)),
            "insider_n_buys_90d": int(is_buy.sum()),
            "insider_n_sells_90d": int(is_sell.sum()),
            "insider_net_dollar_flow_90d": net_flow,
            "insider_distinct_insiders_90d": distinct,
            "insider_max_single_transaction_usd": max_tx,
        }
    return result


def _load_analyst_revisions(store: Any) -> dict[str, dict]:
    """Latest analyst-revision row per ticker (substrate reader, read-only)."""
    try:
        from data.substrate.reader import read_analyst_revisions

        df = read_analyst_revisions(s3_client=store.s3, bucket=store.bucket)
    except Exception as exc:  # noqa: BLE001 — context source, absence recorded
        logger.warning("thinktank context: analyst revisions unreadable: %s", exc)
        return {}
    if df is None or df.empty:
        return {}
    df = df.sort_values("as_of_date").groupby("ticker", as_index=False).last()
    cols = [c for c in _ANALYST_COLS if c in df.columns]
    return {row["ticker"]: {c: row.get(c) for c in cols} for _, row in df.iterrows()}


def _load_inst_ownership(store: Any) -> dict[str, dict]:
    """Latest 13F institutional-ownership row per ticker (derived table, read-only).

    The ``inst_ownership`` table is produced by the alpha-engine-data
    pipeline from SEC quarterly Form 13F bulk data. Returns the latest
    quarter's data per ticker.
    """
    try:
        from data.substrate.reader import read_inst_ownership

        df = read_inst_ownership(s3_client=store.s3, bucket=store.bucket)
    except Exception as exc:  # noqa: BLE001 — context source, absence recorded
        logger.warning("thinktank context: inst_ownership unreadable: %s", exc)
        return {}
    if df is None or df.empty:
        return {}
    df = df.sort_values("quarter").groupby("ticker", as_index=False).last()
    cols = [c for c in _INST_OWNERSHIP_COLS if c in df.columns]
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
