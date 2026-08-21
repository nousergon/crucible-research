"""Context assembly — everything the think tank READS (all pre-existing).

The think tank leverages every weekly-SF output it can (Brian, 2026-07-02)
plus institutional substrate feeds (Brian, 2026-07-13):
- scanner attractiveness board  → ``scanner/universe/latest.json``
- weekly signals                → ``signals/latest.json`` (sector_ratings,
  market_regime, per-ticker stances)
- weekly regime substrate       → ``regime/`` (HMM posterior, composite
  intensity, BOCPD change signal, guardrails — the macro ANCHOR)
- daily news aggregates         → ``data/news_aggregates`` (substrate reader)
- insider transactions          → ``data/insider_transactions`` (90d rollup)
- analyst revisions             → ``data/analyst_revisions`` (consensus deltas)
- institutional ownership (13F) → ``data/inst_ownership`` (QoQ deltas)
- filings corpus                → ``nousergon_lib.rag`` hybrid retrieval

Missing sources degrade the CONTEXT, never silently: each bundle records
which sources were present (surfaced in thesis ``sources_used`` and the run
manifest's ``context_sources_present``), and a WARN is logged per miss.

PRESENCE IS NOT FRESHNESS (alpha-engine-config-I2638). Absence was handled
from the start; staleness was not. Every dated context source is checked
through ``freshness.assert_upstream_fresh`` and every non-fresh verdict is
recorded on ``ContextBundle.freshness``, which the run manifest and the theme
artifacts both carry.

MACRO ANCHOR — ``regime/``, NOT ``archive/macro/macro_report.md``
(Brian ruling 2026-08-21, resolving the open sub-question named inline in
alpha-engine-config-I2638). The Markdown macro report's producer
(``ArchiveManager.save_macro_report``) lost its only call site when the
multi-agent graph retired, so the object froze at 2026-03-16 and Think Tank
reconciled its themes against a five-month-old backdrop for 158 days. The
freshness primitive made that visible; the ruling removes the source rather
than resurrecting a retired agent path. Think Tank now self-anchors on two
inputs that have live producers:

- the weekly ``RegimeSubstrate`` Lambda's artifact (HMM posterior + composite
  intensity z + BOCPD change signal + guardrail flags + raw macro features),
  read through ``nousergon_lib.eval_artifacts`` and dated in-band by its own
  ``calendar_date`` — no object-metadata inference needed; and
- the daily news aggregates, already loaded here and already feeding the
  intraweek ``developments`` leg via ``analyst.sweep``.

Both are freshness-checked. Neither may silently freeze the way the Markdown
report did: a substrate whose ``calendar_date`` stops moving trips the weekly
tolerance, and a news table whose ``aggregate_date`` stops moving trips the
daily one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from freshness import FreshnessVerdict, assert_upstream_fresh

logger = logging.getLogger(__name__)

UNIVERSE_BOARD_KEY = "scanner/universe/latest.json"
SIGNALS_LATEST_KEY = "signals/latest.json"

#: The regime substrate's canonical S3 prefix. ``regime/latest.json`` is a
#: pure pointer sidecar; the payloads live at ``regime/{YYMMDDHHMM}.json``.
#: Resolution is delegated to ``nousergon_lib.eval_artifacts`` so this
#: consumer shares the producer's addressing convention rather than
#: re-deriving it (shared-code-policy — second adoption of the same read).
REGIME_SUBSTRATE_PREFIX = "regime"

#: The substrate is written by the Saturday SF's ``RegimeSubstrate`` state,
#: so a healthy one is at most 7 days old; ``weekly`` tolerance (10d) leaves
#: room for one skipped run.
REGIME_SUBSTRATE_CADENCE = "weekly"

#: News aggregates are a daily producer. They are the intraweek half of the
#: macro anchor (the substrate is the weekly half), so their staleness is
#: load-bearing and gets its own verdict rather than riding on presence.
NEWS_AGGREGATES_CADENCE = "daily"

#: Why Think Tank DEGRADES rather than raises on a stale macro anchor: the
#: daily shadow run's contract with the trading day is explicitly
#: non-blocking, so a hard stop here would take the challenger arm offline
#: over an input whose own producer has independent weekly-SF monitoring.
#: The failure mode accepted is "themes reconciled against an out-of-date
#: macro backdrop"; the surfaces that record it are the ops alert raised by
#: the primitive, the run manifest's ``context_source_freshness``, and
#: ``ThemeThesis.stale_inputs`` on every theme written — plus the banner
#: injected into the macro prompt so the model itself is told how old the
#: anchor is.
_ANCHOR_DEGRADED_REASON = (
    "think tank's daily shadow run is non-blocking by contract; a stale macro "
    "anchor degrades the themes it must not halt the run. Recorded on the run "
    "manifest (context_source_freshness), on every ThemeThesis (stale_inputs), "
    "and alerted via ops_alerts."
)

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
    #: Latest weekly regime-substrate payload — Think Tank's macro anchor.
    regime_substrate: dict | None = None
    news_by_ticker: dict[str, dict] = field(default_factory=dict)
    insider_by_ticker: dict[str, dict] = field(default_factory=dict)
    analyst_by_ticker: dict[str, dict] = field(default_factory=dict)
    inst_ownership_by_ticker: dict[str, dict] = field(default_factory=dict)
    rag_available: bool = False
    sources_present: dict[str, bool] = field(default_factory=dict)
    #: ``{source_name: FreshnessVerdict}`` for every source with a checkable
    #: as-of timestamp — the fresh ones too. A source that emits no freshness
    #: verdict is unobserved, not healthy (observability-policy §3.4/§8.3).
    freshness: dict[str, FreshnessVerdict] = field(default_factory=dict)

    def stale_sources(self) -> list[str]:
        return sorted(n for n, v in self.freshness.items() if not v.is_fresh)

    def freshness_records(self) -> dict[str, dict]:
        """JSON-safe, for the run manifest."""
        return {n: v.as_record() for n, v in self.freshness.items()}

    def staleness_banners(self) -> list[str]:
        return [v.banner() for n, v in sorted(self.freshness.items()) if not v.is_fresh]

    def stale_input_records(self) -> list[dict]:
        """The non-fresh verdicts only — stamped onto Think Tank's own output
        so a downstream reader of a theme can see what anchored it."""
        return [
            v.as_record() for _, v in sorted(self.freshness.items()) if not v.is_fresh
        ]

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
    bundle.regime_substrate = _load_regime_substrate(store)

    # The substrate dates ITSELF (``calendar_date``), so unlike the retired
    # Markdown macro report there is no object-metadata inference here. A
    # payload that cannot be read, or one carrying no usable date, yields an
    # ``undated`` verdict — loud, never silently fresh.
    bundle.freshness["regime_substrate"] = assert_upstream_fresh(
        f"{REGIME_SUBSTRATE_PREFIX}/",
        as_of=_regime_substrate_as_of(bundle.regime_substrate),
        cadence=REGIME_SUBSTRATE_CADENCE,
        on_stale="degrade",
        degraded_reason=_ANCHOR_DEGRADED_REASON,
        source="research:thinktank_daily",
    )

    for name, present in (
        ("universe_board", bundle.board is not None),
        ("signals", bundle.signals is not None),
        ("regime_substrate", bundle.regime_substrate is not None),
    ):
        bundle.sources_present[name] = present
        if not present:
            logger.warning("thinktank context: %s missing", name)

    bundle.news_by_ticker = _load_news(store)
    bundle.sources_present["news_aggregates"] = bool(bundle.news_by_ticker)
    # The intraweek half of the macro anchor. Checked even when EMPTY: an
    # absent news table is ``undated``, which is the loud reading — a macro
    # anchor with no live intraweek leg must not look healthy.
    bundle.freshness["news_aggregates"] = assert_upstream_fresh(
        "data/news_aggregates",
        as_of=_news_aggregates_as_of(bundle.news_by_ticker),
        cadence=NEWS_AGGREGATES_CADENCE,
        on_stale="degrade",
        degraded_reason=_ANCHOR_DEGRADED_REASON,
        source="research:thinktank_daily",
    )

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


def _load_regime_substrate(store: Any) -> dict | None:
    """Latest regime-substrate payload, or None when it cannot be read.

    Swallowed failure mode: the substrate is unreachable (missing sidecar,
    malformed pointer, absent artifact body, parse or S3 error). Swallowed
    because Think Tank's daily shadow run is non-blocking by contract and the
    substrate Lambda carries its own weekly-SF freshness monitoring. The
    recording surfaces are this WARN, the ``undated`` freshness verdict the
    caller derives from the ``None`` (alerted + stamped on every theme), and
    ``sources_present["regime_substrate"]`` on the run manifest.
    """
    try:
        from nousergon_lib.eval_artifacts import load_latest_eval_artifact

        return load_latest_eval_artifact(
            store.s3, bucket=store.bucket, prefix=REGIME_SUBSTRATE_PREFIX
        )
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning("thinktank context: regime substrate unreadable: %s", exc)
        return None


def _regime_substrate_as_of(substrate: dict | None) -> Any | None:
    """The substrate's own as-of date, or None (⇒ ``undated``, never fresh).

    ``calendar_date`` is the dual-tracked wall-clock date the producer stamps
    (``DATE_CONVENTIONS.md``); ``model_metadata.written_at`` is the fallback
    for a payload predating that field. Deliberately NOT derived from the S3
    object's ``LastModified``: a re-upload of an old payload would read as
    fresh, which is the exact failure this consumer just came out of.
    """
    if not isinstance(substrate, dict):
        return None
    stamped = substrate.get("calendar_date")
    if stamped:
        return stamped
    meta = substrate.get("model_metadata")
    if isinstance(meta, dict):
        return meta.get("written_at")
    return None


def _news_aggregates_as_of(news_by_ticker: dict[str, dict]) -> Any | None:
    """Newest ``aggregate_date`` across the loaded news rows, or None.

    None (no rows, or no row carrying a date) is ``undated`` — the loud
    reading. An empty news table is unobserved, not calm.
    """
    dates = [
        row.get("aggregate_date")
        for row in news_by_ticker.values()
        if row.get("aggregate_date")
    ]
    if not dates:
        return None
    return max(str(d) for d in dates)


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
    from datetime import date

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
