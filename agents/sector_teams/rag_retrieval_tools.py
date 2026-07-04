"""Shared RAG retrieval tools — wraps ``nousergon_lib.rag.retrieve``
for use by qual / quant / thesis_update agents.

Wave 1 PR E of the institutional data-revamp arc (plan doc:
``~/Development/alpha-engine-docs/private/data-revamp-260513.md``).

Pairs with the producer-side ingest pipelines in alpha-engine-data:

  - PR A.3 (#229): news ingested with ``doc_type="news"`` and
    ``source="news_polygon"`` / ``"news_gdelt"`` / ``"news_yahoo_rss"``
  - PR B (#230): Form 4 insider transactions (structured parquet —
    not in RAG corpus; queried elsewhere)
  - Existing: 10-K / 10-Q / 8-K / earnings_transcript / thesis

All retrieval goes through the existing hybrid (BM25 + pgvector)
retriever in nousergon_lib.rag — same one the qual analyst's
``query_filings`` tool already calls. This module wraps it with:

  - **Source-filter helpers** so callers can scope to "news only" /
    "filings only" / "all" without each tool reimplementing doc_type
    sets.
  - **Structured INFO log per call** mirroring ``query_filings``'s
    LangSmith-observable shape — decision-capture sees every RAG hit.
  - **Per-stats counters** (attempts / successes / failures) for
    pipeline-level observability.
  - **Graceful degrade** — RAG unavailable returns a structured "no
    data" string rather than crashing the agent.

Tools exposed:

  - ``search_news(ticker, query, days_back)`` — recent news context.
    Doc types: ``["news"]``. Configurable lookback (default 30 days).
  - ``search_filings(ticker, query, forms)`` — SEC filings text.
    Doc types parameterized — caller can narrow to specific forms.
    Default covers the full expanded set: 10-K / 10-Q / 8-K / 14A /
    S-1 / S-4 / 13D / 13G / 13F.
  - ``search_transcripts(ticker, query, quarters_back)`` — earnings
    call transcripts. Doc types: ``["earnings_transcript"]``.

A future PR adds ``get_filing_full_text(accession_number)`` for
deep dives on a single document.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any

log = logging.getLogger(__name__)


# Rerank configuration shared with qual_tools.py.
_RAG_RERANK = os.environ.get("RAG_RERANK", "").strip() or None
_RAG_RERANK_INPUT_N = int(os.environ.get("RAG_RERANK_INPUT_N", "30"))


# RAG usage metrics (module-level, reset per pipeline run).
_rag_stats = {
    "search_news": {"attempted": 0, "succeeded": 0, "failed": 0},
    "search_filings": {"attempted": 0, "succeeded": 0, "failed": 0},
    "search_transcripts": {"attempted": 0, "succeeded": 0, "failed": 0},
}


def get_rag_retrieval_stats() -> dict:
    """Return per-tool stats. Composes with qual_tools.get_rag_stats
    which tracks the legacy ``query_filings`` tool."""
    return {
        tool: dict(counts) for tool, counts in _rag_stats.items()
    }


def reset_rag_retrieval_stats() -> None:
    for counts in _rag_stats.values():
        counts["attempted"] = 0
        counts["succeeded"] = 0
        counts["failed"] = 0


# Default doc_type sets per tool. Defaults narrow the candidate
# corpus before the hybrid retriever scores — significantly faster
# than scoring across the whole corpus.
DEFAULT_FILINGS_DOC_TYPES = (
    "10-K", "10-Q", "8-K", "14A", "S-1", "S-4", "13D", "13G", "13F",
)


# ── Helpers ────────────────────────────────────────────────────────────


def _do_retrieve(
    *,
    query: str,
    ticker: str,
    doc_types: list[str],
    min_date: date,
    top_k: int,
    retrieve_fn: Any = None,
):
    """Run one hybrid retrieval. Centralizes the rerank-toggle +
    structured-log emission so each tool wrapper doesn't repeat it.

    ``retrieve_fn`` is injectable for tests; production uses
    ``nousergon_lib.rag.retrieve``."""
    if retrieve_fn is None:
        from nousergon_lib.rag import retrieve
        retrieve_fn = retrieve
    retrieve_kwargs = {
        "query": query,
        "tickers": [ticker],
        "doc_types": doc_types,
        "min_date": min_date,
        "top_k": top_k,
        "method": "hybrid",
        "vector_weight": 0.7,
    }
    if _RAG_RERANK:
        retrieve_kwargs["rerank"] = _RAG_RERANK
        retrieve_kwargs["rerank_input_n"] = _RAG_RERANK_INPUT_N
    results = retrieve_fn(**retrieve_kwargs)
    return results


def _emit_rag_log(
    *,
    tool: str,
    ticker: str,
    doc_types: list[str],
    n_results: int,
    results,
) -> None:
    """Emit the structured INFO log line — same shape as
    qual_tools.query_filings so existing decision-capture +
    LangSmith parsers pick it up."""
    log.info(
        "RAG_RETRIEVE tool=%s ticker=%s doc_types=%s method=hybrid "
        "vector_weight=0.7 rerank=%s rerank_input_n=%s n_results=%d "
        "component_scores=%s",
        tool, ticker, ",".join(doc_types),
        _RAG_RERANK or "none",
        _RAG_RERANK_INPUT_N if _RAG_RERANK else 0,
        n_results,
        [
            {
                "chunk_id": r.chunk_id,
                "vector_score": r.vector_score,
                "keyword_score": r.keyword_score,
                "combined_score": r.combined_score,
                "rerank_score": getattr(r, "rerank_score", None),
                "rerank_method": getattr(r, "rerank_method", None),
            }
            for r in results
        ],
    )


def _format_results(results, *, include_source: bool = False) -> str:
    """Compose the agent-visible text bundle from the retrieval results.

    Same pattern as query_filings — one section per result, separated
    by --- horizontal rules. ``include_source`` adds the vendor source
    column (useful for news where the same article may appear from
    multiple sources).
    """
    parts = []
    for r in results:
        header_bits = [r.doc_type, str(r.filed_date)]
        if include_source:
            header_bits.append(getattr(r, "source", "") or "?")
        section = getattr(r, "section_label", None) or ""
        if section:
            header_bits.append(section)
        parts.append(
            f"[{' | '.join(header_bits)}]\n{r.content}"
        )
    return "\n\n---\n\n".join(parts)


# ── search_news ─────────────────────────────────────────────────────────


def search_news_impl(
    ticker: str,
    query: str,
    *,
    days_back: int = 30,
    top_k: int = 8,
    retrieve_fn: Any = None,
) -> str:
    """Module-level implementation of the search_news tool.

    Returns a multi-section formatted string suitable as agent context
    text. Graceful degrade: returns a structured "no data" message
    rather than raising.
    """
    _rag_stats["search_news"]["attempted"] += 1
    try:
        results = _do_retrieve(
            query=query,
            ticker=ticker,
            doc_types=["news"],
            min_date=date.today() - timedelta(days=days_back),
            top_k=top_k,
            retrieve_fn=retrieve_fn,
        )
        if not results:
            return f"No recent news found for {ticker} matching '{query}'."
        _rag_stats["search_news"]["succeeded"] += 1
        _emit_rag_log(
            tool="search_news",
            ticker=ticker, doc_types=["news"],
            n_results=len(results), results=results,
        )
        return _format_results(results, include_source=True)
    except Exception as e:
        _rag_stats["search_news"]["failed"] += 1
        log.warning(
            "RAG_UNAVAILABLE tool=search_news ticker=%s "
            "error_type=%s error=%s",
            ticker, type(e).__name__, e,
        )
        return f"News search temporarily unavailable for {ticker}."


# ── search_filings ──────────────────────────────────────────────────────


def search_filings_impl(
    ticker: str,
    query: str,
    *,
    forms: str = ",".join(DEFAULT_FILINGS_DOC_TYPES),
    days_back: int = 730,
    top_k: int = 8,
    retrieve_fn: Any = None,
) -> str:
    """Module-level search_filings.

    Args:
        ticker: stock symbol
        query: natural-language query
        forms: comma-separated form types. Defaults to the full
               expanded SEC set (10-K / 10-Q / 8-K / 14A / S-1 / S-4 /
               13D / 13G / 13F). Caller can narrow to a single form.
        days_back: lookback window (default 2 years for filings).
        top_k: max results to return.
    """
    _rag_stats["search_filings"]["attempted"] += 1
    doc_types = [d.strip() for d in forms.split(",") if d.strip()]
    if not doc_types:
        doc_types = list(DEFAULT_FILINGS_DOC_TYPES)
    try:
        results = _do_retrieve(
            query=query,
            ticker=ticker,
            doc_types=doc_types,
            min_date=date.today() - timedelta(days=days_back),
            top_k=top_k,
            retrieve_fn=retrieve_fn,
        )
        if not results:
            return f"No filings found for {ticker} matching '{query}'."
        _rag_stats["search_filings"]["succeeded"] += 1
        _emit_rag_log(
            tool="search_filings",
            ticker=ticker, doc_types=doc_types,
            n_results=len(results), results=results,
        )
        return _format_results(results, include_source=False)
    except Exception as e:
        _rag_stats["search_filings"]["failed"] += 1
        log.warning(
            "RAG_UNAVAILABLE tool=search_filings ticker=%s "
            "error_type=%s error=%s",
            ticker, type(e).__name__, e,
        )
        return f"Filing search temporarily unavailable for {ticker}."


# ── search_transcripts ──────────────────────────────────────────────────


def search_transcripts_impl(
    ticker: str,
    query: str,
    *,
    quarters_back: int = 4,
    top_k: int = 6,
    retrieve_fn: Any = None,
) -> str:
    """Module-level search_transcripts.

    Limits to ``["earnings_transcript"]`` doc type and a quarter-aware
    lookback (default 4 quarters ≈ 365 days). Use to surface
    management commentary on margins, guidance, or specific business-
    segment dynamics.
    """
    _rag_stats["search_transcripts"]["attempted"] += 1
    days_back = quarters_back * 95  # ~3 months per quarter + slack
    try:
        results = _do_retrieve(
            query=query,
            ticker=ticker,
            doc_types=["earnings_transcript"],
            min_date=date.today() - timedelta(days=days_back),
            top_k=top_k,
            retrieve_fn=retrieve_fn,
        )
        if not results:
            return (
                f"No earnings transcripts found for {ticker} matching "
                f"'{query}' in the last {quarters_back} quarters."
            )
        _rag_stats["search_transcripts"]["succeeded"] += 1
        _emit_rag_log(
            tool="search_transcripts",
            ticker=ticker, doc_types=["earnings_transcript"],
            n_results=len(results), results=results,
        )
        return _format_results(results, include_source=False)
    except Exception as e:
        _rag_stats["search_transcripts"]["failed"] += 1
        log.warning(
            "RAG_UNAVAILABLE tool=search_transcripts ticker=%s "
            "error_type=%s error=%s",
            ticker, type(e).__name__, e,
        )
        return f"Transcript search temporarily unavailable for {ticker}."


# ── @tool wrappers (LangChain agent integration) ──────────────────────


def build_rag_retrieval_tools() -> list:
    """Build the three @tool-decorated callables for inclusion in a
    LangChain ReAct agent's toolset.

    Factory pattern so test code can call the underlying ``_impl``
    functions directly (cleaner than monkeypatching the decorator).
    """
    from langchain_core.tools import tool

    @tool
    def search_news(ticker: str, query: str, days_back: int = 30) -> str:
        """Search recent news articles for a ticker.

        Use this for material-event context, sentiment cross-check, or
        specific event drill-down (M&A rumors, regulatory actions,
        analyst upgrades). Returns recent news excerpts indexed across
        Polygon / GDELT / Yahoo Finance / EDGAR press releases.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            query: What to search for (e.g., 'FDA approval timeline')
            days_back: Lookback window in days (default 30)
        """
        return search_news_impl(ticker, query, days_back=days_back)

    @tool
    def search_filings(
        ticker: str, query: str,
        forms: str = ",".join(DEFAULT_FILINGS_DOC_TYPES),
    ) -> str:
        """Search SEC filings for a ticker across the expanded form set.

        Covers 10-K / 10-Q / 8-K / 14A (proxy) / S-1 (IPO) / S-4 (M&A) /
        13D / 13G (5%+ ownership) / 13F (institutional positioning).
        Use for fundamentals, governance, capital structure, insider/
        institutional positioning context.

        Args:
            ticker: Stock symbol
            query: Natural-language query
            forms: Comma-separated form types to narrow the search
                   (default: all expanded SEC types)
        """
        return search_filings_impl(ticker, query, forms=forms)

    @tool
    def search_transcripts(
        ticker: str, query: str, quarters_back: int = 4,
    ) -> str:
        """Search earnings call transcripts for management commentary.

        Use to surface guidance specifics, margin trajectory, segment
        dynamics, or competitive positioning the company discussed
        with sell-side analysts. Defaults to the last 4 quarters.

        Args:
            ticker: Stock symbol
            query: What to search for (e.g., 'cloud revenue guidance')
            quarters_back: Quarters of history to search (default 4)
        """
        return search_transcripts_impl(
            ticker, query, quarters_back=quarters_back,
        )

    return [search_news, search_filings, search_transcripts]
