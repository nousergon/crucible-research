"""
Tools for the Qualitative Analyst agent — LangChain @tool wrappers.

The qual analyst reviews the quant's top 5 picks using qualitative data sources.
Tools are created via factory that closes over shared context (prior_theses).

S3-first: tries pre-collected alternative data from alpha-engine-data repo,
falls back to direct API calls if S3 data is missing.
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.tools import tool

log = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
_MARKET_DATA_PREFIX = "market_data/"

# Rerank configuration (L1303 ROADMAP — alpha-engine-lib v0.11.0 primitive).
# Default empty string ≡ disabled, preserving the hybrid-only path so this
# PR is safe to merge without a config-side change. Operators flip on by
# setting ``RAG_RERANK=cross_encoder`` (default, local BAAI model, zero
# new API surface) or ``RAG_RERANK=llm_judge`` (Anthropic Haiku) in the
# Lambda environment — no redeploy required for the flip itself, only
# for the install of the ``[rerank]`` extra (deferred to PR 3 of the
# L1303 arc, alongside the eval-validated cutover).
_RAG_RERANK = os.environ.get("RAG_RERANK", "").strip() or None
_RAG_RERANK_INPUT_N = int(os.environ.get("RAG_RERANK_INPUT_N", "30"))


def _load_alternative_from_s3(ticker: str) -> dict | None:
    """Try to load pre-collected alternative data for a ticker from S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        ptr = s3.get_object(Bucket=_S3_BUCKET, Key=f"{_MARKET_DATA_PREFIX}latest_weekly.json")
        pointer = json.loads(ptr["Body"].read())
        prefix = pointer.get("s3_prefix", "")
        if not prefix:
            return None
        key = f"{prefix}alternative/{ticker}.json"
        obj = s3.get_object(Bucket=_S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None

# ── RAG usage metrics (module-level, reset per pipeline run) ─────────────────
_rag_stats = {"attempted": 0, "succeeded": 0, "failed": 0}


def get_rag_stats() -> dict:
    """Return RAG query statistics for the current run."""
    return dict(_rag_stats)


def reset_rag_stats() -> None:
    """Reset RAG stats (call at the start of each pipeline run)."""
    _rag_stats["attempted"] = 0
    _rag_stats["succeeded"] = 0
    _rag_stats["failed"] = 0


def create_qual_tools(context: dict) -> list:
    """
    Create LangChain tools for the qual analyst, closing over shared context.

    Args:
        context: Dict with prior_theses and other shared data.

    Returns:
        List of LangChain tool callables.
    """
    prior_theses = context.get("prior_theses", {})
    price_data = context.get("price_data", {})
    episodic_memories = context.get("episodic_memories", {})
    semantic_memories = context.get("semantic_memories", {})

    @tool
    def get_news_articles(ticker: str, days: int = 7) -> str:
        """Get recent news headlines and excerpts for a ticker. Useful for understanding market narrative."""
        # S3-first
        s3_data = _load_alternative_from_s3(ticker)
        if s3_data and s3_data.get("news"):
            news = s3_data["news"]
            articles = news.get("articles", [])
            if articles:
                trimmed = [
                    {"headline": a.get("headline", ""), "source": a.get("source", ""),
                     "published": a.get("published_utc", ""), "excerpt": ""}
                    for a in articles[:10]
                ]
                return json.dumps({"ticker": ticker, "article_count": len(articles), "articles": trimmed})

        from data.fetchers.news_fetcher import fetch_news_for_ticker

        try:
            articles = fetch_news_for_ticker(ticker, lookback_days=days)
            trimmed = [
                {"headline": a.get("headline", ""), "source": a.get("source", ""),
                 "published": a.get("published_utc", ""),
                 "excerpt": (a.get("article_excerpt", "") or "")[:300]}
                for a in articles[:10]
            ]
            return json.dumps({"ticker": ticker, "article_count": len(articles), "articles": trimmed})
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def get_analyst_reports(ticker: str) -> str:
        """Get analyst consensus, price target, rating changes, earnings surprises for a ticker."""
        # S3-first: try pre-collected data
        s3_data = _load_alternative_from_s3(ticker)
        if s3_data and s3_data.get("analyst_consensus"):
            ac = s3_data["analyst_consensus"]
            return json.dumps({
                "ticker": ticker,
                "consensus_rating": ac.get("rating", "N/A"),
                "num_analysts": ac.get("num_analysts", 0),
                "mean_target": ac.get("target_price"),
                "upside_pct": None,
                "rating_changes": [],
                "earnings_surprises": ac.get("earnings_surprises", [])[:4],
            })

        from data.fetchers.analyst_fetcher import fetch_analyst_consensus

        try:
            cp = None
            df = price_data.get(ticker)
            if df is not None and not df.empty and "Close" in df.columns:
                cp = float(df["Close"].iloc[-1])
            data = fetch_analyst_consensus(ticker, current_price=cp)
            return json.dumps({
                "ticker": ticker,
                "consensus_rating": data.get("consensus_rating", "N/A"),
                "num_analysts": data.get("num_analysts", 0),
                "mean_target": data.get("mean_target"),
                "upside_pct": round(data.get("upside_pct", 0), 1) if data.get("upside_pct") else None,
                "rating_changes": data.get("rating_changes", [])[:5],
                "earnings_surprises": data.get("earnings_surprises", [])[:4],
            })
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def get_insider_activity(ticker: str) -> str:
        """Get insider transactions and cluster buying signals. Cluster buying (3+ insiders in 30d) is strongly bullish."""
        # S3-first
        s3_data = _load_alternative_from_s3(ticker)
        if s3_data and s3_data.get("insider_activity"):
            ia = s3_data["insider_activity"]
            return json.dumps({
                "ticker": ticker,
                "cluster_buy": ia.get("cluster_buying", False),
                "unique_buyers_30d": 0,
                "total_buy_value_30d": 0,
                "net_sentiment": 0,
                "recent_transactions": ia.get("transactions", [])[:5],
            })

        from data.fetchers.insider_fetcher import fetch_insider_activity as _fetch

        try:
            data = _fetch(ticker)
            return json.dumps({
                "ticker": ticker,
                "cluster_buy": data.get("cluster_buy", False),
                "unique_buyers_30d": data.get("unique_buyers_30d", 0),
                "total_buy_value_30d": data.get("total_buy_value_30d", 0),
                "net_sentiment": data.get("net_sentiment", 0),
                "recent_transactions": data.get("transactions", [])[:5],
            })
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def get_sec_filings(ticker: str) -> str:
        """Get recent SEC filings (8-K, 10-K, 10-Q) for corporate actions and disclosures."""
        from data.fetchers.news_fetcher import fetch_sec_filings

        try:
            filings = fetch_sec_filings(ticker)
            trimmed = [{"title": f.get("title", ""), "date": f.get("date", ""),
                        "form_type": f.get("form_type", "")} for f in filings[:5]]
            return json.dumps({"ticker": ticker, "filings": trimmed})
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def get_prior_thesis(ticker: str) -> str:
        """Get prior structured thesis (bull/bear case, catalysts, risks). Returns null if never analyzed."""
        thesis = prior_theses.get(ticker)
        if thesis:
            return json.dumps({
                "ticker": ticker,
                "bull_case": thesis.get("bull_case", ""),
                "bear_case": thesis.get("bear_case", ""),
                "catalysts": thesis.get("catalysts", []),
                "risks": thesis.get("risks", []),
                "conviction": thesis.get("conviction_rationale", ""),
                "last_updated": thesis.get("last_updated", ""),
            })
        return json.dumps({"ticker": ticker, "prior_thesis": None})

    @tool
    def get_options_flow(ticker: str) -> str:
        """Get options market signals: put/call ratio, IV rank, expected move."""
        # S3-first
        s3_data = _load_alternative_from_s3(ticker)
        if s3_data and s3_data.get("options_flow"):
            of = s3_data["options_flow"]
            if of.get("put_call_ratio") is not None:
                return json.dumps({
                    "ticker": ticker,
                    "put_call_ratio": round(of.get("put_call_ratio", 1.0), 2),
                    "iv_rank": round(of.get("iv_rank", 50), 1),
                    "expected_move_pct": round(of.get("expected_move_pct", 0), 2),
                })

        from data.fetchers.options_fetcher import fetch_options_data

        try:
            data = fetch_options_data(ticker)
            return json.dumps({
                "ticker": ticker,
                "put_call_ratio": round(data.get("put_call_ratio", 1.0), 2),
                "iv_rank": round(data.get("iv_rank", 50), 1),
                "expected_move_pct": round(data.get("expected_move_pct", 0), 2),
            })
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def get_institutional_activity(ticker: str) -> str:
        """Get 13F institutional accumulation signals. Shows if large funds are building positions."""
        # S3-first
        s3_data = _load_alternative_from_s3(ticker)
        if s3_data and s3_data.get("institutional"):
            inst = s3_data["institutional"]
            return json.dumps({
                "ticker": ticker,
                "n_funds_accumulating": inst.get("funds_increasing", 0),
                "accumulation_signal": inst.get("accumulation", False),
                "total_new_shares": 0,
            })

        from data.fetchers.institutional_fetcher import fetch_institutional_activity as _fetch

        try:
            data = _fetch(ticker)
            return json.dumps({
                "ticker": ticker,
                "n_funds_accumulating": data.get("n_funds_accumulating", 0),
                "accumulation_signal": data.get("accumulation_signal", False),
                "total_new_shares": data.get("total_new_shares", 0),
            })
        except Exception as e:
            return json.dumps({"ticker": ticker, "error": str(e)})

    @tool
    def query_filings(ticker: str, query: str, doc_types: str = "10-K,10-Q,earnings_transcript") -> str:
        """Search SEC filings and earnings transcripts for deep fundamental context.

        Use this when evaluating competitive position, risk factors, management guidance,
        capital allocation strategy, or business model changes. Returns relevant excerpts
        from 10-K/10-Q filings and earnings call transcripts with source metadata.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            query: What to search for (e.g., 'competitive risks and market position')
            doc_types: Comma-separated filing types (default: '10-K,10-Q,earnings_transcript')
        """
        try:
            from alpha_engine_lib.rag import retrieve
            from datetime import date, timedelta

            _rag_stats["attempted"] += 1
            # Hybrid retrieval (vector + BM25 blend) — pgvector cosine on
            # ``embedding`` plus PostgreSQL Full-Text Search (FTS) on
            # ``content_tsv`` blended at vector_weight=0.7. Strong on both
            # conceptual queries (vector side) AND exact-term surfaces like
            # ticker symbols, filing types, and quantitative line items
            # (keyword side).
            #
            # Rerank toggle (env var ``RAG_RERANK``) wraps the hybrid pool
            # with a cross-encoder or LLM-judge reordering pass when set —
            # widens the candidate fetch to ``rerank_input_n`` (default 30)
            # then truncates to ``top_k=8``. Default unset preserves the
            # pre-rerank behavior so this PR ships safe.
            retrieve_kwargs = {
                "query": query,
                "tickers": [ticker],
                "doc_types": [d.strip() for d in doc_types.split(",")],
                "min_date": date.today() - timedelta(days=730),
                "top_k": 8,
                "method": "hybrid",
                "vector_weight": 0.7,
            }
            if _RAG_RERANK:
                retrieve_kwargs["rerank"] = _RAG_RERANK
                retrieve_kwargs["rerank_input_n"] = _RAG_RERANK_INPUT_N
            results = retrieve(**retrieve_kwargs)
            if not results:
                return f"No filing data found for {ticker}."

            _rag_stats["succeeded"] += 1
            # Structured INFO log for decision-artifact capture + LangSmith
            # observability. Per-result component scores let the eval
            # harness in PR 3 read calibration data straight from prod logs
            # without needing a side-channel; rerank_score is populated
            # post-rerank, ``None`` otherwise.
            log.info(
                "RAG_RETRIEVE ticker=%s method=hybrid vector_weight=0.7 "
                "top_k=8 rerank=%s rerank_input_n=%s n_results=%d component_scores=%s",
                ticker,
                _RAG_RERANK or "none",
                _RAG_RERANK_INPUT_N if _RAG_RERANK else 0,
                len(results),
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
            formatted = []
            for r in results:
                formatted.append(
                    f"[{r.doc_type} | {r.filed_date} | {r.section_label}]\n{r.content}"
                )
            return "\n\n---\n\n".join(formatted)
        except Exception as e:
            _rag_stats["failed"] += 1
            log.warning("RAG_UNAVAILABLE ticker=%s error_type=%s error=%s", ticker, type(e).__name__, e)
            return f"Filing search temporarily unavailable for {ticker}."

    @tool
    def get_lessons(ticker: str) -> str:
        """Get lessons from past signal outcomes for a ticker. Shows what went wrong with previous BUY signals."""
        memories = episodic_memories.get(ticker, [])
        if not memories:
            return json.dumps({"ticker": ticker, "lessons": [], "note": "No prior outcome lessons available"})
        items = []
        for m in memories[:5]:
            items.append({
                "signal_date": m.get("signal_date", ""),
                "score": m.get("score"),
                "outcome_10d": m.get("outcome_10d"),
                "outcome_vs_spy": m.get("outcome_vs_spy"),
                "lesson": m.get("lesson", ""),
                "pattern_tags": m.get("pattern_tags", ""),
            })
        return json.dumps({"ticker": ticker, "lessons": items})

    @tool
    def get_sector_insights(sector: str) -> str:
        """Get cross-team observations and insights for a sector from prior runs."""
        memories = semantic_memories.get(sector, [])
        if not memories:
            return json.dumps({"sector": sector, "insights": [], "note": "No prior sector insights available"})
        items = []
        for m in memories[:5]:
            items.append({
                "source": m.get("source", ""),
                "content": m.get("content", ""),
                "created_date": m.get("created_date", ""),
            })
        return json.dumps({"sector": sector, "insights": items})

    return [
        get_news_articles,
        get_analyst_reports,
        get_insider_activity,
        get_sec_filings,
        get_prior_thesis,
        get_options_flow,
        get_institutional_activity,
        query_filings,
        get_lessons,
        get_sector_insights,
    ]
