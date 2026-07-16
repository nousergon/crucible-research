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
# Default empty string ≡ disabled, preserving the hybrid-only path
# (operator-validated optimal per 2026-05-12 eval — both rerank variants
# regressed against hybrid w=0.7 baseline). Operators flip on by setting
# ``RAG_RERANK=cross_encoder`` (local BAAI bge-reranker-v2-m3, zero API
# surface) in the Lambda environment — no redeploy required for the
# flip itself, only for the install of the ``[rerank]`` extra.
#
# ``RAG_RERANK=llm_judge`` was removed lib v0.34.0 (2026-05-25) per
# ``[[preference_llm_calls_confined_to_research_module]]`` + the no-lift
# finding. CE stays for future domain-finetune retries; if revisiting
# rerank, EXPERIMENTS.md captures the institutional path (finetune CE
# on retrieval-log triples, not LLM-judge).
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


def _opt_float(v) -> float | None:
    """Return float or None for nullable numeric fields (NaN, None, empty)."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
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

        # config#1822: this used to import a `fetch_news_for_ticker` symbol
        # that has never existed in data.fetchers.news_fetcher (drifted
        # since the 2026-03-22 sector-team-agents commit, 22cad19b — the
        # real entry point is `fetch_all_news`, which returns
        # {"yahoo": [...], "edgar_8k": [...]}). The deferred import meant
        # this ImportError only fired at tool-call time (never at module
        # load / deploy), so it went undetected for ~3 months: every
        # qual-analyst call to this tool silently errored, burning tool-
        # call budget for zero information and contributing to the
        # 90-102-tool-call/0-assessment pattern in config#1822.
        from data.fetchers.news_fetcher import fetch_all_news

        try:
            news = fetch_all_news(ticker, hours=days * 24)
            articles = news.get("yahoo", []) + news.get("edgar_8k", [])
            trimmed = [
                {"headline": a.get("headline", "") or a.get("title", ""),
                 "source": a.get("source", ""),
                 "published": a.get("published_utc", "") or a.get("date", ""),
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

        # config#1821 Option B (2026-07-08): fetch_analyst_consensus no
        # longer sources consensus_rating / mean_target / num_analysts /
        # rating_changes — those came from FMP's grades-consensus /
        # price-target-consensus endpoints, which 402'd for every ticker
        # on the current plan and were removed from the feature contract.
        # Only earnings_surprises (a different, still-live v3 endpoint)
        # remains in this fallback path.
        from data.fetchers.analyst_fetcher import fetch_analyst_consensus

        try:
            cp = None
            df = price_data.get(ticker)
            if df is not None and not df.empty and "Close" in df.columns:
                cp = float(df["Close"].iloc[-1])
            data = fetch_analyst_consensus(ticker, current_price=cp)
            return json.dumps({
                "ticker": ticker,
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
        """Get recent SEC filings (8-K) for corporate actions and disclosures."""
        # config#1822: `fetch_sec_filings` has never existed in
        # data.fetchers.news_fetcher (see get_news_articles above for the
        # same drift). The module's only SEC-filings fetcher is
        # `fetch_edgar_8k` (8-K only — there is no 10-K/10-Q fetcher in
        # this module; deep filing text is covered separately by the
        # query_filings/search_filings RAG tools below). Narrowing the
        # docstring to match what this tool actually returns.
        from data.fetchers.news_fetcher import fetch_edgar_8k

        try:
            filings = fetch_edgar_8k(ticker, days=90)
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

        # The yfinance options fetcher was removed (yfinance-centralization
        # arc, 2026-05-16); the live `fetch_options_data` import was already
        # dead (symbol never existed). S3-first above is the working path;
        # graceful-degrade per the tool's `{"error": ...}` contract when the
        # alternative collector hasn't populated options_flow — never raise
        # (all-agents-strict).
        return json.dumps({
            "ticker": ticker,
            "error": "options flow unavailable (no S3 options data; yfinance fetcher removed)",
        })

    @tool
    def get_institutional_activity(ticker: str) -> str:
        """Get 13F institutional ownership signals: fund count, QoQ share/value changes,
        top-fund concentration. Shows if large funds are building or reducing positions."""
        # S3-first: pre-collected alternative data (legacy path)
        s3_data = _load_alternative_from_s3(ticker)
        if s3_data and s3_data.get("institutional"):
            inst = s3_data["institutional"]
            return json.dumps({
                "ticker": ticker,
                "n_funds_accumulating": inst.get("funds_increasing", 0),
                "accumulation_signal": inst.get("accumulation", False),
                "total_new_shares": 0,
            })

        # inst_ownership derived table (built from SEC quarterly bulk Form 13F data)
        try:
            import boto3 as _boto3
            from data.substrate.reader import read_inst_ownership

            s3 = _boto3.client("s3")
            df = read_inst_ownership(s3_client=s3, bucket=_S3_BUCKET)
            if df is not None and len(df) > 0 and "ticker" in df.columns:
                row = df[df["ticker"] == ticker.upper()]
                if len(row) > 0:
                    r = row.iloc[0]
                    return json.dumps({
                        "ticker": ticker.upper(),
                        "n_funds_holding": int(r.get("n_funds_holding", 0)),
                        "total_shares_held": float(r.get("total_shares_held", 0)),
                        "shares_qoq_change": _opt_float(r.get("shares_qoq_change")),
                        "value_qoq_change": _opt_float(r.get("value_qoq_change")),
                        "top5_concentration_pct": _opt_float(r.get("top5_concentration_pct")),
                        "n_funds_increasing": int(r.get("n_funds_increasing", 0)),
                        "n_funds_decreasing": int(r.get("n_funds_decreasing", 0)),
                        "n_funds_new": int(r.get("n_funds_new", 0)),
                        "n_funds_exited": int(r.get("n_funds_exited", 0)),
                        "source": "sec_13f_bulk",
                    })
        except Exception as e:
            log.debug("inst_ownership read failed for %s: %s", ticker, e)

        return json.dumps({
            "ticker": ticker,
            "error": "institutional ownership data unavailable (no 13F data for this ticker)",
        })

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
            from nousergon_lib.rag import retrieve
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
                "outcome_21d": m.get("outcome_21d"),
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

    # Wave 1 PR E (data-revamp-260513.md): RAG retrieval tools that
    # consume the producer-side substrate (news → RAG via PR A.3,
    # filings via existing 8-K/10-K/Q pipelines + PR B Form 4
    # parquet). search_news/filings/transcripts wrap
    # alpha_engine_lib.rag.retrieve with doc_type-scoped queries +
    # shared stats + structured INFO logging.
    from agents.sector_teams.rag_retrieval_tools import (
        build_rag_retrieval_tools,
    )
    rag_tools = build_rag_retrieval_tools()

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
        *rag_tools,  # search_news, search_filings, search_transcripts
    ]
