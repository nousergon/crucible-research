"""
Tools for the Quant Analyst agent — LangChain @tool wrappers around existing fetchers.

Tools are created via factory functions that close over shared context (price_data,
technical_scores). The quant agent calls these via LangGraph's create_react_agent.
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.tools import tool

from scoring.composite import compute_factor_subscore
from scoring.factor_scoring import read_factor_profiles_from_s3
from scoring.focus_list import _assign_stance

log = logging.getLogger(__name__)


def read_fundamentals_from_s3(
    run_date: str | None = None,
    bucket: str | None = None,
) -> dict[str, dict] | None:
    """Read the weekly Finnhub fundamentals snapshot from S3.

    Produced by alpha-engine-data's ``collectors/fundamentals.py`` (weekly
    DataPhase1) and written to ``archive/fundamentals/{date}.json``. The
    predictor's ``inference/stages/fetch_alt_data.py`` already reads the
    same key — this mirrors its read + date-resolution pattern: try the
    given/today's date first, then fall back to scanning the prefix for
    the most-recent snapshot (the snapshot is a slow-moving weekly signal,
    so a stale-by-days fallback is acceptable and matches predictor
    behavior).

    Returns ``{ticker: {pe_ratio, pb_ratio, debt_to_equity,
    revenue_growth_yoy, fcf_yield, gross_margin, roe, current_ratio}}``
    (the data-module's normalized/clipped schema — NOT raw Finnhub keys).
    Returns ``None`` on any read failure; the caller graceful-degrades per
    its existing ``{"error": ...}`` contract (must not raise —
    all-agents-strict).
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket = bucket or os.environ.get(
        "RESEARCH_BUCKET",
        os.environ.get("S3_BUCKET", "alpha-engine-research"),
    )
    try:
        s3 = boto3.client("s3")
        if run_date:
            try:
                obj = s3.get_object(
                    Bucket=bucket, Key=f"archive/fundamentals/{run_date}.json"
                )
                return json.loads(obj["Body"].read())
            except ClientError:
                pass
        # Scan for the most-recent snapshot (mirrors predictor's fallback).
        resp = s3.list_objects_v2(
            Bucket=bucket, Prefix="archive/fundamentals/", MaxKeys=100
        )
        keys = sorted(
            (
                c["Key"]
                for c in resp.get("Contents", [])
                if c["Key"].endswith(".json")
            ),
            reverse=True,
        )
        if not keys:
            return None
        obj = s3.get_object(Bucket=bucket, Key=keys[0])
        log.info("get_balance_sheet: using fundamentals snapshot %s", keys[0])
        return json.loads(obj["Body"].read())
    except Exception as e:
        log.warning("Fundamentals S3 read failed: %s", e)
        return None


def create_quant_tools(context: dict) -> list:
    """
    Create LangChain tools for the quant analyst, closing over shared context.

    Args:
        context: Shared data dict with price_data, technical_scores, and
            optionally factor_profiles (Phase 1c artifact) + market_regime
            + factor_blend_regime_weights for the get_factor_profile tool.
            When factor_profiles isn't in the context, the tool factory
            falls back to reading factors/profiles/latest.json from S3
            once at create-time and closing over the result.

    Returns:
        List of LangChain tool callables.
    """
    price_data = context.get("price_data", {})
    technical_scores = context.get("technical_scores", {})
    # Factor substrate Phase 2: per-ticker within-sector percentile-ranked
    # factor composites (quality/momentum/value/low_vol) + a regime blend
    # for stance / focus_score derivation. Cached at tool-creation time so
    # the ReAct loop doesn't re-read S3 on every tool invocation. One read
    # per sector team per Saturday SF run = 6 reads/week of a single small
    # JSON, well within boto budget.
    factor_profiles: dict[str, dict] | None = context.get("factor_profiles")
    if factor_profiles is None:
        factor_profiles = read_factor_profiles_from_s3() or {}
    # Fundamentals snapshot for get_balance_sheet. Cached once at
    # tool-creation time (one S3 read per sector team per Saturday SF run,
    # same pattern as factor_profiles) so the ReAct loop never re-reads S3.
    # Read from alpha-engine-data's weekly archive/fundamentals/{date}.json
    # (yfinance .info removed — yfinance-centralization arc, 2026-05-16).
    fundamentals_data: dict[str, dict] | None = context.get("fundamentals_data")
    if fundamentals_data is None:
        fundamentals_data = read_fundamentals_from_s3() or {}
    market_regime: str = context.get("market_regime", "neutral")
    factor_blend_regime_weights: dict | None = context.get(
        "factor_blend_regime_weights"
    )
    # PR 4 of scanner-placement arc:
    #   - focus_list_tickers: set of tickers in THIS team's focus list. Tools
    #     consult this set to determine whether a lookup constitutes an
    #     agent_override (tool call on a ticker outside the focus list).
    #     Empty set → no focus list this cycle; override tagging is skipped.
    #   - override_tickers: shared mutable list (passed by reference from
    #     sector_team via run_quant_analyst) that the get_factor_profile
    #     wrapper appends to when invoked on a non-focus ticker. Aggregated
    #     by archive_writer for the scanner_evaluations agent_override
    #     column. Deliberately a list (not a set) so the audit captures
    #     repeated lookups — useful signal about how often the agent
    #     revisits a non-focus name.
    focus_list_tickers: set = set(context.get("focus_list_tickers", set()))
    override_tickers: list = context.get("override_tickers", [])

    # Set of valid tickers the LLM can reference. Used to reject
    # hallucinated tickers (e.g. 2026-04-11 "CARRIER" instead of "CARR")
    # before they hit external APIs.
    #
    # Closes 5/23-SF P0 sweep (q) — the pre-fix `set(price_data.keys())`
    # was the NARROW set: `research_graph._fetch_team_data` excludes
    # feature-store-covered tickers from the raw-OHLCV fetch unless they
    # also live in `population_tickers`, so `price_data` ended up with
    # only ~25-30 population tickers + a few uncovered ones. The sector
    # team's actual scope is ~50 tickers (scanner-universe slice), and
    # `technical_scores` covers the full scope (feature-store + raw-OHLCV
    # union). The 2026-05-24 audit on the trading-day-fix recovery's
    # research Lambda CW Logs saw dozens of `unknown ticker — rejecting`
    # WARNINGs across SBUX/NKE/AAPL/MSFT/NVDA-class S&P names — all of
    # them in feature_store + technical_scores, but missing from
    # price_data because they weren't in population. That mass rejection
    # is the most likely root cause of the Technology + Consumer team
    # 0-picks-across-2-attempts retry exhaustion ((r) in the same sweep).
    #
    # Post-fix: take the UNION of every ticker-source threaded into the
    # tool context. Anything the agent's prompt names is in scope; only
    # genuinely hallucinated tickers (CARRIER, etc.) fall through.
    # focus_list_tickers is the team's tactical focus; factor_profiles
    # + fundamentals_data are pre-loaded ticker→record dicts; price_data
    # + technical_scores cover the raw-OHLCV and feature-store paths.
    _valid_tickers: set[str] = set()
    if price_data:
        _valid_tickers |= set(price_data.keys())
    if technical_scores:
        _valid_tickers |= set(technical_scores.keys())
    if factor_profiles:
        _valid_tickers |= set(factor_profiles.keys())
    if fundamentals_data:
        _valid_tickers |= set(fundamentals_data.keys())
    if focus_list_tickers:
        _valid_tickers |= focus_list_tickers

    def _validate_tickers(tickers: list[str], tool_name: str) -> tuple[list[str], dict[str, str]]:
        """Split incoming tickers into (valid, errors_dict).

        The errors dict maps the rejected ticker to an error message
        suitable for returning directly to the LLM as a tool result.
        """
        valid: list[str] = []
        errors: dict[str, str] = {}
        for t in tickers:
            if not isinstance(t, str) or not t.strip():
                errors[str(t)] = "empty or non-string ticker"
                continue
            t_clean = t.strip().upper()
            if _valid_tickers and t_clean not in _valid_tickers:
                errors[t] = (
                    f"unknown ticker '{t}' — not in the current sector "
                    f"universe. Re-check the ticker symbol (e.g. Carrier "
                    f"Global is CARR, not CARRIER). Valid tickers only."
                )
                log.warning(
                    "[%s] LLM passed unknown ticker '%s' — rejecting",
                    tool_name, t,
                )
                continue
            valid.append(t_clean)
        return valid, errors

    @tool
    def screen_by_volume(tickers: list[str], min_volume: float) -> str:
        """Filter tickers by minimum 20-day average daily volume. Returns tickers meeting threshold."""
        passing = []
        for t in tickers:
            df = price_data.get(t)
            if df is not None and len(df) >= 20 and "Volume" in df.columns:
                avg_vol = df["Volume"].tail(20).mean()
                if avg_vol >= min_volume:
                    passing.append({"ticker": t, "avg_volume_20d": int(avg_vol)})
        return json.dumps({"passing_tickers": len(passing), "tickers": passing[:50]})

    @tool
    def get_technical_indicators(tickers: list[str]) -> str:
        """Get technical indicators: RSI(14), MACD, price vs MA50/MA200, momentum, ATR%, technical_score (0-100)."""
        results = {}
        for t in tickers:
            ts = technical_scores.get(t, {})
            if ts:
                results[t] = {
                    "rsi_14": round(ts.get("rsi_14", 0), 1),
                    "macd_cross": ts.get("macd_cross", False),
                    "price_vs_ma50": round(ts.get("price_vs_ma50", 0), 2),
                    "price_vs_ma200": round(ts.get("price_vs_ma200", 0), 2),
                    "momentum_20d": round(ts.get("momentum_20d", 0), 2),
                    "atr_pct": round(ts.get("atr_pct", 0), 2),
                    "technical_score": round(ts.get("technical_score", 0), 1),
                }
            else:
                results[t] = {"error": "no data available"}
        return json.dumps(results)

    # get_analyst_consensus (analyst ratings / price targets) was removed
    # config#1821 Option B (2026-07-08): it wrapped FMP's grades-consensus
    # / price-target-consensus endpoints, which 402'd for every ticker on
    # the current plan — the tool always returned "N/A"/0/None, burning an
    # LLM tool-call for zero information (the same dead-tool-call pattern
    # as config#1822). Not replaced: no other quant tool consumed this data.

    @tool
    def get_balance_sheet(tickers: list[str]) -> str:
        """Get balance sheet metrics: debt/equity, current ratio, PE, revenue growth, gross margins."""
        # Reads alpha-engine-data's weekly Finnhub fundamentals snapshot
        # (closed over at tool-creation time) instead of yfinance .info
        # (yfinance-centralization arc, 2026-05-16). The data-module schema
        # is normalized/clipped (not raw Finnhub keys) — the quant agent
        # consumes these as soft directional context, not a hard gate, so
        # the trailing-vs-TTM / scale delta is tolerable. D/E is already a
        # ratio in the snapshot — NO yfinance %/100 scaling is applied.
        # forward_pe + market_cap are not persisted by the collector, so
        # they are reported as None per the existing optional-field contract.
        # Graceful-degrade to {"error": ...} on missing ticker/snapshot —
        # never raises (all-agents-strict).
        valid, errors = _validate_tickers(tickers, "get_balance_sheet")
        results: dict = dict(errors)
        for t in valid[:20]:
            f = fundamentals_data.get(t)
            if not f:
                results[t] = {
                    "error": (
                        "no fundamentals snapshot for this ticker "
                        "(archive/fundamentals/{date}.json)"
                    )
                }
                continue
            results[t] = {
                "debt_to_equity": f.get("debt_to_equity"),
                "current_ratio": f.get("current_ratio"),
                "market_cap": None,
                "pe_ratio": f.get("pe_ratio"),
                "forward_pe": None,
                "price_to_book": f.get("pb_ratio"),
                "revenue_growth": f.get("revenue_growth_yoy"),
                "gross_margins": f.get("gross_margin"),
            }
        return json.dumps(results)

    @tool
    def get_price_performance(tickers: list[str]) -> str:
        """Get recent price performance: 5d, 20d, 60d returns and current price."""
        results = {}
        for t in tickers:
            df = price_data.get(t)
            if df is None or len(df) < 5:
                results[t] = {"error": "insufficient price data"}
                continue
            close = df["Close"] if "Close" in df.columns else df["Adj Close"]
            current = float(close.iloc[-1])
            results[t] = {"current_price": round(current, 2)}
            for label, days in [("5d", 5), ("20d", 20), ("60d", 60)]:
                if len(close) >= days:
                    prior = float(close.iloc[-days])
                    results[t][f"return_{label}"] = round((current / prior - 1) * 100, 2)
        return json.dumps(results)

    @tool
    def get_options_flow(tickers: list[str]) -> str:
        """Get options signals: put/call ratio, IV rank, expected move. Gauges market sentiment."""
        # The yfinance options fetcher was removed (yfinance-centralization
        # arc, 2026-05-16); the live `fetch_options_data` import was already
        # dead (symbol never existed) so this tool had no working data path.
        # Graceful-degrade per the tool's existing `{"error": ...}` contract —
        # never raise (all-agents-strict). Options positioning is sourced
        # S3-first by qual_tools.get_options_flow when the alternative
        # collector populates it.
        results = {
            t: {"error": "options flow unavailable (yfinance fetcher removed)"}
            for t in tickers[:10]
        }
        return json.dumps(results)

    @tool
    def get_factor_profile(tickers: list[str]) -> str:
        """Get systematic factor exposures: within-sector percentile-ranked
        quality / momentum / value / low_vol composites (0-100 within-sector,
        higher = stronger on that axis) + the dominant factor stance +
        regime-blended focus score. Use this to reason about WHY a name is
        attractive in factor terms — is it a momentum bet, a quality
        compounder, a value play, or low-vol defensive? In BULL regimes,
        momentum + quality lead; in BEAR, low-vol + quality lead. Pair with
        get_balance_sheet for fundamental cross-check and get_technical_indicators
        for setup-quality confirmation. Coverage flag `_n` per composite
        counts how many raw factors contributed (4 = full data, < 4 =
        partial coverage / wider error band)."""
        valid, errors = _validate_tickers(tickers, "get_factor_profile")
        results: dict = dict(errors)
        for t in valid[:20]:
            # agent_override telemetry (PR 4 of scanner-placement arc):
            # tool call on a ticker outside the team's focus list = the
            # agent reaching outside the curated set. Append even on
            # missing-profile (the intent was to check) — archive_writer
            # aggregates per-team. Skipped when focus_list_tickers is
            # empty (no focus list this cycle).
            if focus_list_tickers and t not in focus_list_tickers:
                override_tickers.append(t)
            profile = factor_profiles.get(t) if factor_profiles else None
            if profile is None:
                results[t] = {"error": "no factor profile available — ticker may be missing from factors/profiles/latest.json"}
                continue

            # Regime-blended focus score (same formula as score_aggregator's
            # factor_subscore + scoring/focus_list.py — single source of
            # truth so tuning one tunes both). None when blend weights
            # aren't configured for the current regime.
            focus_score: float | None = None
            breakdown: dict = {}
            if factor_blend_regime_weights:
                focus_score, details = compute_factor_subscore(
                    profile, market_regime, factor_blend_regime_weights,
                )
                breakdown = details.get("breakdown", {})

            results[t] = {
                "sector": profile.get("sector"),
                "quality_score": profile.get("quality_score"),
                "momentum_score": profile.get("momentum_score"),
                "value_score": profile.get("value_score"),
                "low_vol_score": profile.get("low_vol_score"),
                "quality_n": profile.get("quality_n"),
                "momentum_n": profile.get("momentum_n"),
                "value_n": profile.get("value_n"),
                "low_vol_n": profile.get("low_vol_n"),
                "stance": _assign_stance(profile),
                "focus_score": focus_score,
                "regime": market_regime,
                "factor_blend_breakdown": breakdown,
            }
        return json.dumps(results)

    # Wave 1 PR E (data-revamp-260513.md): RAG retrieval tools so the
    # quant agent can read news context that contextualizes a technical
    # setup (e.g. "RSI oversold — but what news happened in the last
    # week?"). Filings + transcripts available too for cross-check on
    # fundamental triggers behind technical patterns.
    from agents.sector_teams.rag_retrieval_tools import (
        build_rag_retrieval_tools,
    )
    rag_tools = build_rag_retrieval_tools()

    return [
        screen_by_volume,
        get_technical_indicators,
        get_balance_sheet,
        get_price_performance,
        get_options_flow,
        get_factor_profile,
        *rag_tools,
    ]
