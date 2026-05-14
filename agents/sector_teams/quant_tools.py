"""
Tools for the Quant Analyst agent — LangChain @tool wrappers around existing fetchers.

Tools are created via factory functions that close over shared context (price_data,
technical_scores). The quant agent calls these via LangGraph's create_react_agent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from scoring.composite import compute_factor_subscore
from scoring.factor_scoring import read_factor_profiles_from_s3
from scoring.focus_list import _assign_stance

log = logging.getLogger(__name__)


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
    # hallucinated tickers before they hit external APIs. Observed
    # 2026-04-11: a quant ReAct agent called get_balance_sheet(
    # tickers=["CARRIER"]) instead of ["CARR"] and yfinance returned
    # 404 on Carrier Global. Validating here returns a clear error
    # to the LLM so it can retry with the correct symbol.
    _valid_tickers = set(price_data.keys()) if price_data else set()

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

    @tool
    def get_analyst_consensus(tickers: list[str]) -> str:
        """Get analyst ratings, price targets, earnings surprises for up to 5 tickers (FMP daily limit). Pass your top candidates only. Returns consensus_rating, num_analysts, mean_target, upside_pct."""
        from data.fetchers.analyst_fetcher import fetch_analyst_consensus as _fetch

        valid, errors = _validate_tickers(tickers, "get_analyst_consensus")
        results: dict = dict(errors)
        for t in valid[:5]:
            try:
                # Pass yfinance close price to avoid a separate FMP quote call
                cp = None
                df = price_data.get(t)
                if df is not None and not df.empty and "Close" in df.columns:
                    cp = float(df["Close"].iloc[-1])
                data = _fetch(t, current_price=cp)
                results[t] = {
                    "consensus_rating": data.get("consensus_rating", "N/A"),
                    "num_analysts": data.get("num_analysts", 0),
                    "mean_target": data.get("mean_target"),
                    "upside_pct": round(data.get("upside_pct", 0), 1) if data.get("upside_pct") else None,
                }
            except Exception as e:
                results[t] = {"error": str(e)}
        return json.dumps(results)

    @tool
    def get_balance_sheet(tickers: list[str]) -> str:
        """Get balance sheet metrics: debt/equity, current ratio, PE, revenue growth, gross margins."""
        import yfinance as yf

        valid, errors = _validate_tickers(tickers, "get_balance_sheet")
        results: dict = dict(errors)
        for t in valid[:20]:
            try:
                info = yf.Ticker(t).info
                results[t] = {
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "price_to_book": info.get("priceToBook"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "gross_margins": info.get("grossMargins"),
                }
            except Exception as e:
                results[t] = {"error": str(e)}
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
        from data.fetchers.options_fetcher import fetch_options_data

        results = {}
        for t in tickers[:10]:
            try:
                data = fetch_options_data(t)
                results[t] = {
                    "put_call_ratio": round(data.get("put_call_ratio", 1.0), 2),
                    "iv_rank": round(data.get("iv_rank", 50), 1),
                    "expected_move_pct": round(data.get("expected_move_pct", 0), 2),
                }
            except Exception as e:
                results[t] = {"error": str(e)}
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
        get_analyst_consensus,
        get_balance_sheet,
        get_price_performance,
        get_options_flow,
        get_factor_profile,
        *rag_tools,
    ]
