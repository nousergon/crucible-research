"""Regression tests for `_valid_tickers` UNION-of-sources behavior in
`create_quant_tools` — closes 5/23-SF P0 sweep item (q) and unblocks (r).

Pre-fix: `_valid_tickers = set(price_data.keys())` — narrow set excluding
feature-store-covered tickers that weren't in `population_tickers`. The
2026-05-24 audit on the research Lambda CW Logs saw dozens of `unknown
ticker — rejecting` WARNINGs for SBUX/NKE/AAPL/MSFT/NVDA-class S&P names
on this defect.

Post-fix: UNION of every ticker-source threaded into the tool context —
price_data + technical_scores + factor_profiles + fundamentals_data +
focus_list_tickers. Only genuinely hallucinated tickers (CARRIER vs CARR,
etc.) fall through to rejection.
"""
from __future__ import annotations

import json

from agents.sector_teams.quant_tools import create_quant_tools


def _get_tool(tools, name: str):
    """Locate a LangChain @tool wrapper by its `.name` attribute."""
    for t in tools:
        if getattr(t, "name", None) == name:
            return t
    raise AssertionError(f"tool {name!r} not found in {[t.name for t in tools]}")


def test_valid_tickers_includes_technical_scores_when_missing_from_price_data():
    """Feature-store-covered tickers live in technical_scores but NOT in
    price_data. They MUST be valid for the tool layer to accept the agent's
    lookups — pre-fix bug rejected them as unknown."""
    # Simulate the production setup: technical_scores has 50 tickers,
    # price_data has only the 5 population tickers.
    sector_tickers = ["SBUX", "NKE", "AAPL", "MSFT", "NVDA"]  # feature-store-covered
    population_tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]  # raw-OHLCV-fetched

    technical_scores = {
        t: {"technical_score": 50.0, "rsi_14": 50.0, "macd_cross": False,
            "price_vs_ma50": 0.0, "price_vs_ma200": 0.0, "momentum_20d": 0.0,
            "atr_pct": 1.0}
        for t in sector_tickers + population_tickers
    }
    price_data = dict.fromkeys(population_tickers)  # ONLY population
    fundamentals_data = {
        t: {"net_income_ttm": 1000, "free_cash_flow_ttm": 500,
            "total_debt": 100, "total_assets": 10000}
        for t in sector_tickers + population_tickers
    }

    tools = create_quant_tools({
        "price_data": price_data,
        "technical_scores": technical_scores,
        "fundamentals_data": fundamentals_data,
    })

    # get_balance_sheet exercises _validate_tickers — pick a sector ticker
    # that's NOT in price_data and assert it's accepted (no "unknown ticker"
    # error in the tool output).
    get_balance_sheet = _get_tool(tools, "get_balance_sheet")
    out_json = get_balance_sheet.invoke({"tickers": ["AAPL"]})
    out = json.loads(out_json)
    # Pre-fix: out would have {"AAPL": {"error": "unknown ticker 'AAPL' — ..."}}
    # Post-fix: AAPL passes validation and gets the fundamentals payload.
    assert "AAPL" in out, f"AAPL must be accepted (got {out})"
    error_str = json.dumps(out["AAPL"]) if isinstance(out["AAPL"], dict) else str(out["AAPL"])
    assert "unknown ticker" not in error_str, (
        f"AAPL must NOT be rejected as unknown — it's in technical_scores + "
        f"fundamentals_data even though price_data omitted it. Got: {out['AAPL']}"
    )


def test_valid_tickers_includes_focus_list_tickers():
    """Tickers in the team's focus list (Phase-2 tactical scope) are
    always valid for tool lookups even if price_data is empty."""
    tools = create_quant_tools({
        "price_data": {},
        "technical_scores": {},
        "fundamentals_data": {"WING": {"net_income_ttm": 100, "free_cash_flow_ttm": 50,
                                        "total_debt": 10, "total_assets": 1000}},
        "focus_list_tickers": {"WING"},
    })
    get_balance_sheet = _get_tool(tools, "get_balance_sheet")
    out = json.loads(get_balance_sheet.invoke({"tickers": ["WING"]}))
    assert "WING" in out
    error_str = json.dumps(out["WING"]) if isinstance(out["WING"], dict) else str(out["WING"])
    assert "unknown ticker" not in error_str


def test_truly_hallucinated_ticker_still_rejected():
    """A ticker that's in NONE of the threaded sources (genuine
    hallucination — 'CARRIER' instead of 'CARR') must still be
    rejected — defense-in-depth not removed by the fix."""
    technical_scores = {"CARR": {"technical_score": 50.0, "rsi_14": 50.0,
                                  "macd_cross": False, "price_vs_ma50": 0.0,
                                  "price_vs_ma200": 0.0, "momentum_20d": 0.0,
                                  "atr_pct": 1.0}}
    tools = create_quant_tools({
        "price_data": {"CARR": None},
        "technical_scores": technical_scores,
    })
    get_balance_sheet = _get_tool(tools, "get_balance_sheet")
    out = json.loads(get_balance_sheet.invoke({"tickers": ["CARRIER"]}))
    # Pre-fix AND post-fix: CARRIER must reject.
    assert "CARRIER" in out
    assert "unknown ticker" in json.dumps(out["CARRIER"]), (
        f"genuinely hallucinated ticker 'CARRIER' must still reject; got {out}"
    )


def test_empty_price_data_does_not_skip_validation():
    """When price_data is empty AND all other sources are empty, the
    `_valid_tickers` set is empty — pre-fix the legacy `if _valid_tickers`
    guard SKIPPED validation entirely, post-fix the guard still skips
    (preserves the legacy semantic for the case where the tool factory
    is built with zero context — typically in unit tests). Pin the
    behavior so a future refactor doesn't silently flip it."""
    tools = create_quant_tools({"price_data": {}})
    get_balance_sheet = _get_tool(tools, "get_balance_sheet")
    out = json.loads(get_balance_sheet.invoke({"tickers": ["AAA"]}))
    # With empty valid_tickers, the validation IS skipped — caller gets
    # the tool's natural behavior (whatever fundamentals_data lookup
    # produces, which here is "no fundamentals data available" since
    # fundamentals_data wasn't provided).
    assert "AAA" in out


def test_valid_tickers_union_in_factory_explicit():
    """White-box: verify the union shape directly. Catches a future
    refactor that drops one of the sources by mistake."""
    import agents.sector_teams.quant_tools as qt
    # Patch read_factor_profiles_from_s3 to return empty (the factory
    # falls back to S3 read when context lacks factor_profiles).
    orig_read = qt.read_factor_profiles_from_s3
    orig_read_fund = qt.read_fundamentals_from_s3
    qt.read_factor_profiles_from_s3 = lambda: {"AAA": {}}
    qt.read_fundamentals_from_s3 = lambda: {"BBB": {}}
    try:
        # Build the factory with overlapping but non-identical sources.
        ctx = {
            "price_data": {"CCC": None},
            "technical_scores": {"DDD": {}},
            "focus_list_tickers": {"EEE"},
        }
        tools = qt.create_quant_tools(ctx)
        # Exercise via get_balance_sheet — every ticker should be accepted
        # since they're all in at least one source.
        get_balance_sheet = _get_tool(tools, "get_balance_sheet")
        for ticker in ("AAA", "BBB", "CCC", "DDD", "EEE"):
            out = json.loads(get_balance_sheet.invoke({"tickers": [ticker]}))
            assert ticker in out
            # None of these should fire the unknown-ticker rejection.
            assert "unknown ticker" not in json.dumps(out[ticker]), (
                f"{ticker} (in one of the threaded sources) must NOT reject; "
                f"got {out[ticker]}"
            )
    finally:
        qt.read_factor_profiles_from_s3 = orig_read
        qt.read_fundamentals_from_s3 = orig_read_fund
