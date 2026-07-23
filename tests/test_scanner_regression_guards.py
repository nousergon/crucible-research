"""Scanner regression guards.

Complements ``test_scanner_consumer_contract.py`` (which pins the
feature-store read path) by covering two adjacent regression classes
that the consumer-contract test does NOT catch:

1. **OHLCV-fallback path units**: ``data.fetchers.price_fetcher.
   compute_technical_indicators`` is the fallback that produced the
   ONLY real scanner candidates during the 6-month silent-zero-output
   regression. If it gets accidentally normalized to match the
   feature-store ratio convention, the scanner regresses to zero
   candidates everywhere.

2. **Full-gate count regression**: the consumer-contract test pins
   ``liquidity_pass`` count only. A regression in the volatility gate,
   ``compute_technical_score``, or the momentum/deep-value path
   selection could drop production picks from ~60 → ~5 without firing
   that test. The full-gate count test runs the WHOLE scanner pipeline
   against a realistic synthetic universe and asserts ≥ N picks pass
   all gates.

Filed per the gaps analysis on PR #237. See also the plan doc
`~/Development/alpha-engine-docs/private/feature-store-schema-audit-260525.md`.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── OHLCV-fallback path: raw-units pin ───────────────────────────────────────


def _synthetic_ohlcv(
    n: int = 100,
    seed: int = 0,
    volume_low: int = 1_000_000,
    volume_high: int = 10_000_000,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame that ``compute_technical_indicators``
    can consume. Volume is in raw shares (the production scale).
    """
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(daily_returns))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(volume_low, volume_high, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_ohlcv_fallback_returns_avg_volume_in_raw_shares():
    """``compute_technical_indicators`` must emit ``avg_volume_20d`` in
    raw shares — NOT normalized.

    This is the ONLY scanner-feeding code path that was producing real
    candidates during the 6-month silent-zero-output regression (the
    feature-store path was broken; the fallback was correct). If a
    future refactor "harmonizes" this to match the feature store's
    normalized convention, the scanner regresses to zero everywhere.
    """
    from data.fetchers.price_fetcher import compute_technical_indicators

    df = _synthetic_ohlcv(n=100, volume_low=1_000_000, volume_high=10_000_000)
    indicators = compute_technical_indicators(df)

    assert indicators is not None
    avg_vol = indicators.get("avg_volume_20d")
    assert avg_vol is not None, (
        "compute_technical_indicators returned avg_volume_20d=None despite "
        "100 rows of synthetic volume — fallback path is broken."
    )
    # Synthetic volume in [1M, 10M] — rolling-20d-mean must be inside that
    # band. The scanner's MIN_AVG_VOLUME gate is 500_000; we want at least
    # an order of magnitude above the gate so the sniff has real teeth.
    assert avg_vol >= 1_000_000, (
        f"OHLCV-fallback avg_volume_20d = {avg_vol:,.0f}; expected raw "
        f"shares (>= 1M). Likely normalized — check "
        f"data/fetchers/price_fetcher.py::compute_technical_indicators."
    )
    assert avg_vol < 100_000_000, (
        f"OHLCV-fallback avg_volume_20d = {avg_vol:,.0f}; "
        "implausibly large — check unit logic."
    )


def test_ohlcv_fallback_passes_scanner_liquidity_gate():
    """End-to-end fallback path: synthetic OHLCV → compute_technical_indicators
    → scanner liquidity gate. Must pass.

    Mirrors the production fallback chain when feature-store rows are
    missing for a ticker.
    """
    from data.scanner import run_quant_filter

    tickers = [f"T{i:03d}" for i in range(20)]
    price_data = {t: _synthetic_ohlcv(seed=i) for i, t in enumerate(tickers)}
    sector_map = dict.fromkeys(tickers, "Technology")

    # technical_scores empty → scanner falls through to OHLCV path.
    run_quant_filter(
        tickers=tickers,
        price_data=price_data,
        technical_scores={},
        sector_map=sector_map,
    )
    eval_log = run_quant_filter._last_eval_log
    liquidity_pass = sum(1 for r in eval_log if r.get("liquidity_pass") == 1)
    assert liquidity_pass >= 15, (
        f"Only {liquidity_pass} / 20 OHLCV-fallback tickers passed the "
        "scanner liquidity gate. The fallback path's avg_volume_20d "
        "units may have regressed."
    )


# ── Full-gate count regression: realistic fixture, all gates exercised ──────


def _bullish_indicators(
    *,
    ticker_seed: int,
    avg_volume_raw: float = 5_000_000.0,
    current_price: float = 50.0,
    atr_pct: float = 0.018,  # raw decimal (1.8%); scanner ×100 = 1.8
) -> dict:
    """Indicators dict matching the shape returned by
    ``_build_technical_scores_from_feature_store`` for a ticker that
    SHOULD pass all scanner gates when paired with the patched
    ``compute_technical_score`` below.

    Slight per-seed variation so the technical score isn't identical
    across tickers (deduping by tech_score in the scanner top-N would
    otherwise collapse the candidate count).

    Note: ``compute_technical_score`` is patched in the tests rather
    than driven from indicators, so the synthetic indicator values do
    NOT need to actually produce a high score under the production
    ``scoring.yaml`` weights — that's a separate concern tested
    elsewhere. This test is about the scanner GATE behavior, not the
    scorer.
    """
    rng = np.random.default_rng(ticker_seed)
    return {
        "rsi_14": 55.0 + rng.uniform(-5, 10),
        "macd_cross": 1.0,
        "macd_above_zero": True,
        "macd_line_last": 0.5,
        "signal_line_last": 0.3,
        "current_price": current_price + rng.uniform(-10, 30),
        "ma50": None,
        "ma200": None,
        "price_vs_ma50": 0.03 + rng.uniform(-0.01, 0.03),
        "price_vs_ma200": 0.08 + rng.uniform(-0.02, 0.05),
        "momentum_20d": 0.04 + rng.uniform(-0.01, 0.04),
        "momentum_5d": 0.015 + rng.uniform(-0.005, 0.015),
        "avg_volume_20d": avg_volume_raw,
        "atr_14_pct": atr_pct,
        "dist_from_52w_high": -0.04,
        "dist_from_52w_low": 0.25,
    }


def _scanner_thresholds() -> dict:
    """Resolve the live scanner thresholds from production config so
    the synthetic fixtures track config drift instead of hard-coding
    against sample values that don't match what CI sees.
    """
    from config import MAX_ATR_PCT, get_scanner_params
    sp = get_scanner_params()
    return {
        "tech_score_min": sp.get("tech_score_min", 60),
        "max_atr_pct": float(sp.get("max_atr_pct", MAX_ATR_PCT)),
    }


def test_full_scanner_pipeline_produces_realistic_pick_count():
    """End-to-end count regression. Feed a realistic 100-ticker
    bullish-shaped fixture through the scanner; assert most pass all
    gates (liquidity + volatility + MA200 floor + momentum path).

    Catches regressions in:
      - Volatility (ATR) gate — silent if ATR computation broken
      - MA200 floor logic
      - Momentum path selection
      - Top-N truncation logic
      - The liquidity gate plumbing as it flows through to candidate
        selection (separate from the contract test's eval-log assertion)

    Patches ``compute_technical_score`` to return a known-good value
    so this test is INDEPENDENT of ``scoring.yaml`` weight drift —
    that's covered by a separate composite-scoring test elsewhere.
    """
    from data.scanner import run_quant_filter

    thr = _scanner_thresholds()
    tickers = [f"T{i:03d}" for i in range(100)]
    sector_map = dict.fromkeys(tickers, "Technology")

    # ATR comfortably below max_atr_pct (which is 25.0 in current prod
    # config but could change). Pick decimal that ×100 == max/2.
    safe_atr = (thr["max_atr_pct"] / 2.0) / 100.0
    # Tech score comfortably above min (gate is `>=`).
    good_score = thr["tech_score_min"] + 15

    technical_scores: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        indicators = _bullish_indicators(ticker_seed=i, atr_pct=safe_atr)
        technical_scores[ticker] = {**indicators, "technical_score": good_score}

    with patch("data.scanner.compute_technical_score", return_value=good_score):
        candidates = run_quant_filter(
            tickers=tickers,
            price_data={},
            technical_scores=technical_scores,
            sector_map=sector_map,
        )

    eval_log = run_quant_filter._last_eval_log
    gate_eligible = sum(
        1 for r in eval_log
        if r.get("liquidity_pass") == 1 and r.get("volatility_pass") == 1
    )
    assert gate_eligible >= 80, (
        f"Only {gate_eligible} / 100 bullish-fixture tickers passed "
        "liquidity + volatility gates. Regression in one of the gate "
        "computations. Check data/scanner.py."
    )

    # Final post-rank-cutoff candidate list must be non-trivial. The
    # scanner's _momentum_top_n is 60 in current prod config; assert
    # ≥ 30 to leave headroom for config changes and catch a rank-cutoff
    # collapse.
    assert len(candidates) >= 30, (
        f"Scanner returned only {len(candidates)} candidates from a "
        "100-ticker bullish fixture passing all gates. The top-N rank "
        "cutoff or candidate-assembly logic regressed."
    )


def test_scanner_failure_mode_volatility_gate_regression_caught():
    """Negative-control: if every ticker is too volatile, the
    volatility gate SHOULD fire on substantially all of them. Confirms
    the count guard actually exercises the volatility gate (vs silently
    passing through).

    Reads the live ``max_atr_pct`` threshold from production config so
    the test stays correct under config changes.
    """
    from data.scanner import run_quant_filter

    thr = _scanner_thresholds()
    # atr_pct that comfortably exceeds the gate (×100 by scanner adapter).
    bad_atr = (thr["max_atr_pct"] + 10.0) / 100.0
    good_score = thr["tech_score_min"] + 15

    tickers = [f"T{i:03d}" for i in range(100)]
    sector_map = dict.fromkeys(tickers, "Technology")

    technical_scores: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        indicators = _bullish_indicators(ticker_seed=i, atr_pct=bad_atr)
        technical_scores[ticker] = {**indicators, "technical_score": good_score}

    with patch("data.scanner.compute_technical_score", return_value=good_score):
        run_quant_filter(
            tickers=tickers,
            price_data={},
            technical_scores=technical_scores,
            sector_map=sector_map,
        )
    eval_log = run_quant_filter._last_eval_log
    vol_fail_count = sum(
        1 for r in eval_log if r.get("volatility_pass") == 0
    )
    assert vol_fail_count >= 80, (
        f"Negative control: feeding 100 high-ATR (atr_pct={bad_atr:.4f}, "
        f"scanner-scale={bad_atr*100:.1f} > max={thr['max_atr_pct']}) "
        f"tickers should produce >= 80 volatility_pass=0 records; got "
        f"{vol_fail_count}. The volatility gate may be silently passing."
    )


def test_scanner_failure_mode_liquidity_gate_regression_caught():
    """Negative-control: if every ticker's raw volume is below the
    MIN_AVG_VOLUME floor, the scanner SHOULD record liquidity_pass=0.
    Confirms the count guard actually exercises the liquidity gate.
    """
    from config import MIN_AVG_VOLUME
    from data.scanner import run_quant_filter

    thr = _scanner_thresholds()
    # Volume well below the production MIN_AVG_VOLUME floor.
    bad_volume = MIN_AVG_VOLUME / 10.0
    good_score = thr["tech_score_min"] + 15

    tickers = [f"T{i:03d}" for i in range(20)]
    sector_map = dict.fromkeys(tickers, "Technology")

    technical_scores: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        indicators = _bullish_indicators(
            ticker_seed=i, avg_volume_raw=bad_volume,
        )
        technical_scores[ticker] = {**indicators, "technical_score": good_score}

    with patch("data.scanner.compute_technical_score", return_value=good_score):
        run_quant_filter(
            tickers=tickers,
            price_data={},
            technical_scores=technical_scores,
            sector_map=sector_map,
        )
    eval_log = run_quant_filter._last_eval_log
    liq_fail_count = sum(
        1 for r in eval_log if r.get("liquidity_pass") == 0
    )
    assert liq_fail_count == 20, (
        f"Negative control: feeding 20 low-volume tickers "
        f"(avg_volume_raw={bad_volume:,.0f} < MIN_AVG_VOLUME="
        f"{MIN_AVG_VOLUME:,.0f}) should produce 20 liquidity_pass=0 "
        f"records; got {liq_fail_count}. Gate broken."
    )


# ── fetch_data short-history None-guard regression (Saturday SF 2026-06-27) ──
#
# The Saturday pipeline FAILED on 2026-06-27 in research_graph.fetch_data with
# ``AttributeError: 'NoneType' object has no attribute 'get'`` at
# scoring/technical.py compute_technical_score(indicators=None).
#
# Root cause — a threshold mismatch between two contracts:
#   * compute_technical_indicators(df) returns None when ``df.empty or
#     len(df) < 30`` (documented "Returns None if insufficient data").
#   * fetch_data's OHLCV scoring loop admitted any ticker with
#     ``df is not None and len(df) >= 20`` and passed the result straight
#     into compute_technical_score WITHOUT checking for the None sentinel.
# So any ticker whose ArcticDB history has 20–29 rows produced
# ``indicators=None`` → crash. The two other production callers
# (data/scanner.py, local/time_scanner.py) already skip on None; fetch_data
# was the lone unguarded call site.


def test_compute_technical_indicators_returns_none_for_20_to_29_rows():
    """Pin the trigger window: compute_technical_indicators returns None for a
    df with 20–29 rows (admitted by fetch_data's ``len(df) >= 20`` guard but
    below the indicator's own ``len(df) < 30`` floor). If this contract ever
    changes, the fetch_data guard rationale must be revisited."""
    from data.fetchers.price_fetcher import compute_technical_indicators

    for n in (20, 25, 29):
        df = _synthetic_ohlcv(n=n)
        assert len(df) == n
        assert compute_technical_indicators(df) is None, (
            f"compute_technical_indicators({n}-row df) must return the "
            "documented None sentinel (insufficient data)."
        )
    # And it must produce a real dict once there are >= 30 rows.
    assert compute_technical_indicators(_synthetic_ohlcv(n=30)) is not None


def test_fetch_data_skips_short_history_tickers_without_crashing():
    """Reproduce fetch_data's OHLCV scoring loop with the REAL
    compute_technical_indicators + compute_technical_score over a price_data
    map that mixes a 25-row (None-yielding) ticker and a 100-row ticker.

    The short-history ticker must be SKIPPED (not technically scored) and the
    loop must not raise — exactly the production behaviour fixed in
    graph/research_graph.py. A regression here means the pre-2026-06-27 crash
    is back."""
    from data.fetchers.price_fetcher import compute_technical_indicators
    from scoring.technical import compute_technical_score

    price_data = {
        "SHORTY": _synthetic_ohlcv(n=25, seed=1),   # < 30 rows → indicators None
        "GOODY": _synthetic_ohlcv(n=100, seed=2),   # full history → scored
    }
    sector_map = {"SHORTY": "Technology", "GOODY": "Technology"}
    technical_scores: dict = {}

    # Mirror graph/research_graph.py::fetch_data exactly, including the guard.
    for ticker, df in price_data.items():
        if ticker in technical_scores:
            continue
        if df is not None and len(df) >= 20:
            indicators = compute_technical_indicators(df)
            if indicators is None:
                continue
            ts = compute_technical_score(indicators, sector=sector_map.get(ticker))
            technical_scores[ticker] = {**indicators, "technical_score": ts}

    assert "SHORTY" not in technical_scores, (
        "20–29-row ticker must be skipped, not technically scored."
    )
    assert "GOODY" in technical_scores
    assert isinstance(technical_scores["GOODY"]["technical_score"], float)


def test_fetch_data_call_site_guards_none_indicators():
    """Source anchor: the fetch_data OHLCV scoring loop must guard the None
    return of compute_technical_indicators before calling
    compute_technical_score. Pins the literal fix so a future refactor cannot
    silently reintroduce the 2026-06-27 crash."""
    import inspect

    import graph.research_graph as rg

    src = inspect.getsource(rg.fetch_data)
    idx_compute = src.find("compute_technical_indicators(df)")
    assert idx_compute != -1, "expected compute_technical_indicators(df) call in fetch_data"
    # Between computing indicators and scoring them, there must be a None guard.
    tail = src[idx_compute:]
    idx_guard = tail.find("if indicators is None")
    idx_score = tail.find("compute_technical_score(indicators")
    assert idx_guard != -1 and idx_score != -1 and idx_guard < idx_score, (
        "fetch_data must skip (continue) when compute_technical_indicators "
        "returns None BEFORE passing indicators to compute_technical_score — "
        "otherwise a 20–29-row-history ticker crashes the research pipeline."
    )
