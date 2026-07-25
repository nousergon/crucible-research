"""
Market scanner — Stage 1 quantitative filter.

Scans S&P 500 + S&P 400 (~900 stocks) and reduces to ~60 candidates
via momentum and deep-value paths (§6.1, §6.3).

No LLM is used in this module.

The scanner now drives all population selection — no hardcoded universe.
All ~900 stocks are eligible; the exclude_tickers parameter allows the
caller to exclude specific tickers if needed (e.g., recently removed stocks).
"""

from __future__ import annotations

import logging

import pandas as pd

from config import (
    DEEP_VALUE_MAX_ATR_PCT,
    DEEP_VALUE_MAX_CANDIDATES,
    DEEP_VALUE_MAX_RSI,
    DEEP_VALUE_PATH_ENABLED,
    MAX_ATR_PCT,
    MIN_AVG_VOLUME,
    MIN_PRICE,
    get_scanner_params,
)
from data.fetchers.price_fetcher import (
    compute_technical_indicators,
    fetch_sp500_sp400_tickers,
)
from scoring.technical import compute_technical_score

logger = logging.getLogger(__name__)


def get_scanner_universe(exclude_tickers: list[str] | None = None) -> list[str]:
    """
    Return the full scanner candidate universe (S&P 500 + S&P 400).

    All ~900 stocks are eligible since there is no hardcoded static universe.
    Optional exclude_tickers allows caller to filter specific tickers.
    """
    all_tickers = fetch_sp500_sp400_tickers()
    exclude = set(exclude_tickers or [])
    return [t for t in all_tickers if t not in exclude]


def _compute_atr_pct(df: pd.DataFrame, period: int | None = None) -> float | None:
    """Compute Average True Range as % of price over the last `period` days."""
    if period is None:
        from config import get_research_params
        period = get_research_params()["atr_period"]
    if df is None or df.empty or len(df) < period + 1:
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    # True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.tail(period).mean()
    current_price = float(close.iloc[-1])
    if current_price <= 0:
        return None
    return round(float(atr) / current_price * 100, 2)


def run_quant_filter(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    technical_scores: dict[str, dict],
    market_regime: str = "neutral",
    sector_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Stage 1: Apply quantitative filter to reduce ~900 → ~50.

    Filters applied:
      - Liquidity floor (min volume, min price)
      - Volatility screen: ATR% > MAX_ATR_PCT rejected (relaxed for deep value)
      - Momentum or deep value path selection

    Returns list of candidate dicts with fields:
      ticker, path (momentum|deep_value), tech_score, rsi_14,
      price, avg_volume_20d, price_vs_ma200, atr_pct

    Side effect: populates self._scanner_evaluations (list[dict]) with pass/fail
    data for ALL tickers, used for evaluation logging.
    """
    _sector_map = sector_map or {}
    momentum_candidates = []
    deep_value_candidates = []
    # Evaluation log: one entry per ticker, regardless of pass/fail
    _eval_log: list[dict] = []

    # Phase 4a: S3-configurable scanner thresholds (auto-tuned by backtester)
    _sp = get_scanner_params()
    _tech_score_min = _sp.get("tech_score_min", 60)
    _max_atr_pct = _sp.get("max_atr_pct", MAX_ATR_PCT)
    _min_avg_vol = _sp.get("min_avg_volume", MIN_AVG_VOLUME)
    _min_price = _sp.get("min_price", MIN_PRICE)
    _ma200_floor = _sp["momentum_ma200_floor_pct"]
    _momentum_top_n = _sp["momentum_top_n"]
    _min_combined = _sp["min_combined_candidates"]

    for ticker in tickers:
        tech = technical_scores.get(ticker)
        df = price_data.get(ticker)

        # Must have either feature store data or raw price data
        if tech is None and (df is None or df.empty):
            _eval_log.append({
                "ticker": ticker, "sector": _sector_map.get(ticker),
                "quant_filter_pass": 0, "liquidity_pass": 0,
                "filter_fail_reason": "no_data",
            })
            continue
        if tech is None:
            tech = compute_technical_indicators(df)
        if tech is None:
            _eval_log.append({
                "ticker": ticker, "sector": _sector_map.get(ticker),
                "quant_filter_pass": 0, "liquidity_pass": 0,
                "filter_fail_reason": "no_tech_indicators",
            })
            continue

        price = tech.get("current_price", 0)
        avg_vol = tech.get("avg_volume_20d", 0) or 0
        rsi = tech.get("rsi_14", 50)
        price_vs_ma200 = tech.get("price_vs_ma200")

        # Base eval record for this ticker
        eval_rec = {
            "ticker": ticker,
            "sector": _sector_map.get(ticker),
            "tech_score": None,  # computed below if liquidity passes
            "rsi_14": rsi,
            "current_price": price,
            "avg_volume_20d": avg_vol,
            "price_vs_ma200": price_vs_ma200,
            "liquidity_pass": 1,
            "volatility_pass": 1,
            "quant_filter_pass": 0,
        }

        # Basic liquidity + price floor (both paths)
        if avg_vol < _min_avg_vol or (_min_price > 0 and price < _min_price):
            eval_rec["liquidity_pass"] = 0
            eval_rec["filter_fail_reason"] = "liquidity"
            _eval_log.append(eval_rec)
            continue

        # Volatility screen: use feature store ATR if available, else compute from df
        atr_pct = tech.get("atr_14_pct")
        if atr_pct is not None:
            atr_pct = round(atr_pct * 100, 2)  # feature store stores as decimal, scanner expects %
        elif df is not None and not df.empty:
            atr_pct = _compute_atr_pct(df)

        tech_score = compute_technical_score(
            tech,
            market_regime=market_regime,
            sector=_sector_map.get(ticker),
        )
        eval_rec["tech_score"] = tech_score
        eval_rec["atr_pct"] = atr_pct

        candidate = {
            "ticker": ticker,
            "tech_score": tech_score,
            "rsi_14": rsi,
            "current_price": price,
            "avg_volume_20d": avg_vol,
            "price_vs_ma200": price_vs_ma200,
            "atr_pct": atr_pct,
        }

        # ── Momentum path ──────────────────────────────────────────────────
        # Require strong technicals and no severe downtrend.
        # The MA200 floor is configurable (default admits bounce plays near
        # 200-day support; stocks well below MA200 are in sustained downtrends).
        if (
            tech_score >= _tech_score_min
            and (price_vs_ma200 is None or price_vs_ma200 > _ma200_floor)
        ):
            # Volatility gate for momentum path
            if atr_pct is not None and atr_pct > _max_atr_pct:
                eval_rec["volatility_pass"] = 0
                eval_rec["filter_fail_reason"] = "volatility_momentum"
                _eval_log.append(eval_rec)
                continue
            candidate["path"] = "momentum"
            eval_rec["scan_path"] = "momentum"
            momentum_candidates.append(candidate)

        # ── Deep value path (config-gated) ─────────────────────────────────
        elif DEEP_VALUE_PATH_ENABLED and rsi < DEEP_VALUE_MAX_RSI:
            # Relaxed volatility gate for deep value
            if atr_pct is not None and atr_pct > DEEP_VALUE_MAX_ATR_PCT:
                eval_rec["volatility_pass"] = 0
                eval_rec["filter_fail_reason"] = "volatility_deep_value"
                _eval_log.append(eval_rec)
                continue
            # RSI oversold + below 200 MA → potential bottoming
            # Analyst conviction check happens in Stage 2 (after FMP fetch)
            candidate["path"] = "deep_value_pending"  # confirmed after analyst data
            eval_rec["scan_path"] = "deep_value_pending"
            deep_value_candidates.append(candidate)

        else:
            # Didn't qualify for either path
            eval_rec["filter_fail_reason"] = "below_thresholds"
            _eval_log.append(eval_rec)
            continue

        # If we reach here, the ticker is a candidate (momentum or deep value)
        # — don't mark quant_filter_pass yet (rank cutoff applied below)
        _eval_log.append(eval_rec)

    # Sort momentum candidates by tech_score descending; take top N (configurable)
    momentum_candidates.sort(key=lambda x: x["tech_score"], reverse=True)
    momentum_top = momentum_candidates[:_momentum_top_n]

    # Deep value: cap at DEEP_VALUE_MAX_CANDIDATES (default 10)
    deep_value_candidates.sort(key=lambda x: x["rsi_14"])  # most oversold first
    deep_value_top = deep_value_candidates[:DEEP_VALUE_MAX_CANDIDATES]

    combined = momentum_top + deep_value_top

    # Fallback: if fewer than _min_combined candidates passed, fill from all scored tickers
    # sorted by tech_score so the scanner always has at least that many names to work with.
    if len(combined) < _min_combined:
        combined_symbols = {c["ticker"] for c in combined}
        all_scored = []
        for ticker in tickers:
            tech = technical_scores.get(ticker)
            if tech is None:
                df = price_data.get(ticker)
                if df is None or df.empty:
                    continue
                tech = compute_technical_indicators(df)
            if tech is None:
                continue
            price = tech.get("current_price", 0)
            avg_vol = tech.get("avg_volume_20d", 0) or 0
            if avg_vol < MIN_AVG_VOLUME or (MIN_PRICE > 0 and price < MIN_PRICE):
                continue
            if ticker not in combined_symbols:
                all_scored.append({
                    "ticker": ticker,
                    "path": "momentum",
                    "tech_score": compute_technical_score(
                        tech,
                        market_regime=market_regime,
                        sector=_sector_map.get(ticker),
                    ),
                    "rsi_14": tech.get("rsi_14", 50),
                    "current_price": price,
                    "avg_volume_20d": avg_vol,
                    "price_vs_ma200": tech.get("price_vs_ma200"),
                })
        all_scored.sort(key=lambda x: x["tech_score"], reverse=True)
        for c in all_scored:
            if len(combined) >= _min_combined:
                break
            combined.append(c)

    # Deduplicate (a ticker can't be in both paths)
    seen: set[str] = set()
    result = []
    for c in combined:
        if c["ticker"] not in seen:
            seen.add(c["ticker"])
            result.append(c)

    # Mark passing tickers in the eval log
    passing_tickers = {c["ticker"] for c in result}
    for rec in _eval_log:
        if rec["ticker"] in passing_tickers:
            rec["quant_filter_pass"] = 1
        elif not rec.get("filter_fail_reason"):
            rec["filter_fail_reason"] = "rank_cutoff"

    # Attach eval log to module-level for the graph to pick up
    run_quant_filter._last_eval_log = _eval_log

    return result


# confirm_deep_value_with_analyst was removed config#1821 Option B
# (2026-07-08): it was unused (no call sites anywhere in the codebase) and
# keyed off ``analyst_data[...]["consensus_rating"]``, which was sourced
# from FMP's grades-consensus endpoint — 402'd for every ticker on the
# current plan and removed from fetch_analyst_consensus's returned shape.
# Reintroducing a deep-value analyst-confirmation stage would need a new
# data source; wiring "deep_value_pending" candidates to that stage is
# tracked separately (it was never wired to this function to begin with).


def evaluate_candidate_rotation(
    scanner_scores: dict[str, dict],
    active_candidates: list[dict],
    rotation_tiers: list[dict],
    weak_pick_score_threshold: float,
    weak_pick_consecutive_runs: int,
    emergency_rotation_new_score: float,
    run_date: str,
) -> tuple[list[dict], list[dict]]:
    """
    Stage 5: Determine whether to rotate any of the 3 active candidates.

    Args:
        scanner_scores: {ticker: {score, path, ...}} for top-10 scanner candidates
        active_candidates: current 3 dicts with {symbol, entry_date, slot, consecutive_low_runs}
        rotation_tiers: list of {max_tenure_days, min_score_diff}
        ...

    Returns:
        (new_active_candidates, rotation_events)
        rotation_events: list of {out_ticker, in_ticker, reason}
    """
    from datetime import date, datetime

    _sp = get_scanner_params()
    _default_delta = _sp["rotation_default_required_delta"]
    _all_weak_score = _sp["rotation_all_weak_score"]
    _weak_pick_min_tenure = _sp["rotation_weak_pick_min_tenure_days"]
    _weak_pick_min_challenger = _sp["rotation_weak_pick_min_challenger_score"]

    def tenure_days(entry_date_str: str) -> int:
        try:
            ed = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            return (date.fromisoformat(run_date) - ed).days
        except Exception:
            return 0

    def required_delta(tenure: int) -> float:
        for tier in sorted(rotation_tiers, key=lambda t: t["max_tenure_days"]):
            if tenure <= tier["max_tenure_days"]:
                return float(tier["min_score_diff"])
        return _default_delta

    # Build sorted scanner candidate list by score
    scanner_sorted = sorted(
        [{"ticker": t, **v} for t, v in scanner_scores.items()],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    active = list(active_candidates)  # copy
    rotations = []
    rotations_this_run = 0

    # Fill any empty slots first — no score delta required.
    # Guarantees all 3 slots are populated as long as scanner has candidates.
    if len(active) < 3 and scanner_sorted:
        active_symbols = {c["symbol"] for c in active}
        for challenger in scanner_sorted:
            if len(active) >= 3:
                break
            if challenger["ticker"] not in active_symbols:
                slot = len(active) + 1
                active.append({
                    "symbol": challenger["ticker"],
                    "entry_date": run_date,
                    "slot": slot,
                    "score": challenger.get("score", 0),
                    "consecutive_low_runs": 0,
                })
                active_symbols.add(challenger["ticker"])
                rotations.append({
                    "out_ticker": None,
                    "in_ticker": challenger["ticker"],
                    "reason": "fill_empty_slot",
                })

    for challenger in scanner_sorted:
        if rotations_this_run >= 1:
            break  # max 1 rotation per run

        c_score = challenger.get("score", 0)
        c_ticker = challenger["ticker"]

        # Find the weakest active candidate
        weakest = min(active, key=lambda x: x.get("score", 0))
        w_score = weakest.get("score", 0)
        tenure = tenure_days(weakest.get("entry_date", run_date))
        delta_needed = required_delta(tenure)
        consec_low = weakest.get("consecutive_low_runs", 0)

        # Emergency override: all 3 weak + strong challenger
        all_weak = all(a.get("score", 100) < _all_weak_score for a in active)
        if all_weak and c_score >= emergency_rotation_new_score:
            rotations.append({
                "out_ticker": weakest["symbol"],
                "in_ticker": c_ticker,
                "reason": "emergency_rotation",
            })
            active = [a for a in active if a["symbol"] != weakest["symbol"]]
            active.append({
                "symbol": c_ticker,
                "entry_date": run_date,
                "slot": weakest["slot"],
                "score": c_score,
                "consecutive_low_runs": 0,
            })
            rotations_this_run += 1
            continue

        # Weak pick override: long-held low scorer loses tenure protection
        if (
            tenure >= _weak_pick_min_tenure
            and consec_low >= weak_pick_consecutive_runs
            and w_score < weak_pick_score_threshold
            and c_score >= _weak_pick_min_challenger
        ):
            rotations.append({
                "out_ticker": weakest["symbol"],
                "in_ticker": c_ticker,
                "reason": "weak_pick_override",
            })
            active = [a for a in active if a["symbol"] != weakest["symbol"]]
            active.append({
                "symbol": c_ticker,
                "entry_date": run_date,
                "slot": weakest["slot"],
                "score": c_score,
                "consecutive_low_runs": 0,
            })
            rotations_this_run += 1
            continue

        # Standard tiered threshold
        if c_score - w_score >= delta_needed:
            rotations.append({
                "out_ticker": weakest["symbol"],
                "in_ticker": c_ticker,
                "reason": f"score_delta_{c_score - w_score:.1f}_tenure_{tenure}d",
            })
            active = [a for a in active if a["symbol"] != weakest["symbol"]]
            active.append({
                "symbol": c_ticker,
                "entry_date": run_date,
                "slot": weakest["slot"],
                "score": c_score,
                "consecutive_low_runs": 0,
            })
            rotations_this_run += 1

    return active, rotations


