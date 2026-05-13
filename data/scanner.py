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

from typing import Optional

import pandas as pd

from config import (
    MIN_AVG_VOLUME,
    MIN_PRICE,
    DEEP_VALUE_PATH_ENABLED,
    DEEP_VALUE_MAX_RSI,
    DEEP_VALUE_MAX_CANDIDATES,
    MAX_ATR_PCT,
    DEEP_VALUE_MAX_ATR_PCT,
    MAX_DEBT_TO_EQUITY,
    MIN_CURRENT_RATIO,
    BALANCE_SHEET_EXEMPT_SECTORS,
    QUALITY_FLOOR_ENABLED,
    QUALITY_MIN_PROFIT_MARGIN,
    QUALITY_MIN_ROE,
    QUALITY_REQUIRE_BOTH,
    QUALITY_EXEMPT_SECTORS,
    REGIME_ATR_TILT_ENABLED,
    REGIME_ATR_BULL_DROP,
    REGIME_ATR_BEAR_DROP,
    REGIME_ATR_QUARTILE_PCT,
    REGIME_ATR_MIN_SECTOR_SIZE,
    get_scanner_params,
)
from data.fetchers.price_fetcher import (
    fetch_sp500_sp400_tickers,
    fetch_price_data,
    compute_technical_indicators,
)
from scoring.technical import compute_technical_score

import logging

logger = logging.getLogger(__name__)


def get_scanner_universe(exclude_tickers: Optional[list[str]] = None) -> list[str]:
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


def confirm_deep_value_with_analyst(
    candidates: list[dict],
    analyst_data: dict[str, dict],
    min_consensus: str = "Buy",
) -> list[dict]:
    """
    Stage 2 (partial): For deep_value_pending candidates, confirm analyst conviction.
    Removes candidates that don't meet the analyst threshold; promotes path field.
    """
    _CONSENSUS_RANK = {
        "Strong Buy": 5, "Buy": 4, "Hold": 3, "Underperform": 2, "Sell": 1,
    }
    min_rank = _CONSENSUS_RANK.get(min_consensus, 4)

    result = []
    for c in candidates:
        if c.get("path") != "deep_value_pending":
            result.append(c)
            continue

        adata = analyst_data.get(c["ticker"], {})
        consensus = adata.get("consensus_rating", "Hold")
        rank = _CONSENSUS_RANK.get(consensus, 0)

        if rank >= min_rank:
            c = {**c, "path": "deep_value"}
            result.append(c)
        # else: drop this candidate — analyst conviction too low

    return result


def apply_balance_sheet_filter(
    candidates: list[dict],
    sector_map: dict[str, str] | None = None,
    max_debt_to_equity: float = MAX_DEBT_TO_EQUITY,
    min_current_ratio: float = MIN_CURRENT_RATIO,
    exempt_sectors: list[str] | None = None,
) -> list[dict]:
    """
    Task 2B: Reject overleveraged or illiquid-balance-sheet stocks.

    Uses yfinance Ticker.info for debt_to_equity and current_ratio.
    Only fetches for post-quant-filter candidates (~60 stocks), not full universe.
    Financials and Real Estate are exempt from D/E check (leverage is structural).
    """
    import yfinance as yf

    _sector_map = sector_map or {}
    _exempt = set(exempt_sectors if exempt_sectors is not None else BALANCE_SHEET_EXEMPT_SECTORS)

    result = []
    rejected = 0
    fetch_failures = 0
    for c in candidates:
        ticker = c["ticker"]
        sector = _sector_map.get(ticker, c.get("sector", "Unknown"))

        # Exempt sectors skip balance sheet check
        if sector in _exempt:
            result.append(c)
            continue

        try:
            info = yf.Ticker(ticker).info
            de_ratio = info.get("debtToEquity")  # yfinance returns as %, e.g. 150 = 1.5x
            current_ratio = info.get("currentRatio")

            # debtToEquity from yfinance is in % (e.g., 150 means 1.5x)
            if de_ratio is not None:
                de_ratio = de_ratio / 100.0

            if de_ratio is not None and de_ratio > max_debt_to_equity:
                rejected += 1
                logger.debug("[balance_sheet] REJECT %s: D/E=%.1f > %.1f", ticker, de_ratio, max_debt_to_equity)
                continue

            if current_ratio is not None and current_ratio < min_current_ratio:
                rejected += 1
                logger.debug("[balance_sheet] REJECT %s: current_ratio=%.2f < %.2f", ticker, current_ratio, min_current_ratio)
                continue

        except Exception as e:
            # Fail-closed: reject candidate when balance sheet data unavailable (M7 fix)
            logger.warning("[balance_sheet] REJECT %s (fetch error, fail-closed): %s", ticker, e)
            fetch_failures += 1
            continue

        result.append(c)

    if rejected:
        logger.info("[balance_sheet] rejected %d candidates (D/E>%.1f or CR<%.2f)", rejected, max_debt_to_equity, min_current_ratio)
    if fetch_failures:
        logger.warning("[balance_sheet] %d candidates rejected due to fetch failures", fetch_failures)
        if len(candidates) > 0 and fetch_failures > len(candidates) * 0.5:
            logger.warning("[balance_sheet] >50%% fetch failures (%d/%d) — yfinance data source may be down",
                           fetch_failures, len(candidates))
    return result


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
    from datetime import datetime, date

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


def apply_quality_filter(
    candidates: list[dict],
    sector_map: dict[str, str] | None = None,
    min_profit_margin: float = QUALITY_MIN_PROFIT_MARGIN,
    min_roe: float = QUALITY_MIN_ROE,
    require_both: bool = QUALITY_REQUIRE_BOTH,
    exempt_sectors: list[str] | None = None,
) -> list[dict]:
    """Quality floor (Piotroski-lite). Reject names with no profitability signal.

    Default lenient mode (require_both=False): a candidate passes if EITHER
    profitMargins > min_profit_margin OR returnOnEquity > min_roe. Captures
    pre-profit growth names with positive ROE on a small base AND established
    compounders with positive margins. Strict mode (require_both=True) demands
    both — for use in BEAR regimes or when defensiveness matters more.

    Sector exemptions skip the gate where the metrics do not apply
    (Financials, REITs, Utilities by default).

    Uses yfinance Ticker.info — same data source as apply_balance_sheet_filter
    so this function can run in the same fetch pass without doubling API load.
    Fail-closed on fetch error (matches balance-sheet pattern, M7 fix).

    Composes with apply_balance_sheet_filter — call this AFTER balance sheet
    so the candidate set is already smaller (~60 ticker yfinance calls).
    """
    import yfinance as yf

    _sector_map = sector_map or {}
    _exempt = set(exempt_sectors if exempt_sectors is not None else QUALITY_EXEMPT_SECTORS)

    result = []
    rejected = 0
    fetch_failures = 0
    for c in candidates:
        ticker = c["ticker"]
        sector = _sector_map.get(ticker, c.get("sector", "Unknown"))

        if sector in _exempt:
            result.append(c)
            continue

        try:
            info = yf.Ticker(ticker).info
            profit_margin = info.get("profitMargins")
            roe = info.get("returnOnEquity")

            margin_ok = profit_margin is not None and profit_margin > min_profit_margin
            roe_ok = roe is not None and roe > min_roe

            if require_both:
                passes = margin_ok and roe_ok
            else:
                passes = margin_ok or roe_ok

            if not passes:
                rejected += 1
                logger.debug(
                    "[quality_floor] REJECT %s: profit_margin=%s roe=%s "
                    "(require_both=%s, min_pm=%.3f, min_roe=%.3f)",
                    ticker, profit_margin, roe, require_both, min_profit_margin, min_roe,
                )
                continue

        except Exception as e:
            logger.warning("[quality_floor] REJECT %s (fetch error, fail-closed): %s", ticker, e)
            fetch_failures += 1
            continue

        result.append(c)

    if rejected:
        logger.info(
            "[quality_floor] rejected %d candidates (no profitability signal; "
            "min_pm=%.3f, min_roe=%.3f, require_both=%s)",
            rejected, min_profit_margin, min_roe, require_both,
        )
    if fetch_failures:
        logger.warning("[quality_floor] %d candidates rejected due to fetch failures", fetch_failures)
    return result


def apply_regime_atr_tilt(
    candidates: list[dict],
    market_regime: str | None,
    sector_map: dict[str, str] | None = None,
    bull_drop: str = REGIME_ATR_BULL_DROP,
    bear_drop: str = REGIME_ATR_BEAR_DROP,
    quartile_pct: int = REGIME_ATR_QUARTILE_PCT,
    min_sector_size: int = REGIME_ATR_MIN_SECTOR_SIZE,
) -> list[dict]:
    """Regime-conditional ATR tilt within sector.

    In BULL regime: drop bottom-quartile-by-ATR within each sector — those
    names are not contributing risk in a regime that rewards risk. In BEAR
    invert: drop top-quartile to defensive-tilt the population. NEUTRAL
    applies no tilt.

    Sector grouping ensures within-sector composition is preserved (an
    OW Tech sector does not lose all its names just because its average
    ATR is high). Sectors with fewer than min_sector_size candidates skip
    the tilt to avoid degenerate quartile cuts on tiny groups.

    Each candidate must have an ``atr_pct`` field (computed upstream by
    run_quant_filter). Candidates with missing/None ATR are kept as-is
    (cannot rank without the value; fail-open).

    Composes with sector OW/UW (drives WHICH sectors get exposure) and
    max_atr_pct ceiling (drives the absolute upper bound on vol). This
    is the within-sector relative tilt.
    """
    if not candidates:
        return candidates

    regime = (market_regime or "").strip().lower()
    if regime not in ("bull", "bear"):
        return candidates  # neutral / unknown → no tilt

    drop_side = bull_drop if regime == "bull" else bear_drop
    if drop_side not in ("bottom", "top"):
        logger.warning("[regime_atr_tilt] unknown drop_side=%s; no-op", drop_side)
        return candidates

    _sector_map = sector_map or {}

    # Group candidates by sector
    by_sector: dict[str, list[dict]] = {}
    for c in candidates:
        sector = _sector_map.get(c["ticker"], c.get("sector", "Unknown"))
        by_sector.setdefault(sector, []).append(c)

    kept: list[dict] = []
    dropped_total = 0
    for sector, sector_cands in by_sector.items():
        if len(sector_cands) < min_sector_size:
            kept.extend(sector_cands)
            continue

        # Partition by has-ATR vs missing-ATR; missing kept as-is
        with_atr = [c for c in sector_cands if c.get("atr_pct") is not None]
        without_atr = [c for c in sector_cands if c.get("atr_pct") is None]

        if len(with_atr) < min_sector_size:
            kept.extend(sector_cands)
            continue

        with_atr.sort(key=lambda c: c["atr_pct"])
        n_drop = max(1, int(round(len(with_atr) * (quartile_pct / 100.0))))

        if drop_side == "bottom":
            sector_kept = with_atr[n_drop:]   # drop bottom-N
        else:
            sector_kept = with_atr[:-n_drop]  # drop top-N

        dropped_total += len(with_atr) - len(sector_kept)
        kept.extend(sector_kept)
        kept.extend(without_atr)

    if dropped_total:
        logger.info(
            "[regime_atr_tilt] regime=%s drop_side=%s pct=%d → dropped %d/%d "
            "candidates across %d sectors",
            regime, drop_side, quartile_pct, dropped_total, len(candidates), len(by_sector),
        )

    return kept

