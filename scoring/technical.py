"""
Technical scoring engine — deterministic, no LLM.

Computes a 0–100 technical attractiveness score from price-derived indicators
(RSI, MACD, price vs 50/200 MA, 20d momentum). All weights, thresholds, and
anchors are read from config/scoring.yaml via config.TECHNICAL_CFG so the
formula can be tuned without code changes.

See §5.1 for full scoring methodology.
"""

from __future__ import annotations

import logging

from config import TECHNICAL_CFG

logger = logging.getLogger(__name__)

_REQUIRED_WEIGHT_KEYS = ("rsi", "macd", "ma50", "ma200", "momentum")
_WEIGHT_SUM_TOLERANCE = 1e-3
_warned_overrides: set[str] = set()


def _resolve_team_id(team_id: str | None, sector: str | None) -> str | None:
    """Return canonical team_id, resolving from a GICS sector name if needed.

    Caller may pass team_id directly (canonical: technology / healthcare /
    financials / industrials / consumer / defensives) or pass `sector` (GICS
    name) and rely on the team_config.SECTOR_TEAM_MAP for translation.
    Deferred import avoids a scoring → agents layering dependency at import
    time; the mapping is only loaded when an override-eligible lookup
    happens.
    """
    if team_id:
        return team_id
    if not sector:
        return None
    try:
        from agents.sector_teams.team_config import SECTOR_TEAM_MAP
    except ImportError:
        return None
    return SECTOR_TEAM_MAP.get(sector)


def _resolve_composite_weights(team_id: str | None) -> dict:
    """Return composite_weights for the given team_id, falling back to global.

    Sector-level overrides live under `technical.composite_weights_per_sector`
    in scoring.yaml. Each entry MUST contain all 5 sub-keys (rsi/macd/ma50/
    ma200/momentum) and sum to ~1.0. Malformed entries are logged once per
    team_id and ignored (fallback to global weights).
    """
    base = TECHNICAL_CFG.get("composite_weights", {})
    if not team_id:
        return base
    overrides = TECHNICAL_CFG.get("composite_weights_per_sector") or {}
    override = overrides.get(team_id)
    if not override:
        return base
    if not all(k in override for k in _REQUIRED_WEIGHT_KEYS):
        if team_id not in _warned_overrides:
            missing = sorted(set(_REQUIRED_WEIGHT_KEYS) - set(override))
            logger.warning(
                "composite_weights_per_sector[%s] missing keys %s; using global weights",
                team_id, missing,
            )
            _warned_overrides.add(team_id)
        return base
    total = sum(float(override[k]) for k in _REQUIRED_WEIGHT_KEYS)
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        if team_id not in _warned_overrides:
            logger.warning(
                "composite_weights_per_sector[%s] sums to %.4f (expected 1.0); using global weights",
                team_id, total,
            )
            _warned_overrides.add(team_id)
        return base
    return override


# ── Per-signal scoring ────────────────────────────────────────────────────────

def _score_rsi(rsi: float, market_regime: str = "neutral") -> float:
    """
    Score RSI (0–100) with regime-aware overbought/oversold thresholds.

    Bull regime (VIX<15, uptrend): raise overbought threshold.
    Bear/caution regime: raise oversold threshold and cap the oversold-zone
      score (oversold can signal further decline, not necessarily a buy).
    Neutral: standard thresholds.

    Thresholds and max-oversold scores come from scoring.yaml `technical.rsi`.
    """
    rsi_cfg = TECHNICAL_CFG.get("rsi", {})
    # bear and caution share the same parameters
    regime_key = "bear" if market_regime in ("bear", "caution") else market_regime
    regime_cfg = rsi_cfg.get(regime_key, rsi_cfg.get("neutral", {}))

    overbought = regime_cfg["overbought"]
    oversold = regime_cfg["oversold"]
    max_oversold_score = regime_cfg["max_oversold_score"]

    if rsi >= overbought:
        return 0.0
    if rsi <= oversold:
        return max_oversold_score
    # Linear interpolation between oversold (max_oversold_score) and overbought (0)
    return max_oversold_score * (overbought - rsi) / (overbought - oversold)


def _score_macd(macd_cross: float, macd_above_zero: bool) -> float:
    """Score MACD signal cross. Values from scoring.yaml `technical.macd`."""
    macd_cfg = TECHNICAL_CFG.get("macd", {})
    if macd_cross == 1.0:  # bullish cross
        return macd_cfg["bullish_cross_above_zero"] if macd_above_zero else macd_cfg["bullish_cross_below_zero"]
    if macd_cross == -1.0:  # bearish cross
        return macd_cfg["bearish_cross_above_zero"] if macd_above_zero else macd_cfg["bearish_cross_below_zero"]
    return macd_cfg["no_cross_above_zero"] if macd_above_zero else macd_cfg["no_cross_below_zero"]


def _score_price_vs_ma(pct_diff: float | None) -> float:
    """
    Score price relative to a moving average.

    Piecewise linear interpolation with anchors and scales from
    scoring.yaml `technical.price_vs_ma`:
      >= upper_anchor_pct (+5%)  → scaled up to upper_max_score
      0 to upper_anchor_pct      → mid_score + pct * positive_scale
      lower_anchor_pct to 0      → mid_score + pct * negative_scale
      < lower_anchor_pct (-5%)   → lower_anchor_score, decaying to lower_floor_score
    """
    if pct_diff is None:
        cfg = TECHNICAL_CFG.get("price_vs_ma", {})
        return cfg.get("mid_score", 50.0)

    cfg = TECHNICAL_CFG.get("price_vs_ma", {})
    upper_anchor_pct = cfg["upper_anchor_pct"]
    upper_anchor_score = cfg["upper_anchor_score"]
    upper_max_score = cfg["upper_max_score"]
    upper_scale_pct = cfg["upper_scale_pct"]
    mid_score = cfg["mid_score"]
    positive_scale = cfg["positive_scale"]
    lower_anchor_pct = cfg["lower_anchor_pct"]
    lower_anchor_score = cfg["lower_anchor_score"]
    negative_scale = cfg["negative_scale"]
    lower_floor_score = cfg["lower_floor_score"]
    lower_decay = cfg["lower_decay"]

    if pct_diff >= upper_anchor_pct:
        # Scale from upper_anchor_score at +upper_anchor_pct to upper_max_score at +(upper_anchor_pct + upper_scale_pct)
        upper_span = upper_max_score - upper_anchor_score
        return min(
            upper_max_score,
            upper_anchor_score + (pct_diff - upper_anchor_pct) * (upper_span / upper_scale_pct),
        )
    if pct_diff >= 0:
        return mid_score + pct_diff * positive_scale
    if pct_diff > lower_anchor_pct:
        return mid_score + pct_diff * negative_scale
    # pct_diff <= lower_anchor_pct: linear decay from anchor toward floor
    return max(lower_floor_score, lower_anchor_score - (abs(pct_diff) - abs(lower_anchor_pct)) * lower_decay)


def _score_momentum(momentum_20d: float | None, percentile_rank: float | None = None) -> float:
    """
    Score 20-day momentum.
    Ideally uses percentile rank within S&P 500 universe (0–100).
    Falls back to raw return mapping if percentile not available
    (mid_score and raw_scale from scoring.yaml `technical.momentum`).
    """
    if percentile_rank is not None:
        return float(percentile_rank)

    cfg = TECHNICAL_CFG.get("momentum", {})
    mid_score = cfg.get("mid_score", 50.0)

    if momentum_20d is None:
        return mid_score

    score = mid_score + momentum_20d * cfg.get("raw_scale", 3.0)
    return max(0.0, min(100.0, score))


# ── Sub-score breakout (for ablation analysis) ────────────────────────────────


def compute_technical_sub_scores(
    indicators: dict,
    market_regime: str = "neutral",
    momentum_percentile: float | None = None,
) -> dict:
    """Return the 5 per-signal sub-scores that feed compute_technical_score.

    Used by the archive writer to persist per-sub-signal scores into
    team_candidates so the backtester can run weight-ablation analysis
    (re-rank under alternate composite weights) without re-running the
    research pipeline. Each sub-score is in [0, 100].

    Args:
        indicators: dict from price_fetcher.compute_technical_indicators().
        market_regime: 'bull' | 'neutral' | 'caution' | 'bear' — affects
            the RSI thresholds.
        momentum_percentile: percentile rank (0–100) within S&P 500 for
            20d return.

    Returns:
        {rsi: float, macd: float, ma50: float, ma200: float, momentum: float}
    """
    return {
        "rsi": _score_rsi(
            indicators.get("rsi_14", 50.0),
            market_regime=market_regime,
        ),
        "macd": _score_macd(
            indicators.get("macd_cross", 0.0),
            indicators.get("macd_above_zero", False),
        ),
        "ma50": _score_price_vs_ma(indicators.get("price_vs_ma50")),
        "ma200": _score_price_vs_ma(indicators.get("price_vs_ma200")),
        "momentum": _score_momentum(
            indicators.get("momentum_20d"),
            percentile_rank=momentum_percentile,
        ),
    }


# ── Composite score ───────────────────────────────────────────────────────────

def compute_technical_score(
    indicators: dict,
    market_regime: str = "neutral",
    momentum_percentile: float | None = None,
    *,
    sector: str | None = None,
    team_id: str | None = None,
) -> float:
    """
    Compute weighted composite technical score (0–100).

    Weights and predictor-enrichment gate come from scoring.yaml
    `technical.composite_weights` (global) and optional per-sector
    overrides at `technical.composite_weights_per_sector` (keyed by
    sector team_id: technology / healthcare / financials / industrials
    / consumer / defensives). Predictor enrichment is unaffected by
    overrides.

    Args:
        indicators: dict from price_fetcher.compute_technical_indicators()
        market_regime: 'bull' | 'neutral' | 'caution' | 'bear'
        momentum_percentile: percentile rank (0–100) within S&P 500 for 20d return.
                             If None, falls back to raw return mapping.
        sector: GICS sector name (e.g. "Healthcare"); resolves to team_id
            via team_config.SECTOR_TEAM_MAP. Ignored if team_id is given.
        team_id: canonical sector team_id; takes precedence over `sector`.

    Returns: float in [0, 100]
    """
    rsi_score = _score_rsi(
        indicators.get("rsi_14", 50.0),
        market_regime=market_regime,
    )
    macd_score = _score_macd(
        indicators.get("macd_cross", 0.0),
        indicators.get("macd_above_zero", False),
    )
    ma50_score = _score_price_vs_ma(indicators.get("price_vs_ma50"))
    ma200_score = _score_price_vs_ma(indicators.get("price_vs_ma200"))
    momentum_score = _score_momentum(
        indicators.get("momentum_20d"),
        percentile_rank=momentum_percentile,
    )

    resolved_team_id = _resolve_team_id(team_id, sector)
    weights = _resolve_composite_weights(resolved_team_id)
    composite = (
        rsi_score * weights["rsi"]
        + macd_score * weights["macd"]
        + ma50_score * weights["ma50"]
        + ma200_score * weights["ma200"]
        + momentum_score * weights["momentum"]
    )

    # ── Predictor enrichment (optional) ──────────────────────────────────────
    # Keys present only when alpha-engine-predictor has run and written to S3.
    # Falls through to existing composite unchanged if absent or below confidence gate.
    pred_cfg = TECHNICAL_CFG.get("predictor_enrichment", {})
    confidence_gate = pred_cfg.get("confidence_gate", 0.65)
    max_adjustment = pred_cfg.get("max_adjustment", 10.0)

    p_up = indicators.get("p_up")
    p_down = indicators.get("p_down")
    confidence = indicators.get("prediction_confidence", 0.0)
    if p_up is not None and p_down is not None and confidence >= confidence_gate:
        # (p_up - p_down) in [-1, +1]; scale to ±max_adjustment pts weighted by confidence.
        direction_signal = (p_up - p_down) * max_adjustment * confidence
        composite = composite + direction_signal
    # ─────────────────────────────────────────────────────────────────────────

    return round(max(0.0, min(100.0, composite)), 2)


def compute_momentum_percentiles(
    momentum_data: dict[str, float | None],
) -> dict[str, float]:
    """
    Compute percentile ranks for 20d momentum across a universe of tickers.
    Returns {ticker: percentile_rank_0_to_100}.
    """
    import numpy as np

    valid = [(t, m) for t, m in momentum_data.items() if m is not None]
    if not valid:
        return dict.fromkeys(momentum_data, 50.0)

    tickers, values = zip(*valid, strict=True)
    values_arr = np.array(values, dtype=float)
    ranks = (values_arr.argsort().argsort() / max(len(values_arr) - 1, 1)) * 100

    result = {t: round(float(r), 1) for t, r in zip(tickers, ranks, strict=True)}
    # Fill any missing (None momentum) with 50
    for t in momentum_data:
        result.setdefault(t, 50.0)
    return result
