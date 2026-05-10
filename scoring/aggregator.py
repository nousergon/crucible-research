"""
Score aggregator — computes weighted composite attractiveness score.

Formula:
  Base  = quant × w_quant + qual × w_qual   (weights sum to 1.0)
  Score = Base + macro_shift   (clipped to [0, 100])

  macro_shift = (sector_modifier - 1.0) / 0.30 × MACRO_MAX_SHIFT_POINTS
  → modifier 0.70 → -10 pts  |  1.0 → 0 pts  |  1.30 → +10 pts

Horizon separation: Research focuses on 6–12 month fundamental attractiveness
using quant and qual analysis. Technical analysis is handled by the Predictor
(daily GBM inference) and Executor (ATR stops, time exits).

Design rationale: additive bounded shift preserves conviction-level ratings.
A caution regime (modifier ~0.85) nudges scores down ~5 pts rather than
multiplying by 0.85 which would suppress every stock in the universe uniformly.
High-conviction stocks stay BUY through moderate headwinds; only truly marginal
scores flip to HOLD/SELL, which is the desired analyst-aligned behavior.

Uses per-sector macro modifiers from the Macro Agent, not a single global shift.
"""

from __future__ import annotations

import json
import logging
import os

from config import (
    WEIGHT_QUANT,
    WEIGHT_QUAL,
    RATING_BUY_THRESHOLD,
    RATING_SELL_THRESHOLD,
    SECTOR_MAP,
    STALENESS_THRESHOLD_DAYS,
    MATERIAL_SCORE_CHANGE_MIN,
    get_research_params,
)

logger = logging.getLogger(__name__)

DEFAULT_SECTOR_MODIFIER = 1.0   # fallback if sector or macro data unavailable


def _trading_days_between(start_date: str | None, end_date: str) -> int:
    """Count NYSE trading days between two dates (exclusive of start, inclusive of end)."""
    if not start_date:
        return 0
    try:
        from datetime import datetime as _dt, timedelta
        from exchange_calendars import get_calendar
        import pandas as pd
        nyse = get_calendar("XNYS")
        start = pd.Timestamp(_dt.strptime(start_date, "%Y-%m-%d").date()) + timedelta(days=1)
        end = pd.Timestamp(_dt.strptime(end_date, "%Y-%m-%d").date())
        if start > end:
            return 0
        sessions = nyse.sessions_in_range(start, end)
        return len(sessions)
    except Exception as e:
        logger.warning("_trading_days_between failed (%s → %s): %s — falling back to approx",
                        start_date, end_date, e)
        # Fallback: approximate with business days
        from datetime import datetime as _dt
        try:
            delta = (_dt.strptime(end_date, "%Y-%m-%d") - _dt.strptime(start_date, "%Y-%m-%d")).days
            return max(0, delta * 5 // 7)
        except Exception as e2:
            logger.warning(
                "_trading_days_between approx fallback also failed (%s → %s): %s",
                start_date, end_date, e2,
            )
            return 0

# Module-level cache: populated once per Lambda cold-start by _get_weights().
_weights_cache: dict | None = None
# Local cache persists last known optimal across Lambda cold-starts (via /tmp).
_WEIGHTS_CACHE_PATH = os.environ.get(
    "SCORING_WEIGHTS_CACHE", "/tmp/scoring_weights_cache.json"
)


def _load_weights_from_s3() -> dict | None:
    """
    Check S3 for backtester-updated scoring weights.

    Reads s3://{RESEARCH_BUCKET}/config/scoring_weights.json, written by the
    backtester's weight optimizer when it applies an update.

    Fallback chain: S3 → local cache file → None (hardcoded defaults).
    On successful S3 read, writes a local cache so the last known optimal
    weights survive transient S3 failures.
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    key = "config/scoring_weights.json"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        weights = {
            "quant": float(data["quant"]),
            "qual": float(data["qual"]),
        } if "quant" in data and "qual" in data else {}
        if len(weights) == 2:
            logger.info(
                "Scoring weights loaded from S3 (updated %s, n=%s): %s",
                data.get("updated_at", "unknown"),
                data.get("n_samples", "?"),
                weights,
            )
            # Persist to local cache for fault tolerance
            try:
                with open(_WEIGHTS_CACHE_PATH, "w") as f:
                    json.dump(weights, f, indent=2)
            except Exception as e:
                logger.debug("Could not write scoring weights cache: %s", e)
            return weights
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            logger.warning("Could not read scoring weights from S3: %s", e)
    except Exception as e:
        logger.warning("Unexpected error loading S3 scoring weights: %s", e)

    # Fallback: last known optimal from local cache
    try:
        if os.path.exists(_WEIGHTS_CACHE_PATH):
            with open(_WEIGHTS_CACHE_PATH) as f:
                weights = json.load(f)
            if "quant" in weights and "qual" in weights:
                logger.info(
                    "Scoring weights loaded from local cache (last known optimal): %s",
                    weights,
                )
                return weights
    except Exception as e2:
        logger.debug("Could not read local scoring weights cache: %s", e2)

    return None


def _get_weights() -> tuple[float, float]:
    """
    Return current scoring weights (quant, qual), checking S3 override first.

    Result is cached for the lifetime of the Lambda instance (one S3 call
    per cold-start). Falls back to universe.yaml values if S3 file absent.
    """
    global _weights_cache
    if _weights_cache is None:
        s3_weights = _load_weights_from_s3()
        _weights_cache = s3_weights or {
            "quant": WEIGHT_QUANT,
            "qual": WEIGHT_QUAL,
        }
    return _weights_cache["quant"], _weights_cache["qual"]
MACRO_MODIFIER_RANGE = 0.30     # distance from 1.0 to min/max (0.70 and 1.30)
MACRO_MAX_SHIFT_POINTS = 10.0   # max pts added/subtracted by macro shift


def compute_attractiveness_score(
    ticker: str,
    quant_score: float,
    qual_score: float,
    sector_modifiers: dict[str, float],
    sector_map: dict[str, str] | None = None,
) -> dict:
    """
    Compute the final attractiveness score for a ticker.

    Uses quant + qual only (6–12 month fundamental attractiveness).
    Technical analysis is handled by Predictor and Executor.

    Returns dict with:
      quant_score, qual_score, macro_modifier,
      weighted_base, macro_shift, final_score, rating
    """
    sm = sector_map or SECTOR_MAP
    sector = sm.get(ticker, "Technology")
    macro_modifier = sector_modifiers.get(sector, DEFAULT_SECTOR_MODIFIER)
    macro_modifier = max(0.70, min(1.30, macro_modifier))  # clamp to valid range

    w_quant, w_qual = _get_weights()
    weighted_base = (
        quant_score * w_quant
        + qual_score * w_qual
    )

    # Additive bounded shift: (modifier - 1.0) / 0.30 × 10 → range [-10, +10]
    macro_shift = (macro_modifier - 1.0) / MACRO_MODIFIER_RANGE * MACRO_MAX_SHIFT_POINTS
    final_score = max(0.0, min(100.0, weighted_base + macro_shift))

    return {
        "ticker": ticker,
        "sector": sector,
        "quant_score": round(quant_score, 2),
        "qual_score": round(qual_score, 2),
        "macro_modifier": round(macro_modifier, 3),
        "macro_shift": round(macro_shift, 2),
        "weighted_base": round(weighted_base, 2),
        "final_score": round(final_score, 2),
        "rating": score_to_rating(final_score),
    }


def score_to_rating(score: float) -> str:
    """Convert numeric score to BUY / HOLD / SELL (§5.5)."""
    if score >= RATING_BUY_THRESHOLD:
        return "BUY"
    if score >= RATING_SELL_THRESHOLD:
        return "HOLD"
    return "SELL"


def check_consistency(
    quant_score: float | None,
    qual_score: float | None,
    score: float,
) -> bool:
    """
    Flag inconsistency between quant and qual sub-scores.

    Returns True if large divergence detected — one sub-score strongly bullish
    while the other is strongly bearish. Uses numeric comparison rather than
    keyword matching for reliability.
    """
    if quant_score is None or qual_score is None:
        return False

    rp = get_research_params()
    divergence_threshold = rp["consistency_divergence_threshold"]

    if abs(quant_score - qual_score) >= divergence_threshold:
        if (quant_score > RATING_BUY_THRESHOLD and qual_score < RATING_SELL_THRESHOLD) or \
           (qual_score > RATING_BUY_THRESHOLD and quant_score < RATING_SELL_THRESHOLD):
            return True
    return False


def compute_staleness(
    last_material_change_date: str | None,
    trading_days_since: int,
) -> bool:
    """
    Return True if the score is stale (no material change in >= STALENESS_THRESHOLD_DAYS).
    """
    if last_material_change_date is None:
        return False
    return trading_days_since >= STALENESS_THRESHOLD_DAYS


def compute_conviction(score_history: list[float]) -> str:
    """
    Derive conviction from the last 3 scores (most recent first).

    rising:   2+ of the last 2 deltas are positive
    declining: 2+ of the last 2 deltas are negative
    stable:   mixed or insufficient history
    """
    if len(score_history) < 2:
        return "stable"
    # Deltas: [score[0]-score[1], score[1]-score[2]] (recent first)
    deltas = [score_history[i] - score_history[i + 1] for i in range(min(2, len(score_history) - 1))]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    if pos > neg:
        return "rising"
    if neg > pos:
        return "declining"
    return "stable"


def compute_score_velocity_5d(score_history: list[float]) -> float | None:
    """
    Average daily score change over last 5 runs (most recent first).
    Returns None if fewer than 2 data points.
    """
    if len(score_history) < 2:
        return None
    window = score_history[:5]
    return round((window[0] - window[-1]) / (len(window) - 1), 3)


def compute_signal(
    rating: str,
    prior_rating: str | None,
    conviction: str,
    material_changes: bool,
) -> str:
    """
    Derive an actionable signal for the executor.

    ENTER:  rating is BUY (executor skips tickers already in portfolio)
    EXIT:   rating is SELL (regardless of prior)
    REDUCE: BUY → HOLD transition with declining conviction or material change
    HOLD:   all other cases
    """
    if rating == "BUY":
        return "ENTER"
    if rating == "SELL":
        return "EXIT"
    if rating == "HOLD" and prior_rating == "BUY" and (conviction == "declining" or material_changes):
        return "REDUCE"
    return "HOLD"


def compute_long_term_score(
    quant_score_lt: float,
    qual_score_lt: float,
    sector_modifiers: dict[str, float],
    sector: str,
) -> tuple[float, str]:
    """
    Compute a long-term (12-month) composite score and rating.

    Technical indicators are inherently short-term and excluded.
    Weights: quant_lt 50%, qual_lt 50%, plus the same macro sector shift.

    Returns (long_term_score, long_term_rating).
    """
    macro_modifier = sector_modifiers.get(sector, DEFAULT_SECTOR_MODIFIER)
    macro_modifier = max(0.70, min(1.30, macro_modifier))
    macro_shift = (macro_modifier - 1.0) / MACRO_MODIFIER_RANGE * MACRO_MAX_SHIFT_POINTS
    base = quant_score_lt * 0.50 + qual_score_lt * 0.50
    score = round(max(0.0, min(100.0, base + macro_shift)), 2)
    return score, score_to_rating(score)


def _compute_pead_boost(
    ticker: str,
    analyst_data: dict[str, dict],
    run_date: str | None = None,
) -> float:
    """
    PEAD boost (O10): Post-Earnings Announcement Drift.

    Recent positive earnings surprise → score boost.
    Stocks drift in the direction of earnings surprise for up to ~20 days.
    Thresholds and boost values read from research_params (S3 overridable).
    """
    from datetime import datetime

    ticker_analyst = analyst_data.get(ticker, {})
    earnings_data = ticker_analyst.get("earnings_surprises", [])
    if not earnings_data:
        return 0.0

    today = datetime.strptime(run_date, "%Y-%m-%d") if run_date else datetime.now()
    latest = earnings_data[0] if isinstance(earnings_data, list) else earnings_data

    report_date = latest.get("date", "")
    try:
        report_dt = datetime.strptime(report_date, "%Y-%m-%d")
        days_since = (today - report_dt).days
    except (ValueError, TypeError):
        return 0.0

    rp = get_research_params()
    if days_since < rp["pead_window_min_days"] or days_since > rp["pead_window_max_days"]:
        return 0.0

    surprise_pct = latest.get("surprise_pct", 0)
    if surprise_pct is None:
        surprise_pct = 0

    strong_thresh = rp["pead_strong_threshold_pct"]
    if surprise_pct > strong_thresh:
        return rp["pead_strong_boost"]
    elif surprise_pct > 0:
        return rp["pead_modest_boost"]
    elif surprise_pct < -strong_thresh:
        return rp["pead_strong_miss_boost"]
    elif surprise_pct < 0:
        return rp["pead_modest_miss_boost"]
    return 0.0


def _compute_revision_boost(ticker: str, revision_data: dict[str, dict]) -> float:
    """
    Revision boost (O11): EPS revision momentum.

    Rising EPS estimates are one of the strongest short-term bullish signals.
    Streak thresholds and boost values read from research_params.
    """
    data = revision_data.get(ticker, {})
    if not data:
        return 0.0

    rp = get_research_params()
    streak = data.get("revision_streak", 0)
    strong = rp["revision_strong_streak"]
    if streak >= strong:
        return rp["revision_strong_boost"]
    elif streak >= 1:
        return rp["revision_modest_boost"]
    elif streak <= -strong:
        return rp["revision_strong_negative_boost"]
    elif streak <= -1:
        return rp["revision_modest_negative_boost"]
    return 0.0


def _compute_options_adj(ticker: str, options_data: dict[str, dict]) -> float:
    """
    Options adjustment (O12): contrarian signal from options market positioning.

    High put/call ratio → options market bearish, reduce confidence.
    Low IV rank → cheap options → potential catalyst ahead.
    Thresholds and adjustments read from research_params.
    """
    data = options_data.get(ticker, {})
    if not data:
        return 0.0

    rp = get_research_params()
    adj = 0.0
    pc_ratio = data.get("put_call_ratio", 1.0)
    if pc_ratio > rp["options_high_pc_ratio"]:
        adj = rp["options_high_pc_adj"]
    elif pc_ratio < rp["options_low_pc_ratio"]:
        adj = rp["options_low_pc_adj"]

    iv_rank = data.get("iv_rank", 50)
    if iv_rank < rp["options_low_iv_rank"]:
        adj += rp["options_low_iv_adj"]

    return adj


def _compute_insider_boost(ticker: str, insider_data: dict[str, dict]) -> float:
    """
    Insider boost (O13): cluster buying by C-level insiders.

    Cluster buys are one of the strongest bullish signals. Net heavy selling
    is bearish. Thresholds and boost values read from research_params.
    """
    data = insider_data.get(ticker, {})
    if not data:
        return 0.0

    rp = get_research_params()
    boost = 0.0
    if data.get("cluster_buy"):
        boost = rp["insider_cluster_boost"]
    elif data.get("unique_buyers_30d", 0) >= rp["insider_min_unique_buyers"]:
        boost = rp["insider_unique_buyers_boost"]

    if data.get("net_sentiment", 0) < rp["insider_net_sentiment_threshold"]:
        boost = min(boost, rp["insider_net_sentiment_cap"])

    return boost


def _compute_short_interest_adj(ticker: str, short_interest_data: dict[str, dict], rating: str) -> float:
    """
    Short interest adjustment (Task 6A): squeeze potential on BUY signals.

    Thresholds and boost values read from research_params (S3 overridable).
    Note: yfinance SI data is delayed (bi-monthly FINRA).
    """
    data = short_interest_data.get(ticker, {})
    if not data:
        return 0.0

    short_pct = data.get("short_pct_float")  # already in % (e.g. 20 = 20%)
    if short_pct is None:
        return 0.0

    rp = get_research_params()
    buy_thresh = rp["short_interest_buy_threshold_pct"]
    high_thresh = rp["short_interest_high_threshold_pct"]
    buy_boost = rp["short_interest_buy_boost"]
    high_boost = rp["short_interest_high_boost"]

    adj = 0.0
    if rating == "BUY":
        if short_pct > high_thresh:
            adj = high_boost
        elif short_pct > buy_thresh:
            adj = buy_boost

    return adj


def aggregate_all(
    tickers: list[str],
    quant_scores: dict[str, float],
    qual_scores: dict[str, float],
    sector_modifiers: dict[str, float],
    prior_theses: dict[str, dict],
    sector_map: dict[str, str] | None = None,
    run_date: str | None = None,
    score_history: dict[str, list[float]] | None = None,
    price_target_upside: dict[str, float | None] | None = None,
    quant_scores_lt: dict[str, float] | None = None,
    qual_scores_lt: dict[str, float] | None = None,
    analyst_data: dict[str, dict] | None = None,
    revision_data: dict[str, dict] | None = None,
    options_data: dict[str, dict] | None = None,
    insider_data: dict[str, dict] | None = None,
    short_interest_data: dict[str, dict] | None = None,
    institutional_data: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """
    Run aggregation for all tickers in a single pass.

    Returns {ticker: full_result_dict} including conviction, signal,
    score_velocity_5d, price_target_upside, consistency_flag, stale_days,
    and O10-O13 boost values.

    Args:
        score_history: {ticker: [score_t0, score_t-1, score_t-2, ...]} most-recent first
        price_target_upside: {ticker: float | None} precomputed (price_target/price - 1)
        analyst_data: {ticker: dict} from analyst_fetcher — used for PEAD boost (O10)
        revision_data: {ticker: dict} from revision_fetcher — used for revision boost (O11)
        options_data: {ticker: dict} from options_fetcher — used for options adj (O12)
        insider_data: {ticker: dict} from insider_fetcher — used for insider boost (O13)
        short_interest_data: {ticker: dict} from price_fetcher — short squeeze potential
        institutional_data: {ticker: dict} from institutional_fetcher — 13F accumulation
    """
    _score_history = score_history or {}
    _pta = price_target_upside or {}
    _quant_lt = quant_scores_lt or {}
    _qual_lt = qual_scores_lt or {}
    _analyst = analyst_data or {}
    _revisions = revision_data or {}
    _options = options_data or {}
    _insider = insider_data or {}
    _short_interest = short_interest_data or {}
    _institutional = institutional_data or {}
    results = {}

    for ticker in tickers:
        quant_score = quant_scores.get(ticker)
        qual_score = qual_scores.get(ticker)

        # Skip tickers where both LLM scores failed
        if quant_score is None and qual_score is None:
            logger.warning("[aggregator] %s: both quant and qual scores failed — skipping", ticker)
            continue

        # If one score failed, use available sub-score at full weight
        score_partial = False
        if quant_score is None:
            logger.warning("[aggregator] %s: quant score failed — using qual score only", ticker)
            quant_score = qual_score
            score_partial = True
        elif qual_score is None:
            logger.warning("[aggregator] %s: qual score failed — using quant score only", ticker)
            qual_score = quant_score
            score_partial = True

        result = compute_attractiveness_score(
            ticker=ticker,
            quant_score=quant_score,
            qual_score=qual_score,
            sector_modifiers=sector_modifiers,
            sector_map=sector_map,
        )

        # ── O10-O13 signal boosts ──────────────────────────────────────────
        pead_boost = _compute_pead_boost(ticker, _analyst, run_date)
        revision_boost = _compute_revision_boost(ticker, _revisions)
        options_adj = _compute_options_adj(ticker, _options)
        insider_boost = _compute_insider_boost(ticker, _insider)

        # Short interest adjustment (Task 6A)
        short_interest_adj = _compute_short_interest_adj(ticker, _short_interest, result["rating"])

        # 13F institutional accumulation boost (Task 7A)
        institutional_boost = 0.0
        inst = _institutional.get(ticker, {})
        if inst.get("accumulation_signal"):
            institutional_boost = get_research_params()["institutional_boost"]

        # Apply boosts with aggregate cap (M2 fix)
        base_final = result["final_score"]
        total_boost_raw = (
            pead_boost + revision_boost + options_adj
            + insider_boost + short_interest_adj + institutional_boost
        )
        max_boost = get_research_params()["max_aggregate_boost"]
        total_boost = max(-max_boost, min(max_boost, total_boost_raw))
        adjusted_score = base_final + total_boost
        result["final_score"] = round(max(0.0, min(100.0, adjusted_score)), 2)
        result["rating"] = score_to_rating(result["final_score"])

        # Record individual boosts for backtester attribution
        result["pead_boost"] = round(pead_boost, 2)
        result["revision_boost"] = round(revision_boost, 2)
        result["options_adj"] = round(options_adj, 2)
        result["insider_boost"] = round(insider_boost, 2)
        result["short_interest_adj"] = round(short_interest_adj, 2)
        result["institutional_boost"] = round(institutional_boost, 2)
        result["total_boost_raw"] = round(total_boost_raw, 2)
        result["total_boost_capped"] = round(total_boost, 2)
        if score_partial:
            result["score_partial"] = True

        # Long-term composite (12-month horizon, no technical component)
        quant_lt = _quant_lt.get(ticker)
        qual_lt = _qual_lt.get(ticker)
        # Use available score if one failed (mirrors short-term logic above)
        if quant_lt is None and qual_lt is not None:
            quant_lt = qual_lt
        elif qual_lt is None and quant_lt is not None:
            qual_lt = quant_lt
        elif quant_lt is None and qual_lt is None:
            quant_lt, qual_lt = 50.0, 50.0  # both failed — use neutral for LT only
        lt_score, lt_rating = compute_long_term_score(
            quant_score_lt=quant_lt,
            qual_score_lt=qual_lt,
            sector_modifiers=sector_modifiers,
            sector=result["sector"],
        )

        # Prior data for change tracking
        prior = prior_theses.get(ticker, {})
        prior_score = prior.get("score")
        prior_rating = prior.get("rating")

        score_delta = None
        if prior_score is not None:
            score_delta = round(result["final_score"] - prior_score, 2)

        # Staleness tracking (§5.7) — uses actual trading days, not run count (M5 fix)
        last_change_date = prior.get("last_material_change_date")

        material_change = False
        if prior_score is None or abs(result["final_score"] - prior_score) >= MATERIAL_SCORE_CHANGE_MIN:
            material_change = True
            stale_days = 0
            last_change_date = run_date
        else:
            stale_days = _trading_days_between(last_change_date, run_date) if last_change_date else 0

        # Conviction and velocity from score history (§A.4)
        history = _score_history.get(ticker, [])
        if not history:
            logger.debug("[aggregate] %s: no score history — velocity unavailable", ticker)
        conviction = compute_conviction(history)
        velocity_5d = compute_score_velocity_5d(history)

        # Actionable signal for executor — driven by short-term final_score/rating (§A.3)
        signal = compute_signal(
            rating=result["rating"],
            prior_rating=prior_rating,
            conviction=conviction,
            material_changes=material_change,
        )

        # Consistency check (M6: numeric sub-score divergence instead of keyword matching)
        consistency_flag = check_consistency(quant_score, qual_score, result["final_score"])

        results[ticker] = {
            **result,
            "prior_score": prior_score,
            "prior_rating": prior_rating,
            "score_delta": score_delta,
            "last_material_change_date": last_change_date,
            "stale_days": stale_days,
            "material_changes": material_change,
            "consistency_flag": int(consistency_flag),
            "conviction": conviction,
            "score_velocity_5d": velocity_5d,
            "signal": signal,
            "price_target_upside": _pta.get(ticker),
            "long_term_score": lt_score,
            "long_term_rating": lt_rating,
        }

    assign_cross_sectional_ranks(results)
    return results


def assign_cross_sectional_ranks(results: dict[str, dict]) -> None:
    """Attach ``cross_sectional_rank`` + ``percentile`` to each result dict
    in-place.

    PR 1 of the rank-based portfolio-construction restructure. These
    additive fields are emitted on every per-ticker scoring dict so
    downstream consumers (executor, backtester signal_quality, dashboard)
    can migrate to rank-based selection without absolute thresholds.
    No behavior change in this PR — fields are observability-only until
    consumers wire to them.

    Semantics:

    - ``cross_sectional_rank``: 1-indexed integer where 1 = highest
      ``final_score`` in the run's scored population. Ties get the same
      rank via ``min`` semantics (e.g., scores [80, 75, 75, 60] →
      ranks [1, 2, 2, 4]). Stable across runs only if scores are
      deterministic — see typed-state arc 2026-04-30 for reducer
      ordering guarantees.
    - ``percentile``: float in ``[0.0, 1.0]`` where 1.0 = top. Computed
      as ``(n - rank) / max(n - 1, 1)`` so rank 1 → 1.0 and last → 0.0.
      Deterministic given ranks. Single-result population emits 1.0.

    Empty results: no-op. Tickers with non-finite ``final_score`` are
    sorted to the end (lowest rank, percentile 0.0) — defensive
    handling; the [0, 100] clamp upstream should make this unreachable
    in practice.

    Why ``final_score`` (not ``weighted_base``): emits the rank of the
    CURRENT pipeline output so downstream consumers see ranks that
    match the current rating/score they already read. The future
    restructure (PR 2+) may shift to ranking ``weighted_base`` so
    macro_shift can move to a sector-allocation layer rather than a
    per-stock score lever — but that's a behavior change, deferred.
    """
    n = len(results)
    if n == 0:
        return

    # Sort tickers by final_score descending; non-finite scores sink to
    # the bottom. ``-math.inf`` sentinel keeps the sort stable + deterministic.
    import math

    def _key(item: tuple[str, dict]) -> float:
        score = item[1].get("final_score")
        try:
            v = float(score) if score is not None else -math.inf
            return v if math.isfinite(v) else -math.inf
        except (TypeError, ValueError):
            return -math.inf

    sorted_items = sorted(results.items(), key=_key, reverse=True)

    # Assign ranks with ``min`` semantics for ties: scan in order and
    # bump the rank counter when the score strictly drops. First entry
    # gets rank 1; ties to its score also get rank 1.
    prev_score: float | None = None
    rank = 0
    seq = 0  # 1-indexed sequence position; rank only advances on score drop
    for ticker, result in sorted_items:
        seq += 1
        score = result.get("final_score")
        try:
            score_val: float | None = float(score) if score is not None else None
            # Coerce NaN/inf to None so the tie-detection branch below
            # (``score_val < prev_score``) doesn't return False on NaN
            # and silently bucket non-finite tickers into the prior
            # tie group.
            if score_val is not None and not math.isfinite(score_val):
                score_val = None
        except (TypeError, ValueError):
            score_val = None

        if rank == 0:
            rank = 1  # first entry
        elif prev_score is None or score_val is None:
            rank = seq  # non-finite handling: don't tie with valid scores
        elif score_val < prev_score:
            rank = seq  # advanced past prior tie group

        # Percentile: rank 1 → 1.0, rank n → 0.0. n=1 case returns 1.0.
        percentile = 1.0 - (rank - 1) / (n - 1) if n > 1 else 1.0

        result["cross_sectional_rank"] = rank
        result["percentile"] = round(percentile, 4)
        prev_score = score_val
