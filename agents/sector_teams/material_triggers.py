"""
Material event triggers — determines whether a held population stock
needs its thesis updated this week.

All checks are deterministic (no LLM calls). Only stocks with a material
trigger get a thesis update Haiku call; others preserve their prior thesis.
"""

from __future__ import annotations

import logging
from datetime import datetime

log = logging.getLogger(__name__)

# ── Trigger Thresholds ────────────────────────────────────────────────────────

NEWS_VOLUME_THRESHOLD = 3        # novel articles in a week
PRICE_MOVE_ATR_MULTIPLE = 2.0    # price move > 2 × ATR triggers update
EARNINGS_PROXIMITY_DAYS = 5      # days before/after earnings
INSIDER_CLUSTER_THRESHOLD = 2    # insiders transacting in 30d


def check_material_triggers(
    ticker: str,
    news_data: dict | None,
    price_data: dict | None,
    analyst_data: dict | None,
    insider_data: dict | None,
    prior_thesis: dict | None,
    sector_regime_changed: bool,
    run_date: str,
) -> list[str]:
    """
    Check all material event triggers for a held stock.

    Returns:
        List of trigger names that fired. Empty list = no material events,
        thesis should be preserved as-is.
    """
    triggers = []

    # 1. News volume spike
    if news_data:
        article_count = news_data.get("article_count", 0)
        if isinstance(news_data.get("articles"), list):
            article_count = len(news_data["articles"])
        if article_count >= NEWS_VOLUME_THRESHOLD:
            triggers.append("news_volume_spike")

    # 2. Price move > 2 ATR
    if price_data is not None:
        try:
            close = price_data.get("Close", price_data.get("Adj Close"))
            if close is not None and len(close) >= 20:
                current = float(close.iloc[-1])
                prior_close = float(close.iloc[-5]) if len(close) >= 5 else current

                # Compute ATR from high/low/close
                high = price_data.get("High")
                low = price_data.get("Low")
                if high is not None and low is not None and len(high) >= 14:
                    tr_values = []
                    for i in range(-14, 0):
                        h = float(high.iloc[i])
                        lo = float(low.iloc[i])
                        c_prev = float(close.iloc[i - 1]) if abs(i - 1) < len(close) else lo
                        tr = max(h - lo, abs(h - c_prev), abs(lo - c_prev))
                        tr_values.append(tr)
                    atr = sum(tr_values) / len(tr_values) if tr_values else 0

                    if atr > 0:
                        move = abs(current - prior_close)
                        if move > PRICE_MOVE_ATR_MULTIPLE * atr:
                            triggers.append("price_move_gt_2atr")
        except Exception as e:
            log.debug("Price move check failed for %s: %s", ticker, e)

    # 3. Analyst revision — REMOVED (config#1821 Option B, 2026-07-08).
    # This used to key off ``analyst_data["rating_changes"]`` /
    # ``analyst_data["upside_pct"]``, both sourced from FMP's
    # grades-consensus / price-target-consensus endpoints. Those endpoints
    # 402'd for every ticker on the current plan (never populated in
    # practice) and were removed from fetch_analyst_consensus's returned
    # shape entirely, so this check would now always be a no-op.

    # 4. Earnings proximity
    if analyst_data:
        earnings = analyst_data.get("earnings_surprises", [])
        if earnings:
            try:
                latest = earnings[0]
                report_date = latest.get("date", "")
                if report_date:
                    rd = datetime.fromisoformat(run_date)
                    ed = datetime.strptime(report_date, "%Y-%m-%d")
                    days_since = (rd - ed).days
                    if 0 <= days_since <= EARNINGS_PROXIMITY_DAYS:
                        triggers.append("recent_earnings")
            except Exception as e:
                log.debug("Earnings proximity check failed: %s", e)

    # 5. Insider cluster activity
    if insider_data:
        buyers = insider_data.get("unique_buyers_30d", 0)
        if buyers >= INSIDER_CLUSTER_THRESHOLD:
            triggers.append("insider_cluster")
        # Also trigger on significant selling
        net_sentiment = insider_data.get("net_sentiment", 0)
        if net_sentiment <= -0.5:
            triggers.append("insider_selling")

    # 6. Sector regime change
    if sector_regime_changed:
        triggers.append("sector_regime_change")

    if triggers:
        log.info("[triggers:%s] material events: %s", ticker, triggers)

    return triggers
