"""
Thesis updater — converts aggregated scores to buy/sell/hold thesis records.

Reads composite scores from the aggregator and produces the final investment
thesis dict written to S3 and the DB. Operates per-ticker, no LLM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from scoring.aggregator import score_to_rating
from thesis.structured import build_structured_thesis


def build_thesis_record(
    ticker: str,
    run_date: str,
    aggregated: dict,
    agent_outputs: dict,
) -> dict:
    """
    Combine aggregator output and agent JSON outputs into a thesis record
    suitable for S3 JSON storage and DB write.

    Args:
        ticker: stock symbol
        run_date: YYYY-MM-DD string
        aggregated: composite score dict (final_score / weighted_base /
                    macro_shift / rating) from the score_aggregator path
                    (scoring.composite.compute_composite_breakdown)
        agent_outputs: dict with keys 'news_json', 'research_json'
                       (the JSON block at end of each agent report)

    Returns:
        thesis dict matching the investment_thesis DB schema
    """
    news_json = agent_outputs.get("news_json", {})
    research_json = agent_outputs.get("research_json", {})

    # Prefer agent-provided thesis summary; fall back to auto-generated
    thesis_summary = _build_summary(ticker, aggregated, news_json, research_json)

    # Build structured thesis (Phase 2: replaces free-text truncation)
    prior_structured = agent_outputs.get("prior_structured_thesis")
    structured_thesis = build_structured_thesis(
        news_json=news_json,
        research_json=research_json,
        aggregated=aggregated,
        prior_structured=prior_structured,
    )

    return {
        "ticker": ticker,
        "date": run_date,
        "rating": aggregated["rating"],
        "final_score": aggregated["final_score"],
        "quant_score": aggregated["quant_score"],
        "qual_score": aggregated["qual_score"],
        "macro_modifier": aggregated["macro_modifier"],
        "sector": aggregated.get("sector"),
        "thesis_summary": thesis_summary,
        "key_catalyst": news_json.get("key_catalyst") or research_json.get("key_upside"),
        "key_risk": research_json.get("key_risk"),
        "news_sentiment": news_json.get("sentiment"),
        "consensus_direction": research_json.get("consensus_direction"),
        "material_changes": aggregated.get("material_changes", False),
        "last_material_change_date": aggregated.get("last_material_change_date"),
        "stale_days": aggregated.get("stale_days", 0),
        "consistency_flag": aggregated.get("consistency_flag", 0),
        "prior_score": aggregated.get("prior_score"),
        "prior_rating": aggregated.get("prior_rating"),
        "score_delta": aggregated.get("score_delta"),
        # Executor signal fields (§A.1, A.3, A.4)
        "conviction": aggregated.get("conviction", "stable"),
        "signal": aggregated.get("signal", "HOLD"),
        "score_velocity_5d": aggregated.get("score_velocity_5d"),
        "price_target_upside": aggregated.get("price_target_upside"),
        # Long-term (12-month) scores — informational, not used by executor
        "long_term_score": aggregated.get("long_term_score"),
        "long_term_rating": aggregated.get("long_term_rating"),
        # Structured thesis for next run's agent prompts
        "structured_thesis": structured_thesis,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_summary(ticker: str, aggregated: dict, news_json: dict, research_json: dict) -> str:
    """
    Auto-generate a 1–2 sentence thesis summary from structured data.
    Used as thesis_summary in the email ratings table.
    The Consolidator Agent produces a richer version for buy candidates.
    """
    rating = aggregated["rating"]
    score = aggregated["final_score"]
    catalyst = news_json.get("key_catalyst") or research_json.get("key_upside") or ""
    risk = research_json.get("key_risk") or ""
    consensus = research_json.get("consensus_direction", "neutral")
    sentiment = news_json.get("sentiment", "neutral")

    parts = [f"{ticker} rates {rating} (score: {score:.0f})."]

    if consensus == "bullish":
        parts.append(f"Analyst consensus is bullish.")
    elif consensus == "bearish":
        parts.append(f"Analyst consensus is bearish.")

    if catalyst:
        parts.append(f"Key catalyst: {catalyst}")

    if risk and rating != "SELL":
        parts.append(f"Key risk: {risk}")

    return " ".join(parts)


def check_rating_change(thesis: dict, prior_thesis: Optional[dict]) -> Optional[str]:
    """
    Return a string describing a rating change if one occurred, else None.
    e.g., "HOLD → BUY"
    """
    if prior_thesis is None:
        return None
    prior = prior_thesis.get("rating") or prior_thesis.get("prev_rating")
    current = thesis["rating"]
    if prior and prior != current:
        return f"{prior} → {current}"
    return None
