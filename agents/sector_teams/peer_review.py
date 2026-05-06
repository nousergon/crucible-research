"""
Peer Review — Intra-team review between Quant and Qual analysts.

1. If qual added a candidate: quant reviews it (do the numbers support?)
2. Joint finalization: produce final 2-3 recommendations with combined scores.

Uses single Haiku calls (no ReAct) — the peer review is a structured evaluation,
not an open-ended exploration.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from graph.llm_cost_tracker import get_cost_telemetry_callback

from config import (
    ANTHROPIC_API_KEY,
    MAX_TOKENS_PER_STOCK,
    MAX_TOKENS_STRATEGIC,
    PER_STOCK_MODEL,
    TEAM_PICKS_PER_RUN,
)
from agents.prompt_loader import load_prompt

log = logging.getLogger(__name__)


def run_peer_review(
    team_id: str,
    quant_picks: list[dict],
    qual_assessments: list[dict],
    additional_candidate: Optional[dict],
    technical_scores: dict,
    market_regime: str,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run intra-team peer review and produce final recommendations.

    Args:
        team_id: Sector team identifier.
        quant_picks: Quant analyst's top 5 (ticker, quant_score, rationale).
        qual_assessments: Qual analyst's assessments (ticker, qual_score, bull_case, bear_case).
        additional_candidate: Qual's extra candidate (or None).
        technical_scores: {ticker: dict} for quant review of additional.
        market_regime: Current macro regime.

    Returns:
        {
            "team_id": str,
            "recommendations": list[dict],  # final 2-3 picks
            "additional_accepted": bool,
            "peer_review_rationale": str,
        }
    """
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_PER_STOCK,
        callbacks=[get_cost_telemetry_callback()],
    )

    # Step 1: If qual added a candidate, quant reviews it
    additional_accepted = False
    if additional_candidate and additional_candidate.get("ticker"):
        additional_accepted = _quant_reviews_addition(
            llm, team_id, additional_candidate, technical_scores
        )

    # Step 2: Joint finalization — select final 2-3
    all_candidates = _merge_candidates(quant_picks, qual_assessments, additional_candidate, additional_accepted)

    if len(all_candidates) <= TEAM_PICKS_PER_RUN:
        # Not enough to need selection — return all
        return {
            "team_id": team_id,
            "recommendations": all_candidates,
            "additional_accepted": additional_accepted,
            "peer_review_rationale": "All candidates advanced (fewer than max picks).",
        }

    # Joint finalization via single Haiku call
    result = _joint_finalization(llm, team_id, all_candidates, market_regime)

    return {
        "team_id": team_id,
        "recommendations": result["picks"],
        "additional_accepted": additional_accepted,
        "peer_review_rationale": result["rationale"],
    }


def _quant_reviews_addition(
    llm: ChatAnthropic,
    team_id: str,
    candidate: dict,
    technical_scores: dict,
) -> bool:
    """Quant reviews qual's additional candidate. Returns True if accepted."""
    ticker = candidate.get("ticker", "")
    ts = technical_scores.get(ticker, {})

    loaded_prompt = load_prompt("peer_review_quant_addition")
    prompt = loaded_prompt.format(
        team_title=team_id.title(),
        ticker=ticker,
        qual_rationale=candidate.get("rationale", "No rationale provided"),
        qual_score=candidate.get("qual_score", "N/A"),
        rsi_14=ts.get("rsi_14", "N/A"),
        macd_cross=ts.get("macd_cross", "N/A"),
        price_vs_ma50=ts.get("price_vs_ma50", "N/A"),
        price_vs_ma200=ts.get("price_vs_ma200", "N/A"),
        momentum_20d=ts.get("momentum_20d", "N/A"),
        atr_pct=ts.get("atr_pct", "N/A"),
        technical_score=ts.get("technical_score", "N/A"),
    )

    # PR 2.2 Step C: flip _quant_reviews_addition to with_structured_output.
    # Strict mode raises on parse failure; lax mode keeps the silent-False
    # fallback (rejecting the qual addition is the conservative editorial
    # behavior — the team can still produce 2-3 picks from the quant set).
    from graph.state_schemas import QuantAcceptanceVerdict
    from strict_mode import is_strict_validation_enabled

    structured_llm = llm.with_structured_output(QuantAcceptanceVerdict)
    try:
        verdict: QuantAcceptanceVerdict = structured_llm.invoke(
            [HumanMessage(content=prompt)],
            config={"metadata": loaded_prompt.langsmith_metadata()},
        )
        log.info(
            "[peer_review:%s] quant %s qual's addition %s: %s",
            team_id,
            "accepted" if verdict.accept else "rejected",
            ticker,
            verdict.reason,
        )
        return verdict.accept
    except Exception as e:
        if is_strict_validation_enabled():
            raise
        log.warning(
            "[peer_review:%s] quant review of %s failed: %s", team_id, ticker, e
        )

    return False


def _merge_candidates(
    quant_picks: list[dict],
    qual_assessments: list[dict],
    additional: Optional[dict],
    additional_accepted: bool,
) -> list[dict]:
    """Merge quant picks with qual assessments into combined candidates."""
    # Build lookup by ticker
    qual_by_ticker = {a["ticker"]: a for a in qual_assessments}

    merged = []
    for qp in quant_picks:
        ticker = qp["ticker"]
        qa = qual_by_ticker.get(ticker, {})
        merged.append({
            "ticker": ticker,
            "quant_score": qp.get("quant_score", 0),
            "quant_rationale": qp.get("rationale", ""),
            "qual_score": qa.get("qual_score"),
            "bull_case": qa.get("bull_case", ""),
            "bear_case": qa.get("bear_case", ""),
            "catalysts": qa.get("catalysts", []),
            # Option A 2026-04-30: agent-format conviction is int 0-100 or
            # None. The string default ``"medium"`` is gone — qual analyst
            # emits int per qual_analyst_user.txt v1.1.0. None means qual
            # didn't emit a conviction for this ticker; downstream
            # normalize_conviction maps None → "stable".
            "conviction": qa.get("conviction"),
            "resources_used": qa.get("resources_used", []),
        })

    # Add the additional candidate if accepted
    if additional_accepted and additional and additional.get("ticker"):
        ticker = additional["ticker"]
        if ticker not in {m["ticker"] for m in merged}:
            merged.append({
                "ticker": ticker,
                "quant_score": additional.get("quant_score", 0),
                "quant_rationale": "",
                "qual_score": additional.get("qual_score"),
                "bull_case": additional.get("rationale", ""),
                "bear_case": "",
                "catalysts": [],
                # Same int-or-None convention as the merged loop above.
                "conviction": additional.get("conviction"),
                "resources_used": [],
                "is_qual_addition": True,
            })

    return merged


def _joint_finalization(
    llm: ChatAnthropic,
    team_id: str,
    candidates: list[dict],
    market_regime: str,
) -> dict:
    """Two-pass joint finalization (selection then per-ticker rationale).

    Pass 1: Haiku picks final 2-3 tickers from merged candidates and
            emits cross-pick context (team_rationale). Output shape is
            ``JointSelectionOutput`` — list[str] of tickers + a 1-2
            sentence team rationale, ~200 tokens, bounded by
            construction.

    Pass 2: For each selected ticker, a separate Haiku call produces a
            ``JointFinalizationDecision`` (ticker + rationale ≤ 50
            words). Each call is self-bounded; total per-team output
            scales linearly in N picks but no single response can blow
            past max_tokens.

    Replaced single-pass ``JointFinalizationOutput`` call after
    truncation incidents 2026-05-03 + 2026-05-06 where Haiku's
    rationale verbosity drift across 2-3 picks pushed combined output
    past max_tokens_strategic mid-emission, losing the entire
    selection.

    Lax-mode fallback (sort by combined score) is preserved — every
    team MUST produce picks for the downstream merge — and applies if
    Pass 1 fails. If Pass 1 succeeds but a Pass 2 call fails, the
    ticker still ships with an empty rationale (don't lose the pick
    just because rationale generation hiccups).
    """
    from graph.state_schemas import JointSelectionOutput, JointFinalizationDecision
    from strict_mode import is_strict_validation_enabled

    # Pass 1 LLM instance — strategic-tier token budget for selection
    # output. Even though the new schema is small (~200 tokens),
    # MAX_TOKENS_STRATEGIC is the right ceiling: it leaves slack for
    # team_rationale verbosity and matches the prior call shape.
    finalization_llm = ChatAnthropic(
        model=llm.model,
        anthropic_api_key=llm.anthropic_api_key,
        max_tokens=MAX_TOKENS_STRATEGIC,
        callbacks=llm.callbacks,
    )

    # ── Pass 1: selection ────────────────────────────────────────────
    candidates_text = "\n".join(
        f"  {c['ticker']}: quant={c.get('quant_score', '?')}, qual={c.get('qual_score', '?')}, "
        f"conviction={c.get('conviction', '?')}, bull={c.get('bull_case', '')[:80]}"
        for c in candidates
    )
    selection_prompt = load_prompt("peer_review_joint_selection")
    p1_text = selection_prompt.format(
        team_title=team_id.title(),
        market_regime=market_regime,
        candidates_text=candidates_text,
        team_picks_per_run=TEAM_PICKS_PER_RUN,
    )

    selection_structured = finalization_llm.with_structured_output(JointSelectionOutput)
    selection: Optional[JointSelectionOutput] = None
    try:
        selection = selection_structured.invoke(
            [HumanMessage(content=p1_text)],
            config={"metadata": selection_prompt.langsmith_metadata()},
        )
    except Exception as e:
        if is_strict_validation_enabled():
            raise
        log.warning(
            "[peer_review:%s] Pass 1 (selection) failed: %s — applying combined-score fallback",
            team_id, e,
        )

    if selection is None or not selection.selected_tickers:
        # Pass 1 failed entirely — combined-score fallback (preserves
        # invariant that every team produces picks).
        for c in candidates:
            qs = c.get("quant_score") or 0
            qls = c.get("qual_score") or 0
            c["_combined"] = (qs + qls) / 2 if qls else qs
        candidates.sort(key=lambda x: x["_combined"], reverse=True)
        return {
            "picks": candidates[:TEAM_PICKS_PER_RUN],
            "rationale": "Fallback: selected by combined quant+qual score.",
        }

    # Selection succeeded — clamp to TEAM_PICKS_PER_RUN, drop tickers
    # not in the candidate set (LLM hallucination guard).
    candidate_by_ticker = {c["ticker"]: c for c in candidates}
    selected = [
        t for t in selection.selected_tickers if t in candidate_by_ticker
    ][:TEAM_PICKS_PER_RUN]

    # ── Pass 2: per-ticker rationale ─────────────────────────────────
    rationale_prompt = load_prompt("peer_review_per_ticker_rationale")
    rationale_structured = finalization_llm.with_structured_output(JointFinalizationDecision)
    rationale_by_ticker: dict[str, str] = {}

    for ticker in selected:
        candidate = candidate_by_ticker[ticker]
        # Compact candidate context — full quant/qual/conviction fields
        # for this single ticker. ~100-200 tokens.
        candidate_context = (
            f"ticker={ticker}, quant_score={candidate.get('quant_score', '?')}, "
            f"qual_score={candidate.get('qual_score', '?')}, "
            f"conviction={candidate.get('conviction', '?')}, "
            f"rr_ratio={candidate.get('rr_ratio', '?')}, "
            f"bull_case={candidate.get('bull_case', '')[:200]}, "
            f"bear_case={candidate.get('bear_case', '')[:200]}"
        )
        p2_text = rationale_prompt.format(
            team_title=team_id.title(),
            ticker=ticker,
            market_regime=market_regime,
            candidate_context=candidate_context,
            team_rationale=selection.team_rationale,
        )
        try:
            decision: JointFinalizationDecision = rationale_structured.invoke(
                [HumanMessage(content=p2_text)],
                config={"metadata": rationale_prompt.langsmith_metadata()},
            )
            rationale_by_ticker[ticker] = decision.rationale
        except Exception as e:
            # Pass 2 hiccup for one ticker — keep the pick, log the
            # gap. Don't let one rationale failure poison the slate.
            if is_strict_validation_enabled():
                raise
            log.warning(
                "[peer_review:%s] Pass 2 rationale failed for %s: %s",
                team_id, ticker, e,
            )
            rationale_by_ticker[ticker] = ""

    picks = []
    for c in candidates:
        if c["ticker"] in rationale_by_ticker:
            pick = dict(c)
            # Per-pick rationale captured for LLM-as-judge eval — flows
            # into ``recommendations`` and on into decision artifacts.
            pick["peer_review_rationale"] = rationale_by_ticker[c["ticker"]]
            picks.append(pick)
    return {
        "picks": picks[:TEAM_PICKS_PER_RUN],
        "rationale": selection.team_rationale,
    }
