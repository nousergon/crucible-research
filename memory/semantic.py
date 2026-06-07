"""
Semantic memory extraction — extracts cross-agent observations from pipeline outputs.

Runs after the archive writer, extracting sector-level insights from:
1. Sector team recommendations (industry themes, sector dynamics)
2. Macro report (regime reasoning, allocation rationale)
3. CIO decisions (portfolio construction logic)

Cost-capped: MAX_SEMANTIC_EXTRACTIONS per run to prevent runaway costs.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

from graph.state_schemas import ADVANCE_DECISIONS

logger = logging.getLogger(__name__)

MAX_SEMANTIC_EXTRACTIONS = 10  # hard cap per run (~$0.05 max)


def extract_semantic_memories(
    db_conn: sqlite3.Connection,
    sector_team_outputs: dict[str, dict],
    macro_report: str,
    market_regime: str,
    ic_decisions: list[dict],
    run_date: str,
    api_key: str | None = None,
) -> int:
    """
    Extract semantic memories from this run's outputs.

    Uses Haiku to distill sector-level observations, macro reasoning,
    and portfolio construction logic into reusable memories.

    Returns number of new memories created.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from config import ANTHROPIC_API_KEY, PER_STOCK_MODEL
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    # Wire to cost-telemetry stream (Phase 0.2 of the cost-optimization
    # workstream — previously this site was emitting LLM calls with
    # zero ``cost_usd`` attribution since the ``ChatAnthropic`` instance
    # had no telemetry callback). The callback pipes per-call token
    # counts into whatever ``track_llm_cost`` frame is active on the
    # research SF context. If no frame is active (e.g. unit-test path
    # without ``track_llm_cost`` wrap), the callback is a no-op.
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=256,
        callbacks=[get_cost_telemetry_callback()],
    )

    n_created = 0

    # 1. Extract from sector team outputs (1 per team, up to 6)
    for team_id, output in sector_team_outputs.items():
        if n_created >= MAX_SEMANTIC_EXTRACTIONS:
            break

        recs = output.get("recommendations", [])
        if not recs:
            continue

        tickers = [r.get("ticker", "?") for r in recs[:3]]
        bull_cases = [r.get("bull_case", "")[:100] for r in recs[:3]]
        summary = "; ".join(f"{t}: {b}" for t, b in zip(tickers, bull_cases) if b)

        if not summary:
            continue

        prompt = f"""From this sector team ({team_id}) analysis on {run_date}:
Recommendations: {summary}
Market regime: {market_regime}

Extract ONE sector-level observation (not stock-specific) that would be useful for future analysis of this sector. Focus on industry dynamics, sector trends, or thematic patterns.

Respond ONLY with JSON: {{"observation": "...", "sector": "...", "related_tickers": [...]}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                _save_memory(
                    db_conn, "sector_observation", f"team:{team_id}",
                    data.get("observation", ""), data.get("sector"),
                    data.get("related_tickers"), run_date,
                )
                n_created += 1
        except Exception as e:
            logger.debug("[semantic_extractor] team %s extraction failed: %s", team_id, e)

    # 2. Extract from macro report (1 memory)
    if n_created < MAX_SEMANTIC_EXTRACTIONS and macro_report and len(macro_report) > 100:
        prompt = f"""From this macro report ({run_date}, regime: {market_regime}):
{macro_report[:800]}

Extract ONE key regime reasoning observation that explains WHY the current regime classification was chosen. This should help future macro analysis maintain regime continuity.

Respond ONLY with JSON: {{"observation": "...", "sector": null}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                _save_memory(
                    db_conn, "macro_reasoning", "macro",
                    data.get("observation", ""), None, None, run_date,
                )
                n_created += 1
        except Exception as e:
            logger.debug("[semantic_extractor] macro extraction failed: %s", e)

    # 3. Extract cross-sector observation from IC decisions (1 memory)
    if n_created < MAX_SEMANTIC_EXTRACTIONS and ic_decisions:
        advanced = [d for d in ic_decisions if d.get("decision") in ADVANCE_DECISIONS]
        rejected = [d for d in ic_decisions if d.get("decision") == "REJECT"]

        if advanced or rejected:
            adv_text = ", ".join(d.get("ticker", "?") for d in advanced[:5])
            rej_text = ", ".join(d.get("ticker", "?") for d in rejected[:5])

            prompt = f"""CIO decisions on {run_date} (regime: {market_regime}):
Advanced: {adv_text or 'none'}
Rejected: {rej_text or 'none'}

Extract ONE cross-sector portfolio construction insight that would inform future CIO decisions. Focus on what themes or patterns drove the selections.

Respond ONLY with JSON: {{"observation": "...", "related_tickers": [...]}}"""

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                text = response.content
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(text[start:end])
                    _save_memory(
                        db_conn, "cross_sector", "cio",
                        data.get("observation", ""), None,
                        data.get("related_tickers"), run_date,
                    )
                    n_created += 1
            except Exception as e:
                logger.debug("[semantic_extractor] CIO extraction failed: %s", e)

    if n_created:
        logger.info("[semantic_extractor] created %d semantic memories", n_created)
    return n_created


def _save_memory(
    db_conn: sqlite3.Connection,
    category: str,
    source: str,
    content: str,
    sector: str | None,
    related_tickers: list[str] | None,
    run_date: str,
) -> None:
    """Save a semantic memory, handling duplicates gracefully."""
    if not content or len(content) < 10:
        return
    tickers_json = json.dumps(related_tickers) if related_tickers else None
    try:
        db_conn.execute(
            "INSERT INTO memory_semantic "
            "(category, source, content, sector, related_tickers, created_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (category, source, content[:500], sector, tickers_json, run_date),
        )
        db_conn.commit()
    except sqlite3.IntegrityError:
        # Duplicate — reinforce
        db_conn.execute(
            "UPDATE memory_semantic SET reinforced_date = ? "
            "WHERE category = ? AND source = ? AND content = ?",
            (run_date, category, source, content[:500]),
        )
        db_conn.commit()
