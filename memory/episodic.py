"""
Episodic memory extraction — converts failed BUY signal outcomes into lessons.

Runs after the canonical-primary-horizon (21d) outcome lands in the
long-format ``score_performance_outcomes`` store (EPIC config#1483,
consumer cutover config#1530), read here via ``evals.outcome_store`` and
joined onto ``score_performance`` by ``(symbol, score_date)`` — NOT the
retired wide horizon-suffixed score_performance columns. The long store is
DECIMAL-canonical (e.g. 0.043 for +4.3%); the legacy wide return/SPY columns
were 2dp-rounded PERCENT (4.30) while the log-domain alpha column was always
log-domain decimal — a units split this migration retires by reading
decimals uniformly.
Uses Haiku to extract a 1-2 sentence lesson from each failed signal.

Cost-capped: MAX_EPISODIC_EXTRACTIONS per run to prevent runaway costs.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

from evals import outcome_store

logger = logging.getLogger(__name__)

MAX_EPISODIC_EXTRACTIONS = 15  # hard cap per run (~$0.08 max)


def extract_memories(db_conn: sqlite3.Connection, api_key: str | None = None) -> int:
    """
    Extract episodic memories from recently completed score_performance outcomes.

    Finds BUY signals where the canonical-primary-horizon (21d) beat-SPY
    outcome resolved to a loss (underperformed) that don't yet have a
    memory_episodes entry. Uses Haiku to generate a lesson for each.

    Returns number of new memories created.
    """
    # Candidate BUY signals without an existing memory (outcome-agnostic —
    # the beat_spy=0 filter is applied in Python after joining the
    # long-format store, since that store, not score_performance, now
    # carries the outcome). SQL-side LIMIT is a generous upper bound on
    # candidates to scan (steady-state this WHERE clause matches only the
    # last cycle's newly-resolved signals, since memoried rows drop out of
    # candidacy going forward); the Python-side cap right below is the real
    # per-run extraction limit.
    _CANDIDATE_SCAN_LIMIT = MAX_EPISODIC_EXTRACTIONS * 50
    candidates = db_conn.execute("""
        SELECT sp.symbol, sp.score_date, sp.score,
               it.conviction, it.thesis_summary, it.signal,
               ms.market_regime, ms.vix
        FROM score_performance sp
        LEFT JOIN investment_thesis it
            ON sp.symbol = it.symbol AND sp.score_date = it.date
        LEFT JOIN macro_snapshots ms
            ON sp.score_date = ms.date
        LEFT JOIN memory_episodes me
            ON sp.symbol = me.ticker AND sp.score_date = me.signal_date
        WHERE me.id IS NULL
        ORDER BY sp.score_date DESC
        LIMIT ?
    """, (_CANDIDATE_SCAN_LIMIT,)).fetchall()

    outcomes = outcome_store.load_primary_outcomes(db_conn)
    rows = []
    for r in candidates:
        symbol, score_date = r[0], r[1]
        outcome = outcomes.get((symbol, score_date))
        if outcome is None or outcome.beat_spy != 0:
            continue
        rows.append((*r, outcome.stock_return, outcome.spy_return, outcome.log_alpha))
        if len(rows) >= MAX_EPISODIC_EXTRACTIONS * 2:  # fetch more, cap at extraction
            break

    if not rows:
        logger.info("[memory_extractor] no new failed signals to process")
        return 0

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from config import ANTHROPIC_API_KEY, PER_STOCK_MODEL
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    # Wire to cost-telemetry stream — see memory/semantic.py for the
    # rationale (Phase 0.2 of the cost-optimization workstream).
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=256,
        callbacks=[get_cost_telemetry_callback()],
    )

    n_created = 0
    for r in rows:
        if n_created >= MAX_EPISODIC_EXTRACTIONS:
            logger.warning("[memory_extractor] hit cap of %d extractions", MAX_EPISODIC_EXTRACTIONS)
            break

        symbol, score_date, score = r[0], r[1], r[2]
        conviction, thesis_summary, signal = r[3], r[4], r[5]
        regime, vix = r[6], r[7]
        stock_return, spy_return, log_alpha = r[8], r[9], r[10]

        # Canonical market-relative alpha: prefer the log-domain alpha from
        # the long-format store when available, fall back to arithmetic
        # excess. Both terms are decimals (the store's canonical unit), so
        # ``:.1%`` formatting below is correct without any *100 conversion —
        # unlike the retired wide-column read, which mixed a decimal
        # log-domain alpha with percent-scale return columns.
        outcome_vs_spy = (
            log_alpha
            if log_alpha is not None
            else (stock_return or 0) - (spy_return or 0)
        )

        prompt = f"""A BUY signal for {symbol} on {score_date} (score: {score}, conviction: {conviction}) underperformed SPY by {outcome_vs_spy:.1%} over 21 days.

Thesis: {(thesis_summary or 'N/A')[:300]}
Market regime: {regime or 'unknown'}, VIX: {vix or '?'}
Stock return: {stock_return:.1%}, SPY return: {spy_return:.1%}

In 1-2 sentences, what lesson should be remembered for future analysis of {symbol} or similar stocks? Also provide 2-3 pattern tags (e.g., "earnings", "margin_compression", "momentum_reversal").

Respond ONLY with JSON: {{"lesson": "...", "pattern_tags": ["...", "..."]}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content
            # Extract JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                lesson = data.get("lesson", "")
                tags = json.dumps(data.get("pattern_tags", []))
            else:
                lesson = text[:200]
                tags = "[]"

            # Get sector from investment_thesis or sector_map
            sector_row = db_conn.execute(
                "SELECT sector FROM stock_archive WHERE ticker = ? LIMIT 1", (symbol,)
            ).fetchone()
            sector = sector_row[0] if sector_row else None

            # `outcome_21d` (renamed from the stale `outcome_10d` name,
            # config#1480 — the column always stored the canonical 21d
            # realized return; only the name lagged) stores the
            # long-format store's decimal stock_return.
            db_conn.execute(
                "INSERT OR IGNORE INTO memory_episodes "
                "(ticker, signal_date, score, rating, conviction, thesis_summary, "
                "outcome_21d, outcome_vs_spy, lesson, sector, pattern_tags, created_date) "
                "VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, score_date, score, conviction, (thesis_summary or "")[:500],
                 stock_return, outcome_vs_spy, lesson, sector, tags, score_date),
            )
            db_conn.commit()
            n_created += 1
            logger.info("[memory_extractor] created memory for %s (%s): %s",
                       symbol, score_date, lesson[:80])

        except Exception as e:
            logger.warning("[memory_extractor] failed for %s (%s): %s", symbol, score_date, e)

    logger.info("[memory_extractor] created %d new episodic memories", n_created)
    return n_created
