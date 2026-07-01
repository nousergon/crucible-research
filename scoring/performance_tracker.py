"""
Score performance feedback loop (§5.6) — BUY-signal recording.

Records BUY-rated stocks (score >= 70) into score_performance at scoring
time so the canonical 21d outcome (beat_spy_21d / return_21d /
spy_21d_return / log_alpha_21d) can be attached later by the
alpha-engine-data producer (signal_returns._backfill_score_returns,
alpha-engine-data#197).

The legacy 10d/30d evaluation loop (run_performance_checks,
get_trading_day_offset, _get_spy_price_on_date, _compute_accuracy_stats)
was retired here (config#1456 canonical-alpha cutover): it had no
non-test consumer (its accuracy_10d/accuracy_30d/recalibration_flag
output was write-only into unread lambda state), and the canonical 21d
horizon it would otherwise need to migrate to is produced out-of-repo,
not by this module. See alpha-engine-config#1479.

No LLM involved.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from config import RATING_BUY_THRESHOLD


def record_new_buy_scores(
    db_conn: sqlite3.Connection,
    today: str,
    investment_theses: dict[str, dict],
    price_data: dict,
    market_regime: Optional[str] = None,
) -> None:
    """
    At end of run: record any ticker scored >= BUY_THRESHOLD today into score_performance.
    price_data: {ticker: current_price}

    The 5 calibrator-v1 context columns (quant_score, qual_score, conviction,
    sector_modifier, market_regime) are populated when present on the thesis
    dict + market_regime arg; missing values write NULL (older callers without
    the new arg / fields stay backward-compatible). The per-ticker
    sector_modifier is pulled from ``thesis["macro_modifier"]`` — that's the
    per-sector modifier value applied to this ticker at scoring time
    (renamed in the column for clarity; the in-memory field name is a
    legacy artifact). See ROADMAP P0 line ~103 for the v1-GBM upgrade plan.
    """
    cursor = db_conn.cursor()

    for ticker, thesis in investment_theses.items():
        if "final_score" not in thesis:
            raise KeyError(
                f"Thesis for {ticker} missing final_score — build_thesis_record "
                "must set it. Silent default=0 would skip performance tracking."
            )
        score = thesis["final_score"]
        if score < RATING_BUY_THRESHOLD:
            continue

        current_price = price_data.get(ticker)
        if current_price is None:
            continue

        cursor.execute(
            """
            INSERT OR IGNORE INTO score_performance (
                symbol, score_date, score, price_on_date,
                quant_score, qual_score, conviction,
                sector_modifier, market_regime
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, today, score, current_price,
                thesis.get("quant_score"),
                thesis.get("qual_score"),
                thesis.get("conviction"),
                thesis.get("macro_modifier"),
                market_regime,
            ),
        )

    db_conn.commit()
