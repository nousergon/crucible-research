"""
Score performance feedback loop (§5.6).

Tracks whether BUY-rated stocks (score ≥ 70) actually outperform SPY
over subsequent 10 and 30 trading day windows.

Invoked at the start of each daily run before any agents execute.
Reads from investment_thesis and technical_scores tables, fetches current
prices via yfinance, writes to score_performance table.
No LLM involved.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    RATING_BUY_THRESHOLD,
    RECALIBRATION_THRESHOLD,
    RECALIBRATION_LOOKBACK_DAYS,
)


def get_trading_day_offset(from_date: str, n_days: int, db_conn: sqlite3.Connection) -> Optional[str]:
    """
    Find the date N trading days after from_date.
    Uses exchange_calendars (authoritative NYSE schedule) with DB fallback.
    """
    try:
        from exchange_calendars import get_calendar
        nyse = get_calendar("XNYS")
        start = pd.Timestamp(from_date) + pd.Timedelta(days=1)
        sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_days * 2))
        if len(sessions) >= n_days:
            return sessions[n_days - 1].strftime("%Y-%m-%d")
    except Exception:
        pass
    # Fallback: use DB trading dates
    try:
        cursor = db_conn.execute(
            "SELECT DISTINCT date FROM technical_scores WHERE date > ? ORDER BY date ASC LIMIT ?",
            (from_date, n_days),
        )
        rows = cursor.fetchall()
        if len(rows) >= n_days:
            return rows[n_days - 1][0]
    except Exception:
        pass
    return None


def run_performance_checks(db_conn: sqlite3.Connection, today: str) -> dict:
    """
    Main entry point. Checks if any BUY-scored stocks from 10 or 30 trading
    days ago have realized returns to record. Updates score_performance table.

    Returns summary dict with accuracy stats.
    """
    cursor = db_conn.cursor()

    # Find all score_performance rows that need 10d or 30d evaluation
    rows_needing_eval = cursor.execute(
        """
        SELECT symbol, score_date, score, price_on_date
        FROM score_performance
        WHERE (price_10d IS NULL AND eval_date_10d IS NULL)
           OR (price_30d IS NULL AND eval_date_30d IS NULL)
        """,
    ).fetchall()

    if not rows_needing_eval:
        return _compute_accuracy_stats(db_conn, today)

    # Collect all tickers + SPY to fetch current prices
    tickers_needed = list({row[0] for row in rows_needing_eval}) + ["SPY"]

    # Try polygon grouped-daily first (1 API call for all tickers)
    polygon_prices: dict[str, float] = {}
    try:
        from polygon_client import polygon_client
        grouped = polygon_client().get_grouped_daily(today)
        if grouped:
            for t in tickers_needed:
                if t in grouped:
                    polygon_prices[t] = grouped[t]["close"]
    except Exception:
        pass

    # Fallback to yfinance for any missing tickers
    price_data = None
    if len(polygon_prices) < len(tickers_needed):
        try:
            price_data = yf.download(
                tickers=tickers_needed,
                period="2d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            if not polygon_prices:
                return _compute_accuracy_stats(db_conn, today)

    def get_latest_price(ticker: str) -> Optional[float]:
        if ticker in polygon_prices:
            return polygon_prices[ticker]
        if price_data is None:
            return None
        try:
            if len(tickers_needed) == 1:
                return float(price_data["Close"].dropna().iloc[-1])
            return float(price_data[ticker]["Close"].dropna().iloc[-1])
        except Exception:
            return None

    spy_price = get_latest_price("SPY")

    # Compute SPY reference prices from DB
    for row in rows_needing_eval:
        symbol, score_date, score, price_on_date = row
        current_price = get_latest_price(symbol)

        if current_price is None or price_on_date is None:
            continue

        # Check 10d and 30d windows based on trading day offset
        eval_10d = get_trading_day_offset(score_date, 10, db_conn)
        eval_30d = get_trading_day_offset(score_date, 30, db_conn)

        updates = {}

        if eval_10d and eval_10d <= today:
            ret_10d = (current_price / price_on_date) - 1
            spy_10d_return = None
            if spy_price:
                spy_entry_price = _get_spy_price_on_date(score_date, db_conn)
                if spy_entry_price:
                    spy_10d_return = (spy_price / spy_entry_price) - 1

            updates["price_10d"] = current_price
            updates["return_10d"] = round(ret_10d * 100, 2)
            updates["eval_date_10d"] = today
            updates["spy_10d_return"] = round(spy_10d_return * 100, 2) if spy_10d_return else None
            if spy_10d_return is not None:
                updates["beat_spy_10d"] = 1 if ret_10d > spy_10d_return else 0

        if eval_30d and eval_30d <= today:
            ret_30d = (current_price / price_on_date) - 1
            spy_30d_return = None
            if spy_price:
                spy_entry_price = _get_spy_price_on_date(score_date, db_conn)
                if spy_entry_price:
                    spy_30d_return = (spy_price / spy_entry_price) - 1

            updates["price_30d"] = current_price
            updates["return_30d"] = round(ret_30d * 100, 2)
            updates["eval_date_30d"] = today
            updates["spy_30d_return"] = round(spy_30d_return * 100, 2) if spy_30d_return else None
            if spy_30d_return is not None:
                updates["beat_spy_30d"] = 1 if ret_30d > spy_30d_return else 0

        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [symbol, score_date]
            cursor.execute(
                f"UPDATE score_performance SET {set_clause} WHERE symbol = ? AND score_date = ?",
                values,
            )

    db_conn.commit()
    return _compute_accuracy_stats(db_conn, today)


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


def _get_spy_price_on_date(date_str: str, db_conn: sqlite3.Connection) -> Optional[float]:
    """Look up SPY price from macro_snapshots if available."""
    try:
        row = db_conn.execute(
            "SELECT sp500_close FROM macro_snapshots WHERE date = ?", (date_str,)
        ).fetchone()
        return float(row[0]) if row and row[0] else None
    except Exception:
        return None


def _compute_accuracy_stats(db_conn: sqlite3.Connection, today: str) -> dict:
    """
    Compute BUY accuracy stats over trailing RECALIBRATION_LOOKBACK_DAYS.
    Returns dict with accuracy_10d, accuracy_30d, recalibration_flag.
    """
    try:
        rows = db_conn.execute(
            """
            SELECT beat_spy_10d, beat_spy_30d
            FROM score_performance
            WHERE score_date >= date(?, ?)
              AND beat_spy_10d IS NOT NULL
            """,
            (today, f"-{RECALIBRATION_LOOKBACK_DAYS} days"),
        ).fetchall()

        if not rows:
            return {"accuracy_10d": None, "accuracy_30d": None, "recalibration_flag": False}

        beat_10d = [r[0] for r in rows if r[0] is not None]
        beat_30d = [r[1] for r in rows if r[1] is not None]

        acc_10d = sum(beat_10d) / len(beat_10d) if beat_10d else None
        acc_30d = sum(beat_30d) / len(beat_30d) if beat_30d else None

        recal_flag = (
            (acc_10d is not None and acc_10d < RECALIBRATION_THRESHOLD)
            or (acc_30d is not None and acc_30d < RECALIBRATION_THRESHOLD)
        )

        return {
            "accuracy_10d": round(acc_10d * 100, 1) if acc_10d else None,
            "accuracy_30d": round(acc_30d * 100, 1) if acc_30d else None,
            "recalibration_flag": recal_flag,
            "sample_size": len(beat_10d),
        }
    except Exception:
        return {"accuracy_10d": None, "accuracy_30d": None, "recalibration_flag": False}
