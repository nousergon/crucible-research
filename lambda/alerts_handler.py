"""
Intraday price alert Lambda.

Runs every 30 minutes during market hours (9:30am–4:00pm ET, Mon–Fri).
Checks current prices for universe + active candidates and fires an email
alert when any ticker moves ≥5% from prior close.

No LLM, no agents. Execution time: ~15 seconds. Lambda memory: 256 MB.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import tempfile
from typing import Optional

import boto3
import pytz
import yfinance as yf
from botocore.exceptions import ClientError
from exchange_calendars import get_calendar

# Load secrets from SSM Parameter Store (must run before any os.environ.get)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ssm_secrets import load_secrets
load_secrets()

# Structured logging + flow-doctor singleton from alpha-engine-lib.
# See lambda/handler.py for the full rationale. flow-doctor.yaml ships
# in the Lambda task root (Dockerfile.alerts COPY). exclude_patterns
# starts empty by deliberate convention — add patterns only after
# observing real ERROR-level noise from this Lambda.
from alpha_engine_lib.logging import setup_logging
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flow-doctor.yaml")
setup_logging(
    "research-alerts",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

from config import (
    PRICE_MOVE_THRESHOLD_PCT,
    EMAIL_RECIPIENTS,
    EMAIL_SENDER,
    S3_BUCKET,
    AWS_REGION,
)

# In-memory cooldown tracking (resets on cold start — acceptable)
_alerts_fired: dict[str, datetime.datetime] = {}
_COOLDOWN_MINUTES = int(os.environ.get("ALERT_COOLDOWN_MINUTES", "60"))


def is_market_open() -> bool:
    """Return True if NYSE is currently open (approximate check)."""
    nyse = get_calendar("XNYS")
    now_et = datetime.datetime.now(pytz.timezone("America/New_York"))
    today = now_et.date()
    if not nyse.is_session(today):
        return False
    # Check if within 9:30–16:00 ET
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _in_cooldown(ticker: str) -> bool:
    """Return True if an alert was fired for this ticker within the cooldown window."""
    if ticker not in _alerts_fired:
        return False
    elapsed = datetime.datetime.now() - _alerts_fired[ticker]
    return elapsed.total_seconds() < _COOLDOWN_MINUTES * 60


def _get_prior_closes(db_path: str) -> dict[str, float]:
    """Load prior-day closing prices from research.db."""
    import sqlite3
    closes: dict[str, float] = {}
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """SELECT symbol, score FROM investment_thesis
               WHERE date = (SELECT MAX(date) FROM investment_thesis)
               GROUP BY symbol"""
        ).fetchall()
        # Note: investment_thesis doesn't store price directly;
        # we use technical_scores table for prior closes
        rows = conn.execute(
            """SELECT t.symbol, p.current_price
               FROM technical_scores t
               JOIN (
                   SELECT symbol, MAX(date) as max_date FROM technical_scores GROUP BY symbol
               ) p ON t.symbol = p.symbol AND t.date = p.max_date"""
        ).fetchall()
        conn.close()
        for symbol, price in rows:
            if price:
                closes[symbol] = float(price)
    except Exception as e:
        # WARNING because the function returns an empty dict and the
        # caller continues with degraded coverage — not a hard failure.
        logger.warning("Error reading prior closes: %s", e)
    return closes


def _get_active_candidates_from_db(db_path: str) -> list[str]:
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT symbol FROM active_candidates").fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _get_population_tickers_from_db(db_path: str) -> list[str]:
    """Read population tickers from research.db (replaces hardcoded UNIVERSE_TICKERS)."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT symbol FROM population").fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current intraday prices via yfinance."""
    try:
        df = yf.download(
            tickers=tickers,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        prices = {}
        if len(tickers) == 1:
            prices[tickers[0]] = float(df["Close"].dropna().iloc[-1])
        else:
            for t in tickers:
                try:
                    prices[t] = float(df[t]["Close"].dropna().iloc[-1])
                except Exception:
                    pass
        return prices
    except Exception as e:
        # WARNING — caller continues with empty dict (no alerts fire
        # for this 30-min window), not a hard failure for the Lambda.
        logger.warning("Price fetch error: %s", e)
        return {}


def _send_alert_email(alerts: list[dict], prior_closes: dict, ratings: dict) -> None:
    """Send a price alert email via SES."""
    if not alerts:
        return

    lines = ["⚡ PRICE ALERTS\n"]
    for a in alerts:
        ticker = a["ticker"]
        change = a["change_pct"]
        current = a["current_price"]
        prior = prior_closes.get(ticker, 0)
        rating = ratings.get(ticker, {}).get("rating", "N/A")
        score = ratings.get(ticker, {}).get("score", "N/A")
        direction = "+" if change >= 0 else ""
        lines.append(
            f"⚡ {ticker}: {direction}{change:.1f}% from prior close\n"
            f"   Current: ${current:.2f} | Prior close: ${prior:.2f}\n"
            f"   Rating: {rating} | Score: {score}\n"
        )

    body = "\n".join(lines)
    subject = f"⚡ Price Alert — {', '.join(a['ticker'] for a in alerts)}"

    ses = boto3.client("ses", region_name=AWS_REGION)
    try:
        ses.send_email(
            Source=EMAIL_SENDER,
            Destination={"ToAddresses": EMAIL_RECIPIENTS},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Text": {"Data": body}},
            },
        )
        logger.info("Alert sent for: %s", [a["ticker"] for a in alerts])
    except ClientError as e:
        # ERROR — alert delivery failed; flow-doctor should escalate
        # because the operator needs to know the 30-min alert pipeline
        # is dropping price-move signals.
        logger.error(
            "SES alert error: %s", e.response["Error"]["Message"]
        )


def handler(event, context):
    """AWS Lambda handler for intraday price alerts."""
    if not is_market_open():
        return {"status": "SKIPPED", "reason": "market_closed"}

    # Preflight: AWS_REGION + S3 bucket reachable. Runs after the
    # market-open short-circuit so we don't pay the S3 head_bucket call
    # on ~70% of fires that land outside market hours.
    from preflight import ResearchPreflight
    ResearchPreflight(bucket=S3_BUCKET, mode="alerts").run()

    # Download research.db for prior closes + candidate/population tickers
    local_db = os.path.join(tempfile.gettempdir(), "research_alerts.db")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        s3.download_file(S3_BUCKET, "research.db", local_db)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return {"status": "SKIPPED", "reason": "no_db_found"}
        raise

    prior_closes = _get_prior_closes(local_db)
    candidate_tickers = _get_active_candidates_from_db(local_db)
    population_tickers = _get_population_tickers_from_db(local_db)
    all_tickers = list(set(population_tickers + candidate_tickers))

    # Fetch current prices
    current_prices = _get_current_prices(all_tickers)

    # Check for significant moves
    alerts = []
    for ticker in all_tickers:
        if _in_cooldown(ticker):
            continue
        prior = prior_closes.get(ticker)
        current = current_prices.get(ticker)
        if prior and current and prior > 0:
            change_pct = ((current / prior) - 1) * 100
            if abs(change_pct) >= PRICE_MOVE_THRESHOLD_PCT:
                alerts.append({
                    "ticker": ticker,
                    "change_pct": change_pct,
                    "current_price": current,
                })
                _alerts_fired[ticker] = datetime.datetime.now()

    if alerts:
        _send_alert_email(alerts, prior_closes, ratings={})

    return {
        "status": "OK",
        "alerts_fired": len(alerts),
        "tickers_checked": len(all_tickers),
    }
