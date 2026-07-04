"""
Executor surveillance Lambda (formerly intraday price alerts).

Runs every 30 minutes during market hours (9:30am–4:00pm ET, Mon–Fri).
Reads daemon-published intraday snapshots from S3 + research.db, computes
surveillance findings, and emits them as a Telegram rollup to the daemon's
shared channel.

**Reframe (2026-05-13, ROADMAP L1067 PR 3):** This Lambda was historically
a price-alert utility ("ticker moved >5% from prior close → email me"). It
has been re-scoped as an **independent surveillance channel on the
daemon** — same 30-min cadence, but now also catching executor non-decisions
(daemon down, universe drift between research and execution) and routing
all output through Telegram with severity tiering.

**Data sources (post-rewrite):**

- ``s3://{bucket}/intraday/heartbeat.json`` — daemon liveness. Stale = first
  failure mode checked; fires a critical Telegram push and short-circuits
  the rest of the run before reading possibly-stale price data.
- ``s3://{bucket}/intraday/latest_prices.json`` — daemon-published IB
  snapshot. Replaces the prior yfinance fetch path.
- ``research.db`` (downloaded from S3): ``population`` + ``active_candidates``
  + ``technical_scores`` tables. Prior closes still come from
  ``technical_scores`` (unchanged from the legacy handler).

**Surveillance findings emitted as a silent Telegram rollup digest:**

1. **Universe drift** — tickers in research universe but not in daemon's
   subscribed set (daemon's intraday data is incomplete for those tickers).
2. **Price-move alerts** — any ticker that moved
   ``PRICE_MOVE_THRESHOLD_PCT`` (default 5%) from prior close, with 60-min
   in-memory per-ticker cooldown (preserved from the legacy handler).

**Severity tiering:** heartbeat-stale fires loud (push, no
``disable_notification``) and short-circuits. All other findings batch into
a single silent (``disable_notification=True``) rollup digest — visible
in-channel but no phone buzz.

Lambda image: Python 3.12 Docker on ECR. Memory: 256 MB. Execution time:
~5s (no LLM, no yfinance). Deployed via ``./infrastructure/deploy.sh main``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import tempfile
from typing import Optional

import boto3
import pytz
from botocore.exceptions import ClientError
from exchange_calendars import get_calendar

# Ensure the project root is on sys.path so sibling modules can be imported.
# Secrets resolve on-demand via alpha_engine_lib.secrets.get_secret() at
# consumer sites — no module-top SSM fetch required.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Structured logging + flow-doctor singleton from alpha-engine-lib.
# See lambda/handler.py for the full rationale. flow-doctor.yaml ships
# in the Lambda task root (Dockerfile.alerts COPY).
from alpha_engine_lib.logging import monitor_handler, setup_logging
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "research-alerts",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

from alpha_engine_lib.telegram import send_rollup
from config import (
    AWS_REGION,
    PRICE_MOVE_THRESHOLD_PCT,
    S3_BUCKET,
)

# In-memory cooldown tracking (resets on cold start — acceptable).
_alerts_fired: dict[str, datetime.datetime] = {}
_COOLDOWN_MINUTES = int(os.environ.get("ALERT_COOLDOWN_MINUTES", "60"))

# Heartbeat staleness threshold. Daemon writes a heartbeat every
# poll_interval (~60s default); flag stale at 3× that to absorb transient
# S3 read jitter. Configurable for tests + future tuning.
_HEARTBEAT_STALE_SEC = int(os.environ.get("HEARTBEAT_STALE_SEC", "180"))

INTRADAY_HEARTBEAT_KEY = "intraday/heartbeat.json"
INTRADAY_LATEST_PRICES_KEY = "intraday/latest_prices.json"


def is_market_open() -> bool:
    """Return True if NYSE is currently open (approximate check)."""
    nyse = get_calendar("XNYS")
    now_et = datetime.datetime.now(pytz.timezone("America/New_York"))
    today = now_et.date()
    if not nyse.is_session(today):
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _in_cooldown(ticker: str) -> bool:
    """Return True if an alert was fired for this ticker within the cooldown window."""
    if ticker not in _alerts_fired:
        return False
    elapsed = datetime.datetime.now() - _alerts_fired[ticker]
    return elapsed.total_seconds() < _COOLDOWN_MINUTES * 60


def _read_s3_json(s3, bucket: str, key: str) -> Optional[dict]:
    """Best-effort S3 JSON read. Returns None on miss/error (logged)."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            logger.warning("S3 miss: s3://%s/%s", bucket, key)
        else:
            logger.warning("S3 read error for %s: %s", key, code or "unknown")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("S3 JSON decode error for %s: %s", key, e)
        return None


def _heartbeat_stale(heartbeat: dict | None, stale_threshold_sec: int) -> tuple[bool, str]:
    """Check if heartbeat is missing or older than threshold.

    Returns ``(is_stale, reason)``. ``is_stale=True`` if heartbeat is None,
    missing timestamp, malformed timestamp, or older than threshold.
    """
    if heartbeat is None:
        return True, "no heartbeat artifact present in S3"
    ts_str = heartbeat.get("timestamp")
    if not ts_str:
        return True, "heartbeat artifact missing timestamp field"
    try:
        # Accept both "Z" suffix and "+00:00" forms; daemon writes "Z".
        ts = datetime.datetime.fromisoformat(ts_str.rstrip("Z")).replace(
            tzinfo=datetime.timezone.utc,
        )
    except ValueError:
        return True, f"malformed heartbeat timestamp: {ts_str!r}"
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    age_sec = (now_utc - ts).total_seconds()
    if age_sec > stale_threshold_sec:
        return True, (
            f"heartbeat is {age_sec:.0f}s old (threshold {stale_threshold_sec}s); "
            f"last write {ts_str}"
        )
    return False, ""


def _get_prior_closes(db_path: str) -> dict[str, float]:
    """Load prior-day closing prices from research.db's ``technical_scores`` table."""
    import sqlite3
    closes: dict[str, float] = {}
    try:
        conn = sqlite3.connect(db_path)
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
    """Read population tickers from research.db's ``population`` table."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT symbol FROM population").fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _check_universe_drift(
    research_universe: set[str], subscribed_tickers: list[str]
) -> Optional[str]:
    """Compare research universe against daemon's subscribed set.

    Returns a finding string if research_universe ⊄ subscribed_tickers
    (daemon is missing tickers research expects to be surveilled), else
    None. Inverse drift (daemon subscribed to tickers not in research
    universe) is not flagged — SPY + held-but-rotated-out positions
    legitimately stay subscribed even after exiting the active research
    universe.
    """
    subscribed = set(subscribed_tickers)
    missing = research_universe - subscribed
    if not missing:
        return None
    sample = sorted(missing)[:5]
    suffix = f" ... +{len(missing) - 5} more" if len(missing) > 5 else ""
    return (
        f"Universe drift: {len(missing)} research tickers missing from "
        f"daemon subscription — {', '.join(sample)}{suffix}"
    )


def _check_price_moves(
    latest_prices: dict | None,
    prior_closes: dict[str, float],
    research_universe: set[str],
) -> list[str]:
    """Compute price-move findings. Mirrors legacy >X% threshold behavior."""
    findings: list[str] = []
    if not latest_prices:
        return findings
    prices = latest_prices.get("prices") or {}
    for ticker, price_data in prices.items():
        if ticker not in research_universe:
            continue
        if _in_cooldown(ticker):
            continue
        current = (
            price_data.get("last") if isinstance(price_data, dict) else None
        )
        prior = prior_closes.get(ticker)
        if not (current and prior and prior > 0):
            continue
        change_pct = ((current / prior) - 1) * 100
        if abs(change_pct) >= PRICE_MOVE_THRESHOLD_PCT:
            direction = "+" if change_pct >= 0 else ""
            findings.append(
                f"{ticker}: {direction}{change_pct:.1f}% "
                f"(${prior:.2f} → ${current:.2f})"
            )
            _alerts_fired[ticker] = datetime.datetime.now()
    return findings


@monitor_handler
def handler(event, context):
    """AWS Lambda handler — surveillance cadence (every 30 min during market hours)."""
    if not is_market_open():
        return {"status": "SKIPPED", "reason": "market_closed"}

    # Preflight: AWS_REGION + S3 bucket reachable.
    from preflight import ResearchPreflight
    ResearchPreflight(bucket=S3_BUCKET, mode="alerts").run()

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # 1. Heartbeat check — first failure mode. Stale = critical push +
    #    short-circuit (don't waste time reading possibly-stale prices).
    heartbeat = _read_s3_json(s3, S3_BUCKET, INTRADAY_HEARTBEAT_KEY)
    is_stale, stale_reason = _heartbeat_stale(heartbeat, _HEARTBEAT_STALE_SEC)
    if is_stale:
        msg = f"\U0001f6a8 *DAEMON DOWN*\n{stale_reason}"
        from ops_alerts import publish_ops_alert

        publish_ops_alert(
            msg,
            severity="critical",
            source="research:alerts_handler",
            dedup_key=f"daemon_down_{stale_reason[:80]}",
        )
        logger.error("Heartbeat stale: %s", stale_reason)
        return {"status": "DAEMON_DOWN", "reason": stale_reason}

    subscribed_tickers = list(heartbeat.get("subscribed_tickers") or [])

    # 2. Download research.db for universe + prior closes
    local_db = os.path.join(tempfile.gettempdir(), "research_alerts.db")
    try:
        s3.download_file(S3_BUCKET, "research.db", local_db)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return {"status": "SKIPPED", "reason": "no_db_found"}
        raise

    prior_closes = _get_prior_closes(local_db)
    candidate_tickers = _get_active_candidates_from_db(local_db)
    population_tickers = _get_population_tickers_from_db(local_db)
    research_universe = set(population_tickers + candidate_tickers)

    # 3. Universe drift check
    findings: list[str] = []
    drift_finding = _check_universe_drift(research_universe, subscribed_tickers)
    if drift_finding:
        findings.append(drift_finding)

    # 4. Price-move check (uses daemon-published S3 snapshot, not yfinance)
    latest_prices = _read_s3_json(s3, S3_BUCKET, INTRADAY_LATEST_PRICES_KEY)
    findings.extend(_check_price_moves(latest_prices, prior_closes, research_universe))

    # 5. Emit findings as silent rollup digest (no phone buzz unless heartbeat
    #    fired above)
    if findings:
        header = f"Surveillance Digest — {len(findings)} finding{'s' if len(findings) != 1 else ''}"
        send_rollup(findings, header=header)

    return {
        "status": "OK",
        "findings_count": len(findings),
        "research_universe_size": len(research_universe),
        "daemon_subscribed_count": len(subscribed_tickers),
    }
