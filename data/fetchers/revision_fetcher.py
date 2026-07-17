"""
data/fetchers/revision_fetcher.py — EPS revision tracking (O11).

Weekly snapshot of FMP EPS consensus, diff against prior week to create
revision momentum signal. "Estimates going up" is more predictive than
"estimates are high."

FMP free tier: 250 req/day. Each ticker uses 1 call.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

_FMP_V3 = "https://financialmodelingprep.com/api/v3"

# Use the shared FMP rate limiter from analyst_fetcher
from data.fetchers.analyst_fetcher import _fmp_get as _fmp_get_shared


def _fmp_get(endpoint: str, params: Optional[dict] = None) -> dict | list:
    return _fmp_get_shared(endpoint, params=params, base=_FMP_V3)


def fetch_revisions(
    tickers: list[str],
    bucket: str = "alpha-engine-research",
    reference_date: Optional[str] = None,
) -> dict[str, dict]:
    """
    Fetch current EPS consensus and compute revision metrics against prior week.

    For each ticker:
    1. Fetch current EPS estimate from FMP
    2. Load previous week's snapshot from S3
    3. Compute revision direction, magnitude, and streak

    Returns per ticker:
        eps_current: float — current consensus EPS estimate
        eps_previous: float — prior week's estimate (0 if no history)
        revision_pct: float — percent change in EPS estimate
        revision_direction: int — +1 (up), -1 (down), 0 (unchanged)
        revision_streak: int — consecutive weeks of same-direction revisions
    """
    today = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
    results: dict[str, dict] = {}

    # Fetch current estimates
    current_estimates: dict[str, float] = {}
    from data.fetchers.analyst_fetcher import fmp_budget_exhausted
    for ticker in tickers:
        if fmp_budget_exhausted():
            log.info("FMP budget exhausted — skipping remaining EPS estimates (%d/%d done)",
                     len(current_estimates), len(tickers))
            for t in tickers:
                current_estimates.setdefault(t, 0.0)
            break
        try:
            data = _fmp_get(f"analyst-estimates/{ticker}", params={"limit": 1})
            if isinstance(data, list) and data:
                current_estimates[ticker] = data[0].get("estimatedEpsAvg", 0.0)
            else:
                current_estimates[ticker] = 0.0
        except Exception as e:
            log.debug("EPS estimate fetch failed for %s: %s", ticker, e)
            current_estimates[ticker] = 0.0

    # Load prior week's snapshot from S3
    prior_snapshot = _load_best_prior_snapshot(today, bucket, lookback_days=14)

    # Load revision history for streak computation
    revision_history = _load_revision_streaks(today, bucket, lookback_weeks=8)

    for ticker in tickers:
        eps_current = current_estimates.get(ticker, 0.0)
        prior_data = prior_snapshot.get(ticker, {}) if prior_snapshot else {}
        eps_previous = prior_data.get("eps_current", 0.0)

        # Revision computation
        if abs(eps_previous) > 0.001:
            revision_pct = round((eps_current - eps_previous) / abs(eps_previous) * 100, 2)
        else:
            revision_pct = 0.0

        if revision_pct > 0.1:
            revision_direction = 1
        elif revision_pct < -0.1:
            revision_direction = -1
        else:
            revision_direction = 0

        # Streak: how many consecutive weeks has the revision been in the same direction
        streak = _compute_streak(ticker, revision_direction, revision_history)

        results[ticker] = {
            "eps_current": eps_current,
            "eps_previous": eps_previous,
            "revision_pct": revision_pct,
            "revision_direction": revision_direction,
            "revision_streak": streak,
        }

    # Save current snapshot for next week's comparison
    snapshot_data = {
        ticker: {"eps_current": current_estimates.get(ticker, 0.0)}
        for ticker in tickers
    }
    _save_revision_snapshot(snapshot_data, today.strftime("%Y-%m-%d"), bucket)

    log.info("Computed revisions for %d tickers (prior snapshot: %s)",
             len(results), "found" if prior_snapshot else "missing")
    return results


def _load_best_prior_snapshot(
    today: datetime,
    bucket: str,
    lookback_days: int = 14,
) -> Optional[dict]:
    """Try to load the most recent prior revision snapshot from S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        for days_ago in range(7, lookback_days + 1):
            check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            try:
                key = f"archive/revisions/{check_date}.json"
                obj = s3.get_object(Bucket=bucket, Key=key)
                return json.loads(obj["Body"].read())
            except Exception:
                continue
    except Exception as e:
        log.debug("Could not load prior revision snapshot: %s", e)
    return None


def _load_revision_streaks(
    today: datetime,
    bucket: str,
    lookback_weeks: int = 8,
) -> list[tuple[str, dict]]:
    """Load weekly revision snapshots for streak computation."""
    snapshots: list[tuple[str, dict]] = []
    try:
        import boto3
        s3 = boto3.client("s3")
        for weeks_ago in range(1, lookback_weeks + 1):
            target_date = today - timedelta(weeks=weeks_ago)
            for day_offset in range(7):
                check_date = (target_date - timedelta(days=day_offset)).strftime("%Y-%m-%d")
                try:
                    key = f"archive/revisions/{check_date}.json"
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    data = json.loads(obj["Body"].read())
                    snapshots.append((check_date, data))
                    break
                except Exception:
                    # Revision fetch failed for this checkpoint — try the next one
                    # (recorded: S3 list/read failures, JSON decode errors, truncated archives).
                    continue
    except Exception:
        # Outer fetch failed entirely (e.g., S3 list permissions) — empty result set
        # (recorded: boto3 auth, S3 prefix scan failures).
        pass
    snapshots.sort(key=lambda x: x[0])
    return snapshots


def _compute_streak(
    ticker: str,
    current_direction: int,
    history: list[tuple[str, dict]],
) -> int:
    """Compute consecutive weeks of same-direction revisions."""
    if current_direction == 0:
        return 0

    streak = current_direction  # start with current direction as +1 or -1

    if len(history) < 2:
        return streak

    # Walk backward through history
    for i in range(len(history) - 1, 0, -1):
        newer = history[i][1].get(ticker, {}).get("eps_current", 0.0)
        older = history[i - 1][1].get(ticker, {}).get("eps_current", 0.0)

        if abs(older) < 0.001:
            break

        delta_pct = (newer - older) / abs(older) * 100
        if delta_pct > 0.1 and current_direction > 0:
            streak += 1
        elif delta_pct < -0.1 and current_direction < 0:
            streak -= 1
        else:
            break

    return streak


def _save_revision_snapshot(
    data: dict[str, dict],
    date_str: str,
    bucket: str,
) -> None:
    """Save current EPS estimates snapshot to S3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        key = f"archive/revisions/{date_str}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json",
        )
        log.info("Saved revision snapshot to s3://%s/%s", bucket, key)
    except Exception as e:
        log.warning("Failed to save revision snapshot: %s", e)
