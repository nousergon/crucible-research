"""Per-team historical-accuracy producer for adaptive slot allocation (config#926 / config#1422).

Closes the producer gap left by config#926: that PR shipped the CONSUMER
(``archive/manager.py::load_team_accuracy`` + ``ADAPTIVE_SLOT_ALLOCATION_ENABLED``
+ ``compute_team_slots``'s ``team_accuracy`` nudge) but nothing wrote
``config/team_accuracy.json``, so flipping the flag was a no-op (load returns
``None`` -> silent fallback to static allocation).

This module computes each sector team's realized hit rate — the fraction of
its CIO-ADVANCED picks that beat SPY at the canonical 21d horizon — and emits
it to the fixed S3 key the consumer reads.

Substrate join (mirrors ``evals/last_week_scorecard.py``'s established
pattern): ``cio_evaluations`` carries ``team_id`` per ``(ticker, eval_date)``
but no realized outcome; ``score_performance`` carries the realized
``beat_spy_21d`` per ``(symbol, score_date)`` but no team attribution. A
CIO-ADVANCED ticker is scored into ``score_performance`` on the same cycle
(same date) it's ADVANCEd, so joining on ``(ticker=symbol, eval_date=score_date)``
attributes each realized outcome back to the team that recommended it.

Only ``cio_decision = 'ADVANCE'`` rows count: those are the picks that
actually entered the live population (per ``agents/investment_committee/ic_cio.py``'s
``ADVANCE`` / ``REJECT`` / ``NO_ADVANCE_DEADLOCK`` vocabulary) — REJECTed
candidates never traded, so their hypothetical realized return isn't a
reflection of the team's live decision quality.

Output shape: ``{team_id: {"accuracy": float in [0,1], "n_obs": int}}``,
written verbatim to ``config/team_accuracy.json`` — the exact contract
``archive/manager.py::load_team_accuracy`` documents and
``agents/sector_teams/team_config.py::_accuracy_adjustment`` consumes
(gated on ``n_obs >= ADAPTIVE_SLOT_MIN_OBS`` there, so under-sampled teams
are still emitted here — filtering happens once, at the read site — and
the artifact stays a complete audit trail rather than a lossy pre-filtered
view).

Failure posture: like the scorecard, this runs shadow-safe. Callers should
WARN-and-continue on any exception so the primary deliverable (the morning
briefing) never blocks on a secondary-observability producer that the
adaptive-slot consumer already treats as optional (``load_team_accuracy``
returns ``None`` gracefully).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Fixed S3 key the consumer reads (archive/manager.py::load_team_accuracy).
# Not the dated/latest eval-artifacts partition pattern — the consumer
# contract predates that convention and reads this single well-known key.
TEAM_ACCURACY_S3_KEY = "config/team_accuracy.json"

# Lookback window for the accuracy computation. Wider than the 4-week
# scorecard window on purpose: ADAPTIVE_SLOT_MIN_OBS=8 resolved recs/team
# is a much higher bar per-team than the scorecard's aggregate hit rate,
# and 21d-horizon resolution means a 4-week window only has ~1 fully-
# resolved cohort. 26 weeks (~6 months) gives teams a realistic chance to
# clear the minimum-observation gate while still being "recent enough" —
# roughly matches the backtester's own semi-annual recalibration cadence.
DEFAULT_LOOKBACK_WEEKS = 26


def analyze_team_performance(
    conn: sqlite3.Connection,
    as_of_date: date,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> dict[str, dict[str, Any]]:
    """Compute each team's realized 21d-horizon hit rate from research.db.

    ``as_of_date`` is the Saturday this analysis is being built FOR — i.e.
    next cycle's run date. The lookback window ends one day before
    ``as_of_date`` so the current cycle's own (unresolved) picks can't leak
    in, mirroring ``last_week_scorecard.build_scorecard``.

    Returns ``{team_id: {"accuracy": float, "n_obs": int}}`` for every team
    with at least one resolved observation in the window. Teams with zero
    resolved observations are omitted (not zero-filled) so the consumer's
    "team absent from the map" graceful-degrade path — already required by
    ``_accuracy_adjustment`` — is exercised rather than a fabricated 0.0
    accuracy that would look like a real bottom-percentile signal.
    """
    window_end = as_of_date - timedelta(days=1)
    window_start = window_end - timedelta(weeks=lookback_weeks)

    rows = _fetch_team_outcomes(conn, window_start.isoformat(), window_end.isoformat())

    by_team: dict[str, list[int]] = {}
    for r in rows:
        by_team.setdefault(r["team_id"], []).append(r["beat_spy_21d"])

    result: dict[str, dict[str, Any]] = {}
    for team_id, outcomes in sorted(by_team.items()):
        n_obs = len(outcomes)
        accuracy = sum(outcomes) / n_obs
        result[team_id] = {"accuracy": accuracy, "n_obs": n_obs}
    return result


def _fetch_team_outcomes(
    conn: sqlite3.Connection, start: str, end: str
) -> list[dict]:
    """Pull per-pick realized 21d beat-SPY outcomes, attributed to team_id.

    Joins CIO-ADVANCED picks (``cio_evaluations``, has ``team_id``, no
    outcome) against realized signal outcomes (``score_performance``, has
    ``beat_spy_21d``, no team attribution) on ``(ticker, eval_date)`` —
    both are written from the same research cycle for the same date, so
    the pair keys align. Only resolved rows (``beat_spy_21d IS NOT NULL``)
    and only ``team_id IS NOT NULL`` rows (defensive — CIO evaluations for
    exit-only / non-team-sourced candidates may carry a null team) count.
    """
    sql = """
        SELECT
            c.team_id,
            sp.beat_spy_21d
        FROM cio_evaluations c
        JOIN score_performance sp
            ON sp.symbol = c.ticker AND sp.score_date = c.eval_date
        WHERE c.eval_date BETWEEN ? AND ?
          AND c.cio_decision = 'ADVANCE'
          AND c.team_id IS NOT NULL
          AND sp.beat_spy_21d IS NOT NULL
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    return [{"team_id": r[0], "beat_spy_21d": r[1]} for r in rows]


def save_team_accuracy(
    team_accuracy: dict[str, dict[str, Any]],
    *,
    s3_client: Any,
    bucket: str,
    key: str = TEAM_ACCURACY_S3_KEY,
) -> None:
    """Write ``team_accuracy`` to the fixed S3 key the consumer reads.

    Single-key overwrite (like ``population/latest.json``), not a dated +
    latest sidecar pair — ``load_team_accuracy`` only ever reads ``key``
    directly, so there's no dated-history reader to serve.

    Per [[feedback_no_silent_fails]] this raises on any S3 failure — same
    posture as ``emit_scorecard_to_s3``. The caller (Lambda handler) is
    responsible for the shadow-mode WARN-and-continue wrapper so a producer
    failure here never blocks the Saturday morning briefing.
    """
    if not bucket:
        raise ValueError("save_team_accuracy requires a non-empty bucket")
    payload = json.dumps(team_accuracy, indent=2).encode("utf-8")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build the per-team historical-accuracy artifact "
            "(config#1422) from research.db."
        )
    )
    parser.add_argument("--db", required=True, help="Path to research.db.")
    parser.add_argument(
        "--as-of",
        required=True,
        help="ISO date this analysis is being built for (typically the next Saturday).",
    )
    parser.add_argument(
        "--lookback-weeks", type=int, default=DEFAULT_LOOKBACK_WEEKS,
        help=f"Lookback window in weeks (default {DEFAULT_LOOKBACK_WEEKS}).",
    )
    parser.add_argument(
        "--s3-bucket", default=None,
        help="Optional S3 bucket. When provided, writes config/team_accuracy.json.",
    )
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    try:
        as_of = date.fromisoformat(args.as_of)
        team_accuracy = analyze_team_performance(
            conn, as_of_date=as_of, lookback_weeks=args.lookback_weeks
        )
        print(json.dumps(team_accuracy, indent=2))

        if args.s3_bucket:
            import boto3
            save_team_accuracy(
                team_accuracy, s3_client=boto3.client("s3"), bucket=args.s3_bucket,
            )
            logger.info("team_accuracy emitted to s3://%s/%s", args.s3_bucket, TEAM_ACCURACY_S3_KEY)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
