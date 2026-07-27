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
but no realized outcome; the canonical-primary-horizon (21d) beat-SPY outcome
per ``(symbol, score_date)`` now comes from the long-format
``score_performance_outcomes`` store via ``evals.outcome_store``
(config#1483/config#1530 cutover — replaces the retired wide horizon-
suffixed score_performance column read), joined here by team_id having no
direct representation in that store. A CIO-ADVANCED ticker is scored on the
same cycle (same date) it's ADVANCEd, so joining on
``(ticker=symbol, eval_date=score_date)`` attributes each realized outcome
back to the team that recommended it.

Only ``cio_decision = 'ADVANCE'`` rows count: those are the picks that
actually entered the live population (per ``agents/investment_committee/ic_cio.py``'s
``ADVANCE`` / ``REJECT`` / ``NO_ADVANCE_DEADLOCK`` vocabulary) — REJECTed
candidates never traded, so their hypothetical realized return isn't a
reflection of the team's live decision quality.

Output shape (schema_version 1, config#1844 — self-describing envelope)::

    {
      "schema_version": 1,
      "status": "ok" | "insufficient",
      "as_of": "YYYY-MM-DD",
      "n_teams": int,
      "n_advance_picks": int,        # ADVANCE rows in window (resolved or not)
      "n_resolved_outcomes": int,    # join hits == sum of per-team n_obs
      "horizon_days": int,           # canonical primary horizon (21)
      "teams": {team_id: {"accuracy": float in [0,1], "n_obs": int}},
    }

written to ``config/team_accuracy.json``. The per-team payload under
``teams`` is the exact contract ``archive/manager.py::load_team_accuracy``
unwraps and ``agents/sector_teams/team_config.py::_accuracy_adjustment``
consumes (gated on ``n_obs >= ADAPTIVE_SLOT_MIN_OBS`` there, so
under-sampled teams are still emitted here — filtering happens once, at the
read site — and the artifact stays a complete audit trail rather than a
lossy pre-filtered view).

WHY an envelope (config#1844): on 2026-07-03 the live artifact was a bare
``{}`` — indistinguishable between honest insufficiency and silent
starvation (the config#1456/config#1840 bug class). Verified cause: the
long-format ``score_performance_outcomes`` store was first populated by
DataPhase2's signal_returns collector at 16:43 UTC that day, ~7h AFTER this
producer ran (09:46 UTC) inside the Research Lambda — an honest first-
population ordering gap, not a broken join. The envelope makes any future
empty payload carry its reason: counts + ``status="insufficient"`` +
a WARN log, never a bare ``{}``.

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
from typing import Any

from nousergon_lib.quant.horizons import DEFAULT_POLICY

from evals import outcome_store

logger = logging.getLogger(__name__)

# Fixed S3 key the consumer reads (archive/manager.py::load_team_accuracy).
# Not the dated/latest eval-artifacts partition pattern — the consumer
# contract predates that convention and reads this single well-known key.
TEAM_ACCURACY_S3_KEY = "config/team_accuracy.json"

# Envelope schema version (config#1844). Bump only on breaking shape changes;
# additive fields ride the same version per the S3 contract-safety rule.
SCHEMA_VERSION = 1

# Envelope keys save_team_accuracy requires — the structural guard that a
# bare `{}` (or any pre-envelope shape) can never reach S3 again.
_REQUIRED_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "as_of",
        "n_teams",
        "n_advance_picks",
        "n_resolved_outcomes",
        "horizon_days",
        "teams",
    }
)

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
) -> dict[str, Any]:
    """Compute each team's realized 21d-horizon hit rate from research.db.

    ``as_of_date`` is the Saturday this analysis is being built FOR — i.e.
    next cycle's run date. The lookback window ends one day before
    ``as_of_date`` so the current cycle's own (unresolved) picks can't leak
    in, mirroring ``last_week_scorecard.build_scorecard``.

    Returns the schema_version-1 envelope (see module docstring). ``teams``
    carries ``{team_id: {"accuracy": float, "n_obs": int}}`` for every team
    with at least one resolved observation in the window. Teams with zero
    resolved observations are omitted (not zero-filled) so the consumer's
    "team absent from the map" graceful-degrade path — already required by
    ``_accuracy_adjustment`` — is exercised rather than a fabricated 0.0
    accuracy that would look like a real bottom-percentile signal.

    Zero-data windows return ``status="insufficient"`` with the counts that
    explain WHY (``n_advance_picks`` vs ``n_resolved_outcomes``) plus a WARN
    log — never a bare ``{}`` (config#1844, fail-loud doctrine).
    """
    window_end = as_of_date - timedelta(days=1)
    window_start = window_end - timedelta(weeks=lookback_weeks)

    rows, n_advance_picks = _fetch_team_outcomes(
        conn, window_start.isoformat(), window_end.isoformat()
    )

    by_team: dict[str, list[int]] = {}
    for r in rows:
        by_team.setdefault(r["team_id"], []).append(r["beat_spy"])

    teams: dict[str, dict[str, Any]] = {}
    for team_id, outcomes in sorted(by_team.items()):
        n_obs = len(outcomes)
        accuracy = sum(outcomes) / n_obs
        teams[team_id] = {"accuracy": accuracy, "n_obs": n_obs}

    n_resolved_outcomes = len(rows)
    status = "ok" if teams else "insufficient"
    if status == "insufficient":
        logger.warning(
            "team_accuracy: zero resolved observations in window %s..%s — "
            "emitting status=insufficient (n_advance_picks=%d, "
            "n_resolved_outcomes=%d, horizon_days=%d). Either no ADVANCE "
            "picks have reached the primary horizon yet, or the "
            "score_performance_outcomes store hasn't been populated for "
            "this window (it is written by DataPhase2 AFTER this producer "
            "runs — see config#1844).",
            window_start.isoformat(),
            window_end.isoformat(),
            n_advance_picks,
            n_resolved_outcomes,
            DEFAULT_POLICY.primary_horizon,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "as_of": as_of_date.isoformat(),
        "n_teams": len(teams),
        "n_advance_picks": n_advance_picks,
        "n_resolved_outcomes": n_resolved_outcomes,
        "horizon_days": DEFAULT_POLICY.primary_horizon,
        "teams": teams,
    }


def _fetch_team_outcomes(
    conn: sqlite3.Connection, start: str, end: str
) -> tuple[list[dict], int]:
    """Pull per-pick realized 21d beat-SPY outcomes, attributed to team_id.

    Joins CIO-ADVANCED picks (``cio_evaluations``, has ``team_id``, no
    outcome) against realized canonical-primary-horizon (21d) outcomes from
    the long-format ``score_performance_outcomes`` store (via
    ``evals.outcome_store`` — config#1483/config#1530 cutover, replaces the
    retired wide horizon-suffixed score_performance column read) on
    ``(ticker=symbol, eval_date=score_date)`` — both are written from the
    same research cycle for the same date, so the pair keys align. Only
    resolved rows and only ``team_id IS NOT NULL`` rows (defensive — CIO
    evaluations for exit-only / non-team-sourced candidates may carry a null
    team) count.

    Returns ``(rows, n_advance_picks)``: ``rows`` are dicts keyed
    ``team_id``/``beat_spy`` (the long store's field name, NOT the retired
    wide column name) for picks WITH a resolved outcome; ``n_advance_picks``
    is the total ADVANCE-pick count in the window regardless of resolution,
    so the envelope can state how much of the input side survived the join
    (config#1844).
    """
    sql = """
        SELECT
            c.team_id,
            c.ticker,
            c.eval_date
        FROM cio_evaluations c
        WHERE c.eval_date BETWEEN ? AND ?
          AND c.cio_decision = 'ADVANCE'
          AND c.team_id IS NOT NULL
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    outcomes = outcome_store.load_primary_outcomes(conn, start, end)
    result = []
    for team_id, ticker, eval_date in rows:
        outcome = outcomes.get((ticker, eval_date))
        if outcome is None or outcome.beat_spy is None:
            continue
        result.append({"team_id": team_id, "beat_spy": outcome.beat_spy})
    return result, len(rows)


def save_team_accuracy(
    team_accuracy: dict[str, dict[str, Any]],
    *,
    s3_client: Any,
    bucket: str,
    key: str = TEAM_ACCURACY_S3_KEY,
) -> None:
    """Write the ``team_accuracy`` envelope to the fixed S3 key the consumer reads.

    Single-key overwrite (like ``population/latest.json``), not a dated +
    latest sidecar pair — ``load_team_accuracy`` only ever reads ``key``
    directly, so there's no dated-history reader to serve.

    Refuses (ValueError) any payload that is not a complete schema_version-1
    envelope — the structural chokepoint that guarantees a bare ``{}`` (the
    config#1844 defect) can never reach S3 again, whatever the caller does.

    Per [[feedback_no_silent_fails]] this raises on any S3 failure — same
    posture as ``emit_scorecard_to_s3``. The caller (Lambda handler) is
    responsible for the shadow-mode WARN-and-continue wrapper so a producer
    failure here never blocks the Saturday morning briefing.
    """
    if not bucket:
        raise ValueError("save_team_accuracy requires a non-empty bucket")
    if not isinstance(team_accuracy, dict) or not _REQUIRED_ENVELOPE_KEYS.issubset(
        team_accuracy
    ):
        missing = _REQUIRED_ENVELOPE_KEYS - set(
            team_accuracy if isinstance(team_accuracy, dict) else ()
        )
        raise ValueError(
            "save_team_accuracy requires the schema_version-1 envelope "
            f"(config#1844) — missing keys: {sorted(missing)}. A bare "
            "per-team dict (pre-envelope shape) must never be written."
        )
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


def main(argv: list[str] | None = None) -> int:
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
