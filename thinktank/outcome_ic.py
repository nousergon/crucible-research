"""thinktank.outcome_ic — Think Tank rating → realized 21d alpha validation
(config#2467: "validate ratings predict 21d alpha before gating the
Predictor universe on Think Tank output").

WHAT THIS ANSWERS
------------------
``thinktank/ratings.py`` writes an independent 0-100 analyst rating per
covered ticker (``RatingRow.rating``) that deliberately never sees the
scanner's ``attractiveness_score`` (see ``thinktank/analyst.py``). Before
anything downstream (Predictor universe, Executor) is gated on that rating,
this module asks the same question ``evals/judge_outcome_ic.py`` asks of the
judge layer: does the rating cross-sectionally RANK-PREDICT the realized
canonical-primary-horizon (21 trading day, log-domain, SPY-relative) forward
return?

OBSERVATION ONLY — per the issue this module gates nothing. It is a
diagnostic artifact for the human operator, mirroring the anti-Goodhart
stance in ``evals/judge_outcome_ic.py``: computing this is the deliverable,
not a precondition check embedded in any Predictor/Executor code path.

WHY NOT ``evals.outcome_store``
--------------------------------
The repo's M0-disciplined single accessor over resolved outcomes
(``evals.outcome_store.load_primary_outcomes``) reads
``score_performance_outcomes``, which is populated from scanner BUY-signal
attribution (``nousergon-data/collectors/signal_returns.py`` long-format
write) — verified against the live ``research.db`` snapshot (2026-07-14) to
have essentially zero rows keyed by Think Tank tickers/dates (Think Tank
covers a broader, LLM-selected universe than the scanner's signal stream).
Reusing that accessor would silently starve on Think Tank input, exactly the
failure mode ``nousergon_lib.quant.horizons`` exists to prevent. Instead
this module reads the WIDE ``universe_returns`` table directly (same table
``nousergon-data/collectors/signal_returns.py`` populates pre-long-format-
migration) for the two columns needed: ``log_return_21d`` and
``log_spy_return_21d``, joined by ``(ticker, eval_date)`` where
``eval_date`` is the rating's ``thesis_trading_day`` (Think Tank ratings are
already trading-day-stamped at write time — see ``thinktank/run.py`` — so no
capture-date → trading-day remapping is needed here, unlike
``evals/judge_outcome_ic.py``'s judge-artifact join).

ALPHA DEFINITION CAVEAT: ``universe_returns.log_return_21d -
log_spy_return_21d`` is log-domain, SPY-relative alpha but is NOT
sector-neutralized (unlike the ``LabelDefinition(neutralization="sector")``
canonical label ``nousergon_lib.quant.horizons`` documents as the fleet
target, and unlike ``score_performance_outcomes.log_alpha``'s provenance).
This is the best available realized-outcome column for Think Tank's universe
today; the gap is surfaced in the emitted block (``label_note``) rather than
silently presented as the canonical label.

STATISTICS — reuses the ONE engine
------------------------------------
Same date-clustered Spearman IC machinery as ``scoring/leaderboard_scoring``
and ``evals/judge_outcome_ic``: per-rating-date cross-sectional Spearman IC
of rating vs realized alpha, then date-clustered significance (each date is
one cluster). The Student-t two-sided p-value reuses
``evals.judge_outcome_ic.student_t_two_sided_p`` (pure-stdlib, no scipy in
the Lambda layout) rather than a second implementation.

HONEST SMALL-N — this issue's own stated caveat
--------------------------------------------------
The issue text names the sample-size risk directly ("~55 names ... small
sample"). Same floors as ``evals/judge_outcome_ic``: a date contributes only
with >= ``MIN_PAIRS_PER_DATE`` joined pairs; clustered significance requires
>= ``MIN_EVAL_DATES`` contributing dates. Below the floor, or when the
21-trading-day-forward window for a rating's as-of date has not yet closed
in the source data, the block ships ``status="insufficient"`` with real
counts and null metrics — never a fabricated IC.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from typing import Any, Mapping, Optional

from evals.judge_outcome_ic import _pooled_spearman_p, student_t_two_sided_p
from nousergon_lib.quant.horizons import DEFAULT_POLICY, HorizonPolicy
from scoring.leaderboard_scoring import date_clustered_stats, spearman_ic

logger = logging.getLogger(__name__)

# Block schema version — mirrors evals/judge_outcome_ic.SCHEMA_VERSION's
# additive-only-changes discipline should this become a persisted artifact.
SCHEMA_VERSION = 1

# Same floors evals/judge_outcome_ic + scoring/leaderboard_scoring use.
MIN_EVAL_DATES = 2
MIN_PAIRS_PER_DATE = 2

RATINGS_PREFIX = "thinktank/ratings/"
RESEARCH_DB_S3_KEY = "research.db"

# Realized-alpha caveat surfaced in every emitted block (see module docstring).
_LABEL_NOTE = (
    "universe_returns.log_return_21d - log_spy_return_21d: log-domain, "
    "SPY-relative, NOT sector-neutralized (differs from the canonical "
    "LabelDefinition in nousergon_lib.quant.horizons)."
)


# ── Pure compute ────────────────────────────────────────────────────────────


def _null_overall() -> dict[str, Any]:
    return {
        "date_ic_mean": None, "date_ic_t": None, "date_ic_p": None,
        "n_rating_dates": 0, "pooled_ic": None, "pooled_ic_p": None, "n": 0,
    }


def compute_think_tank_outcome_ic(
    ratings: Mapping[tuple[str, str], float],
    realized: Mapping[tuple[str, str], float],
    *,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Compute the Think Tank rating -> realized 21d alpha IC block. PURE —
    no I/O.

    Args:
        ratings: ``{(ticker, thesis_trading_day): rating}`` — one entry per
            distinct rating EVENT (a ticker re-rated on a later trading day
            contributes a second, independent pair; matches how
            ``evals/judge_outcome_ic`` treats repeat evals of the same
            ticker on different eval dates).
        realized: ``{(ticker, eval_date): log_alpha}`` — decimal log-domain,
            SPY-relative 21d forward return (see module docstring for the
            sector-neutralization caveat), already restricted to resolved
            (non-NULL) rows by the caller.
        policy: active ``HorizonPolicy`` — carried through for the emitted
            ``horizon_days`` field only; the realized-alpha join itself is
            fixed to ``universe_returns``'s ``_21d`` columns (see module
            docstring on why this bypasses ``outcome_store``).

    Returns the block shape::

        {
          "schema_version": 1,
          "status": "ok" | "insufficient",
          "horizon_days": <policy.primary_horizon>,
          "label_note": <str>,
          "overall": {"date_ic_mean", "date_ic_t", "date_ic_p",
                      "n_rating_dates", "pooled_ic", "pooled_ic_p", "n"},
          "n_ratings_total": <int>,
          "n_unresolved": <int>,
        }
    """
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for (ticker, rating_date), rating in ratings.items():
        by_date[rating_date][ticker] = float(rating)

    per_date_ics: list[float] = []
    pooled_pairs: list[tuple[float, float]] = []
    for rating_date in sorted(by_date):
        scores = by_date[rating_date]
        paired = [
            (s, realized[(t, rating_date)])
            for t, s in scores.items()
            if (t, rating_date) in realized
        ]
        if len(paired) < MIN_PAIRS_PER_DATE:
            continue
        ic = spearman_ic([p[0] for p in paired], [p[1] for p in paired])
        if ic is not None:
            per_date_ics.append(ic)
        pooled_pairs.extend(paired)

    clustered = date_clustered_stats(per_date_ics)
    n_pooled = len(pooled_pairs)
    pooled_ic = (
        spearman_ic([p[0] for p in pooled_pairs], [p[1] for p in pooled_pairs])
        if n_pooled >= 2 else None
    )

    overall = _null_overall()
    overall["n"] = n_pooled
    if clustered is not None:
        overall["date_ic_mean"] = clustered["mean"]
        overall["date_ic_t"] = clustered["t_stat"]
        overall["n_rating_dates"] = clustered["n_dates"]
        if clustered["t_stat"] is not None:
            overall["date_ic_p"] = round(
                student_t_two_sided_p(clustered["t_stat"], clustered["n_dates"] - 1), 6,
            )
    if pooled_ic is not None:
        overall["pooled_ic"] = round(pooled_ic, 6)
        p = _pooled_spearman_p(pooled_ic, n_pooled)
        overall["pooled_ic_p"] = round(p, 6) if p is not None else None

    n_unresolved = len(ratings) - sum(
        1 for key in ratings if key in realized
    )
    status = (
        "ok"
        if overall["n_rating_dates"] >= MIN_EVAL_DATES
        and overall["date_ic_t"] is not None
        and overall["pooled_ic"] is not None
        else "insufficient"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "horizon_days": policy.primary_horizon,
        "label_note": _LABEL_NOTE,
        "overall": overall,
        "n_ratings_total": len(ratings),
        "n_unresolved": n_unresolved,
    }


# ── I/O + orchestration ─────────────────────────────────────────────────────


def load_rating_events(
    s3: Any, bucket: str, prefix: str = RATINGS_PREFIX,
) -> dict[tuple[str, str], float]:
    """Every distinct (ticker, thesis_trading_day) rating EVENT across all
    dated ``thinktank/ratings/{trading_day}.json`` snapshots (NOT just
    ``latest.json`` — the board is upserted/pruned in place per
    ``thinktank/ratings.update_ratings_board``, so only the union of dated
    snapshots recovers the full rating history; a name re-rated on a later
    trading day contributes a second independent event, matching
    ``evals/judge_outcome_ic``'s per-eval-date treatment of repeat judge
    scores). ``latest.json`` is skipped by name (duplicate of the most
    recent dated snapshot, not new history).

    A rating row with ``rating is None`` (pre-rating-field legacy thesis,
    see ``RatingRow.rating``'s docstring) or an empty ``thesis_trading_day``
    is excluded — it carries no rating opinion to validate."""
    events: dict[tuple[str, str], float] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith(".json") or key.split("/")[-1] == "latest.json":
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            try:
                doc = json.loads(body)
            except ValueError as exc:
                logger.warning(
                    "[thinktank.outcome_ic] unparseable ratings board "
                    "s3://%s/%s skipped: %s", bucket, key, exc,
                )
                continue
            for ticker, row in (doc.get("rows") or {}).items():
                rating = row.get("rating")
                rating_date = row.get("thesis_trading_day")
                if rating is None or not rating_date:
                    continue
                events[(ticker, rating_date)] = float(rating)
    logger.info(
        "[thinktank.outcome_ic] loaded %d distinct rating events from "
        "s3://%s/%s", len(events), bucket, prefix,
    )
    return events


def open_research_db(s3: Any, bucket: str) -> sqlite3.Connection:
    """Download the research.db snapshot and open it (mirrors
    ``evals.judge_outcome_ic.open_research_db``'s pull path — same bucket
    root key, same fail-loud-on-missing posture: the snapshot always exists
    in production, so absence is a broken precondition)."""
    import os
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "thinktank_outcome_ic_research.db")
    s3.download_file(bucket, RESEARCH_DB_S3_KEY, tmp)
    logger.info(
        "[thinktank.outcome_ic] pulled s3://%s/%s for the outcome join",
        bucket, RESEARCH_DB_S3_KEY,
    )
    return sqlite3.connect(tmp)


def load_realized_alpha(
    conn: sqlite3.Connection, keys: Any,
) -> dict[tuple[str, str], float]:
    """Realized decimal log-alpha (``log_return_21d - log_spy_return_21d``)
    for exactly the ``(ticker, eval_date)`` pairs in ``keys``, restricted to
    rows where both source columns are resolved (non-NULL) — an unresolved
    row means the 21-trading-day-forward window for that rating's as-of
    date has not closed yet in ``universe_returns``, which is legitimate
    cohort maturation (see module docstring), never an error."""
    out: dict[tuple[str, str], float] = {}
    cur = conn.cursor()
    for ticker, eval_date in keys:
        cur.execute(
            "SELECT log_return_21d, log_spy_return_21d FROM universe_returns "
            "WHERE ticker = ? AND eval_date = ?",
            (ticker, eval_date),
        )
        row = cur.fetchone()
        if row is None:
            continue
        log_return_21d, log_spy_return_21d = row
        if log_return_21d is None or log_spy_return_21d is None:
            continue
        out[(ticker, eval_date)] = log_return_21d - log_spy_return_21d
    return out


def build_think_tank_outcome_ic_block(
    s3: Any,
    bucket: str,
    *,
    conn: Optional[sqlite3.Connection] = None,
    ratings_prefix: str = RATINGS_PREFIX,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Load -> join -> compute; returns the frozen block. ``conn`` is an
    open research.db connection (injected in tests); when None the S3
    snapshot is pulled via :func:`open_research_db`."""
    ratings = load_rating_events(s3, bucket, prefix=ratings_prefix)
    owns_conn = conn is None
    if owns_conn:
        conn = open_research_db(s3, bucket)
    try:
        realized = load_realized_alpha(conn, ratings.keys())
    finally:
        if owns_conn:
            conn.close()
    return compute_think_tank_outcome_ic(ratings, realized, policy=policy)
