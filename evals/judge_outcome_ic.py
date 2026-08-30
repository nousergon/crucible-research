"""evals.judge_outcome_ic — judge-score → realized-outcome validation
(old ROADMAP L480 re-scope; see the migration note at archive/schema.py
migration 18: "Powers the judge outcome-IC validation").

WHAT THIS ANSWERS
-----------------
The LLM-as-judge layer (``evals/judge.py``) rubric-scores agent outputs,
but until this module there was ZERO linkage between those rubric scores
and realized outcomes — we could not tell whether a high judge score
PREDICTS a good pick. This module correlates per-ticker judge rubric
scores against realized canonical-primary-horizon (21d) market-relative
outcomes (decimal log-alpha) and emits the result as the
``judge_outcome_ic`` block of ``backtest/{date}/agent_quality.json``
(producer: ``scripts/build_agent_quality.py``).

OBSERVABILITY ONLY — the anti-Goodhart stance is deliberate and preserved
--------------------------------------------------------------------------
Two standing invariants in this package govern what this module may NOT do:

* ``evals/judge.py``: "Eval is observability, NOT a gate. Runs proceed
  regardless of eval score."
* ``evals/last_week_scorecard.py``: "Goodhart on rubric outputs: only
  REALIZED outcome data here; Sonnet rubric scores from ``evals/judge.py``
  are intentionally NOT in the scorecard. Feeding rubric back creates a
  Goodhart loop."

This module keeps both: it VALIDATES the judge against outcomes as a
diagnostic artifact for the human operator + report card, it gates
nothing, and judge scores stay excluded from the agent-facing scorecard.
Do not wire this block into any prompt, gate, threshold, or auto-config
loop — the moment agents (or the judge) are optimized against it, the
measurement is destroyed.

ATTRIBUTABILITY — how a RubricEvalArtifact maps to (ticker, eval_date)
----------------------------------------------------------------------
Only per-ticker judge evals can be joined to a per-ticker outcome. The
mapping (see ``evals/judge.resolve_rubric_for_agent`` for the agent_id
taxonomy):

* **ticker** — from ``judged_agent_id`` for the ``thesis_update:{team}:
  {ticker}`` family, the ONLY agent_id family that carries a ticker
  (sector/macro/CIO agents score team- or slate-level outputs; the
  think-tank family is deliberately coarse per config#1579 and carries
  ticker identity only in run_id + snapshot, which is not a stable
  contract to parse). Everything else is counted as unattributable —
  no silent drops (WARN + ``n_unattributable`` in the emitted block).
* **eval_date** — the DecisionArtifact CAPTURE date (authoritative date
  in the ``decision_artifacts/{Y}/{M}/{D}/`` prefix of
  ``judged_artifact_s3_key``; judge wall-clock is NOT it — the judge can
  run hours/days after capture, see ``evals/judge._capture_date_from_s3_key``)
  mapped to the trading day via ``nousergon_lib.dates.expected_last_close``.
  This reproduces the system-wide stamping rule (``lambda/handler.py``:
  "every eval_date ... anchors to 'most recent trading day with data
  available at run time'"), so a Saturday-captured artifact joins the
  Friday-stamped ``score_date`` its research cycle actually wrote.

Skip-marker evals (``judge_skip_reason`` set / empty ``dimension_scores``)
carry no judge opinion and are excluded before attribution (mirroring
``build_agent_quality``'s "real evals" filter); they are logged but not
counted as unattributable.

OUTCOME JOIN
------------
Realized outcomes come from this repo's single accessor over the
long-format store, ``evals.outcome_store.load_primary_outcomes``
(M0 contract discipline — never a second reader), keyed
``(symbol, score_date)`` at the canonical primary horizon from
``nousergon_lib.quant.horizons`` (a PARAMETER — never hardcoded 21, per
the config#1456 lesson). Alpha is the store's ``log_alpha``: decimal
log-domain market-relative return.

STATISTICS
----------
Reuses the date-clustered IC machinery in ``scoring/leaderboard_scoring``
(``spearman_ic`` + ``date_clustered_stats``) — one engine, not a parallel
reimplementation. Per eval date, the cross-sectional Spearman rank-IC of
judge score vs realized log-alpha; significance is date-clustered
(weeks-as-N), never the pooled n that double-counts within-week
correlation. The pooled IC across all pairs is reported alongside as a
descriptive companion.

p-values: the frozen cross-repo block schema carries ``date_ic_p`` /
``pooled_ic_p``. ``date_clustered_stats`` deliberately returns no p (no
scipy in the Lambda layout — this module runs inside the weekly
``eval_rolling_mean`` Lambda via ``build_agent_quality``). The two-sided
Student-t p is therefore computed here with an exact pure-stdlib
regularized-incomplete-beta implementation (validated against scipy
references in tests): the same t-approximation ``scipy.stats.spearmanr``
uses for the pooled p, and the plain t(df=n_dates-1) transform of the
clustered t-stat. Explicit deviation note per the SOTA rule: the SOTA
shortcut skipped is "just import scipy"; it is skipped because the
Lambda layout excludes scipy by standing convention (see
``scoring/leaderboard_scoring.py`` + ``scoring/attractiveness_trajectory.py``),
and the cost of the pure implementation is ~40 lines of a
textbook-standard special function, pinned by tests.

HONEST SMALL-N
--------------
Same floors the leaderboard engine uses: a date contributes only when it
has >= 2 joined (ticker, alpha) pairs (``spearman_ic``'s floor) and the
clustered SE/t/p exist only from >= 2 contributing eval dates
(``date_clustered_stats``'s floor). Below that the block ships
``status="insufficient"`` with the observed counts and null metrics —
an honest "not yet scorable", never a fabricated value.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from nousergon_lib.dates import expected_last_close
from nousergon_lib.quant.horizons import DEFAULT_POLICY, HorizonPolicy

from evals import outcome_store
from scoring.leaderboard_scoring import date_clustered_stats, spearman_ic

logger = logging.getLogger(__name__)

# Frozen cross-repo block schema version (the crucible-evaluator consumer
# is built against exactly this shape — additive changes only, bump on
# any breaking change, per S3 Contract Safety).
SCHEMA_VERSION = 1

# Minimum contributing eval dates for a defined clustered SE/t/p — the
# ``date_clustered_stats`` floor (n=1 yields mean with se/t None).
MIN_EVAL_DATES = 2

# Minimum joined (ticker, alpha) pairs for one date to contribute an IC —
# ``spearman_ic``'s own floor (returns None below 2 pairs).
MIN_PAIRS_PER_DATE = 2

# S3 key of the research SQLite snapshot at the bucket root (see the
# CLAUDE.md S3 layout + scripts/decision_review.py's _DB_S3_KEY).
RESEARCH_DB_S3_KEY = "research.db"

# agent_id family that carries a per-ticker identity (see
# evals/judge.resolve_rubric_for_agent's taxonomy).
_TICKER_BEARING_PREFIX = "thesis_update:"


# ── Eval-scan budget (alpha-engine-config-I9205) ───────────────────────────
#
# ``load_eval_artifacts`` walks the whole ``decision_artifacts/_eval/``
# prefix and issues one ``GetObject`` per artifact. Measured live
# 2026-08-28: 4,679 artifacts / 26,058,127 bytes, sequential ``get_object``
# at ~117.8 ms each → ~551 s for ONE scan. ``build_agent_quality`` ran the
# scan TWICE, so the block consumed ~1,102 s against a 240 s handler
# ceiling and timed out on all three of its first real executions. The
# driver is REQUEST LATENCY, not bytes (26 MB) and not memory (peak 519 MB
# of 1024 MB).
#
# Three bounds, all live on the Lambda handler path and none of them
# disable-able by a caller:
#
# 1. WINDOW — objects are pre-filtered on the listing's ``LastModified``
#    before any ``GetObject`` is issued. The join key is each artifact's
#    DecisionArtifact CAPTURE date, which is always <= the moment the judge
#    WROTE the eval, so an upper-bounded ``LastModified`` filter is a strict
#    SUPERSET of the capture window it stands for: it can never drop an
#    artifact the IC would have joined. 400 days ≈ a 12-month capture window
#    plus a month of judge lag.
# 2. CEILING — a hard object-count assertion that RAISES rather than
#    proceeding. A 240 s timeout is unattributable; ``EvalScanBudgetExceeded``
#    names the number that blew the budget.
# 3. CONCURRENCY — the residual GETs run on a bounded thread pool, which is
#    what turns ~551 s of serialized round-trips into ~20 s of wall clock.
#
# Both knobs resolve from these module constants at CALL time (the public
# parameters default to ``None``), so no call site can pin a stale value and
# a test can lower the ceiling to prove the bound is live.
DEFAULT_EVAL_LOOKBACK_DAYS = 400

# Hard ceiling on the number of eval artifacts one scan may fetch. ~4.7k
# today, accreting ~750/week; 25k is ~7 months of headroom at that rate and
# ~35 s of wall clock at EVAL_SCAN_MAX_WORKERS. Breaching it is a real
# signal (the judge's emission rate changed, or the window regressed), not a
# reason to fetch anyway.
MAX_EVAL_OBJECTS = 25_000

# Bounded fan-out for the residual GETs. botocore clients are thread-safe
# for concurrent API calls; 32 keeps us far below the per-prefix S3 request
# rate and below the Lambda's socket budget.
EVAL_SCAN_MAX_WORKERS = 32


class EvalScanBudgetExceeded(RuntimeError):
    """The ``_eval/`` scan selected more objects than its declared ceiling.

    Raised BEFORE any ``GetObject`` is issued, so it costs nothing and its
    message carries the numbers needed to re-set the budget. Deliberately a
    hard failure: silently fetching an unbounded prefix on a Lambda handler
    path is what produced the ``agent_quality`` timeout (I9205)."""


# ── Pure-stdlib Student-t two-sided p ──────────────────────────────────────


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified
    Lentz's method, the textbook-standard evaluation). Converges in a
    handful of iterations for the (a, b, x) ranges a t-distribution CDF
    produces."""
    max_iter, eps, fpmin = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(
        f"_betacf failed to converge for a={a}, b={b}, x={x} — "
        f"out-of-domain input reached the p-value transform."
    )


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) — regularized incomplete beta, pure stdlib."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_stat: float, df: float) -> float:
    """Two-sided p-value of ``t_stat`` under Student-t with ``df`` degrees
    of freedom: ``p = I_x(df/2, 1/2)`` with ``x = df / (df + t²)``.
    Matches ``2 * scipy.stats.t.sf(|t|, df)`` to double precision
    (pinned by tests)."""
    if df <= 0:
        raise ValueError(f"student_t_two_sided_p requires df > 0, got {df}")
    x = df / (df + t_stat * t_stat)
    return _regularized_incomplete_beta(df / 2.0, 0.5, x)


def _pooled_spearman_p(ic: float, n: int) -> float | None:
    """Two-sided p for a pooled Spearman IC via the standard large-sample
    t-approximation ``t = r·sqrt((n-2)/(1-r²))``, df = n-2 — the same
    default ``scipy.stats.spearmanr`` uses. None when undefined (n < 3);
    0.0 at |r| = 1 (the t diverges)."""
    if n < 3:
        return None
    if abs(ic) >= 1.0:
        return 0.0
    t_stat = ic * math.sqrt((n - 2) / (1.0 - ic * ic))
    return student_t_two_sided_p(t_stat, n - 2)


# ── Attribution ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttributedEval:
    """One real (non-skip) judge eval resolved to a (ticker, eval_date)
    pair, carrying its per-dimension rubric scores."""

    ticker: str
    eval_date: str  # trading-day ISO date — the score_date join key
    dimension_scores: dict[str, float]


@dataclass(frozen=True)
class AttributionResult:
    attributed: list[AttributedEval]
    n_unattributable: int
    n_skip_markers: int


def _ticker_from_agent_id(judged_agent_id: str) -> str | None:
    """Ticker for a ``thesis_update:{team}:{ticker}`` agent_id; None for
    every other family (they score team-/slate-level outputs)."""
    if not judged_agent_id.startswith(_TICKER_BEARING_PREFIX):
        return None
    parts = judged_agent_id.split(":")
    if len(parts) != 3 or not parts[2]:
        return None
    return parts[2]


def _trading_day_for_capture(capture_date_iso: str) -> str:
    """Map a capture CALENDAR date to the trading day the research cycle
    stamped everywhere (``most_recent_trading_day`` in lambda/handler.py):
    Saturday capture → Friday score_date. ``expected_last_close`` is the
    lib-canonical form of that rule."""
    return expected_last_close(capture_date_iso).isoformat()


def attribute_evals(evals: list[dict]) -> AttributionResult:
    """Resolve persisted RubricEvalArtifact dicts to (ticker, eval_date)
    pairs.

    Skip-marker evals (``judge_skip_reason`` set or empty
    ``dimension_scores``) carry no judge opinion and are excluded before
    attribution. Of the REAL evals, one is attributable iff (a) its
    ``judged_agent_id`` carries a ticker (thesis_update family) and (b)
    its ``judged_artifact_s3_key`` yields a capture date. The remainder
    is counted and WARNed — never silently dropped."""
    # Deferred import: evals.judge transitively imports the LangChain/
    # Anthropic stack + config; defer so pure-compute callers and tests of
    # this module's statistics don't pay (or require) that import chain.
    from evals.judge import _capture_date_from_s3_key

    attributed: list[AttributedEval] = []
    n_unattributable = 0
    n_skip = 0
    for doc in evals:
        if doc.get("judge_skip_reason") or not (doc.get("dimension_scores") or []):
            n_skip += 1
            continue
        ticker = _ticker_from_agent_id(str(doc.get("judged_agent_id") or ""))
        capture_date = _capture_date_from_s3_key(doc.get("judged_artifact_s3_key"))
        if ticker is None or capture_date is None:
            n_unattributable += 1
            continue
        dims: dict[str, list[float]] = defaultdict(list)
        for entry in doc["dimension_scores"]:
            name = entry.get("dimension")
            score = entry.get("score")
            if name is None or score is None:
                continue
            dims[str(name)].append(float(score))
        if not dims:
            n_unattributable += 1
            continue
        attributed.append(
            AttributedEval(
                ticker=ticker,
                eval_date=_trading_day_for_capture(capture_date),
                dimension_scores={k: sum(v) / len(v) for k, v in dims.items()},
            )
        )
    if n_unattributable:
        logger.warning(
            "[judge_outcome_ic] %d real eval artifact(s) not attributable to a "
            "(ticker, eval_date) pair (non-thesis_update agent_ids or missing/"
            "unparseable judged_artifact_s3_key) — reported as n_unattributable, "
            "excluded from the IC join.",
            n_unattributable,
        )
    logger.info(
        "[judge_outcome_ic] attribution: %d attributable, %d unattributable, "
        "%d skip-markers excluded (of %d artifacts).",
        len(attributed), n_unattributable, n_skip, len(evals),
    )
    return AttributionResult(attributed, n_unattributable, n_skip)


# ── Core computation (pure — no I/O) ───────────────────────────────────────


def _null_overall() -> dict[str, Any]:
    return {
        "date_ic_mean": None, "date_ic_t": None, "date_ic_p": None,
        "n_eval_dates": 0, "pooled_ic": None, "pooled_ic_p": None, "n": 0,
    }


def _per_date_ics(
    by_date: Mapping[str, dict[str, float]],
    realized: Mapping[tuple[str, str], float],
) -> list[float]:
    """Per-eval-date cross-sectional Spearman ICs of score vs realized
    log-alpha, honoring the >= MIN_PAIRS_PER_DATE floor per date."""
    ics: list[float] = []
    for eval_date in sorted(by_date):
        scores = by_date[eval_date]
        paired = [
            (s, realized[(t, eval_date)])
            for t, s in scores.items()
            if (t, eval_date) in realized
        ]
        if len(paired) < MIN_PAIRS_PER_DATE:
            continue
        ic = spearman_ic([p[0] for p in paired], [p[1] for p in paired])
        if ic is not None:
            ics.append(ic)
    return ics


def compute_judge_outcome_ic(
    attribution: AttributionResult,
    outcomes: Mapping[tuple[str, str], outcome_store.PrimaryOutcome],
    *,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Compute the frozen ``judge_outcome_ic`` block. PURE — no I/O.

    Returns the FROZEN cross-repo shape (schema_version 1)::

        {
          "schema_version": 1,
          "status": "ok" | "insufficient",
          "horizon_days": <policy.primary_horizon>,
          "overall": {"date_ic_mean", "date_ic_t", "date_ic_p",
                      "n_eval_dates", "pooled_ic", "pooled_ic_p", "n"},
          "by_dimension": {<dimension>: {"date_ic_mean", "date_ic_p",
                                         "n_eval_dates"}},
          "n_unattributable": <int>,
        }

    Alpha units: decimal log-alpha (the long store's canonical unit).
    ``overall`` uses the per-(ticker, date) MEAN rubric score across all
    dimensions and all real evals of the pair (Haiku- and Sonnet-tier
    evals of the same artifact average together rather than double-count);
    ``by_dimension`` repeats the date-clustered computation per rubric
    dimension. ``status="insufficient"`` (with null metrics, keys always
    present) below the floors documented in the module docstring; the
    consumer must treat any status != "ok" as not-gradeable."""
    # Aggregate rubric scores per (ticker, eval_date): overall mean +
    # per-dimension means across every real eval of the pair.
    overall_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    dim_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for ev in attribution.attributed:
        all_dims = list(ev.dimension_scores.values())
        overall_scores[ev.eval_date][ev.ticker].append(sum(all_dims) / len(all_dims))
        for dim, score in ev.dimension_scores.items():
            dim_scores[dim][ev.eval_date][ev.ticker].append(score)

    # Realized decimal log-alpha per joined (ticker, eval_date) pair —
    # unresolved outcomes (row absent, or log_alpha still NULL because the
    # forward window hasn't closed) simply don't join yet: legitimate
    # cohort maturation, not an error (see outcome_store's graceful-empty
    # rationale).
    realized: dict[tuple[str, str], float] = {
        key: o.log_alpha for key, o in outcomes.items() if o.log_alpha is not None
    }

    overall_by_date = {
        d: {t: sum(v) / len(v) for t, v in tickers.items()}
        for d, tickers in overall_scores.items()
    }
    overall_ics = _per_date_ics(overall_by_date, realized)
    clustered = date_clustered_stats(overall_ics)

    pooled_pairs = [
        (score, realized[(t, d)])
        for d, tickers in overall_by_date.items()
        for t, score in tickers.items()
        if (t, d) in realized
    ]
    n_pooled = len(pooled_pairs)
    pooled_ic = spearman_ic(
        [p[0] for p in pooled_pairs], [p[1] for p in pooled_pairs],
    ) if n_pooled >= 2 else None

    overall: dict[str, Any] = _null_overall()
    overall["n"] = n_pooled
    if clustered is not None:
        overall["date_ic_mean"] = clustered["mean"]
        overall["date_ic_t"] = clustered["t_stat"]
        overall["n_eval_dates"] = clustered["n_dates"]
        if clustered["t_stat"] is not None:
            overall["date_ic_p"] = round(
                student_t_two_sided_p(clustered["t_stat"], clustered["n_dates"] - 1), 6,
            )
    if pooled_ic is not None:
        overall["pooled_ic"] = round(pooled_ic, 6)
        p = _pooled_spearman_p(pooled_ic, n_pooled)
        overall["pooled_ic_p"] = round(p, 6) if p is not None else None

    by_dimension: dict[str, dict[str, Any]] = {}
    for dim in sorted(dim_scores):
        d_by_date = {
            d: {t: sum(v) / len(v) for t, v in tickers.items()}
            for d, tickers in dim_scores[dim].items()
        }
        d_stats = date_clustered_stats(_per_date_ics(d_by_date, realized))
        if d_stats is None:
            continue  # dimension never had a scoreable date — omitted
        d_p = (
            round(student_t_two_sided_p(d_stats["t_stat"], d_stats["n_dates"] - 1), 6)
            if d_stats["t_stat"] is not None else None
        )
        by_dimension[dim] = {
            "date_ic_mean": d_stats["mean"],
            "date_ic_p": d_p,
            "n_eval_dates": d_stats["n_dates"],
        }

    status = (
        "ok"
        if overall["n_eval_dates"] >= MIN_EVAL_DATES
        and overall["date_ic_t"] is not None
        and overall["pooled_ic"] is not None
        else "insufficient"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "horizon_days": policy.primary_horizon,
        "overall": overall,
        "by_dimension": by_dimension,
        "n_unattributable": attribution.n_unattributable,
    }


# ── I/O + orchestration ────────────────────────────────────────────────────


def load_eval_artifacts(
    s3: Any,
    bucket: str,
    prefix: str | None = None,
    *,
    lookback_days: int | None = None,
    max_objects: int | None = None,
    max_workers: int | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """Every persisted RubricEvalArtifact JSON under the ``_eval/`` prefix,
    windowed, ceilinged and fetched on a bounded thread pool.

    Scans both the canonical flat layout (``{prefix}{judge_run_id}_
    {basename}``, config#793) and the legacy nested ``{prefix}{date}/
    {judge_run_id}/{basename}`` partition, so the whole judged history in
    the window participates. The eval_date used for the outcome join comes
    from each artifact's ``judged_artifact_s3_key`` (layout-independent),
    never from the eval key itself. The ``latest.json`` operator sidecar is
    not an eval artifact and is skipped by name.

    Bounds (alpha-engine-config-I9205 — see the module constants above):

    * ``lookback_days`` — keep only objects whose ``LastModified`` is within
      the window. ``None`` → :data:`DEFAULT_EVAL_LOOKBACK_DAYS`; ``0``
      requests the full history explicitly (still ceilinged). Because a
      judge writes an eval at or after the artifact's capture, this filter
      is a strict superset of the capture window and cannot drop a joinable
      artifact.
    * ``max_objects`` — ``None`` → :data:`MAX_EVAL_OBJECTS`. Exceeding it
      raises :class:`EvalScanBudgetExceeded` before any ``GetObject``.
      There is no value that disables the ceiling.
    * ``max_workers`` — ``None`` → :data:`EVAL_SCAN_MAX_WORKERS`.

    Emits one INFO line per scan carrying the per-phase durations and the
    object/byte counters, so the next budget is set from measurement rather
    than from a comment (I9205 deliverable 6).
    """
    if prefix is None:
        from evals.judge import DEFAULT_EVAL_PREFIX  # deferred — see attribute_evals

        prefix = DEFAULT_EVAL_PREFIX

    lookback = DEFAULT_EVAL_LOOKBACK_DAYS if lookback_days is None else lookback_days
    ceiling = MAX_EVAL_OBJECTS if max_objects is None else max_objects
    workers = EVAL_SCAN_MAX_WORKERS if max_workers is None else max_workers
    cutoff = (
        ((now or datetime.now(UTC)) - timedelta(days=lookback)) if lookback else None
    )

    # ── list phase ────────────────────────────────────────────────────────
    t0 = time.monotonic()
    n_listed = 0
    n_windowed_out = 0
    n_undated = 0
    selected: list[str] = []
    selected_bytes = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith(".json") or key.split("/")[-1] == "latest.json":
                continue
            n_listed += 1
            if cutoff is not None:
                last_modified = obj.get("LastModified")
                if last_modified is None:
                    # A listing row without LastModified is a contract
                    # violation of the S3 API, so this branch should be
                    # unreachable against real S3. It degrades toward
                    # KEEPING the object (never dropping data the IC would
                    # have joined) and the failure mode — an object that
                    # escapes the window and inflates the scan — is recorded
                    # on the WARN below and in the ``n_undated`` counter of
                    # the scan's INFO line.
                    n_undated += 1
                elif last_modified < cutoff:
                    n_windowed_out += 1
                    continue
            selected.append(key)
            selected_bytes += int(obj.get("Size") or 0)
    list_s = time.monotonic() - t0

    if n_undated:
        logger.warning(
            "[judge_outcome_ic] %d eval object(s) under s3://%s/%s had no "
            "LastModified and bypassed the %s-day window (kept, never dropped)",
            n_undated, bucket, prefix, lookback,
        )

    # ── ceiling assertion — before a single GetObject is issued ───────────
    if len(selected) > ceiling:
        raise EvalScanBudgetExceeded(
            f"eval scan selected {len(selected)} objects under "
            f"s3://{bucket}/{prefix} (listed {n_listed}, window "
            f"{lookback or 'ALL'}d) — ceiling is {ceiling}. Refusing to "
            f"issue an unbounded prefix walk on a handler path; narrow the "
            f"window or re-set MAX_EVAL_OBJECTS from a measured scan."
        )

    # ── fetch phase — bounded fan-out ────────────────────────────────────
    t1 = time.monotonic()

    def _fetch(key: str) -> tuple[str, bytes]:
        return key, s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    artifacts: list[dict] = []
    n_unparseable = 0
    if selected:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(selected)))) as pool:
            for key, body in pool.map(_fetch, selected):
                try:
                    doc = json.loads(body)
                except ValueError as exc:
                    # A corrupt artifact is a real defect worth surfacing, but
                    # one bad file must not sink the whole history join —
                    # recorded here (WARN names the key) per no-silent-fails.
                    n_unparseable += 1
                    logger.warning(
                        "[judge_outcome_ic] unparseable eval artifact s3://%s/%s "
                        "skipped: %s", bucket, key, exc,
                    )
                    continue
                if isinstance(doc, dict):
                    artifacts.append(doc)
    fetch_s = time.monotonic() - t1

    logger.info(
        "[judge_outcome_ic] eval scan s3://%s/%s: listed=%d windowed_out=%d "
        "fetched=%d parsed=%d unparseable=%d bytes=%d workers=%d "
        "lookback_days=%s ceiling=%d list_s=%.2f fetch_s=%.2f total_s=%.2f",
        bucket, prefix, n_listed, n_windowed_out, len(selected), len(artifacts),
        n_unparseable, selected_bytes, max(1, min(workers, len(selected) or 1)),
        lookback or "ALL", ceiling, list_s, fetch_s, list_s + fetch_s,
    )
    return artifacts


def open_research_db(s3: Any, bucket: str) -> sqlite3.Connection:
    """Download the research.db snapshot from the bucket root to a temp
    file and open it (mirrors ``scripts/decision_review.open_db``'s pull
    path). Raises loudly on a missing/failed download — the snapshot
    always exists in production, so absence is a broken precondition,
    not an "insufficient data" state."""
    tmp = os.path.join(tempfile.gettempdir(), "judge_outcome_ic_research.db")
    s3.download_file(bucket, RESEARCH_DB_S3_KEY, tmp)
    logger.info(
        "[judge_outcome_ic] pulled s3://%s/%s for the outcome join",
        bucket, RESEARCH_DB_S3_KEY,
    )
    return sqlite3.connect(tmp)


def build_judge_outcome_ic_block(
    s3: Any,
    bucket: str,
    *,
    conn: sqlite3.Connection | None = None,
    eval_prefix: str | None = None,
    evals: list[dict] | None = None,
    policy: HorizonPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Load → attribute → join → compute; returns the frozen block.

    ``conn`` is an open research.db connection (injected in tests); when
    None the S3 snapshot is pulled via :func:`open_research_db`. Raises
    on genuinely broken preconditions (S3 listing failure, missing DB
    snapshot) — the caller decides the isolation posture; absent HISTORY
    (no evals, no attributable pairs, no resolved outcomes) is a
    legitimate ``status="insufficient"``, never an error.

    ``evals`` is an ALREADY-LOADED artifact list. ``scripts/build_agent_quality``
    needs the same list for its rubric metrics and used to load it a second
    time here, doubling a ~551 s prefix scan on the Lambda handler path
    (alpha-engine-config-I9205); it now loads once and injects. When ``evals``
    is None the function loads for itself, so every other caller is unchanged.
    Passing both ``evals`` and ``eval_prefix`` is a contradiction (the prefix
    could not have produced that list) and raises."""
    if evals is not None and eval_prefix is not None:
        raise ValueError(
            "build_judge_outcome_ic_block: pass evals= OR eval_prefix=, not "
            "both — an injected list cannot be re-derived from a prefix."
        )
    injected = evals is not None
    t0 = time.monotonic()
    if evals is None:
        evals = load_eval_artifacts(s3, bucket, prefix=eval_prefix)
    load_s = time.monotonic() - t0

    t1 = time.monotonic()
    attribution = attribute_evals(evals)
    attribute_s = time.monotonic() - t1

    t2 = time.monotonic()
    owns_conn = conn is None
    if owns_conn:
        conn = open_research_db(s3, bucket)
    db_pull_s = time.monotonic() - t2
    t3 = time.monotonic()
    try:
        outcomes = outcome_store.load_primary_outcomes(conn, policy=policy)
    finally:
        if owns_conn:
            conn.close()
    outcomes_s = time.monotonic() - t3

    t4 = time.monotonic()
    block = compute_judge_outcome_ic(attribution, outcomes, policy=policy)
    compute_s = time.monotonic() - t4

    logger.info(
        "[judge_outcome_ic] block built status=%s n_evals=%d injected=%s "
        "n_attributed=%d n_unattributable=%d n_outcomes=%d | load_s=%.2f "
        "attribute_s=%.2f db_pull_s=%.2f outcomes_s=%.2f compute_s=%.2f "
        "total_s=%.2f",
        block.get("status"), len(evals), injected, len(attribution.attributed),
        attribution.n_unattributable, len(outcomes), load_s, attribute_s,
        db_pull_s, outcomes_s, compute_s, time.monotonic() - t0,
    )
    return block
