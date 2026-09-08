"""Statistical control bands on the judge-score time series (L4578(e)).

The eval layer already emits a 4-week rolling mean per
``(judged_agent_id, criterion, judge_model)`` combo
(``evals/rolling_mean.py``) and alarms on a **flat floor** —
``agent_quality_score_4w_mean_min < 3.0``. A flat floor only catches
*absolute-low* scores. It is blind to the two regression shapes that
matter most for an LLM judge:

* a **sustained drift** — a combo sliding 4.8 → 4.3 → 3.8 → 3.4 is a
  real quality regression long before it crosses 3.0; the floor stays
  silent the whole way down.
* a **sudden step** — a one-week 4.6 → 3.3 drop is a process change
  worth paging on even though 3.3 > 3.0.

This module adds the two complementary SOTA control charts that catch
exactly those, operating on each combo's *own* history rather than a
global constant.

**The chart runs on the RAW WEEKLY score, not the 4-week rolling mean**
(Brian's ruling, 2026-09-08, ``alpha-engine-config-I10167``; the option
not taken was leaving the rolling-mean scale and accepting its
sensitivity). Consecutive points of a 4-week rolling mean sampled weekly
share ~75% of their underlying reviews, and an individuals chart
estimates sigma from the average moving range of *consecutive* points on
the assumption that they are independent. They were not, so sigma-hat was
deflated 2-5x -- measured 2026-09-08 across all 20 live combos, 0.028 to
0.155 on the charted mean series against 0.173 to 0.441 on the raw weekly
series of the same combos. 13 of 20 combos had a latest |z| > 3, one at
+14.45 on a bounded 1-5 rubric, and ``agent_quality_score_control_
breach_count`` read 13 on 2026-07-30. A 3-sigma individuals chart is
designed for a ~0.27% two-sided false-alarm rate; that one flagged 65% of
its population.

The two charts:

* **Shewhart individuals (X) chart** — center ± k·σ̂ᵢ. σ is estimated
  from the **average moving range** (σ̂ = MR-bar / d₂, d₂=1.128 for
  ranges of 2 consecutive points), NOT the raw sample stdev: the MR
  estimator measures short-term within-process variation and is robust
  to the sustained shifts we are trying to detect (a raw stdev would be
  inflated by the shift itself and hide it). Catches large sudden steps.

  The limits are **per-observation**, because the observations are not
  exchangeable. Each weekly point is the mean of nᵢ judged reviews and
  nᵢ swings from 2 to 2408 with whole weeks missing
  (``alpha-engine-config-I10169``). A week's mean therefore has variance

      Var(xᵢ) = σ_b² + σ_w²/nᵢ

  where σ_b is the genuine week-to-week PROCESS variation and σ_w is the
  per-review score SD. Charting all of them against one constant limit
  would make a 2-review week and a 2408-review week the same
  observation, which they are not. So:

  * σ_b is recovered from the moving range by SUBTRACTING the sampling
    component it necessarily contains —
    ``E[(xᵢ − xᵢ₋₁)²] = 2σ_b² + σ_w²(1/nᵢ + 1/nᵢ₋₁)``, giving
    ``σ̂_b² = (MR-bar/d₂)² − σ_w²·mean(1/nᵢ + 1/nᵢ₋₁)/2`` (floored at 0);
  * every point is then tested against ``σ̂ᵢ = √(σ̂_b² + σ_w²/nᵢ)``, so a
    thin week automatically gets wide limits rather than being deleted;
  * ``σ_w`` is ``REVIEW_SCORE_SD``, and it is **measured**, not assumed —
    see that constant.

  This also closes the old ``INSUFFICIENT_VARIANCE`` hole: a flat
  baseline no longer means "no scale", because the sampling term is a
  real, known scale.

* **Tabular CUSUM** (standardized, k=0.5, h=5 → ARL₀≈465) — accumulates
  standardized deviations from the in-control center. Catches the small
  sustained drift a Shewhart chart misses.

**Downward-only alarming.** Both charts are two-sided, but only a
*downward* breach (scores falling) is a regression worth paging on.
Upward shifts (scores rising) are recorded for observability but do not
set ``OUT_OF_CONTROL`` — improving judge quality is not an incident.

**A week is observed, or it is not — and "not" is never green.**
Three things make a calendar week unobserved, and all three are handled
the same way: the week is not charted, it is COUNTED, and the count is
published.

1. *Too few reviews.* Below ``MIN_REVIEWS_PER_WEEK`` the week's mean
   cannot resolve one step of the rubric it grades, so it is not an
   observation of quality. It is excluded from the centre, from the
   moving range and from the monitoring window, and counted in
   ``unmeasurable_weeks``. It is never silently dropped.
2. *A missing week.* The calendar slot exists and carries no bucket.
   Moving ranges are formed only between **adjacent** week slots — a
   two-week range is a different quantity and would inflate σ̂_b — and
   the gap is counted in ``missing_weeks``. The CUSUM does **not** reset
   at a gap: accumulated evidence about the process is not erased by
   having failed to look, and resetting would make detection sensitivity
   a function of producer reliability.
3. *A trailing gap.* If the newest observation is more than
   ``DEFAULT_MAX_STALENESS_WEEKS`` behind the newest COMPLETE week, the
   combo is ``STALE``. This is the case that used to read IN_CONTROL:
   two whole weeks (2026-08-12, 2026-08-19) produced no judged reviews
   at all and nothing alarmed (``alpha-engine-config-I10169``).

**Insufficient-history gate.** Control limits computed from too few
points are meaningless and over-alarm. Below ``DEFAULT_MIN_HISTORY``
ADMITTED in-window points a combo returns ``INSUFFICIENT_HISTORY`` (an
honest N/A, mirroring the κ ``MIN_REVIEWS_PER_CELL`` gate) and never
breaches. ``n_points`` is surfaced so the dashboard can show band
maturity, and ``INSUFFICIENT_HISTORY``/``INSUFFICIENT_VARIANCE``/
``STALE`` combos are published as
``agent_quality_score_control_unmeasurable_count`` so an unjudgeable
combo is visible rather than absent.

**Baseline period.** Phase-I is the admitted in-window observations
preceding the monitoring window — currently 2026-06-30/07-07 through
2026-08-04 per combo, i.e. 4-5 weekly points. That is thin: individuals
limits want ~20. It is the only rule available while every combo has 8-9
weekly points, and it is stated here so the successor is explicit — once
a combo reaches ~20 weekly points the baseline should be FROZEN as a
declared Phase-I period rather than sliding forward with every run.

**Re-anchor reset (ties to L4578(a)).** A judge-model change is a regime
break — scores before and after are not comparable, and a baseline that
spans the change would trip every band. The control-band baseline must
not straddle a ``judge_resolved_model`` change. The CloudWatch series
does not carry the resolved model, so ``reset_before`` lets the operator
trim the series to the post-re-anchor points after a judge upgrade.
Automatic reset driven off the artifact corpus' ``judge_resolved_model``
is a noted follow-up.

Run cadence: nominally weekly, from the same ``EvalRollingMean`` Lambda
that emits the rolling mean, AFTER the mean is emitted (it reads the mean
series). **Measured 2026-09-08 it is not one run per week**: the
``agent_quality_score_control_breach_count`` stream carries 25 emissions
between 2026-07-30 and 2026-09-05, six of them on 2026-08-30 alone, with
whole weeks carrying one. Every emission timestamps at *run* time, so a
CloudWatch weekly bucket holds the average of however many reruns landed
in it, and a mid-week run reads a partial bucket. Both are handled here:
the open bucket is dropped (``_weekly_series_by_combo``), and the alarm
watching this metric evaluates a weekly period
(``infrastructure/setup_eval_alarms.sh``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha1
from typing import Any

import boto3

from evals.metrics import DEFAULT_METRIC_NAME, DEFAULT_NAMESPACE
from evals.rolling_mean import (
    _CHANGELOG_BUCKET,
    _CHANGELOG_PREFIX,
    _CHANGELOG_SCHEMA_VERSION,
    _build_metric_data_queries,
    _dims_to_dict,
    _get_metric_data_all,
    _list_metric_combos,
)

# Second consumer of the rolling-mean CloudWatch + changelog substrate.
# Reused rather than re-implemented; lifting these to a shared
# ``evals/_cw_metrics.py`` is a noted follow-up (second-adoption
# consolidation signal) — kept out of this PR to bound its blast radius.
SOURCE_METRIC_NAME = DEFAULT_METRIC_NAME
"""The chart input: the RAW weekly ``agent_quality_score``, one bucket
per calendar week, ``Average`` over that week's judged reviews.

It was ``rolling_mean.DERIVED_METRIC_NAME`` (``agent_quality_score_4w_
mean``) until Brian's 2026-09-08 ruling on ``alpha-engine-config-I10167``
— see the module docstring for the measured sigma deflation that ruling
answers. ``rolling_mean``'s 4-week mean is untouched: it still powers the
absolute-quality floor alarm and the dashboard trend line, which are
statements about level, not about control."""

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────

_WEEK_SECONDS = 7 * 86400
LOOKBACK_WEEKS = 26
"""Baseline+monitoring window. 26 weekly points is enough for stable
individuals-chart limits once the corpus matures; today most combos have
far fewer and fall under the insufficient-history gate."""

DEFAULT_MIN_HISTORY = 8
"""Minimum in-window weekly points before bands go live for a combo.
Below this, limits are unreliable and over-alarm — return
INSUFFICIENT_HISTORY (honest N/A). 8 is a pragmatic floor; individuals
charts give tighter limits nearer ~20, so n_points is surfaced for
maturity. Mirrors calibration_kappa.MIN_REVIEWS_PER_CELL discipline."""

DEFAULT_MONITORING_WINDOW = 4
"""Phase-II window: the most-recent N weekly points the charts actually
*test*. Everything before them is the Phase-I baseline that estimates
the center + σ̂ and is never re-tested against itself. The Shewhart chart
flags the latest point; the CUSUM accumulates from a zero start over
these N points ONLY — the fix for config#2385's self-reference false
positive, where running CUSUM over the *full* series let a transient
early-baseline dip latch a downward breach on a combo whose latest score
was actually its healthiest. 4 matches the CUSUM's own detection latency
(k=0.5, h=5 signals a 1σ shift within ~4-5 samples) while leaving the
baseline the majority of the (young) series."""

_D2_N2 = 1.128
"""Hartley's d₂ constant for the moving range of n=2 consecutive points.
σ̂ = MR-bar / d₂ is the standard individuals-chart σ estimator."""

DEFAULT_K_SIGMA = 3.0
"""Shewhart control-limit width (3σ → ~0.27% two-sided false-alarm)."""

DEFAULT_CUSUM_K = 0.5
"""CUSUM reference value (slack), in σ units. 0.5 is tuned to detect a
1σ sustained shift fastest."""

DEFAULT_CUSUM_H = 5.0
"""CUSUM decision interval, in σ units. h=5 with k=0.5 gives
ARL₀≈465 (low false-alarm) and quick detection of a 1σ shift."""

REVIEW_SCORE_SD = 1.0
"""σ_w — the standard deviation of a SINGLE judged review's score on the
1-5 rubric. **Measured, not assumed** (2026-09-08, all 20 live combos,
26-week window). Two independent estimators over the same 132 adjacent
weekly pairs, both solving
``E[(xᵢ − xᵢ₋₁)²] = 2σ_b² + σ_w²(1/nᵢ + 1/nᵢ₋₁)``:

* OLS of the squared weekly difference on ``(1/nᵢ + 1/nᵢ₋₁)`` — slope
  0.790 → **σ_w = 0.889**, intercept 0.081 → pooled σ_b = 0.20;
* a two-group moment split (44 pairs with ``1/nᵢ+1/nᵢ₋₁ > 0.1`` against
  22 pairs below 0.02) → **σ_w = 1.07**, pooled σ_b = 0.13.

1.0 sits between them and is the round number a 1-5 integer rubric would
be expected to produce. The finding it encodes is the important part:
sampling noise at a typical week (n=12 → 0.29) is TWICE the pooled
process variation (0.13-0.20), so most of the movement in the weekly
series is corpus volume, not agent quality — which is
``alpha-engine-config-I10169``'s point, arrived at from the other end.

This is one global constant standing in for a per-combo, per-week
quantity. CloudWatch offers no ``StandardDeviation`` statistic, so it
cannot be read back from the existing streams; having the judge emit a
per-week sum-of-squares (or the rubric-score histogram) would make it
measurable per combo. Tracked as a follow-up."""

MIN_REVIEWS_PER_WEEK = 4
"""Minimum judged reviews for a week to be an OBSERVATION of the process.

Derivation, not a preference: the rubric is 1-5 in integer steps, so the
finest distinction it can express is half a grade. A week whose mean
carries a standard error above 0.5 rubric points cannot resolve even one
step, and charting it asserts a measurement the corpus did not make. At
the measured ``REVIEW_SCORE_SD`` of 1.0, ``σ_w/√n ≤ 0.5`` gives n ≥ 4.

Stricter than ``rolling_mean._FLOOR_MIN_SAMPLES_DEFAULT`` (3), and
deliberately: that gate protects a MIN-reduction from a single
rubric-minimum score, while this one protects an estimate of VARIATION,
which is the harder thing to measure. The two answers are one review
apart, which is inside the uncertainty on σ_w (0.89-1.07 → 3.2-4.6).

Measured effect 2026-09-08: 4 of 160 combo-weeks are excluded (three
n=2 weeks and one n=3 week, all in ``thinktank_theme`` under
``claude-sonnet-4-6``), which takes those four combos below
``DEFAULT_MIN_HISTORY`` and reports them ``INSUFFICIENT_HISTORY``. That
is the honest state of a corpus with 6 usable weeks for those combos,
and it is published on the unmeasurable stream rather than folded into
the healthy zero."""

DEFAULT_MAX_STALENESS_WEEKS = 1
"""How many complete weeks the newest observation may lag before the
combo is ``STALE`` rather than judged.

One week of grace, because the producer's own cadence is irregular
(measured: 25 emissions 2026-07-30..2026-09-05, six on 2026-08-30 alone,
six-day gaps elsewhere) and a judged week can slip across the Saturday
boundary. TWO consecutive unobserved weeks is not slippage — it is the
2026-08-12/08-19 hole, where the corpus produced nothing, every combo
kept reporting IN_CONTROL against a fortnight-old point, and nothing
alarmed."""

ZSCORE_METRIC_NAME = "agent_quality_score_zscore"
"""Per-combo standardized deviation of the latest 4w-mean from its
in-control center. Powers the dashboard's per-combo drift line."""

UNMEASURABLE_COUNT_METRIC_NAME = "agent_quality_score_control_unmeasurable_count"
"""Dimensionless single-datapoint metric = number of combos the chart
could NOT judge this run (INSUFFICIENT_HISTORY + INSUFFICIENT_VARIANCE +
STALE). Emitted every run, 0 included.

`principles.md` §2.7: a combo that cannot be judged is not a combo in
control, and a breach count of 0 must not be readable as "20 combos are
fine" when 4 of them were never evaluated. This stream is deliberately
NOT alarmed yet: it reads 4 of 20 today, entirely because the judged
corpus is too thin (`alpha-engine-config-I10169`), so an alarm on it
would be red from birth and would be muted rather than acted on. The
threshold belongs with the corpus fix, and is tracked there."""

BREACH_COUNT_METRIC_NAME = "agent_quality_score_control_breach_count"
"""Dimensionless single-datapoint metric = number of combos currently
OUT_OF_CONTROL (downward). The CloudWatch alarm fires on ``>= 1``.
Emitted every run (0 included) so the stream stays alive and the alarm
never sits in INSUFFICIENT_DATA. Same single-stream-for-the-alarm
rationale as rolling_mean's floor metric (alarms reject SEARCH)."""


# ── Statuses ──────────────────────────────────────────────────────────────

STATUS_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
STATUS_INSUFFICIENT_VARIANCE = "INSUFFICIENT_VARIANCE"
STATUS_IN_CONTROL = "IN_CONTROL"
STATUS_OUT_OF_CONTROL = "OUT_OF_CONTROL"
STATUS_STALE = "STALE"


# ── Pure statistics ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeeklyObservation:
    """One calendar week's judged score for one combo.

    ``week_index`` is the integer week slot (``floor(bucket_start /
    7 days)``), so ``b.week_index - a.week_index == 1`` is the test for
    "these two weeks are adjacent" and anything larger is a gap. Carrying
    the slot rather than a timestamp keeps the statistics free of
    calendar arithmetic.

    ``reviews`` is the week's CloudWatch ``SampleCount`` — the number of
    judged reviews behind ``value``. It is not decoration: the limits are
    a function of it (alpha-engine-config-I10167).
    """

    week_index: int
    value: float
    reviews: int


def _adjacent_pairs(
    observations: list[WeeklyObservation],
) -> list[tuple[WeeklyObservation, WeeklyObservation]]:
    """Consecutive-WEEK pairs only.

    A moving range across a gap is the range over two or more weeks — a
    different quantity, larger in expectation, which would inflate σ̂_b
    and blunt every limit derived from it. Excluding those pairs is the
    only honest treatment of a missing week that keeps the rest of the
    series usable.
    """
    return [
        (observations[i - 1], observations[i])
        for i in range(1, len(observations))
        if observations[i].week_index - observations[i - 1].week_index == 1
    ]


def process_sigma(
    observations: list[WeeklyObservation],
    *,
    review_sd: float = REVIEW_SCORE_SD,
) -> float:
    """Estimate σ_b — the week-to-week PROCESS variation — from the
    moving range, with the sampling component removed.

    ``E[(xᵢ − xᵢ₋₁)²] = 2σ_b² + σ_w²(1/nᵢ + 1/nᵢ₋₁)`` and
    ``(MR-bar/d₂)²`` estimates ``E[(xᵢ − xᵢ₋₁)²]/2``, so

        σ̂_b² = (MR-bar/d₂)² − σ_w²·mean(1/nᵢ + 1/nᵢ₋₁)/2

    floored at zero. A zero result is a real answer, not a failure: it
    says every unit of observed week-to-week movement is explained by
    how few reviews each week rested on. The chart still has a scale,
    because ``observation_sigma`` keeps the sampling term.

    Returns 0.0 when there is no adjacent pair to form a range from.
    """
    pairs = _adjacent_pairs(observations)
    if not pairs:
        return 0.0
    mr_bar = sum(abs(b.value - a.value) for a, b in pairs) / len(pairs)
    total_var = (mr_bar / _D2_N2) ** 2
    sampling_var = review_sd ** 2 * sum(
        (1.0 / a.reviews + 1.0 / b.reviews) for a, b in pairs
    ) / len(pairs) / 2.0
    return (max(0.0, total_var - sampling_var)) ** 0.5


def observation_sigma(
    *,
    sigma_b: float,
    reviews: int,
    review_sd: float = REVIEW_SCORE_SD,
) -> float:
    """σ̂ᵢ = √(σ̂_b² + σ_w²/nᵢ) — the SD of one week's mean.

    Strictly positive for any finite ``reviews``, which is what removes
    the old zero-variance degenerate case: there is always a scale,
    because a mean of finitely many reviews always has sampling error.
    """
    if reviews <= 0:
        raise ValueError("observation_sigma needs reviews > 0")
    return (sigma_b ** 2 + review_sd ** 2 / reviews) ** 0.5


def moving_range_sigma(series: list[float]) -> float:
    """Estimate σ from the average moving range (individuals chart).

    σ̂ = mean(|xᵢ − xᵢ₋₁|) / d₂, d₂ = 1.128 for ranges of 2 points.
    Returns 0.0 for a constant series (every moving range is 0) — callers
    handle the zero-variance case explicitly (no division by it).
    Requires ``len(series) >= 2``.
    """
    if len(series) < 2:
        raise ValueError("moving_range_sigma needs >= 2 points")
    mrs = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
    mr_bar = sum(mrs) / len(mrs)
    return mr_bar / _D2_N2


@dataclass(frozen=True)
class CusumResult:
    c_plus: float          # final upward cumulative sum
    c_minus: float         # final downward cumulative sum
    breached_high: bool    # C+ exceeded h at any point (upward shift)
    breached_low: bool     # C- exceeded h at any point (downward shift)


def tabular_cusum(
    series: list[float],
    *,
    target: float,
    sigma: float | list[float],
    k: float = DEFAULT_CUSUM_K,
    h: float = DEFAULT_CUSUM_H,
    reset_after_signal: bool = True,
) -> CusumResult:
    """Standardized two-sided tabular CUSUM.

    zᵢ = (xᵢ − target) / σ
    C⁺ᵢ = max(0, C⁺ᵢ₋₁ + zᵢ − k)   (upward)
    C⁻ᵢ = max(0, C⁻ᵢ₋₁ − zᵢ − k)   (downward)
    A side signals when its statistic exceeds ``h``.

    ``reset_after_signal`` (default True) restores the textbook tabular
    CUSUM: **the accumulator is reset to zero once it signals**
    (Montgomery, *Introduction to Statistical Quality Control*, §9.1.3 —
    a signal means the process is investigated and the chart restarted).
    Without the reset a single deep excursion leaves C⁻ permanently
    inflated, so every later evaluation of the same window re-reports the
    same past excursion as a live "sustained downward drift". Measured on
    2026-09-05: ``thinktank_thesis/context_integration/claude-sonnet-4-6``
    reported ``C- 7.86 > h 5.0`` while its latest 4w-mean (4.002) sat
    **above** its own center (3.913) at z = +1.62 — a six-month high
    reported as a regression (alpha-engine-config-I10165).

    ``sigma`` may be a scalar or a **per-point** list. The per-point form
    is what the raw weekly scale needs: each week's mean has its own
    standard deviation ``√(σ_b² + σ_w²/nᵢ)``, so a thin week contributes
    a correspondingly small standardized deviation instead of the same
    weight as a week with two orders of magnitude more reviews
    (alpha-engine-config-I10167).

    Requires every sigma > 0 (the standardization divides by it) —
    callers gate on the zero-variance case before invoking.
    """
    sigmas = list(sigma) if isinstance(sigma, list) else [sigma] * len(series)
    if len(sigmas) != len(series):
        raise ValueError("tabular_cusum needs one sigma per point")
    if any(sg <= 0 for sg in sigmas):
        raise ValueError("tabular_cusum needs sigma > 0")
    c_plus = 0.0
    c_minus = 0.0
    breached_high = False
    breached_low = False
    for x, sg in zip(series, sigmas, strict=True):
        z = (x - target) / sg
        c_plus = max(0.0, c_plus + z - k)
        c_minus = max(0.0, c_minus - z - k)
        if c_plus > h:
            breached_high = True
            if reset_after_signal:
                c_plus = 0.0
        if c_minus > h:
            breached_low = True
            if reset_after_signal:
                c_minus = 0.0
    return CusumResult(c_plus, c_minus, breached_high, breached_low)


@dataclass(frozen=True)
class ControlBandResult:
    """Outcome of evaluating one combo's weekly 4w-mean series.

    ``status == OUT_OF_CONTROL`` iff a DOWNWARD breach fired (Shewhart
    below LCL, or CUSUM C⁻ over h) — the alarmable regression state.
    Upward shifts populate the ``*_high`` fields for observability but
    never set OUT_OF_CONTROL.
    """

    status: str
    n_points: int                          # ADMITTED weekly observations
    center: float | None = None
    sigma: float | None = None             # sigma-hat of the LATEST week
    sigma_b: float | None = None           # process sigma, sampling removed
    reviews_latest: int | None = None
    unmeasurable_weeks: int = 0            # weeks below MIN_REVIEWS_PER_WEEK
    missing_weeks: int = 0                 # calendar slots with no bucket
    lcl: float | None = None
    ucl: float | None = None
    latest: float | None = None
    latest_z: float | None = None      # None when sigma == 0
    shewhart_low: bool = False            # latest < LCL (regression)
    shewhart_high: bool = False           # latest > UCL (observability)
    cusum_c_minus: float | None = None
    cusum_c_plus: float | None = None
    cusum_low: bool = False               # downward drift, CURRENT (alarms)
    cusum_signal_low: bool = False        # C- signalled in-window (observability)
    cusum_high: bool = False              # upward drift (observability)
    reasons: list[str] = field(default_factory=list)


def evaluate_series(
    observations: list[WeeklyObservation],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    monitoring_window: int = DEFAULT_MONITORING_WINDOW,
    k_sigma: float = DEFAULT_K_SIGMA,
    cusum_k: float = DEFAULT_CUSUM_K,
    cusum_h: float = DEFAULT_CUSUM_H,
    min_reviews: int = MIN_REVIEWS_PER_WEEK,
    review_sd: float = REVIEW_SCORE_SD,
    latest_complete_week: int | None = None,
    max_staleness_weeks: int = DEFAULT_MAX_STALENESS_WEEKS,
) -> ControlBandResult:
    """Run the Shewhart + CUSUM charts on one combo's RAW WEEKLY series.

    ``observations`` are the combo's weekly ``agent_quality_score``
    points, oldest first, each carrying the week slot and the number of
    reviews behind it. Evaluated as a genuine **Phase-I / Phase-II
    split**:

    * **Phase-I baseline** — all admitted weeks except the most-recent
      ``monitoring_window`` — estimates the in-control ``center`` and
      σ̂_b. These weeks ONLY fit the limits.
    * **Phase-II monitoring** — the most-recent ``monitoring_window``
      admitted weeks — is what the charts test. The Shewhart chart flags
      the latest week against limits computed for **that week's** review
      count; the CUSUM accumulates from a zero start over the monitoring
      weeks alone, standardizing each by its own σ̂ᵢ.

    Running CUSUM over the monitoring window (not the full series) is the
    fix for config#2385 failure mode 2: the earlier baseline points that
    *defined* the center are never re-tested against it, so a transient
    early-baseline dip can no longer latch ``breached_low`` on a combo
    whose latest score is actually healthy.

    The **centre is the unweighted mean of the baseline weekly means**,
    not a review-weighted average of the underlying reviews. Each week is
    one observation of the process; review-weighting would let a single
    2408-review week set the centre for a combo whose other weeks carry
    a dozen reviews, which is exactly the volume-mix defect
    ``alpha-engine-config-I10169`` measures on the CloudWatch ``Average``.
    Efficiency is traded for that robustness knowingly: an
    inverse-variance-weighted centre is the more efficient estimator and
    is the right one once weekly volume is stable.

    ``latest_complete_week`` is the newest week slot whose bucket has
    closed. When the newest admitted observation lags it by more than
    ``max_staleness_weeks``, the combo is ``STALE`` — the process has
    gone unmeasured, and that is never reported as in control.
    """
    ordered = sorted(observations, key=lambda o: o.week_index)
    admitted = [o for o in ordered if o.reviews >= min_reviews]
    unmeasurable = len(ordered) - len(admitted)
    missing = (
        sum(
            admitted[i].week_index - admitted[i - 1].week_index - 1
            for i in range(1, len(admitted))
        )
        if admitted else 0
    )
    n = len(admitted)

    def _partial(status: str, reasons: list[str]) -> ControlBandResult:
        return ControlBandResult(
            status=status,
            n_points=n,
            latest=admitted[-1].value if admitted else None,
            reviews_latest=admitted[-1].reviews if admitted else None,
            unmeasurable_weeks=unmeasurable,
            missing_weeks=missing,
            reasons=reasons,
        )

    if not admitted:
        return _partial(
            STATUS_INSUFFICIENT_HISTORY,
            [
                f"insufficient_history: 0 admitted weeks "
                f"({unmeasurable} below min_reviews {min_reviews})"
            ],
        )

    # A trailing gap is the dangerous gap: every other status is an
    # explicit N/A, but a stale series silently answers IN_CONTROL about
    # a point that is weeks old.
    if latest_complete_week is not None:
        lag = latest_complete_week - admitted[-1].week_index
        if lag > max_staleness_weeks:
            return _partial(
                STATUS_STALE,
                [
                    f"stale: newest observed week is {lag} complete weeks "
                    f"behind the newest complete week (grace "
                    f"{max_staleness_weeks}) — the process went "
                    f"unmeasured, which is not in control"
                ],
            )

    if n < min_history:
        return _partial(
            STATUS_INSUFFICIENT_HISTORY,
            [
                f"insufficient_history: {n} < {min_history} admitted "
                f"weeks ({unmeasurable} below min_reviews {min_reviews}, "
                f"{missing} week(s) missing)"
            ],
        )

    # Phase-I / Phase-II split. σ̂_b needs a range, so keep the baseline
    # at >= 2 points even for unusual ``monitoring_window`` values.
    n_monitor = max(1, min(monitoring_window, n - 2))
    baseline = admitted[: n - n_monitor]
    monitoring = admitted[n - n_monitor:]
    latest_obs = admitted[-1]
    latest = latest_obs.value
    center = sum(o.value for o in baseline) / len(baseline)
    sigma_b = process_sigma(baseline, review_sd=review_sd)

    if not _adjacent_pairs(baseline):
        # Every baseline week is isolated by a gap, so no moving range
        # exists and σ̂_b is not estimable. Honest N/A rather than a
        # limit fitted from nothing.
        return ControlBandResult(
            status=STATUS_INSUFFICIENT_VARIANCE,
            n_points=n,
            center=center,
            sigma_b=None,
            latest=latest,
            reviews_latest=latest_obs.reviews,
            unmeasurable_weeks=unmeasurable,
            missing_weeks=missing,
            reasons=[
                f"insufficient_variance: no adjacent-week pair in the "
                f"{len(baseline)}-week baseline — a moving range across a "
                f"gap is not a within-process range, so sigma is not "
                f"estimable (center {center:.3f})"
            ],
        )

    sigma = observation_sigma(
        sigma_b=sigma_b, reviews=latest_obs.reviews, review_sd=review_sd,
    )
    lcl = center - k_sigma * sigma
    ucl = center + k_sigma * sigma

    shewhart_low = latest < lcl
    shewhart_high = latest > ucl
    latest_z: float | None = (latest - center) / sigma

    # CUSUM accumulates over the Phase-II monitoring weeks ONLY, from a
    # zero start, each standardized by its OWN sigma — a 12-review week
    # and a 769-review week do not contribute equally
    # (config#2385 failure mode 2; alpha-engine-config-I10167).
    monitor_sigmas = [
        observation_sigma(
            sigma_b=sigma_b, reviews=o.reviews, review_sd=review_sd,
        )
        for o in monitoring
    ]
    cusum = tabular_cusum(
        [o.value for o in monitoring],
        target=center,
        sigma=monitor_sigmas,
        k=cusum_k,
        h=cusum_h,
    )
    # A CUSUM signal states that deviation ACCUMULATED over the
    # monitoring window. It is not by itself a statement that the process
    # is running low *now*, and the alarm surface it feeds
    # (``agent_quality_score_control_breach_count``) declares the number
    # of combos **currently** OUT_OF_CONTROL. So a downward signal only
    # alarms while the latest point is itself below the in-control
    # center; a signal the latest observation has already reversed is
    # recorded for observability and does not page
    # (alpha-engine-config-I10165). The Shewhart test needs no such gate:
    # ``latest < lcl`` already implies ``latest < center``.
    cusum_signal_low = cusum.breached_low
    cusum_low = cusum_signal_low and latest < center
    cusum_high = cusum.breached_high

    reasons: list[str] = []
    if shewhart_low:
        reasons.append(
            f"shewhart_low: latest {latest:.3f} (n={latest_obs.reviews}) "
            f"< LCL {lcl:.3f} (center {center:.3f}, sigma {sigma:.3f} = "
            f"process {sigma_b:.3f} + sampling)"
        )
    if cusum_low:
        reasons.append(
            f"cusum_low: C- signalled > h {cusum_h:.1f} over the "
            f"{len(monitoring)}-week monitoring window and the latest "
            f"week {latest:.3f} is still below center {center:.3f} "
            f"(sustained downward drift)"
        )
    elif cusum_signal_low:
        reasons.append(
            f"cusum_signal_low (observability, recovered): C- signalled "
            f"> h {cusum_h:.1f} in-window but the latest week "
            f"{latest:.3f} is at/above center {center:.3f} "
            f"(z {latest_z:+.2f}) — the excursion has reversed, not "
            f"alarmed"
        )
    # Upward signals are observability-only — recorded, not alarmed.
    if shewhart_high:
        reasons.append(
            f"shewhart_high (observability): latest {latest:.3f} > "
            f"UCL {ucl:.3f}"
        )
    if cusum_high:
        reasons.append(
            f"cusum_high (observability): C+ {cusum.c_plus:.2f} > h {cusum_h:.1f}"
        )
    if unmeasurable or missing:
        reasons.append(
            f"weeks_not_observed: {unmeasurable} below min_reviews "
            f"{min_reviews}, {missing} missing — excluded from the chart, "
            f"counted here, never charted as an observation"
        )

    out_of_control = shewhart_low or cusum_low
    return ControlBandResult(
        status=STATUS_OUT_OF_CONTROL if out_of_control else STATUS_IN_CONTROL,
        n_points=n,
        center=center,
        sigma=sigma,
        sigma_b=sigma_b,
        reviews_latest=latest_obs.reviews,
        unmeasurable_weeks=unmeasurable,
        missing_weeks=missing,
        lcl=lcl,
        ucl=ucl,
        latest=latest,
        latest_z=latest_z,
        shewhart_low=shewhart_low,
        shewhart_high=shewhart_high,
        cusum_c_minus=cusum.c_minus,
        cusum_c_plus=cusum.c_plus,
        cusum_low=cusum_low,
        cusum_signal_low=cusum_signal_low,
        cusum_high=cusum_high,
        reasons=reasons,
    )


# ── CloudWatch series extraction ──────────────────────────────────────────


def _week_index(ts: datetime) -> int:
    """The integer week slot a CloudWatch weekly bucket start falls in."""
    return int(ts.timestamp()) // _WEEK_SECONDS


def latest_complete_week_index(end_time: datetime) -> int:
    """The newest week slot whose 7-day bucket has fully elapsed."""
    return int(end_time.timestamp()) // _WEEK_SECONDS - 1


def _weekly_series_by_combo(
    metric_data_results: list[dict[str, Any]],
    combos: list[list[dict[str, str]]],
    *,
    reset_before: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[int, list[WeeklyObservation]]:
    """Map combo index → its weekly observations, oldest-first.

    Reads BOTH halves of the paired query ``_build_metric_data_queries``
    already emits: ``m{idx}`` is the week's ``Average`` and ``n{idx}`` is
    its ``SampleCount``. The sample count was previously discarded; on
    the raw weekly scale it is load-bearing, because it sets the week's
    limits (alpha-engine-config-I10167).

    CloudWatch returns Timestamps (descending by default) paired with
    Values; we zip + sort ascending so the moving range / CUSUM see the
    series in chronological order. When ``reset_before`` is set, points
    older than it are dropped so the baseline doesn't straddle a judge
    re-anchor (L4578(a)).

    **The still-open weekly bucket is dropped.** A CloudWatch bucket
    whose window has not yet elapsed holds a partial average, and the
    Shewhart chart tests exactly that latest point. Measured
    2026-08-30: the run at 19:52Z read the open 08-25 bucket for
    ``thinktank_thesis/moat_and_business_quality/claude-haiku-4-5`` as
    3.1807 and flagged ``shewhart_low`` against an LCL of 3.1843 — a
    margin of 0.0036; the run at 23:26Z the same evening no longer
    breached, and that bucket settled at 3.7346. A partial bucket is
    not an observation of the week, so it is never charted
    (alpha-engine-config-I10165). ``end_time`` defaults to now.

    A week whose SampleCount is missing is treated as **zero reviews**,
    never as "assume enough": an unmeasured n is exactly the case the
    admission gate exists for.
    """
    cutoff = end_time or datetime.now(UTC)
    by_id = {r["Id"]: r for r in metric_data_results}
    series_by_combo: dict[int, list[WeeklyObservation]] = {}
    for idx in range(len(combos)):
        result = by_id.get(f"m{idx}")
        if result is None:
            series_by_combo[idx] = []
            continue
        counts_result = by_id.get(f"n{idx}", {})
        counts = dict(zip(
            counts_result.get("Timestamps", []),
            counts_result.get("Values", []),
            strict=True,
        ))
        pairs = list(zip(result.get("Timestamps", []), result.get("Values", []), strict=True))
        # Drop the bucket still accumulating: its window end is in the
        # future relative to this run, so its Average is partial.
        pairs = [
            (t, v) for t, v in pairs
            if t + timedelta(seconds=_WEEK_SECONDS) <= cutoff
        ]
        if reset_before is not None:
            pairs = [(t, v) for t, v in pairs if t >= reset_before]
        pairs.sort(key=lambda tv: tv[0])  # chronological
        series_by_combo[idx] = [
            WeeklyObservation(
                week_index=_week_index(t),
                value=float(v),
                reviews=int(counts.get(t, 0)),
            )
            for t, v in pairs
        ]
    return series_by_combo


# ── Orchestration ─────────────────────────────────────────────────────────


def compute_and_emit_control_bands(
    *,
    end_time: datetime | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    source_metric: str = SOURCE_METRIC_NAME,
    min_history: int = DEFAULT_MIN_HISTORY,
    reset_before: datetime | None = None,
    cloudwatch_client: Any | None = None,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Evaluate control bands per combo and emit metrics + breach entries.

    Reads the raw weekly ``agent_quality_score`` series per combo over
    the trailing ``LOOKBACK_WEEKS`` window, runs the Shewhart + CUSUM
    charts, and emits:

    * per-combo ``agent_quality_score_zscore`` (when σ > 0),
    * a single dimensionless ``agent_quality_score_control_breach_count``
      = number of OUT_OF_CONTROL combos (the alarm surface),
    * a single dimensionless
      ``agent_quality_score_control_unmeasurable_count`` = number of
      combos the chart could not judge, so a zero breach count is never
      read as "every combo is fine",
    * one changelog entry per OUT_OF_CONTROL combo.

    ``reset_before`` (UTC) trims each combo's series to points at/after it
    — the operator sets it after a judge re-anchor (L4578(a)) so the
    baseline doesn't straddle a model change.
    """
    cw = cloudwatch_client or boto3.client("cloudwatch")
    s3 = s3_client or boto3.client("s3")
    end = end_time or datetime.now(UTC)
    start = end - timedelta(days=LOOKBACK_WEEKS * 7)

    combos = _list_metric_combos(
        cw, namespace=namespace, metric_name=source_metric,
    )
    if not combos:
        logger.warning(
            "[control_bands] no %s/%s streams — nothing to evaluate yet "
            "(rolling-mean must run first)",
            namespace, source_metric,
        )
        return _empty_summary(start, end)

    queries = _build_metric_data_queries(
        combos,
        namespace=namespace,
        metric_name=source_metric,
        period_seconds=_WEEK_SECONDS,  # one bucket per weekly run
    )
    metric_data_results = _get_metric_data_all(cw, queries, start, end)
    series_by_combo = _weekly_series_by_combo(
        metric_data_results, combos, reset_before=reset_before,
        end_time=end,
    )

    zscore_data: list[dict[str, Any]] = []
    breaches: list[dict[str, Any]] = []
    breach_emits: list[str] = []
    n_insufficient = 0
    n_insufficient_variance = 0
    n_stale = 0
    n_in_control = 0
    failed: list[dict[str, str]] = []

    latest_week = latest_complete_week_index(end)
    for idx, dims in enumerate(combos):
        series = series_by_combo.get(idx, [])
        try:
            result = evaluate_series(
                series,
                min_history=min_history,
                latest_complete_week=latest_week,
            )
        except Exception as exc:  # noqa: BLE001 — isolate one combo's failure
            failed.append({
                "combo_idx": str(idx),
                "stage": "evaluate_series",
                "error": str(exc),
            })
            continue

        if result.latest_z is not None:
            zscore_data.append({
                "MetricName": ZSCORE_METRIC_NAME,
                "Dimensions": dims,
                "Value": float(result.latest_z),
                "Unit": "None",
                "Timestamp": end,
            })

        if result.status == STATUS_INSUFFICIENT_HISTORY:
            n_insufficient += 1
        elif result.status == STATUS_INSUFFICIENT_VARIANCE:
            n_insufficient_variance += 1
        elif result.status == STATUS_STALE:
            n_stale += 1
        elif result.status == STATUS_IN_CONTROL:
            n_in_control += 1
        elif result.status == STATUS_OUT_OF_CONTROL:
            breaches.append({
                "dims": _dims_to_dict(dims),
                "reasons": result.reasons,
                "latest": result.latest,
                "center": result.center,
                "lcl": result.lcl,
            })
            key = _emit_control_breach_entry(
                dims=dims,
                result=result,
                window_start=start,
                window_end=end,
                s3_client=s3,
            )
            if key:
                breach_emits.append(key)

    # Emit per-combo z-scores (chunked at the PutMetricData 1000 cap).
    for chunk_start in range(0, len(zscore_data), 1000):
        chunk = zscore_data[chunk_start:chunk_start + 1000]
        if chunk:
            cw.put_metric_data(Namespace=namespace, MetricData=chunk)

    # Emit the breach-count alarm surface every run (0 included), and
    # alongside it the count of combos the chart could NOT judge — a
    # breach count of 0 is only good news when the denominator is known
    # (principles.md 2.7).
    breach_count = len(breaches)
    unmeasurable_count = n_insufficient + n_insufficient_variance + n_stale
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                "MetricName": BREACH_COUNT_METRIC_NAME,
                "Value": float(breach_count),
                "Unit": "None",
                "Timestamp": end,
            },
            {
                "MetricName": UNMEASURABLE_COUNT_METRIC_NAME,
                "Value": float(unmeasurable_count),
                "Unit": "None",
                "Timestamp": end,
            },
        ],
    )

    logger.info(
        "[control_bands] done combos=%d in_control=%d insufficient=%d "
        "insufficient_variance=%d stale=%d unmeasurable=%d "
        "out_of_control=%d zscores_emitted=%d failed=%d",
        len(combos), n_in_control, n_insufficient, n_insufficient_variance,
        n_stale, unmeasurable_count, breach_count, len(zscore_data),
        len(failed),
    )

    return {
        "combos_discovered": len(combos),
        "combos_in_control": n_in_control,
        "combos_insufficient_history": n_insufficient,
        "combos_insufficient_variance": n_insufficient_variance,
        "combos_stale": n_stale,
        "combos_unmeasurable": unmeasurable_count,
        "breach_count": breach_count,
        "breaches": breaches,
        "breach_emits": breach_emits,
        "zscores_emitted": len(zscore_data),
        "failed": failed,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }


def _empty_summary(start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "combos_discovered": 0,
        "combos_in_control": 0,
        "combos_insufficient_history": 0,
        "combos_insufficient_variance": 0,
        "breach_count": 0,
        "breaches": [],
        "breach_emits": [],
        "zscores_emitted": 0,
        "failed": [],
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }


def _emit_control_breach_entry(
    *,
    dims: list[dict[str, str]],
    result: ControlBandResult,
    window_start: datetime,
    window_end: datetime,
    s3_client: Any,
) -> str | None:
    """Write one ``eval_score_control_breach`` entry to the system-wide
    changelog corpus (mirrors rolling_mean._emit_regression_entry).

    Best-effort: any failure logs WARN and returns None — it must not
    interrupt sibling combos or the metric emission.
    """
    try:
        ts = window_end
        ts_utc = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        entry_date = ts.strftime("%Y-%m-%d")
        ts_id = ts_utc.replace(":", "-").rstrip("Z")
        actor = "alpha-engine-eval-control-bands"

        dims_flat = _dims_to_dict(dims)
        agent_id = dims_flat.get("judged_agent_id", "unknown")
        criterion = dims_flat.get("criterion", "unknown")
        judge_model = dims_flat.get("judge_model", "unknown")

        digest_input = (
            f"{agent_id}|{criterion}|{judge_model}|"
            f"{window_start.isoformat()}|{window_end.isoformat()}|control"
        ).encode()
        # Short deterministic dedup id, not a security digest.
        event_hash = sha1(digest_input, usedforsecurity=False).hexdigest()[:7]
        event_id = f"{ts_id}_{actor}_{event_hash}"

        summary = (
            f"Eval-score control breach: {agent_id}/{criterion} — "
            f"{'; '.join(result.reasons)[:180]}"
        )[:240]
        description = (
            f"Judged agent: {agent_id}\n"
            f"Criterion: {criterion}\n"
            f"Judge model: {judge_model}\n"
            f"Status: {result.status}\n"
            f"Latest weekly mean: {result.latest} "
            f"(n={result.reviews_latest} reviews)\n"
            f"Center: {result.center}  sigma: {result.sigma} "
            f"(process sigma_b {result.sigma_b})\n"
            f"LCL: {result.lcl}  UCL: {result.ucl}\n"
            f"Latest z: {result.latest_z}\n"
            f"CUSUM C-: {result.cusum_c_minus}  C+: {result.cusum_c_plus}\n"
            f"Reasons: {result.reasons}\n"
            f"n_points: {result.n_points}\n"
            f"Window: {window_start.isoformat()} -> {window_end.isoformat()}\n"
            f"Weeks not observed: {result.unmeasurable_weeks} below "
            f"min_reviews, {result.missing_weeks} missing\n"
            f"Detected by: alpha-engine-research evals/control_bands.py "
            f"(Shewhart individuals + tabular CUSUM on the RAW WEEKLY "
            f"agent_quality_score, variance-adjusted for weekly review "
            f"count — alpha-engine-config-I10167)."
        )

        entry = {
            "schema_version": _CHANGELOG_SCHEMA_VERSION,
            "event_id": event_id,
            "ts_utc": ts_utc,
            "event_type": "eval_score_control_breach",
            "severity": "medium",
            "subsystem": "eval",
            "root_cause_category": "prompt_regression",
            "resolution_type": None,
            "started_at": None,
            "detected_at": ts_utc,
            "resolved_at": None,
            "verified_at": None,
            "summary": summary,
            "description": description,
            "resolution_notes": None,
            "actor": actor,
            "machine": "research:evals/control_bands.py",
            "source": "eval-control-band-autoemit",
            "auto_emitted": True,
            "git_refs": [],
            "prompt_version": None,
            "run_id": window_end.strftime("%Y-%m-%d"),
            "eval_run_ref": (
                f"s3://alpha-engine-research/decision_artifacts/_eval/"
                f"{window_end.strftime('%Y-%m-%d')}/{agent_id}/"
            ),
            "eval_control_breach": {
                "judged_agent_id": agent_id,
                "criterion": criterion,
                "judge_model": judge_model,
                "status": result.status,
                "latest": result.latest,
                "center": result.center,
                "sigma": result.sigma,
                "sigma_b": result.sigma_b,
                "reviews_latest": result.reviews_latest,
                "unmeasurable_weeks": result.unmeasurable_weeks,
                "missing_weeks": result.missing_weeks,
                "lcl": result.lcl,
                "ucl": result.ucl,
                "latest_z": result.latest_z,
                "cusum_c_minus": result.cusum_c_minus,
                "shewhart_low": result.shewhart_low,
                "cusum_low": result.cusum_low,
                "cusum_signal_low": result.cusum_signal_low,
                "n_points": result.n_points,
                "reasons": result.reasons,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
            },
        }
        key = f"{_CHANGELOG_PREFIX}/{entry_date}/{event_id}.json"
        s3_client.put_object(
            Bucket=_CHANGELOG_BUCKET,
            Key=key,
            Body=json.dumps(entry).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "[control_bands] breach auto-emit: s3://%s/%s agent=%s "
            "criterion=%s reasons=%s",
            _CHANGELOG_BUCKET, key, agent_id, criterion, result.reasons,
        )
        return key
    except Exception as e:  # noqa: BLE001 — best-effort, see docstring
        logger.warning(
            "[control_bands] breach auto-emit failed (best-effort, "
            "swallowed): %s",
            e,
        )
        return None
