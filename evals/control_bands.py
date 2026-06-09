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
exactly those, operating on each combo's *own* historical 4w-mean
series rather than a global constant:

* **Shewhart individuals (X) chart** — center ± k·σ̂. Because there is
  one datapoint per weekly run (n=1), σ is estimated from the **average
  moving range** (σ̂ = MR-bar / d₂, d₂=1.128 for ranges of 2 consecutive
  points), NOT the raw sample stdev. The MR estimator measures
  short-term within-process variation and is robust to the sustained
  shifts we are trying to detect (a raw stdev would be inflated by the
  shift itself and hide it). Catches large sudden steps.

* **Tabular CUSUM** (standardized, k=0.5, h=5 → ARL₀≈465) — accumulates
  standardized deviations from the in-control center. Catches the small
  sustained drift a Shewhart chart misses.

**Downward-only alarming.** Both charts are two-sided, but only a
*downward* breach (scores falling) is a regression worth paging on.
Upward shifts (scores rising) are recorded for observability but do not
set ``OUT_OF_CONTROL`` — improving judge quality is not an incident.

**Insufficient-history gate.** Control limits computed from too few
points are meaningless and over-alarm. Below ``DEFAULT_MIN_HISTORY``
in-window points a combo returns ``INSUFFICIENT_HISTORY`` (an honest
N/A, mirroring the κ ``MIN_REVIEWS_PER_CELL`` gate) and never breaches.
The eval corpus is young (canonical cutover 2026-05-09), so most combos
sit here for now; ``n_points`` is surfaced so the dashboard can show
band maturity.

**Re-anchor reset (ties to L4578(a)).** A judge-model change is a regime
break — scores before and after are not comparable, and a baseline that
spans the change would trip every band. The control-band baseline must
not straddle a ``judge_resolved_model`` change. The CloudWatch series
does not carry the resolved model, so ``reset_before`` lets the operator
trim the series to the post-re-anchor points after a judge upgrade.
Automatic reset driven off the artifact corpus' ``judge_resolved_model``
is a noted follow-up.

Run cadence: weekly, from the same ``EvalRollingMean`` Lambda that emits
the rolling mean, AFTER the mean is emitted (it reads the mean series).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any, Optional

import boto3

from evals.metrics import DEFAULT_NAMESPACE
# Second consumer of the rolling-mean CloudWatch + changelog substrate.
# Reused rather than re-implemented; lifting these to a shared
# ``evals/_cw_metrics.py`` is a noted follow-up (second-adoption
# consolidation signal) — kept out of this PR to bound its blast radius.
from evals.rolling_mean import (
    DERIVED_METRIC_NAME as SOURCE_METRIC_NAME,
    _CHANGELOG_BUCKET,
    _CHANGELOG_PREFIX,
    _CHANGELOG_SCHEMA_VERSION,
    _build_metric_data_queries,
    _dims_to_dict,
    _get_metric_data_all,
    _list_metric_combos,
)

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

ZSCORE_METRIC_NAME = "agent_quality_score_zscore"
"""Per-combo standardized deviation of the latest 4w-mean from its
in-control center. Powers the dashboard's per-combo drift line."""

BREACH_COUNT_METRIC_NAME = "agent_quality_score_control_breach_count"
"""Dimensionless single-datapoint metric = number of combos currently
OUT_OF_CONTROL (downward). The CloudWatch alarm fires on ``>= 1``.
Emitted every run (0 included) so the stream stays alive and the alarm
never sits in INSUFFICIENT_DATA. Same single-stream-for-the-alarm
rationale as rolling_mean's floor metric (alarms reject SEARCH)."""


# ── Statuses ──────────────────────────────────────────────────────────────

STATUS_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
STATUS_IN_CONTROL = "IN_CONTROL"
STATUS_OUT_OF_CONTROL = "OUT_OF_CONTROL"


# ── Pure statistics ───────────────────────────────────────────────────────


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
    sigma: float,
    k: float = DEFAULT_CUSUM_K,
    h: float = DEFAULT_CUSUM_H,
) -> CusumResult:
    """Standardized two-sided tabular CUSUM.

    zᵢ = (xᵢ − target) / σ
    C⁺ᵢ = max(0, C⁺ᵢ₋₁ + zᵢ − k)   (upward)
    C⁻ᵢ = max(0, C⁻ᵢ₋₁ − zᵢ − k)   (downward)
    A side signals when its statistic exceeds ``h``.

    Requires ``sigma > 0`` (the standardization divides by it) — callers
    gate on the zero-variance case before invoking.
    """
    if sigma <= 0:
        raise ValueError("tabular_cusum needs sigma > 0")
    c_plus = 0.0
    c_minus = 0.0
    breached_high = False
    breached_low = False
    for x in series:
        z = (x - target) / sigma
        c_plus = max(0.0, c_plus + z - k)
        c_minus = max(0.0, c_minus - z - k)
        if c_plus > h:
            breached_high = True
        if c_minus > h:
            breached_low = True
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
    n_points: int
    center: Optional[float] = None
    sigma: Optional[float] = None
    lcl: Optional[float] = None
    ucl: Optional[float] = None
    latest: Optional[float] = None
    latest_z: Optional[float] = None      # None when sigma == 0
    shewhart_low: bool = False            # latest < LCL (regression)
    shewhart_high: bool = False           # latest > UCL (observability)
    cusum_c_minus: Optional[float] = None
    cusum_c_plus: Optional[float] = None
    cusum_low: bool = False               # downward drift (regression)
    cusum_high: bool = False              # upward drift (observability)
    reasons: list[str] = field(default_factory=list)


def evaluate_series(
    series: list[float],
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    k_sigma: float = DEFAULT_K_SIGMA,
    cusum_k: float = DEFAULT_CUSUM_K,
    cusum_h: float = DEFAULT_CUSUM_H,
) -> ControlBandResult:
    """Run the Shewhart + CUSUM charts on one time-ordered series.

    ``series`` is the combo's weekly ``agent_quality_score_4w_mean``
    points, oldest first. The center + σ are estimated from the baseline
    (all points except the latest), and the latest point is the one
    under test (Phase-I baseline / Phase-II monitoring split).
    """
    n = len(series)
    if n < min_history:
        return ControlBandResult(
            status=STATUS_INSUFFICIENT_HISTORY,
            n_points=n,
            latest=series[-1] if series else None,
            reasons=[f"insufficient_history: {n} < {min_history} points"],
        )

    baseline = series[:-1]
    latest = series[-1]
    center = sum(baseline) / len(baseline)
    sigma = moving_range_sigma(baseline)

    lcl = center - k_sigma * sigma
    ucl = center + k_sigma * sigma

    # Shewhart via direct comparison so it stays correct at sigma == 0
    # (LCL == UCL == center; any deviation is a breach).
    shewhart_low = latest < lcl
    shewhart_high = latest > ucl

    latest_z: Optional[float] = None
    cusum: Optional[CusumResult] = None
    if sigma > 0:
        latest_z = (latest - center) / sigma
        # Run CUSUM over the full series against the baseline center.
        cusum = tabular_cusum(
            series, target=center, sigma=sigma, k=cusum_k, h=cusum_h,
        )

    cusum_low = bool(cusum and cusum.breached_low)
    cusum_high = bool(cusum and cusum.breached_high)

    reasons: list[str] = []
    if shewhart_low:
        reasons.append(
            f"shewhart_low: latest {latest:.3f} < LCL {lcl:.3f} "
            f"(center {center:.3f}, sigma {sigma:.3f})"
        )
    if cusum_low:
        reasons.append(
            f"cusum_low: C- {cusum.c_minus:.2f} > h {cusum_h:.1f} "
            f"(sustained downward drift from center {center:.3f})"
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

    out_of_control = shewhart_low or cusum_low
    return ControlBandResult(
        status=STATUS_OUT_OF_CONTROL if out_of_control else STATUS_IN_CONTROL,
        n_points=n,
        center=center,
        sigma=sigma,
        lcl=lcl,
        ucl=ucl,
        latest=latest,
        latest_z=latest_z,
        shewhart_low=shewhart_low,
        shewhart_high=shewhart_high,
        cusum_c_minus=cusum.c_minus if cusum else None,
        cusum_c_plus=cusum.c_plus if cusum else None,
        cusum_low=cusum_low,
        cusum_high=cusum_high,
        reasons=reasons,
    )


# ── CloudWatch series extraction ──────────────────────────────────────────


def _weekly_series_by_combo(
    metric_data_results: list[dict[str, Any]],
    combos: list[list[dict[str, str]]],
    *,
    reset_before: Optional[datetime] = None,
) -> dict[int, list[float]]:
    """Map combo index → its weekly series, oldest-first.

    CloudWatch returns Timestamps (descending by default) paired with
    Values; we zip + sort ascending so the moving range / CUSUM see the
    series in chronological order. When ``reset_before`` is set, points
    older than it are dropped so the baseline doesn't straddle a judge
    re-anchor (L4578(a)).
    """
    by_id = {r["Id"]: r for r in metric_data_results}
    series_by_combo: dict[int, list[float]] = {}
    for idx in range(len(combos)):
        result = by_id.get(f"m{idx}")
        if result is None:
            series_by_combo[idx] = []
            continue
        pairs = list(zip(result.get("Timestamps", []), result.get("Values", [])))
        if reset_before is not None:
            pairs = [(t, v) for t, v in pairs if t >= reset_before]
        pairs.sort(key=lambda tv: tv[0])  # chronological
        series_by_combo[idx] = [float(v) for _, v in pairs]
    return series_by_combo


# ── Orchestration ─────────────────────────────────────────────────────────


def compute_and_emit_control_bands(
    *,
    end_time: Optional[datetime] = None,
    namespace: str = DEFAULT_NAMESPACE,
    source_metric: str = SOURCE_METRIC_NAME,
    min_history: int = DEFAULT_MIN_HISTORY,
    reset_before: Optional[datetime] = None,
    cloudwatch_client: Optional[Any] = None,
    s3_client: Optional[Any] = None,
) -> dict[str, Any]:
    """Evaluate control bands per combo and emit metrics + breach entries.

    Reads the weekly ``agent_quality_score_4w_mean`` series per combo over
    the trailing ``LOOKBACK_WEEKS`` window, runs the Shewhart + CUSUM
    charts, and emits:

    * per-combo ``agent_quality_score_zscore`` (when σ > 0),
    * a single dimensionless ``agent_quality_score_control_breach_count``
      = number of OUT_OF_CONTROL combos (the alarm surface),
    * one changelog entry per OUT_OF_CONTROL combo.

    ``reset_before`` (UTC) trims each combo's series to points at/after it
    — the operator sets it after a judge re-anchor (L4578(a)) so the
    baseline doesn't straddle a model change.
    """
    cw = cloudwatch_client or boto3.client("cloudwatch")
    s3 = s3_client or boto3.client("s3")
    end = end_time or datetime.now(timezone.utc)
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
    )

    zscore_data: list[dict[str, Any]] = []
    breaches: list[dict[str, Any]] = []
    breach_emits: list[str] = []
    n_insufficient = 0
    n_in_control = 0
    failed: list[dict[str, str]] = []

    for idx, dims in enumerate(combos):
        series = series_by_combo.get(idx, [])
        try:
            result = evaluate_series(series, min_history=min_history)
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

    # Emit the breach-count alarm surface every run (0 included).
    breach_count = len(breaches)
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[{
            "MetricName": BREACH_COUNT_METRIC_NAME,
            "Value": float(breach_count),
            "Unit": "None",
            "Timestamp": end,
        }],
    )

    logger.info(
        "[control_bands] done combos=%d in_control=%d insufficient=%d "
        "out_of_control=%d zscores_emitted=%d failed=%d",
        len(combos), n_in_control, n_insufficient, breach_count,
        len(zscore_data), len(failed),
    )

    return {
        "combos_discovered": len(combos),
        "combos_in_control": n_in_control,
        "combos_insufficient_history": n_insufficient,
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
) -> Optional[str]:
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
        event_hash = sha1(digest_input).hexdigest()[:7]
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
            f"Latest 4w-mean: {result.latest}\n"
            f"Center: {result.center}  sigma: {result.sigma}\n"
            f"LCL: {result.lcl}  UCL: {result.ucl}\n"
            f"Latest z: {result.latest_z}\n"
            f"CUSUM C-: {result.cusum_c_minus}  C+: {result.cusum_c_plus}\n"
            f"Reasons: {result.reasons}\n"
            f"n_points: {result.n_points}\n"
            f"Window: {window_start.isoformat()} -> {window_end.isoformat()}\n"
            f"Detected by: alpha-engine-research evals/control_bands.py "
            f"(Shewhart individuals + tabular CUSUM)."
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
                "lcl": result.lcl,
                "ucl": result.ucl,
                "latest_z": result.latest_z,
                "cusum_c_minus": result.cusum_c_minus,
                "shewhart_low": result.shewhart_low,
                "cusum_low": result.cusum_low,
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
