"""Rolling-4-week-mean derived CloudWatch metric.

CloudWatch alarms cap evaluation period at 24 hours, so a true
4-week rolling alarm on the raw ``AlphaEngine/Eval/agent_quality_score``
metric isn't expressible directly. This module computes the rolling
mean server-side and emits a derived metric
``AlphaEngine/Eval/agent_quality_score_4w_mean`` that a regular
CloudWatch alarm can fire against.

Per ROADMAP §1634:
  Alarm threshold on rolling-4-week-mean < 3.0 emits SNS.

Run cadence: weekly, after the eval-judge Lambda has emitted the
current week's raw metric.

Discovery flow:
  1. ListMetrics paged through ``AlphaEngine/Eval / agent_quality_score``
     to enumerate all (judged_agent_id, criterion, judge_model) combos.
  2. GetMetricData with one query per combo for the 4-week window
     ending at ``end_time``, ``Stat=Average`` and ``Period`` covering
     the full window — returns one mean per combo.
  3. PutMetricData with derived metric ``agent_quality_score_4w_mean``
     under the same namespace, carrying the same dimension shape so
     a single alarm can pivot on dimensions without changing scope.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any, Optional

import boto3

from evals.metrics import DEFAULT_NAMESPACE, DEFAULT_METRIC_NAME

logger = logging.getLogger(__name__)


# Schema-1.0.0 changelog corpus (ROADMAP P0 sub-item 5 eval-regression
# half — Item 6 LLM-as-judge eval pipeline shipped via the 2026-05-06
# evaluator-revamp arc; this is the regression-detection + auto-emit
# layer on top). Mirrors cost-anomaly auto-emit pattern from
# alpha-engine-backtester#152.
_CHANGELOG_BUCKET = os.environ.get("CHANGELOG_BUCKET", "alpha-engine-research")
_CHANGELOG_PREFIX = "changelog/entries"
_CHANGELOG_SCHEMA_VERSION = "1.0.0"

_REGRESSION_THRESHOLD_ENV_VAR = "ALPHA_ENGINE_EVAL_REGRESSION_THRESHOLD"
_REGRESSION_THRESHOLD_DEFAULT = 3.0
"""Per ROADMAP §1634: alarm threshold on rolling-4-week-mean < 3.0."""

_FLOOR_MIN_SAMPLES_ENV_VAR = "ALPHA_ENGINE_EVAL_FLOOR_MIN_N"
_FLOOR_MIN_SAMPLES_DEFAULT = 3
"""Minimum judged samples a combo needs in the window to participate in the
ALARMED floor metric (and the regression auto-emit). Root cause of the
perpetual quality-floor ALARM (investigated 2026-06-11, config#660-era
L4578e-floor-alarm-firing): the combo matrix is per-TICKER
(``thesis_update:financials:BRO`` x criterion x judge), so many combos carry a
SINGLE judged sample per 4-week window — one thesis update scoring the rubric
minimum on one criterion pinned the floor at 1.0 indefinitely. A pager metric
cannot run on single-sample means (mirrors the evaluator MetricRecord
N-honesty discipline). Per-combo derived emission is UNCHANGED — dashboards
still see every combo; only the alarmed floor + auto-emit gain the N gate."""

_REGRESSION_EMIT_CAP_ENV_VAR = "ALPHA_ENGINE_EVAL_REGRESSION_EMIT_CAP"
_REGRESSION_EMIT_CAP_DEFAULT = 20
"""Per-run cap on regression changelog auto-emits (the deployed Lambda had
never run the auto-emit path live; an uncapped first run over the historical
low-N tail would spam dozens of entries). Dropped emits are LOGGED with a
count — never silently truncated."""


DERIVED_METRIC_NAME = "agent_quality_score_4w_mean"
"""Per-combo derived metric — one datapoint per (judged_agent_id,
criterion, judge_model) combo. Powers the dashboard's per-combo
trend lines."""

DERIVED_FLOOR_METRIC_NAME = "agent_quality_score_4w_mean_min"
"""Dimensionless floor metric — single datapoint per run = MIN across
every combo's 4-week mean. The CloudWatch alarm fires against this
metric. Reason: CloudWatch alarms reject SEARCH expressions
(``ValidationError: SEARCH is not supported on Metric Alarms``), so
we can't dynamically reduce across combos at alarm-evaluation time;
we have to pre-compute the floor at emission time and alarm on the
single derived stream. Trade-off: the alarm message says "the floor
dropped below 3.0" but not WHICH combo triggered — operator clicks
dashboard to find out. Same operator workflow a SEARCH-based design
would have produced (alarms on a reduced time series carry no
per-combo identity in the alarm body anyway)."""

ROLLING_WINDOW_DAYS = 28
"""4-week (28-day) window — matches ROADMAP §1634 wording."""


def _list_metric_combos(
    cw: Any,
    *,
    namespace: str,
    metric_name: str,
) -> list[list[dict[str, str]]]:
    """Enumerate every distinct dimension-tuple under (namespace,
    metric_name). Each entry in the returned list is the ordered
    ``Dimensions`` list for that combo, ready to pass back to
    GetMetricData / PutMetricData."""
    paginator = cw.get_paginator("list_metrics")
    combos: list[list[dict[str, str]]] = []
    for page in paginator.paginate(Namespace=namespace, MetricName=metric_name):
        for m in page.get("Metrics", []):
            dims = m.get("Dimensions", [])
            if dims:
                combos.append(dims)
    return combos


_GET_METRIC_DATA_MAX_QUERIES = 500
"""AWS hard cap on ``MetricDataQueries`` per ``GetMetricData`` call.

Exceeding this raises ``ValidationError: The collection
MetricDataQueries must not have a size greater than 500.`` As the
rubric × dimension matrix grew (more sector teams / sub-agents /
criteria / judge models) ``len(queries)`` crossed 500, which made the
Saturday-SF ``EvalRollingMean`` state return ``{"status":"ERROR"}``
(observed on the 2026-05-17 weekend run). ``_get_metric_data_all``
chunks below this cap and paginates each chunk."""


def _get_metric_data_all(
    cw: Any,
    queries: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Run ``GetMetricData`` over arbitrarily many queries.

    AWS caps ``MetricDataQueries`` at ``_GET_METRIC_DATA_MAX_QUERIES``
    (500) per call, and each call may itself paginate via ``NextToken``.
    This chunks ``queries`` into ≤500-query batches, follows
    ``NextToken`` to exhaustion within every chunk, and returns the
    flat-merged ``MetricDataResults`` across all chunks/pages.

    The flat merge is correct because every query Id (``m{idx}``, idx
    over the whole ``combos`` list) is globally unique, so the
    downstream ``by_id = {r["Id"]: r for r in ...}`` mapping stays
    intact regardless of which chunk/page a result came back on. The
    ≤500 case is unchanged behaviourally: a single chunk, paginated
    only if AWS returns a ``NextToken``.
    """
    merged: list[dict[str, Any]] = []
    for chunk_start in range(0, len(queries), _GET_METRIC_DATA_MAX_QUERIES):
        chunk = queries[chunk_start:chunk_start + _GET_METRIC_DATA_MAX_QUERIES]
        next_token: Optional[str] = None
        while True:
            kwargs: dict[str, Any] = {
                "MetricDataQueries": chunk,
                "StartTime": start,
                "EndTime": end,
            }
            if next_token is not None:
                kwargs["NextToken"] = next_token
            response = cw.get_metric_data(**kwargs)
            merged.extend(response.get("MetricDataResults", []))
            next_token = response.get("NextToken")
            if not next_token:
                break
    return merged


def _build_metric_data_queries(
    combos: list[list[dict[str, str]]],
    *,
    namespace: str,
    metric_name: str,
    period_seconds: int,
) -> list[dict[str, Any]]:
    """One GetMetricData query per combo. The query Id is the combo
    index — used to map results back to dimensions on the response side.
    CloudWatch caps queries at 500 per call; ``_get_metric_data_all``
    chunks + paginates so the matrix can grow past 500 combos safely.
    """
    queries: list[dict[str, Any]] = []
    for idx, dims in enumerate(combos):
        metric = {
            "Namespace": namespace,
            "MetricName": metric_name,
            "Dimensions": dims,
        }
        queries.append({
            "Id": f"m{idx}",
            "MetricStat": {"Metric": metric, "Period": period_seconds, "Stat": "Average"},
            "ReturnData": True,
        })
        # Paired sample-count query — powers the floor's min-N gate.
        queries.append({
            "Id": f"n{idx}",
            "MetricStat": {"Metric": metric, "Period": period_seconds, "Stat": "SampleCount"},
            "ReturnData": True,
        })
    return queries


def compute_and_emit_4w_mean(
    *,
    end_time: Optional[datetime] = None,
    namespace: str = DEFAULT_NAMESPACE,
    source_metric: str = DEFAULT_METRIC_NAME,
    derived_metric: str = DERIVED_METRIC_NAME,
    floor_metric: str = DERIVED_FLOOR_METRIC_NAME,
    cloudwatch_client: Optional[Any] = None,
    s3_client: Optional[Any] = None,
) -> dict[str, Any]:
    """Compute and emit the rolling-4-week-mean derived metric.

    Args:
        end_time: window ends at this UTC instant (defaults to now).
            Window starts at ``end_time - ROLLING_WINDOW_DAYS``.
        namespace: source + derived metric namespace.
        source_metric: raw metric to roll up.
        derived_metric: name to push the mean under.
        cloudwatch_client: injected client for tests; production None.
        s3_client: injected client for the regression auto-emit hook
            (ROADMAP P0 sub-item 5 eval half). Tests inject; production
            None → real boto3 client.

    Returns:
        Summary dict: combos discovered, derived datapoints emitted,
        combos skipped (no data in window), any per-combo errors, and
        the list of S3 keys for any regression entries auto-emitted to
        the system-wide changelog corpus.
    """
    cw = cloudwatch_client or boto3.client("cloudwatch")
    s3 = s3_client or boto3.client("s3")
    end = end_time or datetime.now(timezone.utc)
    start = end - timedelta(days=ROLLING_WINDOW_DAYS)
    period_seconds = ROLLING_WINDOW_DAYS * 86400  # one datapoint per window

    combos = _list_metric_combos(
        cw, namespace=namespace, metric_name=source_metric,
    )
    if not combos:
        logger.warning(
            "[rolling_mean] no metric streams found under %s/%s — "
            "first-run state, nothing to roll up yet",
            namespace, source_metric,
        )
        return {
            "combos_discovered": 0,
            "datapoints_emitted": 0,
            "combos_skipped_no_data": 0,
            "failed": [],
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "floor_value": None,
            "floor_metric_emitted": False,
        }

    queries = _build_metric_data_queries(
        combos,
        namespace=namespace,
        metric_name=source_metric,
        period_seconds=period_seconds,
    )

    logger.info(
        "[rolling_mean] querying %d combos window=[%s, %s]",
        len(combos), start.isoformat(), end.isoformat(),
    )

    # AWS caps MetricDataQueries at 500 per GetMetricData call; chunk +
    # paginate so the rubric × dimension matrix can grow past 500
    # combos without raising ValidationError (the 2026-05-17 Sat-SF
    # EvalRollingMean ERROR root cause).
    metric_data_results = _get_metric_data_all(cw, queries, start, end)

    derived_data: list[dict[str, Any]] = []
    combo_stats: list[dict[str, Any]] = []
    skipped_no_data = 0
    failed: list[dict[str, str]] = []

    # Map response Id back to combo dimensions for the derived emission.
    # Ids (m{idx}) are unique across the whole combos list, so a flat
    # merge across chunks/pages is correct.
    by_id = {r["Id"]: r for r in metric_data_results}

    for idx, dims in enumerate(combos):
        result = by_id.get(f"m{idx}")
        if result is None:
            failed.append({
                "combo_idx": str(idx),
                "stage": "get_metric_data",
                "error": "missing result for query Id",
            })
            continue

        values = result.get("Values", [])
        if not values:
            # No data in the window — combo first appeared this week
            # or the agent stopped emitting. Skip silently; alarm
            # later if persistent.
            skipped_no_data += 1
            continue

        # CloudWatch returns multiple values if our Period < window.
        # We use Period == window length so we expect a single value;
        # if for some reason there are multiple, take the latest as
        # the most-current rolling mean.
        mean_value = values[0]

        n_result = by_id.get(f"n{idx}", {})
        n_values = n_result.get("Values", [])
        # Period == window -> a single SampleCount value; sum() tolerates the
        # multi-bucket edge the mean path already tolerates. Missing
        # SampleCount result (e.g. a mock or partial response) -> n=0, which
        # EXCLUDES the combo from the alarmed floor (fail-safe: never let an
        # unverifiable mean page the operator).
        n_samples = int(sum(n_values)) if n_values else 0

        derived_data.append({
            "MetricName": derived_metric,
            "Dimensions": dims,
            "Value": float(mean_value),
            "Unit": "None",
            "Timestamp": end,
        })
        combo_stats.append({"dims": dims, "value": float(mean_value), "n": n_samples})

    if derived_data:
        # Same 1000-entry cap as PutMetricData in metrics.py; chunk
        # defensively in case the corpus grows.
        for chunk_start in range(0, len(derived_data), 1000):
            chunk = derived_data[chunk_start:chunk_start + 1000]
            cw.put_metric_data(Namespace=namespace, MetricData=chunk)

    # Regression auto-emit (ROADMAP P0 sub-item 5 eval half) — for every
    # combo whose 4-week mean falls below the configured threshold,
    # write one schema-1.0.0 entry to the system-wide changelog corpus.
    # Each emit is best-effort + isolated: a write failure on combo N
    # logs WARN but doesn't block combos N+1..M or the metric-emission
    # below.
    threshold = _resolve_regression_threshold()
    min_n = _resolve_floor_min_samples()
    emit_cap = _resolve_regression_emit_cap()
    regression_emits: list[str] = []
    regression_emits_dropped = 0
    if threshold > 0:
        below = [c for c in combo_stats if c["value"] < threshold and c["n"] >= min_n]
        if len(below) > emit_cap:
            regression_emits_dropped = len(below) - emit_cap
            logger.warning(
                "[rolling_mean] regression auto-emit cap hit: %d combos below "
                "threshold (n>=%d), emitting worst %d, dropping %d (cap via %s)",
                len(below), min_n, emit_cap, regression_emits_dropped,
                _REGRESSION_EMIT_CAP_ENV_VAR,
            )
            below = sorted(below, key=lambda c: c["value"])[:emit_cap]
        for c in below:
            key = _emit_regression_entry(
                dims=c["dims"],
                rolling_mean=c["value"],
                threshold=threshold,
                window_start=start,
                window_end=end,
                s3_client=s3,
            )
            if key:
                regression_emits.append(key)

    # Floor metric: single dimensionless datapoint = MIN across every
    # combo's 4-week mean. The alarm fires against this. Only emitted
    # when at least one combo had data — otherwise there's no floor
    # to compute and emitting None would corrupt the alarm.
    eligible = [c for c in combo_stats if c["n"] >= min_n]
    excluded_low_n = len(combo_stats) - len(eligible)
    floor_combo: Optional[dict[str, Any]] = None
    if eligible:
        floor_combo = min(eligible, key=lambda c: c["value"])
    elif combo_stats:
        # No combo clears the N gate (tiny corpus week). Fall back to the
        # ungated min with a WARN rather than silently emitting nothing —
        # the alarm must never go dark without a trace.
        floor_combo = min(combo_stats, key=lambda c: c["value"])
        logger.warning(
            "[rolling_mean] no combo has n>=%d this window — floor falls "
            "back to the ungated min (combo n=%d)",
            min_n, floor_combo["n"],
        )
    floor_value = floor_combo["value"] if floor_combo else None
    if floor_combo is not None:
        dims_flat = {d["Name"]: d["Value"] for d in floor_combo["dims"]}
        logger.info(
            "[rolling_mean] floor combo: agent=%s criterion=%s judge=%s "
            "mean=%.3f n=%d (min_n=%d, %d low-N combos excluded from floor)",
            dims_flat.get("judged_agent_id", "?"), dims_flat.get("criterion", "?"),
            dims_flat.get("judge_model", "?"), floor_combo["value"],
            floor_combo["n"], min_n, excluded_low_n,
        )
    if floor_value is not None:
        cw.put_metric_data(
            Namespace=namespace,
            MetricData=[{
                "MetricName": floor_metric,
                "Value": float(floor_value),
                "Unit": "None",
                "Timestamp": end,
            }],
        )

    logger.info(
        "[rolling_mean] done emitted=%d skipped_no_data=%d failed=%d "
        "floor=%s regression_emits=%d",
        len(derived_data), skipped_no_data, len(failed),
        f"{floor_value:.3f}" if floor_value is not None else "(no data)",
        len(regression_emits),
    )

    return {
        "combos_discovered": len(combos),
        "datapoints_emitted": len(derived_data),
        "combos_skipped_no_data": skipped_no_data,
        "failed": failed,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "floor_value": floor_value,
        "floor_metric_emitted": floor_value is not None,
        "regression_emits": regression_emits,
        "regression_emits_dropped": regression_emits_dropped,
        "regression_threshold": threshold,
        "floor_min_samples": min_n,
        "floor_combo": (
            {d["Name"]: d["Value"] for d in floor_combo["dims"]} | {"n": floor_combo["n"]}
            if floor_combo is not None else None
        ),
        "floor_low_n_excluded": excluded_low_n,
    }


# ── Regression auto-emit (ROADMAP P0 sub-item 5 eval half) ───────────────

def _resolve_regression_threshold() -> float:
    """Resolve the regression-detection threshold.

    Mirrors the ``ALPHA_ENGINE_COST_ANOMALY_RATIO`` pattern from
    alpha-engine-backtester#152: env var override (test/staging),
    sane default at module level, threshold ≤ 0 disables auto-emit.
    """
    raw = os.environ.get(_REGRESSION_THRESHOLD_ENV_VAR)
    if not raw:
        return _REGRESSION_THRESHOLD_DEFAULT
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[rolling_mean] %s=%r unparseable as float; "
            "falling back to default %s",
            _REGRESSION_THRESHOLD_ENV_VAR, raw, _REGRESSION_THRESHOLD_DEFAULT,
        )
        return _REGRESSION_THRESHOLD_DEFAULT


def _resolve_floor_min_samples() -> int:
    """Min samples for floor/auto-emit participation (env-tunable)."""
    raw = os.environ.get(_FLOOR_MIN_SAMPLES_ENV_VAR)
    try:
        return int(raw) if raw is not None else _FLOOR_MIN_SAMPLES_DEFAULT
    except ValueError:
        logger.warning(
            "[rolling_mean] invalid %s=%r — using default %d",
            _FLOOR_MIN_SAMPLES_ENV_VAR, raw, _FLOOR_MIN_SAMPLES_DEFAULT,
        )
        return _FLOOR_MIN_SAMPLES_DEFAULT


def _resolve_regression_emit_cap() -> int:
    """Per-run cap on regression changelog auto-emits (env-tunable)."""
    raw = os.environ.get(_REGRESSION_EMIT_CAP_ENV_VAR)
    try:
        return int(raw) if raw is not None else _REGRESSION_EMIT_CAP_DEFAULT
    except ValueError:
        logger.warning(
            "[rolling_mean] invalid %s=%r — using default %d",
            _REGRESSION_EMIT_CAP_ENV_VAR, raw, _REGRESSION_EMIT_CAP_DEFAULT,
        )
        return _REGRESSION_EMIT_CAP_DEFAULT


def _dims_to_dict(dims: list[dict[str, str]]) -> dict[str, str]:
    """CloudWatch ``Dimensions`` is a list of ``{Name, Value}`` dicts —
    flatten to ``{Name: Value}`` for the changelog entry's diagnostic block."""
    return {d.get("Name", ""): d.get("Value", "") for d in dims}


def _emit_regression_entry(
    *,
    dims: list[dict[str, str]],
    rolling_mean: float,
    threshold: float,
    window_start: datetime,
    window_end: datetime,
    s3_client: Any,
) -> Optional[str]:
    """Write one schema-1.0.0 ``eval_score_regression`` entry to the
    system-wide changelog corpus.

    Best-effort: any failure logs WARN and returns None — does not
    interrupt sibling combo emits or the metric-emission below.
    Returns the structured S3 key on success.
    """
    try:
        ts = datetime.now(timezone.utc)
        ts_utc = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        entry_date = ts.strftime("%Y-%m-%d")
        ts_id = ts_utc.replace(":", "-").rstrip("Z")
        actor = "alpha-engine-eval-rolling-mean"

        dims_flat = _dims_to_dict(dims)
        agent_id = dims_flat.get("judged_agent_id", "unknown")
        criterion = dims_flat.get("criterion", "unknown")
        judge_model = dims_flat.get("judge_model", "unknown")
        # event_id hashes on the combo identity + window so re-running
        # rolling_mean for the same window produces the same id (overwrite,
        # not duplicate).
        digest_input = (
            f"{agent_id}|{criterion}|{judge_model}|"
            f"{window_start.isoformat()}|{window_end.isoformat()}"
        ).encode()
        event_hash = sha1(digest_input).hexdigest()[:7]
        event_id = f"{ts_id}_{actor}_{event_hash}"

        summary = (
            f"Eval-score regression: {agent_id}/{criterion} "
            f"4w mean = {rolling_mean:.2f} < threshold {threshold:.2f}"
        )[:240]
        description = (
            f"Judged agent: {agent_id}\n"
            f"Criterion: {criterion}\n"
            f"Judge model: {judge_model}\n"
            f"4-week rolling mean: {rolling_mean:.4f}\n"
            f"Threshold: {threshold:.4f}\n"
            f"Window: {window_start.isoformat()} → {window_end.isoformat()}\n"
            f"Detected by: alpha-engine-research evals/rolling_mean.py\n"
            f"Notification surface: AlphaEngine/Eval/agent_quality_score_4w_mean "
            f"CloudWatch metric (per-combo) + agent_quality_score_4w_mean_min "
            f"floor metric + this changelog corpus entry."
        )

        entry = {
            "schema_version": _CHANGELOG_SCHEMA_VERSION,
            "event_id": event_id,
            "ts_utc": ts_utc,
            "event_type": "eval_score_regression",
            "severity": "medium",
            "subsystem": "eval",
            "root_cause_category": "prompt_regression",  # most plausible default; operator overrides via follow-up
            "resolution_type": None,
            "started_at": None,
            "detected_at": ts_utc,
            "resolved_at": None,
            "verified_at": None,
            "summary": summary,
            "description": description,
            "resolution_notes": None,
            "actor": actor,
            "machine": "research:evals/rolling_mean.py",
            "source": "eval-regression-autoemit",
            "auto_emitted": True,
            "git_refs": [],
            "prompt_version": None,
            "run_id": window_end.strftime("%Y-%m-%d"),
            "eval_run_ref": (
                f"s3://alpha-engine-research/decision_artifacts/_eval/"
                f"{window_end.strftime('%Y-%m-%d')}/{agent_id}/"
            ),
            "eval_regression": {
                "judged_agent_id": agent_id,
                "criterion": criterion,
                "judge_model": judge_model,
                "rolling_mean": rolling_mean,
                "threshold": threshold,
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
            "[rolling_mean] regression auto-emit: s3://%s/%s "
            "agent=%s criterion=%s mean=%.2f threshold=%.2f",
            _CHANGELOG_BUCKET, key, agent_id, criterion, rolling_mean, threshold,
        )
        return key
    except Exception as e:
        logger.warning(
            "[rolling_mean] regression auto-emit failed (best-effort, "
            "swallowed): %s",
            e,
        )
        return None
