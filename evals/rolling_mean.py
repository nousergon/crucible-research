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

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3

from evals.metrics import DEFAULT_NAMESPACE, DEFAULT_METRIC_NAME

logger = logging.getLogger(__name__)


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


def _build_metric_data_queries(
    combos: list[list[dict[str, str]]],
    *,
    namespace: str,
    metric_name: str,
    period_seconds: int,
) -> list[dict[str, Any]]:
    """One GetMetricData query per combo. The query Id is the combo
    index — used to map results back to dimensions on the response side.
    CloudWatch caps queries at 500 per call; with ~6 sector teams ×
    3 sub-agents × 4-6 criteria × 2 judge models = ~290 we're well below.
    """
    return [
        {
            "Id": f"m{idx}",
            "MetricStat": {
                "Metric": {
                    "Namespace": namespace,
                    "MetricName": metric_name,
                    "Dimensions": dims,
                },
                "Period": period_seconds,
                "Stat": "Average",
            },
            "ReturnData": True,
        }
        for idx, dims in enumerate(combos)
    ]


def compute_and_emit_4w_mean(
    *,
    end_time: Optional[datetime] = None,
    namespace: str = DEFAULT_NAMESPACE,
    source_metric: str = DEFAULT_METRIC_NAME,
    derived_metric: str = DERIVED_METRIC_NAME,
    floor_metric: str = DERIVED_FLOOR_METRIC_NAME,
    cloudwatch_client: Optional[Any] = None,
) -> dict[str, Any]:
    """Compute and emit the rolling-4-week-mean derived metric.

    Args:
        end_time: window ends at this UTC instant (defaults to now).
            Window starts at ``end_time - ROLLING_WINDOW_DAYS``.
        namespace: source + derived metric namespace.
        source_metric: raw metric to roll up.
        derived_metric: name to push the mean under.
        cloudwatch_client: injected client for tests; production None.

    Returns:
        Summary dict: combos discovered, derived datapoints emitted,
        combos skipped (no data in window), and any per-combo errors.
        Same shape as the eval orchestrator so the SF state can pattern-
        match on ``failed`` to decide alarm severity.
    """
    cw = cloudwatch_client or boto3.client("cloudwatch")
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

    response = cw.get_metric_data(
        MetricDataQueries=queries,
        StartTime=start,
        EndTime=end,
    )

    derived_data: list[dict[str, Any]] = []
    skipped_no_data = 0
    failed: list[dict[str, str]] = []

    # Map response Id back to combo dimensions for the derived emission.
    by_id = {r["Id"]: r for r in response.get("MetricDataResults", [])}

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

        derived_data.append({
            "MetricName": derived_metric,
            "Dimensions": dims,
            "Value": float(mean_value),
            "Unit": "None",
            "Timestamp": end,
        })

    if derived_data:
        # Same 1000-entry cap as PutMetricData in metrics.py; chunk
        # defensively in case the corpus grows.
        for chunk_start in range(0, len(derived_data), 1000):
            chunk = derived_data[chunk_start:chunk_start + 1000]
            cw.put_metric_data(Namespace=namespace, MetricData=chunk)

    # Floor metric: single dimensionless datapoint = MIN across every
    # combo's 4-week mean. The alarm fires against this. Only emitted
    # when at least one combo had data — otherwise there's no floor
    # to compute and emitting None would corrupt the alarm.
    floor_value = (
        min(d["Value"] for d in derived_data) if derived_data else None
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
        "floor=%s",
        len(derived_data), skipped_no_data, len(failed),
        f"{floor_value:.3f}" if floor_value is not None else "(no data)",
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
    }
