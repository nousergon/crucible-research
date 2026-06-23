"""
Agent-quality aggregator — emits ``backtest/{date}/agent_quality.json``, the
producer half of the report-card Agent-Quality + Research-output components
(config alpha-engine-config#1149; consumer = crucible-evaluator#59).

Off-hot-path + backfillable by design: it only READS persisted S3 artifacts of
a completed research run (decision_artifacts/_cost_raw, decision_artifacts/_eval,
signals/{date}/signals.json), so it can run after the fact for any past date and
never perturbs the live Saturday pipeline. Mirrors ``scripts/aggregate_costs.py``
(manual CLI now; SF wiring is a follow-up on #1149) and reuses its JSONL readers
so cost is summed identically (same implausible-row filter).

Metrics emitted (each block independently optional — absent input → the block is
omitted and the evaluator grades a precise N/A-MISSING-INPUT, never a fabricated
value):

- ``cost_per_signal``           total run LLM $ / finalized signal count
- ``signal_volume_adequacy``    count of finalized signals (signals.json)
- ``judge_rubric_pass_rate``    % of real judge evals with every rubric dim >= pass
- ``judge_rubric_distribution`` modal-score concentration across all rubric dims
                                (higher = rubric collapse)

Not emitted by this increment (need CloudWatch queries or new instrumentation —
tracked on #1149): ``agent_validation_failure_rate``, ``retry_storm_count``,
``agent_latency_p95`` (CloudWatch ``AlphaEngine/Agents``), ``pillar_emit_coverage``
(``pillar_assessment`` is not persisted in signals.json today). These stay an
honest N/A-MISSING-INPUT on the report card until their increment lands.

Date handling (DATE_CONVENTIONS.md): ``date`` is the TRADING day — it keys the
output path + ``signals/{date}/`` and matches the report card's run_date. The
cost + eval partitions are keyed by the CALENDAR day the run executed, so
``--run-date`` (default = ``date``) selects ``_cost_raw/{run_date}/`` and
``_eval/{run_date}/``. When wired into the pipeline the orchestrator passes both
from ``now_dual()``; for backfill pass both explicitly.

    python scripts/build_agent_quality.py --date 2026-06-12 --run-date 2026-06-13
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections import Counter
from datetime import date as date_type
from typing import Any, Optional

import boto3

# Reuse the cost-aggregator's hardened JSONL readers + implausible-row filter
# so cost is summed identically to the daily cost parquet (SOTA: mirror, do not
# reinvent — CLAUDE.md institutional-default sub-rule).
from scripts.aggregate_costs import (
    _is_plausible_cost_row,
    _list_jsonl_keys,
    _read_jsonl_rows,
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "alpha-engine-research"
_COST_RAW_PREFIX = "decision_artifacts/_cost_raw"
_EVAL_PREFIX = "decision_artifacts/_eval"
_OUTPUT_PREFIX = "backtest"

# A rubric dimension "passes" at score >= 3 (mirrors the judge's own
# escalate-if-any-dim-below-3 gate, evals.orchestrator.DEFAULT_HAIKU_ESCALATE_THRESHOLD);
# an eval passes iff every dimension passes.
_RUBRIC_PASS_THRESHOLD = 3

# Per-agent runtime telemetry namespace (graph/agent_telemetry.py). Dimensioned
# by {agent_id, env} since config#1154 — we read env="prod" only to skip the
# test pollution on the legacy agent_id-only series.
_AGENTS_NAMESPACE = "AlphaEngine/Agents"


def _day_window(run_date: date_type):
    """UTC [00:00, +1d) window for the run day's CW aggregation."""
    from datetime import datetime, timedelta, timezone

    start = datetime(run_date.year, run_date.month, run_date.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _agent_validation_failure_rate(cw: Any, run_date: date_type) -> Optional[dict]:
    """Fleet agent validation-failure rate over the run day, PROD-only (config#1154).

    ``sum(Failures) / sum(Invocations)`` across every ``agent_id`` with
    ``env="prod"`` — a CW metric-math SUM(SEARCH(...)) collapses the per-agent
    series. Returns ``{"value": rate, "n": invocations}`` or ``None`` when there
    are no prod invocations in the window (no run that day / pre-env-dimension
    data). Best-effort: the caller swallows CW errors so a missing
    cloudwatch:GetMetricData grant or throttle never breaks the artifact."""
    start, end = _day_window(run_date)

    def _sum(metric: str) -> float:
        expr = (
            f"SUM(SEARCH('{{{_AGENTS_NAMESPACE},agent_id,env}} "
            f"MetricName=\"{metric}\" env=\"prod\"', 'Sum', 86400))"
        )
        resp = cw.get_metric_data(
            MetricDataQueries=[{"Id": "q", "Expression": expr, "ReturnData": True}],
            StartTime=start, EndTime=end,
        )
        results = resp.get("MetricDataResults") or [{}]
        return float(sum(results[0].get("Values") or []))

    invocations = _sum("Invocations")
    if invocations <= 0:
        return None
    failures = _sum("Failures")
    return {"value": round(failures / invocations, 4), "n": int(invocations)}


def _list_prod_agent_ids(cw: Any, metric: str) -> list[str]:
    """``agent_id`` values that emitted ``metric`` with ``env="prod"`` (per CW
    list_metrics ~2-week retention). The env=prod dimension filter excludes the
    test-polluted agent_id-only series (config#1154)."""
    agents: set[str] = set()
    paginator = cw.get_paginator("list_metrics")
    for page in paginator.paginate(
        Namespace=_AGENTS_NAMESPACE, MetricName=metric,
        Dimensions=[{"Name": "env", "Value": "prod"}],
    ):
        for m in page.get("Metrics", []):
            for d in m.get("Dimensions", []):
                if d.get("Name") == "agent_id":
                    agents.add(d["Value"])
    return sorted(agents)


def _per_agent_stat(cw: Any, metric: str, agent_ids: list[str], stat: str,
                    start, end) -> dict[str, list[float]]:
    """``{agent_id: [values]}`` for ``metric`` at ``stat`` (e.g. ``"Sum"``,
    ``"p95"``) over the window, env=prod. One GetMetricData query per agent
    (≤ ~10 agents ≪ the 500-query cap)."""
    if not agent_ids:
        return {}
    queries = [
        {
            "Id": f"a{i}",
            "MetricStat": {
                "Metric": {
                    "Namespace": _AGENTS_NAMESPACE, "MetricName": metric,
                    "Dimensions": [
                        {"Name": "agent_id", "Value": a},
                        {"Name": "env", "Value": "prod"},
                    ],
                },
                "Period": 86400, "Stat": stat,
            },
            "ReturnData": True,
        }
        for i, a in enumerate(agent_ids)
    ]
    resp = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=end)
    out: dict[str, list[float]] = {}
    for r in resp.get("MetricDataResults") or []:
        try:
            idx = int(str(r.get("Id", "a-1"))[1:])
        except ValueError:
            continue
        vals = r.get("Values") or []
        if vals and 0 <= idx < len(agent_ids):
            out[agent_ids[idx]] = [float(v) for v in vals]
    return out


def _retry_storm_count(cw: Any, run_date: date_type) -> Optional[dict]:
    """# of agents that hit their retry ceiling, PROD-only (config#1149).

    An agent "reached the ceiling" when it fired a retry that did NOT recover —
    i.e. ``sum(RetryAttempts) > sum(RetrySuccesses)`` over the window (a fired
    retry still produced empty output). ``n`` = agents observed. ``None`` when no
    prod retry telemetry exists this window."""
    start, end = _day_window(run_date)
    agents = _list_prod_agent_ids(cw, "RetryAttempts")
    if not agents:
        return None
    attempts = _per_agent_stat(cw, "RetryAttempts", agents, "Sum", start, end)
    successes = _per_agent_stat(cw, "RetrySuccesses", agents, "Sum", start, end)
    storm = sum(
        1 for a in agents if sum(attempts.get(a, [])) > sum(successes.get(a, []))
    )
    return {"value": storm, "n": len(agents)}


def _agent_latency_p95(cw: Any, run_date: date_type) -> Optional[dict]:
    """Worst per-agent-type p95 wall-clock (ms), PROD-only (config#1149).

    CW gives a p95 PER agent_id; the report-card value is the MAX across agent
    types — the slowest agent's tail, which is what flags latency creep. ``n`` =
    agent types. ``None`` when no prod duration telemetry exists this window."""
    start, end = _day_window(run_date)
    agents = _list_prod_agent_ids(cw, "DurationMs")
    if not agents:
        return None
    p95s = _per_agent_stat(cw, "DurationMs", agents, "p95", start, end)
    per_agent_max = [max(v) for v in p95s.values() if v]
    if not per_agent_max:
        return None
    return {"value": round(max(per_agent_max), 1), "n": len(p95s)}


def _get_json(s3: Any, bucket: str, key: str) -> Optional[dict]:
    """Read one JSON object, or None if absent. Raises on any other S3 error."""
    from botocore.exceptions import ClientError

    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


def _total_cost_usd(s3: Any, bucket: str, run_date: date_type) -> Optional[float]:
    """Sum ``cost_usd`` over the run's _cost_raw JSONL (None if no rows)."""
    prefix = f"{_COST_RAW_PREFIX}/{run_date.isoformat()}/"
    keys = _list_jsonl_keys(s3, bucket, prefix)
    if not keys:
        return None
    rows: list[dict] = []
    for key in keys:
        rows.extend(_read_jsonl_rows(s3, bucket, key))
    clean = [r for r in rows if _is_plausible_cost_row(r)[0]]
    if not clean:
        return None
    return float(sum(float(r.get("cost_usd") or 0.0) for r in clean))


def _load_signals(s3: Any, bucket: str, date_str: str) -> Optional[dict]:
    """The run's per-ticker signals dict ({ticker: {...}}), or None."""
    doc = _get_json(s3, bucket, f"signals/{date_str}/signals.json")
    if not doc:
        return None
    sig = doc.get("signals")
    return sig if isinstance(sig, dict) and sig else None


def _load_evals(s3: Any, bucket: str, run_date: date_type) -> list[dict]:
    """All RubricEvalArtifact JSONs under ``_eval/{run_date}/`` (any subdir)."""
    prefix = f"{_EVAL_PREFIX}/{run_date.isoformat()}/"
    artifacts: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            doc = _get_json(s3, bucket, key)
            if doc is not None:
                artifacts.append(doc)
    return artifacts


def build_agent_quality(
    s3: Any,
    bucket: str,
    target_date: date_type,
    *,
    run_date: Optional[date_type] = None,
    cw: Any = None,
) -> dict:
    """Compute the agent-quality artifact for one research run.

    ``target_date`` is the trading day (output path + signals); ``run_date``
    (default = ``target_date``) is the calendar day keying the cost + eval
    partitions.
    """
    run_date = run_date or target_date
    date_str = target_date.isoformat()
    result: dict[str, Any] = {"status": "ok", "date": date_str, "run_date": run_date.isoformat()}

    # signals — finalized per-ticker decisions.
    signals = _load_signals(s3, bucket, date_str)
    n_signals = len(signals) if signals else 0
    if n_signals:
        result["signal_volume_adequacy"] = {"value": n_signals, "n": n_signals}

    # cost_per_signal — total run cost / finalized signal count.
    total_cost = _total_cost_usd(s3, bucket, run_date)
    if total_cost is not None and n_signals:
        result["cost_per_signal"] = {
            "value": round(total_cost / n_signals, 4),
            "n": n_signals,
            "total_cost_usd": round(total_cost, 4),
        }

    # judge rubric metrics — over REAL evals (skip-markers carry an empty
    # dimension_scores + a judge_skip_reason; they are excluded).
    evals = _load_evals(s3, bucket, run_date)
    real = [
        e for e in evals
        if not e.get("judge_skip_reason") and (e.get("dimension_scores") or [])
    ]
    if real:
        n_eval = len(real)
        passes = sum(
            1 for e in real
            if all(int(d.get("score", 0)) >= _RUBRIC_PASS_THRESHOLD for d in e["dimension_scores"])
        )
        result["judge_rubric_pass_rate"] = {"value": round(passes / n_eval, 4), "n": n_eval}

        all_scores = [int(d.get("score", 0)) for e in real for d in e["dimension_scores"]]
        if all_scores:
            modal_concentration = max(Counter(all_scores).values()) / len(all_scores)
            result["judge_rubric_distribution"] = {
                "value": round(modal_concentration, 4),
                "n": n_eval,
            }

    # Agent runtime metrics from the AlphaEngine/Agents prod telemetry
    # (config#1154/#1149): validation-failure rate (fleet), retry-storm count +
    # latency p95 (per-agent). Best-effort — a CW error or absent prod data leaves
    # a component off the artifact → grader renders N/A, never breaks the others.
    try:
        cw_client = cw or boto3.client("cloudwatch", region_name="us-east-1")
        for key, fn in (
            ("agent_validation_failure_rate", _agent_validation_failure_rate),
            ("retry_storm_count", _retry_storm_count),
            ("agent_latency_p95", _agent_latency_p95),
        ):
            try:
                blk = fn(cw_client, run_date)
            except Exception as exc:  # noqa: BLE001 — per-metric isolation
                logger.warning("[agent_quality] %s read failed: %s", key, exc)
                continue
            if blk is not None:
                result[key] = blk
    except Exception as exc:  # noqa: BLE001 — CW client creation failed
        logger.warning("[agent_quality] cloudwatch client unavailable: %s", exc)

    return result


def write_agent_quality(s3: Any, bucket: str, artifact: dict) -> str:
    """Persist the artifact to ``backtest/{date}/agent_quality.json``; returns key."""
    key = f"{_OUTPUT_PREFIX}/{artifact['date']}/agent_quality.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(artifact, indent=2).encode(),
        ContentType="application/json",
    )
    logger.info("[agent_quality] wrote s3://%s/%s", bucket, key)
    return key


def _parse_date(s: str) -> date_type:
    return date_type.fromisoformat(s)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the report-card agent_quality.json artifact.")
    parser.add_argument("--bucket", default=_DEFAULT_BUCKET)
    parser.add_argument("--date", required=True, type=_parse_date,
                        help="Trading day (keys output path + signals/).")
    parser.add_argument("--run-date", default=None, type=_parse_date,
                        help="Calendar run day (keys _cost_raw/ + _eval/). Default: --date.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute + print the artifact but do NOT write to S3.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s3 = boto3.client("s3")
    artifact = build_agent_quality(s3, args.bucket, args.date, run_date=args.run_date)
    graded = [k for k in artifact if isinstance(artifact[k], dict) and "value" in artifact[k]]
    logger.info("[agent_quality] %d component(s) computed: %s", len(graded), ", ".join(graded) or "(none)")
    json.dump(artifact, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.dry_run:
        logger.info("[agent_quality] --dry-run: not writing to S3")
        return 0
    write_agent_quality(s3, args.bucket, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
