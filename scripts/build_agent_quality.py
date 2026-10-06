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
- ``judge_outcome_ic``          judge-score → realized-outcome validation (old
                                ROADMAP L480 re-scope): date-clustered Spearman
                                rank-IC of per-ticker judge rubric scores vs
                                realized canonical-horizon (21d) log-alpha.
                                FROZEN cross-repo schema — see
                                ``evals/judge_outcome_ic.py``. Observability
                                only; anti-Goodhart stance preserved (judge
                                scores stay OUT of the agent-facing scorecard,
                                per evals/last_week_scorecard.py).

Agent runtime metrics — ``agent_validation_failure_rate``, ``retry_storm_count``,
``agent_latency_p95`` — are read from the live agent path's DECLARED telemetry
(alpha-engine-config-I9631 / I9616): the ``agent_telemetry`` block every Think
Tank run manifest carries (``thinktank/runs/{trading_day}/manifest_*.json``,
schema ``thinktank.schemas.AgentTelemetry``), pooled over the trading-day
partitions of the 7 days ending at ``date``. Every artifact also carries an
``agent_telemetry_source`` block that DECLARES what was read — the window, how
many runs were seen and instrumented, the last instrumented run, and a status —
so a reader can tell "no agent ran", "the runs predate the emitter" and "the
read failed" apart instead of inferring them from an absent key. Only when the
manifests yield no agent call does the producer fall back to the legacy
CloudWatch ``AlphaEngine/Agents`` read, whose emitter hangs off the sector-team
graph retired 2026-07-12 and which has never held a datapoint (I9631).

``pillar_emit_coverage`` is not emitted (``pillar_assessment`` is not persisted
in signals.json today) and stays an honest N/A-MISSING-INPUT.

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
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from datetime import date as date_type
from typing import Any

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

# Live-path agent telemetry (alpha-engine-config-I9631): the Think Tank run
# manifests. Keyed by TRADING day, so a weekend run lands in Friday's partition
# and a 7-calendar-day window ending at ``date`` covers one trading week.
_THINKTANK_RUNS_PREFIX = "thinktank/runs"
_AGENT_TELEMETRY_WINDOW_DAYS = 7
_AGENT_TELEMETRY_SOURCE = "thinktank_run_manifest"
_AGENT_TELEMETRY_OWNER = "crucible-research thinktank/client.py (alpha-engine-config-I9631)"
# A dry run makes no LLM call; counting it would dilute nothing but would
# inflate runs_seen with runs that could never have been instrumented.
_AGENT_TELEMETRY_EXCLUDED_MODES = frozenset({"dry_run"})
_AGENT_TELEMETRY_KEYS = (
    "agent_validation_failure_rate",
    "retry_storm_count",
    "agent_latency_p95",
)


def _p95(samples: list[int]) -> float:
    """Nearest-rank 95th percentile (the sample at rank ceil(0.95·n))."""
    import math

    ordered = sorted(samples)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _load_agent_telemetry(s3: Any, bucket: str, target_date: date_type) -> dict:
    """Pool the ``agent_telemetry`` blocks of every Think Tank run manifest in
    the window. Returns ``{"declaration": {...}, "agents": {agent_id: AgentTelemetry}}``.

    Raises on an S3 error other than a missing key and on a manifest whose
    block does not validate — the caller records that as ``status="error"``.
    """
    from datetime import timedelta

    from thinktank.schemas import merge_agent_telemetry

    start = target_date - timedelta(days=_AGENT_TELEMETRY_WINDOW_DAYS - 1)
    paginator = s3.get_paginator("list_objects_v2")
    runs_seen = 0
    runs_instrumented = 0
    last_run: dict | None = None
    blocks: list[dict] = []
    for offset in range(_AGENT_TELEMETRY_WINDOW_DAYS):
        day = (start + timedelta(days=offset)).isoformat()
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{_THINKTANK_RUNS_PREFIX}/{day}/"):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if not key.rsplit("/", 1)[-1].startswith("manifest_") or not key.endswith(".json"):
                    continue
                manifest = _get_json(s3, bucket, key)
                if not isinstance(manifest, dict):
                    continue
                if manifest.get("mode") in _AGENT_TELEMETRY_EXCLUDED_MODES:
                    continue
                runs_seen += 1
                block = manifest.get("agent_telemetry")
                if block is None:
                    continue  # written before the emitter existed — unmeasured, not zero
                runs_instrumented += 1
                blocks.append(block)
                stamp = (manifest.get("trading_day") or day, manifest.get("finished_at") or "")
                if last_run is None or stamp > (last_run["trading_day"], last_run["finished_at"]):
                    last_run = {
                        "trading_day": stamp[0],
                        "finished_at": stamp[1],
                        "run_id": manifest.get("run_id"),
                        "key": key,
                    }

    agents = merge_agent_telemetry(*blocks)
    invocations = sum(t.invocations for t in agents.values())
    if runs_seen == 0:
        status = "no_runs_in_window"
    elif runs_instrumented == 0:
        status = "not_instrumented"
    elif invocations == 0:
        status = "no_agent_calls"
    else:
        status = "ok"
    declaration = {
        "source": _AGENT_TELEMETRY_SOURCE,
        "owner": _AGENT_TELEMETRY_OWNER,
        "status": status,
        "window": {
            "start": start.isoformat(),
            "end": target_date.isoformat(),
            "basis": f"trading-day partitions of s3://{bucket}/{_THINKTANK_RUNS_PREFIX}/",
        },
        "runs_seen": runs_seen,
        "runs_instrumented": runs_instrumented,
        "invocations": invocations,
        "last_instrumented_run": last_run,
    }
    return {"declaration": declaration, "agents": agents}


def _agent_metrics_from_telemetry(agents: dict) -> dict[str, dict]:
    """The three report-card blocks from pooled per-agent telemetry.

    Same definitions the CloudWatch readers below implement, so a component's
    meaning does not change with its source:

    - failure rate = sum(failures) / sum(invocations), ``n`` = invocations;
    - retry storm = agents with >=1 invocation that retried and still failed
      (hit the retry ceiling), ``n`` = agents observed;
    - latency p95 = the slowest agent's p95 wall-clock (ms), ``n`` = agents with
      a duration sample.
    """
    observed = {a: t for a, t in agents.items() if t.invocations > 0}
    if not observed:
        return {}
    invocations = sum(t.invocations for t in observed.values())
    failures = sum(t.failures for t in observed.values())
    out: dict[str, dict] = {
        "agent_validation_failure_rate": {
            "value": round(failures / invocations, 4),
            "n": invocations,
            "failures": failures,
            "source": _AGENT_TELEMETRY_SOURCE,
        },
        "retry_storm_count": {
            "value": sum(1 for t in observed.values() if t.retry_exhausted > 0),
            "n": len(observed),
            "agents_at_ceiling": sorted(a for a, t in observed.items() if t.retry_exhausted > 0),
            "attempts_unreported": sum(t.attempts_unreported for t in observed.values()),
            "source": _AGENT_TELEMETRY_SOURCE,
        },
    }
    p95_by_agent = {a: _p95(t.durations_ms) for a, t in observed.items() if t.durations_ms}
    if p95_by_agent:
        worst = max(p95_by_agent, key=p95_by_agent.__getitem__)
        out["agent_latency_p95"] = {
            "value": round(p95_by_agent[worst], 1),
            "n": len(p95_by_agent),
            "worst_agent": worst,
            "p95_ms_by_agent": {a: round(v, 1) for a, v in sorted(p95_by_agent.items())},
            "source": _AGENT_TELEMETRY_SOURCE,
        }
    return out


def _day_window(run_date: date_type):
    """UTC [00:00, +1d) window for the run day's CW aggregation."""
    from datetime import datetime, timedelta

    start = datetime(run_date.year, run_date.month, run_date.day, tzinfo=UTC)
    return start, start + timedelta(days=1)


def _agent_validation_failure_rate(cw: Any, run_date: date_type) -> dict | None:
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


def _retry_storm_count(cw: Any, run_date: date_type) -> dict | None:
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


def _agent_latency_p95(cw: Any, run_date: date_type) -> dict | None:
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


def _get_json(s3: Any, bucket: str, key: str) -> dict | None:
    """Read one JSON object, or None if absent. Raises on any other S3 error."""
    from botocore.exceptions import ClientError

    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


def _total_cost_usd(s3: Any, bucket: str, run_date: date_type) -> float | None:
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


def _load_signals(s3: Any, bucket: str, date_str: str) -> dict | None:
    """The run's per-ticker signals dict ({ticker: {...}}), or None."""
    doc = _get_json(s3, bucket, f"signals/{date_str}/signals.json")
    if not doc:
        return None
    sig = doc.get("signals")
    return sig if isinstance(sig, dict) and sig else None


def _load_evals(s3: Any, bucket: str, run_date: date_type) -> list[dict]:
    """All RubricEvalArtifact JSONs under ``_eval/`` in the flat layout
    (config#793 — prior nested layout is no longer emitted).

    Reuses the loader from evals/judge_outcome_ic to maintain consistency
    across the rubric + judge_outcome_ic components. The rubric metrics
    report over all evals in the flat prefix; filtering by run_date is no
    longer applicable with the flat layout since execution date is not
    encoded in the S3 partition (the date semantics are now in each artifact's
    timestamps / capture keys). This is correct for the Saturday reporting
    use case (one research run per Saturday writes all weekly evals)."""
    from evals.judge_outcome_ic import load_eval_artifacts

    t0 = time.monotonic()
    artifacts = load_eval_artifacts(s3, bucket)
    logger.info(
        "[build_agent_quality] loaded %d eval artifacts in %.2fs (ONE scan — "
        "the list is reused by the judge_outcome_ic block, I9205)",
        len(artifacts), time.monotonic() - t0,
    )

    if not artifacts:
        logger.warning(
            "[build_agent_quality] zero evals found in the flat _eval/ prefix "
            "(run_date=%s) — this is expected only on the first run after a "
            "research reset; otherwise check that the judge layer is emitting",
            run_date,
        )

    return artifacts


def _coerce_date(value: date_type | str, field: str) -> date_type:
    """Normalize a fleet date carrier to ``datetime.date``, or fail loud.

    Accepts a ``date`` (returned unchanged) or an ISO ``YYYY-MM-DD`` string
    (as produced by ``krepis.dates.now_dual()``). Anything else raises
    ``TypeError`` naming the field and the offending type — a caller passing
    a ``datetime`` or an epoch int is a real contract violation and must not
    be silently reinterpreted.
    """
    if isinstance(value, date_type) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date_type.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"build_agent_quality({field}=...): expected an ISO YYYY-MM-DD "
                f"date string, got {value!r}"
            ) from exc
    raise TypeError(
        f"build_agent_quality({field}=...): expected datetime.date or an ISO "
        f"YYYY-MM-DD string, got {type(value).__name__}"
    )


def build_agent_quality(
    s3: Any,
    bucket: str,
    target_date: date_type | str,
    *,
    run_date: date_type | str | None = None,
    cw: Any = None,
    outcomes_conn: Any = None,
) -> dict:
    """Compute the agent-quality artifact for one research run.

    ``target_date`` is the trading day (output path + signals); ``run_date``
    (default = ``target_date``) is the calendar day keying the cost + eval
    partitions. ``outcomes_conn`` is an optional open research.db connection
    for the ``judge_outcome_ic`` block (injected in tests / ``--db``); when
    None the block pulls the ``research.db`` S3 snapshot itself.

    Both dates accept either a ``datetime.date`` or a fleet-canonical ISO
    ``YYYY-MM-DD`` string and are normalized here, at the boundary
    (alpha-engine-config-I8177). The fleet's canonical date carrier —
    ``krepis.dates.now_dual()`` — returns a ``DualDate`` whose
    ``trading_day`` / ``calendar_date`` are ISO **strings**, and EVERY
    artifact-write site is instructed to call it. A producer that accepted
    only ``date`` therefore swam against the convention, and the mismatch
    was invisible: annotations are unenforced at runtime, so the weekly
    Lambda passed strings straight through and this function died on
    ``'str' object has no attribute 'isoformat'`` every Saturday from
    2026-06-23 to 2026-08-22 while its caller swallowed the exception.
    Normalizing here makes every current and future ``now_dual()`` caller
    correct by construction rather than one call site at a time.
    """
    target_date = _coerce_date(target_date, "target_date")
    run_date = _coerce_date(run_date, "run_date") if run_date is not None else target_date
    date_str = target_date.isoformat()
    result: dict[str, Any] = {"status": "ok", "date": date_str, "run_date": run_date.isoformat()}

    t_build = time.monotonic()

    # signals — finalized per-ticker decisions.
    t0 = time.monotonic()
    signals = _load_signals(s3, bucket, date_str)
    signals_s = time.monotonic() - t0
    n_signals = len(signals) if signals else 0
    if n_signals:
        result["signal_volume_adequacy"] = {"value": n_signals, "n": n_signals}

    # cost_per_signal — total run cost / finalized signal count.
    t0 = time.monotonic()
    total_cost = _total_cost_usd(s3, bucket, run_date)
    cost_s = time.monotonic() - t0
    if total_cost is not None and n_signals:
        result["cost_per_signal"] = {
            "value": round(total_cost / n_signals, 4),
            "n": n_signals,
            "total_cost_usd": round(total_cost, 4),
        }

    # judge rubric metrics — over REAL evals (skip-markers carry an empty
    # dimension_scores + a judge_skip_reason; they are excluded).
    t0 = time.monotonic()
    evals = _load_evals(s3, bucket, run_date)
    evals_s = time.monotonic() - t0
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

    # judge_outcome_ic — judge-score → realized-outcome validation (old ROADMAP
    # L480 re-scope; evals/judge_outcome_ic.py). Unlike the blocks above this
    # one is NEVER silently absent: absent history is an explicit
    # status="insufficient" from the module, and a genuine failure (broken
    # precondition — S3 listing error, missing research.db snapshot) surfaces
    # as status="error" + the exception string, WARNed with traceback, per the
    # per-block isolation pattern below (mirrors the CW blocks: the failure is
    # recorded on a named surface — block status + WARN — while the sibling
    # components and the artifact write survive; raising here would sink the
    # PRIMARY deliverable for a secondary-observability block, and the
    # rolling-mean handler's own fail-soft wrapper would then hide those too).
    # Consumers treat any status != "ok" as not-gradeable. Fleet fail-hard
    # doctrine (config#1684) is honored inside the module: broken
    # preconditions RAISE out of build_judge_outcome_ic_block rather than
    # degrade into a fabricated "insufficient".
    t0 = time.monotonic()
    try:
        from evals.judge_outcome_ic import build_judge_outcome_ic_block

        # ``evals=`` — the SAME list loaded above. Loading it again here was a
        # duplicated unbounded N+1 prefix scan (~551s each, measured
        # 2026-08-28) that put this block ~1,102s over a 240s handler ceiling
        # and timed out every real execution (alpha-engine-config-I9205).
        result["judge_outcome_ic"] = build_judge_outcome_ic_block(
            s3, bucket, conn=outcomes_conn, evals=evals,
        )
    except Exception as exc:  # noqa: BLE001 — per-block isolation, see above
        logger.warning(
            "[agent_quality] judge_outcome_ic failed (recorded as "
            "status=error, other components unaffected): %s", exc, exc_info=True,
        )
        result["judge_outcome_ic"] = {
            "schema_version": 1, "status": "error", "error": str(exc),
        }
    ic_s = time.monotonic() - t0

    # Agent runtime metrics from the LIVE path's declared telemetry — the Think
    # Tank run manifests' agent_telemetry blocks (alpha-engine-config-I9631).
    # The declaration is written on every artifact, whatever it found, so an
    # absent metric always has a stated reason beside it. Per-block isolation,
    # same as judge_outcome_ic above: a failed read is recorded as
    # status="error" + WARN and the sibling components still land.
    t_tel = time.monotonic()
    try:
        telemetry = _load_agent_telemetry(s3, bucket, target_date)
        result["agent_telemetry_source"] = telemetry["declaration"]
        result.update(_agent_metrics_from_telemetry(telemetry["agents"]))
    except Exception as exc:  # noqa: BLE001 — per-block isolation, see above
        logger.warning(
            "[agent_quality] agent telemetry read failed (recorded as "
            "agent_telemetry_source.status=error, other components "
            "unaffected): %s", exc, exc_info=True,
        )
        result["agent_telemetry_source"] = {
            "source": _AGENT_TELEMETRY_SOURCE,
            "owner": _AGENT_TELEMETRY_OWNER,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    telemetry_s = time.monotonic() - t_tel

    # Legacy fallback: the AlphaEngine/Agents CloudWatch namespace
    # (config#1154/#1149). Consulted only for a metric the manifests did not
    # yield. Its emitter is wired to the retired sector-team graph and the
    # namespace has never held a datapoint (I9631), so on the live fleet this
    # adds nothing; deleting it is I9631's option (b), a ruling not taken here.
    # Best-effort — a CW error leaves the component off, never breaks others.
    t_cw = time.monotonic()
    missing = [k for k in _AGENT_TELEMETRY_KEYS if k not in result]
    if missing:
        try:
            cw_client = cw or boto3.client("cloudwatch", region_name="us-east-1")
            for key, fn in (
                ("agent_validation_failure_rate", _agent_validation_failure_rate),
                ("retry_storm_count", _retry_storm_count),
                ("agent_latency_p95", _agent_latency_p95),
            ):
                if key not in missing:
                    continue
                try:
                    blk = fn(cw_client, run_date)
                except Exception as exc:  # noqa: BLE001 — per-metric isolation
                    logger.warning("[agent_quality] %s read failed: %s", key, exc)
                    continue
                if blk is not None:
                    result[key] = blk
        except Exception as exc:  # noqa: BLE001 — CW client creation failed
            logger.warning("[agent_quality] cloudwatch client unavailable: %s", exc)
    cw_s = time.monotonic() - t_cw

    # Per-phase timings + input counters (alpha-engine-config-I9205
    # deliverable 6). The block's ceiling in lambda/eval_rolling_mean_handler
    # is only re-settable from a COMPLETED run's measurement, and before this
    # line neither prefix scan emitted anything at all — the 240s timeout was
    # unattributable from CloudWatch.
    logger.info(
        "[build_agent_quality] built date=%s run_date=%s n_signals=%d "
        "n_evals=%d n_real_evals=%d ic_status=%s telemetry_status=%s | "
        "signals_s=%.2f cost_s=%.2f evals_s=%.2f judge_outcome_ic_s=%.2f "
        "agent_telemetry_s=%.2f cloudwatch_s=%.2f total_s=%.2f",
        date_str, run_date.isoformat(), n_signals, len(evals), len(real),
        (result.get("judge_outcome_ic") or {}).get("status"),
        (result.get("agent_telemetry_source") or {}).get("status"),
        signals_s, cost_s, evals_s, ic_s, telemetry_s, cw_s,
        time.monotonic() - t_build,
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the report-card agent_quality.json artifact.")
    parser.add_argument("--bucket", default=_DEFAULT_BUCKET)
    parser.add_argument("--date", required=True, type=_parse_date,
                        help="Trading day (keys output path + signals/).")
    parser.add_argument("--run-date", default=None, type=_parse_date,
                        help="Calendar run day (keys _cost_raw/ + _eval/). Default: --date.")
    parser.add_argument("--db", default=None,
                        help="Local research.db path for the judge_outcome_ic "
                             "outcome join. Default: pull the S3 snapshot.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute + print the artifact but do NOT write to S3.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    outcomes_conn = None
    if args.db:
        import sqlite3

        if not os.path.exists(args.db):
            parser.error(f"--db path not found: {args.db}")
        outcomes_conn = sqlite3.connect(args.db)

    s3 = boto3.client("s3")
    artifact = build_agent_quality(
        s3, args.bucket, args.date, run_date=args.run_date,
        outcomes_conn=outcomes_conn,
    )
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
