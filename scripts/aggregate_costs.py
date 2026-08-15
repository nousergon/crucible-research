"""
Daily LLM cost aggregator — reads per-call JSONL rows from S3 + writes a
single parquet file for analytics.

Manual CLI for now (PR 3 of the cost-telemetry workstream); SF wiring
is a follow-up. Run after a weekday/Saturday SF completes once the
``decision_artifacts/_cost_raw/{date}/...`` keys have been written by
``graph/llm_cost_tracker._flush_cost_rows_to_s3``::

    python scripts/aggregate_costs.py --date 2026-05-02

Reads:    ``s3://alpha-engine-research/decision_artifacts/_cost_raw/{date}/**/*.jsonl``
Writes:   ``s3://alpha-engine-research/decision_artifacts/_cost/{date}/cost.parquet``
Prints:   total cost, breakdown by sector_team / by model / by run_type, plus
          the underlying token totals so cost can be cross-checked against
          a fresh price-table query if rates change later.

Schema posture (matches the JSONL row shape):

- ``schema_version``, ``timestamp``, ``run_id``, ``agent_id``, ``sector_team_id``,
  ``node_name``, ``run_type``, ``prompt_id``, ``prompt_version``,
  ``prompt_version_hash``, ``model_name``, ``call_seq``, ``input_tokens``,
  ``output_tokens``, ``cache_read_tokens``, ``cache_create_tokens``,
  ``web_search_requests`` (schema v2), ``web_fetch_requests`` (schema v2),
  ``cost_usd``.
- ``execution_id`` (schema v3, additive, config#5504): the owning SF execution
  ID — the per-run grouping key that distinguishes N executions on the same
  date. When absent (pre-v3 rows), aggregator groups under "(unknown)" and
  per-execution drilldown is suppressed.
- ``cost_source`` (schema v3, additive, config#5504): ``"llm"`` (LLM API calls,
  the v1/v2 default), ``"ec2_spot"``, or ``"ec2_on_demand"``. EC2 rows carry
  zero token columns and a single ``cost_usd`` — one row per launched instance.
- All additive going forward — never rename or remove a column without a
  ``schema_version`` bump per CLAUDE.md S3 contract safety rules. v1 rows
  predate the tool-fee columns; the aggregator treats missing as zero so
  v1 + v2 rows can be summed in the same daily parquet. v1/v2 rows predate
  ``execution_id`` + ``cost_source``; both are treated as "llm" rows with an
  unknown execution.

Workstream design: ``alpha-engine-config/private-docs/ROADMAP.md`` line ~1708.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import date as date_type
from datetime import timedelta
from typing import Any

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


class CostAggregationError(RuntimeError):
    """Raised when the aggregator read cost rows but could not produce a
    parquet from them.

    Distinct from an empty input, which is a legitimately quiet week and
    returns ``None``. This exception means rows existed and the pipeline
    still produced nothing — a contract violation the producer may not
    swallow (I5206).
    """


_DEFAULT_BUCKET = "alpha-engine-research"
_INPUT_PREFIX = "decision_artifacts/_cost_raw"
_OUTPUT_PREFIX = "decision_artifacts/_cost"

# CloudWatch metric namespace for cost telemetry (Phase 4 #2 + #3).
# Joins ``AlphaEngine/Agents``, ``AlphaEngine/Predictor``,
# ``AlphaEngine/Eval``, etc. as a load-bearing observability surface.
# Per-agent_id dimension lets alarms rank-by-regressor.
_COST_CW_NAMESPACE = "AlphaEngine/Cost"

# NOTE: ``_RUN_ID_RE`` (an ISO-date-prefix test-pollution discriminator)
# was REMOVED 2026-07-28 — see ``_is_plausible_cost_row`` for the incident
# it caused (I5206). Do not reintroduce a run_id-shape check here.

# cost_source values that represent EC2/spot compute (config#5504) — distinct
# from LLM rows, used by the completeness check below so "spot rows stopped
# arriving" cannot read as "the run was cheap."
_EC2_COST_SOURCES = frozenset({"ec2_spot", "ec2_on_demand"})

# Anthropic's largest context window (Claude Opus 4.7) is ~1M tokens.
# A single API response cannot exceed that. 5M is 5x the API ceiling
# and would mark any single call as impossible-from-real-API. This is
# the implausibility threshold — anything above it is test pollution.
_MAX_PLAUSIBLE_TOKENS_PER_ROW = 5_000_000


def _is_plausible_cost_row(row: dict) -> tuple[bool, str | None]:
    """Reject obvious test pollution before it reaches the daily parquet.

    **One** structural invariant any real production row must satisfy:
    every token-count column is below the API ceiling. The 2026-05-13
    pollution carried ``input_tokens=1_000_000_000`` — ~1000x the real
    ceiling, a number no provider could return.

    **The run_id-shape invariant was removed 2026-07-28 (I5206) because it
    silently killed this pipeline for 17 days.** It required ``run_id`` to
    start with an ISO date, on the reasoning that production used
    date-shaped run_ids and test fixtures used ad-hoc strings. Then the
    production run_id format changed: ``eval_judge`` moved to
    artifact-scoped ids like ``276a5be44c7c-EXEL-v5``. From 2026-07-18 the
    guard classified **100% of production rows** as pollution — real model,
    real tokens, real cost — the aggregator wrote no parquet, and the
    dashboard rendered stale data with no signal. The
    ``decision_artifacts/_cost_raw/2026-07-19/`` partition is the evidence:
    intact raw rows that never became a parquet.

    Two lessons encoded here, not just the one fix:

    - **It was a format coupling wearing an invariant's clothes.** An
      incidental naming convention of one producer became a safety
      property, so an unrelated change in a *different* producer disabled
      the pipeline. A real invariant is one the data cannot violate while
      remaining true; a run_id's shape is a choice, not a law.
    - **It was redundant.** The token ceiling alone catches the 2026-05-13
      incident it was written for. It bought nothing and cost everything.

    Discriminating test from production belongs at emission (an explicit
    ``environment`` stamp), or better, at the boundary that let a test run
    write to the production bucket at all — not in a read-time heuristic.
    Tracked on I5206.

    Returns ``(ok, reason)``. ``ok=False`` → drop the row, log reason.
    Pure function — no I/O, deterministic for the same input.
    """
    for col in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_create_tokens"):
        v = row.get(col)
        if v is not None and v > _MAX_PLAUSIBLE_TOKENS_PER_ROW:
            return False, (f"{col}={v:,} exceeds plausible {_MAX_PLAUSIBLE_TOKENS_PER_ROW:,} (API ceiling)")
    return True, None


# ── S3 read helpers ──────────────────────────────────────────────────────


def _list_jsonl_keys(s3_client: Any, bucket: str, prefix: str) -> list[str]:
    """Return all keys under ``prefix`` ending in ``.jsonl``.

    Uses paginated ``ListObjectsV2`` so prefixes with >1000 entries are
    handled correctly. Empty prefix returns an empty list (caller should
    short-circuit).
    """
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key", "")
            if key.endswith(".jsonl"):
                keys.append(key)
    return keys


def _read_jsonl_rows(s3_client: Any, bucket: str, key: str) -> list[dict]:
    """Read a single JSONL object and return its parsed rows.

    Skips blank lines silently (trailing newlines from the writer are
    common and harmless). Raises if a non-blank line fails to parse —
    the JSONL writer is strict + JSON-encoding always round-trips, so
    a parse error indicates corruption worth surfacing.
    """
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    rows = []
    for i, line in enumerate(body.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSONL at s3://{bucket}/{key} line {i}: {exc}") from exc
    return rows


# ── Aggregation ──────────────────────────────────────────────────────────


def aggregate_day(
    s3_client: Any,
    bucket: str,
    target_date: date_type,
    *,
    output_key_override: str | None = None,
    cw_client: Any | None = None,
) -> dict | None:
    """Read all JSONL files for ``target_date`` and write a parquet.

    Returns a summary dict with ``rows_in``, ``rows_out``, ``output_key``,
    ``total_cost_usd``, ``by_team``, ``by_model``, ``by_run_type``,
    or ``None`` if no JSONL files were found for the date (no parquet
    written in that case — distinguished from an empty-data parquet).
    """
    date_str = target_date.isoformat()
    input_prefix = f"{_INPUT_PREFIX}/{date_str}/"
    keys = _list_jsonl_keys(s3_client, bucket, input_prefix)
    if not keys:
        logger.warning(
            "[aggregate_costs] no JSONL files found at s3://%s/%s — nothing to aggregate for %s",
            bucket,
            input_prefix,
            date_str,
        )
        return None

    logger.info(
        "[aggregate_costs] reading %d JSONL files from s3://%s/%s",
        len(keys),
        bucket,
        input_prefix,
    )
    all_rows: list[dict] = []
    for key in keys:
        all_rows.extend(_read_jsonl_rows(s3_client, bucket, key))

    if not all_rows:
        logger.warning(
            "[aggregate_costs] %d JSONL files contained zero rows — skipping parquet write",
            len(keys),
        )
        return None

    # Drop implausible rows (test pollution). Source: 2026-05-13 incident
    # where a unit-test run with real AWS creds wrote ~$1014 of fake-agent
    # rows into the _cost_raw partition, inflating the dashboard's weekly
    # trend chart 700x. The filter is a single structural invariant (token
    # ceiling) — see _is_plausible_cost_row for why the run_id-shape half
    # was removed.
    clean_rows: list[dict] = []
    drop_reasons: list[str] = []
    for row in all_rows:
        ok, reason = _is_plausible_cost_row(row)
        if ok:
            clean_rows.append(row)
        elif len(drop_reasons) < 10:  # cap log noise
            drop_reasons.append(reason or "implausible")
    n_dropped = len(all_rows) - len(clean_rows)
    if n_dropped:
        logger.warning(
            "[aggregate_costs] dropped %d implausible row(s) from _cost_raw — sample reasons: %s",
            n_dropped,
            "; ".join(drop_reasons[:5]),
        )
    if not clean_rows:
        # FAIL LOUD. This branch used to log a warning and return None,
        # which is how I5206 stayed invisible for 17 days: the guard
        # rejected 100% of production rows, the SF state succeeded, and
        # the dashboard kept rendering stale partitions.
        #
        # "Input had rows but none survived the filter" is categorically
        # different from "input was empty" (handled above, and a
        # legitimately quiet week). It means the filter's notion of a
        # valid row and the producer's have diverged — a contract
        # violation, and exactly the case a producer may not swallow per
        # the fail-loud rule. Raising surfaces it on the SF state.
        raise CostAggregationError(
            f"all {len(all_rows)} row(s) read from "
            f"s3://{bucket}/{input_prefix} were dropped as implausible — "
            f"refusing to write an empty parquet. This means the filter "
            f"and the producer disagree about what a valid row is, not "
            f"that there was no cost. Sample reasons: "
            f"{'; '.join(drop_reasons[:5])}"
        )

    df = pd.DataFrame(clean_rows)

    # Write parquet to a buffer + put_object so we don't need s3fs as a dep.
    output_key = output_key_override or f"{_OUTPUT_PREFIX}/{date_str}/cost.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    logger.info(
        "[aggregate_costs] wrote %d rows to s3://%s/%s",
        len(df),
        bucket,
        output_key,
    )

    summary = _build_summary(df, output_key=output_key, files_read=len(keys))
    # Phase 4 #2 + #3 — emit per-agent_id CloudWatch metrics so alarms +
    # weekly cost-regression alerts can rank by regressor. Best-effort:
    # CW emit failure does NOT block the parquet write (the
    # ``[[feedback_no_silent_fails]]`` rule applies to the load-bearing
    # producer, which is the parquet; CW emit is the observability layer
    # on top, gracefully degraded on permissions / region drift).
    _emit_per_agent_cw_metrics(df, target_date=target_date, s3_client_or_cw=cw_client)
    return summary


def _emit_per_agent_cw_metrics(
    df: pd.DataFrame,
    *,
    target_date: date_type,
    s3_client_or_cw: Any | None = None,
) -> None:
    """Emit per-agent_id cost + cache-hit-ratio metrics to CloudWatch.

    Phase 4 #2 + #3 of the cost-telemetry workstream. Per-agent_id
    dimensioning lets alarms identify which agent regressed when total
    cost spikes (vs the existing total-cost anomaly detection which
    only fires once per parquet).

    Metrics emitted (namespace ``AlphaEngine/Cost``):

    - ``WeeklyCostUsd`` (Unit=None, ``agent_id`` dimension) — sum of
      ``cost_usd`` for the agent in this parquet.
    - ``CacheHitRatio`` (Unit=Percent, ``agent_id`` dimension) —
      ``cache_read_tokens / (cache_read_tokens + input_tokens)`` × 100.
      Only emitted when the agent has non-zero cache activity OR
      non-zero input — otherwise the ratio is undefined.
    - ``ToolFeeRequests`` (Unit=Count, ``agent_id`` + ``tool`` dimensions
      for web_search / web_fetch) — server-tool request counts.

    Best-effort: any CW exception is logged at WARN and swallowed; the
    parquet write upstream is the load-bearing artifact, this is the
    observability layer.

    ``s3_client_or_cw`` parameter is named generically because the test
    pattern in this module passes mocked S3 clients; in production we
    construct a fresh CloudWatch client. Tests pass an explicit CW
    stub via this kwarg.
    """
    if df.empty or "agent_id" not in df.columns:
        return
    try:
        if s3_client_or_cw is None:
            cw = boto3.client("cloudwatch")
        else:
            cw = s3_client_or_cw
    except Exception as exc:
        logger.warning(
            "[aggregate_costs] CloudWatch client construction failed; skipping CW metric emit: %s",
            exc,
        )
        return

    metric_data: list[dict] = []
    df_clean = df.copy()
    df_clean["agent_id"] = df_clean["agent_id"].fillna("(none)").astype(str)

    grouped = df_clean.groupby("agent_id")
    for agent_id, group in grouped:
        cost = float(group["cost_usd"].fillna(0).sum()) if "cost_usd" in group.columns else 0.0
        input_tok = int(group["input_tokens"].fillna(0).sum()) if "input_tokens" in group.columns else 0
        cache_read_tok = int(group["cache_read_tokens"].fillna(0).sum()) if "cache_read_tokens" in group.columns else 0

        metric_data.append(
            {
                "MetricName": "WeeklyCostUsd",
                "Dimensions": [{"Name": "agent_id", "Value": agent_id}],
                "Value": cost,
                "Unit": "None",
            }
        )

        if input_tok + cache_read_tok > 0:
            hit_ratio = 100.0 * cache_read_tok / (input_tok + cache_read_tok)
            metric_data.append(
                {
                    "MetricName": "CacheHitRatio",
                    "Dimensions": [{"Name": "agent_id", "Value": agent_id}],
                    "Value": hit_ratio,
                    "Unit": "Percent",
                }
            )

        # Schema v2: server-tool request counts (per agent + per tool).
        for tool_col, tool_label in (
            ("web_search_requests", "web_search"),
            ("web_fetch_requests", "web_fetch"),
        ):
            if tool_col not in group.columns:
                continue
            count = int(group[tool_col].fillna(0).sum())
            if count <= 0:
                continue
            metric_data.append(
                {
                    "MetricName": "ToolFeeRequests",
                    "Dimensions": [
                        {"Name": "agent_id", "Value": agent_id},
                        {"Name": "tool", "Value": tool_label},
                    ],
                    "Value": count,
                    "Unit": "Count",
                }
            )

    if not metric_data:
        return

    # CloudWatch PutMetricData caps at 20 entries per call.
    try:
        for i in range(0, len(metric_data), 20):
            cw.put_metric_data(
                Namespace=_COST_CW_NAMESPACE,
                MetricData=metric_data[i : i + 20],
            )
        logger.info(
            "[aggregate_costs] emitted %d CW metrics under %s for %d agents",
            len(metric_data),
            _COST_CW_NAMESPACE,
            df_clean["agent_id"].nunique(),
        )
    except Exception as exc:
        logger.warning(
            "[aggregate_costs] CW metric emit failed (parquet write succeeded above; this is observability-only): %s",
            exc,
        )


def _build_summary(df: pd.DataFrame, *, output_key: str, files_read: int) -> dict:
    """Compute drilldown breakdowns on the aggregated DataFrame.

    Hard-codes the key dimensions of interest (sector_team, model,
    run_type). Total cost + total tokens are surfaced separately so
    operators can sanity-check cost vs an expected band without needing
    to load the parquet themselves.
    """
    total_cost = float(df["cost_usd"].fillna(0).sum()) if "cost_usd" in df.columns else 0.0
    total_input = int(df["input_tokens"].fillna(0).sum()) if "input_tokens" in df.columns else 0
    total_output = int(df["output_tokens"].fillna(0).sum()) if "output_tokens" in df.columns else 0
    total_cache_read = int(df["cache_read_tokens"].fillna(0).sum()) if "cache_read_tokens" in df.columns else 0
    total_cache_create = int(df["cache_create_tokens"].fillna(0).sum()) if "cache_create_tokens" in df.columns else 0
    # Schema v2 (additive): server-tool request counts. Missing column on
    # all-v1-rows partitions returns 0 — keeps the path safe during the
    # backfill window where pre-v2 days are read alongside fresh v2 days.
    total_web_search = int(df["web_search_requests"].fillna(0).sum()) if "web_search_requests" in df.columns else 0
    total_web_fetch = int(df["web_fetch_requests"].fillna(0).sum()) if "web_fetch_requests" in df.columns else 0

    def _group_sum(col: str) -> dict:
        if col not in df.columns or "cost_usd" not in df.columns:
            return {}
        # Replace NaN keys with a meaningful label before grouping. Cross-
        # sector agents (macro_economist, ic_cio) have no sector_team_id by
        # design; without this they group under the literal string "nan"
        # and mask in the by-sector breakdown. Same applies to any rows
        # missing model_name / run_type / agent_id.
        col_filled = df[col].fillna("(none)")
        grouped = df.assign(**{col: col_filled}).groupby(col)["cost_usd"].sum().fillna(0)
        return {str(k): float(v) for k, v in grouped.items()}

    return {
        "rows_in": int(len(df)),
        "files_read": files_read,
        "output_key": output_key,
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_create_tokens": total_cache_create,
        "total_web_search_requests": total_web_search,
        "total_web_fetch_requests": total_web_fetch,
        "by_sector_team": _group_sum("sector_team_id"),
        "by_model": _group_sum("model_name"),
        "by_run_type": _group_sum("run_type"),
        "by_agent_id": _group_sum("agent_id"),
        "by_execution_id": _group_sum("execution_id"),
        "by_cost_source": _group_sum("cost_source"),
        # config#5504: the completeness check — a parquet with only LLM rows
        # CANNOT report a per-run total because EC2/spot is the dominant cost
        # term. The consumer (terminal notification, dashboard) reads this
        # flag before displaying "$X.XX per run" — it MUST be False to be
        # meaningful.
        "ec2_cost_present": bool(
            df["cost_source"].fillna("llm").isin(_EC2_COST_SOURCES).any()
        ) if "cost_source" in df.columns else False,
    }


# ── Pretty-printer ────────────────────────────────────────────────────────


def print_summary(summary: dict, *, target_date: date_type) -> None:
    """Render the summary dict as a human-readable report on stdout.

    Format matches the convention in other alpha-engine reporters
    (markdown-ish with totals first, then drilldowns). Operators copy
    this into the weekly cost-report email in PR 4.
    """
    print(f"# LLM cost report — {target_date.isoformat()}\n")
    print(f"- Files read:               {summary['files_read']}")
    print(f"- Per-call rows:            {summary['rows_in']}")
    print(f"- Output:                   s3://{_DEFAULT_BUCKET}/{summary['output_key']}")
    print(f"- Total cost:               ${summary['total_cost_usd']:.4f}")
    print(f"- Total input tokens:       {summary['total_input_tokens']:,}")
    print(f"- Total output tokens:      {summary['total_output_tokens']:,}")
    print(f"- Total cache_read tokens:  {summary['total_cache_read_tokens']:,}")
    print(f"- Total cache_create tokens:{summary['total_cache_create_tokens']:,}")
    print(f"- Total web_search requests:{summary.get('total_web_search_requests', 0):,}")
    print(f"- Total web_fetch requests: {summary.get('total_web_fetch_requests', 0):,}")
    # config#5504: EC2 cost presence — the completeness signal. A per-run
    # total is only meaningful when EC2 rows exist; without them the dominant
    # cost term is absent and the number silently undercounts.
    ec2_present = summary.get("ec2_cost_present", False)
    print(f"- EC2/spot cost rows:       {'present' if ec2_present else 'ABSENT — per-run total UNDERCOUNTED'}")
    print()
    for label, key in (
        ("By sector team", "by_sector_team"),
        ("By model", "by_model"),
        ("By run_type", "by_run_type"),
        ("By agent_id", "by_agent_id"),
        ("By cost source", "by_cost_source"),
    ):
        breakdown = summary.get(key, {})
        if not breakdown:
            continue
        print(f"## {label}")
        for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"  {k:<32s} ${v:.4f}")
        print()
    # Per-execution breakdown (config#5504): the per-run grouping that lets
    # N executions on the same date report N rows. Suppressed when there is
    # only the "(unknown)" sentinel (pre-v3 rows with no execution_id).
    by_exec = summary.get("by_execution_id", {})
    if by_exec and by_exec.keys() != {"(none)"}:
        print(f"## By execution ({len(by_exec)} runs)")
        for k, v in sorted(by_exec.items(), key=lambda x: -x[1]):
            short = k if len(k) <= 48 else k[:45] + "..."
            print(f"  {short:<48s} ${v:.4f}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────


# ── Window aggregation (config-I7407) ───────────────────────────────────────
# `aggregate_day` covers ONE date, and the weekly SF invokes it with exactly
# one: `$.run_date`. Capture, meanwhile, is DAILY -- `krepis.cost_sink` writes
# under the wall-clock date of the call, so the Think Tank's daily runs land on
# dates no SF ever names.
#
# Measured 2026-08-15: `_cost_raw/` held 2026-08-10, 08-11 and 08-12; `_cost/`
# stopped at 08-10. Two days of real cost rows were captured and never
# aggregated, and the Saturday run asked for 2026-08-14, which had no raw at
# all, so the stage "succeeded" having produced nothing. The `cost_telemetry`
# transparency row then failed its 8-day freshness window against a parquet
# from 08-10 -- which is the single [FAIL] that made the 2026-08-15 weekly run
# terminate DEGRADED.
#
# Probing N dates rather than listing the prefix: N is small and bounded, a
# HEAD-shaped probe per date is cheaper and more predictable than paginating a
# prefix that accumulates forever, and the window is the same 8 days the
# consumer's `max_age_days` declares.
DEFAULT_LOOKBACK_DAYS = 8


class CostWindowGapError(CostAggregationError):
    """Raw rows exist for a date and no parquet was produced for it.

    A separate exception from CostAggregationError because the causes differ:
    that one means the filter and the producer disagree about a row, this one
    means aggregation never reached a date that had data. Both are contract
    violations a producer may not swallow; only this one is invisible without
    being named, because the per-date call returns a legitimate None.
    """


def aggregate_window(
    s3_client: Any,
    bucket: str,
    end_date: date_type,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    cw_client: Any = None,
) -> dict:
    """Aggregate every date in ``[end_date - lookback_days, end_date]`` that
    has raw cost rows.

    Returns ``{"aggregated": [...], "skipped": [...], "failed": [...]}`` where
    each entry is an ISO date string. A date with no raw partition is neither
    aggregated nor a failure -- it is a quiet day, and lands in ``skipped``.

    Idempotent by construction: re-aggregating a date rewrites the same
    parquet from the same immutable JSONL, so a re-run over an overlapping
    window is a no-op in effect. That is deliberate -- it means a missed week
    self-heals on the next run instead of needing an operator backfill.
    """
    aggregated: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for offset in range(lookback_days, -1, -1):
        d = end_date - timedelta(days=offset)
        iso = d.isoformat()
        if not _has_raw_rows(s3_client, bucket, d):
            skipped.append(iso)
            continue
        try:
            summary = aggregate_day(
                s3_client=s3_client, bucket=bucket, target_date=d,
                cw_client=cw_client,
            )
        except Exception as exc:  # noqa: BLE001 — recorded, re-raised below
            logger.error(
                "[aggregate_costs] %s had raw rows and failed to aggregate: "
                "%s: %s", iso, type(exc).__name__, exc,
            )
            failed.append((iso, f"{type(exc).__name__}: {exc}"))
            continue
        if summary is None:
            # `_has_raw_rows` said yes and `aggregate_day` produced nothing.
            # That is the silent gap this function exists to make loud.
            failed.append((iso, "raw rows present but no parquet written"))
            continue
        aggregated.append(iso)

    logger.info(
        "[aggregate_costs] window %s..%s: aggregated=%d quiet=%d failed=%d",
        (end_date - timedelta(days=lookback_days)).isoformat(),
        end_date.isoformat(), len(aggregated), len(skipped), len(failed),
    )

    if failed:
        raise CostWindowGapError(
            "cost aggregation left "
            f"{len(failed)} date(s) with raw rows unaggregated: "
            + "; ".join(f"{d}: {why}" for d, why in failed)
            + ". Raw JSONL exists for these dates and no parquet was produced, "
            "so every consumer of decision_artifacts/_cost/ is reading a "
            "window with holes in it."
        )

    return {"aggregated": aggregated, "skipped": skipped, "failed": []}


def _has_raw_rows(s3_client: Any, bucket: str, target_date: date_type) -> bool:
    """Whether ``_cost_raw/{date}/`` holds at least one JSONL object.

    One truncated LIST, not a full enumeration: the question is existence.
    """
    resp = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{_INPUT_PREFIX}/{target_date.isoformat()}/",
        MaxKeys=1,
    )
    return bool(resp.get("KeyCount") or resp.get("Contents"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-call cost JSONL files into a daily parquet.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Target date in ISO format (YYYY-MM-DD) — corresponds to the "
        "decision_artifacts/_cost_raw/{date}/ partition.",
    )
    parser.add_argument(
        "--bucket",
        default=_DEFAULT_BUCKET,
        help=f"S3 bucket (default: {_DEFAULT_BUCKET}).",
    )
    parser.add_argument(
        "--output-key",
        default=None,
        help="Override the output parquet key. Default: decision_artifacts/_cost/{date}/cost.parquet.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary on stdout.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        target_date = date_type.fromisoformat(args.date)
    except ValueError as exc:
        print(f"error: --date must be ISO YYYY-MM-DD ({exc})", file=sys.stderr)
        return 2

    s3_client = boto3.client("s3")
    summary = aggregate_day(
        s3_client,
        args.bucket,
        target_date,
        output_key_override=args.output_key,
    )
    if summary is None:
        print(f"No cost data found for {target_date.isoformat()}", file=sys.stderr)
        return 1

    if not args.quiet:
        print_summary(summary, target_date=target_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
