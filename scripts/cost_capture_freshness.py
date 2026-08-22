"""Freshness of the CAPTURE stream itself — not of the artifact built from it.

**The gap this closes.** ``ARTIFACT_REGISTRY.yaml`` watches
``decision_artifacts/_cost/{date}/cost.parquet`` — the *product* of cost
capture. Nothing watched ``decision_artifacts/_cost_raw/`` — the capture
stream that feeds it. The two fail differently and only one of them was
observable:

- The aggregator dies → the parquet stops appearing → ``llm_cost_parquet``
  ages into STALE and the freshness monitor says so.
- Every producer stops emitting → there is nothing to aggregate → the
  aggregator reports a legitimate ``SKIPPED``, the Step Function succeeds,
  and the parquet's absence is indistinguishable from a quiet week.

The second is the one that actually happened (``alpha-engine-config-I7179``,
``-I7423``): 812 seconds of LLM calls, every record priced and accepted, and
every one of them dead in a frozen Lambda's buffer. Five surfaces reported
it, in five vocabularies, and not one of them said "capture stopped" —
because not one of them measured capture.

``principles.md`` §2.7: *no data is never rendered as green.* A producer
emitting nothing is not a quiet producer, it is an unobserved one.

**Why a sentinel object and not a prefix probe.** The fleet freshness
monitor resolves an artifact with ``head_object`` on a concrete key
(``nousergon-data/infrastructure/lambdas/freshness-monitor/index.py``
:1844, :1871) and is documented there as possibly lacking ``s3:ListBucket``.
Teaching it to probe prefixes would be a change to load-bearing fleet
infrastructure plus an IAM grant, to observe one artifact. Publishing a
concrete key the existing probe already understands costs one PUT per
aggregation cycle and adds nothing to the monitored surface.

**The sentinel is written on every terminal path, including the quiet
one.** That is the whole design. Its ``LastModified`` answers *"did anything
still observe capture?"*; its ``last_capture_date`` answers *"is capture
still happening?"*. A sentinel that stops being written is a dead observer,
which the freshness monitor sees; a sentinel that keeps saying the same
stale ``last_capture_date`` is a dead producer, which
:func:`assert_capture_fresh` raises on.

**What it deliberately does NOT do.** It does not assert that any *named*
call site emitted. That is fan-in coverage, it is a set comparison against
the stages an execution actually entered, and it already exists in
:mod:`scripts.cost_coverage`. This module answers the coarser question that
survives the pipeline not running at all: *has the fleet captured any LLM
spend, anywhere, recently.*
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from datetime import date as date_type
from typing import Any

from scripts.aggregate_costs import _INPUT_PREFIX
from scripts.cost_coverage import observed_producers

logger = logging.getLogger(__name__)

__all__ = [
    "CostCaptureStaleError",
    "SENTINEL_KEY",
    "DEFAULT_MAX_AGE_DAYS",
    "capture_stream_state",
    "assert_capture_fresh",
    "write_capture_sentinel",
]

#: Concrete key the fleet freshness monitor can ``head_object``.
SENTINEL_KEY = f"{_INPUT_PREFIX}/latest.json"

#: Days of total capture silence tolerated before this raises.
#:
#: Matched to ``aggregate_costs.DEFAULT_LOOKBACK_DAYS`` and to the
#: ``cost_telemetry`` transparency row's own max age, so the producer covers
#: exactly the window the consumer asserts. A shorter ceiling would fire on
#: the ordinary case of a week whose LLM stages were skipped; a longer one
#: would let a real capture outage outlive the window the parquet consumer
#: reads, which is the condition that made I7179 invisible.
DEFAULT_MAX_AGE_DAYS = 8


class CostCaptureStaleError(RuntimeError):
    """No LLM spend has been captured anywhere for longer than the ceiling.

    Distinct from :exc:`scripts.aggregate_costs.CostWindowGapError` (raw
    rows exist and did not become a parquet) and from
    :exc:`scripts.cost_coverage.CostCoverageError` (a named stage ran and
    emitted nothing). This one says the stream is empty, which is the
    failure the other two cannot see: both of them need rows to reason
    about.
    """


def _iter_date_partitions(s3_client: Any, bucket: str) -> list[date_type]:
    """Every ``_cost_raw/{date}/`` partition present, as dates, ascending.

    Reads ``CommonPrefixes`` rather than enumerating objects: the question
    is which days exist, and a full listing of a year of per-call JSONL to
    answer it would be a page of S3 traffic per invocation.

    Non-date prefixes are skipped silently and by design — ``latest.json``
    (this module's own sentinel) and the ``unknown-date`` partition
    ``S3JsonlCostSink`` uses for a record with an unparseable ``ts`` both
    live here, and neither is a day.
    """
    dates: list[date_type] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{_INPUT_PREFIX}/", Delimiter="/"
    ):
        for entry in page.get("CommonPrefixes", []) or []:
            leaf = entry.get("Prefix", "").rstrip("/").rsplit("/", 1)[-1]
            try:
                dates.append(date_type.fromisoformat(leaf))
            except ValueError:
                continue
    return sorted(dates)


def _producers_on(s3_client: Any, bucket: str, day: date_type) -> list[str]:
    """Call sites that emitted on ``day``, from the object keys."""
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{_INPUT_PREFIX}/{day.isoformat()}/"
    ):
        for obj in page.get("Contents", []) or []:
            keys.append(obj.get("Key", ""))
    return sorted(observed_producers(keys))


def capture_stream_state(
    s3_client: Any,
    bucket: str,
    *,
    as_of: date_type,
) -> dict:
    """Describe the capture stream as of ``as_of``.

    ``as_of`` is the caller's run date rather than ``date.today()``: a
    backfill or replay must describe the stream as the run it belongs to saw
    it, and a sentinel silently stamped with wall-clock today would report a
    stream that is fresher than the run that wrote it.

    Partitions dated after ``as_of`` are ignored for the age computation but
    still counted in ``dates_present``: capture keys on the UTC date of the
    call while a run date is a trading day, so a partition one day ahead is
    routine and must not read as a negative age.
    """
    dates = _iter_date_partitions(s3_client, bucket)
    at_or_before = [d for d in dates if d <= as_of]
    last = at_or_before[-1] if at_or_before else None
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat(),
        "dates_present": [d.isoformat() for d in dates],
        "last_capture_date": last.isoformat() if last else None,
        "days_since_last_capture": (as_of - last).days if last else None,
        "producers_on_last_capture_date": (
            _producers_on(s3_client, bucket, last) if last else []
        ),
        "max_age_days": DEFAULT_MAX_AGE_DAYS,
    }


def write_capture_sentinel(s3_client: Any, bucket: str, state: dict) -> str:
    """PUT ``state`` at :data:`SENTINEL_KEY`. Returns the key.

    Raises on failure rather than degrading. This object IS the detector's
    only input: a swallowed PUT failure leaves the freshness monitor reading
    a stale sentinel and reporting the capture stream as observed when
    nothing observed it, which is detection blindness dressed as a green
    check. The parquet — the primary deliverable — has already been written
    by the time this runs, so raising costs no data.
    """
    s3_client.put_object(
        Bucket=bucket,
        Key=SENTINEL_KEY,
        Body=json.dumps(state, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(
        "[cost_capture] sentinel written to s3://%s/%s "
        "(last_capture_date=%s, days_since=%s)",
        bucket,
        SENTINEL_KEY,
        state.get("last_capture_date"),
        state.get("days_since_last_capture"),
    )
    return SENTINEL_KEY


def assert_capture_fresh(
    state: dict, *, max_age_days: int = DEFAULT_MAX_AGE_DAYS
) -> None:
    """Raise :exc:`CostCaptureStaleError` when capture has gone silent.

    Silent means either no partition has ever existed at or before the run
    date, or the newest one is older than ``max_age_days``. Both are the
    same finding — *the fleet has been running LLM call sites and paying for
    them, and none of that spend is attributable* — and neither is
    expressible as a wrong value on any dashboard, which is why it has to be
    an exception rather than a metric.
    """
    last = state.get("last_capture_date")
    if last is None:
        raise CostCaptureStaleError(
            "no _cost_raw partition exists at or before "
            f"{state.get('as_of')} — the LLM cost capture stream is empty. "
            "Every active call site resolves a sink from "
            "KREPIS_COST_SINK_BUCKET/_PREFIX, so an empty stream means the "
            "environment wiring, the flush, or both are gone fleet-wide "
            "(alpha-engine-config-I7179, -I7423)."
        )
    age = state.get("days_since_last_capture")
    if age is not None and age > max_age_days:
        raise CostCaptureStaleError(
            f"the newest _cost_raw partition is {last} — {age} day(s) before "
            f"{state.get('as_of')}, past the {max_age_days}-day ceiling. "
            "No LLM spend has been captured anywhere in that window while "
            "the fleet kept making calls, so decision_artifacts/_cost/ is a "
            "window with a hole in it and the weekly cost report, the "
            "dashboard's LLM views and the per-agent CloudWatch dimension "
            "are all rendering an empty set as a quiet stretch."
        )


def evaluate_and_publish(
    s3_client: Any,
    bucket: str,
    *,
    as_of: date_type,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> dict:
    """Measure, publish, then assert — in that order, deliberately.

    The sentinel is written BEFORE the staleness check raises so the failure
    path leaves the same evidence behind as the success path
    (``observability-policy.md`` §3.1). A detector that raises without
    recording what it saw makes its own finding unreconstructable from
    durable artifacts, which is the transparency test.

    Returns the state dict it published.
    """
    state = capture_stream_state(s3_client, bucket, as_of=as_of)
    write_capture_sentinel(s3_client, bucket, state)
    assert_capture_fresh(state, max_age_days=max_age_days)
    return state


def _cli(argv: list[str] | None = None) -> int:
    """Standalone probe. Same verdict as the Lambda, runnable by hand."""
    import argparse

    import boto3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument(
        "--as-of",
        default=None,
        help="ISO date the stream is described as of (default: today UTC).",
    )
    parser.add_argument(
        "--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure and report without writing the sentinel.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    as_of = (
        date_type.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(UTC).date()
    )
    s3_client = boto3.client("s3")
    state = capture_stream_state(s3_client, args.bucket, as_of=as_of)
    print(json.dumps(state, indent=2, sort_keys=True))
    if not args.dry_run:
        write_capture_sentinel(s3_client, args.bucket, state)
    try:
        assert_capture_fresh(state, max_age_days=args.max_age_days)
    except CostCaptureStaleError as exc:
        print(f"STALE: {exc}")
        return 1
    print("FRESH")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI wrapper
    raise SystemExit(_cli())
