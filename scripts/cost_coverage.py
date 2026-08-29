"""Fan-in coverage for the daily cost partition — a SET comparison, not a count.

**The failure this exists to catch.** ``AggregateCosts`` rolls
``decision_artifacts/_cost_raw/{date}/`` into a daily parquet, and it was
doing that correctly. What it was aggregating was exclusively the Think
Tank's spend — a process removed from the weekly pipeline on 2026-08-10 —
while every LLM-calling stage of the pipeline it runs in emitted nothing at
all. Measured on five separate dates (``alpha-engine-config-I7179``).

Five files under the prefix is a perfectly healthy-looking count. A row
count is healthy. A dollar total is non-zero. Every scalar anyone would
put on a dashboard was **right**, and the number was about something else.
The defect is only expressible as a difference between two sets:

    producers that ran   vs   producers that emitted

so that is what this module computes, and it fails in **both** directions:

- **Missing** — a stage entered the execution and produced no cost record.
  The I7179 defect.
- **Undeclared** — a producer appeared that nothing declared. Since
  ``krepis>=0.57.0`` resolves a cost sink from the environment rather than
  from a per-call-site argument, a newly added LLM stage emits *by
  construction*; it therefore shows up here and is refused until the Step
  Function's ``coverage`` declaration names it. That is what stops the next
  stage added from silently reproducing the gap, and it is the half a
  missing-only check would not have.

**Why "stages that entered" and not "stages that exist."** A degraded run
legitimately skips stages. A check that demanded a record from every
declared stage would fire on every degraded run, and a detector that cries
wolf is a detector that gets turned off. The set of stages that actually
entered comes from the execution's own history, which is the only source
that knows.

**Why the read is scoped to the EXECUTION, not to a calendar date.**
``entered`` is a fact about one execution. ``observed`` has to be a fact
about the same execution or the set comparison is between two different
things. It was not, until 2026-08-29: the check listed exactly one
prefix, ``_cost_raw/{run_date}/``, while ``krepis.cost_sink.S3JsonlCostSink``
partitions by the record's own UTC ``ts``. The weekly pipeline starts
09:00 UTC on the day AFTER its ``run_date``, so 100% of its cost records
land one partition ahead of where the check looked, and the verdict it
published — ``covered``, ``observed``, ``undeclared`` — described the
previous day's DAILY pipelines. Measured on execution
``965d925b-f9d6-ce5e-e059-9405433a0724_c0271e3b-a27f-a834-b303-3e3803fffd16``:
``replay-concordance`` was reported as "ran and emitted no cost record"
while its 24 priced rows sat in
``_cost_raw/2026-08-29/krepis-b1444bfa3454/replay-concordance.0.jsonl``,
flushed one second after the stage exited; and the five ``thinktank-*``
producers scored as ``observed`` belonged to a run this execution never
made. See :func:`observed_producers_for_execution`.

**Being unable to measure is not passing.** If the execution history cannot
be read, this raises :exc:`CostCoverageUnmeasured` — a *different*
exception from the coverage breach, so the two are never confused on the
surface that reports them. `principles.md` §2.7: no data is never rendered
as green.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

#: An execution spans at most a couple of UTC dates. A run_date further
#: back than this means the caller is not describing a live run, and
#: enumerating the span would be a guess rather than a measurement.
_MAX_PARTITION_SPAN_DAYS = 3

__all__ = [
    "CostCoverageError",
    "CostCoverageUnmeasured",
    "evaluate_coverage",
    "execution_started_at",
    "observed_producers",
    "observed_producers_for_execution",
    "stages_entered",
]


class CostCoverageError(RuntimeError):
    """A stage ran and emitted nothing, or an undeclared producer appeared.

    A finding about the pipeline. Distinct from
    :exc:`CostCoverageUnmeasured`, which is a fault in this check itself —
    conflating them is how a harness failure gets reported as a domain
    defect, and it is always reported in the alarming direction.
    """


class CostCoverageUnmeasured(RuntimeError):
    """The coverage check could not run. NOT a pass."""


def observed_producers(keys: Iterable[str]) -> set:
    """Distinct producers under a ``_cost_raw/{date}/`` prefix, from the keys.

    ``S3JsonlCostSink`` writes ``{prefix}/{date}/{run_id}/{callsite_id}.{seq}.jsonl``,
    so the callsite id is the object's stem with the flush sequence stripped.
    Derived from the KEY rather than from a row field on purpose: a producer
    that wrote an empty or malformed object still ran, and a check that can
    only see well-formed rows would score its silence as absence.
    """
    producers = set()
    for key in keys:
        if not key.endswith(".jsonl"):
            continue
        stem = key.rsplit("/", 1)[-1][: -len(".jsonl")]
        # Strip the trailing ``.{seq}`` the sink appends per flush. A
        # callsite id may itself contain dots, so only a purely numeric
        # final segment is treated as the sequence.
        head, _, tail = stem.rpartition(".")
        producers.add(head if head and tail.isdigit() else stem)
    return producers


def stages_entered(execution_arn: str, *, sfn_client: Any = None) -> set:
    """Names of every state this execution entered, from its own history.

    Raises :exc:`CostCoverageUnmeasured` rather than returning an empty set
    on any failure. An empty set would make every required producer drop
    out of the expected set, and the check would then pass for precisely
    the reason it should have failed.
    """
    if not execution_arn:
        raise CostCoverageUnmeasured(
            "no execution ARN was supplied, so the set of stages that "
            "actually ran is unknown. The Step Function threads it as "
            "coverage.execution_arn ($$.Execution.Id)."
        )
    if sfn_client is None:
        import boto3

        sfn_client = boto3.client("stepfunctions")

    names = set()
    try:
        paginator = sfn_client.get_paginator("get_execution_history")
        pages = paginator.paginate(
            executionArn=execution_arn, includeExecutionData=False
        )
        for page in pages:
            for event in page.get("events", []) or []:
                details = event.get("stateEnteredEventDetails")
                if details and details.get("name"):
                    names.add(details["name"])
    except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
        raise CostCoverageUnmeasured(
            f"could not read the execution history of {execution_arn}: {exc}. "
            f"This is a fault in the coverage check, NOT a coverage finding — "
            f"do not read it as a silent stage."
        ) from exc
    if not names:
        raise CostCoverageUnmeasured(
            f"the execution history of {execution_arn} named zero states, "
            f"which cannot be true of a running execution — treating it as "
            f"unmeasured rather than as universal absence."
        )
    return names


def execution_started_at(execution_arn: str, *, sfn_client: Any = None):
    """UTC start time of the execution, as a timezone-aware datetime.

    Raises :exc:`CostCoverageUnmeasured` on any failure rather than
    substituting a default. A fabricated start time would silently widen or
    narrow the window the observed set is drawn from, and the check would
    then be wrong in a direction nobody could see — the same shape of
    defect this module exists to catch.
    """
    if not execution_arn:
        raise CostCoverageUnmeasured(
            "no execution ARN was supplied, so the execution's start time — "
            "and therefore which cost objects belong to it — is unknown. "
            "The Step Function threads it as coverage.execution_arn "
            "($$.Execution.Id)."
        )
    if sfn_client is None:
        import boto3

        sfn_client = boto3.client("stepfunctions")
    try:
        started = sfn_client.describe_execution(executionArn=execution_arn)[
            "startDate"
        ]
    except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
        raise CostCoverageUnmeasured(
            f"could not read the start time of {execution_arn}: {exc}. "
            f"This is a fault in the coverage check, NOT a coverage finding."
        ) from exc
    if started is None:
        raise CostCoverageUnmeasured(
            f"describe_execution({execution_arn}) returned no startDate, "
            f"which cannot be true of a running execution."
        )
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started.astimezone(timezone.utc)


def observed_producers_for_execution(
    s3_client: Any,
    bucket: str,
    prefix_root: str,
    *,
    started_at,
    now: Any = None,
) -> set:
    """Producers that wrote a cost object DURING this execution.

    ``prefix_root`` is the ``_cost_raw`` root with no date segment.

    Two things are wrong with reading a single ``{prefix_root}/{run_date}/``
    prefix, and this replaces both:

    1. **The partition is not the run_date.** ``S3JsonlCostSink`` keys by
       the record's own UTC ``ts``, so the partitions an execution can have
       written to are the UTC dates the EXECUTION spans — ``started_at``
       through now — which for the weekly pipeline is ``run_date + 1``.
       ``run_date`` is not consulted at all: it names the trading day the
       run is ABOUT, never the wall-clock day it ran on, and a rerun of an
       old ``run_date`` writes its records today like any other run.
    2. **A partition holds other runs' objects.** The daily pipelines write
       to the same prefix. Filtering on ``LastModified >= started_at`` keeps
       only objects flushed while this execution was running, so a producer
       from another run can no longer supply coverage for a stage THIS
       execution ran — which is what masked the defect on 2026-08-29, where
       ``single-agent-quant`` scored ``covered`` off an object written 14
       hours before the execution began, and five ``thinktank-*`` producers
       were reported as ``observed`` by a run that never entered a thinktank
       stage.

    ``LastModified`` is the flush time, at or after the ``ts`` of every
    record in the object. The sink is per-process and flushes on handler
    exit (the ``finally: flush_default_sink()`` of
    ``alpha-engine-config-I7423``), so an object straddling the execution
    boundary would have to come from a process that started before it and
    flushed during it. That widens ``observed`` only, never narrows it, and
    the ``undeclared`` direction still refuses anything nothing declared.

    Raises :exc:`CostCoverageUnmeasured` if the listing fails: an empty
    observed set is indistinguishable from universal silence, and would be
    reported as the alarming finding this check exists to make.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if started_at is None:
        raise CostCoverageUnmeasured(
            "no execution start time was supplied, so which cost objects "
            "belong to this execution is unknown."
        )
    first = started_at.date()
    last = max(first, now.date())
    if (last - first).days > _MAX_PARTITION_SPAN_DAYS:
        raise CostCoverageUnmeasured(
            f"the execution started {(last - first).days} day(s) ago "
            f"({first} .. {last}); refusing to enumerate that many "
            f"partitions rather than guessing which one the run wrote to."
        )

    root = prefix_root.rstrip("/")
    keys: list[str] = []
    day = first
    while day <= last:
        prefix = f"{root}/{day.isoformat()}/"
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    key = obj.get("Key", "")
                    if not key.endswith(".jsonl"):
                        continue
                    modified = obj.get("LastModified")
                    if modified is None:
                        raise CostCoverageUnmeasured(
                            f"s3://{bucket}/{key} was listed without a "
                            f"LastModified, so it cannot be attributed to "
                            f"an execution."
                        )
                    if modified.tzinfo is None:
                        modified = modified.replace(tzinfo=timezone.utc)
                    if modified >= started_at:
                        keys.append(key)
        except CostCoverageUnmeasured:
            raise
        except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
            raise CostCoverageUnmeasured(
                f"could not list s3://{bucket}/{prefix}: {exc}. This is a "
                f"fault in the coverage check, NOT a coverage finding — do "
                f"not read it as a silent stage."
            ) from exc
        day += timedelta(days=1)

    logger.info(
        "[cost_coverage] observed window %s..%s since %s: %d object(s)",
        first.isoformat(), last.isoformat(), started_at.isoformat(), len(keys),
    )
    return observed_producers(keys)


def evaluate_coverage(
    *,
    observed: set,
    declaration: dict,
    entered: set,
) -> dict:
    """Compare the producers that emitted against the producers that ran.

    ``declaration`` is the Step Function's ``coverage`` block:

        required_producers:    {StageName: [callsite_id, ...]}
        conditional_producers: {StageName: [callsite_id, ...]}
        allowed_producers:     [callsite_id or fnmatch pattern, ...]

    Returns the verdict as data — ``covered`` / ``missing`` / ``undeclared``
    / ``expected`` / ``observed`` — so the caller can log and persist the
    whole picture, then raises on either failure direction. The verdict is
    built before the raise deliberately: an exception message is not a
    surface, and a check whose only output is its own failure text cannot
    be trended.
    """
    required = declaration.get("required_producers") or {}
    conditional = declaration.get("conditional_producers") or {}
    allowed_patterns = list(declaration.get("allowed_producers") or [])

    expected = {
        cid for stage, ids in required.items() if stage in entered for cid in ids
    }
    skipped_stages = sorted(set(required) - entered)
    conditionally_allowed = {
        cid for stage, ids in conditional.items() if stage in entered for cid in ids
    }

    def _is_allowed(producer: str) -> bool:
        if producer in expected or producer in conditionally_allowed:
            return True
        return any(fnmatch.fnmatch(producer, pat) for pat in allowed_patterns)

    missing = sorted(expected - observed)
    undeclared = sorted(p for p in observed if not _is_allowed(p))

    verdict = {
        "expected": sorted(expected),
        "observed": sorted(observed),
        "covered": sorted(expected & observed),
        "missing": missing,
        "undeclared": undeclared,
        "stages_entered": sorted(entered & (set(required) | set(conditional))),
        "stages_not_entered": skipped_stages,
    }

    if missing:
        raise CostCoverageError(
            f"cost fan-in coverage BREACH — {len(missing)} stage(s) ran and "
            f"emitted no cost record: {', '.join(missing)}. Their spend is "
            f"attributed to nothing. Observed producers: "
            f"{', '.join(verdict['observed']) or '(none)'}. "
            f"Verdict: {verdict}"
        )
    if undeclared:
        raise CostCoverageError(
            f"cost fan-in coverage BREACH — {len(undeclared)} producer(s) "
            f"wrote to the partition without being declared: "
            f"{', '.join(undeclared)}. Add them to the Step Function's "
            f"coverage block (required_producers if the stage must always "
            f"emit, allowed_producers if it legitimately writes here without "
            f"being part of this pipeline). Verdict: {verdict}"
        )
    return verdict
