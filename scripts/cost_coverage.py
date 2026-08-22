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
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CostCoverageError",
    "CostCoverageUnmeasured",
    "evaluate_coverage",
    "observed_producers",
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
