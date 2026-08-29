"""Coverage as a pass/fail verdict for the eval judge.

**Why this module exists.** Brian's ruling 2026-08-29, verbatim: *"perhaps if
the lambda times out then we need to put the judge on a spot instance."*
Tracked as ``alpha-engine-config-I9309``.

The reasoning that governs how it is built is his earlier ruling, 2026-08-14,
about the Director: *"director should NEVER time out, if it times out it
FAILS"* — recorded with the rationale that retries and larger ceilings are out
of scope as fixes **because they make a latency regression survivable instead
of visible**.

Measured 2026-08-29: the judge's synchronous rung covers roughly 8-15 of a
~85-artifact corpus inside ``EvalJudgeProcess``' 960s Lambda ceiling. That
shortfall was reported honestly — ``complete=False``, ``budget_stopped_phases``,
``n_skipped_for_budget`` — and the stage still returned success. An honest
field on a successful stage is exactly the anti-pattern above: a tenfold
capacity shortfall that cost the pipeline nothing, moved no alarm, and left a
weekly eval series silently built from a tenth of its corpus.

**So coverage is a verdict, not a field.** The judge grades every artifact it
planned, or the stage fails naming the shortfall.

**What this module does NOT make fatal.** The transport rung
(``degraded_transport``, ``evals/judge_batch_transport.py``) stays a reported
degradation: it is a real, priced, deliberately-taken ladder step, and the
run's output is complete on either rung. Coverage is a different question —
*did every artifact get graded* — and it is the one with a pass/fail answer.
The two are kept apart on purpose; collapsing them would either make a
legitimate transport choice fatal or make an incomplete corpus survivable, and
both mistakes have already happened once each.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: How many ungraded entries the exception message names before truncating.
#: The full list is always on the summary and on the persisted run record;
#: this bounds only the exception STRING, which lands in a Step Functions
#: `cause` field with its own size limit and is frequently the only artifact a
#: reader sees first.
_SHORTFALL_SAMPLE = 10


class JudgeCoverageShortfall(RuntimeError):
    """The judge graded fewer artifacts than it planned to.

    Carries the counts and the shortfall detail as attributes as well as in the
    message, so a caller that catches it can emit structured telemetry without
    re-parsing prose.

    Deliberately a plain ``RuntimeError`` subclass and NOT related to
    ``BatchCapabilityUnavailable``: that one is a ladder signal a caller is
    meant to catch and act on, this one is a terminal verdict. Nothing in this
    repo may catch this to continue — a caller that swallowed it would restore
    exactly the survivable-shortfall behaviour the class exists to end.
    """

    def __init__(
        self,
        *,
        planned: int,
        graded: int,
        ungraded_entries: list[dict[str, Any]],
        budget_stopped_phases: list[str],
        n_skipped_for_budget: dict[str, int],
        failed: list[dict[str, Any]],
    ) -> None:
        self.planned = planned
        self.graded = graded
        self.ungraded_entries = ungraded_entries
        self.budget_stopped_phases = budget_stopped_phases
        self.n_skipped_for_budget = n_skipped_for_budget
        self.failed = failed
        super().__init__(describe_shortfall(
            planned=planned,
            graded=graded,
            ungraded_entries=ungraded_entries,
            budget_stopped_phases=budget_stopped_phases,
            n_skipped_for_budget=n_skipped_for_budget,
            failed=failed,
        ))


def describe_shortfall(
    *,
    planned: int,
    graded: int,
    ungraded_entries: list[dict[str, Any]],
    budget_stopped_phases: list[str],
    n_skipped_for_budget: dict[str, int],
    failed: list[dict[str, Any]],
) -> str:
    """Human- and log-readable statement of what was missed and why.

    Names the counts first (the number an operator acts on), then the CAUSE
    split — a deadline stop and a judge error are different defects with
    different fixes, and a message that reports only "13 of 85" sends a reader
    to the wrong place. Both are stated even when zero, so their absence is
    legible rather than ambiguous.
    """
    missing = planned - graded
    lines = [
        f"eval-judge coverage shortfall: graded {graded} of {planned} planned "
        f"artifacts ({missing} ungraded).",
    ]
    if budget_stopped_phases:
        lines.append(
            "  stopped on deadline in phase(s): "
            + ", ".join(
                f"{p} (+{n_skipped_for_budget.get(p, 0)} skipped)"
                for p in budget_stopped_phases
            )
            + " — the run did not have time to finish; this is a CAPACITY "
              "shortfall, not a judge defect."
        )
    else:
        lines.append("  no phase stopped on a deadline.")
    judge_failures = [
        f for f in failed
        if f.get("stage") in {
            "sync_fallback_judge", "batch_parse_retry", "batch_persist",
            "batch_parse", "process_unknown_custom_id",
        }
    ]
    if judge_failures:
        lines.append(
            f"  {len(judge_failures)} entr(ies) failed in the judge itself: "
            + ", ".join(
                f"{f.get('agent_id')}[{f.get('stage')}]"
                for f in judge_failures[:_SHORTFALL_SAMPLE]
            )
            + ("" if len(judge_failures) <= _SHORTFALL_SAMPLE else ", ...")
        )
    else:
        lines.append("  no entry failed inside the judge.")
    if ungraded_entries:
        sample = ungraded_entries[:_SHORTFALL_SAMPLE]
        lines.append(
            "  ungraded: "
            + ", ".join(
                f"{e.get('agent_id')}/{e.get('run_id')}"
                f"@{e.get('judge_model')}" for e in sample
            )
            + ("" if len(ungraded_entries) <= _SHORTFALL_SAMPLE
               else f", ... (+{len(ungraded_entries) - _SHORTFALL_SAMPLE} more)")
        )
    return "\n".join(lines)


def assess_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    """Reduce a ``process_batch_results`` summary to a coverage verdict.

    Pure: reads, never raises, never writes. Split from
    :func:`enforce_coverage` so the verdict can be recorded on a durable
    artifact and emitted as telemetry BEFORE the raise takes the process down —
    a verdict that only exists inside an exception is a verdict nobody can
    query afterwards.

    A summary missing ``plan_entry_count`` is reported ``UNMEASURED``, never
    ``PASS``. That case means the producer predates this ledger, and treating
    a silent producer as healthy is the failure mode ``principles.md`` §2.7
    exists to forbid: *no data* is never rendered as green.
    """
    if "plan_entry_count" not in summary:
        logger.error(
            "[judge_coverage] summary carries no coverage ledger — the "
            "producer predates alpha-engine-config-I9309. Reporting "
            "UNMEASURED; this is NOT a pass."
        )
        return {
            "status": "UNMEASURED",
            "reason": "summary has no plan_entry_count (producer predates I9309)",
            "planned": None,
            "graded": None,
            "ungraded": None,
        }

    planned = int(summary["plan_entry_count"])
    graded = int(summary.get("plan_entries_graded", 0))
    ungraded_entries = list(summary.get("ungraded_entries", []))
    complete = planned == graded and not ungraded_entries
    return {
        "status": "PASS" if complete else "SHORTFALL",
        "planned": planned,
        "graded": graded,
        "ungraded": planned - graded,
        "ungraded_entries": ungraded_entries,
        "budget_stopped_phases": list(summary.get("budget_stopped_phases", [])),
        "n_skipped_for_budget": dict(summary.get("n_skipped_for_budget", {})),
        # Carried alongside so a reader of the verdict sees the transport the
        # coverage was achieved on. Reported, never fatal — see the module
        # docstring on why the two questions are kept apart.
        "degraded_transport": bool(summary.get("degraded_transport", False)),
        "detail": None if complete else describe_shortfall(
            planned=planned,
            graded=graded,
            ungraded_entries=ungraded_entries,
            budget_stopped_phases=list(summary.get("budget_stopped_phases", [])),
            n_skipped_for_budget=dict(summary.get("n_skipped_for_budget", {})),
            failed=list(summary.get("failed", [])),
        ),
    }


def enforce_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the coverage verdict, or raise :class:`JudgeCoverageShortfall`.

    The single chokepoint that turns the ledger into a stage outcome. Raises on
    ``SHORTFALL`` **and on ``UNMEASURED``** — a producer that cannot say what it
    covered has not demonstrated it covered anything, and the whole point of
    I9309 is that an unproven full sweep may not be reported as one.
    """
    verdict = assess_coverage(summary)

    if verdict["status"] == "PASS":
        logger.info(
            "[judge_coverage] PASS — graded %d of %d planned artifacts "
            "(degraded_transport=%s)",
            verdict["graded"], verdict["planned"], verdict["degraded_transport"],
        )
        return verdict

    if verdict["status"] == "UNMEASURED":
        raise JudgeCoverageShortfall(
            planned=-1,
            graded=-1,
            ungraded_entries=[],
            budget_stopped_phases=[],
            n_skipped_for_budget={},
            failed=[{
                "stage": "coverage_unmeasured",
                "error": verdict["reason"],
            }],
        )

    logger.error("[judge_coverage] SHORTFALL\n%s", verdict["detail"])
    raise JudgeCoverageShortfall(
        planned=verdict["planned"],
        graded=verdict["graded"],
        ungraded_entries=verdict["ungraded_entries"],
        budget_stopped_phases=verdict["budget_stopped_phases"],
        n_skipped_for_budget=verdict["n_skipped_for_budget"],
        failed=list(summary.get("failed", [])),
    )
