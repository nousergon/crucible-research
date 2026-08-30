"""Batch-judge transport resolution — the capability seam that replaced the
three direct ``anthropic.Anthropic(...)`` constructions in the eval-judge
Lambda chain.

**Why this module exists.** Brian's ruling 2026-08-29, verbatim: *"I will not
fund the anthropic account, at this point we shouldn't be using the anthropic
api at all."* It hardens the 2026-08-16 ruling ("we are migrating off anthropic
and onto the router"). Before this module, ``eval_judge_submit_handler``,
``eval_judge_poll_handler`` and ``eval_judge_process_handler`` each built
``anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)`` at the call site — three
copies of a provider SDK client, no router, no fallback ladder. One vendor's
billing state was therefore a single point of failure for a whole branch of the
weekly Step Function, and it failed as one three times:
alpha-engine-config-I7448 (2026-08-16), -I9049 (2026-08-22) and the 2026-08-29
weekly run, whose ``EvalJudgeSubmitWeekly`` stage returned
``400 ... 'Your credit balance is too low to access the Anthropic API'`` and set
``branch_a_degraded`` for the entire run.

**What replaced it — a capability class, never a provider.** Nothing here names
a model id, a base URL, a provider name, or constructs an SDK client
(``principles.md`` §2.8, ``model-router-policy.md`` §2 layer 5). This module
asks the router one question — *can the judge's model group serve a batch
request from here?* — via ``krepis.router.resolve_group_spec`` with
``requires=("batches",)``, the same ``requires`` mechanism
``evals/judge.py::_judge_router_spec_and_route`` already uses for
``tool_choice``. Whichever registry entry answers is a registry decision
resolved above this layer.

**The ladder, stated rather than implied.**

1. *Batch rung.* A router-resolved, batch-capable route. Used when the registry
   declares one reachable from this execution context.
2. *Synchronous rung.* ``evals.judge.evaluate_artifact`` — already fully
   router-addressed through the ``low`` group (alpha-engine-config-I6559) and
   already exercised in production by ``process_batch_results``' parse-retry
   and Sonnet-escalation tails. It is the SAME judge, over the SAME rubric,
   producing the SAME ``RubricEvalArtifact``; only the transport differs.

**A degradation is recorded, never silent.** Dropping from rung 1 to rung 2
costs real money (see ``docs`` on ``BATCH_PRICE_MULTIPLIER`` in
``evals/orchestrator.py``: the batch rung bills at half rate, so the sync rung
is ~2x per judged artifact) and it changes the transport a longitudinal eval
series was produced on. ``build_degradation_record`` produces the durable
record and ``persist_degradation_record`` writes it to S3 next to the batch
plan; the Submit handler additionally returns it on its result so the Step
Function's own stage output carries it. A silent provider switch is precisely
the failure mode this whole module exists to remove — so nothing here swallows:
if the degradation record cannot be written, the write raises and the stage
fails loud rather than degrading unobserved.

**Today's resolved state, measured 2026-08-29.** No batch rung exists.
``krepis.model_registry.ROUTABLE_CAPABILITIES`` is ``("tool_choice",
"streaming")``, so ``batches`` is not yet expressible as a routing requirement;
and the six ``LLM_MODEL_REGISTRY.yaml`` entries that carry ``batches: true``
(``claude-haiku-4-5``, ``claude-sonnet-4-6``, ``claude-sonnet-5``,
``claude-opus-5``, ``claude-opus-4-8``, ``claude-fable-5``) are all served by
an aggregator route that exposes no Message Batches endpoint, so those flags
describe the MODEL rather than the ROUTE that serves it. (Named here by
registry field rather than by provider string: the fleet's direct-linkage
guard cannot distinguish prose from a call site, and the correct response to
that is to stop writing the literal, not to widen an allowlist.) Both gaps are tracked; see
``alpha-engine-config-I9263``. Until one closes, every call to
:func:`resolve_batch_transport` raises :class:`BatchCapabilityUnavailable` and
the pipeline runs rung 2. **That is a resolution outcome, not a hardcoded
switch:** the moment ``batches`` becomes routable and one reachable member
declares it truthfully, the batch rung lights up here with no change to this
file.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import boto3

logger = logging.getLogger(__name__)

BATCH_CAPABILITY = "batches"
"""The capability CLASS the batch judge transport requires.

Named as a capability rather than as a vendor: ``build_batch_plan`` produces a
provider-neutral list of judge requests, and any route that can accept a
fan-out of them off the request path satisfies this call site. The string
matches the ``capabilities.batches`` key already documented in
``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml``'s schema header, so
declaring it truthfully on a route is the only edit needed to serve rung 1."""

SYNC_BATCH_ID_PREFIX = "sync-"
"""Synthetic batch-id prefix marking a run that took the SYNCHRONOUS rung.

Mirrors the existing ``empty-`` sentinel convention: ``poll_batch`` treats it as
already-terminal, and ``process_batch_results`` reads it as "judge every plan
entry through ``evaluate_artifact`` instead of streaming a provider batch". A
prefix rather than a separate field because the batch id is the ONE value that
crosses every Step Function state boundary in this chain, so a run's transport
is legible from any stage's input without a schema change."""


def sync_batch_id(date: str) -> str:
    """Synthetic batch id for a run served by the synchronous rung."""
    return f"{SYNC_BATCH_ID_PREFIX}{date}"


def is_sync_batch_id(batch_id: str) -> bool:
    """True when *batch_id* names a synchronous-rung run."""
    return batch_id.startswith(SYNC_BATCH_ID_PREFIX)


class BatchCapabilityUnavailable(RuntimeError):
    """The router cannot serve a batch request for the judge's model group.

    Carries the group, the capability and the execution context asked for, plus
    the resolver's own stated reason — because for a weekly unattended caller
    the exception message is frequently the only artifact left behind, and
    "no member declares it", "not reachable from here" and "the router is
    unhealthy" are three different answers that must not be flattened into one.

    This is a LADDER signal, not a failure: the caller is expected to catch this
    specific type, record the degradation, and take the synchronous rung. It is
    deliberately NOT a subclass of anything the transport layer raises for a
    genuine fault, so ``except BatchCapabilityUnavailable`` can never
    accidentally swallow a real router or provider error.
    """

    def __init__(
        self,
        *,
        group: str,
        capability: str,
        exec_context: str,
        reason: str,
    ) -> None:
        self.group = group
        self.capability = capability
        self.exec_context = exec_context
        self.reason = reason
        super().__init__(
            f"model group {group!r} cannot serve capability {capability!r} "
            f"from exec_context {exec_context!r}: {reason}"
        )


def resolve_batch_transport(
    *,
    group: str | None = None,
    exec_context: str | None = None,
    max_tokens: int | None = None,
) -> tuple[Any, dict]:
    """Resolve a batch-capable ``(ModelSpec, route)`` for the judge, or raise.

    Asks the router for the judge's capability tier with the batch requirement
    declared. Everything else — model, endpoint, credential — is resolved above
    this layer from the registry; this function states only the tier it wants,
    the shape its request needs, and where it is running
    (``model-router-policy.md`` R29).

    Raises
    ------
    BatchCapabilityUnavailable
        No reachable member of *group* can serve a batch request. Three
        distinguishable causes are all mapped here, each with its own stated
        reason, because from the caller's position they have the identical
        consequence (take rung 2) while meaning different things to a reader:

        * ``CapabilityUnavailableError`` — the group has members, none declares
          the capability. A registry gap.
        * ``ValueError`` naming the capability as not routable — this krepis
          cannot express the requirement at all, so the router demonstrably
          cannot be routing on it. Treating that as "capability available"
          would be the silent-degrade this module exists to prevent.
        * ``ImportError`` — krepis is too old to carry the resolver. Same
          conclusion, different cause.

        Any OTHER exception propagates untouched: a router that is down, a bad
        credential or an unknown group is a fault to fail loud on, not a reason
        to quietly spend twice as much on the sync rung.
    """
    # Imported here, not at module scope, for the same reason
    # ``evals/judge.py::_judge_router_spec_and_route`` does it: the test
    # suite's autouse router stub patches ``krepis.router.resolve_group_spec``,
    # and a module-level ``from krepis.router import ...`` would bind the name
    # before any per-test monkeypatch could intercept it.
    from evals.judge import JUDGE_MODEL_GROUP, judge_exec_context

    group = group or JUDGE_MODEL_GROUP
    # alpha-engine-config-I9309: resolved per call, so the same image asks the
    # router the right question from a Lambda and from the spot box.
    exec_context = exec_context or judge_exec_context()

    try:
        from krepis.model_registry import CapabilityUnavailableError
        from krepis.router import resolve_group_spec
    except ImportError as exc:
        raise BatchCapabilityUnavailable(
            group=group,
            capability=BATCH_CAPABILITY,
            exec_context=exec_context,
            reason=f"krepis router unavailable in this environment: {exc}",
        ) from exc

    try:
        spec, route = resolve_group_spec(
            group,
            exec_context=exec_context,
            requires=(BATCH_CAPABILITY,),
            max_tokens=max_tokens,
        )
    except CapabilityUnavailableError as exc:
        raise BatchCapabilityUnavailable(
            group=group,
            capability=BATCH_CAPABILITY,
            exec_context=exec_context,
            reason=f"no live group member declares it: {exc}",
        ) from exc
    except ValueError as exc:
        # `krepis.router` rejects a `requires` flag absent from
        # `ROUTABLE_CAPABILITIES` with a ValueError. Measured 2026-08-29:
        # ROUTABLE_CAPABILITIES == ("tool_choice", "streaming"), so this is the
        # branch that fires today. An unknown-GROUP ValueError is a real
        # misconfiguration and must not be laundered into a degradation, so the
        # message is checked before the class is claimed.
        if BATCH_CAPABILITY not in str(exc):
            raise
        raise BatchCapabilityUnavailable(
            group=group,
            capability=BATCH_CAPABILITY,
            exec_context=exec_context,
            reason=(
                f"the router cannot route on it in this krepis version, so no "
                f"route can be selected by it: {exc}"
            ),
        ) from exc

    logger.info(
        "[judge_batch_transport] batch rung resolved group=%s route=%s "
        "deployment=%s", group, route.get("route"), spec.model,
    )
    return spec, route


#: What the sync rung costs relative to the batch rung, ON THE SAME MODEL.
#:
#: The batch rung bills every message at half rate — the same figure
#: ``evals/orchestrator.py::_BATCH_PRICE_MULTIPLIER`` (0.5) applies when pricing
#: batch spend — so forfeiting the batch discount doubles the per-token price of
#: an identical request. Derived from the two price cards, not measured from a
#: run: it is a property of the rate cards, and a dollar figure quoted here
#: would go stale the first time the corpus size changed.
SYNC_RUNG_BATCH_DISCOUNT_FORFEITED = 2.0

#: What the sync rung costs relative to the route it actually replaced.
#:
#: **This is the number that answers "did this migration cost us money", and it
#: is not the one above.** The batch rung this ladder replaces did not run the
#: same model: it ran ``claude-haiku-4-5`` against Anthropic's Batches API,
#: while the sync rung runs whatever the ``low`` group resolves to. Measured
#: against ``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml``
#: 2026-08-29:
#:
#: * batch rung — ``claude-haiku-4-5`` at $1.00/$5.00 per M input/output tokens,
#:   halved by the batch discount to **$0.50 / $2.50**;
#: * sync rung — the ``low`` group's tool-capable member ``deepseek-v4-flash``
#:   at **$0.44 / $1.32** (cache reads $0.014/M).
#:
#: So the sync rung is ~12% cheaper on input and ~47% cheaper on output than the
#: route Brian retired. Losing the batch discount is real, and it is
#: more than paid for by leaving the provider. Recorded explicitly because the
#: intuitive reading — "we lost the 50% batch discount, so this costs more" — is
#: the wrong conclusion here, and an unstated wrong conclusion about cost is how
#: a correct migration gets argued backwards later.
SYNC_RUNG_COST_RATIO_VS_RETIRED_BATCH_ROUTE = {
    "input": 0.44 / 0.50,
    "output": 1.32 / 2.50,
    "measured_at": "2026-08-29",
    "batch_route": "claude-haiku-4-5 @ Anthropic Batches (50% discount)",
    "sync_route": "low group -> deepseek-v4-flash via the krepis router",
}


def build_degradation_record(
    *,
    date: str,
    reason: str,
    group: str,
    capability: str,
    exec_context: str,
    request_count: int,
    plan_s3_key: str | None = None,
) -> dict[str, Any]:
    """The durable record of a drop from the batch rung to the sync rung.

    Every field a reader needs to answer "what changed, why, and what did it
    cost" without re-running anything: which rung was asked for, which was
    taken, the resolver's stated reason, how many judged artifacts were
    affected, and the cost multiplier that applies to them.
    """
    return {
        "schema_version": 1,
        "kind": "eval_judge_batch_degradation",
        "date": date,
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "requested_rung": "batch",
        "served_rung": "sync",
        "capability": capability,
        "model_group": group,
        "exec_context": exec_context,
        "reason": reason,
        "request_count": request_count,
        # Two different questions, both answered, because answering only the
        # first invites the wrong conclusion — see the constants' own docs.
        "batch_discount_forfeited_multiplier": SYNC_RUNG_BATCH_DISCOUNT_FORFEITED,
        "cost_ratio_vs_retired_batch_route": (
            SYNC_RUNG_COST_RATIO_VS_RETIRED_BATCH_ROUTE
        ),
        "plan_s3_key": plan_s3_key,
        "ruling": (
            "Brian 2026-08-29: 'I will not fund the anthropic account, at this "
            "point we shouldn't be using the anthropic api at all.' The direct "
            "Anthropic Batches route this stage used is retired; "
            "alpha-engine-config-I9263."
        ),
    }


def degradation_s3_key(*, date: str) -> str:
    """S3 key for a run's degradation record.

    Co-located with the batch plan manifests (``_eval_batch_plans/``) so a
    reader inspecting a run's plan finds the transport record in the same
    listing rather than having to know a second prefix exists."""
    return f"decision_artifacts/_eval_batch_plans/{date}/degradation.json"


def persist_degradation_record(
    record: dict[str, Any],
    *,
    bucket: str,
    s3_client: Any | None = None,
) -> str:
    """Write *record* to S3 and return its key.

    Deliberately unguarded: a failed write RAISES and takes the stage down.
    This record is the ONLY durable evidence that the run served a different
    transport at twice the price, so a best-effort write here would reproduce
    exactly the silent-degradation failure mode this module was written to
    close (``AGENTS.md``: no silent swallows on a producer)."""
    s3 = s3_client or boto3.client("s3")
    key = degradation_s3_key(date=record["date"])
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(record, indent=2, default=str).encode("utf-8"),
    )
    ratio = record["cost_ratio_vs_retired_batch_route"]
    logger.warning(
        "[judge_batch_transport] DEGRADED to the synchronous judge rung for "
        "date=%s (%d requests; %.2fx the same-model batch price, but "
        "%.2fx/%.2fx in/out against the RETIRED Anthropic batch route) — "
        "reason: %s — recorded at s3://%s/%s",
        record["date"], record["request_count"],
        record["batch_discount_forfeited_multiplier"],
        ratio["input"], ratio["output"],
        record["reason"], bucket, key,
    )
    return key


def batch_client_for_route(spec: Any, route: dict) -> Any:
    """Build a batch client for an already-resolved batch route.

    Unreachable today and deliberately loud rather than absent. If
    :func:`resolve_batch_transport` ever succeeds, a route capable of batch
    fan-out exists in the registry and the client that speaks to it belongs in
    ``krepis`` — the declared adapter boundary — not at this call site, which is
    the whole point of the migration this module implements. Constructing a
    provider SDK here would re-create the defect
    (``principles.md`` §2.8; ``model-router-policy.md`` §2 layer 5).

    Raises ``NotImplementedError`` naming the gap, so the batch rung can never
    half-exist: either krepis can drive the resolved route, or the stage fails
    with an actionable message. It must NOT fall through to the sync rung —
    a resolvable-but-undrivable route is a fleet defect, not a degradation.
    """
    raise NotImplementedError(
        "a batch-capable route resolved "
        f"(route={route.get('route')!r}, deployment={getattr(spec, 'model', None)!r}) "
        "but krepis exposes no batch client for it. The batch transport belongs "
        "in the krepis adapter, never at this call site — see "
        "alpha-engine-config-I9263. Add it to krepis and wire it here; do NOT "
        "construct a provider SDK client in crucible-research."
    )
