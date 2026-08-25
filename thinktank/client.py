"""Provider-agnostic LLM client — the plug-and-play chokepoint.

Wire contract: OpenAI-compatible chat completions. Every serving option the
plan cares about (OpenRouter, DeepInfra, Fireworks, Together, Groq, self-hosted
vLLM, OpenAI native, Anthropic via its OpenAI-compat endpoint) speaks it, so
swapping models/providers is a ``thinktank.yaml`` registry edit — never a code
change here.

Structured outputs, portably:
- tiers with ``structured_outputs: true`` send ``response_format=json_schema``
  (strict) — the institutional path where the provider supports it;
- all tiers validate the response against the target Pydantic model, with ONE
  bounded corrective retry (the validation error is fed back), then fail LOUD
  (``ThinktankLLMError``). No silent degradation.

Every call stamps cost (registry prices × usage) and stages an SFT row for the
shared distillation corpus via the ``nousergon_lib.sft`` chokepoint
(producer ``crucible_thinktank``).

**alpha-engine-config-I5223 (2026-07-29):** the inner transport is now
``krepis.llm.LLMClient.structured()`` — the library this client was the fork
of. Per-call cost emission flows through ``krepis.cost_sink.S3JsonlCostSink``
for the shared telemetry stream; thinktank's own cost tracking (budget guard,
manifest, SFT staging) is unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

import boto3
from krepis.cost_sink import S3JsonlCostSink
from krepis.llm import LLMClient, LLMError
from krepis.llm_config import ModelSpec
from krepis.router import resolve_group_spec, served_model_for_deployment
from nousergon_lib import sft
from nousergon_lib.secrets import get_secret
from pydantic import BaseModel

from thinktank import SFT_PRODUCER
from thinktank.schemas import TierUsage
from thinktank.settings import ProviderSpec, ThinktankSettings, TierSpec

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Retry budget handed to ``krepis.llm.LLMClient.structured()``.
#
# The library defaults to ``attempts=2``. The fork this file replaces ran THREE
# transport attempts and THREE non-JSON-body attempts, each on its own budget,
# plus one schema-corrective retry (config#3072). Taking the library default
# would therefore have SHRUNK the budget as a side effect of consolidating —
# and shrunk it on exactly the failure that motivated this work: live incident
# 2026-07-30 (flow-doctor report 019fb37b8b792803c2a92b924042), where OpenRouter
# answered 200 with 8228 bytes of keep-alive padding and no payload, would get
# one retry instead of three.
#
# 3 is the fork's per-class budget, preserved. It now covers every retryable
# cause at once (non-JSON body, null-choices body, schema validation) because
# the library uses one budget for all of them — a consolidation of BUDGETS,
# never a reduction of the number of attempts any single failure gets. krepis
# >= 0.25.0 backs off between attempts (krepis#93), so this is not a tight loop.
_STRUCTURED_ATTEMPTS = 3

THINKTANK_EXEC_CONTEXT = "ec2"
"""Where the Think Tank runs — a fact, never a routing preference (R28/R29).

The daily run is a self-terminating EC2 spot box launched by
``alpha-engine-thinktank-spot-dispatcher`` (ARCHITECTURE §47 / config-I5208).
Declared rather than inferred, and overridable by ``KREPIS_EXEC_CONTEXT`` for
the local/CI case where the same code runs off the box."""


class ThinktankLLMError(RuntimeError):
    """A tier call failed validation after its bounded retry — fail loud."""


class SftCaptureWriteError(RuntimeError):
    """SFT corpus flush failed — loud, never swallowed (no-silent-fails)."""


@dataclass
class LLMCallResult:
    parsed: BaseModel
    raw_text: str
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class _SftRow:
    captured_at: str
    model: str
    call_seq: int
    input_messages: list[dict[str, str]]
    invocation_params: dict[str, Any]
    output_text: str
    structured_output: dict[str, Any] | None
    usage: dict[str, int]
    cost_usd: float
    meta: dict[str, Any]


@dataclass
class ThinktankClient:
    """Per-run LLM client: tier routing, validation, cost meter, SFT staging.

    **alpha-engine-config-I5223:** the inner transport is now
    ``krepis.llm.LLMClient`` — each tier gets its own client instance
    (lazily created and cached per tier name), replacing the forked
    ``OpenAI(...)`` construction this module was extracted from. Cost
    telemetry flows through the shared ``krepis.cost_sink`` chokepoint
    while thinktank's own SFT staging + budget tracking stay unchanged.
    """

    settings: ThinktankSettings
    run_id: str
    run_type: str = "thinktank_daily"
    # test seam: (provider_spec, api_key) -> object exposing .chat.completions.create
    client_factory: Callable[[ProviderSpec, str], Any] | None = None
    # ── I5223: cost_sink for shared telemetry (set by run.py) ────────────
    cost_sink: S3JsonlCostSink | None = None

    _llm_clients: dict[str, LLMClient] = field(default_factory=dict, init=False)
    _usage: dict[str, TierUsage] = field(default_factory=dict, init=False)
    _sft_rows: dict[str, list[_SftRow]] = field(default_factory=dict, init=False)
    _call_seq: int = field(default=0, init=False)

    # ── provider clients ─────────────────────────────────────────────────────

    def _llm_client_for(self, tier: TierSpec, *, callsite_id: str) -> LLMClient:
        """Return (or create and cache) the ``LLMClient`` for *tier*.

        One client per (tier name, callsite_id) — themes-macro and themes-sector
        share the same tier but are distinct call sites. The callsite_id is set
        on the cached client; callers that need a different callsite_id on the
        same tier pass the one they need.
        """
        cache_key = f"{tier.name}:{callsite_id}"
        if cache_key not in self._llm_clients:
            if tier.is_group_addressed:
                # The registry decides model, endpoint and credential; this
                # module decides only which capability tier it wants and
                # where it runs. `structured_outputs` is still ours: it is a
                # per-tier requirement this codebase has live-verified, not a
                # routing preference.
                spec, _route = resolve_group_spec(
                    tier.group,
                    exec_context=THINKTANK_EXEC_CONTEXT,
                    # This client is on the openai transport; asking for the
                    # anthropic wire could hand it a URL it cannot speak.
                    wire="openai",
                    max_tokens=tier.max_tokens,
                    structured_outputs=tier.structured_outputs,
                )
                api_key = None  # resolved by krepis from spec.api_key_env
            else:
                provider = self.settings.provider_for(tier)
                api_key = get_secret(provider.key_secret)
                spec = ModelSpec(
                    provider=provider.name,
                    model=tier.model,
                    max_tokens=tier.max_tokens,
                    base_url=provider.base_url,
                    api_key_env=provider.key_secret,
                    structured_outputs=tier.structured_outputs,
                    # reasoning not used by pinned thinktank tiers — preserve
                    # current behaviour (no explicit reasoning control).
                )
            client = LLMClient(
                spec,
                callsite_id=callsite_id,
                api_key=api_key,
                client_factory=(self._adapt_client_factory() if self.client_factory is not None else None),
                # max_retries=10, was 3 (alpha-engine-config-I8351).
                #
                # THIS BOUNDS A RESTART, NOT A RATE LIMIT. The OpenAI SDK's
                # backoff is 0.5s, 1s, 2s, 4s, then 8s per attempt, so 3
                # retries span 3.2 SECONDS. The dependency it is retrying
                # against — the LiteLLM router — takes about FIFTY seconds to
                # come back (config generation, git pull, prisma migrate), and
                # it is restarted by ordinary merges: `router-config-deploy.yml`
                # fires on any alpha-engine-config registry merge, at whatever
                # hour that merge happens.
                #
                # Measured 2026-08-25. Two runs died this way, 17 minutes
                # apart, each after real work:
                #
                #   deploy 17:02:35 -> router stop 17:04:38 -> 502 -> ABORTED
                #                      after 4 thesis writes
                #   deploy 17:18:03 -> router stop 17:21:03 -> 502 -> ABORTED
                #                      after 6 thesis writes
                #
                # Both retried for 3.2s against a ~50s outage and gave up with
                # 46 seconds still to go. This run is ~50 minutes, NOT
                # resumable, spot-priced, and withholds
                # `challenger_selection/latest.json` on abort — so the
                # asymmetry is total: waiting a minute costs a minute, and not
                # waiting costs the run, the box, the spend already incurred,
                # and another day of the arm's only health signal being stale.
                #
                # 10 retries span ~55s (0.5+1+2+4+8*6), which clears the
                # measured restart with margin. It fits: the box budget is
                # 5400s with a terminal-write reserve, and `timeout=180.0`
                # still bounds each individual attempt.
                #
                # NOT raised fleet-wide, deliberately. A Lambda with a
                # 15-minute ceiling must not sit in a minute of backoff; this
                # is the long-lived box, which is the caller that benefits.
                # `_STRUCTURED_ATTEMPTS` above is a DIFFERENT budget — it
                # bounds schema/body corrective retries, not availability.
                max_retries=10,
                timeout=180.0,
                cost_sink=self.cost_sink,
            )
            self._llm_clients[cache_key] = client
        return self._llm_clients[cache_key]

    def _adapt_client_factory(self) -> Callable[[ModelSpec, str], Any] | None:
        """Adapt the old ``client_factory(ProviderSpec, api_key) → OpenAI``
        test seam into the ``LLMClient`` shape ``(ModelSpec, api_key) → …``.

        Only used when tests inject a factory. Returns None when no factory
        is set, so ``LLMClient`` uses its own default construction.
        """
        if self.client_factory is None:
            return None

        def _adapted(spec: ModelSpec, api_key: str) -> Any:
            # Reconstruct the old ProviderSpec shape for the test seam
            provider = ProviderSpec(
                name=spec.provider,
                base_url=spec.base_url or "",
                key_secret=spec.api_key_env or "",
            )
            return self.client_factory(provider, api_key)  # type: ignore[misc]

        return _adapted

    # ── the one call surface ─────────────────────────────────────────────────

    def complete(
        self,
        tier_name: str,
        *,
        agent_id: str,
        system: str,
        user: str,
        response_model: type[T],
        prompt_id: str = "",
        prompt_version: str = "",
        sft_meta: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        """One structured LLM call on the named tier. Validates or raises.

        **I5223:** routes through ``krepis.llm.LLMClient.structured()``
        instead of the forked ``OpenAI(...).chat.completions.create()``
        this module used before.

        The library collapses into ONE ``attempts`` budget what the fork kept
        as two (a transport-failure budget and a schema-corrective budget).
        That is the intended consolidation — but the budget must not SHRINK on
        the way through, so it is passed explicitly rather than defaulted. See
        ``_STRUCTURED_ATTEMPTS``.
        """
        tier = self.settings.tier(tier_name)
        callsite_id = _callsite_id_for(tier_name, agent_id)
        client = self._llm_client_for(tier, callsite_id=callsite_id)

        try:
            result = client.structured(
                system=system,
                user_content=user,
                schema=response_model,
                schema_name=response_model.__name__,
                max_tokens=tier.max_tokens,
                attempts=_STRUCTURED_ATTEMPTS,
            )
        except LLMError as exc:
            # LLMClient raises LLMError on exhaustion. Record the failed
            # spend (usage is carried on the exception per the lib contract)
            # then re-raise as ThinktankLLMError for caller compatibility.
            if exc.usage is not None:
                total_in = exc.usage.input_tokens
                total_out = exc.usage.output_tokens
            else:
                total_in = total_out = 0
            try:
                self._record(
                    tier,
                    agent_id=agent_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    kwargs={
                        "model": tier.model or tier.group,
                        "max_tokens": tier.max_tokens,
                    },
                    raw_text="",
                    parsed=None,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                    sft_meta=sft_meta,
                    served_model=_priceable_model_for_failed_call(client, tier),
                )
            except Exception as record_exc:
                # NO-SILENT-FAILS DEVIATION, deliberate and narrow.
                #
                # (a) Swallowed: the accounting of a call that has ALREADY
                #     failed. `_cost_for` raises when a group-addressed tier
                #     cannot be priced — and a failed call is precisely the
                #     state that has no served model, so the metering raise
                #     fired ON the failure path and REPLACED the transport
                #     error that caused it. Live: run b4a21acfbbfc
                #     (2026-08-11) aborted reporting "add a price card for
                #     '' in krepis model_pricing.yaml"; the actual pillar-tier
                #     failure it was masking is unrecoverable from that run's
                #     logs. An error handler must never be able to destroy the
                #     error it is handling.
                # (b) The deliverable survives: the original LLMError is
                #     re-raised immediately below, the run still aborts with a
                #     non-zero exit, and the manifest still records the abort.
                # (c) Recorded on: this logger at ERROR (flow-doctor is
                #     attached to the root logger at ERROR, so it alerts), and
                #     the run manifest via run.py's abort handler.
                logger.error(
                    "[thinktank] tier=%s agent=%s: the FAILED call could not "
                    "be metered (%s: %s) — its %d in / %d out tokens are "
                    "absent from this run's spend, so the budget guard reads "
                    "LOW by that much. Continuing to the real cause.",
                    tier.name, agent_id,
                    type(record_exc).__name__, record_exc,
                    total_in, total_out,
                )
            raise ThinktankLLMError(
                f"tier={tier.name} target={tier.model or tier.group} "
                f"agent={agent_id}: response failed after bounded retries: {exc}"
            ) from exc

        usage = result.usage
        cost = self._record(
            tier,
            agent_id=agent_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            kwargs=result.raw_request,
            raw_text=result.text,
            parsed=result.parsed,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            sft_meta=sft_meta,
            served_model=result.model or "",
            provider_cost_usd=usage.provider_cost_usd,
            structured_output_rung=getattr(result, "structured_output_rung", None),
        )
        return LLMCallResult(
            parsed=result.parsed,
            raw_text=result.text,
            model=result.model,
            tier=tier.name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
        )

    # ── cost ─────────────────────────────────────────────────────────────────

    def _cost_for(
        self,
        tier: TierSpec,
        *,
        input_tokens: int,
        output_tokens: int,
        served_model: str,
        provider_cost_usd: float | None,
    ) -> float:
        """Cost for one call, priced from what actually served it.

        A pinned tier carries its own price literals and keeps using them —
        the model is fixed, so the literal cannot drift out from under it.

        A GROUP-addressed tier has no such literal and must not invent one:
        the serving model is not known until the response returns, so pricing
        from a per-tier constant would bill the wrong card the moment the
        chain fell through to a fallback. Order of preference:

        1. ``provider_cost_usd`` — what the provider itself billed. Exact.
        2. ``krepis.cost.PriceCard`` for the model that actually served.
        3. RAISE. There is no third guess worth making: a silently-zero cost
           would make the monthly budget guard fail OPEN, and that guard is
           the only thing bounding this run's spend.
        """
        if not tier.is_group_addressed:
            return (
                input_tokens * (tier.price_in_per_m or 0.0)
                + output_tokens * (tier.price_out_per_m or 0.0)
            ) / 1_000_000

        if provider_cost_usd is not None:
            return float(provider_cost_usd)

        try:
            from krepis.cost import load_default_pricing

            card = load_default_pricing().get(served_model, datetime.now(UTC))
            return (
                input_tokens * card.input_per_1m
                + output_tokens * card.output_per_1m
            ) / 1_000_000
        except Exception as exc:
            # No silent swallow, and deliberately not a zero: the budget guard
            # reads this number, so an unpriced call would raise the run's
            # spend ceiling rather than lower it (no-silent-fails).
            raise ThinktankLLMError(
                f"tier={tier.name} group={tier.group} served_model="
                f"{served_model!r}: the provider returned no cost and no "
                f"price card could be resolved, so this call cannot be "
                f"metered. The monthly budget guard reads this number — "
                f"pricing it as zero would make that guard fail open. "
                f"Add a price card for {served_model!r} in krepis "
                f"model_pricing.yaml."
            ) from exc

    # ── accounting + SFT staging ─────────────────────────────────────────────

    def _record(
        self,
        tier: TierSpec,
        *,
        agent_id: str,
        messages: list[dict[str, str]],
        kwargs: dict[str, Any],
        raw_text: str,
        parsed: BaseModel | None,
        input_tokens: int,
        output_tokens: int,
        prompt_id: str,
        prompt_version: str,
        sft_meta: dict[str, Any] | None = None,
        served_model: str = "",
        provider_cost_usd: float | None = None,
        structured_output_rung: str | None = None,
    ) -> float:
        cost = self._cost_for(
            tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            served_model=served_model,
            provider_cost_usd=provider_cost_usd,
        )
        bucket_usage = self._usage.setdefault(tier.name, TierUsage())
        bucket_usage.calls += 1
        bucket_usage.input_tokens += input_tokens
        bucket_usage.output_tokens += output_tokens
        bucket_usage.cost_usd += cost
        # "unknown" rather than skipping: a call whose rung the transport did
        # not report is a call whose structured-output path is unobserved, and
        # an absent key would read as "no degradation happened".
        rung = structured_output_rung or "unknown"
        bucket_usage.structured_output_rungs[rung] = (
            bucket_usage.structured_output_rungs.get(rung, 0) + 1
        )

        self._call_seq += 1
        self._sft_rows.setdefault(agent_id, []).append(
            _SftRow(
                captured_at=datetime.now(UTC).isoformat(),
                model=served_model or tier.model or tier.group or tier.name,
                call_seq=self._call_seq,
                input_messages=messages,
                invocation_params={k: v for k, v in kwargs.items() if k != "messages"},
                output_text=raw_text,
                structured_output=parsed.model_dump() if parsed is not None else None,
                usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
                cost_usd=cost,
                meta={
                    "run_id": self.run_id,
                    "agent_id": agent_id,
                    "run_type": self.run_type,
                    "tier": tier.name,
                    "prompt_id": prompt_id,
                    "prompt_version": prompt_version,
                    **(sft_meta or {}),
                },
            )
        )
        return cost

    def usage_by_tier(self) -> dict[str, TierUsage]:
        return {k: v.model_copy(deep=True) for k, v in self._usage.items()}

    def total_cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self._usage.values()), 6)

    def flush_sft(self, s3_client: Any | None, bucket: str, trading_day: str) -> int:
        """Flush staged SFT rows to the shared corpus prefix. Loud on failure.

        Gated on ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` upstream (run.py) —
        the same operator switch the rest of the fleet's capture uses.
        """
        import os

        if os.environ.get("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "").lower() not in (
            "1",
            "true",
        ):
            logger.info(
                "thinktank SFT capture disabled — %d rows dropped by flag", sum(len(v) for v in self._sft_rows.values())
            )
            return 0
        client = s3_client or boto3.client("s3")
        flushed = 0
        for agent_id, rows in self._sft_rows.items():
            if not rows:
                continue
            records = [
                sft.build_record(
                    SFT_PRODUCER,
                    captured_at=r.captured_at,
                    model=r.model,
                    call_seq=r.call_seq,
                    input_messages=r.input_messages,
                    invocation_params=r.invocation_params,
                    output_text=r.output_text,
                    structured_output=r.structured_output,
                    usage=r.usage,
                    cost_usd=r.cost_usd,
                    source="live",
                    meta=r.meta,
                )
                for r in rows
            ]
            key = f"decision_artifacts/_sft_raw/{trading_day}/{self.run_id}/{agent_id}.jsonl"
            try:
                client.put_object(Bucket=bucket, Key=key, Body=sft.to_jsonl_bytes(records))
            except Exception as exc:  # noqa: BLE001 — re-raised loud below
                raise SftCaptureWriteError(f"SFT flush failed for s3://{bucket}/{key}: {exc}") from exc
            flushed += len(records)
        self._sft_rows.clear()
        return flushed


def _extract_json(text: str) -> Any:
    """Parse a JSON object out of model text (tolerates markdown fences)."""
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError(f"no JSON object found in response: {text[:200]!r}") from e


def _priceable_model_for_failed_call(client: LLMClient, tier: TierSpec) -> str:
    """The model a FAILED call's tokens are metered against.

    A failed call reports no served model — the response never arrived, or
    arrived unusable. That is not the same as *unknown*: the deployment this
    client addressed was resolved BEFORE the call was made, and for a
    group-addressed tier it is the qualified ``{group}-{mid}`` deployment
    name, which ``krepis.router`` maps back to the upstream model identifier
    price cards are written against.

    Pricing the failed attempt there is an ESTIMATE and is labelled one: a
    fallback deeper in the chain may be what actually burned the tokens, and
    its card may differ in either direction. It is preferred to no number at
    all because the monthly budget guard reads the total — a failed call
    contributing nothing understates the run's real spend.

    Returns ``""`` for a pinned tier, which prices from its own literals and
    never needed a served model. Where the mapping itself fails, returns the
    deployment name unchanged — unpriceable, and deliberately so: the caller
    logs an unpriceable failed call rather than raising on it, so nothing
    here can become the thing that masks a transport error.
    """
    if not tier.is_group_addressed:
        return ""
    addressed = getattr(getattr(client, "spec", None), "model", "") or ""
    if not addressed:
        return ""
    try:
        return served_model_for_deployment(addressed) or addressed
    except Exception as exc:
        logger.warning(
            "[thinktank] could not map addressed deployment %r back to an "
            "upstream model for failed-call metering (%s: %s) — pricing will "
            "fall through to the deployment name",
            addressed, type(exc).__name__, exc,
        )
        return addressed


def _callsite_id_for(tier_name: str, agent_id: str) -> str:
    """Map (tier_name, agent_id) → registered callsite_id.

    The 5 thinktank sites:
      thesis + analyst_thesis  → thinktank-thesis
      pillar + analyst_pillar  → thinktank-pillar
      sweep  + analyst_sweep   → thinktank-sweep
      themes + themes_macro    → thinktank-themes-macro
      themes + themes_sector   → thinktank-themes-sector

    The themes tier is the only one that maps to two different callsite_ids.
    The agent_id disambiguates.
    """
    if tier_name == "thesis":
        return "thinktank-thesis"
    if tier_name == "pillar":
        return "thinktank-pillar"
    if tier_name == "sweep":
        return "thinktank-sweep"
    if tier_name == "themes":
        if "sector" in agent_id:
            return "thinktank-themes-sector"
        return "thinktank-themes-macro"
    # Fallback — should not be reached for registered tiers
    logger.warning(
        "unregistered callsite for tier=%r agent=%r — using derived id",
        tier_name,
        agent_id,
    )
    return f"thinktank-{tier_name}"
