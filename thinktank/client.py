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
            provider = self.settings.provider_for(tier)
            api_key = get_secret(provider.key_secret)
            spec = ModelSpec(
                provider=provider.name,
                model=tier.model,
                max_tokens=tier.max_tokens,
                base_url=provider.base_url,
                api_key_env=provider.key_secret,
                structured_outputs=tier.structured_outputs,
                # reasoning not used by thinktank tiers — preserve current
                # behaviour (no explicit reasoning control).
            )
            client = LLMClient(
                spec,
                callsite_id=callsite_id,
                api_key=api_key,
                client_factory=(self._adapt_client_factory() if self.client_factory is not None else None),
                max_retries=3,
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
            self._record(
                tier,
                agent_id=agent_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                kwargs={"model": tier.model, "max_tokens": tier.max_tokens},
                raw_text="",
                parsed=None,
                input_tokens=total_in,
                output_tokens=total_out,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                sft_meta=sft_meta,
            )
            raise ThinktankLLMError(
                f"tier={tier.name} model={tier.model} agent={agent_id}: response failed after bounded retries: {exc}"
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
    ) -> float:
        cost = (input_tokens * tier.price_in_per_m + output_tokens * tier.price_out_per_m) / 1_000_000
        bucket_usage = self._usage.setdefault(tier.name, TierUsage())
        bucket_usage.calls += 1
        bucket_usage.input_tokens += input_tokens
        bucket_usage.output_tokens += output_tokens
        bucket_usage.cost_usd += cost

        self._call_seq += 1
        self._sft_rows.setdefault(agent_id, []).append(
            _SftRow(
                captured_at=datetime.now(UTC).isoformat(),
                model=tier.model,
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
