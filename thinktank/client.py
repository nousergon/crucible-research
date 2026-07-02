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
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

import boto3
from pydantic import BaseModel, ValidationError

from nousergon_lib.secrets import get_secret
from nousergon_lib import sft

from thinktank import SFT_PRODUCER
from thinktank.schemas import TierUsage
from thinktank.settings import ProviderSpec, ThinktankSettings, TierSpec

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a single JSON object matching this JSON Schema — "
    "no prose, no markdown fences:\n{schema}"
)


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
    """Per-run LLM client: tier routing, validation, cost meter, SFT staging."""

    settings: ThinktankSettings
    run_id: str
    run_type: str = "thinktank_daily"
    # test seam: (provider_spec, api_key) -> object exposing .chat.completions.create
    client_factory: Callable[[ProviderSpec, str], Any] | None = None

    _clients: dict[str, Any] = field(default_factory=dict, init=False)
    _usage: dict[str, TierUsage] = field(default_factory=dict, init=False)
    _sft_rows: dict[str, list[_SftRow]] = field(default_factory=dict, init=False)
    _call_seq: int = field(default=0, init=False)

    # ── provider clients ─────────────────────────────────────────────────────

    def _client_for(self, provider: ProviderSpec) -> Any:
        if provider.name not in self._clients:
            api_key = get_secret(provider.key_secret)
            if self.client_factory is not None:
                self._clients[provider.name] = self.client_factory(provider, api_key)
            else:
                # Imported lazily so the package imports without the openai dep
                # in environments that only read thinktank artifacts.
                from openai import OpenAI

                self._clients[provider.name] = OpenAI(
                    base_url=provider.base_url,
                    api_key=api_key,
                    max_retries=3,
                    timeout=180.0,
                )
        return self._clients[provider.name]

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
        """One structured LLM call on the named tier. Validates or raises."""
        tier = self.settings.tier(tier_name)
        provider = self.settings.provider_for(tier)
        client = self._client_for(provider)
        schema = response_model.model_json_schema()

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        kwargs: dict[str, Any] = {
            "model": tier.model,
            "max_tokens": tier.max_tokens,
        }
        if tier.structured_outputs:
            messages.append({"role": "user", "content": user})
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            messages.append(
                {
                    "role": "user",
                    "content": user
                    + _JSON_INSTRUCTION.format(schema=json.dumps(schema)),
                }
            )

        last_error: Exception | None = None
        raw_text = ""
        total_in = total_out = 0
        for attempt in range(2):  # initial + ONE bounded corrective retry
            response = client.chat.completions.create(messages=messages, **kwargs)
            raw_text = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            total_in += in_tok
            total_out += out_tok
            try:
                parsed = response_model.model_validate(_extract_json(raw_text))
                cost = self._record(
                    tier,
                    agent_id=agent_id,
                    sft_meta=sft_meta,
                    messages=messages,
                    kwargs=kwargs,
                    raw_text=raw_text,
                    parsed=parsed,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                )
                return LLMCallResult(
                    parsed=parsed,
                    raw_text=raw_text,
                    model=tier.model,
                    tier=tier.name,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    cost_usd=cost,
                )
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "thinktank tier=%s agent=%s attempt=%d failed validation: %s",
                    tier.name,
                    agent_id,
                    attempt + 1,
                    exc,
                )
                messages = messages + [
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed schema validation with: "
                            f"{exc}\nReturn ONLY the corrected JSON object."
                        ),
                    },
                ]

        # Record spend for the failed attempts too — tokens were consumed.
        self._record(
            tier,
            agent_id=agent_id,
            sft_meta=sft_meta,
            messages=messages,
            kwargs=kwargs,
            raw_text=raw_text,
            parsed=None,
            input_tokens=total_in,
            output_tokens=total_out,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )
        raise ThinktankLLMError(
            f"tier={tier.name} model={tier.model} agent={agent_id}: response failed "
            f"schema validation after bounded retry: {last_error}"
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
        cost = (
            input_tokens * tier.price_in_per_m + output_tokens * tier.price_out_per_m
        ) / 1_000_000
        bucket_usage = self._usage.setdefault(tier.name, TierUsage())
        bucket_usage.calls += 1
        bucket_usage.input_tokens += input_tokens
        bucket_usage.output_tokens += output_tokens
        bucket_usage.cost_usd += cost

        self._call_seq += 1
        self._sft_rows.setdefault(agent_id, []).append(
            _SftRow(
                captured_at=datetime.now(timezone.utc).isoformat(),
                model=tier.model,
                call_seq=self._call_seq,
                input_messages=messages,
                invocation_params={k: v for k, v in kwargs.items() if k != "messages"},
                output_text=raw_text,
                structured_output=parsed.model_dump() if parsed is not None else None,
                usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
                cost_usd=cost,
                # Entity identifiers (ticker / theme key / version / trading
                # day) ride in sft_meta so corpus rows JOIN to judge scores
                # (via capture_run_id -> RubricEvalArtifact.run_id) and to
                # realized outcomes (ticker + trading_day) — the two joins
                # distillation curation needs (judge-filtered SFT,
                # outcome-weighted selection).
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
            logger.info("thinktank SFT capture disabled — %d rows dropped by flag",
                        sum(len(v) for v in self._sft_rows.values()))
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
                    meta=r.meta,
                )
                for r in rows
            ]
            key = f"decision_artifacts/_sft_raw/{trading_day}/{self.run_id}/{agent_id}.jsonl"
            try:
                client.put_object(
                    Bucket=bucket, Key=key, Body=sft.to_jsonl_bytes(records)
                )
            except Exception as exc:  # noqa: BLE001 — re-raised loud below
                raise SftCaptureWriteError(
                    f"SFT flush failed for s3://{bucket}/{key}: {exc}"
                ) from exc
            flushed += len(records)
        self._sft_rows.clear()
        return flushed


def _extract_json(text: str) -> Any:
    """Parse a JSON object out of model text (tolerates markdown fences)."""
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # last resort: widest brace span (some models add a preamble sentence)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
