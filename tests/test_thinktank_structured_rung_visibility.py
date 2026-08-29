"""Every structured call's degradation rung reaches the run manifest
(alpha-engine-config-I7658).

`krepis.llm` populates `structured_output_rung` on EVERY structured result —
degraded or not — expressly so that a degraded call is visible in the
consumer's artifact. Think Tank read it nowhere.

Measured live on 2026-08-18 against the `low` group this run's `sweep` and
`triage` tiers address:

    llm structured provider=litellm_proxy model=low-deepseek-v4-flash-low:
    the endpoint REFUSED response_format (400 - This response_format type is
    unavailable now). Descending the model-portability-policy §7 ladder
    native -> prompt_only for this call and recording the drop on the result.
    → rung: prompt_only

Every `sweep` and `triage` call in the daily run has been one rung down since
the ladder shipped, and `thinktank/runs/{date}/manifest_*.json` said nothing.

RED on ed3fb223 / 2fc065fc: `TierUsage` has no `structured_output_rungs` field
and `_record` takes no rung, so `extra="forbid"` rejects the construction.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from thinktank.client import ThinktankClient
from thinktank.settings import ThinktankSettings, TierSpec


class _Out(BaseModel):
    model_config = {"extra": "forbid"}
    answer: str
    score: int


def _settings() -> ThinktankSettings:
    return ThinktankSettings(
        bucket="alpha-engine-research",
        daily_new_names=5,
        rank_ceiling=150,
        sweep_chunk_size=25,
        stale_after_days=30,
        monthly_budget_usd_default=25.0,
        budget_ssm_param="/thinktank/monthly_budget_usd",
        tiers={"thesis": TierSpec(
            name="thesis", group="med", max_tokens=100, structured_outputs=True)},
    )


class _FakeCompletions:
    def __init__(self, bodies):
        self._bodies = list(bodies)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self._bodies.pop(0)))],
            usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=500),
            # Distinct from the "med" group alias — the krepis I6543
            # masquerade guard refuses to bill/record a call under the alias.
            model="deepseek-v4-flash-max",
        )


_FAKE_ROUTE = {
    "schema_version": 2,
    "group": "med",
    "route": "litellm_proxy",
    "provider": "litellm",
    "deployment_id": "med",
    "api_base_url": "https://router.example:8443",
    "auth_token_type": "litellm_master_key",
    "registry_id": "litellm:group:med",
    "primary_registry_id": "deepseek-v4-flash-max",
    "params": {},
}


def _client(monkeypatch, bodies):
    from datetime import date as _date

    import krepis.router as _kr
    from krepis.cost import PriceCard, PriceTable

    monkeypatch.setenv("LITELLM_MASTER_KEY", "consumer-test")
    monkeypatch.setattr(_kr, "resolve_group_structured", lambda *a, **k: dict(_FAKE_ROUTE))
    table = PriceTable(cards=[PriceCard(
        model_name="deepseek-v4-flash-max", effective_from=_date(2026, 1, 1),
        input_per_1m=1.0, output_per_1m=2.0,
        cache_read_per_1m=0.0, cache_create_per_1m=0.0,
    )])
    monkeypatch.setattr("krepis.cost.load_default_pricing", lambda: table)

    holder = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(bodies)))
    return ThinktankClient(settings=_settings(), run_id="testrun",
                           client_factory=lambda spec, key: holder)


def _pin_rung(monkeypatch, client, rung):
    """Pin the rung krepis reports, without faking the transport twice."""
    import krepis.llm as _kl

    original = _kl.LLMClient.structured

    def _structured(self, **kwargs):
        result = original(self, **kwargs)
        object.__setattr__(result, "structured_output_rung", rung)
        return result

    monkeypatch.setattr(_kl.LLMClient, "structured", _structured)
    return client


@pytest.mark.parametrize("rung", ["native", "tool_emulation", "prompt_only"])
def test_every_rung_reaches_the_manifests_usage_block(monkeypatch, rung):
    """Including `native`: a tier that emits nothing here is unobserved, not
    healthy — the absence of a degradation and an unreported one must not
    render identically."""
    client = _client(monkeypatch, [json.dumps({"answer": "yes", "score": 1})])
    _pin_rung(monkeypatch, client, rung)

    client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)

    usage = client.usage_by_tier()["thesis"]
    assert usage.structured_output_rungs == {rung: 1}
    assert usage.calls == 1


def test_repeated_calls_accumulate_per_rung(monkeypatch):
    client = _client(monkeypatch, [json.dumps({"answer": "y", "score": 1})] * 3)
    _pin_rung(monkeypatch, client, "prompt_only")

    for _ in range(3):
        client.complete("thesis", agent_id="a", system="s", user="u",
                        response_model=_Out)

    assert client.usage_by_tier()["thesis"].structured_output_rungs == {"prompt_only": 3}


def test_a_transport_reporting_no_rung_is_recorded_as_unknown(monkeypatch):
    """Not skipped. A call whose rung the transport did not report has an
    unobserved structured-output path, and an absent key would read as
    'no degradation happened'."""
    client = _client(monkeypatch, [json.dumps({"answer": "y", "score": 1})])
    _pin_rung(monkeypatch, client, None)

    client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)

    assert client.usage_by_tier()["thesis"].structured_output_rungs == {"unknown": 1}


def test_the_manifest_field_survives_serialisation(monkeypatch):
    """`TierUsage` is `extra="forbid"` and the manifest is a published
    contract — the rung histogram has to round-trip, not just exist in memory."""
    client = _client(monkeypatch, [json.dumps({"answer": "y", "score": 1})])
    _pin_rung(monkeypatch, client, "prompt_only")
    client.complete("thesis", agent_id="a", system="s", user="u", response_model=_Out)

    dumped = json.loads(client.usage_by_tier()["thesis"].model_dump_json())
    assert dumped["structured_output_rungs"] == {"prompt_only": 1}
