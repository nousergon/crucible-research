"""Think Tank tiers addressed by capability group rather than a pinned provider.

Brian's ruling 2026-08-03 (alpha-engine-config-I6367): no agent may be
directly linked to OpenRouter. Before this, all four Think Tank tiers pinned
``provider: openrouter`` with a single ``base_url`` and no fallback — so when
the OpenRouter account balance went negative on 2026-08-02, four consecutive
daily runs aborted mid-loop and ``thinktank/challenger_selection/`` stopped
being written entirely.

The group's chain has no OpenRouter primary, so the same event would not have
touched a group-addressed run at all.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from thinktank.client import ThinktankClient, ThinktankLLMError  # noqa: E402
from thinktank.settings import (  # noqa: E402
    ThinktankSettings,
    TierSpec,
    _parse_tier,
)


# ── config parsing ───────────────────────────────────────────────────────


class TestParseTier:
    def test_group_addressed_tier(self):
        tier = _parse_tier("thesis", {"group": "med", "max_tokens": 8000})
        assert tier.is_group_addressed
        assert tier.group == "med"
        assert tier.provider is None and tier.model is None

    def test_pinned_tier_still_parses(self):
        """The pinned form stays expressible — a tier may be held on a
        specific model deliberately, and that choice should be visible in the
        config rather than implied by its absence."""
        tier = _parse_tier("thesis", {
            "provider": "openrouter", "model": "x/y", "max_tokens": 8000,
            "price_in_per_m": 0.1, "price_out_per_m": 0.2,
        })
        assert not tier.is_group_addressed
        assert tier.model == "x/y"

    def test_both_forms_is_rejected(self):
        """Not resolved by precedence: one of the two would be silently
        ignored, and which one is exactly the sort of fact discovered during
        an incident."""
        with pytest.raises(ValueError, match="BOTH"):
            _parse_tier("thesis", {
                "group": "med", "provider": "openrouter", "model": "x/y",
                "max_tokens": 8000,
            })

    def test_neither_form_is_rejected(self):
        with pytest.raises(ValueError, match="neither"):
            _parse_tier("thesis", {"max_tokens": 8000})

    def test_group_tier_carrying_stale_prices_is_rejected(self):
        """Under group addressing the serving model is a call-time fact. A
        leftover per-tier price literal would read as authoritative and bill
        the wrong card the moment the chain fell through."""
        with pytest.raises(ValueError, match="price_in_per_m"):
            _parse_tier("thesis", {
                "group": "med", "max_tokens": 8000, "price_in_per_m": 0.1,
            })


# ── client: which spec a group-addressed tier produces ───────────────────


def _settings(tier: TierSpec) -> ThinktankSettings:
    return ThinktankSettings(
        bucket="b", daily_new_names=1, rank_ceiling=1, sweep_chunk_size=1,
        stale_after_days=1, monthly_budget_usd_default=1.0,
        budget_ssm_param="/p", providers={}, tiers={tier.name: tier},
    )


def _fake_route(**over):
    route = {
        "schema_version": 2, "group": "med", "route": "litellm_proxy",
        "provider": "litellm", "deployment_id": "med",
        "api_base_url": "https://router.example:8443",
        "auth_token_type": "litellm_master_key",
        "registry_id": "litellm:group:med",
        "primary_registry_id": "deepseek-v4-flash-max", "params": {},
    }
    route.update(over)
    return route


class TestGroupAddressedClient:
    def test_spec_targets_the_router_edge_not_a_provider(self, monkeypatch):
        import krepis.router as _kr
        from krepis.llm_config import TRANSPORT_OPENAI

        monkeypatch.setenv("LITELLM_MASTER_KEY", "consumer-test")
        monkeypatch.setattr(
            _kr, "resolve_group_structured", lambda *a, **k: _fake_route()
        )
        tier = TierSpec(name="thesis", group="med", max_tokens=8000)
        client = ThinktankClient(settings=_settings(tier), run_id="r")

        llm = client._llm_client_for(tier, callsite_id="thinktank-thesis")
        assert llm.spec.model == "med"
        assert llm.spec.base_url == "https://router.example:8443"
        # Provider "litellm" would bind TRANSPORT_LITELLM — krepis' IN-PROCESS
        # Router, which calls each provider directly from this box and reads
        # OPENROUTER_API_KEY from the environment as it goes. That is the
        # linkage the ruling forbids, wearing the router's name.
        assert llm.spec.transport == TRANSPORT_OPENAI
        assert llm.spec.provider != "litellm"

    def test_declares_where_it_runs_and_nothing_else(self, monkeypatch):
        import krepis.router as _kr

        seen = {}
        monkeypatch.setenv("LITELLM_MASTER_KEY", "consumer-test")

        def _resolve(group, *, exec_context=None, wire="openai"):
            seen.update(group=group, exec_context=exec_context, wire=wire)
            return _fake_route()

        monkeypatch.setattr(_kr, "resolve_group_structured", _resolve)
        tier = TierSpec(name="thesis", group="med", max_tokens=8000)
        ThinktankClient(settings=_settings(tier), run_id="r")._llm_client_for(
            tier, callsite_id="thinktank-thesis"
        )
        assert seen == {"group": "med", "exec_context": "ec2", "wire": "openai"}

    def test_structured_outputs_requirement_survives_group_addressing(
        self, monkeypatch
    ):
        """`sweep` is the one tier that needs strict json_schema. It is a
        live-verified per-tier requirement of this codebase, not a routing
        preference, so it must still reach the spec."""
        import krepis.router as _kr

        monkeypatch.setenv("LITELLM_MASTER_KEY", "consumer-test")
        monkeypatch.setattr(
            _kr, "resolve_group_structured",
            lambda *a, **k: _fake_route(params={"structured_outputs": False}),
        )
        tier = TierSpec(
            name="sweep", group="low", max_tokens=4000, structured_outputs=True
        )
        llm = ThinktankClient(
            settings=_settings(tier), run_id="r"
        )._llm_client_for(tier, callsite_id="thinktank-sweep")
        assert llm.spec.structured_outputs is True


# ── cost: priced from what actually served ───────────────────────────────


class TestGroupAddressedCost:
    def _client(self, tier):
        return ThinktankClient(settings=_settings(tier), run_id="r")

    def test_provider_reported_cost_wins(self):
        tier = TierSpec(name="thesis", group="med", max_tokens=8000)
        cost = self._client(tier)._cost_for(
            tier, input_tokens=1000, output_tokens=1000,
            served_model="deepseek-v4-flash", provider_cost_usd=0.0042,
        )
        assert cost == 0.0042

    def test_falls_back_to_the_price_card_for_the_SERVED_model(self):
        """Not the tier's model, and not the group's primary — the model that
        actually answered. Pricing a fallback at the primary's rate is a
        silently wrong number on exactly the runs that degraded."""
        tier = TierSpec(name="thesis", group="med", max_tokens=8000)
        cost = self._client(tier)._cost_for(
            tier, input_tokens=1_000_000, output_tokens=0,
            served_model="deepseek-v4-flash", provider_cost_usd=None,
        )
        assert cost == pytest.approx(0.14, rel=1e-6)

    def test_unpriceable_call_raises_rather_than_costing_zero(self):
        """A zero would make the monthly budget guard fail OPEN — that guard
        is the only thing bounding this run's spend."""
        tier = TierSpec(name="thesis", group="med", max_tokens=8000)
        with pytest.raises(ThinktankLLMError, match="fail open"):
            self._client(tier)._cost_for(
                tier, input_tokens=10, output_tokens=10,
                served_model="no-such-model-anywhere",
                provider_cost_usd=None,
            )

    def test_pinned_tier_still_prices_from_its_literals(self):
        tier = TierSpec(
            name="thesis", provider="openrouter", model="x/y", max_tokens=8000,
            price_in_per_m=1.0, price_out_per_m=2.0,
        )
        cost = self._client(tier)._cost_for(
            tier, input_tokens=1_000_000, output_tokens=1_000_000,
            served_model="ignored", provider_cost_usd=None,
        )
        assert cost == pytest.approx(3.0)
