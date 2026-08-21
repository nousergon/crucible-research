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

        # `requires` is part of resolve_group_structured's signature since
        # krepis v0.59.22 (alpha-engine-config-I7904 — capability-aware
        # selection). A double that omits it does not merely miss the new
        # argument, it TypeErrors the moment any caller passes one, so the
        # double must track the real signature.
        def _resolve(group, *, exec_context=None, wire="openai", requires=()):
            seen.update(
                group=group, exec_context=exec_context, wire=wire, requires=requires
            )
            return _fake_route()

        monkeypatch.setattr(_kr, "resolve_group_structured", _resolve)
        tier = TierSpec(name="thesis", group="med", max_tokens=8000)
        ThinktankClient(settings=_settings(tier), run_id="r")._llm_client_for(
            tier, callsite_id="thinktank-thesis"
        )
        assert seen == {
            "group": "med",
            "exec_context": "ec2",
            "wire": "openai",
            "requires": (),
        }

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


# ── the config file after the flip ───────────────────────────────────────


class TestFullyGroupAddressedConfig:
    """A thinktank.yaml with NO `providers` block at all — the shape the file
    takes once every tier is group-addressed (alpha-engine-config-I6373 step
    6). Requiring an empty block would make the absence of direct provider
    linkage look like a malformed file."""

    def _write(self, tmp_path, body):
        p = tmp_path / "thinktank.yaml"
        p.write_text(body)
        return p

    _NO_PROVIDERS = """
thinktank:
  bucket: alpha-engine-research
  coverage:
    daily_new_names: 5
    rank_ceiling: 150
    sweep_chunk_size: 25
    stale_after_days: 30
  budget:
    monthly_usd_default: 25.0
    ssm_param: /thinktank/monthly_budget_usd
  llm:
    tiers:
      sweep:
        group: low
        max_tokens: 4000
        structured_outputs: true
      thesis:
        group: med
        max_tokens: 8000
        structured_outputs: false
"""

    def test_loads_with_no_providers_block(self, tmp_path, monkeypatch):
        from thinktank.settings import load_settings

        monkeypatch.setenv(
            "THINKTANK_CONFIG_PATH", str(self._write(tmp_path, self._NO_PROVIDERS))
        )
        s = load_settings()
        assert s.providers == {}
        assert {n: t.group for n, t in s.tiers.items()} == {
            "sweep": "low", "thesis": "med",
        }
        assert all(t.is_group_addressed for t in s.tiers.values())

    def test_a_malformed_provider_still_hard_fails(self, tmp_path, monkeypatch):
        """Only the TOTAL absence of the block is meaningful. A provider
        missing base_url is a mistake and must not be tolerated by the same
        leniency."""
        import pytest as _pytest

        from thinktank.settings import load_settings

        body = self._NO_PROVIDERS.replace(
            "  llm:\n    tiers:",
            "  llm:\n    providers:\n      broken:\n        key_secret: X\n    tiers:",
        )
        monkeypatch.setenv("THINKTANK_CONFIG_PATH", str(self._write(tmp_path, body)))
        with _pytest.raises(KeyError):
            load_settings()


# ── The credential half of the same contract (config-I6373) ──────────────


class TestRouterCredentialIsResolvableByKrepis:
    """`_build_client` passes ``api_key=None`` and lets krepis resolve the
    credential from ``spec.api_key_env``. That is a CONTRACT with the library,
    and it is the half that took the Think Tank down.

    The box's credential is a per-consumer SecureString at
    ``/alpha-engine/ROUTER_CONSUMER_THINKTANK``, and it lives only there on
    purpose — the dispatcher hands the box the credential's NAME, never its
    value, so the secret never enters an SSM command string, the CloudWatch
    log that command streams to, or the bootstrap log shipped to S3.

    Through krepis 0.33.0 ``LLMClient._resolve_api_key`` read ``os.environ``
    alone. Route admission resolved the same credential on the full chain, so
    the run was ADMITTED to the edge and then aborted at the first call:

        LLMConfigError: no API key for provider 'litellm_proxy': pass api_key=
        or set the ROUTER_CONSUMER_THINKTANK environment variable

    Live on 2026-08-04 (``thinktank/runs/2026-08-04/manifest_1d6e7a653137``):
    aborted after 5s, 0 theses, ``total_cost_usd`` 0, ``challenger_selection``
    unwritten. krepis>=0.34.0 is the floor that closes it.

    This asserts the library CAPABILITY rather than the version string: a
    floor is satisfiable by a resolver that no longer behaves this way, and the
    failure it guards is a daily run at 14:30 UTC, not a CI red.
    """

    def test_edge_credential_resolves_from_a_non_environment_source(
        self, monkeypatch
    ):
        from krepis.llm import LLMClient
        from krepis.llm_config import ModelSpec
        from krepis.router import ROUTER_EDGE_PROVIDER

        monkeypatch.delenv("ROUTER_CONSUMER_THINKTANK", raising=False)
        monkeypatch.setattr(
            "krepis.router._litellm_master_key_from_ssm",
            lambda name="LITELLM_MASTER_KEY": (
                "sk-live" if name == "ROUTER_CONSUMER_THINKTANK" else None
            ),
        )
        spec = ModelSpec(
            provider=ROUTER_EDGE_PROVIDER,
            model="med",
            base_url="https://router.example.invalid:8443",
            api_key_env="ROUTER_CONSUMER_THINKTANK",
        )
        client = LLMClient(spec=spec, callsite_id="thinktank-contract")
        assert client._resolve_api_key() == "sk-live", (
            "the installed krepis resolves the router-edge credential from the "
            "environment only — the Think Tank's credential is never in the "
            "environment by design, so every tier call will abort the run"
        )

    def test_the_resolver_is_addressable_by_credential_name(self):
        """`resolve_group_spec` stamps a PER-CONSUMER name onto the spec, and
        the edge identifies a consumer BY its credential value. A resolver that
        only answers for the shared ``LITELLM_MASTER_KEY`` would authenticate
        this box as the director and make revocation all-or-nothing."""
        import inspect

        from krepis.router import resolve_router_credential

        assert "name" in inspect.signature(resolve_router_credential).parameters


# ── The served-model half of the same contract (config-I6543 / -I6652) ───────


class TestGroupAliasIsNeverReportedAsTheServedModel:
    """Under group addressing the serving model is unknown until the response
    returns, so ``spec.model`` is a synthetic alias ("low"/"med"/"high"/
    "ultra") that is never itself billable and never carries a price card.

    Through krepis 0.35.0 both litellm-transport paths — ``complete()``'s
    ``TRANSPORT_LITELLM`` branch and ``_structured_litellm()``, the latter
    being the Think Tank's live path for `themes`/`thesis`/`pillar` — fell
    back to ``model=served_model or self.spec.model`` when the response's
    ``model`` field came back empty. The alias then flowed into
    ``thinktank/client.py::_cost_for``, which correctly refused to price a
    fictional model and RAISED::

        ThinktankLLMError: tier=themes group=low served-model='low': the
        provider returned no cost and no price card could be resolved

    Live on 2026-08-04 (``thinktank/runs/2026-08-04/manifest_df20e6c66665``):
    run aborted, 0 theses written. The fail-loud design downstream worked
    exactly as intended; what fed it did not. krepis 0.36.0
    (``_resolve_group_served_model``, krepis-PR115) raises at the source with
    ``_hidden_params`` attached instead of returning the alias.

    Asserted as a library CAPABILITY, not a version string, mirroring
    ``TestRouterCredentialIsResolvableByKrepis`` above: a floor is satisfiable
    by a build that no longer behaves this way, and the failure it guards is a
    daily 14:30 UTC run, not a CI red.

    Deliberately NOT asserted: that a price card exists for "low". Adding one
    would paper over a resolution failure by pretending the alias is billable
    — flagged as the wrong fix in config-I6543 and in the alert-drain issue
    (-I6577) that proposed it, which was closed for that reason.
    """

    def test_a_group_alias_is_never_reported_as_the_served_model(self):
        import pytest as _pytest
        from krepis.llm import _resolve_group_served_model
        from krepis.llm_config import LLMConfigError, ModelSpec
        from krepis.router import ROUTER_EDGE_PROVIDER

        spec = ModelSpec(
            provider=ROUTER_EDGE_PROVIDER,
            model="low",
            base_url="https://router.example.invalid:8443",
            api_key_env="ROUTER_CONSUMER_THINKTANK",
        )

        class _Resp:
            def __init__(self, model):
                self.model = model
                self._hidden_params = {"model_id": "deployment-abc"}

        # A real served model passes straight through.
        assert _resolve_group_served_model(_Resp("deepseek-v4-flash"), spec=spec) == (
            "deepseek-v4-flash"
        )

        # An empty field and the alias echoed back are the two shapes seen
        # live; both must RAISE rather than resolve to the alias.
        for unusable in ("", None, "low"):
            with _pytest.raises(LLMConfigError) as exc:
                _resolve_group_served_model(_Resp(unusable), spec=spec)
            message = str(exc.value)
            assert "low" in message, (
                "the raise must name the group so the operator knows which "
                "tier produced it"
            )
            assert "_hidden_params" in message, (
                "the raise must carry _hidden_params — it is the only "
                "diagnostic that can root-cause WHY the model field was "
                "unusable, and capturing it is deliverable 3 of config-I6543"
            )
