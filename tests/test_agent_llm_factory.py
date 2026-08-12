"""Guards on ``agents.langchain_utils.make_agent_llm`` — the ONE chat-model
constructor every research agent goes through.

The invariant is `model-router-policy` §2 layer 5: a consumer "must not hold a
routing table, a model slug, or an endpoint of their own". Until
alpha-engine-config-I7005 this factory held all three — a
``DIRECT_MODEL_FOR_CLASS`` class->slug map in ``config.py``, the two model-id
literals it was built from, and a configured ``base_url`` — and its DEFAULT
branch (router unset) built a ``ChatAnthropic`` from that map. That is a
fall-through to an unscanned direct provider where §5 step 3 requires a
fail-closed.

The tests below replace the previous file's guards, which asserted properties
of names that no longer exist (``PER_STOCK_MODEL`` is a real model id;
``DIRECT_MODEL_FOR_CLASS`` values are not class names; ``ROUTER_BASE_URL`` is
not localhost). Those defects are now unrepresentable rather than guarded:
there is no consumer-side map for a class name to leak into, and
``test_config_holds_no_routing_facts`` fails the moment one is reintroduced.

What is guarded instead:

1. ``config`` holds no endpoint, slug, credential name or class->model table.
2. The factory resolves through ``krepis.router.resolve_group_spec`` and binds
   the client to what the resolver returned — model, endpoint and credential
   NAME all come from the registry.
3. The execution context is DECLARED (``KREPIS_EXEC_CONTEXT``) and passed
   through verbatim, ``None`` when unset so krepis applies its own default.
   Never sniffed from a hostname or a metadata service (R29).
4. A resolution that reached a direct provider endpoint from a Lambda or the
   weekly Research spot RAISES rather than making a paid, DLP-unscanned call
   (R20/R26; the alpha-engine-config-I6183 shape).
5. A resolver failure propagates — the consumer fails closed rather than
   reaching for whatever happens to be callable.
"""

from __future__ import annotations

import pytest

import config as config_mod

pytest.importorskip("langchain_openai")
pytest.importorskip("krepis.router")


_REMOVED_ROUTING_FACTS = (
    "ROUTER_BASE_URL",
    "ROUTER_KEY_SECRET",
    "DIRECT_MODEL_FOR_CLASS",
    "PER_STOCK_MODEL",
    "STRATEGIC_MODEL",
)


class _FakeSpec:
    """Minimal stand-in for ``krepis.llm_config.ModelSpec``."""

    def __init__(self, *, model, base_url, api_key_env, max_tokens, reasoning=None):
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.provider = "litellm_edge"


def _proxy_route(**over):
    route = {
        "route": "litellm_proxy",
        "registry_id": "litellm:group:low",
        "primary_registry_id": "claude-haiku",
        "deployment_id": "low-claude-haiku",
        "provider": "litellm",
        "api_base_url": "https://router.invalid/v1",
        "exec_context": "lambda",
        "skipped_entries": [],
    }
    route.update(over)
    return route


def _install_resolver(monkeypatch, *, spec, route, recorder=None):
    """Patch ``krepis.router.resolve_group_spec`` (the factory imports it lazily
    from the module, so patching the module attribute is what takes effect)."""
    import krepis.router as krepis_router

    def _fake(group, *, exec_context=None, wire=None, max_tokens=None, **kw):
        if recorder is not None:
            recorder.update(group=group, exec_context=exec_context, wire=wire, max_tokens=max_tokens)
        return spec, route

    monkeypatch.setattr(krepis_router, "resolve_group_spec", _fake, raising=True)


@pytest.fixture()
def _no_cost_callback(monkeypatch):
    """The factory attaches a cost-telemetry callback by default; the tests pass
    ``callbacks=[]`` where they care, and this keeps the default path from
    touching S3 when they do not."""
    import graph.llm_cost_tracker as tracker

    monkeypatch.setattr(tracker, "get_cost_telemetry_callback", lambda *a, **k: None)
    return tracker


# ── 1. no routing facts at layer 5 ───────────────────────────────────────────


def test_config_holds_no_routing_facts():
    """A model slug, an endpoint, a credential name or a class->model table in
    consumer config is the layer-5 defect this change removed. Re-adding one is
    not a style regression — it is a second copy of a registry fact, and §2's
    test says delete the lower copy."""
    present = [name for name in _REMOVED_ROUTING_FACTS if hasattr(config_mod, name)]
    assert not present, (
        f"config re-introduced routing facts at layer 5: {present}. The registry "
        f"owns model, endpoint and credential; this config owns the capability "
        f"class and nothing else (model-router-policy §2)."
    )


def test_capability_classes_survive():
    """The class IS the consumer's legitimate routing statement — removing the
    tables must not have taken it with them."""
    assert config_mod.PER_STOCK_CLASS in {"low", "med", "high", "ultra"}
    assert config_mod.STRATEGIC_CLASS in {"low", "med", "high", "ultra"}


# ── 2. the client is built from what the resolver returned ───────────────────


def test_client_is_bound_to_the_resolved_spec(monkeypatch, _no_cost_callback):
    from agents import langchain_utils

    seen: dict = {}
    spec = _FakeSpec(
        model="low-claude-haiku",
        base_url="https://router.invalid/v1",
        api_key_env=None,  # placeholder auth — the proxy holds the real key
        max_tokens=777,
    )
    _install_resolver(monkeypatch, spec=spec, route=_proxy_route(), recorder=seen)
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")

    llm = langchain_utils.make_agent_llm(model_class="low", max_tokens=777, callbacks=[])

    assert type(llm).__name__ == "ChatOpenAI"
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    assert model == "low-claude-haiku", "the client must address what the registry resolved"
    assert seen["group"] == "low", "the capability class is what goes to the resolver"
    assert seen["wire"] == "openai", "these call sites speak the OpenAI wire format"
    assert seen["max_tokens"] == 777, (
        "the budget must go THROUGH the resolver so the spec, the log and the request all carry one number"
    )
    assert getattr(llm, "max_tokens", None) == 777


def test_credential_is_resolved_by_the_name_the_registry_returned(monkeypatch, _no_cost_callback):
    """The consumer never picks a credential; it resolves the NAME the resolver
    handed back. A locally-derived credential name is a routing fact."""
    from agents import langchain_utils

    spec = _FakeSpec(
        model="low-claude-haiku",
        base_url="https://router.invalid/v1",
        api_key_env="ROUTER_CONSUMER_RESEARCH",
        max_tokens=64,
    )
    _install_resolver(monkeypatch, spec=spec, route=_proxy_route())

    asked: list[str] = []

    import nousergon_lib.secrets as secrets_mod

    def _fake_get_secret(name, required=False, default=""):
        asked.append(name)
        return "sekret"

    monkeypatch.setattr(secrets_mod, "get_secret", _fake_get_secret, raising=True)
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")

    langchain_utils.make_agent_llm(model_class="low", max_tokens=64, callbacks=[])

    assert asked == ["ROUTER_CONSUMER_RESEARCH"], (
        f"credential must be fetched by the resolver-supplied name, asked for {asked}"
    )


# ── 3. execution context is declared, never inferred ─────────────────────────


def test_exec_context_is_passed_through_verbatim(monkeypatch, _no_cost_callback):
    from agents import langchain_utils

    seen: dict = {}
    spec = _FakeSpec(model="m", base_url="https://router.invalid/v1", api_key_env=None, max_tokens=8)
    _install_resolver(monkeypatch, spec=spec, route=_proxy_route(exec_context="ec2"), recorder=seen)
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ec2")

    langchain_utils.make_agent_llm(model_class="low", max_tokens=8, callbacks=[])
    assert seen["exec_context"] == "ec2"


def test_unset_exec_context_defers_to_krepis(monkeypatch, _no_cost_callback):
    """None, not a local default. A default held here would be a second copy of
    krepis' documented resolution order, and a *guessed* one would be the R29
    inference that makes a mis-resolution look like a health failure."""
    from agents import langchain_utils

    seen: dict = {}
    spec = _FakeSpec(model="m", base_url="https://router.invalid/v1", api_key_env=None, max_tokens=8)
    _install_resolver(monkeypatch, spec=spec, route=_proxy_route(), recorder=seen)
    monkeypatch.delenv("KREPIS_EXEC_CONTEXT", raising=False)

    langchain_utils.make_agent_llm(model_class="low", max_tokens=8, callbacks=[])
    assert seen["exec_context"] is None


def test_exec_context_is_not_sniffed(monkeypatch, _no_cost_callback):
    """A Lambda-shaped environment must NOT make the factory declare 'lambda'
    on its own — the context is a value the deployment supplies (R29)."""
    from agents import langchain_utils

    seen: dict = {}
    spec = _FakeSpec(model="m", base_url="https://router.invalid/v1", api_key_env=None, max_tokens=8)
    _install_resolver(monkeypatch, spec=spec, route=_proxy_route(), recorder=seen)
    monkeypatch.delenv("KREPIS_EXEC_CONTEXT", raising=False)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "alpha-engine-research-runner")

    langchain_utils.make_agent_llm(model_class="low", max_tokens=8, callbacks=[])
    assert seen["exec_context"] is None


# ── 4. a non-proxy route off the laptop is refused ───────────────────────────


@pytest.mark.parametrize("ctx", ["lambda", "ec2", None])
def test_direct_provider_route_is_refused_off_the_laptop(monkeypatch, _no_cost_callback, ctx):
    """The alpha-engine-config-I6183 shape: a resolution that reached
    openrouter.ai from a context with no local egress proxy. Raising loses the
    run; making the call bills real money on an unscanned egress and reports
    success (R20 / §5 step 3)."""
    from agents import langchain_utils

    spec = _FakeSpec(
        model="glm-5.2", base_url="https://openrouter.invalid/api/v1", api_key_env="OPENROUTER_API_KEY", max_tokens=8
    )
    route = _proxy_route(
        route="openrouter",
        provider="openrouter",
        registry_id="glm-5.2",
        deployment_id="glm-5.2",
        api_base_url="https://openrouter.invalid/api/v1",
        exec_context=ctx or "laptop",
    )
    _install_resolver(monkeypatch, spec=spec, route=route)
    if ctx is None:
        monkeypatch.delenv("KREPIS_EXEC_CONTEXT", raising=False)
    else:
        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", ctx)

    with pytest.raises(RuntimeError, match="litellm_proxy"):
        langchain_utils.make_agent_llm(model_class="high", max_tokens=8, callbacks=[])


def test_direct_route_is_permitted_on_the_laptop(monkeypatch, _no_cost_callback):
    """R27d: the egress proxy is on loopback there, so a direct route is
    legitimate. The guard refuses a non-conformant resolution, it does not make
    a routing decision — mirrors the Director's."""
    from agents import langchain_utils

    spec = _FakeSpec(
        model="deepseek-v4-flash",
        base_url="http://127.0.0.1:8990",
        api_key_env=None,
        max_tokens=8,
    )
    route = _proxy_route(
        route="egress_proxy",
        provider="deepseek",
        registry_id="deepseek-v4-flash",
        primary_registry_id="deepseek-v4-flash",
        deployment_id="deepseek-v4-flash",
        api_base_url="http://127.0.0.1:8990",
        exec_context="laptop",
    )
    _install_resolver(monkeypatch, spec=spec, route=route)
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "laptop")

    llm = langchain_utils.make_agent_llm(model_class="low", max_tokens=8, callbacks=[])
    assert type(llm).__name__ == "ChatOpenAI"


# ── 5. resolution failure fails closed ───────────────────────────────────────


def test_resolution_failure_propagates(monkeypatch, _no_cost_callback):
    """No fall-through to a default endpoint or an ambient key (R20). krepis'
    own ValueError is the fail-closed, and swallowing it here would rebuild the
    direct-mode branch this change deleted."""
    import krepis.router as krepis_router

    from agents import langchain_utils

    def _boom(*a, **k):
        raise ValueError("no entry in group 'low' is reachable from 'lambda'")

    monkeypatch.setattr(krepis_router, "resolve_group_spec", _boom, raising=True)
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")

    with pytest.raises(ValueError, match="reachable"):
        langchain_utils.make_agent_llm(model_class="low", max_tokens=8, callbacks=[])


# ── degradation telemetry emits on the healthy path too ──────────────────────


def test_route_telemetry_emits_on_the_healthy_path(monkeypatch, _no_cost_callback, caplog):
    """principles.md §2.7: a signal that only appears when something is wrong is
    indistinguishable from a dead emitter."""
    from agents import langchain_utils

    spec = _FakeSpec(model="m", base_url="https://router.invalid/v1", api_key_env=None, max_tokens=8)
    _install_resolver(monkeypatch, spec=spec, route=_proxy_route())
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")

    with caplog.at_level("INFO", logger="agents.langchain_utils"):
        langchain_utils.make_agent_llm(model_class="low", max_tokens=8, callbacks=[])

    assert any("degraded=0" in r.getMessage() for r in caplog.records), (
        "the healthy route must emit an explicit healthy value, not silence"
    )


def test_route_telemetry_warns_when_entries_were_skipped(monkeypatch, _no_cost_callback, caplog):
    """R12: serving past the primary is an ALERT, not a log line."""
    from agents import langchain_utils

    spec = _FakeSpec(model="m", base_url="https://router.invalid/v1", api_key_env=None, max_tokens=8)
    route = _proxy_route(skipped_entries=[{"registry_id": "kimi-k3-direct", "reason": "Egress proxy not reachable"}])
    _install_resolver(monkeypatch, spec=spec, route=route)
    monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")

    with caplog.at_level("WARNING", logger="agents.langchain_utils"):
        langchain_utils.make_agent_llm(model_class="low", max_tokens=8, callbacks=[])

    assert any("DEGRADED" in r.getMessage() for r in caplog.records)
