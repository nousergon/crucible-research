"""Regression guard for the two defects the canary replay caught on
crucible-research-PR507 (alpha-engine-config-I4459).

Both were introduced by the same mistake: making the new router path the
DEFAULT rather than genuinely additive.

1. ``PER_STOCK_MODEL``/``STRATEGIC_MODEL`` were aliased onto the capability
   CLASS names. Every call site still talking to Anthropic by hand
   (canary_replay, ic_cio, memory/, scripts/) then sent the literal string
   ``"low"`` as a model name:

       NotFoundError: 404 {'type': 'not_found_error', 'message': 'model: low'}

   A back-compat shim that keeps a name alive while silently changing what its
   VALUE MEANS is worse than deleting the name — deletion fails loudly at
   import, this failed against the provider at runtime.

2. ``router_base_url`` defaulted to ``http://127.0.0.1:8980/v1``. Only the
   laptop and the dashboard box run a local router; Lambda and the EC2 canary
   box cannot reach localhost at all, so every agent call there became

       APIConnectionError: Connection error.

   An unset router now means DIRECT mode, so no deployment changes behaviour
   until it opts in.

These are cheap to assert and were expensive to find. The canary that caught
them runs per-PR against live infrastructure; these run in milliseconds.
"""

from __future__ import annotations

import importlib

import pytest

import config as config_mod

_CLASS_NAMES = {"low", "med", "high", "ultra"}


def test_legacy_model_aliases_are_real_model_ids_not_class_names():
    """The exact defect: a class name reaching a ``model=`` kwarg."""
    for attr in ("PER_STOCK_MODEL", "STRATEGIC_MODEL"):
        value = getattr(config_mod, attr)
        assert value not in _CLASS_NAMES, (
            f"config.{attr} is the capability class {value!r}, not a model id. "
            f"Every un-migrated ChatAnthropic(model=config.{attr}) site will "
            f"send that string to Anthropic and get a 404."
        )
        assert "-" in value, f"config.{attr}={value!r} does not look like a concrete model id"


def test_direct_model_map_contains_no_class_named_values():
    """The map resolves class -> model. A class on the VALUE side would mean
    a class name is about to be used as a model name."""
    for cls, model in config_mod.DIRECT_MODEL_FOR_CLASS.items():
        assert cls in _CLASS_NAMES, f"{cls!r} is not a capability class"
        assert model not in _CLASS_NAMES, f"DIRECT_MODEL_FOR_CLASS[{cls!r}] is {model!r}, a class name, not a model id"


def test_router_base_url_is_not_defaulted_to_localhost():
    """Localhost is unreachable from Lambda and the EC2 canary box, which is
    most of where this code actually runs. Defaulting to it turns every call
    into a connection error at runtime instead of a config error at load."""
    assert "127.0.0.1" not in config_mod.ROUTER_BASE_URL, (
        "router_base_url must not default to a local router — it is wrong for "
        "every deployment except the laptop and the dashboard box."
    )


def _reload_config(monkeypatch, **llm_overrides):
    cfg = importlib.reload(config_mod)
    for key, value in llm_overrides.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    return cfg


def test_direct_mode_constructs_an_anthropic_client_with_a_concrete_model(monkeypatch):
    """Unset router => pre-migration behaviour, byte-for-byte."""
    pytest.importorskip("langchain_anthropic")
    from agents import langchain_utils

    monkeypatch.setattr(config_mod, "ROUTER_BASE_URL", "", raising=False)
    llm = langchain_utils.make_agent_llm(model_class=config_mod.PER_STOCK_CLASS, max_tokens=64, api_key="test-key")

    assert type(llm).__name__ == "ChatAnthropic", (
        f"unset router_base_url must yield a direct Anthropic client, got {type(llm).__name__}"
    )
    model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    assert model not in _CLASS_NAMES, f"class name {model!r} sent as a model id"
    assert model == config_mod.PER_STOCK_MODEL


def test_router_mode_puts_the_class_on_the_wire(monkeypatch):
    """Configured router => the class name IS the model name; the registry
    resolves it server-side."""
    pytest.importorskip("langchain_openai")
    from agents import langchain_utils

    monkeypatch.setattr(config_mod, "ROUTER_BASE_URL", "http://router.invalid/v1", raising=False)
    llm = langchain_utils.make_agent_llm(model_class="med", max_tokens=64, api_key="test-key")

    assert type(llm).__name__ == "ChatOpenAI", (
        f"configured router_base_url must yield an OpenAI-compatible client, got {type(llm).__name__}"
    )
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    assert model == "med"


def test_direct_mode_fails_loud_on_a_class_with_no_model(monkeypatch):
    """Silently falling through would send the class name to Anthropic — the
    original bug. Raise instead."""
    pytest.importorskip("langchain_anthropic")
    from agents import langchain_utils

    monkeypatch.setattr(config_mod, "ROUTER_BASE_URL", "", raising=False)
    monkeypatch.setattr(config_mod, "DIRECT_MODEL_FOR_CLASS", {"low": "claude-haiku-4-5"}, raising=False)

    with pytest.raises(ValueError, match="no direct-mode model"):
        langchain_utils.make_agent_llm(model_class="ultra", max_tokens=64, api_key="test-key")
