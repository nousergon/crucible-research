"""Tests for the per-call LLM request-timeout guard (config#687).

A single silently-stalled Anthropic call previously had no per-request
ceiling — only the outer 75-min 429 deadline — so one hung agent call could
consume the whole sector-team budget (the 2026-06-06 tail-latency blowout).
``SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS`` bounds each ``.invoke()`` request,
and every agent ChatAnthropic constructor must pass it as
``default_request_timeout``. These tests pin both the resolution/clamp logic
and that no constructor site regresses by dropping the kwarg.
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path

import agents.langchain_utils as L


def _reload_with_env(value):
    if value is None:
        os.environ.pop("SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS", None)
    else:
        os.environ["SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS"] = value
    return importlib.reload(L)


def test_default_is_300s():
    mod = _reload_with_env(None)
    assert mod.SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS == 300.0


def test_valid_override():
    mod = _reload_with_env("120")
    assert mod.SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS == 120.0


def test_clamp_too_low():
    mod = _reload_with_env("5")
    assert mod.SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS == 30.0


def test_clamp_too_high():
    mod = _reload_with_env("99999")
    assert mod.SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS == 1200.0


def test_unparseable_falls_back_to_default():
    mod = _reload_with_env("not-a-number")
    assert mod.SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS == 300.0


def teardown_module(module):
    # Restore the import-time default so other suites see the canonical value.
    _reload_with_env(None)


# ---------------------------------------------------------------------------
# Static guard: every ChatAnthropic(...) in agents/ passes
# default_request_timeout. Catches a future ctor added without the per-call
# bound (the regression #687 fixes).
# ---------------------------------------------------------------------------
_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def _direct_chat_model_calls(tree):
    """Every DIRECT chat-model construction in agents/.

    `make_agent_llm` is deliberately not listed: it sets the timeout itself,
    once, for every caller. The point of this guard is to catch a ctor that
    bypasses the factory — which is exactly how the per-call bound got lost
    the first time (#687).
    """
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in ("ChatAnthropic", "ChatOpenAI"):
                calls.append((name, node))
    return calls


def test_every_direct_chat_model_ctor_sets_request_timeout():
    offenders = []
    for py in _AGENTS_DIR.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for name, call in _direct_chat_model_calls(tree):
            kwargs = {kw.arg for kw in call.keywords if kw.arg}
            # accept either the canonical field or its langchain alias
            if not ({"default_request_timeout", "timeout"} & kwargs):
                offenders.append(f"{py.name}:{call.lineno} ({name})")
    assert not offenders, (
        f"chat model constructed without a per-call request timeout (default_request_timeout / timeout): {offenders}"
    )


def test_the_factory_itself_sets_a_request_timeout():
    """The guard above stops scanning a file the moment it uses the factory,
    so the factory is now the single place the bound can be lost. Without this
    the previous test would pass while guarding nothing."""
    src = (Path(__file__).resolve().parent.parent / "agents" / "langchain_utils.py").read_text()
    tree = ast.parse(src)
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "make_agent_llm")
    ctors = [
        n
        for n in ast.walk(factory)
        if isinstance(n, ast.Call) and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) == "ChatOpenAI"
    ]
    assert ctors, "make_agent_llm no longer constructs a chat model"
    for call in ctors:
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        assert {"timeout", "default_request_timeout"} & kwargs, (
            "make_agent_llm must set a per-call request timeout — every agent now inherits it from here"
        )


def test_agents_do_not_construct_chat_models_directly():
    """Capability-class addressing (model-portability-policy I1): a direct
    ctor in an agent names a model and bypasses the registry, the router,
    fallback chains and cost telemetry.

    ic_cio.py (judge) and canary_replay.py (canary) are the sanctioned
    exceptions — policy §8 requires a shadow/overlap period for a judge and a
    recorded re-baseline for a canary, so they migrate separately.
    """
    # langchain_utils.py IS the factory — it necessarily constructs one, and
    # test_the_factory_itself_sets_a_request_timeout covers it directly.
    SANCTIONED = {"langchain_utils.py", "ic_cio.py", "canary_replay.py"}
    offenders = []
    for py in _AGENTS_DIR.rglob("*.py"):
        if py.name in SANCTIONED:
            continue
        for name, call in _direct_chat_model_calls(ast.parse(py.read_text())):
            offenders.append(f"{py.name}:{call.lineno} ({name})")
    assert not offenders, f"agents must construct chat models via make_agent_llm, not directly: {offenders}"
