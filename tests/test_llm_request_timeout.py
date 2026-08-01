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
    ctor names a model and bypasses the registry, the router, fallback chains
    and cost telemetry.

    SCOPE IS THE WHOLE REPO, NOT agents/. The first version of this guard
    scanned agents/ only. memory/semantic.py, memory/episodic.py and
    scripts/decision_review.py also construct chat models from
    config.PER_STOCK_MODEL, and the narrow scan reported a clean migration
    while three sites were still unmigrated. A guard whose scope is narrower
    than the invariant it claims to enforce is worse than no guard: it reads
    as coverage.

    Sanctioned exceptions are pinned by RELATIVE PATH, so a file named
    ic_cio.py appearing somewhere else does not inherit the exemption.
    """
    # langchain_utils.py IS the factory — it necessarily constructs both a
    # router-mode and a direct-mode client; the two tests above cover it.
    #
    # memory/ and scripts/ were tracked follow-ups in
    # alpha-engine-config-I4459 — migrated off direct construction in the
    # same issue arc. Removing an entry must make this test fail.
    SANCTIONED = {
        "agents/langchain_utils.py",
    }
    repo_root = _AGENTS_DIR.parent
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "tests"}

    offenders = []
    seen_sanctioned = set()
    for py in sorted(repo_root.rglob("*.py")):
        rel = py.relative_to(repo_root).as_posix()
        if any(part in skip_dirs for part in py.relative_to(repo_root).parts):
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        calls = list(_direct_chat_model_calls(tree))
        if rel in SANCTIONED:
            if calls:
                seen_sanctioned.add(rel)
            continue
        for name, call in calls:
            offenders.append(f"{rel}:{call.lineno} ({name})")

    assert not offenders, "chat models must be constructed via make_agent_llm, not directly:\n" + "\n".join(
        f"  - {o}" for o in offenders
    )

    # An exemption for a file that no longer constructs anything is stale and
    # silently widens the guard's blind spot. Same failure mode as the one
    # that let memory/ and scripts/ through.
    stale = SANCTIONED - seen_sanctioned
    assert not stale, f"SANCTIONED entries no longer construct a chat model — remove them: {sorted(stale)}"
