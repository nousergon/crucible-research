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


def _chat_anthropic_calls(tree):
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "ChatAnthropic":
                calls.append(node)
    return calls


def test_every_chatanthropic_ctor_sets_request_timeout():
    offenders = []
    for py in _AGENTS_DIR.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for call in _chat_anthropic_calls(tree):
            kwargs = {kw.arg for kw in call.keywords if kw.arg}
            # accept either the canonical field or its langchain alias
            if not ({"default_request_timeout", "timeout"} & kwargs):
                offenders.append(f"{py.name}:{call.lineno}")
    assert not offenders, (
        "ChatAnthropic constructed without a per-call request timeout "
        f"(default_request_timeout / timeout): {offenders}"
    )
