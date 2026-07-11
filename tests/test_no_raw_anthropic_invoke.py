"""CI guard: no Anthropic-backed structured ``.invoke()`` may bypass the single
send-time pairing chokepoint ``invoke_anthropic_safe`` (config#2255).

The whole point of lifting the tool_use/tool_result pairing invariant to one
chokepoint (config#2255) is that a NAIVE new call site can't silently
reintroduce the "``tool_use`` ids were found without ``tool_result`` blocks"
400 (config#1065 / config#2245) by assembling messages and calling
``structured_llm.invoke(...)`` directly. This AST walk enforces that
structurally: any raw ``*_llm`` / ``*_structured`` handle ``.invoke(...)`` (or a
chained ``.with_structured_output(...).invoke(...)``) in the production packages
is a violation — it must route through ``invoke_anthropic_safe`` (or, for the
ReAct loop, ``invoke_react_with_recovery`` whose ``pre_model_hook`` applies the
same ``repair_tool_use_pairing`` primitive).

Sanctioned exceptions:
  * ``agents/langchain_utils.py`` — the module that DEFINES the chokepoint and
    its wrappers (the one place the raw ``handle.invoke`` legitimately lives).
  * ``agent.invoke({...})`` / ``graph.invoke(state)`` — ReAct/graph entrypoints
    whose input is a fresh state dict, not a hand-assembled message history
    (in-loop pairing is handled by the ReAct ``pre_model_hook``). These do not
    match the ``*_llm`` / ``*_structured`` / ``with_structured_output`` shapes
    below, so they are ignored by construction.
  (``evals/judge.py`` was the last raw structured send; crucible-research#407 /
  config#2237 routed it through ``invoke_structured_with_validation_retry`` —
  itself routed through ``invoke_anthropic_safe`` — so it is now covered
  transitively and needs no allowlist entry.)
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Production packages that assemble Anthropic sends. (``tests/`` is excluded.)
SCAN_DIRS = ("agents", "evals", "producers", "graph", "thinktank")

# Files allowed to contain a raw structured-handle ``.invoke`` (see module doc).
ALLOWLIST = {
    "agents/langchain_utils.py",
}


def _is_llm_handle_name(node: ast.AST) -> bool:
    """True if ``node`` is a Name that looks like a bound LLM/structured handle."""
    if not isinstance(node, ast.Name):
        return False
    n = node.id
    return n == "structured_llm" or n.endswith("_llm") or n.endswith("_structured")


def _is_with_structured_output_chain(node: ast.AST) -> bool:
    """True if ``node`` is a ``X.with_structured_output(...)`` call (a chained
    ``.with_structured_output(...).invoke(...)`` receiver)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "with_structured_output"
    )


def _raw_invoke_violations(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "invoke"):
            continue
        recv = func.value
        if _is_llm_handle_name(recv) or _is_with_structured_output_chain(recv):
            lines.append(getattr(node, "lineno", -1))
    return lines


def _iter_production_py():
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if "/tests/" in f"/{rel}" or path.name.startswith("test_"):
                continue
            yield rel, path


def test_no_raw_structured_invoke_bypasses_chokepoint():
    offenders: list[str] = []
    for rel, path in _iter_production_py():
        if rel in ALLOWLIST:
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        for lineno in _raw_invoke_violations(tree):
            offenders.append(f"{rel}:{lineno}")

    assert not offenders, (
        "Raw Anthropic structured ``.invoke()`` call(s) bypass the "
        "``invoke_anthropic_safe`` send-time pairing chokepoint (config#2255) — "
        "route them through ``invoke_anthropic_safe`` (or "
        "``invoke_react_with_recovery`` for the ReAct loop), or add a justified "
        "entry to ALLOWLIST:\n  " + "\n  ".join(offenders)
    )


def test_allowlisted_files_actually_have_a_raw_invoke():
    """Keep the allowlist honest: an entry that no longer has a raw invoke is
    dead and should be removed (e.g. once crucible-research#407 migrates the
    judge). ``langchain_utils.py`` is exempt — it defines the chokepoint."""
    for rel in ALLOWLIST:
        if rel == "agents/langchain_utils.py":
            continue
        path = ROOT / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        assert _raw_invoke_violations(tree), (
            f"ALLOWLIST entry {rel!r} no longer contains a raw structured "
            "``.invoke()`` — remove it from the allowlist (the bypass it "
            "excused is gone)."
        )
