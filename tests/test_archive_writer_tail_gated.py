"""Static (AST) guard: every UNBOUNDED best-effort tail task in
``archive_writer`` must be invoked ONLY behind the single secondary-work
budget chokepoint (``_run_secondary_within_budget``), never called eagerly
in the node body.

Why this test exists (config#1638) — the recurrence signal
-----------------------------------------------------------
``archive_writer`` runs AFTER signals.json (the primary deliverable) has
already landed, then does a tail of best-effort observability writes and,
crucially, the "must not miss" scanner_eval grade logging + ``upload_db``
finalize. Two SIGKILL incidents came from an *unbounded* tail task (no
internal runtime bound) running past the 900s Lambda ceiling and taking the
whole run down with a spurious ``States.Timeout`` — starving the grade
history that follows:

  * 2026-06-06 — semantic-memory extraction (an unbounded LLM call). Fixed by
    adding an inline deadline gate for that one call.
  * 2026-07-03 — the attractiveness trajectory (a full-universe ArcticDB read
    + digest email) was added to the tail LATER and silently escaped that
    gate. Fixed in crucible-research#366 by introducing
    ``_run_secondary_within_budget`` and routing the trajectory through it.

Two incidents of the SAME class is the signal to lift the invariant to an
*enforced* chokepoint rather than patch the next site. A code review can miss
a newly-added un-gated tail call; a failing test cannot. This guard parses
``archive_writer`` and asserts that every call to a member of the maintained
``_UNBOUNDED_TAIL_CALLS`` allowlist appears ONLY as the deferred ``fn``
(lambda body) passed to ``_run_secondary_within_budget`` — so a future
regression that adds/moves such a call outside the chokepoint fails here.

Maintaining the allowlist
-------------------------
When a NEW unbounded best-effort tail task (another LLM call, another
full-universe ArcticDB read, a network digest, …) is added to
``archive_writer``, add its call name to ``_UNBOUNDED_TAIL_CALLS`` AND route
it through ``_run_secondary_within_budget``. The allowlist is deliberately
small and explicit so the review question "is this new tail call bounded?"
is forced at the point the entry is added.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# The graph node under guard and the single chokepoint every unbounded tail
# task must route through.
_ARCHIVE_WRITER = "archive_writer"
_CHOKEPOINT = "_run_secondary_within_budget"

# Maintained allowlist of UNBOUNDED best-effort tail calls in archive_writer.
# Every one of these MUST be invoked only behind _CHOKEPOINT. Extend this set
# (and route the new call through the chokepoint) whenever a new unbounded
# tail task is added — see the module docstring.
_UNBOUNDED_TAIL_CALLS = frozenset(
    {"extract_semantic_memories", "compute_and_write_trajectory"}
)

_RESEARCH_GRAPH = Path(__file__).resolve().parent.parent / "graph" / "research_graph.py"


def _module_ast() -> ast.Module:
    return ast.parse(_RESEARCH_GRAPH.read_text(encoding="utf-8"), filename=str(_RESEARCH_GRAPH))


def _archive_writer_node(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == _ARCHIVE_WRITER:
            return node  # type: ignore[return-value]
    raise AssertionError(f"{_ARCHIVE_WRITER}() not found in {_RESEARCH_GRAPH.name}")


def _callee_name(call: ast.Call) -> str | None:
    """The simple name of a call's callee (``foo`` or ``obj.foo`` → ``foo``)."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parent_map(root: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(root):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _is_gated(call: ast.Call, root: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """True iff ``call`` is lexically nested inside a Call to the chokepoint
    (i.e. it appears in the deferred ``fn``/lambda argument), climbing parents
    up to — but not past — ``root`` (the archive_writer node)."""
    node: ast.AST | None = parents.get(id(call))
    while node is not None and node is not root:
        if isinstance(node, ast.Call) and _callee_name(node) == _CHOKEPOINT:
            return True
        node = parents.get(id(node))
    return False


def _tail_calls_in_archive_writer() -> list[ast.Call]:
    node = _archive_writer_node(_module_ast())
    return [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and _callee_name(c) in _UNBOUNDED_TAIL_CALLS
    ]


def test_every_unbounded_tail_call_is_budget_gated():
    """No allowlisted unbounded call may be invoked eagerly in archive_writer;
    each must sit behind _run_secondary_within_budget."""
    tree = _module_ast()
    node = _archive_writer_node(tree)
    parents = _parent_map(node)
    ungated = [
        _callee_name(c)
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
        and _callee_name(c) in _UNBOUNDED_TAIL_CALLS
        and not _is_gated(c, node, parents)
    ]
    assert not ungated, (
        "Un-gated UNBOUNDED tail call(s) in archive_writer: "
        f"{sorted(ungated)}. Every call to {sorted(_UNBOUNDED_TAIL_CALLS)} MUST "
        f"be routed through {_CHOKEPOINT}(...) so a tail-latency-slow run SKIPS "
        "the work instead of being SIGKILL'd at the 900s Lambda ceiling "
        "(config#1638). Wrap it as the fn/lambda argument to that chokepoint."
    )


@pytest.mark.parametrize("call_name", sorted(_UNBOUNDED_TAIL_CALLS))
def test_allowlist_is_not_stale(call_name):
    """Each allowlisted call must still appear in archive_writer — if a tail
    task is removed/renamed, prune the allowlist so this guard stays honest
    (and can't pass vacuously)."""
    present = {_callee_name(c) for c in _tail_calls_in_archive_writer()}
    assert call_name in present, (
        f"{call_name!r} is in _UNBOUNDED_TAIL_CALLS but no longer called in "
        f"{_ARCHIVE_WRITER}(). Remove it from the allowlist (or restore the "
        "call) so this guard cannot pass vacuously."
    )


def test_chokepoint_symbol_exists():
    """The guard is meaningless if the chokepoint helper is renamed away —
    pin the symbol the whole tail routes through."""
    from graph import research_graph

    assert hasattr(research_graph, _CHOKEPOINT), (
        f"{_CHOKEPOINT} missing from research_graph — the archive_writer tail "
        "chokepoint was renamed/removed; update this guard and the routing."
    )
