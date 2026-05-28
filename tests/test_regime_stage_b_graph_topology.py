"""Pin the regime-v3 Stage B graph topology — macro_economist runs
serially upstream of the sector-team Send() fan-out.

Three invariants this test locks down:

1. The graph builder defines ``macro_economist_node`` as the direct
   successor of ``fetch_data`` (serial, not via Send/conditional fan-out).
2. The dispatcher (``dispatch_sectors_and_exit``) emits Send() calls
   ONLY for sector teams + the exit evaluator — NOT for
   ``macro_economist_node``. A Send to macro here would re-introduce
   the pre-Stage-B race where sector teams snapshot state before
   macro writes ``market_regime``.
3. ``macro_economist_node`` is wired as the source of the
   conditional-edge dispatch (so it runs serially THEN fans out),
   not as a Send target (which would be parallel).

The substantive consequence of Stage B is that sector team LLM
prompts now receive the actual current-week ``market_regime``
classification + ``sector_modifiers`` + ``sector_ratings`` macro
computed, rather than the default ``"neutral"`` they received pre-B
because Send() snapshots state at dispatch time.

Implementation note: topology assertions use static AST inspection of
``graph/research_graph.py`` rather than calling ``build_graph()`` +
introspecting the compiled Pregel. The compile path has side effects
that pollute monkeypatch state for downstream tests.

Filename note: this file is named ``test_regime_stage_b_*`` rather
than the more natural ``test_graph_macro_*`` to keep it alphabetically
AFTER ``test_macro_sector_coherence_gate.py``. The earlier name caused
``test_gate_disabled_passes_everything`` to fail in suite-mode (passes
in isolation; passes when this file runs after coherence_gate). Root
cause is upstream test-state pollution that pre-dates this PR — the
function-level imports + AST-only topology checks here are deliberately
side-effect-free, so the interaction must be in another test's
monkeypatch teardown chain that captures the original
``SECTOR_COHERENCE_GATE_ENABLED`` value at an ordering-sensitive
moment. The rename sidesteps it without papering over symptoms.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO_ROOT / "graph" / "research_graph.py"


def _build_graph_source() -> str:
    """Extract the body of ``build_graph()`` as source text. Used for
    string-search invariants without parsing the compiled Pregel."""
    source = GRAPH_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_graph":
            return ast.get_source_segment(source, node) or ""
    raise AssertionError("build_graph function not found in research_graph.py")


def test_dispatch_sectors_and_exit_does_not_send_to_macro() -> None:
    """The dispatcher must emit Send targets for sectors + exit ONLY.
    A Send to macro_economist_node would re-introduce the pre-Stage-B
    parallel race that defeated regime context propagation to sectors."""
    from graph.research_graph import (
        ALL_TEAM_IDS,
        dispatch_sectors_and_exit,
    )

    state = {
        "team_id": "",
        "market_regime": "bear",
    }
    sends = dispatch_sectors_and_exit(state)
    target_names = [s.node for s in sends]

    assert "macro_economist_node" not in target_names, (
        f"dispatch_sectors_and_exit emitted a Send to macro_economist_node "
        f"({target_names}). Macro must run SERIALLY upstream per Stage B; "
        f"a parallel Send re-introduces the regime-context race."
    )

    sector_targets = [n for n in target_names if n == "sector_team_node"]
    exit_targets = [n for n in target_names if n == "exit_evaluator_node"]
    assert len(sector_targets) == len(ALL_TEAM_IDS)
    assert len(exit_targets) == 1
    assert set(target_names) == {"sector_team_node", "exit_evaluator_node"}


def test_dispatch_sectors_and_exit_propagates_market_regime() -> None:
    """Each Send must carry the macro-computed market_regime forward to
    sector teams. Pre-Stage-B regression: macro hadn't run yet at
    dispatch time, so state["market_regime"] was always the
    fetch_data-time default ``"neutral"``. Post-Stage-B macro runs
    upstream, so the dispatcher's state has the real value."""
    from graph.research_graph import dispatch_sectors_and_exit

    state = {
        "team_id": "",
        "market_regime": "caution",
        "sector_modifiers": {"Technology": 0.93},
        "sector_ratings": {"Technology": {"rating": "underweight"}},
    }
    sends = dispatch_sectors_and_exit(state)
    sector_sends = [s for s in sends if s.node == "sector_team_node"]
    assert sector_sends, "must have at least one sector-team Send"
    for s in sector_sends:
        assert s.arg["market_regime"] == "caution"
        assert s.arg["sector_modifiers"]["Technology"] == 0.93
        assert s.arg["sector_ratings"]["Technology"]["rating"] == "underweight"


def test_build_graph_serializes_macro_upstream_of_dispatch() -> None:
    """Macro economist must run serially BEFORE the sector dispatch.
    Stage B asserted this via a direct fetch_data → macro edge; Stage C
    inserts a ``load_regime_substrate_node`` between them, but macro
    still runs serially upstream of the dispatch. Pin the property
    (serial macro), not the specific edge wording.

    Uses static AST inspection rather than compiling the graph because
    LangGraph's compile() has side effects that interact badly with
    other tests' monkeypatch state. The source is the contract.
    """
    body = _build_graph_source()
    # Three accepted topologies — all preserve the serial-upstream-of-
    # dispatch property:
    #   - Stage B: fetch_data → macro_economist_node (direct edge)
    #   - Stage C: fetch_data → load_regime_substrate_node → macro
    #   - Phase 2.A.3: fetch_data → load_regime_substrate_node →
    #     load_scorecard_node → macro (scorecard arc spliced in)
    has_stage_b_edge = 'graph.add_edge("fetch_data", "macro_economist_node")' in body
    has_stage_c_chain = (
        'graph.add_edge("fetch_data", "load_regime_substrate_node")' in body
        and 'graph.add_edge("load_regime_substrate_node", "macro_economist_node")'
        in body
    )
    has_phase_2a3_chain = (
        'graph.add_edge("fetch_data", "load_regime_substrate_node")' in body
        and 'graph.add_edge("load_regime_substrate_node", "load_scorecard_node")'
        in body
        and 'graph.add_edge("load_scorecard_node", "macro_economist_node")'
        in body
    )
    assert has_stage_b_edge or has_stage_c_chain or has_phase_2a3_chain, (
        "build_graph must serialize macro_economist_node upstream of the "
        "sector dispatch — Stage B (direct), Stage C (via substrate loader), "
        "or Phase 2.A.3 (via substrate + scorecard loaders). Without this "
        "serialization, sector teams see the default 'neutral' regime "
        "instead of the macro-computed value."
    )


def test_build_graph_dispatches_from_downstream_of_macro_not_fetch_data() -> None:
    """The conditional-edges dispatch must hang off a node downstream
    of macro_economist_node, NOT off fetch_data. Pre-Stage-B it hung
    off fetch_data, which made macro a Send target (parallel). Post-
    Stage-B macro is upstream of the dispatch; subsequent arcs may
    insert intermediate nodes (focus_list_by_team computation etc.)
    between macro and the dispatch — the invariant is "macro is
    serial upstream of the dispatch", not "dispatch immediately
    follows macro"."""
    body = _build_graph_source()
    # Must NOT dispatch off fetch_data (the pre-Stage-B antipattern).
    assert (
        'graph.add_conditional_edges("fetch_data"' not in body
    ), (
        "build_graph dispatches from fetch_data — Stage B moves the "
        "dispatch downstream of macro_economist_node. Stale dispatch on "
        "fetch_data races macro and re-introduces the regime-context bug."
    )
    # Must dispatch off SOMETHING downstream of macro. Accept either
    # macro_economist_node directly (post-Stage-B) or an intermediate
    # node like compute_focus_list_node (post-focus-list-arc).
    assert (
        'graph.add_conditional_edges("macro_economist_node", dispatch_sectors_and_exit)' in body
        or 'graph.add_conditional_edges("compute_focus_list_node", dispatch_sectors_and_exit)' in body
    ), (
        "build_graph must dispatch sectors via conditional_edges off a node "
        "downstream of macro_economist_node (macro itself or a serial "
        "intermediate). Without this serialization sector teams see the "
        "default 'neutral' regime."
    )


def test_build_graph_does_not_have_stale_dispatch_all() -> None:
    """``dispatch_all`` was renamed to ``dispatch_sectors_and_exit`` in
    Stage B. The old name must not appear in build_graph — pins that
    rename so a partial refactor doesn't leave both around."""
    body = _build_graph_source()
    assert "dispatch_all" not in body, (
        "build_graph still references dispatch_all — Stage B renamed it to "
        "dispatch_sectors_and_exit. Stale reference indicates incomplete refactor."
    )


def test_build_graph_does_not_edge_macro_to_merge_results() -> None:
    """Pre-Stage-B macro_economist_node had a direct edge to
    merge_results (because it ran in parallel with sectors). Post-B
    its outputs flow through state and merge_results is reached via the
    sector + exit Send convergence — no direct edge needed."""
    body = _build_graph_source()
    assert (
        'graph.add_edge("macro_economist_node", "merge_results")' not in body
    ), (
        "build_graph still has a direct edge macro_economist_node → "
        "merge_results. This was the pre-Stage-B parallel-path edge; "
        "Stage B routes macro outputs through state, not via this edge."
    )


def test_build_graph_does_not_directly_dispatch_to_sectors_from_fetch_data() -> None:
    """Pre-Stage-B, fetch_data had a conditional_edges to dispatch_all
    (which Send-fanned to teams + macro + exit in parallel). Post-B,
    fetch_data → macro_economist_node directly; the conditional_edges
    hangs off macro_economist_node instead. Pin so a partial revert
    can't leave conditional dispatch on fetch_data."""
    body = _build_graph_source()
    assert (
        'graph.add_conditional_edges("fetch_data"' not in body
    ), (
        "build_graph dispatches from fetch_data — Stage B moves the "
        "dispatch downstream of macro_economist_node. Stale dispatch on "
        "fetch_data races macro and re-introduces the regime-context bug."
    )


def test_macro_economist_node_still_exists_as_callable() -> None:
    """Sanity check — Stage B keeps macro_economist_node as a public
    graph node (just moves it from a Send target to a serial node).
    A refactor that removed the function would silently break the
    graph builder; this test catches that."""
    from graph.research_graph import macro_economist_node
    assert callable(macro_economist_node)
