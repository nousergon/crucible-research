"""Tests for compute_factor_profiles_node — the un-orphan wiring of
``scoring.factor_scoring.compute_and_write_factor_profiles`` into the
research graph.

Covers:
  (a) the node calls compute_and_write_factor_profiles with the state's
      run_date + sector_map and on success returns the observability delta;
  (b) on producer exception (and on missing run_date) the node logs an
      error and RAISES — fail-loud per feedback_no_silent_fails (a
      graceful-degrade here would silently recreate the orphaned-producer
      bug this wiring exists to fix);
  (c) graph-wiring assertions (static AST inspection of build_graph,
      mirroring tests/test_regime_stage_b_graph_topology.py) that the
      new node runs AFTER fetch_data and strictly BEFORE
      compute_focus_list_node AND score_aggregator.

Behavior-safety note: producing the substrate does NOT change scoring
or agent behavior — FACTOR_BLEND_ENABLED / FOCUS_LIST_GATING_ENABLED
both default False; no flag is flipped by the node. These tests assert
the node is substrate-only (returns an observability delta, never
threads profiles through state).
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO_ROOT / "graph" / "research_graph.py"


def _build_graph_source() -> str:
    """Extract the body of ``build_graph()`` as source text (mirrors
    tests/test_regime_stage_b_graph_topology.py — string-search
    invariants without parsing the compiled Pregel)."""
    source = GRAPH_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_graph":
            return ast.get_source_segment(source, node) or ""
    raise AssertionError("build_graph function not found in research_graph.py")


# ── (a) success path ─────────────────────────────────────────────────────────

def test_node_calls_producer_with_state_run_date_and_sector_map():
    """On success the node invokes compute_and_write_factor_profiles with
    exactly the state's run_date + sector_map and returns the
    observability delta (substrate-only — profiles NOT threaded)."""
    from graph.research_graph import compute_factor_profiles_node

    state = {
        "run_date": "2026-05-18",
        "sector_map": {"NVDA": "Tech", "JPM": "Financials"},
    }

    with patch(
        "graph.research_graph.compute_and_write_factor_profiles",
        return_value="factors/profiles/2026-05-18/by_ticker.json",
    ) as mock_producer:
        delta = compute_factor_profiles_node(state)

    mock_producer.assert_called_once_with(
        run_date="2026-05-18",
        sector_map={"NVDA": "Tech", "JPM": "Financials"},
    )
    assert delta == {
        "factor_profiles_written": True,
        "factor_profiles_s3_key": "factors/profiles/2026-05-18/by_ticker.json",
    }
    # Substrate-only: the profiles dict itself must NOT be threaded
    # through state (consumers read from S3 by design).
    assert "factor_profiles" not in delta
    assert "factor_profiles_by_ticker" not in delta


def test_node_defaults_sector_map_to_empty_when_absent():
    """A missing sector_map degrades to {} (producer ranks under
    'Unknown') rather than KeyError-ing the research run."""
    from graph.research_graph import compute_factor_profiles_node

    with patch(
        "graph.research_graph.compute_and_write_factor_profiles",
        return_value="factors/profiles/2026-05-18/by_ticker.json",
    ) as mock_producer:
        delta = compute_factor_profiles_node({"run_date": "2026-05-18"})

    mock_producer.assert_called_once_with(
        run_date="2026-05-18", sector_map={},
    )
    assert delta["factor_profiles_written"] is True


# ── (b) fail-loud paths (feedback_no_silent_fails) ───────────────────────────

def test_node_raises_on_producer_exception(caplog):
    """ANY producer failure (missing features parquet, S3 error, compute
    exception) → node logs an ERROR and RAISES. The Research SF state must
    fail loudly; graceful-degrade would silently recreate the exact
    orphaned-producer class this wiring exists to fix."""
    from graph.research_graph import compute_factor_profiles_node

    state = {"run_date": "2026-05-18", "sector_map": {"NVDA": "Tech"}}

    with patch(
        "graph.research_graph.compute_and_write_factor_profiles",
        side_effect=RuntimeError(
            "NoSuchKey: features/2026-05-18/technical.parquet"
        ),
    ):
        with pytest.raises(RuntimeError, match="NoSuchKey"):
            compute_factor_profiles_node(state)

    # Flow-doctor-visible ERROR log emitted before the re-raise.
    assert any(
        rec.levelno >= 40 and "compute_factor_profiles" in rec.message
        for rec in caplog.records
    ), f"expected an ERROR log mentioning the node; got {caplog.records!r}"


def test_node_raises_when_run_date_missing():
    """No run_date in state → cannot produce; node RAISES without calling
    the producer (fail-loud, not a silent degrade)."""
    from graph.research_graph import compute_factor_profiles_node

    with patch(
        "graph.research_graph.compute_and_write_factor_profiles",
    ) as mock_producer:
        with pytest.raises(RuntimeError, match="no run_date"):
            compute_factor_profiles_node({"sector_map": {"NVDA": "Tech"}})

    # Producer must not even be called without a run_date.
    mock_producer.assert_not_called()


# ── (c) graph-wiring invariants (static AST, mirrors stage-b topology) ────────

def test_build_graph_registers_compute_factor_profiles_node():
    """The node must be registered in build_graph."""
    body = _build_graph_source()
    assert (
        'graph.add_node("compute_factor_profiles_node", compute_factor_profiles_node)'
        in body
    ), "compute_factor_profiles_node not registered in build_graph"


def test_factor_profiles_node_runs_after_fetch_data_via_macro_chain():
    """The node must sit downstream of fetch_data. The serial chain is
    fetch_data → load_regime_substrate_node → load_scorecard_node →
    macro_economist_node → compute_factor_profiles_node, so fetch_data
    (which populates sector_map + run_date) provably precedes it. Pin
    the incoming edge from macro (the established serial-upstream
    node)."""
    body = _build_graph_source()
    assert (
        'graph.add_edge("macro_economist_node", "compute_factor_profiles_node")'
        in body
    ), (
        "compute_factor_profiles_node is not spliced after "
        "macro_economist_node — it needs sector_map + run_date (set in "
        "fetch_data, unchanged by the substrate loader / scorecard loader / "
        "macro) before it can produce factor profiles."
    )
    # And the serial fetch_data → ... → macro chain that guarantees
    # fetch_data ran first must still be intact. Phase 2.A.3 (scorecard)
    # spliced load_scorecard_node between the substrate loader and macro;
    # the chain is now 4 nodes long but the invariant holds.
    assert (
        'graph.add_edge("fetch_data", "load_regime_substrate_node")' in body
        and 'graph.add_edge("load_regime_substrate_node", "load_scorecard_node")'
        in body
        and 'graph.add_edge("load_scorecard_node", "macro_economist_node")'
        in body
    ), "the fetch_data → substrate → scorecard → macro serial chain was broken"


def test_factor_profiles_node_runs_strictly_before_compute_focus_list():
    """compute_focus_list_node reads factors/profiles/latest.json from
    S3 — it must run AFTER the producer node. Pin the direct edge
    compute_factor_profiles_node → compute_focus_list_node and assert
    macro no longer edges straight to the focus list (the producer is
    spliced in between)."""
    body = _build_graph_source()
    assert (
        'graph.add_edge("compute_factor_profiles_node", "compute_focus_list_node")'
        in body
    ), (
        "compute_factor_profiles_node must edge directly into "
        "compute_focus_list_node so the focus-list consumer reads a "
        "freshly-written substrate this same run."
    )
    assert (
        'graph.add_edge("macro_economist_node", "compute_focus_list_node")'
        not in body
    ), (
        "stale direct macro → compute_focus_list_node edge still present; "
        "the producer node must be spliced strictly between them."
    )


def test_factor_profiles_node_runs_strictly_before_score_aggregator():
    """score_aggregator also reads factors/profiles via
    read_factor_profiles_from_s3(). It is downstream of the sector
    dispatch (dispatch → merge_results → score_aggregator), and the
    producer node is upstream of that dispatch (it edges into
    compute_focus_list_node, off which the conditional dispatch hangs).
    Pin that the dispatch hangs off compute_focus_list_node (which the
    producer feeds) and that merge_results → score_aggregator holds, so
    the producer provably precedes score_aggregator."""
    body = _build_graph_source()
    assert (
        'graph.add_conditional_edges("compute_focus_list_node", dispatch_sectors_and_exit)'
        in body
    ), (
        "sector dispatch must hang off compute_focus_list_node (which the "
        "factor-profiles producer feeds) — this is what places the "
        "producer upstream of score_aggregator."
    )
    assert (
        'graph.add_edge("merge_results", "score_aggregator")' in body
    ), "merge_results → score_aggregator edge missing"


def test_factor_profiles_node_does_not_break_dispatch_chain():
    """The new node must not introduce a Send fan-out or hang the
    conditional dispatch off itself — dispatch stays on
    compute_focus_list_node (unchanged), preserving the regime /
    macro / focus-list / dispatch chain."""
    body = _build_graph_source()
    assert (
        'graph.add_conditional_edges("compute_factor_profiles_node"' not in body
    ), (
        "the factor-profiles producer must be a plain serial node, not a "
        "dispatch point — it must not alter the sector fan-out wiring."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
