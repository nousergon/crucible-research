"""Regression guard for config#1458: archive_writer must source
``scanner_eval_log`` from ``ResearchState`` (populated in ``fetch_data`` from
``candidates.json``), NOT from the process-local
``run_quant_filter._last_eval_log`` module-attribute stash.

Root cause this pins
---------------------
PR#344 fixed ``_build_scanner_eval_rows`` to join the scanner's own gate
verdict (``quant_filter_pass`` / ``filter_fail_reason`` / ``scan_path`` /
``liquidity_pass`` / ``volatility_pass``) instead of reconstructing rows from
agent team-picks. But the join source archive_writer read —
``run_quant_filter._last_eval_log`` — is a module attribute set as a side
effect INSIDE the SAME process that calls ``run_quant_filter``. Post
L1995-Phase5, the Research Lambda (where ``archive_writer`` runs) no longer
calls ``run_quant_filter`` itself; only the standalone Scanner SF state does,
in a SEPARATE process (``lambda/scanner_handler.py`` →
``data/scanner_orchestrator.py::build_candidates_artifact``). So that stash
was ALWAYS empty in archive_writer's process, and every cycle after PR#344
merged (2026-06-30) silently degraded to quant_filter_pass=0 for 100% of
rows — confirmed against the live research.db (eval_date=2026-07-02:
quant_filter_pass=0/903, scan_path NULL/903).

The fix threads the eval log through candidates.json (which already crosses
the process boundary, read via ``am.load_candidates_json``) into
``ResearchState["scanner_eval_log"]`` (set in ``fetch_data``), and
archive_writer now reads ``state.get("scanner_eval_log")`` instead of the
stash. ``tests/test_scanner_eval_rows.py`` already pins the PURE
``_build_scanner_eval_rows`` join logic in isolation (with a hand-built eval
log) — it did NOT catch this bug because it never exercised the
candidates.json -> ResearchState -> archive_writer plumbing. This file closes
that gap.
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph.research_graph as rg  # noqa: E402
from graph.research_graph import _build_scanner_eval_rows  # noqa: E402

_RESEARCH_GRAPH = Path(rg.__file__).resolve()


# ── Source-anchored guard: archive_writer must NOT read the process-local
# ── stash, and MUST read it from state ───────────────────────────────────────

def _archive_writer_source() -> str:
    return inspect.getsource(rg.archive_writer)


def test_archive_writer_does_not_import_run_quant_filter():
    """The whole bug was archive_writer importing run_quant_filter to read
    its module-local ``_last_eval_log`` stash. If that import (or any
    attribute access on the callable) ever comes back, the cross-process bug
    is back too. AST-based (not substring) so an explanatory comment
    mentioning the old stash name (as this fix's own NOTE does) can't
    trip a false positive."""
    tree = ast.parse(_archive_writer_source())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name for alias in node.names}
            assert "run_quant_filter" not in names, (
                "archive_writer must not import run_quant_filter — the "
                "run_quant_filter._last_eval_log module-attribute stash is "
                "process-local and is ALWAYS empty in the Research Lambda's "
                "process post-L1995-Phase5 (config#1458). Source "
                "scanner_eval_log from state instead."
            )
        if isinstance(node, ast.Attribute) and node.attr == "_last_eval_log":
            raise AssertionError(
                "archive_writer must not reference "
                "run_quant_filter._last_eval_log (config#1458 cross-process "
                "regression) — found an attribute access on `_last_eval_log`."
            )


def test_archive_writer_reads_scanner_eval_log_from_state():
    """Pin the literal fix: archive_writer must read
    state.get('scanner_eval_log') and feed it to _build_scanner_eval_rows."""
    src = _archive_writer_source()
    assert 'state.get("scanner_eval_log")' in src or "state.get('scanner_eval_log')" in src, (
        "archive_writer must source scanner_eval_log from ResearchState "
        "(populated in fetch_data from candidates.json), not a process-local "
        "stash."
    )


def test_scanner_eval_log_field_declared_on_research_state():
    """ResearchState (the LangGraph state schema) must carry the field so it
    threads from fetch_data -> archive_writer through the graph's normal
    state-passing mechanism."""
    assert "scanner_eval_log" in rg.ResearchState.__annotations__, (
        "ResearchState must declare scanner_eval_log — otherwise it cannot "
        "safely flow from fetch_data to archive_writer as a graph state key."
    )


def test_fetch_data_source_sets_scanner_eval_log_from_resolve_agent_input_set():
    """fetch_data must actually populate the state key it declares, from the
    same _resolve_agent_input_set call that already loads candidates.json
    (avoiding a second S3 round-trip)."""
    src = inspect.getsource(rg.fetch_data)
    assert "scanner_eval_log" in src, (
        "fetch_data must unpack scanner_eval_log from _resolve_agent_input_set "
        "and include it in its returned state dict."
    )


# ── Behavioral guard: state -> _build_scanner_eval_rows join, both branches ──

def _minimal_state(*, scanner_eval_log):
    """A minimal ResearchState-shaped dict covering exactly what the
    scanner_eval_log join in archive_writer needs — NOT a full archive_writer
    invocation (that function does heavy S3/SQLite I/O out of scope here;
    the join logic itself is what this bug lives in)."""
    universe = ["PASS", "FAIL"]
    sector_map = {"PASS": "Tech", "FAIL": "Tech"}
    state: dict = {
        "scanner_universe": universe,
        "sector_map": sector_map,
        "technical_scores": {},
    }
    if scanner_eval_log is not None:
        state["scanner_eval_log"] = scanner_eval_log
    return state, universe, sector_map


def test_state_scanner_eval_log_flows_into_build_scanner_eval_rows():
    """Simulates what candidates.json would have supplied via fetch_data:
    a populated ResearchState["scanner_eval_log"]. Confirms the join
    correctly reflects the scanner's own gate verdict — the exact behavior
    that was silently broken by the cross-process stash read."""
    eval_log = [
        {"ticker": "PASS", "quant_filter_pass": 1, "scan_path": "momentum",
         "liquidity_pass": 1, "volatility_pass": 1},
        {"ticker": "FAIL", "quant_filter_pass": 0,
         "filter_fail_reason": "liquidity", "liquidity_pass": 0},
    ]
    state, universe, sector_map = _minimal_state(scanner_eval_log=eval_log)

    # Mirror archive_writer's exact call shape.
    scanner_eval_log = state.get("scanner_eval_log") or []
    rows = _build_scanner_eval_rows(
        scanner_universe=universe,
        extra_override_tickers=[],
        technical_scores=state["technical_scores"],
        sector_map=sector_map,
        scanner_eval_log=scanner_eval_log,
        focus_lookup={},
        run_date="2026-07-05",
    )
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["PASS"]["quant_filter_pass"] == 1
    assert by_ticker["PASS"]["scan_path"] == "momentum"
    assert by_ticker["FAIL"]["quant_filter_pass"] == 0
    assert by_ticker["FAIL"]["filter_fail_reason"] == "liquidity"


def test_state_missing_scanner_eval_log_degrades_to_fail_soft_zero(caplog):
    """When candidates.json carried no scanner_eval_log (absent key, or an
    empty list), archive_writer's fail-soft posture must be preserved: rows
    degrade to quant_filter_pass=0 / null reason (never fabricated), same as
    the pre-existing empty-eval-log contract in test_scanner_eval_rows.py."""
    state, universe, sector_map = _minimal_state(scanner_eval_log=None)

    scanner_eval_log = state.get("scanner_eval_log") or []
    assert scanner_eval_log == []  # the WARN-triggering condition

    rows = _build_scanner_eval_rows(
        scanner_universe=universe,
        extra_override_tickers=[],
        technical_scores=state["technical_scores"],
        sector_map=sector_map,
        scanner_eval_log=scanner_eval_log,
        focus_lookup={},
        run_date="2026-07-05",
    )
    assert all(r["quant_filter_pass"] == 0 for r in rows)
    assert all(r["filter_fail_reason"] is None for r in rows)


def test_state_empty_list_scanner_eval_log_also_degrades_to_fail_soft_zero():
    """Same as above but with scanner_eval_log explicitly present-but-empty
    (e.g. candidates.json wrote scanner_eval_log: [] because run_quant_filter
    produced no eval records) — must behave identically to the absent-key case."""
    state, universe, sector_map = _minimal_state(scanner_eval_log=[])

    scanner_eval_log = state.get("scanner_eval_log") or []
    assert scanner_eval_log == []

    rows = _build_scanner_eval_rows(
        scanner_universe=universe,
        extra_override_tickers=[],
        technical_scores=state["technical_scores"],
        sector_map=sector_map,
        scanner_eval_log=scanner_eval_log,
        focus_lookup={},
        run_date="2026-07-05",
    )
    assert all(r["quant_filter_pass"] == 0 for r in rows)


def test_archive_writer_warns_when_state_scanner_eval_log_empty():
    """The WARN-on-empty log path itself is source-anchored: a future edit
    must not silently drop the operator-visible warning when candidates.json
    carried no eval log for the cycle."""
    src = _archive_writer_source()
    idx = src.find('scanner_eval_log = state.get("scanner_eval_log")')
    assert idx != -1, "expected the state-sourced scanner_eval_log assignment"
    tail = src[idx:]
    idx_if = tail.find("if not scanner_eval_log:")
    idx_warn = tail.find("logger.warning(")
    assert idx_if != -1 and idx_warn != -1 and idx_if < idx_warn, (
        "archive_writer must WARN when scanner_eval_log is empty/absent from "
        "state, preserving the fail-soft-with-visible-warning contract."
    )
