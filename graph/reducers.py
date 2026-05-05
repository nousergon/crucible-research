"""
LangGraph state reducers — canonical-key-ordering, typed-aware variants.

These reducers are the typed-state successor to ``research_graph._merge_dicts``
(shipped 2026-04-29 as PR #50 with canonical key ordering). They are wired
into ``ResearchState`` via ``Annotated[T, reducer]`` field types — see
``graph.research_graph.ResearchState`` for the per-field reducer assignments.

Three reducer behaviors:

- :func:`take_last` — overwrite-on-write semantics for genuinely scalar fields
  (``run_date``, ``market_regime``, ``open_slots``, …). Single-writer-by-
  convention; if a second writer ever appears, the first write is silently
  lost. Use only for fields whose semantic is "last write wins."

- :func:`merge_typed_dicts` — merge two ``dict[str, T]`` values from parallel
  branches with last-write-wins per key, returning canonical (sorted) key
  order. Use for fields where parallel branches legitimately contribute
  partial-disjoint key sets but overlapping keys are tolerable
  (e.g. per-ticker accumulating dicts where two enrichment sources may
  overwrite each other on the rare overlap).

- :func:`reject_on_conflict` — like :func:`merge_typed_dicts` but raises
  ``RuntimeError`` on overlapping keys. Use for fields where parallel
  branches MUST partition the keyspace (e.g. Send fan-out where each
  branch owns a disjoint ``team_id`` — a duplicate would indicate a
  graph wiring bug, not a legitimate merge).

All dict-merging reducers return canonical (sorted) key order. This was
load-bearing on ``_merge_dicts`` to close the LangGraph Send-completion-order
non-determinism class diagnosed 2026-04-29 (see
``tests/fixtures/BASELINE_README.md``); the same invariant is preserved here
for consistency across reducer variants.

Workstream context: ``~/Development/alpha-engine-docs/private/alpha-engine-
research-typed-state-capture-260429.md`` (Day-1 design doc).
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def take_last(_left, right):
    """Overwrite reducer — last write wins. Use for genuinely scalar fields."""
    return right


def merge_typed_dicts(left: dict | None, right: dict | None) -> dict:
    """
    Merge two ``dict[str, T]`` values from parallel branches.

    Last-write-wins on overlapping keys. Returns canonical (sorted) key
    order so downstream consumers iterating ``.items()`` get a deterministic
    sequence regardless of which branch's update arrived first.
    """
    if left is None:
        merged = dict(right or {})
    elif right is None:
        merged = dict(left)
    else:
        merged = {**left, **right}
    return {k: merged[k] for k in sorted(merged)}


def reject_on_conflict(left: dict | None, right: dict | None) -> dict:
    """
    Strict-merge two ``dict[str, T]`` values from parallel branches.

    Raises ``RuntimeError`` if any key is written by both branches, on the
    assumption that overlapping keys signal a graph-wiring bug (e.g. two
    Send branches both writing the same ``team_id``). For fields where
    overlap is legitimate, use :func:`merge_typed_dicts` instead.

    Returns canonical (sorted) key order.
    """
    if left is None:
        merged = dict(right or {})
    elif right is None:
        merged = dict(left)
    else:
        overlap = set(left) & set(right)
        if overlap:
            raise RuntimeError(
                f"reject_on_conflict: keys written by multiple branches: "
                f"{sorted(overlap)}. Each branch must own a disjoint "
                f"keyspace; an overlap indicates a graph-wiring bug. "
                f"If overlapping keys are legitimate, use merge_typed_dicts."
            )
        merged = {**left, **right}
    return {k: merged[k] for k in sorted(merged)}
