"""
Trajectory validation for the research graph.

After each pipeline run, validates that all expected LangGraph nodes executed
in the correct order. Uses LangSmith traces collected during execution.

This module never blocks the pipeline — all errors are caught and logged.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

logger = logging.getLogger(__name__)

# ── Reference trajectory from graph/research_graph.py ────────────────────────

REQUIRED_NODES = [
    "fetch_data",
    "sector_team_node",
    "macro_economist_node",
    "exit_evaluator_node",
    "merge_results",
    "score_aggregator",
    "cio_node",
    "population_entry_handler",
    "consolidator_node",
    "archive_writer",
    "email_sender_node",
]

# (before, after) — "before" must appear earlier in the trace than "after"
ORDERING_CONSTRAINTS = [
    ("fetch_data", "sector_team_node"),
    ("fetch_data", "macro_economist_node"),
    ("fetch_data", "exit_evaluator_node"),
    ("sector_team_node", "merge_results"),
    ("macro_economist_node", "merge_results"),
    ("exit_evaluator_node", "merge_results"),
    ("merge_results", "score_aggregator"),
    ("score_aggregator", "cio_node"),
    ("cio_node", "population_entry_handler"),
    ("population_entry_handler", "consolidator_node"),
    ("consolidator_node", "archive_writer"),
    ("archive_writer", "email_sender_node"),
]

EXPECTED_SECTOR_TEAM_COUNT = 6

# ── Final-state structural evidence (config#2263) ────────────────────────────
#
# Checkpoint-resumed runs (see graph/research_graph.py's sector_team_node /
# macro_economist_node / cio_node resume short-circuits) can finish fast
# enough to race LangSmith's async span flush — a resumed run that
# short-circuits most/all checkpointable work has far fewer/shorter spans
# than a normal multi-minute run, so the existing completeness poll can
# legitimately time out with ZERO spans landed even though the run itself
# was entirely successful (2026-07-11 watch-rerun-6 incident: 11/11 nodes
# reported missing, sector_team_count 0/6, immediately followed by a
# successful health-status write).
#
# Rather than chase the LangSmith timing further, a resumed run is instead
# validated against ``final_state`` — the LangGraph final state dict, which
# is populated identically by a node's resume-hit branch and its
# fresh-compute branch (see the state-update shapes returned by the three
# resume-capable nodes). This is a reliable, timing-independent completion
# signal. Each entry below is ``node_name -> (final_state) -> bool``,
# True meaning "this node's output is present in final_state".
#
# ``archive_writer`` returns ``{}`` on success (no distinguishing state key)
# so its completion is inferred transitively: the graph wiring is a strict
# unconditional chain ``population_entry_handler -> consolidator_node ->
# archive_writer -> email_sender_node -> END`` (add_edge, not
# add_conditional_edges), so if ``email_sender_node`` ran — evidenced by
# the ``email_sent`` key being PRESENT in final_state regardless of its
# True/False value — ``archive_writer`` necessarily ran first.
STRUCTURAL_EVIDENCE: dict[str, Callable[[dict], bool]] = {
    "fetch_data": lambda s: bool(
        s.get("scanner_universe") or s.get("price_data") or s.get("data_snapshot_id")
    ),
    "sector_team_node": lambda s: (
        len(s.get("sector_team_outputs", {}) or {}) == EXPECTED_SECTOR_TEAM_COUNT
    ),
    "macro_economist_node": lambda s: "market_regime" in s,
    "exit_evaluator_node": lambda s: (
        "remaining_population" in s or "exits" in s or "open_slots" in s
    ),
    "merge_results": lambda s: "team_slot_allocation" in s,
    "score_aggregator": lambda s: "investment_theses" in s,
    "cio_node": lambda s: "ic_decisions" in s,
    "population_entry_handler": lambda s: (
        "new_population" in s or "population_rotation_events" in s
    ),
    "consolidator_node": lambda s: "consolidated_report" in s,
    # Transitive — see module docstring above.
    "archive_writer": lambda s: "email_sent" in s,
    "email_sender_node": lambda s: "email_sent" in s,
}


def _structural_validate(final_state: dict) -> tuple[dict[str, int], list[str]]:
    """Validate node completeness from the final LangGraph state.

    Returns ``(node_counts, failures)`` in the same shape the trace-based
    path produces, so callers don't need to branch on which path ran.
    ``node_counts`` here is evidence-based (0 or 1, except
    ``sector_team_node`` which reports the real team count) rather than a
    true span count — it exists so downstream consumers/tests can still
    read e.g. ``node_counts["sector_team_node"]``.
    """
    node_counts: dict[str, int] = {}
    failures: list[str] = []

    for required in REQUIRED_NODES:
        check = STRUCTURAL_EVIDENCE.get(required)
        if check is None:
            # Should never happen — every REQUIRED_NODES entry has a
            # mapping above. Fail loud rather than silently skip.
            failures.append(f"missing_node: {required}")
            node_counts[required] = 0
            continue
        if required == "sector_team_node":
            team_count = len(final_state.get("sector_team_outputs", {}) or {})
            node_counts["sector_team_node"] = team_count
            if team_count != EXPECTED_SECTOR_TEAM_COUNT:
                failures.append(
                    f"sector_team_count: expected {EXPECTED_SECTOR_TEAM_COUNT}, "
                    f"got {team_count}"
                )
            if team_count == 0:
                failures.append(f"missing_node: {required}")
            continue
        present = bool(check(final_state))
        node_counts[required] = 1 if present else 0
        if not present:
            failures.append(f"missing_node: {required}")

    return node_counts, failures


def validate_trajectory(
    project_name: str = "alpha-research",
    max_wait_seconds: int = 15,
    completeness_timeout_seconds: int = 30,
    final_state: dict | None = None,
) -> dict | None:
    """
    Validate the most recent LangGraph run's trajectory against the reference.

    Queries LangSmith for the latest completed run in the project, extracts
    the child span node names, and checks:
      1. All required nodes are present
      2. sector_team_node appears exactly 6 times (one per Send)
      3. Ordering constraints are satisfied

    Checkpoint-resumed runs (config#2263): when ``final_state`` is provided
    AND it shows at least one node was resumed from an S3 checkpoint
    (``final_state["checkpoint_resumed_nodes"]`` non-empty), node-presence
    and sector-team-count checks fall back to structural evidence in
    ``final_state`` instead of the LangSmith trace — a resumed run can
    finish fast enough to race LangSmith's async span flush, and the
    per-invocation trace legitimately won't show nodes that were
    short-circuited by a checkpoint hit rather than executed this
    invocation. Ordering constraints remain trace-based best-effort (not
    checked when unavailable — never a hard failure on a resumed run,
    since checkpoint hits don't preserve original execution order/timing).
    A fully-fresh run (``final_state`` omitted, or provided with no
    checkpoint-resume evidence) runs the strict trace-only checks below,
    completely unchanged.

    Args:
        project_name: LangSmith project name (matches LANGCHAIN_PROJECT env var)
        max_wait_seconds: Max time to wait for the root run to appear in
            LangSmith
        completeness_timeout_seconds: Max time to wait for ALL required
            child spans to appear after the root run is found. This is the
            propagation race the validator was getting bit by every run —
            ``email_sender_node`` is the last graph node before END, so its
            child span lands in LangSmith after ``graph.invoke()`` returns
            and after the validator's first child-runs query. Default 30s
            covers LangSmith's typical Lambda-side flush latency.
        final_state: Optional LangGraph final state dict (the return value
            of ``graph.invoke()``). When it carries checkpoint-resume
            evidence, enables the structural-completeness fallback
            described above. Omit for byte-identical behavior to before
            this parameter existed.

    Returns:
        {"passed": bool, "failures": list[str], "node_counts": dict, "duration_ms": int}
        or None if tracing is not enabled or validation could not run.
    """
    if os.environ.get("LANGCHAIN_TRACING_V2") != "true":
        logger.info("Trajectory validation skipped — LANGCHAIN_TRACING_V2 not set")
        return None

    try:
        from langsmith import Client
    except ImportError:
        logger.warning("Trajectory validation skipped — langsmith not installed")
        return None

    client = Client()
    failures: list[str] = []

    resumed_nodes = (final_state or {}).get("checkpoint_resumed_nodes") or {}
    is_resumed_run = bool(resumed_nodes)

    # Wait for the most recent run to appear in LangSmith
    run = None
    for attempt in range(max_wait_seconds // 3 + 1):
        if attempt > 0:
            time.sleep(3)
        try:
            runs = list(client.list_runs(
                project_name=project_name,
                is_root=True,
                limit=1,
            ))
            if runs:
                run = runs[0]
                break
        except Exception as e:
            logger.warning("LangSmith query attempt %d failed: %s", attempt + 1, e)

    if run is None:
        logger.warning("No runs found in LangSmith project '%s'", project_name)
        return {"passed": False, "failures": ["no_run_found"], "node_counts": {}, "duration_ms": 0}

    # Fetch child spans (graph node executions). Poll until either all
    # REQUIRED_NODES are present or the completeness timeout fires.
    #
    # Origin: prior to this poll loop, the validator did a single fetch
    # immediately after the root-run lookup. ``email_sender_node`` is the
    # terminal node before ``END`` and its child span propagates to
    # LangSmith *after* ``graph.invoke()`` returns — the validator was
    # racing the async flusher and reporting ``missing_node:
    # email_sender_node`` on every run. This trained operators to ignore
    # the alarm. Keep the poll bounded so a real graph regression
    # (mid-flow node truly missing) still surfaces in <30s.
    #
    # Resumed runs (config#2263) skip the wait entirely: node-presence
    # comes from final_state (see is_resumed_run branch below), so the
    # trace is only consulted best-effort for ORDERING data. Polling for
    # full trace completeness on a resumed run just re-introduces the
    # exact flush-race false-positive class this fallback exists to fix —
    # a single immediate fetch is enough for whatever ordering data is
    # already available.
    poll_interval = 3
    deadline = time.time() if is_resumed_run else time.time() + completeness_timeout_seconds
    child_runs: list = []
    last_fetch_error: str | None = None
    while True:
        try:
            child_runs = list(client.list_runs(
                project_name=project_name,
                trace_id=run.trace_id,
                is_root=False,
            ))
            last_fetch_error = None
        except Exception as e:
            last_fetch_error = str(e)
            logger.warning("Failed to fetch child runs: %s", e)
        observed_names = {c.name for c in child_runs if c.name}
        missing = [n for n in REQUIRED_NODES if n not in observed_names]
        if not missing:
            break
        if time.time() >= deadline:
            break
        logger.debug(
            "Trajectory child-span poll: %d/%d required nodes seen "
            "(missing=%s) — waiting %ds",
            len(REQUIRED_NODES) - len(missing), len(REQUIRED_NODES),
            missing, poll_interval,
        )
        time.sleep(poll_interval)
    if last_fetch_error is not None and not is_resumed_run:
        return {
            "passed": False,
            "failures": [f"fetch_children_failed: {last_fetch_error}"],
            "node_counts": {}, "duration_ms": 0,
        }

    # Extract node names and their earliest start times
    node_names: list[str] = []
    node_first_start: dict[str, float] = {}
    for child in child_runs:
        name = child.name
        if name and child.start_time:
            node_names.append(name)
            ts = child.start_time.timestamp()
            if name not in node_first_start or ts < node_first_start[name]:
                node_first_start[name] = ts

    # Count occurrences
    node_counts: dict[str, int] = {}
    for name in node_names:
        node_counts[name] = node_counts.get(name, 0) + 1

    if is_resumed_run:
        # Structural fallback (config#2263): this invocation used the S3
        # checkpoint short-circuit on at least one node, so the
        # per-invocation LangSmith trace is not a reliable completeness
        # signal — resumed nodes plausibly never emit a span this
        # invocation at all, and/or the whole run finishes fast enough to
        # race the async flush (see module-level comment on
        # STRUCTURAL_EVIDENCE). Validate against final_state instead.
        struct_counts, struct_failures = _structural_validate(final_state or {})
        node_counts = struct_counts
        team_count = struct_counts.get("sector_team_node", 0)
        failures.extend(struct_failures)
        logger.info(
            "Trajectory validation using final-state structural fallback "
            "(checkpoint-resumed nodes this run: %s)", sorted(resumed_nodes),
        )
    else:
        # Check 1: All required nodes present
        for required in REQUIRED_NODES:
            if required not in node_counts:
                failures.append(f"missing_node: {required}")

        # Check 2: sector_team_node count
        team_count = node_counts.get("sector_team_node", 0)
        if team_count != EXPECTED_SECTOR_TEAM_COUNT:
            failures.append(
                f"sector_team_count: expected {EXPECTED_SECTOR_TEAM_COUNT}, got {team_count}"
            )

    # Check 3: Ordering constraints. Best-effort on a resumed run — a
    # checkpoint hit means the node's original execution order/timing
    # isn't reproduced by this invocation's (possibly absent) span, so we
    # only check pairs where BOTH spans happen to be present in the trace
    # we did fetch, and never fail validation solely because ordering
    # couldn't be verified. On a fresh run this is unchanged: both
    # timestamps come from real spans of this invocation.
    for before, after in ORDERING_CONSTRAINTS:
        t_before = node_first_start.get(before)
        t_after = node_first_start.get(after)
        if t_before is not None and t_after is not None:
            if t_before > t_after:
                failures.append(f"ordering_violation: {before} started after {after}")

    # Compute duration
    duration_ms = 0
    if run.end_time and run.start_time:
        duration_ms = int((run.end_time - run.start_time).total_seconds() * 1000)

    passed = len(failures) == 0

    if passed:
        logger.info(
            "Trajectory validation PASSED — %d nodes, %d sector teams, %dms",
            len(node_names), team_count, duration_ms,
        )
    else:
        logger.error(
            "Trajectory validation FAILED — %d failures: %s",
            len(failures), failures,
        )

    return {
        "passed": passed,
        "failures": failures,
        "node_counts": node_counts,
        "duration_ms": duration_ms,
    }
