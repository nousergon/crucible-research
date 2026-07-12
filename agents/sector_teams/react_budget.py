"""Shared workload-derived ReAct step-budget chokepoint (config#1822).

Every sector-team ReAct analyst (``quant_analyst``, ``qual_analyst``, and any
future create_react_agent analyst) must size its LangGraph ``recursion_limit``
to the LIVE workload of the invocation, never to a hand-tuned constant. This
module is the single source of that math so the invariant is enforced in ONE
place instead of being re-derived — and silently drifting — at each call site.

Why a chokepoint and not a per-analyst constant
------------------------------------------------
``recursion_limit`` counts graph supersteps; each ReAct round = 1 LLM node +
1 tool node = 2 supersteps, plus a +2 tail for the stop turn. The budget MUST
cover the bounded worst-case workload: the agent researches each of
``n_work_units`` items (picks for qual, sector tickers for quant) across the
``n_tools`` available tools, and (confirmed in prod traces — config#1822) calls
them roughly ONE-per-round (near-sequential), so the worst case is
``n_work_units * n_tools`` tool rounds before it can synthesize its answer.

A hand-tuned ``*_MAX_ITERATIONS`` constant does NOT scale with the work-unit /
tool counts and silently under-budgets when either grows. This is exactly the
config#1822 failure class: on 2026-07-11 the qual analyst hit it (4 teams, 5
picks × 13 tools = 65 worst-case rounds > the tuned budget → partial →
ALL-AGENTS-STRICT hard-fail) and #404 fixed ONLY qual — leaving the quant
sibling on its own fixed constant, which then hit the SAME cliff on the
industrials team (13 tickers × ~9 tools; 40 legitimate non-repeating research
calls cut off mid-research). Bumping a constant again just moves the cliff;
deriving the ceiling from the live workload makes it provably adequate and
self-adjusting as the pick/ticker or tool set changes. The configured
``*_MAX_ITERATIONS`` is kept as a floor.

This is a HOW-it-runs execution-budget fix: it lets the agent COMPLETE its
designed research and never changes what it concludes. Fail-loud is preserved —
beyond full workload coverage + the synthesis margin, a non-terminating loop is
not a budget problem: langgraph's internal step-budget sentinel still fires and
the team degrades LOUDLY (``partial``), never silently.
"""

from __future__ import annotations

# Synthesis margin: rounds the agent needs AFTER its last tool call to write
# the final answer, plus slack for the occasional legitimate tool re-query.
# Bounded on purpose — see the module docstring on why exceeding
# workload + margin is a loop the sentinel catches, not a budget shortfall.
_REACT_SYNTHESIS_MARGIN_ROUNDS = 10


def workload_derived_recursion_limit(
    n_work_units: int,
    n_tools: int,
    *,
    floor_iterations: int,
    margin_rounds: int = _REACT_SYNTHESIS_MARGIN_ROUNDS,
) -> int:
    """LangGraph ReAct ``recursion_limit`` sized to THIS invocation's workload.

    Floors at ``floor_iterations`` rounds but grows to cover
    ``n_work_units * n_tools`` tool rounds plus a synthesis margin, so the
    agent can research every work unit with every tool AND still synthesize
    its answer. Returns supersteps: ``iterations * 2 + 2`` (one LLM node +
    one tool node per round, +2 stop-turn tail). See the module docstring for
    the full superstep math and the config#1822 rationale.
    """
    worst_case_rounds = n_work_units * n_tools + margin_rounds
    iterations = max(floor_iterations, worst_case_rounds)
    return iterations * 2 + 2
