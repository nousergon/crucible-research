"""Merge-time port of `sf_preflight.py::check_recursion_budget_for_response_format`.

`alpha-engine-config-I11313` deliverable 2 (the CAP_CHECKOUT half of
`alpha-engine-config-I11112` deliverable 3). The assertion is a pure static
scan of files THIS repo owns, with no `run_date` dependency and no live AWS
state — so it belongs on every PR touching them, not in a Saturday preflight
that could not reach them.

## Why the runtime copy could not run, measured 2026-09-22

`sf_preflight.py` resolved these files through
`_sibling_repo("alpha-engine-research")`, and no directory of that name
exists anywhere on the fleet — the repos were renamed to `crucible-*` and the
spot box's checkout directories were not. So the check was dead twice over:
capability-gated out of the Lambda AND pointed at a nonexistent path on any
host that HAS `CAP_CHECKOUT`. It reported `skip`, which rendered as
`status: OK, warn_count: 0` (I11112).

## The failure class

`GraphRecursionError` mid-run. A ReAct agent using `response_format=` spends
graph supersteps on a post-loop structured-extraction call that a bare
`MAX_ITERATIONS * 2` budget does not cover, so the agent dies after
completing its research. The invariant has since been strengthened past what
the preflight asserted: `react_budget.workload_derived_recursion_limit` is
now the single chokepoint, and a hand-tuned constant is a FLOOR, never the
ceiling (`config#1822` — bumping a constant moves the cliff rather than
removing it).

This file asserts BOTH: the original `response_format=`-vs-bare-`* 2` rule
that the preflight carried, and the chokepoint rule that superseded it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SECTOR_TEAMS = Path(__file__).resolve().parents[1] / "agents" / "sector_teams"

#: The ReAct analysts the preflight named. Kept explicit rather than globbed
#: so a DELETED analyst is a test failure that has to be acknowledged, not a
#: silently smaller scan — an assertion whose subject list shrinks to zero is
#: the vacuous-green shape I11112 exists to remove.
ANALYSTS = ("quant_analyst.py", "qual_analyst.py")

#: A `recursion_limit` sized directly off a `*_MAX_ITERATIONS` constant times
#: two, with no `+ N` tail. This is the exact shape that crashed on the
#: structured-extraction call.
BARE_X2 = re.compile(r"recursion_limit[\"']?\s*[:=]\s*\w*MAX_ITERATIONS\s*\*\s*2(?!\s*\+)")

#: The shared chokepoint every analyst must size its budget through.
CHOKEPOINT = "workload_derived_recursion_limit"


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ANALYSTS:
        path = SECTOR_TEAMS / name
        assert path.is_file(), (
            f"{path} is missing. If this analyst was retired, remove it from "
            f"ANALYSTS in this file and say why — a scan that silently loses "
            f"its subject reports green over nothing "
            f"(alpha-engine-config-I11112)."
        )
        out[name] = path.read_text()
    return out


def test_the_scan_has_subjects(sources):
    """Non-vacuity floor. Every assertion below iterates `sources`; if it
    were empty they would all pass having examined nothing."""
    assert len(sources) == len(ANALYSTS)
    assert all(sources.values()), "an analyst source file read as empty"


def test_no_analyst_uses_a_bare_max_iterations_times_two_budget(sources):
    """The original preflight assertion, verbatim in intent: a ReAct site
    whose `recursion_limit` is `MAX_ITERATIONS * 2` with no buffer dies on
    the post-loop structured-extraction call."""
    offenders = [name for name, src in sources.items() if BARE_X2.search(src)]
    assert not offenders, (
        f"{offenders} size recursion_limit as a bare `MAX_ITERATIONS * 2` "
        f"with no `+ N` tail. The post-loop structured-extraction call spends "
        f"from the same budget, so the agent raises GraphRecursionError after "
        f"completing its research. Size it through "
        f"`react_budget.{CHOKEPOINT}` instead."
    )


def test_every_analyst_sizes_its_budget_through_the_shared_chokepoint(sources):
    """`config#1822`: a hand-tuned constant does not scale with the work-unit
    or tool counts and silently under-budgets when either grows — fixing one
    analyst and leaving its sibling on a constant is how the same cliff was
    hit twice. The chokepoint is the invariant; the constant is a floor."""
    missing = [name for name, src in sources.items() if CHOKEPOINT not in src]
    assert not missing, (
        f"{missing} do not route their recursion_limit through "
        f"`react_budget.{CHOKEPOINT}`. Deriving the ceiling from the live "
        f"workload is what makes it provably adequate; bumping a constant "
        f"just moves the cliff (config#1822)."
    )


def test_a_max_iterations_constant_is_only_ever_a_floor(sources):
    """Guards the direction of the regression: `*_MAX_ITERATIONS` may still
    appear, but only as `floor_iterations=`, never as the ceiling."""
    for name, src in sources.items():
        for match in re.finditer(r"\w*MAX_ITERATIONS\b", src):
            line = src[: match.start()].count("\n") + 1
            context = src.splitlines()[line - 1]
            if context.lstrip().startswith("#") or "floor_iterations" in context:
                continue
            assert "import" in context or "from" in context or "*" not in context, (
                f"{name}:{line} multiplies a MAX_ITERATIONS constant outside "
                f"a `floor_iterations=` argument: {context.strip()!r}. The "
                f"constant is a floor, not a budget (config#1822)."
            )


def test_the_bare_pattern_would_actually_be_caught():
    """NEGATIVE CONTROL. Without this, a regex that never matches anything
    makes every assertion above pass for the wrong reason — which is exactly
    how the preflight ancestor reported `ok` while checking nothing."""
    assert BARE_X2.search('        "recursion_limit": QUANT_MAX_ITERATIONS * 2,\n')
    assert BARE_X2.search("    recursion_limit = QUAL_MAX_ITERATIONS * 2\n")
    # …and does NOT fire on the buffered or chokepoint-derived shapes.
    assert not BARE_X2.search('        "recursion_limit": QUANT_MAX_ITERATIONS * 2 + 2,\n')
    assert not BARE_X2.search(
        '        "recursion_limit": workload_derived_recursion_limit(n, t, floor_iterations=Q),\n'
    )


def test_the_chokepoint_still_covers_workload_plus_a_synthesis_margin():
    """The rule the chokepoint encodes, pinned against the implementation:
    the budget must cover every work unit against every tool AND leave room
    to synthesize, or a fully-researching agent dies before answering."""
    from agents.sector_teams.react_budget import workload_derived_recursion_limit

    # 13 tickers x 9 tools is the industrials case that hit the cliff.
    limit = workload_derived_recursion_limit(13, 9, floor_iterations=20)
    rounds = (limit - 2) // 2
    assert rounds >= 13 * 9, (
        f"budget covers {rounds} rounds but the bounded worst case is "
        f"{13 * 9} — this is the config#1822 cliff reopening"
    )
    assert rounds > 13 * 9, "no synthesis margin above the worst-case workload"

    # The floor still applies when the workload is small.
    assert workload_derived_recursion_limit(1, 1, floor_iterations=30) == 30 * 2 + 2
