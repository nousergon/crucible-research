"""The research slot's arm set, and the properties that make it comparable.

Brian, 2026-09-22 (alpha-engine-config-I11393 / -I11398): *"in the weekly sfs I
want to see those arms, and only those arms, competing for the champion
position."*

That requirement is enforced HERE and not in the Step Functions definition, and
the distinction is load-bearing. The weekly SF's ``ChallengerShadow`` state
invokes the runner with ``mode=challengers_only``, and
``producers.runner.run_challengers`` iterates
``buildable_challenger_producers()`` — so the arm set is decided by the
REGISTER. Enumerating arms in the ASL as well would create a second,
hand-maintained list, which is the alpha-engine-config-I9277 defect verbatim:
crucible-backtester's ``VALID_CHAMPIONS`` tuple silently omitted the only two
arms with sufficient evidence to win, and nothing anywhere recorded that they
were excluded or why.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producers.registry import (  # noqa: E402
    PINNED_RESEARCH_PREFILTER,
    RESEARCH_PRODUCERS,
    RESEARCH_SLOT,
    buildable_challenger_producers,
    research_slot_producers,
)

EXPECTED_ARMS = {
    "attractiveness_60": 60,
    "attractiveness_20": 20,
    "tech_score_20": 20,
    "predictor_from_60": 20,
    "thinktank_20": 20,
}


def test_the_slot_holds_exactly_the_five_ruled_arms():
    assert {s.name for s in research_slot_producers()} == set(EXPECTED_ARMS)


def test_the_weekly_sf_builds_those_arms_and_nothing_else():
    """The literal form of Brian's requirement.

    ``thinktank_20`` is absent BY DESIGN — its shadow is written by the Think
    Tank's own daily run, so it carries ``build=None`` and the runner skips it
    while the leaderboard scores it like any other arm.
    """
    built = {s.name for s in buildable_challenger_producers()}
    assert built == set(EXPECTED_ARMS) - {"thinktank_20"}


def test_every_arm_declares_its_own_width():
    for name, width in EXPECTED_ARMS.items():
        assert RESEARCH_PRODUCERS[name].width == width


def test_widths_deliberately_differ():
    """Not a count-matching assertion, and must never become one.

    §4's count-matching binds when the RANKING is under test. This slot runs
    ``attractiveness_60`` against ``attractiveness_20`` on the IDENTICAL
    ranking — the what-does-depth-cost experiment — which is only legible
    because the widths differ. The concentration that buys is priced by the
    slot's ``information_ratio`` primary metric, not by forcing one width.
    """
    widths = {s.width for s in research_slot_producers()}
    assert len(widths) >= 2, (
        "with one width the slot is not testing width, and the simpler "
        "count-matched arrangement it replaced would be the better design"
    )


def test_every_arm_draws_from_the_one_pinned_prefilter():
    """Arms differ in the thing under test — ranking and width — and in
    nothing else. Two arms drawn from different populations cannot be compared
    (§4), which is what made the pre-I11393 board unreadable."""
    for spec in research_slot_producers():
        assert spec.prefilter_cut == PINNED_RESEARCH_PREFILTER


def test_no_arm_is_stateful():
    """A position book's realized alpha embeds turnover and retention — the S
    slot's question. An arm that holds winners longer beats a weekly re-ranker
    without having selected better once (alpha-engine-config-I11390)."""
    for spec in research_slot_producers():
        assert spec.stateful is False


def test_the_superseded_arms_are_retired_not_deleted():
    """§6.3: the register row and the score history are RETAINED permanently —
    they are the record that makes a retirement reviewable."""
    for name in (
        "no_agent_quant",
        "single_agent_quant",
        "scanner_predictor_direct",
        "scanner_top20_predictor",
        "thinktank_coverage",
    ):
        spec = RESEARCH_PRODUCERS[name]
        assert spec.kind == "retired"
        assert spec.retired_date == "2026-09-22"
        assert spec.ineligible_reason, "a retirement carries its reason"


def test_a_retired_arm_is_never_built():
    built = {s.name for s in buildable_challenger_producers()}
    retired = {s.name for s in RESEARCH_PRODUCERS.values() if s.kind == "retired"}
    assert not (built & retired)


def test_registering_a_stateful_research_arm_is_refused_at_import():
    """The guard, exercised. §7.4: a guard that cannot fail is worse than no
    guard, because it reads as coverage."""
    from producers.registry import ProducerSpec, _assert_recipe_declared

    original = dict(RESEARCH_PRODUCERS)
    RESEARCH_PRODUCERS["_probe"] = ProducerSpec(
        name="_probe", kind="challenger", version="v1", description="probe",
        build=None, slot=RESEARCH_SLOT, prefilter_cut=PINNED_RESEARCH_PREFILTER,
        width=20, stateful=True,
    )
    try:
        with pytest.raises(ValueError, match="STATEFUL"):
            _assert_recipe_declared()
    finally:
        RESEARCH_PRODUCERS.clear()
        RESEARCH_PRODUCERS.update(original)


def test_registering_an_unpinned_research_arm_is_refused_at_import():
    from producers.registry import ProducerSpec, _assert_recipe_declared

    original = dict(RESEARCH_PRODUCERS)
    RESEARCH_PRODUCERS["_probe"] = ProducerSpec(
        name="_probe", kind="challenger", version="v1", description="probe",
        build=None, slot=RESEARCH_SLOT, prefilter_cut="tech_score_top_60",
        width=20, stateful=False,
    )
    try:
        with pytest.raises(ValueError, match="pinned pre-filter"):
            _assert_recipe_declared()
    finally:
        RESEARCH_PRODUCERS.clear()
        RESEARCH_PRODUCERS.update(original)


def test_an_undeclared_recipe_is_refused_at_import():
    from producers.registry import ProducerSpec, _assert_recipe_declared

    original = dict(RESEARCH_PRODUCERS)
    RESEARCH_PRODUCERS["_probe"] = ProducerSpec(
        name="_probe", kind="challenger", version="v1", description="probe",
        build=None, slot=RESEARCH_SLOT,
    )
    try:
        with pytest.raises(ValueError, match="width"):
            _assert_recipe_declared()
    finally:
        RESEARCH_PRODUCERS.clear()
        RESEARCH_PRODUCERS.update(original)
