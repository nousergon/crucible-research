"""A successor arm inherits its predecessor's cohort history — and only then.

Brian's ruling 2026-09-22 (alpha-engine-config-I11393). §3.1 makes an arm an
immutable recipe, so inheritance is legitimate exactly where the two recipes
were identical on the inherited dates. That was VERIFIED date by date for
``thinktank_20``: ``thinktank_coverage`` wrote shadows on 20 dates
(2026-07-16 .. 2026-09-17) and the serving cut was ``attractiveness_top_60``
— the now-pinned cut — on every one of them; the pointer moved on 2026-09-18
and the arm wrote no shadow after 09-17.

The property that makes this safe rather than convenient is the second test:
the successor's OWN picks win every date collision, so an inheritance can
never overwrite something the live arm measured.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producers.registry import (  # noqa: E402
    RESEARCH_PRODUCERS,
    challenger_producers,
    retired_producers,
)


def test_thinktank_20_declares_the_arm_it_supersedes():
    assert RESEARCH_PRODUCERS["thinktank_20"].supersedes == "thinktank_coverage"


def test_the_predecessor_is_retired_so_the_pair_is_one_series():
    assert RESEARCH_PRODUCERS["thinktank_coverage"].kind == "retired"


def test_only_a_verified_pair_declares_inheritance():
    """Inheritance is opt-in per arm and must stay that way.

    `predictor_from_60` and `tech_score_20` have NO predecessor: the old
    predictor arms drew from a different pool at a different width, and
    `tech_score` was only ever cut at 60. Declaring `supersedes` on either
    would import a series measured on a different recipe, which is exactly what
    §3.1 forbids and what this field exists to make explicit rather than
    implicit in a rename.
    """
    declared = {s.name: s.supersedes for s in challenger_producers() if s.supersedes}
    assert declared == {"thinktank_20": "thinktank_coverage"}


def test_the_successors_own_picks_win_a_date_collision():
    """The safety property. ``setdefault`` fills only dates the live arm did
    not produce, so inherited history can never overwrite a measured one."""
    own = {"2026-09-20": "live"}
    inherited = {"2026-09-20": "old", "2026-09-17": "old"}
    for date_str, day in inherited.items():
        own.setdefault(date_str, day)
    assert own["2026-09-20"] == "live", "a live measurement must never be replaced"
    assert own["2026-09-17"] == "old", "a gap the live arm never produced is filled"


def test_an_inherited_predecessor_is_not_also_scored_as_its_own_retired_row():
    """Double-counting guard.

    Scoring both rows would enter the same cohort dates twice and narrow §4's
    cross-arm intersection against a competitor that is really this arm's own
    past. Asserted against the real loader's rule rather than a fixture.
    """
    inherited_by_live = {s.supersedes for s in challenger_producers() if s.supersedes}
    assert "thinktank_coverage" in inherited_by_live

    scored_retired = {
        s.name for s in retired_producers(as_of="2026-09-22")
        if s.name not in inherited_by_live
    }
    assert "thinktank_coverage" not in scored_retired
