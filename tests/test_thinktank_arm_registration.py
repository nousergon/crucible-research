"""Think Tank is a scored challenger arm, and the weekly run must not build it.

alpha-engine-config-I5195 scope 4b. Registration was held back on 2026-07-28
because the Think Tank had been dead 11 days on the 900s Lambda ceiling and
registering it would have added a permanently-missing arm. config-I5208 moved
the run to EC2 spot on 2026-07-29 and it resumed writing its shadow view, so
the arm is now registrable.

The trap these tests exist for: ``producers.runner`` calls ``spec.build(...)``
on every challenger and then RAISES if any expected name did not emit. A
``build=None`` arm in that list would raise TypeError inside the per-spec
except, land in ``errors``, and turn every weekly producer run red for a gap
that is not that run's to fill.
"""

from __future__ import annotations

from producers.registry import (
    RESEARCH_PRODUCERS,
    buildable_challenger_producers,
    challenger_producers,
)


def test_thinktank_coverage_is_a_registered_challenger():
    """Without this the leaderboard has no spec for the arm and never scores it
    — the state config-I5195 was filed against."""
    spec = RESEARCH_PRODUCERS.get("thinktank_coverage")
    assert spec is not None
    assert spec.kind == "challenger"


def test_thinktank_coverage_is_scored():
    """challenger_producers() is the SCORING set — the leaderboard reads each
    arm's shadow from S3 and does not care who wrote it."""
    assert "thinktank_coverage" in [s.name for s in challenger_producers()]


def test_thinktank_coverage_is_not_built_by_the_weekly_run():
    """Its shadow is written by the Think Tank's own daily run. If the weekly
    producer pass tried to build it, every run would go red."""
    assert "thinktank_coverage" not in [s.name for s in buildable_challenger_producers()]


def test_every_buildable_challenger_actually_has_a_build():
    """The invariant runner.py depends on. Stated once, here, rather than
    relying on each caller to remember it."""
    for spec in buildable_challenger_producers():
        assert spec.build is not None, spec.name


def test_buildable_is_a_strict_subset_of_scored():
    """An arm that is built but not scored would be wasted compute; the
    inclusion direction must hold."""
    scored = {s.name for s in challenger_producers()}
    buildable = {s.name for s in buildable_challenger_producers()}
    assert buildable <= scored


def test_the_shadow_prefix_matches_what_the_think_tank_writes():
    """The spec name IS the S3 path segment.

    scoring.leaderboard_producers reads
    ``signals_shadow/{producer}/{date}/signals.json`` keyed on spec.name, and
    the Think Tank writes ``signals_shadow/thinktank_coverage/{date}/``. A
    rename on either side silently yields an arm that scores zero cohorts
    forever, which is indistinguishable from an arm that is merely new.
    """
    assert RESEARCH_PRODUCERS["thinktank_coverage"].name == "thinktank_coverage"
