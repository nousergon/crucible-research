"""Every live research arm states where its history comes from — or that it has
none, and why (alpha-engine-config-I11422).

The question this closes: `attractiveness_60` and `attractiveness_20` are, by
construction, the same ticker SETS as the cuts-board arms `attractiveness_top_60`
and `attractiveness_top_20`, which hold 22 and 13 scored dates with their
per-date `topn_alpha_vs_population` series retained. Do they inherit them?

DECIDED: NO. The full comparison is in `producers/registry.py::NO_HISTORY_IMPORT`'s
docstring; the short form is that the scoring formula is shared but the
provenance is not — the cuts loader collapses held decisions onto one
`cut_effective_date` while a research arm writes one shadow per run date, and
`attractiveness_20`'s rank order comes from a LATEST pointer
(`scanner/universe/latest.json`) whose historical state is unrecoverable, so its
past picks cannot be verified at all.

What this file pins is not the decision but its VISIBILITY: an arm's empty
ladder must be a declared fact. Before this guard, adding a research arm with no
history was indistinguishable from an inheritance that silently failed, and
granting one a 22-date head start needed no statement anywhere.

VERIFIED RED (champion-challenger-policy.md §7.4). With `NO_HISTORY_IMPORT`
emptied, importing the registry raises at module scope:

    ValueError: live 'research' arm(s) ['attractiveness_20', 'attractiveness_60',
    'predictor_from_60', 'tech_score_20'] declare neither `supersedes` nor an
    entry in NO_HISTORY_IMPORT.

Observed live while building this guard: it fired first on
`['predictor_from_60', 'tech_score_20']` when only the two funnel arms had been
written, which is the guard doing exactly its job on a half-finished register.
"""

from __future__ import annotations

import dataclasses

import pytest

from producers.registry import (
    NO_HISTORY_IMPORT,
    RESEARCH_PRODUCERS,
    RESEARCH_SLOT,
    _assert_history_provenance_declared,
    research_slot_producers,
)


class TestEveryArmDeclaresItsProvenance:
    def test_every_live_arm_either_inherits_or_states_why_not(self):
        for spec in research_slot_producers():
            assert bool(spec.supersedes) ^ (spec.name in NO_HISTORY_IMPORT), (
                f"{spec.name} must declare exactly one of `supersedes` or a "
                f"NO_HISTORY_IMPORT entry — never both, never neither"
            )

    def test_exactly_one_arm_inherits_and_it_is_the_think_tank(self):
        """The one verified same-surface inheritance (`crucible-research-PR820`),
        pinned so a second one cannot appear without a test changing."""
        inheriting = {
            s.name: s.supersedes for s in research_slot_producers() if s.supersedes
        }
        assert inheriting == {"thinktank_20": "thinktank_coverage"}

    def test_the_two_funnel_arms_do_not_inherit_the_cuts_board(self):
        """The decision itself. `attractiveness_60`/`_20` start at zero dates,
        alongside their siblings, rather than at 22 and 13."""
        for name in ("attractiveness_60", "attractiveness_20"):
            assert RESEARCH_PRODUCERS[name].supersedes is None
            assert name in NO_HISTORY_IMPORT

    def test_each_rationale_names_a_concrete_reason(self):
        """A rationale that says only "not applicable" is an absence wearing a
        declaration's clothes."""
        for name, reason in NO_HISTORY_IMPORT.items():
            assert len(reason) > 120, f"{name}'s rationale is too thin to act on"


class TestTheGuardActuallyBinds:
    def test_an_undeclared_arm_is_refused(self, monkeypatch):
        """A new arm with neither an inheritance nor a stated zero must not
        import — the head-start question answered by nobody."""
        ghost = dataclasses.replace(
            RESEARCH_PRODUCERS["tech_score_20"], name="ghost_arm", supersedes=None
        )
        monkeypatch.setitem(RESEARCH_PRODUCERS, "ghost_arm", ghost)
        with pytest.raises(ValueError, match="ghost_arm"):
            _assert_history_provenance_declared()

    def test_a_rationale_that_outlived_its_arm_is_refused(self, monkeypatch):
        """Stale rationales read as current decisions. A register whose
        explanations no longer match its arms explains the wrong system."""
        monkeypatch.setitem(NO_HISTORY_IMPORT, "arm_that_never_existed", "x" * 200)
        with pytest.raises(ValueError, match="arm_that_never_existed"):
            _assert_history_provenance_declared()

    def test_declaring_both_is_refused(self, monkeypatch):
        """An arm that inherits AND claims no history is a register that
        contradicts itself; the guard must not let either half stand."""
        spec = dataclasses.replace(
            RESEARCH_PRODUCERS["attractiveness_60"], supersedes="thinktank_coverage"
        )
        monkeypatch.setitem(RESEARCH_PRODUCERS, "attractiveness_60", spec)
        with pytest.raises(ValueError, match="attractiveness_60"):
            _assert_history_provenance_declared()


class TestStartingHistoryIsPinned:
    """Deliverable 4 of I11422: a future change cannot silently grant or remove
    a head start."""

    EXPECTED_INHERITED_SOURCE: dict[str, str | None] = {
        "attractiveness_60": None,
        "attractiveness_20": None,
        "tech_score_20": None,
        "predictor_from_60": None,
        "thinktank_20": "thinktank_coverage",
    }

    def test_the_inherited_source_of_every_arm_is_exactly_this(self):
        got = {
            s.name: s.supersedes
            for s in RESEARCH_PRODUCERS.values()
            if s.slot == RESEARCH_SLOT and s.kind != "retired"
        }
        assert got == self.EXPECTED_INHERITED_SOURCE
