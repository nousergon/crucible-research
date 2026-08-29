"""Regression tests for alpha-engine-config-I9277 / -I9279 / -I9282.

Three defects, one arc (Brian's ruling 2026-08-29: "for the research arm, we
should make all arms promote eligible, including think tank"):

* I9282 — the hysteresis EXIT half shipped 2026-08-10 without its RE-ENTRY
  half, so a de-covered name could never become covered again no matter how
  many theses were written for it. Deadlock: `coverage_complete` never True,
  no shadow signal, a dead arm scoring on a frozen cohort.
* I9277 — promotion eligibility lived in a hand-maintained tuple in ANOTHER
  repo, so an arm could be scored-but-ineligible with no reason recorded.
* I9279 — arms were compared on their own cohorts, not the shared one.
"""

from __future__ import annotations

import pytest

from producers.registry import (
    RESEARCH_PRODUCERS,
    ineligible_producers,
    promotion_eligible_producers,
)
from thinktank.ledger import record_thesis_write
from thinktank.schemas import CoverageLedger, LedgerEntry


# ── I9282: the missing re-cover half ────────────────────────────────────────


def _dropped_ledger() -> CoverageLedger:
    """A ledger in the exact live 2026-08-29 state: HALO de-covered on
    2026-08-10 past exit_rank, thesis history intact."""
    return CoverageLedger(
        entries={
            "HALO": LedgerEntry(
                ticker="HALO",
                covered_since="2026-07-02",
                thesis_version=10,
                thesis_updated_on="2026-08-09",
                covered=False,
                dropped_on="2026-08-10",
                attractiveness_rank_at_drop=207,
            )
        }
    )


def test_writing_a_thesis_recovers_a_dropped_name():
    """The defect: this is what ran daily for HALO/FTNT/WEX/HST and left
    `covered` False every time, so each name was re-selected as fresh intake
    the next day and the budget never reached the genuine backlog."""
    ledger = _dropped_ledger()
    record_thesis_write(
        ledger, ticker="HALO", trading_day="2026-08-28", thesis_version=11,
    )
    entry = ledger.entries["HALO"]
    assert entry.thesis_version == 11
    assert entry.thesis_updated_on == "2026-08-28"
    # Fails before the fix — `covered` stayed False forever.
    assert entry.covered is True, (
        "writing a thesis must re-cover a de-covered name; leaving covered=False "
        "re-queues it as new intake every day (alpha-engine-config-I9282)"
    )


def test_recovering_clears_the_drop_record():
    """A re-covered name asserting `dropped_on` is the ledger stating two
    contradictory things about the same entry."""
    ledger = _dropped_ledger()
    record_thesis_write(
        ledger, ticker="HALO", trading_day="2026-08-28", thesis_version=11,
    )
    entry = ledger.entries["HALO"]
    assert entry.dropped_on is None
    assert entry.attractiveness_rank_at_drop is None


def test_recovered_name_leaves_the_uncovered_set():
    """The end-to-end property the deadlock broke: a re-underwritten name must
    stop counting toward `uncovered_count`, or `coverage_complete` can never
    become True again."""
    ledger = _dropped_ledger()
    assert "HALO" not in ledger.covered()
    record_thesis_write(
        ledger, ticker="HALO", trading_day="2026-08-28", thesis_version=11,
    )
    assert "HALO" in ledger.covered()


def test_a_brand_new_name_is_still_covered_on_first_thesis():
    """Guard against fixing the update branch by breaking the create branch."""
    ledger = CoverageLedger(entries={})
    record_thesis_write(
        ledger, ticker="NEWCO", trading_day="2026-08-28", thesis_version=1,
    )
    assert ledger.entries["NEWCO"].covered is True


# ── I9277: eligibility is a recorded property, not an absence ───────────────


def test_every_live_arm_is_promotion_eligible():
    """Brian's ruling 2026-08-29, verbatim: 'for the research arm, we should
    make all arms promote eligible, including think tank.'"""
    eligible = {p.name for p in promotion_eligible_producers()}
    for name in (
        "no_agent_quant",
        "single_agent_quant",
        "thinktank_coverage",
        "scanner_predictor_direct",
        "scanner_top20_predictor",
    ):
        assert name in eligible, f"{name} must be promotion-eligible (I9277)"


def test_the_two_arms_with_evidence_are_eligible():
    """The specific defect: `no_agent_quant` and `single_agent_quant` were the
    ONLY arms with `confidence: ok` on the 2026-08-28 board, and were the only
    two that could not win — because neither was typed into crucible
    -backtester's VALID_CHAMPIONS tuple."""
    eligible = {p.name for p in promotion_eligible_producers()}
    assert {"no_agent_quant", "single_agent_quant"} <= eligible


def test_every_registered_arm_has_a_recorded_eligibility_verdict():
    """No arm may be silently absent from the promotion set: it is either
    eligible, or ineligible WITH a reason. There is no third state."""
    eligible = {p.name for p in promotion_eligible_producers()}
    ineligible = ineligible_producers()
    assert eligible.isdisjoint(ineligible)
    assert eligible | set(ineligible) == set(RESEARCH_PRODUCERS)


def test_an_exclusion_always_carries_a_reason():
    for name, reason in ineligible_producers().items():
        assert reason and reason != "no reason recorded", (
            f"{name} is excluded from promotion with no stated reason — an "
            "exclusion a reader cannot review is the I9277 defect"
        )


def test_retired_arms_are_ineligible():
    """champion-challenger-policy.md §3/§6: retired arms are scored as
    historical evidence but can never be promoted."""
    ineligible = ineligible_producers()
    for spec in RESEARCH_PRODUCERS.values():
        if spec.kind == "retired":
            assert spec.name in ineligible


def test_eligibility_flag_and_reason_cannot_contradict():
    """The import-time guard must reject both incoherent shapes."""
    from producers.registry import ProducerSpec, _assert_eligibility_coherent

    original = dict(RESEARCH_PRODUCERS)
    try:
        RESEARCH_PRODUCERS["bad"] = ProducerSpec(
            name="bad", kind="challenger", version="v1", description="x",
            promotion_eligible=False, ineligible_reason=None,
        )
        with pytest.raises(ValueError, match="no ineligible_reason"):
            _assert_eligibility_coherent()

        RESEARCH_PRODUCERS["bad"] = ProducerSpec(
            name="bad", kind="challenger", version="v1", description="x",
            promotion_eligible=True, ineligible_reason="but why",
        )
        with pytest.raises(ValueError, match="contradict"):
            _assert_eligibility_coherent()
    finally:
        RESEARCH_PRODUCERS.clear()
        RESEARCH_PRODUCERS.update(original)


def test_filling_champion_arms_covers_every_eligible_arm():
    """As a hand-maintained literal this tuple went stale the moment
    `scanner_top20_predictor` became a promotion arm — a promotion onto it
    would have hit the executor coherence assertion with the arm in neither
    the filling nor the noop set."""
    from producers.registry import FILLING_CHAMPION_ARMS

    for spec in promotion_eligible_producers():
        assert spec.name in FILLING_CHAMPION_ARMS


# ── I9279: one cohort, reported alongside the metric ────────────────────────


def _hist(name, dates, *, kind="challenger", eligible=True, reason=None):
    from scoring.leaderboard_scoring import SpecDay, SpecHistory

    return SpecHistory(
        name=name,
        kind=kind,
        by_date={d: SpecDay(ranked=["AAA", "BBB"]) for d in dates},
        promotion_eligible=eligible,
        ineligible_reason=reason,
    )


def _realized(dates):
    return {d: {"AAA": 0.01, "BBB": -0.01, "SPY": 0.0} for d in dates}


def test_intersection_is_the_shared_cohort_not_the_union():
    from scoring.leaderboard_scoring import cohort_intersection

    champ = _hist("champ", ["d1", "d2", "d3"], kind="champion")
    other = _hist("other", ["d2", "d3", "d4"])
    got = cohort_intersection(champ, [other], _realized(["d1", "d2", "d3", "d4"]))
    assert got == ["d2", "d3"]


def test_an_ineligible_arm_does_not_shrink_the_window():
    """A retired arm is historical evidence that can never win, so letting its
    sparse history constrain the window every LIVE arm is judged on would make
    the measurement pay rent for a row that cannot use it."""
    from scoring.leaderboard_scoring import cohort_intersection

    champ = _hist("champ", ["d1", "d2", "d3"], kind="champion")
    live = _hist("live", ["d1", "d2", "d3"])
    retired = _hist(
        "retired", ["d3"], kind="retired", eligible=False, reason="retired",
    )
    got = cohort_intersection(champ, [live, retired], _realized(["d1", "d2", "d3"]))
    assert got == ["d1", "d2", "d3"]


def test_a_dead_arm_does_not_freeze_the_window_for_live_arms():
    """The live 2026-08-29 condition: `thinktank_coverage` stopped writing
    shadows on 2026-08-14, and its 4 stale dates would have cut the shared
    cohort from 18 dates to 4 for every other arm — collapsing every
    comparison to a no-contest because ONE arm had deadlocked."""
    from scoring.leaderboard_scoring import cohort_intersection

    recent = [f"d{i:02d}" for i in range(1, 13)]
    champ = _hist("champ", recent, kind="champion")
    live = _hist("live", recent)
    dead = _hist("dead", ["d01", "d02"])  # stopped long before the cohort end
    got = cohort_intersection(champ, [live, dead], _realized(recent))
    assert got == recent, "a stale arm must not constrain the shared window"


def test_the_metric_the_gate_ranks_on_is_computed_on_the_intersection():
    """champion-challenger-policy.md §4. Before this, a 2-date arm was ranked
    against a 6-date arm as though the two numbers described the same weeks."""
    from scoring.leaderboard_scoring import score_leaderboard

    champ = _hist("champ", ["d1", "d2", "d3"], kind="champion")
    other = _hist("other", ["d2", "d3", "d4"])
    board = score_leaderboard(
        champ, [other], _realized(["d1", "d2", "d3", "d4"]), top_n=2,
    )
    assert board["cohort_intersection"] == ["d2", "d3"]
    assert board["n_dates_intersection"] == 2
    for row in board["specs"]:
        assert "topn_alpha_vs_benchmark_intersection" in row
        assert row["n_dates_in_intersection"] == 2, row["name"]


def test_every_row_reports_its_own_cohort_dates():
    """Two arms reporting n_dates_scored: 6 over DISJOINT weeks were
    indistinguishable on the artifact before this."""
    from scoring.leaderboard_scoring import score_leaderboard

    champ = _hist("champ", ["d1", "d2"], kind="champion")
    other = _hist("other", ["d2", "d3"])
    board = score_leaderboard(champ, [other], _realized(["d1", "d2", "d3"]), top_n=2)
    rows = {r["name"]: r for r in board["specs"]}
    assert rows["champ"]["dates_scored"] == ["d1", "d2"]
    assert rows["other"]["dates_scored"] == ["d2", "d3"]


def test_eligibility_reaches_the_artifact_rows():
    """The projection that lets crucible-backtester drop its own arm list."""
    from scoring.leaderboard_scoring import score_leaderboard

    champ = _hist("champ", ["d1"], kind="champion")
    retired = _hist(
        "old", ["d1"], kind="retired", eligible=False, reason="retired 2026-07-12",
    )
    board = score_leaderboard(champ, [retired], _realized(["d1"]), top_n=2)
    rows = {r["name"]: r for r in board["specs"]}
    assert rows["champ"]["promotion_eligible"] is True
    assert rows["old"]["promotion_eligible"] is False
    assert rows["old"]["ineligible_reason"] == "retired 2026-07-12"
