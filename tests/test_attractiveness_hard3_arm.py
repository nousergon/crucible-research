"""The fundamentals-half arm (alpha-engine-config-I8256).

Third sibling of the weight-vector arms. Each varies ONE thing about the same
ranking, over the same factor profiles, and they must never be collapsed:

    mom121   momentum keeps its 1/6 weight; components move to 12-1  -> HORIZON
    momzero  components unchanged; momentum's weight goes to 0       -> EXPOSURE
    hard3    quality/growth/stewardship go to 0                      -> PROVENANCE

``hard3`` exists because the six pillars split by where their inputs come
from. value / momentum / defensiveness are derived from the price series and
from balance-sheet ratios that carried real cross-sectional spread throughout
2026-08. quality / growth / stewardship come from a vendor fundamentals feed
that was placeholder-saturated until 2026-08-20 (alpha-engine-config-I8255:
``fcf_yield`` held ONE distinct value across 899 names). So for weeks the
champion was a three-pillar ranking by accident, and the question of whether
the vendor half earns its weight has never actually been asked.

This arm asks it on REPAIRED inputs. It is not a reconstruction of the broken
state — see ``test_hard3_is_not_a_replay_of_the_defect``.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import (  # noqa: E402
    ATTRACTIVENESS_FEED_TOP_N,
    CHALLENGER_CUT_PREFIX,
    HARD3_CUT_PREFIX,
    HARD3_PILLAR_WEIGHTS,
    MOMZERO_CUT_PREFIX,
    MOMZERO_PILLAR_WEIGHTS,
    OBSERVE_ONLY_CUTS,
    PILLAR_ORDER_FOR_WEIGHTS,
    PROMOTABLE_CUTS,
    SLOT_ARMS,
    UniverseMembershipError,
    assert_cut_invariants,
    build_universe_membership,
)

HARD3_CUT = f"{HARD3_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"


# ── The weight vector ────────────────────────────────────────────────────────


def test_the_vendor_half_is_zero_not_merely_reduced():
    """Zero is the point, for the same reason it is in MOMZERO_PILLAR_WEIGHTS.

    A partial weight produces a ranking that is neither the champion's nor a
    clean counterfactual, so a small win or loss could not be attributed to
    the half under test.
    """
    for pillar in ("quality", "growth", "stewardship"):
        assert HARD3_PILLAR_WEIGHTS[pillar] == 0.0, pillar


def test_the_surviving_three_split_the_full_weight_evenly():
    """An even split is what makes this a ONE-variable arm.

    Handing the freed weight unevenly — say all of it to value — would make
    this "no fundamentals AND more value", and neither could be attributed.
    """
    assert sum(HARD3_PILLAR_WEIGHTS.values()) == pytest.approx(1.0)
    survivors = [HARD3_PILLAR_WEIGHTS[p] for p in ("value", "momentum", "defensiveness")]
    assert all(v == pytest.approx(1.0 / 3.0) for v in survivors), HARD3_PILLAR_WEIGHTS


def test_weight_vector_covers_every_pillar_exactly_once():
    """A missing pillar is silently dropped by the normalizer, which would
    change two things at once and destroy the arm's attribution."""
    assert set(HARD3_PILLAR_WEIGHTS) == set(PILLAR_ORDER_FOR_WEIGHTS)


def test_the_three_weight_arms_vary_different_things():
    """Guard against a future edit collapsing any two into one experiment."""
    assert len({HARD3_CUT_PREFIX, MOMZERO_CUT_PREFIX, CHALLENGER_CUT_PREFIX}) == 3
    assert HARD3_PILLAR_WEIGHTS != MOMZERO_PILLAR_WEIGHTS


def test_hard3_and_momzero_are_not_the_same_arm_by_arithmetic():
    """Both zero SOME pillars. If they ever zeroed the same set they would be
    one arm wearing two names, and the leaderboard would report a tie between
    an arm and itself while every assertion passed."""
    hard3_zeroed = {p for p, w in HARD3_PILLAR_WEIGHTS.items() if w == 0.0}
    momzero_zeroed = {p for p, w in MOMZERO_PILLAR_WEIGHTS.items() if w == 0.0}
    assert hard3_zeroed != momzero_zeroed
    assert not hard3_zeroed & momzero_zeroed, (
        "the two arms zero DISJOINT halves of the composite — an overlap would "
        "mean one arm is testing part of the other's question"
    )


# ── Registration ─────────────────────────────────────────────────────────────


def test_hard3_is_observe_only_and_cannot_hold_the_feed():
    """Registration is the gate (champion-challenger-policy.md §3): an arm in
    OBSERVE_ONLY_CUTS is scored by construction and can never serve."""
    assert HARD3_CUT in OBSERVE_ONLY_CUTS
    assert HARD3_CUT not in PROMOTABLE_CUTS
    assert HARD3_CUT in SLOT_ARMS


def test_hard3_is_emitted_at_the_champion_width_only():
    """Count-matched at 60 (§4), and deliberately NOT also at 20: a second
    width doubles the arm count without asking a second question."""
    m = build_universe_membership(
        "2026-08-24",
        [f"T{i:03d}" for i in range(60)],
        _attr(),
        hard3_attractiveness=_reversed(),
    )
    hard3_keys = [k for k in m["cuts"] if k.startswith(HARD3_CUT_PREFIX)]
    assert hard3_keys == [HARD3_CUT]
    assert m["cuts"][HARD3_CUT]["size"] == ATTRACTIVENESS_FEED_TOP_N
    assert m["cuts"][HARD3_CUT]["basis"] == "attractiveness_rank_hard3"


# ── The cuts ─────────────────────────────────────────────────────────────────


def _attr(n=80):
    return {f"T{i:03d}": float(1000 - i) for i in range(n)}


def _reversed(n=80):
    return {f"T{i:03d}": float(i) for i in range(n)}


def _shifted(n=80):
    return {f"T{i:03d}": float((i * 37) % n) for i in range(n)}


def test_every_weight_arm_misses_independently():
    """One arm's absence must never suppress another, or a single failure
    silently shrinks the experiment without anything saying so."""
    m = build_universe_membership(
        "2026-08-24",
        [f"T{i:03d}" for i in range(60)],
        _attr(),
        challenger_attractiveness=_reversed(),
        momzero_attractiveness=_shifted(),
        hard3_attractiveness={f"T{i:03d}": float((i * 53) % 80) for i in range(80)},
    )
    assert f"{CHALLENGER_CUT_PREFIX}20" in m["cuts"]
    assert f"{MOMZERO_CUT_PREFIX}20" in m["cuts"]
    assert HARD3_CUT in m["cuts"]

    only_hard3 = build_universe_membership(
        "2026-08-24", [f"T{i:03d}" for i in range(60)], _attr(),
        challenger_attractiveness=None, momzero_attractiveness=None,
        hard3_attractiveness=_shifted(),
    )
    assert HARD3_CUT in only_hard3["cuts"]
    assert not [k for k in only_hard3["cuts"] if k.startswith(MOMZERO_CUT_PREFIX)]


def test_hard3_absent_when_profiles_unavailable():
    m = build_universe_membership(
        "2026-08-24", [f"T{i:03d}" for i in range(60)], _attr(), hard3_attractiveness=None
    )
    assert not [k for k in m["cuts"] if k.startswith(HARD3_CUT_PREFIX)]


def test_vacuous_hard3_is_a_red_run():
    """The failure mode this arm is most exposed to: it shares the champion's
    factor profiles and differs only by the weight vector, so a vector that
    failed to apply yields an IDENTICAL ranking while everything else passes."""
    with pytest.raises(UniverseMembershipError, match="vacuous challenger"):
        build_universe_membership(
            "2026-08-24",
            [f"T{i:03d}" for i in range(60)],
            _attr(),
            hard3_attractiveness=_attr(),
        )


def test_short_hard3_table_is_a_red_run():
    cuts = {
        f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}": {
            "basis": "attractiveness_rank", "size": ATTRACTIVENESS_FEED_TOP_N,
            "tickers": [f"T{i:03d}" for i in range(ATTRACTIVENESS_FEED_TOP_N)],
        },
        HARD3_CUT: {
            "basis": "attractiveness_rank_hard3", "size": 11,
            "tickers": [f"X{i:03d}" for i in range(11)],
        },
    }
    with pytest.raises(UniverseMembershipError, match="count-match broken"):
        assert_cut_invariants({"cuts": cuts}, "2026-08-24")


# ── Near-identity is recorded, not raised (champion-challenger-policy.md §4) ──


def test_champion_overlap_is_recorded_on_every_weight_arm():
    """§4 requires near-identity to be VISIBLE. An equality guard reported 8
    collision dates across a window in which every date was a clone, because
    two call sites reading the factor store separately disagreed about 2 of 60
    names. The fraction is the instrument; the raise is not."""
    m = build_universe_membership(
        "2026-08-24",
        [f"T{i:03d}" for i in range(60)],
        _attr(),
        momzero_attractiveness=_shifted(),
        hard3_attractiveness=_reversed(),
    )
    assert 0.0 <= m["cuts"][HARD3_CUT]["champion_overlap"] <= 1.0
    assert 0.0 <= m["cuts"][f"{MOMZERO_CUT_PREFIX}60"]["champion_overlap"] <= 1.0


def test_overlap_divides_by_the_wider_set_not_the_intersection():
    """A narrow arm fully contained in a wide one is a DIFFERENT (narrower)
    arm, not a perfect clone. Dividing by the intersection would score it as
    1.0 and hide the only thing that distinguishes it."""
    contained = [f"T{i:03d}" for i in range(20)]
    cuts = {
        f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}": {
            "basis": "attractiveness_rank", "size": ATTRACTIVENESS_FEED_TOP_N,
            "tickers": [f"T{i:03d}" for i in range(ATTRACTIVENESS_FEED_TOP_N)],
        },
        HARD3_CUT: {
            "basis": "attractiveness_rank_hard3",
            "size": ATTRACTIVENESS_FEED_TOP_N,
            "tickers": contained + [f"Z{i:03d}" for i in range(ATTRACTIVENESS_FEED_TOP_N - 20)],
        },
    }
    assert_cut_invariants({"cuts": cuts}, "2026-08-24")
    assert cuts[HARD3_CUT]["champion_overlap"] == pytest.approx(20 / ATTRACTIVENESS_FEED_TOP_N, abs=1e-4)


def test_a_high_overlap_does_not_red_the_run():
    """Membership is load-bearing for the predictor's universe. An arm may
    legitimately overlap the champion heavily in a given week; only arithmetic
    identity — which cannot happen unless the definition failed to apply —
    stops the run."""
    near = [f"T{i:03d}" for i in range(ATTRACTIVENESS_FEED_TOP_N - 1)] + ["ZZZ"]
    cuts = {
        f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}": {
            "basis": "attractiveness_rank", "size": ATTRACTIVENESS_FEED_TOP_N,
            "tickers": [f"T{i:03d}" for i in range(ATTRACTIVENESS_FEED_TOP_N)],
        },
        HARD3_CUT: {
            "basis": "attractiveness_rank_hard3",
            "size": ATTRACTIVENESS_FEED_TOP_N, "tickers": near,
        },
    }
    assert_cut_invariants({"cuts": cuts}, "2026-08-24")  # must not raise
    overlap = cuts[HARD3_CUT]["champion_overlap"]
    assert overlap > 0.95, "the fixture is meant to be near-identical"


# ── Applicability (§4: structural identity is refused, not scored) ────────────


def test_hard3_inapplicable_when_the_champion_already_zeroes_the_vendor_half():
    """Then this arm IS the champion by arithmetic. It would emit an identical
    cut, trip the vacuity guard, and turn a legal configuration into a RED
    Scanner run on an artifact the predictor depends on. §4 names this case:
    refused upstream as `inapplicable`, never scored as a tie."""
    import scoring.universe_membership as um

    original = um.champion_pillar_weights
    try:
        um.champion_pillar_weights = lambda *a, **k: {
            "quality": 0.0, "value": 0.25, "momentum": 0.25,
            "growth": 0.0, "stewardship": 0.0, "defensiveness": 0.5,
        }
        assert um.hard3_arm_applicable() is False
    finally:
        um.champion_pillar_weights = original


def test_hard3_applicable_when_any_vendor_pillar_carries_weight():
    import scoring.universe_membership as um

    original = um.champion_pillar_weights
    try:
        um.champion_pillar_weights = lambda *a, **k: dict.fromkeys(PILLAR_ORDER_FOR_WEIGHTS, 1.0)
        assert um.hard3_arm_applicable() is True
        um.champion_pillar_weights = lambda *a, **k: {
            "quality": 0.0, "value": 1.0, "momentum": 1.0,
            "growth": 0.5, "stewardship": 0.0, "defensiveness": 1.0,
        }
        assert um.hard3_arm_applicable() is True, "growth alone is enough"
    finally:
        um.champion_pillar_weights = original


def test_unreadable_weights_suppress_the_arm_rather_than_arming_it():
    """An unreadable config must not silently arm a weight-vector arm against
    a champion nobody has read. All-zero is the conservative direction and it
    only ever suppresses."""
    import scoring.universe_membership as um

    original = um.champion_pillar_weights
    try:
        um.champion_pillar_weights = lambda *a, **k: {}
        assert um.hard3_arm_applicable() is False
    finally:
        um.champion_pillar_weights = original


def test_hard3_is_not_a_replay_of_the_defect():
    """The arm must never be described or built as a reconstruction of the
    pre-2026-08-20 ranking.

    That ranking ran on placeholder fundamentals (I8255) and is not
    point-in-time reproducible. Backfilling this arm onto those dates would
    enter the cohort as a near-clone of the champion, poisoning the exact
    comparison the arm exists to make. This test pins the intent in the
    docstring so a future reader cannot mistake the two.
    """
    import scoring.universe_membership as um

    doc = um.HARD3_PILLAR_WEIGHTS.__doc__ or ""
    module_src = open(um.__file__).read()
    assert "I8256" in module_src
    assert "not point-in-time" in module_src.lower() or "not a reconstruction" in module_src.lower()
    assert doc, "the weight vector must carry its rationale"
