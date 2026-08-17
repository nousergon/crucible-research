"""The zero-momentum-weight arm (alpha-engine-config-I7573).

Sibling to the mom121 arm. The two vary DIFFERENT things about the same
ranking and must never be collapsed into one:

    mom121   momentum keeps its 1/6 weight; components move to 12-1  -> HORIZON
    momzero  components unchanged; momentum's weight goes to 0       -> EXPOSURE

If mom121 wins, momentum belongs in the score and was measured over the wrong
window. If momzero wins, it does not belong at all. Those are different
beliefs about the strategy, and an arm that changed both could not tell them
apart.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import (  # noqa: E402
    CHALLENGER_CUT_PREFIX,
    MOMZERO_CUT_PREFIX,
    MOMZERO_PILLAR_WEIGHTS,
    PILLAR_ORDER_FOR_WEIGHTS,
    UniverseMembershipError,
    assert_cut_invariants,
    build_universe_membership,
)

# ── The weight vector ────────────────────────────────────────────────────────


def test_momentum_weight_is_zero_not_merely_reduced():
    """Zero is the point. A partial weight produces a ranking that is neither
    the champion's nor a clean counterfactual, so a small result cannot be
    attributed to anything."""
    assert MOMZERO_PILLAR_WEIGHTS["momentum"] == 0.0


def test_remaining_pillars_split_the_full_weight_evenly():
    """The vector is declared already-normalized, and evenly.

    NOT because an unnormalized vector would break the ranking — it would
    not. ``nousergon_lib.quant.attractiveness.normalize_pillar_weights``
    rescales whatever it is given, and a vector of five 1/6 values (summing
    to 5/6) was MEASURED to produce a top-20 identical to this one on the
    2026-08-17 profiles. So this is a readability and auditability rule, not
    a correctness one: a reader comparing this arm to the champion's equal
    1/6 weights should be able to see the redistribution directly instead of
    having to know that a normalizer will happen later.

    The EVEN split is the substantive part. Momentum's 1/6 has to go
    somewhere, and giving it to the remaining five equally is the only choice
    that changes exactly one thing. Handing it to (say) value would make this
    a two-variable arm — less momentum AND more value — and neither could be
    attributed.
    """
    assert sum(MOMZERO_PILLAR_WEIGHTS.values()) == pytest.approx(1.0)
    non_momentum = [v for k, v in MOMZERO_PILLAR_WEIGHTS.items() if k != "momentum"]
    assert len(non_momentum) == 5
    assert all(v == pytest.approx(0.2) for v in non_momentum), MOMZERO_PILLAR_WEIGHTS


def test_weight_vector_covers_every_pillar_exactly_once():
    """A missing pillar would be silently dropped by the normalizer, changing
    two things at once and destroying the arm's attribution."""
    assert set(MOMZERO_PILLAR_WEIGHTS) == set(PILLAR_ORDER_FOR_WEIGHTS)


def test_the_two_arms_vary_different_things():
    """Guard against a future edit collapsing them into one experiment."""
    assert MOMZERO_CUT_PREFIX != CHALLENGER_CUT_PREFIX


# ── The cuts ─────────────────────────────────────────────────────────────────


def _attr(n=80):
    return {f"T{i:03d}": float(1000 - i) for i in range(n)}


def _reversed(n=80):
    return {f"T{i:03d}": float(i) for i in range(n)}


def _shifted(n=80):
    # Distinct from both the champion and the mom121 fixture.
    return {f"T{i:03d}": float((i * 37) % n) for i in range(n)}


def test_momzero_cuts_are_emitted_count_matched():
    m = build_universe_membership(
        "2026-08-17",
        [f"T{i:03d}" for i in range(60)],
        _attr(),
        momzero_attractiveness=_reversed(),
    )
    assert m["cuts"][f"{MOMZERO_CUT_PREFIX}20"]["size"] == 20
    assert m["cuts"][f"{MOMZERO_CUT_PREFIX}60"]["size"] == 60
    assert m["cuts"][f"{MOMZERO_CUT_PREFIX}20"]["basis"] == "attractiveness_rank_momzero"


def test_both_challenger_arms_coexist_independently():
    """Each arm's absence is its OWN miss — one missing must not suppress the
    other, or a single failure would silently halve the experiment."""
    m = build_universe_membership(
        "2026-08-17",
        [f"T{i:03d}" for i in range(60)],
        _attr(),
        challenger_attractiveness=_reversed(),
        momzero_attractiveness=_shifted(),
    )
    assert f"{CHALLENGER_CUT_PREFIX}20" in m["cuts"]
    assert f"{MOMZERO_CUT_PREFIX}20" in m["cuts"]

    only_momzero = build_universe_membership(
        "2026-08-17", [f"T{i:03d}" for i in range(60)], _attr(),
        challenger_attractiveness=None, momzero_attractiveness=_shifted(),
    )
    assert f"{CHALLENGER_CUT_PREFIX}20" not in only_momzero["cuts"]
    assert f"{MOMZERO_CUT_PREFIX}20" in only_momzero["cuts"]


def test_momzero_absent_when_profiles_unavailable():
    m = build_universe_membership(
        "2026-08-17", [f"T{i:03d}" for i in range(60)], _attr(), momzero_attractiveness=None
    )
    assert not [k for k in m["cuts"] if k.startswith(MOMZERO_CUT_PREFIX)]


def test_vacuous_momzero_is_a_red_run():
    """The failure mode this arm is most exposed to.

    momzero shares the champion's factor profiles — only the weight vector
    differs — so a weight vector that failed to apply yields a ranking
    IDENTICAL to the champion's while every other assertion passes.
    """
    with pytest.raises(UniverseMembershipError, match="vacuous challenger"):
        build_universe_membership(
            "2026-08-17",
            [f"T{i:03d}" for i in range(60)],
            _attr(),
            momzero_attractiveness=_attr(),
        )


def test_short_momzero_table_is_a_red_run():
    cuts = {
        "attractiveness_top_20": {
            "basis": "attractiveness_rank", "size": 20,
            "tickers": [f"T{i:03d}" for i in range(20)],
        },
        f"{MOMZERO_CUT_PREFIX}20": {
            "basis": "attractiveness_rank_momzero", "size": 11,
            "tickers": [f"X{i:03d}" for i in range(11)],
        },
    }
    with pytest.raises(UniverseMembershipError, match="count-match broken"):
        assert_cut_invariants({"cuts": cuts}, "2026-08-17")
