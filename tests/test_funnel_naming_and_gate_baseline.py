"""The cut named "scanner" is not the funnel (alpha-engine-config-I7578).

Brian ruling 2026-08-17: only the funnel-60 advances to the top-20 and to Think
Tank. The funnel INVARIANT already enforced that (I6630). What was unguarded is
the NAMING — the 60-wide cut that feeds nothing was the one whose name said
"scanner", and it was read backwards three times, each caught by hand.

Measured 2026-08-17 on the live artifact: the two 60-wide cuts overlap on 12 of
60, and the champion 20 overlaps the gate cut on 3 of 20. Near-disjoint sets,
and only one of them advances.

RED on origin/main at aff64bd8: every test here except the alias ones fails,
because neither the new cut name, the `feed_cut` field, the `funnel` block nor
the routing guard exists.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import (  # noqa: E402
    FEED_CUT_NAME,
    GATE_BASELINE_CUT,
    GATE_LEGACY_CUT,
    PREDICTOR_UNIVERSE_CUT,
    UniverseMembershipError,
    assert_gate_cut_feeds_nothing_live,
    build_universe_membership,
)

GATE = [f"G{i:03d}" for i in range(60)]
ATTRACTIVENESS = {f"T{i:03d}": float(1000 - i) for i in range(200)}


@pytest.fixture
def membership():
    return build_universe_membership(
        "2026-08-18", scanner_tickers=GATE, attractiveness=ATTRACTIVENESS,
    )


# ── The rename ───────────────────────────────────────────────────────────────


def test_gate_cut_name_contains_baseline():
    """Reading this name as the funnel must require ignoring a word."""
    assert "baseline" in GATE_BASELINE_CUT


def test_gate_cut_is_emitted_under_the_new_name(membership):
    cut = membership["cuts"][GATE_BASELINE_CUT]
    assert cut["basis"] == "scanner_gate"
    assert cut["size"] == 60
    assert cut["feeds"] == []
    assert "feeds nothing live" in cut["role"].lower()


def test_legacy_key_still_emitted_and_marked_deprecated(membership):
    """A consumer pinned on the old name must break loudly at a stated date,
    never silently read a missing key as an empty cut. Known live reader:
    crucible-dashboard/loaders/universe_churn.py."""
    legacy = membership["cuts"][GATE_LEGACY_CUT]
    assert legacy["deprecated_alias_for"] == GATE_BASELINE_CUT
    assert legacy["removal_tracked_by"]


def test_alias_and_canonical_carry_identical_membership(membership):
    """An alias that could drift from its target is a second source of truth."""
    a = membership["cuts"][GATE_BASELINE_CUT]
    b = membership["cuts"][GATE_LEGACY_CUT]
    assert a["tickers"] == b["tickers"]
    assert a["size"] == b["size"]
    assert a["basis"] == b["basis"]


# ── The funnel, stated in the artifact ───────────────────────────────────────


def test_feed_cut_is_named_at_the_top_level(membership):
    assert membership["feed_cut"] == FEED_CUT_NAME


def test_funnel_block_states_the_whole_chain(membership):
    f = membership["funnel"]
    assert f["population"] == len(ATTRACTIVENESS)
    assert f["feed_cut"] == {"name": FEED_CUT_NAME, "size": 60}
    assert f["champion_cut"] == {"name": PREDICTOR_UNIVERSE_CUT, "size": 20}


def test_funnel_names_all_three_live_consumers(membership):
    """Brian's ruling in one place: the top-20 goes to the predictor, and the
    SAME 60 is both the RAG scope and Think Tank's window."""
    advances = membership["funnel"]["advances_to"]
    assert advances["predictor_universe"] == PREDICTOR_UNIVERSE_CUT
    assert advances["rag_corpus_scope"] == FEED_CUT_NAME
    assert advances["thinktank_coverage_window"] == FEED_CUT_NAME


def test_funnel_says_which_cuts_feed_nothing(membership):
    assert set(membership["funnel"]["feeds_nothing_live"]) == {
        GATE_BASELINE_CUT, GATE_LEGACY_CUT,
    }


def test_every_name_in_the_funnel_block_is_a_real_cut(membership):
    """A funnel block naming a cut the artifact does not carry is a map to a
    place that does not exist."""
    cuts = membership["cuts"]
    f = membership["funnel"]
    for name in (f["feed_cut"]["name"], f["champion_cut"]["name"],
                 *f["advances_to"].values(), *f["feeds_nothing_live"]):
        assert name in cuts, name


# ── The guard: the gate cut may never be wired to a live consumer ────────────


@pytest.mark.parametrize("field", ["predictor_universe_cut", "feed_cut"])
@pytest.mark.parametrize("gate_name", [GATE_BASELINE_CUT, GATE_LEGACY_CUT])
def test_routing_a_live_consumer_at_the_gate_cut_raises(membership, field, gate_name):
    """This is the check whose absence produced I6630 — the RAG corpus scoped
    to the gate cut against a champion drawn from the rank, overlapping 2 of
    20, with nothing asserting the two were related."""
    broken = {**membership, field: gate_name}
    with pytest.raises(UniverseMembershipError, match="feeds nothing live"):
        assert_gate_cut_feeds_nothing_live(broken, "2026-08-18")


def test_routing_a_funnel_consumer_at_the_gate_cut_raises(membership):
    broken = {
        **membership,
        "funnel": {
            **membership["funnel"],
            "advances_to": {**membership["funnel"]["advances_to"],
                            "rag_corpus_scope": GATE_BASELINE_CUT},
        },
    }
    with pytest.raises(UniverseMembershipError, match="feeds nothing live"):
        assert_gate_cut_feeds_nothing_live(broken, "2026-08-18")


def test_the_real_artifact_passes_the_guard(membership):
    assert_gate_cut_feeds_nothing_live(membership, "2026-08-18")


# ── The property the ruling is actually about ────────────────────────────────


def test_only_the_feed_cut_advances(membership):
    """The champion is the head of the feed cut, and the gate cut is not
    involved in either."""
    cuts = membership["cuts"]
    champion = set(cuts[PREDICTOR_UNIVERSE_CUT]["tickers"])
    feed = set(cuts[FEED_CUT_NAME]["tickers"])
    gate = set(cuts[GATE_BASELINE_CUT]["tickers"])
    assert champion <= feed
    assert not (champion & gate)
