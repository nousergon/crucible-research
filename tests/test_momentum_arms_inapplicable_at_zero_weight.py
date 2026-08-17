"""Both momentum arms are INAPPLICABLE — not failed, not tied — once the live
champion weights momentum at 0 (Brian ruling 2026-08-17).

The §4 vacuity guard in ``assert_cut_invariants`` raises when a challenger cut
resolves to the champion's own names, on the premise that identity means the
arm's override silently did not apply. That premise holds only while momentum
carries weight in the champion.

When the champion weights momentum at 0:

    momzero  same profiles, same weights as the champion  -> identical
    mom121   a re-composed momentum pillar times 0        -> identical

so BOTH arms emit champion membership and BOTH trip the guard. Membership is
load-bearing for the predictor's universe, so that turns a correct
configuration into a red Scanner run. These tests pin the applicability test
that suppresses the arms upstream of the guard.

RED on origin/main at b30e9b04: every test below except the last two fails,
because the arms are computed unconditionally.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import universe_membership as um  # noqa: E402

EQUAL = {p: 1 / 6 for p in um.PILLAR_ORDER_FOR_WEIGHTS}
MOM_ZERO = dict(um.MOMZERO_PILLAR_WEIGHTS)


@pytest.fixture
def weights(monkeypatch):
    """Patch the champion weight chokepoint the applicability test reads."""

    def _set(vector):
        import scoring.universe_board as board

        monkeypatch.setattr(
            board, "_load_pillar_weights", lambda *a, **k: dict(vector)
        )

    return _set


# ── The applicability test itself ────────────────────────────────────────────


def test_arms_applicable_when_momentum_carries_weight(weights):
    weights(EQUAL)
    assert um.champion_momentum_weight() == pytest.approx(1 / 6)
    assert um.momentum_arms_applicable() is True


def test_arms_inapplicable_when_champion_zeroes_momentum(weights):
    weights(MOM_ZERO)
    assert um.champion_momentum_weight() == 0.0
    assert um.momentum_arms_applicable() is False


def test_unreadable_weights_suppress_the_arms_rather_than_raising(monkeypatch):
    """Conservative direction: a read failure suppresses observe-only arms.

    The opposite default would let an unreadable config red a Scanner run over
    an arm that is not load-bearing.
    """
    import scoring.universe_board as board

    def _boom(*a, **k):
        raise RuntimeError("s3 unavailable")

    monkeypatch.setattr(board, "_load_pillar_weights", _boom)
    assert um.champion_momentum_weight() == 0.0
    assert um.momentum_arms_applicable() is False


# ── Both producers honour it ─────────────────────────────────────────────────


def test_momzero_producer_returns_none_at_zero_weight(weights):
    weights(MOM_ZERO)
    assert um.momzero_attractiveness_for_run("2026-08-18") is None


def test_mom121_producer_returns_none_at_zero_weight(weights):
    """Asserted WITHOUT any S3 stub: the applicability test must short-circuit
    before the shadow-profile read, or a network call would be needed to
    discover the arm is inapplicable."""
    weights(MOM_ZERO)
    assert um.challenger_attractiveness_for_run("2026-08-18") is None


# ── The failure this prevents ────────────────────────────────────────────────


def _membership_with(champion, challenger=None, momzero=None):
    return um.build_universe_membership(
        "2026-08-18",
        scanner_tickers=list(champion)[:60] or ["AAA"],
        attractiveness=champion,
        challenger_attractiveness=challenger,
        momzero_attractiveness=momzero,
    )


def test_identical_arm_membership_still_raises_when_momentum_has_weight():
    """The guard must KEEP firing in the case it was written for — an arm whose
    override silently did not apply while momentum still counts."""
    champion = {f"T{i:03d}": float(1000 - i) for i in range(200)}
    with pytest.raises(um.UniverseMembershipError, match="vacuous challenger"):
        _membership_with(champion, momzero=dict(champion))


def test_arms_omitted_rather_than_raising_at_zero_weight(weights):
    """The end-to-end property: with the champion at momentum 0, a Scanner run
    produces a clean artifact carrying no arm cuts, instead of raising."""
    weights(MOM_ZERO)
    champion = {f"T{i:03d}": float(1000 - i) for i in range(200)}
    membership = um.build_universe_membership(
        "2026-08-18",
        scanner_tickers=list(champion)[:60],
        attractiveness=champion,
        challenger_attractiveness=um.challenger_attractiveness_for_run("2026-08-18"),
        momzero_attractiveness=um.momzero_attractiveness_for_run("2026-08-18"),
    )
    cuts = membership["cuts"]
    assert not [k for k in cuts if k.startswith(um.MOMZERO_CUT_PREFIX)]
    assert not [k for k in cuts if k.startswith(um.CHALLENGER_CUT_PREFIX)]
    # The champion funnel is untouched by the suppression.
    assert cuts[um.PREDICTOR_UNIVERSE_CUT]["size"] == 20
    assert cuts[um.FEED_CUT_NAME]["size"] == 60
    assert set(cuts[um.PREDICTOR_UNIVERSE_CUT]["tickers"]) <= set(
        cuts[um.FEED_CUT_NAME]["tickers"]
    )
