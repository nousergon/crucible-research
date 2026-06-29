"""Tests for adaptive slot allocation (config#926).

compute_team_slots gains an optional team_accuracy input that nudges each team's
eligible-pick count ±1 by historical accuracy, gated on a min observation count
and degrading gracefully (byte-identical to static) when data is absent.
"""

import pytest

from agents.sector_teams.team_config import (
    ADAPTIVE_SLOT_MIN_OBS,
    _accuracy_adjustment,
    compute_team_slots,
    ALL_TEAM_IDS,
)


def _ratings_all_market_weight():
    # every sector market_weight → base only, no sector adjustment
    return {}


class TestAccuracyAdjustment:
    def test_none_or_empty(self):
        assert _accuracy_adjustment(None) == 0
        assert _accuracy_adjustment({}) == 0

    def test_under_sampled_no_nudge(self):
        assert _accuracy_adjustment(
            {"accuracy": 0.9, "n_obs": ADAPTIVE_SLOT_MIN_OBS - 1}
        ) == 0

    def test_top_percentile_plus_one(self):
        assert _accuracy_adjustment({"accuracy": 0.70, "n_obs": 20}) == 1

    def test_bottom_percentile_minus_one(self):
        assert _accuracy_adjustment({"accuracy": 0.30, "n_obs": 20}) == -1

    def test_mid_percentile_zero(self):
        assert _accuracy_adjustment({"accuracy": 0.50, "n_obs": 20}) == 0

    def test_missing_accuracy_value(self):
        assert _accuracy_adjustment({"n_obs": 50}) == 0

    def test_non_numeric_accuracy(self):
        assert _accuracy_adjustment({"accuracy": "high", "n_obs": 50}) == 0


class TestComputeTeamSlotsGracefulDegrade:
    def test_no_accuracy_equals_static(self):
        ratings = _ratings_all_market_weight()
        static = compute_team_slots(6, ratings)
        with_none = compute_team_slots(6, ratings, team_accuracy=None)
        with_empty = compute_team_slots(6, ratings, team_accuracy={})
        assert static == with_none == with_empty

    def test_all_teams_present(self):
        alloc = compute_team_slots(6, _ratings_all_market_weight())
        assert set(alloc) == set(ALL_TEAM_IDS)


class TestComputeTeamSlotsAdaptive:
    def test_high_accuracy_team_gets_more(self):
        ratings = _ratings_all_market_weight()
        base = compute_team_slots(6, ratings)
        team = ALL_TEAM_IDS[0]
        adaptive = compute_team_slots(
            6, ratings, team_accuracy={team: {"accuracy": 0.75, "n_obs": 20}}
        )
        assert adaptive[team] == base[team] + 1
        # other teams unchanged
        for t in ALL_TEAM_IDS[1:]:
            assert adaptive[t] == base[t]

    def test_low_accuracy_team_gets_fewer(self):
        ratings = _ratings_all_market_weight()
        base = compute_team_slots(6, ratings)
        team = ALL_TEAM_IDS[0]
        adaptive = compute_team_slots(
            6, ratings, team_accuracy={team: {"accuracy": 0.20, "n_obs": 20}}
        )
        assert adaptive[team] == base[team] - 1

    def test_never_negative(self):
        # 1 open slot → base 1; a poor team would go to 0, never below.
        ratings = _ratings_all_market_weight()
        team = ALL_TEAM_IDS[0]
        adaptive = compute_team_slots(
            1, ratings, team_accuracy={team: {"accuracy": 0.10, "n_obs": 50}}
        )
        assert adaptive[team] >= 0

    def test_under_sampled_team_not_nudged(self):
        ratings = _ratings_all_market_weight()
        base = compute_team_slots(6, ratings)
        team = ALL_TEAM_IDS[0]
        adaptive = compute_team_slots(
            6, ratings,
            team_accuracy={team: {"accuracy": 0.95, "n_obs": 2}},
        )
        assert adaptive[team] == base[team]
