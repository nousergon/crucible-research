"""Tests for regime-v3 Stage D' Wire 1 — sector teams allowed to emit
0 picks under a regime-conditional composite-score gate.

Covers:
- ``regime_conditional_min_score`` threshold formula
  (base + max(0, -intensity_z) * scale), with None intensity_z
  degrading gracefully to base
- ``_candidate_composite_score`` (mean of quant + qual when both
  present; quant only otherwise)
- ``_apply_regime_pick_gate`` filtering behavior with the flag on
- No-op when SECTOR_REGIME_PICK_GATE_ENABLED is False (default)
- 0-pick path allowed when no candidate clears the bar
- Pass-through when score field is missing (defensive)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


from agents.sector_teams.peer_review import (
    _apply_regime_pick_gate,
    _candidate_composite_score,
    regime_conditional_min_score,
)


# ---------------------------------------------------------------------------
# regime_conditional_min_score
# ---------------------------------------------------------------------------


class TestRegimeConditionalMinScore:
    def test_none_intensity_returns_base(self):
        assert regime_conditional_min_score(None, base_min_score=50.0, intensity_scale=10.0) == 50.0

    def test_positive_intensity_returns_base_no_premium(self):
        """Bull regime (intensity_z > 0) doesn't raise the bar above
        base — the gate is asymmetric, only adds friction in risk-off
        conditions."""
        assert regime_conditional_min_score(1.5, base_min_score=50.0, intensity_scale=10.0) == 50.0
        assert regime_conditional_min_score(0.0, base_min_score=50.0, intensity_scale=10.0) == 50.0

    def test_negative_intensity_raises_threshold(self):
        """Bear regime (intensity_z < 0) raises the bar linearly with
        the magnitude of risk-off conditions."""
        # intensity_z=-1.0, scale=10.0 → premium=10 → threshold=60
        assert regime_conditional_min_score(-1.0, base_min_score=50.0, intensity_scale=10.0) == 60.0
        # Deeper risk-off raises more
        assert regime_conditional_min_score(-2.5, base_min_score=50.0, intensity_scale=10.0) == 75.0

    def test_uses_config_defaults_when_not_overridden(self):
        """The config-loaded SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE +
        _INTENSITY_SCALE are picked up when overrides aren't supplied.
        Pins that the helper doesn't silently fall back to hard-coded
        constants if the config changes."""
        from config import (
            SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE,
            SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE,
        )
        # intensity_z=-1.0 → threshold = base + scale
        expected = SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE + SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE
        assert regime_conditional_min_score(-1.0) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _candidate_composite_score
# ---------------------------------------------------------------------------


class TestCandidateCompositeScore:
    def test_average_when_both_present(self):
        assert _candidate_composite_score({"quant_score": 70, "qual_score": 80}) == 75.0

    def test_quant_only_when_qual_missing(self):
        """Mirrors the fallback formula at _joint_finalization line 287."""
        assert _candidate_composite_score({"quant_score": 70}) == 70.0
        assert _candidate_composite_score({"quant_score": 70, "qual_score": None}) == 70.0

    def test_none_when_no_quant(self):
        """No quant score → gate has no basis to judge; helper returns
        None and caller passes the candidate through unfiltered."""
        assert _candidate_composite_score({"qual_score": 80}) is None
        assert _candidate_composite_score({}) is None


# ---------------------------------------------------------------------------
# _apply_regime_pick_gate
# ---------------------------------------------------------------------------


class TestApplyRegimePickGate:
    def test_no_op_when_flag_disabled(self):
        """Default state — gate is OFF. peer_review behavior unchanged."""
        picks = [{"ticker": "A", "quant_score": 50}, {"ticker": "B", "quant_score": 30}]
        with patch("config.SECTOR_REGIME_PICK_GATE_ENABLED", False):
            result = _apply_regime_pick_gate(picks, "bear", -2.0, "team_x")
        assert result == picks

    def test_filters_picks_below_threshold_when_enabled(self):
        """Bear regime with deep risk-off (intensity_z=-2) raises
        threshold; low-score picks are dropped."""
        picks = [
            {"ticker": "HIGH", "quant_score": 85, "qual_score": 80},  # composite 82.5
            {"ticker": "MED",  "quant_score": 60, "qual_score": 55},  # composite 57.5
            {"ticker": "LOW",  "quant_score": 45, "qual_score": 50},  # composite 47.5
        ]
        with patch("config.SECTOR_REGIME_PICK_GATE_ENABLED", True), \
             patch("config.SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE", 50.0), \
             patch("config.SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE", 10.0):
            # threshold = 50 + max(0, 2.0) * 10 = 70
            result = _apply_regime_pick_gate(picks, "bear", -2.0, "team_x")
        result_tickers = [c["ticker"] for c in result]
        assert result_tickers == ["HIGH"]

    def test_allows_zero_picks_when_no_candidate_clears_bar(self):
        """The signature institutional change — sector teams can now
        return EMPTY picks when bear/caution conditions raise the bar
        above every candidate. Previously the team would forcibly
        return 2-3 picks regardless of quality."""
        picks = [
            {"ticker": "A", "quant_score": 40, "qual_score": 30},
            {"ticker": "B", "quant_score": 35, "qual_score": 25},
        ]
        with patch("config.SECTOR_REGIME_PICK_GATE_ENABLED", True), \
             patch("config.SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE", 50.0), \
             patch("config.SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE", 10.0):
            result = _apply_regime_pick_gate(picks, "bear", -1.5, "team_x")
        assert result == []

    def test_passes_through_picks_missing_score(self):
        """Defensive: a candidate with no quant_score has no basis for
        the gate to judge → kept rather than silently dropped."""
        picks = [
            {"ticker": "NOSCORE", "qual_score": 90},  # quant missing
            {"ticker": "LOW", "quant_score": 30, "qual_score": 25},
        ]
        with patch("config.SECTOR_REGIME_PICK_GATE_ENABLED", True), \
             patch("config.SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE", 50.0), \
             patch("config.SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE", 10.0):
            result = _apply_regime_pick_gate(picks, "bear", -1.5, "team_x")
        # NOSCORE kept (no basis to judge); LOW filtered out
        assert [c["ticker"] for c in result] == ["NOSCORE"]

    def test_bull_regime_passes_all_picks_when_base_zero(self):
        """Bull regime with base=0 threshold → threshold=0 → no
        filtering."""
        picks = [
            {"ticker": "A", "quant_score": 40, "qual_score": 30},
            {"ticker": "B", "quant_score": 10, "qual_score": 5},
        ]
        with patch("config.SECTOR_REGIME_PICK_GATE_ENABLED", True), \
             patch("config.SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE", 0.0), \
             patch("config.SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE", 10.0):
            result = _apply_regime_pick_gate(picks, "bull", 1.5, "team_x")
        assert result == picks

    def test_none_intensity_degrades_to_base_threshold(self):
        """No substrate available — gate still applies the base
        threshold (no regime premium). Allows the gate to still do
        useful baseline filtering even in Stage A pre-deploy state."""
        picks = [
            {"ticker": "HIGH", "quant_score": 85, "qual_score": 80},
            {"ticker": "LOW", "quant_score": 30, "qual_score": 25},
        ]
        with patch("config.SECTOR_REGIME_PICK_GATE_ENABLED", True), \
             patch("config.SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE", 50.0), \
             patch("config.SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE", 10.0):
            # intensity_z=None → threshold=base=50; LOW filtered
            result = _apply_regime_pick_gate(picks, "neutral", None, "team_x")
        assert [c["ticker"] for c in result] == ["HIGH"]


# ---------------------------------------------------------------------------
# Default config pins
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_gate_off_by_default(self):
        """Stage D' Wire 1 ships behind a flag — default OFF until
        operator validates via the next Saturday SF observation cycle."""
        from config import SECTOR_REGIME_PICK_GATE_ENABLED
        assert SECTOR_REGIME_PICK_GATE_ENABLED is False

    def test_base_min_score_default_zero(self):
        from config import SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE
        assert SECTOR_REGIME_PICK_GATE_BASE_MIN_SCORE == 0.0

    def test_intensity_scale_default(self):
        from config import SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE
        assert SECTOR_REGIME_PICK_GATE_INTENSITY_SCALE == 8.0


# ---------------------------------------------------------------------------
# SectorTeamContext schema
# ---------------------------------------------------------------------------


class TestSectorTeamContextHasIntensityZField:
    def test_field_present_with_default_none(self):
        """The SectorTeamContext dataclass carries the substrate
        intensity_z so the sector_team_node → peer_review path can
        thread it through. Default None handles pre-deploy state."""
        import dataclasses
        from agents.sector_teams.sector_team import SectorTeamContext
        fields = {f.name: f for f in dataclasses.fields(SectorTeamContext)}
        assert "regime_intensity_z" in fields
        assert fields["regime_intensity_z"].default is None
