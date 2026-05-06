"""Tests for the Risk Posture regime-indicator section.

Pins:
  1. Builds avg/median/max ATR over the population.
  2. Reports n_high_vol vs the top-quartile threshold of the fetched universe.
  3. Empty population → empty list (consolidator skips the section).
  4. Empty technical_scores → empty list.
  5. Population picks with no ATR data → empty list (no spurious section).
  6. Universe of < 4 tickers → omits the high-vol-quartile line (insufficient
     resolution to define a top quartile).
  7. Both atr_pct and atr_14_pct keys are recognised (pipeline parity).
"""
from __future__ import annotations

import pytest

from graph.research_graph import _build_risk_posture


def _ts(atr: float | None = None, key: str = "atr_pct") -> dict:
    """Synthetic technical_scores entry with optional ATR."""
    if atr is None:
        return {"current_price": 100.0}
    return {key: atr, "current_price": 100.0}


class TestRiskPosture:
    def test_avg_median_max_reported(self):
        new_pop = [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}]
        technical_scores = {
            "AAA": _ts(2.0),
            "BBB": _ts(4.0),
            "CCC": _ts(6.0),
        }
        state = {"new_population": new_pop, "technical_scores": technical_scores}
        lines = _build_risk_posture(state)
        joined = "\n".join(lines)
        assert "avg 4.00" in joined
        assert "median 4.00" in joined
        assert "max 6.00" in joined

    def test_high_vol_quartile_count(self):
        new_pop = [{"ticker": "HV1"}, {"ticker": "LV1"}]
        technical_scores = {
            "HV1": _ts(8.0),
            "LV1": _ts(1.0),
            # Universe-only tickers (not in pop) define the quartile threshold.
            "U1": _ts(1.0),
            "U2": _ts(2.0),
            "U3": _ts(3.0),
            "U4": _ts(4.0),
        }
        state = {"new_population": new_pop, "technical_scores": technical_scores}
        lines = _build_risk_posture(state)
        joined = "\n".join(lines)
        # 6 universe ATRs sorted = [1, 1, 2, 3, 4, 8]; idx 4 (75th pct) = 4.
        # HV1 (8.0) ≥ 4 → counted; LV1 (1.0) < 4 → not.
        assert "1/2" in joined  # n_high_vol/len(pop)

    def test_empty_population_returns_empty(self):
        assert _build_risk_posture({"new_population": [], "technical_scores": {}}) == []
        assert _build_risk_posture(
            {"new_population": [], "technical_scores": {"AAA": _ts(2.0)}},
        ) == []

    def test_empty_technical_scores_returns_empty(self):
        assert _build_risk_posture(
            {"new_population": [{"ticker": "AAA"}], "technical_scores": {}},
        ) == []

    def test_population_with_no_atr_data_returns_empty(self):
        # technical_scores exists but ATR fields are missing → no useful section.
        assert _build_risk_posture({
            "new_population": [{"ticker": "AAA"}],
            "technical_scores": {"AAA": {"current_price": 100.0}},
        }) == []

    def test_small_universe_omits_quartile_line(self):
        # Only 3 universe tickers — threshold can't be reliably set.
        new_pop = [{"ticker": "AAA"}]
        technical_scores = {
            "AAA": _ts(3.0),
            "BBB": _ts(2.0),
            "CCC": _ts(4.0),
        }
        state = {"new_population": new_pop, "technical_scores": technical_scores}
        lines = _build_risk_posture(state)
        joined = "\n".join(lines)
        assert "avg 3.00" in joined  # avg/median/max still rendered
        assert "top vol-quartile" not in joined  # quartile line skipped

    def test_both_atr_key_variants_recognised(self):
        new_pop = [{"ticker": "AAA"}, {"ticker": "BBB"}]
        technical_scores = {
            "AAA": _ts(3.0, key="atr_pct"),
            "BBB": _ts(5.0, key="atr_14_pct"),
        }
        state = {"new_population": new_pop, "technical_scores": technical_scores}
        lines = _build_risk_posture(state)
        joined = "\n".join(lines)
        assert "avg 4.00" in joined  # both ATRs picked up
