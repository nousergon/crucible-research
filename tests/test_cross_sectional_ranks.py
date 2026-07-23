"""Tests for ``assign_cross_sectional_ranks`` — PR 1 of the rank-based
portfolio-construction restructure.

Pins:

1. Rank 1 goes to the highest ``final_score``; rank N to the lowest.
2. Ties get the same rank with ``min`` semantics, percentile follows.
3. Percentile range is [0.0, 1.0]; rank 1 → 1.0, last → 0.0.
4. Single-ticker population emits rank=1, percentile=1.0.
5. Empty population is a no-op (no exception).
6. Non-finite or missing ``final_score`` sinks to the bottom without
   crashing the rank assignment for the rest of the population.
7. Mutates in-place — caller's dict carries the new fields.
"""

from __future__ import annotations

import pytest

from scoring.aggregator import assign_cross_sectional_ranks


def _result(score: float | None) -> dict:
    """Minimal result dict with just ``final_score``. Other fields aren't
    read by the rank assigner."""
    return {"final_score": score}


class TestRankSemantics:
    def test_descending_rank_by_final_score(self):
        results = {
            "A": _result(80.0),
            "B": _result(60.0),
            "C": _result(70.0),
        }
        assign_cross_sectional_ranks(results)
        assert results["A"]["cross_sectional_rank"] == 1  # highest
        assert results["C"]["cross_sectional_rank"] == 2
        assert results["B"]["cross_sectional_rank"] == 3  # lowest

    def test_percentile_extremes(self):
        results = {
            "A": _result(80.0),
            "B": _result(60.0),
            "C": _result(70.0),
        }
        assign_cross_sectional_ranks(results)
        assert results["A"]["percentile"] == 1.0  # top
        assert results["B"]["percentile"] == 0.0  # bottom
        # Middle rank's percentile lies strictly between 0 and 1.
        assert 0.0 < results["C"]["percentile"] < 1.0

    def test_percentile_uniform_for_evenly_spaced(self):
        """For 4 tickers, percentiles step by 1/3."""
        results = {
            "A": _result(90.0),
            "B": _result(80.0),
            "C": _result(70.0),
            "D": _result(60.0),
        }
        assign_cross_sectional_ranks(results)
        assert results["A"]["percentile"] == 1.0
        assert results["B"]["percentile"] == pytest.approx(2 / 3, abs=1e-4)
        assert results["C"]["percentile"] == pytest.approx(1 / 3, abs=1e-4)
        assert results["D"]["percentile"] == 0.0


class TestTies:
    def test_ties_get_same_rank_min_semantics(self):
        """Scores [80, 75, 75, 60] → ranks [1, 2, 2, 4]. The third
        ticker's rank does NOT advance to 3 — that's ``min`` tie
        semantics. Next non-tie advances to the sequence position."""
        results = {
            "A": _result(80.0),
            "B": _result(75.0),
            "C": _result(75.0),
            "D": _result(60.0),
        }
        assign_cross_sectional_ranks(results)
        assert results["A"]["cross_sectional_rank"] == 1
        assert results["B"]["cross_sectional_rank"] == 2
        assert results["C"]["cross_sectional_rank"] == 2  # tied with B
        assert results["D"]["cross_sectional_rank"] == 4  # advances past tie

    def test_all_tied(self):
        """If everyone has the same score, they all get rank 1.
        Percentiles are uniform at 1.0 (top)."""
        results = {ch: _result(50.0) for ch in "ABCD"}
        assign_cross_sectional_ranks(results)
        for ch in "ABCD":
            assert results[ch]["cross_sectional_rank"] == 1
            assert results[ch]["percentile"] == 1.0


class TestEdgeCases:
    def test_empty_results_is_noop(self):
        results: dict = {}
        assign_cross_sectional_ranks(results)
        # Should not raise; nothing to assert (dict still empty).
        assert results == {}

    def test_single_ticker_gets_rank_1_pct_1(self):
        results = {"A": _result(42.0)}
        assign_cross_sectional_ranks(results)
        assert results["A"]["cross_sectional_rank"] == 1
        assert results["A"]["percentile"] == 1.0

    def test_none_score_sinks_to_bottom(self):
        """A ticker with ``final_score=None`` shouldn't crash; it sinks
        to the lowest rank (highest rank number) with percentile 0.0
        when the population has finite scores."""
        results = {
            "A": _result(80.0),
            "B": _result(None),
            "C": _result(60.0),
        }
        assign_cross_sectional_ranks(results)
        assert results["A"]["cross_sectional_rank"] == 1
        assert results["C"]["cross_sectional_rank"] == 2
        assert results["B"]["cross_sectional_rank"] == 3
        assert results["B"]["percentile"] == 0.0

    def test_nan_score_sinks_to_bottom(self):
        results = {
            "A": _result(80.0),
            "B": _result(float("nan")),
        }
        assign_cross_sectional_ranks(results)
        assert results["A"]["cross_sectional_rank"] == 1
        assert results["B"]["cross_sectional_rank"] == 2
        assert results["A"]["percentile"] == 1.0
        assert results["B"]["percentile"] == 0.0


class TestNoBehaviorChange:
    """PR 1 explicitly does NOT change rating / score / signal — the
    only addition is two new observability fields. Pin that other
    fields aren't perturbed.
    """

    def test_does_not_perturb_other_fields(self):
        results = {
            "A": {
                "final_score": 80.0,
                "rating": "BUY",
                "signal": "ENTER",
                "quant_score": 75.0,
                "qual_score": 60.0,
                "macro_modifier": 1.05,
                "macro_shift": 1.67,
            },
            "B": {
                "final_score": 50.0,
                "rating": "HOLD",
                "signal": "HOLD",
                "quant_score": 55.0,
                "qual_score": 45.0,
            },
        }
        assign_cross_sectional_ranks(results)

        # Rank fields added.
        assert "cross_sectional_rank" in results["A"]
        assert "percentile" in results["A"]

        # All original fields preserved unchanged.
        assert results["A"]["final_score"] == 80.0
        assert results["A"]["rating"] == "BUY"
        assert results["A"]["signal"] == "ENTER"
        assert results["A"]["quant_score"] == 75.0
        assert results["A"]["macro_modifier"] == 1.05
        assert results["A"]["macro_shift"] == 1.67
        assert results["B"]["rating"] == "HOLD"


class TestProductionShape:
    """Sanity check on a 30-ticker population mirroring the live signals.json
    typical size. Pins that ranks span 1..30 and percentiles cover [0, 1]
    when scores are non-degenerate."""

    def test_30_ticker_population(self):
        results = {
            f"T{i:02d}": _result(float(100 - i * 2.5))
            for i in range(30)
        }
        assign_cross_sectional_ranks(results)
        ranks = [r["cross_sectional_rank"] for r in results.values()]
        pcts = [r["percentile"] for r in results.values()]
        assert min(ranks) == 1
        assert max(ranks) == 30
        assert min(pcts) == 0.0
        assert max(pcts) == 1.0
