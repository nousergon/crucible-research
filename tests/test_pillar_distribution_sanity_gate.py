"""Tests for the Σ pillar_weights gate on _check_pillar_distribution_sanity.

Surfaced 2026-05-21 via `python local/run.py --dry-run`: the sanity check
fired `low_coverage: 0/1 picks (0.0%) have populated pillar_contributions`
on the dry-run smoke even though the active scoring.yaml has
Phase-4-cutover-defaults (all pillar_weights = 0), where empty
pillar_contributions are harmless because the composite reduces to legacy
by construction.

The gate: skip the check entirely when Σ PILLAR_COMPOSITE_WEIGHTS ≈ 0,
since pillar coverage is not load-bearing under those weights.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from graph.research_graph import _check_pillar_distribution_sanity


_ZERO_WEIGHTS = {
    "quality": 0.0, "value": 0.0, "momentum": 0.0,
    "growth": 0.0, "stewardship": 0.0, "defensiveness": 0.0,
}


_AQR_WEIGHTS = {
    "quality": 0.25, "value": 0.20, "momentum": 0.20,
    "growth": 0.15, "defensiveness": 0.10, "stewardship": 0.10,
}


def _thesis_without_pillar() -> dict:
    """Dry-run / Phase-4-cutover thesis: legacy composite only."""
    return {
        "composite_breakdown": {
            "final_score": 65.0,
            "pillar_contributions": [],
            "legacy_blend": {"quant_component": 60.0, "qual_component": 70.0},
        },
    }


def _thesis_with_pillar(seed: int = 0) -> dict:
    """Live LLM thesis under PILLAR_EMIT_ENABLED=true.

    `seed` shifts the per-pillar qual scores so multi-thesis fixtures
    keep std above the collapsed_pillar threshold (5.0) — a real LLM
    rubric assigns different values to different tickers.
    """
    return {
        "composite_breakdown": {
            "final_score": 70.0,
            "pillar_contributions": [
                {
                    "pillar": p,
                    "qual_component": 50.0 + 5.0 * i + 7.0 * seed,
                    "quant_component": 55.0,
                }
                for i, p in enumerate(("quality", "value", "momentum", "growth", "stewardship", "defensiveness"))
            ],
        },
    }


def test_skips_when_pillar_weights_zero():
    """Phase-4-cutover-defaults: empty pillar_contributions are by design."""
    theses = {"AAPL": _thesis_without_pillar()}
    with patch("graph.research_graph.PILLAR_COMPOSITE_WEIGHTS", _ZERO_WEIGHTS), \
         patch("alpha_engine_lib.alerts.publish") as mock_publish:
        _check_pillar_distribution_sanity(theses)
    mock_publish.assert_not_called()


def test_skips_dry_run_smoke_at_phase4_defaults():
    """Dry-run path: stub qual_analyst produces no pillar_assessments; at
    Phase-4-cutover-defaults this must not fire (was the 5/21 false alarm).
    """
    theses = {"NVDA": _thesis_without_pillar()}
    with patch("graph.research_graph.PILLAR_COMPOSITE_WEIGHTS", _ZERO_WEIGHTS), \
         patch("alpha_engine_lib.alerts.publish") as mock_publish:
        _check_pillar_distribution_sanity(theses)
    mock_publish.assert_not_called()


def test_fires_when_aqr_weights_live_and_coverage_low():
    """Genuine AQR-cutover regression: weights load-bearing, pillar emit empty."""
    theses = {
        "AAPL": _thesis_without_pillar(),
        "NVDA": _thesis_without_pillar(),
        "MSFT": _thesis_without_pillar(),
    }
    with patch("graph.research_graph.PILLAR_COMPOSITE_WEIGHTS", _AQR_WEIGHTS), \
         patch("alpha_engine_lib.alerts.publish") as mock_publish:
        _check_pillar_distribution_sanity(theses)
    mock_publish.assert_called_once()
    call_msg = mock_publish.call_args.kwargs["message"]
    assert "low_coverage: 0/3" in call_msg
    assert "AQR-prior cutover sanity FAIL" in call_msg


def test_passes_when_aqr_weights_live_and_coverage_high():
    """Healthy AQR-cutover: live LLM emitted pillar_assessments on every pick."""
    theses = {
        f"TICK{i}": _thesis_with_pillar(seed=i) for i in range(5)
    }
    with patch("graph.research_graph.PILLAR_COMPOSITE_WEIGHTS", _AQR_WEIGHTS), \
         patch("alpha_engine_lib.alerts.publish") as mock_publish:
        _check_pillar_distribution_sanity(theses)
    mock_publish.assert_not_called()


def test_zero_theses_short_circuits_at_any_weight_state():
    """No theses → nothing to check, regardless of weights."""
    for weights in (_ZERO_WEIGHTS, _AQR_WEIGHTS):
        with patch("graph.research_graph.PILLAR_COMPOSITE_WEIGHTS", weights), \
             patch("alpha_engine_lib.alerts.publish") as mock_publish:
            _check_pillar_distribution_sanity({})
        mock_publish.assert_not_called()


def test_epsilon_boundary_treats_near_zero_weights_as_zero():
    """Float-residual weights (e.g. 1e-9) should not trip the load-bearing gate."""
    tiny_weights = {p: 1e-9 / 6 for p in _ZERO_WEIGHTS}
    theses = {"AAPL": _thesis_without_pillar()}
    with patch("graph.research_graph.PILLAR_COMPOSITE_WEIGHTS", tiny_weights), \
         patch("alpha_engine_lib.alerts.publish") as mock_publish:
        _check_pillar_distribution_sanity(theses)
    mock_publish.assert_not_called()


def test_half_pillar_ramp_with_low_coverage_still_fires():
    """The 50/50 half-pillar-ramp config (cutover-diagnostic-recipe.md): Σ
    pillar = 0.5 IS load-bearing, so coverage failures must surface.
    """
    half_weights = {p: 0.5 / 6 for p in _ZERO_WEIGHTS}
    theses = {f"TICK{i}": _thesis_without_pillar() for i in range(5)}
    with patch("graph.research_graph.PILLAR_COMPOSITE_WEIGHTS", half_weights), \
         patch("alpha_engine_lib.alerts.publish") as mock_publish:
        _check_pillar_distribution_sanity(theses)
    mock_publish.assert_called_once()
    assert "low_coverage: 0/5" in mock_publish.call_args.kwargs["message"]
