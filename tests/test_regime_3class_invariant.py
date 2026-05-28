"""
test_regime_3class_invariant.py

Pin the 3-class Ang-Bekaert macro regime taxonomy retired the legacy
"caution" tier in v0.42.0 (plan: caution-regime-retirement-260528.md).

Covers:
  - REGIME_VALUES enum is exactly ("bull", "neutral", "bear")
  - macro_agent._REGIME_SEVERITY is 3-key
  - macro_agent._validate_regime coerces legacy LLM "caution" → "neutral"
    (migration shim) with WARN log
  - hard-override → bear still fires on extreme stress (VIX > 30 AND
    SPY 30d < -10%)
  - soft-override → caution path is gone (elevated stress no longer
    discretized; signal flows through regime_intensity_z continuously)

Why this matters: the focus_list silently emptied when regime=caution
because alpha-engine-config/research/scoring.yaml had no caution block
and the consumer hardcoded {bull,bear,neutral} dropped any YAML caution
block on load. Rather than band-aid the config to 4 keys, the SOTA
realignment retires the macro-agent's rule-based caution override
entirely — its driving signals (VIX, HY OAS, SPY 30d return) are
already weighted into regime_intensity_z via alpha-engine-predictor/
regime/composite.py::DEFAULT_WEIGHTS, the continuous SOTA path.
"""

from __future__ import annotations

import logging

import pytest


# ── REGIME_VALUES enum ───────────────────────────────────────────────────


def test_regime_values_is_3class():
    from graph.state_schemas import REGIME_VALUES

    assert REGIME_VALUES == ("bull", "neutral", "bear")
    assert "caution" not in REGIME_VALUES


# ── _REGIME_SEVERITY ordering ────────────────────────────────────────────


def test_regime_severity_is_3key():
    from agents.macro_agent import _REGIME_SEVERITY

    assert set(_REGIME_SEVERITY.keys()) == {"bull", "neutral", "bear"}
    assert _REGIME_SEVERITY["bull"] < _REGIME_SEVERITY["neutral"]
    assert _REGIME_SEVERITY["neutral"] < _REGIME_SEVERITY["bear"]


# ── _validate_regime migration shim ──────────────────────────────────────


class TestValidateRegimeCoercion:
    """Migration shim: legacy LLM `caution` coerces to `neutral`.
    The stress signal that drove the historical `caution` call is
    preserved end-to-end through regime_intensity_z, so no information
    is lost — just no longer discretized into a redundant 4th category.
    """

    def test_legacy_caution_coerces_to_neutral(self, caplog):
        from agents.macro_agent import _validate_regime

        with caplog.at_level(logging.WARNING):
            out = _validate_regime("caution", {})

        assert out == "neutral"
        assert any(
            "LEGACY COERCION" in rec.message and "caution" in rec.message
            for rec in caplog.records
        ), "expected LEGACY COERCION WARN log"

    def test_case_insensitive_coercion(self):
        from agents.macro_agent import _validate_regime

        assert _validate_regime("CAUTION", {}) == "neutral"
        assert _validate_regime("Caution", {}) == "neutral"

    def test_non_legacy_values_passthrough_when_no_threshold_breach(self):
        from agents.macro_agent import _validate_regime

        # No vix / spy_30d → no hard override → returns input unchanged.
        for regime in ("bull", "neutral", "bear"):
            assert _validate_regime(regime, {}) == regime


# ── Hard override → bear ─────────────────────────────────────────────────


class TestHardOverrideStillFires:
    """The hard override (VIX > 30 AND SPY 30d < -10% → bear) survives
    the caution retirement. It's the catastrophic guardrail; only the
    soft-override → caution branches retired."""

    def test_extreme_stress_forces_bear(self, caplog):
        from agents.macro_agent import _validate_regime

        macro = {"vix": 35.0, "sp500_30d_return": -12.0}
        with caplog.at_level(logging.WARNING):
            out = _validate_regime("neutral", macro)

        assert out == "bear"
        assert any(
            "OVERRIDE neutral → bear" in rec.message for rec in caplog.records
        )

    def test_extreme_stress_from_legacy_caution_still_coerces_then_overrides(self):
        from agents.macro_agent import _validate_regime

        # Legacy LLM caution → coerced to neutral → hard override → bear.
        macro = {"vix": 40.0, "sp500_30d_return": -15.0}
        assert _validate_regime("caution", macro) == "bear"

    def test_below_bear_threshold_does_not_override(self):
        from agents.macro_agent import _validate_regime

        # Elevated stress but below the bear hard-override threshold —
        # used to escalate to caution; now passes through (signal flows
        # via regime_intensity_z continuously, not via discrete category).
        macro = {"vix": 27.0, "sp500_30d_return": -7.0, "hy_credit_spread_oas": 600.0}
        assert _validate_regime("neutral", macro) == "neutral"
        assert _validate_regime("bull", macro) == "bull"

    def test_hy_oas_alone_no_longer_overrides(self):
        # The HY-OAS soft override retired with the caution branches.
        # Wide credit spreads flow into regime_intensity_z (hy_oas_bps
        # weighted at +1.0 in alpha-engine-predictor/regime/composite.py
        # DEFAULT_WEIGHTS) rather than discretizing into a regime tier.
        from agents.macro_agent import _validate_regime

        macro = {"vix": 18.0, "sp500_30d_return": 1.0, "hy_credit_spread_oas": 800.0}
        assert _validate_regime("neutral", macro) == "neutral"


# ── No-regression on existing non-override paths ─────────────────────────


def test_no_guardrails_config_passes_through():
    """When REGIME_GUARDRAILS is unset, _validate_regime returns input
    (except the legacy coercion shim, which fires unconditionally)."""
    from agents import macro_agent

    # Save + clear the guardrails cfg.
    saved = macro_agent.REGIME_GUARDRAILS
    macro_agent.REGIME_GUARDRAILS = {}
    try:
        assert macro_agent._validate_regime("bull", {"vix": 50, "sp500_30d_return": -20}) == "bull"
        assert macro_agent._validate_regime("bear", {}) == "bear"
        # Coercion still fires (it's pre-cfg check).
        assert macro_agent._validate_regime("caution", {}) == "neutral"
    finally:
        macro_agent.REGIME_GUARDRAILS = saved
