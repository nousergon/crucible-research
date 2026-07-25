"""Tests for ``graph.research_graph._validate`` and the
``STRICT_VALIDATION`` env-var escape hatch.

The validator helper is the single chokepoint for typed-state validation.
PR 2.1 introduces the ``strict`` parameter (and the env-var helper that
defaults it) without changing default behavior — warn-mode stays
warn-mode until Step F flips ``_strict_validation_enabled()``'s default.

These tests lock both halves of the contract:
  - Behavior under each combination of ``strict`` + env var
  - The env-var helper itself parses values per-spec

Plus a few sanity checks on extra-field tolerance, since PR 2's
``with_structured_output`` integration relies on ``extra="allow"``
surviving strict mode.
"""

from __future__ import annotations

import pytest

from graph.research_graph import _strict_validation_enabled, _validate
from graph.state_schemas import (
    CIODecision,
    InvestmentThesis,
    SectorRecommendation,
)


class TestStrictValidationEnvVar:
    """``_strict_validation_enabled()`` parses the env var per-spec."""

    def test_unset_defaults_to_true_post_step_f(self, monkeypatch):
        """Step F of PR 2 (2026-04-30 evening) flipped the default from
        False to True. Strict-by-default is now the post-flip steady state;
        STRICT_VALIDATION=false is the emergency override path."""
        monkeypatch.delenv("STRICT_VALIDATION", raising=False)
        assert _strict_validation_enabled() is True

    def test_true_string_enables_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "true")
        assert _strict_validation_enabled() is True

    def test_TRUE_uppercase_enables_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "TRUE")
        assert _strict_validation_enabled() is True

    def test_one_string_enables_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "1")
        assert _strict_validation_enabled() is True

    def test_yes_string_enables_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "yes")
        assert _strict_validation_enabled() is True

    def test_false_string_disables_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "false")
        assert _strict_validation_enabled() is False

    def test_arbitrary_non_truthy_string_disables_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "maybe")
        assert _strict_validation_enabled() is False

    def test_empty_string_disables_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "")
        assert _strict_validation_enabled() is False


class TestValidateBehavior:
    """``_validate`` behaves correctly under each strict/payload combo."""

    def test_valid_payload_passes_in_strict_default(self, monkeypatch, caplog):
        """Default is strict post-Step F. Valid payload still passes silently."""
        monkeypatch.delenv("STRICT_VALIDATION", raising=False)
        good = {"ticker": "JPM", "thesis_type": "ADVANCE", "conviction": 78}
        # Should not raise, should not log a schema warning
        _validate(CIODecision, good, context="test")
        assert "schema-warn" not in caplog.text
        assert "schema-fail" not in caplog.text

    def test_invalid_payload_raises_in_strict_default(self, monkeypatch):
        """Default is strict post-Step F. Invalid payload raises rather than
        warns. To recover the warn-mode log path, set STRICT_VALIDATION=false."""
        monkeypatch.delenv("STRICT_VALIDATION", raising=False)
        bad = {"ticker": "JPM", "conviction": 999}  # out of [0, 100]
        with pytest.raises(RuntimeError, match=r"schema-fail:test_default_strict"):
            _validate(CIODecision, bad, context="test_default_strict")

    def test_invalid_payload_raises_when_strict_param_true(self, monkeypatch):
        monkeypatch.delenv("STRICT_VALIDATION", raising=False)
        bad = {"ticker": "JPM", "conviction": 999}
        with pytest.raises(RuntimeError, match=r"schema-fail:test_strict"):
            _validate(CIODecision, bad, context="test_strict", strict=True)

    def test_invalid_payload_raises_when_env_var_true(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "true")
        bad = {"ticker": "JPM", "conviction": 999}
        with pytest.raises(RuntimeError, match=r"schema-fail"):
            _validate(CIODecision, bad, context="env_var_strict")

    def test_invalid_payload_warns_when_env_var_false(self, monkeypatch, caplog):
        monkeypatch.setenv("STRICT_VALIDATION", "false")
        bad = {"ticker": "JPM", "conviction": 999}
        import logging
        with caplog.at_level(logging.WARNING):
            _validate(CIODecision, bad, context="env_var_lax")
        assert "schema-warn:env_var_lax" in caplog.text

    def test_param_overrides_env_var_when_strict_false(self, monkeypatch, caplog):
        """Explicit ``strict=False`` overrides env-var-true (useful for tests)."""
        monkeypatch.setenv("STRICT_VALIDATION", "true")
        bad = {"ticker": "JPM", "conviction": 999}
        import logging
        with caplog.at_level(logging.WARNING):
            _validate(CIODecision, bad, context="param_override_lax", strict=False)
        assert "schema-warn:param_override_lax" in caplog.text

    def test_extra_fields_accepted_in_strict_mode(self, monkeypatch):
        """PR 2 integration relies on extra='allow' surviving strict mode.
        State + LLM-extraction schemas all use extra='allow' so prompts
        emitting unexpected keys do not crash hard-fail validation."""
        monkeypatch.setenv("STRICT_VALIDATION", "true")
        payload_with_extra = {
            "ticker": "AAPL",
            "quant_score": 70.0,
            "qual_score": 65.0,
            "undocumented_field": "value",
        }
        # Should NOT raise — extra fields are allowed even strict
        _validate(SectorRecommendation, payload_with_extra, context="extra_test")

    def test_required_field_missing_raises_in_strict(self, monkeypatch):
        monkeypatch.setenv("STRICT_VALIDATION", "true")
        # InvestmentThesis requires ticker, final_score, rating
        missing = {"ticker": "AAPL"}  # missing final_score + rating
        with pytest.raises(RuntimeError, match=r"schema-fail"):
            _validate(InvestmentThesis, missing, context="missing_required")


class TestBackwardCompatAlias:
    """``_warn_validate`` is preserved as an alias to ``_validate`` for any
    external imports we missed during the rename."""

    def test_warn_validate_alias_points_to_validate(self):
        from graph.research_graph import _validate, _warn_validate
        assert _warn_validate is _validate
