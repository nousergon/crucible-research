"""
Tests for ResearchPreflight mode composition.

BasePreflight primitives are tested in alpha-engine-lib. These tests
only verify that each research mode calls the expected primitives.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preflight import ResearchPreflight


class TestResearchPreflight:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            ResearchPreflight(bucket="b", mode="bogus")

    def test_weekly_mode_checks_anthropic_key_and_arcticdb(self):
        pf = ResearchPreflight(bucket="b", mode="weekly")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "_check_arcticdb_universe") as arctic, \
             patch.object(pf, "_check_deferred_imports") as deferred:
            pf.run()
        # Two check_env_vars calls: AWS_REGION then ANTHROPIC_API_KEY.
        assert env.call_args_list[0].args == ("AWS_REGION",)
        assert env.call_args_list[1].args == ("ANTHROPIC_API_KEY",)
        s3.assert_called_once()
        deferred.assert_called_once()
        arctic.assert_called_once()

    def test_alerts_mode_skips_anthropic_key_arcticdb_and_deferred_imports(self):
        pf = ResearchPreflight(bucket="b", mode="alerts")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "_check_arcticdb_universe") as arctic, \
             patch.object(pf, "_check_deferred_imports") as deferred:
            pf.run()
        assert env.call_count == 1
        assert env.call_args_list[0].args == ("AWS_REGION",)
        s3.assert_called_once()
        arctic.assert_not_called()
        # Alerts handler doesn't import scripts.aggregate_costs — no
        # need to pay the eager-import cost in that mode.
        deferred.assert_not_called()


class TestDeferredImportPreflight:
    """Locks ``ResearchPreflight._check_deferred_imports`` (PR
    fix/preflight-eager-imports-and-cio-min-length, 2026-05-02)."""

    def test_resolves_when_module_present(self):
        """All deferred imports must be resolvable in the test
        environment (mirrors the Lambda image post-fix-PR-#85)."""
        pf = ResearchPreflight(bucket="b", mode="weekly")
        # Should NOT raise — scripts/aggregate_costs is in the repo
        # alongside the explicit __init__.py marker.
        pf._check_deferred_imports()

    def test_raises_with_actionable_message_on_module_not_found(self):
        """A missing module must surface as a RuntimeError naming the
        Dockerfile + __init__.py contract — that's the trail every
        future contributor needs to fix the regression."""
        pf = ResearchPreflight(bucket="b", mode="weekly")
        with patch.object(
            ResearchPreflight, "_DEFERRED_IMPORTS",
            (("ghost_module_does_not_exist", "any_attr"),),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                pf._check_deferred_imports()
        msg = str(exc_info.value)
        assert "ghost_module_does_not_exist" in msg
        assert "Dockerfile" in msg, (
            "Error must point at the Docker COPY contract so a future "
            "contributor knows where to apply the fix."
        )

    def test_raises_on_missing_attribute(self):
        """Symbol renamed inside an existing module must also surface
        — same regression class as a missing module from the consumer's
        perspective (the lazy import would crash either way)."""
        pf = ResearchPreflight(bucket="b", mode="weekly")
        with patch.object(
            ResearchPreflight, "_DEFERRED_IMPORTS",
            (("scripts.aggregate_costs", "no_such_function"),),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                pf._check_deferred_imports()
        assert "no_such_function" in str(exc_info.value)

    def test_aggregate_day_is_in_default_deferred_imports(self):
        """Lock the canonical entry. If a refactor renames or moves
        ``scripts.aggregate_costs.aggregate_day``, this test fires
        before the next deploy hits the WARN-at-end-of-run regression."""
        assert (
            "scripts.aggregate_costs",
            "aggregate_day",
        ) in ResearchPreflight._DEFERRED_IMPORTS
