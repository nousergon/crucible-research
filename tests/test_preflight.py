"""
Tests for ResearchPreflight mode composition.

BasePreflight primitives are tested in alpha-engine-lib. These tests
only verify that each research mode calls the expected primitives.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preflight import ResearchPreflight


class TestResearchPreflight:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            ResearchPreflight(bucket="b", mode="bogus")

    def test_weekly_mode_checks_router_reachability_and_arcticdb(self):
        """The weekly gate is ROUTER REACHABILITY, not a vendor credential.

        alpha-engine-config-I9302: this used to assert ANTHROPIC_API_KEY.
        Direct Anthropic is retired, so that check would pass on a box with no
        router and fail on a correctly configured one — wrong in both
        directions. What the run needs is krepis' env contract plus both model
        classes actually resolving.
        """
        pf = ResearchPreflight(bucket="b", mode="weekly")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "_check_arcticdb_universe") as arctic, \
             patch.object(pf, "_check_router_resolves_weekly_classes") as router, \
             patch.object(pf, "_check_deferred_imports") as deferred:
            pf.run()
        assert env.call_args_list[0].args == ("AWS_REGION",)
        assert env.call_args_list[1].args == (
            "KREPIS_LITELLM_PROXY_URL", "KREPIS_ROUTER_CREDENTIAL_SECRET"
        )
        assert not any(
            "ANTHROPIC_API_KEY" in c.args for c in env.call_args_list
        ), "the retired vendor's credential must not gate the weekly run"
        router.assert_called_once()
        s3.assert_called_once()
        deferred.assert_called_once()
        arctic.assert_called_once()

    def test_alerts_mode_skips_router_check_arcticdb_and_deferred_imports(self):
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
        # Alerts make no LLM call at all — resolving a model class there would
        # fail a healthy run for a dependency it does not have.
        # Alerts handler doesn't import scripts.aggregate_costs — no
        # need to pay the eager-import cost in that mode.
        deferred.assert_not_called()


class TestRouterReachabilityPreflight:
    """Locks ``ResearchPreflight._check_router_resolves_weekly_classes``
    (alpha-engine-config-I9302)."""

    def test_resolves_both_weekly_model_classes(self):
        pf = ResearchPreflight(bucket="b", mode="weekly")
        spec = SimpleNamespace(model="m", api_key_env="K")
        resolve = MagicMock(return_value=(spec, {}))
        with patch.dict(sys.modules, {"krepis.router": SimpleNamespace(
            resolve_group_spec=resolve
        )}):
            pf._check_router_resolves_weekly_classes()
        # One resolution per weekly model class — PER_STOCK and STRATEGIC.
        import config
        assert [c.args[0] for c in resolve.call_args_list] == [
            config.PER_STOCK_CLASS, config.STRATEGIC_CLASS
        ]

    def test_raises_when_a_class_cannot_be_served_from_here(self):
        """Fails loud. A weekly run that cannot resolve its model classes has
        nothing to fall back to, so degrading here would hide the outage."""
        pf = ResearchPreflight(bucket="b", mode="weekly")
        resolve = MagicMock(side_effect=ValueError("no reachable member"))
        with patch.dict(sys.modules, {"krepis.router": SimpleNamespace(
            resolve_group_spec=resolve
        )}), pytest.raises(RuntimeError, match="does not resolve"):
            pf._check_router_resolves_weekly_classes()


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
