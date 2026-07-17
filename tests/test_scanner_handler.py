"""Unit tests for the scanner Lambda handler (ROADMAP L1995 Phase 1)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "scanner_handler.py"


def _load_handler_module():
    """Import lambda/scanner_handler.py without using ``lambda`` as a
    package name (Python keyword)."""
    module_name = "lambda_scanner_handler"
    spec = importlib.util.spec_from_file_location(module_name, _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def handler_mod():
    mod = _load_handler_module()
    mod._init_done = False
    yield mod


def _ok_artifact() -> dict:
    return {
        "run_date": "2026-05-30",
        "scanner_version": "v1.0",
        "generated_at": "2026-05-30T09:00:00+00:00",
        "population_tickers": ["AAPL", "GOOG"],
        "scanner_tickers": ["AMD", "BNY", "SN"],
        "agent_input_set": ["AAPL", "GOOG", "AMD", "BNY", "SN"],
        "filters_applied": {"min_avg_volume": 500000},
        "stats": {
            "universe_size": 903,
            "post_scanner": 3,
            "population_size": 2,
            "agent_input_size": 5,
            "feature_store_enriched": 903,
            "feature_store_missing": 0,
            "new_vs_prior_cycle": ["BNY", "SN"],
            "dropped_vs_prior_cycle": ["PSTG"],
            "prior_run_date": "2026-05-23",
            "baseline_missing": False,
        },
    }


class TestHandler:
    def test_ok_when_orchestrator_succeeds(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   return_value=_ok_artifact()), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-30/candidates.json"), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"}, context=None,
            )
        assert result["status"] == "OK"
        # 2026-05-30 (Sat) normalizes to the 2026-05-29 (Fri) trading day —
        # candidates.json keys by trading day to match Research (DATE_CONVENTIONS).
        assert result["date"] == "2026-05-29"
        # Summary surfaces the operationally interesting counts.
        assert result["summary"]["scanner_tickers"] == 3
        assert result["summary"]["population_tickers"] == 2
        assert result["summary"]["new_vs_prior_cycle"] == 2
        assert result["summary"]["dropped_vs_prior_cycle"] == 1
        assert result["summary"]["s3_key"] == "candidates/2026-05-30/candidates.json"

    def test_error_when_orchestrator_precondition_fails(self, handler_mod):
        from data.scanner_orchestrator import ScannerOrchestratorError

        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   side_effect=ScannerOrchestratorError(
                       "feature store empty"
                   )), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"}, context=None,
            )
        assert result["status"] == "ERROR"
        assert "feature store" in result["error"]

    def test_error_when_orchestrator_raises_unexpected(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   side_effect=RuntimeError("S3 unreachable")), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"}, context=None,
            )
        assert result["status"] == "ERROR"
        assert "S3 unreachable" in result["error"]

    def test_error_when_s3_write_fails(self, handler_mod):
        # Build succeeded but the S3 write itself blew up — must surface
        # as ERROR (the artifact was lost; SF Catch handles).
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   return_value=_ok_artifact()), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   side_effect=RuntimeError("PutObject denied")), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"}, context=None,
            )
        assert result["status"] == "ERROR"
        assert "S3 write failed" in result["error"]
        assert "PutObject denied" in result["error"]

    def test_error_when_run_date_missing(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "run_date" in result["error"]

    def test_error_when_run_date_invalid(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler({"run_date": "bad"}, context=None)
        assert result["status"] == "ERROR"
        assert "run_date" in result["error"] or "bad" in result["error"]

    def test_dry_run_short_circuits_before_s3(self, handler_mod):
        # dry_run_llm shell-run path must NOT touch S3 or call the
        # orchestrator. Mirrors the rationale_clustering / eval_judge
        # dry path used by Friday-Preflight shell runs.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact") as mock_build, \
             patch("boto3.client") as mock_boto:
            result = handler_mod.handler(
                {"dry_run_llm": True, "run_date": "2026-05-30"},
                context=None,
            )
        assert result["status"] == "OK"
        assert result["dry_run"] is True
        mock_build.assert_not_called()
        mock_boto.assert_not_called()

    def test_run_date_threaded_through_to_orchestrator(self, handler_mod):
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return _ok_artifact()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   side_effect=fake_build), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-30/candidates.json"), \
             patch("boto3.client", return_value=MagicMock()):
            handler_mod.handler(
                {"run_date": "2026-05-30"}, context=None,
            )
        # Normalized Sat→Fri trading day before reaching the orchestrator.
        assert captured["run_date"] == "2026-05-29"

    def test_bucket_and_market_regime_overrides(self, handler_mod):
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return _ok_artifact()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   side_effect=fake_build), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-30/candidates.json"), \
             patch("boto3.client", return_value=MagicMock()):
            handler_mod.handler(
                {
                    "run_date": "2026-05-30",
                    "bucket": "test-bucket",
                    "market_regime": "bull",
                },
                context=None,
            )
        assert captured["bucket"] == "test-bucket"
        assert captured["market_regime"] == "bull"

    def test_run_date_normalized_to_trading_day(self, handler_mod):
        """A weekend/holiday calendar run_date is normalized to the most
        recent trading day so candidates.json lands on the SAME key Research
        reads (DATE_CONVENTIONS). The 2026-05-30 recovery failed because the
        Scanner keyed by calendar date (Sat) while Research read trading day
        (Fri). Saturday 2026-05-30 → Friday 2026-05-29; a trading-day input
        passes through unchanged."""
        for given, expected in [("2026-05-30", "2026-05-29"),   # Sat → Fri
                                ("2026-05-31", "2026-05-29"),   # Sun → Fri
                                ("2026-05-29", "2026-05-29")]:   # Fri → Fri
            captured = {}

            def fake_build(**kwargs):
                captured.update(kwargs)
                return _ok_artifact()

            with patch.object(handler_mod, "_ensure_init"), \
                 patch("data.scanner_orchestrator.build_candidates_artifact",
                       side_effect=fake_build), \
                 patch("data.scanner_orchestrator.write_candidates_artifact",
                       return_value=f"candidates/{expected}/candidates.json"), \
                 patch("boto3.client", return_value=MagicMock()):
                result = handler_mod.handler({"run_date": given}, context=None)
            assert captured["run_date"] == expected, (
                f"run_date {given} must normalize to trading day {expected}, "
                f"got {captured.get('run_date')}"
            )
            assert result["date"] == expected

    def test_shadow_specs_written_and_summarized(self, handler_mod):
        # Champion/challenger OBSERVE shadows (config#1221): the handler writes
        # each challenger artifact to the isolated shadow prefix and records the
        # keys in summary.shadows — without disturbing the live OK path.
        shadow_art = {"momentum_sleeve": {"run_date": "2026-05-29", "scanner_tickers": ["A"]}}
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   return_value=_ok_artifact()), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-29/candidates.json"), \
             patch("data.scanner_orchestrator.build_shadow_candidate_artifacts",
                   return_value=shadow_art), \
             patch("data.scanner_orchestrator.write_shadow_candidates_artifact",
                   return_value="candidates_shadow/momentum_sleeve/2026-05-29/candidates.json"), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["shadows"] == {
            "momentum_sleeve": "candidates_shadow/momentum_sleeve/2026-05-29/candidates.json"
        }
        assert "shadow_error" not in result["summary"]

    def test_shadow_failure_does_not_downgrade_live_ok(self, handler_mod):
        # A shadow build/write failure is WHOLLY fail-soft: live stays OK, the
        # failure is recorded in summary.shadow_error (no-silent-fails).
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   return_value=_ok_artifact()), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-29/candidates.json"), \
             patch("data.scanner_orchestrator.build_shadow_candidate_artifacts",
                   side_effect=RuntimeError("loadings exploded")), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["shadows"] == {}
        assert "loadings exploded" in result["summary"]["shadow_error"]

    def test_universe_board_written_and_summarized(self, handler_mod):
        # alpha-engine-config-I2515: the standalone Scanner path becomes a
        # universe-board producer. The handler records the written key in
        # summary.universe_board without disturbing the live OK path.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   return_value=_ok_artifact()), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-29/candidates.json"), \
             patch("data.scanner_orchestrator.write_universe_board_for_scanner_run",
                   return_value="scanner/universe/2026-05-29/universe.json"), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["universe_board"] == {
            "status": "OK", "key": "scanner/universe/2026-05-29/universe.json",
        }
        assert "universe_board_error" not in result["summary"]

    def test_universe_board_failure_does_not_downgrade_live_ok(self, handler_mod):
        # A board build/write failure is WHOLLY fail-soft: live stays OK, the
        # failure is recorded in summary.universe_board_error (no-silent-fails)
        # — mirrors the shadow-artifact fail-soft contract above.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   return_value=_ok_artifact()), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-29/candidates.json"), \
             patch("data.scanner_orchestrator.write_universe_board_for_scanner_run",
                   side_effect=RuntimeError("factor profiles unreadable")), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["universe_board"] == {"status": "error", "key": None}
        assert "factor profiles unreadable" in result["summary"]["universe_board_error"]

    def test_universe_board_receives_market_regime_and_artifact(self, handler_mod):
        # market_regime must thread through so build_pure_quant_focus_lookup
        # blends on the SAME regime the scanner used, not a stale default.
        captured = {}
        ok_artifact = _ok_artifact()

        def fake_write(artifact, **kwargs):
            captured.update(kwargs)
            captured["artifact"] = artifact
            return "scanner/universe/2026-05-29/universe.json"

        with patch.object(handler_mod, "_ensure_init"), \
             patch("data.scanner_orchestrator.build_candidates_artifact",
                   return_value=ok_artifact), \
             patch("data.scanner_orchestrator.write_candidates_artifact",
                   return_value="candidates/2026-05-29/candidates.json"), \
             patch("data.scanner_orchestrator.write_universe_board_for_scanner_run",
                   side_effect=fake_write), \
             patch("boto3.client", return_value=MagicMock()):
            handler_mod.handler(
                {"run_date": "2026-05-30", "market_regime": "bull"}, context=None,
            )
        assert captured["market_regime"] == "bull"
        assert captured["artifact"] is ok_artifact
