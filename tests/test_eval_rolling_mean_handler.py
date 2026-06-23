"""Unit tests for the rolling-mean Lambda handler (PR 4b)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "eval_rolling_mean_handler.py"


def _load_handler_module():
    """Import lambda/eval_rolling_mean_handler.py without using ``lambda``
    as a package name (Python keyword)."""
    module_name = "lambda_eval_rolling_mean_handler"
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


def _ok_summary() -> dict:
    return {
        "combos_discovered": 12,
        "datapoints_emitted": 12,
        "combos_skipped_no_data": 0,
        "failed": [],
        "window_start": "2026-05-09T00:00:00+00:00",
        "window_end": "2026-06-06T00:00:00+00:00",
    }


def _partial_summary() -> dict:
    s = _ok_summary()
    s["failed"] = [{
        "combo_idx": "5", "stage": "get_metric_data",
        "error": "missing result for query Id",
    }]
    return s


def _calib_report(status: str = "empty") -> dict:
    return {
        "status": status, "n_cells": 0,
        "n_cells_sufficient": 0, "n_paired_reviews": 0,
    }


class TestHandler:
    @pytest.fixture(autouse=True)
    def _stub_calibration(self):
        """Stub the κ-report side path so the handler tests never touch
        AWS. The two dedicated tests below override this."""
        with patch("evals.calibration_kappa.emit_calibration_report",
                   return_value=_calib_report()):
            yield

    def test_ok_when_no_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   return_value=_ok_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["datapoints_emitted"] == 12

    def test_calibration_report_surfaced_in_result(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   return_value=_ok_summary()), \
             patch("evals.calibration_kappa.emit_calibration_report",
                   return_value=_calib_report("ok")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["calibration"]["status"] == "ok"

    def test_calibration_failure_is_non_fatal(self, handler_mod):
        # κ side path failing must NOT change the primary rolling-mean
        # status — it is recorded in the calibration field instead.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   return_value=_ok_summary()), \
             patch("evals.calibration_kappa.emit_calibration_report",
                   side_effect=RuntimeError("S3 down")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["calibration"]["status"] == "ERROR"
        assert "S3 down" in result["calibration"]["error"]

    def test_partial_when_any_failure(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   return_value=_partial_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "PARTIAL"
        assert len(result["summary"]["failed"]) == 1

    def test_error_when_compute_raises(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   side_effect=RuntimeError("CW throttled")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "CW throttled" in result["error"]

    def test_end_time_iso_passed_through(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"end_time_iso": "2026-06-06T00:00:00Z"}, context=None,
            )

        assert captured["end_time"] == datetime(
            2026, 6, 6, 0, 0, tzinfo=timezone.utc,
        )

    def test_end_time_defaults_to_none_when_unset(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean",
                   side_effect=fake_compute):
            handler_mod.handler({}, context=None)

        # None means rolling_mean will default to now-UTC.
        assert captured.get("end_time") is None

    # ── agent_quality producer wiring (config#1149 Batch A) ─────────────────
    def test_agent_quality_surfaced_in_result(self, handler_mod):
        # The previously-unwired build_agent_quality producer now runs here;
        # its artifact + graded-component list rides in the result.
        artifact = {"status": "ok", "date": "2026-06-22",
                    "signal_volume_adequacy": {"value": 25, "n": 25},
                    "judge_rubric_pass_rate": {"value": 0.8, "n": 30}}
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()), \
             patch("scripts.build_agent_quality.build_agent_quality", return_value=artifact), \
             patch("scripts.build_agent_quality.write_agent_quality",
                   return_value="backtest/2026-06-22/agent_quality.json"):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["agent_quality"]["status"] == "OK"
        assert result["agent_quality"]["key"] == "backtest/2026-06-22/agent_quality.json"
        assert set(result["agent_quality"]["graded_components"]) == {
            "signal_volume_adequacy", "judge_rubric_pass_rate"}

    def test_agent_quality_failure_is_non_fatal(self, handler_mod):
        # A producer failure MUST NOT change the primary rolling-mean status —
        # it is recorded in the agent_quality field instead.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()), \
             patch("scripts.build_agent_quality.build_agent_quality",
                   side_effect=RuntimeError("S3 list failed")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["agent_quality"]["status"] == "ERROR"
        assert "S3 list failed" in result["agent_quality"]["error"]
