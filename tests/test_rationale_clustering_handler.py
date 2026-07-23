"""Unit tests for the cross-week rationale clustering Lambda handler."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "rationale_clustering_handler.py"


def _load_handler_module():
    """Import lambda/rationale_clustering_handler.py without using
    ``lambda`` as a package name (Python keyword)."""
    module_name = "lambda_rationale_clustering_handler"
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
        "window_start": "2026-03-14T00:00:00+00:00",
        "window_end": "2026-05-09T00:00:00+00:00",
        "artifacts_discovered": 48,
        "agents_analyzed": 6,
        "agents_skipped_thin_sample": [],
        "load_failures": [],
        "cluster_failures": [],
        "per_agent": [
            {
                "agent_id": "sector_quant",
                "n_rationales": 40,
                "n_clusters": 5,
                "top3_concentration": 0.85,
                "analysis_key": "decision_artifacts/_analysis/sector_quant/2026-W19.json",
            },
        ],
    }


def _partial_summary() -> dict:
    s = _ok_summary()
    s["load_failures"] = [{"key": "x.json", "error": "NoSuchKey"}]
    return s


class TestHandler:
    def test_ok_when_no_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rationale_clustering.compute_and_emit",
                   return_value=_ok_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["agents_analyzed"] == 6

    def test_partial_when_load_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rationale_clustering.compute_and_emit",
                   return_value=_partial_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "PARTIAL"
        assert len(result["summary"]["load_failures"]) == 1

    def test_error_when_compute_raises(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rationale_clustering.compute_and_emit",
                   side_effect=RuntimeError("S3 unreachable")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "S3 unreachable" in result["error"]

    def test_end_time_iso_passed_through(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rationale_clustering.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"end_time_iso": "2026-05-09T00:00:00Z"}, context=None,
            )

        assert captured["end_time"] == datetime(
            2026, 5, 9, 0, 0, tzinfo=UTC,
        )

    def test_window_days_override(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rationale_clustering.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler({"window_days": 28}, context=None)

        assert captured["window_days"] == 28

    def test_dry_run_disables_metric_emission(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.rationale_clustering.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler({"dry_run": True}, context=None)

        assert captured["emit_metrics"] is False
