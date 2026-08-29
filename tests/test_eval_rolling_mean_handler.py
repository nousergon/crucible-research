"""Unit tests for the rolling-mean Lambda handler (PR 4b)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
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
    s["failed"] = [
        {
            "combo_idx": "5",
            "stage": "get_metric_data",
            "error": "missing result for query Id",
        }
    ]
    return s


def _calib_report(status: str = "empty") -> dict:
    return {
        "status": status,
        "n_cells": 0,
        "n_cells_sufficient": 0,
        "n_paired_reviews": 0,
    }


class TestHandler:
    @pytest.fixture(autouse=True)
    def _stub_calibration(self):
        """Stub the κ-report side path so the handler tests never touch
        AWS. The two dedicated tests below override this."""
        with patch("evals.calibration_kappa.emit_calibration_report", return_value=_calib_report()):
            yield

    @pytest.fixture(autouse=True)
    def _stub_producer_leaderboard(self):
        """Same rationale for the producer-leaderboard side path
        (alpha-engine-config-I5195).

        These are HANDLER-WIRING tests; the leaderboard is a fail-soft
        post-step they should not execute for real. That was always true, but
        it went unnoticed while the scorer read `staging/daily_closes/` through
        a mocked S3 and harmlessly returned nothing. The scorer now reads
        ArcticDB, so an unstubbed call opens a real connection — which on macOS
        aborts the interpreter natively (SIGABRT from the bundled allocator, not
        a catchable Python exception), taking the whole suite down.

        The two dedicated leaderboard tests below override this.
        """
        with patch(
            "scoring.leaderboard_producers.build_producer_leaderboard",
            return_value={"status": "ok", "key": None, "leaderboard": {"n_dates": 0}},
        ):
            yield

    def test_ok_when_no_failures(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["datapoints_emitted"] == 12

    def test_calibration_report_surfaced_in_result(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch("evals.calibration_kappa.emit_calibration_report", return_value=_calib_report("ok")),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["calibration"]["status"] == "ok"

    def test_calibration_failure_is_non_fatal(self, handler_mod):
        # κ side path failing must NOT change the primary rolling-mean
        # status — it is recorded in the calibration field instead.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch("evals.calibration_kappa.emit_calibration_report", side_effect=RuntimeError("S3 down")),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["calibration"]["status"] == "ERROR"
        assert "S3 down" in result["calibration"]["error"]

    def test_partial_when_any_failure(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_partial_summary()),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "PARTIAL"
        assert len(result["summary"]["failed"]) == 1

    def test_error_when_compute_raises(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", side_effect=RuntimeError("CW throttled")),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "CW throttled" in result["error"]

    def test_end_time_iso_passed_through(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", side_effect=fake_compute),
        ):
            handler_mod.handler(
                {"end_time_iso": "2026-06-06T00:00:00Z"},
                context=None,
            )

        assert captured["end_time"] == datetime(
            2026,
            6,
            6,
            0,
            0,
            tzinfo=UTC,
        )

    def test_end_time_defaults_to_none_when_unset(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", side_effect=fake_compute),
        ):
            handler_mod.handler({}, context=None)

        # None means rolling_mean will default to now-UTC.
        assert captured.get("end_time") is None

    # ── agent_quality producer wiring (config#1149 Batch A) ─────────────────
    def test_agent_quality_surfaced_in_result(self, handler_mod):
        # The previously-unwired build_agent_quality producer now runs here;
        # its artifact + graded-component list rides in the result.
        artifact = {
            "status": "ok",
            "date": "2026-06-22",
            "signal_volume_adequacy": {"value": 25, "n": 25},
            "judge_rubric_pass_rate": {"value": 0.8, "n": 30},
        }
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch("scripts.build_agent_quality.build_agent_quality", return_value=artifact),
            patch(
                "scripts.build_agent_quality.write_agent_quality", return_value="backtest/2026-06-22/agent_quality.json"
            ),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["agent_quality"]["status"] == "OK"
        assert result["agent_quality"]["key"] == "backtest/2026-06-22/agent_quality.json"
        assert set(result["agent_quality"]["graded_components"]) == {"signal_volume_adequacy", "judge_rubric_pass_rate"}

    def test_agent_quality_failure_is_non_fatal(self, handler_mod):
        # A producer failure MUST NOT change the primary rolling-mean status —
        # it is recorded in the agent_quality field instead.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch("scripts.build_agent_quality.build_agent_quality", side_effect=RuntimeError("S3 list failed")),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["agent_quality"]["status"] == "ERROR"
        assert "S3 list failed" in result["agent_quality"]["error"]

    # ── producer leaderboard wiring (config#1223 B4 / #1221 shared scorer) ────
    def test_producer_leaderboard_surfaced_in_result(self, handler_mod):
        # The producer champion/challenger leaderboard scorer now runs here as a
        # fail-soft post-step; its status + key + cohort count ride in the result.
        lb = {
            "status": "ok",
            "key": "research/producer_leaderboard/2026-06-22.json",
            "leaderboard": {"n_dates": 3, "champion": "agentic_sector_teams"},
        }
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch("scoring.leaderboard_producers.build_producer_leaderboard", return_value=lb),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["producer_leaderboard"]["status"] == "ok"
        assert result["producer_leaderboard"]["key"] == "research/producer_leaderboard/2026-06-22.json"
        assert result["producer_leaderboard"]["n_dates"] == 3

    def test_producer_leaderboard_failure_is_non_fatal(self, handler_mod):
        # A leaderboard failure MUST NOT change the primary rolling-mean status.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch(
                "scoring.leaderboard_producers.build_producer_leaderboard",
                side_effect=RuntimeError("closes read failed"),
            ),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["producer_leaderboard"]["status"] == "ERROR"
        assert "closes read failed" in result["producer_leaderboard"]["error"]


# ── alpha-engine-config-I9321 ────────────────────────────────────────────


class TestFloorUnmeasurablePropagates:
    """The handler must not convert an unmeasurable floor into a green stage.

    `{"status": "ERROR"}` is a SUCCESSFUL Lambda return, and `EvalRollingMean`
    has no Choice state reading `status` — its only success exit goes straight
    to `CheckSkipRationaleClustering`. So the handler's blanket
    `except Exception -> return {"status": "ERROR"}` made every computation
    failure indistinguishable from a clean run at the Step Function level.

    Brian, 2026-08-29: *"if an agent is downgraded from a poor answer, how am i
    notified?"* With the floor unpublished the alarm has nothing to evaluate,
    and per `champion-challenger-policy.md` §8 the judge's scores may never
    demote anything — so notification is not one response among several, it is
    the only one. A silent stage removes it entirely.
    """

    def test_floor_unmeasurable_raises_out_of_the_handler(self, handler_mod):
        from evals.rolling_mean import EvalFloorUnmeasurable

        boom = EvalFloorUnmeasurable(
            reason="no metric streams exist under the source metric",
            namespace="AlphaEngine/Eval",
            source_metric="agent_quality_score",
            combos_discovered=0,
            combos_skipped_no_data=0,
        )
        with patch(
            "evals.rolling_mean.compute_and_emit_4w_mean", side_effect=boom,
        ):
            with pytest.raises(EvalFloorUnmeasurable):
                handler_mod.handler(
                    {"end_time_iso": "2026-08-29T05:11:40Z"}, None,
                )

    def test_other_computation_failures_still_return_status_error(
        self, handler_mod,
    ):
        """The blanket catch is NARROWED, not removed.

        Only the coverage-blinding case escalates to a stage failure. A
        transient GetMetricData fault keeps the pre-existing fail-soft posture
        (eval is observability; it must not halt the Saturday pipeline), so
        this change cannot be read as making the whole stage brittle.
        """
        with patch(
            "evals.rolling_mean.compute_and_emit_4w_mean",
            side_effect=RuntimeError("throttled"),
        ):
            result = handler_mod.handler(
                {"end_time_iso": "2026-08-29T05:11:40Z"}, None,
            )
        assert result["status"] == "ERROR"
        assert "throttled" in result["error"]
