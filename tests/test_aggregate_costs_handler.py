"""Unit tests for the aggregate_costs Lambda handler (ROADMAP L1146)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date as date_type
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "aggregate_costs_handler.py"


def _load_handler_module():
    """Import lambda/aggregate_costs_handler.py without using ``lambda``
    as a package name (Python keyword)."""
    module_name = "lambda_aggregate_costs_handler"
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
        "rows_in": 1234,
        "files_read": 87,
        "output_key": "decision_artifacts/_cost/2026-05-25/cost.parquet",
        "total_cost_usd": 12.3456,
        "total_input_tokens": 5_000_000,
        "total_output_tokens": 300_000,
        "total_cache_read_tokens": 200_000,
        "total_cache_create_tokens": 50_000,
        "total_web_search_requests": 0,
        "total_web_fetch_requests": 0,
        "by_sector_team": {"tech": 4.0, "financials": 3.5},
        "by_model": {"claude-sonnet-4-6": 8.0, "claude-haiku-4-5": 4.0},
        "by_run_type": {"saturday_research": 12.0},
        "by_agent_id": {},
    }


class TestHandler:
    def test_ok_when_aggregate_returns_summary(self, handler_mod):
        # Real boto3 clients are constructed inside the handler before
        # aggregate_day is called; patch the boto3 module wholesale to
        # avoid network credential errors in CI.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("scripts.aggregate_costs.aggregate_day",
                   return_value=_ok_summary()), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler(
                {"date": "2026-05-25"}, context=None,
            )
        assert result["status"] == "OK"
        assert result["summary"]["rows_in"] == 1234
        assert result["date"] == "2026-05-25"

    def test_skipped_when_aggregate_returns_none(self, handler_mod):
        # aggregate_day returns None for "no JSONL files for the date" —
        # legitimate upstream no-op. Per L3277 audit + data #295 pattern
        # the SF consumer (canary or task state) must accept SKIPPED.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("scripts.aggregate_costs.aggregate_day",
                   return_value=None), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler(
                {"date": "2026-05-25"}, context=None,
            )
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "no_cost_raw_for_date"
        assert result["date"] == "2026-05-25"

    def test_error_when_aggregate_raises(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("scripts.aggregate_costs.aggregate_day",
                   side_effect=RuntimeError("S3 unreachable")), \
             patch("boto3.client", return_value=MagicMock()):
            result = handler_mod.handler(
                {"date": "2026-05-25"}, context=None,
            )
        assert result["status"] == "ERROR"
        assert "S3 unreachable" in result["error"]

    def test_error_when_date_missing(self, handler_mod):
        # Hard contract: SF state MUST thread `state.run_date` into the
        # event. Empty event triggers an explicit ERROR with a clear
        # message rather than a silent default to "today" (which would
        # silently aggregate the wrong partition on a recovery SF that
        # re-runs an older date).
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "date" in result["error"]

    def test_error_when_date_invalid(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler(
                {"date": "not-a-date"}, context=None,
            )
        assert result["status"] == "ERROR"
        assert "not-a-date" in result["error"]

    def test_dry_run_short_circuits_before_s3(self, handler_mod):
        # dry_run_llm shell-run path must NOT touch S3 or call
        # aggregate_day. Mirrors the rationale_clustering / eval_judge
        # dry path used by Friday-Preflight shell runs.
        with patch.object(handler_mod, "_ensure_init"), \
             patch("scripts.aggregate_costs.aggregate_day") as mock_agg, \
             patch("boto3.client") as mock_boto:
            result = handler_mod.handler(
                {"dry_run_llm": True, "date": "2026-05-25"},
                context=None,
            )
        assert result["status"] == "OK"
        assert result["dry_run"] is True
        mock_agg.assert_not_called()
        mock_boto.assert_not_called()

    def test_target_date_threaded_through(self, handler_mod):
        # The handler must pass the parsed date_type instance through —
        # not the raw string. aggregate_day's signature requires
        # date_type so a string would TypeError at the call site.
        captured = {}

        def fake_aggregate(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("scripts.aggregate_costs.aggregate_day",
                   side_effect=fake_aggregate), \
             patch("boto3.client", return_value=MagicMock()):
            handler_mod.handler({"date": "2026-05-25"}, context=None)

        assert captured["target_date"] == date_type(2026, 5, 25)
        # Bucket defaults to the configured RESEARCH_BUCKET env var
        # (or the fallback constant) — confirms the kwarg is wired.
        assert captured["bucket"] in (
            "alpha-engine-research",
            handler_mod._DEFAULT_BUCKET,
        )

    def test_bucket_override(self, handler_mod):
        captured = {}

        def fake_aggregate(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("scripts.aggregate_costs.aggregate_day",
                   side_effect=fake_aggregate), \
             patch("boto3.client", return_value=MagicMock()):
            handler_mod.handler(
                {"date": "2026-05-25", "bucket": "test-bucket"},
                context=None,
            )

        assert captured["bucket"] == "test-bucket"
