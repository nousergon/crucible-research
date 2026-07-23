"""Unit tests for the eval-judge Lambda handler (PR 3b of P3.1).

Covers:
- ``handler`` end-to-end with a stubbed orchestrator: status code
  derivation (OK / PARTIAL / ERROR), event parameter mapping, default
  date fallback.
- The orchestrator itself is exercised in ``test_eval_orchestrator``;
  these tests pin only the thin handler-shaped contract that the
  Saturday SF state will rely on.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest


def _ok_summary() -> dict:
    return {
        "date": "2026-05-09",
        "capture_keys_total": 4,
        "haiku_evaluated": 4,
        "sonnet_evaluated": 0,
        "skipped_unmapped": 1,
        "metric_emission_failures": 0,
        "failed": [],
        "persisted_keys": ["k1", "k2", "k3", "k4"],
        "haiku_model": "claude-haiku-4-5",
        "sonnet_model": "claude-sonnet-4-6",
        "force_sonnet_pass": False,
        "dry_run": False,
        "judge_only": False,
        "eval_prefix": "decision_artifacts/_eval/",
        "cw_namespace": "AlphaEngine/Eval",
        "would_evaluate": [],
    }


def _partial_summary() -> dict:
    s = _ok_summary()
    s["failed"] = [{
        "key": "decision_artifacts/2026/05/09/macro_economist/r.json",
        "agent_id": "macro_economist",
        "stage": "haiku",
        "error": "anthropic 5xx",
    }]
    return s


# ``lambda`` is a Python keyword so the Lambda directory can't be
# imported as a normal package. The handler is loaded by file path
# via importlib in the ``handler_mod`` fixture below.


import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "eval_judge_handler.py"


def _load_handler_module():
    """Import lambda/eval_judge_handler.py without using ``lambda`` as
    a package name (it's a Python keyword)."""
    module_name = "lambda_eval_judge_handler"
    spec = importlib.util.spec_from_file_location(module_name, _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def handler_mod():
    """Fresh handler module per test — ``_init_done`` resets, and
    monkey-patches don't leak across tests."""
    mod = _load_handler_module()
    mod._init_done = False
    yield mod


# ── handler ───────────────────────────────────────────────────────────────


class TestHandler:
    def test_ok_when_no_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.orchestrator.evaluate_corpus", return_value=_ok_summary()):
            result = handler_mod.handler(
                {"date": "2026-05-09"}, context=None,
            )
        assert result["status"] == "OK"
        assert result["summary"]["haiku_evaluated"] == 4
        assert result["summary"]["sonnet_evaluated"] == 0

    def test_partial_when_any_failure(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.orchestrator.evaluate_corpus", return_value=_partial_summary()):
            result = handler_mod.handler(
                {"date": "2026-05-09"}, context=None,
            )
        assert result["status"] == "PARTIAL"
        assert len(result["summary"]["failed"]) == 1

    def test_error_when_orchestrator_raises(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.orchestrator.evaluate_corpus",
                   side_effect=RuntimeError("S3 listing blew up")):
            result = handler_mod.handler(
                {"date": "2026-05-09"}, context=None,
            )
        assert result["status"] == "ERROR"
        assert "S3 listing blew up" in result["error"]
        assert "summary" not in result

    def test_default_date_falls_back_to_today_utc(self, handler_mod):
        captured = {}

        def fake_corpus(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.orchestrator.evaluate_corpus", side_effect=fake_corpus):
            handler_mod.handler({}, context=None)

        # Default == today UTC formatted YYYY-MM-DD
        today = str(datetime.date.today())
        assert captured["date"] == today

    def test_event_passes_through_to_orchestrator(self, handler_mod):
        captured = {}

        def fake_corpus(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.orchestrator.evaluate_corpus", side_effect=fake_corpus):
            handler_mod.handler({
                "date": "2026-05-09",
                "force_sonnet_pass": True,
                "haiku_model": "claude-haiku-4-5-test",
                "sonnet_model": "claude-sonnet-4-6-test",
                "haiku_escalate_threshold": 4,
            }, context=None)

        assert captured["date"] == "2026-05-09"
        assert captured["force_sonnet_pass"] is True
        assert captured["haiku_model"] == "claude-haiku-4-5-test"
        assert captured["sonnet_model"] == "claude-sonnet-4-6-test"
        assert captured["haiku_escalate_threshold"] == 4

    def test_dry_run_and_judge_only_flags_default_false(self, handler_mod):
        """Production Saturday SF invocations don't pass dry_run /
        judge_only — both must default to False so the SF path stays
        on the prod track."""
        captured = {}

        def fake_corpus(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.orchestrator.evaluate_corpus", side_effect=fake_corpus):
            handler_mod.handler({"date": "2026-05-09"}, context=None)

        assert captured["dry_run"] is False
        assert captured["judge_only"] is False

    def test_dry_run_and_judge_only_flags_pass_through(self, handler_mod):
        captured = {}

        def fake_corpus(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("evals.orchestrator.evaluate_corpus", side_effect=fake_corpus):
            handler_mod.handler({
                "date": "2026-05-09",
                "dry_run": True,
                "judge_only": True,
            }, context=None)

        assert captured["dry_run"] is True
        assert captured["judge_only"] is True
