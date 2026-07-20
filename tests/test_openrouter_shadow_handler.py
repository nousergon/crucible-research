"""Unit tests for the OpenRouter shadow-judge scheduled Lambda handler
(alpha-engine-config#2934).

Pins the handler's load-bearing contracts, mirroring
test_thinktank_handler.py's structure for the same handler family:

1. **Dry path** — ``dry_run_llm`` (shell-run smoke) returns before any
   run_shadow_judge_over_date call — no S3/CloudWatch access, no LLM
   calls.
2. **Argument translation** — the event's ``date``/``bucket`` (or their
   defaults) reach ``run_shadow_judge_over_date`` correctly; a missing
   ``date`` defaults to yesterday UTC (the schedule fires the morning
   after the Saturday capture partition it targets closes).
3. **Raise-on-failure** — the handler must PROPAGATE exceptions, never
   convert them to an ERROR-dict return. This Lambda is invoked async by
   EventBridge (no SF Catch above it): an ERROR-dict return counts as a
   *successful* invocation, so the AWS/Lambda Errors metric stays flat,
   no async retry fires, and the run fails silently — the exact bug
   class the setup-openrouter-shadow-schedule.sh alarm (Errors >= 3 /
   week) exists to page on.
"""

from __future__ import annotations

import datetime
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "openrouter_shadow_handler.py"


def _load_handler_module():
    """Import lambda/openrouter_shadow_handler.py without using
    ``lambda`` as a package name (Python keyword). Mirrors
    test_thinktank_handler.py / test_scanner_handler.py."""
    module_name = "lambda_openrouter_shadow_handler"
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
    mod._init_done = False


def _summary_mock(**overrides):
    summary = {
        "date": "2026-07-18",
        "judge_model": "openrouter-shadow",
        "judge_run_id": "run-abc123",
        "capture_keys_seen": 12,
        "evaluated": 10,
        "skipped_unmapped": 1,
        "skipped_empty_or_degenerate": 1,
        "failed": [],
        "metric_emission_failures": 0,
        "persisted_keys": ["decision_artifacts/_eval/2026-07-18/x.json"],
        "shadow_only": True,
    }
    summary.update(overrides)
    return summary


class TestDryPath:
    def test_shell_run_dry_short_circuits_before_run(self, handler_mod):
        """dry_run_llm must return before _ensure_init / the runner call —
        no S3/CloudWatch, no LLM. The Friday shell-run keystone contract
        (shared evals.lambda_dry.is_dry substrate)."""
        with patch.object(handler_mod, "_ensure_init") as init, \
             patch("evals.openrouter_shadow.run_shadow_judge_over_date") as run:
            result = handler_mod.handler({"dry_run_llm": True}, None)
        assert result == {"status": "OK", "dry_run": True}
        init.assert_not_called()
        run.assert_not_called()


class TestArgumentTranslation:
    def test_explicit_date_and_bucket_passed_through(self, handler_mod):
        summary = _summary_mock()
        with patch.object(handler_mod, "_ensure_init"), \
             patch(
                 "evals.openrouter_shadow.run_shadow_judge_over_date",
                 return_value=summary,
             ) as run:
            result = handler_mod.handler(
                {"date": "2026-07-18", "bucket": "some-other-bucket"}, None
            )
        run.assert_called_once_with(date="2026-07-18", bucket="some-other-bucket")
        assert result == {"status": "OK", "summary": summary}

    def test_missing_date_defaults_to_yesterday_utc(self, handler_mod):
        """No explicit date -> yesterday UTC, since the Sunday schedule
        targets Saturday's just-closed capture partition."""
        summary = _summary_mock()
        expected_date = str(datetime.date.today() - datetime.timedelta(days=1))
        with patch.object(handler_mod, "_ensure_init"), \
             patch(
                 "evals.openrouter_shadow.run_shadow_judge_over_date",
                 return_value=summary,
             ) as run:
            handler_mod.handler({}, None)
        run.assert_called_once_with(date=expected_date, bucket="alpha-engine-research")

    def test_non_dict_event_treated_as_real_run_with_defaults(self, handler_mod):
        """EventBridge scheduled events are dicts, but a None/odd payload
        from a manual invoke must not crash the flag probe or arg
        resolution."""
        summary = _summary_mock()
        expected_date = str(datetime.date.today() - datetime.timedelta(days=1))
        with patch.object(handler_mod, "_ensure_init"), \
             patch(
                 "evals.openrouter_shadow.run_shadow_judge_over_date",
                 return_value=summary,
             ) as run:
            result = handler_mod.handler(None, None)
        run.assert_called_once_with(date=expected_date, bucket="alpha-engine-research")
        assert result["status"] == "OK"

    def test_missing_bucket_defaults_to_env_or_constant(self, handler_mod, monkeypatch):
        monkeypatch.setenv("RESEARCH_BUCKET", "env-bucket")
        summary = _summary_mock()
        with patch.object(handler_mod, "_ensure_init"), \
             patch(
                 "evals.openrouter_shadow.run_shadow_judge_over_date",
                 return_value=summary,
             ) as run:
            handler_mod.handler({"date": "2026-07-18"}, None)
        run.assert_called_once_with(date="2026-07-18", bucket="env-bucket")


class TestRaiseOnFailure:
    def test_run_shadow_judge_failure_propagates(self, handler_mod):
        """The core contract: NO error-dict conversion. See module doc."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch(
                 "evals.openrouter_shadow.run_shadow_judge_over_date",
                 side_effect=RuntimeError("boom"),
             ):
            with pytest.raises(RuntimeError, match="boom"):
                handler_mod.handler({"date": "2026-07-18"}, None)

    def test_handler_source_never_returns_error_status(self):
        """Belt-and-suspenders source pin: the SF-handler idiom
        ``return {"status": "ERROR", ...}`` must never appear here — an
        async-invoked Lambda that returns ERROR is a silent failure."""
        import re

        text = _HANDLER_PATH.read_text(encoding="utf-8")
        assert not re.search(r'return\s*\{\s*"status":\s*"ERROR"', text)


class TestColdStartInit:
    def test_init_runs_once(self, handler_mod):
        assert handler_mod._init_done is False
        handler_mod._ensure_init()
        assert handler_mod._init_done is True
        # Second call must be a no-op (no exception, no state change).
        handler_mod._ensure_init()
        assert handler_mod._init_done is True

    def test_init_sets_xdg_cache_home(self, handler_mod, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        handler_mod._ensure_init()
        import os

        assert os.environ["XDG_CACHE_HOME"] == "/tmp"
