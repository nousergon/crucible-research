"""Unit tests for the think-tank Lambda handler (config#1579 P1).

Pins the handler's three load-bearing contracts:

1. **Dry paths** — ``dry_run_llm`` (shell-run smoke) returns before secrets
   hydration / any S3 access; ``dry_run`` routes to the plan-only mode.
2. **Raise-on-failure** — the handler must PROPAGATE exceptions, never
   convert them to an ERROR-dict return. Invoked by the Saturday SF's
   ``ThinkTankCoverage`` Task (``arn:aws:states:::lambda:invoke``): the
   SF Catch only triggers on an actual raised Lambda error — a normal
   return value (even an error-shaped dict) is a *successful* Task
   completion, so the non-blocking Catch would never fire and the run
   would fail silently — the exact bug class the
   setup-thinktank-alarm.sh alarm (Errors >= 2/day) exists to page on.
3. **Cold-start hydration** — RAG_DATABASE_URL + VOYAGE_API_KEY are BOTH
   hydrated from the get_secret chokepoint (the 2026-07-02 first-run
   gotcha: the RAG availability probe passes on DB URL alone while every
   per-ticker retrieve fails without the Voyage key), and decision
   capture is forced on (judge coverage requires it).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "thinktank_handler.py"


def _load_handler_module():
    """Import lambda/thinktank_handler.py without using ``lambda`` as a
    package name (Python keyword). Mirrors test_scanner_handler.py."""
    module_name = "lambda_thinktank_handler"
    spec = importlib.util.spec_from_file_location(module_name, _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def handler_mod(monkeypatch):
    mod = _load_handler_module()
    mod._init_done = False
    # Never let a unit test hit SSM: tests that exercise the non-dry path
    # patch get_secret explicitly; this guard makes an unpatched escape
    # loud instead of a network call.
    monkeypatch.delenv("RAG_DATABASE_URL", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
    yield mod
    mod._init_done = False


def _manifest_mock():
    manifest = MagicMock()
    manifest.run_id = "abc123def456"
    manifest.mode = "daily"
    manifest.trading_day = "2026-07-02"
    manifest.theses_written = 5
    manifest.sweep_tickers = 10
    manifest.theme_updates_written = 0
    manifest.total_cost_usd = 0.0034
    manifest.budget_month_spent_usd = 0.05
    manifest.budget_month_limit_usd = 25.0
    manifest.model_dump.return_value = {"run_id": "abc123def456", "mode": "daily"}
    return manifest


class TestDryPaths:
    def test_shell_run_dry_short_circuits_before_secrets(self, handler_mod):
        """dry_run_llm must return before _ensure_init — no SSM fetch,
        no S3, no LLM. The Friday shell-run keystone contract."""
        with patch.object(handler_mod, "_ensure_init") as init:
            result = handler_mod.handler({"dry_run_llm": True}, None)
        assert result == {"status": "OK", "dry_run": True}
        init.assert_not_called()

    def test_plan_only_dry_run_routes_to_run_daily_dry(self, handler_mod):
        manifest = _manifest_mock()
        manifest.mode = "dry_run"
        with patch.object(handler_mod, "_ensure_init"), \
             patch("thinktank.run.run_daily", return_value=manifest) as run:
            result = handler_mod.handler({"dry_run": True}, None)
        run.assert_called_once_with(dry_run=True, refresh_tickers=None)
        assert result["status"] == "OK"


class TestSuccessPath:
    def test_ok_returns_manifest_dump(self, handler_mod):
        manifest = _manifest_mock()
        with patch.object(handler_mod, "_ensure_init"), \
             patch("thinktank.run.run_daily", return_value=manifest) as run:
            result = handler_mod.handler({}, None)
        run.assert_called_once_with(dry_run=False, refresh_tickers=None)
        assert result == {
            "status": "OK",
            "manifest": {"run_id": "abc123def456", "mode": "daily"},
        }

    def test_non_dict_event_treated_as_real_run(self, handler_mod):
        """EventBridge scheduled events are dicts, but a None/odd payload
        from a manual invoke must not crash the flag probe."""
        manifest = _manifest_mock()
        with patch.object(handler_mod, "_ensure_init"), \
             patch("thinktank.run.run_daily", return_value=manifest) as run:
            result = handler_mod.handler(None, None)
        run.assert_called_once_with(dry_run=False, refresh_tickers=None)
        assert result["status"] == "OK"


class TestRaiseOnFailure:
    def test_run_daily_failure_propagates(self, handler_mod):
        """The core contract: NO error-dict conversion. See module doc."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("thinktank.run.run_daily", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                handler_mod.handler({}, None)

    def test_handler_source_never_returns_error_status(self):
        """Belt-and-suspenders source pin: the SF-handler idiom
        ``return {"status": "ERROR", ...}`` must never appear here —
        an async-invoked Lambda that returns ERROR is a silent failure."""
        import re

        text = _HANDLER_PATH.read_text(encoding="utf-8")
        assert not re.search(r'return\s*\{\s*"status":\s*"ERROR"', text)


class TestColdStartHydration:
    def test_hydrates_both_rag_secrets_and_capture_flag(
        self, handler_mod, monkeypatch
    ):
        secrets = {"RAG_DATABASE_URL": "postgres://x", "VOYAGE_API_KEY": "vk"}
        with patch(
            "nousergon_lib.secrets.get_secret", side_effect=secrets.__getitem__
        ) as get_secret:
            handler_mod._ensure_init()
        assert {c.args[0] for c in get_secret.call_args_list} == set(secrets)
        import os

        assert os.environ["RAG_DATABASE_URL"] == "postgres://x"
        assert os.environ["VOYAGE_API_KEY"] == "vk"
        assert os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] == "true"

    def test_existing_env_values_not_refetched(self, handler_mod, monkeypatch):
        monkeypatch.setenv("RAG_DATABASE_URL", "postgres://already")
        monkeypatch.setenv("VOYAGE_API_KEY", "already")
        with patch("nousergon_lib.secrets.get_secret") as get_secret:
            handler_mod._ensure_init()
        get_secret.assert_not_called()

    def test_init_runs_once(self, handler_mod):
        with patch(
            "nousergon_lib.secrets.get_secret", return_value="v"
        ) as get_secret:
            handler_mod._ensure_init()
            handler_mod._ensure_init()
        assert get_secret.call_count == 2  # two secrets, fetched exactly once
