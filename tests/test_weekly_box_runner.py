"""The weekly-Research spot-EC2 box entrypoint (config#1687).

Proves the box path invokes the SAME production orchestration the Lambda
handler drives — via ``handler(event, None)`` with the SF weekly Payload —
and honors the fail-loud exit contract (non-zero on ERROR so the
``krepis.ssm_dispatcher`` poller + SF ``ExtractResearchError`` surface it).

Also mirrors ``alpha-engine-predictor/tests/test_spot_train_krepis_cli_executes.py``:
the entrypoint must EXECUTE as a CLI (parse argv, dispatch), not be an
import-only shim — a guard-less module prints nothing and exits 0 for
``--help``; a real CLI prints usage and exits 0 for ``--help`` / non-zero
for a bad flag.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENTRYPOINT = _REPO_ROOT / "infrastructure" / "weekly_box_runner.py"


def _load_module():
    """Load infrastructure/weekly_box_runner.py by path (not a package)."""
    spec = importlib.util.spec_from_file_location("weekly_box_runner_under_test", _ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def wbr():
    return _load_module()


class TestWeeklyEventShape:
    def test_build_event_mirrors_sf_research_payload(self, wbr):
        # SF ``Research`` Payload: {weekly_run, force, skip_dry_run_gate, dry_run_llm}
        ev = wbr.build_event()
        assert ev == {
            "weekly_run": True,
            "force": False,
            "skip_dry_run_gate": True,  # Lambda weekly production optimization carries over
            "dry_run_llm": False,
        }

    def test_flags_propagate(self, wbr):
        ev = wbr.build_event(force=True, skip_dry_run_gate=False, dry_run_llm=True)
        assert ev["force"] is True
        assert ev["skip_dry_run_gate"] is False
        assert ev["dry_run_llm"] is True


class _Recorder:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def handler(self, event, context):
        self.calls.append((event, context))
        return self.result


def _patch_handler(wbr, recorder):
    wbr._import_handler = lambda: types.SimpleNamespace(handler=recorder.handler)


class TestRunInvokesHandler:
    def test_invokes_handler_with_weekly_event_context_none_and_returns_0_on_ok(self, wbr):
        rec = _Recorder({"status": "OK", "date": "2026-07-03"})
        _patch_handler(wbr, rec)
        rc = wbr.run([])
        assert rc == 0
        assert len(rec.calls) == 1
        event, context = rec.calls[0]
        # Reuses production orchestration verbatim: handler never reads context.
        assert context is None
        assert event["weekly_run"] is True
        assert event["skip_dry_run_gate"] is True
        assert event["force"] is False

    def test_force_flag_threads_into_event(self, wbr):
        rec = _Recorder({"status": "OK"})
        _patch_handler(wbr, rec)
        wbr.run(["--force"])
        assert rec.calls[0][0]["force"] is True

    def test_no_skip_dry_run_gate_flag_threads_into_event(self, wbr):
        rec = _Recorder({"status": "OK"})
        _patch_handler(wbr, rec)
        wbr.run(["--no-skip-dry-run-gate"])
        assert rec.calls[0][0]["skip_dry_run_gate"] is False


class TestFailLoudExitContract:
    def test_error_status_returns_nonzero(self, wbr):
        rec = _Recorder({"status": "ERROR", "reason": "boom"})
        _patch_handler(wbr, rec)
        assert wbr.run([]) == 1

    def test_missing_status_treated_as_error(self, wbr):
        rec = _Recorder({})  # defensively, an unshaped result must NOT read as green
        _patch_handler(wbr, rec)
        assert wbr.run([]) == 1

    def test_raise_propagates(self, wbr):
        def _boom():
            m = types.SimpleNamespace()

            def handler(event, context):
                raise RuntimeError("handler blew up")

            m.handler = handler
            return m

        wbr._import_handler = _boom
        with pytest.raises(RuntimeError):
            wbr.run([])

    def test_skipped_is_benign_zero(self, wbr):
        rec = _Recorder({"status": "SKIPPED", "reason": "already_run", "date": "2026-07-03"})
        _patch_handler(wbr, rec)
        assert wbr.run([]) == 0


class TestCliExecutes:
    """Mirror of the predictor's krepis-cli-executes guard: the entrypoint
    must PARSE argv and dispatch, not import-and-fall-off-the-end."""

    def test_help_executes_and_exits_zero(self):
        out = subprocess.run(
            [sys.executable, str(_ENTRYPOINT), "--help"],
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0
        assert "weekly_box_runner" in out.stdout
        assert "--preflight-only" in out.stdout

    def test_bad_flag_exits_nonzero(self):
        out = subprocess.run(
            [sys.executable, str(_ENTRYPOINT), "--definitely-not-a-flag"],
            capture_output=True,
            text=True,
        )
        assert out.returncode != 0
