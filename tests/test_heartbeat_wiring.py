"""Tests for the end-of-run flow-doctor heartbeat wire-up (config#646).

The handler success path calls ``_emit_flow_doctor_heartbeat()`` right
before returning ``{"status": "OK", ...}``. The helper resolves the
flow-doctor singleton via ``get_flow_doctor()`` and, when active, calls
``fd.emit_heartbeat(bucket=<research bucket>)`` so the console System
Health consumer can read the daily research producer's end-of-run
status() snapshot from the **research** bucket.

The console reads heartbeats from ``alpha-engine-research``, so the write
MUST target that bucket. These tests lock both the bucket and the
soft-fail / inactive-flow-doctor posture at the helper surface (mirrors
test_scorecard_lambda_wiring's importlib approach so the heavy graph
imports aren't paid to exercise a tail-of-run wire-up).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_handler():
    """Import lambda/handler.py — the dir name collides with the `lambda`
    Python keyword, so ``from lambda import handler`` is a SyntaxError.
    Load it via importlib instead (same shim as
    test_scorecard_lambda_wiring)."""
    import importlib.util
    handler_path = Path(__file__).parent.parent / "lambda" / "handler.py"
    spec = importlib.util.spec_from_file_location(
        "research_handler_under_test", handler_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def handler_mod():
    return _import_handler()


def test_heartbeat_emitted_to_research_bucket(handler_mod, monkeypatch):
    """When flow-doctor is active, the helper emits the heartbeat to the
    research bucket. Removing the wire-up (or pointing it at the wrong
    bucket) makes the console System Health panel go stale for the daily
    research producer."""
    monkeypatch.delenv("RESEARCH_BUCKET", raising=False)  # exercise default
    mock_fd = MagicMock()
    with patch.object(handler_mod, "get_flow_doctor", return_value=mock_fd):
        handler_mod._emit_flow_doctor_heartbeat()
    mock_fd.emit_heartbeat.assert_called_once_with(bucket="alpha-engine-research")


def test_heartbeat_respects_research_bucket_override(handler_mod, monkeypatch):
    """The bucket comes from the same ``RESEARCH_BUCKET`` override every
    other artifact in this handler uses, defaulting to alpha-engine-research."""
    monkeypatch.setenv("RESEARCH_BUCKET", "alpha-engine-research-override")
    mock_fd = MagicMock()
    with patch.object(handler_mod, "get_flow_doctor", return_value=mock_fd):
        handler_mod._emit_flow_doctor_heartbeat()
    mock_fd.emit_heartbeat.assert_called_once_with(
        bucket="alpha-engine-research-override"
    )


def test_heartbeat_noop_when_flow_doctor_inactive(handler_mod):
    """No flow-doctor singleton (local dev / disabled) -> the helper is a
    clean no-op and never raises."""
    with patch.object(handler_mod, "get_flow_doctor", return_value=None):
        # Must not raise.
        handler_mod._emit_flow_doctor_heartbeat()


def test_heartbeat_noop_when_lib_lacks_emit_heartbeat(handler_mod):
    """Version-skew safety (config#646): emit_heartbeat only exists in
    flow-doctor >=0.6.2. A skewed deploy pinning an older lib returns a
    singleton WITHOUT the method — the hasattr guard must make that a
    silent no-op, never an AttributeError that crashes end-of-run."""
    class _OldFlowDoctor:
        pass  # no emit_heartbeat — mimics <0.6.2

    with patch.object(handler_mod, "get_flow_doctor", return_value=_OldFlowDoctor()):
        # Must not raise (would AttributeError without the hasattr guard).
        handler_mod._emit_flow_doctor_heartbeat()


def test_success_path_calls_heartbeat_helper():
    """Source-text lock: the ``{"status": "OK"}`` success return is
    immediately preceded by the ``_emit_flow_doctor_heartbeat()`` call, so
    a refactor can't silently drop the heartbeat from the tail of a
    successful run."""
    src = (Path(__file__).parent.parent / "lambda" / "handler.py").read_text()
    call_idx = src.find("_emit_flow_doctor_heartbeat()\n\n        return {")
    assert call_idx != -1, (
        "the success-return path must call _emit_flow_doctor_heartbeat() "
        "immediately before returning {'status': 'OK', ...} (config#646)"
    )
