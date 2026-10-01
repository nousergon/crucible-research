"""Pins the on-box deadline wiring (config-I5208 / nous-ergon-ops-I162).

The migration off Lambda is only correct if the box supplies a deadline. The
naive box entry ``handler(event, None)`` — which is what nous-ergon-ops-I162
originally specified — leaves ``seconds_remaining=None``, and
``thinktank.run._out_of_time`` reads that as "never truncate". The run then
executes unbounded until the SSM execution timeout or a spot reclaim kills it
mid-loop, discarding every terminal write: the exact 2026-07-17 failure this
migration exists to fix, reproduced on the hardware chosen to fix it.

These tests exist so that regression cannot land silently.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_runner():
    path = os.path.join(_REPO_ROOT, "infrastructure", "thinktank_box_runner.py")
    spec = importlib.util.spec_from_file_location("thinktank_box_runner", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["thinktank_box_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


class _StubWatcher:
    def __init__(self, interrupted: bool = False) -> None:
        self.interrupted = interrupted


def test_box_context_exposes_the_lambda_context_duck_type():
    """``thinktank_handler.handler`` gates on this exact attribute name; if it
    is missing the deadline silently stays ``None`` and the guard disarms."""
    ctx = runner.BoxContext(100.0, _StubWatcher())
    assert hasattr(ctx, "get_remaining_time_in_millis")
    assert ctx.get_remaining_time_in_millis() > 0


def test_remaining_time_decreases_and_floors_at_zero():
    ctx = runner.BoxContext(0.0, _StubWatcher())
    assert ctx.get_remaining_time_in_millis() == 0.0


def test_spot_interruption_collapses_remaining_time_immediately():
    """A reclaim notice must drive ``_out_of_time`` true on the next check so
    the terminal writes happen inside EC2's ~120s window — a budget-only clock
    would still be reporting hours remaining when the box disappears."""
    watcher = _StubWatcher(interrupted=False)
    ctx = runner.BoxContext(10_000.0, watcher)
    assert ctx.get_remaining_time_in_millis() > 120_000
    watcher.interrupted = True
    assert ctx.get_remaining_time_in_millis() == 0.0


def test_collapsed_remaining_time_trips_the_run_module_reserve():
    """Wire the real predicate, not a restatement of it: the collapsed clock
    must actually satisfy ``thinktank.run._out_of_time``."""
    sys.path.insert(0, _REPO_ROOT)
    from thinktank.run import _TERMINAL_WRITE_RESERVE_S, _out_of_time

    watcher = _StubWatcher(interrupted=True)
    ctx = runner.BoxContext(10_000.0, watcher)
    seconds_remaining = lambda: ctx.get_remaining_time_in_millis() / 1000.0  # noqa: E731
    assert _out_of_time(seconds_remaining) is True
    assert _TERMINAL_WRITE_RESERVE_S >= 120.0


def test_zero_or_absent_budget_raises_rather_than_running_unbounded(monkeypatch):
    """A missing/zero budget must fail loud. Defaulting to "no deadline" is the
    disarmed state this whole module exists to prevent."""
    monkeypatch.setenv("THINKTANK_RUN_BUDGET_SECONDS", "0")
    with pytest.raises(ValueError, match="disarms the deadline guard"):
        runner.main()


def test_imds_404_is_the_steady_state_not_an_interruption():
    """A 404 from the spot/instance-action endpoint means "no reclaim
    scheduled" — misreading it as a notice would truncate every run instantly."""
    import urllib.error

    watcher = runner.SpotInterruptionWatcher()

    def _raise_404(*_args, **_kwargs):
        raise urllib.error.HTTPError(runner._IMDS_ACTION_URL, 404, "Not Found", hdrs=None, fp=None)

    original = runner.urllib.request.urlopen
    runner.urllib.request.urlopen = _raise_404
    try:
        assert watcher._notice_present() is False
        assert watcher.interrupted is False
    finally:
        runner.urllib.request.urlopen = original


def test_imds_failure_degrades_to_budget_only_never_to_no_deadline():
    """A broken IMDS must not resurrect the run's remaining time, and must not
    latch a false interruption — the budget stays the independent floor."""
    watcher = runner.SpotInterruptionWatcher()

    def _boom(*_args, **_kwargs):
        raise OSError("imds unreachable")

    original = runner.urllib.request.urlopen
    runner.urllib.request.urlopen = _boom
    try:
        assert watcher._notice_present() is False
    finally:
        runner.urllib.request.urlopen = original

    ctx = runner.BoxContext(600.0, watcher)
    assert ctx.get_remaining_time_in_millis() > 0


# ── run mode (alpha-engine-config-I11378) ───────────────────────────────────


@pytest.mark.parametrize("mode", [None, "", "  ", "daily"])
def test_absent_or_daily_mode_runs_the_daily_pass(mode):
    """The daily EventBridge rule sends no mode; its box must keep running the
    exact event it always ran (``{}``)."""
    assert runner.resolve_event(mode) == {}


def test_gap_fill_mode_reaches_the_handler_as_gap_fill():
    """``thinktank_handler`` reads ``event["mode"] == "gap_fill"`` to set
    ``gap_fill_only`` -- the weekly SF's bulk pass over the pinned window."""
    assert runner.resolve_event("gap_fill") == {"mode": "gap_fill"}


@pytest.mark.parametrize("mode", ["gapfill", "gap_fill_plan", "weekly", "GAP_FILL"])
def test_unknown_mode_refuses_rather_than_running_daily(mode):
    with pytest.raises(ValueError, match="THINKTANK_RUN_MODE"):
        runner.resolve_event(mode)


def test_main_passes_the_resolved_event_to_the_handler(monkeypatch):
    """End to end through ``main``: the env var is what the dispatcher sets."""
    seen = {}

    def _fake_handler(event, context):
        seen["event"] = event
        seen["has_clock"] = hasattr(context, "get_remaining_time_in_millis")
        return {"status": "OK"}

    fake = type(sys)("thinktank_handler")
    fake.handler = _fake_handler
    monkeypatch.setitem(sys.modules, "thinktank_handler", fake)

    class _Watcher(_StubWatcher):
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(runner, "SpotInterruptionWatcher", _Watcher)
    monkeypatch.setenv("THINKTANK_RUN_BUDGET_SECONDS", "60")
    monkeypatch.setenv("THINKTANK_RUN_MODE", "gap_fill")
    assert runner.main() == 0
    assert seen == {"event": {"mode": "gap_fill"}, "has_clock": True}
