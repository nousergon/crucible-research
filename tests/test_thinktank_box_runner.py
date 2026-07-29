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
