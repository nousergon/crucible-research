"""Tests for the archive_writer secondary-work deadline guard.

Context — 2026-06-06 Saturday SF: research's full graph ran to ~14.9 of
the 15-min Lambda ceiling (sector teams alone ate ~13 min). signals.json
was persisted, but the Lambda was SIGKILL'd seconds later while extracting
semantic memories (unbounded LLM call), so the Step Function saw a TIMEOUT
and failed the Research branch for a run whose primary deliverable had
already landed — and the kill preceded the "must not miss" scanner_eval
logging.

The guard skips the lowest-priority best-effort semantic extraction once a
wall-clock budget (measured from run_time) is exhausted, so the run returns
OK with signals delivered. It fails OPEN (work still runs) when disabled or
when run_time is unparseable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from graph.research_graph import (
    _run_secondary_within_budget,
    _SECONDARY_DEADLINE_DEFAULT_S,
    _SECONDARY_DEADLINE_ENV,
    _secondary_deadline_budget_s,
    _secondary_work_deadline_exhausted,
)


def _state_started_seconds_ago(seconds: float) -> dict:
    start = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return {"run_time": start.isoformat()}


class TestBudgetParsing:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(_SECONDARY_DEADLINE_ENV, raising=False)
        assert _secondary_deadline_budget_s() == _SECONDARY_DEADLINE_DEFAULT_S

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "600")
        assert _secondary_deadline_budget_s() == 600.0

    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "not-a-number")
        assert _secondary_deadline_budget_s() == _SECONDARY_DEADLINE_DEFAULT_S


class TestDeadlineGuard:
    def test_fresh_run_not_exhausted(self, monkeypatch):
        monkeypatch.delenv(_SECONDARY_DEADLINE_ENV, raising=False)
        hit, elapsed = _secondary_work_deadline_exhausted(_state_started_seconds_ago(10))
        assert hit is False
        assert elapsed >= 0

    def test_slow_run_exhausted(self, monkeypatch):
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "780")
        hit, elapsed = _secondary_work_deadline_exhausted(_state_started_seconds_ago(890))
        assert hit is True
        assert elapsed >= 780

    def test_disabled_budget_never_exhausts(self, monkeypatch):
        """Budget <= 0 disables the guard — secondary work always runs."""
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "0")
        hit, _ = _secondary_work_deadline_exhausted(_state_started_seconds_ago(100000))
        assert hit is False

    def test_unparseable_run_time_fails_open(self, monkeypatch):
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "780")
        hit, elapsed = _secondary_work_deadline_exhausted({"run_time": "not-a-date"})
        assert hit is False
        assert elapsed == 0.0

    def test_missing_run_time_fails_open(self, monkeypatch):
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "780")
        hit, elapsed = _secondary_work_deadline_exhausted({})
        assert hit is False

    def test_naive_run_time_treated_as_utc(self, monkeypatch):
        """run_time without tzinfo must not raise (defensive)."""
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "780")
        naive = (datetime.now(timezone.utc) - timedelta(seconds=890)).replace(tzinfo=None)
        hit, elapsed = _secondary_work_deadline_exhausted({"run_time": naive.isoformat()})
        assert hit is True


class TestSecondaryWithinBudgetChokepoint:
    """The single chokepoint archive_writer's UNBOUNDED tail tasks route
    through. Regression for the 2026-07-03 SF TIMEOUT: the attractiveness
    trajectory (an unbounded ArcticDB-read + digest-email tail task) was NOT
    deadline-gated, so on a tail-latency-slow run it was SIGKILL'd at the 900s
    Lambda ceiling — returning a spurious States.Timeout to the Step Function
    for a run whose signals.json had already landed AND starving the
    must-not-miss upload_db that follows. Routing every unbounded tail task
    through this helper means a slow run SKIPS the observability instead.
    """

    def test_runs_fn_and_returns_result_when_budget_remains(self, monkeypatch):
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "780")
        calls = []

        def fn():
            calls.append(1)
            return "artifact-key"

        ran, result = _run_secondary_within_budget(
            _state_started_seconds_ago(10), "unit task", fn,
        )
        assert ran is True
        assert result == "artifact-key"
        assert calls == [1]

    def test_deadline_exhausted_skips_fn_entirely(self, monkeypatch):
        """The whole point: on a slow run fn must NOT be invoked, so no
        unbounded work can push the Lambda past 900s. fn raises if called."""
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "780")

        def fn():
            raise AssertionError("fn must not run once the budget is exhausted")

        ran, result = _run_secondary_within_budget(
            _state_started_seconds_ago(890), "unit task", fn,
        )
        assert ran is False
        assert result is None

    def test_fn_exception_is_swallowed_fail_soft(self, monkeypatch):
        """Best-effort observability must never fail a delivered-signals run:
        an exception from fn is swallowed and reported as ran=True, result=None
        (so the must-not-miss finalize still proceeds)."""
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "780")

        def fn():
            raise RuntimeError("arcticdb read blew up")

        ran, result = _run_secondary_within_budget(
            _state_started_seconds_ago(10), "unit task", fn,
        )
        assert ran is True
        assert result is None

    def test_disabled_budget_still_runs_fn(self, monkeypatch):
        """Budget<=0 disables the gate (fails OPEN) — fn always runs."""
        monkeypatch.setenv(_SECONDARY_DEADLINE_ENV, "0")
        calls = []
        ran, result = _run_secondary_within_budget(
            _state_started_seconds_ago(100000), "unit task",
            lambda: calls.append(1) or "ok",
        )
        assert ran is True
        assert result == "ok"
        assert calls == [1]
