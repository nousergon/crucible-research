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
