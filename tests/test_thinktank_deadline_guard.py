"""A bounded runtime must not cost the run its terminal artifacts.

alpha-engine-config-I5208. The Think Tank ran on a 900s Lambda with NO deadline
awareness. Every run from 2026-07-17 hit the ceiling mid-loop and died before
its terminal writes, so ~15 theses of real work per day were completed and then
discarded: `thinktank/ratings/`, `thinktank/challenger_selection/` and the
leaderboard shadow view all froze on 2026-07-17 while the logs looked busy.

The guard is not a Lambda workaround — a spot box's reclaim notice is 2 minutes,
squarely inside the reserve — so it belongs in the run loop wherever it runs.

Pinned here:
  1. Work stops with enough wall-clock left to persist.
  2. Truncation is RECORDED, never silent — partial coverage must be
     distinguishable from complete coverage.
  3. No deadline (local/operator run) never truncates.
  4. A broken clock never truncates a healthy run.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinktank.run import _TERMINAL_WRITE_RESERVE_S, _out_of_time  # noqa: E402
from thinktank.schemas import RunManifest  # noqa: E402

# ── The guard ────────────────────────────────────────────────────────────────


def test_stops_when_inside_the_terminal_write_reserve() -> None:
    assert _out_of_time(lambda: _TERMINAL_WRITE_RESERVE_S - 1) is True
    assert _out_of_time(lambda: 0.0) is True


def test_continues_while_ample_time_remains() -> None:
    assert _out_of_time(lambda: _TERMINAL_WRITE_RESERVE_S + 1) is False
    assert _out_of_time(lambda: 900.0) is False


def test_boundary_is_inclusive_of_the_reserve() -> None:
    """Exactly at the reserve must STOP: the reserve is what the terminal
    writes need, so spending it on another thesis defeats the guard."""
    assert _out_of_time(lambda: _TERMINAL_WRITE_RESERVE_S) is True


def test_no_deadline_never_truncates() -> None:
    """Local and operator invocations pass no clock; they must run to
    completion rather than being silently cut short."""
    assert _out_of_time(None) is False


def test_a_broken_clock_never_truncates() -> None:
    """A raising clock is 'unknown', not 'out of time'. Treating an exception
    as a stop would let a monitoring bug silently halve every run's output —
    the same 'unverified is not false' trap as ARCHITECTURE §132."""

    def boom() -> float:
        raise RuntimeError("no context")

    assert _out_of_time(boom) is False


def test_reserve_is_large_enough_for_a_spot_reclaim_notice() -> None:
    """EC2 spot gives a 2-minute interruption notice. The reserve must cover
    it, or migrating this job to spot (the I5208 durable fix) reintroduces the
    exact failure on reclaim."""
    assert _TERMINAL_WRITE_RESERVE_S >= 120.0


# ── Truncation is recorded, not silent ───────────────────────────────────────


def test_manifest_defaults_to_untruncated() -> None:
    m = RunManifest(
        run_id="abc",
        mode="daily",
        trading_day="2026-07-28",
        calendar_date="2026-07-28",
        started_at="2026-07-28T14:30:00Z",
    )
    assert m.deadline_truncated is False
    assert m.deadline_skipped_new == []
    assert m.deadline_skipped_refresh == []
    assert m.deadline_skipped_sweep is False


def test_manifest_carries_what_was_skipped() -> None:
    """A partial run must say WHICH names it did not reach. 'Truncated' alone
    would leave a reader unable to tell partial coverage from complete."""
    m = RunManifest(
        run_id="abc",
        mode="daily",
        trading_day="2026-07-28",
        calendar_date="2026-07-28",
        started_at="2026-07-28T14:30:00Z",
        deadline_truncated=True,
        deadline_skipped_new=["AAPL", "MSFT"],
        deadline_skipped_refresh=["NVDA"],
        deadline_skipped_sweep=True,
    )
    round_tripped = RunManifest.model_validate(m.model_dump())
    assert round_tripped.deadline_truncated is True
    assert round_tripped.deadline_skipped_new == ["AAPL", "MSFT"]
    assert round_tripped.deadline_skipped_refresh == ["NVDA"]
    assert round_tripped.deadline_skipped_sweep is True


def test_truncation_fields_are_additive_to_the_schema() -> None:
    """Old manifests (written before I5208) must still parse — the artifact is
    versioned and consumers read historical runs."""
    legacy = {
        "run_id": "old",
        "mode": "daily",
        "trading_day": "2026-07-10",
        "calendar_date": "2026-07-10",
        "started_at": "2026-07-10T14:30:00Z",
        "theses_written": 5,
    }
    m = RunManifest.model_validate(legacy)
    assert m.deadline_truncated is False
    assert m.theses_written == 5
