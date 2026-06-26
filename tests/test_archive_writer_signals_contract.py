"""archive_writer signals.json write-failure contract — regression guard.

crucible-research#312 made the archive_writer node re-raise on a
signals.json write/validation failure (killing the "ghost success" mode
where the research SF task returned status=OK while signals.json was
absent, so the whole downstream cycle silently ran on stale signals).
That fix shipped with no dedicated unit test because the archive_writer
node had no clean unit seam (it reads dozens of state keys).

config#1235 extracted the validate+write into ``_validate_and_write_signals``
so the contract is now directly testable. These tests pin both halves of
the fail-loud guarantee:

  1. A failing ``write_signals_json`` PROPAGATES (raises) — never a silent
     ghost-success.
  2. An invalid payload PROPAGATES from ``_validate_signals_payload`` AND
     the write is never attempted (an invalid payload is as fatal as a
     failed write — neither yields a usable signals.json).

And the happy path: a valid payload is written exactly once.

Uses a hand-rolled FakeArchive (mirrors tests/test_stub_quarantine.py).
"""

from __future__ import annotations

import pytest

from graph.research_graph import _validate_and_write_signals


class _RaisingArchive:
    """write_signals_json fails (e.g. S3 down / validation in the writer)."""

    def __init__(self) -> None:
        self.write_attempts: list[tuple] = []

    def write_signals_json(self, run_date, run_time, payload):
        self.write_attempts.append((run_date, run_time, payload))
        raise OSError("S3 PutObject failed")


class _RecordingArchive:
    """write_signals_json succeeds and records the call."""

    def __init__(self) -> None:
        self.writes: list[tuple] = []

    def write_signals_json(self, run_date, run_time, payload):
        self.writes.append((run_date, run_time, payload))


# An empty signals map has no ENTER signals, so it passes validation.
_VALID_PAYLOAD = {"signals": {}, "buy_candidates": []}

# An ENTER signal with an unresolved sector is a blocking validation failure
# (the 2026-05-04 EOG/NVT incident shape) — _validate_signals_payload raises.
_INVALID_PAYLOAD = {
    "signals": {"FAKE": {"signal": "ENTER", "sector": "Unknown"}},
    "buy_candidates": [],
}


def test_write_failure_propagates():
    """A failing write_signals_json must re-raise — NOT ghost-succeed."""
    arch = _RaisingArchive()
    with pytest.raises(OSError, match="S3 PutObject failed"):
        _validate_and_write_signals(
            arch, "2026-06-26", "2026-06-26T17:00:00Z", _VALID_PAYLOAD
        )
    assert arch.write_attempts, "the write must have been attempted before raising"


def test_invalid_payload_propagates_and_skips_write():
    """An invalid payload must propagate from validation AND never reach
    the write (an invalid payload is as fatal as a failed write)."""
    arch = _RecordingArchive()
    with pytest.raises(RuntimeError, match="BLOCKED"):
        _validate_and_write_signals(
            arch,
            "2026-06-26",
            "2026-06-26T17:00:00Z",
            _INVALID_PAYLOAD,
            scanner_universe={"FAKE"},  # in-universe, so the sector check is what blocks
        )
    assert arch.writes == [], "signals.json must NOT be written on an invalid payload"


def test_out_of_universe_enter_propagates_and_skips_write():
    """An ENTER signal outside the current scanner universe is a blocking
    validation failure — propagates and the write is skipped."""
    arch = _RecordingArchive()
    payload = {
        "signals": {"ZZZZ": {"signal": "ENTER", "sector": "Technology"}},
        "buy_candidates": [],
    }
    with pytest.raises(RuntimeError, match="outside current"):
        _validate_and_write_signals(
            arch,
            "2026-06-26",
            "2026-06-26T17:00:00Z",
            payload,
            scanner_universe={"AAPL", "MSFT"},  # ZZZZ not a member
        )
    assert arch.writes == []


def test_valid_payload_writes_once():
    """The happy path: a valid payload is persisted exactly once and the
    helper returns without raising."""
    arch = _RecordingArchive()
    _validate_and_write_signals(
        arch, "2026-06-26", "2026-06-26T17:00:00Z", _VALID_PAYLOAD
    )
    assert len(arch.writes) == 1
    run_date, run_time, payload = arch.writes[0]
    assert run_date == "2026-06-26"
    assert payload is _VALID_PAYLOAD
