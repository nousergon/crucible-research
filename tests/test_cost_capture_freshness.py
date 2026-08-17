"""Contract tests for the capture-stream freshness detector (config-I7407 D4).

These assert the CONTRACT, not the values: that a silent stream raises, that
a live one does not, that the sentinel is written before the raise, and that
the sentinel's key is the concrete one the fleet freshness monitor can
``head_object``. A test pinning today's dates would go red on the calendar
rather than on a regression.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from scripts.cost_capture_freshness import (
    DEFAULT_MAX_AGE_DAYS,
    SENTINEL_KEY,
    CostCaptureStaleError,
    assert_capture_fresh,
    capture_stream_state,
    evaluate_and_publish,
    write_capture_sentinel,
)


class FakeS3:
    """Minimal S3 double: date partitions in, listings and PUTs out."""

    def __init__(self, partitions: dict[str, list[str]] | None = None,
                 put_error: Exception | None = None):
        # {"2026-08-16": ["decision_artifacts/_cost_raw/2026-08-16/run/x.0.jsonl"]}
        self.partitions = partitions or {}
        self.puts: list[dict] = []
        self.put_error = put_error

    def get_paginator(self, _name):
        return _FakePaginator(self)

    def put_object(self, **kwargs):
        if self.put_error is not None:
            raise self.put_error
        self.puts.append(kwargs)
        return {}


class _FakePaginator:
    def __init__(self, s3: FakeS3):
        self.s3 = s3

    def paginate(self, *, Bucket, Prefix, Delimiter=None):  # noqa: N803
        if Delimiter == "/":
            # Top-level listing: date partitions plus the sentinel object,
            # which must NOT be read as a partition.
            prefixes = [
                {"Prefix": f"{Prefix}{d}/"} for d in sorted(self.s3.partitions)
            ]
            prefixes.append({"Prefix": f"{Prefix}unknown-date/"})
            yield {"CommonPrefixes": prefixes,
                   "Contents": [{"Key": f"{Prefix}latest.json"}]}
            return
        day = Prefix.rstrip("/").rsplit("/", 1)[-1]
        yield {"Contents": [{"Key": k} for k in self.s3.partitions.get(day, [])]}


def _keys(day: str, *callsites: str) -> list[str]:
    return [
        f"decision_artifacts/_cost_raw/{day}/krepis-abc/{c}.0.jsonl"
        for c in callsites
    ]


# ── the state it reports ────────────────────────────────────────────────


def test_non_date_prefixes_are_not_partitions():
    """``latest.json`` and ``unknown-date/`` are not days.

    The sentinel lives under the same prefix as the partitions it describes,
    so a naive listing would count the detector's own output as capture and
    report a dead stream as fresh — the detector grading itself.
    """
    s3 = FakeS3({"2026-08-16": _keys("2026-08-16", "replay-concordance")})
    state = capture_stream_state(s3, "b", as_of=date(2026, 8, 16))
    assert state["dates_present"] == ["2026-08-16"]
    assert state["last_capture_date"] == "2026-08-16"
    assert state["days_since_last_capture"] == 0


def test_producers_are_named_from_the_keys():
    s3 = FakeS3({
        "2026-08-16": _keys("2026-08-16", "replay-concordance", "single-agent-quant"),
    })
    state = capture_stream_state(s3, "b", as_of=date(2026, 8, 16))
    assert state["producers_on_last_capture_date"] == [
        "replay-concordance", "single-agent-quant",
    ]


def test_a_partition_ahead_of_the_run_date_is_not_a_negative_age():
    """Capture keys on the UTC date of the call; a run date is a trading day.

    A partition one day ahead is routine, and computing the age against it
    would report a negative staleness — which compares as fresh forever.
    """
    s3 = FakeS3({
        "2026-08-10": _keys("2026-08-10", "thinktank-sweep"),
        "2026-08-17": _keys("2026-08-17", "replay-concordance"),
    })
    state = capture_stream_state(s3, "b", as_of=date(2026, 8, 16))
    assert state["last_capture_date"] == "2026-08-10"
    assert state["days_since_last_capture"] == 6
    assert "2026-08-17" in state["dates_present"]


# ── the verdict it reaches ──────────────────────────────────────────────


def test_fresh_stream_does_not_raise():
    s3 = FakeS3({"2026-08-16": _keys("2026-08-16", "replay-concordance")})
    state = capture_stream_state(s3, "b", as_of=date(2026, 8, 16))
    assert_capture_fresh(state)  # no exception


def test_stream_silent_past_the_ceiling_raises():
    s3 = FakeS3({"2026-08-01": _keys("2026-08-01", "thinktank-sweep")})
    state = capture_stream_state(s3, "b", as_of=date(2026, 8, 16))
    with pytest.raises(CostCaptureStaleError) as exc:
        assert_capture_fresh(state)
    assert "2026-08-01" in str(exc.value)
    assert "15 day(s)" in str(exc.value)


def test_exactly_at_the_ceiling_is_still_fresh():
    """The ceiling is inclusive — a boundary that raises one day early would
    fire on the ordinary week whose LLM stages were all skipped."""
    s3 = FakeS3({"2026-08-08": _keys("2026-08-08", "thinktank-sweep")})
    state = capture_stream_state(s3, "b", as_of=date(2026, 8, 8))
    state["days_since_last_capture"] = DEFAULT_MAX_AGE_DAYS
    assert_capture_fresh(state)
    state["days_since_last_capture"] = DEFAULT_MAX_AGE_DAYS + 1
    with pytest.raises(CostCaptureStaleError):
        assert_capture_fresh(state)


def test_an_empty_stream_raises_rather_than_passing_vacuously():
    """No partitions at all is the loudest case, not the quietest.

    `principles.md` §2.7. An emptiness check written as "every partition
    present is recent" is satisfied vacuously by having none.
    """
    state = capture_stream_state(FakeS3({}), "b", as_of=date(2026, 8, 16))
    assert state["last_capture_date"] is None
    with pytest.raises(CostCaptureStaleError) as exc:
        assert_capture_fresh(state)
    assert "empty" in str(exc.value)


# ── the sentinel it publishes ───────────────────────────────────────────


def test_sentinel_key_is_the_concrete_key_the_monitor_probes():
    """The fleet freshness monitor resolves an artifact by ``head_object``
    on a concrete key. A prefix would be unprobeable, and the row would
    render as permanently absent."""
    assert SENTINEL_KEY == "decision_artifacts/_cost_raw/latest.json"
    assert not SENTINEL_KEY.endswith("/")


def test_sentinel_is_written_before_the_staleness_raise():
    """The failure path leaves the same evidence as the success path.

    observability-policy.md §3.1. A detector that raises without recording
    what it saw makes its own finding unreconstructable.
    """
    s3 = FakeS3({"2026-08-01": _keys("2026-08-01", "thinktank-sweep")})
    with pytest.raises(CostCaptureStaleError):
        evaluate_and_publish(s3, "b", as_of=date(2026, 8, 16))
    assert len(s3.puts) == 1
    body = json.loads(s3.puts[0]["Body"].decode("utf-8"))
    assert s3.puts[0]["Key"] == SENTINEL_KEY
    assert body["last_capture_date"] == "2026-08-01"
    assert body["days_since_last_capture"] == 15


def test_sentinel_write_failure_raises_and_is_not_swallowed():
    """The sentinel IS the detector's only input.

    A swallowed PUT failure leaves the monitor reading a stale sentinel and
    reporting the stream as observed when nothing observed it.
    """
    s3 = FakeS3({"2026-08-16": _keys("2026-08-16", "replay-concordance")},
                put_error=RuntimeError("AccessDenied"))
    state = capture_stream_state(s3, "b", as_of=date(2026, 8, 16))
    with pytest.raises(RuntimeError, match="AccessDenied"):
        write_capture_sentinel(s3, "b", state)


def test_sentinel_carries_a_schema_version():
    s3 = FakeS3({"2026-08-16": _keys("2026-08-16", "replay-concordance")})
    evaluate_and_publish(s3, "b", as_of=date(2026, 8, 16))
    body = json.loads(s3.puts[0]["Body"].decode("utf-8"))
    assert body["schema_version"] == 1
    assert body["max_age_days"] == DEFAULT_MAX_AGE_DAYS
    assert body["as_of"] == "2026-08-16"
