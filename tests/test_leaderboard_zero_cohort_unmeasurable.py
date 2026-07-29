"""A leaderboard that scores zero cohorts must say so, not render as success.

champion-challenger-policy §7.2 / alpha-engine-config-I5195: ``n_dates == 0``
across every arm is a defect, not a result.

The source-failure path already routed to ``unmeasurable``. The path where the
closes source reads FINE and still scores nothing did not — it wrote
``status: "ok"`` with every metric null, byte-comparable to a healthy
leaderboard. Verified live 2026-07-29 against real S3: the producer leaderboard
returned ``status="ok", n_dates=0`` with all four arms null, two days before
the earliest cohort could possibly mature.

That is the fleet's dominant bug class — a record asserting an action that
never happened — sitting inside the very function written to prevent it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scoring.leaderboard_producers import (
    _overdue_zero_cohort_reason,
    build_producer_leaderboard,
    build_scanner_leaderboard,
)


class _FakeS3:
    """Enough S3 for the builders to reach scoring with zero matured cohorts."""

    def __init__(self) -> None:
        self.put_calls: list[dict] = []

    def put_object(self, **kw):
        self.put_calls.append(kw)
        return {}

    def get_object(self, **kw):  # noqa: ARG002 - absent objects are the point
        raise FileNotFoundError("no such key")


@pytest.fixture
def _no_matured_cohorts():
    """Cohorts old enough to have matured, resolving nothing — the OVERDUE
    case. An immature cohort is covered in test_leaderboard_realized_returns."""
    with (
        patch("scoring.leaderboard_producers._cohort_dates", return_value=["2026-06-01", "2026-06-10"]),
        patch("scoring.leaderboard_producers._resolve_realized_returns", return_value={}),
        patch("scoring.leaderboard_producers._get_json", return_value=None),
    ):
        yield


@pytest.mark.parametrize("builder", [build_producer_leaderboard, build_scanner_leaderboard])
def test_zero_scored_cohorts_is_unmeasurable_not_ok(builder, _no_matured_cohorts):
    res = builder(_FakeS3(), "bkt", "2026-07-29", write=False)
    assert res["status"] == "unmeasurable", (
        "a leaderboard that scored nothing must not report success — that is indistinguishable from a healthy one"
    )
    assert res["leaderboard"]["unmeasurable_reason"]


@pytest.mark.parametrize("builder", [build_producer_leaderboard, build_scanner_leaderboard])
def test_unmeasurable_artifact_is_still_written(builder, _no_matured_cohorts):
    """Downstream consumers depend on the artifact EXISTING. Fail-loud must not
    become fail-absent — that is a different silent gap (config#1403)."""
    s3 = _FakeS3()
    res = builder(s3, "bkt", "2026-07-29", write=True)
    assert res["status"] == "unmeasurable"
    assert s3.put_calls, "the artifact must still be written, with an honest verdict"


def test_immature_returns_no_reason_so_nothing_alerts():
    """Zero scored cohorts is EXPECTED while nothing has aged past the horizon.
    Alerting there would fire every cycle for weeks on a healthy system."""
    assert _overdue_zero_cohort_reason(["2026-07-24"], 21, "2026-07-28") is None


def test_overdue_returns_a_reason_naming_the_oldest_cohort():
    reason = _overdue_zero_cohort_reason(["2026-06-01"], 21, "2026-07-28")
    assert reason and "2026-06-01" in reason and "should have matured" in reason


def test_no_cohorts_at_all_is_its_own_reason():
    """Different operator response: the producers are not emitting, go look."""
    reason = _overdue_zero_cohort_reason([], 21, "2026-07-28")
    assert reason and "no cohort dates" in reason


def test_a_measurable_leaderboard_still_reports_ok():
    """The guard must not swallow real results — n_dates > 0 stays ok."""
    scored = {"leaderboard_id": "x", "n_dates": 3, "specs": [{"name": "a"}]}
    with (
        patch("scoring.leaderboard_producers._cohort_dates", return_value=["2026-07-02"]),
        patch("scoring.leaderboard_producers._resolve_realized_returns", return_value={"2026-07-02": {}}),
        patch("scoring.leaderboard_producers.score_leaderboard", return_value=scored),
        patch("scoring.leaderboard_producers._get_json", return_value=None),
    ):
        res = build_producer_leaderboard(_FakeS3(), "bkt", "2026-07-29", write=False)
    assert res["status"] == "ok"
