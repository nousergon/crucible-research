"""Runtime vacuity guard: champion vs. challenger identical membership must
ALARM (alpha-engine-config-I6429, champion-challenger-policy.md §4).

"A vacuity guard is required. If two arms resolve to the same membership, the
comparison is worthless while every other assertion still passes. Assert that
competing arms actually differ."

Before this fix, a guard existed ONLY as a fixture-based unit test
(``tests/test_universe_membership.py::
test_incumbent_arm_disagrees_with_the_champion``) — it locks static test
fixtures, and exercises nothing at runtime. If a live Saturday cycle produced
identical champion/challenger membership, nothing would flag it: a
well-formed, green leaderboard comparing an arm to itself.

Per policy §7.4 ("a guard must be verified to fail without the fix"): these
tests were run against the pre-fix ``scoring/leaderboard_producers.py`` (no
``_vacuous_membership_collisions``/``_alert_vacuous_collisions``, no call
site in either builder) and FAILED — ``test_identical_membership_triggers_the_alarm``
found zero matching ``publish_observe_alert`` calls for both builders. That is
the red-guard evidence this PR's body cites.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scoring.leaderboard_producers import (
    build_producer_leaderboard,
    build_scanner_leaderboard,
)
from scoring.leaderboard_scoring import SpecDay, SpecHistory

_COHORT_DATE = "2026-07-24"


class _FakeS3:
    """Enough S3 for the builders to reach scoring without touching AWS."""

    def __init__(self) -> None:
        self.put_calls: list[dict] = []

    def put_object(self, **kw):
        self.put_calls.append(kw)
        return {}

    def get_object(self, **kw):  # noqa: ARG002 - absent objects are the point
        raise FileNotFoundError("no such key")


def _identical_specs() -> tuple[SpecHistory, list[SpecHistory]]:
    """Champion and challenger resolve to the EXACT same picked-ticker set —
    the vacuous case a live cycle must never let through silently."""
    champion = SpecHistory(
        name="champion_arm",
        kind="champion",
        by_date={_COHORT_DATE: SpecDay(ranked=["AAPL", "MSFT", "GOOG"])},
    )
    challenger = SpecHistory(
        name="colliding_challenger",
        kind="challenger",
        by_date={_COHORT_DATE: SpecDay(ranked=["GOOG", "AAPL", "MSFT"])},  # same SET, different order
    )
    return champion, [challenger]


def _distinct_specs() -> tuple[SpecHistory, list[SpecHistory]]:
    """Champion and challenger genuinely disagree — the healthy, expected
    case that must never false-positive."""
    champion = SpecHistory(
        name="champion_arm",
        kind="champion",
        by_date={_COHORT_DATE: SpecDay(ranked=["AAPL", "MSFT", "GOOG"])},
    )
    challenger = SpecHistory(
        name="distinct_challenger",
        kind="challenger",
        by_date={_COHORT_DATE: SpecDay(ranked=["NVDA", "AMD", "INTC"])},
    )
    return champion, [challenger]


def _vacuity_alert_calls(mock_alert):
    return [
        c
        for c in mock_alert.call_args_list
        if "vacuous" in c.kwargs.get("message", "").lower() or "identical" in c.kwargs.get("message", "").lower()
    ]


@pytest.mark.parametrize(
    "builder,loader_target",
    [
        (build_scanner_leaderboard, "scoring.leaderboard_producers._load_scanner_specs"),
        (build_producer_leaderboard, "scoring.leaderboard_producers._load_producer_specs"),
    ],
)
def test_identical_membership_triggers_the_alarm(builder, loader_target):
    """FAULT INJECTION: two arms configured to resolve to the SAME membership
    on a live cycle must alarm. This is the assertion that fails on pre-fix
    code — no call site compares champion vs. challenger membership at all."""
    champion, challengers = _identical_specs()
    with (
        patch(loader_target, return_value=(champion, challengers)),
        patch("scoring.leaderboard_producers._cohort_dates", return_value=[_COHORT_DATE]),
        patch("scoring.leaderboard_producers._resolve_realized_returns", return_value={}),
        patch("scoring.leaderboard_producers.publish_observe_alert") as mock_alert,
    ):
        res = builder(_FakeS3(), "bkt", "2026-07-29", write=False)

    calls = _vacuity_alert_calls(mock_alert)
    assert calls, "identical champion/challenger membership must publish an explicit alarm"
    msg = calls[0].kwargs["message"]
    assert "colliding_challenger" in msg, "the alarm must name which arm collided"
    assert "champion_arm" in msg, "the alarm must name the champion it collided with"

    # Fail-soft posture (CC-7.2 mirror): an alarm, never an exception — the
    # build still returns a normal status dict.
    assert res["status"] in ("ok", "unmeasurable")


@pytest.mark.parametrize(
    "builder,loader_target",
    [
        (build_scanner_leaderboard, "scoring.leaderboard_producers._load_scanner_specs"),
        (build_producer_leaderboard, "scoring.leaderboard_producers._load_producer_specs"),
    ],
)
def test_distinct_membership_does_not_alarm(builder, loader_target):
    """Guard against a false-positive: genuinely different arms must score
    cleanly with no vacuity alert."""
    champion, challengers = _distinct_specs()
    with (
        patch(loader_target, return_value=(champion, challengers)),
        patch("scoring.leaderboard_producers._cohort_dates", return_value=[_COHORT_DATE]),
        patch("scoring.leaderboard_producers._resolve_realized_returns", return_value={}),
        patch("scoring.leaderboard_producers.publish_observe_alert") as mock_alert,
    ):
        builder(_FakeS3(), "bkt", "2026-07-29", write=False)

    assert not _vacuity_alert_calls(mock_alert), "genuinely distinct arms must never false-positive"


def test_collision_is_recorded_on_the_leaderboard_artifact():
    """The artifact itself carries the collision, not only a transient alert
    — a downstream reader (dashboard, sweep) must be able to see it without
    depending on the alert channel."""
    champion, challengers = _identical_specs()
    with (
        patch("scoring.leaderboard_producers._load_scanner_specs", return_value=(champion, challengers)),
        patch("scoring.leaderboard_producers._cohort_dates", return_value=[_COHORT_DATE]),
        patch("scoring.leaderboard_producers._resolve_realized_returns", return_value={}),
        patch("scoring.leaderboard_producers.publish_observe_alert"),
    ):
        res = build_scanner_leaderboard(_FakeS3(), "bkt", "2026-07-29", write=False)

    assert res["status"] == "ok"
    collisions = res["leaderboard"]["vacuous_membership_collisions"]
    assert collisions == [{"challenger": "colliding_challenger", "date": _COHORT_DATE, "n_tickers": 3}]


def test_no_collision_is_an_empty_list_not_absent():
    """A healthy cycle still carries the field — absence must never be
    conflated with "checked and clean"."""
    champion, challengers = _distinct_specs()
    with (
        patch("scoring.leaderboard_producers._load_scanner_specs", return_value=(champion, challengers)),
        patch("scoring.leaderboard_producers._cohort_dates", return_value=[_COHORT_DATE]),
        patch("scoring.leaderboard_producers._resolve_realized_returns", return_value={}),
        patch("scoring.leaderboard_producers.publish_observe_alert"),
    ):
        res = build_scanner_leaderboard(_FakeS3(), "bkt", "2026-07-29", write=False)

    assert res["leaderboard"]["vacuous_membership_collisions"] == []


def test_producer_leaderboard_with_no_champion_never_raises():
    """config-I2993/I2998: a producer leaderboard with no registered champion
    is an honest, expected state. The vacuity guard must degrade to a no-op,
    never crash the champion-free path."""
    challenger = SpecHistory(
        name="only_challenger",
        kind="challenger",
        by_date={_COHORT_DATE: SpecDay(ranked=["AAPL"])},
    )
    with (
        patch("scoring.leaderboard_producers._load_producer_specs", return_value=(None, [challenger])),
        patch("scoring.leaderboard_producers._cohort_dates", return_value=[_COHORT_DATE]),
        patch("scoring.leaderboard_producers._resolve_realized_returns", return_value={}),
        patch("scoring.leaderboard_producers.publish_observe_alert") as mock_alert,
    ):
        res = build_producer_leaderboard(_FakeS3(), "bkt", "2026-07-29", write=False)

    assert res["status"] == "ok"
    assert not _vacuity_alert_calls(mock_alert)
    assert res["leaderboard"]["vacuous_membership_collisions"] == []


def test_unit_helper_flags_collisions_by_challenger_and_date():
    """Direct unit coverage of the comparison primitive, independent of the
    S3/builder plumbing."""
    from scoring.leaderboard_producers import _vacuous_membership_collisions

    champion, challengers = _identical_specs()
    collisions = _vacuous_membership_collisions(champion, challengers)
    assert collisions == [{"challenger": "colliding_challenger", "date": _COHORT_DATE, "n_tickers": 3}]


def test_unit_helper_ignores_empty_membership():
    """Two arms both resolving to an EMPTY set are not a vacuity collision —
    that is a different (unmeasurable/no-picks) condition, already covered
    elsewhere, and must not be double-reported here."""
    from scoring.leaderboard_producers import _vacuous_membership_collisions

    champion = SpecHistory(name="c", kind="champion", by_date={_COHORT_DATE: SpecDay(ranked=[])})
    challenger = SpecHistory(name="x", kind="challenger", by_date={_COHORT_DATE: SpecDay(ranked=[])})
    assert _vacuous_membership_collisions(champion, [challenger]) == []


def test_unit_helper_none_champion_is_a_no_op():
    from scoring.leaderboard_producers import _vacuous_membership_collisions

    challenger = SpecHistory(name="x", kind="challenger", by_date={_COHORT_DATE: SpecDay(ranked=["AAPL"])})
    assert _vacuous_membership_collisions(None, [challenger]) == []
