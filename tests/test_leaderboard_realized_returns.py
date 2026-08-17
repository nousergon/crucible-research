"""The horizon-vs-retention invariant, and the loud `unmeasurable` verdict.

alpha-engine-config-I5195. Both leaderboards computed a 21-SESSION forward
return from `staging/daily_closes/`, a prefix under an
`expire-staging-after-7-days` S3 lifecycle rule. The horizon was 3x the
retention, so the join could never resolve — and because every failure mode was
best-effort, the impossibility rendered as an ordinary empty artifact
(`n_dates: 0`, all metrics null) written weekly for over a month.

Two things are pinned here:

  1. The relationship is asserted DIRECTLY (`_assert_horizon_is_satisfiable`)
     rather than inferred from a downstream count of zero.
  2. "Could not measure" and "nothing has matured yet" produce DIFFERENT
     outputs. Conflating them is what made the defect invisible.

`test_the_original_i5195_configuration_is_now_rejected` is the regression: it
reconstructs the exact live shape (7 sessions retained, 21-session horizon) and
requires it to raise. Against the pre-fix code that configuration returned `{}`
and the leaderboard reported a clean, empty result.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.leaderboard_producers import (  # noqa: E402
    LeaderboardUnmeasurableError,
    _assert_horizon_is_satisfiable,
    _horizon_date,
    _resolve_realized_returns,
    build_producer_leaderboard,
)


def _sessions(n: int, start_day: int = 1) -> list[str]:
    """`n` consecutive pseudo-session dates. Calendar realism is irrelevant —
    the scorer treats the panel's own key set AS the trading calendar."""
    return [f"2026-06-{start_day + i:02d}" for i in range(n)]


# ── 1. The invariant itself ──────────────────────────────────────────────────


def test_the_original_i5195_configuration_is_now_rejected() -> None:
    """THE REGRESSION TEST. A 7-session retention window against a 21-session
    horizon — the live configuration on 2026-07-28. Must raise; previously it
    returned empty and the leaderboard reported a clean, healthy-looking run."""
    retained = _sessions(7)
    with pytest.raises(LeaderboardUnmeasurableError) as exc:
        _assert_horizon_is_satisfiable(retained, entry_dates=[retained[0]], horizon_days=21)

    msg = str(exc.value)
    assert "21" in msg and "I5195" in msg, "the error must name the horizon and the issue"


def test_passes_when_the_source_can_support_the_horizon() -> None:
    calendar = _sessions(30)
    _assert_horizon_is_satisfiable(calendar, entry_dates=[calendar[0]], horizon_days=21)


def test_boundary_exactly_horizon_many_sessions_is_sufficient() -> None:
    calendar = _sessions(21)
    _assert_horizon_is_satisfiable(calendar, entry_dates=[calendar[0]], horizon_days=21)
    with pytest.raises(LeaderboardUnmeasurableError):
        _assert_horizon_is_satisfiable(calendar, entry_dates=[calendar[0]], horizon_days=22)


def test_a_fresh_cohort_against_a_capable_source_does_NOT_raise() -> None:
    """The distinction that matters. A capable source (ArcticDB holds years)
    plus a cohort entered days ago is the normal Monday state — immature, not
    broken. An earlier formulation of this invariant counted only sessions
    AFTER the earliest entry and fired here, which would have converted every
    fresh cohort into a false alarm."""
    calendar = _sessions(28, start_day=1)  # 2026-06-01 .. 2026-06-28
    _assert_horizon_is_satisfiable(calendar, entry_dates=["2026-06-26"], horizon_days=21)


def test_cohort_entirely_before_the_calendar_is_unmeasurable() -> None:
    """The retention window slid past the cohort: the source is long enough in
    the abstract, but it can no longer see any entry date, so no entry close
    can be read."""
    calendar = _sessions(28, start_day=1)  # begins 2026-06-01
    with pytest.raises(LeaderboardUnmeasurableError) as exc:
        _assert_horizon_is_satisfiable(calendar, entry_dates=["2026-05-02"], horizon_days=21)
    assert "precedes the closes calendar" in str(exc.value)


def test_empty_calendar_raises_rather_than_returning_nothing() -> None:
    with pytest.raises(LeaderboardUnmeasurableError):
        _assert_horizon_is_satisfiable([], entry_dates=["2026-06-01"], horizon_days=21)


# ── 2. An empty source raises instead of scoring zero dates ──────────────────


def test_empty_closes_panel_raises() -> None:
    with patch(
        "scoring.leaderboard_producers._closes_panel_from_arcticdb",
        return_value={},
    ):
        with pytest.raises(LeaderboardUnmeasurableError):
            _resolve_realized_returns(None, "b", ["2026-06-01"], 21)


def test_no_entry_dates_is_not_an_error() -> None:
    """An empty cohort is a legitimate "nothing to score", NOT an unmeasurable
    source — the distinction this whole module exists to preserve."""
    assert _resolve_realized_returns(None, "b", [], 21) == {}


# ── 3. Forward returns compute correctly once the source is durable ──────────


def test_forward_returns_use_the_horizon_session() -> None:
    cal = _sessions(10)
    panel = {d: {"AAPL": 100.0, "MSFT": 50.0} for d in cal}
    panel[cal[5]] = {"AAPL": 110.0, "MSFT": 45.0}  # entry cal[0], horizon 5

    with patch(
        "scoring.leaderboard_producers._closes_panel_from_arcticdb",
        return_value=panel,
    ):
        realized = _resolve_realized_returns(None, "b", [cal[0]], 5)

    assert realized[cal[0]]["AAPL"] == pytest.approx(0.10)
    assert realized[cal[0]]["MSFT"] == pytest.approx(-0.10)


def test_immature_cohort_date_is_omitted_not_raised() -> None:
    """An honest `None`: the source is fine, this date just has not matured.
    Must NOT raise — that would make a healthy young cohort look like a defect."""
    cal = _sessions(30)
    panel = {d: {"AAPL": 100.0} for d in cal}
    entries = [cal[0], cal[28]]  # second has only 1 session after it

    with patch(
        "scoring.leaderboard_producers._closes_panel_from_arcticdb",
        return_value=panel,
    ):
        realized = _resolve_realized_returns(None, "b", entries, 21)

    assert cal[0] in realized
    assert cal[28] not in realized


def test_horizon_date_selects_the_nth_session_after_entry() -> None:
    cal = _sessions(10)
    assert _horizon_date(cal, cal[0], 3) == cal[3]
    assert _horizon_date(cal, cal[7], 5) is None


# ── 4. The verdict is LOUD, and distinguishable from an immature cohort ──────


class _S3:
    def __init__(self):
        self.written = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.written[Key] = Body


def test_unmeasurable_leaderboard_is_written_labelled_and_alerted() -> None:
    s3 = _S3()
    with (
        patch(
            "scoring.leaderboard_producers._cohort_dates",
            return_value=["2026-06-01"],
        ),
        patch(
            "scoring.leaderboard_producers._load_producer_specs",
            return_value=(None, []),
        ),
        patch(
            "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
            side_effect=LeaderboardUnmeasurableError("horizon 21 > retention 7"),
        ),
        patch(
            "scoring.leaderboard_producers.publish_observe_alert",
        ) as alert,
    ):
        out = build_producer_leaderboard(s3, "b", "2026-07-28")

    assert out["status"] == "unmeasurable", "must not report success"
    assert out["leaderboard"]["status"] == "unmeasurable"
    assert "horizon 21 > retention 7" in out["leaderboard"]["unmeasurable_reason"]

    # The artifact still exists — consumers depend on it — but it SAYS so.
    assert "research/producer_leaderboard/2026-07-28.json" in s3.written

    # And it is loud. Silence was the defect.
    assert alert.call_count == 1
    assert "UNMEASURABLE" in alert.call_args.kwargs["message"]


def test_unmeasurable_is_not_confusable_with_an_immature_cohort() -> None:
    """The core I5195 lesson: these two states must not render identically.
    A healthy-but-immature run reports `ok`; an unmeasurable one does not.

    Cohort date corrected 2026-07-29: this previously used 2026-06-01 against
    an as-of of 2026-07-28 — **39 trading days** on a 21-day horizon. That is
    not an immature cohort, it is a cohort that should long since have
    matured, i.e. precisely the defect I5195 was filed about. The test was
    asserting `ok` for the bug. It now uses a genuinely immature cohort.
    """
    s3 = _S3()
    with (
        patch(
            "scoring.leaderboard_producers._cohort_dates",
            return_value=["2026-07-24"],  # 2 trading days before as-of
        ),
        patch(
            "scoring.leaderboard_producers._load_producer_specs",
            return_value=(None, []),
        ),
        patch(
            "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
            return_value=({}, {}),
        ),
        patch(
            "scoring.leaderboard_producers.publish_observe_alert",
        ) as alert,
    ):
        out = build_producer_leaderboard(s3, "b", "2026-07-28")

    assert out["status"] == "ok"
    assert out["leaderboard"].get("status") != "unmeasurable"
    assert alert.call_count == 0, (
        "an immature cohort must NOT alert — it resolves itself, and alerting "
        "every cycle for weeks is how alert fatigue is manufactured"
    )


def test_zero_cohorts_past_the_horizon_escalates_to_unmeasurable() -> None:
    """Immaturity is BOUNDED. Once the oldest cohort is older than the horizon
    and still nothing scores, that is no longer waiting — it is the four-weeks
    -of-empty-artifacts defect I5195 was filed about, and it must say so."""
    s3 = _S3()
    with (
        patch(
            "scoring.leaderboard_producers._cohort_dates",
            return_value=["2026-06-01"],  # ~39 trading days before as-of
        ),
        patch(
            "scoring.leaderboard_producers._load_producer_specs",
            return_value=(None, []),
        ),
        patch(
            "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
            return_value=({}, {}),
        ),
        patch("scoring.leaderboard_producers.publish_observe_alert") as alert,
    ):
        out = build_producer_leaderboard(s3, "b", "2026-07-28")

    assert out["status"] == "unmeasurable"
    assert "should have matured" in out["leaderboard"]["unmeasurable_reason"]
    assert alert.call_count == 1


# ── 5. The champion's identity comes from the live pointer, not the registry ──
#
# alpha-engine-config-I5195. `producers/registry.py` lost its champion spec when
# `agentic_sector_teams` retired (config-I2993) and nothing replaced it, so every
# producer leaderboard reported `champion: null` — while `config/producer_champion.json`
# had named `scanner_predictor_direct` since 2026-07-13. Two registries for one
# fact is the multi-writer drift class; the pointer wins.


def _s3_returning(objects: dict):
    """Minimal S3 stub: `objects` maps key -> dict body (absent keys 404)."""
    import json as _json

    class _Err(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "NoSuchKey"}}

    class _S3Get:
        def get_object(self, Bucket, Key):  # noqa: N803
            if Key not in objects:
                from botocore.exceptions import ClientError

                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

            class _B:
                @staticmethod
                def read():
                    return _json.dumps(objects[Key]).encode()

            return {"Body": _B()}

    return _S3Get()


def test_champion_name_comes_from_the_live_pointer() -> None:
    from scoring.leaderboard_producers import _resolve_champion_name

    s3 = _s3_returning(
        {
            "config/producer_champion.json": {
                "schema_version": 1,
                "champion": "scanner_predictor_direct",
                "promoted_at": "2026-07-13T22:07:09Z",
            },
        }
    )
    assert _resolve_champion_name(s3, "b") == "scanner_predictor_direct"


def test_champion_falls_back_to_registry_when_pointer_absent() -> None:
    """An S3 hiccup must degrade to prior behaviour, not drop the champion arm."""
    from scoring.leaderboard_producers import _resolve_champion_name

    with patch("producers.registry.champion_producer") as champ:
        champ.return_value = type("S", (), {"name": "registry_champ"})()
        assert _resolve_champion_name(_s3_returning({}), "b") == "registry_champ"


def test_champion_is_none_only_when_neither_source_names_one() -> None:
    from scoring.leaderboard_producers import _resolve_champion_name

    with patch("producers.registry.champion_producer", return_value=None):
        assert _resolve_champion_name(_s3_returning({}), "b") is None


def test_malformed_pointer_does_not_yield_a_junk_champion_name() -> None:
    from scoring.leaderboard_producers import _resolve_champion_name

    with patch("producers.registry.champion_producer", return_value=None):
        for bad in ({"champion": ""}, {"champion": None}, {"champion": 7}, {}):
            s3 = _s3_returning({"config/producer_champion.json": bad})
            assert _resolve_champion_name(s3, "b") is None, bad
