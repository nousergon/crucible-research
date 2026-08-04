"""Retired-arm trailing-window scoring (alpha-engine-config-I6427 /
champion-challenger-policy.md §3): "A retired arm is scored for a trailing
window (default: 8 cycles past retired_date)."

Before this PR, ``producers.registry.challenger_producers()`` filtered
``RESEARCH_PRODUCERS`` to ``kind == "challenger"`` only — every
``kind == "retired"`` row (e.g. ``agentic_sector_teams``, retired
2026-07-12) was excluded from scoring entirely, in violation of §3. There
was no retired-arm scoring path anywhere in the repo.

Locks down:
- ``retired_producers()`` selects ``kind=="retired"`` rows whose
  ``retired_date`` falls inside the named trailing-window constants,
  against the live registry instance (``agentic_sector_teams``,
  retired_date="2026-07-12" — 8 cycles * 7 days = 56 days later is
  2026-09-06, matching the issue's own worked example).
- The producer leaderboard (``scoring.leaderboard_producers``) scores an
  in-window retired arm alongside champion + challengers, and tags its
  spec row ``"kind": "retired"`` so a leaderboard reader (and any
  downstream promotion engine filtering on ``kind=="challenger"``) can
  tell it apart from a live challenger.
- An out-of-window retired arm is NOT scored — it drops out of both the
  registry selector and the leaderboard's ``specs`` list.

Guard verified RED on pre-fix code (policy §7.4): before
``retired_producers()`` existed, ``ImportError`` at collection; after
adding the selector but before wiring it into
``scoring/leaderboard_producers.py::_load_producer_specs``, the
leaderboard integration tests below failed with
``"agentic_sector_teams" not in names`` (retired arm silently absent from
``specs``). See PR body for the literal pre-fix pytest output.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

_BUCKET = "alpha-engine-research"

_RETIRED_ARM = "agentic_sector_teams"
_RETIRED_DATE = "2026-07-12"
# 8 cycles * 7 days/cycle = 56 days past retired_date → 2026-09-06, exactly
# the window end the issue's own worked example names.
_WINDOW_END = "2026-09-06"
_IN_WINDOW_DATE = "2026-08-04"  # inside the window (issue's "still open" date)
_OUT_OF_WINDOW_DATE = "2026-09-07"  # one day past the window end


# ── Registry-level selector ────────────────────────────────────────────────


class TestRetiredProducersSelector:
    def test_named_window_constants_multiply_to_56_days(self):
        from producers.registry import (
            RETIRED_TRAILING_WINDOW_CYCLE_DAYS,
            RETIRED_TRAILING_WINDOW_CYCLES,
            RETIRED_TRAILING_WINDOW_DAYS,
        )

        assert RETIRED_TRAILING_WINDOW_CYCLES == 8
        assert RETIRED_TRAILING_WINDOW_DAYS == (RETIRED_TRAILING_WINDOW_CYCLES * RETIRED_TRAILING_WINDOW_CYCLE_DAYS)
        assert RETIRED_TRAILING_WINDOW_DAYS == 56

    def test_in_window_retired_arm_is_selected(self):
        from producers.registry import retired_producers

        names = {p.name for p in retired_producers(as_of=_IN_WINDOW_DATE)}
        assert _RETIRED_ARM in names

    def test_window_end_boundary_is_inclusive(self):
        from producers.registry import retired_producers

        names = {p.name for p in retired_producers(as_of=_WINDOW_END)}
        assert _RETIRED_ARM in names

    def test_out_of_window_retired_arm_is_not_selected(self):
        from producers.registry import retired_producers

        names = {p.name for p in retired_producers(as_of=_OUT_OF_WINDOW_DATE)}
        assert _RETIRED_ARM not in names

    def test_selector_never_returns_champion_or_challenger_kinds(self):
        from producers.registry import retired_producers

        for p in retired_producers(as_of=_IN_WINDOW_DATE):
            assert p.kind == "retired"

    def test_retired_producers_accepts_date_object(self):
        from datetime import date

        from producers.registry import retired_producers

        names = {p.name for p in retired_producers(as_of=date(2026, 8, 4))}
        assert _RETIRED_ARM in names


# ── Producer-leaderboard wiring (moto S3) ──────────────────────────────────


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_json(s3, key, obj):
    s3.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


class _Panel:
    """Builds `{date: {ticker: close}}` and hands it back as a loader."""

    def __init__(self):
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update({t: float(c) for t, c in closes.items()})
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


def _seed_retired_arm_shadow(s3, entry: str) -> _Panel:
    """A retired arm's shadow signals for one cohort date, plus a matured
    21d-forward closes panel so the cohort actually scores (not just
    "present but immature")."""
    _put_json(
        s3,
        f"signals_shadow/{_RETIRED_ARM}/{entry}/signals.json",
        {
            "signals": {
                "A": {"signal": "ENTER", "score": 90},
                "B": {"signal": "ENTER", "score": 70},
            }
        },
    )
    panel = _Panel().put(entry, {"A": 100, "B": 100, "SPY": 100})
    for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
        panel.put(d, {"A": 110, "B": 102, "SPY": 105})
    return panel


class TestRetiredArmProducerLeaderboardScoring:
    def test_in_window_retired_arm_is_scored_and_tagged_retired(self, s3):
        from scoring.leaderboard_producers import build_producer_leaderboard

        entry = "2026-06-01"
        panel = _seed_retired_arm_shadow(s3, entry)

        res = build_producer_leaderboard(
            s3,
            _BUCKET,
            _IN_WINDOW_DATE,
            top_n=1,
            closes_panel_loader=panel.loader(),
        )
        assert res["status"] == "ok"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        names = {s["name"]: s for s in got["specs"]}
        assert _RETIRED_ARM in names, (
            f"retired arm {_RETIRED_ARM} missing from an in-window leaderboard "
            f"— retired_producers() must feed scoring/leaderboard_producers.py"
        )
        row = names[_RETIRED_ARM]
        # Distinctly tagged so a leaderboard reader (and a downstream
        # promotion engine filtering on kind=="challenger") can tell a
        # historical-evidence-only retired arm apart from a live challenger.
        assert row["kind"] == "retired"
        # It is actually SCORED, not just present with null metrics.
        assert row["n_dates_scored"] >= 1
        assert row["realized_rank_ic"] is not None

    def test_out_of_window_retired_arm_is_not_scored(self, s3):
        from scoring.leaderboard_producers import build_producer_leaderboard

        entry = "2026-06-01"
        panel = _seed_retired_arm_shadow(s3, entry)
        # A live challenger with real, matured data too — otherwise, once the
        # retired arm correctly drops out, NOTHING scores and the leaderboard
        # reports "unmeasurable" for an unrelated reason (zero cohorts at
        # all), which would mask what this test is actually checking.
        _put_json(
            s3,
            f"signals_shadow/no_agent_quant/{entry}/signals.json",
            {"signals": {"A": {"signal": "ENTER", "score": 60}, "B": {"signal": "ENTER", "score": 88}}},
        )

        res = build_producer_leaderboard(
            s3,
            _BUCKET,
            _OUT_OF_WINDOW_DATE,
            top_n=1,
            closes_panel_loader=panel.loader(),
        )
        assert res["status"] == "ok"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        names = {s["name"]: s for s in got["specs"]}
        assert _RETIRED_ARM not in names, (
            f"retired arm {_RETIRED_ARM} scored past its trailing window "
            "— retired_producers() must exclude out-of-window rows"
        )

    def test_live_challenger_still_scored_alongside_retired_arm(self, s3):
        """Challengers must remain unaffected: retired-arm wiring is additive."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        entry = "2026-06-01"
        panel = _seed_retired_arm_shadow(s3, entry)
        _put_json(
            s3,
            f"signals_shadow/no_agent_quant/{entry}/signals.json",
            {"signals": {"A": {"signal": "ENTER", "score": 60}, "B": {"signal": "ENTER", "score": 88}}},
        )

        res = build_producer_leaderboard(
            s3,
            _BUCKET,
            _IN_WINDOW_DATE,
            top_n=1,
            closes_panel_loader=panel.loader(),
        )
        assert res["status"] == "ok"
        got = res["leaderboard"]
        names = {s["name"]: s for s in got["specs"]}
        assert names["no_agent_quant"]["kind"] == "challenger"
        assert names[_RETIRED_ARM]["kind"] == "retired"
