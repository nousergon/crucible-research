"""The mis-wired predictor arms' history is MARKED unusable, per date
(alpha-engine-config-I11423, deliverable 3 of -I11396).

`-I11396` fixed `load_predictor_cut_pool` (`crucible-research-PR816`). Its
deliverable 3 — re-running the backfill — was out of scope there. Until that
history is either regenerated or marked, the affected arms' record silently
spans two wirings, and any comparison crossing the boundary measures the wiring
change rather than the arms.

WHY MARKED RATHER THAN REGENERATED. Re-running was the preferred remedy and is
not possible. Measured 2026-09-22 against live S3: the fixed loader reads
`predictor/predictions_research_free/{date}.json`, and that prefix's retention
begins on 2026-09-14 — so the great majority of affected arm-dates have no
input left to replay. Even 2026-09-18, which is inside retention, does not
reproduce its stored result (4 of 20 cut names overlap that day's file now), so
the replay is not idempotent where it is executable at all. The audit is on the
issue.

WHY IT STILL MATTERS AFTER RETIREMENT. Both arms are `kind="retired"`
(`-PR819`, retired_date 2026-09-22), and §3 scores a retired arm for a trailing
window of 8 cycles — 56 days, through 2026-11-17. This history is READ, not
inert. A retired arm's record exists so that "we retired the wrong one" stays
answerable, and a record produced by a mis-wired pool cannot answer it.

THE CONDITION IS THE POOL SOURCE, NOT A DATE LIST. The flag is derived from each
day's own `arm_pool.pool_source` (the provenance `-I11424` carries onto the
row). A hand-kept date list would have to be maintained in step with a set of S3
objects by hand — the failure class that put one role in an IAM Resource list
while five were declared.

VERIFIED RED (champion-challenger-policy.md §7.4). Removing the one line that
wires `_annotate_degraded_inputs` into `build_producer_leaderboard` and
re-running this file, verbatim:

    FAILED ...::test_a_mis_wired_pool_date_is_marked_on_the_producer_board
      AssertionError: the mis-wired date must be named on the row
    FAILED ...::test_the_board_names_every_affected_arm_at_the_top_level
      KeyError: 'input_quality'
    FAILED ...::test_a_correctly_wired_arm_carries_an_explicit_null
      KeyError: 'degraded_input'

`_annotate_degraded_inputs` ran on the CUTS board alone — so a defect in what an
arm was ranked ON was visible on one board and invisible on the other two, and
the board holding both mis-wired arms was one of the blind ones.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

_BUCKET = "alpha-engine-research"
_ENTRY = "2026-06-01"
_AS_OF = "2026-08-04"
_TICKERS = [f"T{i:02d}" for i in range(60)]


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


class _Panel:
    def __init__(self):
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update(
            {t: float(c) for t, c in closes.items()}
        )
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


def _shadow(s3, arm: str, date_str: str, n: int, pool_source: str, pool_size: int):
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"signals_shadow/{arm}/{date_str}/signals.json",
        Body=json.dumps({
            "signals": {
                t: {"signal": "ENTER", "score": float(100 - i)}
                for i, t in enumerate(_TICKERS[:n])
            },
            "arm_pool": {
                "pool_source": pool_source, "pool_size": pool_size, "n_selected": n,
            },
        }).encode(),
    )


def _panel(*dates: str) -> _Panel:
    p = _Panel()
    for d in dates:
        p.put(d, {**dict.fromkeys(_TICKERS, 100.0), "SPY": 100.0})
    for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
        p.put(d, {
            **{t: 100.0 + (60 - i) * 0.5 for i, t in enumerate(_TICKERS)},
            "SPY": 105.0,
        })
    return p


def _board(s3, panel) -> dict:
    from scoring.leaderboard_producers import build_producer_leaderboard

    res = build_producer_leaderboard(
        s3, _BUCKET, _AS_OF, closes_panel_loader=panel.loader()
    )
    assert res["status"] == "ok", res
    return json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())


def _finding(row: dict, reason: str) -> dict | None:
    return next(
        (f for f in (row.get("degraded_input") or []) if f["reason"] == reason), None
    )


class TestTheMarkIsOnTheProducerBoard:
    def test_a_mis_wired_pool_date_is_marked_on_the_producer_board(self, s3):
        _shadow(s3, "predictor_from_60", _ENTRY, 20,
                "predictor_cut:attractiveness_top_20", 1)
        row = {
            r["name"]: r for r in _board(s3, _panel(_ENTRY))["specs"]
        }["predictor_from_60"]
        f = _finding(row, "mis_wired_predictor_pool")
        assert f is not None, "the mis-wired date must be named on the row"
        assert f["dates"] == [_ENTRY]
        assert f["fraction_of_scored"] == 1.0
        assert f["issue"] == "alpha-engine-config-I11396"
        assert f["remedy"] == "marked_unusable"
        assert "no input left to replay" in f["remedy_reason"]

    def test_the_board_names_every_affected_arm_at_the_top_level(self, s3):
        _shadow(s3, "predictor_from_60", _ENTRY, 20,
                "predictor_cut:attractiveness_top_20", 1)
        _shadow(s3, "tech_score_20", _ENTRY, 20,
                "pinned_cut:attractiveness_top_60:tech_score", 60)
        board = _board(s3, _panel(_ENTRY))
        assert board["input_quality"]["mis_wired_predictor_pool"]["arms_affected"] == [
            "predictor_from_60"
        ]

    def test_a_correctly_wired_arm_carries_an_explicit_null(self, s3):
        """"checked, clean" and "never checked" must never render identically
        (§7.2) — and an arm that never touched the bad join must not be tarred
        by a flag belonging to its co-arms."""
        _shadow(s3, "tech_score_20", _ENTRY, 20,
                "pinned_cut:attractiveness_top_60:tech_score", 60)
        row = {
            r["name"]: r for r in _board(s3, _panel(_ENTRY))["specs"]
        }["tech_score_20"]
        assert row["degraded_input"] is None

    def test_a_partly_affected_arm_reports_the_fraction_not_a_boolean(self, s3):
        """"1 of 2 dates" and "2 of 2" are different verdicts about the same
        arm, and a promotion consumer must be able to discount rather than only
        refuse."""
        _shadow(s3, "predictor_from_60", _ENTRY, 20,
                "predictor_cut:attractiveness_top_20", 1)
        _shadow(s3, "predictor_from_60", "2026-06-02", 20,
                "pinned_cut:attractiveness_top_60:predicted_alpha", 60)
        row = {
            r["name"]: r
            for r in _board(s3, _panel(_ENTRY, "2026-06-02"))["specs"]
        }["predictor_from_60"]
        f = _finding(row, "mis_wired_predictor_pool")
        assert f["dates"] == [_ENTRY]
        assert f["n_dates_scored"] == 2
        assert f["fraction_of_scored"] == 0.5


class TestTheConditionIsDerivedNotListed:
    def test_no_affected_date_literal_is_hardcoded_in_the_scorer(self):
        """A hand-kept date list would drift from the S3 objects it describes.
        The condition IS the pool source, so that is what the code tests."""
        import inspect

        import scoring.leaderboard_producers as lp

        src = inspect.getsource(lp._mis_wired_predictor_dates)
        assert "2026-" not in src

    def test_the_prefix_is_named_once_and_consumed(self):
        from scoring.leaderboard_producers import (
            _MIS_WIRED_PREDICTOR_POOL_PREFIX,
            _mis_wired_predictor_dates,
        )
        from scoring.leaderboard_scoring import SpecDay, SpecHistory

        assert _MIS_WIRED_PREDICTOR_POOL_PREFIX == "predictor_cut:"
        arm = SpecHistory(name="x", kind="retired")
        arm.by_date["d1"] = SpecDay(ranked=["A"], pool_source="predictor_cut:foo")
        arm.by_date["d2"] = SpecDay(ranked=["A"], pool_source="pinned_cut:foo")
        arm.by_date["d3"] = SpecDay(ranked=["A"])  # surface records no pool
        assert _mis_wired_predictor_dates(arm, ["d1", "d2", "d3"]) == ["d1"]
