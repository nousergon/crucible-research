"""The funnel's own cuts, scored against the population they narrowed
(alpha-engine-config-I7584).

Brian, 2026-08-17: *"track the performance of scanner top 60, scanner top 20 and
think tank over time."*

Deliberately a SEPARATE slot from `scanner`. That slot count-matches every arm
at 50 because its arms are competing candidate-generation rules and §4 forbids
confounding a win between rule and breadth. These arms are the funnel's own
stages, which differ in breadth by definition — `attractiveness_top_20` is the
HEAD of `attractiveness_top_60` — so asking which beats the other is incoherent,
and truncating both to a common width measures an artifact nobody consumes.

RED on origin/main at d05279db: neither the slot nor the producer exists.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.leaderboard_producers import (  # noqa: E402
    _load_cut_specs,
    build_cuts_leaderboard,
)
from scoring.leaderboard_scoring import LEADERBOARD_SLOTS, slot_spec  # noqa: E402
from scoring.universe_membership import (  # noqa: E402
    FEED_CUT_NAME,
    GATE_BASELINE_CUT,
    PREDICTOR_UNIVERSE_CUT,
)

DATES = ["2026-07-01", "2026-07-02"]
FEED = [f"T{i:03d}" for i in range(60)]
CHAMP = FEED[:20]
GATE = [f"G{i:03d}" for i in range(60)]


class _S3:
    """Minimal S3 double: dated membership objects plus a paginator."""

    def __init__(self):
        self.written: dict[str, dict] = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            body = json.dumps({
                "cuts": {
                    FEED_CUT_NAME: {"tickers": FEED},
                    PREDICTOR_UNIVERSE_CUT: {"tickers": CHAMP},
                    GATE_BASELINE_CUT: {"tickers": GATE},
                }
            }).encode()
            return {"Body": _Body(body)}
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def get_paginator(self, _op):
        return _Paginator()

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803
        self.written[Key] = json.loads(Body)


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class _Paginator:
    def paginate(self, Bucket, Prefix):  # noqa: N803
        yield {"Contents": [
            {"Key": f"{Prefix}{d}/membership.json"} for d in DATES
        ] + [{"Key": f"{Prefix}latest.json"}]}


# ── The slot ─────────────────────────────────────────────────────────────────


def test_cuts_slot_exists_and_ranks_on_population():
    spec = slot_spec("cuts")
    assert spec.primary_metric == "topn_alpha_vs_population"


def test_cuts_slot_opts_out_of_count_matching_explicitly():
    """Mixed widths must be a declared property, never an accident."""
    assert slot_spec("cuts").per_arm_width is True


def test_competing_slots_keep_count_matching():
    """The guard that keeps this split from quietly collapsing back: a slot
    whose arms ARE competing selection rules must stay count-matched."""
    for slot_id in ("scanner", "producer"):
        spec = LEADERBOARD_SLOTS[slot_id]
        assert spec.per_arm_width is False, slot_id
        assert spec.top_n > 0, slot_id


# ── The loader ───────────────────────────────────────────────────────────────


def test_loader_reads_all_three_cuts_with_their_widths():
    arms, widths = _load_cut_specs(_S3(), "b", DATES)
    assert {a.name for a in arms} == {
        FEED_CUT_NAME, PREDICTOR_UNIVERSE_CUT, GATE_BASELINE_CUT,
    }
    assert widths[FEED_CUT_NAME] == 60
    assert widths[PREDICTOR_UNIVERSE_CUT] == 20
    assert widths[GATE_BASELINE_CUT] == 60


def test_loader_takes_cut_names_from_constants_not_literals():
    """PR648 renamed the gate cut. A literal in the loader would have stopped
    matching silently, and the arm would have vanished from the board with no
    error anywhere."""
    import inspect

    import scoring.leaderboard_producers as lp

    src = inspect.getsource(lp._load_cut_specs)
    assert "scanner_candidates" not in src
    assert "GATE_BASELINE_CUT" in src


# ── The artifact ─────────────────────────────────────────────────────────────


@pytest.fixture
def built():
    s3 = _S3()
    realized = {d: dict.fromkeys(FEED + GATE, 0.01) | {"SPY": 0.005} for d in DATES}
    population = dict.fromkeys(DATES, 0.002)
    with patch(
        "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
        return_value=(
            {21: realized, 126: realized, 252: realized},
            {},
            {21: population, 126: population, 252: population},
        ),
    ):
        out = build_cuts_leaderboard(s3, "b", "2026-07-03")
    return out, s3


def test_artifact_is_written_to_its_own_key(built):
    out, s3 = built
    assert out["status"] == "ok"
    assert out["key"] == "research/cuts_leaderboard/2026-07-03.json"
    assert "research/cuts_leaderboard/2026-07-03.json" in s3.written


def test_every_row_records_its_own_width(built):
    """Without this a reader cannot tell that two rows in one table were
    measured at different breadths."""
    out, _ = built
    for block in out["leaderboard"]["horizons"]:
        for row in block["specs"]:
            assert row["top_n"] in (20, 60), row


def test_slot_level_top_n_is_null_not_a_number(built):
    """A single number here would be read as the width every row used."""
    lb = out_lb(built)
    assert lb["top_n"] is None
    assert lb["per_arm_width"] is True


def out_lb(built):
    return built[0]["leaderboard"]


def test_all_three_arms_present_at_every_horizon(built):
    lb = out_lb(built)
    for block in lb["horizons"]:
        names = {r["name"] for r in block["specs"]}
        assert names == {FEED_CUT_NAME, PREDICTOR_UNIVERSE_CUT, GATE_BASELINE_CUT}


def test_no_champion_so_vs_champion_is_null_by_construction(built):
    """These arms are not competing — a null here is the design, not a miss."""
    lb = out_lb(built)
    assert lb["champion"] is None
    for block in lb["horizons"]:
        for row in block["specs"]:
            assert row["topn_alpha_vs_champion"] is None


def test_population_metric_is_populated(built):
    lb = out_lb(built)
    for block in lb["horizons"]:
        for row in block["specs"]:
            assert row["topn_alpha_vs_population"] is not None
            # every ticker returned 0.01 against a 0.002 population
            assert row["topn_alpha_vs_population"]["mean"] == pytest.approx(0.008)


# ── Fail-soft ────────────────────────────────────────────────────────────────


def test_build_never_raises_into_the_live_path():
    class _Broken(_S3):
        def get_paginator(self, _op):
            raise RuntimeError("s3 down")

    with patch("scoring.leaderboard_producers.publish_observe_alert") as alert:
        out = build_cuts_leaderboard(_Broken(), "b", "2026-07-03")
    assert out["status"] == "error"
    assert alert.called, "a failed observe build must be loud, never silent"


# ── The panel cache (alpha-engine-config-I7584) ──────────────────────────────
#
# No manual ``lp._PANEL_CACHE.clear()`` before each test (removed,
# alpha-engine-config-I7643): every fresh ``_loader`` below is a distinct
# object, the cache key now holds the loader OBJECT (PR657), and
# ``_load_closes_panel`` unconditionally clears+repopulates the (size-bounded-
# to-1) cache on every miss — so a leftover entry from a prior test can never
# match a new test's key and is always overwritten on first use. A cache that
# still needed manual clearing here would be the workaround this class
# accumulated, not a fix.


def test_panel_is_read_once_across_leaderboards_in_one_invocation():
    """Three leaderboards now each need the FULL universe panel. Without the
    in-process dedup that is three identical ~904-symbol ArcticDB slices sized
    to 252 sessions — the dominant cost of the observe path, paid three times
    for one answer."""
    import scoring.leaderboard_producers as lp

    calls = []

    def _loader(bucket, entry_dates, horizon_days, symbols):
        calls.append((bucket, tuple(entry_dates), horizon_days, symbols))
        return {d: {"AAA": 1.0} for d in entry_dates}

    for _ in range(3):
        lp._load_closes_panel("b", DATES, 252, None, _loader)
    assert len(calls) == 1


def test_panel_cache_does_not_serve_a_different_cohort():
    """A warm Lambda crossing a trading day must not be handed yesterday's
    panel."""
    import scoring.leaderboard_producers as lp

    calls = []

    def _loader(bucket, entry_dates, horizon_days, symbols):
        calls.append(tuple(entry_dates))
        return {d: {"AAA": 1.0} for d in entry_dates}

    lp._load_closes_panel("b", DATES, 252, None, _loader)
    lp._load_closes_panel("b", [*DATES, "2026-07-03"], 252, None, _loader)
    assert len(calls) == 2


def test_panel_cache_does_not_serve_a_narrowed_panel_as_a_full_one():
    """The population baseline is only valid over a full-universe panel
    (config-I7587); a cache collision here would resurrect that defect."""
    import scoring.leaderboard_producers as lp

    calls = []

    def _loader(bucket, entry_dates, horizon_days, symbols):
        calls.append(symbols)
        return {d: {"AAA": 1.0} for d in entry_dates}

    lp._load_closes_panel("b", DATES, 252, {"AAA"}, _loader)
    lp._load_closes_panel("b", DATES, 252, None, _loader)
    assert len(calls) == 2
