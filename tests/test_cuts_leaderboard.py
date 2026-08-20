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
    CHAMPION_CUT,
    FEED_CUT_NAME,
    GATE_BASELINE_CUT,
    GATE_LEGACY_CUT,
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
                    CHAMPION_CUT: {"tickers": GATE},
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
        FEED_CUT_NAME, PREDICTOR_UNIVERSE_CUT, CHAMPION_CUT,
    }
    assert widths[FEED_CUT_NAME] == 60
    assert widths[PREDICTOR_UNIVERSE_CUT] == 20
    assert widths[CHAMPION_CUT] == 60


def test_loader_takes_cut_names_from_constants_not_literals():
    """PR648 renamed the gate cut, then alpha-engine-config-I7818 renamed it
    again. A literal in the loader would have stopped matching silently, and
    the arm would have vanished from the board with no error anywhere."""
    import inspect

    import scoring.leaderboard_producers as lp

    src = inspect.getsource(lp._load_cut_specs)
    assert "scanner_candidates" not in src
    assert "scanner_gate_baseline_60" not in src
    assert "scanner_champion_60" not in src
    assert "CHAMPION_CUT" in src


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
        assert names == {FEED_CUT_NAME, PREDICTOR_UNIVERSE_CUT, CHAMPION_CUT}


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


# ── The live artifact's three defects (alpha-engine-config-I7631) ────────────
#
# Measured on `research/cuts_leaderboard/2026-08-18.json`, the board's FIRST
# live artifact: 5 rows in the 21-day block for 3 arms (the two arms after the
# first appear twice), a `realized_rank_ic` for every cut computed over an
# ALPHABETICAL ticker list, and the gate baseline reporting `insufficient`
# while 20 cohort dates of its history sit in S3 under its documented alias.


class _RankedS3(_S3):
    """Membership as the writer actually emits it: cut tickers SORTED (set
    semantics, `universe_membership._top_n`), with the rank order carried
    separately in the `ranks` / `scanner_ranks` tables.

    The rank tables are DELIBERATELY the reverse of alphabetical order, so a
    consumer reading `tickers` as a ranking is distinguishable from one reading
    the rank table.
    """

    #: dates on which the gate cut exists ONLY under its pre-PR648 name
    #: (``scanner_candidates``) — the oldest of the three spellings.
    LEGACY_DATES = (DATES[0],)
    #: dates on which the gate cut exists ONLY under its PR648-through-I7818
    #: name (``scanner_gate_baseline_60``) — the middle spelling, current
    #: dates use the newest, :data:`CHAMPION_CUT`.
    BASELINE_DATES: tuple[str, ...] = ()

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            date_str = Key.split("/")[1]
            if date_str in self.LEGACY_DATES:
                gate_name = GATE_LEGACY_CUT
            elif date_str in self.BASELINE_DATES:
                gate_name = GATE_BASELINE_CUT
            else:
                gate_name = CHAMPION_CUT
            body = json.dumps({
                "cuts": {
                    FEED_CUT_NAME: {"tickers": sorted(FEED)},
                    PREDICTOR_UNIVERSE_CUT: {"tickers": sorted(CHAMP)},
                    gate_name: {"tickers": sorted(GATE)},
                },
                "ranks": {
                    t: {"attractiveness_rank": i + 1}
                    for i, t in enumerate(sorted(FEED, reverse=True))
                },
                "scanner_ranks": {
                    t: {"tech_score_rank": i + 1}
                    for i, t in enumerate(sorted(GATE, reverse=True))
                },
            }).encode()
            return {"Body": _Body(body)}
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")


def test_cut_order_comes_from_the_rank_table_not_the_ticker_list():
    """`universe_membership._top_n` returns `sorted(...)` — EVERY cut's
    `tickers` is alphabetical, not rank order. Reading it as a ranking makes
    `realized_rank_ic` a Spearman correlation against the alphabet."""
    arms, _ = _load_cut_specs(_RankedS3(), "b", DATES)
    feed = next(a for a in arms if a.name == FEED_CUT_NAME)
    day = feed.by_date[DATES[-1]]
    assert day.ranked == sorted(FEED, reverse=True)
    assert day.rank_ordered is True


def test_a_cut_with_no_rank_source_is_not_passed_off_as_ranked():
    """Absent a rank table the order is unknown, and an unknown order must
    produce a MISSING rank-IC, never one computed over the alphabet."""
    arms, _ = _load_cut_specs(_S3(), "b", DATES)  # no `ranks` block at all
    for arm in arms:
        for day in arm.by_date.values():
            assert day.rank_ordered is False


def test_rank_ic_is_null_when_the_days_order_is_not_a_ranking():
    from scoring.leaderboard_scoring import SpecDay, SpecHistory, score_leaderboard

    hist = SpecHistory(name="x", kind="challenger", by_date={
        d: SpecDay(ranked=sorted(FEED), rank_ordered=False) for d in DATES
    })
    realized = {d: dict.fromkeys(FEED, 0.01) for d in DATES}
    row = score_leaderboard(None, [hist], realized, top_n=60)["specs"][0]
    assert row["realized_rank_ic"] is None
    assert row["n_dates_scored"] == len(DATES)


def test_gate_history_under_the_legacy_alias_is_the_same_arm():
    """PR648 renamed the gate cut; every membership artifact written before it
    carries the incumbent baseline under `scanner_candidates`. Reading only the
    newest name discards that history and reports `insufficient` for a
    baseline whose evidence is sitting in S3."""
    arms, widths = _load_cut_specs(_RankedS3(), "b", DATES)
    gate = next(a for a in arms if a.name == CHAMPION_CUT)
    assert set(gate.by_date) == set(DATES)
    assert widths[CHAMPION_CUT] == 60
    assert GATE_LEGACY_CUT not in {a.name for a in arms}
    assert GATE_BASELINE_CUT not in {a.name for a in arms}


def test_gate_history_spans_all_three_spellings():
    """alpha-engine-config-I7818 added a third spelling in four weeks
    (``scanner_candidates`` -> ``scanner_gate_baseline_60`` ->
    ``scanner_champion_60``). Reading only the newest two would still discard
    the oldest artifacts' history the same way I7631 did for two."""
    s3 = _RankedS3()
    s3.BASELINE_DATES = (DATES[1],)  # DATES[0] stays LEGACY_DATES (oldest)
    arms, widths = _load_cut_specs(s3, "b", DATES)
    gate = next(a for a in arms if a.name == CHAMPION_CUT)
    assert set(gate.by_date) == set(DATES)
    assert widths[CHAMPION_CUT] == 60


@pytest.fixture
def built_ranked():
    s3 = _RankedS3()
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


def test_no_arm_is_scored_twice_in_any_horizon_block(built_ranked):
    """Live 2026-08-18: the 21-day block carried 5 rows for 3 arms. A reader
    averaging or counting that table gets a double-weighted answer."""
    lb = out_lb(built_ranked)
    for block in lb["horizons"]:
        names = [r["name"] for r in block["specs"]]
        assert len(names) == len(set(names)), (block["horizon_days"], names)


def test_top_level_spread_matches_the_primary_block_exactly(built_ranked):
    """The top level is the primary horizon's block spread for continuity —
    it must carry the same rows, not the same LIST OBJECT."""
    lb = out_lb(built_ranked)
    primary = next(b for b in lb["horizons"] if b["horizon_days"] == 21)
    assert lb["specs"] is not primary["specs"]
    assert [r["name"] for r in lb["specs"]] == [r["name"] for r in primary["specs"]]


def test_block_n_dates_counts_every_arms_cohort_not_the_first_arms(built_ranked):
    """Each arm is scored in its own pass, so the first arm's `n_dates` is not
    the board's. An arm whose history starts later must not shrink the count,
    and an arm with more history must not be hidden behind the first one's."""
    lb = out_lb(built_ranked)
    for block in lb["horizons"]:
        assert block["n_dates"] == len(DATES), block["horizon_days"]


# ── Arm-level measurement gaps hidden under a board-level "ok"
# (alpha-engine-config-I7819) ─────────────────────────────────────────────────
#
# `block["n_dates"]` is a UNION across arms: one arm with real history reads
# the whole block as `status: ok` while co-arms sit at `confidence:
# insufficient` and contribute nothing. Measured live on
# `research/cuts_leaderboard/2026-08-19.json`: n_dates=8, board reads
# "measured", and 2 of 3 arms individually scored 0 dates.

_GAP_GATE_DATE = "2026-05-01"  # old enough to have matured by 07-03 — a
# missing score here is a DEFECT (overdue), not a young cohort.
_GAP_PREDICTOR_DATE = "2026-07-02"  # one trading day before as-of — too
# recent to have matured; a missing score here is honest immaturity.
_GAP_FEED_DATES = ("2026-06-01", "2026-06-02")
_GAP_AS_OF = "2026-07-03"
_GAP_DATES = (_GAP_GATE_DATE, *_GAP_FEED_DATES, _GAP_PREDICTOR_DATE)


class _GapS3(_S3):
    """Three arms, three DIFFERENT measurement states on the same board:
    FEED scores normally, GATE's one cohort date is old enough to have
    matured but the realized-return join never produced it (simulated
    defect), PREDICTOR's one cohort date is too recent to have matured yet
    (honest immaturity)."""

    _CUTS_BY_DATE = {
        _GAP_GATE_DATE: {CHAMPION_CUT: {"tickers": GATE}},
        _GAP_FEED_DATES[0]: {FEED_CUT_NAME: {"tickers": FEED}},
        _GAP_FEED_DATES[1]: {FEED_CUT_NAME: {"tickers": FEED}},
        _GAP_PREDICTOR_DATE: {PREDICTOR_UNIVERSE_CUT: {"tickers": CHAMP}},
    }

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            date_str = Key.split("/")[1]
            body = json.dumps({"cuts": self._CUTS_BY_DATE[date_str]}).encode()
            return {"Body": _Body(body)}
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def get_paginator(self, _op):
        class _P:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {"Contents": [
                    {"Key": f"{Prefix}{d}/membership.json"} for d in _GAP_DATES
                ] + [{"Key": f"{Prefix}latest.json"}]}

        return _P()


@pytest.fixture
def built_with_gap():
    s3 = _GapS3()
    realized = {d: dict.fromkeys(FEED, 0.01) | {"SPY": 0.005} for d in _GAP_FEED_DATES}
    population = dict.fromkeys(_GAP_FEED_DATES, 0.002)
    with patch(
        "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
        return_value=(
            {21: realized, 126: realized, 252: realized},
            {},
            {21: population, 126: population, 252: population},
        ),
    ), patch("scoring.leaderboard_producers.publish_observe_alert") as alert:
        out = build_cuts_leaderboard(s3, "b", _GAP_AS_OF)
    return out, s3, alert


def test_block_names_which_arms_it_scored_nothing_for(built_with_gap):
    """The board must not rely on a reader diffing every row's `confidence`
    against every other row's — the gap is named on the block directly."""
    out, _, _ = built_with_gap
    lb = out["leaderboard"]
    block21 = next(b for b in lb["horizons"] if b["horizon_days"] == 21)
    assert block21["status"] == "ok"  # FEED's history keeps the block itself "ok"
    assert block21["n_dates"] > 0  # exactly the "looks measured" state
    assert block21["arms_total"] == 3
    assert block21["arms_unmeasured"] == sorted(
        [CHAMPION_CUT, PREDICTOR_UNIVERSE_CUT],
    )
    assert FEED_CUT_NAME not in block21["arms_unmeasured"]


def test_an_overdue_arm_alerts_but_an_immature_arm_does_not(built_with_gap):
    """GATE's only cohort date is a month old — it should have matured, and
    scoring zero for it is a defect, alerted once. PREDICTOR's only cohort
    date is one trading day old — scoring zero for it is expected and must
    stay silent, or every genuinely new arm pages on day one."""
    _, _, alert = built_with_gap
    messages = [call.kwargs.get("message", "") for call in alert.call_args_list]
    assert any(CHAMPION_CUT in m for m in messages)
    assert not any(PREDICTOR_UNIVERSE_CUT in m for m in messages)
