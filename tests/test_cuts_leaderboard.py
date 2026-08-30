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


def test_champion_names_the_live_cut_and_carries_the_paired_difference(built):
    """The board names the arm that is actually serving, and now holds the
    arm-vs-arm difference it never used to.

    Before alpha-engine-config-I8026 this asserted ``champion is None``, on the
    reading that the board's arms are funnel STAGES and stages do not compete.
    That was true of the three arms the board then carried, and it stopped
    being true when the universe-cut slot's arms joined them: the slot has a
    champion, ``cut_promotion.decide_cut_champion`` decides from this board,
    and §3 calls a null champion field on a slot that has one a broken
    leaderboard.

    Until alpha-engine-config-I8263 this ALSO asserted that
    ``topn_alpha_vs_champion`` stays null, because each arm is scored in its
    own single-arm pass and no pass ever held two arms to difference. That was
    a faithful description of the code and a defect in it: the paired
    difference is the highest-power statistic the count-matched same-date
    design makes available, and discarding it left the slot comparing two
    independently-estimated means at ~52 observations a year. It is now
    computed outside the pass, where both arms are in scope.

    The CHAMPION's own row keeps a null — an arm has no difference against
    itself, and a zero there would read as a measured tie.
    """
    lb = out_lb(built)
    assert lb["champion"] == FEED_CUT_NAME
    champion_rows = [
        r for block in lb["horizons"] for r in block["specs"] if r["name"] == FEED_CUT_NAME
    ]
    assert champion_rows, "the champion arm must still carry a scored row (§3)"
    assert all(r["kind"] == "champion" for r in champion_rows)
    assert all(r["topn_alpha_vs_champion"] is None for r in champion_rows), (
        "an arm has no paired difference against itself"
    )

    # Every count-matched challenger that shares cohort dates with the champion
    # must now carry one. A null here is the I8263 defect returning.
    champion_width = next(
        r["top_n"] for block in lb["horizons"] for r in block["specs"]
        if r["name"] == FEED_CUT_NAME
    )
    paired_rows = [
        r for block in lb["horizons"] for r in block["specs"]
        if r["kind"] == "challenger"
        and r["top_n"] == champion_width
        and (r.get("n_dates_scored") or 0) > 0
    ]
    assert paired_rows, "fixture must carry at least one count-matched challenger"
    for row in paired_rows:
        assert row["topn_alpha_vs_champion"] is not None, row["name"]


def test_a_challenger_at_a_different_width_gets_no_paired_difference(built):
    """Width-gated, not width-adjusted (champion-challenger-policy.md §4).

    A paired difference between a 60-wide arm and a 20-wide champion measures
    BREADTH, not the selection rule — the question the slot is asking. A null
    is the correct output for such a row, and it must not be filled in with a
    number that answers a different question.
    """
    lb = out_lb(built)
    champion_width = next(
        r["top_n"] for block in lb["horizons"] for r in block["specs"]
        if r["name"] == FEED_CUT_NAME
    )
    for block in lb["horizons"]:
        for row in block["specs"]:
            if row["kind"] == "challenger" and row["top_n"] != champion_width:
                assert row["topn_alpha_vs_champion"] is None, row["name"]


def test_every_horizon_declares_whether_its_observations_overlap(built):
    """The dependence assumption is stated ON the artifact, never inferred.

    An SE is only interpretable against the assumption behind it, and no
    consumer can recover that from the number. ``overlap_lags: 0`` means
    genuinely non-overlapping — not "not checked" (alpha-engine-config-I8263).
    """
    lb = out_lb(built)
    for block in lb["horizons"]:
        assert "overlap_lags" in block, block["horizon_days"]
        assert isinstance(block["overlap_lags"], int)
        assert block["observations_overlap"] == (block["overlap_lags"] > 0)


def test_a_longer_horizon_never_gets_a_smaller_overlap_correction(built):
    """Overlap is monotone in the horizon at fixed cohort spacing.

    The 126- and 252-session blocks are the MORE distorted ones, not less, so a
    single lag count applied across all horizons would under-correct exactly
    where the correction matters most.
    """
    lb = out_lb(built)
    by_h = sorted(
        ((b["horizon_days"], b["overlap_lags"]) for b in lb["horizons"]),
        key=lambda p: p[0],
    )
    lags = [x[1] for x in by_h]
    assert lags == sorted(lags), by_h


def test_an_overlapping_metric_never_publishes_an_iid_t_stat(built):
    """The defect in one assertion.

    A t-stat computed with ``sd/sqrt(n)`` over overlapping windows understates
    the SE by roughly ``sqrt(lags+1)`` and reads as significance. Any block
    declaring overlap must carry a HAC method, or no SE at all — never `iid`.
    """
    lb = out_lb(built)
    for block in lb["horizons"]:
        if not block["observations_overlap"]:
            continue
        for row in block["specs"]:
            for key in (
                "realized_rank_ic", "topn_alpha_vs_champion",
                "topn_alpha_vs_benchmark", "topn_alpha_vs_population",
            ):
                metric = row.get(key)
                if not isinstance(metric, dict):
                    continue
                assert metric.get("se_method") != "iid", (
                    f"{row['name']}.{key} at {block['horizon_days']}d publishes "
                    "an iid SE over overlapping windows"
                )


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


# ── The universe-cut SLOT's arms (alpha-engine-config-I8026) ─────────────────
#
# RED on origin/main at 869a95c4: `_load_cut_specs` resolved its arm list from
# three constants — the funnel's stages — so `tech_score_top_60`,
# `attractiveness_momzero_top_60` and `attractiveness_mom121_top_60` appeared
# on no board anywhere, and `cut_promotion.decide_cut_champion` read a board
# that had never carried one of the two arms it compares. Every one of these
# tests fails there: the rows do not exist.

from scoring.cut_promotion import (  # noqa: E402
    CUT_PROMOTION_SLOT,
    _veto_horizon,
    decide_cut_champion,
)
from scoring.universe_membership import (  # noqa: E402
    ATTRACTIVENESS_FEED_TOP_N,
    CHALLENGER_CUT_PREFIX,
    HARD3_CUT_PREFIX,
    MOMZERO_CUT_PREFIX,
    PROMOTABLE_CUTS,
    TECH_SCORE_CUT_PREFIX,
    TECH_SCORE_RANKS_FIELD,
)

TECH_CUT = f"{TECH_SCORE_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"
MOMZERO_CUT = f"{MOMZERO_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"
MOM121_CUT = f"{CHALLENGER_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"
# The fifth arm. Absent from this fixture until alpha-engine-config-I9272,
# which is itself the finding: it has been a registered arm of the slot since
# 2026-08-28 and no test in this file seeded it, so every "every arm reaches
# every horizon" assertion below was passing over four arms out of five.
HARD3_CUT = f"{HARD3_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"

# Each slot arm picks a different 60 — a real comparison, not four clones.
TECH = [f"H{i:03d}" for i in range(60)]
MOMZERO = FEED[:30] + [f"Z{i:03d}" for i in range(30)]
MOM121 = FEED[:10] + [f"M{i:03d}" for i in range(50)]
HARD3 = FEED[:20] + [f"D{i:03d}" for i in range(40)]
SLOT_UNIVERSE = sorted(set(FEED + GATE + TECH + MOMZERO + MOM121 + HARD3))


class _SlotS3(_S3):
    """Membership carrying the funnel's stages AND all five of the slot's arms."""

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            body = json.dumps({
                "cuts": {
                    FEED_CUT_NAME: {"tickers": FEED},
                    PREDICTOR_UNIVERSE_CUT: {"tickers": CHAMP},
                    CHAMPION_CUT: {"tickers": GATE},
                    TECH_CUT: {"tickers": TECH},
                    MOMZERO_CUT: {"tickers": MOMZERO},
                    MOM121_CUT: {"tickers": MOM121},
                    HARD3_CUT: {"tickers": HARD3},
                },
                # Only the tech_score arm publishes a rank table; the two
                # momentum arms deliberately do not (see the loader's comment).
                TECH_SCORE_RANKS_FIELD: {
                    t: {"tech_score_rank": i + 1} for i, t in enumerate(TECH)
                },
            }).encode()
            return {"Body": _Body(body)}
        return super().get_object(Bucket=Bucket, Key=Key)


@pytest.fixture
def slot_built():
    s3 = _SlotS3()
    # Returns VARY by ticker here, unlike the funnel-stage fixture above. A
    # constant panel makes every rank-IC null for want of variance, which would
    # hide the difference this fixture exists to show: an arm WITH a rank table
    # gets a real rank-IC, an arm without one gets a missing value.
    realized = {
        d: {t: 0.01 + (i * 1e-4) for i, t in enumerate(SLOT_UNIVERSE)} | {"SPY": 0.005}
        for d in DATES
    }
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


def test_loader_reads_the_slot_arms_too():
    arms, widths = _load_cut_specs(_SlotS3(), "b", DATES)
    names = {a.name for a in arms}
    assert {TECH_CUT, MOMZERO_CUT, MOM121_CUT, HARD3_CUT} <= names
    # §4: the slot is count-matched by construction — every arm at 60.
    for arm in (FEED_CUT_NAME, TECH_CUT, MOMZERO_CUT, MOM121_CUT, HARD3_CUT):
        assert widths[arm] == ATTRACTIVENESS_FEED_TOP_N, arm


def test_every_promotable_arm_reaches_every_horizon(slot_built):
    """The failure this exists for: `cut_promotion` decides between the two
    members of PROMOTABLE_CUTS at 126 sessions, and one of them was on no
    board at any horizon."""
    lb = out_lb(slot_built)
    for block in lb["horizons"]:
        names = [r["name"] for r in block["specs"]]
        for arm in PROMOTABLE_CUTS:
            assert arm in names, (block["horizon_days"], arm)
        # §3 + the I7645 regression: exactly one row per arm per horizon.
        assert len(names) == len(set(names)), (block["horizon_days"], names)


def test_the_shadow_arms_are_scored_not_merely_registered(slot_built):
    """An arm that writes shadow output but is scored nowhere is a rumour
    (champion-challenger-policy.md §3)."""
    lb = out_lb(slot_built)
    block = next(b for b in lb["horizons"] if b["horizon_days"] == 21)
    rows = {r["name"]: r for r in block["specs"]}
    for arm in (MOMZERO_CUT, MOM121_CUT):
        assert rows[arm]["topn_alpha_vs_population"] is not None, arm
        assert rows[arm]["n_dates_scored"] == len(DATES), arm


def test_momentum_arms_report_a_missing_rank_ic_not_an_alphabetical_one(slot_built):
    """Their membership carries no rank table, so there is no order to
    correlate. A number here would be a Spearman against the alphabet."""
    lb = out_lb(slot_built)
    block = next(b for b in lb["horizons"] if b["horizon_days"] == 21)
    rows = {r["name"]: r for r in block["specs"]}
    for arm in (MOMZERO_CUT, MOM121_CUT):
        assert rows[arm]["realized_rank_ic"] is None, arm
    # The tech_score arm DOES publish one, so its rank-IC is real.
    assert rows[TECH_CUT]["realized_rank_ic"] is not None


def test_board_names_the_serving_arm_as_champion(slot_built):
    lb = out_lb(slot_built)
    assert lb["champion"] == FEED_CUT_NAME
    block = next(b for b in lb["horizons"] if b["horizon_days"] == 21)
    kinds = {r["name"]: r["kind"] for r in block["specs"]}
    assert kinds[FEED_CUT_NAME] == "champion"
    assert kinds[TECH_CUT] == "challenger"


def test_every_promotable_arm_is_readable_by_the_long_horizon_veto(slot_built):
    """End to end onto the post-I8261 consumer.

    The DECISION no longer comes from this board (alpha-engine-config-I8261,
    Brian's ruling 2026-08-24 — it comes from the weekly ledger), but the
    corroborating VETO still does, and a veto cannot be taken on a horizon
    where an arm it compares has no row at all. That is what a board without
    `tech_score_top_60` forced on every single evaluation.

    An IMMATURE block is still a legitimate non-blocking outcome here and is
    what this fixture produces at 126/252 — the property under test is that the
    reason is maturity, never a missing row.
    """
    lb = out_lb(slot_built)
    for horizon in CUT_PROMOTION_SLOT.corroborating_horizons_days:
        entry = _veto_horizon(lb, CUT_PROMOTION_SLOT, FEED_CUT_NAME, horizon)
        assert "rows at" not in entry["note"], (horizon, entry["note"])
    # And the decision path itself still produces a full arms block.
    decision = decide_cut_champion(
        ledger_rows=None, board=lb, champion_before=FEED_CUT_NAME,
        decided_on="2026-08-29",
    )
    assert set(decision.arms) == set(CUT_PROMOTION_SLOT.arms)


def test_unreadable_membership_writes_the_verdict_instead_of_going_silent():
    """§7.2 / ARCHITECTURE §133: an unmeasurable cycle is a written verdict.

    A board that stops appearing and a board reporting "nothing was scorable"
    are the same S3 state to every consumer, and `cut_promotion` renders the
    first as `board_missing` — a hold whose stated reason is wrong. RED before
    alpha-engine-config-I8026: this path returned the dict to its caller and
    wrote nothing.
    """

    class _EmptyS3(_S3):
        def get_object(self, Bucket, Key):  # noqa: N803
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    s3 = _EmptyS3()
    out = build_cuts_leaderboard(s3, "b", "2026-07-03")
    assert out["status"] == "unmeasurable"
    written = s3.written.get("research/cuts_leaderboard/2026-07-03.json")
    assert written is not None, "the verdict must reach S3, not just the caller"
    assert written["status"] == "unmeasurable"
    assert "no membership cuts readable" in written["unmeasurable_reason"]


# ── Degraded fundamental inputs, per arm and per date (I8255 d2) ─────────────
#
# RED on origin/main at c38c2350: no row on any cuts board carries a
# `degraded_input` key at all, and nothing in `leaderboard_producers` imports
# the measured window — so a promotion consumer reading the board cannot tell a
# cut ranked on the saturated-placeholder cross-section from one ranked on real
# fundamentals.

from scoring.universe_board import (  # noqa: E402
    DEGENERATE_FUNDAMENTALS_RUN_DATE_WINDOW,
)

IN_WINDOW = ["2026-08-05", "2026-08-06"]
OUT_WINDOW = ["2026-08-25", "2026-08-26"]
MIXED_DATES = IN_WINDOW + OUT_WINDOW


class _MixedWindowS3(_S3):
    """One arm straddles the degenerate window; another sits entirely outside.

    The board-level boolean this replaces would tar both identically.
    """

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            d = Key.split("/")[1]
            cuts = {FEED_CUT_NAME: {"tickers": FEED}, CHAMPION_CUT: {"tickers": GATE}}
            if d in OUT_WINDOW:
                # Only present after the fundamentals repair — no degraded date.
                cuts[PREDICTOR_UNIVERSE_CUT] = {"tickers": CHAMP}
            return {"Body": _Body(json.dumps({"cuts": cuts}).encode())}
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def get_paginator(self, _op):
        return _DatedPaginator(MIXED_DATES)


class _DatedPaginator:
    def __init__(self, dates):
        self._dates = dates

    def paginate(self, Bucket, Prefix):  # noqa: N803
        yield {"Contents": [
            {"Key": f"{Prefix}{d}/membership.json"} for d in self._dates
        ] + [{"Key": f"{Prefix}latest.json"}]}


@pytest.fixture
def mixed_built():
    s3 = _MixedWindowS3()
    realized = {
        d: dict.fromkeys(FEED + GATE + CHAMP, 0.01) | {"SPY": 0.005}
        for d in MIXED_DATES
    }
    population = dict.fromkeys(MIXED_DATES, 0.002)
    with patch(
        "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
        return_value=(
            {21: realized, 126: realized, 252: realized},
            {},
            {21: population, 126: population, 252: population},
        ),
    ):
        out = build_cuts_leaderboard(s3, "b", "2026-08-27")
    return out, s3


def test_degraded_input_is_flagged_per_arm_and_per_date(mixed_built):
    """Five vendor fundamental fields were saturated placeholders — as few as
    ONE distinct value across ~899 names — on every scanner board dated
    2026-08-03..2026-08-19 (measured 2026-08-24). An arm ranked on those dates
    was ranked on three near-constant pillars, and the board must say so
    without also condemning an arm that never scored a date in the window."""
    lb = out_lb(mixed_built)
    block = lb["horizons"][0]
    rows = {r["name"]: r for r in block["specs"]}

    straddling = rows[FEED_CUT_NAME]["degraded_input"]
    assert straddling is not None
    assert straddling["dates"] == IN_WINDOW
    assert straddling["n_dates"] == 2
    assert straddling["n_dates_scored"] == 4
    assert straddling["fraction_of_scored"] == 0.5
    assert straddling["reason"] == "degenerate_fundamentals"

    # The arm that only ever scored post-repair dates carries an explicit null:
    # "checked, clean" must never render as "never checked" (§7.2).
    assert PREDICTOR_UNIVERSE_CUT in rows
    assert rows[PREDICTOR_UNIVERSE_CUT]["degraded_input"] is None

    assert lb["input_quality"]["degenerate_fundamentals"]["arms_affected"] == sorted(
        {FEED_CUT_NAME, CHAMPION_CUT}
    )


def test_the_degenerate_window_is_consumed_not_restated(mixed_built):
    """A second copy of the dates is how the producer-side distinctness floor
    and the consumer-side flag stop agreeing. The window is imported from
    `scoring.universe_board`, where PR735 put it."""
    import inspect

    import scoring.leaderboard_producers as lp

    src = inspect.getsource(lp)
    assert "2026-08-03" not in src, "the window literal must not be restated here"
    assert "is_degenerate_fundamentals_run_date" in src

    lb = out_lb(mixed_built)
    assert lb["input_quality"]["degenerate_fundamentals"]["window"] == list(
        DEGENERATE_FUNDAMENTALS_RUN_DATE_WINDOW
    )


# ── One cut, one observation (alpha-engine-config-I8269) ────────────────────
#
# RED on origin/main at c38c2350: `_load_cut_specs` writes one SpecDay per
# dated prefix, so three prefixes carrying ONE held cut score as three
# independent cohort dates and every horizon's evidence is overstated by the
# hold length — on the board `cut_promotion.decide_cut_champion` reads.

HELD_DATES = ["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-13"]
FRESH_CUT = [f"F{i:03d}" for i in range(60)]


class _HeldCutS3(_S3):
    """Weekly-cadence membership: three prefixes carry ONE cut, then a re-cut.

    Shaped exactly as `carry_forward_cuts` writes it under
    `CUT_REFRESH_CADENCE=weekly` — the cuts are byte-identical and
    `cut_effective_date` points back at the date the cut was FORMED.
    """

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            d = Key.split("/")[1]
            fresh = d == "2026-07-13"
            body = {
                "run_date": d,
                "cut_effective_date": "2026-07-13" if fresh else "2026-07-06",
                "cut_refresh_cadence": "weekly",
                "cuts": {
                    FEED_CUT_NAME: {"tickers": FRESH_CUT if fresh else FEED},
                    CHAMPION_CUT: {"tickers": GATE},
                },
            }
            return {"Body": _Body(json.dumps(body).encode())}
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def get_paginator(self, _op):
        return _DatedPaginator(HELD_DATES)


class _NoEffectiveDateS3(_HeldCutS3):
    """Pre-2026-08-10 artifacts carry NO `cut_effective_date` at all."""

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            d = Key.split("/")[1]
            body = {
                "run_date": d,
                "cuts": {FEED_CUT_NAME: {"tickers": FEED + [f"X{d[-2:]}"]}},
            }
            return {"Body": _Body(json.dumps(body).encode())}
        return super().get_object(Bucket=Bucket, Key=Key)


def test_a_held_cut_is_one_cohort_date_not_one_per_prefix():
    """`universe_membership/{date}` is written on EVERY run; the cut is
    re-derived on `cut_refresh_cadence()`. Three prefixes, one cut, one
    observation — and the hold recorded rather than performed silently."""
    arms, widths = _load_cut_specs(_HeldCutS3(), "b", HELD_DATES)
    feed = next(a for a in arms if a.name == FEED_CUT_NAME)
    assert sorted(feed.by_date) == ["2026-07-06", "2026-07-13"]
    assert feed.held_dates == {
        "2026-07-06": ["2026-07-06", "2026-07-07", "2026-07-08"],
        "2026-07-13": ["2026-07-13"],
    }
    # The width is unchanged by the collapse — it is a property of the cut.
    assert widths[FEED_CUT_NAME] == 60
    # The collapse keeps the EARLIEST prefix carrying the effective date, so
    # the observation still joins a realized-return map keyed by cohort date.
    assert feed.by_date["2026-07-06"].ranked, "the scored day must carry picks"


def test_an_artifact_with_no_cut_effective_date_falls_back_to_its_run_date():
    """A `None` key would collapse the whole pre-field history into ONE cohort
    date and destroy 20+ dates of champion evidence — the alpha-engine-config
    -I7631 failure in a new guise. Every such date is counted, and counted
    once."""
    arms, _ = _load_cut_specs(_NoEffectiveDateS3(), "b", HELD_DATES)
    feed = next(a for a in arms if a.name == FEED_CUT_NAME)
    assert sorted(feed.by_date) == sorted(HELD_DATES)
    assert feed.held_dates == {d: [d] for d in HELD_DATES}


@pytest.fixture
def held_built():
    s3 = _HeldCutS3()
    realized = {
        d: dict.fromkeys(FEED + GATE + FRESH_CUT, 0.01) | {"SPY": 0.005}
        for d in HELD_DATES
    }
    population = dict.fromkeys(HELD_DATES, 0.002)
    with patch(
        "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
        return_value=(
            {21: realized, 126: realized, 252: realized},
            {},
            {21: population, 126: population, 252: population},
        ),
    ):
        out = build_cuts_leaderboard(s3, "b", "2026-07-14")
    return out, s3


def test_the_board_reports_held_versus_scored_per_arm(held_built):
    """Reported, not performed silently: `n_dates_scored` counts DECISIONS,
    `n_dates_held` counts the calendar prefixes they spanned. A gap between
    them is the cut being HELD, legible on the artifact instead of inferred
    from the cadence config."""
    lb = out_lb(held_built)
    block = lb["horizons"][0]
    rows = {r["name"]: r for r in block["specs"]}
    feed = rows[FEED_CUT_NAME]
    assert feed["n_dates_scored"] == 2
    assert feed["n_dates_held"] == 4
    assert feed["cut_carry_forward"] is True
    # The gate cut is byte-identical on every prefix here too, so it collapses
    # onto the same two effective dates.
    assert rows[CHAMPION_CUT]["n_dates_held"] == 4


def test_the_overlap_correction_uses_the_dates_the_horizon_actually_scored():
    """Overlap is a property of the observations that ENTERED the mean.

    On the live board the two differ by 5x: the enumerated cohort runs daily
    through August (median gap 1 session) while the 21-session horizon had
    matured only on 8 WEEKLY dates. Deriving the lag from the whole calendar
    asks for ceil(21/1)-1 = 20 lags on 8 observations, which the HAC estimator
    truncates to n-1 and answers from almost no independent information
    (measured 2026-08-24, alpha-engine-config-I8263 d4).
    """
    daily = ["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10",
             "2026-07-17", "2026-07-24", "2026-07-31"]
    scored = ["2026-07-10", "2026-07-17", "2026-07-24", "2026-07-31"]  # weekly

    class _S(_S3):
        def get_object(self, Bucket, Key):  # noqa: N803
            if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
                d = Key.split("/")[1]
                body = {"run_date": d, "cut_effective_date": d,
                        "cuts": {FEED_CUT_NAME: {"tickers": FEED + [f"Y{d[-2:]}"]}}}
                return {"Body": _Body(json.dumps(body).encode())}
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        def get_paginator(self, _op):
            return _DatedPaginator(daily)

    # Only the weekly subset has a matured 21-session window.
    realized = {
        d: dict.fromkeys(FEED + [f"Y{d[-2:]}"], 0.01) | {"SPY": 0.005} for d in scored
    }
    population = dict.fromkeys(scored, 0.002)
    with patch(
        "scoring.leaderboard_producers._resolve_realized_returns_by_horizon",
        return_value=(
            {21: realized, 126: {}, 252: {}},
            {},
            {21: population, 126: {}, 252: {}},
        ),
    ):
        out = build_cuts_leaderboard(_S(), "b", "2026-08-01")

    block = next(b for b in out["leaderboard"]["horizons"] if b["horizon_days"] == 21)
    assert block["cohort_spacing_days"] == 5, block
    assert block["overlap_lags"] == 4, block  # ceil(21/5) - 1


class _MovedUnderHoldS3(_HeldCutS3):
    """The artifact stamps a HELD effective date while this arm's picks move.

    An arm re-derived on a different cadence from the one the doc stamps. The
    collapse must not eat a real observation on the strength of a field that
    describes a different arm.
    """

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key.startswith("universe_membership/") and Key.endswith("membership.json"):
            d = Key.split("/")[1]
            body = {
                "run_date": d,
                "cut_effective_date": "2026-07-06",
                "cuts": {FEED_CUT_NAME: {"tickers": FEED + [f"W{d[-2:]}"]}},
            }
            return {"Body": _Body(json.dumps(body).encode())}
        return super().get_object(Bucket=Bucket, Key=Key)


def test_a_cut_that_actually_moved_is_never_collapsed_into_the_held_one():
    """Discarding a real observation is the opposite failure to counting a held
    one twice, and the more expensive of the two: the evidence is gone rather
    than overstated. The collapse is gated on the arm's picks being identical,
    not on the stamped field alone."""
    arms, _ = _load_cut_specs(_MovedUnderHoldS3(), "b", HELD_DATES)
    feed = next(a for a in arms if a.name == FEED_CUT_NAME)
    assert sorted(feed.by_date) == sorted(HELD_DATES)
    assert feed.held_dates == {d: [d] for d in HELD_DATES}
