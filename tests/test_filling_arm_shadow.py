"""alpha-engine-config-I9307 — the champion arm must be measurable on the same
basis as its challengers, and an arm that CANNOT be measured must never render
as one that is merely thin.

Every test here was verified RED against the pre-fix tree
(champion-challenger-policy.md §7.4): a guard that cannot fail is worse than no
guard, because it reads as coverage.
"""

from __future__ import annotations

import json

import pytest

from producers.filling_arms import (
    CHAMPION_TOP_N_DEFAULT,
    FillingShadowError,
    build_shadow_payload,
    rank_by_alpha,
    rank_to_score,
)
from producers.registry import (
    RESEARCH_PRODUCERS,
    SCORE_SOURCE_SIGNALS_LIVE,
    buildable_challenger_producers,
    score_source_for,
)
from scoring.leaderboard_scoring import (
    COHORT_LAG_UNMEASURABLE_DATES,
    MEASURABILITY_MEASURED,
    MEASURABILITY_UNMEASURABLE,
    SpecDay,
    SpecHistory,
    measurability_for,
    score_leaderboard,
)

# ``scanner_predictor_direct`` / ``scanner_top20_predictor`` (the original
# I9307 subject) retired 2026-09-22, superseded by the research slot's own
# board/predictor-ranked arms (alpha-engine-config-I11393). Their successors
# carry the same obligation this file locks down: an arm must build its own
# comparable shadow rather than being scored from somewhere else — every
# arm here except ``thinktank_20``, whose shadow is written by the Think
# Tank's own daily run, the same legitimate build=None split
# ``thinktank_coverage`` used (see producers/research_arms.py).
FILLING_ARMS = (
    "attractiveness_60",
    "attractiveness_20",
    "tech_score_20",
    "predictor_from_60",
)


# ── 1. The arms now build a comparable artifact, unconditionally ─────────────


def test_every_filling_arm_builds_its_own_shadow():
    """RED before the fix: the original two arms carried ``build=None``, so
    ``producers.runner`` skipped them and no ``signals_shadow/`` prefix ever
    existed for either. The champion was therefore scored from an artifact its
    picks are not in. Their research-slot successors must not regress the
    same property."""
    buildable = {spec.name for spec in buildable_challenger_producers()}
    for arm in FILLING_ARMS:
        assert arm in RESEARCH_PRODUCERS, f"{arm} is not registered"
        assert arm in buildable, (
            f"{arm} does not build a shadow, so it can only ever be scored from "
            "somewhere other than where every challenger is scored — which is "
            "the I9279 asymmetry that hid I9307"
        )


def test_no_arm_is_scored_from_the_empty_by_contract_live_artifact():
    """The champion arm was scored from ``signals/{date}/signals.json`` while
    that artifact's live producer emits no ENTER pick at all. Every arm must
    now be scored from its own shadow."""
    for name in RESEARCH_PRODUCERS:
        assert score_source_for(name) != SCORE_SOURCE_SIGNALS_LIVE, (
            f"{name} is scored from the live signals artifact, whose producer is "
            "empty-by-contract — it can never score (alpha-engine-config-I9307)"
        )


def test_registry_refuses_an_arm_scored_from_a_source_that_cannot_carry_it():
    """The structural guard, exercised directly: declaring ``signals_live`` on a
    live arm while the live producer is empty-by-contract must RAISE at
    registry-validation time, not produce a thin row for seven weeks.

    ``no_agent_quant`` (the original I9307 regression's subject) is retired as
    of 2026-09-22, and the guard deliberately skips retired rows — a retired
    arm's declared score_source is historical record, not a live hazard. The
    guard is exercised against ``attractiveness_20``, a LIVE research-slot arm
    (alpha-engine-config-I11393), so the check still proves the guard fires on
    the case that matters: a live arm, not a retired one.
    """
    import dataclasses

    from producers import registry as reg

    original = dict(reg.RESEARCH_PRODUCERS)
    try:
        reg.RESEARCH_PRODUCERS["attractiveness_20"] = dataclasses.replace(
            original["attractiveness_20"], score_source=SCORE_SOURCE_SIGNALS_LIVE,
        )
        with pytest.raises(ValueError, match="EMPTY-BY-CONTRACT"):
            reg._assert_score_source_can_carry_output()
    finally:
        reg.RESEARCH_PRODUCERS.clear()
        reg.RESEARCH_PRODUCERS.update(original)
    # and the real registry is still coherent afterwards
    reg._assert_score_source_can_carry_output()


# ── 2. The shadow payload is on the challengers' contract ───────────────────


def _ranked(n: int) -> list[tuple[str, float]]:
    return [(f"T{i:03d}", 1.0 - i / 100.0) for i in range(n)]


def test_shadow_payload_is_readable_by_the_shared_scorer():
    """The whole point: the artifact this writes must reduce to ENTER picks via
    the same ``_enter_ranked_and_scores`` every challenger goes through."""
    from scoring.leaderboard_producers import _enter_ranked_and_scores

    payload = build_shadow_payload(
        "scanner_predictor_direct", "2026-08-28", _ranked(60),
        pool_size=60, pool_source="predictions_research_free",
    )
    day = _enter_ranked_and_scores(payload)
    assert day.ranked, "the shared scorer read ZERO picks out of the shadow"
    assert len(day.ranked) == CHAMPION_TOP_N_DEFAULT
    assert day.ranked[0] == "T000", "the shadow must preserve the arm's ranking"
    assert day.scores and all(isinstance(v, float) for v in day.scores.values())


def test_shadow_ranking_is_the_executors_rule():
    """Fidelity pin. The executor sorts the pool by ``predicted_alpha``
    descending and maps rank onto [floor, ceiling] with a monotone transform.
    A divergence here means the shadow measures a rule the executor does not
    serve, which is a worse defect than the one being fixed."""
    rows = [("AAA", 0.01), ("BBB", 0.05), ("CCC", 0.03)]
    assert [t for t, _ in rank_by_alpha(rows)] == ["BBB", "CCC", "AAA"]
    # ties break on ticker so the record is reproducible
    assert [t for t, _ in rank_by_alpha([("ZZ", 1.0), ("AA", 1.0)])] == ["AA", "ZZ"]
    assert rank_to_score(0.0, 60.0, 95.0) == 95.0
    assert rank_to_score(1.0, 60.0, 95.0) == 60.0
    payload = build_shadow_payload(
        "scanner_predictor_direct", "2026-08-28", rank_by_alpha(rows),
        pool_size=3, pool_source="predictions_research_free",
    )
    scores = [payload["signals"][t]["score"] for t in ("BBB", "CCC", "AAA")]
    assert scores == sorted(scores, reverse=True), "score must be monotone in rank"


def test_an_empty_pool_raises_rather_than_writing_a_healthy_looking_shadow():
    """No silent degrade on a producer. An empty-but-well-formed shadow is
    exactly the artifact whose existence caused this issue."""
    with pytest.raises(FillingShadowError, match="ZERO ranked names"):
        build_shadow_payload(
            "scanner_predictor_direct", "2026-08-28", [],
            pool_size=0, pool_source="predictions_research_free",
        )


def test_shadow_payload_matches_the_versioned_contract():
    """M0 contract discipline: the artifact has a versioned schema and a
    producer-side contract test at birth."""
    import pathlib

    schema = json.loads(
        (pathlib.Path(__file__).resolve().parents[1]
         / "contracts/arm_shadow_signals.schema.json").read_text()
    )
    payload = build_shadow_payload(
        "scanner_top20_predictor", "2026-08-28", _ranked(20),
        pool_size=20, pool_source="predictor_cut:attractiveness_top_20",
    )
    for field in schema["required"]:
        assert field in payload, f"payload is missing required field {field!r}"
    entry = next(iter(payload["signals"].values()))
    for field in schema["$defs"]["shadow_signal_entry"]["required"]:
        assert field in entry, f"signal entry is missing required field {field!r}"
    assert payload["producer"] == "scanner_top20_predictor", (
        "the artifact must NAME the arm that produced it (§7.5), never carry a "
        "literal that goes stale when the pointer moves"
    )


# ── 3. unmeasurable is distinguishable from thin ────────────────────────────


def _hist(name, dates, kind="challenger", reason=None):
    h = SpecHistory(name=name, kind=kind, unmeasurable_reason=reason)
    for d in dates:
        h.by_date[d] = SpecDay(ranked=["AAA", "BBB"], scores={"AAA": 2.0, "BBB": 1.0})
    return h


COHORT = ["2026-07-17", "2026-07-30", "2026-08-10", "2026-08-14", "2026-08-21", "2026-08-28"]


def test_an_arm_that_stopped_producing_is_unmeasurable_not_thin():
    """THE regression. The champion scored 2026-07-17 and nothing after, while
    every other arm scored through 2026-08-28 — and its row said ``thin``,
    a word meaning "not enough evidence YET", i.e. a state that resolves by
    waiting. Nothing was coming."""
    dead = _hist("scanner_predictor_direct", ["2026-07-17"], kind="champion")
    m, reason = measurability_for(dead, sorted(dead.by_date), COHORT)
    assert m == MEASURABILITY_UNMEASURABLE
    assert reason and "2026-07-17" in reason


def test_a_genuinely_thin_but_live_arm_stays_measured():
    """The other half, and the one that makes the guard worth having: an arm
    with little evidence that is still producing must NOT be flagged. Otherwise
    the new status is noise and gets ignored, which is how the old one died."""
    live = _hist("thinktank_coverage", ["2026-08-21", "2026-08-28"])
    m, reason = measurability_for(live, sorted(live.by_date), COHORT)
    assert m == MEASURABILITY_MEASURED
    assert reason is None


def test_an_arm_scoring_nothing_at_all_is_unmeasurable():
    empty = _hist("scanner_top20_predictor", [])
    m, reason = measurability_for(empty, [], COHORT)
    assert m == MEASURABILITY_UNMEASURABLE
    assert reason and "0 of" in reason


def test_a_loader_supplied_reason_wins():
    """The loader knows more than a date count when it knows anything at all."""
    h = _hist("x", COHORT, reason="declared score source cannot carry this arm")
    m, reason = measurability_for(h, COHORT, COHORT)
    assert m == MEASURABILITY_UNMEASURABLE
    assert reason == "declared score source cannot carry this arm"


def test_the_cohort_lag_window_is_short_enough_to_matter():
    """Seven weeks of silence went unnoticed. The window must be short enough
    that three cycles is the most this class can ever hide."""
    assert COHORT_LAG_UNMEASURABLE_DATES <= 3


def test_the_leaderboard_names_its_unmeasurable_arms():
    """Board level, so a consumer does not have to walk ``specs`` to discover
    the comparison is missing a side — and so an alert can be raised on it."""
    champion = _hist("scanner_predictor_direct", ["2026-07-17"], kind="champion")
    challengers = [_hist("no_agent_quant", COHORT[-3:])]
    realized = {d: {"AAA": 0.01, "BBB": -0.01} for d in COHORT}
    board = score_leaderboard(champion, challengers, realized, benchmark_ticker=None)

    assert "unmeasurable_arms" in board, (
        "the board must state this every cycle — an absent field is unmeasured, "
        "not healthy"
    )
    names = {a["name"] for a in board["unmeasurable_arms"]}
    assert names == {"scanner_predictor_direct"}
    champ_row = next(r for r in board["specs"] if r["kind"] == "champion")
    assert champ_row["measurability"] == MEASURABILITY_UNMEASURABLE
    assert champ_row["unmeasurable_reason"]
    live_row = next(r for r in board["specs"] if r["name"] == "no_agent_quant")
    assert live_row["measurability"] == MEASURABILITY_MEASURED


def test_every_spec_row_carries_measurability():
    """Emitted on every row on every cycle, healthy included."""
    board = score_leaderboard(
        _hist("c", COHORT, kind="champion"),
        [_hist("a", COHORT)],
        {d: {"AAA": 0.01, "BBB": -0.01} for d in COHORT},
        benchmark_ticker=None,
    )
    for row in board["specs"]:
        assert "measurability" in row and "unmeasurable_reason" in row


# ── 4. The loader reads every arm through ONE path ──────────────────────────


class _FakeS3:
    """Records every key read so the test can assert champion and challengers
    were read from the SAME prefix family."""

    def __init__(self, docs):
        self.docs = docs
        self.reads: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        self.reads.append(Key)
        if Key not in self.docs:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": _Body(self.docs[Key])}

    def get_paginator(self, _op):
        """`_load_producer_specs` enumerates `universe_membership/` to resolve
        the cross-surface history import (alpha-engine-config-I11422). This
        fake holds no membership artifacts, so an EMPTY listing is the honest
        answer — and it keeps these tests about what they are about, which is
        which PREFIX FAMILY the champion and challengers are read from."""
        return _EmptyPaginator()


class _EmptyPaginator:
    def paginate(self, **_kw):
        return iter([{"Contents": []}])


class _Body:
    def __init__(self, doc):
        self._doc = doc

    def read(self):
        return json.dumps(self._doc).encode()


def _shadow_doc():
    return {"signals": {"AAA": {"signal": "ENTER", "score": 90.0}}}


def test_champion_and_challengers_are_read_from_the_same_source_family():
    """§4: both sides of the comparison scored from the same source. Before the
    fix the champion alone was read from ``signals/{date}/signals.json`` while
    every challenger was read from ``signals_shadow/{arm}/{date}/signals.json``
    — inside the same function, two lines apart."""
    from scoring.leaderboard_producers import _load_producer_specs

    dates = ["2026-08-21", "2026-08-28"]
    docs = {"config/producer_champion.json": {"champion": "scanner_predictor_direct"}}
    for arm in ("scanner_predictor_direct", "no_agent_quant", "single_agent_quant",
                "scanner_top20_predictor", "thinktank_coverage"):
        for d in dates:
            docs[f"signals_shadow/{arm}/{d}/signals.json"] = _shadow_doc()

    s3 = _FakeS3(docs)
    champion, challengers = _load_producer_specs(s3, "b", dates, as_of="2026-08-28")

    assert champion is not None and champion.name == "scanner_predictor_direct"
    assert sorted(champion.by_date) == dates, (
        "the champion scored nothing — it is still being read from a source "
        "that cannot carry its picks"
    )
    assert not any(k.startswith("signals/") for k in s3.reads), (
        f"the loader still reads the live signals artifact: "
        f"{[k for k in s3.reads if k.startswith('signals/')]}"
    )
    for ch in challengers:
        assert ch.name != "scanner_predictor_direct", "champion emitted twice"


def test_a_champion_with_no_picks_anywhere_is_marked_unmeasurable_by_the_loader():
    from scoring.leaderboard_producers import _load_producer_specs

    s3 = _FakeS3({"config/producer_champion.json": {"champion": "scanner_predictor_direct"}})
    champion, _ = _load_producer_specs(s3, "b", ["2026-08-28"], as_of="2026-08-28")
    assert champion is not None
    assert champion.unmeasurable_reason, (
        "a champion that produced nothing on any cohort date must say so — "
        "the slot has no measured incumbent, which is a blocking fact, not a "
        "thin one"
    )


# ── 5. The page fires for a live arm, and NOT for a retired one ─────────────


def _board(arms):
    return {"champion": "c", "unmeasurable_arms": arms}


def test_a_stalled_challenger_reaches_the_DIGEST_and_never_the_page():
    """observability-policy.md §7.2a. A challenger that stopped producing is a
    standing structural fact whose remedy is a tracked item — surveillance
    tier, never the operator's phone. Getting this wrong is how a fix for
    silence ships as noise."""
    from unittest.mock import patch

    from scoring.leaderboard_producers import _alert_unmeasurable_arms

    with patch("ops_alerts.publish_ops_digest") as digest, \
            patch("scoring.leaderboard_producers.publish_observe_alert") as page:
        _alert_unmeasurable_arms(
            "producer", "2026-08-28",
            _board([{
                "name": "scanner_top20_predictor", "kind": "challenger",
                "reason": "last scored 2026-07-30, but scored none of the 3 most recent",
            }]),
        )
    assert digest.call_count == 1
    assert page.call_count == 0, "a stalled challenger must not page"


def test_an_arm_that_never_produced_is_not_reported_twice():
    """`_annotate_arm_measurement_gaps` (alpha-engine-config-I9276) already
    digests the never-wrote-anything case. Reporting it here as well would
    trade one silence for two duplicate notifications."""
    from unittest.mock import patch

    from scoring.leaderboard_producers import _alert_unmeasurable_arms

    with patch("ops_alerts.publish_ops_digest") as digest, \
            patch("scoring.leaderboard_producers.publish_observe_alert") as page:
        _alert_unmeasurable_arms(
            "producer", "2026-08-28",
            _board([{
                "name": "scanner_top20_predictor", "kind": "challenger",
                "reason": "scored 0 of 6 cohort date(s) — the arm produced no comparable output",
            }]),
        )
    assert digest.call_count == 0 and page.call_count == 0


def test_an_unmeasurable_champion_says_the_slot_cannot_promote():
    """The operative consequence, in the alert itself. A challenger cannot be
    promoted over an incumbent nobody measured."""
    from unittest.mock import patch

    from scoring.leaderboard_producers import _alert_unmeasurable_arms

    with patch("scoring.leaderboard_producers.publish_observe_alert") as alert:
        _alert_unmeasurable_arms(
            "producer", "2026-08-28",
            {"champion": "c", "unmeasurable_arms": [{"name": "c", "kind": "champion", "reason": "r"}]},
        )
    assert alert.call_count == 1, (
        "the champion being unmeasurable IS actionable now — it is the one rung "
        "of this class that belongs on the phone"
    )
    assert "cannot promote" in alert.call_args.kwargs["message"]


def test_a_retired_arm_does_not_page():
    """§6 deletes a retired arm's code, so producing nothing is its EXPECTED
    state for the whole trailing window. Paging would fire every cycle for
    eight cycles after every retirement — and an alert that fires when nothing
    is wrong is how the previous detector died."""
    from unittest.mock import patch

    from scoring.leaderboard_producers import _alert_unmeasurable_arms

    with patch("ops_alerts.publish_ops_digest") as digest, \
            patch("scoring.leaderboard_producers.publish_observe_alert") as page:
        _alert_unmeasurable_arms(
            "producer", "2026-08-28",
            _board([{
                "name": "agentic_sector_teams", "kind": "retired",
                "reason": "last scored 2026-07-11, but scored none of the 3 most recent",
            }]),
        )
    assert page.call_count == 0 and digest.call_count == 0


def test_a_retired_arm_still_appears_on_the_artifact():
    """Not paging is not the same as not recording. The row must stay legible."""
    board = score_leaderboard(
        _hist("c", COHORT, kind="champion"),
        [_hist("gone", [], kind="retired")],
        {d: {"AAA": 0.01, "BBB": -0.01} for d in COHORT},
        benchmark_ticker=None,
    )
    assert {a["name"] for a in board["unmeasurable_arms"]} == {"gone"}


# ── 2026-09-05: the filling arms' rows carried no sector, so the promotion guard
# refused every write ─────────────────────────────────────────────────────────
#
# Measured on weekly SF watch-rerun-2026-09-04-{1,2,3} (ChallengerShadow):
#   UNRESOLVED-SECTOR: 10 actionable signal(s) in signals_shadow/
#   scanner_top20_predictor/2026-09-04/signals.json carry no resolved sector:
#   [('ANET', None), ('ANF', None), ...] — REFUSING the write.
# build_shadow_payload (PR767, 2026-08-29 — first live Saturday was 9/5) never
# set `sector` on a row; scoring.promotion_guards.assert_promotable (8/20)
# refuses any actionable row without one. Both arms therefore failed on every
# run, and a degraded ChallengerShadow fails the weekly run at its terminal.
# no_agent / single_agent resolve sector from the same constituents artifact
# (fetch_sp500_sp400_with_sectors) — the filling arms now do the same.

_SECTORS = {f"T{i:03d}": ("Technology" if i % 2 else "Healthcare") for i in range(60)}


def test_shadow_rows_carry_the_constituents_sector():
    from producers.filling_arms import build_shadow_payload
    payload = build_shadow_payload(
        "scanner_top20_predictor", "2026-09-04", _ranked(20),
        pool_size=20, pool_source="predictor_cut:attractiveness_top_20",
        sector_map=_SECTORS,
    )
    for row in payload["buy_candidates"]:
        assert row["sector"] == _SECTORS[row["ticker"]]
    for ticker, sig in payload["signals"].items():
        assert sig["sector"] == _SECTORS[ticker]


def test_shadow_payload_with_sectors_passes_the_promotion_guard():
    from producers.filling_arms import build_shadow_payload
    from scoring.promotion_guards import assert_promotable
    payload = build_shadow_payload(
        "scanner_predictor_direct", "2026-09-04", _ranked(20),
        pool_size=20, pool_source="predictions_research_free", sector_map=_SECTORS,
    )
    assert_promotable(payload, surface="signals_shadow/scanner_predictor_direct/2026-09-04/signals.json")


def test_a_ticker_outside_the_constituents_map_is_still_refused_by_the_guard():
    """Mirrors no_agent: an unknown ticker gets 'Unknown', and the guard — not
    this module — is what refuses it, with its established message."""
    from producers.filling_arms import build_shadow_payload
    from scoring.promotion_guards import UnresolvedSectorError, assert_promotable
    payload = build_shadow_payload(
        "scanner_top20_predictor", "2026-09-04", _ranked(20),
        pool_size=20, pool_source="predictor_cut:attractiveness_top_20",
        sector_map={k: v for k, v in _SECTORS.items() if k != "T000"},
    )
    assert payload["signals"]["T000"]["sector"] == "Unknown"
    with pytest.raises(UnresolvedSectorError, match="T000"):
        assert_promotable(payload, surface="test")


# ── alpha-engine-config-I11396: the cut and its ranking artifact must share a
# population, and a near-disjoint join must RAISE ─────────────────────────────
#
# Measured: `scanner_top20_predictor`'s cut (`attractiveness_top_20`, 20
# scanner-swept names) was ranked against `predictor/predictions/{date}.json`,
# whose universe is the ~25 thesis-covered tickers
# (crucible-predictor/inference/stages/load_universe.py, target_size=25) — a
# DIFFERENT population from the scanner candidate pool the cut is drawn from.
# On 2026-08-03/2026-08-04 the intersection was exactly {CRUS}: pool_size=1
# out of a 20-name cut. `load_research_free_pool` was already reading the
# correct population (`predictor/predictions_research_free/{date}.json`,
# 76-78 names measured on 08-03/04/07); `load_predictor_cut_pool` now reads
# the SAME artifact so the two arms differ only in cut width.


def test_load_predictor_cut_pool_reads_the_research_free_artifact_not_predictions():
    """The repoint. Before this fix the loader read
    `predictor/predictions/{date}.json` (`PREDICTIONS_KEY`); it must now read
    `predictor/predictions_research_free/{date}.json`
    (`PREDICTIONS_RESEARCH_FREE_KEY`) — the same artifact
    `load_research_free_pool` reads for `scanner_predictor_direct`."""
    from producers.filling_arms import load_predictor_cut_pool

    docs = {
        "universe_membership/2026-09-04/membership.json": {
            "predictor_universe_cut": "attractiveness_top_20",
            "cuts": {"attractiveness_top_20": {"tickers": [f"T{i:03d}" for i in range(20)]}},
        },
        # Deliberately NOT under predictor/predictions/ — if the loader still
        # reads the old key this document is invisible to it and the call
        # raises "no s3://.../predictor/predictions_research_free/...".
        "predictor/predictions_research_free/2026-09-04.json": {
            "predictions": [
                {"ticker": f"T{i:03d}", "predicted_alpha": 0.5 - i * 0.01} for i in range(20)
            ],
        },
    }
    s3 = _FakeS3(docs)
    ranked, pool_source = load_predictor_cut_pool(s3, "b", "2026-09-04")
    assert len(ranked) == 20
    assert pool_source == "predictor_cut:attractiveness_top_20"
    assert not any(k.startswith("predictor/predictions/") for k in s3.reads), (
        f"the loader still reads the thesis-coverage-scoped predictions artifact: "
        f"{[k for k in s3.reads if k.startswith('predictor/predictions/')]}"
    )


def test_load_predictor_cut_pool_raises_on_a_near_disjoint_join():
    """The join-fraction guard, exercised directly against the EXACT
    historical shape: a 20-name cut, a ranking artifact carrying only 1 of
    those 20 tickers (the measured {CRUS}-only 2026-08-03/04 case). Before
    this fix nothing raised here — the loader logged a WARNING and happily
    returned a pool of 1, which is the defect this guard exists to close.
    RAISES rather than silently synthesizing a short pool (fleet rule: no
    silent swallow in a producer)."""
    from producers.filling_arms import FillingShadowError, load_predictor_cut_pool

    cut = [f"T{i:03d}" for i in range(20)]
    docs = {
        "universe_membership/2026-09-04/membership.json": {
            "predictor_universe_cut": "attractiveness_top_20",
            "cuts": {"attractiveness_top_20": {"tickers": cut}},
        },
        # Only ONE of the 20 cut tickers appears in the ranking artifact —
        # 1/20 = 5%, the measured pre-fix join fraction.
        "predictor/predictions_research_free/2026-09-04.json": {
            "predictions": [{"ticker": "T000", "predicted_alpha": 0.5}],
        },
    }
    s3 = _FakeS3(docs)
    with pytest.raises(FillingShadowError, match=r"near-disjoint"):
        load_predictor_cut_pool(s3, "b", "2026-09-04")
    # the message names the cut, the artifact, both population sizes and the
    # measured fraction, per the fleet no-silent-swallow rule
    try:
        load_predictor_cut_pool(s3, "b", "2026-09-04")
    except FillingShadowError as e:
        msg = str(e)
        assert "attractiveness_top_20" in msg
        assert "predictor/predictions_research_free/2026-09-04.json" in msg
        assert "20 ticker(s)" in msg  # cut size
        assert "1 ticker(s)" in msg  # artifact population size
        assert "5%" in msg  # measured fraction


def test_load_predictor_cut_pool_passes_on_a_well_joined_cut():
    """The other half: a cut that IS mostly present in the ranking artifact
    (18/20 = 90%, above the 80% floor) must not raise — the guard fires on
    the disjoint-population defect, not on an ordinary handful of misses."""
    from producers.filling_arms import load_predictor_cut_pool

    cut = [f"T{i:03d}" for i in range(20)]
    preds = [{"ticker": f"T{i:03d}", "predicted_alpha": 0.5 - i * 0.01} for i in range(18)]
    docs = {
        "universe_membership/2026-09-04/membership.json": {
            "predictor_universe_cut": "attractiveness_top_20",
            "cuts": {"attractiveness_top_20": {"tickers": cut}},
        },
        "predictor/predictions_research_free/2026-09-04.json": {"predictions": preds},
    }
    s3 = _FakeS3(docs)
    ranked, _ = load_predictor_cut_pool(s3, "b", "2026-09-04")
    assert len(ranked) == 18


def test_build_filling_shadow_resolves_sectors_from_the_constituents_source(monkeypatch):
    import producers.filling_arms as fa
    docs = {
        "universe_membership/2026-09-04/membership.json": {
            "predictor_universe_cut": "attractiveness_top_20",
            "cuts": {"attractiveness_top_20": {"tickers": [f"T{i:03d}" for i in range(20)]}},
        },
        "predictor/predictions_research_free/2026-09-04.json": {
            "predictions": [{"ticker": f"T{i:03d}", "predicted_alpha": 0.5 - i * 0.01} for i in range(20)],
        },
    }
    calls = []
    def _fake_fetch():
        calls.append(1)
        return list(_SECTORS), dict(_SECTORS)
    monkeypatch.setattr(fa, "fetch_sp500_sp400_with_sectors", _fake_fetch)

    class _AM:
        s3 = _FakeS3(docs)
        bucket = "b"

    payload = fa.build_filling_shadow("scanner_top20_predictor", "2026-09-04", _AM())
    assert calls == [1]
    assert all(r["sector"] in ("Technology", "Healthcare") for r in payload["buy_candidates"])
    # ctx-supplied map wins over the fetch (the runner may already hold one)
    payload2 = fa.build_filling_shadow(
        "scanner_top20_predictor", "2026-09-04", _AM(), sector_map=dict.fromkeys(_SECTORS, "Energy"),
    )
    assert calls == [1]
    assert {r["sector"] for r in payload2["buy_candidates"]} == {"Energy"}
