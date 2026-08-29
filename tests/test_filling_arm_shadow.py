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

from producers import filling_arms
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

FILLING_ARMS = ("scanner_predictor_direct", "scanner_top20_predictor")


# ── 1. The arms now build a comparable artifact, unconditionally ─────────────


def test_every_filling_arm_builds_its_own_shadow():
    """RED before the fix: both arms carried ``build=None``, so
    ``producers.runner`` skipped them and no ``signals_shadow/`` prefix ever
    existed for either. The champion was therefore scored from an artifact its
    picks are not in."""
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
    registry-validation time, not produce a thin row for seven weeks."""
    import dataclasses

    from producers import registry as reg

    original = dict(reg.RESEARCH_PRODUCERS)
    try:
        reg.RESEARCH_PRODUCERS["no_agent_quant"] = dataclasses.replace(
            original["no_agent_quant"], score_source=SCORE_SOURCE_SIGNALS_LIVE,
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
        pool_size=60, pool_source="research_free_parquet",
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
        pool_size=3, pool_source="research_free_parquet",
    )
    scores = [payload["signals"][t]["score"] for t in ("BBB", "CCC", "AAA")]
    assert scores == sorted(scores, reverse=True), "score must be monotone in rank"


def test_an_empty_pool_raises_rather_than_writing_a_healthy_looking_shadow():
    """No silent degrade on a producer. An empty-but-well-formed shadow is
    exactly the artifact whose existence caused this issue."""
    with pytest.raises(FillingShadowError, match="ZERO ranked names"):
        build_shadow_payload(
            "scanner_predictor_direct", "2026-08-28", [],
            pool_size=0, pool_source="research_free_parquet",
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


def test_an_unmeasurable_live_arm_pages():
    from unittest.mock import patch

    from scoring.leaderboard_producers import _alert_unmeasurable_arms

    with patch("scoring.leaderboard_producers.publish_observe_alert") as alert:
        _alert_unmeasurable_arms(
            "producer", "2026-08-28",
            _board([{"name": "scanner_top20_predictor", "kind": "challenger", "reason": "r"}]),
        )
    assert alert.call_count == 1
    assert "UNMEASURABLE" in alert.call_args.kwargs["message"]


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
    assert "cannot promote" in alert.call_args.kwargs["message"]


def test_a_retired_arm_does_not_page():
    """§6 deletes a retired arm's code, so producing nothing is its EXPECTED
    state for the whole trailing window. Paging would fire every cycle for
    eight cycles after every retirement — and an alert that fires when nothing
    is wrong is how the previous detector died."""
    from unittest.mock import patch

    from scoring.leaderboard_producers import _alert_unmeasurable_arms

    with patch("scoring.leaderboard_producers.publish_observe_alert") as alert:
        _alert_unmeasurable_arms(
            "producer", "2026-08-28",
            _board([{"name": "agentic_sector_teams", "kind": "retired", "reason": "r"}]),
        )
    assert alert.call_count == 0


def test_a_retired_arm_still_appears_on_the_artifact():
    """Not paging is not the same as not recording. The row must stay legible."""
    board = score_leaderboard(
        _hist("c", COHORT, kind="champion"),
        [_hist("gone", [], kind="retired")],
        {d: {"AAA": 0.01, "BBB": -0.01} for d in COHORT},
        benchmark_ticker=None,
    )
    assert {a["name"] for a in board["unmeasurable_arms"]} == {"gone"}
