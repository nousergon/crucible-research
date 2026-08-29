"""Every board must report its cohort INTERSECTION, and every cross-arm figure
must be computed over it (alpha-engine-config-I9274, -I9281).

champion-challenger-policy.md §4: *"arms are scored over the intersection of
dates where all arms produced output, and the intersection is reported
alongside the metric."* No board carried that number. Every board reported a
single ``n_dates`` that is the UNION across arms, and compared a champion with
a live cohort running back to July against challengers registered last week.

MEASURED 2026-08-29, the state this file reproduces:

* ``s3://alpha-engine-research/scanner/leaderboard/2026-08-28.json`` — 21d
  block ``status: "ok"``, ``n_dates: 7``; ``momentum_sleeve`` (champion) 7
  dates at ``topn_alpha_vs_population`` mean 0.031348, t 11.41;
  ``tech_score_gate`` 0 dates; ``mom_12_1_sleeve`` 0 dates. **Cohort
  intersection 0**, and nothing on the artifact said so.
* ``research/cuts_leaderboard/2026-08-28.json`` — 21d block ``n_dates: 11``
  with 4 of 7 arms at zero, and an ELEVEN-date ``topn_alpha_vs_champion``
  published for ``scanner_champion_60`` on a block whose all-arm intersection
  is 0.

The immaturity behind both is legitimate — the challengers' shadows genuinely
start in mid-August. The defect is that the artifact renders that immaturity as
a champion RESULT.

GUARD PROVENANCE (champion-challenger-policy.md §7.4 — "a guard must be
verified to fail without the fix"). Every assertion here was run against the
pre-fix tree and observed RED; the run is recorded in the PR body. The module
imports only symbols that exist pre-fix, so the pre-fix collection is clean and
each guard fails on its own assertion rather than on an ImportError.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from scoring.leaderboard_producers import (
    build_cuts_leaderboard,
    build_producer_leaderboard,
    build_scanner_leaderboard,
)
from scoring.leaderboard_scoring import SpecDay, SpecHistory, score_leaderboard

_BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_json(client, key, obj):
    client.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


def _session_calendar(n: int, start: str = "2025-02-03") -> list[str]:
    import pandas as pd

    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=n)]


class _Panel:
    def __init__(self) -> None:
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update({t: float(c) for t, c in closes.items()})
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


def _panel_and_entries() -> tuple[_Panel, list[str]]:
    cal = _session_calendar(400)
    panel = _Panel()
    for i, d in enumerate(cal):
        panel.put(d, {"A": 100 + i, "B": 100 + 0.5 * i, "C": 100 - 0.2 * i, "SPY": 100 + 0.3 * i})
    return panel, [cal[250], cal[260]]


# ── The unit the whole issue is about ────────────────────────────────────────


def _arm(name: str, kind: str, dates: list[str], order: list[str]) -> SpecHistory:
    return SpecHistory(
        name=name, kind=kind,
        by_date={d: SpecDay(ranked=list(order)) for d in dates},
    )


class TestTheIntersectionIsReportedAndBinding:
    def test_a_champion_on_n_dates_and_a_challenger_on_zero_yields_no_comparison(self):
        """THE defect, in the smallest form that exhibits it: the champion has
        a real cohort, the challenger has none, and the block must report the
        intersection as EMPTY and refuse to publish a paired figure for either
        side.

        PRE-FIX: RED — ``KeyError: 'cohort_intersection_dates'``; the board
        carries only the union and no row says the comparison has no common
        ground."""
        dates = ["2026-08-03", "2026-08-04", "2026-08-05"]
        realized = {d: {"A": 0.05, "B": 0.01, "C": -0.02} for d in dates}
        champ = _arm("champ", "champion", dates, ["A", "B", "C"])
        chal = _arm("chal", "challenger", [], ["C", "B", "A"])

        lb = score_leaderboard(champ, [chal], realized, top_n=2, benchmark_ticker="SPY")

        assert lb["cohort_intersection_dates"] == 0
        assert lb["cohort_intersection_first"] is None
        assert lb["cohort_intersection_last"] is None
        # The UNION keeps its name and its meaning — the dashboard and the
        # live gate:data predicates poll `n_dates`.
        assert lb["n_dates"] == 3
        assert lb["cohort_union_dates"] == 3

        rows = {r["name"]: r for r in lb["specs"]}
        assert rows["champ"]["topn_alpha_vs_champion"] is None
        assert rows["chal"]["topn_alpha_vs_champion"] is None
        assert rows["chal"]["comparison_status"] == "no_common_cohort", (
            "`confidence: insufficient` already carries 'this arm scored "
            "little'; an empty intersection is a DIFFERENT condition and must "
            "not be read through the same field"
        )

    def test_the_champions_own_history_is_never_dropped_to_make_the_comparison(self):
        """§3: a promoted arm keeps its series. Narrowing the CROSS-ARM figure
        must not narrow the champion's own unpaired metrics.

        PRE-FIX: RED — ``KeyError: 'cohort_intersection_dates'``; the guard
        asserts the narrowing exists AND is confined to the paired figure, so
        it cannot pass on a tree that narrows nothing."""
        dates = ["2026-08-03", "2026-08-04", "2026-08-05"]
        realized = {d: {"A": 0.05, "B": 0.01, "C": -0.02, "SPY": 0.0} for d in dates}
        champ = _arm("champ", "champion", dates, ["A", "B", "C"])
        chal = _arm("chal", "challenger", dates[-1:], ["C", "B", "A"])

        lb = score_leaderboard(champ, [chal], realized, top_n=2, benchmark_ticker="SPY")
        assert lb["cohort_intersection_dates"] == 1
        champ_row = next(r for r in lb["specs"] if r["name"] == "champ")
        assert champ_row["n_dates_scored"] == 3, (
            "only the CROSS-ARM comparison narrows; the champion's own history "
            "is kept (champion-challenger-policy.md §3)"
        )
        assert champ_row["realized_rank_ic"]["n_dates"] == 3
        assert champ_row["topn_alpha_vs_benchmark"]["n_dates"] == 3

    def test_the_paired_figure_reports_the_intersection_as_its_own_n_dates(self):
        """Closes-when: every non-null ``topn_alpha_vs_champion`` reports
        ``n_dates == cohort_intersection_dates`` for its block.

        The champion here has FIVE dates and the challenger THREE, so the union
        is 5 and the intersection is 3. Pre-fix the paired figure was taken
        over whatever the pair happened to share and the board reported 5.

        PRE-FIX: RED — ``KeyError: 'cohort_intersection_dates'``."""
        dates = _session_calendar(5, start="2026-08-03")
        realized = {d: {"A": 0.05, "B": 0.01, "C": -0.02} for d in dates}
        champ = _arm("champ", "champion", dates, ["A", "B", "C"])
        chal = _arm("chal", "challenger", dates[2:], ["C", "B", "A"])

        lb = score_leaderboard(champ, [chal], realized, top_n=2, benchmark_ticker="SPY")
        assert lb["n_dates"] == 5
        assert lb["cohort_intersection_dates"] == 3
        assert lb["cohort_intersection_first"] == dates[2]
        assert lb["cohort_intersection_last"] == dates[4]
        row = next(r for r in lb["specs"] if r["name"] == "chal")
        assert row["comparison_status"] == "ok"
        assert row["topn_alpha_vs_champion"]["n_dates"] == 3

    def test_a_third_arm_narrows_the_intersection_for_everyone(self):
        """The intersection is over EVERY arm on the block, not over the pair
        being compared. Two arms sharing five dates are not comparable on five
        when a third arm on the same block scored only two of them — the board
        publishes one comparison basis, and it has to be the same basis for
        every row on it.

        PRE-FIX: RED — ``KeyError: 'cohort_intersection_dates'``."""
        dates = _session_calendar(5, start="2026-08-03")
        realized = {d: {"A": 0.05, "B": 0.01, "C": -0.02} for d in dates}
        champ = _arm("champ", "champion", dates, ["A", "B", "C"])
        wide = _arm("wide", "challenger", dates, ["C", "B", "A"])
        narrow = _arm("narrow", "challenger", dates[3:], ["B", "A", "C"])

        lb = score_leaderboard(champ, [wide, narrow], realized, top_n=2, benchmark_ticker="SPY")
        assert lb["cohort_intersection_dates"] == 2
        for name in ("wide", "narrow"):
            row = next(r for r in lb["specs"] if r["name"] == name)
            assert row["topn_alpha_vs_champion"]["n_dates"] == 2, (
                f"{name} was compared on a basis the block does not report"
            )

    def test_the_champion_row_says_self_not_no_common_cohort(self):
        """The champion has nothing to compare against by definition, which is
        not the same finding as an empty intersection. Four states, four
        answers.

        PRE-FIX: RED — ``KeyError: 'comparison_status'``."""
        dates = _session_calendar(5, start="2026-08-03")
        realized = {d: {"A": 0.05, "B": 0.01, "C": -0.02} for d in dates}
        champ = _arm("champ", "champion", dates, ["A", "B", "C"])
        chal = _arm("chal", "challenger", dates, ["C", "B", "A"])
        lb = score_leaderboard(champ, [chal], realized, top_n=2, benchmark_ticker="SPY")
        assert next(r for r in lb["specs"] if r["name"] == "champ")["comparison_status"] == "self"

        no_champ = score_leaderboard(None, [chal], realized, top_n=2, benchmark_ticker="SPY")
        assert no_champ["specs"][0]["comparison_status"] == "no_champion"


# ── Applied to ALL THREE boards, PER HORIZON ─────────────────────────────────


def _seed_all_slots(client, entries: list[str], *, shadow_entries: list[str] | None = None) -> None:
    """The live shape: the champion carries the whole cohort, the shadow
    challenger carries a strict subset (or none at all)."""
    shadow = entries if shadow_entries is None else shadow_entries
    for d in entries:
        _put_json(
            client, f"signals/{d}/signals.json",
            {"signals": {t: {"signal": "ENTER", "score": s}
                         for t, s in [("A", 0.9), ("B", 0.5), ("C", 0.1)]}},
        )
        _put_json(client, f"candidates/{d}/candidates.json", {"scanner_tickers": ["A", "B", "C"]})
        _put_json(
            client, f"universe_membership/{d}/universe.json",
            {"tickers": ["A", "B", "C"]},
        )
    # The shadow-writing arms are resolved from the REGISTER, never typed out.
    # A hardcoded list goes stale the next time an arm is registered, and it
    # goes stale SILENTLY: the new arm simply has no cohort, which this file's
    # own routing tests then read as a live measurement gap. That happened —
    # alpha-engine-config-I9277 registered `scanner_predictor_direct` and
    # `scanner_top20_predictor`, and `test_a_retired_arm_is_never_routed_as_a_gap`
    # went red asserting "every LIVE arm scored" against a fixture that had
    # stopped seeding two of them. Deriving the set is the fix for the CLASS.
    #
    # Every registered challenger gets a shadow, including whichever one is
    # currently the live champion: the champion is scored from `signals/` and
    # its shadow prefix is simply never read, so seeding it is harmless and
    # removes the need for this fixture to resolve the pointer at all.
    from producers.registry import challenger_producers

    _shadow_arms = tuple(s.name for s in challenger_producers())
    for d in shadow:
        for arm in _shadow_arms:
            _put_json(
                client, f"signals_shadow/{arm}/{d}/signals.json",
                {"signals": {t: {"signal": "ENTER", "score": s}
                             for t, s in [("C", 0.9), ("B", 0.5), ("A", 0.1)]}},
            )
        for arm in ("mom_12_1_sleeve", "tech_score_gate"):
            _put_json(
                client, f"candidates_shadow/{arm}/{d}/candidates.json",
                {"scanner_tickers": ["C", "B", "A"]},
            )


@pytest.mark.parametrize(
    ("builder", "leaderboard_id"),
    [
        (build_scanner_leaderboard, "scanner"),
        (build_producer_leaderboard, "producer"),
    ],
)
def test_every_board_carries_the_intersection_on_every_horizon_block(
    s3, builder, leaderboard_id,
):
    """Closes-when, deliverable 3: all three boards, and deliverable 4: the
    field is on the BLOCK, so an arm can be inside the 21d intersection and
    outside the 126d one.

    PRE-FIX: RED — ``cohort_intersection_dates`` is absent from every block on
    every board."""
    panel, entries = _panel_and_entries()
    _seed_all_slots(s3, entries)
    res = builder(s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader())
    assert res["status"] == "ok", res
    blocks = res["leaderboard"]["horizons"]
    assert blocks
    for block in blocks:
        assert "cohort_intersection_dates" in block, (
            f"{leaderboard_id} {block['horizon_days']}d block reports a union "
            "and calls it the cohort (champion-challenger-policy.md §4)"
        )
        assert "cohort_intersection_first" in block
        assert "cohort_intersection_last" in block
        assert block["cohort_union_dates"] == block["n_dates"], (
            "`n_dates` must keep its name and its meaning — crucible-dashboard "
            "and the live gate:data predicates poll it"
        )
        assert block["cohort_intersection_dates"] <= block["n_dates"]
        for row in block["specs"]:
            paired = row.get("topn_alpha_vs_champion")
            if paired is not None and row.get("comparison_status") == "ok":
                assert paired["n_dates"] == block["cohort_intersection_dates"], (
                    "a paired figure published under a denominator the block "
                    "does not report is the whole defect, one level down"
                )


def test_a_board_whose_challengers_never_wrote_publishes_no_comparison(s3):
    """The live 2026-08-28 scanner board, reproduced end to end: the champion
    carries the cohort, both challengers wrote nothing, and the block must say
    the intersection is empty rather than render as a champion result.

    PRE-FIX: RED — ``KeyError: 'cohort_intersection_dates'``."""
    panel, entries = _panel_and_entries()
    _seed_all_slots(s3, entries, shadow_entries=[])
    # The scanner board takes its cohort from `candidates_shadow/` prefixes, so
    # ONE challenger is seeded to give the board a cohort at all; the other
    # contributes nothing, exactly as on 2026-08-28.
    for d in entries:
        _put_json(
            s3, f"candidates_shadow/mom_12_1_sleeve/{d}/candidates.json",
            {"scanner_tickers": ["C", "B", "A"]},
        )
    res = build_scanner_leaderboard(
        s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
    )
    block21 = next(b for b in res["leaderboard"]["horizons"] if b["horizon_days"] == 21)
    assert block21["status"] == "ok"
    assert block21["n_dates"] > 0, "the block reads measured off the champion"
    assert block21["cohort_intersection_dates"] == 0, (
        "one challenger never wrote — there is no date every arm scored"
    )
    for row in block21["specs"]:
        assert row["topn_alpha_vs_champion"] is None
    assert any(
        row.get("comparison_status") == "no_common_cohort" for row in block21["specs"]
    ), "the empty intersection must fail LOUD on the row, not render as an empty success"


# ── I9281: routing the "registered arm wrote nothing" class ──────────────────


class TestNoCohortArmsAreRoutedOnceOnTheSurveillanceTier:
    def test_a_registered_arm_that_wrote_nothing_reaches_the_digest_not_the_page(self, s3):
        """observability-policy.md §7.2: a page is for what is actionable NOW.
        A registered arm that has never written a shadow is a standing
        structural fact whose remedy is a tracked item — it goes to the
        surveillance surface, never to the operator's phone.

        PRE-FIX: RED — nothing routes this class anywhere; the digest is never
        called."""
        panel, entries = _panel_and_entries()
        # Only ONE of the three registered producer challengers writes.
        _seed_all_slots(s3, entries, shadow_entries=[])
        for d in entries:
            _put_json(
                s3, f"signals_shadow/no_agent_quant/{d}/signals.json",
                {"signals": {t: {"signal": "ENTER", "score": s}
                             for t, s in [("C", 0.9), ("B", 0.5), ("A", 0.1)]}},
            )
        with patch("ops_alerts.publish_ops_digest") as digest, \
                patch("scoring.leaderboard_producers.publish_observe_alert") as page:
            build_producer_leaderboard(
                s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
            )
        assert digest.call_count == 1, (
            "observability-policy.md §7.2a: one grouped notification for the "
            "group, not one per member and not one per horizon"
        )
        findings = digest.call_args.args[0]
        assert {"single_agent_quant", "thinktank_coverage"} <= {
            f.split("'")[1] for f in findings
        }, "a group notification that omits its member list has traded spam for uselessness"
        assert page.call_count == 0, (
            "this class must not reach the paging channel — a standing daily "
            "page for a condition with an open owning item is §7.4a's defect"
        )

    def test_a_retired_arm_is_never_routed_as_a_gap(self, s3):
        """``agentic_sector_teams`` is ``kind="retired"`` (2026-07-12, code
        deleted under §6) and has no ``signals_shadow/`` prefix at all — it is
        permanently zero-cohort BY CONSTRUCTION. Routing it would make this
        class a guaranteed daily false positive on its first run, which §7.2
        rates worse than no watchdog.

        PRE-FIX: RED — the pre-fix tree routes nothing, so this passes
        vacuously; it is asserted here against the ARMED behaviour and was
        confirmed to fail when the arming was written without the kind gate."""
        panel, entries = _panel_and_entries()
        _seed_all_slots(s3, entries)
        with patch("ops_alerts.publish_ops_digest") as digest:
            res = build_producer_leaderboard(
                s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
            )
        # Two assertions, deliberately. The COUNT proves the gate is not
        # passing vacuously; the MEMBERSHIP says which arm broke it when it
        # does break. A bare count fails as `1 != 0` and tells a future reader
        # nothing — which is how this test read when the register grew two
        # challengers under it (alpha-engine-config-I9277) and the fixture
        # stopped seeding them. `_seed_all_slots` now derives its arms from the
        # register so that cannot recur; this message is the backstop if it
        # somehow does.
        routed = [
            n for call in digest.call_args_list
            for n in (call.kwargs.get("members") or [])
        ]
        assert "agentic_sector_teams" not in routed, (
            "a RETIRED arm was routed as a measurement gap — it is permanently "
            "zero-cohort by construction, so this class would page every day "
            "from its first run (observability-policy §7.2 rates that worse "
            "than no watchdog)"
        )
        assert digest.call_count == 0, (
            "every LIVE arm scored and the only zero-cohort arm is retired, so "
            f"nothing should route; routed instead: {routed or digest.call_args_list}"
        )
        # Suppression is a DELIVERY decision, never a recording one (§7.2a):
        # the retired arm still appears on the artifact.
        block21 = next(
            b for b in res["leaderboard"]["horizons"] if b["horizon_days"] == 21
        )
        assert "agentic_sector_teams" in block21["arms_no_cohort"]


# ── The cuts board: same defect, its own loader ──────────────────────────────
#
# `build_cuts_leaderboard` reads `universe_membership/{date}/membership.json`
# rather than a shadow prefix and takes no `top_n` (its arms carry per-arm
# widths), so it needs its own double rather than the shared seeder above.

_CUT_DATES = ["2026-07-01", "2026-07-02", "2026-07-03"]
_CUT_FEED = [f"T{i:03d}" for i in range(60)]


class _CutsS3:
    """Membership double where the champion cut spans EVERY date and one
    co-arm appears on only the last — the live 2026-08-28 shape, where 4 of 7
    arms had no cohort at all and the board still published an 11-date paired
    figure."""

    def __init__(self, sparse_from: int) -> None:
        from scoring.universe_membership import (
            CHAMPION_CUT,
            FEED_CUT_NAME,
            PREDICTOR_UNIVERSE_CUT,
        )

        self._feed, self._champ_cut, self._gate = (
            FEED_CUT_NAME, PREDICTOR_UNIVERSE_CUT, CHAMPION_CUT,
        )
        self._sparse_from = sparse_from
        self.written: dict[str, dict] = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        date_str = Key.split("/")[1]
        cuts = {
            self._feed: {"tickers": _CUT_FEED},
            self._champ_cut: {"tickers": _CUT_FEED[:20]},
        }
        if _CUT_DATES.index(date_str) >= self._sparse_from:
            cuts[self._gate] = {"tickers": [f"G{i:03d}" for i in range(60)]}
        return {"Body": _CutsBody(json.dumps({"cuts": cuts}).encode())}

    def get_paginator(self, _op):
        return _CutsPaginator()

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803
        self.written[Key] = json.loads(Body)


class _CutsBody:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class _CutsPaginator:
    def paginate(self, Bucket, Prefix):  # noqa: N803
        yield {"Contents": [{"Key": f"{Prefix}{d}/membership.json"} for d in _CUT_DATES]
               + [{"Key": f"{Prefix}latest.json"}]}


def _cuts_panel() -> _Panel:
    cal = _session_calendar(400, start="2026-07-01")
    panel = _Panel()
    for i, d in enumerate(cal):
        closes = {t: 100 + i * (0.1 + 0.001 * n) for n, t in enumerate(_CUT_FEED)}
        closes.update({f"G{n:03d}": 100 - 0.05 * i for n in range(60)})
        closes["SPY"] = 100 + 0.3 * i
        panel.put(d, closes)
    return panel


def test_the_cuts_board_carries_the_intersection_and_narrows_the_paired_figure():
    """The cuts board scores each arm in its OWN single-arm pass, so the
    intersection cannot be taken inside the pass — it is applied once after the
    merge, where every arm is finally in scope.

    PRE-FIX: RED — ``KeyError: 'cohort_intersection_dates'``, and the paired
    figure was taken over whatever the PAIR shared rather than over the
    block."""
    s3 = _CutsS3(sparse_from=len(_CUT_DATES) - 1)
    res = build_cuts_leaderboard(
        s3, _BUCKET, "2026-08-17", closes_panel_loader=_cuts_panel().loader(),
    )
    assert res["status"] == "ok", res
    for block in res["leaderboard"]["horizons"]:
        assert "cohort_intersection_dates" in block
        assert block["cohort_union_dates"] == block["n_dates"]
        assert block["cohort_intersection_dates"] <= block["n_dates"]
        for row in block["specs"]:
            assert "comparison_status" in row
            paired = row.get("topn_alpha_vs_champion")
            if paired is not None and row["comparison_status"] == "ok":
                assert paired["n_dates"] == block["cohort_intersection_dates"]
    block21 = next(b for b in res["leaderboard"]["horizons"] if b["horizon_days"] == 21)
    assert block21["cohort_intersection_dates"] < block21["n_dates"], (
        "one arm appears on only the last date; the union must exceed the "
        "intersection or this fixture does not exhibit the defect"
    )
