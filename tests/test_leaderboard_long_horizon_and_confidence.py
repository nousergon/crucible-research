"""Long-horizon scoring (alpha-engine-config-I7540) + per-row confidence
(alpha-engine-config-I7542) — the two defects are one artifact contract.

**I7540 — the leaderboards could not answer the question they exist to answer.**
Every arm was scored at a single 21-session horizon. The scanner's stated
objective is to surface names attractive over ~1 year, comparable to sell-side
coverage, so every arm was graded on a horizon ~12x shorter than the thing it
is trying to do: an arm built for a long view loses whether or not its view is
correct, and a momentum-tilted arm wins whether or not the tilt serves the
objective. Live evidence — ``research/producer_leaderboard/2026-08-14.json``
marked ``thinktank_coverage``, which writes a ONE-YEAR analyst thesis, on 21-day
returns off ONE date.

**I7542 — a one-date mean rendered as an ordinary result.** The same artifact
showed ``realized_rank_ic: {mean: -0.107, se: null, t_stat: null, n_dates: 1}``
in the identical shape as the champion's real multi-date row. Nothing marked it
low-confidence. Read by a human or an agent that says "the Think Tank arm is
losing badly"; it says nothing of the kind.

They land together because I7540 makes I7542 the DOMINANT rendering: a
252-session horizon needs ~252 trading days of shadow history before any date
scores, so most rows at the long horizons are insufficient or thin by
construction for months. Honest immaturity must render as immaturity — never as
a numeric result and never as a zero (champion-challenger-policy.md §7.2).

GUARD PROVENANCE (§7.4 — "a guard must be verified to fail without the fix").
Every assertion in this file was run against ``origin/main`` before the fix and
observed RED; the run is recorded in the PR body. This file deliberately imports
only symbols that EXIST on the pre-fix commit at module scope, so the pre-fix
run collects cleanly and each guard fails on its own assertion rather than on an
ImportError — an import failure would prove the module changed, not that the
guard bites. The one test that needs a new symbol imports it inside its own
body, and its pre-fix failure is named as such.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from scoring.leaderboard_producers import (
    build_producer_leaderboard,
    build_scanner_leaderboard,
)
from scoring.leaderboard_scoring import (
    SpecDay,
    SpecHistory,
    score_leaderboard,
)

_BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_json(client, key, obj):
    client.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _session_calendar(n: int, start: str = "2024-01-01") -> list[str]:
    """``n`` synthetic TRADING sessions. Business days only, which is what the
    real panel holds (non-sessions are absent by construction), so the
    horizon arithmetic under test is the arithmetic production runs."""
    import pandas as pd

    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=n)]


class _Panel:
    """Builds ``{date: {ticker: close}}`` and hands it back as a loader,
    mirroring ``tests/test_leaderboard_scoring.py::_Panel``. The loader accepts
    ``horizon_days`` and ignores it: the point of the multi-horizon read is
    that ONE panel serves every horizon."""

    def __init__(self) -> None:
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update({t: float(c) for t, c in closes.items()})
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


# A source CAPABLE of every horizon (ArcticDB holds years and carries no
# lifecycle expiry — verified live 2026-08-17), on which the 21- and
# 126-session horizons have matured for the cohort and the 252-session one has
# not. That is the state the fleet is actually in, and the state I7540's
# constraint section names: honest immaturity, not a defect.
# The panel spans 400 sessions and ENDS at the as-of date, so it is capable of
# every horizon (§7.1 satisfied at 252). The cohort sits ~150 sessions from the
# end: matured at 21 and 126, genuinely immature at 252. That is a different
# state from "the source cannot serve 252 sessions at all", and the two must
# not be conflated in either direction — which is the whole point of
# ``_assert_horizon_is_satisfiable``.
_N_SESSIONS = 400
_CAL_START = "2025-02-03"


def _matured_to_126_panel() -> tuple[_Panel, list[str]]:
    cal = _session_calendar(_N_SESSIONS, start=_CAL_START)
    panel = _Panel()
    for i, d in enumerate(cal):
        # A strictly outperforms B strictly outperforms C; SPY sits between.
        panel.put(d, {"A": 100 + i, "B": 100 + 0.5 * i, "C": 100 - 0.2 * i, "SPY": 100 + 0.3 * i})
    entries = [cal[250], cal[260]]
    return panel, entries


# EVERY registered challenger arm, not just one (alpha-engine-config-I9281).
# The fixture used to seed `signals_shadow/thinktank_coverage/` alone while
# `RESEARCH_PRODUCERS` also registers `no_agent_quant` and
# `single_agent_quant` — so the "an immature long horizon does not alert" lock
# below was standing over a fixture that genuinely exhibited a DIFFERENT
# condition (two registered arms with no cohort artifact at all). The lock is
# about an immature HORIZON; it must be seeded with a cohort that is immature
# rather than one that is missing, or it locks the wrong thing in place.
# DERIVED, never typed out. The literal below used to read
# ("thinktank_coverage", "no_agent_quant", "single_agent_quant") under a comment
# claiming it was "EVERY registered challenger arm" — true when written, false
# the moment alpha-engine-config-I9277 registered `scanner_predictor_direct` and
# `scanner_top20_predictor`. The two new arms then had no cohort, which is the
# very condition this lock asserts is absent, so the lock went red for a reason
# that had nothing to do with the property it names.
#
# That is the SECOND fixture in this repo to go stale the same way this session
# (the first: `tests/test_cuts_leaderboard.py::_SlotS3`, which never seeded
# `attractiveness_hard3_top_60`). A register is a moving target and a test that
# restates it is a test that silently stops covering what it claims. Resolve it.
def _seeded_producer_arms() -> tuple[str, ...]:
    from producers.registry import challenger_producers

    return tuple(s.name for s in challenger_producers())

# The arm the pointer names in this fixture. Seeded explicitly rather than
# left to the registry fallback, so the test states which arm it is scoring —
# and so the champion's picks land at the same prefix every challenger's do
# (alpha-engine-config-I9307).
_STANDIN_CHAMPION = "scanner_predictor_direct"


def _seed_producer_cohort(client, entries: list[str]) -> None:
    _put_json(
        client,
        "config/producer_champion.json",
        {"schema_version": 1, "champion": _STANDIN_CHAMPION},
    )
    for d in entries:
        # alpha-engine-config-I9307: the champion is read from its own shadow
        # prefix like every other arm, never from the live signals artifact.
        _put_json(
            client,
            f"signals_shadow/{_STANDIN_CHAMPION}/{d}/signals.json",
            {"signals": {t: {"signal": "ENTER", "score": s} for t, s in [("A", 0.9), ("B", 0.5), ("C", 0.1)]}},
        )
        # Their derived arm list (I9281) — never a literal that goes stale when
        # the register moves. Each arm gets a membership that DIFFERS from the
        # champion's rather than being a re-ordering of it: a challenger that
        # resolves the champion's exact set is a vacuous comparison
        # (champion-challenger-policy.md §4) and the board's own vacuity guard
        # correctly alerts on it. Before alpha-engine-config-I9307 the champion
        # resolved to None here, so that guard was never exercised.
        _members = {"A", "B", "C"}
        for _i, arm in enumerate(a for a in _seeded_producer_arms() if a != _STANDIN_CHAMPION):
            picks = sorted(_members - {sorted(_members)[_i % len(_members)]})
            _put_json(
                client,
                f"signals_shadow/{arm}/{d}/signals.json",
                {"signals": {t: {"signal": "ENTER", "score": 0.9 - 0.4 * j}
                             for j, t in enumerate(picks)}},
            )
            )


# ── I7540: every horizon is scored, each with its own n_dates ─────────────────


class TestEveryHorizonIsScored:
    def test_both_artifacts_carry_a_block_per_registered_horizon(self, s3):
        """Closes-when #1: ``scanner/leaderboard/{date}.json`` and
        ``research/producer_leaderboard/{date}.json`` each carry scored blocks
        at 21, 126 and 252 sessions, with per-horizon ``n_dates``.

        PRE-FIX: RED — ``horizons`` does not exist on the artifact at all; the
        whole artifact is one 21-day block."""
        panel, entries = _matured_to_126_panel()
        _seed_producer_cohort(s3, entries)
        for d in entries:
            _put_json(s3, f"candidates/{d}/candidates.json", {"scanner_tickers": ["A", "B", "C"]})
            _put_json(
                s3,
                f"candidates_shadow/momentum_sleeve/{d}/candidates.json",
                {"scanner_tickers": ["C", "B", "A"]},
            )

        for builder, key in (
            (build_producer_leaderboard, "research/producer_leaderboard/2026-08-17.json"),
            (build_scanner_leaderboard, "scanner/leaderboard/2026-08-17.json"),
        ):
            res = builder(s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader())
            assert res["status"] == "ok", res
            got = json.loads(s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read())
            assert [b["horizon_days"] for b in got["horizons"]] == [21, 126, 252], (
                "an arm graded only at 21 sessions cannot answer the ~1-year "
                "question the scanner exists to answer (I7540)"
            )
            for block in got["horizons"]:
                assert "n_dates" in block, "each horizon must carry its OWN cohort count"
                assert block["specs"], "every arm is scored at every horizon (§3)"

    def test_each_horizon_carries_its_own_cohort_count(self, s3):
        """A shared ``n_dates`` across horizons would be a lie: a cohort that
        has matured 21 sessions has not necessarily matured 126.

        PRE-FIX: RED — there is exactly one ``n_dates`` on the artifact."""
        panel, entries = _matured_to_126_panel()
        _seed_producer_cohort(s3, entries)
        res = build_producer_leaderboard(
            s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
        )
        blocks = {b["horizon_days"]: b for b in res["leaderboard"]["horizons"]}
        assert blocks[21]["n_dates"] == 2
        assert blocks[126]["n_dates"] == 2
        assert blocks[252]["n_dates"] == 0, (
            "a 252-session horizon cannot have matured against a 200-session panel"
        )

    def test_one_closes_panel_read_serves_every_horizon(self, s3):
        """Three ArcticDB reads for three horizons would triple the dominant
        cost of the Lambda this runs in, for data already in memory.

        PRE-FIX: RED — the pre-fix builder loads the panel sized to the 21-day
        horizon, so ``calls == [21]``."""
        panel, entries = _matured_to_126_panel()
        _seed_producer_cohort(s3, entries)
        calls: list[int] = []

        def counting_loader(bucket, entry_dates, horizon_days, symbols=None):
            calls.append(horizon_days)
            return panel.panel

        build_producer_leaderboard(
            s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=counting_loader
        )
        assert calls == [252], (
            "the panel must be loaded exactly once, sized to the LONGEST horizon"
        )


# ── I7540: immaturity renders as immaturity, never as a zero ─────────────────


class TestLongHorizonImmaturityIsHonest:
    def test_an_unmatured_horizon_is_immature_with_a_reason_not_a_zero(self, s3):
        """Closes-when #2: a 252d block with insufficient cohort history renders
        as immature/unmeasurable with a reason string, never as a numeric
        result.

        PRE-FIX: RED — no 252d block exists, so there is nothing to be honest
        about."""
        panel, entries = _matured_to_126_panel()
        _seed_producer_cohort(s3, entries)
        res = build_producer_leaderboard(
            s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
        )
        block = next(b for b in res["leaderboard"]["horizons"] if b["horizon_days"] == 252)

        assert block["status"] == "immature"
        assert block["reason"] and "252" in block["reason"]
        for row in block["specs"]:
            assert row["confidence"] == "insufficient"
            assert row["n_dates_scored"] == 0
            assert row["realized_rank_ic"] is None, (
                "an unmatured horizon must not fabricate a mean"
            )
            assert row["topn_alpha_vs_benchmark"] is None

    def test_an_immature_long_horizon_does_not_alert(self, s3):
        """A 252d block sits immature for the better part of a year. Alerting on
        it every cycle manufactures exactly the fatigue that hides real
        findings — the same reasoning that keeps the primary horizon's
        immature case silent.

        The fixture seeds EVERY registered challenger arm (alpha-engine-config
        -I9281): a fixture where two of the three registered arms wrote no
        shadow at all exhibits the registered-arm-never-wrote condition, not an
        immature horizon, so "no alert" over it locked a claim this test does
        not make. Both channels are asserted silent — the page AND the
        surveillance digest — because the honest-immaturity rule is about the
        condition, not about which channel it would have reached.

        PRE-FIX: GREEN — no long horizon exists to alert about. A lock against
        the wrong fix (alerting on honest immaturity), stated as such rather
        than counted among the red guards."""
        from unittest.mock import patch

        panel, entries = _matured_to_126_panel()
        _seed_producer_cohort(s3, entries)
        with patch("scoring.leaderboard_producers.publish_observe_alert") as alert, \
                patch("ops_alerts.publish_ops_digest") as digest:
            build_producer_leaderboard(
                s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
            )
        assert alert.call_count == 0, [c.kwargs.get('message') for c in alert.call_args_list]
        assert digest.call_count == 0, (
            "every registered arm wrote a shadow here; the only zero-cohort "
            "arm is the RETIRED one, which is permanently zero by construction"
        )

    def test_a_long_horizon_the_source_cannot_serve_never_sinks_the_21d_series(self, s3):
        """§3 continuity: failing the whole leaderboard because the panel cannot
        span 252 sessions would take the WORKING 21-day series down with it —
        and the 21-day series is a promoted arm's history.

        PRE-FIX: RED — ``KeyError: 'horizons'``; there is no per-horizon
        failure isolation to assert against."""
        cal = _session_calendar(30)
        panel = _Panel()
        for i, d in enumerate(cal):
            panel.put(d, {"A": 100 + i, "B": 100 + 0.5 * i, "C": 100 - 0.2 * i, "SPY": 100 + 0.3 * i})
        entries = [cal[0]]
        _seed_producer_cohort(s3, entries)

        res = build_producer_leaderboard(
            s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
        )
        assert res["status"] == "ok", "a 21-session horizon this panel CAN serve must still score"
        blocks = {b["horizon_days"]: b for b in res["leaderboard"]["horizons"]}
        assert blocks[21]["n_dates"] == 1
        for h in (126, 252):
            assert blocks[h]["status"] == "unmeasurable"
            assert blocks[h]["reason"], "an unmeasurable horizon must say why (§7.2)"
            assert blocks[h]["n_dates"] == 0


# ── I7540: §3 continuity — the 21-day series does not move ───────────────────


# Captured from ``origin/main`` (commit 33428e4c) by running the PRE-CHANGE
# ``score_leaderboard`` against ``_continuity_fixture()`` below. These are the
# exact values the pre-change scorer produced. champion-challenger-policy.md §3:
# a promoted arm keeps its history — promotion does not reset a series, and
# neither may a scorer change. If this fixture's numbers move, the 21-day
# leaderboard series has been silently re-based and every comparison across the
# change date is invalid.
_PRE_CHANGE_21D = {
    "benchmark_ticker": "SPY",
    "champion": "champ",
    "horizon_days": 21,
    "n_dates": 6,
    "top_n": 2,
    "specs": [
        {
            "kind": "champion",
            "n_dates_scored": 6,
            "name": "champ",
            "realized_rank_ic": {"mean": 0.533333, "n_dates": 6, "se": 0.098883, "t_stat": 5.3936},
            "topn_alpha_vs_benchmark": {"mean": 0.01725, "n_dates": 6, "se": 0.001909, "t_stat": 9.0342},
            "topn_alpha_vs_champion": None,
        },
        {
            "kind": "challenger",
            "n_dates_scored": 6,
            "name": "chal",
            "realized_rank_ic": {"mean": -0.533333, "n_dates": 6, "se": 0.176383, "t_stat": -3.0237},
            "topn_alpha_vs_benchmark": {"mean": -0.00275, "n_dates": 6, "se": 0.001909, "t_stat": -1.4402},
            "topn_alpha_vs_champion": {"mean": -0.02, "n_dates": 6, "se": 0.0, "t_stat": None},
        },
    ],
}


def _continuity_fixture():
    dates = ["2026-05-04", "2026-05-11", "2026-05-18", "2026-05-25", "2026-06-01", "2026-06-08"]
    champ = SpecHistory(
        name="champ", kind="champion", by_date={d: SpecDay(ranked=["A", "B", "C", "D"]) for d in dates}
    )
    chal = SpecHistory(
        name="chal", kind="challenger", by_date={d: SpecDay(ranked=["C", "D", "A", "B"]) for d in dates}
    )
    realized = {
        d: {
            "A": round(0.01 * (i + 1), 6),
            "B": round(0.02 - 0.005 * i, 6),
            "C": round(-0.01 + 0.002 * i, 6),
            "D": round(0.003 * i, 6),
            "SPY": 0.004,
        }
        for i, d in enumerate(dates)
    }
    return champ, [chal], realized


# Fields added AFTER _PRE_CHANGE_21D was captured. §3 protects the pre-change
# numbers from moving; it does not forbid a row carrying new information
# alongside them. Each entry must be genuinely additive — a field that REPLACES
# or rescales an existing one does not belong here, it belongs in a new pin.
#
#   confidence                 alpha-engine-config-I7542 — evidence behind a row
#   topn_alpha_vs_population   alpha-engine-config-I7576 — lift vs the scored
#                              population, alongside (never instead of) the
#                              unchanged SPY series
#   dates_scored / n_dates_in_intersection / topn_alpha_vs_benchmark_intersection
#   / promotion_eligible / ineligible_reason
#                              alpha-engine-config-I9277, -I9279 — WHICH dates
#                              a row's number was computed over, the same metric
#                              restricted to the cohort every eligible arm
#                              shares, and whether the arm may be promoted.
#   comparison_status          alpha-engine-config-I9274 — WHY this row holds
#                              the paired figure it holds.
#
# Every protected mean/se/t_stat above is arithmetically untouched by BOTH
# changes. On this pinned fixture all arms score all six dates, so the cohort
# intersection IS the union and `topn_alpha_vs_champion` is unchanged (the
# pinned {-0.02, se 0.0, n 6} still stands) — which is the correct value, not a
# coincidence being papered over. What the new fields add is a STATED basis
# where the basis was previously unstated.
_ADDITIVE_SINCE_CAPTURE = (
    "confidence",
    "topn_alpha_vs_population",
    "dates_scored",
    "topn_alpha_vs_benchmark_intersection",
    "n_dates_in_intersection",
    "promotion_eligible",
    "ineligible_reason",
    "comparison_status",
    # alpha-engine-config-I9307 — whether the arm CAN produce comparable output
    # at all. A different question from `comparison_status`, which asks whether
    # this row can be compared against the CHAMPION; see the note in
    # scoring/leaderboard_scoring.py. Purely additive: it says something new
    # about the row without touching any protected number.
    "measurability",
    "unmeasurable_reason",
)

# Additive keys at the TOP level of the leaderboard dict, same rule. This is the
# UNION of what both merged changes emit, and it is exhaustive on purpose: the
# lock's whole value is that an unlisted new key FAILS. `n_dates` — the field
# crucible-dashboard's s3_loader and the live gate:data predicates poll — is
# absent from this list because it is untouched, and that is what this lock
# proves (alpha-engine-config-I9274).
#
# Two intersection vocabularies coexist here deliberately; see the
# "TWO windows" note in scoring/leaderboard_scoring.py. `cohort_intersection` /
# `n_dates_intersection` carry the RELAXED promotion window; the
# `cohort_intersection_*` trio carries the STRICT reported one.
_ADDITIVE_TOP_LEVEL_SINCE_CAPTURE = (
    "cohort_intersection",
    "n_dates_intersection",
    "cohort_union_dates",
    "cohort_intersection_dates",
    "cohort_intersection_first",
    "cohort_intersection_last",
    "unmeasurable_arms",  # alpha-engine-config-I9307
)

# Additive fields added INSIDE a metric block, rather than alongside it. Same
# rule as above and the same test: each must carry new information about the
# protected number without changing it.
#
#   se_method / overlap_lags   alpha-engine-config-I8263 — WHICH standard error
#                              produced this block, and whether its observations
#                              overlapped. The pinned series is scored with no
#                              declared cohort spacing, so it takes the iid
#                              branch and every protected mean/se/t_stat is
#                              arithmetically untouched — which is exactly what
#                              this lock is here to prove, and why the keys are
#                              stripped rather than the pin re-captured.
_ADDITIVE_IN_METRIC_SINCE_CAPTURE = ("se_method", "overlap_lags")


def _numeric_only(row: dict) -> dict:
    """A spec row reduced to the fields that existed when the pin was captured.

    Deliberately a subtraction from the LIVE row rather than a re-capture of the
    literals: re-pinning would let a future change to the protected numbers slide
    through under cover of an unrelated addition."""
    out = {}
    for k, v in row.items():
        if k in _ADDITIVE_SINCE_CAPTURE:
            continue
        if isinstance(v, dict):
            v = {
                mk: mv for mk, mv in v.items()
                if mk not in _ADDITIVE_IN_METRIC_SINCE_CAPTURE
            }
        out[k] = v
    return out


class TestTwentyOneDayContinuity:
    def test_the_21d_numbers_are_unchanged_by_this_work(self):
        """Closes-when #3: the 21d numbers are byte-identical to the pre-change
        series for a replayed date.

        PRE-FIX: GREEN by construction — this is a regression lock on the
        pre-change values, not a defect guard, and it is the assertion that
        makes the rest of I7540 safe to land. Stated plainly rather than
        counted among the red guards."""
        champ, chals, realized = _continuity_fixture()
        lb = score_leaderboard(champ, chals, realized, top_n=2, horizon_days=21, benchmark_ticker="SPY")
        got = {
            k: v for k, v in lb.items()
            if k != "specs" and k not in _ADDITIVE_TOP_LEVEL_SINCE_CAPTURE
        }
        expected = {k: v for k, v in _PRE_CHANGE_21D.items() if k != "specs"}
        assert got == expected
        assert [_numeric_only(r) for r in lb["specs"]] == _PRE_CHANGE_21D["specs"]

    def test_the_primary_horizon_block_matches_the_top_level_block(self):
        """The top level IS the 21-day block. A consumer reading either surface
        must see the same series, or the compatibility surface is a fiction.

        PRE-FIX: RED — ``score_multi_horizon`` does not exist."""
        from scoring.leaderboard_scoring import score_multi_horizon

        champ, chals, realized = _continuity_fixture()
        out = score_multi_horizon(
            champ,
            chals,
            {21: realized, 126: {}, 252: {}},
            top_n=2,
            horizons_days=[21, 126, 252],
            benchmark_ticker="SPY",
        )
        assert out["horizon_days"] == 21
        assert out["n_dates"] == _PRE_CHANGE_21D["n_dates"]
        assert [_numeric_only(r) for r in out["specs"]] == _PRE_CHANGE_21D["specs"]
        primary = out["horizons"][0]
        assert primary["horizon_days"] == 21
        assert primary["specs"] == out["specs"]

    def test_the_primary_horizon_is_always_present_and_first(self):
        """A caller cannot drop the primary horizon: §3 continuity is not an
        option a call site gets to decline.

        PRE-FIX: RED — ``_resolve_horizons`` does not exist."""
        from scoring.leaderboard_producers import _resolve_horizons

        assert _resolve_horizons(None, 21, (21, 126, 252)) == [21, 126, 252]
        assert _resolve_horizons([126, 252], 21, (21, 126, 252)) == [21, 126, 252]
        assert _resolve_horizons([252], 21, (21, 126, 252)) == [21, 252]


# ── I7542: per-row confidence ────────────────────────────────────────────────


def _one_date_fixture(n_dates: int):
    """``n_dates`` scored dates for one challenger — the ``thinktank_coverage``
    shape from the live 2026-08-14 artifact when ``n_dates == 1``."""
    dates = [f"2026-06-{d:02d}" for d in range(1, n_dates + 1)]
    chal = SpecHistory(
        name="thinktank_coverage",
        kind="challenger",
        by_date={d: SpecDay(ranked=["A", "B", "C"], scores={"A": 0.9, "B": 0.5, "C": 0.1}) for d in dates},
    )
    realized = {d: {"A": -0.02, "B": 0.01, "C": 0.03, "SPY": 0.005} for d in dates}
    return chal, realized


class TestPerRowConfidence:
    def test_a_single_scored_date_is_thin(self):
        """Closes-when: a fixture with a single scored date produces ``thin``.

        This is THE I7542 defect, reproduced from the live artifact: a mean
        over one observation, ``se: null``, ``t_stat: null``, rendered in the
        same shape as the champion's real result.

        PRE-FIX: RED — ``KeyError: 'confidence'``; the row carries no status at
        all."""
        chal, realized = _one_date_fixture(1)
        lb = score_leaderboard(None, [chal], realized, top_n=2, benchmark_ticker="SPY")
        row = lb["specs"][0]
        assert row["n_dates_scored"] == 1
        assert row["confidence"] == "thin"
        # The nulls stay null — the fix is the STATUS, not manufacturing a
        # standard error that does not exist at n=1.
        assert row["realized_rank_ic"]["se"] is None
        assert row["realized_rank_ic"]["t_stat"] is None

    def test_a_thin_row_is_never_suppressed_from_the_artifact(self):
        """§3: a cycle where an arm produces little is recorded as a miss, not
        omitted. Silent absence and a genuine result must never render
        identically — so the fix must not be "hide thin rows".

        PRE-FIX: GREEN (rows were never suppressed). A lock against the wrong
        fix, stated as such."""
        chal, realized = _one_date_fixture(1)
        lb = score_leaderboard(None, [chal], realized, top_n=2, benchmark_ticker="SPY")
        assert [r["name"] for r in lb["specs"]] == ["thinktank_coverage"]
        assert lb["specs"][0]["realized_rank_ic"]["mean"] is not None

    def test_zero_scored_dates_is_insufficient(self):
        """PRE-FIX: RED — ``KeyError: 'confidence'``."""
        chal, _ = _one_date_fixture(1)
        lb = score_leaderboard(None, [chal], {}, top_n=2, benchmark_ticker="SPY")
        row = lb["specs"][0]
        assert row["n_dates_scored"] == 0
        assert row["confidence"] == "insufficient"
        assert row["realized_rank_ic"] is None

    def test_enough_dates_is_ok(self):
        """PRE-FIX: RED — ``KeyError: 'confidence'``."""
        chal, realized = _one_date_fixture(6)
        lb = score_leaderboard(None, [chal], realized, top_n=2, benchmark_ticker="SPY")
        assert lb["specs"][0]["n_dates_scored"] == 6
        assert lb["specs"][0]["confidence"] == "ok"

    def test_the_boundary_is_the_registered_threshold_not_a_call_site_literal(self):
        """champion-challenger-policy.md §10: the slot names its own evidence
        floor in its registry. A literal here would be the same
        two-registries-for-one-fact drift that made the producer leaderboard
        report ``champion: null`` for a month.

        PRE-FIX: RED with an ImportError — ``slot_spec`` and
        ``MIN_DATES_FOR_INFERENCE`` do not exist. Imported inside the test body
        so this is the ONLY test in the file that fails that way."""
        from scoring.leaderboard_scoring import MIN_DATES_FOR_INFERENCE, slot_spec

        for slot_id in ("scanner", "producer"):
            spec = slot_spec(slot_id)
            assert spec.min_dates_for_inference == MIN_DATES_FOR_INFERENCE
            assert spec.horizons_days[0] == 21, "the primary horizon is the continuity surface"
            assert spec.horizons_days == (21, 126, 252)
            assert spec.benchmark_ticker == "SPY"

        floor = MIN_DATES_FOR_INFERENCE
        chal, realized = _one_date_fixture(floor - 1)
        assert score_leaderboard(None, [chal], realized)["specs"][0]["confidence"] == "thin"
        chal, realized = _one_date_fixture(floor)
        assert score_leaderboard(None, [chal], realized)["specs"][0]["confidence"] == "ok"

    def test_a_spec_whose_scoring_raised_is_insufficient_not_silently_comparable(self):
        """The per-spec fail-soft row already emitted null metrics; it must not
        also read as a comparable row.

        PRE-FIX: RED — ``KeyError: 'confidence'``."""

        class _Boom(dict):
            """A by_date mapping that raises when the scorer walks it — the
            per-spec fail-soft path, exercised without stubbing the scorer."""

            def items(self):
                raise RuntimeError("boom")

        lb = score_leaderboard(None, [SpecHistory(name="bad", kind="challenger", by_date=_Boom())], {})
        row = lb["specs"][0]
        assert row["confidence"] == "insufficient"
        assert "boom" in row["error"]

    def test_the_threshold_is_reported_on_the_artifact(self, s3):
        """A status is uninterpretable without the threshold that produced it.
        A reader must be able to tell ``thin`` at 4/5 from ``thin`` at 4/20
        without reading this repo.

        PRE-FIX: RED — the artifact carries no threshold."""
        panel, entries = _matured_to_126_panel()
        _seed_producer_cohort(s3, entries)
        res = build_producer_leaderboard(
            s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
        )
        assert res["leaderboard"]["min_dates_for_inference"] == 5
        assert res["leaderboard"]["horizons_days"] == [21, 126, 252]

    def test_every_row_at_every_horizon_carries_a_confidence(self, s3):
        """Closes-when: EVERY spec row in BOTH leaderboard artifacts carries an
        explicit confidence status — including the long-horizon blocks, which
        is where most rows will be non-``ok`` for months.

        PRE-FIX: RED — neither the field nor the long-horizon blocks exist."""
        panel, entries = _matured_to_126_panel()
        _seed_producer_cohort(s3, entries)
        for d in entries:
            _put_json(s3, f"candidates/{d}/candidates.json", {"scanner_tickers": ["A", "B", "C"]})
            _put_json(
                s3,
                f"candidates_shadow/momentum_sleeve/{d}/candidates.json",
                {"scanner_tickers": ["C", "B", "A"]},
            )
        for builder in (build_producer_leaderboard, build_scanner_leaderboard):
            res = builder(s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader())
            lb = res["leaderboard"]
            rows = list(lb["specs"]) + [r for b in lb["horizons"] for r in b["specs"]]
            assert rows
            for row in rows:
                assert row["confidence"] in {"ok", "thin", "insufficient"}, row
