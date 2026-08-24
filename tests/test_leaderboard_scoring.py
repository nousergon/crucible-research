"""Unit tests for the shared champion/challenger leaderboard scorer
(scoring/leaderboard_scoring.py) + the two thin producers
(scoring/leaderboard_producers.py) — config#1221 (scanner) + config#1223
(producer), ONE engine.

Locks down:
- Pure statistics: average-rank Spearman IC + date-clustered (weeks-as-N)
  significance, against hand-computed values.
- The scorer reads shadow picks joined to realized 21d outcomes and emits the
  exact leaderboard contract (champion + per-spec rank-IC + top-N alpha).
- Cohort gate: a date with no matured realized return does not join — its
  metric is an honest None (never fabricated), and a fresh run ships n_dates=0.
- Observe-only fail-soft: every producer entry point returns a status dict and
  never raises into the live path.
- The S3 producers fixture-drive shadow candidates/signals + daily_closes
  parquets through moto and assert the written leaderboard JSON shape.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from scoring.leaderboard_scoring import (
    SpecDay,
    SpecHistory,
    date_clustered_stats,
    score_leaderboard,
    spearman_ic,
)

_BUCKET = "alpha-engine-research"


# ── Pure statistics ───────────────────────────────────────────────────────────


class TestSpearmanIC:
    def test_perfect_positive(self):
        assert spearman_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert spearman_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_ties_use_average_rank(self):
        # signal ties on the first two; realized strictly increasing.
        ic = spearman_ic([1, 1, 3, 4], [10, 20, 30, 40])
        assert ic is not None and 0.0 < ic < 1.0

    def test_undefined_when_too_few(self):
        assert spearman_ic([1], [2]) is None

    def test_undefined_on_zero_variance(self):
        assert spearman_ic([5, 5, 5], [1, 2, 3]) is None


class TestDateClusteredStats:
    def test_mean_se_tstat(self):
        # per-date ICs 0.1, 0.2, 0.3 → mean 0.2, sd 0.1, se 0.1/sqrt(3).
        s = date_clustered_stats([0.1, 0.2, 0.3])
        assert s["n_dates"] == 3
        assert s["mean"] == pytest.approx(0.2)
        assert s["se"] == pytest.approx(0.1 / (3**0.5), rel=1e-4)
        assert s["t_stat"] == pytest.approx(0.2 / (0.1 / (3**0.5)), rel=1e-4)

    def test_single_date_has_no_se(self):
        s = date_clustered_stats([0.42])
        assert s == {
            "mean": 0.42, "se": None, "t_stat": None, "n_dates": 1,
            # No SE exists for one observation, and the block says so rather
            # than omitting the field — a reader must never have to infer
            # WHY an SE is absent (alpha-engine-config-I8263).
            "se_method": "unavailable",
        }

    def test_empty_is_none(self):
        assert date_clustered_stats([]) is None

    def test_independent_observations_keep_the_iid_se_and_say_so(self):
        """The original weeks-as-N contract, unchanged and now DECLARED.

        ``overlap_lags=0`` is a positive statement that the observations do not
        overlap — not "not checked" — so the iid SE it labels is correct rather
        than merely conventional.
        """
        s = date_clustered_stats([0.1, 0.2, 0.3], overlap_lags=0)
        assert s["se_method"] == "iid"
        assert s["se"] == pytest.approx(0.1 / (3**0.5), rel=1e-4)

    def test_overlapping_observations_get_a_hac_se_not_an_iid_one(self):
        """The defect this parameter exists to remove.

        ``sd/sqrt(n)`` over overlapping forward windows understates the SE by
        roughly ``sqrt(lags+1)`` and reads as significance. A block declaring
        overlap must carry a HAC method.
        """
        series = [0.05, 0.06, 0.05, 0.07, 0.06, 0.05, 0.06, 0.07]
        s = date_clustered_stats(series, overlap_lags=3)
        assert s["se_method"] == "newey-west"
        assert s["overlap_lags"] == 3
        assert s["se"] is not None

    def test_a_positively_autocorrelated_series_gets_a_LARGER_se_under_hac(self):
        """Direction matters, not just the label.

        Overlapping windows share most of their span, which induces strong
        POSITIVE autocorrelation. HAC must widen the SE relative to iid — a
        correction that narrowed it would make the very numbers it was added to
        discipline look more significant, not less.
        """
        series = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        iid = date_clustered_stats(series, overlap_lags=0)
        hac = date_clustered_stats(series, overlap_lags=4)
        assert hac["se_method"] == "newey-west"
        assert hac["se"] > iid["se"], (iid["se"], hac["se"])
        assert abs(hac["t_stat"]) < abs(iid["t_stat"])

    def test_an_uncomputable_hac_se_is_null_never_an_iid_substitute(self):
        """Silently downgrading to iid would be indistinguishable from a
        correct number, which is the whole failure mode. `unavailable` is the
        honest output (champion-challenger-policy.md §7.2)."""
        import scoring.leaderboard_scoring as ls

        original = ls._hac_se
        try:
            ls._hac_se = lambda *a, **k: None
            s = date_clustered_stats([0.1, 0.2, 0.3, 0.4], overlap_lags=2)
        finally:
            ls._hac_se = original
        assert s["se_method"] == "unavailable"
        assert s["se"] is None
        assert s["t_stat"] is None
        assert s["mean"] == pytest.approx(0.25)


class TestOverlapLags:
    """``overlap_lags_for`` — the (horizon, cohort spacing) → lag mapping."""

    def test_weekly_cohort_on_a_21_session_horizon_overlaps_four_neighbours(self):
        """The live configuration. Cohort dates ~5 sessions apart, forward
        window 21 sessions: each window shares span with four neighbours."""
        from scoring.leaderboard_scoring import overlap_lags_for

        assert overlap_lags_for(21, 5) == 4

    def test_a_longer_horizon_overlaps_more_at_the_same_spacing(self):
        """Monotone in the horizon — the 126- and 252-session blocks are the
        MORE distorted ones, which is the opposite of how they are usually
        read."""
        from scoring.leaderboard_scoring import overlap_lags_for

        assert overlap_lags_for(126, 5) > overlap_lags_for(21, 5)
        assert overlap_lags_for(252, 5) > overlap_lags_for(126, 5)

    def test_a_weekly_holding_period_on_weekly_cohorts_does_not_overlap(self):
        """The measurement Brian ruled for on 2026-08-24: hold the cut for the
        week it is formed. Horizon == spacing, so the windows abut rather than
        overlap and the iid SE is correct as written."""
        from scoring.leaderboard_scoring import overlap_lags_for

        assert overlap_lags_for(5, 5) == 0

    def test_unknown_spacing_applies_no_correction(self):
        """Fewer than two cohort dates means no spacing exists. Inventing one
        would be a fabricated correction; applying none is honest, and the
        block's declared `overlap_lags: 0` is then a statement about what was
        measurable."""
        from scoring.leaderboard_scoring import overlap_lags_for

        assert overlap_lags_for(21, None) == 0
        assert overlap_lags_for(21, 0) == 0


# ── Scorer ────────────────────────────────────────────────────────────────────


def _scanner_specs():
    champ = SpecHistory(
        name="champ",
        kind="champion",
        by_date={
            "2026-06-01": SpecDay(ranked=["A", "B", "C", "D"]),
            "2026-06-08": SpecDay(ranked=["A", "B", "C", "D"]),
        },
    )
    # challenger inverts the champion order each date.
    chal = SpecHistory(
        name="chal",
        kind="challenger",
        by_date={
            "2026-06-01": SpecDay(ranked=["D", "C", "B", "A"]),
            "2026-06-08": SpecDay(ranked=["D", "C", "B", "A"]),
        },
    )
    realized = {
        # A best, D worst → champion rank order tracks returns (high IC),
        # challenger inverts it (negative IC).
        "2026-06-01": {"A": 0.10, "B": 0.05, "C": 0.00, "D": -0.05},
        "2026-06-08": {"A": 0.08, "B": 0.04, "C": 0.01, "D": -0.03},
    }
    return champ, [chal], realized


class TestScoreLeaderboard:
    def test_shape_and_champion(self):
        champ, chals, realized = _scanner_specs()
        lb = score_leaderboard(champ, chals, realized, top_n=2)
        assert lb["champion"] == "champ"
        assert lb["horizon_days"] == 21
        assert lb["top_n"] == 2
        assert lb["n_dates"] == 2
        names = {s["name"]: s for s in lb["specs"]}
        assert set(names) == {"champ", "chal"}
        # champion never gets a top-N-alpha-vs-itself.
        assert names["champ"]["topn_alpha_vs_champion"] is None

    def test_champion_positive_ic_challenger_negative(self):
        champ, chals, realized = _scanner_specs()
        lb = score_leaderboard(champ, chals, realized, top_n=2)
        names = {s["name"]: s for s in lb["specs"]}
        assert names["champ"]["realized_rank_ic"]["mean"] == pytest.approx(1.0)
        assert names["chal"]["realized_rank_ic"]["mean"] == pytest.approx(-1.0)
        assert names["champ"]["realized_rank_ic"]["n_dates"] == 2

    def test_topn_alpha_vs_champion(self):
        champ, chals, realized = _scanner_specs()
        lb = score_leaderboard(champ, chals, realized, top_n=2)
        names = {s["name"]: s for s in lb["specs"]}
        alpha = names["chal"]["topn_alpha_vs_champion"]
        # champion top-2 {A,B} vs challenger top-2 {D,C}.
        # date1: chal mean(-0.05,0.00)=-0.025, champ mean(0.10,0.05)=0.075 → -0.10
        # date2: chal mean(-0.03,0.01)=-0.010, champ mean(0.08,0.04)=0.060 → -0.07
        assert alpha["n_dates"] == 2
        assert alpha["mean"] == pytest.approx((-0.10 + -0.07) / 2)

    def test_producer_scores_use_explicit_scores(self):
        # producer specs carry per-ticker scores → rank-IC uses them directly.
        champ = SpecHistory(
            name="agentic",
            kind="champion",
            by_date={"2026-06-01": SpecDay(ranked=["A", "B"], scores={"A": 90, "B": 70})},
        )
        chal = SpecHistory(
            name="quant",
            kind="challenger",
            by_date={"2026-06-01": SpecDay(ranked=["B", "A"], scores={"B": 88, "A": 60})},
        )
        realized = {"2026-06-01": {"A": 0.10, "B": 0.02}}
        lb = score_leaderboard(champ, [chal], realized, top_n=1)
        names = {s["name"]: s for s in lb["specs"]}
        # single date → IC defined (2 names), se None.
        assert names["agentic"]["realized_rank_ic"]["mean"] == pytest.approx(1.0)
        assert names["quant"]["realized_rank_ic"]["mean"] == pytest.approx(-1.0)

    def test_champion_none_disables_topn_alpha_vs_champion_only(self):
        """alpha-engine-config-I2998: no champion registered (config-I2993)
        must not block scoring — realized_rank_ic stays champion-free-computable,
        only topn_alpha_vs_champion (which needs a live comparator) goes null,
        and no champion row is emitted."""
        champ, chals, realized = _scanner_specs()
        lb = score_leaderboard(None, chals, realized, top_n=2)
        assert lb["champion"] is None
        names = {s["name"]: s for s in lb["specs"]}
        assert set(names) == {"chal"}
        assert names["chal"]["topn_alpha_vs_champion"] is None
        assert names["chal"]["realized_rank_ic"]["mean"] == pytest.approx(-1.0)

    def test_topn_alpha_vs_benchmark(self):
        # champion-free direct lift: spec top-N mean realized return minus the
        # benchmark ticker's own realized return, per date.
        champ, chals, realized = _scanner_specs()
        realized = {d: {**r, "SPY": 0.01} for d, r in realized.items()}
        lb = score_leaderboard(champ, chals, realized, top_n=2)
        names = {s["name"]: s for s in lb["specs"]}
        # champ top-2 {A,B}: date1 mean(0.10,0.05)-0.01=0.065; date2 mean(0.08,0.04)-0.01=0.05
        alpha = names["champ"]["topn_alpha_vs_benchmark"]
        assert alpha["n_dates"] == 2
        assert alpha["mean"] == pytest.approx((0.065 + 0.05) / 2)

    def test_benchmark_ticker_none_disables_metric(self):
        champ, chals, realized = _scanner_specs()
        lb = score_leaderboard(champ, chals, realized, top_n=2, benchmark_ticker=None)
        names = {s["name"]: s for s in lb["specs"]}
        assert names["champ"]["topn_alpha_vs_benchmark"] is None


class TestCohortGate:
    def test_unmatured_date_does_not_join(self):
        champ = SpecHistory(
            name="c",
            kind="champion",
            by_date={"2026-06-01": SpecDay(ranked=["A", "B"]), "2026-06-08": SpecDay(ranked=["A", "B"])},
        )
        # only the first date has realized returns (second hasn't matured).
        realized = {"2026-06-01": {"A": 0.05, "B": -0.01}}
        lb = score_leaderboard(champ, [], realized)
        assert lb["n_dates"] == 1
        c = lb["specs"][0]
        assert c["realized_rank_ic"]["n_dates"] == 1
        assert c["n_dates_scored"] == 1

    def test_no_realized_yields_null_metrics(self):
        champ = SpecHistory(
            name="c",
            kind="champion",
            by_date={"2026-06-08": SpecDay(ranked=["A", "B"])},
        )
        lb = score_leaderboard(champ, [], {})  # nothing matured
        assert lb["n_dates"] == 0
        assert lb["specs"][0]["realized_rank_ic"] is None


# ── S3 producers (moto) ───────────────────────────────────────────────────────


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_json(s3, key, obj):
    s3.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


# Closes are no longer seeded into S3 (alpha-engine-config-I5195). The scorer
# reads its panel from the DURABLE ArcticDB store via an injectable loader, so
# these tests build a panel in memory and pass it as `closes_panel_loader`.
# Seeding `staging/daily_closes/` here would test a source production no longer
# uses — and that prefix's 7-day expiry vs a 21-session horizon was the defect.


class _Panel:
    """Builds `{date: {ticker: close}}` and hands it back as a loader."""

    def __init__(self):
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update({t: float(c) for t, c in closes.items()})
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


class TestScannerLeaderboardProducer:
    def test_writes_leaderboard_with_realized_join(self, s3):
        from scoring.leaderboard_producers import build_scanner_leaderboard

        # live champion candidates + one shadow challenger for one cohort date.
        entry = "2026-06-01"
        # `scanner_tickers` IS the champion arm's ranking, best first: the live
        # path applies SCANNER_SPECS[LIVE_CHAMPION].rank, which sorts by its own
        # score descending and slices (SCANNER_CONTRACT.md §4). `tech_score` is
        # the ranking signal of the `tech_score_gate` CHALLENGER, and is seeded
        # here in the exact reverse order so a producer that re-sorts the
        # champion by a rival arm's signal — as it did between
        # alpha-engine-config-I7645 and I7808 — cannot pass.
        _put_json(s3, f"candidates/{entry}/candidates.json", {
            "scanner_tickers": ["A", "B", "C", "D"],
            "scanner_eval_log": [
                {"ticker": "A", "tech_score": 0.3},
                {"ticker": "B", "tech_score": 0.5},
                {"ticker": "C", "tech_score": 0.7},
                {"ticker": "D", "tech_score": 0.9},
            ],
        })
        _put_json(
            s3, f"candidates_shadow/tech_score_gate/{entry}/candidates.json", {"scanner_tickers": ["D", "C", "B", "A"]}
        )

        # daily_closes: entry-date closes + a matured horizon close 21 sessions on.
        panel = _Panel().put(entry, {"A": 100, "B": 100, "C": 100, "D": 100})
        # 21 trading dates after entry (use July placeholders); the 21st is horizon.
        for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
            # A rises most, D falls — champion order tracks returns.
            panel.put(d, {"A": 110, "B": 105, "C": 100, "D": 95})

        res = build_scanner_leaderboard(
            s3,
            _BUCKET,
            "2026-06-27",
            top_n=2,
            closes_panel_loader=panel.loader(),
        )
        assert res["status"] == "ok"
        assert res["key"] == "scanner/leaderboard/2026-06-27.json"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        assert got["leaderboard_id"] == "scanner"
        assert got["date"] == "2026-06-27"
        assert got["champion"] == "momentum_sleeve"
        names = {s["name"]: s for s in got["specs"]}
        assert "momentum_sleeve" in names and "tech_score_gate" in names
        # The champion's own list order matches realized returns → IC = 1.0 on
        # the one date. `tech_score` here is its exact reverse, so a producer
        # ranking the champion by the challenger's signal would score -1.0.
        assert names["momentum_sleeve"]["realized_rank_ic"]["mean"] == pytest.approx(1.0)
        assert got["n_dates"] == 1

    def test_champion_with_an_empty_cut_reports_no_rank_ic(self, s3):
        """An empty champion cut has no order to correlate, so the rank-IC must
        be MISSING rather than computed over nothing.

        This replaces a test asserting that a champion WITHOUT a
        `scanner_eval_log` reports no rank-IC (alpha-engine-config-I7645). That
        premise was wrong: it treated `tech_score` as the champion's ranking
        signal, which it had not been since the 2026-07-22 cutover, and so it
        locked in re-sorting the champion's cut by a rival arm's signal
        (alpha-engine-config-I7808). A cut's list order IS its ranking; the only
        cut with no ranking is an empty one.
        """
        from scoring.leaderboard_producers import build_scanner_leaderboard

        entry = "2026-06-01"
        _put_json(s3, f"candidates/{entry}/candidates.json", {"scanner_tickers": []})
        _put_json(
            s3, f"candidates_shadow/tech_score_gate/{entry}/candidates.json",
            {"scanner_tickers": ["D", "C", "B", "A"]},
        )
        panel = _Panel().put(entry, {"A": 100, "B": 100, "C": 100, "D": 100})
        for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
            panel.put(d, {"A": 110, "B": 105, "C": 100, "D": 95})

        res = build_scanner_leaderboard(
            s3, _BUCKET, "2026-06-27", top_n=2, closes_panel_loader=panel.loader(),
        )
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        names = {s["name"]: s for s in got["specs"]}
        # The champion contributed no scorable day at all; the challenger still
        # scores, which is the point of the fail-soft substrate.
        assert names.get("momentum_sleeve", {}).get("realized_rank_ic") is None
        assert names["tech_score_gate"]["n_dates_scored"] == 1

    def test_fresh_date_ships_null_metrics(self, s3):
        from scoring.leaderboard_producers import build_scanner_leaderboard

        entry = "2026-06-20"
        _put_json(s3, f"candidates/{entry}/candidates.json", {"scanner_tickers": ["A", "B"]})
        _put_json(s3, f"candidates_shadow/tech_score_gate/{entry}/candidates.json", {"scanner_tickers": ["B", "A"]})
        # A CAPABLE source (ArcticDB holds years) in which THIS cohort simply
        # has not matured: 5 sessions after entry, horizon 21. That is the
        # honest "fresh date" state and must stay `ok` — distinct from an
        # unmeasurable source (alpha-engine-config-I5195). Seeding only the
        # entry date, as this fixture once did, is indistinguishable from a
        # 1-session retention window and no longer represents reality.
        panel = _Panel().put(entry, {"A": 100, "B": 100})
        for d in [f"2026-05-{d:02d}" for d in range(1, 21)]:  # history before entry
            panel.put(d, {"A": 100, "B": 100})
        for d in [f"2026-06-{d:02d}" for d in range(21, 26)]:  # only 5 sessions after
            panel.put(d, {"A": 101, "B": 101})

        res = build_scanner_leaderboard(
            s3,
            _BUCKET,
            "2026-06-27",
            closes_panel_loader=panel.loader(),
        )
        assert res["status"] == "ok"
        got = res["leaderboard"]
        assert got["n_dates"] == 0
        names = {s["name"]: s for s in got["specs"]}
        assert names["momentum_sleeve"]["realized_rank_ic"] is None

    def test_fail_soft_never_raises_and_alerts_loud(self, s3, monkeypatch):
        import scoring.leaderboard_producers as lp

        # No bucket objects + a deliberately broken client call path: the
        # function must return a status dict, never raise.
        class _BoomS3:
            def get_paginator(self, *a, **k):
                raise RuntimeError("AccessDenied")

        alerts = []
        monkeypatch.setattr(lp, "publish_observe_alert", lambda message, **kw: alerts.append((message, kw)) or True)

        res = lp.build_scanner_leaderboard(_BoomS3(), _BUCKET, "2026-06-27")
        assert res["status"] == "error"
        # config#1403: a build failure means the artifact is NOT written → LOUD.
        assert len(alerts) == 1
        msg, kw = alerts[0]
        assert "scanner leaderboard build FAILED" in msg
        assert kw["dedup_key"] == "scanner_leaderboard_build_error:2026-06-27"
        assert kw["source"] == "research:scanner_leaderboard"


class TestProducerLeaderboardProducer:
    def test_writes_with_enter_scores(self, s3, monkeypatch):
        from producers.registry import ProducerSpec
        from scoring.leaderboard_producers import build_producer_leaderboard

        # config-I2993: agentic_sector_teams is retired — no producer is
        # currently registered kind=="champion". This test still needs to lock
        # down the join/scoring path end-to-end, so it monkeypatches in a
        # standin champion spec (as if a successor champion were registered)
        # rather than asserting on the now-retired name. The no-champion-
        # registered CURRENT state is covered separately below.
        standin = ProducerSpec(
            name="standin_champion",
            kind="champion",
            version="v1",
            description="test standin — not the real registry",
            build=None,
        )
        monkeypatch.setattr("producers.registry.champion_producer", lambda: standin)

        entry = "2026-06-01"
        # live champion signals.json + one shadow challenger.
        _put_json(
            s3,
            f"signals/{entry}/signals.json",
            {
                "signals": {
                    "A": {"signal": "ENTER", "score": 90},
                    "B": {"signal": "ENTER", "score": 70},
                    "Z": {"signal": "HOLD", "score": 99},  # excluded (not ENTER)
                }
            },
        )
        _put_json(
            s3,
            f"signals_shadow/no_agent_quant/{entry}/signals.json",
            {
                "signals": {
                    "B": {"signal": "ENTER", "score": 88},
                    "A": {"signal": "ENTER", "score": 60},
                }
            },
        )
        panel = _Panel().put(entry, {"A": 100, "B": 100})
        for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
            panel.put(d, {"A": 110, "B": 102})  # A outperforms

        res = build_producer_leaderboard(
            s3,
            _BUCKET,
            "2026-06-27",
            top_n=1,
            closes_panel_loader=panel.loader(),
        )
        assert res["status"] == "ok"
        assert res["key"] == "research/producer_leaderboard/2026-06-27.json"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        assert got["leaderboard_id"] == "producer"
        assert got["champion"] == "standin_champion"
        names = {s["name"]: s for s in got["specs"]}
        assert "no_agent_quant" in names
        # champion ranks A>B and A outperforms → IC = 1.0.
        assert names["standin_champion"]["realized_rank_ic"]["mean"] == pytest.approx(1.0)

    def test_no_champion_registered_still_scores_challengers(self, s3):
        # Current real-registry state (config-I2993): agentic_sector_teams is
        # retired and no successor champion is registered. Per alpha-engine
        # -config-I2998, this must NOT block the leaderboard from being
        # written or challengers from being scored — the prior behavior
        # (writing nothing at all) meant crucible-backtester's
        # champion_promotion.py could never observe this arm's evidence,
        # exactly the "simultaneous no-contest" defect I2998 was filed to
        # close. Champion-free metrics (realized_rank_ic,
        # topn_alpha_vs_benchmark) must still be computed.
        from scoring.leaderboard_producers import build_producer_leaderboard

        entry = "2026-06-01"
        _put_json(
            s3,
            f"signals_shadow/no_agent_quant/{entry}/signals.json",
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

        res = build_producer_leaderboard(
            s3,
            _BUCKET,
            "2026-06-27",
            top_n=1,
            closes_panel_loader=panel.loader(),
        )
        assert res["status"] == "ok"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        assert got["champion"] is None
        names = {s["name"]: s for s in got["specs"]}
        assert "no_agent_quant" in names
        row = names["no_agent_quant"]
        assert row["topn_alpha_vs_champion"] is None
        # top-1 pick is A (score 90): realized 0.10 vs SPY's realized 0.05.
        assert row["topn_alpha_vs_benchmark"]["mean"] == pytest.approx(0.05)
        assert row["realized_rank_ic"]["mean"] == pytest.approx(1.0)

    def test_fail_soft_never_raises_and_alerts_loud(self, s3, monkeypatch):
        import scoring.leaderboard_producers as lp

        class _BoomS3:
            def get_paginator(self, *a, **k):
                raise RuntimeError("AccessDenied")

        alerts = []
        monkeypatch.setattr(lp, "publish_observe_alert", lambda message, **kw: alerts.append((message, kw)) or True)

        res = lp.build_producer_leaderboard(_BoomS3(), _BUCKET, "2026-06-27")
        assert res["status"] == "error"
        assert len(alerts) == 1
        msg, kw = alerts[0]
        assert "producer leaderboard build FAILED" in msg
        assert kw["dedup_key"] == "producer_leaderboard_build_error:2026-06-27"
        assert kw["source"] == "research:producer_leaderboard"
