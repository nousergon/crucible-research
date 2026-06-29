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
from typing import Iterable

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
        assert s["se"] == pytest.approx(0.1 / (3 ** 0.5), rel=1e-4)
        assert s["t_stat"] == pytest.approx(0.2 / (0.1 / (3 ** 0.5)), rel=1e-4)

    def test_single_date_has_no_se(self):
        s = date_clustered_stats([0.42])
        assert s == {"mean": 0.42, "se": None, "t_stat": None, "n_dates": 1}

    def test_empty_is_none(self):
        assert date_clustered_stats([]) is None


# ── Scorer ────────────────────────────────────────────────────────────────────

def _scanner_specs():
    champ = SpecHistory(
        name="champ", kind="champion",
        by_date={
            "2026-06-01": SpecDay(ranked=["A", "B", "C", "D"]),
            "2026-06-08": SpecDay(ranked=["A", "B", "C", "D"]),
        },
    )
    # challenger inverts the champion order each date.
    chal = SpecHistory(
        name="chal", kind="challenger",
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
            name="agentic", kind="champion",
            by_date={"2026-06-01": SpecDay(ranked=["A", "B"], scores={"A": 90, "B": 70})},
        )
        chal = SpecHistory(
            name="quant", kind="challenger",
            by_date={"2026-06-01": SpecDay(ranked=["B", "A"], scores={"B": 88, "A": 60})},
        )
        realized = {"2026-06-01": {"A": 0.10, "B": 0.02}}
        lb = score_leaderboard(champ, [chal], realized, top_n=1)
        names = {s["name"]: s for s in lb["specs"]}
        # single date → IC defined (2 names), se None.
        assert names["agentic"]["realized_rank_ic"]["mean"] == pytest.approx(1.0)
        assert names["quant"]["realized_rank_ic"]["mean"] == pytest.approx(-1.0)


class TestCohortGate:
    def test_unmatured_date_does_not_join(self):
        champ = SpecHistory(
            name="c", kind="champion",
            by_date={"2026-06-01": SpecDay(ranked=["A", "B"]),
                     "2026-06-08": SpecDay(ranked=["A", "B"])},
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
            name="c", kind="champion",
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


def _put_closes(s3, date_str, closes: dict):
    import io

    import pandas as pd

    df = pd.DataFrame({"ticker": list(closes), "close": list(closes.values())})
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    s3.put_object(Bucket=_BUCKET, Key=f"staging/daily_closes/{date_str}.parquet",
                  Body=buf.getvalue())


def _seed_horizon_calendar(s3, entry_dates: Iterable[str], horizon: int):
    """Put empty placeholder closes for enough trading dates after each entry so
    the horizon resolves (the join only reads entry + horizon-date closes)."""
    # use a simple incrementing calendar of weekday-ish strings.
    extra = [f"2026-07-{d:02d}" for d in range(1, horizon + 5)]
    for d in extra:
        _put_closes(s3, d, {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})


class TestScannerLeaderboardProducer:
    def test_writes_leaderboard_with_realized_join(self, s3):
        from scoring.leaderboard_producers import build_scanner_leaderboard

        # live champion candidates + one shadow challenger for one cohort date.
        entry = "2026-06-01"
        _put_json(s3, f"candidates/{entry}/candidates.json",
                  {"scanner_tickers": ["A", "B", "C", "D"]})
        _put_json(s3, f"candidates_shadow/momentum_sleeve/{entry}/candidates.json",
                  {"scanner_tickers": ["D", "C", "B", "A"]})

        # daily_closes: entry-date closes + a matured horizon close 21 sessions on.
        _put_closes(s3, entry, {"A": 100, "B": 100, "C": 100, "D": 100})
        # 21 trading dates after entry (use July placeholders); the 21st is horizon.
        horizon_dates = [f"2026-07-{d:02d}" for d in range(1, 25)]
        for i, d in enumerate(horizon_dates):
            # A rises most, D falls — champion order tracks returns.
            mult = {0: ("A", 1.10), 1: ("B", 1.05), 2: ("C", 1.00), 3: ("D", 0.95)}
            _put_closes(s3, d, {"A": 110, "B": 105, "C": 100, "D": 95})

        res = build_scanner_leaderboard(s3, _BUCKET, "2026-06-27", top_n=2)
        assert res["status"] == "ok"
        assert res["key"] == "scanner/leaderboard/2026-06-27.json"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        assert got["leaderboard_id"] == "scanner"
        assert got["date"] == "2026-06-27"
        assert got["champion"] == "tech_score_momentum"
        names = {s["name"]: s for s in got["specs"]}
        assert "tech_score_momentum" in names and "momentum_sleeve" in names
        # champion rank order matches realized returns → IC = 1.0 on the one date.
        assert names["tech_score_momentum"]["realized_rank_ic"]["mean"] == pytest.approx(1.0)
        assert got["n_dates"] == 1

    def test_fresh_date_ships_null_metrics(self, s3):
        from scoring.leaderboard_producers import build_scanner_leaderboard

        entry = "2026-06-20"
        _put_json(s3, f"candidates/{entry}/candidates.json",
                  {"scanner_tickers": ["A", "B"]})
        _put_json(s3, f"candidates_shadow/momentum_sleeve/{entry}/candidates.json",
                  {"scanner_tickers": ["B", "A"]})
        _put_closes(s3, entry, {"A": 100, "B": 100})  # no horizon close → no join

        res = build_scanner_leaderboard(s3, _BUCKET, "2026-06-27")
        assert res["status"] == "ok"
        got = res["leaderboard"]
        assert got["n_dates"] == 0
        names = {s["name"]: s for s in got["specs"]}
        assert names["tech_score_momentum"]["realized_rank_ic"] is None

    def test_fail_soft_never_raises_and_alerts_loud(self, s3, monkeypatch):
        import scoring.leaderboard_producers as lp

        # No bucket objects + a deliberately broken client call path: the
        # function must return a status dict, never raise.
        class _BoomS3:
            def get_paginator(self, *a, **k):
                raise RuntimeError("AccessDenied")

        alerts = []
        monkeypatch.setattr(lp, "publish_observe_alert",
                            lambda message, **kw: alerts.append((message, kw)) or True)

        res = lp.build_scanner_leaderboard(_BoomS3(), _BUCKET, "2026-06-27")
        assert res["status"] == "error"
        # config#1403: a build failure means the artifact is NOT written → LOUD.
        assert len(alerts) == 1
        msg, kw = alerts[0]
        assert "scanner leaderboard build FAILED" in msg
        assert kw["dedup_key"] == "scanner_leaderboard_build_error:2026-06-27"
        assert kw["source"] == "research:scanner_leaderboard"


class TestProducerLeaderboardProducer:
    def test_writes_with_enter_scores(self, s3):
        from scoring.leaderboard_producers import build_producer_leaderboard

        entry = "2026-06-01"
        # live champion signals.json + one shadow challenger.
        _put_json(s3, f"signals/{entry}/signals.json", {"signals": {
            "A": {"signal": "ENTER", "score": 90},
            "B": {"signal": "ENTER", "score": 70},
            "Z": {"signal": "HOLD", "score": 99},  # excluded (not ENTER)
        }})
        _put_json(s3, f"signals_shadow/no_agent_quant/{entry}/signals.json", {"signals": {
            "B": {"signal": "ENTER", "score": 88},
            "A": {"signal": "ENTER", "score": 60},
        }})
        _put_closes(s3, entry, {"A": 100, "B": 100})
        for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
            _put_closes(s3, d, {"A": 110, "B": 102})  # A outperforms

        res = build_producer_leaderboard(s3, _BUCKET, "2026-06-27", top_n=1)
        assert res["status"] == "ok"
        assert res["key"] == "research/producer_leaderboard/2026-06-27.json"
        got = json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())
        assert got["leaderboard_id"] == "producer"
        assert got["champion"] == "agentic_sector_teams"
        names = {s["name"]: s for s in got["specs"]}
        assert "no_agent_quant" in names
        # champion ranks A>B and A outperforms → IC = 1.0.
        assert names["agentic_sector_teams"]["realized_rank_ic"]["mean"] == pytest.approx(1.0)

    def test_fail_soft_never_raises_and_alerts_loud(self, s3, monkeypatch):
        import scoring.leaderboard_producers as lp

        class _BoomS3:
            def get_paginator(self, *a, **k):
                raise RuntimeError("AccessDenied")

        alerts = []
        monkeypatch.setattr(lp, "publish_observe_alert",
                            lambda message, **kw: alerts.append((message, kw)) or True)

        res = lp.build_producer_leaderboard(_BoomS3(), _BUCKET, "2026-06-27")
        assert res["status"] == "error"
        assert len(alerts) == 1
        msg, kw = alerts[0]
        assert "producer leaderboard build FAILED" in msg
        assert kw["dedup_key"] == "producer_leaderboard_build_error:2026-06-27"
        assert kw["source"] == "research:producer_leaderboard"
