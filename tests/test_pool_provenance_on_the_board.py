"""Pool provenance reaches the surface where arms are COMPARED
(alpha-engine-config-I11424).

Every research arm's shadow records where its pool came from —
``arm_pool.pool_source``, plus ``pool_size`` / ``n_selected``
(``producers/filling_arms.py::build_shadow_payload``,
``producers/research_arms.py``). None of it reached the leaderboard row, so a
divergence in what an arm ranked FROM was visible only to someone who opened a
raw shadow artifact for a specific date and knew to look.

Not hypothetical. `alpha-engine-config-I11396` is exactly this defect going
unnoticed: `scanner_top20_predictor` and `scanner_predictor_direct` ranked from
near-disjoint populations for weeks, one of them collapsing to a **pool of 1**
on 2026-08-03 and 2026-08-04, and the board rendered both as ordinary rows. The
evidence that cracked it was `arm_pool.pool_source` read off S3 by hand.

§4's vacuity guard checks whether two arms' PICKS collide. Nothing checked
whether their POOLS differ — the upstream question, and the one that decides
whether a collision means anything.

VERIFIED RED (champion-challenger-policy.md §7.4). Removing the single
`"pool_provenance": pool_provenance_for(...)` line from `score_leaderboard._row`
and re-running this file: 5 of the 7 fail with `KeyError: 'pool_provenance'`,
and `test_the_key_exists_on_every_row` fails with `AssertionError:
attractiveness_60` — the row the board renders without saying what it ranked
from.

This is an observability change and must not move any metric. The 21d
continuity lock in ``tests/test_leaderboard_long_horizon_and_confidence.py``
guards that; ``pool_provenance`` is listed in its ``_ADDITIVE_SINCE_CAPTURE``
allowlist rather than the protected literals being re-pinned.
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


def _shadow(s3, arm: str, date_str: str, n: int, pool: dict | None) -> None:
    doc: dict = {
        "signals": {
            t: {"signal": "ENTER", "score": float(100 - i)}
            for i, t in enumerate(_TICKERS[:n])
        }
    }
    if pool is not None:
        doc["arm_pool"] = pool
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"signals_shadow/{arm}/{date_str}/signals.json",
        Body=json.dumps(doc).encode(),
    )


def _matured_panel() -> _Panel:
    panel = _Panel().put(_ENTRY, {**dict.fromkeys(_TICKERS, 100.0), "SPY": 100.0})
    for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
        panel.put(d, {
            **{t: 100.0 + (60 - i) * 0.5 for i, t in enumerate(_TICKERS)},
            "SPY": 105.0,
        })
    return panel


def _board(s3, panel) -> dict:
    from scoring.leaderboard_producers import build_producer_leaderboard

    res = build_producer_leaderboard(
        s3, _BUCKET, _AS_OF, closes_panel_loader=panel.loader()
    )
    assert res["status"] == "ok", res
    return json.loads(s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read())


class TestProvenanceIsOnTheComparisonSurface:
    def test_two_arms_with_different_pools_are_distinguishable_on_the_board(self, s3):
        """The I11396 shape, reduced: two arms, two pools, one board. Before
        this, both rendered identically."""
        _shadow(s3, "attractiveness_60", _ENTRY, 60, {
            "pool_source": "pinned_cut:attractiveness_top_60:attractiveness_score",
            "pool_size": 60, "n_selected": 60,
        })
        _shadow(s3, "predictor_from_60", _ENTRY, 20, {
            "pool_source": "predictor_cut:attractiveness_top_20",
            "pool_size": 20, "n_selected": 20,
        })
        rows = {r["name"]: r for r in _board(s3, _matured_panel())["specs"]}
        a = rows["attractiveness_60"]["pool_provenance"]
        b = rows["predictor_from_60"]["pool_provenance"]
        assert a["pool_source"] != b["pool_source"]
        assert a["pool_source"].startswith("pinned_cut:")
        assert b["pool_source"].startswith("predictor_cut:")

    def test_a_pool_that_collapsed_on_one_date_is_visible(self, s3):
        """The literal 2026-08-03/04 finding: a pool of ONE among healthy
        dates. Reported as a MINIMUM, because an average hides it."""
        _shadow(s3, "tech_score_20", _ENTRY, 20, {
            "pool_source": "pinned_cut:attractiveness_top_60:tech_score",
            "pool_size": 1, "n_selected": 1,
        })
        _shadow(s3, "tech_score_20", "2026-06-02", 20, {
            "pool_source": "pinned_cut:attractiveness_top_60:tech_score",
            "pool_size": 60, "n_selected": 20,
        })
        panel = _matured_panel()
        panel.put("2026-06-02", {**dict.fromkeys(_TICKERS, 100.0), "SPY": 100.0})
        prov = {
            r["name"]: r for r in _board(s3, panel)["specs"]
        }["tech_score_20"]["pool_provenance"]
        assert prov["pool_size_min"] == 1
        assert prov["pool_size_max"] == 60

    def test_the_selection_ratio_separates_20_of_60_from_20_of_20(self, s3):
        """Both arms render as "20 picks" and are doing different things. A
        ratio of 1.0 means the arm's selection selected nothing — its pool WAS
        its pick list (alpha-engine-config-I11390)."""
        _shadow(s3, "tech_score_20", _ENTRY, 20, {
            "pool_source": "pinned_cut:attractiveness_top_60:tech_score",
            "pool_size": 60, "n_selected": 20,
        })
        _shadow(s3, "predictor_from_60", _ENTRY, 20, {
            "pool_source": "predictor_cut:attractiveness_top_20",
            "pool_size": 20, "n_selected": 20,
        })
        rows = {r["name"]: r for r in _board(s3, _matured_panel())["specs"]}
        assert rows["tech_score_20"]["pool_provenance"]["selection_ratio"] == 0.3333
        assert rows["predictor_from_60"]["pool_provenance"]["selection_ratio"] == 1.0

    def test_a_pool_that_changed_mid_window_is_named_not_flattened(self, s3):
        """Silently taking the last source would have rendered the
        2026-09-18 universe_cut move — which replaced 100% of a downstream
        arm's input population — as an ordinary row."""
        _shadow(s3, "tech_score_20", _ENTRY, 20, {
            "pool_source": "pinned_cut:attractiveness_top_60:tech_score",
            "pool_size": 60, "n_selected": 20,
        })
        _shadow(s3, "tech_score_20", "2026-06-02", 20, {
            "pool_source": "pinned_cut:tech_score_top_60:tech_score",
            "pool_size": 60, "n_selected": 20,
        })
        panel = _matured_panel()
        panel.put("2026-06-02", {**dict.fromkeys(_TICKERS, 100.0), "SPY": 100.0})
        prov = {
            r["name"]: r for r in _board(s3, panel)["specs"]
        }["tech_score_20"]["pool_provenance"]
        assert prov["pool_source_changed"] is True
        assert prov["pool_source"] is None
        assert len(prov["pool_sources"]) == 2


class TestAbsenceIsNotZero:
    def test_a_shadow_with_no_pool_block_reports_none_not_a_fabricated_pool(self, s3):
        """"This surface does not record it" and "the pool was empty" are
        different facts and must not render alike (§7.2)."""
        _shadow(s3, "attractiveness_20", _ENTRY, 20, None)
        rows = {r["name"]: r for r in _board(s3, _matured_panel())["specs"]}
        assert rows["attractiveness_20"]["pool_provenance"] is None

    def test_the_key_exists_on_every_row(self, s3):
        """A consumer must be able to tell "not recorded" from "this row threw"
        — so the key is present even where the value is null."""
        _shadow(s3, "attractiveness_20", _ENTRY, 20, None)
        for row in _board(s3, _matured_panel())["specs"]:
            assert "pool_provenance" in row, row["name"]

    def test_a_malformed_pool_field_degrades_without_losing_the_day(self, s3):
        """The picks are the measurement; provenance is observability on top.
        A bad value must not take out the arm's cohort date."""
        _shadow(s3, "attractiveness_20", _ENTRY, 20, {
            "pool_source": "pinned_cut:attractiveness_top_60:attractiveness_score",
            "pool_size": "sixty", "n_selected": None,
        })
        row = {
            r["name"]: r for r in _board(s3, _matured_panel())["specs"]
        }["attractiveness_20"]
        assert row["n_dates_scored"] >= 1
        assert row["pool_provenance"]["pool_size_min"] is None
        assert row["pool_provenance"]["pool_source"] is not None
