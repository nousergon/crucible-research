"""Population-relative scoring (alpha-engine-config-I7576).

Every arm was graded against SPY and against nothing else. A SELECTION stage's
job is not to beat the market — it is to beat the population it narrowed, and
in the measured regime the two give opposite verdicts: over 903 tracked names,
SPY returned +0.73% across 21 sessions against the equal-weight population's
+2.13% (+1.95% vs +4.59% across 42). An arm can book a win against the
benchmark while losing to the universe it selected from.

These tests pin the new metric, the scanner slot's primary moving to it, and —
most importantly — that the SPY series is untouched (§3 continuity), proven
against literals captured from `origin/main` at aff64bd8 rather than asserted.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.leaderboard_scoring import (  # noqa: E402
    LEADERBOARD_SLOTS,
    SpecDay,
    SpecHistory,
    _population_return_by_date,
    _topn_alpha_vs_population_metric,
    score_leaderboard,
    slot_spec,
)

# The fixture the baseline literals below were captured against.
REALIZED = {
    "2026-08-03": {"AAA": 0.05, "BBB": 0.02, "CCC": -0.01, "DDD": 0.03,
                   "EEE": -0.04, "SPY": 0.01},
    "2026-08-04": {"AAA": -0.02, "BBB": 0.06, "CCC": 0.00, "DDD": -0.03,
                   "EEE": 0.08, "SPY": 0.02},
    "2026-08-05": {"AAA": 0.01, "BBB": -0.05, "CCC": 0.04, "DDD": 0.07,
                   "EEE": 0.02, "SPY": -0.01},
}


def _hist(name, kind, picks):
    return SpecHistory(
        name=name, kind=kind,
        by_date={d: SpecDay(ranked=list(r)) for d, r in picks.items()},
    )


CHAMP = _hist("champ", "champion", {
    "2026-08-03": ["AAA", "BBB"],
    "2026-08-04": ["BBB", "CCC"],
    "2026-08-05": ["CCC", "DDD"],
})
CHAL = _hist("chal", "challenger", {
    "2026-08-03": ["DDD", "EEE"],
    "2026-08-04": ["EEE", "AAA"],
    "2026-08-05": ["AAA", "BBB"],
})


@pytest.fixture
def scored():
    return score_leaderboard(CHAMP, [CHAL], REALIZED, top_n=2, horizon_days=21)


def _row(board, name):
    return next(r for r in board["specs"] if r["name"] == name)


# ── §3 continuity: the SPY series is unchanged, proven against captured literals

BASELINE_ON_MAIN = {
    "champ": {
        "realized_rank_ic": {"mean": 0.333333, "n_dates": 3, "se": 0.666667,
                             "t_stat": 0.5},
        "topn_alpha_vs_benchmark": {"mean": 0.033333, "n_dates": 3,
                                    "se": 0.016415, "t_stat": 2.0307},
        "topn_alpha_vs_champion": None,
        "n_dates_scored": 3,
        "confidence": "thin",
    },
    "chal": {
        "realized_rank_ic": {"mean": 1.0, "n_dates": 3, "se": 0.0,
                             "t_stat": None},
        "topn_alpha_vs_benchmark": {"mean": -0.005, "n_dates": 3,
                                    "se": 0.007638, "t_stat": -0.6547},
        "topn_alpha_vs_champion": {"mean": -0.038333, "n_dates": 3,
                                   "se": 0.021667, "t_stat": -1.7692},
        "n_dates_scored": 3,
        "confidence": "thin",
    },
}


@pytest.mark.parametrize("name", sorted(BASELINE_ON_MAIN))
def test_pre_change_fields_are_byte_identical(scored, name):
    """Captured from `origin/main` at aff64bd8 on this exact fixture. Every
    field that existed before this change must still equal its old value —
    adding a metric must not perturb one."""
    row = _row(scored, name)
    for field, expected in BASELINE_ON_MAIN[name].items():
        assert row[field] == expected, f"{name}.{field} changed"


def test_leaderboard_level_fields_unchanged(scored):
    assert scored["champion"] == "champ"
    assert scored["horizon_days"] == 21
    assert scored["top_n"] == 2
    assert scored["benchmark_ticker"] == "SPY"
    assert scored["n_dates"] == 3


# ── The new metric ───────────────────────────────────────────────────────────


def test_population_excludes_the_benchmark():
    """SPY is an index proxy, not a name the arm could have picked. Including
    it would make the baseline mean something other than 'the universe I
    narrowed'."""
    ret = REALIZED["2026-08-04"]
    with_spy = sum(ret.values()) / len(ret)
    without = _population_return_by_date(ret, "SPY")
    assert without == pytest.approx((-0.02 + 0.06 + 0.00 - 0.03 + 0.08) / 5)
    assert without != pytest.approx(with_spy)


def test_population_metric_is_the_arithmetic_it_claims():
    """Hand-computed, not golden-filed: champ's top-2 minus the 5-name
    population mean, per date, then averaged."""
    per_date = [
        (0.05 + 0.02) / 2 - (0.05 + 0.02 - 0.01 + 0.03 - 0.04) / 5,
        (0.06 + 0.00) / 2 - (-0.02 + 0.06 + 0.00 - 0.03 + 0.08) / 5,
        (0.04 + 0.07) / 2 - (0.01 - 0.05 + 0.04 + 0.07 + 0.02) / 5,
    ]
    got = _topn_alpha_vs_population_metric(CHAMP, REALIZED, 2, "SPY")
    assert got["mean"] == pytest.approx(sum(per_date) / 3, abs=1e-6)
    assert got["n_dates"] == 3


def test_metric_present_on_every_row(scored):
    for row in scored["specs"]:
        assert "topn_alpha_vs_population" in row
        assert row["topn_alpha_vs_population"]["n_dates"] == 3


def test_metric_needs_no_benchmark():
    """Available on any slot and any date the other metrics are — the
    population comes from the realized join itself, not a separate fetch."""
    board = score_leaderboard(CHAMP, [CHAL], REALIZED, top_n=2,
                              benchmark_ticker=None)
    for row in board["specs"]:
        assert row["topn_alpha_vs_benchmark"] is None
        assert row["topn_alpha_vs_population"] is not None


def test_benchmark_and_population_can_disagree_in_sign():
    """The whole reason the metric exists. A single date where the arm beats a
    weak benchmark and loses to the population it drew from."""
    realized = {"2026-08-03": {"A": 0.01, "B": 0.01, "C": 0.09, "D": 0.09,
                               "SPY": -0.02}}
    arm = _hist("arm", "champion", {"2026-08-03": ["A", "B"]})
    board = score_leaderboard(arm, [], realized, top_n=2)
    row = _row(board, "arm")
    assert row["topn_alpha_vs_benchmark"]["mean"] > 0   # beat SPY
    assert row["topn_alpha_vs_population"]["mean"] < 0  # lost to the universe


# ── Slot registry ────────────────────────────────────────────────────────────


def test_scanner_slot_ranks_on_population():
    assert slot_spec("scanner").primary_metric == "topn_alpha_vs_population"


def test_producer_slot_deliberately_unchanged():
    """Moving it is a separate argument — the producer slot mixes selection
    with sizing."""
    assert slot_spec("producer").primary_metric == "topn_alpha_vs_benchmark"


def test_every_slot_primary_is_a_metric_rows_actually_carry(scored):
    """A registry naming a metric no row emits is the thinktank_coverage defect
    repeating in a new place."""
    emitted = set(scored["specs"][0])
    for spec in LEADERBOARD_SLOTS.values():
        assert spec.primary_metric in emitted, spec.slot_id


# ── Failure isolation ────────────────────────────────────────────────────────


def test_failed_row_carries_the_new_field_as_null():
    """The fail-soft row must have the same shape as a good one, or a consumer
    reading the primary metric hits a KeyError on exactly the rows that already
    went wrong."""
    # ``ranked=None`` raises inside the metric functions on a date that DOES
    # join, which is the real fail-soft path — a spec whose dates simply do not
    # join produces a clean empty row, not an error.
    broken = SpecHistory(
        name="broken", kind="challenger",
        by_date={"2026-08-03": SpecDay(ranked=None)},
    )
    board = score_leaderboard(CHAMP, [broken], REALIZED, top_n=2)
    row = _row(board, "broken")
    assert "error" in row
    assert row["topn_alpha_vs_population"] is None
