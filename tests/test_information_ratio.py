"""Information ratio — the research slot's primary metric.

Brian's ruling 2026-09-22 (alpha-engine-config-I11393): the research slot
carries arms at DIFFERING declared widths, so its primary statistic must price
concentration. These tests pin the property that motivated the change — a raw
mean prefers the narrower arm whenever a ranking carries signal, and the
information ratio does not — plus the degenerate cases that must render as a
stated absence rather than a substituted number.
"""

from __future__ import annotations

import math

import pytest

from scoring.leaderboard_scoring import (
    IR_INSUFFICIENT,
    IR_ZERO_DISPERSION,
    LEADERBOARD_SLOTS,
    information_ratio_stats,
    slot_spec,
)


def test_empty_series_returns_none():
    assert information_ratio_stats([]) is None


def test_single_date_is_insufficient_not_a_ratio():
    block = information_ratio_stats([0.01])
    assert block["information_ratio"] is None
    assert block["reason"] == IR_INSUFFICIENT
    assert block["n_dates"] == 1
    # The mean is still reported — the absence is of the RATIO, not the series.
    assert block["mean"] == pytest.approx(0.01)


def test_zero_dispersion_is_stated_not_infinite():
    """An arm whose active return never varied is unrateable, not infinitely
    good. A substituted or infinite value here would rank it first forever."""
    block = information_ratio_stats([0.02, 0.02, 0.02, 0.02])
    assert block["information_ratio"] is None
    assert block["reason"] == IR_ZERO_DISPERSION
    assert block["sd"] == pytest.approx(0.0)


def test_ratio_is_mean_over_sample_sd():
    vals = [0.01, 0.03, -0.01, 0.05]
    block = information_ratio_stats(vals)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    assert block["information_ratio"] == pytest.approx(mean / math.sqrt(var), abs=1e-4)
    assert block["n_dates"] == 4
    assert block["reason"] is None


def test_never_annualised():
    """Overlapping observations (weekly cohort, 21-session horizon) make an
    annualisation factor wrong by roughly sqrt(overlap). The block says so."""
    block = information_ratio_stats([0.01, 0.02, 0.03])
    assert block["annualised"] is False


def test_narrow_arm_does_not_win_on_depth_alone():
    """THE property the metric change exists for.

    Two arms, same ranking, different depth. The narrow arm's per-date alpha is
    uniformly higher because it stops earlier — a mean ranks it first on depth
    alone. Give it proportionally more dispersion (the concentration it took)
    and the information ratio declines to refuse that free win.
    """
    # Narrow arm: higher mean AND proportionally higher dispersion.
    narrow = [0.04, 0.00, 0.08, -0.02]
    # Broad arm: lower mean, much tighter.
    broad = [0.02, 0.01, 0.03, 0.005]

    mean_narrow = sum(narrow) / len(narrow)
    mean_broad = sum(broad) / len(broad)
    assert mean_narrow > mean_broad, "fixture must reproduce the depth advantage"

    ir_narrow = information_ratio_stats(narrow)["information_ratio"]
    ir_broad = information_ratio_stats(broad)["information_ratio"]
    assert ir_broad > ir_narrow, (
        "the information ratio must charge the narrow arm for the variance its "
        "concentration created; otherwise free widths select for concentration "
        "rather than skill (alpha-engine-config-I11393)"
    )


def test_research_slot_is_registered_with_per_arm_width_and_ir_primary():
    spec = slot_spec("research")
    assert spec.primary_metric == "information_ratio"
    assert spec.per_arm_width is True, (
        "the research slot carries arms at differing declared widths by design"
    )
    assert spec.slot_id == "research"


def test_research_slot_is_not_graded_against_spy_as_primary():
    """§4: a selection stage is graded against the population it drew from.
    The IR is computed over the population series, so the primary metric must
    not be the SPY-relative one."""
    spec = LEADERBOARD_SLOTS["research"]
    assert spec.primary_metric != "topn_alpha_vs_benchmark"
