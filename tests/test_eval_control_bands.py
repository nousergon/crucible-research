"""Tests for the judge-score control bands (L4578(e))."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from evals import control_bands as cb

_UTC = UTC


def _dims(agent: str, criterion: str, judge: str = "claude-haiku-4-5") -> list[dict]:
    return [
        {"Name": "judged_agent_id", "Value": agent},
        {"Name": "criterion", "Value": criterion},
        {"Name": "judge_model", "Value": judge},
    ]


def _make_cw(
    combos: list[list[dict]],
    series_by_idx: dict[int, list[float]],
    reviews: dict[int, list[int]] | None = None,
):
    """MagicMock CloudWatch backed by combos + per-combo weekly series.

    Synthesizes weekly Timestamps (descending, as CloudWatch returns
    them) so the control-band extractor's zip+sort path is exercised,
    and the paired ``n{idx}`` SampleCount result the raw-weekly chart
    reads its review counts from (alpha-engine-config-I10167). Review
    counts default well above ``MIN_REVIEWS_PER_WEEK`` so a test that
    says nothing about volume is not silently gated by it.
    """
    cw = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Metrics": [{"Dimensions": d} for d in combos]},
    ]
    cw.get_paginator.return_value = paginator

    # `base` is the instant the NEWEST weekly bucket closes, so a test
    # passing end_time=base sees every bucket as complete. A test that
    # wants the newest bucket still open passes an earlier end_time
    # (alpha-engine-config-I10165).
    base = datetime(2026, 6, 9, tzinfo=_UTC)
    results = []
    for idx in range(len(combos)):
        vals = series_by_idx.get(idx, [])
        # oldest-first values → descending timestamps (newest first) to
        # mimic CloudWatch default ScanBy.
        n = len(vals)
        ts = [base - timedelta(weeks=(n - i)) for i in range(n)]
        results.append({
            "Id": f"m{idx}",
            "Timestamps": list(reversed(ts)),
            "Values": list(reversed(vals)),
        })
        ns = (reviews or {}).get(idx) or [1000] * n
        results.append({
            "Id": f"n{idx}",
            "Timestamps": list(reversed(ts)),
            "Values": [float(x) for x in reversed(ns)],
        })
    cw.get_metric_data.return_value = {"MetricDataResults": results}
    return cw


# ── moving_range_sigma ────────────────────────────────────────────────────


class TestMovingRangeSigma:
    def test_known_series(self):
        # MRs of [1,2,3,4] = [1,1,1]; MR-bar=1; sigma=1/1.128.
        assert cb.moving_range_sigma([1, 2, 3, 4]) == pytest.approx(1 / 1.128)

    def test_constant_series_is_zero(self):
        assert cb.moving_range_sigma([5.0, 5.0, 5.0]) == 0.0

    def test_needs_two_points(self):
        with pytest.raises(ValueError):
            cb.moving_range_sigma([4.2])


# ── tabular_cusum ─────────────────────────────────────────────────────────


class TestTabularCusum:
    def test_in_control_series_does_not_breach(self):
        series = [4.5] * 15
        r = cb.tabular_cusum(series, target=4.5, sigma=0.3)
        assert r.breached_low is False
        assert r.breached_high is False

    def test_sustained_downward_drift_breaches_low_not_shewhart(self):
        # 1.5σ sustained downward shift — no single point is a 3σ outlier,
        # but CUSUM accumulates the drift. This is the case the flat floor
        # AND a Shewhart chart would miss.
        series = [4.5] * 5 + [4.05] * 10  # 4.05 = 4.5 - 1.5*0.3
        r = cb.tabular_cusum(series, target=4.5, sigma=0.3)
        assert r.breached_low is True
        # Each shifted point is only 1.5σ out — within Shewhart's 3σ band.
        assert abs(4.05 - 4.5) / 0.3 < cb.DEFAULT_K_SIGMA

    def test_sigma_must_be_positive(self):
        with pytest.raises(ValueError):
            cb.tabular_cusum([4.5, 4.5], target=4.5, sigma=0.0)


# ── evaluate_series ───────────────────────────────────────────────────────


class TestEvaluateSeries:
    def test_insufficient_history(self):
        r = cb.evaluate_series(_obs([4.5, 4.4, 4.6]), min_history=8)
        assert r.status == cb.STATUS_INSUFFICIENT_HISTORY
        assert r.n_points == 3
        assert r.latest == 4.6
        assert r.latest_z is None

    def test_stable_series_in_control(self):
        series = [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.5]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.status == cb.STATUS_IN_CONTROL
        assert r.sigma > 0
        assert not r.shewhart_low
        assert not r.cusum_low

    def test_sudden_drop_is_shewhart_low(self):
        series = [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 3.0]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.status == cb.STATUS_OUT_OF_CONTROL
        assert r.shewhart_low is True
        assert r.latest_z is not None and r.latest_z < 0
        assert any("shewhart_low" in reason for reason in r.reasons)

    def test_zero_process_variance_baseline_is_a_tiny_but_real_scale(self):
        # config#2385 failure mode 1 was: a combo graded a CONSTANT
        # baseline had sigma-hat 0, so any non-identical next score
        # looked like an infinite-sigma breach, and the module answered
        # with INSUFFICIENT_VARIANCE — an honest N/A, but a hole: a
        # genuine collapse from a flat baseline was unjudgeable.
        #
        # On the raw weekly scale (alpha-engine-config-I10167) that hole
        # closes without reopening the false positive. A week's mean has
        # a KNOWN sampling variance sigma_w^2/n even when the process
        # variance is zero, so there is always a scale, and it is the
        # right one. A 0.1 wobble on 400 reviews is still in control;
        # `test_a_flat_baseline_is_no_longer_a_variance_hole` covers the
        # collapse case.
        series = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9]
        r = cb.evaluate_series(_obs(series, reviews=400), min_history=8)
        assert r.sigma_b == 0.0
        assert r.sigma == pytest.approx(0.05)
        assert r.status == cb.STATUS_IN_CONTROL
        assert r.shewhart_low is False
        assert r.cusum_low is False

    def test_transient_early_baseline_dip_does_not_self_reference_breach(self):
        # config#2385 failure mode 2: a transient dip in the EARLY baseline
        # (which defines the center) must not latch cusum_low when the
        # combo's recent/latest scores are healthy — the exact
        # `cusum_low: C- 0.00` + `latest > UCL` contradiction the old
        # full-series CUSUM produced. The dip lives in the Phase-I
        # baseline; CUSUM only walks the Phase-II monitoring window.
        series = [4.5, 3.6, 4.5, 4.5, 4.5, 4.6, 4.6, 4.7, 4.8]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.status == cb.STATUS_IN_CONTROL
        assert r.cusum_low is False        # not re-tested against baseline
        assert r.shewhart_low is False     # latest is above center

    def test_sustained_downtrend_out_of_control(self):
        series = [4.6, 4.5, 4.5, 4.4, 4.0, 3.8, 3.6, 3.5, 3.4]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.status == cb.STATUS_OUT_OF_CONTROL
        assert r.cusum_low is True

    def test_upward_shift_not_alarmed(self):
        # Scores rising is observability, not a regression.
        series = [4.0, 4.0, 4.1, 4.0, 4.1, 4.0, 4.1, 4.9]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.status == cb.STATUS_IN_CONTROL
        assert r.shewhart_high is True
        assert r.shewhart_low is False


# ── compute_and_emit_control_bands ────────────────────────────────────────


class TestComputeAndEmit:
    def test_no_streams_returns_empty_summary(self):
        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Metrics": []}]
        cw.get_paginator.return_value = paginator
        s3 = MagicMock()

        out = cb.compute_and_emit_control_bands(
            cloudwatch_client=cw, s3_client=s3,
        )
        assert out["combos_discovered"] == 0
        assert out["breach_count"] == 0
        s3.put_object.assert_not_called()

    def test_breach_emits_metric_zscore_and_changelog(self):
        combos = [_dims("alpha", "c1"), _dims("beta", "c1")]
        series = {
            0: [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.6, 2.5],  # breach
            1: [4.5, 4.4, 4.6],                                  # insufficient
        }
        cw = _make_cw(combos, series)
        s3 = MagicMock()

        out = cb.compute_and_emit_control_bands(
            end_time=datetime(2026, 6, 9, tzinfo=_UTC),
            cloudwatch_client=cw, s3_client=s3,
        )

        assert out["combos_discovered"] == 2
        assert out["breach_count"] == 1
        assert out["combos_insufficient_history"] == 1
        assert out["zscores_emitted"] == 1  # only the breaching combo (sigma>0)

        # One changelog breach entry written.
        s3.put_object.assert_called_once()
        key = s3.put_object.call_args.kwargs["Key"]
        assert key.startswith("changelog/entries/")

        # The breach-count alarm surface was emitted with value 1.
        emitted = [
            md
            for call in cw.put_metric_data.call_args_list
            for md in call.kwargs["MetricData"]
        ]
        breach_metric = [
            m for m in emitted
            if m["MetricName"] == cb.BREACH_COUNT_METRIC_NAME
        ]
        assert len(breach_metric) == 1
        assert breach_metric[0]["Value"] == 1.0
        # And a per-combo z-score was emitted for the breaching combo.
        assert any(
            m["MetricName"] == cb.ZSCORE_METRIC_NAME for m in emitted
        )

    def test_breach_count_emitted_even_when_zero(self):
        # The alarm stream must stay alive (no INSUFFICIENT_DATA) when no
        # combo breaches.
        combos = [_dims("alpha", "c1")]
        series = {0: [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.5]}  # in control
        cw = _make_cw(combos, series)
        s3 = MagicMock()

        out = cb.compute_and_emit_control_bands(
            end_time=datetime(2026, 6, 9, tzinfo=_UTC),
            cloudwatch_client=cw, s3_client=s3,
        )
        assert out["breach_count"] == 0
        s3.put_object.assert_not_called()
        emitted = [
            md
            for call in cw.put_metric_data.call_args_list
            for md in call.kwargs["MetricData"]
        ]
        breach_metric = [
            m for m in emitted
            if m["MetricName"] == cb.BREACH_COUNT_METRIC_NAME
        ]
        assert len(breach_metric) == 1
        assert breach_metric[0]["Value"] == 0.0

    def test_reset_before_trims_pre_reanchor_points(self):
        combos = [_dims("alpha", "c1")]
        # 9 points; the older 5 are a different (lower) regime that a
        # judge re-anchor invalidated. reset_before drops them, leaving 4
        # post-reset points → insufficient history (correctly).
        series = {0: [2.0, 2.0, 2.0, 2.0, 2.0, 4.5, 4.5, 4.5, 4.5]}
        cw = _make_cw(combos, series)
        s3 = MagicMock()

        # base timestamp in _make_cw is 2026-06-09; weeks count back.
        # Reset to keep only the last 4 weekly points.
        reset = datetime(2026, 6, 9, tzinfo=_UTC) - timedelta(weeks=4, days=1)
        out = cb.compute_and_emit_control_bands(
            end_time=datetime(2026, 6, 9, tzinfo=_UTC),
            reset_before=reset,
            cloudwatch_client=cw, s3_client=s3,
        )
        assert out["combos_insufficient_history"] == 1
        assert out["breach_count"] == 0


# ── alpha-engine-config-I10165 ────────────────────────────────────────────
#
# Three defects, all measured against the live series on 2026-09-08, all
# of which made the alarm `alpha-engine-eval-control-breach` report a
# condition that was not true at the time it reported it.


class TestCusumResetAfterSignal:
    """A tabular CUSUM is reset once it signals (Montgomery §9.1.3).

    Without the reset the accumulator carries a past excursion forward
    indefinitely, so every later evaluation of the same window
    re-reports the same excursion as a live drift.
    """

    def test_accumulator_resets_on_signal(self):
        # A deep 3-point excursion signals, then one recovering point.
        # Without reset C- stays ~9-12; with reset it drains to a small
        # residue.
        series = [3.65, 3.67, 3.69, 4.00]
        no_reset = cb.tabular_cusum(
            series, target=3.913, sigma=0.0553, reset_after_signal=False,
        )
        with_reset = cb.tabular_cusum(
            series, target=3.913, sigma=0.0553, reset_after_signal=True,
        )
        assert no_reset.breached_low is True
        assert with_reset.breached_low is True
        assert no_reset.c_minus > 8.0
        assert with_reset.c_minus < no_reset.c_minus
        assert with_reset.c_minus < cb.DEFAULT_CUSUM_H

    def test_reset_is_the_default(self):
        series = [3.65, 3.67, 3.69, 4.00]
        assert (
            cb.tabular_cusum(series, target=3.913, sigma=0.0553).c_minus
            == cb.tabular_cusum(
                series, target=3.913, sigma=0.0553, reset_after_signal=True,
            ).c_minus
        )

    def test_upward_side_resets_too(self):
        series = [4.35, 4.33, 4.31, 4.00]
        no_reset = cb.tabular_cusum(
            series, target=4.0, sigma=0.05, reset_after_signal=False,
        )
        with_reset = cb.tabular_cusum(
            series, target=4.0, sigma=0.05, reset_after_signal=True,
        )
        assert no_reset.breached_high and with_reset.breached_high
        assert with_reset.c_plus < no_reset.c_plus


class TestCusumCurrencyGate:
    """A CUSUM signal the latest point has already reversed is not a
    CURRENT out-of-control state, and the metric it feeds counts combos
    that are *currently* out of control.

    Live case, 2026-09-05:
    ``thinktank_thesis/context_integration/claude-sonnet-4-6`` reported
    ``cusum_low: C- 7.86 > h 5.0 (sustained downward drift from center
    3.913)`` while its latest 4w-mean was 4.002 — ABOVE its own center,
    at z = +1.62, and its highest value in six months. The alarm had
    been latched on that report for 8.7 days.
    """

    def test_recovered_excursion_is_not_out_of_control(self):
        # Baseline ~3.92 (flat), then a deep dip, then full recovery.
        series = [3.933, 3.909, 3.936, 3.885, 3.645, 3.675, 3.687, 4.002]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.latest_z is not None and r.latest_z > 0, (
            "precondition: the latest point sits above the baseline center"
        )
        assert r.cusum_signal_low is True, (
            "the in-window excursion is still recorded for observability"
        )
        assert r.cusum_low is False
        assert r.shewhart_low is False
        assert r.status == cb.STATUS_IN_CONTROL
        assert any("recovered" in x for x in r.reasons)

    def test_ongoing_drift_still_alarms(self):
        # Same excursion, but the latest point is still below center:
        # detection must NOT be blunted.
        series = [3.933, 3.909, 3.936, 3.885, 3.645, 3.675, 3.687, 3.660]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.latest_z is not None and r.latest_z < 0
        assert r.cusum_low is True
        assert r.status == cb.STATUS_OUT_OF_CONTROL

    def test_shewhart_low_needs_no_currency_gate(self):
        # latest < LCL implies latest < center, so a Shewhart breach is
        # current by construction and alarms regardless of the CUSUM.
        series = [4.0, 4.05, 3.95, 4.0, 4.05, 3.95, 4.0, 2.0]
        r = cb.evaluate_series(_obs(series), min_history=8)
        assert r.shewhart_low is True
        assert r.status == cb.STATUS_OUT_OF_CONTROL


class TestOpenWeeklyBucketIsDropped:
    """The still-accumulating CloudWatch bucket is a partial average and
    is never charted.

    Live case, 2026-08-30: the 19:52Z run read the open 08-25 bucket for
    ``thinktank_thesis/moat_and_business_quality/claude-haiku-4-5`` as
    3.1807 and flagged ``shewhart_low`` against LCL 3.1843 — a margin of
    0.0036. The 23:26Z run the same evening no longer breached, and the
    bucket settled at 3.7346.
    """

    def _results(self, values, end_time):
        combos = [_dims("alpha", "c1")]
        cw = _make_cw(combos, values)
        return cb._weekly_series_by_combo(
            cw.get_metric_data.return_value["MetricDataResults"],
            combos,
            end_time=end_time,
        )

    def test_open_bucket_excluded(self):
        # _make_cw lays weekly buckets back from 2026-06-09. With
        # end_time only 2 days past the newest bucket start, that bucket
        # has not closed.
        vals = {0: [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.9]}
        got = self._results(vals, datetime(2026, 6, 6, tzinfo=_UTC))
        assert 9.9 not in [o.value for o in got[0]]
        assert len(got[0]) == 7

    def test_closed_bucket_retained(self):
        vals = {0: [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.9]}
        got = self._results(vals, datetime(2026, 6, 9, tzinfo=_UTC))
        assert got[0][-1].value == 9.9
        assert len(got[0]) == 8

    def test_partial_bucket_cannot_produce_a_breach(self):
        # A flat baseline plus one partial low bucket: charting it would
        # be a Shewhart breach; dropping it leaves an honest N/A.
        combos = [_dims("alpha", "c1")]
        series = {0: [4.0, 4.05, 3.95, 4.0, 4.05, 3.95, 4.0, 4.02, 2.0]}
        cw = _make_cw(combos, series)
        s3 = MagicMock()
        out = cb.compute_and_emit_control_bands(
            end_time=datetime(2026, 6, 6, tzinfo=_UTC),
            cloudwatch_client=cw, s3_client=s3,
        )
        assert out["breach_count"] == 0
        s3.put_object.assert_not_called()


# ── alpha-engine-config-I10167 — the RAW WEEKLY scale ─────────────────────
#
# Brian ruled 2026-09-08: chart the raw weekly `agent_quality_score`
# series, not the 4-week rolling mean. The option NOT taken was leaving
# the rolling-mean scale and accepting its sensitivity.
#
# Every test below was verified failing against the pre-ruling module.


def _obs(values, reviews=1000, start_week=0, skip=()):
    """Build a weekly-observation series, oldest first.

    ``reviews`` is an int (applied to every week) or a per-week list.
    ``skip`` names 0-based positions whose week slot is SKIPPED, i.e. the
    calendar week exists but carries no observation — a real gap.
    """
    out = []
    week = start_week
    for i, v in enumerate(values):
        if i in skip:
            week += 1
        n = reviews[i] if isinstance(reviews, list) else reviews
        out.append(cb.WeeklyObservation(week_index=week, value=v, reviews=n))
        week += 1
    return out


class TestWeeklyObservationScale:
    """The chart input is one week's mean and the n behind it."""

    def test_evaluate_series_takes_weekly_observations(self):
        r = cb.evaluate_series(_obs([4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.5]))
        assert r.status == cb.STATUS_IN_CONTROL
        assert r.reviews_latest == 1000

    def test_the_source_metric_is_the_raw_weekly_score(self):
        # Not `agent_quality_score_4w_mean`: consecutive 4-week means
        # share ~75% of their reviews, which deflates the moving-range
        # sigma 2-5x (measured 0.028-0.155 vs 0.173-0.441).
        assert cb.SOURCE_METRIC_NAME == "agent_quality_score"


class TestVarianceScalesWithReviewCount:
    """A 2-review week and a 2408-review week are not the same
    observation. Var(week mean) = sigma_b^2 + sigma_w^2 / n."""

    def test_the_same_drop_is_less_significant_on_fewer_reviews(self):
        base = [4.0, 4.1, 3.9, 4.0, 4.1, 3.9, 4.0]
        many = cb.evaluate_series(_obs(base + [3.6], reviews=[400] * 8))
        few = cb.evaluate_series(_obs(base + [3.6], reviews=[400] * 7 + [6]))
        assert abs(many.latest_z) > abs(few.latest_z)
        assert many.sigma < few.sigma

    def test_observation_sigma_never_falls_below_the_sampling_floor(self):
        s = cb.observation_sigma(sigma_b=0.0, reviews=25, review_sd=1.0)
        assert s == pytest.approx(0.2)

    def test_process_sigma_removes_the_sampling_component(self):
        # A baseline whose whole moving range is explained by sampling
        # noise has NO demonstrated process variation.
        obs = _obs([4.0, 3.7, 4.3, 3.8], reviews=[4, 4, 4, 4])
        sigma_b = cb.process_sigma(obs)
        assert sigma_b == 0.0

    def test_a_flat_baseline_is_no_longer_a_variance_hole(self):
        # Pre-ruling a constant baseline returned INSUFFICIENT_VARIANCE
        # (sigma 0, no scale). The sampling term supplies a real scale,
        # so the chart can now judge a drop from a flat baseline.
        obs = _obs([5.0] * 7 + [3.0], reviews=400)
        r = cb.evaluate_series(obs)
        assert r.status == cb.STATUS_OUT_OF_CONTROL
        assert r.sigma_b == 0.0
        assert r.sigma > 0


class TestMinimumReviewsAdmission:
    """A week below MIN_REVIEWS_PER_WEEK is not an observation of the
    process. It is recorded, never silently dropped."""

    def test_a_low_n_week_is_excluded_and_counted(self):
        obs = _obs(
            [4.0, 4.1, 3.9, 4.0, 4.1, 3.9, 4.0, 4.0, 4.1],
            reviews=[400, 400, 2, 400, 400, 400, 400, 400, 400],
        )
        r = cb.evaluate_series(obs)
        assert r.unmeasurable_weeks == 1
        assert r.n_points == 8          # admitted weeks only

    def test_a_low_n_week_can_never_produce_a_breach(self):
        # A 2-review week scoring the rubric minimum is sampling noise,
        # not a regression.
        obs = _obs(
            [4.0, 4.1, 3.9, 4.0, 4.1, 3.9, 4.0, 4.0, 1.0],
            reviews=[400] * 8 + [2],
        )
        r = cb.evaluate_series(obs)
        assert r.status != cb.STATUS_OUT_OF_CONTROL
        assert r.unmeasurable_weeks == 1

    def test_the_gate_is_derived_from_the_rubric_resolution(self):
        # sigma_w / sqrt(n) <= 0.5 rubric points at the MEASURED
        # sigma_w = 1.0 gives n >= 4.
        assert cb.MIN_REVIEWS_PER_WEEK == 4
        assert cb.REVIEW_SCORE_SD == 1.0

    def test_too_few_admitted_weeks_is_insufficient_history(self):
        obs = _obs(
            [4.0, 4.1, 3.9, 4.0, 4.1, 3.9, 4.0, 4.0],
            reviews=[2, 2, 2, 400, 400, 400, 400, 400],
        )
        r = cb.evaluate_series(obs)
        assert r.status == cb.STATUS_INSUFFICIENT_HISTORY
        assert r.unmeasurable_weeks == 3


class TestGapsAreNotClosedOver:
    """A missing week is an unobserved week, not a shorter interval."""

    def test_moving_range_skips_pairs_spanning_a_gap(self):
        # Same four values; in the second series a whole week is missing
        # between points 1 and 2, so that pair is not a moving range.
        contiguous = _obs([4.0, 4.2, 3.0, 3.2], reviews=100000)
        gapped = _obs([4.0, 4.2, 3.0, 3.2], reviews=100000, skip=(2,))
        assert cb.process_sigma(gapped) < cb.process_sigma(contiguous)

    def test_a_gap_does_not_reset_the_cusum(self):
        # Drift evidence is about the process, not about whether we
        # happened to look. A producer hiccup must not erase it.
        vals = [4.5, 4.5, 4.5, 4.5, 4.5, 3.9, 3.9, 3.9, 3.9]
        r = cb.evaluate_series(_obs(vals, reviews=400, skip=(7,)))
        assert r.cusum_low is True
        assert r.missing_weeks == 1

    def test_a_trailing_gap_is_stale_not_in_control(self):
        # The chart's newest observation is two complete weeks old: the
        # process has gone unmeasured and that is never reported green.
        obs = _obs([4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.5], reviews=400)
        r = cb.evaluate_series(obs, latest_complete_week=obs[-1].week_index + 2)
        assert r.status == cb.STATUS_STALE
        assert r.status != cb.STATUS_IN_CONTROL

    def test_one_week_of_slippage_is_within_grace(self):
        obs = _obs([4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.5], reviews=400)
        r = cb.evaluate_series(obs, latest_complete_week=obs[-1].week_index + 1)
        assert r.status == cb.STATUS_IN_CONTROL


class TestCenterIsNotReviewWeighted:
    """alpha-engine-config-I10169: a review-weighted centre tracks the
    volume mix. Each WEEK is one observation of the process."""

    def test_one_huge_week_does_not_dominate_the_centre(self):
        obs = _obs(
            [3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            reviews=[2408, 100, 100, 100, 100, 100, 100, 100],
        )
        r = cb.evaluate_series(obs)
        # Unweighted over the 4 baseline weeks: (3+4+4+4)/4 = 3.75.
        assert r.center == pytest.approx(3.75)


class TestUnmeasurableIsPublished:
    """principles.md 2.7 — a combo the chart cannot judge is counted on
    its own stream, never folded into the healthy zero."""

    def test_unmeasurable_count_emitted_every_run(self):
        combos = [_dims("alpha", "c1"), _dims("beta", "c1")]
        series = {
            0: [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.6],   # chartable
            1: [4.5, 4.4, 4.6],                             # insufficient
        }
        cw = _make_cw(combos, series)
        s3 = MagicMock()
        out = cb.compute_and_emit_control_bands(
            end_time=datetime(2026, 6, 9, tzinfo=_UTC),
            cloudwatch_client=cw, s3_client=s3,
        )
        emitted = [
            md for call in cw.put_metric_data.call_args_list
            for md in call.kwargs["MetricData"]
        ]
        unm = [
            m for m in emitted
            if m["MetricName"] == cb.UNMEASURABLE_COUNT_METRIC_NAME
        ]
        assert len(unm) == 1
        assert unm[0]["Value"] == 1.0
        assert out["combos_unmeasurable"] == 1

    def test_sample_counts_are_read_from_the_paired_query(self):
        combos = [_dims("alpha", "c1")]
        cw = _make_cw(combos, {0: [4.0, 4.1]}, reviews={0: [7, 9]})
        got = cb._weekly_series_by_combo(
            cw.get_metric_data.return_value["MetricDataResults"],
            combos,
            end_time=datetime(2026, 6, 9, tzinfo=_UTC),
        )
        assert [o.reviews for o in got[0]] == [7, 9]


class TestLiveSeries20260908:
    """Replayed against the live raw weekly series, measured 2026-09-08."""

    def test_the_false_positive_that_latched_the_alarm_stays_in_control(self):
        # thinktank_thesis/context_integration/claude-sonnet-4-6.
        vals = [3.93, 3.86, 3.33, 4.53, 3.58, 3.79, 4.46, 4.58, 4.53]
        ns = [15, 7, 6, 17, 277, 38, 13, 12, 72]
        r = cb.evaluate_series(_obs(vals, reviews=ns, skip=(7,)))
        assert r.status != cb.STATUS_OUT_OF_CONTROL

    def test_grounding_in_inputs_haiku_still_breaches(self):
        # thinktank_theme/grounding_in_inputs/claude-haiku-4-5 — a real
        # sustained decline carried by a 189-review week at z = -4.8.
        vals = [3.75, 3.75, 4.16, 4.18, 4.29, 4.00, 3.61, 3.33]
        ns = [12, 24, 234, 98, 59, 12, 189, 12]
        r = cb.evaluate_series(_obs(vals, reviews=ns, skip=(6,)))
        assert r.status == cb.STATUS_OUT_OF_CONTROL
        assert r.cusum_low is True

    def test_churn_discipline_haiku_is_no_longer_out_of_control(self):
        # Old scale: z = -3.06 (sigma-hat 0.155, a smoothing artifact).
        # Raw weekly: its own week-to-week sigma is 0.30 and the drop is
        # 0.6 -> under 2 sigma. The old answer was wrong.
        vals = [3.83, 3.54, 4.27, 4.13, 4.58, 4.67, 3.68, 3.33]
        ns = [12, 24, 234, 98, 59, 12, 189, 12]
        r = cb.evaluate_series(_obs(vals, reviews=ns, skip=(6,)))
        assert r.status == cb.STATUS_IN_CONTROL
        assert abs(r.latest_z) < 3.0
