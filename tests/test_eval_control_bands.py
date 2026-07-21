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


def _make_cw(combos: list[list[dict]], series_by_idx: dict[int, list[float]]):
    """MagicMock CloudWatch backed by combos + per-combo weekly series.

    Synthesizes weekly Timestamps (descending, as CloudWatch returns
    them) so the control-band extractor's zip+sort path is exercised.
    """
    cw = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Metrics": [{"Dimensions": d} for d in combos]},
    ]
    cw.get_paginator.return_value = paginator

    base = datetime(2026, 6, 9, tzinfo=_UTC)
    results = []
    for idx in range(len(combos)):
        vals = series_by_idx.get(idx, [])
        # oldest-first values → descending timestamps (newest first) to
        # mimic CloudWatch default ScanBy.
        n = len(vals)
        ts = [base - timedelta(weeks=(n - 1 - i)) for i in range(n)]
        results.append({
            "Id": f"m{idx}",
            "Timestamps": list(reversed(ts)),
            "Values": list(reversed(vals)),
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
        r = cb.evaluate_series([4.5, 4.4, 4.6], min_history=8)
        assert r.status == cb.STATUS_INSUFFICIENT_HISTORY
        assert r.n_points == 3
        assert r.latest == 4.6
        assert r.latest_z is None

    def test_stable_series_in_control(self):
        series = [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 4.5]
        r = cb.evaluate_series(series, min_history=8)
        assert r.status == cb.STATUS_IN_CONTROL
        assert r.sigma > 0
        assert not r.shewhart_low
        assert not r.cusum_low

    def test_sudden_drop_is_shewhart_low(self):
        series = [4.5, 4.6, 4.4, 4.5, 4.6, 4.4, 4.5, 3.0]
        r = cb.evaluate_series(series, min_history=8)
        assert r.status == cb.STATUS_OUT_OF_CONTROL
        assert r.shewhart_low is True
        assert r.latest_z is not None and r.latest_z < 0
        assert any("shewhart_low" in reason for reason in r.reasons)

    def test_zero_variance_baseline_is_insufficient_variance(self):
        # config#2385 failure mode 1: a combo graded a constant baseline
        # has sigma 0 → no scale to judge a deviation. It must NOT flag
        # OUT_OF_CONTROL on any non-identical next score (the spurious
        # breach the alarm first fired on). Honest N/A instead; the
        # flat-floor alarm backstops absolute-low.
        series = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9]
        r = cb.evaluate_series(series, min_history=8)
        assert r.status == cb.STATUS_INSUFFICIENT_VARIANCE
        assert r.sigma == 0.0
        assert r.latest_z is None          # undefined at sigma 0, not a crash
        assert r.shewhart_low is False     # never breaches
        assert r.cusum_low is False
        assert any("insufficient_variance" in reason for reason in r.reasons)

    def test_zero_variance_baseline_large_drop_still_not_flagged(self):
        # Even a large drop from a flat baseline stays INSUFFICIENT_VARIANCE
        # — there is genuinely no variance scale, and the separate
        # flat-floor alarm (< 3.0) is the backstop for absolute-low.
        series = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0]
        r = cb.evaluate_series(series, min_history=8)
        assert r.status == cb.STATUS_INSUFFICIENT_VARIANCE
        assert r.status != cb.STATUS_OUT_OF_CONTROL

    def test_transient_early_baseline_dip_does_not_self_reference_breach(self):
        # config#2385 failure mode 2: a transient dip in the EARLY baseline
        # (which defines the center) must not latch cusum_low when the
        # combo's recent/latest scores are healthy — the exact
        # `cusum_low: C- 0.00` + `latest > UCL` contradiction the old
        # full-series CUSUM produced. The dip lives in the Phase-I
        # baseline; CUSUM only walks the Phase-II monitoring window.
        series = [4.5, 3.6, 4.5, 4.5, 4.5, 4.6, 4.6, 4.7, 4.8]
        r = cb.evaluate_series(series, min_history=8)
        assert r.status == cb.STATUS_IN_CONTROL
        assert r.cusum_low is False        # not re-tested against baseline
        assert r.shewhart_low is False     # latest is above center

    def test_sustained_downtrend_out_of_control(self):
        series = [4.6, 4.5, 4.5, 4.4, 4.0, 3.8, 3.6, 3.5, 3.4]
        r = cb.evaluate_series(series, min_history=8)
        assert r.status == cb.STATUS_OUT_OF_CONTROL
        assert r.cusum_low is True

    def test_upward_shift_not_alarmed(self):
        # Scores rising is observability, not a regression.
        series = [4.0, 4.0, 4.1, 4.0, 4.1, 4.0, 4.1, 4.9]
        r = cb.evaluate_series(series, min_history=8)
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
        reset = datetime(2026, 6, 9, tzinfo=_UTC) - timedelta(weeks=3, days=1)
        out = cb.compute_and_emit_control_bands(
            end_time=datetime(2026, 6, 9, tzinfo=_UTC),
            reset_before=reset,
            cloudwatch_client=cw, s3_client=s3,
        )
        assert out["combos_insufficient_history"] == 1
        assert out["breach_count"] == 0
