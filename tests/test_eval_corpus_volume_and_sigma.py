"""The judged corpus behind the control charts: volume, sigma_w, sigma_b.

Three defects, one corpus (measured 2026-09-08):

* **alpha-engine-config-I10169** — weekly judged volume swung from 2 to
  2408 reviews with whole weeks empty. Two proven causes, both here:
  the dedup index (`_eval_by_capture`) was written by
  `lambda/eval_judge_process_handler.py`, which was DELETED on 2026-08-29
  when the judge moved to the spot runner, and the replacement never
  carried the write over — so the seven judge runs of 2026-08-30/31
  graded 876 artifacts covering 142 distinct triples, 6.17x re-judging;
  and the judge's window was a fixed lookback that cannot reach a cycle
  that was skipped, which is how the weeks of 2026-08-12 and 2026-08-19
  came to hold nothing at all.
* **alpha-engine-config-I10186** — sigma_w was one global constant
  because CloudWatch has no `StandardDeviation` statistic. It is now
  measured per combo from a squared-score companion stream.
* **alpha-engine-config-I10188 (deliverable 2)** — sigma_b floored at
  exactly 0.000 for 7 of 20 combos fitted on three moving ranges. It is
  now shrunk toward a pooled prior with weight set by the ranges each
  combo actually has.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from evals import control_bands as cb
from evals import orchestrator as orch
from evals.rolling_mean import _build_metric_data_queries

BUCKET = "alpha-engine-research"


def _dims(agent: str, criterion: str, judge: str = "claude-haiku-4-5") -> list[dict]:
    return [
        {"Name": "judged_agent_id", "Value": agent},
        {"Name": "criterion", "Value": criterion},
        {"Name": "judge_model", "Value": judge},
    ]


# ══ I10169 — the judge window resumes from what was actually judged ══


class TestJudgeWindowCatchUp:
    def test_without_resume_from_the_window_is_the_fixed_lookback(self):
        """The default path is byte-identical to the old behaviour — the
        catch-up must not silently widen every ordinary run."""
        assert orch.compute_judge_window_dates("2026-09-05", 6) == sorted(
            set(
                [orch.effective_capture_ceiling_date("2026-09-05")]
                + orch.expand_lookback_dates(
                    orch.effective_capture_ceiling_date("2026-09-05"), 6,
                )
            ),
            reverse=True,
        )

    def test_resume_from_extends_the_window_back_to_the_last_judged_day(self):
        """The 2026-08-12/08-19 hole, in one assertion.

        Two weekly cycles produced no judged reviews. Under the fixed
        6-trading-day lookback the following Saturday's window reaches
        back roughly eight calendar days and those captures are simply
        lost. Resuming from the last INDEXED capture date recovers them.
        """
        fixed = orch.compute_judge_window_dates("2026-08-29", 6)
        caught_up = orch.compute_judge_window_dates(
            "2026-08-29", 6, resume_from="2026-08-11",
        )
        assert min(fixed) > "2026-08-12"          # the hole is unreachable
        assert min(caught_up) <= "2026-08-12"     # ...and now it is reached
        assert set(fixed) <= set(caught_up)       # nothing was dropped

    def test_catch_up_is_capped(self):
        """A long outage must not turn one run into an unbounded backfill."""
        window = orch.compute_judge_window_dates(
            "2026-09-05", 6, resume_from="2024-01-01", max_catchup_days=25,
        )
        assert len(window) <= 25

    def test_resume_from_newer_than_the_window_changes_nothing(self):
        fixed = orch.compute_judge_window_dates("2026-09-05", 6)
        assert (
            orch.compute_judge_window_dates(
                "2026-09-05", 6, resume_from="2026-09-04",
            )
            == fixed
        )


class TestLatestIndexedCaptureDate:
    def test_returns_the_newest_manifest_date(self):
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket=BUCKET)
            for d in ("2026-08-03", "2026-08-11", "2026-07-24"):
                s3.put_object(
                    Bucket=BUCKET,
                    Key=f"decision_artifacts/_eval_by_capture/{d}/manifest.json",
                    Body=json.dumps({"evals": []}),
                )
            assert (
                orch.latest_indexed_capture_date(s3, bucket=BUCKET)
                == "2026-08-11"
            )

    def test_empty_index_is_none_not_an_ancient_date(self):
        """None keeps the fixed lookback. Returning a sentinel epoch would
        make the FIRST run of a fresh corpus a full backfill."""
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket=BUCKET)
            assert orch.latest_indexed_capture_date(s3, bucket=BUCKET) is None


class TestSpotRunnerWritesTheDedupIndex:
    """`evals/judge_spot_run.py` must rebuild `_eval_by_capture`.

    The deleted `lambda/eval_judge_process_handler.py` did, with a comment
    naming config#4776 as the incident that put it at that point in the
    flow. Its replacement did not, and the very next judge runs re-graded
    one corpus 6.17 times.
    """

    def _run(self, *, manifest_raises: bool = False):
        import evals.judge_spot_run as jsr

        summary = {"persisted_keys": ["k"], "degraded_transport": None}
        built = {}

        def _fake_build(**kwargs):
            if manifest_raises:
                raise RuntimeError("s3 exploded")
            built.update(kwargs)
            return {"2026-09-05": {"eval_count": 3}}

        records = []
        with (
            patch("evals.orchestrator.process_batch_results", return_value=summary),
            patch("evals.eval_manifest.build_manifests", side_effect=_fake_build),
            patch("evals.judge_coverage.assess_coverage", return_value={
                "graded": 1, "planned": 1, "complete": True,
            }),
            patch("evals.judge_coverage.enforce_coverage", return_value=None),
            patch.object(
                jsr, "_persist_run_record",
                side_effect=lambda rec, **_kw: records.append(rec),
            ),
            patch.object(jsr, "_stage_coverage", return_value={}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            rc = jsr.main([
                "--date", "2026-09-05",
                "--bucket", BUCKET,
                "--plan-s3-key", "p", "--batch-id", "b",
            ])
        return rc, records[0], built

    def test_manifest_is_rebuilt_on_every_run(self):
        rc, record, built = self._run()
        assert rc == 0
        assert built["bucket"] == BUCKET
        assert record["manifest_capture_dates"] == ["2026-09-05"]
        assert record["manifest_error"] is None

    def test_manifest_failure_is_recorded_loudly_not_swallowed(self):
        """It must not fail the stage — the graded artifacts are already
        persisted — but it must be legible on the run record, because a
        missing index is what makes the NEXT run re-judge this corpus."""
        rc, record, _ = self._run(manifest_raises=True)
        assert rc == 0
        assert record["manifest_capture_dates"] == []
        assert "s3 exploded" in record["manifest_error"]


# ══ I10169 — the judged-corpus floor is a published, alarmable number ══


def _cw_for(combos, values, reviews, sumsq=None, base=None):
    """CloudWatch double: weekly buckets, newest closing at ``base``."""
    base = base or datetime(2026, 6, 9, tzinfo=UTC)
    cw = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Metrics": [{"Dimensions": d} for d in combos]},
    ]
    cw.get_paginator.return_value = paginator
    results = []
    for idx in range(len(combos)):
        vals = values.get(idx, [])
        n = len(vals)
        ts = [base - timedelta(weeks=(n - i)) for i in range(n)]
        results.append({"Id": f"m{idx}", "Timestamps": ts, "Values": vals})
        results.append({
            "Id": f"n{idx}", "Timestamps": ts,
            "Values": [float(x) for x in reviews.get(idx, [1000] * n)],
        })
        if sumsq and idx in sumsq:
            results.append({
                "Id": f"s{idx}", "Timestamps": ts,
                "Values": [float(x) for x in sumsq[idx]],
            })
    cw.get_metric_data.return_value = {"MetricDataResults": results}
    return cw


def _emitted(cw, name):
    for call in cw.put_metric_data.call_args_list:
        for d in call.kwargs["MetricData"]:
            if d.get("MetricName") == name:
                return d["Value"]
    return None


class TestJudgedCorpusFloorMetric:
    def test_a_week_nobody_judged_reads_the_whole_matrix(self):
        """The 2026-08-15 and 2026-08-22 shape: the SF SUCCEEDED, every
        rerun carried ``skip_eval_judge: true``, and no combo has a bucket
        for the newest complete week. That is 20 of 20, not silence."""
        base = datetime(2026, 6, 9, tzinfo=UTC)
        combos = [_dims("a", "c1"), _dims("a", "c2")]
        # Series ends TWO weeks before `base`, so the newest complete week
        # slot carries nothing for either combo.
        cw = _cw_for(
            combos,
            {0: [4.0] * 6, 1: [4.0] * 6},
            {0: [50] * 6, 1: [50] * 6},
            base=base - timedelta(weeks=2),
        )
        cb.compute_and_emit_control_bands(
            end_time=base, cloudwatch_client=cw, s3_client=MagicMock(),
        )
        assert _emitted(cw, cb.REVIEW_FLOOR_BREACH_METRIC_NAME) == 2.0

    def test_a_thin_combo_is_counted_and_a_healthy_one_is_not(self):
        base = datetime(2026, 6, 9, tzinfo=UTC)
        combos = [_dims("a", "thin"), _dims("a", "fat")]
        cw = _cw_for(
            combos,
            {0: [4.0] * 6, 1: [4.0] * 6},
            # combo 0's newest complete week rests on 2 reviews
            {0: [50, 50, 50, 50, 50, 2], 1: [50] * 6},
            base=base,
        )
        cb.compute_and_emit_control_bands(
            end_time=base, cloudwatch_client=cw, s3_client=MagicMock(),
        )
        assert _emitted(cw, cb.REVIEW_FLOOR_BREACH_METRIC_NAME) == 1.0

    def test_a_healthy_matrix_publishes_zero_not_nothing(self):
        base = datetime(2026, 6, 9, tzinfo=UTC)
        combos = [_dims("a", "c1")]
        cw = _cw_for(combos, {0: [4.0] * 6}, {0: [50] * 6}, base=base)
        cb.compute_and_emit_control_bands(
            end_time=base, cloudwatch_client=cw, s3_client=MagicMock(),
        )
        assert _emitted(cw, cb.REVIEW_FLOOR_BREACH_METRIC_NAME) == 0.0

    def test_combos_discovered_is_published(self):
        """The denominator (alpha-engine-config-I10187 deliverable 1) —
        an unmeasurable count of 4 means one thing against 20 combos and
        another against 200, and a fraction threshold is unexpressible
        without it."""
        base = datetime(2026, 6, 9, tzinfo=UTC)
        combos = [_dims("a", "c1"), _dims("a", "c2"), _dims("a", "c3")]
        cw = _cw_for(
            combos, {i: [4.0] * 6 for i in range(3)},
            {i: [50] * 6 for i in range(3)}, base=base,
        )
        cb.compute_and_emit_control_bands(
            end_time=base, cloudwatch_client=cw, s3_client=MagicMock(),
        )
        assert _emitted(cw, cb.COMBOS_DISCOVERED_METRIC_NAME) == 3.0


# ══ I10186 — sigma_w is measured, and the fallback is recorded ══


class TestWithinWeekSd:
    def test_recovers_a_known_sd_exactly(self):
        # Four reviews scoring 2,4,4,6: mean 4, SS = 8, s^2 = 8/3.
        obs = [cb.WeeklyObservation(
            week_index=1, value=4.0, reviews=4,
            sumsq=2 ** 2 + 4 ** 2 + 4 ** 2 + 6 ** 2,
        )]
        assert cb.within_week_sd(obs) == pytest.approx((8 / 3) ** 0.5)

    def test_pools_across_the_combos_weeks(self):
        obs = [
            cb.WeeklyObservation(1, 4.0, 4, sumsq=72.0),   # SS 8, df 3
            cb.WeeklyObservation(2, 4.0, 4, sumsq=72.0),   # SS 8, df 3
        ]
        assert cb.within_week_sd(obs) == pytest.approx((16 / 6) ** 0.5)

    def test_none_when_the_companion_stream_is_absent(self):
        obs = [cb.WeeklyObservation(1, 4.0, 50), cb.WeeklyObservation(2, 4.1, 50)]
        assert cb.within_week_sd(obs) is None

    def test_a_single_review_week_contributes_no_degrees_of_freedom(self):
        obs = [cb.WeeklyObservation(1, 4.0, 1, sumsq=16.0)]
        assert cb.within_week_sd(obs) is None

    def test_float_noise_cannot_produce_a_negative_variance(self):
        obs = [cb.WeeklyObservation(1, 4.0, 10, sumsq=10 * 4.0 ** 2 - 1e-12)]
        assert cb.within_week_sd(obs) == pytest.approx(0.0, abs=1e-6)


class TestResolveReviewSd:
    def test_measured_is_reported_as_measured(self):
        obs = [cb.WeeklyObservation(1, 4.0, 4, sumsq=72.0)]
        sd, source = cb.resolve_review_sd(obs)
        assert source == cb.REVIEW_SD_SOURCE_MEASURED
        assert sd == pytest.approx((8 / 3) ** 0.5)

    def test_absent_falls_back_to_the_global_constant_and_says_so(self):
        obs = [cb.WeeklyObservation(1, 4.0, 50)]
        sd, source = cb.resolve_review_sd(obs)
        assert (sd, source) == (
            cb.REVIEW_SCORE_SD, cb.REVIEW_SD_SOURCE_FALLBACK,
        )


class TestMeasuredSigmaWChangesTheLimits:
    """The issue's own closes-when: the fallback path and the measured
    path must produce DIFFERENT limits on the SAME series."""

    def _series(self, *, with_sumsq: bool):
        vals = [4.0, 4.2, 3.9, 4.1, 4.0, 4.3, 3.8, 4.0, 4.1, 3.95]
        out = []
        for i, v in enumerate(vals):
            n = 10
            # A very TIGHT judge: sigma_w ~ 0.2, five times below the 1.0
            # global constant.
            ss = n * v ** 2 + 0.04 * (n - 1) if with_sumsq else None
            out.append(cb.WeeklyObservation(i, v, n, sumsq=ss))
        return out

    def test_limits_differ(self):
        measured = cb.evaluate_series(self._series(with_sumsq=True))
        fallback = cb.evaluate_series(self._series(with_sumsq=False))
        assert measured.review_sd_source == cb.REVIEW_SD_SOURCE_MEASURED
        assert fallback.review_sd_source == cb.REVIEW_SD_SOURCE_FALLBACK
        assert measured.review_sd == pytest.approx(0.2, abs=0.01)
        assert fallback.review_sd == cb.REVIEW_SCORE_SD
        # The tighter judge earns tighter limits on the same scores.
        assert measured.sigma < fallback.sigma
        assert measured.lcl > fallback.lcl

    def test_the_source_reaches_the_breach_reason(self):
        """A breach must be attributable to a KNOWN sigma."""
        series = [
            cb.WeeklyObservation(i, v, 200, sumsq=200 * v ** 2 + 0.01 * 199)
            for i, v in enumerate(
                [4.0, 4.02, 3.98, 4.01, 4.0, 3.99, 4.01, 4.0, 3.0]
            )
        ]
        result = cb.evaluate_series(series)
        assert result.status == cb.STATUS_OUT_OF_CONTROL
        assert any("sigma_provenance" in r for r in result.reasons)
        assert any(cb.REVIEW_SD_SOURCE_MEASURED in r for r in result.reasons)


class TestSumsqQueryPlumbing:
    def test_third_query_is_opt_in(self):
        combos = [_dims("a", "c1")]
        without = _build_metric_data_queries(
            combos, namespace="N", metric_name="m", period_seconds=604800,
        )
        with_ss = _build_metric_data_queries(
            combos, namespace="N", metric_name="m", period_seconds=604800,
            sumsq_metric_name="m_sumsq",
        )
        assert [q["Id"] for q in without] == ["m0", "n0"]
        assert [q["Id"] for q in with_ss] == ["m0", "n0", "s0"]
        assert with_ss[2]["MetricStat"]["Stat"] == "Sum"
        assert with_ss[2]["MetricStat"]["Metric"]["MetricName"] == "m_sumsq"

    def test_the_live_matrix_stays_under_the_getmetricdata_cap(self):
        """20 combos x 3 queries = 60. The cap binds at 166 combos — the
        constraint I10186 names, asserted rather than asserted-in-prose."""
        combos = [_dims(f"a{i}", "c") for i in range(20)]
        queries = _build_metric_data_queries(
            combos, namespace="N", metric_name="m", period_seconds=604800,
            sumsq_metric_name="m_sumsq",
        )
        assert len(queries) == 60
        assert len(queries) <= 500

    def test_the_extractor_reads_the_sumsq_result(self):
        combos = [_dims("a", "c1")]
        base = datetime(2026, 6, 9, tzinfo=UTC)
        ts = [base - timedelta(weeks=2), base - timedelta(weeks=1)]
        results = [
            {"Id": "m0", "Timestamps": ts, "Values": [4.0, 4.0]},
            {"Id": "n0", "Timestamps": ts, "Values": [10.0, 10.0]},
            {"Id": "s0", "Timestamps": ts, "Values": [161.0, 162.0]},
        ]
        series = cb._weekly_series_by_combo(results, combos, end_time=base)
        assert [o.sumsq for o in series[0]] == [161.0, 162.0]

    def test_a_week_with_no_sumsq_bucket_is_none_not_zero(self):
        """Zero would say 'every review scored 0', which is a claim; None
        says 'not measured', which is the truth for every week judged
        before the companion stream existed."""
        combos = [_dims("a", "c1")]
        base = datetime(2026, 6, 9, tzinfo=UTC)
        ts = [base - timedelta(weeks=2), base - timedelta(weeks=1)]
        results = [
            {"Id": "m0", "Timestamps": ts, "Values": [4.0, 4.0]},
            {"Id": "n0", "Timestamps": ts, "Values": [10.0, 10.0]},
            {"Id": "s0", "Timestamps": ts[1:], "Values": [162.0]},
        ]
        series = cb._weekly_series_by_combo(results, combos, end_time=base)
        assert [o.sumsq for o in series[0]] == [None, 162.0]


# ══ I10188 d2 — sigma_b is shrunk toward a pooled prior, never floored ══


class TestPooledProcessSigma:
    def test_pools_in_variance_space_weighted_by_ranges(self):
        pooled = cb.pooled_process_sigma([(0.2, 3), (0.4, 9)])
        expected = ((3 * 0.04 + 9 * 0.16) / 12) ** 0.5
        assert pooled == pytest.approx(expected)

    def test_none_when_nothing_has_a_range(self):
        assert cb.pooled_process_sigma([(0.3, 0)]) is None
        assert cb.pooled_process_sigma([]) is None


class TestShrinkProcessSigma:
    def test_three_ranges_is_half_own_half_pooled_in_variance_space(self):
        out = cb.shrink_process_sigma(
            0.0, n_ranges=3, pooled=0.2, prior_ranges=3,
        )
        assert out == pytest.approx((0.5 * 0.04) ** 0.5)

    def test_a_long_series_is_barely_shrunk(self):
        out = cb.shrink_process_sigma(
            0.40, n_ranges=20, pooled=0.10, prior_ranges=3,
        )
        w = 20 / 23
        assert out == pytest.approx((w * 0.16 + (1 - w) * 0.01) ** 0.5)
        assert out > 0.37     # ~87% of its own estimate survives

    def test_no_pool_leaves_the_estimate_alone(self):
        assert cb.shrink_process_sigma(0.31, n_ranges=5, pooled=None) == 0.31

    def test_shrinkage_is_monotone_in_the_range_count(self):
        vals = [
            cb.shrink_process_sigma(0.0, n_ranges=m, pooled=0.2)
            for m in (3, 6, 12, 24)
        ]
        assert vals == sorted(vals, reverse=True)


class TestNoComboFloorsAtZeroOnThreeRanges:
    """The required assertion (alpha-engine-config-I10188 d2).

    Seven of twenty combos reported sigma_b of EXACTLY 0.000 on 2026-09-08,
    every one of them fitted on three moving ranges. Zero is a legitimate
    answer from a long series; from three ranges it is an estimate with no
    degrees of freedom, and it hands the combo limits set entirely by the
    sampling term.
    """

    def _flat_combo(self):
        # A perfectly flat baseline — the case that produced 0.000.
        return [
            cb.WeeklyObservation(i, 4.0, 40)
            for i in range(9)
        ]

    def test_unpooled_this_combo_really_does_report_zero(self):
        result = cb.evaluate_series(self._flat_combo(), sigma_b_pooled=None)
        assert result.sigma_b_raw == 0.0
        assert result.sigma_b == 0.0          # the defect, reproduced

    def test_pooled_it_never_does(self):
        result = cb.evaluate_series(self._flat_combo(), sigma_b_pooled=0.17)
        assert result.sigma_b_raw == 0.0      # its own estimate is unchanged
        assert result.sigma_b > 0.0           # what the chart USES is not
        # 9 weekly points, a 4-week monitoring window → a 5-point
        # baseline → 4 moving ranges, the thinnest the gate admits.
        assert result.moving_ranges == 4

    def test_pooling_widens_the_limits_on_a_thin_combo(self):
        """Reported, not tuned around: this is the direction that costs
        sensitivity on exactly the combos hardest to judge."""
        unpooled = cb.evaluate_series(self._flat_combo(), sigma_b_pooled=None)
        pooled = cb.evaluate_series(self._flat_combo(), sigma_b_pooled=0.17)
        assert pooled.lcl < unpooled.lcl
        assert pooled.sigma > unpooled.sigma

    def test_pooling_is_measured_but_not_applied(self):
        """The stop signal, asserted (alpha-engine-config-I10188 d2).

        Applying the pool retires
        `thinktank_theme/grounding_in_inputs/claude-haiku-4-5` — the only
        downward breach this corpus carries — measured 2026-09-08. Losing
        sensitivity to a real regression is a stop signal, not a knob, so
        the estimator ships measured-and-published and switched OFF."""
        assert cb.POOLED_SIGMA_B_ENABLED is False

    def test_the_orchestrator_pools_across_combos_and_reports_it(self):
        base = datetime(2026, 6, 9, tzinfo=UTC)
        combos = [_dims("a", "flat"), _dims("a", "moving")]
        cw = _cw_for(
            combos,
            {
                0: [4.0] * 9,
                1: [3.6, 4.2, 3.7, 4.3, 3.65, 4.25, 3.7, 4.2, 3.8],
            },
            {0: [40] * 9, 1: [40] * 9},
            base=base,
        )
        summary = cb.compute_and_emit_control_bands(
            end_time=base, cloudwatch_client=cw, s3_client=MagicMock(),
        )
        assert summary["sigma_b_pooled"] is not None
        assert summary["sigma_b_pooled"] > 0.0
        # Measured and published, not applied.
        assert summary["sigma_b_pooled_applied"] is False
