"""Unit tests for the rolling-4-week-mean derived metric (PR 4b).

Covers:
- ``_list_metric_combos`` — pagination over ListMetrics + dimension
  preservation.
- ``_build_metric_data_queries`` — query shape correctness.
- ``compute_and_emit_4w_mean`` end-to-end with a stubbed CloudWatch
  client: the empty-corpus first-run case, the happy path, the
  partial-no-data case, and the result-mapping path back from
  GetMetricData Ids to dimensions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


def _dims(agent: str, criterion: str, judge: str = "claude-haiku-4-5") -> list[dict]:
    return [
        {"Name": "judged_agent_id", "Value": agent},
        {"Name": "criterion", "Value": criterion},
        {"Name": "judge_model", "Value": judge},
    ]


def _make_cw_with_combos(combos: list[list[dict]], values_by_idx: dict[int, list[float]]):
    """Build a MagicMock CloudWatch client backed by the given combos +
    GetMetricData values keyed by query Id index."""
    cw = MagicMock()

    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Metrics": [{"Dimensions": d} for d in combos]},
    ]
    cw.get_paginator.return_value = paginator

    cw.get_metric_data.return_value = {
        "MetricDataResults": [
            {"Id": f"m{idx}", "Values": values_by_idx.get(idx, [])}
            for idx in range(len(combos))
        ],
    }
    return cw


# ── _list_metric_combos ───────────────────────────────────────────────────


class TestListMetricCombos:
    def test_paginated_results_aggregate(self):
        from evals.rolling_mean import _list_metric_combos

        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Metrics": [{"Dimensions": _dims("a", "c1")}]},
            {"Metrics": [
                {"Dimensions": _dims("a", "c2")},
                {"Dimensions": _dims("b", "c1")},
            ]},
        ]
        cw.get_paginator.return_value = paginator

        out = _list_metric_combos(
            cw, namespace="AlphaEngine/Eval", metric_name="agent_quality_score",
        )
        assert len(out) == 3

    def test_drops_streams_with_no_dimensions(self):
        from evals.rolling_mean import _list_metric_combos

        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Metrics": [
                {"Dimensions": _dims("a", "c1")},
                {"Dimensions": []},  # the no-dim aggregate emission, if any
            ]},
        ]
        cw.get_paginator.return_value = paginator

        out = _list_metric_combos(
            cw, namespace="AlphaEngine/Eval", metric_name="agent_quality_score",
        )
        assert len(out) == 1


# ── _build_metric_data_queries ────────────────────────────────────────────


class TestBuildMetricDataQueries:
    def test_one_query_per_combo_with_indexed_id(self):
        from evals.rolling_mean import _build_metric_data_queries

        combos = [_dims("a", "c1"), _dims("a", "c2"), _dims("b", "c1")]
        queries = _build_metric_data_queries(
            combos,
            namespace="AlphaEngine/Eval",
            metric_name="agent_quality_score",
            period_seconds=2419200,
        )
        assert [q["Id"] for q in queries] == ["m0", "m1", "m2"]
        # Stat is Average; Period covers the full window so we get a
        # single mean datapoint per combo.
        for q in queries:
            assert q["MetricStat"]["Stat"] == "Average"
            assert q["MetricStat"]["Period"] == 2419200
            assert q["ReturnData"] is True


# ── _get_metric_data_all (chunk + paginate, AWS 500 cap) ──────────────────


class TestGetMetricDataAll:
    """The GetMetricData MetricDataQueries collection is hard-capped at
    500 by AWS. As the rubric × dimension matrix grew past 500 combos
    the single un-chunked call raised ``ValidationError: The collection
    MetricDataQueries must not have a size greater than 500.`` — which
    made the Sat-SF EvalRollingMean state ERROR on 2026-05-17. This
    helper chunks ≤500 + follows NextToken pagination."""

    def test_cap_constant_is_500(self):
        from evals.rolling_mean import _GET_METRIC_DATA_MAX_QUERIES

        assert _GET_METRIC_DATA_MAX_QUERIES == 500

    def test_over_500_queries_split_into_multiple_calls_each_le_500(self):
        from evals.rolling_mean import _get_metric_data_all

        # 1201 queries → 3 chunks: 500, 500, 201.
        queries = [{"Id": f"m{i}"} for i in range(1201)]
        cw = MagicMock()
        cw.get_metric_data.side_effect = lambda **kw: {
            "MetricDataResults": [
                {"Id": q["Id"], "Values": [1.0]}
                for q in kw["MetricDataQueries"]
            ],
        }

        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = datetime(2026, 5, 29, tzinfo=timezone.utc)
        out = _get_metric_data_all(cw, queries, start, end)

        # Three calls, every chunk ≤ 500.
        assert cw.get_metric_data.call_count == 3
        chunk_sizes = [
            len(c.kwargs["MetricDataQueries"])
            for c in cw.get_metric_data.call_args_list
        ]
        assert chunk_sizes == [500, 500, 201]
        assert all(n <= 500 for n in chunk_sizes)
        # Every query's result merged exactly once, order preserved.
        assert [r["Id"] for r in out] == [f"m{i}" for i in range(1201)]
        # StartTime/EndTime threaded through unchanged on every call.
        for c in cw.get_metric_data.call_args_list:
            assert c.kwargs["StartTime"] == start
            assert c.kwargs["EndTime"] == end
            assert "NextToken" not in c.kwargs

    def test_next_token_pagination_followed_within_chunk(self):
        from evals.rolling_mean import _get_metric_data_all

        queries = [{"Id": f"m{i}"} for i in range(3)]  # one chunk
        responses = [
            {"MetricDataResults": [{"Id": "m0", "Values": [1.0]}],
             "NextToken": "pg2"},
            {"MetricDataResults": [{"Id": "m1", "Values": [2.0]}],
             "NextToken": "pg3"},
            {"MetricDataResults": [{"Id": "m2", "Values": [3.0]}]},  # no token → stop
        ]
        cw = MagicMock()
        cw.get_metric_data.side_effect = responses

        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = datetime(2026, 5, 29, tzinfo=timezone.utc)
        out = _get_metric_data_all(cw, queries, start, end)

        assert cw.get_metric_data.call_count == 3
        # First call has no NextToken; subsequent calls carry the prior token.
        calls = cw.get_metric_data.call_args_list
        assert "NextToken" not in calls[0].kwargs
        assert calls[1].kwargs["NextToken"] == "pg2"
        assert calls[2].kwargs["NextToken"] == "pg3"
        assert [r["Id"] for r in out] == ["m0", "m1", "m2"]

    def test_le_500_single_call_unchanged_path(self):
        from evals.rolling_mean import _get_metric_data_all

        queries = [{"Id": f"m{i}"} for i in range(7)]
        cw = MagicMock()
        cw.get_metric_data.return_value = {
            "MetricDataResults": [{"Id": q["Id"], "Values": [4.0]} for q in queries],
        }

        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = datetime(2026, 5, 29, tzinfo=timezone.utc)
        out = _get_metric_data_all(cw, queries, start, end)

        # Exactly one call, no NextToken, all results returned.
        assert cw.get_metric_data.call_count == 1
        assert "NextToken" not in cw.get_metric_data.call_args.kwargs
        assert len(out) == 7

    def test_over_500_end_to_end_results_merged_back_to_combos(self):
        """compute_and_emit_4w_mean with >500 combos must split the
        GetMetricData calls AND still map every result back to its
        combo via the unique m{idx} Id scheme."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        n = 1100
        combos = [_dims(f"agent{i}", f"c{i}") for i in range(n)]
        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Metrics": [{"Dimensions": d} for d in combos]},
        ]
        cw.get_paginator.return_value = paginator
        # Echo back one result per query in the chunk.
        cw.get_metric_data.side_effect = lambda **kw: {
            "MetricDataResults": [
                {"Id": q["Id"], "Values": [3.5]}
                for q in kw["MetricDataQueries"]
            ],
        }
        s3 = MagicMock()

        result = compute_and_emit_4w_mean(cloudwatch_client=cw, s3_client=s3)

        # 1100 combos → 3 GetMetricData calls (500 + 500 + 100), each ≤ 500.
        assert cw.get_metric_data.call_count == 3
        for c in cw.get_metric_data.call_args_list:
            assert len(c.kwargs["MetricDataQueries"]) <= 500
        # All combos mapped back + emitted (no "missing result" failures).
        assert result["combos_discovered"] == n
        assert result["datapoints_emitted"] == n
        assert result["failed"] == []


# ── compute_and_emit_4w_mean ──────────────────────────────────────────────


class TestComputeAndEmit4wMean:
    def test_empty_corpus_first_run(self):
        from evals.rolling_mean import compute_and_emit_4w_mean

        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Metrics": []}]
        cw.get_paginator.return_value = paginator

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 0
        assert result["datapoints_emitted"] == 0
        cw.get_metric_data.assert_not_called()
        cw.put_metric_data.assert_not_called()
        # No combos → no floor emission. The alarm correctly stays in
        # INSUFFICIENT_DATA rather than firing on a None.
        assert result["floor_value"] is None
        assert result["floor_metric_emitted"] is False

    def test_happy_path_emits_one_per_combo(self):
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [
            _dims("ic_cio", "decision_coherence"),
            _dims("ic_cio", "rationale_quality"),
            _dims("macro_economist", "regime_grounding"),
        ]
        cw = _make_cw_with_combos(combos, values_by_idx={
            0: [4.2],  # mean over the 4w window
            1: [3.8],
            2: [4.5],
        })

        end = datetime(2026, 6, 6, 0, 0, tzinfo=timezone.utc)
        result = compute_and_emit_4w_mean(end_time=end, cloudwatch_client=cw)

        assert result["combos_discovered"] == 3
        assert result["datapoints_emitted"] == 3
        assert result["combos_skipped_no_data"] == 0
        assert result["failed"] == []

        # Two put_metric_data calls now: per-combo batch + floor.
        assert cw.put_metric_data.call_count == 2
        per_combo_call = cw.put_metric_data.call_args_list[0]
        floor_call = cw.put_metric_data.call_args_list[1]

        # Per-combo batch
        assert per_combo_call.kwargs["Namespace"] == "AlphaEngine/Eval"
        per_combo_metrics = per_combo_call.kwargs["MetricData"]
        assert all(
            d["MetricName"] == "agent_quality_score_4w_mean"
            for d in per_combo_metrics
        )
        values_by_agent = {
            d["Dimensions"][0]["Value"] + "/" + d["Dimensions"][1]["Value"]: d["Value"]
            for d in per_combo_metrics
        }
        assert values_by_agent["ic_cio/decision_coherence"] == 4.2
        assert values_by_agent["ic_cio/rationale_quality"] == 3.8
        assert values_by_agent["macro_economist/regime_grounding"] == 4.5

        # Floor metric: single dimensionless datapoint = MIN across combos.
        floor_metrics = floor_call.kwargs["MetricData"]
        assert len(floor_metrics) == 1
        assert floor_metrics[0]["MetricName"] == "agent_quality_score_4w_mean_min"
        assert "Dimensions" not in floor_metrics[0]
        assert floor_metrics[0]["Value"] == 3.8  # min of 4.2, 3.8, 4.5
        assert result["floor_value"] == 3.8
        assert result["floor_metric_emitted"] is True

    def test_skips_combos_with_no_data_in_window(self):
        """A combo that ListMetrics returns but GetMetricData has no
        Values for (e.g., agent stopped emitting; combo first appeared
        this week with no prior data) should be counted as skipped,
        not failed."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [
            _dims("ic_cio", "decision_coherence"),
            _dims("ic_cio", "deprecated_dim"),
        ]
        cw = _make_cw_with_combos(combos, values_by_idx={
            0: [4.0],
            1: [],  # no data
        })

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 2
        assert result["datapoints_emitted"] == 1
        assert result["combos_skipped_no_data"] == 1
        assert result["failed"] == []

    def test_window_is_28_days_ending_at_end_time(self):
        from evals.rolling_mean import compute_and_emit_4w_mean, ROLLING_WINDOW_DAYS

        combos = [_dims("a", "c1")]
        cw = _make_cw_with_combos(combos, values_by_idx={0: [4.0]})

        end = datetime(2026, 6, 6, 0, 0, tzinfo=timezone.utc)
        compute_and_emit_4w_mean(end_time=end, cloudwatch_client=cw)

        # Inspect the GetMetricData call's StartTime/EndTime.
        kwargs = cw.get_metric_data.call_args.kwargs
        assert kwargs["EndTime"] == end
        assert kwargs["StartTime"] == end - timedelta(days=ROLLING_WINDOW_DAYS)

    def test_dimension_shape_preserved_on_derived_emission(self):
        """The derived per-combo metric must carry the SAME dimension
        shape as the source so the dashboard's per-combo trend lines
        work without further translation. (The floor metric is
        intentionally dimensionless — see
        ``test_floor_metric_is_dimensionless``.)"""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("sector_quant:technology", "numerical_grounding", "claude-sonnet-4-6")]
        cw = _make_cw_with_combos(combos, values_by_idx={0: [4.5]})

        compute_and_emit_4w_mean(cloudwatch_client=cw)

        # Index 0 = per-combo emission; index 1 = floor.
        per_combo_call = cw.put_metric_data.call_args_list[0]
        emitted = per_combo_call.kwargs["MetricData"][0]
        emitted_dims = {d["Name"]: d["Value"] for d in emitted["Dimensions"]}
        assert emitted_dims == {
            "judged_agent_id": "sector_quant:technology",
            "criterion": "numerical_grounding",
            "judge_model": "claude-sonnet-4-6",
        }

    def test_floor_metric_is_dimensionless(self):
        """The floor metric carries NO dimensions — that's what lets a
        single CloudWatch alarm fire on it (alarms can't use SEARCH
        to reduce across dimensions). One alarm, one stream, one
        threshold."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [
            _dims("a", "c1"),
            _dims("a", "c2"),
        ]
        cw = _make_cw_with_combos(combos, values_by_idx={0: [4.5], 1: [3.2]})

        compute_and_emit_4w_mean(cloudwatch_client=cw)

        floor_call = cw.put_metric_data.call_args_list[1]
        floor_metric = floor_call.kwargs["MetricData"][0]
        assert floor_metric["MetricName"] == "agent_quality_score_4w_mean_min"
        assert "Dimensions" not in floor_metric
        # MIN across the two combo means.
        assert floor_metric["Value"] == 3.2

    def test_floor_emitted_only_when_at_least_one_combo_had_data(self):
        """All combos return empty Values → no floor emission even
        though combos were discovered. Alarm stays in
        INSUFFICIENT_DATA rather than getting a None datapoint."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("a", "c1"), _dims("a", "c2")]
        cw = _make_cw_with_combos(combos, values_by_idx={
            0: [],  # no data
            1: [],
        })

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 2
        assert result["datapoints_emitted"] == 0
        assert result["combos_skipped_no_data"] == 2
        assert result["floor_value"] is None
        assert result["floor_metric_emitted"] is False
        # Only the combo discovery happened — no put calls at all
        # because no per-combo data AND no floor to emit.
        cw.put_metric_data.assert_not_called()

    def test_missing_query_result_recorded_as_failure(self):
        """If GetMetricData drops a query result (shouldn't happen
        in practice, but defensive), record it in failed list."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("a", "c1"), _dims("a", "c2")]
        cw = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Metrics": [{"Dimensions": d} for d in combos]}]
        cw.get_paginator.return_value = paginator
        # Only one result returned (m0); m1 missing.
        cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "m0", "Values": [4.0]},
            ],
        }

        result = compute_and_emit_4w_mean(cloudwatch_client=cw)

        assert result["combos_discovered"] == 2
        assert result["datapoints_emitted"] == 1
        assert len(result["failed"]) == 1
        assert result["failed"][0]["combo_idx"] == "1"
        assert result["failed"][0]["stage"] == "get_metric_data"


# ── Regression auto-emit (ROADMAP P0 sub-item 5 eval half) ───────────────


@pytest.fixture
def reset_regression_threshold(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_EVAL_REGRESSION_THRESHOLD", raising=False)
    yield


class TestResolveRegressionThreshold:
    def test_default_is_3x(self, reset_regression_threshold):
        from evals.rolling_mean import _resolve_regression_threshold
        assert _resolve_regression_threshold() == 3.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_EVAL_REGRESSION_THRESHOLD", "2.5")
        from evals.rolling_mean import _resolve_regression_threshold
        assert _resolve_regression_threshold() == 2.5

    def test_env_zero_disables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_EVAL_REGRESSION_THRESHOLD", "0")
        from evals.rolling_mean import _resolve_regression_threshold
        assert _resolve_regression_threshold() == 0.0

    def test_env_unparseable_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_EVAL_REGRESSION_THRESHOLD", "garbage")
        from evals.rolling_mean import _resolve_regression_threshold
        assert _resolve_regression_threshold() == 3.0


class TestRegressionAutoEmit:
    """End-to-end auto-emit when a combo's 4-week mean falls below threshold."""

    def test_below_threshold_combo_writes_changelog_entry(
        self, reset_regression_threshold,
    ):
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("sector_quant", "calibration"), _dims("sector_qual", "depth")]
        # First combo's mean is 2.5 < 3.0 → regression. Second is 4.0 → ok.
        cw = _make_cw_with_combos(combos, {0: [2.5], 1: [4.0]})
        s3 = MagicMock()

        result = compute_and_emit_4w_mean(cloudwatch_client=cw, s3_client=s3)

        # Exactly one regression entry written (the first combo)
        assert s3.put_object.call_count == 1
        call = s3.put_object.call_args_list[0]
        assert call.kwargs["Bucket"] == "alpha-engine-research"
        assert call.kwargs["Key"].startswith("changelog/entries/")
        assert call.kwargs["Key"].endswith(".json")
        assert call.kwargs["ContentType"] == "application/json"

        # Result summary carries the regression-emit count + threshold
        assert len(result["regression_emits"]) == 1
        assert result["regression_threshold"] == 3.0

    def test_above_threshold_no_emit(self, reset_regression_threshold):
        """All combos above threshold → no auto-emit (quiet weeks stay quiet)."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("sector_quant", "c1"), _dims("sector_qual", "c2")]
        cw = _make_cw_with_combos(combos, {0: [4.0], 1: [4.5]})
        s3 = MagicMock()

        result = compute_and_emit_4w_mean(cloudwatch_client=cw, s3_client=s3)

        assert s3.put_object.call_count == 0
        assert result["regression_emits"] == []

    def test_disabled_threshold_no_emit(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_EVAL_REGRESSION_THRESHOLD", "0")
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("sector_quant", "c1")]
        cw = _make_cw_with_combos(combos, {0: [1.0]})  # would normally fire
        s3 = MagicMock()

        result = compute_and_emit_4w_mean(cloudwatch_client=cw, s3_client=s3)

        assert s3.put_object.call_count == 0
        assert result["regression_emits"] == []
        assert result["regression_threshold"] == 0.0

    def test_entry_payload_shape(self, reset_regression_threshold):
        """Auto-emitted entry carries every schema-1.0.0 field + the
        eval_regression diagnostic block."""
        import json as _json
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [_dims("sector_quant", "calibration", "claude-haiku-4-5")]
        cw = _make_cw_with_combos(combos, {0: [2.0]})
        s3 = MagicMock()

        compute_and_emit_4w_mean(cloudwatch_client=cw, s3_client=s3)

        body = _json.loads(s3.put_object.call_args.kwargs["Body"].decode())
        assert body["schema_version"] == "1.0.0"
        assert body["event_type"] == "eval_score_regression"
        assert body["severity"] == "medium"  # operational, not capital-at-risk
        assert body["subsystem"] == "eval"
        assert body["root_cause_category"] == "prompt_regression"
        assert body["source"] == "eval-regression-autoemit"
        assert body["actor"] == "alpha-engine-eval-rolling-mean"
        assert body["machine"] == "research:evals/rolling_mean.py"
        assert body["auto_emitted"] is True
        # eval_regression diagnostic block
        er = body["eval_regression"]
        assert er["judged_agent_id"] == "sector_quant"
        assert er["criterion"] == "calibration"
        assert er["judge_model"] == "claude-haiku-4-5"
        assert er["rolling_mean"] == 2.0
        assert er["threshold"] == 3.0
        # eval_run_ref points at the per-day decision_artifacts path
        assert "decision_artifacts/_eval/" in body["eval_run_ref"]
        assert "/sector_quant/" in body["eval_run_ref"]
        # event_id format mirrors SNS-mirror + cloudwatch-mirror scheme
        parts = body["event_id"].split("_")
        assert parts[1] == "alpha-engine-eval-rolling-mean"
        assert len(parts[2]) == 7

    def test_s3_write_failure_swallowed_other_combos_continue(
        self, reset_regression_threshold,
    ):
        """If put_object raises on combo N, combo N+1 still gets a write
        attempt + the floor metric still emits + the function returns."""
        from evals.rolling_mean import compute_and_emit_4w_mean

        combos = [
            _dims("a", "c1"),  # below threshold → tries to emit, fails
            _dims("a", "c2"),  # below threshold → tries to emit, succeeds
        ]
        cw = _make_cw_with_combos(combos, {0: [2.0], 1: [2.5]})
        s3 = MagicMock()

        # First put_object raises, second succeeds
        s3.put_object.side_effect = [
            Exception("AccessDenied first call"),
            {"ETag": '"deadbeef"'},
        ]

        # Should NOT raise — error is swallowed
        result = compute_and_emit_4w_mean(cloudwatch_client=cw, s3_client=s3)

        # Both put_object calls attempted
        assert s3.put_object.call_count == 2
        # Only the second succeeded → only one S3 key in regression_emits
        assert len(result["regression_emits"]) == 1
        # CloudWatch metric emission happened in spite of S3 failure
        # (combo data + floor were both put)
        assert cw.put_metric_data.call_count >= 2

    def test_event_id_idempotent_on_same_window_combo(
        self, reset_regression_threshold,
    ):
        """Re-running with the same window + combo identity produces the
        same event_id hash (overwrite, not duplicate)."""
        from evals.rolling_mean import _emit_regression_entry

        end = datetime(2026, 5, 9, 0, 0, 0, tzinfo=timezone.utc)
        start = end - timedelta(days=28)
        dims = _dims("sector_quant", "calibration")

        s3_a = MagicMock()
        s3_b = MagicMock()
        _emit_regression_entry(
            dims=dims, rolling_mean=2.0, threshold=3.0,
            window_start=start, window_end=end, s3_client=s3_a,
        )
        _emit_regression_entry(
            dims=dims, rolling_mean=2.0, threshold=3.0,
            window_start=start, window_end=end, s3_client=s3_b,
        )

        key_a = s3_a.put_object.call_args.kwargs["Key"]
        key_b = s3_b.put_object.call_args.kwargs["Key"]
        # Hash segment (last 7 hex before .json) identical
        hash_a = key_a.rsplit("_", 1)[-1].split(".")[0]
        hash_b = key_b.rsplit("_", 1)[-1].split(".")[0]
        assert hash_a == hash_b
