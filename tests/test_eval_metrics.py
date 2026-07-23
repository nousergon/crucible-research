"""Unit tests for ``evals.metrics.emit_eval_metric`` (PR 4a)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from graph.state_schemas import (
    RubricDimensionScore,
    RubricEvalArtifact,
)


def _make_eval(
    *,
    judged_agent_id: str = "ic_cio",
    judge_model: str = "claude-haiku-4-5",
    rubric_id: str = "eval_rubric_ic_cio",
    timestamp: str = "2026-05-09T22:30:00.000Z",
    scores: list[tuple[str, int]] | None = None,
) -> RubricEvalArtifact:
    scores = scores or [
        ("decision_coherence", 4),
        ("rationale_quality", 3),
    ]
    return RubricEvalArtifact(
        run_id="run-1",
        judge_run_id="test-batch-uuid",
        timestamp=timestamp,
        judged_agent_id=judged_agent_id,
        rubric_id=rubric_id,
        rubric_version="1.0.0",
        judge_model=judge_model,
        dimension_scores=[
            RubricDimensionScore(dimension=d, score=s, reasoning="r")
            for d, s in scores
        ],
        overall_reasoning="ok",
    )


class TestEmitEvalMetric:
    def test_emits_one_datapoint_per_dimension(self):
        from evals.metrics import emit_eval_metric

        cw = MagicMock()
        eval_ = _make_eval(scores=[
            ("decision_coherence", 4),
            ("rationale_quality", 3),
            ("macro_integration", 5),
        ])
        sent = emit_eval_metric(eval_, cloudwatch_client=cw)

        assert sent == 3
        cw.put_metric_data.assert_called_once()
        kwargs = cw.put_metric_data.call_args.kwargs
        assert kwargs["Namespace"] == "AlphaEngine/Eval"
        assert len(kwargs["MetricData"]) == 3
        assert all(d["MetricName"] == "agent_quality_score" for d in kwargs["MetricData"])

    def test_dimensions_carry_agent_criterion_judge_model(self):
        from evals.metrics import emit_eval_metric

        cw = MagicMock()
        eval_ = _make_eval(
            judged_agent_id="sector_quant:technology",
            judge_model="claude-sonnet-4-6",
            scores=[("numerical_grounding", 5)],
        )
        emit_eval_metric(eval_, cloudwatch_client=cw)

        datapoint = cw.put_metric_data.call_args.kwargs["MetricData"][0]
        # Dimensions are a list of {Name, Value} dicts; build a lookup
        # so order isn't load-bearing.
        dims = {d["Name"]: d["Value"] for d in datapoint["Dimensions"]}
        assert dims["judged_agent_id"] == "sector_quant:technology"
        assert dims["criterion"] == "numerical_grounding"
        assert dims["judge_model"] == "claude-sonnet-4-6"

    def test_value_is_score_as_float(self):
        from evals.metrics import emit_eval_metric

        cw = MagicMock()
        eval_ = _make_eval(scores=[("d1", 2)])
        emit_eval_metric(eval_, cloudwatch_client=cw)
        datapoint = cw.put_metric_data.call_args.kwargs["MetricData"][0]
        assert datapoint["Value"] == 2.0
        assert isinstance(datapoint["Value"], float)

    def test_timestamp_is_parsed_from_artifact(self):
        from evals.metrics import emit_eval_metric

        cw = MagicMock()
        eval_ = _make_eval(timestamp="2026-04-25T03:14:00.000Z")
        emit_eval_metric(eval_, cloudwatch_client=cw)
        ts = cw.put_metric_data.call_args.kwargs["MetricData"][0]["Timestamp"]
        # Datetime, with the tz set so CloudWatch interprets it correctly.
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None
        assert ts == datetime(2026, 4, 25, 3, 14, tzinfo=UTC)

    def test_namespace_and_metric_name_overridable(self):
        from evals.metrics import emit_eval_metric

        cw = MagicMock()
        eval_ = _make_eval(scores=[("d1", 4)])
        emit_eval_metric(
            eval_,
            namespace="Test/Custom",
            metric_name="custom_metric",
            cloudwatch_client=cw,
        )
        kwargs = cw.put_metric_data.call_args.kwargs
        assert kwargs["Namespace"] == "Test/Custom"
        assert kwargs["MetricData"][0]["MetricName"] == "custom_metric"

    def test_empty_dimension_scores_skips_call(self):
        # Defensive: pydantic schema enforces min_length=1 on
        # dimension_scores, but if a future loosening allows zero we
        # should NOT make a put_metric_data call with empty MetricData
        # (CloudWatch rejects that with a validation error).
        from evals.metrics import emit_eval_metric

        cw = MagicMock()
        eval_ = _make_eval(scores=[("d1", 4)])
        # Manually empty — bypasses the schema validator.
        eval_.dimension_scores = []
        sent = emit_eval_metric(eval_, cloudwatch_client=cw)
        assert sent == 0
        cw.put_metric_data.assert_not_called()
