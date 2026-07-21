"""CloudWatch metric emission for LLM-as-judge eval results (PR 4a).

Each persisted ``RubricEvalArtifact`` produces one CloudWatch
``AlphaEngine/Eval/agent_quality_score`` datapoint per rubric
dimension, dimensioned by ``judged_agent_id`` + ``criterion`` +
``judge_model``. The dashboard (PR 4b) and the rolling-4-week-mean
SNS alarm (PR 4b) read from this metric stream.

Metric emission is observability OF observability — a CloudWatch
hiccup must NOT cause the eval pipeline to alert. Callers wrap
``emit_eval_metric`` in try/except (``evals/orchestrator.py``) and
accumulate failures into the run summary so they're visible without
halting the run.

ROADMAP §1634:
  CloudWatch metric AlphaEngine/Eval/agent_quality_score (Dimensions:
  agent_id, criterion). Alarm threshold on rolling-4-week-mean < 3.0
  emits SNS.

We add ``judge_model`` as a third CloudWatch dimension so Haiku-tier
and Sonnet-tier scores are tracked as separate streams — useful for
spotting systematic Haiku/Sonnet disagreement (the calibration
question §1627 asks).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import boto3

from graph.state_schemas import RubricEvalArtifact

logger = logging.getLogger(__name__)


DEFAULT_NAMESPACE = "AlphaEngine/Eval"
DEFAULT_METRIC_NAME = "agent_quality_score"


def emit_eval_metric(
    eval_artifact: RubricEvalArtifact,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    metric_name: str = DEFAULT_METRIC_NAME,
    cloudwatch_client: Any | None = None,
) -> int:
    """Emit one CloudWatch datapoint per rubric dimension. Returns the
    number of datapoints sent (matches len(eval_artifact.dimension_scores)).

    ``Timestamp`` on each datapoint is taken from the eval artifact's
    own stamped time so a delayed metric write still lands on the
    correct evaluation date.
    """
    cw = cloudwatch_client or boto3.client("cloudwatch")

    artifact_ts = datetime.fromisoformat(
        eval_artifact.timestamp.replace("Z", "+00:00")
    )

    metric_data = [
        {
            "MetricName": metric_name,
            "Dimensions": [
                {"Name": "judged_agent_id", "Value": eval_artifact.judged_agent_id},
                {"Name": "criterion", "Value": dim.dimension},
                {"Name": "judge_model", "Value": eval_artifact.judge_model},
            ],
            "Value": float(dim.score),
            "Unit": "None",
            "Timestamp": artifact_ts,
        }
        for dim in eval_artifact.dimension_scores
    ]

    if not metric_data:
        return 0

    # CloudWatch put_metric_data caps at 1000 entries per call. A
    # single rubric has 4-5 dimensions so we never approach that
    # ceiling — but if a future rubric is much larger this would need
    # batching.
    cw.put_metric_data(Namespace=namespace, MetricData=metric_data)
    logger.info(
        "[eval_metrics] emitted %d datapoints namespace=%s "
        "agent_id=%s judge=%s",
        len(metric_data), namespace,
        eval_artifact.judged_agent_id, eval_artifact.judge_model,
    )
    return len(metric_data)
