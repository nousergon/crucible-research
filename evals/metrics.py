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

SUMSQ_METRIC_NAME = "agent_quality_score_sumsq"
"""Companion stream carrying the SQUARE of each judged review's score,
under the SAME three dimensions and the SAME timestamp as
``agent_quality_score``.

**Why it exists (alpha-engine-config-I10186).** ``evals/control_bands.py``
models a weekly observation as ``Var(x_i) = sigma_b^2 + sigma_w^2/n_i``,
where sigma_w is the SD of a single judged review on the 1-5 rubric. It
sets every control limit and the whole ``MIN_REVIEWS_PER_WEEK``
derivation, and it was ONE GLOBAL CONSTANT (``REVIEW_SCORE_SD = 1.0``)
standing in for a quantity that plainly varies by agent, criterion and
judge — measured 0.889 (OLS) and 1.07 (two-group moment split) over 132
adjacent weekly pairs, but POOLED across all 20 combos.

It was pooled for a mechanical reason: **CloudWatch has no
``StandardDeviation`` statistic.** ``GetMetricData`` offers Average, Sum,
Minimum, Maximum, SampleCount and percentiles, so the within-week spread
cannot be read back from ``agent_quality_score`` at all.

Emitting the squares as their own stream removes the constraint without
working around it, because the three statistics CloudWatch DOES offer are
jointly sufficient::

    sigma_w^2 = ( Sum(sumsq) - n * Average(score)^2 ) / (n - 1)
    n         = SampleCount(score)

**Chosen over the five-bin rubric histogram**, the other shape the issue
offers, on three counts: one extra metric name against five; one extra
``GetMetricData`` query per combo (3 x 20 = 60, against the 500-query cap
``_get_metric_data_all`` chunks at) against five; and a bin count is not
more informative for a variance estimate — it is more informative about
the judge's score DISTRIBUTION, which is a different question and belongs
with I7038 rather than here. The histogram stays the right answer there.

Cost: one extra datapoint per rubric dimension per judged review, on a
stream nothing alarms on. ``put_metric_data`` caps at 1000 entries per
call; a rubric has 4-5 dimensions, so doubling stays three orders of
magnitude clear of it."""


def emit_eval_metric(
    eval_artifact: RubricEvalArtifact,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    metric_name: str = DEFAULT_METRIC_NAME,
    sumsq_metric_name: str = SUMSQ_METRIC_NAME,
    cloudwatch_client: Any | None = None,
) -> int:
    """Emit two CloudWatch datapoints per rubric dimension — the score and
    its square — and return the number of SCORE datapoints (i.e.
    ``len(eval_artifact.dimension_scores)``).

    The return value deliberately counts scores, not datapoints: callers
    (``evals/orchestrator.py``) accumulate it as "reviews emitted", and the
    ``sumsq`` companion (``alpha-engine-config-I10186``) is a statistic
    about those reviews, not more of them.

    ``Timestamp`` on each datapoint is taken from the eval artifact's
    own stamped time so a delayed metric write still lands on the
    correct evaluation date.
    """
    cw = cloudwatch_client or boto3.client("cloudwatch")

    artifact_ts = datetime.fromisoformat(
        eval_artifact.timestamp.replace("Z", "+00:00")
    )

    def dimensions_for(dim: Any) -> list[dict[str, str]]:
        """One dimension shape, used by both streams — they MUST match or
        the paired GetMetricData query in control_bands returns nothing
        for the sumsq half and sigma_w silently falls back."""
        return [
            {"Name": "judged_agent_id", "Value": eval_artifact.judged_agent_id},
            {"Name": "criterion", "Value": dim.dimension},
            {"Name": "judge_model", "Value": eval_artifact.judge_model},
        ]

    metric_data = [
        {
            "MetricName": metric_name,
            "Dimensions": dimensions_for(dim),
            "Value": float(dim.score),
            "Unit": "None",
            "Timestamp": artifact_ts,
        }
        for dim in eval_artifact.dimension_scores
    ]

    # The sufficient statistic for the within-week review SD
    # (alpha-engine-config-I10186) — same dimensions, same timestamp, so a
    # weekly bucket of `Sum(sumsq)` pairs exactly with the score bucket's
    # `Average` and `SampleCount`.
    metric_data.extend(
        {
            "MetricName": sumsq_metric_name,
            "Dimensions": dimensions_for(dim),
            "Value": float(dim.score) ** 2,
            "Unit": "None",
            "Timestamp": artifact_ts,
        }
        for dim in eval_artifact.dimension_scores
    )

    if not metric_data:
        return 0

    # CloudWatch put_metric_data caps at 1000 entries per call. A
    # single rubric has 4-5 dimensions so we never approach that
    # ceiling — but if a future rubric is much larger this would need
    # batching.
    cw.put_metric_data(Namespace=namespace, MetricData=metric_data)
    logger.info(
        "[eval_metrics] emitted %d datapoints (%d scores + %d sumsq) "
        "namespace=%s agent_id=%s judge=%s",
        len(metric_data), len(eval_artifact.dimension_scores),
        len(eval_artifact.dimension_scores), namespace,
        eval_artifact.judged_agent_id, eval_artifact.judge_model,
    )
    return len(eval_artifact.dimension_scores)
