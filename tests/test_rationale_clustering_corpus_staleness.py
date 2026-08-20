"""Instance 2 of the frozen-upstream bug class (alpha-engine-config-I2638).

The rationale-clustering rollup reads a TRAILING window of DecisionArtifact
captures. Measured 2026-08-15: the ``sector_quant`` captures had been frozen at
2026-07-11 since the multi-agent graph retired, and this module still wrote a
freshly-dated ``_analysis/sector_quant/2026-W33.json`` and stamped a CloudWatch
datapoint at that day's timestamp. A template-concentration figure computed
over a dead corpus is a claim about July presented as a claim about this week,
and nothing on the artifact distinguished the two.

It DEGRADES rather than raises: the stage spans every agent corpus at once, and
a retired agent's frozen captures must not fail the stage for live agents.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import boto3
from moto import mock_aws

from evals.rationale_clustering import (
    _capture_date_from_key,
    _newest_capture_date,
    compute_and_emit,
)

BUCKET = "alpha-engine-research"
END = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
FROZEN_DAY = datetime(2026, 7, 11, tzinfo=UTC)  # the measured freeze date


def _rationales(n: int) -> list[str]:
    return [
        f"P/E of {i} is below the sector median of {i + 6}; momentum confirms."
        for i in range(n)
    ]


def _agent_output(agent_id: str, i: int) -> dict:
    """Real capture shapes — ``sector_quant`` carries ``ranked_picks[*].rationale``,
    ``sector_qual`` carries ``assessments[*].bull_case`` (see
    ``extract_rationales``)."""
    text = _rationales(6)[i % 6]
    if agent_id == "sector_qual":
        return {"assessments": [{"ticker": f"T{i}", "bull_case": text}]}
    return {"ranked_picks": [{"ticker": f"T{i}", "rationale": text}]}


def _seed(s3, *, agent_id: str, day: datetime, n_artifacts: int = 6) -> None:
    for i in range(n_artifacts):
        key = (
            f"decision_artifacts/{day:%Y}/{day:%m}/{day:%d}/"
            f"{agent_id}/run{i}.json"
        )
        body = {"agent_id": agent_id, "agent_output": _agent_output(agent_id, i)}
        s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(body).encode())


def _run(s3, **kw):
    return compute_and_emit(
        end_time=END,
        bucket=BUCKET,
        s3_client=s3,
        cloudwatch_client=kw.pop("cw", MagicMock()),
        **kw,
    )


def test_capture_date_parsing_is_none_on_an_unexpected_layout():
    key = "decision_artifacts/2026/07/11/sector_quant/run0.json"
    assert _capture_date_from_key(key) == FROZEN_DAY
    assert _capture_date_from_key("decision_artifacts/nope.json") is None
    assert _capture_date_from_key("a/b/c/d/e/f.json") is None  # non-numeric
    assert _newest_capture_date([]) is None


@mock_aws
def test_frozen_corpus_still_writes_but_the_artifact_says_so(freshness_alerts):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    _seed(s3, agent_id="sector_quant", day=FROZEN_DAY)

    summary = _run(s3)

    # The rollup is still produced — a retired agent must not fail the stage.
    assert summary["agents_analyzed"] == 1
    row = summary["per_agent"][0]
    assert row["agent_id"] == "sector_quant"
    assert row["corpus_freshness"] == "stale"
    assert row["corpus_age_days"] > 30

    # The staleness rides on the PERSISTED artifact, not only the summary —
    # a downstream reader of the analysis JSON sees it without the SF payload.
    body = json.loads(
        s3.get_object(Bucket=BUCKET, Key=row["analysis_key"])["Body"].read()
    )
    assert body["degraded"] is True
    assert body["upstream_freshness"]["status"] == "stale"
    assert body["upstream_freshness"]["as_of"].startswith("2026-07-11")
    assert body["upstream_freshness"]["age_days"] > 30
    # ...next to the fresh-looking date that made this invisible.
    assert body["computed_at"]

    assert [r["status"] for r in summary["agents_stale_corpus"]] == ["stale"]
    assert len(freshness_alerts) == 1  # one per frozen producer, deduped by artifact


@mock_aws
def test_live_corpus_is_marked_fresh_and_does_not_alert(freshness_alerts):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    _seed(s3, agent_id="sector_qual", day=END - timedelta(days=2))

    summary = _run(s3)

    row = summary["per_agent"][0]
    assert row["corpus_freshness"] == "fresh"
    body = json.loads(
        s3.get_object(Bucket=BUCKET, Key=row["analysis_key"])["Body"].read()
    )
    # The fresh case is recorded too — silence would be indistinguishable from
    # "never checked".
    assert body["degraded"] is False
    assert body["upstream_freshness"]["status"] == "fresh"
    assert summary["agents_stale_corpus"] == []
    assert freshness_alerts == []


@mock_aws
def test_one_frozen_agent_does_not_degrade_a_live_one(freshness_alerts):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    _seed(s3, agent_id="sector_quant", day=FROZEN_DAY)
    _seed(s3, agent_id="sector_qual", day=END - timedelta(days=1))

    summary = _run(s3)

    by_agent = {r["agent_id"]: r for r in summary["per_agent"]}
    assert by_agent["sector_quant"]["corpus_freshness"] == "stale"
    assert by_agent["sector_qual"]["corpus_freshness"] == "fresh"
    # One notification per frozen producer, grouped on the causal key.
    assert len(freshness_alerts) == 1


@mock_aws
def test_empty_window_reports_undated_rather_than_a_clean_no_op(freshness_alerts):
    """Zero artifacts is the loudest form of a frozen upstream, and it used to
    render as a successful run with nothing to do."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)

    summary = _run(s3)

    assert summary["artifacts_discovered"] == 0
    assert summary["corpus_freshness"]["status"] == "undated"
    assert len(freshness_alerts) == 1


@mock_aws
def test_corpus_age_is_emitted_as_its_own_metric_beside_the_value_it_explains():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    _seed(s3, agent_id="sector_quant", day=FROZEN_DAY)
    cw = MagicMock()

    _run(s3, cw=cw)

    emitted = {
        datum["MetricName"]
        for call in cw.put_metric_data.call_args_list
        for datum in call.kwargs["MetricData"]
    }
    assert "agent_rationale_template_concentration" in emitted
    assert "agent_rationale_template_concentration_corpus_age_days" in emitted
    age_datum = next(
        datum
        for call in cw.put_metric_data.call_args_list
        for datum in call.kwargs["MetricData"]
        if datum["MetricName"].endswith("_corpus_age_days")
    )
    assert age_datum["Value"] > 30
    assert age_datum["Dimensions"] == [
        {"Name": "judged_agent_id", "Value": "sector_quant"}
    ]
