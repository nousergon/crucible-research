"""Unit tests for the upstream-freshness primitive (alpha-engine-config-I2638).

The primitive exists because presence checks are not freshness checks: a
consumer that reads a frozen artifact successfully produces fresh-dated output
from dead input, and nothing anywhere says so. These pin the decisions that
make that impossible to miss — fail-loud by default, an undated artifact is
never fresh, and a deliberate degradation must name the failure mode it
accepts.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import boto3
import pytest
from moto import mock_aws

from freshness import (
    CADENCE_TOLERANCE_DAYS,
    FreshnessVerdict,
    UpstreamStaleError,
    assert_upstream_fresh,
    s3_last_modified,
)

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def test_within_tolerance_is_fresh_and_never_alerts(freshness_alerts):
    verdict = assert_upstream_fresh(
        "archive/macro/macro_report.md",
        as_of=NOW - timedelta(days=6),
        cadence="weekly",
        checked_at=NOW,
    )
    assert verdict.is_fresh
    assert verdict.status == "fresh"
    assert verdict.age_days == pytest.approx(6.0)
    assert verdict.tolerance_days == CADENCE_TOLERANCE_DAYS["weekly"]
    assert freshness_alerts == []


def test_beyond_tolerance_raises_by_default(freshness_alerts):
    """Fail-loud is the DEFAULT — a consumer gets the raise unless it opts out
    in writing."""
    with pytest.raises(UpstreamStaleError) as exc:
        assert_upstream_fresh(
            "archive/macro/macro_report.md",
            as_of=datetime(2026, 3, 16, tzinfo=UTC),  # the real measured value
            cadence="weekly",
            checked_at=NOW,
        )
    verdict = exc.value.verdict
    assert verdict.status == "stale"
    assert verdict.age_days > 150
    # The staleness still reaches the alert channel on the raising path.
    assert len(freshness_alerts) == 1
    assert freshness_alerts[0]["severity"] == "error"


def test_undated_artifact_is_not_fresh(freshness_alerts):
    """No timestamp is UNOBSERVED, never healthy — no data is not green."""
    with pytest.raises(UpstreamStaleError) as exc:
        assert_upstream_fresh(
            "some/blob.md", as_of=None, cadence="weekly", checked_at=NOW
        )
    assert exc.value.verdict.status == "undated"
    assert exc.value.verdict.age_days is None
    assert len(freshness_alerts) == 1


def test_unparseable_timestamp_is_undated_not_fresh():
    with pytest.raises(UpstreamStaleError) as exc:
        assert_upstream_fresh(
            "some/blob.md", as_of="not-a-date", cadence="weekly", checked_at=NOW
        )
    assert exc.value.verdict.status == "undated"


def test_degrade_requires_a_named_failure_mode():
    """A deliberate swallow names what it swallows, or it is not allowed."""
    with pytest.raises(ValueError, match="degraded_reason"):
        assert_upstream_fresh(
            "x", as_of=NOW, cadence="weekly", checked_at=NOW, on_stale="degrade"
        )
    with pytest.raises(ValueError, match="degraded_reason"):
        assert_upstream_fresh(
            "x",
            as_of=NOW,
            cadence="weekly",
            checked_at=NOW,
            on_stale="degrade",
            degraded_reason="   ",
        )


def test_degrade_returns_a_recordable_verdict_and_alerts(freshness_alerts, caplog):
    verdict = assert_upstream_fresh(
        "archive/macro/macro_report.md",
        as_of=date(2026, 3, 16),
        cadence="weekly",
        checked_at=NOW,
        on_stale="degrade",
        degraded_reason="non-blocking shadow run; recorded on the manifest",
    )
    assert verdict.status == "stale"
    assert verdict.degraded_reason
    assert len(freshness_alerts) == 1
    assert any(rec.levelname == "ERROR" for rec in caplog.records)

    record = verdict.as_record()
    assert record["status"] == "stale"
    assert record["artifact"] == "archive/macro/macro_report.md"
    assert record["as_of"].startswith("2026-03-16")
    assert record["age_days"] > 150
    assert record["degraded_reason"]
    # JSON-safe: every value must survive a round trip onto an S3 artifact.
    import json

    assert json.loads(json.dumps(record)) == record


def test_explicit_tolerance_overrides_the_cadence_default():
    verdict = assert_upstream_fresh(
        "x",
        as_of=NOW - timedelta(days=9),
        cadence="weekly",
        tolerance_days=30.0,
        checked_at=NOW,
    )
    assert verdict.is_fresh


def test_unknown_cadence_without_tolerance_is_a_programming_error():
    with pytest.raises(ValueError, match="unknown cadence"):
        assert_upstream_fresh("x", as_of=NOW, cadence="fortnightly", checked_at=NOW)


def test_naive_timestamps_are_treated_as_utc():
    verdict = assert_upstream_fresh(
        "x",
        as_of=datetime(2026, 8, 20, 0, 0),  # naive on purpose — the case under test
        cadence="daily",
        checked_at=NOW,
    )
    assert verdict.is_fresh


def test_banner_names_the_artifact_the_age_and_the_anchor_date():
    verdict = FreshnessVerdict(
        artifact="archive/macro/macro_report.md",
        cadence="weekly",
        tolerance_days=10.0,
        checked_at=NOW,
        status="stale",
        as_of=datetime(2026, 3, 16, tzinfo=UTC),
        age_days=157.5,
    )
    banner = verdict.banner()
    assert "archive/macro/macro_report.md" in banner
    assert "157.5" in banner
    assert "2026-03-16" in banner
    assert "2026-08-20" in banner


def test_alert_transport_failure_does_not_lose_the_detection(
    monkeypatch, caplog, live_freshness_alerts
):
    """The only declared swallow in the module: the notification may fail, the
    detection may not."""
    import sys
    import types

    fake = types.ModuleType("ops_alerts")

    def _boom(*args, **kwargs):
        raise RuntimeError("sns unavailable")

    fake.publish_ops_alert = _boom
    monkeypatch.setitem(sys.modules, "ops_alerts", fake)

    verdict = assert_upstream_fresh(
        "x",
        as_of=NOW - timedelta(days=400),
        cadence="weekly",
        checked_at=NOW,
        on_stale="degrade",
        degraded_reason="test",
    )
    assert verdict.status == "stale"
    assert any("ops alert publish failed" in rec.message for rec in caplog.records)


@mock_aws
def test_s3_last_modified_reads_object_metadata():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="alpha-engine-research")
    s3.put_object(Bucket="alpha-engine-research", Key="k.md", Body=b"x")
    stamp = s3_last_modified(s3, bucket="alpha-engine-research", key="k.md")
    assert stamp is not None
    assert stamp.tzinfo is not None


@mock_aws
def test_s3_last_modified_missing_object_is_none_not_an_exception():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="alpha-engine-research")
    assert s3_last_modified(s3, bucket="alpha-engine-research", key="absent.md") is None
