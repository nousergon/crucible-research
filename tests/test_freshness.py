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
    StaleGroup,
    UpstreamStaleError,
    assert_upstream_fresh,
    group_stale_verdicts,
    publish_grouped_alerts,
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


# ── Grouping / fan-out — alpha-engine-config-I8173's own bug class ─────────
#
# The historical incident this section pins: one dead producer
# (``decision_artifacts/`` last write 2026-07-11) fanned out into 47
# independent ERROR pages, one per artifact subtype, because each call to
# ``assert_upstream_fresh`` alerted on its own. These tests prove the
# grouping mechanism collapses that fan-out to one page, that genuinely
# distinct causes still page separately, and that one unbroken episode does
# not re-page across repeat evaluations.


def _stale_batch(n: int, *, as_of: datetime, checked_at: datetime) -> list[FreshnessVerdict]:
    """N findings that all trace to the SAME frozen producer — the shape of
    the real 47-subtype incident (``decision_artifacts/*/<subtype>``)."""
    return [
        assert_upstream_fresh(
            f"decision_artifacts/*/subtype_{i}",
            as_of=as_of,
            cadence="weekly",
            checked_at=checked_at,
            on_stale="degrade",
            degraded_reason="weekly rollup skips this subtype's contribution",
            remediation="a new decision_artifacts/{Y}/{M}/{D}/<subtype>/*.json write",
            source="crucible-research.evals.rationale_clustering",
            alert=False,
        )
        for i in range(n)
    ]


def test_47_same_cause_findings_collapse_to_one_alert(freshness_alerts):
    as_of = datetime(2026, 7, 11, tzinfo=UTC)
    checked_at = datetime(2026, 8, 22, tzinfo=UTC)  # 42.4 days — the real measured age
    verdicts = _stale_batch(47, as_of=as_of, checked_at=checked_at)
    assert all(v.status == "stale" for v in verdicts)
    assert all(v.driver == "producer-halted" for v in verdicts)

    groups = publish_grouped_alerts(
        verdicts, source="crucible-research.evals.rationale_clustering"
    )

    assert len(groups) == 1
    assert len(groups[0].members) == 47
    # One notification for the group, not 47 (observability-policy §7.2a).
    assert len(freshness_alerts) == 1
    message = freshness_alerts[0]["message"]
    assert "47" in message  # exact count named, even though it is truncated
    assert "decision_artifacts" in message
    # The group NAMES its members — the first ones verbatim, not "see logs".
    assert "subtype_0" in message
    assert "subtype_9" in message
    # Truncation is marked in-band, and the count stays exact (never a
    # sample presented as a census, observability-policy.md §5.1).
    assert "more" in message
    assert "47 total" in message
    assert freshness_alerts[0]["source"] == "crucible-research.evals.rationale_clustering"


def test_distinct_causes_still_publish_separately(freshness_alerts):
    """Two findings that do NOT share a cause must not be merged into one
    alert — grouping is legitimate only when the evidence says one failure."""
    checked_at = datetime(2026, 8, 22, tzinfo=UTC)
    same_producer = _stale_batch(3, as_of=datetime(2026, 7, 11, tzinfo=UTC), checked_at=checked_at)
    unrelated = assert_upstream_fresh(
        "regime/",
        as_of=None,  # never written — a DIFFERENT driver entirely
        cadence="weekly",
        checked_at=checked_at,
        on_stale="degrade",
        degraded_reason="macro anchor missing",
        source="research:thinktank_daily",
        alert=False,
    )

    groups = publish_grouped_alerts(
        [*same_producer, unrelated], source="crucible-research.evals.rationale_clustering"
    )

    assert len(groups) == 2
    assert len(freshness_alerts) == 2
    dedup_keys = {rec["dedup_key"] for rec in freshness_alerts}
    assert len(dedup_keys) == 2  # distinct episodes, distinct dedup keys


def test_distinct_causes_same_prefix_different_as_of_still_separate(freshness_alerts):
    """Same upstream family, same driver, but a DIFFERENT as-of — e.g. one
    subtype's producer resumed writing while the rest stayed frozen. That is
    two failures, not one, even though the artifact prefix matches."""
    checked_at = datetime(2026, 8, 22, tzinfo=UTC)
    old_episode = _stale_batch(2, as_of=datetime(2026, 7, 11, tzinfo=UTC), checked_at=checked_at)
    newer_but_still_stale = assert_upstream_fresh(
        "decision_artifacts/*/subtype_recovered_then_refroze",
        as_of=datetime(2026, 8, 1, tzinfo=UTC),
        cadence="weekly",
        checked_at=checked_at,
        on_stale="degrade",
        degraded_reason="weekly rollup skips this subtype's contribution",
        source="crucible-research.evals.rationale_clustering",
        alert=False,
    )

    groups = group_stale_verdicts([*old_episode, newer_but_still_stale])
    assert len(groups) == 2


def test_episode_key_is_stable_across_repeat_evaluations_until_the_producer_writes():
    """One unbroken episode pages once, not once per evaluation. No new
    persisted state: the SAME as-of timestamp yields the SAME dedup key on a
    second, independent evaluation run — which is what lets the alert
    transport's own dedup marker suppress the repeat. The moment the
    producer writes again (a new as-of), the key moves."""
    checked_run_1 = datetime(2026, 8, 15, tzinfo=UTC)
    checked_run_2 = datetime(2026, 8, 22, tzinfo=UTC)  # next weekly evaluation
    as_of = datetime(2026, 7, 11, tzinfo=UTC)

    run_1 = group_stale_verdicts(_stale_batch(5, as_of=as_of, checked_at=checked_run_1))
    run_2 = group_stale_verdicts(_stale_batch(5, as_of=as_of, checked_at=checked_run_2))
    assert run_1[0].episode_key() == run_2[0].episode_key()

    # Still stale against the weekly tolerance (21d > 10d), but a NEWER
    # write than the frozen 2026-07-11 — a different episode, not a healed
    # producer.
    producer_wrote_again = group_stale_verdicts(
        _stale_batch(5, as_of=datetime(2026, 8, 1, tzinfo=UTC), checked_at=checked_run_2)
    )
    assert producer_wrote_again[0].episode_key() != run_2[0].episode_key()


def test_group_names_its_members_not_just_a_count():
    group = StaleGroup(
        driver="producer-halted",
        upstream_prefix="decision_artifacts",
        as_of=datetime(2026, 7, 11, tzinfo=UTC),
        members=tuple(_stale_batch(3, as_of=datetime(2026, 7, 11, tzinfo=UTC), checked_at=datetime(2026, 8, 22, tzinfo=UTC))),
    )
    banner = group.banner()
    for i in range(3):
        assert f"subtype_{i}" in banner
    assert "3 findings" in banner


# ── Driver attribution — the closed set, computed on every evaluation ──────


def test_driver_producer_halted_when_a_stale_as_of_exists():
    verdict = assert_upstream_fresh(
        "x", as_of=NOW - timedelta(days=400), cadence="weekly", checked_at=NOW,
        on_stale="degrade", degraded_reason="test",
    )
    assert verdict.driver == "producer-halted"


def test_driver_never_written_when_as_of_is_none():
    verdict = assert_upstream_fresh(
        "x", as_of=None, cadence="weekly", checked_at=NOW,
        on_stale="degrade", degraded_reason="test",
    )
    assert verdict.driver == "never-written"


def test_driver_timestamp_unreadable_when_as_of_does_not_parse():
    verdict = assert_upstream_fresh(
        "x", as_of="not-a-date", cadence="weekly", checked_at=NOW,
        on_stale="degrade", degraded_reason="test",
    )
    assert verdict.driver == "timestamp-unreadable"


def test_driver_is_none_only_for_fresh():
    verdict = assert_upstream_fresh(
        "x", as_of=NOW - timedelta(days=1), cadence="weekly", checked_at=NOW
    )
    assert verdict.is_fresh
    assert verdict.driver is None


def test_driver_unattributed_is_a_reachable_terminal_branch_not_dead_code():
    """The closed set's defensive branch — reached whenever ``status`` and
    the coerced timestamp disagree in a way the three named branches don't
    cover. Exercised directly since ``assert_upstream_fresh`` itself cannot
    construct this combination (its own contract keeps status/stamp
    consistent) — this pins that the terminal branch exists and is not a
    silent fallthrough that would otherwise be untestable dead code."""
    from freshness import _attribute_driver

    assert (
        _attribute_driver(status="stale", as_of_input=NOW, stamp=None)
        == "unattributed"
    )


def test_driver_and_remediation_are_recorded_on_every_evaluation_not_only_when_alerting():
    """Attribution must not be alert-conditional — it is computed and
    persisted even when ``alert=False`` (the batched-fan-out path)."""
    verdict = assert_upstream_fresh(
        "decision_artifacts/*/subtype_x",
        as_of=datetime(2026, 7, 11, tzinfo=UTC),
        cadence="weekly",
        checked_at=datetime(2026, 8, 22, tzinfo=UTC),
        on_stale="degrade",
        degraded_reason="test",
        remediation="a fresh write under decision_artifacts/*/subtype_x",
        alert=False,
    )
    record = verdict.as_record()
    assert record["driver"] == "producer-halted"
    assert record["remediation"] == "a fresh write under decision_artifacts/*/subtype_x"
    import json

    assert json.loads(json.dumps(record)) == record
