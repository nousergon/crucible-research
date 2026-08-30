"""Tests for scripts/backfill_orphaned_judge_partitions.py
(alpha-engine-config-I9331 deliverable 2)."""

from __future__ import annotations

import json

import boto3
from moto import mock_aws

from scripts.backfill_orphaned_judge_partitions import backfill, find_orphaned_dates

BUCKET = "alpha-engine-research"


def _put_capture(s3, *, date: str, agent_id: str, run_id: str):
    y, m, d = date.split("-")
    key = f"decision_artifacts/{y}/{m}/{d}/{agent_id}/{run_id}.json"
    body = {
        "schema_version": 2,
        "run_id": run_id,
        "timestamp": f"{date}T12:00:00+00:00",
        "agent_id": agent_id,
        "model_metadata": None,
        "full_prompt_context": None,
        "input_data_snapshot": {"x": 1},
        "agent_output": {"y": 2},
    }
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(body))
    return key


def test_find_orphaned_dates_flags_only_never_judged_days():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        # 2026-07-01 (Wed) — captured, never judged: ORPHANED.
        _put_capture(s3, date="2026-07-01", agent_id="ic_cio", run_id="r1")
        # 2026-07-02 (Thu) — captured AND judged: not orphaned.
        k2 = _put_capture(s3, date="2026-07-02", agent_id="ic_cio", run_id="r2")
        s3.put_object(
            Bucket=BUCKET,
            Key="decision_artifacts/_eval_by_capture/2026-07-02/manifest.json",
            Body=json.dumps({"entries": [{"judged_artifact_s3_key": k2}]}),
        )
        # 2026-07-04 (Sat) — no captures at all: not orphaned (nothing to judge).

        orphaned = find_orphaned_dates(
            s3, bucket=BUCKET, since="2026-06-29", until="2026-07-04",
        )
        assert orphaned == ["2026-07-01"]


def test_backfill_dry_run_makes_no_writes_and_reports_the_plan():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        _put_capture(s3, date="2026-07-01", agent_id="ic_cio", run_id="r1")

        import scripts.backfill_orphaned_judge_partitions as mod

        called = {}

        def _fake_boto3_client(*a, **kw):
            called["used_real_client"] = True
            return s3

        # backfill() constructs its own boto3.client("s3") internally —
        # patch it to the moto-backed client so the dry-run exercises the
        # real listing path against the fixture bucket.
        orig = mod.boto3.client
        mod.boto3.client = lambda *a, **kw: s3
        try:
            summary = backfill(
                bucket=BUCKET, since="2026-06-29", until="2026-07-02",
                dry_run=True,
            )
        finally:
            mod.boto3.client = orig

        assert summary["orphaned_dates"] == ["2026-07-01"]
        assert len(summary["results"]) == 1
        result = summary["results"][0]
        assert result["date"] == "2026-07-01"
        assert result["action"] == "would_submit"
        assert result["capture_keys_total"] == 1

        # No manifest was written — dry-run touched nothing beyond reads.
        listing = s3.list_objects_v2(
            Bucket=BUCKET, Prefix="decision_artifacts/_eval",
        )
        assert listing.get("KeyCount", 0) == 0
