"""Tests for scripts/backfill_eval_option_b.py — historical eval
corpus migration into Option B partition layout (PR 3 of the arc).

Three contract layers exercised:

1. Old-path detection — _parse_old_key recognizes legacy shape and
   rejects new-shape (UUID middle segment) + non-eval keys.

2. migrate_one — reads old, augments with judge_run_id + schema bump,
   writes to new shape, deletes old. Idempotent (re-run after a
   partial failure picks up where it left off, since old keys are
   removed on success).

3. backfill_corpus — end-to-end via mocked S3: scan, group by date,
   mint per-date synthetic judge_run_ids, migrate every file. Summary
   counts match.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from scripts.backfill_eval_option_b import (
    OLD_EVAL_PREFIX,
    _looks_like_uuid,
    _parse_old_key,
    backfill_corpus,
    group_by_date,
    list_old_shape_keys,
    migrate_one,
)


@pytest.fixture
def mocked_s3():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")
        yield s3


def _write_old_eval(
    s3, *, date: str, agent_id: str, run_id: str,
    judge_model: str = "claude-haiku-4-5",
    capture_date: str | None = None,
    bucket: str = "alpha-engine-research",
) -> str:
    """Write an old-shape (schema_version=1, no judge_run_id) eval file."""
    artifact = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": f"{date}T22:30:00.000Z",
        "judged_agent_id": agent_id,
        "rubric_id": "eval_rubric_test",
        "rubric_version": "1.0.0",
        "judge_model": judge_model,
        "dimension_scores": [
            {"dimension": "d1", "score": 4, "reasoning": "ok"},
        ],
        "overall_reasoning": "ok",
    }
    if capture_date:
        artifact["judged_artifact_s3_key"] = (
            f"decision_artifacts/{capture_date.replace('-', '/')}/"
            f"{agent_id}/{run_id}.json"
        )
    key = f"{OLD_EVAL_PREFIX}{date}/{agent_id}/{run_id}.{judge_model}.json"
    s3.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(artifact).encode(),
    )
    return key


# ── _looks_like_uuid ────────────────────────────────────────────────────


class TestLooksLikeUuid:
    def test_real_uuid_v4(self):
        assert _looks_like_uuid("d1e2f3a4-1234-5678-9abc-def012345678")

    def test_too_short(self):
        assert not _looks_like_uuid("abc-123")

    def test_wrong_segment_lengths(self):
        # Right hyphens, wrong char counts.
        assert not _looks_like_uuid("12345678-1234-1234-1234-12345678901")

    def test_non_hex_chars(self):
        assert not _looks_like_uuid("zzzzzzzz-1234-5678-9abc-def012345678")

    def test_agent_id_with_colons_is_not_uuid(self):
        assert not _looks_like_uuid("sector_quant:technology")


# ── _parse_old_key ──────────────────────────────────────────────────────


class TestParseOldKey:
    def test_canonical_old_key_parses(self):
        result = _parse_old_key(
            "decision_artifacts/_eval/2026-05-06/ic_cio/run-1.claude-haiku-4-5.json",
        )
        assert result == {
            "date": "2026-05-06",
            "agent_id": "ic_cio",
            "run_id": "run-1",
            "judge_model": "claude-haiku-4-5",
        }

    def test_agent_id_with_colons_is_recognized(self):
        """Sector / thesis_update agent_ids contain colons in the
        directory name. Must still match the old shape."""
        result = _parse_old_key(
            "decision_artifacts/_eval/2026-05-06/sector_quant:technology/r1.claude-sonnet-4-6.json",
        )
        assert result is not None
        assert result["agent_id"] == "sector_quant:technology"

    def test_new_shape_key_does_not_parse(self):
        """New-shape has a UUID middle segment, NOT an agent_id.
        _parse_old_key must reject so we don't double-migrate."""
        result = _parse_old_key(
            "decision_artifacts/_eval/2026-05-09/"
            "d1e2f3a4-1234-5678-9abc-def012345678/"
            "ic_cio.r1.claude-haiku-4-5.json",
        )
        # Note the new-shape filename ic_cio.r1.claude-haiku-4-5.json
        # would also match the {run_id}.{judge_model} pattern (since
        # there are dots), but the UUID middle segment makes it new.
        assert result is None

    def test_non_eval_key_does_not_parse(self):
        assert _parse_old_key("some/other/path/file.json") is None
        assert _parse_old_key("decision_artifacts/2026/05/06/ic_cio/r1.json") is None


# ── migrate_one ─────────────────────────────────────────────────────────


class TestMigrateOne:
    def test_writes_new_shape_and_deletes_old(self, mocked_s3):
        old_key = _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="ic_cio", run_id="run-1",
        )
        result = migrate_one(
            mocked_s3, bucket="alpha-engine-research",
            old_key=old_key, judge_run_id="batch-test-1", dry_run=False,
        )
        assert result is not None
        new_key, migrated = result
        assert new_key == (
            "decision_artifacts/_eval/2026-05-06/batch-test-1/"
            "ic_cio.run-1.claude-haiku-4-5.json"
        )
        assert migrated["judge_run_id"] == "batch-test-1"
        assert migrated["schema_version"] == 2
        # Old key is gone.
        from botocore.exceptions import ClientError
        with pytest.raises(ClientError):
            mocked_s3.head_object(
                Bucket="alpha-engine-research", Key=old_key,
            )
        # New key has the migrated artifact.
        body = mocked_s3.get_object(
            Bucket="alpha-engine-research", Key=new_key,
        )["Body"].read()
        loaded = json.loads(body)
        assert loaded["judge_run_id"] == "batch-test-1"
        assert loaded["schema_version"] == 2

    def test_dry_run_does_not_write_or_delete(self, mocked_s3):
        old_key = _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="ic_cio", run_id="run-1",
        )
        result = migrate_one(
            mocked_s3, bucket="alpha-engine-research",
            old_key=old_key, judge_run_id="batch-test-1", dry_run=True,
        )
        assert result is not None
        new_key, _ = result
        # Old key still exists.
        mocked_s3.head_object(
            Bucket="alpha-engine-research", Key=old_key,
        )
        # New key NOT written.
        from botocore.exceptions import ClientError
        with pytest.raises(ClientError):
            mocked_s3.head_object(
                Bucket="alpha-engine-research", Key=new_key,
            )

    def test_preserves_judged_artifact_s3_key(self, mocked_s3):
        """FK to the judged artifact must survive the migration —
        the manifest aggregator (PR 2) reads it to group by capture_date."""
        old_key = _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="thesis_update:financials:CBOE",
            run_id="run-thesis-1", capture_date="2026-05-06",
        )
        result = migrate_one(
            mocked_s3, bucket="alpha-engine-research",
            old_key=old_key, judge_run_id="batch-test-1", dry_run=False,
        )
        assert result is not None
        _, migrated = result
        assert migrated["judged_artifact_s3_key"] == (
            "decision_artifacts/2026/05/06/"
            "thesis_update:financials:CBOE/run-thesis-1.json"
        )


# ── group_by_date + list_old_shape_keys ─────────────────────────────────


class TestListAndGroup:
    def test_lists_old_keys_only(self, mocked_s3):
        old_key = _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="ic_cio", run_id="run-1",
        )
        # Also write a "new shape" key — should NOT be listed.
        new_key = (
            "decision_artifacts/_eval/2026-05-09/"
            "d1e2f3a4-1234-5678-9abc-def012345678/"
            "ic_cio.run-2.claude-haiku-4-5.json"
        )
        mocked_s3.put_object(
            Bucket="alpha-engine-research", Key=new_key,
            Body=json.dumps({"schema_version": 2}).encode(),
        )

        keys = list_old_shape_keys(mocked_s3, bucket="alpha-engine-research")
        assert old_key in keys
        assert new_key not in keys

    def test_group_by_date(self, mocked_s3):
        _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="ic_cio", run_id="run-1",
        )
        _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="sector_quant:tech", run_id="run-2",
        )
        _write_old_eval(
            mocked_s3, date="2026-04-25",
            agent_id="ic_cio", run_id="run-3",
        )
        keys = list_old_shape_keys(mocked_s3, bucket="alpha-engine-research")
        grouped = group_by_date(keys)
        assert sorted(grouped.keys()) == ["2026-04-25", "2026-05-06"]
        assert len(grouped["2026-05-06"]) == 2
        assert len(grouped["2026-04-25"]) == 1


# ── backfill_corpus end-to-end ─────────────────────────────────────────


class TestBackfillCorpus:
    def test_migrates_full_corpus(self, mocked_s3):
        for agent in ["ic_cio", "macro_economist", "sector_quant:tech"]:
            _write_old_eval(
                mocked_s3, date="2026-05-06",
                agent_id=agent, run_id="run-1",
            )
        summary = backfill_corpus(
            mocked_s3, bucket="alpha-engine-research", dry_run=False,
        )
        assert summary["old_shape_keys_total"] == 3
        assert summary["dates_processed"] == 1
        assert summary["migrated"] == 3
        assert summary["failed"] == 0
        # No old-shape keys left.
        residue = list_old_shape_keys(
            mocked_s3, bucket="alpha-engine-research",
        )
        assert residue == []

    def test_one_judge_run_id_per_date(self, mocked_s3):
        """All evals on the same source date share one synthetic
        judge_run_id (we don't have batch-boundary info to do finer
        grouping; pre-fix all evals on a date came from one batch)."""
        for agent in ["ic_cio", "macro_economist"]:
            _write_old_eval(
                mocked_s3, date="2026-05-06",
                agent_id=agent, run_id="run-1",
            )
        backfill_corpus(
            mocked_s3, bucket="alpha-engine-research", dry_run=False,
        )
        # Inspect the post-migration keys: all 5/06 evals must share
        # the same judge_run_id directory.
        paginator = mocked_s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(
            Bucket="alpha-engine-research",
            Prefix="decision_artifacts/_eval/2026-05-06/",
        ):
            for obj in page.get("Contents", []) or []:
                keys.append(obj["Key"])
        # Both keys should have the same judge_run_id segment.
        # Path: decision_artifacts/_eval/{date}/{judge_run_id}/{filename}
        judge_run_ids = {k.split("/")[3] for k in keys}
        assert len(judge_run_ids) == 1

    def test_different_dates_get_different_judge_run_ids(self, mocked_s3):
        _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="ic_cio", run_id="run-1",
        )
        _write_old_eval(
            mocked_s3, date="2026-04-25",
            agent_id="ic_cio", run_id="run-2",
        )
        backfill_corpus(
            mocked_s3, bucket="alpha-engine-research", dry_run=False,
        )
        # Inspect the migrated keys and verify the per-date judge_run_id
        # directory names differ.
        paginator = mocked_s3.get_paginator("list_objects_v2")
        all_keys = []
        for page in paginator.paginate(
            Bucket="alpha-engine-research",
            Prefix="decision_artifacts/_eval/",
        ):
            for obj in page.get("Contents", []) or []:
                all_keys.append(obj["Key"])
        date_to_uuid = {}
        for k in all_keys:
            parts = k.split("/")
            date_to_uuid[parts[2]] = parts[3]
        assert date_to_uuid["2026-05-06"] != date_to_uuid["2026-04-25"]

    def test_dry_run_summary_no_writes(self, mocked_s3):
        _write_old_eval(
            mocked_s3, date="2026-05-06",
            agent_id="ic_cio", run_id="run-1",
        )
        summary = backfill_corpus(
            mocked_s3, bucket="alpha-engine-research", dry_run=True,
        )
        assert summary["migrated"] == 1
        assert summary["dry_run"] is True
        # Old key still exists; no new key written.
        assert list_old_shape_keys(
            mocked_s3, bucket="alpha-engine-research",
        ) == [
            "decision_artifacts/_eval/2026-05-06/ic_cio/run-1.claude-haiku-4-5.json"
        ]
