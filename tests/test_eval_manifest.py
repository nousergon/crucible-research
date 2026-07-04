"""Tests for evals/eval_manifest.py — capture-date manifest aggregator
(Option B PR 2).

Three contract layers exercised:

1. Capture-date extraction from ``judged_artifact_s3_key`` — pulls
   ``YYYY-MM-DD`` from the canonical decision_artifacts/{Y}/{M}/{D}/
   prefix; tolerates missing / non-canonical / malformed keys.

2. Manifest build groups eval-artifact JSON files under their
   capture_date. One manifest per capture_date with the eval list
   sorted deterministically (idempotent — repeated runs produce
   byte-identical bodies).

3. End-to-end aggregator scans a (judge_run_dates) window in mocked
   S3, loads each artifact, groups, writes manifests. Skips
   in-memory / synthetic artifacts that lack a capture S3 backref.
"""

from __future__ import annotations

import json
from datetime import date

import boto3
import pytest
from moto import mock_aws

from evals.eval_manifest import (
    DEFAULT_EVAL_PREFIX,
    DEFAULT_MANIFEST_PREFIX,
    MANIFEST_SCHEMA_VERSION,
    _capture_date_from_s3_key,
    _make_manifest_entry,
    build_manifests,
)


@pytest.fixture
def mocked_s3():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="alpha-engine-research")
        yield s3


def _write_eval(
    s3, *, judge_run_date: str, judge_run_id: str,
    judged_agent_id: str, judged_run_id: str,
    judge_model: str = "claude-haiku-4-5",
    capture_date: str = "2026-05-09",
    bucket: str = "alpha-engine-research",
    rubric_id: str = "eval_rubric_test",
    rubric_version: str = "1.0.0",
    judge_skip_reason: str | None = None,
) -> str:
    """Write a v2 RubricEvalArtifact-shaped JSON to mocked S3 + return key."""
    eval_artifact = {
        "schema_version": 2,
        "run_id": judged_run_id,
        "judge_run_id": judge_run_id,
        "timestamp": f"{judge_run_date}T22:30:00.000Z",
        "judged_agent_id": judged_agent_id,
        "judged_artifact_s3_key": (
            f"decision_artifacts/{capture_date.replace('-', '/')}/"
            f"{judged_agent_id}/{judged_run_id}.json"
        ),
        "rubric_id": rubric_id,
        "rubric_version": rubric_version,
        "judge_model": judge_model,
        "dimension_scores": [
            {"dimension": "d1", "score": 4, "reasoning": "ok"},
        ],
        "overall_reasoning": "ok",
        "judge_skip_reason": judge_skip_reason,
    }
    key = (
        f"{DEFAULT_EVAL_PREFIX}{judge_run_date}/{judge_run_id}/"
        f"{judged_agent_id}.{judged_run_id}.{judge_model}.json"
    )
    s3.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(eval_artifact).encode(),
    )
    return key


def _write_eval_flat(
    s3, *, judge_run_id: str,
    judged_agent_id: str, judged_run_id: str,
    judge_model: str = "claude-haiku-4-5",
    capture_date: str = "2026-05-09",
    bucket: str = "alpha-engine-research",
    rubric_id: str = "eval_rubric_test",
    rubric_version: str = "1.0.0",
    judge_skip_reason: str | None = None,
) -> str:
    """Write a CANONICAL FLAT (config#793) eval artifact to mocked S3.

    Path: ``_eval/{judge_run_id}_{agent}.{run}.{model}.json`` where
    judge_run_id is a YYMMDDHHMM timestamp. The timestamp field is
    derived from the run_id's date so judge_run_date extraction in the
    manifest entry stays consistent.
    """
    # Derive an ISO timestamp from the YYMMDDHHMM run_id for the payload.
    yy, mm, dd, hh, mi = (
        judge_run_id[0:2], judge_run_id[2:4], judge_run_id[4:6],
        judge_run_id[6:8], judge_run_id[8:10],
    )
    timestamp = f"20{yy}-{mm}-{dd}T{hh}:{mi}:00.000Z"
    eval_artifact = {
        "schema_version": 2,
        "run_id": judged_run_id,
        "judge_run_id": judge_run_id,
        "timestamp": timestamp,
        "judged_agent_id": judged_agent_id,
        "judged_artifact_s3_key": (
            f"decision_artifacts/{capture_date.replace('-', '/')}/"
            f"{judged_agent_id}/{judged_run_id}.json"
        ),
        "rubric_id": rubric_id,
        "rubric_version": rubric_version,
        "judge_model": judge_model,
        "dimension_scores": [
            {"dimension": "d1", "score": 4, "reasoning": "ok"},
        ],
        "overall_reasoning": "ok",
        "judge_skip_reason": judge_skip_reason,
    }
    key = (
        f"{DEFAULT_EVAL_PREFIX}{judge_run_id}_"
        f"{judged_agent_id}.{judged_run_id}.{judge_model}.json"
    )
    s3.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(eval_artifact).encode(),
    )
    return key


# ── Capture-date extraction ─────────────────────────────────────────────


class TestCaptureDateExtraction:
    def test_canonical_prefix_extracts(self):
        result = _capture_date_from_s3_key(
            "decision_artifacts/2026/05/09/ic_cio/run-1.json",
        )
        assert result == "2026-05-09"

    def test_none_returns_none(self):
        assert _capture_date_from_s3_key(None) is None

    def test_empty_returns_none(self):
        assert _capture_date_from_s3_key("") is None

    def test_non_canonical_prefix_returns_none(self):
        assert _capture_date_from_s3_key("some/other/path/file.json") is None

    def test_malformed_date_returns_none(self):
        # February 31 — strict-parse rejects.
        assert _capture_date_from_s3_key(
            "decision_artifacts/2026/02/31/sector_quant:tech/r1.json",
        ) is None


# ── Manifest entry projection ──────────────────────────────────────────


class TestMakeManifestEntry:
    def test_pulls_canonical_fields(self):
        artifact = {
            "schema_version": 2,
            "run_id": "agent-run-1",
            "judge_run_id": "batch-uuid-1",
            "timestamp": "2026-05-09T22:30:00Z",
            "judged_agent_id": "ic_cio",
            "judged_artifact_s3_key": "decision_artifacts/2026/05/09/ic_cio/agent-run-1.json",
            "rubric_id": "eval_rubric_ic_cio",
            "rubric_version": "1.0.0",
            "judge_model": "claude-haiku-4-5",
            "dimension_scores": [],
            "overall_reasoning": "ok",
            "judge_skip_reason": None,
        }
        eval_s3_key = (
            "decision_artifacts/_eval/2026-05-09/batch-uuid-1/"
            "ic_cio.agent-run-1.claude-haiku-4-5.json"
        )
        entry = _make_manifest_entry(artifact, eval_s3_key)
        assert entry == {
            "judge_run_id": "batch-uuid-1",
            "judge_run_date": "2026-05-09",
            "judged_agent_id": "ic_cio",
            "judged_run_id": "agent-run-1",
            "judge_model": "claude-haiku-4-5",
            "rubric_id": "eval_rubric_ic_cio",
            "rubric_version": "1.0.0",
            "eval_s3_key": eval_s3_key,
            "judged_artifact_s3_key": "decision_artifacts/2026/05/09/ic_cio/agent-run-1.json",
            "judge_skip_reason": None,
        }


# ── End-to-end aggregator ───────────────────────────────────────────────


class TestBuildManifests:
    def test_groups_evals_by_capture_date(self, mocked_s3):
        # Three evals on different captures within same judge run.
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="ic_cio",
            judged_run_id="r1", capture_date="2026-05-09",
        )
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="sector_quant:tech",
            judged_run_id="r2", capture_date="2026-05-09",
        )
        # Cross-day eval: the judge_run_date is 5/9 but the capture is 5/8.
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="ic_cio",
            judged_run_id="r3", capture_date="2026-05-08",
        )

        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"],
        )

        # Two capture_dates with different counts.
        assert set(manifests.keys()) == {"2026-05-09", "2026-05-08"}
        assert manifests["2026-05-09"]["eval_count"] == 2
        assert manifests["2026-05-08"]["eval_count"] == 1
        # Manifest schema version pinned.
        assert manifests["2026-05-09"]["schema_version"] == MANIFEST_SCHEMA_VERSION

    def test_writes_to_s3_at_canonical_key(self, mocked_s3):
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="ic_cio",
            judged_run_id="r1", capture_date="2026-05-09",
        )
        build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"],
        )

        manifest_key = f"{DEFAULT_MANIFEST_PREFIX}2026-05-09/manifest.json"
        obj = mocked_s3.get_object(
            Bucket="alpha-engine-research", Key=manifest_key,
        )
        body = json.loads(obj["Body"].read())
        assert body["capture_date"] == "2026-05-09"
        assert body["eval_count"] == 1
        assert body["evals"][0]["judged_agent_id"] == "ic_cio"

    def test_dry_run_does_not_write(self, mocked_s3):
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="ic_cio",
            judged_run_id="r1", capture_date="2026-05-09",
        )
        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"], write=False,
        )
        # Manifest computed but no S3 write.
        assert "2026-05-09" in manifests
        from botocore.exceptions import ClientError
        with pytest.raises(ClientError):
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=f"{DEFAULT_MANIFEST_PREFIX}2026-05-09/manifest.json",
            )

    def test_skips_evals_without_judged_artifact_s3_key(self, mocked_s3):
        # Simulate an in-memory / synthetic artifact: write the eval
        # but with judged_artifact_s3_key=None.
        eval_artifact = {
            "schema_version": 2,
            "run_id": "synthetic-1",
            "judge_run_id": "batch-1",
            "timestamp": "2026-05-09T22:30:00Z",
            "judged_agent_id": "ic_cio",
            "judged_artifact_s3_key": None,
            "rubric_id": "eval_rubric_ic_cio",
            "rubric_version": "1.0.0",
            "judge_model": "claude-haiku-4-5",
            "dimension_scores": [],
            "overall_reasoning": "ok",
            "judge_skip_reason": None,
        }
        key = (
            f"{DEFAULT_EVAL_PREFIX}2026-05-09/batch-1/"
            f"ic_cio.synthetic-1.claude-haiku-4-5.json"
        )
        mocked_s3.put_object(
            Bucket="alpha-engine-research", Key=key,
            Body=json.dumps(eval_artifact).encode(),
        )

        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"],
        )
        # Synthetic artifact didn't enter any capture_date manifest —
        # still discoverable at its judge_run_id directory.
        assert manifests == {}

    def test_idempotent_repeat_run_produces_same_manifest(self, mocked_s3):
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="ic_cio",
            judged_run_id="r1", capture_date="2026-05-09",
        )
        m1 = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"], write=False,
        )
        m2 = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"], write=False,
        )
        # Body excluding generated_at (which advances each call) is
        # byte-identical — sorted ordering pins the stable shape.
        for cd in m1:
            m1[cd].pop("generated_at", None)
            m2[cd].pop("generated_at", None)
        assert m1 == m2

    def test_sort_order_is_deterministic(self, mocked_s3):
        # Insert evals out of order; manifest should sort by
        # (judged_agent_id, judged_run_id, judge_model).
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="zzz_agent",
            judged_run_id="r1", capture_date="2026-05-09",
        )
        _write_eval(
            mocked_s3, judge_run_date="2026-05-09",
            judge_run_id="batch-1", judged_agent_id="aaa_agent",
            judged_run_id="r1", capture_date="2026-05-09",
        )
        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"],
        )
        agents_in_order = [
            e["judged_agent_id"] for e in manifests["2026-05-09"]["evals"]
        ]
        assert agents_in_order == ["aaa_agent", "zzz_agent"]

    # ── Dual-layout tolerance (config#793) ─────────────────────────────

    def test_reads_canonical_flat_layout(self, mocked_s3):
        """New-layout artifacts (flat ``{run_id}_{basename}``) are scanned
        and indexed by capture_date."""
        # judge_run_id 2605091430 = 2026-05-09 14:30 UTC.
        _write_eval_flat(
            mocked_s3, judge_run_id="2605091430",
            judged_agent_id="ic_cio", judged_run_id="r1",
            capture_date="2026-05-09",
        )
        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"],
        )
        assert "2026-05-09" in manifests
        assert manifests["2026-05-09"]["eval_count"] == 1
        entry = manifests["2026-05-09"]["evals"][0]
        assert entry["judge_run_id"] == "2605091430"
        assert entry["eval_s3_key"].startswith(
            f"{DEFAULT_EVAL_PREFIX}2605091430_"
        )

    def test_reads_both_layouts_together(self, mocked_s3):
        """A corpus mid-migration holds BOTH legacy nested and new flat
        artifacts under _eval/. The scanner reads every one — the swap
        strands no historical forensic data."""
        # Legacy nested for a 5/8 capture.
        _write_eval(
            mocked_s3, judge_run_date="2026-05-08",
            judge_run_id="batch-legacy-uuid", judged_agent_id="ic_cio",
            judged_run_id="r-old", capture_date="2026-05-08",
        )
        # New flat for a 5/9 capture.
        _write_eval_flat(
            mocked_s3, judge_run_id="2605091430",
            judged_agent_id="sector_quant:tech", judged_run_id="r-new",
            capture_date="2026-05-09",
        )
        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-08", "2026-05-09"],
        )
        assert set(manifests.keys()) == {"2026-05-08", "2026-05-09"}
        assert manifests["2026-05-08"]["eval_count"] == 1
        assert manifests["2026-05-09"]["eval_count"] == 1
        # Legacy entry carries the UUID run_id; flat entry the timestamp.
        assert manifests["2026-05-08"]["evals"][0]["judge_run_id"] == "batch-legacy-uuid"
        assert manifests["2026-05-09"]["evals"][0]["judge_run_id"] == "2605091430"

    def test_flat_run_id_outside_window_excluded(self, mocked_s3):
        """A flat artifact whose timestamp-encoded run_id falls outside
        the requested judge_run_dates window is not indexed — the scanner
        date-filters flat keys by their run_id prefix."""
        # run_id date 2026-05-09, but we only ask for 2026-05-10.
        _write_eval_flat(
            mocked_s3, judge_run_id="2605091430",
            judged_agent_id="ic_cio", judged_run_id="r1",
            capture_date="2026-05-09",
        )
        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-10"],
        )
        assert manifests == {}

    def test_latest_sidecar_is_not_indexed(self, mocked_s3):
        """The operator-UX latest.json sidecar is a pointer, not an eval
        artifact — the scanner must skip it."""
        from nousergon_lib.eval_artifacts import eval_latest_key
        _write_eval_flat(
            mocked_s3, judge_run_id="2605091430",
            judged_agent_id="ic_cio", judged_run_id="r1",
            capture_date="2026-05-09",
        )
        # Write a sidecar alongside.
        mocked_s3.put_object(
            Bucket="alpha-engine-research",
            Key=eval_latest_key(DEFAULT_EVAL_PREFIX),
            Body=json.dumps({"artifact_key": "whatever"}).encode(),
        )
        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            judge_run_dates=["2026-05-09"],
        )
        # Only the real eval indexed; sidecar excluded (no crash).
        assert manifests["2026-05-09"]["eval_count"] == 1

    def test_lookback_window_default(self, mocked_s3):
        # Eval at the edge of the lookback window (10 days back) is
        # included; eval beyond (20 days back) is missed.
        _write_eval(
            mocked_s3, judge_run_date="2026-04-29",  # 10 days back
            judge_run_id="batch-recent", judged_agent_id="ic_cio",
            judged_run_id="r1", capture_date="2026-04-29",
        )
        _write_eval(
            mocked_s3, judge_run_date="2026-04-19",  # 20 days back
            judge_run_id="batch-old", judged_agent_id="ic_cio",
            judged_run_id="r2", capture_date="2026-04-19",
        )
        manifests = build_manifests(
            s3_client=mocked_s3, bucket="alpha-engine-research",
            today=date(2026, 5, 9), lookback_days=14,
        )
        assert "2026-04-29" in manifests
        assert "2026-04-19" not in manifests
