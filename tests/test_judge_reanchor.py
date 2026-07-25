"""Unit tests for the judge re-anchor marker mechanism (config#2575 item 7).

The mechanism is unit-tested here but NOT invoked from any live
promotion path in this codebase yet — see ``evals/judge_reanchor.py``'s
module docstring for the documented promotion-time call contract a
future change should follow.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock


class TestLogJudgeReanchorMarker:
    def test_writes_entry_with_expected_shape(self):
        from evals.judge_reanchor import EVENT_TYPE_JUDGE_REANCHOR, log_judge_reanchor_marker

        s3 = MagicMock()
        key = log_judge_reanchor_marker(
            logical_key="openrouter-shadow",
            old_resolved_model=None,
            new_resolved_model="deepseek/deepseek-v4-flash",
            reason="config#2575 item 7 promotion (test)",
            s3_client=s3,
            now=datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC),
        )

        assert key is not None
        assert key.startswith("changelog/entries/2026-08-01/")
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args
        assert call.kwargs["Bucket"] == "alpha-engine-research"
        assert call.kwargs["Key"] == key

        entry = json.loads(call.kwargs["Body"])
        assert entry["schema_version"] == "1.0.0"
        assert entry["event_type"] == EVENT_TYPE_JUDGE_REANCHOR
        assert entry["judge_reanchor"]["logical_key"] == "openrouter-shadow"
        assert entry["judge_reanchor"]["old_resolved_model"] is None
        assert entry["judge_reanchor"]["new_resolved_model"] == "deepseek/deepseek-v4-flash"
        assert "config#2575" in entry["judge_reanchor"]["reason"]
        assert "regime" in entry["description"].lower()

    def test_old_and_new_model_both_present(self):
        from evals.judge_reanchor import log_judge_reanchor_marker

        s3 = MagicMock()
        log_judge_reanchor_marker(
            logical_key="claude-haiku-4-5",
            old_resolved_model="claude-haiku-4-5-20251001",
            new_resolved_model="claude-haiku-4-5-20260201",
            reason="snapshot repin",
            s3_client=s3,
        )
        entry = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert entry["judge_reanchor"]["old_resolved_model"] == "claude-haiku-4-5-20251001"
        assert "claude-haiku-4-5-20251001" in entry["summary"]
        assert "claude-haiku-4-5-20260201" in entry["summary"]

    def test_write_failure_is_swallowed_and_returns_none(self):
        from evals.judge_reanchor import log_judge_reanchor_marker

        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 down")
        result = log_judge_reanchor_marker(
            logical_key="openrouter-shadow",
            old_resolved_model=None,
            new_resolved_model="deepseek/deepseek-v4-flash",
            reason="test",
            s3_client=s3,
        )
        assert result is None  # never raises

    def test_event_id_idempotent_for_same_inputs_and_timestamp(self):
        """Mirrors evals.rolling_mean's regression-entry idempotency
        convention — same (logical_key, old, new, ts) hashes identically
        so a retried write overwrites rather than duplicates."""
        from evals.judge_reanchor import log_judge_reanchor_marker

        now = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
        s3_a, s3_b = MagicMock(), MagicMock()
        key_a = log_judge_reanchor_marker(
            logical_key="openrouter-shadow", old_resolved_model=None,
            new_resolved_model="deepseek/deepseek-v4-flash",
            reason="r", s3_client=s3_a, now=now,
        )
        key_b = log_judge_reanchor_marker(
            logical_key="openrouter-shadow", old_resolved_model=None,
            new_resolved_model="deepseek/deepseek-v4-flash",
            reason="r", s3_client=s3_b, now=now,
        )
        assert key_a == key_b

    def test_uses_same_changelog_corpus_as_regression_autoemit(self):
        """The re-anchor marker must land in the SAME corpus
        evals.rolling_mean's regression auto-emit uses, so corpus readers
        keyed on the changelog/entries/ prefix pick it up without a
        second integration."""
        from evals.judge_reanchor import _CHANGELOG_BUCKET, _CHANGELOG_PREFIX
        from evals.rolling_mean import _CHANGELOG_BUCKET as rm_bucket
        from evals.rolling_mean import _CHANGELOG_PREFIX as rm_prefix

        assert _CHANGELOG_BUCKET == rm_bucket
        assert _CHANGELOG_PREFIX == rm_prefix
