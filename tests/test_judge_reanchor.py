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


class TestDeclaredReanchors:
    """alpha-engine-config-I11484: the 2026-08-30 judge switch is declared in
    code and binds the control-band reset on merge."""

    def test_the_08_30_break_is_declared_for_both_authoritative_tiers(self):
        from evals.judge_models import HAIKU, SONNET
        from evals.judge_reanchor import DECLARED_REANCHORS

        by_key = {r.logical_key: r for r in DECLARED_REANCHORS}
        for spec in (HAIKU, SONNET):
            r = by_key[spec.logical_key]
            assert r.effective_utc == datetime(2026, 8, 30, tzinfo=UTC)
            assert r.new_resolved_model == "deepseek-v4-flash"
            assert "I11484" in r.reason

    def test_every_declared_key_is_a_registered_judge(self):
        from evals.judge_models import resolve
        from evals.judge_reanchor import DECLARED_REANCHORS

        for r in DECLARED_REANCHORS:
            resolve(r.logical_key)  # raises KeyError on an unknown key

    def test_no_old_model_judging_shares_the_reset_slot(self):
        """The reset is floored to the week slot holding the change, which is
        only sound while the last old-model run lands in an EARLIER slot."""
        from evals.judge_reanchor import DECLARED_REANCHORS, _slot_start

        for r in DECLARED_REANCHORS:
            assert r.last_old_run_utc < _slot_start(r.effective_utc) <= r.effective_utc

    def test_reset_applies_from_the_change_slot(self):
        from evals.judge_reanchor import _slot_start, reset_before_by_judge

        out = reset_before_by_judge(datetime(2026, 10, 29, tzinfo=UTC))
        slot = _slot_start(datetime(2026, 8, 30, tzinfo=UTC))
        assert slot == datetime(2026, 8, 27, tzinfo=UTC)  # epoch slots start Thu
        assert out == {"claude-haiku-4-5": slot, "claude-sonnet-4-6": slot}
        assert "openrouter-shadow" not in out

    def test_a_replay_before_the_change_sees_no_reset(self):
        from evals.judge_reanchor import reset_before_by_judge

        assert reset_before_by_judge(datetime(2026, 8, 29, tzinfo=UTC)) == {}

    def test_the_latest_reanchor_per_key_wins(self):
        from evals.judge_reanchor import DeclaredReanchor, reset_before_by_judge

        def _r(day):
            return DeclaredReanchor(
                logical_key="k", old_resolved_model="a", new_resolved_model="b",
                last_old_run_utc=datetime(2026, 1, 1, tzinfo=UTC),
                effective_utc=day, reason="t",
            )

        early = _r(datetime(2026, 3, 5, tzinfo=UTC))
        late = _r(datetime(2026, 6, 4, tzinfo=UTC))
        out = reset_before_by_judge(
            datetime(2026, 7, 1, tzinfo=UTC), reanchors=(late, early),
        )
        assert out == {"k": datetime(2026, 6, 4, tzinfo=UTC)}
