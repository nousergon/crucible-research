"""End-to-end batch flow tests for the LLM-as-judge orchestrator
(ROADMAP §1642). Covers ``build_batch_plan`` → ``submit_batch`` →
``poll_batch`` → ``process_batch_results`` with a stubbed Anthropic
client + moto-mocked S3.

The Anthropic client is stubbed at the call boundary so these tests
pin the wiring (custom_id round-trip, plan-manifest persistence,
empty-input client-side skip, Sonnet escalation tail) without making
real API calls. Real-LLM smoke is deferred to the deploy canary.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from alpha_engine_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_capture_dict(
    agent_id: str,
    *,
    run_id: str = "run-1",
    agent_output: dict | None = None,
    input_data_snapshot: dict | None = None,
) -> dict:
    """Builds a DecisionArtifact dict for test fixtures.

    The ``input_data_snapshot`` defaults to a per-rubric "non-degenerate"
    shape (post-2026-05-13 input-sufficiency gate) so the existing
    batch-plan tests' shape contracts (n mapped → n requests) still
    hold. Tests that want to exercise the degenerate-input skip path
    pass an explicit empty/sparse snapshot.
    """
    if input_data_snapshot is None:
        if agent_id.startswith("sector_quant:"):
            input_data_snapshot = {
                "sector_tickers": ["AAPL"],
                "sector_tickers_count": 1,
                "technical_scores_team": {"AAPL": {"rsi_14": 55}},
            }
        elif agent_id.startswith("sector_qual:"):
            input_data_snapshot = {
                "sector_tickers": ["AAPL"],
                "sector_tickers_count": 1,
                "sector_population": ["AAPL"],
            }
        elif agent_id.startswith("sector_peer_review:"):
            input_data_snapshot = {
                "quant_picks": [{"ticker": "AAPL"}],
                "qual_picks": [{"ticker": "AAPL"}],
            }
        elif agent_id.startswith("thesis_update:"):
            input_data_snapshot = {
                "prior_thesis": {"thesis_summary": "real thesis text"},
                "news_data": {"articles": [{"headline": "h"}]},
                "analyst_data": {"consensus_rating": "buy"},
            }
        else:
            input_data_snapshot = {"k": "v"}
    return DecisionArtifact(
        run_id=run_id,
        timestamp="2026-05-09T22:30:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
        full_prompt_context=FullPromptContext(
            system_prompt="<see config/prompts>",
            user_prompt="<rendered>",
        ),
        input_data_snapshot=input_data_snapshot,
        input_data_summary="k=v",
        agent_output=agent_output if agent_output is not None else {"out": "ok"},
    ).model_dump()


@pytest.fixture
def mocked_s3():
    """Yields an S3 client with a small mixed-corpus capture set:
    3 mapped agents (ic_cio + sector_quant + macro_economist) +
    1 unmapped agent (skipped) + 1 empty-input mapped agent (client-
    side skip-marker write)."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        prefix = "decision_artifacts/2026/05/09"
        captures = {
            f"{prefix}/ic_cio/run-1.json":
                _make_capture_dict("ic_cio"),
            f"{prefix}/sector_quant:technology/run-1.json":
                _make_capture_dict("sector_quant:technology"),
            f"{prefix}/macro_economist/run-1.json":
                _make_capture_dict("macro_economist"),
            f"{prefix}/unknown_xyz/run-1.json":
                _make_capture_dict("unknown_xyz"),
            f"{prefix}/sector_qual:technology/run-1.json":
                _make_capture_dict(
                    "sector_qual:technology", agent_output={},
                ),
        }
        for key, payload in captures.items():
            client.put_object(
                Bucket="alpha-engine-research",
                Key=key,
                Body=json.dumps(payload, default=str).encode("utf-8"),
            )
        yield client


def _make_batch_result_succeeded(
    custom_id: str,
    *,
    scores: list[int] | None = None,
):
    """Synthesize one batch result entry as the SDK would yield."""
    scores = scores if scores is not None else [4, 4, 4]
    # Note: ``MagicMock(name=...)`` reserves ``name`` for the mock's
    # identity. We assign attributes after construction so block.name
    # returns the literal string instead of the mock's repr.
    block = MagicMock(spec=["type", "name", "input"])
    block.type = "tool_use"
    block.name = "RubricEvalLLMOutput"
    block.input = {
        "dimension_scores": [
            {
                "dimension": f"dim_{i}",
                "score": s,
                "reasoning": f"r_{i}",
            }
            for i, s in enumerate(scores)
        ],
        "overall_reasoning": "ok",
    }
    msg = MagicMock(spec=["content"])
    msg.content = [block]
    result = MagicMock()
    result.custom_id = custom_id
    result.result = MagicMock()
    result.result.type = "succeeded"
    result.result.message = msg
    return result


def _make_batch_result_malformed_stringified(custom_id: str):
    """Batch result whose tool input carries dimension_scores as a
    MALFORMED (unterminated) JSON string and no overall_reasoning —
    the exact 2026-07-03 Haiku non-conformance shape (config#1650:
    stop_reason=tool_use, NOT truncation; json.loads fails so the
    schema's mode='before' salvage validator correctly falls through)."""
    block = MagicMock(spec=["type", "name", "input"])
    block.type = "tool_use"
    block.name = "RubricEvalLLMOutput"
    block.input = {
        "dimension_scores": (
            '[\n  {\n    "dimension": "d0",\n    "score": 4,\n'
            '    "reasoning": "unterminated'
        ),
    }
    msg = MagicMock(spec=["content"])
    msg.content = [block]
    result = MagicMock()
    result.custom_id = custom_id
    result.result = MagicMock()
    result.result.type = "succeeded"
    result.result.message = msg
    return result


def _make_batch_result_errored(custom_id: str, error_type: str = "server_error"):
    result = MagicMock()
    result.custom_id = custom_id
    result.result = MagicMock()
    result.result.type = "errored"
    result.result.error = MagicMock()
    result.result.error.type = error_type
    return result


# ── build_batch_plan ──────────────────────────────────────────────────────


class TestBuildBatchPlan:
    def test_weekly_path_haiku_only_per_mapped_artifact(self, mocked_s3):
        from evals.orchestrator import build_batch_plan

        plan = build_batch_plan(
            date="2026-05-09",
            bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        # 3 mapped + 1 empty-input client-side skip + 1 unmapped.
        # Weekly path: each mapped non-empty artifact gets ONE Haiku
        # entry. Empty-input is bucketed in client_side_skips.
        assert len(plan["requests"]) == 3
        assert plan["skipped_unmapped"] == 1
        empty_skips = [
            s for s in plan["client_side_skips"]
            if s.get("stage") == "empty_input_skip"
        ]
        assert len(empty_skips) == 1
        # All requests target Haiku for the weekly cadence. The API
        # request is PINNED to the dated snapshot (L4578(a))...
        models = {r["params"]["model"] for r in plan["requests"]}
        assert models == {"claude-haiku-4-5-20251001"}
        # ...while the custom_id still carries the STABLE logical key, so
        # persistence + the CloudWatch dimension don't move on a pin.
        from evals.judge import decode_custom_id

        logical = {decode_custom_id(r["custom_id"])[2] for r in plan["requests"]}
        assert logical == {"claude-haiku-4-5"}

    def test_first_saturday_path_haiku_plus_sonnet_per_mapped_artifact(
        self, mocked_s3,
    ):
        from evals.orchestrator import build_batch_plan

        plan = build_batch_plan(
            date="2026-05-09",
            bucket="alpha-engine-research",
            force_sonnet_pass=True,
            s3_client=mocked_s3,
        )
        # 3 mapped artifacts × 2 tiers = 6 requests. Haiku is pinned to
        # its dated snapshot; Sonnet 4.6 has no snapshot to pin to so it
        # requests the alias (L4578(a)).
        assert len(plan["requests"]) == 6
        models = {r["params"]["model"] for r in plan["requests"]}
        assert models == {"claude-haiku-4-5-20251001", "claude-sonnet-4-6"}

    def test_judge_only_routes_to_isolated_eval_prefix(self, mocked_s3):
        from evals.orchestrator import (
            build_batch_plan,
            JUDGE_ONLY_EVAL_PREFIX,
            JUDGE_ONLY_CW_NAMESPACE,
        )

        plan = build_batch_plan(
            date="2026-05-09",
            bucket="alpha-engine-research",
            judge_only=True,
            s3_client=mocked_s3,
        )
        assert plan["eval_prefix"] == JUDGE_ONLY_EVAL_PREFIX
        assert plan["cw_namespace"] == JUDGE_ONLY_CW_NAMESPACE


# ── submit_batch ──────────────────────────────────────────────────────────


class TestSubmitBatch:
    def test_persists_plan_manifest_keyed_by_batch_id(self, mocked_s3):
        from evals.orchestrator import build_batch_plan, submit_batch

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )

        # Stub Anthropic client returning a fake batch_id.
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_test_001"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch

        result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )
        assert result["batch_id"] == "msgbatch_test_001"
        assert result["processing_status"] == "in_progress"
        assert result["request_count"] == 3
        # Manifest was persisted at the canonical location.
        assert (
            result["plan_s3_key"]
            == "decision_artifacts/_eval_batch_plans/2026-05-09/msgbatch_test_001.json"
        )
        manifest_raw = mocked_s3.get_object(
            Bucket="alpha-engine-research", Key=result["plan_s3_key"],
        )["Body"].read()
        manifest = json.loads(manifest_raw)
        assert manifest["force_sonnet_pass"] is False
        assert len(manifest["plan_entries"]) == 3

    def test_empty_plan_short_circuits_anthropic_call(self, mocked_s3):
        """Nothing to submit → no API call → synthetic batch_id +
        ``ended_empty`` status. Avoids paying for a batch slot on a
        Saturday with no captures (unlikely but defensively handled)."""
        from evals.orchestrator import submit_batch

        empty_plan = {
            "date": "2026-05-09",
            "bucket": "alpha-engine-research",
            "eval_prefix": "decision_artifacts/_eval/",
            "cw_namespace": "AlphaEngine/Eval",
            "haiku_model": "claude-haiku-4-5",
            "sonnet_model": "claude-sonnet-4-6",
            "force_sonnet_pass": False,
            "judge_only": False,
            "max_tokens": 4096,
            "capture_keys_total": 0,
            "skipped_unmapped": 0,
            "client_side_skips": [],
            "plan_entries": [],
            "requests": [],
        }
        fake_client = MagicMock()
        result = submit_batch(
            empty_plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )
        assert result["processing_status"] == "ended_empty"
        assert result["batch_id"].startswith("empty-")
        assert result["request_count"] == 0
        # Anthropic API never called for an empty plan.
        fake_client.messages.batches.create.assert_not_called()


# ── poll_batch ────────────────────────────────────────────────────────────


class TestPollBatch:
    def test_synthetic_empty_batch_returns_ended(self):
        from evals.orchestrator import poll_batch

        # No client needed — empty path short-circuits.
        result = poll_batch(
            batch_id="empty-2026-05-09", anthropic_client=MagicMock(),
        )
        assert result["processing_status"] == "ended"

    def test_in_progress_propagates(self):
        from evals.orchestrator import poll_batch

        fake_client = MagicMock()
        fake_batch = MagicMock(spec=["processing_status", "request_counts"])
        fake_batch.processing_status = "in_progress"
        fake_batch.request_counts = {"processing": 5, "succeeded": 2}
        fake_client.messages.batches.retrieve.return_value = fake_batch
        result = poll_batch(
            batch_id="msgbatch_xyz", anthropic_client=fake_client,
        )
        assert result["processing_status"] == "in_progress"
        assert result["request_counts"]["processing"] == 5

    def test_pydantic_batch_with_datetime_serializes_for_lambda_marshaller(self):
        """Anthropic's real batch object is a Pydantic model with
        ``datetime`` fields (``created_at`` / ``ended_at`` / ``expires_at``).
        Plain ``model_dump()`` returns Python datetime objects, which
        Lambda's JSON response marshaller cannot serialize — every poll
        crashes with ``Object of type datetime is not JSON serializable``.

        Surfaced 2026-05-07 against a real Anthropic batch retrieval:
        the original unit test (above) used a MagicMock that bypassed
        Pydantic and missed the bug. Pinning behavior with a real
        Pydantic model that mirrors Anthropic's MessageBatch shape so
        the regression class can't recur.
        """
        from datetime import datetime, timezone
        from pydantic import BaseModel
        from evals.orchestrator import poll_batch

        class FakePydanticBatch(BaseModel):
            id: str
            processing_status: str
            request_counts: dict
            created_at: datetime
            ended_at: datetime | None
            expires_at: datetime

        fake_batch = FakePydanticBatch(
            id="msgbatch_xyz",
            processing_status="ended",
            request_counts={"succeeded": 24, "errored": 0},
            created_at=datetime(2026, 5, 7, 23, 30, tzinfo=timezone.utc),
            ended_at=datetime(2026, 5, 7, 23, 35, tzinfo=timezone.utc),
            expires_at=datetime(2026, 5, 8, 23, 30, tzinfo=timezone.utc),
        )
        fake_client = MagicMock()
        fake_client.messages.batches.retrieve.return_value = fake_batch

        result = poll_batch(
            batch_id="msgbatch_xyz", anthropic_client=fake_client,
        )

        # Result must be JSON-serializable — Lambda's runtime marshaller
        # is `json.dumps`, no `default=` fallback.
        json.dumps(result)

        assert result["processing_status"] == "ended"
        assert isinstance(result["ended_at"], str)  # ISO string, not datetime
        assert result["ended_at"].startswith("2026-05-07T23:35")


# ── process_batch_results ─────────────────────────────────────────────────


class TestProcessBatchResults:
    def test_weekly_path_persists_haiku_evals_to_s3(self, mocked_s3):
        """Submit weekly plan → fake successful batch results → Process
        persists per-artifact eval JSONs at decision_artifacts/_eval/."""
        from evals.orchestrator import (
            build_batch_plan, submit_batch, process_batch_results,
        )

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_test_002"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch

        submit_result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )

        # Synthesize the batch results — one succeeded entry per
        # custom_id from the plan. All scores are 4 so no Sonnet
        # escalation fires.
        plan_entries = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        fake_results = [
            _make_batch_result_succeeded(e["custom_id"], scores=[4, 4, 4])
            for e in plan_entries
        ]
        fake_client.messages.batches.results.return_value = iter(fake_results)

        summary = process_batch_results(
            batch_id=submit_result["batch_id"],
            plan_s3_key=submit_result["plan_s3_key"],
            bucket="alpha-engine-research",
            anthropic_client=fake_client,
            s3_client=mocked_s3,
            emit_metrics=False,
        )
        assert summary["haiku_evaluated"] == 3
        assert summary["sonnet_evaluated"] == 0
        assert summary["failed"] == []
        assert summary["skipped_unmapped"] == 1
        # Each persisted_key lives under decision_artifacts/_eval/.
        for k in summary["persisted_keys"]:
            assert k.startswith("decision_artifacts/_eval/")

    def test_weekly_path_runs_sonnet_escalation_tail_on_borderline_haiku(
        self, mocked_s3,
    ):
        """A Haiku result with a dimension <3 must trigger a synchronous
        Sonnet escalation call inside Process. ``evaluate_artifact`` is
        the call site for the tail; pin its invocation."""
        from evals.orchestrator import (
            build_batch_plan, submit_batch, process_batch_results,
        )
        from unittest.mock import patch

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_test_003"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch
        submit_result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )
        plan_entries = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        # Build batch results: ic_cio gets a borderline 2 → escalate.
        fake_results = []
        for e in plan_entries:
            scores = [2, 4, 4] if e["agent_id"] == "ic_cio" else [4, 4, 4]
            fake_results.append(
                _make_batch_result_succeeded(e["custom_id"], scores=scores),
            )
        fake_client.messages.batches.results.return_value = iter(fake_results)

        # Stub the synchronous escalation call.
        from evals import orchestrator as orch
        from graph.state_schemas import (
            RubricEvalArtifact, RubricDimensionScore,
        )

        def fake_evaluate(artifact, *, judge_run_id, judge_model, judged_artifact_s3_key, **kw):
            return RubricEvalArtifact(
                run_id=artifact.run_id,
                judge_run_id=judge_run_id,
                timestamp="2026-05-09T22:30:00Z",
                judged_agent_id=artifact.agent_id,
                rubric_id="eval_rubric_test",
                rubric_version="1.0.0",
                judge_model=judge_model,
                dimension_scores=[
                    RubricDimensionScore(
                        dimension="d", score=4, reasoning="r",
                    ),
                ],
                overall_reasoning="ok",
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_evaluate):
            summary = process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                anthropic_client=fake_client,
                s3_client=mocked_s3,
                emit_metrics=False,
            )
        assert summary["haiku_evaluated"] == 3
        # Exactly the ic_cio escalation should have run.
        assert summary["sonnet_evaluated"] == 1

    def test_first_saturday_path_skips_escalation_tail(self, mocked_s3):
        """force_sonnet_pass=True submits both tiers in the batch — the
        Process Lambda must NOT also run the synchronous escalation
        tail (would double-bill Sonnet on every borderline)."""
        from evals.orchestrator import (
            build_batch_plan, submit_batch, process_batch_results,
        )
        from unittest.mock import patch

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            force_sonnet_pass=True, s3_client=mocked_s3,
        )
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_test_004"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch
        submit_result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )
        plan_entries = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        # Borderline scores everywhere — the tail would bill again
        # if not gated by force_sonnet_pass.
        fake_results = [
            _make_batch_result_succeeded(e["custom_id"], scores=[2, 4, 4])
            for e in plan_entries
        ]
        fake_client.messages.batches.results.return_value = iter(fake_results)

        from evals import orchestrator as orch
        with patch.object(orch, "evaluate_artifact") as ea_mock:
            summary = process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                anthropic_client=fake_client,
                s3_client=mocked_s3,
                emit_metrics=False,
            )
            ea_mock.assert_not_called()
        # 3 mapped × 2 tiers = 6 entries; all should persist.
        assert summary["haiku_evaluated"] + summary["sonnet_evaluated"] == 6
        assert summary["sonnet_evaluated"] == 3

    def test_errored_batch_result_is_recorded_as_failed(self, mocked_s3):
        from evals.orchestrator import (
            build_batch_plan, submit_batch, process_batch_results,
        )

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_test_005"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch
        submit_result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )
        plan_entries = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        # Mix one errored result with two successes.
        fake_results = []
        for i, e in enumerate(plan_entries):
            if i == 0:
                fake_results.append(_make_batch_result_errored(e["custom_id"]))
            else:
                fake_results.append(
                    _make_batch_result_succeeded(e["custom_id"]),
                )
        fake_client.messages.batches.results.return_value = iter(fake_results)

        summary = process_batch_results(
            batch_id=submit_result["batch_id"],
            plan_s3_key=submit_result["plan_s3_key"],
            bucket="alpha-engine-research",
            anthropic_client=fake_client,
            s3_client=mocked_s3,
            emit_metrics=False,
        )
        assert summary["haiku_evaluated"] == 2
        assert len(summary["failed"]) == 1
        assert summary["failed"][0]["stage"] == "batch_errored"

    def test_parse_failure_recovers_via_sync_retry_tail(self, mocked_s3):
        """config#1650: a batch result whose tool output fails schema parse
        (malformed stringified dimension_scores) must get ONE synchronous
        evaluate_artifact retry — recovered evals persist, count, and leave
        the failed list EMPTY instead of silently thinning the corpus."""
        from evals.orchestrator import (
            build_batch_plan, submit_batch, process_batch_results,
        )
        from unittest.mock import patch
        from evals import orchestrator as orch
        from graph.state_schemas import (
            RubricEvalArtifact, RubricDimensionScore,
        )

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_test_retry"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch
        submit_result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )
        plan_entries = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        # ic_cio's result is malformed; the rest parse clean.
        fake_results = [
            _make_batch_result_malformed_stringified(e["custom_id"])
            if e["agent_id"] == "ic_cio"
            else _make_batch_result_succeeded(e["custom_id"], scores=[4, 4, 4])
            for e in plan_entries
        ]
        fake_client.messages.batches.results.return_value = iter(fake_results)

        retried = []

        def fake_evaluate(artifact, *, judge_run_id, judge_model,
                          judged_artifact_s3_key, **kw):
            retried.append((artifact.agent_id, judge_model))
            return RubricEvalArtifact(
                run_id=artifact.run_id,
                judge_run_id=judge_run_id,
                timestamp="2026-05-09T22:30:00Z",
                judged_agent_id=artifact.agent_id,
                rubric_id="eval_rubric_test",
                rubric_version="1.0.0",
                judge_model=judge_model,
                dimension_scores=[
                    RubricDimensionScore(dimension="d", score=4, reasoning="r"),
                ],
                overall_reasoning="ok",
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_evaluate):
            summary = process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                anthropic_client=fake_client,
                s3_client=mocked_s3,
                emit_metrics=False,
            )
        # The retry ran for exactly the failed item, with the SAME judge.
        assert retried == [("ic_cio", summary["haiku_model"])]
        assert summary["parse_retry_recovered"] == 1
        assert summary["failed"] == []
        # Recovered eval counts toward the haiku total like any other.
        assert summary["haiku_evaluated"] == 3

    def test_parse_retry_exhaustion_is_terminal_failed(self, mocked_s3):
        """If the sync retry ALSO fails, the item is terminal-failed with
        a stage naming the retry (fail-loud, run goes PARTIAL) — never
        silently dropped and never retried unboundedly."""
        from evals.orchestrator import (
            build_batch_plan, submit_batch, process_batch_results,
        )
        from unittest.mock import patch
        from evals import orchestrator as orch

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_test_retry_fail"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch
        submit_result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )
        plan_entries = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        fake_results = [
            _make_batch_result_malformed_stringified(e["custom_id"])
            if e["agent_id"] == "ic_cio"
            else _make_batch_result_succeeded(e["custom_id"], scores=[4, 4, 4])
            for e in plan_entries
        ]
        fake_client.messages.batches.results.return_value = iter(fake_results)

        with patch.object(
            orch, "evaluate_artifact",
            side_effect=RuntimeError("still non-conformant"),
        ):
            summary = process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                anthropic_client=fake_client,
                s3_client=mocked_s3,
                emit_metrics=False,
            )
        assert summary["parse_retry_recovered"] == 0
        stages = [f["stage"] for f in summary["failed"]]
        assert stages == ["batch_parse_retry"]
        assert "still non-conformant" in summary["failed"][0]["error"]
        assert "original:" in summary["failed"][0]["error"]

    def test_empty_input_skip_marker_persisted_in_submit(self, mocked_s3):
        """The empty-input artifact (sector_qual:technology with
        agent_output={}) gets a skip-marker eval written client-side
        in Submit (no batch slot consumed). Pin the persistence here
        so the rolling-mean alarm doesn't see a missing data point."""
        from evals.orchestrator import (
            build_batch_plan,
            _persist_client_side_skips,
        )

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        skip_count, degenerate_count, persisted, failed = _persist_client_side_skips(
            plan, s3=mocked_s3, bucket="alpha-engine-research",
        )
        assert skip_count == 1
        assert degenerate_count == 0
        assert len(failed) == 0
        # The skip-marker eval is at decision_artifacts/_eval/{date}/
        # Canonical flat layout (config#793):
        # {prefix}{judge_run_id}_{agent_id}.{run_id}.{judge_model}.json
        # where judge_run_id is a YYMMDDHHMM timestamp. We pin the
        # agent_id + judge_model basename segments only.
        assert any(
            "/_eval/" in k
            and "_sector_qual:technology." in k
            and ".claude-haiku-4-5.json" in k
            for k in persisted
        )


# ── End-to-end Submit → Poll → Process integration ───────────────────────


class TestBatchChainIntegration:
    def test_submit_then_poll_then_process_round_trips_eval_artifacts(
        self, mocked_s3,
    ):
        """Pin the full round-trip: plan → submit → poll → process
        produces the same per-agent eval artifacts the legacy sync
        path produced. No real Anthropic call; just shape-validates the
        in-memory wiring."""
        from evals.orchestrator import (
            build_batch_plan, submit_batch, poll_batch,
            process_batch_results, _persist_client_side_skips,
        )

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=mocked_s3,
        )
        # Persist client-side skips first (mirrors Submit Lambda flow).
        _persist_client_side_skips(
            plan, s3=mocked_s3, bucket="alpha-engine-research",
        )

        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_integration"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch
        submit_result = submit_batch(
            plan, anthropic_client=fake_client, s3_client=mocked_s3,
        )

        # Simulate one poll cycle returning ``ended``.
        fake_done_batch = MagicMock(
            spec=["processing_status", "request_counts", "ended_at"]
        )
        fake_done_batch.processing_status = "ended"
        fake_done_batch.request_counts = {"succeeded": 3}
        fake_done_batch.ended_at = "2026-05-09T22:35:00Z"
        fake_client.messages.batches.retrieve.return_value = fake_done_batch
        poll_result = poll_batch(
            batch_id=submit_result["batch_id"], anthropic_client=fake_client,
        )
        assert poll_result["processing_status"] == "ended"

        plan_entries = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        fake_client.messages.batches.results.return_value = iter([
            _make_batch_result_succeeded(e["custom_id"], scores=[4, 4, 4])
            for e in plan_entries
        ])

        summary = process_batch_results(
            batch_id=submit_result["batch_id"],
            plan_s3_key=submit_result["plan_s3_key"],
            bucket="alpha-engine-research",
            anthropic_client=fake_client,
            s3_client=mocked_s3,
            emit_metrics=False,
        )
        # 3 mapped agents → 3 Haiku evals; 1 empty-input handled via
        # client-side skip marker (counted in skipped_empty_input).
        assert summary["haiku_evaluated"] == 3
        assert summary["skipped_empty_input"] == 1
