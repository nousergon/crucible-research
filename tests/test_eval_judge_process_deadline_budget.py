"""Deadline discipline for EvalJudgeProcess (alpha-engine-config-I6920).

``alpha-engine-research-eval-judge-process`` was killed at its 900s wall in
9 of 28 observed real invocations (32%), every one of them inside
``process_batch_results``'s parse-retry tail. Measured on the live function
2026-07-26: each synchronous re-judge took 45-105s against a cap whose whole
justification was "40 × ≲8s ≪ the Process Lambda's 15-min ceiling". 40 × 75s
is 3000s. The cap could never bind on time, and nothing in the module
consulted the clock.

Killed at the wall means no manifest build, no summary, no cause — the SF
sees ``States.Timeout`` and learns nothing about what the pass covered. So
the fix is not a bigger ceiling: it is sizing the work to the budget.

These tests pin the four properties that makes true:

  1. The estimator measures the workload rather than trusting a literal,
     and each phase estimates from its OWN items.
  2. Each of the three loops stops before the deadline instead of being
     killed, and keeps the work already done.
  3. A truncated pass says it is truncated — ``complete=False`` plus an
     exact residue count — in a field a MACHINE reads, not only a log line
     (``sf-pipeline-policy.md`` §2.3a: a missing verdict propagates as
     UNKNOWN, never as pass).
  4. Callers with no deadline behave exactly as they did before.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from evals.orchestrator import (
    PROCESS_LLM_ITEM_FLOOR_S,
    PROCESS_STREAM_ITEM_FLOOR_S,
    PROCESS_WRITE_RESERVE_S,
    _next_item_affordable,
)
from tests.test_eval_orchestrator_batch import (
    _make_batch_result_malformed_stringified,
    _make_batch_result_succeeded,
    _make_capture_dict,
)


@pytest.fixture
def budget_s3():
    """Same mixed corpus as ``test_eval_orchestrator_batch::mocked_s3`` —
    3 mapped agents, 1 unmapped, 1 empty-input — built from that module's
    own capture helper.

    Declared here rather than imported: importing a fixture makes every
    test parameter that uses it a redefinition of the imported name (ruff
    F811), and silencing that on eight signatures is more noise than the
    six lines below.
    """
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        prefix = "decision_artifacts/2026/05/09"
        captures = {
            f"{prefix}/ic_cio/run-1.json": _make_capture_dict("ic_cio"),
            f"{prefix}/sector_quant:technology/run-1.json":
                _make_capture_dict("sector_quant:technology"),
            f"{prefix}/macro_economist/run-1.json":
                _make_capture_dict("macro_economist"),
            f"{prefix}/unknown_xyz/run-1.json":
                _make_capture_dict("unknown_xyz"),
            f"{prefix}/sector_qual:technology/run-1.json":
                _make_capture_dict("sector_qual:technology", agent_output={}),
        }
        for key, payload in captures.items():
            client.put_object(
                Bucket="alpha-engine-research",
                Key=key,
                Body=json.dumps(payload, default=str).encode("utf-8"),
            )
        yield client


# ── 1. The estimator ──────────────────────────────────────────────────────


class TestNextItemAffordable:
    def test_no_deadline_is_always_affordable(self):
        """The CLI, spot runs and tests pass no deadline and must be
        completely unaffected — the pre-I6920 behaviour."""
        affordable, needed = _next_item_affordable(
            None, [999_000], floor_s=PROCESS_LLM_ITEM_FLOOR_S,
        )
        assert affordable is True
        assert needed == 0.0

    def test_first_item_uses_the_floor_not_zero(self):
        """With no observations yet the estimate is the phase floor. Zero
        would let an empty sample authorise an item with no time left."""
        _affordable, needed = _next_item_affordable(
            lambda: 10.0, [], floor_s=PROCESS_LLM_ITEM_FLOOR_S,
        )
        assert needed == PROCESS_LLM_ITEM_FLOOR_S + PROCESS_WRITE_RESERVE_S

    def test_estimate_tracks_observed_p90_once_items_are_slow(self):
        """The workload measures itself. Ten observed items at 100s must
        move the estimate off the 30s floor — the exact failure of the
        literal-cap approach this replaces."""
        latencies = [100_000] * 10
        _affordable, needed = _next_item_affordable(
            lambda: 10.0, latencies, floor_s=PROCESS_LLM_ITEM_FLOOR_S,
        )
        assert needed == pytest.approx(100.0 + PROCESS_WRITE_RESERVE_S)

    def test_floor_governs_when_observed_items_are_faster(self):
        """A fast sample may not talk the estimate BELOW the floor: the
        next item is not obliged to resemble the last ones."""
        _affordable, needed = _next_item_affordable(
            lambda: 10.0, [1_000] * 5, floor_s=PROCESS_LLM_ITEM_FLOOR_S,
        )
        assert needed == PROCESS_LLM_ITEM_FLOOR_S + PROCESS_WRITE_RESERVE_S

    def test_write_reserve_is_held_back_from_every_phase(self):
        """Time left equal to one item's cost is NOT affordable — the
        manifest build and the summary still have to run, and a pass that
        spends its last second on an eval reports nothing about itself."""
        remaining = PROCESS_LLM_ITEM_FLOOR_S + 1.0
        affordable, _needed = _next_item_affordable(
            lambda: remaining, [], floor_s=PROCESS_LLM_ITEM_FLOOR_S,
        )
        assert affordable is False

    def test_stream_floor_is_lower_than_the_llm_floor(self):
        """A phase estimates from its own items. Sharing one sample would
        let the stream's sub-second parses licence a 100s judge call."""
        assert PROCESS_STREAM_ITEM_FLOOR_S < PROCESS_LLM_ITEM_FLOOR_S


# ── 2-3. The loops stop, and say so ───────────────────────────────────────


def _plan_and_results(budget_s3, *, scores):
    """Build a real plan + submit, and synthesize one batch result per
    plan entry with the given per-artifact scores."""
    from evals.orchestrator import build_batch_plan, submit_batch

    plan = build_batch_plan(
        date="2026-05-09", bucket="alpha-engine-research", s3_client=budget_s3,
    )
    fake_batch = MagicMock()
    fake_batch.id = "msgbatch_budget"
    fake_client = MagicMock()
    fake_client.messages.batches.create.return_value = fake_batch
    submit_result = submit_batch(
        plan, batch_client=fake_client, s3_client=budget_s3,
    )
    plan_entries = json.loads(
        budget_s3.get_object(
            Bucket="alpha-engine-research",
            Key=submit_result["plan_s3_key"],
        )["Body"].read()
    )["plan_entries"]
    fake_client.messages.batches.results.return_value = iter([
        _make_batch_result_succeeded(e["custom_id"], scores=scores)
        for e in plan_entries
    ])
    return fake_client, submit_result, plan_entries


class TestStreamPhaseBudget:
    def test_exhausted_budget_stops_the_stream_and_marks_incomplete(
        self, budget_s3,
    ):
        """With no time left the stream processes nothing, and the summary
        says so with an exact residue count rather than reporting a clean
        zero-failure pass."""
        from evals.orchestrator import process_batch_results

        fake_client, submit_result, plan_entries = _plan_and_results(
            budget_s3, scores=[4, 4, 4],
        )
        summary = process_batch_results(
            batch_id=submit_result["batch_id"],
            plan_s3_key=submit_result["plan_s3_key"],
            bucket="alpha-engine-research",
            batch_client=fake_client,
            s3_client=budget_s3,
            emit_metrics=False,
            remaining_s=lambda: 0.0,
        )
        assert summary["complete"] is False
        assert summary["budget_stopped"] is True
        assert summary["budget_stopped_phases"] == ["batch_stream"]
        assert summary["n_skipped_for_budget"]["batch_stream"] > 0
        assert summary["haiku_evaluated"] == 0

    def test_ample_budget_completes_and_reports_complete(self, budget_s3):
        """A deadline that is never threatened must produce exactly the
        pre-I6920 result, plus complete=True."""
        from evals.orchestrator import process_batch_results

        fake_client, submit_result, _ = _plan_and_results(
            budget_s3, scores=[4, 4, 4],
        )
        summary = process_batch_results(
            batch_id=submit_result["batch_id"],
            plan_s3_key=submit_result["plan_s3_key"],
            bucket="alpha-engine-research",
            batch_client=fake_client,
            s3_client=budget_s3,
            emit_metrics=False,
            remaining_s=lambda: 10_000.0,
        )
        assert summary["complete"] is True
        assert summary["budget_stopped"] is False
        assert summary["budget_stopped_phases"] == []
        assert summary["haiku_evaluated"] == 3
        assert summary["failed"] == []

    def test_no_deadline_matches_the_deadline_free_contract(self, budget_s3):
        """Callers passing nothing keep the old behaviour verbatim."""
        from evals.orchestrator import process_batch_results

        fake_client, submit_result, _ = _plan_and_results(
            budget_s3, scores=[4, 4, 4],
        )
        summary = process_batch_results(
            batch_id=submit_result["batch_id"],
            plan_s3_key=submit_result["plan_s3_key"],
            bucket="alpha-engine-research",
            batch_client=fake_client,
            s3_client=budget_s3,
            emit_metrics=False,
        )
        assert summary["haiku_evaluated"] == 3
        assert summary["complete"] is True


class TestParseRetryPhaseBudget:
    """The phase that actually killed the function, nine times.

    Every batch result here is malformed, so every entry lands in the
    parse-retry queue — the exact shape of the 2026-07-26 runs, whose logs
    show an unbroken sequence of `parse-retry recovered` lines running
    straight into `Status: timeout`.
    """

    def test_parse_retry_tail_stops_on_budget_and_reports_the_residue(
        self, budget_s3,
    ):
        from evals.orchestrator import (
            build_batch_plan,
            process_batch_results,
            submit_batch,
        )

        plan = build_batch_plan(
            date="2026-05-09", bucket="alpha-engine-research",
            s3_client=budget_s3,
        )
        fake_batch = MagicMock()
        fake_batch.id = "msgbatch_retry_budget"
        fake_client = MagicMock()
        fake_client.messages.batches.create.return_value = fake_batch
        submit_result = submit_batch(
            plan, batch_client=fake_client, s3_client=budget_s3,
        )
        plan_entries = json.loads(
            budget_s3.get_object(
                Bucket="alpha-engine-research",
                Key=submit_result["plan_s3_key"],
            )["Body"].read()
        )["plan_entries"]
        fake_client.messages.batches.results.return_value = iter([
            _make_batch_result_malformed_stringified(e["custom_id"])
            for e in plan_entries
        ])

        # Three stream items, then nothing fits — see the sibling test.
        budget = iter([10_000.0] * 3 + [1.0] * 500)
        with patch("evals.orchestrator.evaluate_artifact") as ev:
            summary = process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                batch_client=fake_client,
                s3_client=budget_s3,
                emit_metrics=False,
                remaining_s=lambda: next(budget),
            )

        assert summary["budget_stopped"] is True
        assert "parse_retry" in summary["budget_stopped_phases"]
        assert summary["n_skipped_for_budget"]["parse_retry"] == 3
        assert summary["complete"] is False
        # Not one paid re-judge was made past the deadline. Before this
        # change the loop would have run all three at 45-105s apiece and
        # been killed with no summary at all.
        assert ev.call_count == 0


class TestEscalationPhaseBudget:
    def test_escalation_tail_stops_on_budget_instead_of_running_unbounded(
        self, budget_s3,
    ):
        """The Sonnet tail's cardinality is DATA-dependent — every
        borderline Haiku eval adds a synchronous judge call, so a week the
        judged agents degrade is a week the tail grows. It must stop on the
        clock, keep what it did, and report the exact residue.

        The budget here is generous enough for the (cheap, mocked) stream
        but not for any escalation item, which is the real shape: the
        stream is sub-second per item and each escalation is tens of
        seconds.
        """
        from evals.orchestrator import process_batch_results

        # scores below the escalate threshold (3) put every entry in the
        # escalation queue.
        fake_client, submit_result, _ = _plan_and_results(
            budget_s3, scores=[1, 1, 1],
        )

        # The stream asks once per result and this plan yields three, so
        # the first three answers cover the stream phase in full; every
        # answer after that lands in the escalation phase, where nothing
        # fits. The parse-retry queue is empty on this path (no parse
        # failures) so it asks nothing.
        budget = iter([10_000.0] * 3 + [1.0] * 500)
        with patch("evals.orchestrator.evaluate_artifact") as ev:
            summary = process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                batch_client=fake_client,
                s3_client=budget_s3,
                emit_metrics=False,
                remaining_s=lambda: next(budget),
            )

        assert summary["budget_stopped"] is True
        assert "sonnet_escalation" in summary["budget_stopped_phases"]
        assert summary["n_skipped_for_budget"]["sonnet_escalation"] == 3
        assert summary["complete"] is False
        # The Haiku work already done is KEPT — stopping early is not
        # discarding, which is exactly what the wall-kill did.
        assert summary["haiku_evaluated"] == 3
        # And not one paid escalation call was made past the deadline.
        assert ev.call_count == 0

    def test_escalation_runs_normally_when_the_budget_allows(
        self, budget_s3,
    ):
        """The guard may not suppress the tail on a healthy run."""
        from evals.orchestrator import process_batch_results
        from graph.state_schemas import RubricEvalArtifact

        fake_client, submit_result, _ = _plan_and_results(
            budget_s3, scores=[1, 1, 1],
        )

        def _fake_eval(artifact, **kwargs):
            return RubricEvalArtifact(
                run_id="run-1",
                judge_run_id=kwargs.get("judge_run_id"),
                timestamp="2026-05-09T00:00:00Z",
                judged_agent_id=artifact.agent_id,
                judged_artifact_s3_key=kwargs.get("judged_artifact_s3_key"),
                rubric_id="r",
                rubric_version="1",
                judge_model=kwargs.get("judge_model"),
                judge_request_model="rm",
                judge_resolved_model="rm",
                dimension_scores=[],
                overall_reasoning="ok",
            )

        with patch(
            "evals.orchestrator.evaluate_artifact", side_effect=_fake_eval,
        ) as ev:
            summary = process_batch_results(
                batch_id=submit_result["batch_id"],
                plan_s3_key=submit_result["plan_s3_key"],
                bucket="alpha-engine-research",
                batch_client=fake_client,
                s3_client=budget_s3,
                emit_metrics=False,
                remaining_s=lambda: 10_000.0,
            )
        assert ev.call_count == 3
        assert summary["sonnet_evaluated"] == 3
        assert summary["complete"] is True


# ── 4. The envelope a machine reads ───────────────────────────────────────


class TestHandlerEnvelope:
    """§2.3a — visibility to an operator and propagation to a machine are
    independent properties. The Step Functions definition can only branch
    on a TOP-LEVEL field, so `complete` may not live only inside
    `$.summary`, and the dry path must carry the same keys or the Choice's
    JSONPath breaks on the Friday preflight run.
    """

    def _handler(self):
        import importlib.util
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "lambda" / "eval_judge_process_handler.py"
        )
        spec = importlib.util.spec_from_file_location(
            "eval_judge_process_handler_under_test", path,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_remaining_seconds_reads_the_lambda_context(self):
        mod = self._handler()
        ctx = MagicMock()
        ctx.get_remaining_time_in_millis.return_value = 123_000
        assert mod._remaining_seconds(ctx)() == pytest.approx(123.0)

    def test_remaining_seconds_is_none_without_a_context(self):
        """Local runs and tests get no deadline, not a crash."""
        mod = self._handler()
        assert mod._remaining_seconds(None) is None
        assert mod._remaining_seconds(object()) is None

    def test_monitor_handler_wraps_the_entry_point_not_a_helper(self):
        """The crash-capture decorator belongs on `handler`. Putting it on
        a helper leaves the real entry point unwrapped — the exact defect
        this file's sibling fix names in crucible-backtester."""
        mod = self._handler()
        assert hasattr(mod.handler, "__wrapped__")
        assert not hasattr(mod._remaining_seconds, "__wrapped__")

    def _invoke(self, summary):
        """Drive the real handler with a stubbed orchestrator so the
        envelope under test is the one the SF actually receives."""
        mod = self._handler()
        ctx = MagicMock()
        ctx.get_remaining_time_in_millis.return_value = 900_000
        with patch(
            "evals.orchestrator.process_batch_results", return_value=summary,
        ), patch(
            "evals.eval_manifest.build_manifests", return_value=[],
        ):
            return mod.handler(
                {"batch_id": "msgbatch_x", "plan_s3_key": "k.json"}, ctx,
            )

    @staticmethod
    def _summary(**over):
        base = {
            "failed": [],
            "haiku_evaluated": 2,
            "sonnet_evaluated": 0,
            "skipped_unmapped": 0,
            "skipped_empty_input": 0,
            "complete": True,
            "budget_stopped": False,
            "budget_stopped_phases": [],
            "n_skipped_for_budget": {},
        }
        base.update(over)
        return base

    def test_budget_stopped_summary_downgrades_status_to_partial(self):
        """A truncated pass with ZERO failures must not report OK — that
        is precisely how a partial sweep reads as a full one."""
        out = self._invoke(self._summary(
            complete=False, budget_stopped=True,
            budget_stopped_phases=["parse_retry"],
            n_skipped_for_budget={"parse_retry": 17},
        ))
        assert out["status"] == "PARTIAL"
        assert out["complete"] is False
        assert out["budget_stopped"] is True

    def test_complete_run_reports_ok_and_complete_at_the_top_level(self):
        out = self._invoke(self._summary())
        assert out["status"] == "OK"
        assert out["complete"] is True
        assert out["budget_stopped"] is False

    def test_dry_result_carries_the_completeness_keys(self):
        """A Choice on `$.eval_judge_result.Payload.complete` must resolve
        on the Friday preflight run too."""
        from evals.lambda_dry import dry_process_result

        result = dry_process_result("dry-batch")
        assert result["complete"] is True
        assert result["budget_stopped"] is False
        assert result["dry_run"] is True
