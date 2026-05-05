"""Unit tests for the LLM-as-judge orchestrator (PR 3b of P3.1).

Covers:
- ``should_escalate_to_sonnet`` — Haiku-tier flag-out logic.
- ``list_capture_keys`` — S3 listing of captures, exclusion of the
  ``_eval/`` subtree, and ``.json``-only filtering.
- ``evaluate_corpus`` end-to-end with moto-mocked S3 + a stubbed
  ``evaluate_artifact``: per-artifact escalation, force_sonnet_pass,
  unmapped-agent skipping, mid-batch error containment.

Real LLM is never called — all eval invocations go through a stubbed
``evaluate_artifact`` patched on the orchestrator module.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from alpha_engine_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)
from graph.state_schemas import (
    RubricDimensionScore,
    RubricEvalArtifact,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_capture(agent_id: str, run_id: str = "run-1") -> dict:
    """Build a captured-artifact dict (the JSON shape S3 stores)."""
    return DecisionArtifact(
        run_id=run_id,
        timestamp="2026-05-09T22:30:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
        full_prompt_context=FullPromptContext(
            system_prompt="<see config/prompts>",
            user_prompt="<rendered>",
        ),
        input_data_snapshot={"k": "v"},
        input_data_summary="k=v",
        agent_output={"out": "ok"},
    ).model_dump()


def _make_eval(
    judged_agent_id: str,
    *,
    run_id: str = "run-1",
    judge_model: str = "claude-haiku-4-5",
    scores: list[int] | None = None,
) -> RubricEvalArtifact:
    """Build a RubricEvalArtifact with controllable dimension scores
    so escalation logic can be exercised."""
    scores = scores if scores is not None else [4, 4, 4, 4]
    return RubricEvalArtifact(
        run_id=run_id,
        timestamp="2026-05-09T22:30:00.000Z",
        judged_agent_id=judged_agent_id,
        rubric_id="eval_rubric_test",
        rubric_version="1.0.0",
        judge_model=judge_model,
        dimension_scores=[
            RubricDimensionScore(
                dimension=f"dim_{i}", score=s, reasoning=f"r_{i}",
            )
            for i, s in enumerate(scores)
        ],
        overall_reasoning="ok",
    )


@pytest.fixture
def mocked_s3_with_captures():
    """Yields an S3 client + bucket pre-populated with a small Sat-5/9-style
    capture set: 5 mapped agent captures (sector_quant + sector_qual +
    macro_economist + ic_cio + thesis_update — the latter mapped 2026-05-05
    after the executor's behavioral dependency on conviction + bull_case
    was confirmed) + 1 unknown_agent (defensive coverage of the
    unmapped-skip path) + 1 stray non-JSON file (defensive listing case)."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")

        prefix = "decision_artifacts/2026/05/09"
        captures = {
            f"{prefix}/sector_quant:technology/run-1.json":
                _make_capture("sector_quant:technology"),
            f"{prefix}/sector_qual:technology/run-1.json":
                _make_capture("sector_qual:technology"),
            f"{prefix}/macro_economist/run-1.json":
                _make_capture("macro_economist"),
            f"{prefix}/ic_cio/run-1.json":
                _make_capture("ic_cio"),
            f"{prefix}/thesis_update:technology:AAPL/run-1.json":
                _make_capture("thesis_update:technology:AAPL"),
            f"{prefix}/unknown_agent_xyz/run-1.json":
                _make_capture("unknown_agent_xyz"),
        }
        for key, payload in captures.items():
            client.put_object(
                Bucket="alpha-engine-research", Key=key,
                Body=json.dumps(payload, default=str).encode("utf-8"),
            )

        # Stray non-JSON file under the partition — must be ignored
        # by list_capture_keys.
        client.put_object(
            Bucket="alpha-engine-research",
            Key=f"{prefix}/README.txt",
            Body=b"not an artifact",
        )

        # A pre-existing eval artifact under _eval/ — must NOT be
        # picked up as input.
        client.put_object(
            Bucket="alpha-engine-research",
            Key=f"decision_artifacts/_eval/2026-05-09/ic_cio/run-prev.claude-haiku-4-5.json",
            Body=b"{}",
        )

        yield client


# ── should_escalate_to_sonnet ─────────────────────────────────────────────


class TestShouldEscalateToSonnet:
    def test_no_dimension_below_threshold(self):
        from evals.orchestrator import should_escalate_to_sonnet
        eval_ = _make_eval("ic_cio", scores=[5, 4, 3, 4])
        assert should_escalate_to_sonnet(eval_, threshold=3) is False

    def test_one_dimension_below_threshold(self):
        from evals.orchestrator import should_escalate_to_sonnet
        eval_ = _make_eval("ic_cio", scores=[5, 4, 2, 4])
        assert should_escalate_to_sonnet(eval_, threshold=3) is True

    def test_threshold_is_strict_less_than(self):
        # score == threshold should NOT escalate (the rubric midpoint
        # is acceptable; only "below 3" triggers).
        from evals.orchestrator import should_escalate_to_sonnet
        eval_ = _make_eval("ic_cio", scores=[3, 3, 3, 3])
        assert should_escalate_to_sonnet(eval_, threshold=3) is False

    def test_custom_threshold(self):
        from evals.orchestrator import should_escalate_to_sonnet
        eval_ = _make_eval("ic_cio", scores=[4, 4, 4, 4])
        # Tighter gate: below 5 → escalate.
        assert should_escalate_to_sonnet(eval_, threshold=5) is True


# ── list_capture_keys ─────────────────────────────────────────────────────


class TestListCaptureKeys:
    def test_lists_only_capture_jsons(self, mocked_s3_with_captures):
        from evals.orchestrator import list_capture_keys
        keys = list_capture_keys(
            mocked_s3_with_captures,
            date="2026-05-09",
            bucket="alpha-engine-research",
        )
        # 6 capture artifacts (5 mapped + 1 unknown_agent); README.txt
        # + _eval/ entry both excluded.
        assert len(keys) == 6
        assert all(k.endswith(".json") for k in keys)
        assert all("/_eval/" not in k for k in keys)
        assert all(k.startswith("decision_artifacts/2026/05/09/") for k in keys)


# ── evaluate_corpus ───────────────────────────────────────────────────────


class TestEvaluateCorpus:
    def test_haiku_only_when_no_escalation(self, mocked_s3_with_captures):
        """All Haiku scores are 4; no force_sonnet_pass → only Haiku
        evals are written. Sonnet count = 0."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        # 5 mapped agents (sector_quant + sector_qual + macro_economist +
        # ic_cio + thesis_update); unknown_agent_xyz is unmapped → skipped.
        assert result["haiku_evaluated"] == 5
        assert result["sonnet_evaluated"] == 0
        assert result["skipped_unmapped"] == 1
        assert result["failed"] == []
        assert len(result["persisted_keys"]) == 5
        assert all(
            ".claude-haiku-4-5.json" in k for k in result["persisted_keys"]
        )

    def test_force_sonnet_pass_runs_both_tiers(self, mocked_s3_with_captures):
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[5, 5, 5, 5],  # No per-artifact escalation reason.
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                force_sonnet_pass=True,
                s3_client=mocked_s3_with_captures,
            )

        # Every mapped artifact gets BOTH Haiku and Sonnet evals.
        assert result["haiku_evaluated"] == 5
        assert result["sonnet_evaluated"] == 5
        assert len(result["persisted_keys"]) == 10
        haiku_keys = [k for k in result["persisted_keys"] if "claude-haiku-4-5" in k]
        sonnet_keys = [k for k in result["persisted_keys"] if "claude-sonnet-4-6" in k]
        assert len(haiku_keys) == 5
        assert len(sonnet_keys) == 5

    def test_per_artifact_escalation_when_haiku_score_below_threshold(
        self, mocked_s3_with_captures,
    ):
        """ic_cio gets a Haiku score of 2 → escalates to Sonnet.
        Other agents stay at 4 → no escalation."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            scores = [2, 4, 4, 4] if artifact.agent_id == "ic_cio" else [4, 4, 4, 4]
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=scores,
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        assert result["haiku_evaluated"] == 5
        assert result["sonnet_evaluated"] == 1
        # The Sonnet escalation should be for ic_cio specifically.
        sonnet_keys = [k for k in result["persisted_keys"] if "claude-sonnet-4-6" in k]
        assert len(sonnet_keys) == 1
        assert "/ic_cio/" in sonnet_keys[0]

    def test_haiku_failure_is_contained(self, mocked_s3_with_captures):
        """LLM raising on one artifact must not halt evaluation of others."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            if artifact.agent_id == "macro_economist":
                raise RuntimeError("anthropic 5xx")
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        # 4 succeed (sector_quant + sector_qual + ic_cio + thesis_update);
        # macro fails.
        assert result["haiku_evaluated"] == 4
        assert result["sonnet_evaluated"] == 0
        assert len(result["failed"]) == 1
        assert result["failed"][0]["agent_id"] == "macro_economist"
        assert result["failed"][0]["stage"] == "haiku"
        assert "anthropic 5xx" in result["failed"][0]["error"]

    def test_sonnet_failure_does_not_invalidate_haiku(self, mocked_s3_with_captures):
        """If Haiku succeeds + Sonnet fails, the Haiku eval is still
        persisted; only Sonnet is marked failed."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            if judge_model == "claude-sonnet-4-6":
                raise RuntimeError("anthropic 429")
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[2, 4, 4, 4],  # triggers escalation
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        # Every mapped artifact has Haiku score 2 → all 5 escalate.
        assert result["haiku_evaluated"] == 5
        assert result["sonnet_evaluated"] == 0
        # All 5 Sonnet attempts failed; no Haiku failures.
        sonnet_failures = [f for f in result["failed"] if f["stage"] == "sonnet"]
        haiku_failures = [f for f in result["failed"] if f["stage"] == "haiku"]
        assert len(sonnet_failures) == 5
        assert len(haiku_failures) == 0

    def test_skipped_unmapped_agents_not_in_failed(
        self, mocked_s3_with_captures,
    ):
        """unknown_agent_xyz has no rubric; it should be counted as
        skipped, NOT as a failure. (thesis_update:* was unmapped pre
        2026-05-05; rubric shipped that day after confirming executor's
        behavioral dependency, so the unmapped-skip path is now
        exercised by the unknown_agent fixture entry.)"""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        assert result["skipped_unmapped"] == 1
        assert result["failed"] == []

    def test_skipped_empty_input_counted_separately(
        self, mocked_s3_with_captures,
    ):
        """When the judge short-circuits on empty agent_output, the
        eval persists with judge_skip_reason set + dimension_scores=[].
        The orchestrator must (a) count it under skipped_empty_input,
        (b) NOT escalate to Sonnet, (c) NOT mark it as failed."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            # Simulate the empty-input short-circuit on sector_qual —
            # judge.evaluate_artifact would return a skip-marker eval.
            if artifact.agent_id.startswith("sector_qual"):
                ev = _make_eval(
                    artifact.agent_id,
                    run_id=artifact.run_id,
                    judge_model=judge_model,
                    scores=[],  # empty dimensions on skip path
                )
                # Manually set the skip reason post-construction; the
                # _make_eval helper doesn't expose it.
                object.__setattr__(
                    ev, "judge_skip_reason", "precluded_by_empty_upstream",
                )
                return ev
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        # Exactly 1 sector_qual capture in the fixture set → exactly
        # 1 skipped_empty_input.
        assert result["skipped_empty_input"] == 1
        # The skipped eval still counts under haiku_evaluated (it WAS
        # processed end-to-end, just without an LLM call). What we
        # verify here is that Sonnet did NOT escalate on it — that
        # would be wasted work scoring an empty artifact at Sonnet's
        # cost tier.
        assert result["sonnet_evaluated"] == 0
        assert result["failed"] == []

    def test_emit_metrics_false_skips_cloudwatch(
        self, mocked_s3_with_captures,
    ):
        """`emit_metrics=False` should completely bypass CloudWatch —
        no client built, no failures counted."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        emit_calls = []

        def fake_emit(*args, **kwargs):
            emit_calls.append((args, kwargs))

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval), \
             patch.object(orch, "emit_eval_metric", side_effect=fake_emit):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
                emit_metrics=False,
            )

        assert emit_calls == []
        assert result["metric_emission_failures"] == 0

    def test_metric_emission_called_per_persisted_eval(
        self, mocked_s3_with_captures,
    ):
        """Default ``emit_metrics=True``: every successful persist should
        trigger one ``emit_eval_metric`` call. With 5 mapped agents +
        no escalation, that's 5 emission calls."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        emit_calls = []

        def fake_emit(eval_artifact, **kwargs):
            emit_calls.append(eval_artifact.judged_agent_id)

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval), \
             patch.object(orch, "emit_eval_metric", side_effect=fake_emit):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        assert len(emit_calls) == 5
        assert result["metric_emission_failures"] == 0
        assert sorted(emit_calls) == sorted([
            "sector_quant:technology",
            "sector_qual:technology",
            "macro_economist",
            "ic_cio",
            "thesis_update:technology:AAPL",
        ])

    def test_metric_emission_failure_does_not_halt_run(
        self, mocked_s3_with_captures,
    ):
        """CloudWatch hiccup must not cascade into the eval pipeline.
        Failures get counted in the summary; persisted_keys is unchanged."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval), \
             patch.object(orch, "emit_eval_metric",
                          side_effect=RuntimeError("CW throttled")):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        assert result["haiku_evaluated"] == 5
        assert result["metric_emission_failures"] == 5
        assert result["failed"] == []  # not promoted to artifact-level failure
        assert len(result["persisted_keys"]) == 5

    def test_dry_run_skips_llm_persist_and_metrics(
        self, mocked_s3_with_captures,
    ):
        """dry_run lists artifacts + resolves rubrics + populates
        ``would_evaluate``, but DOES NOT call evaluate_artifact /
        persist / emit metrics. Cost: $0."""
        from evals import orchestrator as orch

        eval_mock = MagicMock(side_effect=AssertionError(
            "evaluate_artifact must NOT be called in dry_run"
        ))
        emit_mock = MagicMock(side_effect=AssertionError(
            "emit_eval_metric must NOT be called in dry_run"
        ))

        with patch.object(orch, "evaluate_artifact", eval_mock), \
             patch.object(orch, "emit_eval_metric", emit_mock):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
                dry_run=True,
            )

        assert result["dry_run"] is True
        assert result["haiku_evaluated"] == 0
        assert result["sonnet_evaluated"] == 0
        # 5 mapped agents (sector_quant + sector_qual + macro + cio +
        # thesis_update); unknown_agent_xyz is unmapped → skipped, not
        # in would_evaluate.
        assert len(result["would_evaluate"]) == 5
        assert result["skipped_unmapped"] == 1
        assert result["persisted_keys"] == []
        # Each entry carries enough info to manually verify what would
        # have run before paying for real LLM.
        for entry in result["would_evaluate"]:
            assert "key" in entry
            assert "agent_id" in entry
            assert "rubric" in entry

    def test_judge_only_redirects_persist_to_isolated_prefix(
        self, mocked_s3_with_captures,
    ):
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
                judge_only=True,
            )

        assert result["judge_only"] is True
        assert result["eval_prefix"] == "decision_artifacts/_eval_judge_only/"
        # Every persisted key lands under the isolated prefix.
        for key in result["persisted_keys"]:
            assert key.startswith("decision_artifacts/_eval_judge_only/")

    def test_judge_only_redirects_cw_namespace(self, mocked_s3_with_captures):
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        emit_calls = []

        def fake_emit(eval_artifact, *, namespace=None, **kwargs):
            emit_calls.append(namespace)

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval), \
             patch.object(orch, "emit_eval_metric", side_effect=fake_emit):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
                judge_only=True,
            )

        assert result["cw_namespace"] == "AlphaEngine/EvalJudgeOnly"
        assert all(ns == "AlphaEngine/EvalJudgeOnly" for ns in emit_calls)
        assert "AlphaEngine/Eval" not in emit_calls

    def test_dry_run_and_judge_only_compose(self, mocked_s3_with_captures):
        """The cheapest end-to-end smoke: lists + renders against prod
        captures, no LLM, no writes anywhere — and the result still
        carries the judge_only namespace + prefix flags so operators
        can confirm WHERE the run WOULD have written if dry_run were
        flipped off."""
        from evals import orchestrator as orch

        eval_mock = MagicMock(side_effect=AssertionError("must not be called"))

        with patch.object(orch, "evaluate_artifact", eval_mock):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
                dry_run=True,
                judge_only=True,
            )

        assert result["dry_run"] is True
        assert result["judge_only"] is True
        assert result["eval_prefix"] == "decision_artifacts/_eval_judge_only/"
        assert result["cw_namespace"] == "AlphaEngine/EvalJudgeOnly"
        assert result["haiku_evaluated"] == 0
        assert len(result["would_evaluate"]) == 5

    def test_default_keeps_prod_paths(self, mocked_s3_with_captures):
        """Default invocation (no flags) must NOT redirect anything —
        prod Saturday SF runs depend on these defaults."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        assert result["dry_run"] is False
        assert result["judge_only"] is False
        assert result["eval_prefix"] == "decision_artifacts/_eval/"
        assert result["cw_namespace"] == "AlphaEngine/Eval"
        for key in result["persisted_keys"]:
            assert key.startswith("decision_artifacts/_eval/")

    def test_persisted_keys_reflect_two_tier_naming(
        self, mocked_s3_with_captures,
    ):
        """Persisted keys must include the judge_model segment so
        Haiku and Sonnet writes for the same artifact don't collide."""
        from evals import orchestrator as orch

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=judge_model,
                scores=[2, 4, 4, 4] if artifact.agent_id == "ic_cio" else [4, 4, 4, 4],
            )

        with patch.object(orch, "evaluate_artifact", side_effect=fake_eval):
            result = orch.evaluate_corpus(
                date="2026-05-09",
                bucket="alpha-engine-research",
                s3_client=mocked_s3_with_captures,
            )

        # Find the two ic_cio keys (Haiku + Sonnet).
        ic_cio_keys = [k for k in result["persisted_keys"] if "/ic_cio/" in k]
        assert len(ic_cio_keys) == 2
        assert any(".claude-haiku-4-5.json" in k for k in ic_cio_keys)
        assert any(".claude-sonnet-4-6.json" in k for k in ic_cio_keys)
