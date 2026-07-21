"""Unit tests for the OpenRouter shadow-judge runner (config#2575 items 4-5).

Covers:
- ``run_shadow_judge_over_date`` — moto-mocked S3 end-to-end with a
  stubbed ``evaluate_artifact_openrouter``, mirroring
  ``tests/test_eval_orchestrator.py``'s ``TestEvaluateCorpus`` pattern.
- ``compute_shadow_agreement`` — manifest-driven pairing + reuse of
  ``evals.cross_validation.summarize_agreement``.
- The ``SHADOW_LOGICAL_KEYS`` refusal guard.

Real LLM/OpenRouter is never called here — see
``scripts/live_validate_openrouter_judge.py`` for the live-API validation
run (config#2575 acceptance criteria).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from graph.state_schemas import RubricDimensionScore, RubricEvalArtifact
from tests.test_eval_orchestrator import _make_capture, _make_eval

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mocked_s3_with_captures():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        prefix = "decision_artifacts/2026/05/09"
        captures = {
            f"{prefix}/sector_quant:technology/run-1.json":
                _make_capture("sector_quant:technology"),
            f"{prefix}/ic_cio/run-1.json": _make_capture("ic_cio"),
            f"{prefix}/unknown_agent_xyz/run-1.json":
                _make_capture("unknown_agent_xyz"),
        }
        for key, payload in captures.items():
            client.put_object(
                Bucket="alpha-engine-research", Key=key,
                Body=json.dumps(payload, default=str).encode("utf-8"),
            )
        yield client


class TestRunShadowJudgeOverDate:
    def test_refuses_non_shadow_judge_model(self, mocked_s3_with_captures):
        from evals.openrouter_shadow import run_shadow_judge_over_date

        with pytest.raises(ValueError, match="SHADOW_LOGICAL_KEYS"):
            run_shadow_judge_over_date(
                date="2026-05-09",
                judge_model="claude-haiku-4-5",  # NOT a shadow key
                s3_client=mocked_s3_with_captures,
                emit_metrics=False,
            )

    def test_evaluates_and_persists_mapped_artifacts(self, mocked_s3_with_captures):
        from evals import openrouter_shadow as shadow_mod

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id, run_id=artifact.run_id,
                judge_model=judge_model, scores=[4, 4, 4, 4],
            )

        with patch.object(shadow_mod, "evaluate_artifact_openrouter", side_effect=fake_eval):
            result = shadow_mod.run_shadow_judge_over_date(
                date="2026-05-09",
                s3_client=mocked_s3_with_captures,
                emit_metrics=False,
            )

        # 2 mapped agents (sector_quant, ic_cio); unknown_agent_xyz skipped.
        assert result["evaluated"] == 2
        assert result["skipped_unmapped"] == 1
        assert result["failed"] == []
        assert result["shadow_only"] is True
        assert result["judge_model"] == "openrouter-shadow"
        assert len(result["persisted_keys"]) == 2
        assert all(".openrouter-shadow.json" in k for k in result["persisted_keys"])

    def test_eval_failure_is_contained(self, mocked_s3_with_captures):
        from evals import openrouter_shadow as shadow_mod

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            if artifact.agent_id == "ic_cio":
                raise RuntimeError("simulated OpenRouter failure")
            return _make_eval(
                artifact.agent_id, run_id=artifact.run_id,
                judge_model=judge_model, scores=[4, 4, 4, 4],
            )

        with patch.object(shadow_mod, "evaluate_artifact_openrouter", side_effect=fake_eval):
            result = shadow_mod.run_shadow_judge_over_date(
                date="2026-05-09",
                s3_client=mocked_s3_with_captures,
                emit_metrics=False,
            )

        assert result["evaluated"] == 1
        assert len(result["failed"]) == 1
        assert result["failed"][0]["agent_id"] == "ic_cio"
        assert result["failed"][0]["stage"] == "eval_openrouter_shadow"

    def test_persists_to_same_prefix_as_primary_judges(self, mocked_s3_with_captures):
        """Shadow evals must be queryable ALONGSIDE Haiku/Sonnet (item 4)
        — same DEFAULT_EVAL_PREFIX, distinguished only by judge_model."""
        from evals import openrouter_shadow as shadow_mod
        from evals.judge import DEFAULT_EVAL_PREFIX

        def fake_eval(artifact, *, judge_model, judged_artifact_s3_key, **kw):
            return _make_eval(
                artifact.agent_id, run_id=artifact.run_id,
                judge_model=judge_model, scores=[4, 4, 4, 4],
            )

        with patch.object(shadow_mod, "evaluate_artifact_openrouter", side_effect=fake_eval):
            result = shadow_mod.run_shadow_judge_over_date(
                date="2026-05-09",
                s3_client=mocked_s3_with_captures,
                emit_metrics=False,
            )
        assert all(k.startswith(DEFAULT_EVAL_PREFIX) for k in result["persisted_keys"])


class TestComputeShadowAgreement:
    def test_empty_manifest_returns_empty_list(self, mocked_s3_with_captures):
        from evals.openrouter_shadow import compute_shadow_agreement

        agreements = compute_shadow_agreement(
            date="2026-05-09", s3_client=mocked_s3_with_captures,
        )
        assert agreements == []

    def test_pairs_haiku_and_shadow_scores(self, mocked_s3_with_captures):
        from evals.openrouter_shadow import compute_shadow_agreement

        client = mocked_s3_with_captures

        haiku_art = RubricEvalArtifact(
            run_id="run-1", judge_run_id="jr-1",
            timestamp="2026-05-09T22:30:00.000Z",
            judged_agent_id="ic_cio", rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0", judge_model="claude-haiku-4-5",
            dimension_scores=[
                RubricDimensionScore(dimension="d1", score=4, reasoning="r"),
                RubricDimensionScore(dimension="d2", score=3, reasoning="r"),
            ],
            overall_reasoning="ok",
        )
        shadow_art = RubricEvalArtifact(
            run_id="run-1", judge_run_id="jr-2",
            timestamp="2026-05-09T22:31:00.000Z",
            judged_agent_id="ic_cio", rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0", judge_model="openrouter-shadow",
            dimension_scores=[
                RubricDimensionScore(dimension="d1", score=4, reasoning="r"),
                RubricDimensionScore(dimension="d2", score=2, reasoning="r"),
            ],
            overall_reasoning="ok",
        )

        haiku_key = "decision_artifacts/_eval/jr-1_ic_cio.run-1.claude-haiku-4-5.json"
        shadow_key = "decision_artifacts/_eval/jr-2_ic_cio.run-1.openrouter-shadow.json"
        client.put_object(Bucket="alpha-engine-research", Key=haiku_key,
                           Body=haiku_art.model_dump_json().encode("utf-8"))
        client.put_object(Bucket="alpha-engine-research", Key=shadow_key,
                           Body=shadow_art.model_dump_json().encode("utf-8"))

        manifest = {
            "entries": [
                {"eval_s3_key": haiku_key, "judged_agent_id": "ic_cio",
                 "judged_run_id": "run-1", "judge_model": "claude-haiku-4-5"},
                {"eval_s3_key": shadow_key, "judged_agent_id": "ic_cio",
                 "judged_run_id": "run-1", "judge_model": "openrouter-shadow"},
            ],
        }
        client.put_object(
            Bucket="alpha-engine-research",
            Key="decision_artifacts/_eval_by_capture/2026-05-09/manifest.json",
            Body=json.dumps(manifest).encode("utf-8"),
        )

        agreements = compute_shadow_agreement(date="2026-05-09", s3_client=client)
        assert len(agreements) == 2  # d1, d2
        by_dim = {a.dimension: a for a in agreements}
        assert by_dim["d1"].n == 1
        assert by_dim["d1"].exact_match_rate == 1.0  # 4 == 4
        assert by_dim["d2"].n == 1
        assert by_dim["d2"].mean_abs_diff == 1.0  # |3 - 2|

    def test_skip_marker_evals_excluded(self, mocked_s3_with_captures):
        """Skip-marker evals (empty dimension_scores) must not be paired
        — nothing to compare."""
        from evals.openrouter_shadow import compute_shadow_agreement

        client = mocked_s3_with_captures
        haiku_skip = RubricEvalArtifact(
            run_id="run-1", judge_run_id="jr-1",
            timestamp="2026-05-09T22:30:00.000Z",
            judged_agent_id="ic_cio", rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0", judge_model="claude-haiku-4-5",
            dimension_scores=[], overall_reasoning="skipped",
            judge_skip_reason="precluded_by_empty_upstream",
        )
        shadow_art = RubricEvalArtifact(
            run_id="run-1", judge_run_id="jr-2",
            timestamp="2026-05-09T22:31:00.000Z",
            judged_agent_id="ic_cio", rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0", judge_model="openrouter-shadow",
            dimension_scores=[
                RubricDimensionScore(dimension="d1", score=4, reasoning="r"),
            ],
            overall_reasoning="ok",
        )
        haiku_key = "decision_artifacts/_eval/jr-1_ic_cio.run-1.claude-haiku-4-5.json"
        shadow_key = "decision_artifacts/_eval/jr-2_ic_cio.run-1.openrouter-shadow.json"
        client.put_object(Bucket="alpha-engine-research", Key=haiku_key,
                           Body=haiku_skip.model_dump_json().encode("utf-8"))
        client.put_object(Bucket="alpha-engine-research", Key=shadow_key,
                           Body=shadow_art.model_dump_json().encode("utf-8"))
        manifest = {
            "entries": [
                {"eval_s3_key": haiku_key, "judge_model": "claude-haiku-4-5"},
                {"eval_s3_key": shadow_key, "judge_model": "openrouter-shadow"},
            ],
        }
        client.put_object(
            Bucket="alpha-engine-research",
            Key="decision_artifacts/_eval_by_capture/2026-05-09/manifest.json",
            Body=json.dumps(manifest).encode("utf-8"),
        )
        agreements = compute_shadow_agreement(date="2026-05-09", s3_client=client)
        assert agreements == []
