"""
Unit tests for the LLM-as-judge eval pipeline (PR 2 of ROADMAP P3.1).

Covers:
- ``resolve_rubric_for_agent`` — agent_id → rubric_name mapping (and
  the intentionally-unevaluated cases).
- ``build_eval_s3_key`` — canonical S3 path layout.
- ``evaluate_artifact`` — end-to-end with a mocked judge LLM, asserting
  the rendered prompt, the wrapped artifact metadata, and the cost
  tracker integration.
- ``persist_eval_artifact`` — moto-mocked S3 round-trip, including
  re-validating from S3 bytes.

The judge LLM is mocked across all eval tests — we don't make real
Anthropic calls in the unit suite. Real-LLM smoke tests live with the
SF-wiring PR.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from nousergon_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)
from graph.state_schemas import (
    RubricDimensionScore,
    RubricEvalArtifact,
    RubricEvalLLMOutput,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_artifact(agent_id: str, *, run_id: str = "test-run-001") -> DecisionArtifact:
    """Build a DecisionArtifact with shape-realistic input + output."""
    return DecisionArtifact(
        run_id=run_id,
        timestamp="2026-05-09T22:30:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
        full_prompt_context=FullPromptContext(
            system_prompt="<see config/prompts>",
            user_prompt="<rendered at run time>",
        ),
        input_data_snapshot={
            "team_id": "technology",
            "run_date": "2026-05-09",
            "market_regime": "neutral",
            "sector_tickers": ["AAPL", "MSFT"],
            "technical_scores_team": {
                "AAPL": {"rsi_14": 55, "technical_score": 70},
                "MSFT": {"rsi_14": 50, "technical_score": 65},
            },
        },
        input_data_summary="team_id=technology, sector_tickers=2",
        agent_output={
            "ranked_picks": [
                {"ticker": "AAPL", "quant_score": 70, "quant_rationale": "RSI 55, TS 70."},
                {"ticker": "MSFT", "quant_score": 65, "quant_rationale": "RSI 50, TS 65."},
            ],
        },
    )


def _make_llm_output() -> RubricEvalLLMOutput:
    """Build a realistic judge response."""
    return RubricEvalLLMOutput(
        dimension_scores=[
            RubricDimensionScore(
                dimension="numerical_grounding", score=4,
                reasoning="Both picks cite specific RSI + TS values.",
            ),
            RubricDimensionScore(
                dimension="signal_calibration", score=3,
                reasoning="Score gradient is directional but tight.",
            ),
            RubricDimensionScore(
                dimension="ranking_coherence", score=4,
                reasoning="Rank matches scores; reasoning differentiates picks.",
            ),
            RubricDimensionScore(
                dimension="regime_awareness", score=3,
                reasoning="Regime mentioned once but doesn't shape picks.",
            ),
            RubricDimensionScore(
                dimension="reasoning_complexity", score=2,
                reasoning="Threshold-summing pattern; reproducible by short script.",
            ),
            RubricDimensionScore(
                dimension="output_completeness", score=4,
                reasoning="2 picks emitted with full rationales — adequate coverage.",
            ),
        ],
        overall_reasoning="Solid grounding; regime engagement weakest.",
    )


# ── Rubric mapping ────────────────────────────────────────────────────────


class TestResolveRubricForAgent:
    def test_sector_quant_with_team(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("sector_quant:technology") == "eval_rubric_sector_quant"
        assert resolve_rubric_for_agent("sector_quant:financials") == "eval_rubric_sector_quant"

    def test_sector_qual_with_team(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("sector_qual:healthcare") == "eval_rubric_sector_qual"

    def test_sector_peer_review_with_team(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("sector_peer_review:industrials") == "eval_rubric_sector_peer_review"

    def test_macro_economist_exact_match(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("macro_economist") == "eval_rubric_macro_economist"

    def test_ic_cio_exact_match(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("ic_cio") == "eval_rubric_ic_cio"

    def test_thesis_update_with_team_and_ticker(self):
        # Held-stock thesis update rubric promoted from "deferred"
        # to "shipped" 2026-05-05 after confirming behavioral
        # load-bearing-ness in the executor (position sizing reads
        # conviction; EOD email reads bull_case). agent_id shape is
        # ``thesis_update:{team}:{ticker}`` per
        # research_graph._capture_if_enabled.
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("thesis_update:technology:AAPL") == "eval_rubric_thesis_update"
        assert resolve_rubric_for_agent("thesis_update:financials:JHG") == "eval_rubric_thesis_update"
        assert resolve_rubric_for_agent("thesis_update:healthcare:LLY") == "eval_rubric_thesis_update"

    def test_unknown_agent_returns_none(self):
        from evals.judge import resolve_rubric_for_agent
        assert resolve_rubric_for_agent("totally_made_up_agent") is None
        assert resolve_rubric_for_agent("") is None


# ── S3 key shape ──────────────────────────────────────────────────────────


class TestBuildEvalS3Key:
    """Pins the canonical ``alpha_engine_lib.eval_artifacts`` flat layout
    (config#793 swap): ``_eval/{judge_run_id}_{agent_id}.{run_id}.{judge_model}.json``
    where ``judge_run_id`` is a ``YYMMDDHHMM`` structured timestamp. The
    multi-file-per-run grouping prefix keeps every artifact from one
    batch grouped by the shared judge_run_id. No date sub-partition —
    the timestamp-encoded run_id makes it redundant (lib rationale)."""

    def test_canonical_flat_path_groups_on_judge_run_id(self):
        from evals.judge import build_eval_s3_key
        key = build_eval_s3_key(
            judged_agent_id="sector_quant:technology",
            run_id="run-abc-123",
            judge_run_id="2605092230",
            judge_model="claude-haiku-4-5",
        )
        # Flat: {prefix}{judge_run_id}_{agent_id}.{run_id}.{judge_model}.json
        assert key == (
            "decision_artifacts/_eval/2605092230_"
            "sector_quant:technology.run-abc-123.claude-haiku-4-5.json"
        )

    def test_uses_lib_helper_as_single_source_of_truth(self):
        """The key is built by alpha_engine_lib.eval_artifacts.eval_artifact_key
        — we do NOT hand-roll the format. Pin the equivalence so a drift
        in either side is caught."""
        from nousergon_lib.eval_artifacts import eval_artifact_key
        from evals.judge import build_eval_s3_key, DEFAULT_EVAL_PREFIX
        key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605092230",
            judge_model="claude-haiku-4-5",
        )
        expected = eval_artifact_key(
            DEFAULT_EVAL_PREFIX, "2605092230",
            basename="ic_cio.r1.claude-haiku-4-5.json",
        )
        assert key == expected

    def test_judge_run_id_required(self):
        """Empty string raises — production paths must mint a run_id per
        batch and propagate. Solo callers go through evaluate_artifact
        which defaults one; build_eval_s3_key itself is strict."""
        from evals.judge import build_eval_s3_key
        with pytest.raises(ValueError, match="judge_run_id"):
            build_eval_s3_key(
                judged_agent_id="ic_cio", run_id="r1",
                judge_run_id="",
                judge_model="claude-haiku-4-5",
            )

    def test_no_date_subpartition(self):
        from evals.judge import build_eval_s3_key
        key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605092230",
            judge_model="claude-haiku-4-5",
        )
        # Flat — the relative key under the prefix has no further "/".
        rel = key[len("decision_artifacts/_eval/"):]
        assert "/" not in rel
        assert rel.startswith("2605092230_")

    def test_judge_model_disambiguates_two_tier(self):
        """Haiku + Sonnet of same (judge_run_id, agent_id, run_id) must
        coexist — the judge_model segment in the basename is what keeps
        the two writes from clobbering each other."""
        from evals.judge import build_eval_s3_key
        haiku_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605092230",
            judge_model="claude-haiku-4-5",
        )
        sonnet_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605092230",
            judge_model="claude-sonnet-4-6",
        )
        assert haiku_key != sonnet_key
        assert haiku_key.endswith(".claude-haiku-4-5.json")
        assert sonnet_key.endswith(".claude-sonnet-4-6.json")

    def test_batch_cohesion_same_judge_run_id_grouped(self):
        """Institutional invariant carried into the flat layout: every
        artifact emitted by ONE batch invocation shares the SAME
        judge_run_id prefix, even across different judged_agent_ids and
        run_ids. Operator query 'show me batch X's outputs' =
        `aws s3 ls _eval/ | grep {judge_run_id}`."""
        from evals.judge import build_eval_s3_key
        keys = [
            build_eval_s3_key(
                judged_agent_id=aid, run_id=rid,
                judge_run_id="2605092230",
                judge_model="claude-haiku-4-5",
            )
            for aid, rid in [
                ("sector_quant:technology", "agent-run-1"),
                ("ic_cio", "agent-run-2"),
                ("thesis_update:financials:CBOE", "agent-run-3"),
            ]
        ]
        common_prefix = "decision_artifacts/_eval/2605092230_"
        for k in keys:
            assert k.startswith(common_prefix)

    def test_different_batches_different_run_id_prefix(self):
        """Re-judging an artifact in a separate batch lands under a
        DIFFERENT judge_run_id prefix — preserves audit history of
        re-runs (vs an overwrite-on-rerun shape)."""
        from evals.judge import build_eval_s3_key
        original = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605092230",
            judge_model="claude-haiku-4-5",
        )
        rerun = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605100915",
            judge_model="claude-haiku-4-5",
        )
        assert original != rerun
        assert "/2605092230_" in original
        assert "/2605100915_" in rerun

    def test_prefix_override_for_judge_only_mode(self):
        """``judge_only=True`` test runs persist under a non-prod prefix."""
        from evals.judge import build_eval_s3_key
        prod_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605092230",
            judge_model="claude-haiku-4-5",
        )
        test_key = build_eval_s3_key(
            judged_agent_id="ic_cio", run_id="r1",
            judge_run_id="2605092230",
            judge_model="claude-haiku-4-5",
            prefix="decision_artifacts/_eval_judge_only/",
        )
        assert prod_key.startswith("decision_artifacts/_eval/")
        assert test_key.startswith("decision_artifacts/_eval_judge_only/")
        assert prod_key != test_key


class TestNewJudgeRunId:
    """``_new_judge_run_id`` delegates to the lib's ``new_eval_run_id``
    (config#793) — YYMMDDHHMM structured timestamp, not a UUID."""

    def test_returns_yymmddhhmm_shape(self):
        from evals.judge import _new_judge_run_id
        rid = _new_judge_run_id()
        assert rid.isdigit()
        assert len(rid) == 10

    def test_sortable_chronologically(self):
        from datetime import datetime, timezone
        from nousergon_lib.eval_artifacts import new_eval_run_id
        earlier = new_eval_run_id(now=datetime(2026, 5, 9, 22, 30, tzinfo=timezone.utc))
        later = new_eval_run_id(now=datetime(2026, 5, 10, 9, 15, tzinfo=timezone.utc))
        assert earlier < later  # lexicographic = chronological


class TestBuildLegacyEvalS3Key:
    """Backward-compat: the legacy nested Option B key builder is retained
    so readers/tests can reconstruct pre-config#793 historical paths."""

    def test_legacy_nested_shape(self):
        from evals.judge import build_legacy_eval_s3_key
        ts = datetime(2026, 5, 9, 22, 30, tzinfo=timezone.utc)
        key = build_legacy_eval_s3_key(
            judged_agent_id="sector_quant:technology",
            run_id="run-abc-123",
            judge_run_id="batch-uuid-xyz",
            judge_model="claude-haiku-4-5",
            timestamp=ts,
        )
        assert key == (
            "decision_artifacts/_eval/2026-05-09/batch-uuid-xyz/"
            "sector_quant:technology.run-abc-123.claude-haiku-4-5.json"
        )


# ── evaluate_artifact end-to-end ──────────────────────────────────────────


class TestEvaluateArtifact:
    def test_unmapped_agent_raises(self):
        from evals.judge import evaluate_artifact
        artifact = _make_artifact("totally_made_up_agent")
        with pytest.raises(ValueError, match="No rubric mapped"):
            evaluate_artifact(artifact)

    def test_full_pipeline_with_mocked_llm(self, monkeypatch):
        from evals import judge as judge_mod

        # PR #106: with_structured_output(include_raw=True) returns a
        # runnable whose .invoke() yields a dict with parsed/raw/parsing_error.
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "parsed": _make_llm_output(),
            "raw": MagicMock(content="ok"),
            "parsing_error": None,
        }

        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(
            judge_mod, "ChatAnthropic", return_value=fake_llm,
        ) as mock_anthropic:
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact,
                judge_model="claude-haiku-4-5",
                api_key="sk-test",
                judged_artifact_s3_key="decision_artifacts/2026/05/09/sector_quant:technology/r1.json",
            )

        # Result wrapping
        assert isinstance(result, RubricEvalArtifact)
        assert result.judged_agent_id == "sector_quant:technology"
        assert result.run_id == artifact.run_id
        assert result.rubric_id == "eval_rubric_sector_quant"
        # judge_model is the STABLE logical key (persistence/dimension)...
        assert result.judge_model == "claude-haiku-4-5"
        # ...while the API call is PINNED to the dated snapshot (L4578(a)).
        assert mock_anthropic.call_args.kwargs["model"] == (
            "claude-haiku-4-5-20251001"
        )
        assert result.judge_request_model == "claude-haiku-4-5-20251001"
        # MagicMock raw has no dict response_metadata → resolved is None,
        # not a crash (defensive capture).
        assert result.judge_resolved_model is None
        assert result.judged_artifact_s3_key.endswith("/r1.json")
        # Rubric version comes from the loaded prompt's frontmatter; we
        # don't pin to a specific semver here so prompt updates don't
        # break this test.
        assert result.rubric_version  # non-empty
        # Dimension scores propagated
        assert len(result.dimension_scores) == 6
        assert result.dimension_scores[0].dimension == "numerical_grounding"
        # Overall reasoning propagated
        assert "regime engagement" in result.overall_reasoning
        # First-attempt success — should NOT have called invoke more than once
        assert fake_structured.invoke.call_count == 1
        # And must have requested include_raw=True
        fake_llm.with_structured_output.assert_called_once()
        kwargs = fake_llm.with_structured_output.call_args.kwargs
        assert kwargs.get("include_raw") is True

    def test_records_resolved_model_from_response_metadata(self):
        """L4578(a): the API-resolved model is captured per-artifact for
        drift detection / the re-anchor protocol."""
        from evals import judge as judge_mod

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "parsed": _make_llm_output(),
            # langchain_anthropic populates response_metadata['model']
            # with the snapshot Anthropic resolved the request to.
            "raw": MagicMock(
                response_metadata={"model": "claude-haiku-4-5-20251001"},
            ),
            "parsing_error": None,
        }
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            result = judge_mod.evaluate_artifact(
                _make_artifact("sector_quant:technology"),
                judge_model="claude-haiku-4-5",
                api_key="sk-test",
            )
        assert result.judge_resolved_model == "claude-haiku-4-5-20251001"

    def test_renders_artifact_payload_into_prompt(self, monkeypatch):
        """Verify the rubric prompt is rendered with the artifact's
        input_data_snapshot + agent_output, not placeholder strings."""
        from evals import judge as judge_mod

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "parsed": _make_llm_output(),
            "raw": MagicMock(content="ok"),
            "parsing_error": None,
        }
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            artifact = _make_artifact("sector_quant:technology")
            judge_mod.evaluate_artifact(artifact, api_key="sk-test")

        # Inspect the rendered prompt passed to invoke.
        call_args = fake_structured.invoke.call_args
        messages = call_args[0][0]
        rendered = messages[0].content
        # Specific values from the snapshot must appear in the rendered prompt
        assert "AAPL" in rendered
        assert "technology" in rendered
        # And specific values from the agent output
        assert "ranked_picks" in rendered
        assert "RSI 55, TS 70." in rendered
        # Substitution variables should NOT remain unrendered
        assert "{agent_input}" not in rendered
        assert "{agent_output}" not in rendered

    def test_retries_on_parse_failure_and_succeeds(self, monkeypatch, caplog):
        """First attempt returns parsing_error; second attempt succeeds.
        Pin: retry loop catches the parse failure, logs WARNING with
        raw payload head, retries, returns the parsed result. Counts
        2 invokes total."""
        import logging
        from evals import judge as judge_mod
        from pydantic import ValidationError

        bad_resp = {
            "parsed": None,
            "raw": MagicMock(content='[{"dimension": "x", "score": 4'),
            "parsing_error": ValidationError.from_exception_data(
                "RubricEvalLLMOutput", []
            ),
        }
        good_resp = {
            "parsed": _make_llm_output(),
            "raw": MagicMock(content="ok"),
            "parsing_error": None,
        }

        fake_structured = MagicMock()
        fake_structured.invoke.side_effect = [bad_resp, good_resp]
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm), \
             patch("time.sleep"), \
             caplog.at_level(logging.WARNING):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact, judge_model="claude-haiku-4-5", api_key="sk-test",
            )

        assert isinstance(result, RubricEvalArtifact)
        assert fake_structured.invoke.call_count == 2
        # Loud-log the parse failure with raw head for diagnostic.
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("parse attempt 1/3 failed" in m for m in warn_msgs)
        assert any("raw head=" in m for m in warn_msgs)

    def test_raises_after_max_retries_exhausted(self, monkeypatch):
        """All 3 attempts return parsing_error → raises RuntimeError
        with diagnostic context. Bounds worst-case latency + makes
        structural failures loud."""
        from evals import judge as judge_mod
        from pydantic import ValidationError

        bad_resp = {
            "parsed": None,
            "raw": MagicMock(content="malformed"),
            "parsing_error": ValidationError.from_exception_data(
                "RubricEvalLLMOutput", []
            ),
        }

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = bad_resp
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm), \
             patch("time.sleep"):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError, match="parse attempts failed"):
                judge_mod.evaluate_artifact(
                    artifact, judge_model="claude-haiku-4-5", api_key="sk-test",
                )

        # All 3 attempts fired (default MAX_JUDGE_RETRIES=3).
        assert fake_structured.invoke.call_count == 3

    def test_retry_count_param_overrides_default(self, monkeypatch):
        """``max_retries`` param lets callers tune the budget — useful
        for the test-track flag and for ad-hoc replay scripts that
        want to fail fast."""
        from evals import judge as judge_mod
        from pydantic import ValidationError

        bad_resp = {
            "parsed": None,
            "raw": MagicMock(content="x"),
            "parsing_error": ValidationError.from_exception_data(
                "RubricEvalLLMOutput", []
            ),
        }

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = bad_resp
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm), \
             patch("time.sleep"):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError):
                judge_mod.evaluate_artifact(
                    artifact, max_retries=1, api_key="sk-test",
                )

        # Single attempt only.
        assert fake_structured.invoke.call_count == 1

    def test_recovers_deterministic_tool_xml_leak_via_correction_feedback(self):
        """config#2237: the judge routes through the shared
        ``invoke_structured_with_validation_retry`` chokepoint (was a bespoke
        bare re-roll). A DETERMINISTIC tool-XML leak — the model emitting a
        literal ``<parameter name="dimension_scores">`` tag, captured by
        langchain as a ``str`` where ``list[RubricDimensionScore]`` is required
        — re-rolled IDENTICALLY under the old loop and could never recover (the
        2026-07-11 CRUS-class hard-fail). The chokepoint feeds the Pydantic
        ``ValidationError`` plus the model's own prior malformed output back as
        correction context, so a later attempt corrects the field. Mirrors
        ``test_held_thesis_strict::test_reroll_recovers_when_a_later_attempt_valid``.
        """
        from evals import judge as judge_mod
        from graph.state_schemas import RubricEvalLLMOutput

        # The deterministic leak: a literal tool-XML tag string where a list is
        # required. Constructing the schema with it raises the SAME
        # ValidationError every attempt — a bare re-roll can't fix it; only the
        # correction-feedback retry can.
        leaked_xml = '\n<parameter name="dimension_scores">numerical_grounding: 4\n'

        def _leak_response():
            try:
                RubricEvalLLMOutput(
                    dimension_scores=leaked_xml, overall_reasoning="x",
                )
                raise AssertionError(
                    "expected RubricEvalLLMOutput to reject a str dimension_scores"
                )
            except Exception as exc:  # pydantic.ValidationError
                parsing_error = exc
            return {"raw": object(), "parsed": None, "parsing_error": parsing_error}

        def _valid_response():
            return {
                "raw": MagicMock(
                    response_metadata={"model": "claude-haiku-4-5-20251001"},
                ),
                "parsed": _make_llm_output(),
                "parsing_error": None,
            }

        class _RecordingStructuredLLM:
            def __init__(self, responses):
                self._responses = responses
                self.calls = []  # captured message list per invoke

            def invoke(self, messages, config=None):
                self.calls.append(messages)
                idx = min(len(self.calls) - 1, len(self._responses) - 1)
                return self._responses[idx]

        # [leak, leak, valid] — recovers on the 3rd attempt under the preserved
        # budget (MAX_JUDGE_RETRIES=3 → chokepoint max_retries=2 → 3 attempts).
        structured = _RecordingStructuredLLM(
            [_leak_response(), _leak_response(), _valid_response()]
        )
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact, judge_model="claude-haiku-4-5", api_key="sk-test",
            )

        # Recovered: a valid RubricEvalArtifact carrying the 3rd attempt's scores.
        assert isinstance(result, RubricEvalArtifact)
        assert len(result.dimension_scores) == 6
        assert result.overall_reasoning == (
            "Solid grounding; regime engagement weakest."
        )
        # resolved_model extraction (L4578(a)) preserved across the migration.
        assert result.judge_resolved_model == "claude-haiku-4-5-20251001"

        # 3 total attempts (fail-loud budget preserved).
        assert len(structured.calls) == 3

        # CORRECTION FEEDBACK — the discriminating assertion vs. a bare re-roll:
        # attempt 1 sends only the rendered prompt; the retry does NOT re-send
        # the identical single message but grows it with the prior raw output +
        # a correction that names the schema violation.
        assert len(structured.calls[0]) == 1
        assert len(structured.calls[1]) > 1
        retry_text = " ".join(
            m.content for m in structured.calls[1]
            if isinstance(getattr(m, "content", None), str)
        )
        assert "schema validation" in retry_text.lower()

    def test_deterministic_leak_that_never_corrects_still_fails_loud(self):
        """FAIL-LOUD is absolute (config#2237): a deterministic leak the model
        never corrects across the full attempt budget must still ``raise`` — no
        carry-forward, no swallow, no widened gate."""
        from evals import judge as judge_mod
        from graph.state_schemas import RubricEvalLLMOutput

        leaked_xml = '\n<parameter name="dimension_scores">numerical_grounding: 4\n'

        def _leak_response():
            try:
                RubricEvalLLMOutput(
                    dimension_scores=leaked_xml, overall_reasoning="x",
                )
                raise AssertionError("expected ValidationError")
            except Exception as exc:
                parsing_error = exc
            return {"raw": object(), "parsed": None, "parsing_error": parsing_error}

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = _leak_response()
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError, match="parse attempts failed"):
                judge_mod.evaluate_artifact(
                    artifact, judge_model="claude-haiku-4-5", api_key="sk-test",
                )

        # Full budget exhausted (MAX_JUDGE_RETRIES=3 → 3 attempts).
        assert fake_structured.invoke.call_count == 3


# ── persist_eval_artifact ─────────────────────────────────────────────────


class TestEmptyInputShortCircuit:
    """Empty-input structural-skip handling.

    Sector_qual + sector_peer_review captures land with ``agent_output={}``
    when graph design bypasses them (e.g. qual loop is skipped when
    upstream ``quant_top5`` is empty). Pre-fix, the judge LLM was asked
    to score nothing and uniformly returned 1/1/1/1 across dimensions —
    dragging the rolling-mean alarm threshold toward the floor without
    any real quality regression. The fix detects empty agent_output,
    short-circuits BEFORE the LLM call, and persists a skip marker
    with empty dimension_scores + judge_skip_reason set.
    """

    def _make_empty_qual_artifact(self):
        return DecisionArtifact(
            run_id="run-empty-1",
            timestamp="2026-05-04T13:00:00.000Z",
            agent_id="sector_qual:financials",
            model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
            full_prompt_context=FullPromptContext(
                system_prompt="<see config/prompts>",
                user_prompt="<rendered>",
            ),
            input_data_snapshot={
                "team_id": "financials",
                "run_date": "2026-05-04",
                "quant_top5": [],
                "quant_top5_tickers": [],
                "held_in_top5": [],
            },
            input_data_summary="team_id=financials, top5=0",
            agent_output={},  # The empty-input pattern
        )

    def test_empty_agent_output_short_circuits_before_llm(self, monkeypatch):
        """The LLM call is never made when agent_output is empty —
        zero token cost, zero retry surface."""
        from evals import judge as judge_mod

        fake_llm = MagicMock()
        fake_structured = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            result = judge_mod.evaluate_artifact(
                self._make_empty_qual_artifact(),
                judge_model="claude-haiku-4-5",
                api_key="sk-test",
            )

        # ChatAnthropic must have been instantiated only by load_prompt's
        # downstream paths, NOT for an LLM call. The structured-output
        # invoke path must never have fired.
        assert fake_structured.invoke.call_count == 0, (
            "Empty-input short-circuit must skip the LLM invoke entirely. "
            f"Got invoke_count={fake_structured.invoke.call_count}"
        )

    def test_empty_agent_output_returns_skip_marker_artifact(self):
        """Skip path persists a RubricEvalArtifact with empty dimensions
        + judge_skip_reason set + a non-empty overall_reasoning string
        explaining why."""
        from evals.judge import evaluate_artifact

        result = evaluate_artifact(
            self._make_empty_qual_artifact(),
            judge_model="claude-haiku-4-5",
            api_key="sk-test",
        )

        assert isinstance(result, RubricEvalArtifact)
        assert result.judge_skip_reason == "precluded_by_empty_upstream"
        assert result.dimension_scores == []
        assert result.overall_reasoning  # non-empty
        assert "short-circuited" in result.overall_reasoning.lower()
        # Metadata must still be populated so the audit trail traces
        # back to the judged artifact + rubric + judge model.
        assert result.judged_agent_id == "sector_qual:financials"
        assert result.rubric_id == "eval_rubric_sector_qual"
        assert result.rubric_version  # frontmatter version
        assert result.judge_model == "claude-haiku-4-5"
        assert result.run_id == "run-empty-1"

    def test_none_agent_output_is_also_short_circuited(self):
        """``agent_output=None`` is treated identically to ``{}`` — both
        are falsy and indicate the agent never ran. Otherwise an
        upstream nullability change in DecisionArtifact would silently
        bypass the skip path."""
        from evals.judge import evaluate_artifact

        artifact = self._make_empty_qual_artifact()
        # DecisionArtifact.agent_output is typed dict but tests can poke
        # the model_dump path; bypass with object.__setattr__ for the
        # null edge case.
        object.__setattr__(artifact, "agent_output", None)

        result = evaluate_artifact(artifact, api_key="sk-test")
        assert result.judge_skip_reason == "precluded_by_empty_upstream"
        assert result.dimension_scores == []

    def test_non_empty_agent_output_does_not_short_circuit(self, monkeypatch):
        """A non-empty agent_output (even one with empty inner lists,
        e.g. quant returning ranked_picks=[]) is NOT a structural skip
        — the agent ran. The judge runs as normal and the empty inner
        result becomes the agent-failure signal we WANT to surface."""
        from evals import judge as judge_mod

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "parsed": _make_llm_output(),
            "raw": MagicMock(content="ok"),
            "parsing_error": None,
        }
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        artifact = _make_artifact("sector_quant:technology")
        # Quant ran, did 22 tool calls, but returned no qualifying picks
        # — the agent-failure pattern from workstream #2 (separate).
        artifact.agent_output = {"ranked_picks": [], "tool_calls": [{}] * 22, "iterations": 22}

        with patch.object(judge_mod, "ChatAnthropic", return_value=fake_llm):
            result = judge_mod.evaluate_artifact(artifact, api_key="sk-test")

        # Judge MUST have run — empty ranked_picks is not the structural
        # skip pattern; it's an agent-quality signal we want surfaced.
        assert fake_structured.invoke.call_count == 1
        assert result.judge_skip_reason is None
        assert len(result.dimension_scores) > 0


@pytest.fixture
def mocked_s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        yield client


class TestPersistEvalArtifact:
    """Pins persistence under the canonical flat ``eval_artifacts`` layout
    (config#793): judge_run_id is required and is the path's
    batch-grouping prefix; a ``latest.json`` operator-UX sidecar mirrors
    the most-recently-written key."""

    def test_writes_at_canonical_flat_key(self, mocked_s3):
        from evals.judge import persist_eval_artifact

        artifact = RubricEvalArtifact(
            run_id="run-1",
            judge_run_id="2605092230",
            timestamp="2026-05-09T22:30:00.000Z",
            judged_agent_id="sector_quant:technology",
            rubric_id="eval_rubric_sector_quant",
            rubric_version="1.0.0",
            judge_model="claude-haiku-4-5",
            dimension_scores=_make_llm_output().dimension_scores,
            overall_reasoning="solid grounding",
        )
        key = persist_eval_artifact(
            artifact, s3_client=mocked_s3, bucket="alpha-engine-research",
        )

        # Flat: {prefix}{judge_run_id}_{agent_id}.{run_id}.{judge_model}.json
        assert key == (
            "decision_artifacts/_eval/2605092230_"
            "sector_quant:technology.run-1.claude-haiku-4-5.json"
        )
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=key)
        roundtrip = RubricEvalArtifact.model_validate(json.loads(obj["Body"].read()))
        assert roundtrip.judge_model == "claude-haiku-4-5"
        assert roundtrip.judge_run_id == "2605092230"
        assert roundtrip.rubric_version == "1.0.0"
        assert len(roundtrip.dimension_scores) == 6

    def test_writes_latest_sidecar_pointing_at_artifact(self, mocked_s3):
        """The latest.json sidecar mirrors the most-recently-written key
        and is resolvable by the lib's load_latest_eval_artifact reader."""
        from nousergon_lib.eval_artifacts import (
            eval_latest_key, load_latest_eval_artifact,
        )
        from evals.judge import DEFAULT_EVAL_PREFIX, persist_eval_artifact

        artifact = RubricEvalArtifact(
            run_id="run-1",
            judge_run_id="2605092230",
            timestamp="2026-05-09T22:30:00.000Z",
            judged_agent_id="ic_cio",
            rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0",
            judge_model="claude-haiku-4-5",
            dimension_scores=_make_llm_output().dimension_scores,
            overall_reasoning="x",
        )
        key = persist_eval_artifact(
            artifact, s3_client=mocked_s3, bucket="alpha-engine-research",
        )
        sidecar_key = eval_latest_key(DEFAULT_EVAL_PREFIX)
        sidecar = json.loads(
            mocked_s3.get_object(
                Bucket="alpha-engine-research", Key=sidecar_key,
            )["Body"].read()
        )
        assert sidecar["artifact_key"] == key
        assert sidecar["judge_run_id"] == "2605092230"
        # The lib reader resolves sidecar → artifact body end-to-end.
        loaded = load_latest_eval_artifact(
            mocked_s3, bucket="alpha-engine-research",
            prefix=DEFAULT_EVAL_PREFIX,
        )
        assert loaded is not None
        assert loaded["judged_agent_id"] == "ic_cio"

    def test_update_latest_false_skips_sidecar(self, mocked_s3):
        from nousergon_lib.eval_artifacts import eval_latest_key
        from botocore.exceptions import ClientError
        from evals.judge import DEFAULT_EVAL_PREFIX, persist_eval_artifact

        artifact = RubricEvalArtifact(
            run_id="run-1",
            judge_run_id="2605092230",
            timestamp="2026-05-09T22:30:00.000Z",
            judged_agent_id="ic_cio",
            rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0",
            judge_model="claude-haiku-4-5",
            dimension_scores=_make_llm_output().dimension_scores,
            overall_reasoning="x",
        )
        persist_eval_artifact(
            artifact, s3_client=mocked_s3, bucket="alpha-engine-research",
            update_latest=False,
        )
        with pytest.raises(ClientError):
            mocked_s3.get_object(
                Bucket="alpha-engine-research",
                Key=eval_latest_key(DEFAULT_EVAL_PREFIX),
            )

    def test_batch_cohesion_under_one_judge_run_id(self, mocked_s3):
        """Two artifacts sharing a judge_run_id land under the same
        run_id prefix — the institutional batch-cohesion property
        carried into the flat layout (config#793)."""
        from evals.judge import persist_eval_artifact

        shared_batch = "2605092230"
        artifact_a = RubricEvalArtifact(
            run_id="run-a",
            judge_run_id=shared_batch,
            timestamp="2026-05-09T22:30:00.000Z",
            judged_agent_id="ic_cio",
            rubric_id="eval_rubric_ic_cio",
            rubric_version="1.0.0",
            judge_model="claude-haiku-4-5",
            dimension_scores=_make_llm_output().dimension_scores,
            overall_reasoning="x",
        )
        artifact_b = RubricEvalArtifact(
            run_id="run-b",
            judge_run_id=shared_batch,
            timestamp="2026-05-09T22:31:00.000Z",
            judged_agent_id="thesis_update:financials:CBOE",
            rubric_id="eval_rubric_thesis_update",
            rubric_version="1.0.0",
            judge_model="claude-haiku-4-5",
            dimension_scores=_make_llm_output().dimension_scores,
            overall_reasoning="x",
        )
        key_a = persist_eval_artifact(
            artifact_a, s3_client=mocked_s3, bucket="alpha-engine-research",
        )
        key_b = persist_eval_artifact(
            artifact_b, s3_client=mocked_s3, bucket="alpha-engine-research",
        )
        # Same judge_run_id prefix; different basenames.
        common_prefix = f"decision_artifacts/_eval/{shared_batch}_"
        assert key_a.startswith(common_prefix)
        assert key_b.startswith(common_prefix)
        assert key_a != key_b


# ── RubricEvalLLMOutput stringify defense ─────────────────────────────────


class TestRubricEvalLLMOutputStringDefense:
    """Pins the 2026-05-03 fix for an observed Haiku failure mode:
    ``dimension_scores`` returned as a JSON-encoded string instead of
    a structured array. First surfaced in the judge_only smoke against
    Sat 5/3 captures (5/32 evals failed at the schema boundary).
    Mirrors the JointFinalizationOutput defense pattern from PR #99.
    """

    def test_actual_list_passes_through_unchanged(self):
        out = RubricEvalLLMOutput(
            dimension_scores=[
                RubricDimensionScore(
                    dimension="numerical_grounding", score=4,
                    reasoning="cited multiples match filing",
                ),
                RubricDimensionScore(
                    dimension="signal_calibration", score=3,
                    reasoning="confidence within range",
                ),
            ],
            overall_reasoning="balanced",
        )
        assert len(out.dimension_scores) == 2
        assert out.dimension_scores[0].dimension == "numerical_grounding"

    def test_json_string_of_list_is_parsed_and_logged(self, caplog):
        """The exact failure shape Haiku produced 2026-05-03: a string
        whose contents are valid JSON for the expected list."""
        import logging
        payload = json.dumps([
            {"dimension": "numerical_grounding", "score": 4, "reasoning": "ok"},
            {"dimension": "signal_calibration", "score": 3, "reasoning": "tight"},
        ])

        with caplog.at_level(logging.WARNING):
            out = RubricEvalLLMOutput(
                dimension_scores=payload,  # type: ignore[arg-type]
                overall_reasoning="ok",
            )

        assert len(out.dimension_scores) == 2
        assert out.dimension_scores[0].dimension == "numerical_grounding"
        assert any(
            "schema-vs-LLM drift" in rec.message
            or "JSON-string" in rec.message
            for rec in caplog.records
        )

    def test_invalid_json_string_raises_normal_pydantic_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="list_type|valid list"):
            RubricEvalLLMOutput(
                dimension_scores="not even close to valid json",  # type: ignore[arg-type]
                overall_reasoning="ok",
            )

    def test_truncated_jsonlist_string_still_raises_loud(self):
        """Truncated mid-string from max_tokens hit. Validator's
        json.loads can't parse incomplete JSON → falls through to loud
        Pydantic list_type error. Pin so silent rescue can't regress."""
        from pydantic import ValidationError

        truncated = (
            '[\n  {\n    "dimension": "numerical_grounding",\n    "score": 4,\n'
            '    "reasoning": "Extensive citation analysis was attem'
            # truncated mid-string, no closing
        )
        with pytest.raises(ValidationError, match="list_type|valid list"):
            RubricEvalLLMOutput(
                dimension_scores=truncated,  # type: ignore[arg-type]
                overall_reasoning="ok",
            )

    def test_field_description_present(self):
        """Pin the 'structured array, NOT JSON-encoded string' hint
        in the tool-use spec — same pattern as JointFinalizationOutput."""
        field_info = RubricEvalLLMOutput.model_fields["dimension_scores"]
        assert "structured array" in (field_info.description or "")
        assert "NOT" in (field_info.description or "")
