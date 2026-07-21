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
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws
from nousergon_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)

from evals.judge_models import OPENROUTER_SHADOW
from graph.state_schemas import (
    RubricDimensionScore,
    RubricEvalArtifact,
    RubricEvalLLMOutput,
)

# ── OpenRouter transport fakes (alpha-engine-config-I2997) — mirrors
#    tests/test_eval_judge_openrouter.py's helpers; duplicated locally
#    (rather than imported) since that module imports fixtures FROM this
#    one and a back-import would cycle. ──────────────────────────────────


def _openai_tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _openai_response(
    *, finish_reason: str, tool_calls=None, content=None,
    model="deepseek/deepseek-v4-flash", cost: float = 0.0001,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    usage = SimpleNamespace(cost=cost)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _valid_tool_args() -> dict:
    return _make_llm_output().model_dump()


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

        from evals.judge import DEFAULT_EVAL_PREFIX, build_eval_s3_key
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
        from datetime import datetime

        from nousergon_lib.eval_artifacts import new_eval_run_id
        earlier = new_eval_run_id(now=datetime(2026, 5, 9, 22, 30, tzinfo=UTC))
        later = new_eval_run_id(now=datetime(2026, 5, 10, 9, 15, tzinfo=UTC))
        assert earlier < later  # lexicographic = chronological


class TestBuildLegacyEvalS3Key:
    """Backward-compat: the legacy nested Option B key builder is retained
    so readers/tests can reconstruct pre-config#793 historical paths."""

    def test_legacy_nested_shape(self):
        from evals.judge import build_legacy_eval_s3_key
        ts = datetime(2026, 5, 9, 22, 30, tzinfo=UTC)
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
    """alpha-engine-config-I2997 (2026-07-19): ``evaluate_artifact`` migrated
    off direct Anthropic (``ChatAnthropic``) to the SAME OpenRouter
    forced-tool-call transport core ``evaluate_artifact_openrouter`` already
    used (config#2575) — mocks ``judge_mod.OpenAI`` (mirrors
    ``tests/test_eval_judge_openrouter.py``), not ``ChatAnthropic``."""

    def test_unmapped_agent_raises(self):
        from evals.judge import evaluate_artifact
        artifact = _make_artifact("totally_made_up_agent")
        with pytest.raises(ValueError, match="No rubric mapped"):
            evaluate_artifact(artifact)

    def test_full_pipeline_with_mocked_llm(self, monkeypatch):
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="tool_calls",
            tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
        )

        with patch.object(judge_mod, "OpenAI", return_value=fake_client) as mock_openai_cls:
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact,
                judge_model="claude-haiku-4-5",
                api_key="sk-or-test",
                judged_artifact_s3_key="decision_artifacts/2026/05/09/sector_quant:technology/r1.json",
            )

        # Result wrapping
        assert isinstance(result, RubricEvalArtifact)
        assert result.judged_agent_id == "sector_quant:technology"
        assert result.run_id == artifact.run_id
        assert result.rubric_id == "eval_rubric_sector_quant"
        # judge_model is the STABLE logical key (persistence/dimension) —
        # PRESERVED across the migration (S3 path / CloudWatch dimension /
        # rolling-mean identity; see evaluate_artifact's docstring).
        assert result.judge_model == "claude-haiku-4-5"
        # ...while the ACTUAL API call now goes to the OpenRouter default
        # evaluate_artifact_openrouter already uses (Brian's ruling: "keep
        # consistent" rather than a bespoke new pin) — NOT the old Anthropic
        # dated-snapshot pin.
        assert result.judge_request_model == OPENROUTER_SHADOW.request_model
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == OPENROUTER_SHADOW.request_model
        assert result.judge_resolved_model == "deepseek/deepseek-v4-flash"
        assert result.judged_artifact_s3_key.endswith("/r1.json")
        assert result.rubric_version  # non-empty
        assert len(result.dimension_scores) == 6
        assert result.dimension_scores[0].dimension == "numerical_grounding"
        assert "regime engagement" in result.overall_reasoning
        # First-attempt success — should NOT have called create more than once
        assert fake_client.chat.completions.create.call_count == 1
        # base_url must resolve to the OpenRouter endpoint.
        mock_openai_cls.assert_called_once()
        assert mock_openai_cls.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert mock_openai_cls.call_args.kwargs["api_key"] == "sk-or-test"
        # reasoning={"exclude": True} forwarded (truncation-avoidance default).
        assert call_kwargs["extra_body"] == {"reasoning": {"exclude": True}}

    def test_sonnet_tier_also_routes_to_the_same_openrouter_default(self):
        """Brian's ruling collapses BOTH tiers' physical call onto the SAME
        OpenRouter default — the Sonnet ``judge_model`` logical key is
        preserved (persisted identity) but the request model is identical
        to the Haiku tier's, not a separate Sonnet-equivalent pin."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="tool_calls",
            tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
        )
        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact, judge_model="claude-sonnet-4-6", api_key="sk-or-test",
            )

        assert result.judge_model == "claude-sonnet-4-6"
        assert result.judge_request_model == OPENROUTER_SHADOW.request_model

    def test_records_resolved_model_from_response(self):
        """L4578(a): the API-resolved model is captured per-artifact for
        drift detection / the re-anchor protocol — now sourced from the
        OpenRouter response's ``model`` field rather than langchain's
        ``response_metadata``."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="tool_calls",
            tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
            model="deepseek/deepseek-v4-flash",
        )
        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            result = judge_mod.evaluate_artifact(
                _make_artifact("sector_quant:technology"),
                judge_model="claude-haiku-4-5",
                api_key="sk-or-test",
            )
        assert result.judge_resolved_model == "deepseek/deepseek-v4-flash"

    def test_renders_artifact_payload_into_prompt(self, monkeypatch):
        """Verify the rubric prompt is rendered with the artifact's
        input_data_snapshot + agent_output, not placeholder strings."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="tool_calls",
            tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
        )
        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            judge_mod.evaluate_artifact(artifact, api_key="sk-or-test")

        # Inspect the rendered user-turn content passed to the API.
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        rendered = call_kwargs["messages"][1]["content"]
        assert "AAPL" in rendered
        assert "technology" in rendered
        assert "ranked_picks" in rendered
        assert "RSI 55, TS 70." in rendered
        assert "{agent_input}" not in rendered
        assert "{agent_output}" not in rendered

    def test_retries_on_parse_failure_and_succeeds(self, monkeypatch, caplog):
        """First attempt has no tool call (ordinary non-conformance);
        second attempt succeeds. Counts 2 creates total."""
        import logging

        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _openai_response(finish_reason="stop", tool_calls=None, content="I refuse."),
            _openai_response(
                finish_reason="tool_calls",
                tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
            ),
        ]

        with patch.object(judge_mod, "OpenAI", return_value=fake_client), \
             caplog.at_level(logging.WARNING):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(
                artifact, judge_model="claude-haiku-4-5", api_key="sk-or-test",
            )

        assert isinstance(result, RubricEvalArtifact)
        assert fake_client.chat.completions.create.call_count == 2
        warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("attempt 1/3" in m for m in warn_msgs)

    def test_raises_after_max_retries_exhausted(self, monkeypatch):
        """All 3 attempts fail (leak/truncation) → raises RuntimeError with
        diagnostic context. Bounds worst-case latency + makes structural
        failures loud."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="length", tool_calls=None, content=None,
        )

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError, match="attempts failed"):
                judge_mod.evaluate_artifact(
                    artifact, judge_model="claude-haiku-4-5", api_key="sk-or-test",
                )

        # All 3 attempts fired (default MAX_JUDGE_RETRIES=3).
        assert fake_client.chat.completions.create.call_count == 3

    def test_retry_count_param_overrides_default(self, monkeypatch):
        """``max_retries`` param lets callers tune the budget — useful
        for the test-track flag and for ad-hoc replay scripts that
        want to fail fast."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="length", tool_calls=None, content=None,
        )

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError):
                judge_mod.evaluate_artifact(
                    artifact, max_retries=1, api_key="sk-or-test",
                )

        # Single attempt only.
        assert fake_client.chat.completions.create.call_count == 1

    def test_control_token_leak_recovers_on_retry(self, caplog):
        """Mirrors ``TestEvaluateArtifactOpenRouter``'s leak-guard coverage
        — a control-token leak on attempt 1 consumes a retry (fresh
        decoder sample), not an immediate failure; attempt 2 succeeds."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _openai_response(
                finish_reason="stop",
                content="<|tool_calls_section_begin|>garbage",
                tool_calls=None,
            ),
            _openai_response(
                finish_reason="tool_calls",
                tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
            ),
        ]

        with patch.object(judge_mod, "OpenAI", return_value=fake_client), \
             caplog.at_level("WARNING"):
            artifact = _make_artifact("sector_quant:technology")
            result = judge_mod.evaluate_artifact(artifact, api_key="sk-or-test")

        assert isinstance(result, RubricEvalArtifact)
        assert len(result.dimension_scores) == 6
        assert result.overall_reasoning == (
            "Solid grounding; regime engagement weakest."
        )
        assert fake_client.chat.completions.create.call_count == 2
        assert any("leak_guard_triggered" in rec.message for rec in caplog.records)

    def test_leak_that_never_recovers_still_fails_loud(self):
        """FAIL-LOUD is absolute: a leak the model never corrects across
        the full attempt budget must still ``raise`` — no carry-forward,
        no swallow, no widened gate."""
        from evals import judge as judge_mod

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="length", tool_calls=None, content=None,
        )

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            artifact = _make_artifact("sector_quant:technology")
            with pytest.raises(RuntimeError, match="attempts failed"):
                judge_mod.evaluate_artifact(
                    artifact, judge_model="claude-haiku-4-5", api_key="sk-or-test",
                )

        # Full budget exhausted (MAX_JUDGE_RETRIES=3 → 3 attempts).
        assert fake_client.chat.completions.create.call_count == 3


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

        fake_client = MagicMock()

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            result = judge_mod.evaluate_artifact(
                self._make_empty_qual_artifact(),
                judge_model="claude-haiku-4-5",
                api_key="sk-or-test",
            )

        assert result.judge_skip_reason == "precluded_by_empty_upstream"
        fake_client.chat.completions.create.assert_not_called()

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

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _openai_response(
            finish_reason="tool_calls",
            tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
        )

        artifact = _make_artifact("sector_quant:technology")
        # Quant ran, did 22 tool calls, but returned no qualifying picks
        # — the agent-failure pattern from workstream #2 (separate).
        artifact.agent_output = {"ranked_picks": [], "tool_calls": [{}] * 22, "iterations": 22}

        with patch.object(judge_mod, "OpenAI", return_value=fake_client):
            result = judge_mod.evaluate_artifact(artifact, api_key="sk-or-test")

        # Judge MUST have run — empty ranked_picks is not the structural
        # skip pattern; it's an agent-quality signal we want surfaced.
        assert fake_client.chat.completions.create.call_count == 1
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
            eval_latest_key,
            load_latest_eval_artifact,
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
        from botocore.exceptions import ClientError
        from nousergon_lib.eval_artifacts import eval_latest_key

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
