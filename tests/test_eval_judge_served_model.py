"""alpha-engine-config-I11484: a judge record names the model that SERVED it.

On rehearsal-2026-09-23-2 all 33 evals were served by the router's ``low``
group (DeepSeek v4 flash through the LiteLLM proxy), and every one was
persisted as ``judge_model=claude-haiku-4-5`` / ``claude-sonnet-4-6``. The two
provenance fields beside it did not correct that:

* ``judge_request_model`` carried a module constant
  (``OPENROUTER_SHADOW.request_model``), which since I6559 selects nothing;
* ``judge_resolved_model`` carried ``raw_response.model`` — on the proxy route
  that is the DEPLOYMENT NAME LiteLLM echoes back
  (``low-deepseek-v4-flash-tools``), not a model.

``krepis`` already resolves the served model onto ``LLMResult.model``
(``_resolve_group_served_model``); the judge threw it away. The same fact is
what the Sonnet escalation needs to show whether it was a second opinion.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import krepis.llm as _kllm
import pytest

from evals import judge as judge_mod
from evals import orchestrator as orch
from graph.state_schemas import RubricDimensionScore, RubricEvalArtifact
from tests.test_eval_judge import (
    _make_artifact,
    _make_llm_output,
    _openai_response,
    _openai_tool_call,
    _patch_llm_client,
    _valid_tool_args,
)

DEPLOYMENT = "low-deepseek-v4-flash-tools"
UPSTREAM = "deepseek-v4-flash"


@pytest.fixture(autouse=True)
def live_router_resolution(monkeypatch):
    """The production shape: ``low`` resolves to a QUALIFIED deployment, the
    wire carries that name, and LiteLLM echoes it back as ``model``."""
    import krepis.router as _kr

    def fake_resolve_structured(group, *, exec_context=None, wire="openai", requires=()):
        return {
            "schema_version": 2,
            "group": group,
            "route": "litellm_proxy",
            "provider": "litellm",
            "deployment_id": DEPLOYMENT,
            "api_base_url": "https://router.invalid/v1",
            "auth_token_type": "litellm_master_key",
            "registry_id": f"litellm:group:{group}",
            "primary_registry_id": UPSTREAM,
            "params": {},
        }

    monkeypatch.setattr(_kr, "resolve_group_structured", fake_resolve_structured)
    # The registry read krepis does to map a deployment back to its upstream
    # model — stubbed so the test does not depend on a registry file.
    monkeypatch.setattr(
        _kllm,
        "served_model_for_deployment",
        lambda name: UPSTREAM if name == DEPLOYMENT else None,
    )
    monkeypatch.setenv("LITELLM_MASTER_KEY", "router-test-key")
    return True


def _evaluate(judge_model: str, *, cost: float | None = 0.0001):
    fake_client = MagicMock()
    response = _openai_response(
        finish_reason="tool_calls",
        tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
        model=DEPLOYMENT,
        cost=cost if cost is not None else 0.0,
    )
    if cost is None:
        response.usage = SimpleNamespace()
    fake_client.chat.completions.create.return_value = response
    with _patch_llm_client(judge_mod, fake_client):
        return judge_mod.evaluate_artifact(
            _make_artifact("thinktank_thesis"),
            judge_model=judge_model,
            api_key="sk-test",
        )


@pytest.mark.parametrize("tier", ["claude-haiku-4-5", "claude-sonnet-4-6"])
def test_the_record_names_the_model_that_served_not_the_deployment_echo(tier):
    result = _evaluate(tier)

    # The tier key is unchanged — it is the S3 path / CW dimension identity.
    assert result.judge_model == tier
    # What graded: the upstream model krepis resolved, never the echo.
    assert result.judge_resolved_model == UPSTREAM
    assert result.judge_resolved_model != DEPLOYMENT
    # What was addressed: the deployment the router resolved `low` to.
    assert result.judge_request_model == DEPLOYMENT
    # Neither provenance field claims Claude graded it.
    assert "claude" not in (result.judge_resolved_model or "")
    assert "claude" not in (result.judge_request_model or "")


def test_the_log_line_names_the_tier_as_a_tier_and_the_served_model(caplog):
    with caplog.at_level(logging.INFO, logger="evals.judge"):
        _evaluate("claude-haiku-4-5")
    line = next(r.getMessage() for r in caplog.records if "persisted-cost" in r.getMessage())
    assert "judge_tier=claude-haiku-4-5" in line
    assert f"served_model={UPSTREAM}" in line
    assert f"addressed={DEPLOYMENT}" in line
    assert "judge_model=" not in line


def test_an_unreported_cost_is_logged_as_unreported_not_as_zero(caplog):
    with caplog.at_level(logging.INFO, logger="evals.judge"):
        _evaluate("claude-haiku-4-5", cost=None)
    line = next(r.getMessage() for r in caplog.records if "persisted-cost" in r.getMessage())
    assert "provider_cost_usd=unreported" in line
    assert "0.000000" not in line


def test_a_reported_zero_cost_is_still_zero(caplog):
    with caplog.at_level(logging.INFO, logger="evals.judge"):
        _evaluate("claude-haiku-4-5", cost=0.0)
    line = next(r.getMessage() for r in caplog.records if "persisted-cost" in r.getMessage())
    assert "provider_cost_usd=0.000000" in line


def test_the_call_core_prefers_the_resolved_model_over_the_raw_echo(monkeypatch):
    """Unit-level: ``LLMResult.model`` wins over ``raw_response.model``."""
    good = json.dumps(_make_llm_output().model_dump())
    call = SimpleNamespace(
        id="c1",
        type="function",
        function=SimpleNamespace(name=judge_mod._build_rubric_tool_spec()["name"], arguments=good),
    )
    choice = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(content=None, tool_calls=[call]),
    )

    class _Client:
        def __init__(self, *a, **k):
            pass

        def complete(self, **kwargs):
            return SimpleNamespace(
                model=UPSTREAM,
                raw_response=SimpleNamespace(choices=[choice], model=DEPLOYMENT),
                usage=SimpleNamespace(provider_cost_usd=None),
            )

    monkeypatch.setattr(judge_mod, "LLMClient", _Client)
    monkeypatch.setattr(
        judge_mod,
        "_judge_router_spec_and_route",
        lambda **kw: (SimpleNamespace(model=DEPLOYMENT), {"route": "litellm_proxy"}),
    )
    out = judge_mod._call_openrouter_judge_llm(
        "rendered",
        agent_id="a",
        request_model="declared/constant",
        max_tokens=64,
        api_key=None,
        max_retries=1,
        log_prefix="[t]",
    )
    assert out.resolved_model == UPSTREAM
    assert out.addressed_model == DEPLOYMENT
    assert out.total_usd is None


# ── Escalation: was it a second opinion? ─────────────────────────────────


def _eval(judge_model: str, served: str | None) -> RubricEvalArtifact:
    return RubricEvalArtifact(
        run_id="r1",
        judge_run_id="j1",
        timestamp="2026-09-23T00:00:00Z",
        judged_agent_id="ic_cio",
        rubric_id="eval_rubric_test",
        rubric_version="1.0.0",
        judge_model=judge_model,
        judge_resolved_model=served,
        dimension_scores=[RubricDimensionScore(dimension="d", score=4, reasoning="r")],
        overall_reasoning="ok",
    )


@pytest.mark.parametrize(
    ("first", "second", "verdict"),
    [
        (UPSTREAM, UPSTREAM, orch.ESCALATION_SAME_MODEL),
        # Same weights through an aggregator is still the same judge.
        ("deepseek/deepseek-v4-flash", UPSTREAM, orch.ESCALATION_SAME_MODEL),
        (UPSTREAM, "claude-sonnet-4-6", orch.ESCALATION_DISTINCT),
        (None, UPSTREAM, orch.ESCALATION_UNKNOWN),
        (UPSTREAM, None, orch.ESCALATION_UNKNOWN),
    ],
)
def test_escalation_verdict(first, second, verdict):
    assert (
        orch.escalation_served_model_verdict(
            _eval("claude-haiku-4-5", first),
            _eval("claude-sonnet-4-6", second),
        )
        == verdict
    )


def test_a_same_model_escalation_warns_and_is_counted(caplog):
    counts: dict[str, int] = {}
    with caplog.at_level(logging.WARNING, logger="evals.orchestrator"):
        orch._note_escalation_verdict(
            counts,
            _eval("claude-haiku-4-5", UPSTREAM),
            _eval("claude-sonnet-4-6", UPSTREAM),
        )
    assert orch._escalation_result_fields(counts) == {
        "escalation_distinct_served_model": 0,
        "escalation_same_served_model": 1,
        "escalation_served_model_unknown": 0,
    }
    assert any("escalation_same_served_model" in r.getMessage() for r in caplog.records)


def test_the_escalation_fields_are_reported_when_nothing_escalated():
    assert orch._escalation_result_fields({}) == {
        "escalation_distinct_served_model": 0,
        "escalation_same_served_model": 0,
        "escalation_served_model_unknown": 0,
    }
