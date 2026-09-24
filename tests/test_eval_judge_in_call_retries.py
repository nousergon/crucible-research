"""alpha-engine-config-I11485: the sync rung's retries and its reason are
legible on the run log.

On rehearsal-2026-09-23-2 every eval took the sync rung. The log showed one
schema retry recovering inside ``evaluate_artifact`` while the Process summary
said ``parse_retry_recovered=0``, because that counter only ever counted the
batch rung's parse-retry TAIL. And ``degraded_transport=True`` named the rung
without saying why the batch route was unavailable.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from evals import judge as _judge
from evals import judge_batch_transport as jbt
from tests.test_eval_judge import _make_llm_output


class _FakeSpec:
    model = "low-deepseek-v4-flash-tools"
    provider = "litellm"
    registry_id = "deepseek-v4-flash"


def _response(arguments: str):
    call = SimpleNamespace(
        id="c1", type="function",
        function=SimpleNamespace(name=_judge._build_rubric_tool_spec()["name"],
                                 arguments=arguments),
    )
    choice = SimpleNamespace(
        finish_reason="tool_calls",
        message=SimpleNamespace(content=None, tool_calls=[call]),
    )
    return SimpleNamespace(
        raw_response=SimpleNamespace(choices=[choice], model="deepseek/deepseek-v4-flash"),
        usage=SimpleNamespace(provider_cost_usd=0.0),
    )


def _client_answering(monkeypatch, answers: list[str]):
    queue = list(answers)

    class _Client:
        def __init__(self, *a, **k):
            pass

        def complete(self, **kwargs):
            return _response(queue.pop(0))

    monkeypatch.setattr(_judge, "LLMClient", _Client)
    monkeypatch.setattr(
        _judge, "_judge_router_spec_and_route",
        lambda **kw: (_FakeSpec(), {"route": "litellm_proxy"}),
    )


def _call():
    return _judge._call_openrouter_judge_llm(
        "rendered", agent_id="a", request_model="m", max_tokens=64,
        api_key=None, max_retries=3, log_prefix="[t]",
    )


def test_a_call_recovered_on_retry_is_counted(monkeypatch):
    good = json.dumps(_make_llm_output().model_dump())
    _client_answering(monkeypatch, ['{"dimension_scores": "not a list"}', good])
    before = _judge.in_call_retries_recovered()
    _call()
    assert _judge.in_call_retries_recovered() - before == 1


def test_a_first_attempt_success_is_not_counted(monkeypatch):
    good = json.dumps(_make_llm_output().model_dump())
    _client_answering(monkeypatch, [good])
    before = _judge.in_call_retries_recovered()
    _call()
    assert _judge.in_call_retries_recovered() == before


def test_the_sync_rung_logs_why_the_batch_route_was_unavailable(monkeypatch, caplog):
    import evals.orchestrator as orch

    def _unavailable(**kwargs):
        raise jbt.BatchCapabilityUnavailable(
            group="low", capability="batches", exec_context="ec2",
            reason="batches is not a routable capability",
        )

    monkeypatch.setattr(orch, "resolve_batch_transport", _unavailable)
    plan = {
        "date": "2026-09-23", "bucket": "b", "force_sonnet_pass": False,
        "requests": [{"custom_id": "x"}], "plan_entries": [],
        "haiku_model": "h", "sonnet_model": "s", "judge_run_id": "jr",
        "client_side_skips": [], "capture_keys_total": 1, "skipped_unmapped": 0,
    }
    with caplog.at_level(logging.WARNING, logger=orch.logger.name):
        orch.submit_batch(plan, s3_client=MagicMock())
    line = next(r.getMessage() for r in caplog.records if "sync rung" in r.getMessage())
    assert "reason=batches is not a routable capability" in line
    assert "exec_context=ec2" in line
