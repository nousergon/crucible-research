"""Router group-addressing tests for the judge call sites
(alpha-engine-config-I6559 — no agent may be directly linked to
OpenRouter, I6367 ruling).

Mirrors ``tests/test_single_agent_producer.py``'s addressing tests (the
sibling call site migrated first): fakes the RESOLVER
(``krepis.router.resolve_group_structured``), not the adapter
(``resolve_group_spec`` itself), so the provider/transport decision made by
``krepis.router`` is still exercised — only the registry read is stubbed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evals import judge as judge_mod
from tests.test_eval_judge import _make_artifact, _make_llm_output


def _fake_route(**over):
    """What ``krepis.router.resolve_group_spec`` returns for the ``low``
    group from a Lambda: the synthesised litellm_proxy route through the
    krepis router edge."""
    route = {
        "schema_version": 2,
        "group": "low",
        "route": "litellm_proxy",
        "provider": "litellm",
        "deployment_id": "low",
        "api_base_url": "https://router.example:8443",
        "auth_token_type": "litellm_master_key",
        "registry_id": "litellm:group:low",
        "primary_registry_id": "deepseek-v4-flash",
        "params": {},
    }
    route.update(over)
    return route


def _patch_router(monkeypatch, *, route=None, captured=None):
    """Fake the RESOLVER, not the adapter — see module docstring."""
    import krepis.router as _kr

    the_route = route or _fake_route()

    def fake_resolve_structured(group, *, exec_context=None, wire="openai", requires=()):
        if captured is not None:
            captured.append({
                "group": group, "exec_context": exec_context,
                "wire": wire, "requires": requires,
            })
        return the_route

    monkeypatch.setattr(_kr, "resolve_group_structured", fake_resolve_structured)


def _openai_tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _openai_response(*, finish_reason: str, tool_calls=None, content=None,
                      model="deepseek/deepseek-v4-flash", cost: float = 0.0001):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    usage = SimpleNamespace(cost=cost)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _valid_tool_args() -> dict:
    return _make_llm_output().model_dump()


def _patch_llm_client(fake_client):
    """Same style as test_eval_judge.py/_openrouter.py's helper, local copy
    to keep this module's fixtures self-contained."""
    from unittest.mock import patch

    from krepis.llm import LLMClient as _RealLLMClient

    def _factory(spec, **kwargs):
        kwargs.pop("client_factory", None)
        return _RealLLMClient(spec, client_factory=lambda _s, _k: fake_client, **kwargs)

    return patch.object(judge_mod, "LLMClient", side_effect=_factory)


def test_sync_judge_addresses_the_low_group_from_lambda(monkeypatch, live_router_resolution):
    """``evaluate_artifact`` (evaljudge-sync) resolves the ``low`` router
    group from the ``lambda`` exec context, openai wire, non-strict
    structured outputs — LLM_CALLSITE_REGISTRY.yaml's declared shape for
    the ``evaljudge-sync`` row."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _openai_response(
        finish_reason="tool_calls",
        tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
    )
    captured = []
    _patch_router(monkeypatch, captured=captured)

    with _patch_llm_client(fake_client) as mock_llm_cls:
        artifact = _make_artifact("thinktank_thesis")
        judge_mod.evaluate_artifact(artifact, api_key="sk-test")

    assert len(captured) == 1
    ask = captured[0]
    assert ask["group"] == judge_mod.JUDGE_MODEL_GROUP == "low"
    assert ask["exec_context"] == judge_mod.JUDGE_EXEC_CONTEXT == "lambda"
    assert ask["wire"] == "openai"

    spec = mock_llm_cls.call_args.args[0]
    # The router decides provider and model. This call site must not.
    assert spec.provider == "litellm_proxy"
    assert spec.structured_outputs is False
    # Truncation-avoidance default re-applied on top of the resolved spec
    # (frozen dataclass — see _judge_router_spec_and_route).
    assert spec.reasoning == {"exclude": True}


def test_shadow_judge_addresses_the_same_low_group_from_lambda(monkeypatch, live_router_resolution):
    """``evaluate_artifact_openrouter`` (evaljudge-shadow) resolves the
    IDENTICAL router group/context/wire as the sync path — the shadow
    tier's serving path has not been distinct from the primary's since
    alpha-engine-config-I2997 (2026-07-19); this migration preserves that,
    it does not introduce it. See evaluate_artifact_openrouter's docstring
    for the full champion-challenger note."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _openai_response(
        finish_reason="tool_calls",
        tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
    )
    captured = []
    _patch_router(monkeypatch, captured=captured)

    with _patch_llm_client(fake_client):
        artifact = _make_artifact("thinktank_thesis")
        judge_mod.evaluate_artifact_openrouter(artifact, api_key="sk-test")

    assert len(captured) == 1
    ask = captured[0]
    assert ask["group"] == judge_mod.JUDGE_MODEL_GROUP == "low"
    assert ask["exec_context"] == judge_mod.JUDGE_EXEC_CONTEXT == "lambda"
    assert ask["wire"] == "openai"


def test_sync_and_shadow_use_distinct_callsite_ids(monkeypatch, live_router_resolution):
    """The router group is shared, but per-callsite cost attribution must
    not collapse: `evaljudge-sync` and `evaljudge-shadow` are separate
    LLM_CALLSITE_REGISTRY.yaml rows and must reach LLMClient as such."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _openai_response(
        finish_reason="tool_calls",
        tool_calls=[_openai_tool_call("RubricEvalLLMOutput", _valid_tool_args())],
    )
    _patch_router(monkeypatch)

    with _patch_llm_client(fake_client) as mock_llm_cls:
        judge_mod.evaluate_artifact(_make_artifact("thinktank_thesis"), api_key="sk-test")
    assert mock_llm_cls.call_args.kwargs["callsite_id"] == "evaljudge-sync"

    with _patch_llm_client(fake_client) as mock_llm_cls:
        judge_mod.evaluate_artifact_openrouter(_make_artifact("thinktank_thesis"), api_key="sk-test")
    assert mock_llm_cls.call_args.kwargs["callsite_id"] == "evaljudge-shadow"


def test_module_constructs_no_provider_pinned_spec():
    """The pin this migration removes would be silent if it came back — the
    call would simply start working again against a provider nobody chose
    and outside the DLP-scanned router edge.

    Structural, not textual: the prose in this module legitimately NAMES
    the old direct-OpenRouter construction to explain the migration. What
    must not come back is a ``ModelSpec(provider="openrouter", ...)`` call,
    or a binding of ``OPENROUTER_API_KEY``. Both are AST facts."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "evals" / "judge.py"
    tree = ast.parse(src.read_text())

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ModelSpec"):
            continue
        for kw in node.keywords:
            assert not (kw.arg == "provider" and isinstance(kw.value, ast.Constant) and kw.value.value == "openrouter"), (
                "evals/judge.py constructs a ModelSpec(provider='openrouter', ...) — "
                "model, endpoint and credential are registry decisions resolved by "
                "krepis.router.resolve_group_spec (alpha-engine-config-I6559)"
            )

    bound = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "OPENROUTER_API_KEY" not in bound, (
        "evals/judge.py binds OPENROUTER_API_KEY — no agent may be directly "
        "linked to OpenRouter (Brian's ruling 2026-08-03, I6367)"
    )


# ── alpha-engine-config-I7904 ────────────────────────────────────────────
#
# The judge FORCES a tool call. `low`'s declared primary refuses one — the
# registry says so in `capabilities.tool_choice: false`, and this module's own
# LLM_CALLSITE_REGISTRY row says so from the other side in
# `requires_forced_tool_call: true`. Neither fact reached routing, so the judge
# resolved to the member that cannot serve it and took an identical permanent
# 400 on all three attempts, which the router then failed over and reported as
# a rate limit on a different model entirely.

def test_the_judge_declares_the_call_shape_it_cannot_do_without(monkeypatch):
    """Resolution must be asked for a tool-capable member, not just a tier."""
    captured: dict = {}

    def _spy(group, **kwargs):
        captured["group"] = group
        captured.update(kwargs)
        raise _StopResolution()

    class _StopResolution(Exception):
        pass

    import krepis.router as _kr
    monkeypatch.setattr(_kr, "resolve_group_spec", _spy)

    from evals import judge as _judge
    with pytest.raises(_StopResolution):
        _judge._judge_router_spec_and_route(max_tokens=1024)

    assert captured["group"] == "low"
    assert captured["requires"] == ("tool_choice",)


def test_a_permanent_contract_error_is_not_retried(monkeypatch):
    """A 400 refusing the request shape is identical on every attempt.

    Spending the retry budget on it turns one loud, correct error into three
    and delays it, which is exactly what the I7904 log shows.
    """
    from krepis.llm_errors import PermanentContractError

    from evals import judge as _judge

    attempts: list[int] = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        def complete(self, **kwargs):
            attempts.append(1)
            raise PermanentContractError(
                "Thinking mode does not support this tool_choice",
                status_code=400,
                deployment="low-deepseek-v4-flash-low",
            )

    monkeypatch.setattr(_judge, "LLMClient", _Client)
    monkeypatch.setattr(
        _judge, "_judge_router_spec_and_route",
        lambda **kw: (_FakeSpec(), {"route": "litellm_proxy"}),
    )

    with pytest.raises(PermanentContractError) as exc:
        _judge._call_openrouter_judge_llm(
            "rendered", agent_id="a", request_model="m", max_tokens=64,
            api_key=None, max_retries=3, log_prefix="[t]",
        )

    assert len(attempts) == 1, f"a permanent 400 consumed {len(attempts)} attempts"
    assert "Thinking mode does not support this tool_choice" in str(exc.value)


class _FakeSpec:
    model = "low-glm-4.7-flash"
    provider = "litellm"
    registry_id = "glm-4.7-flash"
