"""Tests for the single-agent research producer (config#1223 / M3). The single
agent provides qualitative scores via ONE LLM call; the payload must satisfy the
SAME producer contract as the champion. LLM is never hit — the builder takes
assessments directly, and the run test injects assess_fn."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from producers.registry import RESEARCH_PRODUCERS  # noqa: E402
from producers.single_agent import (  # noqa: E402
    CHALLENGER_EXEC_CONTEXT,
    CHALLENGER_GROUP,
    RankingProducerOutput,
    assess_candidates,
    build_single_agent_signals,
    run_single_agent_producer,
)
from tests.test_signals_producer_contract import (  # noqa: E402
    _REQUIRED_PER_ITEM,
    _REQUIRED_TOP_LEVEL,
)


def _inputs():
    scanner_tickers = ["AAA", "BBB", "CCC", "DDD"]
    technical_scores = {
        "AAA": {"technical_score": 80.0, "momentum_20d": 0.05},
        "BBB": {"technical_score": 75.0, "momentum_20d": -0.05},
        "CCC": {"technical_score": 30.0, "momentum_20d": 0.0},
        "DDD": {"technical_score": 85.0, "momentum_20d": 0.01},
    }
    assessments = [
        {"ticker": "AAA", "qual_score": 90.0, "conviction": "rising", "brief_thesis": "strong"},
        {"ticker": "BBB", "qual_score": 70.0, "conviction": "stable", "brief_thesis": "ok"},
        {"ticker": "CCC", "qual_score": 20.0, "conviction": "declining", "brief_thesis": "weak"},
        {"ticker": "DDD", "qual_score": 88.0, "conviction": "stable", "brief_thesis": "held"},
    ]
    population = [
        {"ticker": "DDD", "sector": "Technology", "long_term_rating": "BUY",
         "long_term_score": 86.0, "conviction": "stable", "price_target_upside": 0.1},
    ]
    sector_map = {"AAA": "Technology", "BBB": "Healthcare", "CCC": "Technology", "DDD": "Technology"}
    return scanner_tickers, technical_scores, assessments, population, sector_map


def _build():
    st, ts, asmt, pop, sm = _inputs()
    return build_single_agent_signals(
        "2026-06-19", scanner_tickers=st, assessments=asmt, technical_scores=ts,
        population=pop, prior_theses={}, sector_map=sm,
    )


def test_single_agent_payload_satisfies_producer_contract():
    payload = _build()
    assert _REQUIRED_TOP_LEVEL <= set(payload), _REQUIRED_TOP_LEVEL - set(payload)
    for section in ("universe", "buy_candidates"):
        for item in payload[section]:
            assert _REQUIRED_PER_ITEM <= set(item), (section, _REQUIRED_PER_ITEM - set(item))


def test_enter_signals_carry_numeric_score():
    for e in (u for u in _build()["universe"] if u["signal"] == "ENTER"):
        assert isinstance(e["score"], (int, float)) and e["score"] is not None, e


def test_single_agent_emits_qualitative_score():
    # The defining difference vs the no-agent floor: qual sub-scores ARE present.
    payload = _build()
    aaa = payload["signals"]["AAA"]
    assert aaa["sub_scores"]["qual"] == 90.0, aaa
    assert aaa["sub_scores"]["quant"] == 80.0, aaa


def test_gate_and_provenance():
    sig = _build()["signals"]
    assert sig["AAA"]["signal"] == "ENTER" and sig["AAA"]["stance_source"] == "cio_entrant"
    assert sig["DDD"]["signal"] == "ENTER" and sig["DDD"]["stance_source"] == "reaffirmed_hold"
    # CCC: weak quant (30) + weak qual (20) → below threshold, not held → dropped.
    assert "CCC" not in sig


def test_ranking_output_schema_rejects_empty():
    RankingProducerOutput(assessments=[{"ticker": "X", "qual_score": 50}])  # ok
    with pytest.raises(ValidationError):
        RankingProducerOutput(assessments=[])  # min_length=1


def test_run_injects_assess_fn(monkeypatch):
    from unittest.mock import MagicMock

    import data.fetchers.price_fetcher as pf
    import data.scanner_orchestrator as so
    import scoring.universe_membership as um

    # The feed is the champion CUT, not candidates.json
    # (alpha-engine-config-I7823). Unpatched, this reaches S3 — which is the
    # point: the producer no longer has a fail-soft empty-list path to fall
    # into when the artifact is missing.
    monkeypatch.setattr(um, "resolve_feed_cut", lambda **kw: (
        ["AAA", "BBB"],
        {"cut": "attractiveness_top_60", "run_date": "2026-06-19",
         "cut_effective_date": "2026-06-19", "basis": "attractiveness_rank", "size": 2},
    ))

    am = MagicMock()
    am.load_population.return_value = []
    am.load_latest_theses.return_value = {}
    monkeypatch.setattr(pf, "fetch_sp500_sp400_with_sectors",
                        lambda: (["AAA", "BBB"], {"AAA": "Technology", "BBB": "Healthcare"}))
    monkeypatch.setattr(so, "_build_technical_scores_from_feature_store",
                        lambda c, s: ({"AAA": {"technical_score": 80.0, "momentum_20d": 0.05},
                                       "BBB": {"technical_score": 75.0, "momentum_20d": 0.0}}, 2))
    captured = {}

    def fake_assess(scanner_tickers, technical_scores, sector_map):
        captured["called"] = True
        return [{"ticker": "AAA", "qual_score": 90.0, "conviction": "rising", "brief_thesis": ""},
                {"ticker": "BBB", "qual_score": 80.0, "conviction": "stable", "brief_thesis": ""}]

    payload = run_single_agent_producer("2026-06-19", am, assess_fn=fake_assess)
    assert captured.get("called") is True
    assert _REQUIRED_TOP_LEVEL <= set(payload)
    assert {u["ticker"] for u in payload["universe"]} == {"AAA", "BBB"}


def test_registry_has_single_agent_challenger():
    spec = RESEARCH_PRODUCERS.get("single_agent_quant")
    assert spec is not None and spec.kind == "challenger" and spec.build is not None


# ── assess_candidates (alpha-engine-config-I6367: router group addressing,
#    fake krepis.llm transport — no real network) ───────────────────────────


def _fake_route(**over):
    """What krepis.router.resolve_group_spec returns for the `high` group from
    a Lambda: the synthesised litellm_proxy route, because the registry
    declares `lambda` on no model entry."""
    route = {
        "schema_version": 2,
        "group": "high",
        "route": "litellm_proxy",
        "provider": "litellm",
        "deployment_id": "high",
        "api_base_url": "https://router.example:8443",
        "auth_token_type": "litellm_master_key",
        "registry_id": "litellm:group:high",
        "primary_registry_id": "deepseek-v4-pro-max",
        "params": {},
    }
    route.update(over)
    return route


def _patch_router(monkeypatch, *, route=None, captured=None):
    """Fake the RESOLVER, not the adapter.

    `resolve_group_spec` is the thing under test at this call site — faking it
    would leave the provider/transport decision (the defect that made the
    proxy route import litellm in a Lambda) untested. So only
    `resolve_group_structured`, krepis' registry read, is replaced.
    """
    import krepis.router as _kr

    the_route = route or _fake_route()

    def fake_resolve_structured(group, *, exec_context=None, wire="openai"):
        if captured is not None:
            captured.append(
                {"group": group, "exec_context": exec_context, "wire": wire}
            )
        return the_route

    monkeypatch.setattr(_kr, "resolve_group_structured", fake_resolve_structured)


class _FakeAgentPrompt:
    def __init__(self):
        self.text = "SYSTEM PROMPT PLACEHOLDER"

    def langsmith_metadata(self):
        return {}


def test_assess_candidates_addresses_the_high_group_non_strict(monkeypatch, live_router_resolution):
    import agents.prompt_loader as prompt_loader

    monkeypatch.setattr(prompt_loader, "load_prompt", lambda name: _FakeAgentPrompt())

    # The router credential is resolved by NAME at call time — the registry
    # says which name, this test provides a value for it.
    monkeypatch.setenv("LITELLM_MASTER_KEY", "router-consumer-test")
    captured_resolves = []
    _patch_router(monkeypatch, captured=captured_resolves)
    captured_specs = []

    class FakeCompletions:
        def create(self, **kwargs):
            return type("Resp", (), {
                "choices": [type("Choice", (), {
                    "message": type("Msg", (), {
                        "content": (
                            '{"assessments": ['
                            '{"ticker": "AAA", "qual_score": 91, "conviction": "rising", '
                            '"brief_thesis": "strong"}]}'
                        ),
                    })(),
                })()],
                "model": "deepseek/deepseek-v4-pro",
                "usage": None,
            })()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    def factory(spec, api_key):
        captured_specs.append(spec)
        return FakeClient()

    out = assess_candidates(
        ["AAA"], {"AAA": {"technical_score": 80.0}}, {"AAA": "Technology"},
        api_key="sk-test", client_factory=factory,
    )
    assert out == [
        {"ticker": "AAA", "qual_score": 91.0, "conviction": "rising", "brief_thesis": "strong"}
    ]
    assert len(captured_specs) == 1
    spec = captured_specs[0]
    # The router decides provider and model. This call site must not.
    assert spec.provider == "litellm_proxy"
    assert spec.model == "high"

    # The edge is a custom OpenAI-compatible endpoint. Provider "litellm"
    # would bind TRANSPORT_LITELLM — the in-process Router, which calls
    # providers directly from the consumer and needs litellm importable
    # inside the Lambda.
    from krepis.llm_config import TRANSPORT_OPENAI
    assert spec.transport == TRANSPORT_OPENAI
    assert spec.base_url == "https://router.example:8443"

    assert len(captured_resolves) == 1
    ask = captured_resolves[0]
    assert ask["group"] == CHALLENGER_GROUP == "high"
    assert ask["exec_context"] == CHALLENGER_EXEC_CONTEXT == "lambda"
    # An anthropic-wire fallback would hand this openai-transport client a URL
    # it cannot speak.
    assert ask["wire"] == "openai"
    # REQUIRED, not incidental — see producers/single_agent.py: strict
    # json_schema is live-verified unreliable for DeepSeek-family models
    # against this exact schema. Passed EXPLICITLY so a registry default
    # cannot re-enable it.
    assert spec.structured_outputs is False


def test_assess_candidates_reads_no_openrouter_key(monkeypatch, live_router_resolution):
    """Brian's ruling 2026-08-03 (alpha-engine-config-I6367): no agent
    directly linked to OpenRouter. This call site used to RAISE when
    config.OPENROUTER_API_KEY was empty — the key was load-bearing. It must
    now run without one, because the credential is named by the registry and
    resolved by krepis at call time."""
    import agents.prompt_loader as prompt_loader
    monkeypatch.setattr(prompt_loader, "load_prompt", lambda name: _FakeAgentPrompt())
    import config
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "router-consumer-test")
    _patch_router(monkeypatch)

    class FakeCompletions:
        def create(self, **kwargs):
            return type("Resp", (), {
                "choices": [type("Choice", (), {
                    "message": type("Msg", (), {
                        "content": (
                            '{"assessments": [{"ticker": "AAA", '
                            '"qual_score": 50, "conviction": "stable", '
                            '"brief_thesis": "x"}]}'
                        ),
                    })(),
                })()],
                "model": "deepseek-v4-pro-max",
                "usage": None,
            })()

    def factory(spec, api_key):
        return type("C", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()})()

    out = assess_candidates(["AAA"], {}, {}, api_key=None, client_factory=factory)
    assert [a["ticker"] for a in out] == ["AAA"]


def test_module_constructs_no_provider_pinned_spec():
    """The pin is what died on 2026-08-02 when the OpenRouter balance went
    negative, and its reintroduction would be silent — the call would simply
    start working again against a provider nobody chose.

    Structural, not textual: the prose in this module legitimately NAMES the
    old pin to explain why it is gone. What must not come back is a
    ``ModelSpec(...)`` built here, or a binding of ``OPENROUTER_API_KEY``.
    Both are AST facts."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "producers" / "single_agent.py"
    tree = ast.parse(src.read_text())

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "ModelSpec" not in called, (
        "producers/single_agent.py constructs a ModelSpec — model, endpoint "
        "and credential are registry decisions resolved by "
        "krepis.router.resolve_group_spec (alpha-engine-config-I6367)"
    )

    bound = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "OPENROUTER_API_KEY" not in bound, (
        "producers/single_agent.py binds OPENROUTER_API_KEY — no agent may be "
        "directly linked to OpenRouter (Brian's ruling 2026-08-03)"
    )
