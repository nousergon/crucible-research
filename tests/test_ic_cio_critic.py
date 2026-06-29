"""Tests for the IC critic reflection loop (config#927).

The critic is a cheap Haiku reviewer that challenges the CIO's advance set
before finalization, mirroring the macro-agent reflection loop. These tests pin:
  - CIOCriticOutput schema shape,
  - _summarize_advanced formatting,
  - run_cio_critic accept / revise / strict-vs-lax-fallback,
  - run_cio_with_reflection: accept short-circuits, revise re-runs run_cio once,
    and reflection_log is populated — all without real LLM/network calls.
"""

import pytest

from agents.investment_committee import ic_cio
from agents.investment_committee.ic_cio import (
    CIOCriticOutput,
    _summarize_advanced,
    run_cio_critic,
    run_cio_with_reflection,
)


def _candidates():
    return [
        {"ticker": "AAA", "team_id": "technology", "conviction": 80,
         "bull_case": "strong moat"},
        {"ticker": "BBB", "team_id": "healthcare", "conviction": 55,
         "bull_case": "thin thesis"},
    ]


class TestCIOCriticOutput:
    def test_minimal(self):
        v = CIOCriticOutput(action="accept")
        assert v.action == "accept"
        assert v.flagged_tickers == [] and v.suggested_drops == []

    def test_revise_with_fields(self):
        v = CIOCriticOutput(
            action="revise", critique="BBB is weak",
            flagged_tickers=["BBB"], suggested_drops=["BBB"],
        )
        assert v.action == "revise" and v.flagged_tickers == ["BBB"]

    def test_rejects_bad_action(self):
        with pytest.raises(Exception):
            CIOCriticOutput(action="maybe")


def test_summarize_advanced():
    cio_result = {
        "advanced_tickers": ["AAA"],
        "entry_theses": {"AAA": {"thesis_summary": "great setup"}},
    }
    out = _summarize_advanced(cio_result, _candidates())
    assert "AAA" in out and "conviction=80" in out and "great setup" in out


def test_summarize_advanced_empty():
    assert "no tickers advanced" in _summarize_advanced(
        {"advanced_tickers": []}, _candidates()
    )


class _FakeStructuredLLM:
    def __init__(self, verdict):
        self._verdict = verdict

    def invoke(self, *_a, **_k):
        return self._verdict


class _FakeLLM:
    def __init__(self, verdict):
        self._verdict = verdict

    def with_structured_output(self, _schema):
        return _FakeStructuredLLM(self._verdict)


def _patch_llm(monkeypatch, verdict):
    monkeypatch.setattr(ic_cio, "ChatAnthropic", lambda **kw: _FakeLLM(verdict))
    # cost-telemetry callback factory is imported lazily inside run_cio_critic
    import graph.llm_cost_tracker as lct
    monkeypatch.setattr(lct, "get_cost_telemetry_callback", lambda: None)
    # invoke_with_rate_limit_retry just calls the thunk in tests
    monkeypatch.setattr(
        ic_cio, "invoke_with_rate_limit_retry",
        lambda fn, label=None: fn(),
    )


class TestRunCioCritic:
    def test_accept(self, monkeypatch):
        _patch_llm(monkeypatch, CIOCriticOutput(action="accept", critique="ok"))
        out = run_cio_critic(
            {"advanced_tickers": ["AAA"], "entry_theses": {}},
            _candidates(), {"market_regime": "neutral"}, {},
        )
        assert out["action"] == "accept"

    def test_revise(self, monkeypatch):
        _patch_llm(monkeypatch, CIOCriticOutput(
            action="revise", critique="BBB weak", flagged_tickers=["BBB"],
            suggested_drops=["BBB"],
        ))
        out = run_cio_critic(
            {"advanced_tickers": ["AAA", "BBB"], "entry_theses": {}},
            _candidates(), {"market_regime": "neutral"}, {},
        )
        assert out["action"] == "revise"
        assert out["flagged_tickers"] == ["BBB"]
        assert out["suggested_drops"] == ["BBB"]

    def test_lax_fallback_on_error(self, monkeypatch):
        # Failure at invoke time (the realistic LLM error path the try guards).
        _patch_llm(monkeypatch, CIOCriticOutput(action="accept"))
        monkeypatch.setattr(
            ic_cio, "invoke_with_rate_limit_retry",
            lambda fn, label=None: (_ for _ in ()).throw(RuntimeError("api down")),
        )
        monkeypatch.setattr(ic_cio, "is_strict_validation_enabled", lambda: False)
        out = run_cio_critic(
            {"advanced_tickers": ["AAA"], "entry_theses": {}},
            _candidates(), {"market_regime": "neutral"}, {},
        )
        assert out["action"] == "accept"
        assert "unavailable" in out["critique"].lower()

    def test_strict_reraises_on_error(self, monkeypatch):
        _patch_llm(monkeypatch, CIOCriticOutput(action="accept"))
        monkeypatch.setattr(
            ic_cio, "invoke_with_rate_limit_retry",
            lambda fn, label=None: (_ for _ in ()).throw(RuntimeError("api down")),
        )
        monkeypatch.setattr(ic_cio, "is_strict_validation_enabled", lambda: True)
        with pytest.raises(RuntimeError):
            run_cio_critic(
                {"advanced_tickers": ["AAA"], "entry_theses": {}},
                _candidates(), {"market_regime": "neutral"}, {},
            )


class TestRunCioWithReflection:
    def _base_kwargs(self):
        return dict(
            candidates=_candidates(),
            macro_context={"market_regime": "neutral", "macro_report": ""},
            sector_ratings={},
            current_population=[],
            open_slots=5,
            exits=[],
            run_date="2026-06-15",
        )

    def test_accept_does_not_rerun(self, monkeypatch):
        calls = {"cio": 0}

        def fake_run_cio(**kw):
            calls["cio"] += 1
            return {"decisions": [], "advanced_tickers": ["AAA"], "entry_theses": {}}

        monkeypatch.setattr(ic_cio, "run_cio", fake_run_cio)
        monkeypatch.setattr(
            ic_cio, "run_cio_critic",
            lambda *a, **k: {"action": "accept", "critique": "ok",
                             "flagged_tickers": [], "suggested_drops": [],
                             "suggested_adds": []},
        )
        result, log = run_cio_with_reflection(**self._base_kwargs())
        assert calls["cio"] == 1  # no re-run on accept
        assert log["critic_action"] == "accept"
        assert log["iterations"] == 1
        assert log["final_advanced"] == ["AAA"]

    def test_revise_reruns_cio_once(self, monkeypatch):
        calls = {"cio": 0}
        seen_prior = {}

        def fake_run_cio(**kw):
            calls["cio"] += 1
            seen_prior["last"] = kw.get("prior_decisions")
            adv = ["AAA", "BBB"] if calls["cio"] == 1 else ["AAA"]
            return {"decisions": [], "advanced_tickers": adv, "entry_theses": {}}

        monkeypatch.setattr(ic_cio, "run_cio", fake_run_cio)
        monkeypatch.setattr(
            ic_cio, "run_cio_critic",
            lambda *a, **k: {"action": "revise", "critique": "drop BBB",
                             "flagged_tickers": ["BBB"], "suggested_drops": ["BBB"],
                             "suggested_adds": []},
        )
        result, log = run_cio_with_reflection(max_iterations=2, **self._base_kwargs())
        assert calls["cio"] == 2  # initial + one revision
        assert log["critic_action"] == "revise"
        assert log["initial_advanced"] == ["AAA", "BBB"]
        assert log["final_advanced"] == ["AAA"]
        # critique threaded into the re-run's prior_decisions
        assert any(
            d.get("ticker") == "__CRITIC__" for d in (seen_prior["last"] or [])
        )

    def test_empty_candidates_skips_critic(self, monkeypatch):
        called = {"critic": 0}
        monkeypatch.setattr(
            ic_cio, "run_cio",
            lambda **kw: {"decisions": [], "advanced_tickers": [], "entry_theses": {}},
        )

        def _critic(*a, **k):
            called["critic"] += 1
            return {"action": "accept"}

        monkeypatch.setattr(ic_cio, "run_cio_critic", _critic)
        kw = self._base_kwargs()
        kw["candidates"] = []
        result, log = run_cio_with_reflection(**kw)
        assert called["critic"] == 0
