"""All-agents-strict contract for held-stock thesis updates.

CONTRACT CHANGE (Brian, 2026-05-16) — this file REPLACES the former
``test_held_thesis_isolation.py``, whose tests pinned #193's
"degrade gracefully — carry the prior thesis forward, run continues"
behavior. That behavior is INTENTIONALLY REMOVED, not a regression to
preserve:

  "We don't get anything from this process if the sectors, or any other
   agent for that matter, fail/don't run."

A held-thesis update is one of the agents in scope. Under the rework:

  * A transient tool-XML schema leak (the 2026-05-16 ``catalysts``
    string-not-list nondeterminism) gets a small bounded parse re-roll
    (``_MAX_PARSE_ATTEMPTS`` = 3) — these recover on a re-roll.
  * If it STILL fails after the re-rolls, ``_update_thesis_for_held_stock``
    RAISES (it no longer carries the prior thesis forward). The raise
    propagates to the team's hard ``error`` and ``score_aggregator``
    hard-fails the whole run (no signals.json / email / DB write).
  * A 429 is handled by the deadline-bounded
    ``invoke_with_rate_limit_retry`` wrapper inside the thunk; once it
    has exhausted the ~75-min window the held-thesis path fails fast
    (re-rolling can't fix an org TPM ceiling).

Pre-existing tests rewritten here (was test_held_thesis_isolation.py):
  - test_malformed_catalysts_degrades_to_prior_thesis
        -> test_malformed_catalysts_raises_after_reroll
  - test_retry_recovers_when_second_attempt_valid
        -> test_reroll_recovers_when_a_later_attempt_valid (kept;
          3-attempt parse budget, post-budget = raise not carry)
  - test_validation_error_does_not_propagate_in_strict_mode
        -> test_validation_error_DOES_propagate (OPPOSITE assertion)
  - test_no_prior_thesis_isolation_marks_score_failed
        -> test_no_prior_thesis_still_raises (no isolation fallback)

Uses the ``monkeypatch`` fixture (NOT ``unittest.mock.patch``) per the
documented full-suite bleed in this repo.
"""

from __future__ import annotations

import pytest

from graph.state_schemas import HeldThesisUpdateLLMOutput

_LEAKED_TOOL_XML = (
    '\n<parameter name="catalysts">- Q3 earnings beat\n'
    "- new defense contract awards\n- expansion into European "
    "satellite programs\n"
)


class _FakeStructuredLLM:
    def __init__(self, responses):
        self._responses = responses
        self.invoke_calls = 0

    def invoke(self, *args, **kwargs):
        idx = min(self.invoke_calls, len(self._responses) - 1)
        self.invoke_calls += 1
        return self._responses[idx]


class _FakeLLM:
    def __init__(self, structured):
        self._structured = structured
        self.with_structured_output_kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.with_structured_output_kwargs = kwargs
        return self._structured


def _validation_error_response():
    try:
        HeldThesisUpdateLLMOutput(catalysts=_LEAKED_TOOL_XML)
        raise AssertionError(
            "Expected HeldThesisUpdateLLMOutput to reject a str catalysts"
        )
    except Exception as exc:
        parsing_error = exc
    return {"raw": object(), "parsed": None, "parsing_error": parsing_error}


def _valid_response(**fields):
    return {
        "raw": object(),
        "parsed": HeldThesisUpdateLLMOutput(**fields),
        "parsing_error": None,
    }


def _patch_call_site(monkeypatch, fake_llm):
    from agents.sector_teams import sector_team

    monkeypatch.setattr(
        sector_team, "ChatAnthropic", lambda *a, **k: fake_llm
    )

    class _FakePrompt:
        def format(self, **kwargs):
            return "prompt"

        def langsmith_metadata(self):
            return {}

    monkeypatch.setattr(
        sector_team, "load_prompt", lambda name: _FakePrompt()
    )
    monkeypatch.setattr(
        sector_team, "format_structured_thesis_for_prompt",
        lambda *a, **k: "",
    )


_PRIOR = {
    "ticker": "LMT",
    "sector": "Industrials",
    "team_id": "industrials",
    "final_score": 62.0,
    "quant_score": 60.0,
    "qual_score": 64.0,
    "rating": "HOLD",
    "conviction": "stable",
    "bull_case": "old bull",
    "bear_case": "old bear",
}


def test_malformed_catalysts_raises_after_reroll(monkeypatch):
    """REWRITTEN from test_malformed_catalysts_degrades_to_prior_thesis.
    Old (#193): carry prior thesis forward, no raise. New: RAISE.

    config#2247: the deterministic per-ticker failure now raises a
    ``QuarantinableThesisError`` (a RuntimeError subclass) rather than a bare
    RuntimeError — ``_update_thesis_for_held_stock`` STILL raises; the caller
    (``run_sector_team``) catches it to quarantine the single ticker instead of
    killing the whole run. No prior-thesis carry-forward either way."""
    from agents.sector_teams import sector_team
    from agents.sector_teams.sector_team import QuarantinableThesisError

    structured = _FakeStructuredLLM([_validation_error_response()] * 3)
    fake_llm = _FakeLLM(structured)
    _patch_call_site(monkeypatch, fake_llm)

    with pytest.raises(QuarantinableThesisError, match="per-ticker quarantine"):
        sector_team._update_thesis_for_held_stock(
            ticker="LMT",
            triggers=["earnings"],
            prior_thesis=dict(_PRIOR),
            news_data=None,
            analyst_data=None,
            run_date="2026-05-15",
            team_id="industrials",
            api_key="test-key",
        )

    assert structured.invoke_calls == 3
    assert fake_llm.with_structured_output_kwargs == {"include_raw": True}


def test_reroll_recovers_when_a_later_attempt_valid(monkeypatch):
    """KEPT (was test_retry_recovers_when_second_attempt_valid): the
    transient tool-XML leak is still recovered by a parse re-roll."""
    from agents.sector_teams import sector_team

    structured = _FakeStructuredLLM([
        _validation_error_response(),
        _validation_error_response(),
        _valid_response(
            bull_case="new bull narrative",
            bear_case="new bear narrative",
            catalysts=["Q3 earnings beat", "new defense contract"],
            conviction=70,
        ),
    ])
    fake_llm = _FakeLLM(structured)
    _patch_call_site(monkeypatch, fake_llm)

    result = sector_team._update_thesis_for_held_stock(
        ticker="LMT",
        triggers=["earnings"],
        prior_thesis=dict(_PRIOR),
        news_data=None,
        analyst_data=None,
        run_date="2026-05-15",
        team_id="industrials",
        api_key="test-key",
    )

    assert structured.invoke_calls == 3
    assert result["bull_case"] == "new bull narrative"
    assert result["bear_case"] == "new bear narrative"
    assert result["catalysts"] == ["Q3 earnings beat", "new defense contract"]
    assert result["conviction"] == 70
    assert result["final_score"] == 62.0
    assert result["rating"] == "HOLD"
    assert result["last_updated"] == "2026-05-15"
    assert result["stale_days"] == 0


def test_validation_error_DOES_propagate(monkeypatch):
    """REWRITTEN — OPPOSITE of the old
    test_validation_error_does_not_propagate_in_strict_mode."""
    from agents.sector_teams import sector_team

    monkeypatch.setenv("STRICT_VALIDATION", "true")

    structured = _FakeStructuredLLM([_validation_error_response()] * 3)
    fake_llm = _FakeLLM(structured)
    _patch_call_site(monkeypatch, fake_llm)

    with pytest.raises(RuntimeError, match="cannot produce real output"):
        sector_team._update_thesis_for_held_stock(
            ticker="LMT",
            triggers=["earnings"],
            prior_thesis=dict(_PRIOR),
            news_data=None,
            analyst_data=None,
            run_date="2026-05-15",
            team_id="industrials",
            api_key="test-key",
        )


def test_no_prior_thesis_still_raises(monkeypatch):
    """REWRITTEN from test_no_prior_thesis_isolation_marks_score_failed.
    No isolation fallback at all — raises regardless of prior thesis.
    config#2247: the raise is now a quarantine-eligible QuarantinableThesisError
    (caught upstream to quarantine), never a carry-forward."""
    from agents.sector_teams import sector_team
    from agents.sector_teams.sector_team import QuarantinableThesisError

    structured = _FakeStructuredLLM([_validation_error_response()] * 3)
    fake_llm = _FakeLLM(structured)
    _patch_call_site(monkeypatch, fake_llm)

    with pytest.raises(QuarantinableThesisError, match="per-ticker quarantine"):
        sector_team._update_thesis_for_held_stock(
            ticker="NEW",
            triggers=["earnings"],
            prior_thesis=None,
            news_data=None,
            analyst_data=None,
            run_date="2026-05-15",
            team_id="industrials",
            api_key="test-key",
        )


def test_429_past_deadline_fails_fast_no_carry_forward(monkeypatch):
    """A 429 that survives the deadline-bounded wrapper propagates
    immediately (no parse re-roll spin, no carry-forward)."""
    from agents import langchain_utils
    from agents.sector_teams import sector_team

    monkeypatch.setattr(
        langchain_utils, "RATE_LIMIT_RETRY_DEADLINE_SECONDS", 0.01
    )
    monkeypatch.setattr(langchain_utils.time, "sleep", lambda s: None)

    class _FakeResp:
        headers: dict = {}

    def _make_429():
        import anthropic

        try:
            return anthropic.RateLimitError(
                message="org rate limit",
                response=_FakeResp(),  # type: ignore[arg-type]
                body=None,
            )
        except Exception:
            class _Duck(Exception):
                status_code = 429

                def __init__(self):
                    super().__init__("rate limit 429")
                    self.response = _FakeResp()

            return _Duck()

    class _StructuredAlways429:
        invoke_calls = 0

        def invoke(self, *a, **k):
            type(self).invoke_calls += 1
            raise _make_429()

    class _LLM429:
        def with_structured_output(self, schema, **kwargs):
            return _StructuredAlways429()

    monkeypatch.setattr(
        sector_team, "ChatAnthropic", lambda *a, **k: _LLM429()
    )

    class _FakePrompt:
        def format(self, **kwargs):
            return "prompt"

        def langsmith_metadata(self):
            return {}

    monkeypatch.setattr(
        sector_team, "load_prompt", lambda name: _FakePrompt()
    )
    monkeypatch.setattr(
        sector_team, "format_structured_thesis_for_prompt",
        lambda *a, **k: "",
    )

    with pytest.raises(Exception) as exc:
        sector_team._update_thesis_for_held_stock(
            ticker="LMT",
            triggers=["earnings"],
            prior_thesis=dict(_PRIOR),
            news_data=None,
            analyst_data=None,
            run_date="2026-05-15",
            team_id="industrials",
            api_key="test-key",
        )

    from agents.langchain_utils import _is_rate_limit_error

    assert _is_rate_limit_error(exc.value)
    assert _StructuredAlways429.invoke_calls == 1
