"""Per-thesis isolation + structured-output retry for held-stock updates.

Regression + behavior pins for the 2026-05-16 Saturday SF recovery
failure. The Research Lambda (alpha-engine-research-runner,
ExecutedVersion 209) ran but returned::

    {"status":"ERROR","date":"2026-05-15",
     "error":"1 validation error for HeldThesisUpdateLLMOutput\\n"
             "catalysts\\n  Input should be a valid list "
             "[type=list_type, input_value='\\n<parameter "
             "name=\\"catal...en satellite programs\\n', "
             "input_type=str]"}

Root cause: a single held-thesis-update LLM call emitted ``catalysts``
as a *string* of leaked Anthropic tool-use XML
(``<parameter name="catalysts">…satellite programs``) instead of a
JSON list. ``HeldThesisUpdateLLMOutput.catalysts`` is typed
``list[str]``, so Pydantic raised ``list_type`` ValidationError. In
strict mode (``STRICT_VALIDATION`` default = true) that error
re-raised inside ``_update_thesis_for_held_stock`` and propagated
uncaught through the sector-team held-thesis loop to
``lambda/handler.py:520`` (``return {"status":"ERROR", ...}``), so ONE
malformed thesis update aborted the ENTIRE weekly run (~25 theses +
sector teams + CIO all discarded).

The fix (matches the standing "one item must never fail the whole
batch" rule):

  1. drive ``with_structured_output(..., include_raw=True)`` and retry
     the call ONCE on a parse/validation error (these tool-XML leaks
     are transient model nondeterminism — a single retry recovers most
     cases), then
  2. if it STILL fails, degrade THIS thesis only — carry the prior
     thesis forward and continue the run. The function NEVER re-raises
     (even in strict mode), so a single held-thesis-update failure can
     no longer change the Lambda's overall status away from
     OK/SKIPPED.

These tests use the ``monkeypatch`` fixture (NOT ``unittest.mock.patch``)
to match the convention in ``test_macro_sector_coherence_gate.py`` —
this repo has a known test-bleed issue with ``unittest.mock.patch``
(a test passes alone but fails in the full alphabetical suite run).
"""

from __future__ import annotations

import pytest

from graph.state_schemas import HeldThesisUpdateLLMOutput


# The exact malformed payload from the 2026-05-16 failure: ``catalysts``
# arrives as a string containing leaked Anthropic tool-use XML instead
# of a JSON list. Reproduced verbatim (truncated tail) so the test
# fails for the same reason production did.
_LEAKED_TOOL_XML = (
    '\n<parameter name="catalysts">- Q3 earnings beat\n'
    "- new defense contract awards\n- expansion into European "
    "satellite programs\n"
)


class _FakeStructuredLLM:
    """Mimics ``llm.with_structured_output(Schema, include_raw=True)``.

    ``responses`` is a list of dicts in the include_raw contract shape
    (``{"raw": ..., "parsed": ..., "parsing_error": ...}``) consumed
    one per ``.invoke()`` call, so a test can script "attempt 1 fails,
    attempt 2 succeeds".
    """

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.invoke_calls = 0

    def invoke(self, *args, **kwargs):
        idx = min(self.invoke_calls, len(self._responses) - 1)
        self.invoke_calls += 1
        return self._responses[idx]


class _FakeLLM:
    def __init__(self, structured: _FakeStructuredLLM):
        self._structured = structured
        self.with_structured_output_kwargs: dict | None = None

    def with_structured_output(self, schema, **kwargs):
        self.with_structured_output_kwargs = kwargs
        return self._structured


def _validation_error_response() -> dict:
    """An include_raw response whose ``parsed`` is None because Pydantic
    rejected the leaked tool-XML string assigned to ``catalysts``."""
    try:
        HeldThesisUpdateLLMOutput(catalysts=_LEAKED_TOOL_XML)
        raise AssertionError(
            "Expected HeldThesisUpdateLLMOutput to reject a str catalysts"
        )
    except Exception as exc:  # pydantic.ValidationError
        parsing_error = exc
    return {"raw": object(), "parsed": None, "parsing_error": parsing_error}


def _valid_response(**fields) -> dict:
    return {
        "raw": object(),
        "parsed": HeldThesisUpdateLLMOutput(**fields),
        "parsing_error": None,
    }


def _patch_call_site(monkeypatch, fake_llm: _FakeLLM) -> None:
    """Patch ChatAnthropic + prompt loading in the held-thesis call site."""
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


def test_malformed_catalysts_degrades_to_prior_thesis(monkeypatch):
    """(a) A held-thesis-update returning the malformed ``catalysts``
    string on BOTH attempts must degrade gracefully: carry the prior
    thesis forward, do NOT raise. The overall run is unaffected.

    This is the exact 2026-05-16 production payload.
    """
    from agents.sector_teams import sector_team

    structured = _FakeStructuredLLM([
        _validation_error_response(),  # attempt 1
        _validation_error_response(),  # attempt 2 (retry)
    ])
    fake_llm = _FakeLLM(structured)
    _patch_call_site(monkeypatch, fake_llm)

    # Must NOT raise — per-thesis isolation. Pre-fix this re-raised the
    # ValidationError (strict mode default) up to lambda/handler.py:520.
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

    # Prior thesis carried forward (no-update path) — scores intact.
    assert result["ticker"] == "LMT"
    assert result["final_score"] == 62.0
    assert result["rating"] == "HOLD"
    assert result["bull_case"] == "old bull"
    assert result["triggers"] == ["earnings"]
    assert result["stale_days"] == 0
    # It retried once (2 total attempts) before degrading.
    assert structured.invoke_calls == 2
    # And it used the include_raw contract.
    assert fake_llm.with_structured_output_kwargs == {"include_raw": True}


def test_retry_recovers_when_second_attempt_valid(monkeypatch):
    """(b) First attempt emits the malformed ``catalysts`` string,
    second attempt is valid → the thesis updates normally (the transient
    tool-XML leak is recovered by the single retry)."""
    from agents.sector_teams import sector_team

    structured = _FakeStructuredLLM([
        _validation_error_response(),  # attempt 1: tool-XML leak
        _valid_response(  # attempt 2: clean structured output
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

    assert structured.invoke_calls == 2  # retried exactly once
    # Narrative fields updated from the successful retry.
    assert result["bull_case"] == "new bull narrative"
    assert result["bear_case"] == "new bear narrative"
    assert result["catalysts"] == ["Q3 earnings beat", "new defense contract"]
    assert result["conviction"] == 70
    # Prior scores still preserved (schema has no score fields).
    assert result["final_score"] == 62.0
    assert result["rating"] == "HOLD"
    assert result["last_updated"] == "2026-05-15"
    assert result["stale_days"] == 0


def test_validation_error_does_not_propagate_in_strict_mode(monkeypatch):
    """(c) Regression: a single held-thesis ValidationError must NOT
    propagate, even with STRICT_VALIDATION=true (the production default
    that caused the 2026-05-16 abort). The escape path to
    ``lambda/handler.py:520`` only exists if this function RAISES — so
    "returns instead of raising under strict mode" is the regression
    guarantee. Pre-fix, ``is_strict_validation_enabled()`` → ``raise``
    here aborted the whole weekly run."""
    from agents.sector_teams import sector_team

    monkeypatch.setenv("STRICT_VALIDATION", "true")

    structured = _FakeStructuredLLM([
        _validation_error_response(),
        _validation_error_response(),
    ])
    fake_llm = _FakeLLM(structured)
    _patch_call_site(monkeypatch, fake_llm)

    # The assertion IS that this call returns rather than raising.
    try:
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
    except Exception as exc:  # pragma: no cover — this is the bug
        pytest.fail(
            "Held-thesis ValidationError propagated under strict mode — "
            f"this aborts the whole weekly run (handler.py:520): {exc!r}"
        )

    # Degraded to prior thesis, run continues.
    assert result["final_score"] == 62.0
    assert result["ticker"] == "LMT"


def test_no_prior_thesis_isolation_marks_score_failed(monkeypatch):
    """Defensive: if the LLM keeps failing AND there is no prior thesis
    to carry forward, the isolation fallback still returns (does not
    raise) and marks the thesis ``score_failed`` so downstream guards
    downgrade it rather than emitting an unscored BUY."""
    from agents.sector_teams import sector_team

    structured = _FakeStructuredLLM([
        _validation_error_response(),
        _validation_error_response(),
    ])
    fake_llm = _FakeLLM(structured)
    _patch_call_site(monkeypatch, fake_llm)

    result = sector_team._update_thesis_for_held_stock(
        ticker="NEW",
        triggers=["earnings"],
        prior_thesis=None,
        news_data=None,
        analyst_data=None,
        run_date="2026-05-15",
        team_id="industrials",
        api_key="test-key",
    )

    assert result.get("score_failed") is True
    assert "final_score" not in result
