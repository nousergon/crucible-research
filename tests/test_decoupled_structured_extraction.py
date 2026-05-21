"""Tests for the decoupled structured-output extraction in quant + qual
analysts (2026-05-02 refactor).

Background: the prior pattern used ``create_react_agent(response_format=
QuantAnalystOutput)`` which adds a post-loop extraction call inside the
LangGraph subgraph. That call is not constrained — Haiku occasionally
returns markdown-fenced JSON text instead of using the structured-output
tool, crashing with a ``ValidationError`` (input_value is the entire
string-with-fences assigned to a Pydantic field).

The 2026-05-02 refactor decouples the extraction: ReAct loop runs to a
free-text final answer, then a separate ``with_structured_output(
include_raw=True)`` call drives the parsing. The extraction call is
constrained at the API boundary (Anthropic tool-use) so the markdown-
fence failure mode is structurally impossible.

These tests lock the new contract:
  - Happy path: ReAct + extraction → typed picks
  - Empty final_text → loud error
  - Extraction parse error in strict-mode → loud error
  - Extraction parse error in lax-mode → empty fallback
  - PR #78's GraphRecursionError handling preserved (extraction not reached)
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from langgraph.errors import GraphRecursionError


@pytest.fixture
def fresh_modules():
    """Force-reload analyst modules to defeat MagicMock pollution from
    test_dry_run.py's sentinel pattern (cross-test order dependency)."""
    from agents.sector_teams import quant_analyst, qual_analyst, sector_team
    importlib.reload(quant_analyst)
    importlib.reload(qual_analyst)
    importlib.reload(sector_team)
    yield


# ── Test fixtures ─────────────────────────────────────────────────────────────


def _react_result(final_text: str) -> dict:
    """Build a minimal LangGraph ReAct result with a final AI message."""
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=final_text)]}


def _quant_kwargs():
    return {
        "team_id": "technology",
        "sector_tickers": ["AAPL", "MSFT"],
        "market_regime": "neutral",
        "price_data": {},
        "technical_scores": {},
        "run_date": "2026-05-02",
        "api_key": "test-key",
    }


def _qual_kwargs():
    return {
        "team_id": "technology",
        "quant_top5": [{"ticker": "AAPL"}],
        "prior_theses": {},
        "market_regime": "neutral",
        "run_date": "2026-05-02",
        "api_key": "test-key",
        "price_data": {},
    }


# ── Quant analyst: decoupled extraction happy path ────────────────────────────


def test_quant_extraction_happy_path(fresh_modules):
    """ReAct loop completes with a final-text answer; structured_llm.invoke
    returns parsed picks. Picks land in the result dict."""
    from agents.sector_teams import quant_analyst as _qa
    from graph.state_schemas import QuantAnalystOutput, QuantPick

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result(
        "Top picks: AAPL (75), MSFT (70). Both showing momentum."
    )

    parsed = QuantAnalystOutput(ranked_picks=[
        QuantPick(ticker="AAPL", quant_score=75.0, rationale="momentum"),
        QuantPick(ticker="MSFT", quant_score=70.0, rationale="momentum"),
    ])
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = {
        "raw": MagicMock(content="..."),
        "parsed": parsed,
        "parsing_error": None,
    }

    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm

    with patch.object(_qa, "create_react_agent", return_value=fake_agent), \
         patch.object(_qa, "ChatAnthropic", return_value=fake_llm):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is None
    assert len(result["ranked_picks"]) == 2
    assert {p["ticker"] for p in result["ranked_picks"]} == {"AAPL", "MSFT"}
    # Verify the extraction call happened with QuantAnalystOutput + include_raw
    fake_llm.with_structured_output.assert_called_once_with(
        QuantAnalystOutput, include_raw=True,
    )


def test_quant_extraction_empty_final_text_raises(fresh_modules):
    """ReAct loop ended with no AI message (or empty content) — extraction
    has nothing to work with. Loud error rather than silent empty picks."""
    from agents.sector_teams import quant_analyst as _qa

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result("")  # empty final
    fake_llm = MagicMock()

    with patch.object(_qa, "create_react_agent", return_value=fake_agent), \
         patch.object(_qa, "ChatAnthropic", return_value=fake_llm):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is not None
    assert "empty final_text" in result["error"]


def test_quant_extraction_parsing_error_strict_raises(fresh_modules, monkeypatch):
    """Strict-mode (default): parsing_error from structured-output extraction
    must surface as a hard error. Mirrors macro_agent.py's contract."""
    monkeypatch.setenv("STRICT_VALIDATION", "true")
    from agents.sector_teams import quant_analyst as _qa
    importlib.reload(_qa)  # re-eval is_strict_validation_enabled at import time

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result("Top picks: AAPL.")
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = {
        "raw": MagicMock(content="..."),
        "parsed": None,
        "parsing_error": ValueError("Pydantic validation failed: bad shape"),
    }
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm

    with patch.object(_qa, "create_react_agent", return_value=fake_agent), \
         patch.object(_qa, "ChatAnthropic", return_value=fake_llm):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is not None
    assert "parse failed" in result["error"]
    assert "ValueError" in result["error"]


def test_quant_extraction_parsing_error_lax_falls_back(fresh_modules, monkeypatch):
    """Lax-mode: parsing_error becomes a WARN log + empty picks. Same
    fallback semantics as macro_agent."""
    monkeypatch.setenv("STRICT_VALIDATION", "false")
    from agents.sector_teams import quant_analyst as _qa
    importlib.reload(_qa)

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result("Top picks: AAPL.")
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = {
        "raw": MagicMock(content="..."),
        "parsed": None,
        "parsing_error": ValueError("schema mismatch"),
    }
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm

    with patch.object(_qa, "create_react_agent", return_value=fake_agent), \
         patch.object(_qa, "ChatAnthropic", return_value=fake_llm):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is None
    assert result["ranked_picks"] == []
    assert result["partial"] is False  # lax-mode degrades silently to empty
    monkeypatch.setenv("STRICT_VALIDATION", "true")  # restore for other tests


def test_quant_recursion_error_unchanged_by_refactor(fresh_modules):
    """PR #78's GraphRecursionError → partial=True path is preserved.
    The ReAct loop raises before extraction is reached, so the new
    decoupled-extraction code doesn't touch this path."""
    from agents.sector_teams import quant_analyst as _qa

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = GraphRecursionError(
        "Recursion limit of 18 reached"
    )
    fake_llm = MagicMock()

    with patch.object(_qa, "create_react_agent", return_value=fake_agent), \
         patch.object(_qa, "ChatAnthropic", return_value=fake_llm):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is None
    assert result["partial"] is True
    assert result["partial_reason"] == "recursion_limit_exhausted"


# ── Qual analyst: same contract ───────────────────────────────────────────────


def test_qual_extraction_happy_path(fresh_modules, monkeypatch):
    from agents.sector_teams import qual_analyst as _qual
    from graph.state_schemas import QualAnalystOutput, QualAssessment

    # PILLAR_EMIT_ENABLED flipped to True at the Phase 4 cutover
    # (config #258, 2026-05-21). The single shared structured_llm mock
    # in this test returns a QualAnalystOutput shape — fine for the
    # legacy extraction (call #1), but the pillar extraction (call #2)
    # expects a _QualPillarBatch with an `items` attribute. This test
    # exercises the LEGACY extraction path only; disable pillar emit
    # locally so the second call doesn't fire. Pillar extraction has
    # its own dedicated test coverage in test_pillar_emit*.py.
    monkeypatch.setattr(_qual, "PILLAR_EMIT_ENABLED", False)

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result(
        "Assessment: AAPL strong fundamentals, $200 PT."
    )
    parsed = QualAnalystOutput(assessments=[
        QualAssessment(ticker="AAPL", qual_score=80.0,
                       bull_case="strong fundamentals"),
    ])
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = {
        "raw": MagicMock(content="..."),
        "parsed": parsed,
        "parsing_error": None,
    }
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm

    with patch.object(_qual, "create_react_agent", return_value=fake_agent), \
         patch.object(_qual, "ChatAnthropic", return_value=fake_llm):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is None
    assert len(result["assessments"]) == 1
    assert result["assessments"][0]["ticker"] == "AAPL"


def test_qual_extraction_empty_final_text_raises(fresh_modules):
    from agents.sector_teams import qual_analyst as _qual

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result("")
    fake_llm = MagicMock()

    with patch.object(_qual, "create_react_agent", return_value=fake_agent), \
         patch.object(_qual, "ChatAnthropic", return_value=fake_llm):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is not None
    assert "empty final_text" in result["error"]


def test_qual_extraction_parsing_error_strict_raises(fresh_modules, monkeypatch):
    monkeypatch.setenv("STRICT_VALIDATION", "true")
    from agents.sector_teams import qual_analyst as _qual
    importlib.reload(_qual)

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result("Some assessment text.")
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = {
        "raw": MagicMock(content="..."),
        "parsed": None,
        "parsing_error": ValueError("schema mismatch"),
    }
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm

    with patch.object(_qual, "create_react_agent", return_value=fake_agent), \
         patch.object(_qual, "ChatAnthropic", return_value=fake_llm):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is not None
    assert "parse failed" in result["error"]


def test_qual_extraction_parsing_error_lax_falls_back(fresh_modules, monkeypatch):
    monkeypatch.setenv("STRICT_VALIDATION", "false")
    from agents.sector_teams import qual_analyst as _qual
    importlib.reload(_qual)

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _react_result("Some assessment text.")
    fake_structured_llm = MagicMock()
    fake_structured_llm.invoke.return_value = {
        "raw": MagicMock(content="..."),
        "parsed": None,
        "parsing_error": ValueError("schema mismatch"),
    }
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured_llm

    with patch.object(_qual, "create_react_agent", return_value=fake_agent), \
         patch.object(_qual, "ChatAnthropic", return_value=fake_llm):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is None
    assert result["assessments"] == []
    monkeypatch.setenv("STRICT_VALIDATION", "true")


# ── Architectural invariant: response_format must NOT reappear ────────────────


def _strip_comments_and_strings(src: str) -> str:
    """Drop line comments and triple-quoted docstrings/strings so
    architectural-invariant checks aren't tripped by mention of a forbidden
    pattern in a comment explaining WHY it was removed."""
    import re
    # Drop triple-quoted strings (greedy across newlines)
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    # Drop line comments
    src = re.sub(r"(?m)^\s*#.*$", "", src)
    # Drop trailing line comments (preserve code before #)
    src = re.sub(r"(?m)\s*#[^\n]*$", "", src)
    return src


def test_quant_analyst_does_not_use_response_format():
    """Locks the architectural decision: structured extraction is decoupled
    from the ReAct loop. Reverting to ``create_react_agent(response_format=
    ...)`` would resurrect the markdown-fence ValidationError class that
    crashed today's SF replay."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "agents" / "sector_teams"
           / "quant_analyst.py").read_text()
    code = _strip_comments_and_strings(src)
    assert "response_format=" not in code, (
        "quant_analyst.py uses response_format= as a code argument — this "
        "resurrects the 2026-05-02 ValidationError class. Use the decoupled "
        "with_structured_output pattern (see macro_agent.py)."
    )
    # Defensive — the decoupled call site MUST be present.
    assert "with_structured_output(" in code


def test_qual_analyst_does_not_use_response_format():
    """Same architectural lock for qual_analyst."""
    from pathlib import Path
    src = (Path(__file__).parent.parent / "agents" / "sector_teams"
           / "qual_analyst.py").read_text()
    code = _strip_comments_and_strings(src)
    assert "response_format=" not in code
    assert "with_structured_output(" in code
