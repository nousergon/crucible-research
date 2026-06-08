"""Tests for the agent-decision review CLI (scripts/decision_review.py, L4567).

The CLI's query layer is pure: it takes a ``sqlite3.Connection`` and returns
plain dicts. These tests build an in-memory research.db via the real
``archive.schema.ensure_schema`` (so column drift is caught), seed the
decision tables directly, and assert the funnel logic of ``why-not`` at each
stage plus the ticker/date views.
"""

from __future__ import annotations

import sqlite3

import pytest

from archive.schema import ensure_schema
from scripts.decision_review import (
    _has_decision_tables,
    _parse_rule_tags,
    answer_question,
    build_qa_prompt,
    explain_why_not,
    gather_evidence,
    has_evidence,
    latest_eval_date,
    review_date,
    review_ticker,
)

DATE = "2026-05-16"


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    ensure_schema(c)
    yield c
    c.close()


# ── seed helpers ────────────────────────────────────────────────────────────


def _scanner(c, ticker, *, passed, reason=None, tech=50.0, sector="technology",
             liquidity=1, volatility=1, balance=1, date=DATE):
    c.execute(
        "INSERT INTO scanner_evaluations "
        "(ticker, eval_date, sector, tech_score, scan_path, quant_filter_pass, "
        " liquidity_pass, volatility_pass, balance_sheet_pass, filter_fail_reason) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (ticker, date, sector, tech, "momentum", 1 if passed else 0,
         liquidity, volatility, balance, reason),
    )
    c.commit()


def _team(c, ticker, *, team_id, rank, quant, qual, recommended, date=DATE):
    c.execute(
        "INSERT INTO team_candidates "
        "(ticker, eval_date, team_id, quant_rank, quant_score, qual_score, "
        " team_recommended, momentum_sub_score) VALUES (?,?,?,?,?,?,?,?)",
        (ticker, date, team_id, rank, quant, qual, 1 if recommended else 0, 60.0),
    )
    c.commit()


def _cio(c, ticker, *, decision, rank=None, conviction=None, final=None,
         rationale="", rule_tags=None, team_id="technology", date=DATE):
    c.execute(
        "INSERT INTO cio_evaluations "
        "(ticker, eval_date, team_id, final_score, cio_decision, cio_conviction, "
        " cio_rank, rationale, rule_tags) VALUES (?,?,?,?,?,?,?,?,?)",
        (ticker, date, team_id, final, decision, conviction, rank, rationale, rule_tags),
    )
    c.commit()


def _thesis(c, symbol, *, rating="BUY", score=70.0, summary="", date=DATE):
    c.execute(
        "INSERT INTO investment_thesis "
        "(symbol, date, run_time, rating, score, thesis_summary) "
        "VALUES (?,?,?,?,?,?)",
        (symbol, date, "2026-05-16T09:00:00Z", rating, score, summary),
    )
    c.commit()


# ── infra/helpers ───────────────────────────────────────────────────────────


def test_has_decision_tables(conn):
    assert _has_decision_tables(conn) is True


def test_has_decision_tables_false_on_bare_db():
    bare = sqlite3.connect(":memory:")
    try:
        assert _has_decision_tables(bare) is False
    finally:
        bare.close()


def test_parse_rule_tags():
    assert _parse_rule_tags(None) == []
    assert _parse_rule_tags("") == []
    assert _parse_rule_tags('["floor_fill", "conviction_bar"]') == [
        "floor_fill", "conviction_bar"]
    assert _parse_rule_tags(["already", "list"]) == ["already", "list"]
    assert _parse_rule_tags("not json") == []


def test_latest_eval_date(conn):
    assert latest_eval_date(conn) is None
    _scanner(conn, "AAA", passed=True, date="2026-05-09")
    _scanner(conn, "BBB", passed=True, date="2026-05-16")
    assert latest_eval_date(conn) == "2026-05-16"


# ── review_ticker ───────────────────────────────────────────────────────────


def test_review_ticker_full_record(conn):
    _scanner(conn, "NVDA", passed=True, tech=88.0)
    _team(conn, "NVDA", team_id="technology", rank=1, quant=85, qual=70, recommended=True)
    _cio(conn, "NVDA", decision="ADVANCE", rank=1, conviction=80, final=82.0,
         rationale="AI infra cycle", rule_tags='["rubric_advance"]')
    _thesis(conn, "NVDA", summary="Datacenter demand")

    r = review_ticker(conn, "nvda")  # lowercase → normalized
    assert r["ticker"] == "NVDA"
    assert r["eval_date"] == DATE
    assert r["scanner"]["quant_filter_pass"] == 1
    assert len(r["team_candidates"]) == 1
    assert r["team_candidates"][0]["team_recommended"] == 1
    assert r["cio"]["cio_decision"] == "ADVANCE"
    assert r["cio"]["rule_tags"] == ["rubric_advance"]  # parsed from JSON
    assert r["thesis"]["rating"] == "BUY"


def test_review_ticker_empty_sections(conn):
    _scanner(conn, "ZZZ", passed=True)  # only a scanner row
    r = review_ticker(conn, "ZZZ")
    assert r["scanner"] is not None
    assert r["team_candidates"] == []
    assert r["cio"] is None
    assert r["thesis"] is None


# ── explain_why_not — one assertion per funnel stage ────────────────────────


def test_why_not_no_record(conn):
    r = explain_why_not(conn, "GOOG")
    assert r["stage"] == "no_record"


def test_why_not_scanner_stage(conn):
    _scanner(conn, "PENNY", passed=False, reason="liquidity", liquidity=0, tech=20.0)
    r = explain_why_not(conn, "PENNY")
    assert r["stage"] == "scanner"
    assert "liquidity" in r["verdict"]
    assert r["detail"]["scanner"]["filter_fail_reason"] == "liquidity"


def test_why_not_team_stage_not_recommended(conn):
    _scanner(conn, "MSFT", passed=True)
    # MSFT ranked but not recommended; AAPL recommended for comparison context.
    _team(conn, "MSFT", team_id="technology", rank=11, quant=47, qual=40, recommended=False)
    _team(conn, "AAPL", team_id="technology", rank=1, quant=70, qual=65, recommended=True)
    r = explain_why_not(conn, "MSFT")
    assert r["stage"] == "team"
    assert "not recommended" in r["verdict"].lower()
    assert "quant_rank" in r["verdict"]
    assert r["detail"]["team_context"]["recommended_count"] == 1
    assert "AAPL" in r["detail"]["team_context"]["recommended_tickers"]


def test_why_not_passed_scanner_but_no_team_row(conn):
    _scanner(conn, "ORCL", passed=True)
    r = explain_why_not(conn, "ORCL")
    assert r["stage"] == "team"
    assert "did not appear" in r["verdict"]


def test_why_not_cio_reject(conn):
    _scanner(conn, "TSLA", passed=True)
    _team(conn, "TSLA", team_id="consumer", rank=2, quant=66, qual=60, recommended=True)
    _cio(conn, "TSLA", decision="REJECT", final=55.0, team_id="consumer",
         rationale="Valuation risk", rule_tags='["valuation"]')
    r = explain_why_not(conn, "TSLA")
    assert r["stage"] == "cio"
    assert "Valuation risk" in r["verdict"]


def test_why_not_chosen(conn):
    _scanner(conn, "NVDA", passed=True)
    _team(conn, "NVDA", team_id="technology", rank=1, quant=85, qual=70, recommended=True)
    _cio(conn, "NVDA", decision="ADVANCE", rank=1, conviction=80, final=82.0,
         rationale="AI infra")
    r = explain_why_not(conn, "NVDA")
    assert r["stage"] == "chosen"
    assert "WAS chosen" in r["verdict"]


def test_why_not_chosen_via_advance_forced(conn):
    """ADVANCE_FORCED (floor-fill) must count as chosen, not a rejection."""
    _scanner(conn, "WING", passed=True)
    _team(conn, "WING", team_id="consumer", rank=3, quant=58, qual=55, recommended=True)
    _cio(conn, "WING", decision="ADVANCE_FORCED", rank=3, conviction=62, final=60.0,
         rationale="min_new_entrants floor", team_id="consumer")
    r = explain_why_not(conn, "WING")
    assert r["stage"] == "chosen"


# ── review_date ─────────────────────────────────────────────────────────────


def test_review_date_funnel_counts(conn):
    _scanner(conn, "AAA", passed=True)
    _scanner(conn, "BBB", passed=False, reason="volatility", volatility=0)
    _scanner(conn, "CCC", passed=True)
    _team(conn, "AAA", team_id="technology", rank=1, quant=80, qual=70, recommended=True)
    _team(conn, "CCC", team_id="technology", rank=2, quant=60, qual=55, recommended=False)
    _cio(conn, "AAA", decision="ADVANCE", rank=1, conviction=75, final=78.0)

    s = review_date(conn)
    assert s["eval_date"] == DATE
    assert s["scanner_screened"] == 3
    assert s["scanner_passed"] == 2
    assert s["team_ranked"] == 2
    assert s["team_recommended"] == 1
    assert s["cio_evaluated"] == 1
    assert s["cio_advanced"] == 1
    assert s["advanced"][0]["ticker"] == "AAA"


# ── Phase 2: LLM-fallback Q&A (injected llm_fn — no real API calls) ─────────


def test_has_evidence(conn):
    ev_empty = gather_evidence(conn, "NONE")
    assert has_evidence(ev_empty) is False
    _scanner(conn, "MSFT", passed=True)
    assert has_evidence(gather_evidence(conn, "MSFT")) is True


def test_build_qa_prompt_grounds_in_evidence(conn):
    _scanner(conn, "MSFT", passed=True, tech=61.0)
    _team(conn, "MSFT", team_id="technology", rank=11, quant=47, qual=40, recommended=False)
    ev = gather_evidence(conn, "MSFT")
    system, user = build_qa_prompt("MSFT", DATE, "why not a higher rank?", ev)
    # System message carries the no-fabrication guardrail.
    assert "ONLY this evidence" in system
    assert "do NOT speculate" in system
    # User message embeds the recorded numbers + the question.
    assert "why not a higher rank?" in user
    assert "47" in user  # quant_score present in serialized evidence
    assert "RECORDED EVIDENCE" in user


def test_answer_question_skips_llm_when_no_evidence(conn):
    calls = []

    def fake_llm(system, user):
        calls.append((system, user))
        return "should not be called"

    result = answer_question(conn, "GHOST", "why not?", llm_fn=fake_llm)
    assert result["llm_called"] is False
    assert result["model"] is None
    assert calls == []  # no LLM call when nothing is recorded
    assert "No decision evidence" in result["answer"]


def test_answer_question_calls_llm_with_evidence(conn):
    _scanner(conn, "MSFT", passed=True)
    _team(conn, "MSFT", team_id="technology", rank=11, quant=47, qual=40, recommended=False)
    captured = {}

    def fake_llm(system, user):
        captured["system"] = system
        captured["user"] = user
        return "Ranked #11 (quant_score 47); the team recommended higher-scoring names."

    result = answer_question(
        conn, "msft", "why didn't the team pick it?", model="claude-test",
        llm_fn=fake_llm,
    )
    assert result["llm_called"] is True
    assert result["model"] == "claude-test"
    assert result["ticker"] == "MSFT"
    assert "Ranked #11" in result["answer"]
    assert "47" in captured["user"]  # evidence was passed to the model
    assert result["artifacts_used"] == []  # with_artifacts defaulted off
