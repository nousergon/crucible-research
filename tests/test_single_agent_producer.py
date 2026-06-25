"""Tests for the single-agent research producer (config#1223 / M3). The single
agent provides qualitative scores via ONE LLM call; the payload must satisfy the
SAME producer contract as the champion. LLM is never hit — the builder takes
assessments directly, and the run test injects assess_fn."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from tests.test_signals_producer_contract import (  # noqa: E402
    _REQUIRED_PER_ITEM,
    _REQUIRED_TOP_LEVEL,
)
from producers.single_agent import (  # noqa: E402
    RankingProducerOutput,
    build_single_agent_signals,
    run_single_agent_producer,
)
from producers.registry import RESEARCH_PRODUCERS  # noqa: E402


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
    with pytest.raises(Exception):
        RankingProducerOutput(assessments=[])  # min_length=1


def test_run_injects_assess_fn(monkeypatch):
    from unittest.mock import MagicMock
    import data.fetchers.price_fetcher as pf
    import data.scanner_orchestrator as so

    am = MagicMock()
    am.load_candidates_json.return_value = {"scanner_tickers": ["AAA", "BBB"]}
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
