"""Tests for the research producer substrate — the no-agent (pure-quant)
producer must emit a signals.json that satisfies the SAME producer contract as
the live agentic champion (config#1221 / M3)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the contract field sets the champion is pinned to — the whole point is
# that a challenger producer is contract-identical, only its BELIEF differs.
from tests.test_signals_producer_contract import (  # noqa: E402
    _REQUIRED_PER_ITEM,
    _REQUIRED_TOP_LEVEL,
)
from producers.no_agent import build_no_agent_signals  # noqa: E402
from producers.registry import (  # noqa: E402
    RESEARCH_PRODUCERS,
    champion_producer,
    challenger_producers,
)


def _inputs():
    scanner_tickers = ["AAA", "BBB", "CCC", "DDD"]
    technical_scores = {
        "AAA": {"technical_score": 85.0, "momentum_20d": 0.05},   # BUY, rising, new → ENTER
        "BBB": {"technical_score": 70.0, "momentum_20d": -0.05},  # BUY, declining, new → ENTER
        "CCC": {"technical_score": 40.0, "momentum_20d": 0.0},    # < threshold → HOLD, dropped
        "DDD": {"technical_score": 90.0, "momentum_20d": 0.01},   # BUY, HELD → reaffirmed ENTER
    }
    population = [
        {
            "ticker": "DDD", "sector": "Technology", "long_term_rating": "BUY",
            "long_term_score": 88.0, "conviction": "stable", "price_target_upside": 0.1,
        }
    ]
    sector_map = {"AAA": "Technology", "BBB": "Healthcare", "CCC": "Technology", "DDD": "Technology"}
    return scanner_tickers, technical_scores, population, sector_map


def _build():
    scanner_tickers, technical_scores, population, sector_map = _inputs()
    return build_no_agent_signals(
        "2026-06-19",
        scanner_tickers=scanner_tickers,
        population=population,
        prior_theses={},
        technical_scores=technical_scores,
        sector_map=sector_map,
    )


def test_no_agent_payload_satisfies_producer_contract():
    payload = _build()
    assert _REQUIRED_TOP_LEVEL <= set(payload), (
        "missing top-level contract keys: " + str(_REQUIRED_TOP_LEVEL - set(payload))
    )
    for section in ("universe", "buy_candidates"):
        for item in payload[section]:
            assert _REQUIRED_PER_ITEM <= set(item), (
                f"{section} item missing fields: {_REQUIRED_PER_ITEM - set(item)} ({item.get('ticker')})"
            )


def test_enter_signals_carry_numeric_score():
    payload = _build()
    enters = [u for u in payload["universe"] if u["signal"] == "ENTER"]
    assert enters, "expected at least one ENTER"
    for e in enters:
        assert isinstance(e["score"], (int, float)) and e["score"] is not None, e


def test_no_agent_gate_and_provenance():
    payload = _build()
    sig = payload["signals"]
    # AAA/BBB: new BUY above threshold → ENTER as cio_entrant (the quant gate).
    assert sig["AAA"]["signal"] == "ENTER" and sig["AAA"]["stance_source"] == "cio_entrant"
    assert sig["BBB"]["signal"] == "ENTER"
    # DDD: held BUY → reaffirmed ENTER.
    assert sig["DDD"]["signal"] == "ENTER" and sig["DDD"]["stance_source"] == "reaffirmed_hold"
    # CCC: below the BUY threshold and not held → dropped entirely.
    assert "CCC" not in sig


def test_no_agent_emits_no_qualitative_score():
    payload = _build()
    # No LLM → every qual sub-score is None (the defining property of the floor).
    for u in payload["universe"]:
        assert (u.get("sub_scores") or {}).get("qual") is None, u


def test_threshold_controls_entry():
    scanner_tickers, technical_scores, population, sector_map = _inputs()
    # Raise the bar above AAA/BBB/DDD's scores → no new entrants, DDD still held-HOLD?
    payload = build_no_agent_signals(
        "2026-06-19", scanner_tickers=scanner_tickers, population=population,
        prior_theses={}, technical_scores=technical_scores, sector_map=sector_map,
        buy_score_threshold=95.0,
    )
    # No candidate clears 95 → no fresh BUY theses → AAA/BBB dropped; DDD carries
    # over from population (prior BUY + score) as a reaffirmed/held name.
    assert "AAA" not in payload["signals"] and "BBB" not in payload["signals"]
    assert "DDD" in payload["signals"]


def test_registry_invariants():
    # config-I2993: agentic_sector_teams is RETIRED — no producer is currently
    # registered kind=="champion" (retiring a spec does not auto-promote a
    # successor; that registration is tracked separately). champion_producer()
    # must return None rather than raising, and no OTHER spec may silently
    # carry kind=="champion" only because it happens to be first in dict order.
    champs = [p for p in RESEARCH_PRODUCERS.values() if p.kind == "champion"]
    assert champs == []
    assert champion_producer() is None
    chals = challenger_producers()
    assert chals and all(p.build is not None for p in chals)
    assert "no_agent_quant" in {p.name for p in chals}


def test_agentic_sector_teams_is_retired_with_date():
    # Root-cause fix for config-I2993: the retired six-team+CIO producer must
    # be a queryable fact (kind + retired_date), not a stale "champion" label
    # a downstream reader (evaluator/backtester e2e_lift) has to re-derive.
    spec = RESEARCH_PRODUCERS["agentic_sector_teams"]
    assert spec.kind == "retired"
    assert spec.retired_date == "2026-07-12"
    assert spec.build is None


def test_only_agentic_sector_teams_is_retired():
    # Every OTHER registered producer must be untouched by the additive
    # retired_date field (defaults to None) and keep its pre-existing kind.
    retired = {name: p for name, p in RESEARCH_PRODUCERS.items() if p.kind == "retired"}
    assert set(retired) == {"agentic_sector_teams"}
    for name, p in RESEARCH_PRODUCERS.items():
        if name == "agentic_sector_teams":
            continue
        assert p.retired_date is None
