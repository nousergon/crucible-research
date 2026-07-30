"""Producer-side contract test for ``signals.json::rag_scope`` (config-I5700).

Research is the PRODUCER of the RAG corpus's fetch set. Before this contract
existed, every ``rag/pipelines/*`` step in nousergon-data resolved its tickers
from ``signals.json["universe"]`` — the v1 *executor* compatibility key, built
from every ENTER/HOLD/EXIT signal. That is not a research scope, and reusing it
grew the corpus fetch set 27 -> 903 tickers (~3.1h of Polygon crawl at the
account-wide 5 req/min).

``rag-corpus-policy.md`` §2.1: the fetch set is the decision set — the scanner
focus list ∪ the held population ∪ open exits — never the widest list
available. This test pins that:

  * ``rag_scope`` is emitted and carries its declared shape;
  * it is built from ``agent_input_set`` ∪ population ∪ EXIT signals;
  * it is a SUBSET of nothing in particular but is NOT silently equal to the
    wider ``universe`` key — the two are deliberately different, and a future
    refactor that collapses them re-creates the 903-ticker regression.

The consumer-side guard (nousergon-data's ``_signals_universe``) fails loud
when this key is absent. That is the other half of the contract: a producer
that stops emitting it must break the consumer, never silently widen it back.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.research_graph import _build_signals_payload  # noqa: E402


def _synthetic_state() -> dict:
    """Minimal ResearchState driving one ENTER, one HOLD, one EXIT, plus an
    ``agent_input_set`` that is deliberately WIDER than the population (the
    scanner focus list contributes names that are not yet held) and a ticker
    (``DDD``) that appears ONLY in the focus list — so a builder that ignored
    ``agent_input_set`` would be caught."""
    return {
        "run_date": "2026-06-15",
        "run_time": "2026-06-13T09:00:00Z",
        "market_regime": "bull",
        "sector_modifiers": {"Technology": 1.1},
        "sector_ratings": {
            "Technology": {"rating": "overweight", "modifier": 1.1, "rationale": "x"},
        },
        "sector_map": {"AAA": "Technology", "BBB": "Technology"},
        "advanced_tickers": ["AAA"],
        "agent_input_set": ["AAA", "BBB", "DDD"],
        "new_population": [
            {"ticker": "AAA", "sector": "Technology", "price_target_upside": 0.18},
            {"ticker": "BBB", "sector": "Technology", "long_term_rating": "HOLD"},
        ],
        "investment_theses": {
            "AAA": {
                "rating": "BUY",
                "final_score": 82.0,
                "conviction": "rising",
                "bull_case": "thesis",
                "sector": "Technology",
                "team_id": "tech",
                "quant_score": 80.0,
                "qual_score": 84.0,
            },
        },
        "prior_theses": {},
        "entry_theses": {},
        "exits": [{"ticker_out": "CCC", "score_out": 30, "reason": "dropped"}],
    }


def test_rag_scope_is_emitted_with_its_declared_shape():
    scope = _build_signals_payload(_synthetic_state())["rag_scope"]
    assert set(scope) == {"tickers", "counts", "source"}
    assert set(scope["counts"]) == {"total", "agent_input_set", "population", "exits"}
    assert scope["counts"]["total"] == len(scope["tickers"])


def test_rag_scope_is_the_decision_set_not_the_executor_universe():
    payload = _build_signals_payload(_synthetic_state())
    scope = set(payload["rag_scope"]["tickers"])

    # agent_input_set contributes DDD, which is in NO signal (not held, not
    # advanced, not exiting) — so it is absent from `universe` by construction.
    # A builder that derived rag_scope from `universe` would miss it.
    assert "DDD" in scope, (
        "rag_scope must include scanner focus-list names that carry no signal "
        "yet — they are exactly what research is about to reason about"
    )
    assert "DDD" not in {e["ticker"] for e in payload["universe"]}

    # Exits still need evidence (a sell needs a rationale).
    assert "CCC" in scope, "a ticker with an open EXIT signal must stay in scope"

    # Held population is always in scope regardless of ranking.
    assert {"AAA", "BBB"} <= scope


def test_rag_scope_tickers_are_sorted_and_unique():
    # Consumers key watermarks off this list; an unstable order or a duplicate
    # turns a no-op week into spurious work.
    tickers = _build_signals_payload(_synthetic_state())["rag_scope"]["tickers"]
    assert tickers == sorted(set(tickers))


def test_rag_scope_survives_a_missing_agent_input_set():
    # Cold start / a state that never ran the resolver: the scope degrades to
    # population ∪ exits rather than raising. It must NOT silently widen to
    # `universe` — that is the 903-ticker regression this contract exists to
    # prevent, and the consumer's own guard is sized against this shape.
    state = _synthetic_state()
    del state["agent_input_set"]
    scope = _build_signals_payload(state)["rag_scope"]
    assert scope["counts"]["agent_input_set"] == 0
    assert "DDD" not in scope["tickers"]
    assert {"AAA", "BBB", "CCC"} <= set(scope["tickers"])
