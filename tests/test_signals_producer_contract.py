"""Producer-side contract test for the research → signals.json boundary (L4520).

Research is the PRODUCER of ``signals/{trading_day}/signals.json``, consumed
cross-repo by the executor (``alpha-engine/executor/signal_reader.py``) and the
predictor (``alpha-engine-predictor/inference/stages/load_universe.py``). Those
consumers read the payload FAIL-SOFT (``.get(...)`` defaults), so a producer
that silently stops emitting a contract field would trade on a default with no
error — the exact silent-structural-break class L4520 exists to kill (cf. the
``avg_volume_20d`` 901/903 silent-zero pinned by
``test_scanner_consumer_contract.py``).

This test pins the field set the producer MUST keep emitting. It builds a real
payload via the pure ``_build_signals_payload`` and fails LOUDLY if a future PR
drops any contract field from the envelope or from a per-ticker ``universe`` /
``buy_candidates`` entry.

Contract source of truth: ``alpha-engine-config/private-docs/PIPELINE_CONTRACT.yaml``
boundary ``signals`` (kept in sync by convention — the drift-proof lib-hosted
form is a filed follow-up).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.research_graph import _build_signals_payload  # noqa: E402

# ── Contract field sets (MUST match PIPELINE_CONTRACT.yaml boundary `signals`) ──

# Top-level envelope keys the consumers rely on.
_REQUIRED_TOP_LEVEL = {
    "date",
    "market_regime",      # executor regime-aware sizing (get_actionable_signals)
    "sector_ratings",     # executor sector modifiers
    "sector_modifiers",   # predictor macro features
    "signals",            # predictor universe extraction
    "universe",           # executor read_signals
    "buy_candidates",     # executor + predictor
    "population",
}

# Per-ticker fields on every `universe` / `buy_candidates` entry (executor reads
# ticker + signal as load-bearing; the rest size/annotate positions).
_REQUIRED_PER_ITEM = {
    "ticker",
    "signal",
    "score",
    "rating",
    "conviction",
    "sector",
    "sector_rating",
    "price_target_upside",
}


def _synthetic_state() -> dict:
    """A minimal ResearchState that drives one ENTER, one HOLD, one EXIT through
    the pure builder — enough to exercise every per-ticker construction branch."""
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
        "new_population": [
            {"ticker": "AAA", "sector": "Technology", "price_target_upside": 0.18},
            {"ticker": "BBB", "sector": "Technology", "long_term_rating": "HOLD"},
        ],
        "investment_theses": {
            "AAA": {
                "rating": "BUY", "final_score": 82.0, "conviction": "rising",
                "bull_case": "thesis", "sector": "Technology", "team_id": "tech",
                "quant_score": 80.0, "qual_score": 84.0,
            },
        },
        "prior_theses": {},
        "entry_theses": {},
        "exits": [{"ticker_out": "CCC", "score_out": 30, "reason": "dropped"}],
    }


def test_envelope_carries_every_required_top_level_field():
    payload = _build_signals_payload(_synthetic_state())
    missing = _REQUIRED_TOP_LEVEL - payload.keys()
    assert not missing, (
        f"signals.json producer dropped required top-level field(s): {sorted(missing)}. "
        "Consumers (executor/predictor) read these fail-soft, so a drop is a SILENT "
        "structural break. Update PIPELINE_CONTRACT.yaml deliberately if intended."
    )


def test_universe_entries_carry_every_required_per_item_field():
    payload = _build_signals_payload(_synthetic_state())
    assert payload["universe"], "synthetic state should yield a non-empty universe"
    for entry in payload["universe"]:
        missing = _REQUIRED_PER_ITEM - entry.keys()
        assert not missing, (
            f"universe entry {entry.get('ticker')!r} missing contract field(s): "
            f"{sorted(missing)}"
        )


def test_buy_candidates_entries_carry_every_required_per_item_field():
    payload = _build_signals_payload(_synthetic_state())
    assert payload["buy_candidates"], "AAA (BUY+advanced) should be a buy_candidate"
    for entry in payload["buy_candidates"]:
        missing = _REQUIRED_PER_ITEM - entry.keys()
        assert not missing, (
            f"buy_candidate {entry.get('ticker')!r} missing contract field(s): "
            f"{sorted(missing)}"
        )


def test_enter_signal_carries_a_numeric_score():
    # An ENTER with score=None is the 2026-04-04 broken-thesis bug; the contract
    # requires ENTER entries to carry a numeric score the executor can size on.
    payload = _build_signals_payload(_synthetic_state())
    enters = [e for e in payload["universe"] if e["signal"] == "ENTER"]
    assert enters, "synthetic AAA should produce an ENTER"
    for e in enters:
        assert e["score"] is not None, f"ENTER {e['ticker']} must carry a numeric score"


def test_universe_entries_carry_stance_source_provenance():
    # config#859: every universe entry must carry a non-null stance_source so the
    # evaluator's stance_source_provenance grader can score pick-provenance
    # coverage (was N/A-MISSING-INPUT because the field did not exist).
    payload = _build_signals_payload(_synthetic_state())
    by_ticker = {e["ticker"]: e for e in payload["universe"]}
    # AAA: BUY + CIO-advanced fresh thesis; BBB: population carryover (no fresh
    # thesis); CCC: dropped from population.
    assert by_ticker["AAA"]["stance_source"] == "cio_entrant"
    assert by_ticker["BBB"]["stance_source"] == "carryover"
    assert by_ticker["CCC"]["stance_source"] == "exit"
    for e in payload["universe"]:
        assert e.get("stance_source"), (
            f"universe entry {e.get('ticker')!r} has empty stance_source — "
            "the provenance grader would under-count coverage"
        )
