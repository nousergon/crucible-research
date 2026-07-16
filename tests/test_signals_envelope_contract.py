"""Producer-side contract test for ``scoring/signals_envelope.py`` (config#2515).

This is the SECOND implementation of the Slot R ``signals`` contract (the
first being the multi-agent ``graph.research_graph._build_signals_payload``,
pinned by ``test_signals_producer_contract.py``). M0 discipline treats slots
as product contracts that may have more than one implementation; this test
pins the SAME consumer-required field set for this producer and additionally
validates every built envelope against the existing versioned JSON Schema in
``nousergon_lib.contracts`` (``SLOT_SCHEMAS["signals"]``) — the schema is not
re-invented here, it is the shared source of truth both producers target.

Consumed-field inventory (traced 2026-07-14, field -> consumer file:line):

Top-level envelope fields:
  date              -> alpha-engine/executor/signal_reader.py (S3 key date;
                       _warn_if_stale); alpha-engine-predictor/inference/
                       stages/load_universe.py (fallback-chain freshness)
  market_regime     -> alpha-engine/executor/signal_reader.py:763
                       (get_actionable_signals); alpha-engine/executor/
                       main.py:1171,1257-1344 (bear-override gating,
                       _macro_rank); alpha-engine/executor/risk_guard.py:443,
                       457 (bear_block_underweight); alpha-engine-research/
                       thinktank/context.py:98-99 (market_regime());
                       alpha-engine-predictor/inference/stages/
                       load_universe.py:27-29 (_CANONICAL_SIGNALS_MACRO_FIELDS)
  sector_ratings    -> alpha-engine/executor/signal_reader.py:764
                       (get_actionable_signals); alpha-engine/executor/
                       deciders.py:568 (sector_info.get("rating")); alpha-
                       engine-research/thinktank/context.py:95-96
                       (sector_ratings()); predictor load_universe.py
                       (_CANONICAL_SIGNALS_MACRO_FIELDS)
  sector_modifiers  -> alpha-engine-predictor/model/research_features.py:105
                       (sector_macro_modifier feature, .get(sector, 1.0))
  universe[]        -> alpha-engine/executor/signal_reader.py:337-353
                       (read_signals); alpha-engine-data/collectors/
                       alternative.py:816-855 (_load_promoted_tickers);
                       alpha-engine-predictor/inference/stages/
                       load_universe.py:367-368 (load_watchlist)
  buy_candidates    -> alpha-engine/executor/signal_reader.py (filter_*
                       functions); alpha-engine/executor/champion.py:280-288
                       (count-fallback trigger); alpha-engine-data/
                       collectors/alternative.py:816-855
  population        -> alpha-engine-predictor/inference/stages/
                       load_universe.py:358-364 (load_watchlist local-file
                       fallback path only)
  signals           -> alpha-engine-predictor/inference/stages/
                       load_universe.py:180 (get_universe_tickers)

Per-ticker fields (universe / buy_candidates entries):
  ticker               -> everywhere above; alpha-engine-data alternative.py
  signal               -> alpha-engine/executor/signal_reader.py
                          get_actionable_signals (enter/exit/reduce/hold
                          buckets); exit_manager.py:826 research_action
                          derivation (`.get("signal", "HOLD")`)
  score                -> alpha-engine/executor/risk_guard.py:393-406
                          (min_score_to_enter, ENTER only); main.py:1650-1657
                          (_conviction_rank drawdown-forced-exit ordering,
                          `.get("score") or 50`)
  rating               -> alpha-engine/executor/decision_capture.py:342,559
                          (research_rating, narrative/logging only)
  conviction           -> alpha-engine/executor/position_sizer.py:110-112
                          (conviction_adj: "declining" -> 0.70x)
  sector               -> alpha-engine/executor/signal_reader.py:680-730
                          (patch_unknown_sectors_with_constituents)
  sector_rating        -> alpha-engine/executor/risk_guard.py:458 (bear-
                          regime underweight block); decision_capture.py:536;
                          position_sizer.py:63,108 (sector_adj_map lookup)
  price_target_upside  -> alpha-engine/executor/position_sizer.py:114-117
                          (upside_adj: below floor -> config multiplier)
  thesis_summary       -> alpha-engine/executor/decision_capture.py,
                          order_book_rationale.py (narrative/display only)
  stance_source        -> evaluator stance_source_provenance grader
                          (config#859); non-empty required
  quant_score/qual_score/sub_scores/factor_quality_score
                       -> evaluator grading tiles (research metric tile);
                          non-gating

This test file pins the frozen field sets and asserts every consumer field
survives a real built envelope.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nousergon_lib.contracts import ContractViolation, conformance_errors, validate

from scoring.signals_envelope import (
    build_sector_modifiers,
    build_sector_ratings,
    build_signals_envelope,
    build_universe_entries,
    derive_market_regime,
)

# ── Contract field sets (must match the multi-agent producer's own contract,
#    test_signals_producer_contract.py, so BOTH implementations of Slot R
#    satisfy the same consumer set) ───────────────────────────────────────────

_REQUIRED_TOP_LEVEL = {
    "date",
    "market_regime",
    "sector_ratings",
    "sector_modifiers",
    "signals",
    "universe",
    "buy_candidates",
    "population",
}

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


def _sample_board(n: int = 3) -> dict:
    tickers = ["AAA", "BBB", "CCC"][:n]
    sectors = ["Technology", "Technology", "Healthcare"][:n]
    scores = [82.5, 61.0, None]
    return {
        "schema_version": 3,
        "as_of": "2026-07-14",
        "universe_count": n,
        "stocks": [
            {
                "ticker": t,
                "sector": sectors[i],
                "attractiveness_score": scores[i],
                "attractiveness_raw": 0.5 if scores[i] is not None else None,
                "pillars": {"quality": 70.0 + i, "value": 55.0, "momentum": 60.0,
                            "growth": 65.0, "stewardship": 50.0, "defensiveness": 45.0},
                "tradeability": {"expected_cost_bps": 12.0, "tradeability_score": 80.0,
                                  "adv_usd": 5_000_000.0, "reference_notional_usd": 100_000.0},
                "gate": {"quant_filter_pass": 1, "filter_fail_reason": None},
                "gate_stage": "passed",
                "metrics": {"current_price": 100.0 + i},
            }
            for i, t in enumerate(tickers)
        ],
    }


def _sample_substrate(intensity_z: float | None) -> dict | None:
    if intensity_z is None:
        return None
    return {"composite": {"intensity_z": intensity_z}, "run_id": "20260713-0900"}


# ── Builder unit tests ───────────────────────────────────────────────────────


def test_market_regime_thresholds():
    assert derive_market_regime(_sample_substrate(0.6)) == "bull"
    assert derive_market_regime(_sample_substrate(0.5)) == "bull"
    assert derive_market_regime(_sample_substrate(-0.5)) == "bear"
    assert derive_market_regime(_sample_substrate(-0.6)) == "bear"
    assert derive_market_regime(_sample_substrate(0.0)) == "neutral"
    assert derive_market_regime(_sample_substrate(0.49)) == "neutral"


def test_market_regime_defaults_neutral_when_substrate_missing(caplog):
    assert derive_market_regime(None) == "neutral"
    assert derive_market_regime({"composite": {}}) == "neutral"
    assert derive_market_regime({"composite": {"intensity_z": "not-a-number"}}) == "neutral"


def test_market_regime_never_emits_risk_on_or_risk_off():
    # Corrected verification (2026-07-14): the executor's real 3-class
    # taxonomy is bull/neutral/bear (main.py's _macro_rank, risk_guard.py's
    # market_regime == "bear" gates); "risk_on"/"risk_off" appears nowhere
    # in the executor or research codebases. This producer must never emit
    # that vocabulary — a bear-market string-equality gate would silently
    # never fire against it.
    for z in (-5.0, -0.5, -0.1, 0.0, 0.1, 0.5, 5.0, None):
        regime = derive_market_regime(_sample_substrate(z) if z is not None else None)
        assert regime in ("bull", "neutral", "bear")


def test_sector_ratings_neutral_for_every_board_sector():
    board = _sample_board()
    sectors = sorted({s["sector"] for s in board["stocks"]})
    ratings = build_sector_ratings(sectors)
    assert set(ratings) == set(sectors)
    for row in ratings.values():
        assert row["rating"] == "market_weight"
        assert row["modifier"] == 1.0


def test_sector_modifiers_neutral_for_every_board_sector():
    board = _sample_board()
    sectors = sorted({s["sector"] for s in board["stocks"]})
    modifiers = build_sector_modifiers(sectors)
    assert modifiers == {s: 1.0 for s in sectors}


def test_universe_entries_are_neutral_and_carry_real_score():
    board = _sample_board()
    entries = build_universe_entries(board["stocks"])
    assert len(entries) == 3
    by_ticker = {e["ticker"]: e for e in entries}
    assert by_ticker["AAA"]["score"] == 82.5  # real quant fact, not fabricated
    assert by_ticker["CCC"]["score"] is None  # honest coverage gap preserved
    for e in entries:
        assert e["signal"] == "HOLD"
        assert e["rating"] == "HOLD"
        assert e["conviction"] == "stable"
        assert e["sector_rating"] == "market_weight"
        assert e["price_target_upside"] is None
        assert e["stance_source"] == "quant_envelope_producer"


# ── Full envelope: field completeness + schema contract ─────────────────────


def _built_envelope() -> dict:
    return build_signals_envelope("2026-07-14", _sample_board(), _sample_substrate(0.2))


def test_envelope_carries_every_required_top_level_field():
    payload = _built_envelope()
    missing = _REQUIRED_TOP_LEVEL - payload.keys()
    assert not missing, f"signals_envelope dropped required top-level field(s): {sorted(missing)}"


def test_universe_entries_carry_every_required_per_item_field():
    payload = _built_envelope()
    assert payload["universe"], "sample board should yield a non-empty universe"
    for entry in payload["universe"]:
        missing = _REQUIRED_PER_ITEM - entry.keys()
        assert not missing, f"universe entry {entry.get('ticker')!r} missing: {sorted(missing)}"


def test_buy_candidates_always_empty():
    payload = _built_envelope()
    assert payload["buy_candidates"] == []


def test_envelope_stamps_producer_and_schema_version():
    payload = _built_envelope()
    assert payload["producer"] == "signals_envelope"
    assert payload["schema_version"] == 1


def test_envelope_conforms_to_existing_signals_contract_schema():
    # The load-bearing M0 assertion: this producer's output must validate
    # clean against the SAME nousergon_lib.contracts signals v1 schema the
    # multi-agent producer targets — no new/parallel schema was invented.
    errors = conformance_errors("signals", _built_envelope())
    assert errors == [], f"signals_envelope output violates the signals v1 contract: {errors}"


def test_envelope_raises_contract_violation_on_broken_shape(monkeypatch):
    # Sanity-check that validate() is actually load-bearing inside the
    # builder — sabotage a required per-item field and confirm it raises
    # loud rather than silently emitting a non-conformant envelope.
    import scoring.signals_envelope as mod

    original_build_universe_entries = mod.build_universe_entries

    def _broken_entries(stocks):
        entries = [dict(e) for e in original_build_universe_entries(stocks)]
        for e in entries:
            e.pop("sector_rating", None)
        return entries

    monkeypatch.setattr(mod, "build_universe_entries", _broken_entries)
    with pytest.raises(ContractViolation):
        mod.build_signals_envelope("2026-07-14", _sample_board(), None)


def test_board_with_empty_stocks_raises():
    empty_board = {"schema_version": 3, "stocks": []}
    with pytest.raises(ValueError):
        build_signals_envelope("2026-07-14", empty_board, None)
