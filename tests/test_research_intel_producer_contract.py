"""Producer-side contract test for the research → research_intel.json boundary.

config#1500 (Phase 0 of EPIC config#1499). Research is the PRODUCER of
``research_intel/{date}.json`` (+ ``research_intel/latest.json``), a NEW
neutral, product-facing sibling to ``signals.json``. Consumers (Metron Advisor
and product surfaces, landing in Phase 1) read this artifact for neutral intel
WITHOUT touching the edge-mixed signals.json — so a producer that silently
stops emitting a contract field, or that LEAKS an edge field (tuned pillar/blend
weights), is exactly the class this test kills.

This test:
  1. Pins the neutral allowlist the producer MUST keep emitting (envelope +
     per-ticker attractiveness fields), mirroring
     ``test_signals_producer_contract.py``.
  2. Asserts the payload VALIDATES against the ``research_intel`` schema hosted
     in ``nousergon_lib.contracts`` — the single cross-repo source of truth
     (same conformance-kit primitive the signals + predictions contracts use).
  3. Asserts the edge stays PRIVATE: the published attractiveness breakdown
     carries component VALUES only, never the tuned blend/pillar weights.

Contract source of truth: ``nousergon_lib.contracts`` schema ``research_intel``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.research_graph import _build_research_intel_payload  # noqa: E402


# ── Contract field sets (MUST match research_intel.schema.json) ────────────────

_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "date",
    "generated_at",
    "market_regime",
    "sector_ratings",
    "sector_modifiers",
    "market_breadth",
    "attractiveness",
}

_REQUIRED_ATTRACTIVENESS_ITEM = {"ticker", "score"}

# Edge fields that MUST NOT appear in the published breakdown — the tuned
# pillar/blend weights are the private edge (config#1499). If a future refactor
# widened the breakdown to spread the whole composite_breakdown, these would
# leak and this test fails LOUDLY.
_FORBIDDEN_BREAKDOWN_KEYS = {
    "w_legacy_quant",
    "w_legacy_qual",
    "w_factor",
    "pillar_weight",
    "pillar_weights",
    "pillar_contributions",
    "legacy_blend",
    "contribution",
}


def _synthetic_state() -> dict:
    """A minimal ResearchState carrying exactly the already-computed node
    outputs ``_build_research_intel_payload`` reads: macro (regime + narrative
    + sector ratings/modifiers), fetch_data breadth (in macro_data), and
    score_aggregator's investment_theses (attractiveness + thesis)."""
    return {
        "run_date": "2026-06-15",
        "run_time": "2026-06-13T09:00:00Z",
        "market_regime": "bull",
        "macro_report": "Constructive macro backdrop; breadth broadening.",
        "sector_modifiers": {"Technology": 1.1, "Energy": 0.85},
        "sector_ratings": {
            "Technology": {
                "rating": "overweight",
                "rationale": "AI capex tailwind",
                # an extra key that must be DROPPED by the neutral carve:
                "modifier": 1.1,
            },
            "Energy": {"rating": "underweight", "rationale": "demand softening"},
        },
        "macro_data": {
            "pct_above_50d_ma": 58.0,
            "pct_above_200d_ma": 62.0,
            "advance_decline_ratio": 1.4,
            # non-allowlisted macro inputs that MUST NOT leak into breadth:
            "vix": 14.2,
            "treasury_10yr": 4.3,
        },
        "investment_theses": {
            "AAA": {
                "ticker": "AAA",
                "sector": "Technology",
                "final_score": 82.0,
                "quant_score": 80.0,
                "qual_score": 84.0,
                "weighted_base": 80.0,
                "macro_shift": 1.0,
                "bull_case": "Durable moat + secular demand.",
                # bear_case / rating present in state but MUST NOT be published:
                "bear_case": "Valuation rich.",
                "rating": "BUY",
                "composite_breakdown": {
                    "final_score": 82.0,
                    "weighted_base": 80.0,
                    "macro_shift": 1.0,
                    "legacy_blend": {
                        "quant_score": 80.0,
                        "qual_score": 84.0,
                        "factor_subscore": None,
                        "w_legacy_quant": 0.35,
                        "w_legacy_qual": 0.35,
                        "w_factor": 0.30,
                        "contribution": 80.0,
                    },
                    "pillar_contributions": [],
                    "score_failed": False,
                },
            },
            # A held-stock recompute path thesis with no composite_breakdown —
            # exercises the flat-field fallback branch.
            "BBB": {
                "ticker": "BBB",
                "sector": "Energy",
                "final_score": 41.0,
                "quant_score": 40.0,
                "qual_score": 42.0,
                "weighted_base": 41.0,
                "macro_shift": -0.5,
                "bull_case": "Cost discipline.",
            },
        },
    }


def _payload() -> dict:
    return _build_research_intel_payload(_synthetic_state())


def test_envelope_carries_every_required_top_level_field():
    payload = _payload()
    missing = _REQUIRED_TOP_LEVEL - payload.keys()
    assert not missing, (
        f"research_intel producer dropped required top-level field(s): "
        f"{sorted(missing)}. Consumers read this neutral artifact; a drop is a "
        "silent structural break. Update research_intel.schema.json deliberately "
        "if intended."
    )


def test_attractiveness_entries_carry_required_fields():
    payload = _payload()
    assert payload["attractiveness"], "synthetic state should yield attractiveness"
    for ticker, entry in payload["attractiveness"].items():
        missing = _REQUIRED_ATTRACTIVENESS_ITEM - entry.keys()
        assert not missing, (
            f"attractiveness[{ticker!r}] missing contract field(s): {sorted(missing)}"
        )


def test_market_breadth_is_the_strict_allowlist():
    # breadth is lifted from macro_data — must carry ONLY the three allowlisted
    # keys, never vix/treasury/etc. that also live in macro_data.
    payload = _payload()
    assert set(payload["market_breadth"]) == {
        "pct_above_50d_ma",
        "pct_above_200d_ma",
        "advance_decline_ratio",
    }


def test_sector_ratings_carry_only_rating_and_rationale():
    # The neutral carve drops any extra keys (e.g. a leaked 'modifier').
    payload = _payload()
    for sector, entry in payload["sector_ratings"].items():
        assert set(entry) <= {"rating", "rationale"}, (
            f"sector_ratings[{sector!r}] leaked non-neutral key(s): "
            f"{set(entry) - {'rating', 'rationale'}}"
        )


def test_edge_weights_never_leak_into_breakdown():
    # The published breakdown must carry component VALUES only — never the
    # tuned pillar/blend weights (config#1499 edge boundary).
    payload = _payload()
    for ticker, entry in payload["attractiveness"].items():
        breakdown = entry.get("breakdown", {})
        leaked = _FORBIDDEN_BREAKDOWN_KEYS & set(breakdown)
        assert not leaked, (
            f"attractiveness[{ticker!r}].breakdown LEAKED edge field(s): "
            f"{sorted(leaked)} — tuned weights must stay private."
        )


def test_thesis_carries_no_position_judgment():
    # thesis is the generic sector-team narrative — NOT the ENTER/EXIT/rating.
    payload = _payload()
    for ticker, entry in payload["attractiveness"].items():
        thesis = entry.get("thesis", {})
        assert set(thesis) <= {"bull_case", "sector"}, (
            f"attractiveness[{ticker!r}].thesis leaked a non-generic key: "
            f"{set(thesis) - {'bull_case', 'sector'}}"
        )
        assert "rating" not in thesis and "signal" not in thesis
        assert "bear_case" not in thesis


def test_payload_validates_against_lib_contract():
    # The cross-repo source of truth: nousergon_lib.contracts research_intel
    # schema. Same conformance-kit primitive as signals + predictions.
    from nousergon_lib import contracts

    errors = contracts.conformance_errors("research_intel", _payload())
    assert errors == [], (
        "research_intel payload violates the nousergon_lib contract:\n  "
        + "\n  ".join(errors)
    )


def test_held_stock_fallback_breakdown_populated():
    # BBB has no composite_breakdown — the flat-field fallback must still
    # populate the neutral breakdown values.
    payload = _payload()
    bbb = payload["attractiveness"]["BBB"]["breakdown"]
    assert bbb["quant_score"] == 40.0
    assert bbb["weighted_base"] == 41.0
    assert bbb["macro_shift"] == -0.5
