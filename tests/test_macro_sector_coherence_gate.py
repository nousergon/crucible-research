"""Tests for the macro-sector coherence gate (PR feat/macro-sector-coherence-gate, 2026-05-13).

Pins behavior: NEW buy_candidates from UNDERWEIGHT sectors are blocked when
composite score is below SECTOR_COHERENCE_UW_MIN_SCORE. Existing HOLDs and
EXITs are unaffected — gate only filters ENTER signals at buy_candidates
construction.

Wired into ``_build_signals_payload`` in graph/research_graph.py. Threshold
lives in alpha-engine-config/research/scoring.yaml under
``aggregator.macro_sector_coherence_gate``.
"""

from graph.research_graph import _build_signals_payload


def _make_thesis(ticker, rating, score, sector="Technology"):
    return {
        "ticker": ticker,
        "rating": rating,
        "final_score": score,
        "quant_score": score,
        "qual_score": score,
        "sector": sector,
        "team_id": sector.lower(),
        "conviction": "stable",
        "bull_case": "",
    }


def test_uw_sector_below_threshold_dropped_from_buy_candidates(monkeypatch):
    """A BUY in an UNDERWEIGHT sector with score below the threshold must
    NOT appear in buy_candidates even when the CIO advanced it. This is
    the 2026-05-13 HD/MCD-in-Consumer-Disc case."""
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", True, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_UW_MIN_SCORE", 80.0, raising=False
    )

    state = {
        "investment_theses": {
            "MCD": _make_thesis("MCD", "BUY", 61.0, sector="Consumer Discretionary"),
        },
        "prior_theses": {},
        "new_population": [],
        "sector_map": {"MCD": "Consumer Discretionary"},
        "sector_ratings": {"Consumer Discretionary": {"rating": "underweight"}},
        "entry_theses": {},
        "advanced_tickers": ["MCD"],
    }

    payload = _build_signals_payload(state)
    candidate_tickers = {c["ticker"] for c in payload["buy_candidates"]}
    assert "MCD" not in candidate_tickers, (
        f"UW-sector BUY below threshold leaked into buy_candidates: {candidate_tickers}"
    )


def test_uw_sector_above_threshold_kept_in_buy_candidates(monkeypatch):
    """A BUY in an UNDERWEIGHT sector with score ABOVE the threshold is kept
    — the gate only blocks weak picks in fighting-the-macro-call sectors,
    not strong-conviction ones."""
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", True, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_UW_MIN_SCORE", 80.0, raising=False
    )

    state = {
        "investment_theses": {
            "AMZN": _make_thesis("AMZN", "BUY", 85.0, sector="Consumer Discretionary"),
        },
        "prior_theses": {},
        "new_population": [],
        "sector_map": {"AMZN": "Consumer Discretionary"},
        "sector_ratings": {"Consumer Discretionary": {"rating": "underweight"}},
        "entry_theses": {},
        "advanced_tickers": ["AMZN"],
    }

    payload = _build_signals_payload(state)
    candidate_tickers = {c["ticker"] for c in payload["buy_candidates"]}
    assert "AMZN" in candidate_tickers, (
        f"UW-sector BUY ABOVE threshold was wrongly blocked: {candidate_tickers}"
    )


def test_ow_sector_not_affected_by_gate(monkeypatch):
    """OVERWEIGHT sectors are not gated — any score that earned ENTER passes."""
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", True, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_UW_MIN_SCORE", 80.0, raising=False
    )

    state = {
        "investment_theses": {
            "NVDA": _make_thesis("NVDA", "BUY", 65.0, sector="Technology"),
        },
        "prior_theses": {},
        "new_population": [],
        "sector_map": {"NVDA": "Technology"},
        "sector_ratings": {"Technology": {"rating": "overweight"}},
        "entry_theses": {},
        "advanced_tickers": ["NVDA"],
    }

    payload = _build_signals_payload(state)
    candidate_tickers = {c["ticker"] for c in payload["buy_candidates"]}
    assert "NVDA" in candidate_tickers


def test_market_weight_sector_not_affected_by_gate(monkeypatch):
    """MARKET_WEIGHT sectors are not gated either — gate only fires for UW."""
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", True, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_UW_MIN_SCORE", 80.0, raising=False
    )

    state = {
        "investment_theses": {
            "JNJ": _make_thesis("JNJ", "BUY", 65.0, sector="Healthcare"),
        },
        "prior_theses": {},
        "new_population": [],
        "sector_map": {"JNJ": "Healthcare"},
        "sector_ratings": {"Healthcare": {"rating": "market_weight"}},
        "entry_theses": {},
        "advanced_tickers": ["JNJ"],
    }

    payload = _build_signals_payload(state)
    candidate_tickers = {c["ticker"] for c in payload["buy_candidates"]}
    assert "JNJ" in candidate_tickers


def test_gate_disabled_passes_everything(monkeypatch):
    """When the gate is disabled at config level, UW-sector low-score picks
    pass through. Provides the kill-switch behavior for rollback."""
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", False, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_UW_MIN_SCORE", 80.0, raising=False
    )

    state = {
        "investment_theses": {
            "MCD": _make_thesis("MCD", "BUY", 61.0, sector="Consumer Discretionary"),
        },
        "prior_theses": {},
        "new_population": [],
        "sector_map": {"MCD": "Consumer Discretionary"},
        "sector_ratings": {"Consumer Discretionary": {"rating": "underweight"}},
        "entry_theses": {},
        "advanced_tickers": ["MCD"],
    }

    payload = _build_signals_payload(state)
    candidate_tickers = {c["ticker"] for c in payload["buy_candidates"]}
    assert "MCD" in candidate_tickers, (
        "Gate disabled — should not block UW picks"
    )


def test_held_uw_sector_position_not_dropped(monkeypatch):
    """A HELD position in a UW sector keeps its HOLD/EXIT semantics — the
    gate only blocks NEW entries (signal='ENTER'). This pins that existing
    portfolio exposures aren't force-liquidated by a sector downgrade."""
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", True, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_UW_MIN_SCORE", 80.0, raising=False
    )

    state = {
        "investment_theses": {
            "HD": _make_thesis("HD", "HOLD", 62.0, sector="Consumer Discretionary"),
        },
        "prior_theses": {
            "HD": _make_thesis("HD", "BUY", 70.0, sector="Consumer Discretionary"),
        },
        "new_population": [
            {"ticker": "HD", "sector": "Consumer Discretionary"}
        ],
        "sector_map": {"HD": "Consumer Discretionary"},
        "sector_ratings": {"Consumer Discretionary": {"rating": "underweight"}},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    # HD as held position should appear in signals (not buy_candidates).
    # The gate operates only on ENTER signals → no impact on HOLD reaffirms.
    assert "HD" in payload["signals"], "Held HD position should still be in signals"
