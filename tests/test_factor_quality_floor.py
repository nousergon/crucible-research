"""Tests for the Phase 4 factor-based structural quality floor in
``_build_signals_payload``.

The floor blocks NEW ENTER signals whose within-sector ``quality_score``
(produced by the Phase 1c factor scoring module + threaded through
``score_aggregator``) falls below the configured percentile. Replaces the
dormant Piotroski-lite ``data/scanner.py:apply_quality_filter`` retired in
the same PR.

Plan: ~/Development/alpha-engine-docs/private/factor-substrate-260513.md (Phase 4).
"""


from graph.research_graph import _build_signals_payload


def _make_thesis(
    ticker,
    *,
    sector="Technology",
    final_score=75.0,
    quality_pct=None,
):
    return {
        "ticker": ticker,
        "sector": sector,
        "team_id": "technology",
        "final_score": final_score,
        "quant_score": final_score,
        "qual_score": final_score,
        "factor_quality_score": quality_pct,
        "bull_case": "",
        "conviction": "stable",
        "rating": "BUY",
    }


def _make_state(
    *,
    theses,
    sector_map=None,
    sector_ratings=None,
    advanced_tickers=None,
):
    return {
        "investment_theses": theses,
        "prior_theses": {},
        "new_population": [{"ticker": t} for t in theses.keys()],
        "sector_map": sector_map or {},
        "sector_ratings": sector_ratings or {},
        "entry_theses": {},
        "advanced_tickers": advanced_tickers or list(theses.keys()),
        "exits": [],
        "run_date": "2026-05-14",
        "run_time": "2026-05-14T12:00:00Z",
        "market_regime": "bull",
        "sector_modifiers": {},
    }


def _apply_floor_only(monkeypatch, *, enabled=True, min_pct=10.0, exempt=None):
    """Pin the quality floor + disable the macro-sector coherence gate so
    tests can isolate floor behavior without bleed-through from the other
    structural gate that runs in the same buy_candidates loop."""
    monkeypatch.setattr(
        "graph.research_graph.FACTOR_QUALITY_FLOOR_ENABLED", enabled, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.FACTOR_QUALITY_FLOOR_MIN_PERCENTILE", min_pct, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.FACTOR_QUALITY_FLOOR_EXEMPT_SECTORS",
        exempt if exempt is not None else [],
        raising=False,
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", False, raising=False
    )


def test_blocks_below_floor(monkeypatch):
    _apply_floor_only(monkeypatch, min_pct=10.0)

    state = _make_state(
        theses={
            "GOOD": _make_thesis("GOOD", quality_pct=80.0),
            "BAD": _make_thesis("BAD", quality_pct=5.0),  # bottom decile
        },
        sector_map={"GOOD": "Technology", "BAD": "Technology"},
    )
    payload = _build_signals_payload(state)

    tickers = [c["ticker"] for c in payload["buy_candidates"]]
    assert "GOOD" in tickers
    assert "BAD" not in tickers


def test_allows_at_or_above_floor(monkeypatch):
    _apply_floor_only(monkeypatch, min_pct=10.0)

    state = _make_state(
        theses={
            "EDGE": _make_thesis("EDGE", quality_pct=10.0),
            "JUST_OVER": _make_thesis("JUST_OVER", quality_pct=10.1),
        },
        sector_map={"EDGE": "Technology", "JUST_OVER": "Technology"},
    )
    payload = _build_signals_payload(state)

    tickers = [c["ticker"] for c in payload["buy_candidates"]]
    # Strictly-less-than predicate → 10.0 percentile passes
    assert "EDGE" in tickers
    assert "JUST_OVER" in tickers


def test_exempt_sector_bypasses_floor(monkeypatch):
    _apply_floor_only(
        monkeypatch, min_pct=10.0, exempt=["Financial", "Real Estate", "Utilities"]
    )

    state = _make_state(
        theses={
            "BANK": _make_thesis("BANK", sector="Financial", quality_pct=2.0),
            "TECH_BAD": _make_thesis("TECH_BAD", sector="Technology", quality_pct=2.0),
        },
        sector_map={"BANK": "Financial", "TECH_BAD": "Technology"},
    )
    payload = _build_signals_payload(state)

    tickers = [c["ticker"] for c in payload["buy_candidates"]]
    assert "BANK" in tickers       # exempt sector → bypass
    assert "TECH_BAD" not in tickers  # non-exempt → blocked


def test_missing_quality_score_lets_through(monkeypatch):
    # Graceful degrade — same pattern as the rest of the factor blend.
    # When a ticker has no factor profile (e.g. Phase 1c writer hasn't
    # fired yet for new constituents), the floor lets the ticker through.
    _apply_floor_only(monkeypatch, min_pct=10.0)

    state = _make_state(
        theses={"UNCOVERED": _make_thesis("UNCOVERED", quality_pct=None)},
        sector_map={"UNCOVERED": "Technology"},
    )
    payload = _build_signals_payload(state)

    tickers = [c["ticker"] for c in payload["buy_candidates"]]
    assert "UNCOVERED" in tickers


def test_disabled_is_no_op(monkeypatch):
    _apply_floor_only(monkeypatch, enabled=False)

    state = _make_state(
        theses={"BAD": _make_thesis("BAD", quality_pct=1.0)},
        sector_map={"BAD": "Technology"},
    )
    payload = _build_signals_payload(state)

    tickers = [c["ticker"] for c in payload["buy_candidates"]]
    assert "BAD" in tickers  # floor disabled → not blocked


def test_gate_only_affects_enter(monkeypatch):
    # HOLD / EXIT signals are not gated — the floor is a NEW-buy guard
    _apply_floor_only(monkeypatch, min_pct=99.0)

    # Held ticker without a fresh thesis → carryover path produces HOLD.
    state = {
        "investment_theses": {},
        "prior_theses": {"HELD_BAD": {"score": 75.0, "rating": "HOLD", "team_id": None}},
        "new_population": [{"ticker": "HELD_BAD", "sector": "Technology"}],
        "sector_map": {"HELD_BAD": "Technology"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
        "exits": [],
        "run_date": "2026-05-14",
        "run_time": "2026-05-14T12:00:00Z",
        "market_regime": "bull",
        "sector_modifiers": {},
    }
    payload = _build_signals_payload(state)

    signals = payload["signals"]
    assert signals["HELD_BAD"]["signal"] == "HOLD"
    # HOLD doesn't enter buy_candidates regardless of the floor
    assert "HELD_BAD" not in [c["ticker"] for c in payload["buy_candidates"]]


def test_coherence_gate_runs_alongside_quality_floor(monkeypatch):
    # Composition with macro-sector-coherence-gate — UW sector + low score
    # is filtered by the coherence gate; the quality floor is the second
    # layer. Both produce the same end result here: not in buy_candidates.
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_GATE_ENABLED", True, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.SECTOR_COHERENCE_UW_MIN_SCORE", 80.0, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.FACTOR_QUALITY_FLOOR_ENABLED", True, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.FACTOR_QUALITY_FLOOR_MIN_PERCENTILE", 10.0, raising=False
    )
    monkeypatch.setattr(
        "graph.research_graph.FACTOR_QUALITY_FLOOR_EXEMPT_SECTORS", [], raising=False
    )

    state = _make_state(
        theses={
            "UW_LOW_SCORE_BAD_Q": _make_thesis(
                "UW_LOW_SCORE_BAD_Q",
                sector="Consumer Discretionary",
                final_score=60.0,
                quality_pct=5.0,
            ),
        },
        sector_map={"UW_LOW_SCORE_BAD_Q": "Consumer Discretionary"},
        sector_ratings={"Consumer Discretionary": {"rating": "underweight"}},
    )
    payload = _build_signals_payload(state)

    # Coherence gate fires first (score 60 < 80 in UW sector). Either
    # gate alone would block this candidate.
    assert "UW_LOW_SCORE_BAD_Q" not in [c["ticker"] for c in payload["buy_candidates"]]
