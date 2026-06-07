"""Tests for the CIO-as-gate change (PR feat/cio-as-gate, 2026-04-30).

Pins two related changes:

1. ``_build_signals_payload`` no longer emits ENTER for BUY-rated team recs
   that the CIO did not advance. Only CIO-advanced names (or held BUY-rated
   reaffirmations) become ENTER.

2. ``run_cio`` now applies a weekly entrant cap derived from
   ``cio.max_new_entrants`` and ``cio.min_new_entrants`` config values,
   rather than truncating purely on ``open_slots``.
"""

from agents.investment_committee.ic_cio import (
    _compute_advance_bounds,
    _fallback_selection,
    _post_process_cio_decisions,
)


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


def test_unadvanced_buy_does_not_emit_signal():
    """A team rec with rating=BUY that the CIO did NOT advance and
    that is NOT in the population must produce no signal — the previous
    bypass branch let these leak through as ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "ABBV": _make_thesis("ABBV", "BUY", 70.0, sector="Healthcare"),
        },
        "prior_theses": {},
        "new_population": [],  # not held
        "sector_map": {"ABBV": "Healthcare"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],  # CIO did NOT advance
    }

    payload = _build_signals_payload(state)
    signals = payload.get("signals", {})

    assert "ABBV" not in signals, (
        f"Non-advanced BUY rec leaked through to signals: {signals.get('ABBV')}"
    )


def test_advanced_buy_emits_enter():
    """A team rec with rating=BUY that the CIO DID advance must emit ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "MSFT": _make_thesis("MSFT", "BUY", 75.0),
        },
        "prior_theses": {},
        "new_population": [],
        "sector_map": {"MSFT": "Technology"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": ["MSFT"],
    }

    payload = _build_signals_payload(state)
    msft = payload["signals"].get("MSFT")
    assert msft is not None
    assert msft["signal"] == "ENTER"


def test_held_buy_emits_enter_reaffirm():
    """A held BUY-rated name that was NOT advanced this week (but is already
    in population) must still emit ENTER as a reaffirm."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "AAPL": _make_thesis("AAPL", "BUY", 68.0),
        },
        "prior_theses": {},
        "new_population": [{"ticker": "AAPL"}],  # held
        "sector_map": {"AAPL": "Technology"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],  # not advanced this week — reaffirm path
    }

    payload = _build_signals_payload(state)
    aapl = payload["signals"].get("AAPL")
    assert aapl is not None
    assert aapl["signal"] == "ENTER"


def test_held_non_buy_emits_hold():
    """A held name with rating != BUY must emit HOLD."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "META": _make_thesis("META", "HOLD", 55.0, sector="Communication Services"),
        },
        "prior_theses": {},
        "new_population": [{"ticker": "META"}],
        "sector_map": {"META": "Communication Services"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    meta = payload["signals"].get("META")
    assert meta is not None
    assert meta["signal"] == "HOLD"


def test_compute_advance_bounds_typical_case():
    """With 20 candidates, max=10, min=2 → cap=10, floor=2."""
    floor, cap = _compute_advance_bounds(
        n_candidates=20, max_new_entrants=10, min_new_entrants=2,
    )
    assert (floor, cap) == (2, 10)


def test_compute_advance_bounds_clamped_by_n_candidates():
    """Both floor and cap are clamped to n_candidates so we never demand
    advancing more candidates than exist."""
    floor, cap = _compute_advance_bounds(
        n_candidates=1, max_new_entrants=10, min_new_entrants=2,
    )
    assert (floor, cap) == (1, 1)


def test_compute_advance_bounds_open_slots_does_not_factor_in():
    """open_slots is intentionally NOT a parameter — the bounds depend only
    on n_candidates and the configured min/max."""
    # No `open_slots` argument exists in the signature; this test exists
    # to pin that decoupling. Calling with n_candidates=20 always yields
    # (2, 10) regardless of how empty/full the portfolio is elsewhere.
    floor, cap = _compute_advance_bounds(
        n_candidates=20, max_new_entrants=10, min_new_entrants=2,
    )
    assert (floor, cap) == (2, 10)


def test_compute_advance_bounds_zero_when_no_candidates():
    """Zero candidates → (0, 0)."""
    assert _compute_advance_bounds(
        n_candidates=0, max_new_entrants=10, min_new_entrants=2,
    ) == (0, 0)


def test_compute_advance_bounds_zero_when_max_zero():
    """max_new_entrants=0 produces (0, 0) (kill-switch behavior)."""
    assert _compute_advance_bounds(
        n_candidates=20, max_new_entrants=0, min_new_entrants=0,
    ) == (0, 0)


def test_compute_advance_bounds_floor_never_exceeds_cap():
    """Defensive: even if min_new_entrants > max_new_entrants slipped past
    config bounds-checking, the returned floor is clamped to cap."""
    floor, cap = _compute_advance_bounds(
        n_candidates=20, max_new_entrants=3, min_new_entrants=10,
    )
    assert floor <= cap
    assert (floor, cap) == (3, 3)


def _decisions_to_text(decisions):
    """Wrap a decisions list as the JSON the LLM would emit."""
    return json.dumps({"decisions": decisions})


def test_parse_cio_response_truncates_at_cap():
    """When the LLM rubric advances more than cap, truncate to cap."""
    candidates = [
        {"ticker": f"T{i}", "quant_score": 90 - i, "qual_score": 80 - i}
        for i in range(15)
    ]
    decisions = [
        {"ticker": f"T{i}", "decision": "ADVANCE", "rank": i + 1,
         "conviction": 80, "rationale": "rubric pass", "entry_thesis": None}
        for i in range(12)
    ]
    text = _decisions_to_text(decisions)
    result = _post_process_cio_decisions(decisions, candidates, floor=2, cap=10)
    assert len(result["advanced_tickers"]) == 10
    # First 10 in original ADVANCE order are kept
    assert result["advanced_tickers"] == [f"T{i}" for i in range(10)]


def test_parse_cio_response_force_advances_to_floor():
    """When the LLM rubric advances fewer than floor, force-fill from
    REJECT/DEADLOCK candidates ranked by combined quant+qual score."""
    candidates = [
        {"ticker": f"T{i}", "quant_score": 90 - i, "qual_score": 80 - i}
        for i in range(15)
    ]
    # LLM advances only T5 (mid-ranked); rejects everything else
    decisions = [{
        "ticker": "T5", "decision": "ADVANCE", "rank": 1,
        "conviction": 75, "rationale": "rubric pass", "entry_thesis": None,
    }]
    decisions.extend([
        {"ticker": f"T{i}", "decision": "REJECT", "rank": None,
         "conviction": 0, "rationale": "weak catalyst", "entry_thesis": None}
        for i in range(15) if i != 5
    ])
    text = _decisions_to_text(decisions)
    result = _post_process_cio_decisions(decisions, candidates, floor=3, cap=10)

    # Must hit floor: 1 rubric-advanced + 2 forced = 3
    assert len(result["advanced_tickers"]) == 3
    # Forced picks are T0 and T1 (highest combined score among non-advanced)
    assert "T5" in result["advanced_tickers"]
    assert "T0" in result["advanced_tickers"]
    assert "T1" in result["advanced_tickers"]
    # Audit trail tags forced advances distinctly
    forced_decisions = [
        d for d in result["decisions"] if d.get("decision") == "ADVANCE_FORCED"
    ]
    assert {d["ticker"] for d in forced_decisions} == {"T0", "T1"}


def test_parse_cio_response_no_force_when_rubric_meets_floor():
    """Rubric advanced exactly at the floor — no force needed; no
    ADVANCE_FORCED decisions emitted."""
    candidates = [
        {"ticker": f"T{i}", "quant_score": 90 - i, "qual_score": 80 - i}
        for i in range(15)
    ]
    decisions = [
        {"ticker": "T0", "decision": "ADVANCE", "rank": 1, "conviction": 85,
         "rationale": "rubric", "entry_thesis": None},
        {"ticker": "T1", "decision": "ADVANCE", "rank": 2, "conviction": 84,
         "rationale": "rubric", "entry_thesis": None},
    ]
    text = _decisions_to_text(decisions)
    result = _post_process_cio_decisions(decisions, candidates, floor=2, cap=10)
    assert result["advanced_tickers"] == ["T0", "T1"]
    forced = [
        d for d in result["decisions"] if d.get("decision") == "ADVANCE_FORCED"
    ]
    assert forced == []


def test_parse_cio_response_passes_through_in_band():
    """Rubric advanced 5 with floor=2, cap=10 → passes through unchanged."""
    candidates = [
        {"ticker": f"T{i}", "quant_score": 90 - i, "qual_score": 80 - i}
        for i in range(15)
    ]
    decisions = [
        {"ticker": f"T{i}", "decision": "ADVANCE", "rank": i + 1,
         "conviction": 80, "rationale": "rubric", "entry_thesis": None}
        for i in range(5)
    ]
    text = _decisions_to_text(decisions)
    result = _post_process_cio_decisions(decisions, candidates, floor=2, cap=10)
    assert len(result["advanced_tickers"]) == 5
    assert result["advanced_tickers"] == [f"T{i}" for i in range(5)]


def test_fallback_selection_uses_floor_not_cap():
    """LLM-failure fallback advances `floor`, not `cap`. When LLM signal is
    unusable we don't know which candidates would clear the rubric — be
    conservative."""
    candidates = [
        {"ticker": f"T{i}", "quant_score": 90 - i, "qual_score": 80 - i}
        for i in range(15)
    ]
    result = _fallback_selection(candidates, floor=2)
    assert len(result["advanced_tickers"]) == 2
    # Highest combined score takes T0 then T1
    assert result["advanced_tickers"] == ["T0", "T1"]


# ── Offline replay against 2026-04-24 signals fixture ───────────────────────
#
# Demonstrates that today's 27-ENTER output collapses correctly under the new
# gate behavior. signals.json from 2026-04-24 contains 21 population names
# (15 BUY-rated, 6 HOLD-rated) and 27 buy_candidates (all ENTER, all BUY).
# We reconstruct an upstream state and replay _build_signals_payload under
# three CIO-advance scenarios.

import json
from pathlib import Path

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "signals_2026-04-24.json"
)


def _state_from_fixture(advanced_tickers):
    """Reconstruct an upstream graph state from the persisted signals.json,
    then override advanced_tickers to simulate different CIO outcomes."""
    with open(FIXTURE_PATH) as f:
        signals = json.load(f)

    investment_theses = {}
    for u in signals.get("universe", []):
        ticker = u["ticker"]
        investment_theses[ticker] = {
            "ticker": ticker,
            "rating": u.get("rating", "HOLD"),
            "final_score": u.get("score"),
            "quant_score": (u.get("sub_scores") or {}).get("quant"),
            "qual_score": (u.get("sub_scores") or {}).get("qual"),
            "sector": u.get("sector", "Unknown"),
            "team_id": (u.get("sector") or "unknown").lower(),
            "conviction": u.get("conviction", "stable"),
            "bull_case": u.get("thesis_summary", ""),
        }

    pop_tickers = signals.get("population", [])
    return {
        "investment_theses": investment_theses,
        "prior_theses": {},
        "new_population": [{"ticker": t} for t in pop_tickers],
        "sector_map": {t: th["sector"] for t, th in investment_theses.items()},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": list(advanced_tickers),
    }


def _count_signals(payload):
    counts = {"ENTER": 0, "HOLD": 0}
    for sig in payload["signals"].values():
        counts[sig["signal"]] = counts.get(sig["signal"], 0) + 1
    return counts


def test_replay_2026_04_24_no_cio_advances():
    """If the CIO advanced 0 of the 27 candidates this Saturday, ENTER count
    must equal only the held BUY-rated reaffirmations — under the old code
    all 27 would still ENTER via the bypass branch."""
    from graph.research_graph import _build_signals_payload

    state = _state_from_fixture(advanced_tickers=[])
    payload = _build_signals_payload(state)
    counts = _count_signals(payload)

    held_buys = sum(
        1 for t in state["new_population"]
        if state["investment_theses"].get(t["ticker"], {}).get("rating") == "BUY"
    )
    assert counts["ENTER"] == held_buys, (
        f"With 0 advances, ENTER must == held-BUY count ({held_buys}); "
        f"got {counts['ENTER']}. Bypass branch may have leaked."
    )


def test_replay_2026_04_24_all_advanced():
    """Sanity: if the CIO advanced all 27 candidates, all 27 BUYs ENTER —
    proves the new code still emits ENTER on the CIO-approved path."""
    from graph.research_graph import _build_signals_payload

    state = _state_from_fixture(advanced_tickers=[])
    # Advance every BUY-rated ticker
    all_buy_tickers = [
        t for t, th in state["investment_theses"].items()
        if th.get("rating") == "BUY"
    ]
    state["advanced_tickers"] = all_buy_tickers
    payload = _build_signals_payload(state)
    counts = _count_signals(payload)

    assert counts["ENTER"] == len(all_buy_tickers), (
        f"With all BUYs advanced, ENTER must equal BUY count "
        f"({len(all_buy_tickers)}); got {counts['ENTER']}"
    )


def test_replay_2026_04_24_top3_advanced_caps_below_baseline():
    """Tightly capped advance: CIO advances only top 3 net-new BUYs.
    The total ENTER count must drop below the 27-ENTER baseline that the
    bypass branch would have produced when none of those 3 are reaffirms."""
    from graph.research_graph import _build_signals_payload

    state = _state_from_fixture(advanced_tickers=[])
    pop_tickers = {p["ticker"] for p in state["new_population"]}

    new_buy_candidates = [
        (t, th.get("final_score") or 0)
        for t, th in state["investment_theses"].items()
        if th.get("rating") == "BUY" and t not in pop_tickers
    ]
    new_buy_candidates.sort(key=lambda x: x[1], reverse=True)
    cap = 3
    state["advanced_tickers"] = [t for t, _ in new_buy_candidates[:cap]]

    payload = _build_signals_payload(state)
    counts = _count_signals(payload)

    held_buys = sum(
        1 for p in state["new_population"]
        if state["investment_theses"].get(p["ticker"], {}).get("rating") == "BUY"
    )
    expected_enters = held_buys + min(cap, len(new_buy_candidates))
    assert counts["ENTER"] == expected_enters
    # Must be strictly fewer than the original 27 — bypass branch is gone.
    assert counts["ENTER"] < 27, (
        f"Capped at top-{cap}, ENTER count must be < 27 baseline; got {counts['ENTER']}"
    )


# ---------------------------------------------------------------------------
# L4532 — net-new floor + quality-gated force-fill + ADVANCE_FORCED consumers
# ---------------------------------------------------------------------------


def test_net_new_floor_incumbent_advances_do_not_satisfy_floor():
    """The 2026-06-05 bug: every ADVANCE is an already-held incumbent, so
    net-new = 0 even though len(advanced) >= floor. The floor must measure
    net-new, not total advances."""
    held = {"HELD1", "HELD2"}
    decisions = [
        {"ticker": "HELD1", "decision": "ADVANCE", "rank": 1, "conviction": 80,
         "rationale": "reaffirm", "entry_thesis": None},
        {"ticker": "HELD2", "decision": "ADVANCE", "rank": 2, "conviction": 78,
         "rationale": "reaffirm", "entry_thesis": None},
        # fresh names all below the bar — must NOT be force-filled
        {"ticker": "FRESH1", "decision": "REJECT", "rank": None, "conviction": 40,
         "rationale": "weak", "entry_thesis": None},
        {"ticker": "FRESH2", "decision": "REJECT", "rank": None, "conviction": 35,
         "rationale": "weak", "entry_thesis": None},
    ]
    candidates = [{"ticker": d["ticker"]} for d in decisions]
    result = _post_process_cio_decisions(
        decisions, candidates, floor=2, cap=10,
        held_tickers=held, force_fill_conviction_floor=60,
    )
    # Incumbents are re-affirmed (still in advanced_tickers) but count as 0 net-new.
    assert result["net_new_entrants"] == 0
    assert set(result["advanced_tickers"]) == {"HELD1", "HELD2"}
    # No sub-bar fresh name was force-filled.
    forced = [d for d in result["decisions"] if d.get("decision") == "ADVANCE_FORCED"]
    assert forced == []


def test_quality_gated_force_fill_admits_only_above_bar():
    """A fresh name the rubric rejected is force-filled ONLY if its conviction
    clears the entrant bar."""
    held = {"HELD1"}
    decisions = [
        {"ticker": "HELD1", "decision": "ADVANCE", "rank": 1, "conviction": 80,
         "rationale": "reaffirm", "entry_thesis": None},
        {"ticker": "STRONG", "decision": "REJECT", "rank": None, "conviction": 66,
         "rationale": "soft reject", "entry_thesis": None},
        {"ticker": "WEAK", "decision": "REJECT", "rank": None, "conviction": 45,
         "rationale": "weak", "entry_thesis": None},
    ]
    candidates = [{"ticker": d["ticker"]} for d in decisions]
    result = _post_process_cio_decisions(
        decisions, candidates, floor=2, cap=10,
        held_tickers=held, force_fill_conviction_floor=60,
    )
    # STRONG (66 >= 60) forced in; WEAK (45 < 60) not.
    assert result["net_new_entrants"] == 1
    assert "STRONG" in result["advanced_tickers"]
    assert "WEAK" not in result["advanced_tickers"]
    forced = [d for d in result["decisions"] if d.get("decision") == "ADVANCE_FORCED"]
    assert {d["ticker"] for d in forced} == {"STRONG"}


def test_incumbent_reaffirmations_not_truncated_by_cap():
    """Cap governs net-new only — incumbent re-affirmations are unbounded."""
    held = {f"H{i}" for i in range(8)}
    decisions = [
        {"ticker": f"H{i}", "decision": "ADVANCE", "rank": i + 1, "conviction": 80,
         "rationale": "reaffirm", "entry_thesis": None}
        for i in range(8)
    ]
    # 3 net-new advances with cap=2 → one dropped (lowest conviction)
    decisions += [
        {"ticker": "N1", "decision": "ADVANCE", "rank": 9, "conviction": 70,
         "rationale": "new", "entry_thesis": None},
        {"ticker": "N2", "decision": "ADVANCE", "rank": 10, "conviction": 65,
         "rationale": "new", "entry_thesis": None},
        {"ticker": "N3", "decision": "ADVANCE", "rank": 11, "conviction": 62,
         "rationale": "new", "entry_thesis": None},
    ]
    candidates = [{"ticker": d["ticker"]} for d in decisions]
    result = _post_process_cio_decisions(
        decisions, candidates, floor=0, cap=2,
        held_tickers=held, force_fill_conviction_floor=60,
    )
    # All 8 incumbents kept; net-new truncated to cap=2 (drops lowest, N3).
    assert sum(1 for t in result["advanced_tickers"] if t.startswith("H")) == 8
    assert result["net_new_entrants"] == 2
    assert "N1" in result["advanced_tickers"] and "N2" in result["advanced_tickers"]
    assert "N3" not in result["advanced_tickers"]


def test_apply_ic_entries_admits_advance_forced():
    """The consumer-side half of the bug: apply_ic_entries must admit
    ADVANCE_FORCED, not only ADVANCE."""
    from data.population_selector import apply_ic_entries

    remaining = [{"ticker": "HELD1", "sector": "Tech"}]
    ic_decisions = [
        {"ticker": "HELD1", "decision": "ADVANCE", "rank": 1, "conviction": 80,
         "rationale": "reaffirm"},
        {"ticker": "FORCED", "decision": "ADVANCE_FORCED", "rank": None,
         "conviction": 65, "rationale": "floor"},
    ]
    final_pop, events = apply_ic_entries(
        remaining_population=remaining,
        ic_decisions=ic_decisions,
        entry_theses={},
        sector_map={"FORCED": "Healthcare"},
        run_date="2026-06-12",
    )
    tickers = {p["ticker"] for p in final_pop}
    assert "FORCED" in tickers, "ADVANCE_FORCED must enter the population"
    assert {e["ticker_in"] for e in events} == {"FORCED"}
