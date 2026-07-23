"""
Tests for the UNIVERSE_DROP guardrail in compute_exits_and_open_slots.

Origin: 2026-04-20 — TSM + ASML persisted as population incumbents
despite being absent from the current S&P 500+400 constituents list
(ADRs are excluded by S&P index rules). Every weekly run, the exit
evaluator re-included them because there was no constituents check —
only score/tenure gates. ArcticDB DataPhase1 only writes the S&P 900,
so downstream executor reads for those tickers raised
NoSuchVersionException deep in simulate mode.

These tests verify the constituents-whitelist behavior without
requiring the full graph state.
"""

from __future__ import annotations

from data.population_selector import (
    apply_ic_entries,
    compute_exits_and_open_slots,
)

DEFAULT_CONFIG = {
    "target_size": 25,
    "rotation": {
        "min_long_term_score": 45,
        "min_tenure_weeks": 2,
        "thesis_collapse_threshold": 40,
        "min_rotation_pct": 0.0,  # disable min-rotation floor for deterministic tests
        "max_rotations_per_run": 10,
    },
}


def _make_incumbent(ticker: str, score: float = 70.0, sector: str = "Tech", tenure_weeks: int = 1) -> dict:
    """Tenure=1 keeps incumbents below min_tenure_weeks=2 so the force-rotation
    floor doesn't accidentally rotate them — isolates UNIVERSE_DROP behavior
    from the min_rotation_pct logic."""
    return {
        "ticker": ticker,
        "sector": sector,
        "long_term_score": score,
        "tenure_weeks": tenure_weeks,
        "entry_date": "2026-04-15",  # recent enough that tenure stays ~1 vs run_date
    }


class TestUniverseDropGuardrail:
    def test_out_of_universe_incumbent_is_dropped(self):
        current = [_make_incumbent("TSM"), _make_incumbent("AAPL")]
        constituents = {"AAPL", "MSFT"}  # TSM not included
        theses = {"AAPL": {"long_term_score": 70}, "TSM": {"long_term_score": 70}}

        _remaining, exits, _open = compute_exits_and_open_slots(
            current_population=current,
            investment_theses=theses,
            config=DEFAULT_CONFIG,
            run_date="2026-04-20",
            constituents=constituents,
        )

        universe_drops = [e for e in exits if e["type"] == "UNIVERSE_DROP"]
        assert len(universe_drops) == 1
        assert universe_drops[0]["ticker_out"] == "TSM"
        assert "not in current S&P 500+400 constituents" in universe_drops[0]["reason"]
        # TSM definitely gone from remaining; AAPL's fate depends on unrelated
        # rotation logic which this test doesn't exercise.
        assert "TSM" not in {p["ticker"] for p in _remaining}

    def test_no_constituents_kwarg_preserves_legacy_behavior(self):
        """When constituents=None, no UNIVERSE_DROP events are emitted — identical
        to pre-guardrail behavior for callers that don't supply constituents."""
        current = [_make_incumbent("TSM"), _make_incumbent("AAPL")]
        theses = {"AAPL": {"long_term_score": 70}, "TSM": {"long_term_score": 70}}

        _remaining, exits, _open = compute_exits_and_open_slots(
            current_population=current,
            investment_theses=theses,
            config=DEFAULT_CONFIG,
            run_date="2026-04-20",
            # constituents omitted → None → no universe check
        )

        assert not [e for e in exits if e["type"] == "UNIVERSE_DROP"]

    def test_universe_drops_do_not_count_toward_max_rotations(self):
        """UNIVERSE_DROP is reconciliation, not volitional rotation."""
        # 5 grandfathered outliers + 1 healthy incumbent
        current = [
            _make_incumbent("TSM"),
            _make_incumbent("ASML"),
            _make_incumbent("BABA"),
            _make_incumbent("NVO"),
            _make_incumbent("SONY"),
            _make_incumbent("AAPL"),
        ]
        constituents = {"AAPL", "MSFT"}
        theses = {t["ticker"]: {"long_term_score": 70} for t in current}

        config = dict(DEFAULT_CONFIG)
        config["rotation"] = dict(DEFAULT_CONFIG["rotation"])
        config["rotation"]["max_rotations_per_run"] = 1  # very tight

        _remaining, exits, _open = compute_exits_and_open_slots(
            current_population=current,
            investment_theses=theses,
            config=config,
            run_date="2026-04-20",
            constituents=constituents,
        )

        # All 5 outliers dropped despite max_rotations_per_run=1 because
        # UNIVERSE_DROP doesn't increment rotations_used.
        universe_drops = [e for e in exits if e["type"] == "UNIVERSE_DROP"]
        assert len(universe_drops) == 5
        assert {e["ticker_out"] for e in universe_drops} == {"TSM", "ASML", "BABA", "NVO", "SONY"}

    def test_accepts_set_or_list_or_frozenset(self):
        """constituents kwarg is tolerant of any iterable container."""
        current = [_make_incumbent("AAPL"), _make_incumbent("TSM")]
        theses = {"AAPL": {"long_term_score": 70}, "TSM": {"long_term_score": 70}}

        for container in (
            {"AAPL", "MSFT"},             # set
            frozenset({"AAPL", "MSFT"}),  # frozenset
            ["AAPL", "MSFT"],             # list (coerced internally)
        ):
            _rem, exits, _open = compute_exits_and_open_slots(
                current_population=current,
                investment_theses=theses,
                config=DEFAULT_CONFIG,
                run_date="2026-04-20",
                constituents=container,
            )
            assert len([e for e in exits if e["type"] == "UNIVERSE_DROP"]) == 1

    def test_all_incumbents_in_universe_no_drops(self):
        current = [_make_incumbent("AAPL"), _make_incumbent("MSFT")]
        constituents = {"AAPL", "MSFT", "GOOG"}
        theses = {"AAPL": {"long_term_score": 70}, "MSFT": {"long_term_score": 70}}

        _remaining, exits, _open = compute_exits_and_open_slots(
            current_population=current,
            investment_theses=theses,
            config=DEFAULT_CONFIG,
            run_date="2026-04-20",
            constituents=constituents,
        )

        assert not [e for e in exits if e["type"] == "UNIVERSE_DROP"]


# ---------------------------------------------------------------------------
# L4534 — unconditional min_rotation_floor removed + replacement-aware swap
# ---------------------------------------------------------------------------

_ROT_CONFIG = {
    "target_size": 3,
    "rotation": {
        "min_long_term_score": 45,
        "min_tenure_weeks": 2,
        "thesis_collapse_threshold": 40,
        "min_rotation_pct": 0.50,  # would have force-rotated under the old floor
        "max_rotations_per_run": 10,
    },
}


def test_no_unconditional_forced_rotation_of_healthy_names():
    """The removed min_rotation_floor must NOT eject healthy incumbents — even
    with min_rotation_pct high, names >= min_long_term_score stay."""
    current = [
        {"ticker": "AAA", "sector": "Tech", "long_term_score": 60, "entry_date": "2026-01-01"},
        {"ticker": "BBB", "sector": "Tech", "long_term_score": 55, "entry_date": "2026-01-01"},
        {"ticker": "CCC", "sector": "Tech", "long_term_score": 61, "entry_date": "2026-01-01"},
    ]
    theses = {t["ticker"]: {"long_term_score": t["long_term_score"]} for t in current}
    remaining, exits, _open = compute_exits_and_open_slots(
        current_population=current,
        investment_theses=theses,
        config=_ROT_CONFIG,
        run_date="2026-06-12",
    )
    assert exits == []  # old floor would have force-rotated ~half
    assert {p["ticker"] for p in remaining} == {"AAA", "BBB", "CCC"}


def test_conditional_swap_rotates_out_only_on_upgrade():
    """Over target after a strong entrant → the weakest incumbent (below the
    entrant) is swapped out as FORCED_ROTATION."""
    remaining = [
        {"ticker": "AAA", "sector": "Tech", "long_term_score": 80},
        {"ticker": "WEAK", "sector": "Tech", "long_term_score": 50},
        {"ticker": "BBB", "sector": "Tech", "long_term_score": 75},
    ]
    pop, events = apply_ic_entries(
        remaining_population=remaining,
        ic_decisions=[{"ticker": "NEW", "decision": "ADVANCE", "rank": 1, "conviction": 70}],
        entry_theses={},
        sector_map={"NEW": "Tech"},
        run_date="2026-06-12",
        target_size=3,
    )
    tickers = {p["ticker"] for p in pop}
    assert tickers == {"AAA", "BBB", "NEW"}   # WEAK (50 < 70) swapped out, back to target
    swaps = [e for e in events if e.get("type") == "FORCED_ROTATION"]
    assert {e["ticker_out"] for e in swaps} == {"WEAK"}


def test_no_swap_when_entrant_does_not_upgrade():
    """Saturation: the entrant is weaker than every incumbent → no rotation,
    book holds over target (no erosion)."""
    remaining = [
        {"ticker": "AAA", "sector": "Tech", "long_term_score": 80},
        {"ticker": "BBB", "sector": "Tech", "long_term_score": 75},
        {"ticker": "CCC", "sector": "Tech", "long_term_score": 70},
    ]
    pop, events = apply_ic_entries(
        remaining_population=remaining,
        ic_decisions=[{"ticker": "WEAKNEW", "decision": "ADVANCE", "rank": 1, "conviction": 40}],
        entry_theses={},
        sector_map={"WEAKNEW": "Tech"},
        run_date="2026-06-12",
        target_size=3,
    )
    assert {p["ticker"] for p in pop} == {"AAA", "BBB", "CCC", "WEAKNEW"}
    assert [e for e in events if e.get("type") == "FORCED_ROTATION"] == []


def test_no_swap_when_no_entrants():
    """No net-new entrants (the saturated 0-add week) → no rotation at all."""
    remaining = [
        {"ticker": "AAA", "sector": "Tech", "long_term_score": 80},
        {"ticker": "BBB", "sector": "Tech", "long_term_score": 50},
    ]
    pop, events = apply_ic_entries(
        remaining_population=remaining,
        ic_decisions=[{"ticker": "X", "decision": "REJECT"}],
        entry_theses={},
        sector_map={},
        run_date="2026-06-12",
        target_size=1,  # over target, but no entrants → still no rotation
    )
    assert {p["ticker"] for p in pop} == {"AAA", "BBB"}
    assert [e for e in events if e.get("type") == "FORCED_ROTATION"] == []
