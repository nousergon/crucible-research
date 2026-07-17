"""Tests for scoring/focus_list.py — shadow focus-list substrate (PR 1 of
the scanner-placement arc, ``alpha-engine-docs/private/scanner-260514.md``).

Validates the regime-blended per-team prescreen built on top of the
Phase 1c factor composites + Phase 3 blend formula. Pure helper tests
— no S3, no graph, no agent.
"""

import pytest

from unittest.mock import patch

from scoring.focus_list import (
    FOCUS_LIST_DEFAULT_SIZE,
    FOCUS_LIST_HARD_CAP,
    FOCUS_LIST_MIN_SECTOR_SIZE,
    FocusListEntry,
    _assign_stance,
    build_focus_list,
    build_focus_list_audit_lookup,
    build_pure_quant_focus_lookup,
    compute_focus_scores,
    summarize_focus_list,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def regime_weights():
    """Mirrors ``alpha-engine-config/research/scoring.yaml`` factor_blend block."""
    return {
        "bull": {
            "momentum_score": 0.40,
            "quality_score": 0.30,
            "value_score": 0.20,
            "low_vol_score": -0.10,
        },
        "bear": {
            "low_vol_score": 0.40,
            "quality_score": 0.30,
            "momentum_score": -0.20,
            "value_score": 0.10,
        },
        "neutral": {
            "momentum_score": 0.25,
            "quality_score": 0.25,
            "value_score": 0.25,
            "low_vol_score": 0.25,
        },
    }


@pytest.fixture
def sector_team_map():
    return {
        "Technology": "technology",
        "Communication Services": "technology",
        "Health Care": "healthcare",
        "Financials": "financials",
    }


def _profile(sector, q=None, m=None, v=None, lv=None):
    """Helper for compact profile dicts."""
    out = {"sector": sector}
    if q is not None:
        out["quality_score"] = q
    if m is not None:
        out["momentum_score"] = m
    if v is not None:
        out["value_score"] = v
    if lv is not None:
        out["low_vol_score"] = lv
    return out


# ── _assign_stance ──────────────────────────────────────────────────────────


def test_stance_momentum_dominant():
    profile = _profile("Technology", q=40, m=90, v=20, lv=30)
    assert _assign_stance(profile) == "momentum"


def test_stance_quality_dominant():
    profile = _profile("Technology", q=88, m=30, v=20, lv=40)
    assert _assign_stance(profile) == "quality"


def test_stance_value_dominant():
    profile = _profile("Financials", q=50, m=20, v=85, lv=40)
    assert _assign_stance(profile) == "value"


def test_stance_low_vol_dominant():
    profile = _profile("Health Care", q=40, m=20, v=30, lv=92)
    assert _assign_stance(profile) == "low_vol"


def test_stance_unknown_when_all_missing():
    """No factor scores → stance = unknown (graceful degrade)."""
    profile = {"sector": "Technology"}
    assert _assign_stance(profile) == "unknown"


def test_stance_tie_break_favors_momentum():
    """Tied scores → iteration order in _STANCE_BY_FACTOR (momentum first =
    mild pro-growth tie-break)."""
    profile = _profile("Technology", q=70, m=70, v=70, lv=70)
    assert _assign_stance(profile) == "momentum"


# ── compute_focus_scores ────────────────────────────────────────────────────


def test_compute_focus_scores_empty_input(regime_weights):
    assert compute_focus_scores({}, "bull", regime_weights) == {}
    assert compute_focus_scores(None, "bull", regime_weights) == {}


def test_compute_focus_scores_bull_favors_momentum(regime_weights):
    """In BULL, a high-momentum / low-vol ticker should outrank a high-low-vol
    / low-momentum ticker — the low_vol coefficient is negative."""
    profiles = {
        "MOMO": _profile("Technology", q=50, m=90, v=50, lv=20),
        "DEFO": _profile("Technology", q=50, m=20, v=50, lv=90),
    }
    scores = compute_focus_scores(profiles, "bull", regime_weights)
    assert scores["MOMO"]["focus_score"] > scores["DEFO"]["focus_score"]


def test_compute_focus_scores_bear_favors_low_vol(regime_weights):
    """In BEAR the sign flips — momentum penalized, low_vol rewarded."""
    profiles = {
        "MOMO": _profile("Technology", q=50, m=90, v=50, lv=20),
        "DEFO": _profile("Technology", q=50, m=20, v=50, lv=90),
    }
    scores = compute_focus_scores(profiles, "bear", regime_weights)
    assert scores["DEFO"]["focus_score"] > scores["MOMO"]["focus_score"]


def test_compute_focus_scores_neutral_balanced(regime_weights):
    """In NEUTRAL all 4 weights equal — score = mean of the 4 composites."""
    profiles = {
        "T1": _profile("Technology", q=50, m=50, v=50, lv=50),
    }
    scores = compute_focus_scores(profiles, "neutral", regime_weights)
    # Equal weights, equal composites → focus_score == 50
    assert scores["T1"]["focus_score"] == 50.0


def test_compute_focus_scores_partial_coverage(regime_weights):
    """Missing factor → its weight reallocates pro-rata to available factors."""
    full = {"FULL": _profile("Technology", q=80, m=80, v=80, lv=80)}
    partial = {"PARTIAL": _profile("Technology", q=80, m=80)}  # missing value + low_vol
    scores_full = compute_focus_scores(full, "bull", regime_weights)
    scores_partial = compute_focus_scores(partial, "bull", regime_weights)
    # Both should be on the 0-100 scale after renormalization
    assert 0.0 <= scores_full["FULL"]["focus_score"] <= 100.0
    assert 0.0 <= scores_partial["PARTIAL"]["focus_score"] <= 100.0


def test_compute_focus_scores_skips_no_components(regime_weights):
    """Profile with no factor scores at all → skipped entirely."""
    profiles = {"EMPTY": {"sector": "Technology"}}
    scores = compute_focus_scores(profiles, "bull", regime_weights)
    assert "EMPTY" not in scores


def test_compute_focus_scores_unknown_regime_returns_empty(regime_weights):
    """Regime not in weights config → skip all tickers (don't synthesize)."""
    profiles = {"T1": _profile("Technology", q=80, m=80, v=80, lv=80)}
    scores = compute_focus_scores(profiles, "irrational_exuberance", regime_weights)
    assert scores == {}


def test_compute_focus_scores_carries_breakdown(regime_weights):
    """Per-factor breakdown is propagated to the output entry."""
    profiles = {"T1": _profile("Technology", q=80, m=80, v=80, lv=80)}
    scores = compute_focus_scores(profiles, "bull", regime_weights)
    breakdown = scores["T1"]["factor_blend_breakdown"]
    # 4 BULL weights are all non-zero, so all 4 should contribute
    assert set(breakdown.keys()) == {
        "momentum_score", "quality_score", "value_score", "low_vol_score",
    }


# ── build_focus_list ────────────────────────────────────────────────────────


def _scores(*tuples):
    """Helper: build a focus_scores dict from (ticker, sector, score, stance) tuples."""
    return {
        ticker: {
            "sector": sector,
            "focus_score": score,
            "stance": stance,
            "quality_score": None,
            "momentum_score": None,
            "value_score": None,
            "low_vol_score": None,
            "factor_blend_breakdown": {},
        }
        for ticker, sector, score, stance in tuples
    }


def test_build_focus_list_empty_scores_returns_empty_per_team(sector_team_map):
    """Empty input → every team gets an empty list (not missing keys)."""
    result = build_focus_list({}, sector_team_map)
    assert set(result.keys()) == set(sector_team_map.values())
    for team_id, entries in result.items():
        assert entries == []


def test_build_focus_list_top_n_per_team(sector_team_map):
    """Default size cap applied per team."""
    scores = _scores(*[
        (f"T{i}", "Technology", 100 - i, "momentum")
        for i in range(30)
    ])
    result = build_focus_list(scores, sector_team_map)
    assert len(result["technology"]) == FOCUS_LIST_DEFAULT_SIZE
    # Highest score first
    assert result["technology"][0].ticker == "T0"
    assert result["technology"][0].focus_score == 100.0
    assert result["technology"][0].rank_in_team == 1
    assert result["technology"][-1].rank_in_team == FOCUS_LIST_DEFAULT_SIZE


def test_build_focus_list_hard_cap_enforced(sector_team_map):
    """Per-team override capped at FOCUS_LIST_HARD_CAP (20)."""
    scores = _scores(*[
        (f"T{i}", "Technology", 100 - i, "momentum")
        for i in range(50)
    ])
    result = build_focus_list(
        scores, sector_team_map,
        per_team_size={"technology": 999},  # try to bypass the cap
    )
    assert len(result["technology"]) == FOCUS_LIST_HARD_CAP


def test_build_focus_list_per_team_override_honored(sector_team_map):
    """Within-cap override is respected."""
    scores = _scores(*[
        (f"T{i}", "Technology", 100 - i, "momentum")
        for i in range(30)
    ])
    result = build_focus_list(
        scores, sector_team_map,
        per_team_size={"technology": 12},
    )
    assert len(result["technology"]) == 12


def test_build_focus_list_min_carveout_for_thin_sector(sector_team_map):
    """Sector with < FOCUS_LIST_MIN_SECTOR_SIZE candidates passes them all through."""
    scores = _scores(
        ("F1", "Financials", 80, "value"),
        ("F2", "Financials", 60, "value"),
    )
    result = build_focus_list(scores, sector_team_map, per_team_size={"financials": 5})
    # Only 2 candidates, both under the team cap, so both kept
    assert len(result["financials"]) == 2


def test_build_focus_list_sector_to_team_grouping(sector_team_map):
    """Multiple sectors mapping to one team (Technology + Communication Services
    → technology) coalesce into a single ranked list."""
    scores = _scores(
        ("TECH1", "Technology", 90, "momentum"),
        ("COMM1", "Communication Services", 95, "momentum"),
        ("TECH2", "Technology", 70, "quality"),
        ("COMM2", "Communication Services", 60, "quality"),
    )
    result = build_focus_list(scores, sector_team_map, per_team_size={"technology": 10})
    assert len(result["technology"]) == 4
    # Ranked by focus_score across both sectors
    assert [e.ticker for e in result["technology"]] == ["COMM1", "TECH1", "TECH2", "COMM2"]
    # rank_in_team is across both sectors
    assert result["technology"][0].rank_in_team == 1
    # rank_in_sector is within each sector individually
    comm1 = next(e for e in result["technology"] if e.ticker == "COMM1")
    assert comm1.rank_in_sector == 1  # top of its sector


def test_build_focus_list_unknown_sector_dropped(sector_team_map):
    """Sector not in sector_team_map → ticker dropped (no team to assign to)."""
    scores = _scores(
        ("KNOWN", "Technology", 80, "momentum"),
        ("ORPHAN", "Materials", 95, "momentum"),
    )
    result = build_focus_list(scores, sector_team_map)
    all_tickers = {e.ticker for entries in result.values() for e in entries}
    assert "KNOWN" in all_tickers
    assert "ORPHAN" not in all_tickers


def test_build_focus_list_entry_carries_factor_scores(sector_team_map):
    """The 4 underlying factor scores are preserved on the entry for downstream
    audit + the @tool get_factor_profile boundary."""
    scores = {
        "T1": {
            "sector": "Technology",
            "focus_score": 75.0,
            "stance": "momentum",
            "quality_score": 70.0,
            "momentum_score": 90.0,
            "value_score": 50.0,
            "low_vol_score": 30.0,
            "factor_blend_breakdown": {"momentum_score": 36.0, "quality_score": 21.0},
        },
    }
    result = build_focus_list(scores, sector_team_map)
    entry = result["technology"][0]
    assert entry.quality_score == 70.0
    assert entry.momentum_score == 90.0
    assert entry.value_score == 50.0
    assert entry.low_vol_score == 30.0
    assert entry.factor_blend_breakdown == {
        "momentum_score": 36.0, "quality_score": 21.0,
    }


def test_focus_list_entry_to_dict_serializable():
    """FocusListEntry.to_dict() returns plain dict — JSON-serializable for
    scanner_evals / dashboard surface."""
    entry = FocusListEntry(
        ticker="NVDA",
        sector="Technology",
        team_id="technology",
        focus_score=78.5,
        stance="momentum",
        rank_in_sector=2,
        rank_in_team=2,
        quality_score=71.0,
        momentum_score=92.0,
        value_score=18.0,
        low_vol_score=22.0,
    )
    d = entry.to_dict()
    assert d["ticker"] == "NVDA"
    assert d["focus_score"] == 78.5
    assert d["stance"] == "momentum"
    assert d["factor_blend_breakdown"] == {}


# ── End-to-end: factor profiles → focus list ────────────────────────────────


def test_end_to_end_bull_regime_focus_list(regime_weights, sector_team_map):
    """Full pipeline: factor profiles → blended scores → top-N per team."""
    profiles = {
        "NVDA": _profile("Technology", q=80, m=95, v=20, lv=25),  # momentum BULL favorite
        "META": _profile("Technology", q=85, m=80, v=40, lv=40),
        "AAPL": _profile("Technology", q=90, m=60, v=35, lv=55),
        "T":    _profile("Communication Services", q=40, m=30, v=70, lv=80),  # defensive
        "PFE":  _profile("Health Care", q=60, m=40, v=70, lv=70),
        "JPM":  _profile("Financials", q=75, m=70, v=80, lv=50),
    }
    scores = compute_focus_scores(profiles, "bull", regime_weights)
    fl = build_focus_list(scores, sector_team_map, per_team_size={
        "technology": 3, "healthcare": 2, "financials": 2,
    })
    # Tech team should have NVDA at the top (momentum BULL favorite)
    assert fl["technology"][0].ticker == "NVDA"
    # T (defensive) should rank below NVDA/META/AAPL in BULL
    if len(fl["technology"]) >= 4:
        non_t_scores = [e.focus_score for e in fl["technology"] if e.ticker != "T"]
        t_score = next((e.focus_score for e in fl["technology"] if e.ticker == "T"), None)
        if t_score is not None:
            assert t_score < max(non_t_scores)


def test_end_to_end_bear_regime_inverts_ranking(regime_weights, sector_team_map):
    """Same tickers, BEAR regime → defensive names rank higher."""
    profiles = {
        "MOMO": _profile("Technology", q=60, m=95, v=30, lv=20),
        "DEFO": _profile("Technology", q=70, m=20, v=60, lv=95),
    }
    bull_fl = build_focus_list(
        compute_focus_scores(profiles, "bull", regime_weights),
        sector_team_map,
    )
    bear_fl = build_focus_list(
        compute_focus_scores(profiles, "bear", regime_weights),
        sector_team_map,
    )
    assert bull_fl["technology"][0].ticker == "MOMO"
    assert bear_fl["technology"][0].ticker == "DEFO"


# ── summarize_focus_list ────────────────────────────────────────────────────


def test_summarize_focus_list_shape():
    """Summary surfaces n / top_3 / stance_mix per team."""
    fl = {
        "technology": [
            FocusListEntry(ticker="NVDA", sector="Technology", team_id="technology",
                           focus_score=80, stance="momentum",
                           rank_in_sector=1, rank_in_team=1),
            FocusListEntry(ticker="MSFT", sector="Technology", team_id="technology",
                           focus_score=75, stance="quality",
                           rank_in_sector=2, rank_in_team=2),
            FocusListEntry(ticker="AAPL", sector="Technology", team_id="technology",
                           focus_score=72, stance="quality",
                           rank_in_sector=3, rank_in_team=3),
            FocusListEntry(ticker="ANET", sector="Technology", team_id="technology",
                           focus_score=70, stance="momentum",
                           rank_in_sector=4, rank_in_team=4),
        ],
        "healthcare": [],
    }
    s = summarize_focus_list(fl)
    assert s["technology"]["n"] == 4
    assert s["technology"]["top_3"] == ["NVDA", "MSFT", "AAPL"]
    assert s["technology"]["stance_mix"] == {"momentum": 2, "quality": 2}
    assert s["healthcare"]["n"] == 0
    assert s["healthcare"]["top_3"] == []


# ── build_focus_list_audit_lookup (alpha-engine-config-I2515) ───────────────
#
# Shared projection extracted from graph.research_graph's legacy fallback
# branch so BOTH callers (the graph's agent-state-absent fallback AND the
# standalone Scanner path) build the flat scanner_evaluations audit-lookup
# from exactly one implementation.


def test_build_focus_list_audit_lookup_passed_and_near_miss(regime_weights, sector_team_map):
    # 3 Tech tickers, team size 2 → NVDA/META pass, AAPL is a near miss.
    profiles = {
        "NVDA": _profile("Technology", q=60, m=95, v=30, lv=20),
        "META": _profile("Technology", q=70, m=80, v=40, lv=30),
        "AAPL": _profile("Technology", q=80, m=50, v=60, lv=50),
    }
    scores = compute_focus_scores(profiles, "bull", regime_weights)
    fl = build_focus_list(scores, sector_team_map, per_team_size={"technology": 2})

    lookup = build_focus_list_audit_lookup(scores, fl, sector_team_map)

    assert lookup["NVDA"]["focus_list_passed"] == 1
    assert lookup["NVDA"]["focus_rank_in_team"] == 1
    assert lookup["NVDA"]["focus_team_id"] == "technology"
    assert lookup["META"]["focus_list_passed"] == 1
    # AAPL scored but cut by the team-size-2 cap → near-miss row.
    assert lookup["AAPL"]["focus_list_passed"] == 0
    assert lookup["AAPL"]["focus_rank_in_team"] is None
    assert lookup["AAPL"]["focus_rank_in_sector"] is None
    assert lookup["AAPL"]["focus_team_id"] == "technology"
    # Pure-quant projection never attributes an agent — always 0/None.
    for row in lookup.values():
        assert row["agent_override"] == 0
        assert row["override_team_id"] is None


def test_build_focus_list_audit_lookup_drops_unmapped_sector(regime_weights):
    """A ticker whose sector has no team mapping is dropped from the
    near-miss branch rather than synthesizing a fake team_id."""
    profiles = {
        "ORPHAN": _profile("Aerospace & Crypto Hybrids", q=80, m=80, v=80, lv=80),
    }
    scores = compute_focus_scores(profiles, "bull", regime_weights)
    fl = build_focus_list(scores, sector_team_map={})  # no team mapping at all
    lookup = build_focus_list_audit_lookup(scores, fl, sector_team_map={})
    assert "ORPHAN" not in lookup


# ── build_pure_quant_focus_lookup (alpha-engine-config-I2515) ───────────────
#
# The standalone Scanner path's entry point — no agent/graph run backs it,
# so this IS the only focus-list audit path in that Lambda (not a fallback).


class TestBuildPureQuantFocusLookup:
    def test_empty_when_factor_blend_disabled(self):
        with patch("config.FACTOR_BLEND_ENABLED", False):
            result = build_pure_quant_focus_lookup(market_regime="bull")
        assert result == {}

    def test_empty_when_factor_profiles_unreadable(self):
        with patch("config.FACTOR_BLEND_ENABLED", True), \
             patch("scoring.factor_scoring.read_factor_profiles_from_s3",
                    return_value=None):
            result = build_pure_quant_focus_lookup(
                market_regime="bull", run_date="2026-06-06",
            )
        assert result == {}

    def test_populates_lookup_from_factor_profiles(self):
        profiles = {
            "NVDA": {"sector": "Technology", "quality_score": 70.0,
                     "momentum_score": 95.0, "value_score": 20.0,
                     "low_vol_score": 25.0},
            "MSFT": {"sector": "Technology", "quality_score": 90.0,
                     "momentum_score": 60.0, "value_score": 40.0,
                     "low_vol_score": 55.0},
        }
        with patch("config.FACTOR_BLEND_ENABLED", True), \
             patch("scoring.factor_scoring.read_factor_profiles_from_s3",
                   return_value=profiles):
            result = build_pure_quant_focus_lookup(
                market_regime="bull", run_date="2026-06-06",
            )
        assert "NVDA" in result and "MSFT" in result
        assert result["NVDA"]["focus_list_passed"] == 1
        assert result["NVDA"]["agent_override"] == 0
        assert result["NVDA"]["override_team_id"] is None
        # NVDA (momentum=95) outranks MSFT (momentum=60) in a BULL regime.
        assert result["NVDA"]["focus_rank_in_team"] == 1

    def test_reads_dated_profile_key_not_latest(self):
        """run_date is threaded through to read_factor_profiles_from_s3 so
        the Scanner path reads THIS cycle's freshly-written profile rather
        than an arbitrary latest.json (which could be a stale prior week
        if this Lambda invocation races another writer)."""
        captured = {}

        def fake_read(run_date=None, bucket=None):
            captured["run_date"] = run_date
            captured["bucket"] = bucket
            return None

        with patch("config.FACTOR_BLEND_ENABLED", True), \
             patch("scoring.factor_scoring.read_factor_profiles_from_s3",
                   side_effect=fake_read):
            build_pure_quant_focus_lookup(
                market_regime="bull", run_date="2026-06-06", bucket="test-bucket",
            )
        assert captured["run_date"] == "2026-06-06"
        assert captured["bucket"] == "test-bucket"
