"""Contract tests for SCANNER_CONTRACT.md.

The scanner's output contract used to exist only as the emergent sum of four
code sites, so a ranking cutover on 2026-07-22 left three separate labels
asserting a ranking the live path had stopped using and nothing failed for four
weeks (alpha-engine-config-I7808 / I7809). These tests are what makes the
contract enforceable rather than aspirational: each one fails on a specific way
the code and the document can drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from data import scanner_specs
from data.scanner_specs import (
    LIVE_CHAMPION,
    RETIRED_SPEC_NAMES,
    SCANNER_SPECS,
    ScannerSpec,
    assert_registry_coherent,
    challenger_specs,
    live_champion_spec,
)
from scoring.leaderboard_producers import (
    _VACUITY_OVERLAP_THRESHOLD,
    _champion_scanner_day,
    _membership_overlap,
    _vacuous_membership_collisions,
)
from scoring.leaderboard_scoring import SpecDay, SpecHistory
from scoring.universe_membership import (
    ATTRACTIVENESS_FEED_TOP_N,
    TECH_SCORE_CUT_PREFIX,
    build_universe_membership,
    momentum_path_tech_scores,
    tech_scores_from_eval_log,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "SCANNER_CONTRACT.md"


# ── §3 the register and the live path cannot disagree ────────────────────────


def test_registry_is_coherent_at_import():
    assert_registry_coherent()


def test_exactly_one_champion_and_live_champion_names_it():
    champions = [s for s in SCANNER_SPECS.values() if s.kind == "champion"]
    assert len(champions) == 1
    assert champions[0].name == LIVE_CHAMPION
    assert live_champion_spec() is champions[0]


def test_no_challenger_shares_the_champions_ranking_function():
    """The defect verbatim: momentum_sleeve was registered as a challenger
    while the live path already applied it, so the leaderboard scored the
    champion against itself for four weeks (alpha-engine-config-I7808)."""
    champion = live_champion_spec()
    assert champion.rank is not None
    for spec in challenger_specs():
        assert spec.rank is not champion.rank, spec.name


def test_registry_refuses_a_challenger_that_clones_the_champion(monkeypatch):
    champion = live_champion_spec()
    clone = ScannerSpec(
        name="clone",
        kind="challenger",
        version="v1",
        description="the champion under another name",
        rank=champion.rank,
    )
    monkeypatch.setitem(SCANNER_SPECS, "clone", clone)
    with pytest.raises(ValueError, match="vacuous by construction"):
        assert_registry_coherent()


def test_orchestrator_resolves_the_ranker_through_the_register():
    """A direct ``_rank_*`` import in the live path is how the register and the
    live ranking drifted apart. The indirection is the fix, so it is asserted
    at the source level — a future edit that reintroduces the import fails
    here rather than four weeks later on a leaderboard."""
    src = (REPO_ROOT / "data" / "scanner_orchestrator.py").read_text()
    assert "live_champion_spec" in src
    offenders = re.findall(r"from data\.scanner_specs import[^\n]*\b(_rank_\w+)", src)
    assert not offenders, f"live path imports a ranker directly: {offenders}"


def test_retired_names_are_not_also_live():
    assert not set(RETIRED_SPEC_NAMES) & set(SCANNER_SPECS)
    assert "tech_score_momentum" in RETIRED_SPEC_NAMES


# ── §3 the displaced incumbent is a real, scored arm ─────────────────────────


def _eval_log() -> list[dict]:
    return [
        {"ticker": "AAA", "tech_score": 90.0, "scan_path": "momentum", "liquidity_pass": 1},
        {"ticker": "BBB", "tech_score": 80.0, "scan_path": "momentum", "liquidity_pass": 1},
        {"ticker": "CCC", "tech_score": 70.0, "scan_path": "momentum", "liquidity_pass": 1},
        # cleared liquidity but NOT the momentum path's own gates
        {"ticker": "DDD", "tech_score": 99.0, "liquidity_pass": 1, "filter_fail_reason": "below_thresholds"},
        {"ticker": "EEE", "tech_score": 60.0, "liquidity_pass": 0, "filter_fail_reason": "liquidity"},
    ]


def test_tech_score_arm_ranks_the_momentum_path_only():
    """DDD has the highest tech_score in the universe and is excluded: the
    incumbent rule is the gates PLUS the ranking, not the ranking alone."""
    picks = scanner_specs._rank_tech_score(_eval_log(), None, {"momentum_top_n": 3})
    assert picks == ["AAA", "BBB", "CCC"]


def test_tech_score_arm_is_deterministic_under_ties():
    log = [
        {"ticker": "ZZZ", "tech_score": 50.0, "scan_path": "momentum"},
        {"ticker": "AAA", "tech_score": 50.0, "scan_path": "momentum"},
    ]
    assert scanner_specs._rank_tech_score(log, None, {"momentum_top_n": 2}) == ["AAA", "ZZZ"]
    assert scanner_specs._rank_tech_score(list(reversed(log)), None, {"momentum_top_n": 2}) == ["AAA", "ZZZ"]


# ── §2 every cut's basis matches its producer ────────────────────────────────

EXPECTED_BASES = {
    "scanner_gate_baseline_60": "scanner_champion_rank",
    "scanner_candidates": "scanner_champion_rank",
    "attractiveness_top_20": "attractiveness_rank",
    "attractiveness_top_25": "attractiveness_rank",
    "attractiveness_top_60": "attractiveness_rank",
    f"{TECH_SCORE_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}": "tech_score_rank",
    "scanner_top_20": "tech_score_rank",
}


def _membership(n: int = 80) -> dict:
    tickers = [f"T{i:03d}" for i in range(n)]
    scanner_cut = tickers[:60]
    attractiveness = {t: float(n - i) for i, t in enumerate(tickers)}
    eval_log = [
        {"ticker": t, "tech_score": float(i), "scan_path": "momentum", "liquidity_pass": 1}
        for i, t in enumerate(tickers)
    ]
    return build_universe_membership(
        "2026-08-20",
        scanner_cut,
        attractiveness,
        tech_scores=tech_scores_from_eval_log(eval_log),
        gate_eligible_tech_scores=momentum_path_tech_scores(eval_log),
    )


def test_every_cut_basis_matches_its_producer():
    cuts = _membership()["cuts"]
    for name, basis in EXPECTED_BASES.items():
        assert name in cuts, f"contract declares {name} but it was not emitted"
        assert cuts[name]["basis"] == basis, name


def test_no_cut_claims_a_basis_the_contract_does_not_declare():
    """The failure this closes: `scanner_gate_baseline_60` carried
    basis=`scanner_gate` and a docstring calling it a tech_score ranking, while
    resolving to the momentum sleeve. An undeclared basis is now a test
    failure, so a label can no longer outlive the thing it labels."""
    declared = set(EXPECTED_BASES.values()) | {
        "attractiveness_rank_mom121",
        "attractiveness_rank_momzero",
    }
    for name, cut in _membership()["cuts"].items():
        assert cut["basis"] in declared, f"{name} carries undeclared basis {cut['basis']!r}"


def test_tech_score_cut_is_the_universe_not_the_champions_cut():
    """`scanner_top_20` ranks WITHIN the champion's 60; `tech_score_top_60`
    ranks over the whole momentum-path-eligible universe. Conflating the two is
    what made "the tech score top 60" name nothing real."""
    cuts = _membership()["cuts"]
    ts_cut = set(cuts[f"{TECH_SCORE_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"]["tickers"])
    live_cut = set(cuts["scanner_gate_baseline_60"]["tickers"])
    assert ts_cut - live_cut, "tech_score cut is a subset of the live cut — it did not rank the universe"
    assert set(cuts["scanner_top_20"]["tickers"]) <= live_cut


def test_live_cut_names_the_arm_that_ranked_it():
    cut = _membership()["cuts"]["scanner_gate_baseline_60"]
    assert cut["ranked_by"] == LIVE_CHAMPION
    # It feeds nothing: the sector teams read the champion CUT, not this
    # candidate-generation arm's artifact (alpha-engine-config-I7823).
    assert cut["feeds"] == []


def test_the_feed_cut_declares_its_consumers():
    cuts = _membership()["cuts"]
    assert cuts["attractiveness_top_60"]["feeds"] == [
        "sector_teams", "rag_corpus_scope", "thinktank_window",
    ]
    assert cuts["attractiveness_top_20"]["feeds"] == ["predictor_universe"]


def test_only_count_matched_arms_may_hold_the_feed():
    """The pointer is writable by an automated promotion engine, so the set of
    values it may take is closed and both members are 60 wide."""
    from scoring.universe_membership import DEFAULT_CUT_CHAMPION, PROMOTABLE_CUTS

    assert DEFAULT_CUT_CHAMPION in PROMOTABLE_CUTS
    assert set(PROMOTABLE_CUTS) == {"attractiveness_top_60", "tech_score_top_60"}
    assert all(name.endswith(f"_{ATTRACTIVENESS_FEED_TOP_N}") for name in PROMOTABLE_CUTS)


def test_tech_score_cut_flags_when_it_equals_the_live_cut():
    """The documented degrade — loadings unavailable, live cut falls back to
    the incumbent ordering — is annotated rather than raised on."""
    tickers = [f"T{i:03d}" for i in range(60)]
    eval_log = [
        {"ticker": t, "tech_score": float(60 - i), "scan_path": "momentum", "liquidity_pass": 1}
        for i, t in enumerate(tickers)
    ]
    m = build_universe_membership(
        "2026-08-20",
        tickers,
        {t: float(60 - i) for i, t in enumerate(tickers)},
        gate_eligible_tech_scores=momentum_path_tech_scores(eval_log),
    )
    assert m["cuts"][f"{TECH_SCORE_CUT_PREFIX}{ATTRACTIVENESS_FEED_TOP_N}"]["equals_live_cut"] is True


# ── §4 the champion's list order IS its ranking ──────────────────────────────


def test_champion_day_keeps_the_artifacts_order():
    """The live list is the champion arm's own descending ranking. Re-sorting
    it by tech_score — a rival arm's signal — made the champion's rank-IC a
    correlation against another arm's ordering (I7645 → I7808)."""
    doc = {
        "scanner_tickers": ["CCC", "AAA", "BBB"],
        "scanner_eval_log": [
            {"ticker": "AAA", "tech_score": 90.0},
            {"ticker": "BBB", "tech_score": 80.0},
            {"ticker": "CCC", "tech_score": 10.0},
        ],
    }
    day = _champion_scanner_day(doc)
    assert day.ranked == ["CCC", "AAA", "BBB"]
    assert day.rank_ordered is True


def test_champion_day_with_no_tickers_is_unranked():
    day = _champion_scanner_day({"scanner_tickers": []})
    assert day.ranked == []
    assert day.rank_ordered is False


# ── §3 near-identity is vacuous too ──────────────────────────────────────────


def _history(name: str, kind: str, picks: list[str]) -> SpecHistory:
    return SpecHistory(name=name, kind=kind, by_date={"2026-08-19": SpecDay(ranked=picks)})


def test_near_identical_arms_are_reported_vacuous():
    """58 of 60 shared — the state the exact-equality guard passed as a live
    comparison on 2026-08-19 while both arms were the same function."""
    champ = [f"T{i:03d}" for i in range(60)]
    near = champ[:58] + ["X01", "X02"]
    found = _vacuous_membership_collisions(
        _history("momentum_sleeve", "champion", champ),
        [_history("clone", "challenger", near)],
    )
    assert len(found) == 1
    assert found[0]["identical"] is False
    assert found[0]["n_shared"] == 58
    assert found[0]["overlap"] >= _VACUITY_OVERLAP_THRESHOLD


def test_genuinely_different_arms_are_not_reported():
    champ = [f"T{i:03d}" for i in range(60)]
    other = [f"U{i:03d}" for i in range(60)]
    assert _vacuous_membership_collisions(
        _history("momentum_sleeve", "champion", champ),
        [_history("mom_12_1_sleeve", "challenger", other)],
    ) == []


def test_a_narrow_subset_arm_is_not_a_clone():
    """Overlap divides by the WIDER set, so a 30-name arm inside a 60-name
    champion reads as a different arm rather than a perfect clone."""
    champ = [f"T{i:03d}" for i in range(60)]
    assert _membership_overlap(set(champ), set(champ[:30])) == pytest.approx(0.5)
    assert _vacuous_membership_collisions(
        _history("momentum_sleeve", "champion", champ),
        [_history("narrow", "challenger", champ[:30])],
    ) == []


# ── the document itself stays in step ────────────────────────────────────────


def test_contract_document_exists_and_names_every_live_spec():
    text = CONTRACT.read_text()
    for name in SCANNER_SPECS:
        assert name in text, f"SCANNER_CONTRACT.md does not describe arm {name!r}"
    for name in EXPECTED_BASES:
        assert name in text, f"SCANNER_CONTRACT.md does not describe cut {name!r}"
