"""Unit tests for the synthetic-perturbation judge validator harness
(ROADMAP L480, 2026-05-29).

Two layers, both runnable in regular (mocked, no-API-key) CI:
  1. Corruption determinism — each corruption degrades exactly its
     targeted construct (digits gone, ranking reversed, scores flattened,
     reasoning collapsed) and is a pure function over a deep copy.
  2. Battery logic — with an injected fake judge, `caught` / `drop` /
     `caught_rate` are computed correctly and the reference is judged
     once per rubric (cached).

The live `reference > corrupted` assertion against the real judge LLM
lives in tests/live_smoke/judge_perturbation_smoke.py (paths-filtered,
needs ANTHROPIC_API_KEY + the gitignored rubric prompts).
"""

from __future__ import annotations

import copy

from evals.perturbation import (
    CORRUPTIONS,
    REFERENCE_FIXTURES,
    Corruption,
    _break_anchor_fidelity,
    _break_ranking_coherence,
    _contradict_stance,
    _flatten_reasoning_depth,
    _flatten_signal_calibration,
    _gut_output_completeness,
    _misalign_evidence,
    _strip_citation_grounding,
    _strip_input_groundedness,
    _strip_numerical_grounding,
    _unearned_material_change,
    _vacuous_moat,
    _verbosity_pad,
    format_scorecard,
    run_perturbation_battery,
)

_NUM = __import__("re").compile(r"\d")


def _quant() -> dict:
    return copy.deepcopy(REFERENCE_FIXTURES["eval_rubric_sector_quant"]["agent_output"])


def _qual() -> dict:
    return copy.deepcopy(REFERENCE_FIXTURES["eval_rubric_sector_qual"]["agent_output"])


def _thesis() -> dict:
    return copy.deepcopy(REFERENCE_FIXTURES["eval_rubric_thinktank_thesis"]["agent_output"])


def _theme() -> dict:
    return copy.deepcopy(REFERENCE_FIXTURES["eval_rubric_thinktank_theme"]["agent_output"])


# ── Corruption determinism ─────────────────────────────────────────────────


class TestQuantCorruptions:
    def test_strip_numerical_grounding_removes_all_digits(self):
        out = _strip_numerical_grounding(_quant())
        for p in out["ranked_picks"]:
            assert p["key_metrics"] == {}
            assert not _NUM.search(p["rationale"]), "digits survived the scrub"

    def test_break_ranking_coherence_inverts_score_vs_rank(self):
        ref = _quant()
        out = _break_ranking_coherence(_quant())
        # ticker order + rationales preserved...
        assert [p["ticker"] for p in out["ranked_picks"]] == \
            [p["ticker"] for p in ref["ranked_picks"]]
        # ...but scores now ASCEND down the list — the first-listed pick
        # (described as strongest) carries the lowest score.
        scores = [p["quant_score"] for p in out["ranked_picks"]]
        assert scores == sorted(scores), "scores should ascend down the list"
        assert scores[0] < scores[-1], "first-listed must now be the lowest"

    def test_flatten_signal_calibration_collapses_gradient(self):
        out = _flatten_signal_calibration(_quant())
        scores = {p["quant_score"] for p in out["ranked_picks"]}
        assert len(scores) == 1, "scores should be identical (no gradient)"

    def test_gut_output_completeness_drops_to_one_empty_pick(self):
        out = _gut_output_completeness(_quant())
        assert len(out["ranked_picks"]) == 1
        assert out["ranked_picks"][0]["rationale"] == ""

    def test_verbosity_pad_is_longer_but_still_ungrounded(self):
        ref = _quant()
        out = _verbosity_pad(_quant())
        for rp, op in zip(ref["ranked_picks"], out["ranked_picks"]):
            assert len(op["rationale"]) > len(rp["rationale"]), "should be longer"
            assert not _NUM.search(op["rationale"]), "still no real numbers"
            assert op["key_metrics"] == {}


class TestQualCorruptions:
    def test_strip_citation_grounding_removes_specific_facts(self):
        ref = _qual()
        out = _strip_citation_grounding(_qual())
        for ra, oa in zip(ref["assessments"], out["assessments"]):
            assert oa["bull_case"] != ra["bull_case"]
            assert not _NUM.search(oa["bull_case"])
            assert not _NUM.search(oa["bear_case"])

    def test_flatten_reasoning_depth_collapses_to_single_clause(self):
        out = _flatten_reasoning_depth(_qual())
        for a in out["assessments"]:
            assert len(a["bull_case"].split()) <= 5
            assert len(a["bear_case"].split()) <= 5

    def test_misalign_evidence_inflates_score_vs_thin_bull(self):
        ref = _qual()
        out = _misalign_evidence(_qual())
        for ra, oa in zip(ref["assessments"], out["assessments"]):
            assert oa["qual_score"] > ra["qual_score"]
            # bull weakened, bear (substantive) preserved → misalignment
            assert len(oa["bull_case"]) < len(ra["bull_case"])
            assert oa["bear_case"] == ra["bear_case"]


class TestThinktankThesisCorruptions:
    def test_strip_input_groundedness_removes_specific_references(self):
        ref = _thesis()
        out = _strip_input_groundedness(_thesis())
        for field in ("business_summary", "filings_review", "news_sentiment",
                      "valuation", "market_dynamics"):
            assert out[field] != ref[field]
            assert not _NUM.search(out[field]), f"{field} still cites a number"
        # untouched — only groundedness should degrade
        assert out["stance"] == ref["stance"]
        assert out["risks"] == ref["risks"]

    def test_vacuous_moat_replaces_with_marketing_language(self):
        ref = _thesis()
        out = _vacuous_moat(_thesis())
        assert out["moat"] != ref["moat"]
        assert not _NUM.search(out["moat"])
        # untouched
        assert out["business_summary"] == ref["business_summary"]

    def test_contradict_stance_flips_stance_but_keeps_bullish_body(self):
        ref = _thesis()
        out = _contradict_stance(_thesis())
        assert ref["stance"] == "attractive"
        assert out["stance"] == "avoid"
        # body (the bullish evidence) is untouched — the contradiction is
        # purely stance-vs-body, not a rewrite of the evidence itself.
        assert out["summary"] == ref["summary"]
        assert out["moat"] == ref["moat"]


class TestThinktankThemeCorruptions:
    def test_unearned_material_change_flags_true_for_a_restatement(self):
        ref = _theme()
        out = _unearned_material_change(_theme())
        assert ref["material_change"] is False
        assert out["material_change"] is True
        # narrative/drivers still read as a restatement, not an actual shift
        assert out["narrative"] == ref["narrative"]
        assert out["drivers"] == ref["drivers"]

    def test_break_anchor_fidelity_silently_flips_stance(self):
        ref = _theme()
        out = _break_anchor_fidelity(_theme())
        assert ref["stance"] == "overweight"
        assert out["stance"] == "underweight"
        # no divergence acknowledgment — change_summary stays empty
        assert out["change_summary"] == ""


def test_corruptions_do_not_mutate_the_shared_fixture():
    """Battery deep-copies before corrupting; confirm the module-level
    reference fixture is unchanged after running every corruption."""
    before = copy.deepcopy(REFERENCE_FIXTURES)
    for c in CORRUPTIONS:
        c.fn(copy.deepcopy(REFERENCE_FIXTURES[c.rubric]["agent_output"]))
    assert REFERENCE_FIXTURES == before


def test_every_corruption_targets_a_real_registered_rubric():
    for c in CORRUPTIONS:
        assert c.rubric in REFERENCE_FIXTURES, f"{c.name} → unknown rubric {c.rubric}"


# ── Battery logic (injected fake judge — no live LLM) ──────────────────────


def _seq_judge(*score_dicts):
    """Fake judge_fn returning the given dicts in call order."""
    calls = {"n": 0}

    def fake(_artifact, *, judge_model, api_key):
        d = score_dicts[min(calls["n"], len(score_dicts) - 1)]
        calls["n"] += 1
        return dict(d)

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


_ONE_QUANT = [Corruption("t", "eval_rubric_sector_quant",
                         "numerical_grounding", _strip_numerical_grounding)]


class TestBatteryLogic:
    def test_caught_when_targeted_dimension_drops(self):
        fake = _seq_judge(
            {"numerical_grounding": 5, "ranking_coherence": 4},   # reference
            {"numerical_grounding": 2, "ranking_coherence": 4},   # corrupted
        )
        rep = run_perturbation_battery(corruptions=_ONE_QUANT, judge_fn=fake)
        assert rep["cases"][0]["caught"] is True
        assert rep["cases"][0]["drop"] == 3
        assert rep["caught_rate"] == 1.0

    def test_not_caught_when_judge_insensitive(self):
        fake = _seq_judge(
            {"numerical_grounding": 4},   # reference
            {"numerical_grounding": 4},   # corrupted — judge didn't notice
        )
        rep = run_perturbation_battery(corruptions=_ONE_QUANT, judge_fn=fake)
        assert rep["cases"][0]["caught"] is False
        assert rep["cases"][0]["drop"] == 0
        assert rep["n_caught"] == 0

    def test_min_drop_threshold_respected(self):
        fake = _seq_judge(
            {"numerical_grounding": 4},
            {"numerical_grounding": 3},   # drop of 1
        )
        rep = run_perturbation_battery(
            corruptions=_ONE_QUANT, judge_fn=fake, min_drop=2,
        )
        assert rep["cases"][0]["caught"] is False  # 1 < 2

    def test_reference_judged_once_per_rubric(self):
        two_same_rubric = [
            Corruption("a", "eval_rubric_sector_quant", "numerical_grounding",
                       _strip_numerical_grounding),
            Corruption("b", "eval_rubric_sector_quant", "ranking_coherence",
                       _break_ranking_coherence),
        ]
        fake = _seq_judge({"numerical_grounding": 5, "ranking_coherence": 5})
        run_perturbation_battery(corruptions=two_same_rubric, judge_fn=fake)
        # 1 reference judging (cached) + 2 corrupted = 3, not 4
        assert fake.calls["n"] == 3

    def test_missing_dimension_yields_uncaught_none_drop(self):
        fake = _seq_judge(
            {"some_other_dim": 5},   # targeted dim absent from judge output
            {"some_other_dim": 5},
        )
        rep = run_perturbation_battery(corruptions=_ONE_QUANT, judge_fn=fake)
        assert rep["cases"][0]["drop"] is None
        assert rep["cases"][0]["caught"] is False

    def test_scorecard_renders_caught_and_missed(self):
        fake = _seq_judge(
            {"numerical_grounding": 5},
            {"numerical_grounding": 1},
        )
        rep = run_perturbation_battery(corruptions=_ONE_QUANT, judge_fn=fake)
        md = format_scorecard(rep)
        assert "Judge sensitivity" in md
        assert "1/1" in md
        assert "✅" in md
