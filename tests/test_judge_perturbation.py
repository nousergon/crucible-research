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
    _contradict_stance,
    _strip_input_groundedness,
    _unearned_material_change,
    _vacuous_moat,
    emit_perturbation_report,
    format_scorecard,
    run_perturbation_battery,
)

_NUM = __import__("re").compile(r"\d")


def _thesis() -> dict:
    return copy.deepcopy(REFERENCE_FIXTURES["eval_rubric_thinktank_thesis"]["agent_output"])


def _theme() -> dict:
    return copy.deepcopy(REFERENCE_FIXTURES["eval_rubric_thinktank_theme"]["agent_output"])


# ── Corruption determinism ─────────────────────────────────────────────────


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


_ONE_CASE = [Corruption("t", "eval_rubric_thinktank_thesis",
                        "input_groundedness", _strip_input_groundedness)]


class TestBatteryLogic:
    def test_caught_when_targeted_dimension_drops(self):
        fake = _seq_judge(
            {"input_groundedness": 5, "moat_and_business_quality": 4},   # reference
            {"input_groundedness": 2, "moat_and_business_quality": 4},   # corrupted
        )
        rep = run_perturbation_battery(corruptions=_ONE_CASE, judge_fn=fake)
        assert rep["cases"][0]["caught"] is True
        assert rep["cases"][0]["drop"] == 3
        assert rep["caught_rate"] == 1.0

    def test_not_caught_when_judge_insensitive(self):
        fake = _seq_judge(
            {"input_groundedness": 4},   # reference
            {"input_groundedness": 4},   # corrupted — judge didn't notice
        )
        rep = run_perturbation_battery(corruptions=_ONE_CASE, judge_fn=fake)
        assert rep["cases"][0]["caught"] is False
        assert rep["cases"][0]["drop"] == 0
        assert rep["n_caught"] == 0

    def test_min_drop_threshold_respected(self):
        fake = _seq_judge(
            {"input_groundedness": 4},
            {"input_groundedness": 3},   # drop of 1
        )
        rep = run_perturbation_battery(
            corruptions=_ONE_CASE, judge_fn=fake, min_drop=2,
        )
        assert rep["cases"][0]["caught"] is False  # 1 < 2

    def test_reference_judged_once_per_rubric(self):
        two_same_rubric = [
            Corruption("a", "eval_rubric_thinktank_thesis", "input_groundedness",
                       _strip_input_groundedness),
            Corruption("b", "eval_rubric_thinktank_thesis", "moat_and_business_quality",
                       _vacuous_moat),
        ]
        fake = _seq_judge({"input_groundedness": 5, "moat_and_business_quality": 5})
        run_perturbation_battery(corruptions=two_same_rubric, judge_fn=fake)
        # 1 reference judging (cached) + 2 corrupted = 3, not 4
        assert fake.calls["n"] == 3

    def test_missing_dimension_yields_uncaught_none_drop(self):
        fake = _seq_judge(
            {"some_other_dim": 5},   # targeted dim absent from judge output
            {"some_other_dim": 5},
        )
        rep = run_perturbation_battery(corruptions=_ONE_CASE, judge_fn=fake)
        assert rep["cases"][0]["drop"] is None
        assert rep["cases"][0]["caught"] is False

    def test_scorecard_renders_caught_and_missed(self):
        fake = _seq_judge(
            {"input_groundedness": 5},
            {"input_groundedness": 1},
        )
        rep = run_perturbation_battery(corruptions=_ONE_CASE, judge_fn=fake)
        md = format_scorecard(rep)
        assert "Judge sensitivity" in md
        assert "1/1" in md
        assert "✅" in md


# ── Weekly sensitivity scorecard emit (Phase B, config#752) ────────────────


class _FakeS3:
    """Captures put_object calls without touching AWS."""

    def __init__(self):
        self.puts: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803
        self.puts[Key] = Body
        self.content_types[Key] = ContentType


_PRECOMPUTED_REPORT = {
    "judge_model": "claude-haiku-4-5-20251001",
    "n": 2,
    "n_caught": 1,
    "caught_rate": 0.5,
    "cases": [
        {"name": "a", "rubric": "eval_rubric_thinktank_thesis",
         "target_dimension": "input_groundedness", "ref_score": 5,
         "corrupted_score": 1, "drop": 4, "caught": True,
         "ref_mean": 4.5, "corrupted_mean": 3.0},
        {"name": "b", "rubric": "eval_rubric_thinktank_theme",
         "target_dimension": "churn_discipline", "ref_score": 4,
         "corrupted_score": 4, "drop": 0, "caught": False,
         "ref_mean": 4.0, "corrupted_mean": 4.0},
    ],
}


class TestEmitPerturbationReport:
    def test_writes_dated_and_latest_json_and_md(self):
        s3 = _FakeS3()
        out = emit_perturbation_report(
            report=copy.deepcopy(_PRECOMPUTED_REPORT),
            s3_client=s3, report_date="2026-07-11", bucket="test-bucket",
        )
        assert set(out["report_keys"]) == set(s3.puts)
        assert s3.puts.keys() >= {
            "decision_artifacts/_perturbation/_report/2026-07-11/sensitivity.json",
            "decision_artifacts/_perturbation/_report/2026-07-11/sensitivity.md",
            "decision_artifacts/_perturbation/_report/latest/sensitivity.json",
            "decision_artifacts/_perturbation/_report/latest/sensitivity.md",
        }
        # latest pointers mirror the dated copies byte-for-byte
        dated_md = s3.puts["decision_artifacts/_perturbation/_report/2026-07-11/sensitivity.md"]
        latest_md = s3.puts["decision_artifacts/_perturbation/_report/latest/sensitivity.md"]
        assert dated_md == latest_md
        assert b"Judge sensitivity" in latest_md

    def test_content_types_and_provenance(self):
        s3 = _FakeS3()
        out = emit_perturbation_report(
            report=copy.deepcopy(_PRECOMPUTED_REPORT),
            s3_client=s3, report_date="2026-07-11", bucket="test-bucket",
        )
        for k, ct in s3.content_types.items():
            assert ct == ("text/markdown" if k.endswith(".md") else "application/json")
        assert out["report_date"] == "2026-07-11"
        assert "generated_at" in out  # stamped for drift tracking

    def test_runs_battery_when_no_report_injected(self):
        # End-to-end: emit runs the battery itself with an injected fake judge
        # (no live LLM), then writes the scorecard.
        s3 = _FakeS3()
        fake = _seq_judge(
            {"input_groundedness": 5, "moat_and_business_quality": 5},   # reference
            {"input_groundedness": 1, "moat_and_business_quality": 5},   # corrupted
        )
        # Restrict to a single corruption via monkeypatching CORRUPTIONS is
        # unnecessary — the full battery runs fine offline with the fake judge.
        out = emit_perturbation_report(
            s3_client=s3, report_date="2026-07-11", bucket="test-bucket",
            judge_fn=fake,
        )
        assert out["n"] == len(CORRUPTIONS)
        assert len(s3.puts) == 4
