"""Unit tests for ``evals.cross_validation`` (L83 manual judge cross-val)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evals.cross_validation import (
    DimensionAgreement,
    collect_rating_pairs,
    parse_judge_scores,
    parse_worksheet,
    quadratic_weighted_kappa,
    render_markdown_report,
    run_cross_validation,
    summarize_agreement,
)


# ── parse_worksheet ─────────────────────────────────────────────────────


def _write_worksheet(tmp_path: Path, body: str, name: str = "01_x_2026-05-06.md") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_parse_worksheet_extracts_filled_scores(tmp_path):
    path = _write_worksheet(tmp_path, """# Worksheet 01

## Your scores

Rubric family: `ic_cio`

- **decision_coherence**: 4  (1-5)
  - reasoning: tracks scores
- **rationale_quality**: 3
  - reasoning: mixed

## Rubric (anchors)

  5 — perfect
  3 — adequate
  1 — broken
""")
    scores = parse_worksheet(path)
    assert scores == {"decision_coherence": 4, "rationale_quality": 3}


def test_parse_worksheet_skips_unfilled_placeholder(tmp_path):
    path = _write_worksheet(tmp_path, """## Your scores

- **decision_coherence**: 4  (1-5)
- **rationale_quality**: SCORE_HERE  (1-5)
""")
    assert parse_worksheet(path) == {"decision_coherence": 4}


def test_parse_worksheet_ignores_rubric_section_numbers(tmp_path):
    # Important: rubric anchors contain lines like "  5 — perfect" or
    # "- **dimension_name**: ..." that must NOT be parsed as operator scores.
    path = _write_worksheet(tmp_path, """## Your scores

- **decision_coherence**: 4

## Rubric (anchors)

- **fake_dim**: 5
  5 — ADVANCE/REJECT pattern tracks
  1 — broken
""")
    assert parse_worksheet(path) == {"decision_coherence": 4}


def test_parse_worksheet_rejects_out_of_range_scores(tmp_path):
    path = _write_worksheet(tmp_path, """## Your scores

- **decision_coherence**: 7  (1-5)
""")
    with pytest.raises(ValueError, match="out of range"):
        parse_worksheet(path)


def test_parse_worksheet_empty_returns_empty_dict(tmp_path):
    path = _write_worksheet(tmp_path, "no scores here")
    assert parse_worksheet(path) == {}


# ── parse_judge_scores ──────────────────────────────────────────────────


def test_parse_judge_scores_multi_model(tmp_path):
    path = tmp_path / "judge.json"
    path.write_text(json.dumps({
        "claude-haiku-4-5": {
            "rubric_id": "eval_rubric_ic_cio",
            "rubric_version": "1.2.0",
            "dimension_scores": [
                {"dimension": "decision_coherence", "score": 4, "reasoning": "ok"},
                {"dimension": "rationale_quality", "score": 3, "reasoning": "ok"},
            ],
        },
        "claude-sonnet-4-6": {
            "dimension_scores": [
                {"dimension": "decision_coherence", "score": 5},
            ],
        },
    }))
    out = parse_judge_scores(path)
    assert out == {
        "claude-haiku-4-5": {"decision_coherence": 4, "rationale_quality": 3},
        "claude-sonnet-4-6": {"decision_coherence": 5},
    }


# ── quadratic_weighted_kappa ────────────────────────────────────────────


def test_kappa_perfect_agreement_is_one():
    pairs = [(1, 1), (3, 3), (5, 5), (4, 4), (2, 2)]
    assert quadratic_weighted_kappa(pairs) == pytest.approx(1.0)


def test_kappa_max_disagreement_is_negative():
    # All raters max-disagree across the 1-5 ordinal range.
    pairs = [(1, 5), (5, 1), (1, 5), (5, 1)]
    k = quadratic_weighted_kappa(pairs)
    assert k < 0  # worse than chance


def test_kappa_chance_agreement_near_zero():
    # When ratings are independent of each other, kappa should be near 0.
    # Construct a confusion that exactly equals the chance expectation:
    # marginals (10, 10) on a 2-class problem with all 4 cells = 5 ratings
    # is the independence case, giving kappa = 0.
    pairs = (
        [(1, 1)] * 5
        + [(1, 2)] * 5
        + [(2, 1)] * 5
        + [(2, 2)] * 5
    )
    k = quadratic_weighted_kappa(pairs, n_classes=2)
    assert abs(k) < 1e-9


def test_kappa_empty_returns_nan():
    assert math.isnan(quadratic_weighted_kappa([]))


def test_kappa_rejects_out_of_range_score():
    with pytest.raises(ValueError, match="outside"):
        quadratic_weighted_kappa([(0, 3)])


def test_kappa_constant_rater_returns_zero():
    # All judges give 3; humans vary. Expected disagreement is also 0
    # because one rater is constant. Kappa is undefined; we return 0.0.
    pairs = [(1, 3), (3, 3), (5, 3), (4, 3)]
    assert quadratic_weighted_kappa(pairs) == 0.0


# ── summarize_agreement ─────────────────────────────────────────────────


def _pair(*, agent_id="ic_cio", family="ic_cio", dim="decision_coherence",
          h=3, j=3, judge="claude-haiku-4-5", nn="01"):
    from evals.cross_validation import RatingPair
    return RatingPair(
        artifact_nn=nn, agent_id=agent_id, rubric_family=family,
        run_id="2026-05-06", dimension=dim, human_score=h, judge_score=j,
        judge_model=judge,
    )


def test_summarize_groups_by_cell():
    pairs = [
        _pair(dim="decision_coherence", h=4, j=4, nn="01"),
        _pair(dim="decision_coherence", h=3, j=4, nn="02"),
        _pair(dim="rationale_quality", h=2, j=4, nn="01"),
    ]
    agreements = summarize_agreement(pairs)
    assert len(agreements) == 2
    by_dim = {a.dimension: a for a in agreements}
    dc = by_dim["decision_coherence"]
    assert dc.n == 2
    assert dc.exact_match_rate == 0.5  # 1 of 2 exact
    assert dc.within_one_rate == 1.0   # both within 1
    assert dc.mean_abs_diff == 0.5
    rq = by_dim["rationale_quality"]
    assert rq.n == 1
    assert rq.exact_match_rate == 0.0
    assert rq.mean_abs_diff == 2.0


def test_summarize_separates_judge_models():
    # Same dimension, different judge models -> separate cells.
    pairs = [
        _pair(h=4, j=4, judge="claude-haiku-4-5"),
        _pair(h=4, j=5, judge="claude-sonnet-4-6"),
    ]
    agreements = summarize_agreement(pairs)
    assert len(agreements) == 2
    models = sorted(a.judge_model for a in agreements)
    assert models == ["claude-haiku-4-5", "claude-sonnet-4-6"]


# ── Bundle traversal (collect_rating_pairs) ─────────────────────────────


def _make_bundle(tmp_path: Path) -> Path:
    """Build a tiny rating bundle on disk: 2 artifacts, 1 with filled
    worksheet + 1 unfilled."""
    bundle = tmp_path / "bundle"
    (bundle / "worksheets").mkdir(parents=True)
    (bundle / ".judge_scores").mkdir()

    # Artifact 01: filled
    (bundle / "worksheets" / "01_x_2026-05-06.md").write_text("""## Your scores

- **decision_coherence**: 4  (1-5)
- **rationale_quality**: 3
""")
    (bundle / ".judge_scores" / "01_x_2026-05-06.json").write_text(json.dumps({
        "claude-haiku-4-5": {
            "dimension_scores": [
                {"dimension": "decision_coherence", "score": 4},
                {"dimension": "rationale_quality", "score": 5},
            ],
        },
    }))

    # Artifact 02: blank worksheet
    (bundle / "worksheets" / "02_y_2026-05-06.md").write_text("""## Your scores

- **decision_coherence**: SCORE_HERE
""")
    (bundle / ".judge_scores" / "02_y_2026-05-06.json").write_text(json.dumps({
        "claude-haiku-4-5": {
            "dimension_scores": [{"dimension": "decision_coherence", "score": 3}],
        },
    }))

    (bundle / "index.json").write_text(json.dumps({
        "seed": 1,
        "items": [
            {
                "nn": "01", "agent_id": "ic_cio", "rubric_family": "ic_cio",
                "run_id": "2026-05-06",
                "worksheet_path": "worksheets/01_x_2026-05-06.md",
                "judge_scores_path": ".judge_scores/01_x_2026-05-06.json",
            },
            {
                "nn": "02", "agent_id": "ic_cio", "rubric_family": "ic_cio",
                "run_id": "2026-05-06",
                "worksheet_path": "worksheets/02_y_2026-05-06.md",
                "judge_scores_path": ".judge_scores/02_y_2026-05-06.json",
            },
        ],
    }))
    return bundle


def test_collect_rating_pairs_skips_blank_worksheet(tmp_path):
    bundle = _make_bundle(tmp_path)
    pairs = collect_rating_pairs(bundle)
    # Artifact 01 has 2 dimensions × 1 judge model = 2 pairs.
    # Artifact 02 was blank — skipped.
    assert len(pairs) == 2
    assert {p.artifact_nn for p in pairs} == {"01"}
    dims = {p.dimension: p for p in pairs}
    assert dims["decision_coherence"].human_score == 4
    assert dims["decision_coherence"].judge_score == 4
    assert dims["rationale_quality"].human_score == 3
    assert dims["rationale_quality"].judge_score == 5


def test_collect_rating_pairs_warns_on_dimension_mismatch(tmp_path, caplog):
    """If worksheet has a dimension the judge eval doesn't, log + skip
    that pair rather than raising — the eval-judge framework may evolve
    rubrics over time and we want partial reports to still render."""
    bundle = _make_bundle(tmp_path)
    # Add a phantom dimension to worksheet 01
    ws = bundle / "worksheets" / "01_x_2026-05-06.md"
    ws.write_text(ws.read_text() + "\n- **phantom_dim**: 5\n")
    with caplog.at_level("WARNING"):
        pairs = collect_rating_pairs(bundle)
    assert any("phantom_dim" in r.message for r in caplog.records)
    # The 2 real dimensions still come through
    assert len(pairs) == 2


# ── End-to-end report rendering ────────────────────────────────────────


def test_run_cross_validation_renders_report(tmp_path):
    bundle = _make_bundle(tmp_path)
    report, agreements = run_cross_validation(bundle)
    assert "cross-validation report" in report.lower()
    assert "ic_cio" in report
    assert "decision_coherence" in report
    assert "rationale_quality" in report
    # Overall block present when there's data
    assert "Overall" in report
    # The 2 dimension cells we expect
    assert len(agreements) == 2


def test_run_cross_validation_empty_bundle(tmp_path):
    """No filled worksheets -> non-empty report explaining what's needed,
    not a crash."""
    bundle = tmp_path / "bundle"
    (bundle / "worksheets").mkdir(parents=True)
    (bundle / ".judge_scores").mkdir()
    (bundle / "worksheets" / "01_x.md").write_text("## Your scores\n\n- **dim**: SCORE_HERE\n")
    (bundle / ".judge_scores" / "01_x.json").write_text(json.dumps({
        "claude-haiku-4-5": {"dimension_scores": [{"dimension": "dim", "score": 3}]}
    }))
    (bundle / "index.json").write_text(json.dumps({"seed": 1, "items": [{
        "nn": "01", "agent_id": "x", "rubric_family": "x", "run_id": "r",
        "worksheet_path": "worksheets/01_x.md",
        "judge_scores_path": ".judge_scores/01_x.json",
    }]}))
    report, agreements = run_cross_validation(bundle)
    assert agreements == []
    assert "no filled worksheets" in report.lower()


def test_render_markdown_report_has_expected_columns(tmp_path):
    pairs = [
        _pair(h=4, j=4),
        _pair(dim="rationale_quality", h=2, j=4),
    ]
    agreements = summarize_agreement(pairs)
    md = render_markdown_report(
        agreements,
        bundle_dir=Path("/tmp/bundle-x"),
        n_artifacts_rated=1,
    )
    # The column header line is present
    assert "| dimension | judge_model | n | exact | ±1 | MAD | κ (quad) |" in md
    assert "decision_coherence" in md
    assert "rationale_quality" in md


def test_dimension_agreement_dataclass_roundtrip():
    a = DimensionAgreement(
        rubric_family="ic_cio",
        dimension="decision_coherence",
        judge_model="claude-haiku-4-5",
        n=3,
        exact_match_rate=0.67,
        within_one_rate=1.0,
        mean_abs_diff=0.33,
        quadratic_weighted_kappa=0.85,
    )
    assert a.score_pairs == []
    assert a.n == 3
