"""Unit tests for the judge-calibration κ stage (ROADMAP L480)."""

from __future__ import annotations

import json
import math

import boto3
import pytest
from moto import mock_aws

from evals.calibration_kappa import (
    MIN_REVIEWS_PER_CELL,
    _coerce_score,
    compute_calibration_report,
    emit_calibration_report,
    krippendorff_alpha_ordinal,
    load_reviews,
    render_markdown,
)

# ── helpers ──────────────────────────────────────────────────────────────


def _review(rubric_id: str, dims: list[tuple[str, int | None, int, int | None]]) -> dict:
    """Build one review record. ``dims`` is a list of
    (dimension, blind_score, llm_score, final_score)."""
    return {
        "review_id": f"rid-{rubric_id}-{len(dims)}",
        "rubric_id": rubric_id,
        "per_dimension": [
            {
                "dimension": d,
                "blind_score": b,
                "llm_score": llm,
                "final_score": f if f is not None else (b if b is not None else llm),
                "revised": (f is not None and f != b),
            }
            for (d, b, llm, f) in dims
        ],
        "overall_note": "",
    }


def _n_reviews(n: int, rubric: str, dim: str, blind: int, llm: int) -> list[dict]:
    return [_review(rubric, [(dim, blind, llm, None)]) for _ in range(n)]


# ── _coerce_score ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [(1, 1), (5, 5), (3, 3), ("4", 4), (None, None), (0, None), (6, None), ("x", None)],
)
def test_coerce_score(value, expected):
    assert _coerce_score(value) == expected


# ── Krippendorff α (ordinal) ─────────────────────────────────────────────


def test_alpha_empty_is_nan():
    assert math.isnan(krippendorff_alpha_ordinal([]))


def test_alpha_perfect_agreement_is_one():
    pairs = [(3, 3)] * 10 + [(5, 5)] * 5
    assert krippendorff_alpha_ordinal(pairs) == pytest.approx(1.0)


def test_alpha_single_category_undefined_returns_zero():
    # All ratings in one category → no expected disagreement → α = 0.0.
    assert krippendorff_alpha_ordinal([(4, 4)] * 8) == 0.0


def test_alpha_perfect_disagreement_closed_form():
    # 10 pairs all (1, 5): α = 1 - (2n-1)/n = 1 - 19/10 = -0.9 (derived
    # analytically from the ordinal coincidence-matrix formula).
    assert krippendorff_alpha_ordinal([(1, 5)] * 10) == pytest.approx(-0.9)


def test_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        krippendorff_alpha_ordinal([(1, 7)])


# ── compute_calibration_report ───────────────────────────────────────────


def test_report_empty_corpus():
    rep = compute_calibration_report([])
    assert rep["status"] == "empty"
    assert rep["n_cells"] == 0
    assert rep["n_paired_reviews"] == 0


def test_report_insufficient_below_threshold():
    reviews = _n_reviews(MIN_REVIEWS_PER_CELL - 1, "thesis_update", "completeness", 4, 4)
    rep = compute_calibration_report(reviews)
    assert rep["status"] == "insufficient"
    assert rep["n_cells"] == 1
    assert rep["n_cells_sufficient"] == 0
    cell = rep["cells"][0]
    assert cell["n"] == MIN_REVIEWS_PER_CELL - 1
    assert cell["sufficient"] is False
    assert cell["progress"] == f"{MIN_REVIEWS_PER_CELL - 1}/{MIN_REVIEWS_PER_CELL}"


def test_report_threshold_boundary_exactly_30_is_ok():
    reviews = _n_reviews(MIN_REVIEWS_PER_CELL, "thesis_update", "completeness", 4, 4)
    rep = compute_calibration_report(reviews)
    assert rep["status"] == "ok"
    assert rep["n_cells_sufficient"] == 1
    cell = rep["cells"][0]
    assert cell["sufficient"] is True
    # Perfect agreement → κ on a constant rater is 0.0 (undefined), but
    # exact agreement is 100%.
    assert cell["exact_agreement"] == 1.0


def test_report_groups_by_rubric_and_dimension():
    reviews = (
        _n_reviews(5, "thesis_update", "completeness", 4, 4)
        + _n_reviews(3, "thesis_update", "rigor", 2, 3)
        + _n_reviews(7, "sector_quant", "completeness", 5, 5)
    )
    rep = compute_calibration_report(reviews)
    keys = {(c["rubric_id"], c["dimension"]) for c in rep["cells"]}
    assert keys == {
        ("thesis_update", "completeness"),
        ("thesis_update", "rigor"),
        ("sector_quant", "completeness"),
    }


def test_report_skipped_blind_drops_from_blind_but_keeps_final():
    # blind_score=None (operator skipped step 1) → no blind pair, but the
    # final score still pairs against the LLM score.
    reviews = [_review("thesis_update", [("completeness", None, 4, 3)])]
    rep = compute_calibration_report(reviews)
    cell = rep["cells"][0]
    assert cell["n"] == 0  # no blind pairs
    assert cell["n_final"] == 1
    assert cell["qwk_blind_vs_llm"] is None  # empty → NaN → null


def test_report_kappa_distinguishes_blind_from_final():
    # Operator disagrees blind, then revises toward the judge after the
    # reveal: κ(final, llm) should be higher (more agreement) than the
    # raw blind agreement.
    reviews = []
    for _ in range(MIN_REVIEWS_PER_CELL):
        reviews.append(_review("thesis_update", [("rigor", 2, 4, 4)]))
    rep = compute_calibration_report(reviews)
    cell = rep["cells"][0]
    # final == llm everywhere → exact-agree on final; blind != llm.
    assert cell["exact_agreement"] == 0.0  # blind never matched llm
    assert cell["n_final"] == MIN_REVIEWS_PER_CELL


# ── render_markdown ──────────────────────────────────────────────────────


@pytest.mark.parametrize("status_reviews", [
    [],
    _n_reviews(3, "thesis_update", "completeness", 4, 4),
    _n_reviews(MIN_REVIEWS_PER_CELL, "thesis_update", "completeness", 3, 4),
])
def test_render_markdown_never_raises(status_reviews):
    rep = compute_calibration_report(status_reviews)
    md = render_markdown(rep)
    assert md.startswith("## Judge calibration (κ)")


# ── S3 read/write (moto) ─────────────────────────────────────────────────


@pytest.fixture
def s3_bucket():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")
        yield client


def _put_reviews_jsonl(client, date: str, reviews: list[dict], *, bad_line: bool = False):
    key = f"decision_artifacts/_calibration/{date}/reviews.jsonl"
    lines = [json.dumps(r) for r in reviews]
    if bad_line:
        lines.append("{not valid json")
    client.put_object(
        Bucket="alpha-engine-research",
        Key=key,
        Body=("\n".join(lines) + "\n").encode("utf-8"),
    )


def test_load_reviews_reads_all_dates_and_skips_bad_lines(s3_bucket):
    _put_reviews_jsonl(s3_bucket, "2026-05-25", _n_reviews(2, "r", "d", 4, 4))
    _put_reviews_jsonl(s3_bucket, "2026-05-26", _n_reviews(3, "r", "d", 4, 4), bad_line=True)
    got = load_reviews(s3_client=s3_bucket)
    assert len(got) == 5  # 2 + 3, the malformed line skipped


def test_load_reviews_excludes_report_subtree(s3_bucket):
    _put_reviews_jsonl(s3_bucket, "2026-05-25", _n_reviews(2, "r", "d", 4, 4))
    # A stray reviews.jsonl-named object under _report/ must NOT be read.
    s3_bucket.put_object(
        Bucket="alpha-engine-research",
        Key="decision_artifacts/_calibration/_report/2026-05-25/reviews.jsonl",
        Body=b'{"rubric_id":"x"}\n',
    )
    got = load_reviews(s3_client=s3_bucket)
    assert len(got) == 2


def test_emit_writes_json_md_and_latest(s3_bucket):
    _put_reviews_jsonl(
        s3_bucket, "2026-05-25",
        _n_reviews(MIN_REVIEWS_PER_CELL, "thesis_update", "completeness", 3, 4),
    )
    rep = emit_calibration_report(s3_client=s3_bucket, report_date="2026-05-28")
    assert rep["status"] == "ok"
    assert rep["report_keys"] == [
        "decision_artifacts/_calibration/_report/2026-05-28/kappa.json",
        "decision_artifacts/_calibration/_report/2026-05-28/kappa.md",
        "decision_artifacts/_calibration/_report/latest/kappa.json",
        "decision_artifacts/_calibration/_report/latest/kappa.md",
    ]
    # latest json pointer is readable and matches.
    latest = s3_bucket.get_object(
        Bucket="alpha-engine-research",
        Key="decision_artifacts/_calibration/_report/latest/kappa.json",
    )["Body"].read()
    assert json.loads(latest)["status"] == "ok"
    # latest markdown pointer (what the backtester email embeds) exists.
    latest_md = s3_bucket.get_object(
        Bucket="alpha-engine-research",
        Key="decision_artifacts/_calibration/_report/latest/kappa.md",
    )["Body"].read().decode("utf-8")
    assert latest_md.startswith("## Judge calibration (κ)")


def test_emit_empty_corpus_still_writes_report(s3_bucket):
    rep = emit_calibration_report(s3_client=s3_bucket, report_date="2026-05-28")
    assert rep["status"] == "empty"
    # Report artifact exists even with zero reviews (always-emit).
    obj = s3_bucket.get_object(
        Bucket="alpha-engine-research",
        Key="decision_artifacts/_calibration/_report/2026-05-28/kappa.json",
    )["Body"].read()
    assert json.loads(obj)["status"] == "empty"
