"""Judge-calibration κ stage — the metric leg of the SOTA judge-anchored
review (ROADMAP L480, formerly "Manual judge cross-validation sample").

The UI substrate (alpha-engine-dashboard PR #115, "Calibrate" tab on
page 8) captures a two-step, bias-isolated review per eval artifact:

    Step 1 — operator scores each rubric dimension BLIND (rubric + agent
             output only, judge verdict hidden) → the un-anchored anchor.
    Step 2 — judge score + reasoning revealed; operator may REVISE to a
             final score with a one-line note → residual after anchoring.

Each submitted review lands as one JSONL line under
``decision_artifacts/_calibration/{date}/reviews.jsonl`` carrying a
``per_dimension`` list of ``{dimension, blind_score, llm_score,
final_score, revised}``.

This module is the **metric leg**: it walks the whole reviews corpus,
groups paired scores by ``(rubric_id, dimension)`` cell, and computes —
per cell — the inter-rater reliability between operator and judge.

Headline metric: **quadratic-weighted Cohen's κ on (blind_score,
llm_score)**. The blind score is the whole point of the two-step design
— it is the operator's un-anchored judgment, so κ(blind, llm) is the
clean calibration estimate the anchored spot-check methodology (the
superseded PR #169 path in ``cross_validation.py``) could not produce.
A second κ(final, llm) measures the residual disagreement that survives
even after the operator sees the judge's reasoning — large residual on a
dimension is the signal to escalate it to pairwise / Bradley-Terry mode.

Companions per cell: raw exact-agreement, within-one rate, mean absolute
difference, Krippendorff's α (ordinal) — α is reported alongside κ
because it generalizes cleanly to >2 raters and missing data, so a
future multi-operator panel needs no metric swap.

Gate: a cell's κ is only reported as load-bearing once it has
``MIN_REVIEWS_PER_CELL`` (30) paired reviews — below that the κ estimate
is too noisy to act on. Per [[feedback_observational_stages_always_emit_artifact]]
the report is emitted on EVERY run regardless: an under-threshold run
emits ``status="insufficient"`` with the per-cell ``n``/30 progress, so
the operator can see exactly how much more reviewing each cell needs.

Library-shaped: the pure compute (``compute_calibration_report``) takes
the raw review records and returns the report dict with no network calls;
``emit_calibration_report`` is the thin S3 read/write wrapper. The QWK
primitive is reused verbatim from ``evals.cross_validation`` so there is
exactly one κ implementation in the repo.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import boto3

from evals.cross_validation import quadratic_weighted_kappa

logger = logging.getLogger(__name__)


_RESEARCH_BUCKET = os.environ.get("CHANGELOG_BUCKET", "alpha-engine-research")
_CALIBRATION_PREFIX = "decision_artifacts/_calibration/"
_REVIEWS_FILENAME = "reviews.jsonl"
_REPORT_PREFIX = "decision_artifacts/_calibration/_report/"

MIN_REVIEWS_PER_CELL = 30
"""Per ROADMAP L480 acceptance: a (rubric_id, dimension) cell's κ is
only load-bearing once it has ≥30 paired reviews. Below that the report
still renders the cell, flagged ``sufficient=False``, so the operator
sees the n/30 progress."""

N_CLASSES = 5
"""Rubric dimensions are scored on an integer 1..5 ordinal scale."""

SCHEMA_VERSION = "1.0.0"


# ── Pure compute ─────────────────────────────────────────────────────────


def _coerce_score(value: Any) -> int | None:
    """Return an int 1..N_CLASSES, or None when the score is absent /
    out of range. Operator may leave a blind score unset (None); judge
    scores are always present but we stay defensive."""
    if value is None:
        return None
    try:
        s = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= s <= N_CLASSES:
        return s
    return None


def _iter_pairs(
    reviews: Iterable[dict],
) -> dict[tuple[str, str], dict[str, list[tuple[int, int]]]]:
    """Explode review records into per-cell paired-score lists.

    Returns ``{(rubric_id, dimension): {"blind": [(blind, llm), ...],
    "final": [(final, llm), ...]}}``. A pair is only emitted when BOTH
    sides coerce to a valid 1..5 score, so a skipped blind score drops
    that dimension from the blind list (but it may still contribute to
    the final list)."""
    cells: dict[tuple[str, str], dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {"blind": [], "final": []}
    )
    for rec in reviews:
        if not isinstance(rec, dict):
            continue
        rubric_id = str(rec.get("rubric_id", "") or "unknown")
        per_dim = rec.get("per_dimension") or []
        if not isinstance(per_dim, list):
            continue
        for dim in per_dim:
            if not isinstance(dim, dict):
                continue
            dim_name = str(dim.get("dimension", "") or "")
            if not dim_name:
                continue
            llm = _coerce_score(dim.get("llm_score"))
            if llm is None:
                continue
            cell = cells[(rubric_id, dim_name)]
            blind = _coerce_score(dim.get("blind_score"))
            if blind is not None:
                cell["blind"].append((blind, llm))
            final = _coerce_score(dim.get("final_score"))
            if final is not None:
                cell["final"].append((final, llm))
    return cells


def krippendorff_alpha_ordinal(
    pairs: Iterable[tuple[int, int]], n_classes: int = N_CLASSES
) -> float:
    """Krippendorff's α with the ordinal difference metric, for the
    fully-paired two-rater case (each unit = one (operator, judge) pair).

    Coincidence-matrix form::

        α = 1 - (n - 1) · Σ_{c<k} o_ck · δ²_ck
                          ─────────────────────
                          Σ_{c<k} n_c · n_k · δ²_ck

    where ``o`` is the symmetric coincidence matrix (each unit's two
    values contribute o_{v1,v2} += 1 and o_{v2,v1} += 1, i.e. ÷(m_u−1)
    with m_u=2), ``n_c`` are its marginals, ``n = Σ n_c``, and the
    ordinal metric is δ²_ck = ( Σ_{g=c..k} n_g − (n_c + n_k)/2 )².

    Returns 1.0 on perfect agreement, NaN on empty input, and 0.0 when
    expected disagreement is zero (one value used throughout — α
    undefined, reported as no-reliability).
    """
    pair_list = [(int(a), int(b)) for a, b in pairs]
    if not pair_list:
        return float("nan")

    # Coincidence matrix (1-indexed scores → 0-indexed matrix).
    o = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for a, b in pair_list:
        if not (1 <= a <= n_classes and 1 <= b <= n_classes):
            raise ValueError(f"score outside 1..{n_classes}: ({a}, {b})")
        o[a - 1][b - 1] += 1
        o[b - 1][a - 1] += 1

    marg = [sum(o[c]) for c in range(n_classes)]
    n_total = sum(marg)
    if n_total == 0:
        return float("nan")

    # Ordinal metric δ²_ck depends on the marginals.
    def delta_sq(c: int, k: int) -> float:
        lo, hi = (c, k) if c <= k else (k, c)
        inner = sum(marg[g] for g in range(lo, hi + 1)) - (marg[c] + marg[k]) / 2.0
        return inner * inner

    num = 0.0
    den = 0.0
    for c in range(n_classes):
        for k in range(c + 1, n_classes):
            d2 = delta_sq(c, k)
            num += o[c][k] * d2
            den += marg[c] * marg[k] * d2

    if den == 0:
        # All ratings landed in a single category — no expected
        # disagreement, α undefined.
        return 0.0
    return 1.0 - (n_total - 1) * num / den


def _summarize_pairs(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Raw-agreement companions for one paired-score list."""
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "exact_agreement": None,
            "within_one": None,
            "mean_abs_diff": None,
        }
    diffs = [abs(a - b) for a, b in pairs]
    return {
        "n": n,
        "exact_agreement": sum(1 for d in diffs if d == 0) / n,
        "within_one": sum(1 for d in diffs if d <= 1) / n,
        "mean_abs_diff": sum(diffs) / n,
    }


def _nan_to_none(x: float) -> float | None:
    """JSON has no NaN — render undefined κ/α as null."""
    return None if x != x else round(x, 4)


def compute_calibration_report(
    reviews: Iterable[dict],
    *,
    generated_at: str | None = None,
    min_reviews_per_cell: int = MIN_REVIEWS_PER_CELL,
) -> dict[str, Any]:
    """Pure compute — no network. Turn raw review records into the κ report.

    Returns a dict with overall ``status`` (``empty`` | ``insufficient``
    | ``ok``), the per-cell metrics, and roll-up counts. ``status`` is
    ``ok`` once at least one cell clears ``min_reviews_per_cell`` — those
    cells carry load-bearing κ; under-threshold cells are still listed
    with their n/threshold progress.
    """
    cells_raw = _iter_pairs(reviews)

    cells: list[dict[str, Any]] = []
    n_sufficient = 0
    for (rubric_id, dimension), pair_sets in sorted(cells_raw.items()):
        blind = pair_sets["blind"]
        final = pair_sets["final"]
        n_blind = len(blind)
        sufficient = n_blind >= min_reviews_per_cell
        if sufficient:
            n_sufficient += 1
        blind_summary = _summarize_pairs(blind)
        cells.append(
            {
                "rubric_id": rubric_id,
                "dimension": dimension,
                "n": n_blind,
                "sufficient": sufficient,
                "progress": f"{n_blind}/{min_reviews_per_cell}",
                # Headline: un-anchored operator score vs judge.
                "qwk_blind_vs_llm": _nan_to_none(quadratic_weighted_kappa(blind, N_CLASSES)),
                # Residual after the operator sees the judge's reasoning.
                "qwk_final_vs_llm": _nan_to_none(quadratic_weighted_kappa(final, N_CLASSES)),
                "krippendorff_alpha_blind": _nan_to_none(
                    krippendorff_alpha_ordinal(blind, N_CLASSES)
                ),
                "exact_agreement": (
                    None if blind_summary["exact_agreement"] is None
                    else round(blind_summary["exact_agreement"], 4)
                ),
                "within_one": (
                    None if blind_summary["within_one"] is None
                    else round(blind_summary["within_one"], 4)
                ),
                "mean_abs_diff": (
                    None if blind_summary["mean_abs_diff"] is None
                    else round(blind_summary["mean_abs_diff"], 4)
                ),
                "n_final": len(final),
            }
        )

    total_reviews = sum(c["n"] for c in cells)
    if not cells or total_reviews == 0:
        status = "empty"
    elif n_sufficient == 0:
        status = "insufficient"
    else:
        status = "ok"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "min_reviews_per_cell": min_reviews_per_cell,
        "n_cells": len(cells),
        "n_cells_sufficient": n_sufficient,
        "n_paired_reviews": total_reviews,
        "cells": cells,
    }


# ── Markdown rendering (consumed by the backtester evaluator email) ───────


def render_markdown(report: dict[str, Any]) -> str:
    """Compact markdown for the weekly evaluator email's
    ``## Judge calibration (κ)`` section. Safe to render at any status."""
    status = report.get("status", "empty")
    n_cells = report.get("n_cells", 0)
    n_ok = report.get("n_cells_sufficient", 0)
    thresh = report.get("min_reviews_per_cell", MIN_REVIEWS_PER_CELL)

    lines = ["## Judge calibration (κ)"]
    if status == "empty":
        lines.append(
            "_No operator calibration reviews submitted yet. Use the "
            "Calibrate tab on dashboard page 8 to seed the corpus "
            f"(≥{thresh} reviews per rubric×dimension cell needed)._"
        )
        return "\n".join(lines)

    if status == "insufficient":
        lines.append(
            f"_Corpus building: {n_cells} cell(s) seen, 0 at the ≥{thresh} "
            "threshold yet. Per-cell progress below._"
        )
    else:
        lines.append(
            f"_{n_ok}/{n_cells} cell(s) at the ≥{thresh}-review threshold. "
            "κ(blind, llm) is the un-anchored calibration estimate; "
            "κ(final, llm) is the residual after the judge's reasoning "
            "is revealed._"
        )

    lines.append("")
    lines.append("| rubric · dimension | n | κ blind | κ final | exact | α |")
    lines.append("|---|---|---|---|---|---|")
    for c in report.get("cells", []):
        flag = "" if c.get("sufficient") else " ⏳"
        def fmt(v: Any) -> str:
            return "—" if v is None else f"{v:.2f}"
        exact = c.get("exact_agreement")
        exact_s = "—" if exact is None else f"{exact:.0%}"
        lines.append(
            f"| `{c['rubric_id']}` · `{c['dimension']}`{flag} "
            f"| {c['progress']} "
            f"| {fmt(c.get('qwk_blind_vs_llm'))} "
            f"| {fmt(c.get('qwk_final_vs_llm'))} "
            f"| {exact_s} "
            f"| {fmt(c.get('krippendorff_alpha_blind'))} |"
        )
    return "\n".join(lines)


# ── S3 read / write wrapper ──────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def load_reviews(
    *, bucket: str | None = None, s3_client: Any = None
) -> list[dict]:
    """Read every ``_calibration/{date}/reviews.jsonl`` line in the
    bucket into a flat list of review records. The ``_report/`` subtree
    is excluded. Tolerant of malformed lines (logged + skipped) so one
    bad row never sinks the run."""
    bkt = bucket or _RESEARCH_BUCKET
    client = s3_client or boto3.client("s3")

    reviews: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bkt, Prefix=_CALIBRATION_PREFIX):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith(_REVIEWS_FILENAME):
                continue
            if key.startswith(_REPORT_PREFIX):
                continue
            try:
                body = client.get_object(Bucket=bkt, Key=key)["Body"].read()
            except Exception:  # noqa: BLE001 — one unreadable file ≠ run failure
                logger.warning("[calibration_kappa] could not read %s", key)
                continue
            for ln, raw in enumerate(body.decode("utf-8").splitlines(), start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(
                        "[calibration_kappa] bad JSONL line %s:%d — skipped",
                        key, ln,
                    )
                    continue
                reviews.append(rec)
    return reviews


def emit_calibration_report(
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    report_date: str | None = None,
    min_reviews_per_cell: int = MIN_REVIEWS_PER_CELL,
) -> dict[str, Any]:
    """Load the reviews corpus, compute the κ report, and write it to
    ``_calibration/_report/{date}/kappa.json`` + ``kappa.md``.

    The report is written on EVERY run (including ``status="empty"``) so
    the backtester evaluator email always has a current artifact to
    surface and the operator can track corpus progress. Returns the
    report dict (also carries ``report_keys`` for the written S3 keys).
    """
    bkt = bucket or _RESEARCH_BUCKET
    client = s3_client or boto3.client("s3")
    now = _utc_now_iso()
    report_date = report_date or now[:10]

    reviews = load_reviews(bucket=bkt, s3_client=client)
    report = compute_calibration_report(
        reviews, generated_at=now, min_reviews_per_cell=min_reviews_per_cell
    )

    json_key = f"{_REPORT_PREFIX}{report_date}/kappa.json"
    md_key = f"{_REPORT_PREFIX}{report_date}/kappa.md"
    # Stable "latest" pointers so consumers (the backtester evaluator
    # email) don't have to date-walk. The markdown pointer is what the
    # email surfaces verbatim — research owns the rendering, the
    # backtester just embeds it.
    latest_json_key = f"{_REPORT_PREFIX}latest/kappa.json"
    latest_md_key = f"{_REPORT_PREFIX}latest/kappa.md"

    body = json.dumps(report, indent=2, default=str).encode("utf-8")
    md = render_markdown(report).encode("utf-8")
    client.put_object(
        Bucket=bkt, Key=json_key, Body=body, ContentType="application/json"
    )
    client.put_object(
        Bucket=bkt, Key=md_key, Body=md, ContentType="text/markdown"
    )
    client.put_object(
        Bucket=bkt, Key=latest_json_key, Body=body, ContentType="application/json"
    )
    client.put_object(
        Bucket=bkt, Key=latest_md_key, Body=md, ContentType="text/markdown"
    )

    report["report_keys"] = [json_key, md_key, latest_json_key, latest_md_key]
    logger.info(
        "[calibration_kappa] status=%s cells=%d sufficient=%d reviews=%d → s3://%s/%s",
        report["status"],
        report["n_cells"],
        report["n_cells_sufficient"],
        report["n_paired_reviews"],
        bkt,
        json_key,
    )
    return report
