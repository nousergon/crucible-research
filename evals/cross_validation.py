"""Manual judge cross-validation — calibration anchor for the LLM-as-judge.

ROADMAP L83 (P1):
    Manual judge cross-validation sample (LLM-as-judge calibration anchor).
    Manually rate 10-20 captured decision artifacts on each rubric dimension;
    compare against judge scores; document per-dimension agreement rate.
    Re-validate quarterly + on every judge model upgrade.

The eval-judge framework (``evals/judge.py`` + ``evals/orchestrator.py``)
emits per-agent rubric scores. Without a periodic human cross-validation
anchor, judge-score drift across model upgrades and prompt revisions is
undetectable. This module is the comparison layer:

    1. Operator rates a stratified sample of decision artifacts on each
       rubric dimension (see ``judge-crossval-260513/`` bundle for the
       worksheet format — agent input + rubric anchors + blank score fields).
    2. This module parses the filled worksheets and joins them against the
       judge's scores stored under ``decision_artifacts/_eval/`` in S3 (or
       a local hidden copy).
    3. Emits per-dimension agreement metrics: exact-match rate, ±1-tolerance
       rate, mean absolute difference, quadratic-weighted Cohen's kappa.
       Quadratic kappa is the standard ordinal-rating agreement metric:
       penalizes large disagreements (1↔5) more than small (3↔4) and
       chance-corrects, so it's robust to score-distribution skew.

The module is operator-driven and library-shaped: no network calls, no
boto3, deterministic given input file paths. The companion operator
script in ``scripts/run_judge_cross_validation.py`` is the thin CLI that
locates the bundle directory and emits the markdown report.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Domain types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RatingPair:
    """One (human, judge) score pair for one artifact × dimension × judge_model."""
    artifact_nn: str
    agent_id: str
    rubric_family: str
    run_id: str
    dimension: str
    human_score: int
    judge_score: int
    judge_model: str


@dataclass
class DimensionAgreement:
    """Per-dimension agreement summary over all paired ratings."""
    rubric_family: str
    dimension: str
    judge_model: str
    n: int
    exact_match_rate: float
    within_one_rate: float
    mean_abs_diff: float
    quadratic_weighted_kappa: float
    score_pairs: list[tuple[int, int]] = field(default_factory=list)


# ── Worksheet parsing ───────────────────────────────────────────────────


# Match worksheet lines like:
#   - **decision_coherence**: 4  (1-5)
#   - **decision_coherence**: 4
# Tolerates the placeholder "SCORE_HERE" so an unfilled worksheet just
# produces no row for that dimension (rather than raising).
_SCORE_LINE = re.compile(
    r"^\s*-\s+\*\*(?P<dim>[A-Za-z_][A-Za-z0-9_]*)\*\*:\s*"
    r"(?P<val>\d|SCORE_HERE)\b"
)


def parse_worksheet(path: Path) -> dict[str, int]:
    """Return a mapping dimension_name -> integer score for one filled worksheet.

    Lines with the placeholder ``SCORE_HERE`` are skipped (treated as not yet
    rated). Lines with values outside 1-5 raise ``ValueError`` so a typo
    surfaces immediately instead of polluting the aggregate.
    """
    scores: dict[str, int] = {}
    in_scores_section = False
    text = path.read_text()
    for line in text.splitlines():
        # The rubric section also has lines like "  5 — ..." that match our
        # pattern's digit class if we're not careful. Gate on the
        # "## Your scores" header so we only parse the operator's score
        # block, not the rubric anchors.
        if line.strip().startswith("## "):
            in_scores_section = line.strip() == "## Your scores"
            continue
        if not in_scores_section:
            continue
        m = _SCORE_LINE.match(line)
        if not m:
            continue
        val = m.group("val")
        if val == "SCORE_HERE":
            continue
        score = int(val)
        if score < 1 or score > 5:
            raise ValueError(
                f"{path.name}: dimension {m.group('dim')!r} score "
                f"{score} out of range 1-5"
            )
        scores[m.group("dim")] = score
    return scores


def parse_judge_scores(path: Path) -> dict[str, dict[str, int]]:
    """Return ``{judge_model: {dimension: score}}`` from a hidden judge-scores file.

    The hidden file layout (produced by ``sample_and_bundle.py``):

        {
          "claude-haiku-4-5": {
              "rubric_id": "eval_rubric_ic_cio",
              "rubric_version": "1.2.0",
              "dimension_scores": [{"dimension": "...", "score": 4, ...}, ...],
              ...
          },
          "claude-sonnet-4-6": {...}
        }
    """
    payload = json.loads(path.read_text())
    out: dict[str, dict[str, int]] = {}
    for judge_model, entry in payload.items():
        out[judge_model] = {
            d["dimension"]: int(d["score"])
            for d in entry.get("dimension_scores", [])
        }
    return out


# ── Bundle traversal ────────────────────────────────────────────────────


def load_index(bundle_dir: Path) -> list[dict]:
    """Read the ``index.json`` manifest produced by the sampler."""
    payload = json.loads((bundle_dir / "index.json").read_text())
    return payload["items"]


def collect_rating_pairs(bundle_dir: Path) -> list[RatingPair]:
    """Walk a filled rating bundle and emit one ``RatingPair`` per
    (artifact, dimension, judge_model) where both human + judge scores
    exist. Artifacts the operator hasn't rated yet are skipped silently.
    """
    items = load_index(bundle_dir)
    pairs: list[RatingPair] = []
    for item in items:
        ws_path = bundle_dir / item["worksheet_path"]
        judge_path = bundle_dir / item["judge_scores_path"]
        if not ws_path.exists():
            logger.warning("worksheet %s missing — skip", ws_path)
            continue
        if not judge_path.exists():
            logger.warning("judge_scores %s missing — skip", judge_path)
            continue
        human = parse_worksheet(ws_path)
        if not human:
            # operator has not rated this one yet
            continue
        judge_by_model = parse_judge_scores(judge_path)
        for judge_model, judge in judge_by_model.items():
            for dim, h_score in human.items():
                j_score = judge.get(dim)
                if j_score is None:
                    logger.warning(
                        "artifact %s: dimension %r in worksheet but not in "
                        "judge eval (%s) — skip",
                        item["nn"], dim, judge_model,
                    )
                    continue
                pairs.append(RatingPair(
                    artifact_nn=item["nn"],
                    agent_id=item["agent_id"],
                    rubric_family=item["rubric_family"],
                    run_id=item["run_id"],
                    dimension=dim,
                    human_score=h_score,
                    judge_score=j_score,
                    judge_model=judge_model,
                ))
    return pairs


# ── Agreement metrics ───────────────────────────────────────────────────


def quadratic_weighted_kappa(
    pairs: Iterable[tuple[int, int]], n_classes: int = 5
) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal 1..n_classes ratings.

    Standard formula:

        kappa = 1 - sum_{i,j} w_{ij} * O_{ij} / sum_{i,j} w_{ij} * E_{ij}

    where w_{ij} = (i - j)^2 / (n_classes - 1)^2 (quadratic weights),
    O is the normalized observed confusion matrix, E is the
    expected-by-chance matrix from the marginals.

    Returns 0.0 when there is no disagreement to weigh (pairs all-equal or
    one rater has zero variance) — kappa is undefined there. Returns
    NaN-equivalent (``float('nan')``) for empty input.
    """
    pair_list = list(pairs)
    if not pair_list:
        return float("nan")

    # Build confusion matrix indexed 0..n_classes-1
    obs = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for h, j in pair_list:
        if not (1 <= h <= n_classes and 1 <= j <= n_classes):
            raise ValueError(f"score outside 1..{n_classes}: ({h}, {j})")
        obs[h - 1][j - 1] += 1
    total = len(pair_list)

    # Marginals
    row_marginal = [sum(obs[i]) for i in range(n_classes)]
    col_marginal = [sum(obs[i][j] for i in range(n_classes))
                    for j in range(n_classes)]

    # Quadratic weights w_{ij} = (i-j)^2 / (n-1)^2
    denom_sq = (n_classes - 1) ** 2
    num_obs = 0.0
    num_exp = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            w = ((i - j) ** 2) / denom_sq
            num_obs += w * obs[i][j] / total
            num_exp += w * row_marginal[i] * col_marginal[j] / (total * total)

    if num_exp == 0:
        # No expected disagreement (one rater is constant). Kappa undefined.
        return 0.0
    return 1.0 - num_obs / num_exp


def summarize_agreement(pairs: list[RatingPair]) -> list[DimensionAgreement]:
    """Group pairs by (rubric_family, dimension, judge_model) and compute
    the four agreement metrics for each cell.
    """
    by_cell: dict[tuple[str, str, str], list[RatingPair]] = {}
    for p in pairs:
        key = (p.rubric_family, p.dimension, p.judge_model)
        by_cell.setdefault(key, []).append(p)

    out: list[DimensionAgreement] = []
    for (family, dim, judge_model), cell in sorted(by_cell.items()):
        score_pairs = [(p.human_score, p.judge_score) for p in cell]
        diffs = [abs(h - j) for h, j in score_pairs]
        n = len(cell)
        exact = sum(1 for d in diffs if d == 0) / n
        within_one = sum(1 for d in diffs if d <= 1) / n
        mad = mean(diffs)
        kappa = quadratic_weighted_kappa(score_pairs)
        out.append(DimensionAgreement(
            rubric_family=family,
            dimension=dim,
            judge_model=judge_model,
            n=n,
            exact_match_rate=exact,
            within_one_rate=within_one,
            mean_abs_diff=mad,
            quadratic_weighted_kappa=kappa,
            score_pairs=score_pairs,
        ))
    return out


# ── Report rendering ────────────────────────────────────────────────────


def render_markdown_report(
    agreements: list[DimensionAgreement],
    *,
    bundle_dir: Path,
    n_artifacts_rated: int,
) -> str:
    """Render the per-dimension agreement table to markdown."""
    lines: list[str] = []
    lines.append("# LLM-as-judge cross-validation report")
    lines.append("")
    lines.append(f"- Bundle: `{bundle_dir.name}`")
    lines.append(f"- Artifacts rated: {n_artifacts_rated}")
    lines.append(f"- Dimension cells: {len(agreements)}")
    lines.append("")
    lines.append("Quadratic-weighted Cohen's kappa interpretation (Landis & Koch 1977):")
    lines.append("`<0` worse than chance · `0.01-0.20` slight · `0.21-0.40` fair · "
                 "`0.41-0.60` moderate · `0.61-0.80` substantial · `0.81-1.0` almost perfect.")
    lines.append("")

    # Group by rubric_family
    by_family: dict[str, list[DimensionAgreement]] = {}
    for a in agreements:
        by_family.setdefault(a.rubric_family, []).append(a)

    for family, rows in sorted(by_family.items()):
        lines.append(f"## `{family}`")
        lines.append("")
        lines.append("| dimension | judge_model | n | exact | ±1 | MAD | κ (quad) |")
        lines.append("|---|---|--:|--:|--:|--:|--:|")
        for r in sorted(rows, key=lambda x: (x.dimension, x.judge_model)):
            lines.append(
                f"| {r.dimension} | {r.judge_model} | {r.n} | "
                f"{r.exact_match_rate:.0%} | {r.within_one_rate:.0%} | "
                f"{r.mean_abs_diff:.2f} | {r.quadratic_weighted_kappa:.2f} |"
            )
        lines.append("")

    # Overall numbers (collapse across cells, weighted by n)
    all_pairs: list[tuple[int, int]] = []
    for a in agreements:
        all_pairs.extend(a.score_pairs)
    if all_pairs:
        diffs = [abs(h - j) for h, j in all_pairs]
        n = len(all_pairs)
        lines.append("## Overall (all dimensions, all judge models)")
        lines.append("")
        lines.append(f"- n score-pairs: {n}")
        lines.append(f"- exact agreement: {sum(1 for d in diffs if d == 0)/n:.0%}")
        lines.append(f"- ±1 tolerance: {sum(1 for d in diffs if d <= 1)/n:.0%}")
        lines.append(f"- mean absolute diff: {mean(diffs):.2f}")
        lines.append(
            f"- quadratic-weighted κ: "
            f"{quadratic_weighted_kappa(all_pairs):.2f}"
        )
        lines.append("")

    return "\n".join(lines)


def run_cross_validation(bundle_dir: Path) -> tuple[str, list[DimensionAgreement]]:
    """End-to-end: parse bundle, compute agreement, render markdown.

    Returns ``(markdown_report, agreements)``. The caller decides where to
    persist (local file, S3, both). Returns an empty report if no
    worksheets are filled yet.
    """
    pairs = collect_rating_pairs(bundle_dir)
    if not pairs:
        return (
            "# LLM-as-judge cross-validation report\n\n"
            "No filled worksheets found in the bundle. Rate at least one "
            "worksheet before running the report.\n",
            [],
        )
    agreements = summarize_agreement(pairs)
    n_artifacts = len({p.artifact_nn for p in pairs})
    md = render_markdown_report(
        agreements,
        bundle_dir=bundle_dir,
        n_artifacts_rated=n_artifacts,
    )
    return md, agreements
