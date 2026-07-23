"""Manual judge cross-validation — calibration anchor for the LLM-as-judge.

ROADMAP L83 (P1):
    Manual judge cross-validation sample (LLM-as-judge calibration anchor).
    Manually rate captured decision artifacts on each rubric dimension;
    compare against judge scores; document per-dimension agreement rate.
    Re-validate quarterly + on every judge model upgrade.

Spot-check methodology (chosen over blind rating, 2026-05-13):
    The eval-judge framework emits per-agent rubric scores with a reasoning
    string per dimension. The methodology shows the operator the judge's
    score + reasoning inline and asks for a verdict (``agree`` / ``disagree``
    / ``partial``) per dimension, plus an override score + notes only when
    the operator flags a dispute.

    Trade-off: ratings are anchored to the judge's score (kappa is inflated),
    BUT the deliverable becomes the dispute log — qualitative reasoning
    comparisons that are more actionable for tuning the rubric than a kappa
    number. Concurrence rate + dispute appendix is the headline output.

This module is the parsing + summary layer:
    1. Operator fills worksheets in a rating bundle (see
       ``judge-crossval-260513/`` for the format — judge scores rendered
       inline + verdict field per dimension).
    2. ``collect_review_outcomes`` walks the bundle, parses each
       worksheet, joins to the hidden judge-score files, and emits one
       ``ReviewOutcome`` per (artifact, dimension, judge_model).
    3. ``render_markdown_report`` produces the concurrence summary + dispute
       appendix.

Library-shaped: no network calls, no boto3, deterministic given input
file paths. The companion operator script
``scripts/run_judge_cross_validation.py`` is the thin CLI that locates
the bundle directory and emits the markdown report.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

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

# Spot-check verdict line:
#   - **decision_coherence**: agree
#   - **decision_coherence**: disagree
#   - **decision_coherence**: partial
#   - **decision_coherence**: VERDICT_HERE   (unfilled — skipped)
#
# The regex matches any word so we can catch typos ("agreee", "maybe") and
# raise an explicit error rather than silently skipping the dimension.
_VERDICT_LINE = re.compile(
    r"^\s*-\s+\*\*(?P<dim>[A-Za-z_][A-Za-z0-9_]*)\*\*:\s*"
    r"(?P<val>[A-Za-z_]+)\b"
)

# Spot-check sub-line for override_score / notes:
#   - override_score: 3
#   - notes: judge missed the regime override on EOG
_OVERRIDE_LINE = re.compile(
    r"^\s*-\s+override_score:\s*(?P<val>\d|SCORE_HERE)\b"
)
_NOTES_LINE = re.compile(
    r"^\s*-\s+notes:\s*(?P<val>.*?)\s*$"
)


VALID_VERDICTS = {"agree", "disagree", "partial"}


def parse_worksheet(path: Path) -> dict[str, int]:
    """Return a mapping dimension_name -> integer score for one blind-rating
    worksheet (legacy blind-mode format with ``- **dim**: 4`` score lines).

    Spot-check worksheets (current format) use :func:`parse_spotcheck_worksheet`.
    """
    scores: dict[str, int] = {}
    in_scores_section = False
    text = path.read_text()
    for line in text.splitlines():
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


@dataclass(frozen=True)
class SpotcheckEntry:
    """One dimension's spot-check verdict + optional override."""
    dimension: str
    verdict: str                 # "agree" | "disagree" | "partial"
    override_score: int | None   # only set when verdict != "agree"
    notes: str | None


def parse_spotcheck_worksheet(path: Path) -> dict[str, SpotcheckEntry]:
    """Parse a spot-check worksheet → mapping dimension_name → SpotcheckEntry.

    Worksheet shape (per dimension):

        - **dimension_name**: agree   (or disagree / partial)
          - override_score: 3   (1-5; only required when verdict != agree)
          - notes: free text

    Dimensions left at ``VERDICT_HERE`` are skipped (not yet reviewed).
    A ``disagree`` or ``partial`` verdict without an override_score raises
    ``ValueError`` — the operator flagged a dispute but didn't say what
    score they'd give instead, which would leave the dispute appendix
    half-empty.
    """
    text = path.read_text()
    in_section = False
    # Walk the file once; whenever we hit a verdict line, look ahead at
    # the next several indented lines for override_score / notes belonging
    # to the same dimension.
    lines = text.splitlines()
    out: dict[str, SpotcheckEntry] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("## "):
            in_section = line.strip() == "## Your spot-check"
            i += 1
            continue
        if not in_section:
            i += 1
            continue
        m = _VERDICT_LINE.match(line)
        if not m:
            i += 1
            continue
        dim = m.group("dim")
        verdict_raw = m.group("val").lower()
        if verdict_raw == "verdict_here":
            i += 1
            continue
        if verdict_raw not in VALID_VERDICTS:
            raise ValueError(
                f"{path.name}: dimension {dim!r} verdict {verdict_raw!r} "
                f"not one of {sorted(VALID_VERDICTS)}"
            )

        override_score: int | None = None
        notes: str | None = None
        # Look at the next few lines for override / notes (only the
        # immediately following indented sub-bullets belong to this
        # dimension; stop at blank line or next dimension verdict).
        j = i + 1
        while j < len(lines):
            sub = lines[j]
            stripped = sub.strip()
            if not stripped:
                break
            if _VERDICT_LINE.match(sub):
                break
            om = _OVERRIDE_LINE.match(sub)
            if om:
                v = om.group("val")
                if v != "SCORE_HERE":
                    s = int(v)
                    if s < 1 or s > 5:
                        raise ValueError(
                            f"{path.name}: dimension {dim!r} "
                            f"override_score {s} out of range 1-5"
                        )
                    override_score = s
                j += 1
                continue
            nm = _NOTES_LINE.match(sub)
            if nm:
                raw = nm.group("val")
                if raw and raw != "WRITE_HERE  (optional — fill if you flagged anything)" \
                        and raw != "WRITE_HERE":
                    notes = raw
                j += 1
                continue
            # Anything else — break out (probably a markdown subheader or
            # the next dimension's anchors).
            break

        if verdict_raw in {"disagree", "partial"} and override_score is None:
            raise ValueError(
                f"{path.name}: dimension {dim!r} marked {verdict_raw!r} "
                f"but no override_score given. Fill override_score 1-5 so "
                f"the dispute appendix has a comparable number."
            )

        out[dim] = SpotcheckEntry(
            dimension=dim,
            verdict=verdict_raw,
            override_score=override_score,
            notes=notes,
        )
        i = j
    return out


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
    """End-to-end (blind-rating mode): parse bundle, compute agreement,
    render markdown. Returns ``(markdown_report, agreements)``.

    For spot-check mode (current default), use
    :func:`run_spotcheck_review` instead.
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


# ── Spot-check mode (current default) ───────────────────────────────────


@dataclass(frozen=True)
class ReviewOutcome:
    """One operator-vs-judge spot-check on one (artifact, dimension, judge_model)."""
    artifact_nn: str
    agent_id: str
    rubric_family: str
    run_id: str
    dimension: str
    judge_model: str
    verdict: str                   # "agree" | "disagree" | "partial"
    judge_score: int
    judge_reasoning: str
    override_score: int | None     # operator's score when verdict != "agree"
    operator_notes: str | None


def collect_review_outcomes(bundle_dir: Path) -> list[ReviewOutcome]:
    """Walk the bundle, parse each spot-check worksheet, join to the
    hidden judge-score files, return one ``ReviewOutcome`` per
    (artifact, dimension, judge_model). Unreviewed dimensions are
    skipped silently.
    """
    items = load_index(bundle_dir)
    outcomes: list[ReviewOutcome] = []
    for item in items:
        ws_path = bundle_dir / item["worksheet_path"]
        judge_path = bundle_dir / item["judge_scores_path"]
        if not ws_path.exists() or not judge_path.exists():
            logger.warning("missing files for artifact %s — skip", item["nn"])
            continue
        entries = parse_spotcheck_worksheet(ws_path)
        if not entries:
            continue  # not yet reviewed
        judge_payload = json.loads(judge_path.read_text())
        for judge_model, jp in judge_payload.items():
            judge_dims = {d["dimension"]: d for d in jp.get("dimension_scores", [])}
            for dim, entry in entries.items():
                jd = judge_dims.get(dim)
                if jd is None:
                    logger.warning(
                        "artifact %s: dimension %r reviewed but not in "
                        "judge eval (%s) — skip",
                        item["nn"], dim, judge_model,
                    )
                    continue
                outcomes.append(ReviewOutcome(
                    artifact_nn=item["nn"],
                    agent_id=item["agent_id"],
                    rubric_family=item["rubric_family"],
                    run_id=item["run_id"],
                    dimension=dim,
                    judge_model=judge_model,
                    verdict=entry.verdict,
                    judge_score=int(jd["score"]),
                    judge_reasoning=(jd.get("reasoning") or "").strip(),
                    override_score=entry.override_score,
                    operator_notes=entry.notes,
                ))
    return outcomes


def render_spotcheck_report(
    outcomes: list[ReviewOutcome],
    *,
    bundle_dir: Path,
) -> str:
    """Render the spot-check report — concurrence summary + dispute appendix.

    Concurrence is the headline metric (% of reviewed dimensions where
    operator marked ``agree``). Dispute appendix lists every
    ``disagree`` / ``partial`` with judge + operator scores +
    judge-reasoning + operator-notes side-by-side. This is the
    actionable artifact for tuning the rubric.
    """
    n = len(outcomes)
    by_verdict = {"agree": 0, "disagree": 0, "partial": 0}
    for o in outcomes:
        by_verdict[o.verdict] = by_verdict.get(o.verdict, 0) + 1

    # Per-rubric-family + per-judge-model concurrence
    cells: dict[tuple[str, str], list[ReviewOutcome]] = {}
    for o in outcomes:
        cells.setdefault((o.rubric_family, o.judge_model), []).append(o)

    lines: list[str] = []
    lines.append("# LLM-as-judge spot-check report")
    lines.append("")
    lines.append(f"- Bundle: `{bundle_dir.name}`")
    lines.append(f"- Dimensions reviewed: {n}")
    lines.append(
        f"- Concurrence: **{by_verdict['agree']}/{n}** "
        f"({by_verdict['agree']/n:.0%} agree · "
        f"{by_verdict['partial']/n:.0%} partial · "
        f"{by_verdict['disagree']/n:.0%} disagree)"
        if n else "- Concurrence: n/a (no reviewed dimensions)"
    )
    if n:
        # Override deltas (signed: operator_score - judge_score) on disputes
        deltas = [
            o.override_score - o.judge_score
            for o in outcomes
            if o.verdict != "agree" and o.override_score is not None
        ]
        if deltas:
            lines.append(
                f"- Override delta (operator - judge) over "
                f"{len(deltas)} disputes: "
                f"mean {mean(deltas):+.2f}, "
                f"range [{min(deltas):+d}, {max(deltas):+d}]"
            )
    lines.append("")

    if not outcomes:
        return "\n".join(lines) + "\nNo spot-check verdicts found. Mark `agree` / `disagree` / `partial` on at least one dimension.\n"

    lines.append("## Concurrence by `(rubric_family, judge_model)`")
    lines.append("")
    lines.append("| rubric_family | judge_model | n | agree | partial | disagree |")
    lines.append("|---|---|--:|--:|--:|--:|")
    for (family, jm), cell in sorted(cells.items()):
        cn = len(cell)
        ca = sum(1 for o in cell if o.verdict == "agree")
        cp = sum(1 for o in cell if o.verdict == "partial")
        cd = sum(1 for o in cell if o.verdict == "disagree")
        lines.append(
            f"| {family} | {jm} | {cn} | "
            f"{ca/cn:.0%} | {cp/cn:.0%} | {cd/cn:.0%} |"
        )
    lines.append("")

    # Dispute appendix
    disputes = [o for o in outcomes if o.verdict != "agree"]
    lines.append(f"## Dispute appendix ({len(disputes)} dimensions)")
    lines.append("")
    if not disputes:
        lines.append("_No disputes flagged. Either the judge is well-calibrated on this sample, or the sample was easy._")
        lines.append("")
        return "\n".join(lines)

    for o in sorted(disputes, key=lambda x: (x.artifact_nn, x.dimension, x.judge_model)):
        delta = (
            f"{o.override_score - o.judge_score:+d}"
            if o.override_score is not None else "n/a"
        )
        lines.append(
            f"### #{o.artifact_nn} `{o.agent_id}` ({o.run_id}) · "
            f"`{o.dimension}` · `{o.judge_model}`"
        )
        lines.append("")
        lines.append(
            f"- **Verdict:** `{o.verdict}` · "
            f"judge={o.judge_score} → operator={o.override_score} (Δ={delta})"
        )
        lines.append(f"- **Judge reasoning:** {o.judge_reasoning or '_(none)_'}")
        if o.operator_notes:
            lines.append(f"- **Operator notes:** {o.operator_notes}")
        lines.append("")

    return "\n".join(lines)


def run_spotcheck_review(bundle_dir: Path) -> tuple[str, list[ReviewOutcome]]:
    """End-to-end (spot-check mode): parse bundle, render markdown report.

    Returns ``(markdown_report, outcomes)``. Empty report if no
    worksheets have any verdicts filled yet.
    """
    outcomes = collect_review_outcomes(bundle_dir)
    if not outcomes:
        return (
            "# LLM-as-judge spot-check report\n\n"
            "No verdicts filled in the bundle yet. Mark `agree` / "
            "`disagree` / `partial` on at least one dimension.\n",
            [],
        )
    md = render_spotcheck_report(outcomes, bundle_dir=bundle_dir)
    return md, outcomes
