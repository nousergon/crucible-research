"""RAG (Retrieval-Augmented Generation) hybrid-retrieval evaluation harness.

Calibrates ``vector_weight`` and confirms hybrid retrieval beats pure
vector and pure keyword on a hand-curated query set. Output drives
PR 5's default-weight choice + per-category lift documentation.

The harness is split into three layers so the math is pytest-testable
without a live Neon connection:

1. Pure functions — ``recall_at_k`` / ``aggregate_recall`` /
   ``render_markdown_report``. No I/O, no DB.
2. Harness orchestrator — ``run_eval`` calls the pure functions over
   each (query, method, vector_weight) condition. Takes a callable
   ``retrieve_fn`` so live tests use real ``nousergon_lib.rag.retrieve``
   and unit tests inject a fake.
3. CLI runner — ``scripts/run_rag_retrieval_eval.py`` wires the live
   retrieve_fn + reads the YAML test set + writes the markdown report.

Acceptance gates (per ROADMAP P1 line 1242):
- Hybrid recall@10 ≥ pure-vector recall@10 across the test set.
- Positive gap on at least the ticker / filing-type / quantitative
  categories.
- Documented findings in ``alpha-engine-docs/private/rag-retrieval-eval-{date}.md``.

Out of scope here: the actual (query → expected_chunk_id) curation —
that needs domain knowledge and lives in
``evals/rag_retrieval_queries.yaml``. This module ships the harness
mechanics; the fixture starts empty (with placeholder schema + curation
notes) and PR 4's curation step populates it before the report runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Sequence


# ── Test set schema ─────────────────────────────────────────────────────────


# Categories the harness reports against. Curated query coverage should
# fall into these buckets so we can document per-category lift.
CATEGORIES: tuple[str, ...] = (
    "ticker_named_entity",   # "ABBV's competitive position", "PFE's pipeline"
    "filing_type",           # "10-Q risk factors for SYK", "earnings call from BIIB Q3"
    "date_range",            # "VRTX 2024 fiscal year R&D"
    "quantitative_line_item",  # "ABBV R&D spend $", "PFE FCF margin"
    "abstract_thesis",       # "company's competitive moat", "management quality"
    "conceptual_narrative",  # "drug pipeline transition risks", "patent cliff"
)


@dataclass(frozen=True)
class EvalQuery:
    """One curated test query with its expected-relevant chunk_ids.

    Multi-relevant queries supported (e.g. "ABBV 2024 10-Q risk factors"
    might have 3 equally-correct chunks); single-relevant is the common
    case (90%+ of curation).
    """

    query: str
    expected_chunk_ids: tuple[str, ...]
    category: str
    note: str = ""  # operator commentary on what makes this query hard / canonical


# ── Pure functions ──────────────────────────────────────────────────────────


def recall_at_k(retrieved: Sequence[str], expected: Iterable[str], k: int) -> float:
    """Fraction of expected chunks that appear in the retrieved top-k.

    For single-relevant queries this is binary {0.0, 1.0}; for
    multi-relevant queries it's the standard recall fraction.

    >>> recall_at_k(["a", "b", "c"], ["b"], k=3)
    1.0
    >>> recall_at_k(["a", "b", "c"], ["d"], k=3)
    0.0
    >>> recall_at_k(["a", "b", "c"], ["a", "d"], k=3)
    0.5
    >>> recall_at_k(["a", "b", "c"], ["a"], k=2)
    1.0
    >>> recall_at_k(["a", "b", "c"], ["c"], k=2)
    0.0
    """
    expected_set = set(expected)
    if not expected_set:
        # Defensive: a query with no expected chunks is a curation error,
        # but don't crash the whole eval — return 0.0 + leave it to the
        # operator to fix the YAML.
        return 0.0
    retrieved_top_k = list(retrieved)[:k]
    hits = sum(1 for cid in retrieved_top_k if cid in expected_set)
    return hits / len(expected_set)


@dataclass
class Condition:
    """One method × vector_weight × rerank combo to sweep over.

    ``rerank`` extends the method-only condition matrix introduced in
    PR 4 with the L1303 reranking dimension. When set, ``retrieve_kwargs``
    threads ``rerank=`` + ``rerank_input_n=`` into the underlying
    ``retrieve()`` call so the harness can measure pre- vs post-rerank
    recall@k from the same call site the production qual_tools uses.
    """

    name: str          # display name in the report ("hybrid w=0.7", "hybrid w=0.7 + ce rerank", ...)
    method: str        # "vector" | "keyword" | "hybrid"
    vector_weight: float | None  # None for non-hybrid
    rerank: str | None = None        # None | "cross_encoder" | "llm_judge"
    rerank_input_n: int | None = None  # pre-rerank candidate pool size; None when rerank=None

    @property
    def retrieve_kwargs(self) -> dict:
        kw: dict = {"method": self.method}
        if self.method == "hybrid":
            kw["vector_weight"] = self.vector_weight
        if self.rerank is not None:
            kw["rerank"] = self.rerank
            if self.rerank_input_n is not None:
                kw["rerank_input_n"] = self.rerank_input_n
        return kw


# Default sweep — six baseline conditions (vector + keyword + four
# hybrid weights, established 2026-05-08) plus one rerank condition
# layered on the winning hybrid w=0.7 baseline.
#
# **LLM-judge rerank condition removed 2026-05-25** (lib v0.34.0 dropped
# the ``LLMJudgeReranker`` class). The 2026-05-12 operator eval found
# both rerank variants regressed against the hybrid w=0.7 baseline
# (-33.3% CE, -14.2% LLM-judge recall@10 on SEC filings); LLM-judge was
# deleted outright per ``[[preference_llm_calls_confined_to_research_module]]``,
# CE stays for future domain-finetune retries. EXPERIMENTS.md captures
# the institutional rerank-revisit path.
#
# Rerank conditions require the ``[rerank]`` extras installed on the
# eval runner; the CLI emits a clear ImportError pointing at the
# install path if missing (see ``scripts/run_rag_retrieval_eval.py``
# docstring).
DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    Condition("vector", "vector", None),
    Condition("keyword", "keyword", None),
    Condition("hybrid w=0.3", "hybrid", 0.3),
    Condition("hybrid w=0.5", "hybrid", 0.5),
    Condition("hybrid w=0.7", "hybrid", 0.7),
    Condition("hybrid w=0.9", "hybrid", 0.9),
    Condition("hybrid w=0.7 + ce rerank", "hybrid", 0.7,
              rerank="cross_encoder", rerank_input_n=30),
)


DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)


@dataclass
class QueryResult:
    """Recall numbers for one (query, condition) pair."""

    query: EvalQuery
    condition_name: str
    recalls: dict[int, float]    # k → recall@k
    retrieved_chunk_ids: tuple[str, ...]  # for trace / debugging


@dataclass
class AggregateRow:
    """One row in the per-condition recall table (overall or per-category)."""

    condition_name: str
    n_queries: int
    mean_recall: dict[int, float]  # k → mean recall@k


def aggregate_recall(
    results: list[QueryResult],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, AggregateRow]:
    """Aggregate per-query results into per-condition mean recall.

    Returns a dict keyed by condition name; each AggregateRow holds the
    mean recall@k for each k. Empty results → empty dict (caller decides
    whether to render a placeholder section).
    """
    by_condition: dict[str, list[QueryResult]] = {}
    for r in results:
        by_condition.setdefault(r.condition_name, []).append(r)

    agg: dict[str, AggregateRow] = {}
    for cond_name, rs in by_condition.items():
        means: dict[int, float] = {}
        for k in k_values:
            vals = [r.recalls[k] for r in rs if k in r.recalls]
            means[k] = sum(vals) / len(vals) if vals else 0.0
        agg[cond_name] = AggregateRow(
            condition_name=cond_name,
            n_queries=len(rs),
            mean_recall=means,
        )
    return agg


def aggregate_by_category(
    results: list[QueryResult],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, dict[str, AggregateRow]]:
    """Two-level aggregation: category → condition → AggregateRow.

    Useful for the per-category lift table — surfaces which categories
    benefit most from hybrid (PR 5's documentation target).
    """
    by_cat: dict[str, list[QueryResult]] = {}
    for r in results:
        by_cat.setdefault(r.query.category, []).append(r)

    return {cat: aggregate_recall(rs, k_values) for cat, rs in by_cat.items()}


# ── Markdown report writer ──────────────────────────────────────────────────


def render_markdown_report(
    *,
    run_date: date,
    queries: list[EvalQuery],
    results: list[QueryResult],
    conditions: Sequence[Condition] = DEFAULT_CONDITIONS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> str:
    """Render a self-contained markdown report.

    Sections:
    1. Run metadata (date, query count, condition list)
    2. Overall recall@k table
    3. Per-category recall@k tables
    4. Recommendation (auto-generated from the data)
    5. Per-query trace (collapsed by default — long table for debugging)
    """
    if not results:
        return _render_empty_report(run_date, queries)

    overall = aggregate_recall(results, k_values)
    per_cat = aggregate_by_category(results, k_values)

    parts: list[str] = []
    parts.append(f"# RAG retrieval eval — {run_date.isoformat()}\n")
    parts.append(_render_run_metadata(queries, conditions))
    parts.append("\n## Overall recall@k\n")
    parts.append(_render_recall_table(overall, conditions, k_values))
    parts.append("\n## By category\n")
    for cat in sorted(per_cat.keys()):
        n = sum(1 for q in queries if q.category == cat)
        parts.append(f"\n### {cat} ({n} queries)\n")
        parts.append(_render_recall_table(per_cat[cat], conditions, k_values))
    parts.append("\n## Recommendation\n")
    parts.append(_render_recommendation(overall, per_cat, k_values))
    return "\n".join(parts) + "\n"


def _render_empty_report(run_date: date, queries: list[EvalQuery]) -> str:
    if not queries:
        body = (
            "No queries curated yet. Populate "
            "``evals/rag_retrieval_queries.yaml`` with at least 30 "
            "``(query → expected_chunk_id)`` pairs across the documented "
            "categories before running this report.\n"
        )
    else:
        body = (
            f"{len(queries)} queries curated but no results — looks like "
            "the live runner failed. Check the runner's stderr.\n"
        )
    return f"# RAG retrieval eval — {run_date.isoformat()}\n\n{body}"


def _render_run_metadata(
    queries: list[EvalQuery],
    conditions: Sequence[Condition],
) -> str:
    cat_counts = {c: sum(1 for q in queries if q.category == c) for c in CATEGORIES}
    lines = [
        f"**Queries:** {len(queries)} total",
    ]
    cat_breakdown = ", ".join(
        f"{c}={cat_counts[c]}" for c in CATEGORIES if cat_counts[c] > 0
    )
    if cat_breakdown:
        lines.append(f"**Category breakdown:** {cat_breakdown}")
    lines.append(
        "**Conditions:** " + ", ".join(c.name for c in conditions)
    )
    return "\n".join(lines)


def _render_recall_table(
    rows: dict[str, AggregateRow],
    conditions: Sequence[Condition],
    k_values: Sequence[int],
) -> str:
    header = "| Method | n | " + " | ".join(f"Recall@{k}" for k in k_values) + " |"
    sep = "|" + "---|" * (2 + len(k_values))
    body_lines: list[str] = []
    for cond in conditions:
        row = rows.get(cond.name)
        if row is None:
            body_lines.append(
                "| " + cond.name + " | 0 | " + " | ".join("—" for _ in k_values) + " |"
            )
            continue
        cells = [f"{row.mean_recall.get(k, 0.0):.3f}" for k in k_values]
        body_lines.append(
            f"| {cond.name} | {row.n_queries} | " + " | ".join(cells) + " |"
        )
    return "\n".join([header, sep] + body_lines)


def _render_recommendation(
    overall: dict[str, AggregateRow],
    per_cat: dict[str, dict[str, AggregateRow]],
    k_values: Sequence[int],
) -> str:
    """Auto-generated text suggesting the winning condition and weight.

    Always picks recall@10 as the canonical decision metric (mid-corpus
    relevance — ``top_k=10`` is the qual analyst's call).
    """
    decision_k = 10 if 10 in k_values else max(k_values)
    if not overall:
        return "_No data — recommendation skipped._\n"
    best_name, best_row = max(
        overall.items(), key=lambda kv: kv[1].mean_recall.get(decision_k, 0.0)
    )
    best_score = best_row.mean_recall.get(decision_k, 0.0)
    vector_score = overall.get("vector", AggregateRow("vector", 0, {})).mean_recall.get(decision_k, 0.0)
    lift = best_score - vector_score
    lines = [
        f"**Best condition (overall recall@{decision_k}):** `{best_name}` "
        f"with mean recall@{decision_k} = {best_score:.3f}",
        f"**Lift over pure vector:** {lift:+.3f} ({lift / vector_score * 100:+.1f}%)"
        if vector_score > 0
        else f"**Lift over pure vector:** {lift:+.3f}",
    ]
    # Per-category gating — the ROADMAP entry requires positive gap on
    # at least ticker / filing-type / quantitative categories.
    gated_categories = ("ticker_named_entity", "filing_type", "quantitative_line_item")
    gated_lifts: list[str] = []
    for cat in gated_categories:
        cat_rows = per_cat.get(cat)
        if not cat_rows:
            continue
        v = cat_rows.get("vector", AggregateRow("vector", 0, {})).mean_recall.get(decision_k, 0.0)
        b = cat_rows.get(best_name, AggregateRow(best_name, 0, {})).mean_recall.get(decision_k, 0.0)
        gated_lifts.append(f"- {cat}: {b - v:+.3f}")
    if gated_lifts:
        lines.append("\n**Gated-category lifts (vs vector, recall@%d):**\n%s" % (
            decision_k, "\n".join(gated_lifts)
        ))
    lines.append(
        "\n_PR 5 should adopt the winning condition's `vector_weight` "
        "as the default in `agents/sector_teams/qual_tools.py::query_filings`."
        " If lift is negative or the gated categories regress, switch "
        "score-normalization to RRF (Reciprocal Rank Fusion) and rerun."
    )
    return "\n".join(lines)


# ── Harness orchestrator ────────────────────────────────────────────────────


def run_eval(
    *,
    queries: list[EvalQuery],
    retrieve_fn: Callable,
    conditions: Sequence[Condition] = DEFAULT_CONDITIONS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    top_k: int | None = None,
) -> list[QueryResult]:
    """Run every (query, condition) combo through ``retrieve_fn`` and
    compute recall@k for each k in ``k_values``.

    ``retrieve_fn`` must accept the kwargs from
    ``Condition.retrieve_kwargs`` plus ``query=str`` and ``top_k=int``,
    and return an iterable of objects with a ``chunk_id`` attribute.

    ``top_k`` defaults to ``max(k_values)`` so we retrieve enough
    candidates to compute every recall@k from a single API call per
    condition.
    """
    if top_k is None:
        top_k = max(k_values)

    results: list[QueryResult] = []
    for query in queries:
        for cond in conditions:
            retrieved = retrieve_fn(
                query=query.query,
                top_k=top_k,
                **cond.retrieve_kwargs,
            )
            chunk_ids = tuple(getattr(r, "chunk_id", None) or "" for r in retrieved)
            recalls = {
                k: recall_at_k(chunk_ids, query.expected_chunk_ids, k)
                for k in k_values
            }
            results.append(
                QueryResult(
                    query=query,
                    condition_name=cond.name,
                    recalls=recalls,
                    retrieved_chunk_ids=chunk_ids,
                )
            )
    return results
