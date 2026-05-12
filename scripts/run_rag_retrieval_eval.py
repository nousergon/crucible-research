"""CLI entry point for the RAG hybrid-retrieval eval harness.

Reads ``evals/rag_retrieval_queries.yaml``, runs every query through
each (method × vector_weight × rerank) condition against the live
Neon pgvector database, and writes the markdown report to
``alpha-engine-docs/private/rag-retrieval-eval-{date}.md`` (default)
or wherever ``--out`` points.

Requires ``RAG_DATABASE_URL`` + ``VOYAGE_API_KEY`` in the environment.
Rerank conditions (cross_encoder + llm_judge) additionally require:

    pip install 'alpha-engine-lib[rerank] @ git+https://github.com/cipher813/alpha-engine-lib@v0.11.0'
    # (LLM-judge path also needs ANTHROPIC_API_KEY)

When the ``[rerank]`` extras aren't installed on the eval runner, pass
``--skip-rerank`` to drop the rerank conditions from the sweep and only
exercise the original hybrid-only matrix. Otherwise the import error
surfaces loudly on the first rerank condition's call.

The report is the source of truth for the L1303 ROADMAP P1 cutover
decision — flip ``RAG_RERANK=cross_encoder`` (or ``llm_judge``) in
the alpha-engine-config Lambda env iff the rerank conditions show
material recall@10 lift over the hybrid w=0.7 baseline.

Usage:

    # Curate evals/rag_retrieval_queries.yaml first (see file header).

    # Run with defaults (full sweep, including rerank conditions).
    python scripts/run_rag_retrieval_eval.py

    # Drop rerank conditions (operator without [rerank] extras installed).
    python scripts/run_rag_retrieval_eval.py --skip-rerank

    # Custom output path:
    python scripts/run_rag_retrieval_eval.py --out /tmp/eval.md

    # Smaller sweep (e.g. just overall, no per-category):
    python scripts/run_rag_retrieval_eval.py --queries-file custom.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.rag_retrieval import (  # noqa: E402
    CATEGORIES,
    DEFAULT_CONDITIONS,
    DEFAULT_K_VALUES,
    EvalQuery,
    render_markdown_report,
    run_eval,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_queries(path: Path) -> list[EvalQuery]:
    """Read the YAML test set + validate schema.

    Skipped entries (``expected_chunk_ids == ["TODO"]``) are silently
    dropped from the eval — the curate workflow uses TODO as the
    "skip this query" sentinel when no candidate in top-10 was
    relevant. Skipping is signal: it means neither retrieval method
    surfaced the right chunk, OR the corpus doesn't contain it. The
    eval just doesn't include those queries in the recall numbers.

    Other malformed entries raise loudly so the operator fixes
    curation rather than getting a silently-empty report.
    """
    if not path.exists():
        raise FileNotFoundError(f"queries file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        raise ValueError(f"queries must be a list, got {type(raw_queries).__name__}")

    out: list[EvalQuery] = []
    skipped_todo = 0
    for i, entry in enumerate(raw_queries):
        if not isinstance(entry, dict):
            raise ValueError(f"queries[{i}] must be a dict, got {type(entry).__name__}")
        try:
            query_text = entry["query"]
            expected_ids = tuple(entry["expected_chunk_ids"])
            category = entry["category"]
        except KeyError as e:
            raise ValueError(
                f"queries[{i}] missing required field {e}; required: "
                "query, expected_chunk_ids, category"
            ) from e
        if category not in CATEGORIES:
            raise ValueError(
                f"queries[{i}] category={category!r} not in {CATEGORIES}"
            )
        # TODO sentinel = skip; drop from eval silently.
        if expected_ids == ("TODO",):
            skipped_todo += 1
            continue
        if not expected_ids:
            raise ValueError(
                f"queries[{i}] expected_chunk_ids must be non-empty — "
                f"a query with no expected chunks is a curation error "
                f"(use [\"TODO\"] sentinel to skip on purpose)"
            )
        out.append(
            EvalQuery(
                query=query_text,
                expected_chunk_ids=expected_ids,
                category=category,
                note=entry.get("note", ""),
            )
        )
    if skipped_todo:
        log.info(
            "Skipped %d query/queries with TODO chunk_ids — these stay "
            "uncurated and don't contribute to the recall numbers.",
            skipped_todo,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries-file",
        default=str(_REPO_ROOT / "evals" / "rag_retrieval_queries.yaml"),
        help="Path to the YAML test set",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output markdown path. Default: "
            "~/Development/alpha-engine-docs/private/rag-retrieval-eval-{date}.md"
        ),
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help=(
            "Drop rerank conditions from the sweep (use when the eval "
            "runner doesn't have the alpha-engine-lib[rerank] extras "
            "installed). Default: include all conditions."
        ),
    )
    args = parser.parse_args()

    queries = load_queries(Path(args.queries_file))
    log.info("Loaded %d queries from %s", len(queries), args.queries_file)
    if not queries:
        log.warning(
            "No queries curated. Populate %s with at least 30 entries "
            "across the documented categories before re-running.",
            args.queries_file,
        )
        # Still write the placeholder report so the operator sees the
        # empty-state message rather than a missing file.

    # Live retrieve_fn — defer the lib import to runtime so the harness
    # is importable for unit tests without RAG_DATABASE_URL set.
    from alpha_engine_lib.rag import retrieve as live_retrieve

    if args.skip_rerank:
        conditions = tuple(c for c in DEFAULT_CONDITIONS if c.rerank is None)
        log.info(
            "Skipping rerank conditions (--skip-rerank); running %d of %d "
            "default conditions.", len(conditions), len(DEFAULT_CONDITIONS),
        )
    else:
        conditions = DEFAULT_CONDITIONS

    if queries:
        log.info("Running %d queries × %d conditions = %d retrievals",
                 len(queries), len(conditions),
                 len(queries) * len(conditions))
        results = run_eval(queries=queries, retrieve_fn=live_retrieve,
                           conditions=conditions)
    else:
        results = []

    run_date = date.today()
    report_md = render_markdown_report(
        run_date=run_date,
        queries=queries,
        results=results,
        conditions=conditions,
        k_values=DEFAULT_K_VALUES,
    )

    if args.out is None:
        out_path = Path.home() / "Development" / "alpha-engine-docs" / "private" / (
            f"rag-retrieval-eval-{run_date.isoformat()}.md"
        )
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md)
    log.info("Wrote report to %s (%d bytes)", out_path, len(report_md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
