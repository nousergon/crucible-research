"""CLI entry point for the RAG hybrid-retrieval eval harness.

Reads ``evals/rag_retrieval_queries.yaml``, runs every query through
each (method × vector_weight) condition against the live Neon
pgvector database, and writes the markdown report to
``alpha-engine-docs/private/rag-retrieval-eval-{date}.md`` (default)
or wherever ``--out`` points.

Requires ``RAG_DATABASE_URL`` + ``VOYAGE_API_KEY`` in the environment.
The report is the source of truth for PR 5's default-weight choice.

Usage:

    # Curate evals/rag_retrieval_queries.yaml first (see file header).

    # Run with defaults (writes to alpha-engine-docs/private/...).
    python scripts/run_rag_retrieval_eval.py

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
    """Read the YAML test set + validate schema. Raises on malformed
    entries so the operator fixes curation rather than getting a
    silently-empty report.
    """
    if not path.exists():
        raise FileNotFoundError(f"queries file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    raw_queries = data.get("queries", [])
    if not isinstance(raw_queries, list):
        raise ValueError(f"queries must be a list, got {type(raw_queries).__name__}")

    out: list[EvalQuery] = []
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
        if not expected_ids:
            raise ValueError(
                f"queries[{i}] expected_chunk_ids must be non-empty — "
                f"a query with no expected chunks is a curation error"
            )
        out.append(
            EvalQuery(
                query=query_text,
                expected_chunk_ids=expected_ids,
                category=category,
                note=entry.get("note", ""),
            )
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

    if queries:
        log.info("Running %d queries × %d conditions = %d retrievals",
                 len(queries), len(DEFAULT_CONDITIONS),
                 len(queries) * len(DEFAULT_CONDITIONS))
        results = run_eval(queries=queries, retrieve_fn=live_retrieve)
    else:
        results = []

    run_date = date.today()
    report_md = render_markdown_report(
        run_date=run_date,
        queries=queries,
        results=results,
        conditions=DEFAULT_CONDITIONS,
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
