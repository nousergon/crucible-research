"""CLI entry point for the RAG hybrid-retrieval eval harness.

Reads ``evals/rag_retrieval_queries.yaml``, runs every query through
each (method × vector_weight × rerank) condition against the live
Neon pgvector database, and writes the markdown report to
``alpha-engine-docs/private/rag-retrieval-eval-{date}.md`` (default)
or wherever ``--out`` points.

Requires ``RAG_DATABASE_URL`` + ``VOYAGE_API_KEY`` in the environment.
Rerank conditions (cross_encoder + llm_judge) additionally require:

    pip install 'alpha-engine-lib[rerank] @ git+https://github.com/nousergon/nousergon-lib@v0.11.0'
    # (LLM-judge path also needs ANTHROPIC_API_KEY)

When the ``[rerank]`` extras aren't installed on the eval runner, pass
``--skip-rerank`` to drop the rerank conditions from the sweep and only
exercise the original hybrid-only matrix. Otherwise the import error
surfaces loudly on the first rerank condition's call.

Resource bounds (added 2026-05-12 after two local-machine lockups):
the cross-encoder model (``BAAI/bge-reranker-v2-m3``, ~600MB resident
with torch + transformers) defaults to pegging every CPU core via
sentence-transformers + torch intra-op parallelism. The CLI applies
``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and ``torch.set_num_threads``
caps (``--limit-threads``, default 2) and offers ``--limit-queries`` +
``--rerank-only`` so the operator can split the heavy CE and LLM-judge
passes across separate processes (each one releases the cross-encoder
when it terminates). For unattended cloud runs use
``infrastructure/spot_rag_eval.sh`` instead — c5.xlarge has the RAM
headroom to lift the thread cap.

The report is the source of truth for the L1303 ROADMAP P1 cutover
decision — flip ``RAG_RERANK=cross_encoder`` (or ``llm_judge``) in
the alpha-engine-config Lambda env iff the rerank conditions show
material recall@10 lift over the hybrid w=0.7 baseline.

Usage:

    # Curate evals/rag_retrieval_queries.yaml first (see file header).

    # Default: full sweep, 2-thread cap, includes both rerank conditions.
    python scripts/run_rag_retrieval_eval.py

    # Drop rerank conditions (operator without [rerank] extras installed).
    python scripts/run_rag_retrieval_eval.py --skip-rerank

    # Local two-pass — splits the heavy work across processes so the
    # CE model and LLM-judge cache lifetimes don't overlap:
    python scripts/run_rag_retrieval_eval.py --rerank-only cross_encoder --out /tmp/ce.md
    python scripts/run_rag_retrieval_eval.py --rerank-only llm_judge   --out /tmp/llm.md

    # Smoke-test the plumbing before the full run:
    python scripts/run_rag_retrieval_eval.py --limit-queries 3 --skip-rerank

    # Lift the thread cap on a dedicated cloud runner:
    python scripts/run_rag_retrieval_eval.py --limit-threads 0
"""

from __future__ import annotations

# Thread caps must be applied BEFORE any import that pulls torch /
# sentence-transformers — OMP_NUM_THREADS / MKL_NUM_THREADS are read
# once at process start. We set conservative defaults here and let the
# CLI flag override them after argparse has run (which is too late for
# OMP but still effective for torch.set_num_threads). The "2" default
# is sized for a 4-core Mac running other dev tools concurrently; cloud
# runners override via ``--limit-threads 0``.
import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "2")
_os.environ.setdefault("MKL_NUM_THREADS", "2")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    Condition,
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


SUPPORTED_RERANK_NAMES: tuple[str, ...] = ("cross_encoder", "llm_judge")


def filter_conditions(
    *,
    skip_rerank: bool,
    rerank_only: str | None,
    source: tuple[Condition, ...] = DEFAULT_CONDITIONS,
) -> tuple[Condition, ...]:
    """Pick which conditions run this invocation.

    Three modes:

    - ``skip_rerank=True`` — drop every rerank condition (operator without
      the ``[rerank]`` extras installed). Returns the 6 hybrid baselines.
    - ``rerank_only=NAME`` — keep all non-rerank conditions PLUS the
      named rerank condition only. Used to split the CE and LLM-judge
      passes across separate processes so each one starts with a clean
      model registry and the cross-encoder weights don't sit resident
      while the LLM-judge condition runs.
    - Both unset — full default sweep (8 conditions).

    The two flags are mutually exclusive; the CLI enforces that and
    argparse surfaces the conflict directly. Defined here (not inline in
    ``main``) so the filtering logic is unit-testable without spinning
    up the live ``retrieve_fn``.
    """
    if skip_rerank and rerank_only is not None:
        raise ValueError(
            "filter_conditions: skip_rerank and rerank_only are mutually exclusive"
        )
    if skip_rerank:
        return tuple(c for c in source if c.rerank is None)
    if rerank_only is not None:
        if rerank_only not in SUPPORTED_RERANK_NAMES:
            raise ValueError(
                f"rerank_only={rerank_only!r} not in {SUPPORTED_RERANK_NAMES}"
            )
        return tuple(
            c for c in source if c.rerank is None or c.rerank == rerank_only
        )
    return source


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


def _apply_torch_thread_cap(limit_threads: int) -> None:
    """Best-effort cap on torch intra-op parallelism.

    Composes with the module-top ``OMP_NUM_THREADS`` env var (which torch
    reads at first import). Calling ``torch.set_num_threads`` after
    import is the documented override path and matches what the
    cross-encoder runner expects. ``limit_threads=0`` opts out entirely
    (cloud runner with dedicated cores). Silently no-ops if torch isn't
    importable — the operator only hits a rerank condition if the
    ``[rerank]`` extras are installed.
    """
    if limit_threads <= 0:
        log.info("Thread cap disabled (--limit-threads 0); using torch defaults.")
        return
    try:
        import torch
    except ImportError:
        log.debug("torch not importable; skipping thread cap (no rerank conditions).")
        return
    torch.set_num_threads(limit_threads)
    log.info(
        "Capped torch.set_num_threads(%d); OMP_NUM_THREADS=%s, MKL_NUM_THREADS=%s.",
        limit_threads,
        _os.environ.get("OMP_NUM_THREADS"),
        _os.environ.get("MKL_NUM_THREADS"),
    )


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
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--skip-rerank",
        action="store_true",
        help=(
            "Drop rerank conditions from the sweep (use when the eval "
            "runner doesn't have the alpha-engine-lib[rerank] extras "
            "installed). Default: include all conditions."
        ),
    )
    rerank_group.add_argument(
        "--rerank-only",
        choices=SUPPORTED_RERANK_NAMES,
        default=None,
        help=(
            "Run only the named rerank condition alongside the non-rerank "
            "baselines. Use to split the CE and LLM-judge passes across "
            "separate processes so each one starts with a clean model "
            "registry. Mutually exclusive with --skip-rerank."
        ),
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help=(
            "Truncate the loaded query set to the first N entries. Use "
            "for a smoke test before the full ~25-query sweep."
        ),
    )
    parser.add_argument(
        "--limit-threads",
        type=int,
        default=2,
        help=(
            "Cap torch intra-op parallelism to N threads (default 2; "
            "0 disables the cap). Sized for a 4-core Mac running other "
            "dev tools concurrently; cloud runners with dedicated cores "
            "should pass 0."
        ),
    )
    args = parser.parse_args()

    queries = load_queries(Path(args.queries_file))
    log.info("Loaded %d queries from %s", len(queries), args.queries_file)
    if args.limit_queries is not None:
        if args.limit_queries <= 0:
            parser.error("--limit-queries must be a positive integer")
        queries = queries[: args.limit_queries]
        log.info("Truncated to first %d queries (--limit-queries).", len(queries))
    if not queries:
        log.warning(
            "No queries curated. Populate %s with at least 30 entries "
            "across the documented categories before re-running.",
            args.queries_file,
        )
        # Still write the placeholder report so the operator sees the
        # empty-state message rather than a missing file.

    conditions = filter_conditions(
        skip_rerank=args.skip_rerank,
        rerank_only=args.rerank_only,
    )
    if args.skip_rerank:
        log.info(
            "Skipping rerank conditions (--skip-rerank); running %d of %d "
            "default conditions.", len(conditions), len(DEFAULT_CONDITIONS),
        )
    elif args.rerank_only is not None:
        log.info(
            "Filtered to rerank=%s only; running %d of %d default conditions.",
            args.rerank_only, len(conditions), len(DEFAULT_CONDITIONS),
        )

    # Apply the torch thread cap before any rerank-enabled retrieve()
    # call loads the cross-encoder weights. No-op when no rerank
    # condition is in play (torch isn't pulled in for vector/keyword).
    _apply_torch_thread_cap(args.limit_threads)

    # Live retrieve_fn — defer the lib import to runtime so the harness
    # is importable for unit tests without RAG_DATABASE_URL set.
    from alpha_engine_lib.rag import retrieve as live_retrieve

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
