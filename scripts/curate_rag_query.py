"""Interactive curation helper for ``evals/rag_retrieval_queries.yaml``.

For each entry where ``expected_chunk_ids == ["TODO"]``:

1. Runs vector + keyword retrieval against live Neon (top 10 each).
2. Prints both lists side-by-side with letter shortcuts.
3. Prompts: pick a letter (a-j vector, A-J keyword), paste a UUID
   directly, skip (``x``), or quit (``q``).
4. Writes the picked chunk_id back into the YAML in place.

Re-runnable + saves progress after each pick — kill it any time
without losing earlier work.

Requires ``RAG_DATABASE_URL`` + ``VOYAGE_API_KEY`` in env or .env.

Usage:

    python scripts/curate_rag_query.py
    python scripts/curate_rag_query.py --queries-file custom.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml
from dotenv import dotenv_values

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_CANDIDATES = [
    _REPO_ROOT / ".env",
    Path.home() / "Development" / "alpha-engine-research" / ".env",
]
_DEFAULT_YAML = _REPO_ROOT / "evals" / "rag_retrieval_queries.yaml"

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _load_env() -> None:
    """Populate os.environ from whichever .env we can find."""
    import os
    for path in _ENV_CANDIDATES:
        if path.exists():
            for k, v in dotenv_values(path).items():
                if k in ("RAG_DATABASE_URL", "VOYAGE_API_KEY") and v:
                    os.environ.setdefault(k, v)
            return


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"queries file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data.get("queries"), list):
        raise SystemExit("queries: must be a list")
    return data


def _save_yaml(path: Path, data: dict, header: str) -> None:
    """Write back preserving the file header comments + key order."""
    with path.open("w") as f:
        f.write(header)
        yaml.safe_dump(
            data, f, sort_keys=False, default_flow_style=False,
            allow_unicode=True, width=120,
        )


def _read_header(path: Path) -> str:
    """Read leading ``#`` comment lines + blank lines through the
    first non-comment line. Preserves the doc header on round-trip.
    """
    lines: list[str] = []
    with path.open() as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                lines.append(line)
            else:
                break
    return "".join(lines)


def _print_side_by_side(vec_results, kw_results, query: str) -> None:
    """Render top-10 vector + top-10 keyword candidates side-by-side."""
    print()
    print("=" * 110)
    print(f"QUERY: {query}")
    print("=" * 110)

    def _fmt_row(label: str, r) -> str:
        cid = (r.chunk_id or "")[:8]
        ticker = (r.ticker or "")[:6]
        dtype = (r.doc_type or "")[:18]
        date = str(r.filed_date)[:10]
        score = r.similarity if r.similarity is not None else 0.0
        preview = re.sub(r"\s+", " ", (r.content or ""))[:60]
        return f"[{label}] {cid:<8} {ticker:<6} {dtype:<18} {date} {score:.3f}  {preview!r}"

    vec_letters = "abcdefghij"
    kw_letters = "ABCDEFGHIJ"
    print("\n-- VECTOR top 10 (cosine) --")
    for i, r in enumerate(vec_results[:10]):
        print(_fmt_row(vec_letters[i], r))
    print("\n-- KEYWORD top 10 (FTS ts_rank_cd) --")
    for i, r in enumerate(kw_results[:10]):
        print(_fmt_row(kw_letters[i], r))
    print()


def _prompt_pick(vec_results, kw_results) -> str | None:
    """Returns the picked chunk_id, or None to skip, or 'QUIT'."""
    while True:
        choice = input(
            "Pick: [a-j]=vector  [A-J]=keyword  [paste UUID]  [x]=skip  [q]=quit  > "
        ).strip()
        if choice in ("q", "Q"):
            return "QUIT"
        if choice in ("x", "X", ""):
            return None
        if len(choice) == 1 and choice in "abcdefghij":
            idx = "abcdefghij".index(choice)
            if idx < len(vec_results):
                return vec_results[idx].chunk_id
            print("  (no vector result at that letter)")
            continue
        if len(choice) == 1 and choice in "ABCDEFGHIJ":
            idx = "ABCDEFGHIJ".index(choice)
            if idx < len(kw_results):
                return kw_results[idx].chunk_id
            print("  (no keyword result at that letter)")
            continue
        if _UUID_RE.match(choice):
            return choice
        print(
            "  (invalid — letter a-j or A-J, full UUID, x to skip, q to quit)"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries-file", default=str(_DEFAULT_YAML))
    p.add_argument("--top-k", type=int, default=10,
                   help="Candidates per side (default 10)")
    args = p.parse_args()

    yaml_path = Path(args.queries_file)
    header = _read_header(yaml_path)
    data = _load_yaml(yaml_path)

    _load_env()
    # Defer lib import until env is loaded.
    from alpha_engine_lib.rag import retrieve

    todos = [
        (i, q) for i, q in enumerate(data["queries"])
        if q.get("expected_chunk_ids") == ["TODO"]
        or not q.get("expected_chunk_ids")
    ]
    if not todos:
        print("No TODO entries in", yaml_path)
        return 0

    print(f"Found {len(todos)} TODO entries to curate. Saving after each pick.")
    quit_now = False

    for n, (i, q) in enumerate(todos, start=1):
        if quit_now:
            break
        print(f"\n[{n}/{len(todos)}] category={q['category']}")
        try:
            vec = retrieve(query=q["query"], top_k=args.top_k, method="vector")
            kw = retrieve(query=q["query"], top_k=args.top_k, method="keyword")
        except Exception as e:
            print(f"  retrieve() failed: {type(e).__name__}: {e}")
            continue

        _print_side_by_side(vec, kw, q["query"])
        picked = _prompt_pick(vec, kw)
        if picked == "QUIT":
            print("Quitting.")
            quit_now = True
            break
        if picked is None:
            print("  skipped.")
            continue

        # Write back + persist after each pick — kill-safe.
        data["queries"][i]["expected_chunk_ids"] = [picked]
        _save_yaml(yaml_path, data, header)
        print(f"  picked: {picked}")

    remaining = sum(
        1 for q in data["queries"]
        if q.get("expected_chunk_ids") == ["TODO"]
        or not q.get("expected_chunk_ids")
    )
    done = len(data["queries"]) - remaining
    print(f"\n{done}/{len(data['queries'])} queries curated. {remaining} TODO remaining.")
    if remaining == 0:
        print("All done — run: python scripts/run_rag_retrieval_eval.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
