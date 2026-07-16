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


_PREVIEW_CHARS = 350
_TOP_N = 5  # show top-5 per side (10 candidates) — tighter than top-10
_VEC_LETTERS = "abcde"
_KW_LETTERS = "ABCDE"


def _print_side_by_side(vec_results, kw_results, query: str) -> None:
    """Render top-N vector + top-N keyword candidates with enough
    preview text to actually judge relevance.
    """
    print()
    print("=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    def _fmt_block(label: str, r) -> str:
        cid = (r.chunk_id or "")[:36]
        ticker = (r.ticker or "")
        dtype = (r.doc_type or "")
        date = str(r.filed_date)[:10]
        section = (r.section_label or "—")[:60]
        score = r.similarity if r.similarity is not None else 0.0
        preview = re.sub(r"\s+", " ", (r.content or ""))[:_PREVIEW_CHARS]
        head = f"[{label}] {ticker} {dtype} {date} | section={section} | score={score:.3f}"
        return f"{head}\n      {cid}\n      {preview}"

    print("\n── VECTOR top 5 (cosine) " + "─" * 73)
    for i, r in enumerate(vec_results[:_TOP_N]):
        print(_fmt_block(_VEC_LETTERS[i], r))
    print("\n── KEYWORD top 5 (FTS ts_rank_cd) " + "─" * 65)
    for i, r in enumerate(kw_results[:_TOP_N]):
        print(_fmt_block(_KW_LETTERS[i], r))
    print()


def _print_full_chunk(r) -> None:
    """Expand a single candidate to full content for closer inspection."""
    print()
    print("─" * 100)
    print(f"FULL CHUNK: {r.chunk_id}")
    print(f"  {r.ticker} {r.doc_type} {r.filed_date} | section={r.section_label}")
    print("─" * 100)
    print(r.content or "(empty content)")
    print("─" * 100)
    print()


def _prompt_pick(vec_results, kw_results) -> str | None:
    """Returns the picked chunk_id, None to skip, or 'QUIT'.

    Commands:
        a-e          pick vector candidate at that letter
        A-E          pick keyword candidate at that letter
        ?<letter>    expand that candidate to full chunk content
        <UUID>       paste a chunk UUID directly (escape hatch)
        x / Enter    skip this query (leaves TODO in place)
        q            quit (already-saved picks preserved)
    """
    while True:
        choice = input(
            "Pick: [a-e]=vec  [A-E]=kw  [?<letter>]=expand  [UUID]  [x]=skip  [q]=quit  > "
        ).strip()
        if choice in ("q", "Q"):
            return "QUIT"
        if choice in ("x", "X", ""):
            return None

        # Expand-to-full-content commands: ?a, ?A, etc.
        if choice.startswith("?") and len(choice) == 2:
            letter = choice[1]
            if letter in _VEC_LETTERS:
                idx = _VEC_LETTERS.index(letter)
                if idx < len(vec_results):
                    _print_full_chunk(vec_results[idx])
                    continue
            if letter in _KW_LETTERS:
                idx = _KW_LETTERS.index(letter)
                if idx < len(kw_results):
                    _print_full_chunk(kw_results[idx])
                    continue
            print(f"  (no candidate at letter {letter!r})")
            continue

        if len(choice) == 1 and choice in _VEC_LETTERS:
            idx = _VEC_LETTERS.index(choice)
            if idx < len(vec_results):
                return vec_results[idx].chunk_id
            print("  (no vector result at that letter)")
            continue
        if len(choice) == 1 and choice in _KW_LETTERS:
            idx = _KW_LETTERS.index(choice)
            if idx < len(kw_results):
                return kw_results[idx].chunk_id
            print("  (no keyword result at that letter)")
            continue
        if _UUID_RE.match(choice):
            return choice
        print(
            "  (invalid — letter a-e or A-E, ?<letter> to expand, UUID, "
            "x to skip, q to quit)"
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
    from nousergon_lib.rag import retrieve

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
