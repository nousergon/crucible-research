"""The news-aggregates prefix names a producer that is still WRITING.

alpha-engine-config-I8174. Think Tank self-anchors on two live-producer
inputs (Brian ruling 2026-08-21, I7831); the intraweek leg is the daily news
aggregates. It was wired to ``data/news_aggregates`` — the SATURDAY
full-universe artifact — whose producer was retired on 2026-07-30 by
nousergon-data#1168, when the weekly RAG chain stopped filling the corpus and
began only verifying it (config-I5702, rag-corpus-policy.md §2.3).

Consequence measured 2026-08-22: the prefix's last write was
2026-07-30T17:57:29Z, the daily Think Tank run read 23-day-old news, and run
``7ff54f603bba`` went DEGRADED at 14:32Z then ABORTED at 14:37Z with zero
thesis writes. The live producer — ``collectors/daily_news.py`` in
nousergon-data, writing ``data/news_aggregates_daily`` through the SAME
``aggregate_and_write`` writer — had been healthy the whole time.

This is the exact defect the anchor rewrite existed to remove: an input
pointed at a prefix nothing produces, looking healthy right up until its
freshness verdict is read. Two guards, because the repoint alone would not
have caught it:

1. the prefix constant names the live daily producer, and
2. NOTHING restates that prefix as a literal. The verdict was published under
   ``data/news_aggregates`` while the reader read the same dead key, so the
   label and the key agreed with each other and disagreed with reality. A
   second copy of the string is how a future repoint half-lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

from data.substrate.reader import NEWS_AGGREGATES_PREFIX

#: The retired Saturday full-universe artifact. Nothing may read it or name it.
RETIRED_PREFIX = "data/news_aggregates"

#: The live weekday producer (nousergon-data ``collectors/daily_news.py``).
LIVE_PREFIX = "data/news_aggregates_daily"

_REPO = Path(__file__).resolve().parent.parent


def test_news_prefix_names_the_live_daily_producer():
    assert NEWS_AGGREGATES_PREFIX == LIVE_PREFIX, (
        f"news aggregates are read from {NEWS_AGGREGATES_PREFIX!r}; the only "
        f"live producer writes {LIVE_PREFIX!r}. {RETIRED_PREFIX!r} was retired "
        "2026-07-30 by nousergon-data#1168 and has had no writer since."
    )


def _string_literals(path: Path) -> list[str]:
    """Every string CONSTANT in a module, excluding docstrings.

    Docstrings are excluded deliberately: prose may — and does — name the
    retired prefix to explain why it is retired. What must not exist is a
    second executable copy of the key.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_module_restates_the_news_prefix_as_a_literal():
    """The reader's constant is the single source of truth for this key.

    Both the S3 read and the freshness verdict's artifact identity must derive
    from ``NEWS_AGGREGATES_PREFIX``. When the identity was its own literal, the
    verdict kept publishing ``STALE-INPUT[data/news_aggregates]`` for a key the
    fleet had stopped writing, and the two agreeing strings read as corroboration.
    """
    offenders: list[str] = []
    searched = [
        p
        for d in ("thinktank", "data/substrate", "evals")
        for p in sorted((_REPO / d).rglob("*.py"))
    ]
    assert searched, "found no modules to scan — the guard would pass vacuously"

    for path in searched:
        for literal in _string_literals(path):
            if literal in (RETIRED_PREFIX, LIVE_PREFIX):
                if (
                    path.name == "reader.py"
                    and literal == LIVE_PREFIX
                ):
                    continue  # the one definition
                offenders.append(f"{path.relative_to(_REPO)}: {literal!r}")

    assert not offenders, (
        "the news-aggregates prefix is restated as a literal in:\n  "
        + "\n  ".join(offenders)
        + "\nImport NEWS_AGGREGATES_PREFIX from data.substrate.reader instead — "
        "a second copy is how alpha-engine-config-I8174 outlived its producer."
    )
