"""Packaging guard: every local package the Lambda handler imports must be
COPY'd into the image (config#1403).

The bug this prevents: `producers/` (the challenger research producers) was added
and wired into lambda/handler.py, but the Dockerfile never COPY'd it — so
`from producers.runner import run_challengers` raised ModuleNotFoundError on
every Saturday run, silently swallowed by the best-effort guard, and
signals_shadow/ stayed empty for weeks. A static check catches the whole
orphaned-package bug class at CI time instead of at runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _top_level_imports(py_path: Path) -> set[str]:
    tree = ast.parse(py_path.read_text())
    tops: set[str] = set()
    for node in ast.walk(tree):  # walk → catches function-level/lazy imports too
        if isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                tops.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                tops.add(alias.name.split(".")[0])
    return tops


def _local_packages_imported(py_path: Path) -> set[str]:
    """Top-level names imported by py_path that are local PACKAGES in the repo
    (a directory with __init__.py)."""
    return {
        t for t in _top_level_imports(py_path)
        if (_REPO / t).is_dir() and (_REPO / t / "__init__.py").exists()
    }


def test_dockerfile_copies_local_packages_imported_by_handler():
    dockerfile = (_REPO / "Dockerfile").read_text()
    handler = _REPO / "lambda" / "handler.py"
    missing = sorted(
        pkg for pkg in _local_packages_imported(handler)
        if f"COPY {pkg}/" not in dockerfile
    )
    assert not missing, (
        "Dockerfile is missing COPY for local package(s) imported by "
        f"lambda/handler.py: {missing}. Add `COPY {missing[0]}/ "
        "${LAMBDA_TASK_ROOT}/" + (missing[0] if missing else "") + "/` — "
        "an un-COPY'd package ModuleNotFounds silently at runtime "
        "(config#1403: this is exactly how producers/ stayed orphaned)."
    )


def test_producers_package_is_copied():
    """Explicit pin for the config#1403 regression."""
    dockerfile = (_REPO / "Dockerfile").read_text()
    assert "COPY producers/" in dockerfile, (
        "producers/ must be COPY'd into the Lambda image — the challenger "
        "research producers (no_agent_quant / single_agent_quant) import-fail "
        "without it and signals_shadow/ goes empty (config#1403)."
    )
