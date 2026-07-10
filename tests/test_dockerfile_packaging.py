"""Packaging guard: every LOCAL module the Lambda image can import — directly
or transitively from any entrypoint COPY'd into the image — must itself be
COPY'd (config#1403, config#1683, config#2132).

The bug class this prevents (repeatedly realized):
- 2026-06: `producers/` was wired into lambda/handler.py but never COPY'd —
  `from producers.runner import run_challengers` ModuleNotFound'd every
  Saturday, silently swallowed, and signals_shadow/ stayed empty for weeks.
- 2026-07-03: `observe_alerts.py` (a repo-ROOT single-file MODULE, imported
  TRANSITIVELY via producers/runner.py) was not COPY'd — the whole challenger
  post-step import-failed again. The v1 guard was blind on both axes: it
  checked only PACKAGES (dir + __init__.py) imported DIRECTLY by
  lambda/handler.py.

The v2 guard closes both blind spots structurally:
- entrypoints are derived FROM the Dockerfile (every `COPY lambda/<x>.py`),
  so a new handler is covered the moment it is added;
- the walk is TRANSITIVE over the local import graph (AST, lazy/function-level
  imports included);
- local single-file root modules (`<name>.py` at repo root) are first-class
  alongside packages.

config#2132: this repo builds TWO separate Lambda images from TWO
Dockerfiles — `Dockerfile` (main image, 10 shared handlers) and
`Dockerfile.alerts` (standalone `lambda/alerts_handler.py` surveillance
Lambda). The v1/v2 guard only ever read `Dockerfile`, leaving
`Dockerfile.alerts` with zero packaging coverage — the exact blind spot
crucible-predictor PR #352 closed for the predictor's single-Dockerfile
image. The guard is now parametrized over every Dockerfile in the repo so
new images are covered without another rewrite.

Note: gitignored-but-local files (private prompts/scoring) may exist in a dev
checkout and not in CI — the walk simply covers whatever is present, so the
local run checks a superset of CI. Both catch this bug class.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# Every Dockerfile that builds a Lambda image, each independently subject to
# the "every local import reachable from a COPY'd entrypoint must itself be
# COPY'd" packaging guard.
_DOCKERFILES = ["Dockerfile", "Dockerfile.alerts"]


def _dockerfile(name: str = "Dockerfile") -> str:
    return (_REPO / name).read_text()


def _image_entrypoints(dockerfile: str) -> list[Path]:
    """Every lambda/*.py handler COPY'd into the image root."""
    eps = [
        _REPO / "lambda" / m.group(1)
        for m in re.finditer(r"^COPY lambda/(\w+\.py)\s", dockerfile, re.M)
    ]
    assert eps, "no `COPY lambda/<handler>.py` lines found — Dockerfile moved?"
    return [p for p in eps if p.exists()]


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


def _resolve_local(name: str) -> list[Path]:
    """Map a top-level import name to the local file(s) that implement it.

    A package resolves to every .py under its directory (so the walk follows
    imports made anywhere inside it); a root module resolves to its file.
    Non-local names (stdlib / site-packages) resolve to [].
    """
    pkg_dir = _REPO / name
    if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
        return sorted(pkg_dir.rglob("*.py"))
    mod = _REPO / f"{name}.py"
    if mod.exists():
        return [mod]
    return []


def _transitive_local_imports(entrypoints: list[Path]) -> set[str]:
    """All local top-level names reachable from the entrypoints."""
    seen_names: set[str] = set()
    seen_files: set[Path] = set()
    frontier: list[Path] = list(entrypoints)
    while frontier:
        f = frontier.pop()
        if f in seen_files:
            continue
        seen_files.add(f)
        for name in _top_level_imports(f):
            files = _resolve_local(name)
            if not files:
                continue
            seen_names.add(name)
            frontier.extend(files)
    return seen_names


def _copied(name: str, dockerfile: str) -> bool:
    if (_REPO / name).is_dir():
        return f"COPY {name}/" in dockerfile
    return f"COPY {name}.py" in dockerfile


@pytest.mark.parametrize("dockerfile_name", _DOCKERFILES)
def test_dockerfile_copies_transitive_local_imports(dockerfile_name: str):
    dockerfile = _dockerfile(dockerfile_name)
    entrypoints = _image_entrypoints(dockerfile)
    missing = sorted(
        name
        for name in _transitive_local_imports(entrypoints)
        if not _copied(name, dockerfile)
    )
    assert not missing, (
        f"{dockerfile_name} is missing COPY for local module(s)/package(s) "
        f"reachable from the image's Lambda entrypoints: {missing}. Add "
        f"`COPY {missing[0]}{'/' if (_REPO / missing[0]).is_dir() else '.py'} "
        "${LAMBDA_TASK_ROOT}/...` — an un-COPY'd import ModuleNotFounds at "
        "runtime (config#1403: producers/ direct-import miss; config#1683: "
        "observe_alerts.py transitive single-file miss; config#2132: "
        "Dockerfile.alerts coverage)."
    )


def test_producers_package_is_copied():
    """Explicit pin for the config#1403 regression."""
    assert "COPY producers/" in _dockerfile(), (
        "producers/ must be COPY'd into the Lambda image — the challenger "
        "research producers (no_agent_quant / single_agent_quant) import-fail "
        "without it and signals_shadow/ goes empty (config#1403)."
    )


def test_observe_alerts_module_is_copied():
    """Explicit pin for the config#1683 regression (transitive single-file
    root module imported by producers/runner.py + scoring/leaderboard_producers.py)."""
    assert "COPY observe_alerts.py" in _dockerfile(), (
        "observe_alerts.py must be COPY'd into the Lambda image — "
        "producers/runner.py imports it at module level, so its absence "
        "import-kills the entire challenger post-step (config#1683)."
    )


def test_guard_catches_the_1683_shape():
    """Self-test: prove the transitive walk actually reaches observe_alerts
    from an image entrypoint (i.e. the guard would have caught 2026-07-03)."""
    names = _transitive_local_imports(_image_entrypoints(_dockerfile()))
    assert "producers" in names, "walk lost the direct producers/ import"
    assert "observe_alerts" in names, (
        "walk failed to reach observe_alerts transitively via producers/ — "
        "the config#1683 blind spot has been re-introduced"
    )


def test_dockerfile_alerts_entrypoint_is_discovered():
    """Explicit pin for config#2132: the alerts image's sole handler must be
    picked up by the same `COPY lambda/<x>.py` entrypoint-discovery regex
    used for the main image, so it rides the shared guard above."""
    entrypoints = _image_entrypoints(_dockerfile("Dockerfile.alerts"))
    assert [p.name for p in entrypoints] == ["alerts_handler.py"], (
        "Dockerfile.alerts entrypoint discovery drifted — expected exactly "
        "lambda/alerts_handler.py"
    )


def test_dockerfile_alerts_copies_its_local_imports():
    """Explicit pin for config#2132: lambda/alerts_handler.py's local imports
    (config, preflight, ops_alerts as of writing) must all be COPY'd into
    Dockerfile.alerts. This is the same assertion the parametrized guard
    above makes generically; kept explicit so a future regression names the
    exact modules involved, mirroring test_producers_package_is_copied /
    test_observe_alerts_module_is_copied for the main image."""
    dockerfile = _dockerfile("Dockerfile.alerts")
    names = _transitive_local_imports(_image_entrypoints(dockerfile))
    assert {"config", "preflight", "ops_alerts"} <= names, (
        "walk failed to reach the expected local imports from "
        "lambda/alerts_handler.py — entrypoint or import graph changed"
    )
    missing = sorted(n for n in names if not _copied(n, dockerfile))
    assert not missing, (
        f"Dockerfile.alerts is missing COPY for: {missing} — an un-COPY'd "
        "import ModuleNotFounds at runtime (config#2132, same class as "
        "config#1403/#1683 but previously unguarded for the alerts image)."
    )
