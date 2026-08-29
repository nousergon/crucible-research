"""Repo-wide guard against a guarded FIRST-PARTY import swallowing a
rename into permanent silence (alpha-engine-config-I9339).

THE CLASS. ``try: from <first-party module> import X except ImportError:
...`` makes a MISSING SYMBOL and a MISSING MODULE raise the exact same
exception. If X is ever renamed, moved, or removed, the guard swallows
that forever. ``scoring/technical.py::_resolve_team_id`` did exactly
this — ``from agents.sector_teams.team_config import SECTOR_TEAM_MAP``
behind ``except ImportError: return None`` — silently returning ``None``
for every sector with nothing logged, which made every
``technical.composite_weights_per_sector`` override in ``scoring.yaml``
stop applying invisibly. The same shape nearly cost the fleet a
delivered verdict in ``scoring/spec_promotion.py`` (alpha-engine-
config-I9278): ``verdict_digest.py`` landed on a sibling branch and
exported a differently-named symbol, and the guard there would have
logged "not present yet" forever with the file sitting right there.
That near-miss produced a single-module AST check
(``tests/test_verdict_digest.py::
test_every_guarded_import_of_this_module_names_a_symbol_that_exists``).
That guard was pinned to one module — this generalises it to every
first-party module in the repo.

For every ``try``-wrapped ``ImportFrom`` whose module resolves to a
file INSIDE this repo, assert every name it imports is actually defined
in that file. A THIRD-PARTY guarded import (``import pandas``,
``from pandas import ...``) is correctly OUT of scope — the risk this
guards against is specific to symbols we own and can rename out from
under a guard; a third-party library's own API stability is a
different, already-versioned contract.

Resolution is via ``ast.parse``, never ``import`` — importing every
first-party module as a side effect of a static-analysis test would be
its own hazard (import ordering, env/secret side effects at collection
time, agent prompt files that raise on import when gitignored content
is absent). A guarded import inside a FUNCTION body (not just module
scope) is walked too — both known live instances guard inside a
function, invisible to a naive module-level-only scan.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Source roots scanned for guarded first-party imports. Deliberately
# excludes tests/ (guard-testing fixtures legitimately construct fake
# ImportErrors to exercise the guarded branch) and anything not
# first-party source (.venv, build output, vendored code).
_SOURCE_DIRS = [
    "agents", "archive", "config", "contracts", "data", "emailer",
    "evals", "graph", "infrastructure", "lambda", "local", "memory",
    "producers", "rag", "scoring", "scripts", "thesis", "thinktank",
]
_ROOT_MODULES = [
    "config.py", "conftest.py", "dry_run.py", "freshness.py",
    "invocation_budget.py", "main.py", "observe_alerts.py",
    "ops_alerts.py", "polygon_client.py", "preflight.py", "retry.py",
    "strict_mode.py",
]


def _iter_source_files():
    for d in _SOURCE_DIRS:
        p = REPO_ROOT / d
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
    for f in _ROOT_MODULES:
        p = REPO_ROOT / f
        if p.is_file():
            yield p


def _module_file(module: str, *, level: int, from_path: pathlib.Path) -> pathlib.Path | None:
    """Resolve a ``from <module> import ...`` to its file, or ``None`` if
    ``module`` is not first-party (third-party / stdlib — out of scope
    by construction: we only ever return a path that actually exists in
    this repo)."""
    if level > 0:
        # Relative import — resolve against the importing file's own
        # package directory. Not seen in this repo's style (everything
        # is absolute from the PYTHONPATH root) but handled for safety.
        base = from_path.parent
        for _ in range(level - 1):
            base = base.parent
        parts = module.split(".") if module else []
    else:
        if not module:
            return None
        parts = module.split(".")
        if parts[0] not in _SOURCE_DIRS and not (REPO_ROOT / f"{parts[0]}.py").is_file():
            return None
        base = REPO_ROOT

    candidate = base.joinpath(*parts) if parts else base
    py_file = candidate.with_suffix(".py")
    if py_file.is_file():
        return py_file
    init_file = candidate / "__init__.py"
    if init_file.is_file():
        return init_file
    return None


def _defined_names(module_path: pathlib.Path) -> set[str]:
    """Top-level names a module makes available, by static AST
    inspection only — functions, classes, module-level assignments, and
    names re-exported via its own top-level imports."""
    tree = ast.parse(module_path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    names.update(e.id for e in t.elts if isinstance(e, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _handles_import_error(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True  # bare `except:` also swallows ImportError.
    candidates = [t] if isinstance(t, ast.Name) else (
        t.elts if isinstance(t, ast.Tuple) else []
    )
    return any(
        isinstance(n, ast.Name) and n.id in ("ImportError", "ModuleNotFoundError")
        for n in candidates
    )


def _find_guarded_first_party_imports() -> list[tuple[pathlib.Path, ast.ImportFrom, pathlib.Path]]:
    """Every ``try``-wrapped ``ImportFrom`` whose module resolves
    first-party, repo-wide, at any nesting depth (module or function
    body)."""
    found: list[tuple[pathlib.Path, ast.ImportFrom, pathlib.Path]] = []
    for path in _iter_source_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if not any(_handles_import_error(h) for h in node.handlers):
                continue
            for stmt in node.body:
                for inner in ast.walk(stmt):
                    if not isinstance(inner, ast.ImportFrom):
                        continue
                    target = _module_file(
                        inner.module or "", level=inner.level, from_path=path,
                    )
                    if target is not None:
                        found.append((path, inner, target))
    return found


def test_every_guarded_first_party_import_names_a_symbol_that_exists():
    """THE guard for this class, repo-wide.

    RED before alpha-engine-config-I9339's fix: ``scoring/technical.py``
    guarded ``SECTOR_TEAM_MAP`` behind ``except ImportError`` even though
    the import cannot legitimately fail (first-party, always bundled) —
    that guard is now deleted, not narrowed, so this sweep's job is to
    make sure no *other* module reintroduces the same shape, and that any
    genuinely-guarded first-party import (``scoring/spec_promotion.py``'s
    sibling-branch ``verdict_digest`` import) actually names a symbol
    that exists.
    """
    instances = _find_guarded_first_party_imports()
    failures = []
    for path, node, target in instances:
        defined = _defined_names(target)
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name not in defined:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}: guarded import of "
                    f"{alias.name!r} from {node.module!r} "
                    f"(resolves to {target.relative_to(REPO_ROOT)}) — no "
                    f"such symbol is defined there. A missing MODULE and a "
                    f"missing SYMBOL raise the same ImportError, so this "
                    f"guard would swallow a rename forever "
                    f"(alpha-engine-config-I9339)."
                )
    assert not failures, "\n" + "\n".join(failures)


def test_sweep_finds_the_known_legitimate_instance():
    """Pins the sweep isn't accidentally matching zero files (a
    trivially-passing empty sweep reads as coverage while covering
    nothing — champion-challenger-policy.md §7.4). As of this PR the one
    legitimate guarded first-party import left in the repo is
    ``scoring/spec_promotion.py``'s sibling-branch ``verdict_digest``
    import (alpha-engine-config-I9278, already covered by its own
    narrower AST test in ``tests/test_verdict_digest.py``); the
    ``technical.py`` instance this issue exists to fix is GONE (import
    moved to module scope, guard deleted, per I9339 deliverable 1)."""
    instances = _find_guarded_first_party_imports()
    resolved = {str(target.relative_to(REPO_ROOT)) for _, _, target in instances}
    assert "scoring/verdict_digest.py" in resolved, sorted(resolved)
    guarded_files = {str(path.relative_to(REPO_ROOT)) for path, _, _ in instances}
    assert "scoring/technical.py" not in guarded_files, (
        "technical.py still guards a first-party import — I9339 expected "
        "the guard deleted, not narrowed, since the import cannot "
        "legitimately fail."
    )
