"""Class guard: no `except Exception` handler in this repo's core source
may swallow the failure into `logger.debug(...)` or a bare `pass` with
nothing else recording it (alpha-engine-config-I10226, following
`crucible-executor`'s original `alpha-engine-config-I10031`,
`crucible-executor-PR547`).

The detector itself now lives in `nousergon_lib.testing.debug_swallow_guard`
(lifted on second adoption per `policy-shared-code`, `nousergon-lib-PR398`)
-- this file is a thin per-repo call-site: it names the directories that
hold this repo's core source, diffs the live scan against
`.debug-swallow-allowlist.yaml`, and fails the build on drift. See that
module's docstring for the exact AST shape matched.

Unlike `crucible-executor` (a single `executor/` package),
`crucible-research`'s core source is spread across several top-level
package directories plus a handful of scripts directly at the repo root,
so `_SOURCE_DIRS` lists each one explicitly -- `find_debug_only_swallows`
is non-recursive per directory, matching the original `executor/*.py`
shape, so a package with its own subpackages (e.g. `agents/sector_teams/`,
`data/fetchers/`) is listed as its own additional entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nousergon_lib.testing.debug_swallow_guard import (
    check_against_allowlist,
    check_allowlist_entries_self_contained,
    find_debug_only_swallows,
    load_allowlist,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ALLOWLIST_PATH = _REPO_ROOT / ".debug-swallow-allowlist.yaml"

# Every top-level package directory with *.py files directly under it, plus
# the repo root itself (main.py, config.py, preflight.py, ... live there
# unpackaged), plus each nested subpackage that carries its own *.py files
# (find_debug_only_swallows is non-recursive, so each such directory is
# scanned separately). config/prompts/ and config/scoring.yaml are
# proprietary and gitignored, but config/ itself carries no *.py at its top
# level, so it needs no entry here.
_SOURCE_DIRS = [
    _REPO_ROOT,
    _REPO_ROOT / "agents",
    _REPO_ROOT / "agents" / "investment_committee",
    _REPO_ROOT / "agents" / "sector_teams",
    _REPO_ROOT / "data",
    _REPO_ROOT / "data" / "fetchers",
    _REPO_ROOT / "data" / "substrate",
    _REPO_ROOT / "emailer",
    _REPO_ROOT / "evals",
    _REPO_ROOT / "graph",
    _REPO_ROOT / "infrastructure",
    _REPO_ROOT / "lambda",
    _REPO_ROOT / "local",
    _REPO_ROOT / "memory",
    _REPO_ROOT / "producers",
    _REPO_ROOT / "rag",
    _REPO_ROOT / "scoring",
    _REPO_ROOT / "scripts",
    _REPO_ROOT / "thesis",
    _REPO_ROOT / "thinktank",
]


def _all_swallow_sites() -> dict[str, set[int]]:
    live_sites: dict[str, set[int]] = {}
    for source_dir in _SOURCE_DIRS:
        for path, lines in find_debug_only_swallows(source_dir, repo_root=_REPO_ROOT).items():
            if lines:
                live_sites.setdefault(path, set()).update(lines)
    return live_sites


def test_no_new_debug_only_swallows_outside_allowlist():
    """Every debug-only-or-pass `except Exception` swallow in this repo's
    core source is either fixed (raised, or recorded at WARNING/ERROR+) or
    has a non-expired, matching entry in `.debug-swallow-allowlist.yaml`."""
    live_sites = _all_swallow_sites()
    allowlist = load_allowlist(_ALLOWLIST_PATH)
    failures = check_against_allowlist(live_sites, allowlist)
    assert not failures, "\n".join(failures)


def test_allowlist_entries_are_self_contained():
    """Every entry names a reason, an expiry, and a tracking issue -- a
    swallow with no named recording surface is not a swallow, it is a
    deletion (alpha-engine-config-I10226 deliverable, mirroring
    alpha-engine-config-I10031 deliverable 2)."""
    allowlist = load_allowlist(_ALLOWLIST_PATH)
    failures = check_allowlist_entries_self_contained(allowlist)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
