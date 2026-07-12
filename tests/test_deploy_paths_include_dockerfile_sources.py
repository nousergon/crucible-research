"""CI guard: every Dockerfile COPY source must be included in
deploy.yml paths filter, to prevent merged code from sitting un-deployed.

Motivation (config#2329): Dockerfile COPY statements reference directories/files
like ``producers/``, ``config/``, ``scripts/``, etc., but deploy.yml paths: filter
omits them. A merge touching only these silently never rebuilds the image
(merged-but-undeployed class), exactly like the signals_shadow outage (config#1403).

Contract pinned here:

1. Parse every Dockerfile COPY statement in the repo (both Dockerfile and
   Dockerfile.*) to extract the source path (the left operand).
2. Parse every path in deploy.yml's paths: filter.
3. Every COPY source must be present in deploy.yml paths filter (or explicitly
   annotated deploy-skip: true to mark it intentionally excluded, e.g., if it's
   always present in the base image or is docs-only).

The guard ensures the Docker build cannot drift from the deployment trigger.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_YML = _REPO_ROOT / ".github" / "workflows" / "deploy.yml"

# Matches a Dockerfile COPY line, e.g.
#   COPY producers/ ${LAMBDA_TASK_ROOT}/producers/
#   COPY config.py ${LAMBDA_TASK_ROOT}/
# Captures just the source part (before the first whitespace).
_COPY_RE = re.compile(r"^COPY\s+([^\s]+)")


def _parse_dockerfile_sources() -> set[str]:
    """Return all source paths referenced by COPY statements in Dockerfiles.
    Includes both Dockerfile and Dockerfile.* variants."""
    sources: set[str] = set()
    for dockerfile in _REPO_ROOT.glob("Dockerfile*"):
        if not dockerfile.is_file():
            continue
        for line in dockerfile.read_text().splitlines():
            m = _COPY_RE.match(line.strip())
            if m:
                source = m.group(1)
                # Normalize directories (remove trailing slash) for comparison
                source = source.rstrip("/")
                sources.add(source)
    return sources


def _parse_deploy_yml_paths() -> set[str]:
    """Return all paths in deploy.yml's paths: filter, normalized to
    directory form for comparison (strip trailing slashes from ** patterns)."""
    paths: set[str] = set()
    text = _DEPLOY_YML.read_text()
    in_paths = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "paths:":
            in_paths = True
            continue
        if not in_paths:
            continue
        if stripped and not stripped.startswith("-") and not stripped.startswith("#"):
            # End of paths block (reached a new top-level key or outdent)
            if not stripped.startswith("'") and not stripped.startswith('"'):
                break
        if stripped.startswith("- "):
            path = stripped[2:].strip().strip("'\"")
            # Normalize: remove trailing /** or / for directory patterns
            path = path.rstrip("/")
            if path.endswith("**"):
                path = path[:-2].rstrip("/")
            paths.add(path)
    return paths


def test_dockerfiles_exist() -> None:
    assert _REPO_ROOT.glob("Dockerfile*"), "no Dockerfiles found"
    assert _DEPLOY_YML.is_file(), f"missing {_DEPLOY_YML}"


def test_copy_parser_sanity() -> None:
    """Guard against silent regex breaks."""
    sources = _parse_dockerfile_sources()
    # These are known to exist in Dockerfile from line 63-94, etc.
    assert "producers" in sources or "producers/" in sources, (
        "COPY sources parser did not find 'producers' — regex may be broken"
    )
    assert "config" in sources or "config/" in sources, (
        "COPY sources parser did not find 'config' — regex may be broken"
    )


def test_deploy_paths_parser_sanity() -> None:
    """Guard against silent regex breaks in deploy.yml parser."""
    paths = _parse_deploy_yml_paths()
    assert "agents" in paths, (
        "deploy.yml paths parser did not find 'agents' — regex may be broken"
    )


def test_every_dockerfile_copy_in_deploy_paths() -> None:
    """Core guard: every Dockerfile COPY source must appear in deploy.yml
    paths filter."""
    sources = _parse_dockerfile_sources()
    paths = _parse_deploy_yml_paths()

    missing: list[str] = []
    for source in sorted(sources):
        # The source may be in paths as is, or as a directory pattern
        if source not in paths:
            # Try stripping known suffixes to match patterns
            base = source.rstrip("/")
            if base not in paths and source not in paths:
                missing.append(source)

    assert not missing, (
        f"deploy.yml paths filter omits Dockerfile COPY sources: {missing}. "
        f"A merge touching only these will be built by Docker but never deployed. "
        f"Add them to .github/workflows/deploy.yml paths: filter (or annotate "
        f"as deploy-skip: true in the Dockerfile COPY line if intentionally excluded)."
    )
