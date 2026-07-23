"""CI guard: every Dockerfile COPY source must be included in
deploy.yml paths filter, to prevent merged code from sitting un-deployed.

Motivation (config#2329): Dockerfile COPY statements reference directories/files
like ``producers/``, ``config/``, ``scripts/``, etc., but deploy.yml paths: filter
omits them. A merge touching only these silently never rebuilds the image
(merged-but-undeployed class), exactly like the signals_shadow outage (config#1403).

Contract pinned here:

1. Parse every Dockerfile COPY statement in the repo (both Dockerfile and
   Dockerfile.*) to extract the source path (the left operand).
2. Parse every path PATTERN in deploy.yml's paths: filter, in its raw glob
   form (e.g. ``lambda/**``, ``requirements*.txt``) — the exact string
   GitHub Actions evaluates against changed file paths.
3. Every COPY source must be COVERED by deploy.yml's paths filter: either an
   exact/glob match (file sources, e.g. ``requirements.txt`` matches
   ``requirements*.txt``), or, for directory COPY sources, matched by a
   directory-bucket glob that would catch a file underneath it (e.g.
   ``lambda/handler.py`` is covered by ``lambda/**``) — or explicitly
   annotated deploy-skip: true to mark it intentionally excluded, e.g., if
   it's always present in the base image or is docs-only.

Coverage is evaluated with fnmatch-style glob semantics (mirroring how
GitHub Actions' paths: filter itself matches, where both ``*`` and ``**``
span across ``/``) rather than plain string equality — a purely literal
comparison flags every individually-named file already covered by a
directory-bucket or mid-string-wildcard pattern as "missing", which is a
false positive, not a real un-deployed-code gap. (Found live 2026-07-17:
the original literal-comparison version of this guard failed on 13
pre-existing, already-covered sources — e.g. every lambda/*_handler.py
file under the 'lambda/**' bucket, requirements.txt/-alerts.txt under
'requirements*.txt' — from its own origin commit, well before the 7/14
lambda-handler additions that were assumed to be the sole cause.)

The guard ensures the Docker build cannot drift from the deployment trigger.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

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


def _parse_deploy_yml_paths() -> list[str]:
    """Return the raw glob PATTERNS from deploy.yml's paths: filter, exactly
    as GitHub Actions evaluates them (e.g. ``'agents/**'``,
    ``'requirements*.txt'``, ``'config.py'``) — no normalization, so callers
    can do real glob matching against them."""
    patterns: list[str] = []
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
            patterns.append(stripped[2:].strip().strip("'\""))
    return patterns


def _source_is_covered(source: str, patterns: list[str]) -> bool:
    """True if some deploy.yml path pattern would trigger a rebuild for a
    change to this Dockerfile COPY source.

    Two ways to be covered:
    - As a FILE: the source itself matches a pattern, literally or via glob
      (e.g. source 'requirements.txt' matches pattern 'requirements*.txt').
    - As a DIRECTORY: a synthetic file just inside the source directory
      matches a pattern (e.g. probe 'lambda/__probe__' matches pattern
      'lambda/**'), meaning any real file added under that directory would
      trigger a rebuild too — which is what a directory-bucket COPY source
      (Dockerfile's ``COPY lambda/ ...``, stripped of its trailing slash by
      ``_parse_dockerfile_sources``) actually needs.

    fnmatch is used (not a path-segment-aware matcher) because it mirrors
    GitHub Actions' own paths: filter semantics, where '*' and '**' both
    span '/' freely — this is a deliberate departure from POSIX shell glob.
    """
    probe = source.rstrip("/") + "/__probe__"
    for pattern in patterns:
        if source == pattern or fnmatch.fnmatch(source, pattern):
            return True
        if fnmatch.fnmatch(probe, pattern):
            return True
    return False


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
    patterns = _parse_deploy_yml_paths()
    assert "agents/**" in patterns, (
        "deploy.yml paths parser did not find 'agents/**' — regex may be broken"
    )


def test_every_dockerfile_copy_in_deploy_paths() -> None:
    """Core guard: every Dockerfile COPY source must be covered by deploy.yml
    paths filter (see _source_is_covered for exact/glob/directory-bucket
    matching semantics)."""
    sources = _parse_dockerfile_sources()
    patterns = _parse_deploy_yml_paths()

    missing = [s for s in sorted(sources) if not _source_is_covered(s, patterns)]

    assert not missing, (
        f"deploy.yml paths filter omits Dockerfile COPY sources: {missing}. "
        f"A merge touching only these will be built by Docker but never deployed. "
        f"Add them to .github/workflows/deploy.yml paths: filter (or annotate "
        f"as deploy-skip: true in the Dockerfile COPY line if intentionally excluded)."
    )
