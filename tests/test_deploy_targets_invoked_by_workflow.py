"""CI guard: every ``infrastructure/deploy.sh`` dispatch target must be
invoked by ``.github/workflows/deploy.yml`` (or be explicitly annotated
manual).

Motivation (config#993 — "merged-but-not-deployed", recurrence #3): a
deploy.sh ``case`` target gets added but the matching
``run: bash infrastructure/deploy.sh <target>`` step never lands in
deploy.yml, so the Lambda silently rots while its siblings redeploy on
every merge. This bit ``eval_rolling_mean`` for 5+ weeks (the perpetual
quality-floor ALARM, found 2026-06-11) and the historical
``eval_judge_batch`` case documented inside deploy.yml itself.

Contract pinned here:

1. Parse every dispatch target from deploy.sh's ``case "$TARGET" in``
   block (the leftmost ``token)`` labels), skipping the ``*)`` wildcard.
2. Parse every target invoked by a
   ``run: bash infrastructure/deploy.sh <target>`` line in deploy.yml.
3. Every deploy.sh target must EITHER be invoked by deploy.yml OR carry
   an inline ``# ci-deploy-guard: manual`` annotation on its case line
   (deliberately-manual / aggregate convenience targets).

The allowlist is annotation-driven (not a hardcoded set in this test) so
that the decision to skip a target lives next to the target itself, in
deploy.sh, where the author adding it will see it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"
_DEPLOY_YML = _REPO_ROOT / ".github" / "workflows" / "deploy.yml"

# Inline annotation that marks a deploy.sh target as deliberately not
# auto-invoked by deploy.yml (manual-only or an aggregate convenience
# target like ``both`` / ``all``).
_MANUAL_ANNOTATION = "ci-deploy-guard: manual"

# Matches a case-dispatch label line, e.g.
#   ``  eval_rolling_mean)     deploy_eval_rolling_mean ;;``
# Captures the target token. Anchored to indentation + ``token)`` so it
# does not match the ``case "$TARGET" in`` header or arbitrary ``)``.
_CASE_LABEL_RE = re.compile(r"^\s+([A-Za-z0-9_]+)\)")

# Matches a workflow invocation, e.g.
#   ``run: bash infrastructure/deploy.sh eval_rolling_mean``
_INVOKE_RE = re.compile(r"bash\s+infrastructure/deploy\.sh\s+([A-Za-z0-9_]+)")


def _parse_case_targets() -> dict[str, bool]:
    """Return {target: is_manual_annotated} for every label in the
    deploy.sh ``case "$TARGET" in`` block. Skips the ``*)`` wildcard
    default."""
    text = _DEPLOY_SH.read_text().splitlines()
    targets: dict[str, bool] = {}
    in_case = False
    for line in text:
        stripped = line.strip()
        if stripped.startswith('case "$TARGET"') or stripped.startswith("case $TARGET"):
            in_case = True
            continue
        if not in_case:
            continue
        if stripped == "esac":
            break
        if stripped.startswith("*)"):
            continue  # wildcard default — not a real target
        m = _CASE_LABEL_RE.match(line)
        if m:
            target = m.group(1)
            targets[target] = _MANUAL_ANNOTATION in line
    return targets


def _parse_workflow_invocations() -> set[str]:
    """Return the set of targets invoked by a
    ``bash infrastructure/deploy.sh <target>`` line in deploy.yml.

    Lines inside ``#`` comments are ignored so that a target merely
    *mentioned* in a comment (e.g. the eval_judge_batch backstory) does
    not count as an invocation."""
    invoked: set[str] = set()
    for line in _DEPLOY_YML.read_text().splitlines():
        # Drop comment-only lines and trailing comments before matching.
        code = line.split("#", 1)[0]
        m = _INVOKE_RE.search(code)
        if m:
            invoked.add(m.group(1))
    return invoked


def test_deploy_sh_and_yml_exist() -> None:
    assert _DEPLOY_SH.is_file(), f"missing {_DEPLOY_SH}"
    assert _DEPLOY_YML.is_file(), f"missing {_DEPLOY_YML}"


def test_case_block_parses_known_targets() -> None:
    """Sanity: the parser finds the targets we know exist. Guards
    against a silent regex break that would make the real assertion
    vacuously pass."""
    targets = _parse_case_targets()
    assert "main" in targets
    assert "eval_rolling_mean" in targets
    # The wildcard default must never be treated as a target.
    assert "*" not in targets


def test_workflow_invocations_parse() -> None:
    invoked = _parse_workflow_invocations()
    # main is unconditionally deployed and lives in a code line, not a
    # comment — if this is empty the comment-stripping is too aggressive.
    assert "main" in invoked, (
        "expected deploy.yml to invoke 'main'; invocation parser may be broken"
    )


def test_every_target_invoked_or_marked_manual() -> None:
    """The core guard. Every deploy.sh dispatch target must be invoked by
    deploy.yml unless it carries the ``# ci-deploy-guard: manual``
    annotation on its case line."""
    targets = _parse_case_targets()
    invoked = _parse_workflow_invocations()

    rotting: list[str] = []
    for target, is_manual in targets.items():
        if is_manual:
            continue
        if target not in invoked:
            rotting.append(target)

    assert not rotting, (
        "deploy.sh defines target(s) never invoked by deploy.yml and not "
        f"annotated manual: {sorted(rotting)}. Either add a "
        f"`run: bash infrastructure/deploy.sh <target>` step to "
        f".github/workflows/deploy.yml, or annotate the case line with "
        f"`# {_MANUAL_ANNOTATION}` if the target is deliberately "
        f"manual/aggregate."
    )


def test_manual_annotation_is_honored() -> None:
    """Targets marked manual must be excluded even if uninvoked — this
    is the escape hatch for aggregate convenience targets (both/all) and
    the standalone alerts deploy. Asserts at least one such annotation
    exists so the mechanism is exercised, not dead code."""
    targets = _parse_case_targets()
    manual = [t for t, is_manual in targets.items() if is_manual]
    assert manual, (
        "expected at least one target annotated "
        f"`# {_MANUAL_ANNOTATION}` (e.g. both/all/alerts). If none are "
        "genuinely manual, this assertion can be relaxed."
    )
