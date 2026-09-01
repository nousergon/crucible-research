"""Pin ``requirements.txt`` + ``Dockerfile`` + ``Dockerfile.alerts`` to the
same nousergon-lib version.

The Dockerfile strips nousergon-lib from ``requirements.txt`` before
``pip install`` (see the ``grep -vE ...nousergon-lib`` line in the
Dockerfile RUN block) and instead installs the lib via a hardcoded
``pip install "nousergon-lib@vX.Y.Z"`` line ABOVE that grep. So
bumping ``requirements.txt`` alone does NOT propagate to the Lambda
image — the Dockerfile's hardcoded pin wins.

This drift class has bitten production twice:

  - 2026-05-06: ``requirements.txt`` bumped @v0.4.0 → @v0.5.1 but the
    Dockerfile kept installing v0.3.0; Research Lambda canary failed
    with ``ModuleNotFoundError: alpha_engine_lib.agent_schemas``.
  - 2026-05-12: predictor PR #147 bumped ``requirements.txt`` →
    v0.12.0 but missed ``requirements-lambda.txt``; predictor Lambda
    canary failed with ``ModuleNotFoundError: alpha_engine_lib.secrets``.

The Dockerfile comment block warns about this — clearly not enough.
This test re-greps all three files on every CI run so a future
single-file bump fails here, not in a canary.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_REQUIREMENTS_PIN_RE = re.compile(
    r"nousergon-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"
)
# Dockerfile pin lives inside a quoted RUN argument.
_DOCKERFILE_PIN_RE = re.compile(
    r'"nousergon-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"'
)


def _read_pin(filename: str, regex: re.Pattern[str]) -> str:
    text = (_REPO_ROOT / filename).read_text()
    match = regex.search(text)
    assert match is not None, (
        f"could not find nousergon-lib pin in {filename} — pattern "
        f"{regex.pattern!r} matched nothing"
    )
    return match.group(1)


def test_requirements_and_dockerfile_pins_match():
    """All three files must pin nousergon-lib to the same tag."""
    req_pin = _read_pin("requirements.txt", _REQUIREMENTS_PIN_RE)
    main_pin = _read_pin("Dockerfile", _DOCKERFILE_PIN_RE)
    alerts_pin = _read_pin("Dockerfile.alerts", _DOCKERFILE_PIN_RE)

    pins = {
        "requirements.txt": req_pin,
        "Dockerfile": main_pin,
        "Dockerfile.alerts": alerts_pin,
    }
    unique = set(pins.values())
    assert len(unique) == 1, (
        "nousergon-lib pin drift across deploy artifacts:\n"
        + "\n".join(f"  {name}: {pin}" for name, pin in pins.items())
        + "\n\nAll three must move in lockstep — bumping requirements.txt "
        "alone does NOT propagate to the Lambda image because the Dockerfile "
        "strips the lib pin from requirements.txt before pip install."
    )


# ── A commit-SHA pin is a CI failure here, not a Step Functions degradation ──
#
# alpha-engine-config-I7301 deliverable 3. `crucible-predictor` pinned
# nousergon-lib to a bare commit SHA on 2026-07-31 (crucible-predictor#422).
# The commit was not an ancestor of `main` — the branch was squash-merged — so
# production installed a tree that never landed, and co-install parity with
# `crucible-backtester` went unverified for thirteen days. Nothing in CI said
# so. The first signal was the weekly Step Function's `LibPinDriftCheck`
# reporting `reason: sha_pinned` and degrading, which is the wrong layer: a
# committed repo state that will recur on every run until a human edits the
# file is a CI failure, not a runtime gate that fires forever and stops being
# read.
#
# The tag-shaped `_read_pin` assertions above already fail on a SHA pin, but
# they fail as *"could not find nousergon-lib pin"* — a message that reads like
# a moved file or a renamed dependency and sends the reader looking for the
# wrong thing. This test names the actual condition, so the red build states
# the fix.
_SHA_PIN_RE = re.compile(
    r"nousergon-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/"
    r"nousergon-lib@([0-9a-f]{7,40})\b"
)

_PIN_FILES = ("requirements.txt", "Dockerfile", "Dockerfile.alerts")


def test_no_deploy_artifact_pins_the_lib_by_commit_sha():
    """Every nousergon-lib pin is a `vX.Y.Z` release tag.

    A commit SHA is not comparable to another repo's tag or to the weekly
    pipeline's `MIN_LIB_VERSION` floor, so it makes the cross-repo lib-pin
    invariant permanently unverifiable rather than merely unsatisfied.
    """
    offenders = {
        name: match.group(1)
        for name in _PIN_FILES
        if (match := _SHA_PIN_RE.search((_REPO_ROOT / name).read_text()))
    }
    assert not offenders, (
        "nousergon-lib is pinned by commit SHA in:\n"
        + "\n".join(f"  {name}: {sha}" for name, sha in offenders.items())
        + "\n\nPin a released vX.Y.Z tag instead. A SHA pin cannot be compared "
        "to another repo's pin or to the weekly pipeline's MIN_LIB_VERSION "
        "floor, so `LibPinDriftCheck` reports `sha_pinned` and DEGRADES the "
        "Saturday run on every invocation until the file is edited "
        "(alpha-engine-config-I7301)."
    )
