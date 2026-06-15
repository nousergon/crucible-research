"""Pin ``requirements.txt`` + ``Dockerfile`` + ``Dockerfile.alerts`` to the
same alpha-engine-lib version.

The Dockerfile strips alpha-engine-lib from ``requirements.txt`` before
``pip install`` (see the ``grep -vE ...alpha-engine-lib`` line in the
Dockerfile RUN block) and instead installs the lib via a hardcoded
``pip install "alpha-engine-lib@vX.Y.Z"`` line ABOVE that grep. So
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
    r"alpha-engine-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"
)
# Dockerfile pin lives inside a quoted RUN argument.
_DOCKERFILE_PIN_RE = re.compile(
    r'"alpha-engine-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"'
)


def _read_pin(filename: str, regex: re.Pattern[str]) -> str:
    text = (_REPO_ROOT / filename).read_text()
    match = regex.search(text)
    assert match is not None, (
        f"could not find alpha-engine-lib pin in {filename} — pattern "
        f"{regex.pattern!r} matched nothing"
    )
    return match.group(1)


def test_requirements_and_dockerfile_pins_match():
    """All three files must pin alpha-engine-lib to the same tag."""
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
        f"alpha-engine-lib pin drift across deploy artifacts:\n"
        + "\n".join(f"  {name}: {pin}" for name, pin in pins.items())
        + "\n\nAll three must move in lockstep — bumping requirements.txt "
        f"alone does NOT propagate to the Lambda image because the Dockerfile "
        f"strips the lib pin from requirements.txt before pip install."
    )
