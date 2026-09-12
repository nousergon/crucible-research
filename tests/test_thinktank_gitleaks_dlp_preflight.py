"""Contract test for the gitleaks install + DLP preflight gate in
``infrastructure/thinktank_spot_bootstrap.sh`` (alpha-engine-config-I10370).

## Why this exists

``krepis.session_dlp`` (the LLMClient DLP hook) shells out to the gitleaks
BINARY on every LLM call and fails CLOSED when it is absent. This bootstrap
already installed a pinned, checksum-verified gitleaks binary — see the
in-file "DLP scan prerequisite" block — but never asserted the scanner was
actually READY before starting the daily run. The measured failure this
issue was filed from (2026-08-19T14:32Z) was discovered on the box's FIRST
LLM call, deep into the run: ``ThinktankLLMError: ... DLP scan blocked
outbound request: gitleaks binary not found on PATH — failing closed``.

``python -m krepis.session_dlp preflight`` (krepis-PR211) now runs as a BOOT
GATE right after the venv/requirements install and before
``thinktank_box_runner.py`` starts, so a scanner that is not ready aborts
the boot loudly instead.

Style matches ``test_eval_judge_spot_bootstrap.py``'s grep-shaped contract
for the sibling bootstrap in this same repo.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "infrastructure" / "thinktank_spot_bootstrap.sh"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _code_lines() -> list[str]:
    return [
        line for line in _script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]


def test_bootstrap_script_exists():
    assert _SCRIPT.is_file(), f"{_SCRIPT} is missing"


def test_bootstrap_installs_pinned_checksum_verified_gitleaks():
    text = _script_text()
    assert re.search(r"GITLEAKS_VERSION=8\.30\.1\b", text), (
        "gitleaks version must be pinned (8.30.1, matching the fleet standard)"
    )
    assert re.search(r"GITLEAKS_SHA256=[0-9a-f]{64}", text), (
        "the gitleaks download must be checksum-verified, not merely versioned"
    )
    assert re.search(
        r'echo\s+"\$\{GITLEAKS_SHA256\}\s+/tmp/gitleaks\.tar\.gz"\s*\|\s*sha256sum\s+-c\s+-',
        text,
    ), "the downloaded tarball must actually be verified against the pinned sha256"
    assert "gitleaks binary unavailable after install" in text, (
        "the post-install check must be fail-closed, not a silent best-effort"
    )


def test_bootstrap_runs_dlp_preflight_after_deps_before_the_run():
    code = "\n".join(_code_lines())
    deps_idx = code.index("pip install --quiet -r requirements.txt")
    preflight_idx = code.index("python -m krepis.session_dlp preflight")
    run_idx = code.index("python infrastructure/thinktank_box_runner.py")
    assert deps_idx < preflight_idx < run_idx, (
        "the DLP preflight gate must run after krepis is installed "
        "(requirements.txt) and before the daily run starts — running it "
        "earlier would test an environment without krepis installed, and "
        "running it later defeats the point of a BOOT gate"
    )


def test_dlp_preflight_failure_aborts_the_boot():
    """The gate must be load-bearing — a non-zero preflight must fail the
    script loudly (fail-closed), not merely log a warning."""
    m = re.search(
        r"python -m krepis\.session_dlp preflight\s*\|\|\s*fail\s+\"([^\"]*)\"",
        _script_text(),
    )
    assert m, (
        "`python -m krepis.session_dlp preflight` must be guarded by "
        "`|| fail \"...\"` — the same fail-closed pattern every other "
        "load-bearing step in this bootstrap uses"
    )
    assert "DLP preflight" in m.group(1)


def test_dlp_preflight_does_not_read_the_disable_escape_hatch():
    """Non-inferable gotcha named in alpha-engine-config-I10370:
    KREPIS_DLP_DISABLED=1 would make preflight() report readiness while the
    fleet's only DLP control is silently off. This bootstrap must never set
    or read it.
    """
    assert "KREPIS_DLP_DISABLED" not in "\n".join(_code_lines()), (
        "thinktank_spot_bootstrap.sh must not set KREPIS_DLP_DISABLED — that "
        "would silently defeat the preflight gate this test protects"
    )


def test_bootstrap_fails_loud():
    code = "\n".join(_code_lines())
    assert "set -euo pipefail" in code
    assert code.count("|| fail") >= 8, (
        "every load-bearing step, including the new preflight gate, needs an "
        "explicit `|| fail`"
    )
