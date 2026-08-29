"""Contract tests for ``infrastructure/eval_judge_spot_bootstrap.sh``
(alpha-engine-config-I9329).

The script is SETUP-ONLY: it prepares the eval-judge spot box and leaves it
ALIVE for a LATER ``ssm:sendCommand`` issued by the Step Function, which is
what actually runs ``evals.judge_spot_run``. That inverts the sibling
``thinktank_spot_bootstrap.sh`` contract, whose trap shuts the box down on
every exit path, so the difference is pinned here rather than left to a header
comment: a `shutdown` reintroduced by someone pattern-matching on the sibling
script would kill the SF stage that is about to use the box, and it would do so
only in production — the SF stage would report an unreachable instance and read
as a judge defect.

The venv path assertion exists for the matching reason: the SF's judge command
addresses the interpreter by absolute path, so a bootstrap that builds the venv
somewhere else "succeeds" and leaves a box the run cannot use.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "infrastructure" / "eval_judge_spot_bootstrap.sh"

#: The single source of truth for the venv path in this file. The SF's
#: `ssm:sendCommand` addresses this same absolute path; stated once so the
#: assertions below cannot drift apart from each other.
BOX_REPO_DIR = "/home/ec2-user/crucible-research"
BOX_VENV_DIR = f"{BOX_REPO_DIR}/.venv"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _code_lines() -> list[str]:
    """Script body with comment-only lines stripped.

    Every assertion below is about what the script DOES. The header documents
    the no-shutdown decision in prose and would otherwise trip the very check
    that enforces it.
    """
    return [
        line for line in _script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]


def test_bootstrap_script_exists_and_is_executable():
    assert _SCRIPT.is_file(), f"{_SCRIPT} is missing"
    assert os.access(_SCRIPT, os.X_OK), (
        f"{_SCRIPT} is not executable — the dispatcher prelude execs it "
        "directly"
    )


def test_bootstrap_never_shuts_the_box_down():
    """The deliberate difference from thinktank_spot_bootstrap.sh.

    This script hands a LIVE box to the Step Function, which then issues its
    own `ssm:sendCommand` to run the judge. A `shutdown` on any path — success,
    failure, or trap — pulls the box out from under that stage. Teardown
    belongs to the dispatcher's `krepis.spot_bootstrap` deadman and
    `max_runtime_seconds` timers, which are armed before this script starts.
    """
    offenders = [
        line.strip() for line in _code_lines()
        if re.search(r"\b(shutdown|halt|poweroff)\b", line)
        or re.search(r"\bterminate-instances\b", line)
    ]
    assert offenders == [], (
        "the eval-judge bootstrap must never terminate its own box — the SF "
        f"runs the judge on it afterwards. Found: {offenders}"
    )


def test_bootstrap_installs_gitleaks():
    """The krepis DLP hook fails CLOSED without the binary, so every routed
    LLM call the judge makes would die. Same install as the sibling script.
    """
    text = _script_text()
    assert "gitleaks" in text
    assert "/opt/llm-routing/gitleaks-egress.toml" in text, (
        "gitleaks config must land at session_dlp.py's first standard-path "
        "fallback"
    )
    assert re.search(r"GITLEAKS_SHA256=[0-9a-f]{64}", text), (
        "the gitleaks download must be checksum-verified"
    )


def test_bootstrap_builds_the_venv_where_the_sf_command_looks_for_it():
    code = "\n".join(_code_lines())
    assert 'VENV_DIR:-${REPO_DIR}/.venv' in code or BOX_VENV_DIR in code, (
        f"the venv must resolve to {BOX_VENV_DIR}"
    )
    assert f'REPO_DIR:-{BOX_REPO_DIR}' in code, (
        f"the repo checkout must default to {BOX_REPO_DIR} — the dispatcher "
        "prelude clones there"
    )
    assert "python3.12 -m venv" in code, (
        "the interpreter must be python3.12 strictly; requirements.txt is "
        "resolved against it and the wheels differ"
    )


def test_bootstrap_proves_readiness_before_exiting_zero():
    """A bootstrap that succeeds onto a box the SF's command cannot run on is
    indistinguishable, from the SF's side, from a judge bug.
    """
    code = "\n".join(_code_lines())
    assert "import krepis.ssm_log_capture" in code
    assert "import evals.judge_spot_run" in code
    assert "-m evals.judge_spot_run --help" in code
    assert "READY" in code


def test_bootstrap_requires_the_private_eval_rubrics():
    """The judge's rubrics are ``eval_rubric_*.txt`` in alpha-engine-config
    (``evals/judge.py::resolve_rubric_for_agent``), NOT the ``thinktank_*.txt``
    the sibling script checks. ``load_prompt`` hard-fails on a miss, so the
    absence must be caught here rather than mid-run.
    """
    code = "\n".join(_code_lines())
    assert "eval_rubric_" in code
    assert "research/prompts" in code


def test_bootstrap_fails_loud():
    code = "\n".join(_code_lines())
    assert "set -euo pipefail" in code
    # Only the best-effort outcome REPORTING may swallow (its carve-out is in
    # the header); nothing that prepares the box may.
    assert code.count("|| fail") >= 8, (
        "every load-bearing step needs an explicit `|| fail`"
    )
