"""Tests for the ``_lambda_function_exists`` helper in
infrastructure/deploy.sh.

Exercises the helper via subprocess by sourcing deploy.sh and stubbing
the ``aws`` command. Three contracts pinned:

1. Successful get-function → return 0 (caller proceeds to update path).
2. ResourceNotFoundException stderr → return 1 (caller proceeds to
   create path — current behavior).
3. Any other stderr (AccessDenied / 504 / throttle / network) → exit
   the script with stderr surfaced to operator. Pre-fix this class was
   silently swallowed by ``&>/dev/null`` and fell through to
   create-function, surfacing as a misleading
   ``ResourceConflictException: Function already exist``. Closes
   ROADMAP P3 line ~133.

The deploy.sh script does heavy work at top-level (env staging, ECR
login, etc.) that we don't want to run during tests. To isolate the
helper, we extract just the ``_lambda_function_exists`` function
definition by sed and source THAT into a temporary harness script.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_DEPLOY_SH = Path(__file__).resolve().parent.parent / "infrastructure" / "deploy.sh"


def _extract_helper_to_tmp() -> str:
    """Pull the ``_lambda_function_exists`` function definition out of
    deploy.sh and return its body as a string. Sourcing the full
    deploy.sh would trigger its top-level env staging + ECR login —
    not what we want in unit tests."""
    text = _DEPLOY_SH.read_text()
    start = text.find("_lambda_function_exists() {")
    assert start >= 0, "helper definition not found in deploy.sh"
    # Match the closing brace at column 0 of a line — the helper has
    # no nested unindented braces.
    end = text.find("\n}\n", start)
    assert end >= 0, "helper closing brace not found"
    return text[start:end + 3]


@pytest.fixture
def harness_dir(tmp_path: Path) -> Path:
    """Build an isolated harness: a tmp dir with a fake ``aws`` script
    on PATH and a small bash file that sources the extracted helper
    and calls it once. The fake ``aws`` reads its behavior from env
    vars (``MOCK_AWS_EXIT`` / ``MOCK_AWS_STDERR``) so each test case
    can configure the failure mode it cares about."""
    aws_stub = tmp_path / "aws"
    aws_stub.write_text(
        "#!/usr/bin/env bash\n"
        "if [ -n \"${MOCK_AWS_STDERR:-}\" ]; then\n"
        "  echo \"$MOCK_AWS_STDERR\" >&2\n"
        "fi\n"
        "exit \"${MOCK_AWS_EXIT:-0}\"\n"
    )
    aws_stub.chmod(aws_stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    helper_body = _extract_helper_to_tmp()
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"  # -e disabled so we can observe rc=1 from helper
        "REGION=us-east-1\n"
        + helper_body + "\n"
        + "_lambda_function_exists \"$1\"\n"
        + "echo \"RC=$?\"\n"
    )
    harness.chmod(harness.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return tmp_path


def _run(harness_dir: Path, fn_name: str, *,
         exit_code: int = 0, stderr_text: str = "") -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Prepend our stub aws to PATH.
    env["PATH"] = f"{harness_dir}:{env['PATH']}"
    env["MOCK_AWS_EXIT"] = str(exit_code)
    env["MOCK_AWS_STDERR"] = stderr_text
    return subprocess.run(
        ["bash", str(harness_dir / "harness.sh"), fn_name],
        env=env, capture_output=True, text=True,
    )


# ── Contract 1: success → return 0 ────────────────────────────────────────


def test_success_returns_zero(harness_dir: Path) -> None:
    """get-function returns 0 → helper returns 0 (caller goes to
    update path). Stdout swallowed by helper; only RC line surfaces."""
    result = _run(harness_dir, "alpha-engine-research-eval-judge", exit_code=0)
    assert result.returncode == 0
    assert "RC=0" in result.stdout


# ── Contract 2: ResourceNotFound → return 1 ───────────────────────────────


def test_resource_not_found_returns_one(harness_dir: Path) -> None:
    """ResourceNotFoundException stderr → helper returns 1 (caller goes
    to create path). The harness wrapper prints RC=1 and exits 0."""
    result = _run(
        harness_dir, "alpha-engine-research-new-fn",
        exit_code=255,
        stderr_text="An error occurred (ResourceNotFoundException) when calling the GetFunction operation: Function not found: arn:...",
    )
    assert result.returncode == 0  # harness reaches its echo
    assert "RC=1" in result.stdout


def test_function_not_found_phrase_alone_returns_one(harness_dir: Path) -> None:
    """Some AWS SDK versions emit ``Function not found`` without the
    full exception name. Helper accepts either phrasing."""
    result = _run(
        harness_dir, "alpha-engine-research-new-fn",
        exit_code=255,
        stderr_text="Function not found: arn:aws:lambda:...",
    )
    assert result.returncode == 0
    assert "RC=1" in result.stdout


# ── Contract 3: any other failure → fail loud ─────────────────────────────


def test_access_denied_fails_loud(harness_dir: Path) -> None:
    """AccessDenied was the original incident's silent failure mode
    (alpha-engine-data#149). Pre-fix it fell through to create-function
    and surfaced as ``Function already exist``. Post-fix the script
    exits with the real error visible."""
    result = _run(
        harness_dir, "alpha-engine-research-eval-judge",
        exit_code=255,
        stderr_text="An error occurred (AccessDeniedException) when calling the GetFunction operation: User: arn:aws:iam::... is not authorized to perform: lambda:GetFunction",
    )
    assert result.returncode != 0
    assert "AccessDenied" in result.stderr
    assert "ResourceConflictException" not in result.stderr
    # Helper hint about the cause is surfaced.
    assert "AccessDenied" in result.stderr
    assert "lambda:GetFunction" in result.stderr or "AccessDenied" in result.stderr


def test_504_gateway_timeout_fails_loud(harness_dir: Path) -> None:
    """The 2026-05-08 eval-judge deploy hit AWS 504 Gateway Timeout,
    exhausted retries. Pre-fix the script swallowed it and fell
    through to create-function, surfacing as ``Function already
    exist``. Post-fix the operator sees the real cause + a retry hint."""
    result = _run(
        harness_dir, "alpha-engine-research-eval-judge",
        exit_code=255,
        stderr_text="aws: [ERROR]: An error occurred (504) when calling the GetFunction operation (reached max retries: 2): Gateway Timeout",
    )
    assert result.returncode != 0
    assert "504" in result.stderr or "Gateway Timeout" in result.stderr
    # Operator hint mentions retry for transient.
    assert "retry" in result.stderr.lower() or "transient" in result.stderr.lower()


def test_throttle_fails_loud(harness_dir: Path) -> None:
    """Throttle errors are similar — transient, retry-able. The helper
    surfaces stderr and exits non-zero rather than silently masking."""
    result = _run(
        harness_dir, "alpha-engine-research-eval-judge",
        exit_code=255,
        stderr_text="An error occurred (TooManyRequestsException) when calling the GetFunction operation: Rate exceeded",
    )
    assert result.returncode != 0
    assert "TooManyRequests" in result.stderr or "Rate exceeded" in result.stderr


def test_unknown_error_fails_loud_with_stderr(harness_dir: Path) -> None:
    """Defensive: any unrecognized stderr surface fails loud with the
    raw error text + retry hint, never silently masks. Catches future
    AWS error classes we haven't enumerated."""
    result = _run(
        harness_dir, "alpha-engine-research-eval-judge",
        exit_code=1,
        stderr_text="Connection refused",
    )
    assert result.returncode != 0
    assert "Connection refused" in result.stderr
