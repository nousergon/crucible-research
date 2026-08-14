"""``_spot_failure_reason`` must correctly classify both `--json` outcomes.

BACKGROUND

``krepis.ec2_spot relaunch-decision`` grew ``--json`` (krepis-PR133, released
as krepis 0.51.0). With ``--json``, the verdict is a JSON field
(``{"relaunch": bool, ...}``) on stdout, and the CLI exits ``0`` whenever it
reached ANY decision — hold included. A non-zero exit with ``--json`` means
only "the CLI could not answer", never a verdict (alpha-engine-config-I7009).

This retires this repo's previous exit-code contract
(``NO_RELAUNCH_EXIT_CODE`` = 75 = hold, 0 = relaunch) in favour of reading the
JSON payload. The caller-side errexit-suppression convention this test
originally guarded —

    reason="$(_spot_failure_reason "$rc")" || reason=""

— is UNCHANGED and still load-bearing: it is a property of this function's
own return-1/return-0 contract to its caller, not of how the function talks
to the CLI. This file keeps exercising that convention (errexit ACTIVE at the
call site, no ``||`` to hide behind) while updating what the stubbed CLI
speaks.

WHAT THIS TEST ASSERTS

(a) The CLI answers "hold" via ``--json`` (``{"relaunch": false, ...}``,
    exit 0) -> the function returns 1 (not a reclaim; caller proceeds to
    terminate, no relaunch).
(b) The CLI itself fails to answer (non-zero exit, no reliable JSON) -> the
    function ALSO returns 1. A CLI failure is not evidence of a reclaim, so
    it is treated the same as hold, not retried as if it were confirmed.
(c) The CLI answers "relaunch" via ``--json`` (``{"relaunch": true, ...}``,
    exit 0) -> the function returns 0 and echoes ``confirmed-reclaim``.

Demonstrated against the pre-migration file (stash, run, pop) that (a) and
(b) both FAIL there: the old code read the CLI's *exit code* as the verdict,
so a ``--json`` invocation returning ``rc=0`` with a hold payload was
misread as a confirmed reclaim.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"

#: (script, instance-id variable read by that copy of the function).
_LAUNCHERS = (("spot_research_weekly.sh", "INSTANCE_ID"),)


def _function_text(source: str, name: str) -> str:
    """Return a shell function's full text, brace-matched."""
    marker = "\n" + name + "() {"
    assert marker in source, f"{name}() not found"
    start = source.index(marker) + 1
    depth = 0
    for idx in range(start, len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"unbalanced braces in {name}()")


@pytest.fixture(autouse=True)
def _requires_bash():
    if shutil.which("bash") is None:  # pragma: no cover - bash is a hard dep
        pytest.skip("bash unavailable")


def _stub_python(tmp_path: Path, name: str, *, json_body: str, rc: int) -> Path:
    """A stand-in for $LIB_PYTHON.

    Intercepts the ``-m krepis.ec2_spot relaunch-decision`` invocation (prints
    the canned ``--json`` payload, exits ``rc``) and delegates every other
    invocation — notably the function's own ``-c '...json.load...'`` parse
    step — to the real ``python3`` so the JSON-decoding logic under test
    actually runs.
    """
    stub = tmp_path / name
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "krepis.ec2_spot" ]; then\n'
        f"  printf '%s\\n' '{json_body}'\n"
        f"  exit {rc}\n"
        "else\n"
        '  exec python3 "$@"\n'
        "fi\n"
    )
    stub.chmod(0o755)
    return stub


def _run(
    script_name: str,
    instance_var: str,
    tmp_path: Path,
    lib_python: Path,
    *,
    rc_arg: int = 3,
) -> subprocess.CompletedProcess:
    script = _INFRA / script_name
    lifted = _function_text(script.read_text(), "_spot_failure_reason")

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        # errexit ACTIVE at the call — no `|| reason=""` to hide behind. This
        # is the whole point: the function must be correct on its own terms.
        "set -euo pipefail\n"
        f"{lifted}\n"
        "AWS_REGION=us-east-1\n"
        f"{instance_var}=i-0000000000test0000\n"
        "MAX_RUNTIME_SECONDS=5400\n"
        "SF_EXECUTION_TIMEOUT=''\n"
        "SPOT_ATTEMPT=1\n"
        "MAX_SPOT_ATTEMPTS=2\n"
        f"LIB_PYTHON={lib_python}\n"
        # rc passed to the function — a workload failure, not the
        # launch-capacity 64 shortcut.
        f"_spot_failure_reason {rc_arg}\n"
        "echo UNREACHABLE_IF_NOT_A_RECLAIM\n"
    )
    harness.chmod(0o755)

    return subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path)},
        timeout=60,
    )


@pytest.mark.parametrize(
    ("script_name", "instance_var"),
    _LAUNCHERS,
    ids=[s for s, _ in _LAUNCHERS],
)
def test_hold_verdict_via_json_is_not_a_reclaim(
    script_name: str, instance_var: str, tmp_path: Path
) -> None:
    """CLI answers via --json, exit 0, relaunch=false -> hold, return 1."""
    lib_python = _stub_python(
        tmp_path,
        "hold-python",
        json_body='{"relaunch": false, "reason": "other"}',
        rc=0,
    )
    proc = _run(script_name, instance_var, tmp_path, lib_python)

    assert proc.returncode != 0, (
        "a --json hold verdict (relaunch=false, rc=0) must not read as a "
        f"reclaim.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "confirmed-reclaim" not in proc.stdout, proc.stdout
    assert "UNREACHABLE_IF_NOT_A_RECLAIM" not in proc.stdout, proc.stdout


@pytest.mark.parametrize(
    ("script_name", "instance_var"),
    _LAUNCHERS,
    ids=[s for s, _ in _LAUNCHERS],
)
def test_cli_failure_is_treated_as_hold_not_a_reclaim(
    script_name: str, instance_var: str, tmp_path: Path
) -> None:
    """CLI itself fails (non-zero exit) -> could not answer, treat as hold."""
    lib_python = _stub_python(
        tmp_path,
        "fail-python",
        json_body="",
        rc=1,
    )
    proc = _run(script_name, instance_var, tmp_path, lib_python)

    assert proc.returncode != 0, (
        "a CLI failure to answer (non-zero exit under --json) must be "
        "treated as hold, never as a confirmed reclaim.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "confirmed-reclaim" not in proc.stdout, proc.stdout
    assert "UNREACHABLE_IF_NOT_A_RECLAIM" not in proc.stdout, proc.stdout


@pytest.mark.parametrize(
    ("script_name", "instance_var"),
    _LAUNCHERS,
    ids=[s for s, _ in _LAUNCHERS],
)
def test_relaunch_verdict_via_json_is_a_confirmed_reclaim(
    script_name: str, instance_var: str, tmp_path: Path
) -> None:
    """CLI answers via --json, exit 0, relaunch=true -> confirmed reclaim."""
    lib_python = _stub_python(
        tmp_path,
        "relaunch-python",
        json_body='{"relaunch": true, "reason": "spot-reclaim"}',
        rc=0,
    )
    proc = _run(script_name, instance_var, tmp_path, lib_python)

    assert proc.returncode == 0, (
        f"a --json relaunch verdict must return 0.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "confirmed-reclaim" in proc.stdout, proc.stdout
