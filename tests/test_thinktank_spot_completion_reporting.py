"""The Think Tank box must report its own terminal outcome before it dies.

alpha-engine-config-I5752. The box self-terminates on every exit path, so
anything not shipped off it before ``shutdown`` is gone. Measured 2026-07-30,
the coverage this closes:

===================  ==============================  ======================
stage                before                          after
===================  ==============================  ======================
dispatcher prelude   log → S3, no alert              unchanged (own trap)
this bootstrap       **nothing** — no log, no alert  log → S3 + SNS on fail
the Python run       flow-doctor Telegram + email    unchanged
overran the cap      reaped, no incomplete-reap row  marker + WatchKind
===================  ==============================  ======================

The middle row was the gap. ``fail()`` in that script is what refuses to start
when the private prompt config is absent — deliberately, before any LLM spend —
and that refusal was silent: stderr died with the box, and the SSM command
status is watched by nothing.

The load-bearing assertion here is **not** that a marker is written. It is that
a marker is written on the SUCCESS PATH ONLY (``principles.md`` §2.7). The
marker is what ``spot-orphan-reaper``'s Think-Tank ``WatchKind`` reads to decide
whether a box it reaped at the fleet age cap had finished; a marker a failed run
wrote would make the failure read as a completed run to the one detector that
can see a hung box.

These run the real ``on_exit`` in a sandbox with ``aws`` and ``shutdown``
stubbed, so they assert the script's behaviour rather than its source text — a
grep-based test would pass on a script whose branches were inverted.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOOTSTRAP = _REPO_ROOT / "infrastructure" / "thinktank_spot_bootstrap.sh"


def _run_on_exit(tmp_path: Path, rc: int) -> tuple[str, list[str]]:
    """Execute the bootstrap's ``on_exit`` with a given exit code.

    The trap body is extracted and sourced rather than re-implemented — a
    hand-copied copy would drift from the thing under test on the first edit.
    Returns ``(stdout, aws_argv_lines)``.
    """
    src = _BOOTSTRAP.read_text(encoding="utf-8")
    start = src.index("SNS_TOPIC_ARN=")
    end = src.index("trap on_exit EXIT")
    prelude = src[start:end]

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    aws_log = tmp_path / "aws.log"
    (bin_dir / "aws").write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> {aws_log}\nexit 0\n')
    (bin_dir / "shutdown").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "hostname").write_text("#!/usr/bin/env bash\necho test-box\n")
    for stub in ("aws", "shutdown", "hostname"):
        (bin_dir / stub).chmod(0o755)

    log_file = tmp_path / "bootstrap.log"
    log_file.write_text("line one\nline two\n")

    script = tmp_path / "harness.sh"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            REGION=us-east-1
            THINKTANK_SPOT_RUN_TOKEN=tok123
            BOOTSTRAP_LOG={log_file}
            log() {{ echo "[thinktank-bootstrap] $*"; }}
            {prelude}
            ( exit {rc} )
            on_exit
            """
        )
    )
    script.chmod(0o755)

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=30)
    calls = aws_log.read_text().splitlines() if aws_log.exists() else []
    return proc.stdout, calls


def test_success_writes_the_completion_marker(tmp_path):
    stdout, calls = _run_on_exit(tmp_path, rc=0)
    marker_writes = [c for c in calls if "thinktank/_control/completed/" in c]
    assert marker_writes, f"no completion marker written on success: {calls}"
    assert "completion marker written" in stdout


def test_failure_writes_NO_completion_marker(tmp_path):
    """principles.md §2.7 — the operative assertion of this module."""
    stdout, calls = _run_on_exit(tmp_path, rc=1)
    marker_writes = [c for c in calls if "thinktank/_control/completed/" in c]
    assert not marker_writes, (
        "a FAILED run wrote a completion marker — spot-orphan-reaper would read "
        f"the failure as a completed run: {marker_writes}"
    )
    assert "NO completion marker" in stdout


def test_failure_publishes_an_sns_alert_naming_the_exit_code(tmp_path):
    """The window flow-doctor cannot cover: a failure before the Python run
    configures it, where stderr dies with the box."""
    _, calls = _run_on_exit(tmp_path, rc=7)
    published = [c for c in calls if c.startswith("sns publish")]
    assert published, f"no SNS alert on failure: {calls}"
    assert "alpha-engine-alerts" in published[0]
    assert "rc=7" in published[0]


def test_success_publishes_no_alert(tmp_path):
    _, calls = _run_on_exit(tmp_path, rc=0)
    assert not [c for c in calls if c.startswith("sns publish")]


@pytest.mark.parametrize("rc", [0, 1])
def test_the_log_ships_on_every_exit_path(tmp_path, rc):
    """A successful run's log is the baseline the next failure gets diffed
    against, so this is not gated on failure."""
    _, calls = _run_on_exit(tmp_path, rc=rc)
    assert [c for c in calls if "_ssm_logs/thinktank-spot/" in c], calls


@pytest.mark.parametrize("rc", [0, 3])
def test_reporting_never_changes_the_run_outcome(tmp_path, rc):
    """`on_exit` re-raises the original rc. A failure to REPORT an outcome must
    never alter it — every reporting step is `|| true` for that reason."""
    src = _BOOTSTRAP.read_text(encoding="utf-8")
    assert 'exit "$rc"' in src
    stdout, _ = _run_on_exit(tmp_path, rc=rc)
    assert f"run finished rc={rc}" in stdout
