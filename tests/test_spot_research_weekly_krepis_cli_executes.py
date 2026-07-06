"""The spot_research_weekly.sh launcher's ``-m krepis.*`` CLI callsites must
EXECUTE, not merely import (config#1687; the config#1646/#1649 guard class).

Mirrors ``alpha-engine-predictor/tests/test_spot_train_krepis_cli_executes.py``.
Context: nousergon-lib turned ``nousergon_lib.ec2_spot`` / ``ssm_dispatcher`` /
``ssm_log_capture`` into guard-less re-export shims. Under ``python -m``
(runpy) such a shim imports, rebinds, falls off the end — **exit 0, nothing
runs, no output** — which is exactly how the 2026-07-03 silent-success
incident hid. This launcher invokes the canonical ``krepis.*`` modules; this
test proves each one's CLI actually parses argv and dispatches, entirely
offline (no AWS calls): a guard-less shim prints NOTHING and exits 0 for a
bare invocation; a real CLI prints usage/errors and/or exits non-zero.

String-pinning alone cannot prove executability (importable-but-not-executable
is the failure mode), so this test both (a) pins that the launcher NAMES the
krepis chokepoints on real callsites and (b) executes each under ``python -m``.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "spot_research_weekly.sh"

# `python -m krepis.{ec2_spot,ssm_dispatcher,ssm_log_capture}` as invoked by
# the launcher (LIB_PYTHON / PYTHON_BIN is a runtime path, not the module).
_MODULE_RE = re.compile(r"-m\s+(krepis\.(?:ec2_spot|ssm_dispatcher|ssm_log_capture))\b")


def _invoked_modules() -> set[str]:
    """Every krepis chokepoint the launcher ACTUALLY invokes on a non-comment
    line (comment-only mentions must not be mistaken for real callsites)."""
    modules: set[str] = set()
    for raw in _SCRIPT.read_text().splitlines():
        if raw.strip().startswith("#"):
            continue
        modules.update(_MODULE_RE.findall(raw))
    return modules


def test_launcher_exists_and_is_syntactically_valid():
    assert _SCRIPT.exists()
    out = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_launcher_invokes_all_three_krepis_chokepoints():
    modules = _invoked_modules()
    assert "krepis.ec2_spot" in modules
    assert "krepis.ssm_dispatcher" in modules
    assert "krepis.ssm_log_capture" in modules


def test_launcher_runs_the_box_entrypoint_not_local_dev_runner():
    text = _SCRIPT.read_text()
    # Root-cause design: the box path reuses the production handler via the box
    # entrypoint — NOT local/run.py (the non-production dev runner missing the
    # FAIL-HARD challenger post-step + prior-population snapshot).
    assert "infrastructure/weekly_box_runner.py" in text
    assert "local/run.py" not in text


def test_launcher_fails_loud_on_empty_instance_id():
    # config#1646: rc=0 with an empty id must fail loud, never record a silent
    # success (the 2026-07-03 guard-less-shim no-op).
    text = _SCRIPT.read_text()
    assert "without an instance id" in text
    assert "config#1646" in text


@pytest.mark.parametrize("module", sorted(["krepis.ec2_spot", "krepis.ssm_dispatcher", "krepis.ssm_log_capture"]))
def test_each_krepis_module_executes_under_runpy(module):
    """A guard-less shim exits 0 with NO output for a bare ``python -m``; a real
    argparse CLI either prints usage/errors or exits non-zero (missing
    subcommand/args). Skips if the module isn't importable in this env."""
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not importable in this environment")
    out = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
    )
    produced_output = bool(out.stdout.strip() or out.stderr.strip())
    # Executing CLI: either non-zero exit OR it printed something. The shim
    # signature (exit 0 AND empty output) must NOT occur.
    assert out.returncode != 0 or produced_output, (
        f"{module} behaved like a guard-less shim (exit 0, no output) under python -m"
    )


def test_launcher_stages_private_research_config_for_the_box():
    """A fresh public clone has NO prompts and NO real YAMLs (gitignored;
    prompt_loader hard-fails with no .example fallback — the 2026-04-11
    silent-sample-fallback incident class). The launcher MUST stage the
    private config-repo research/ subtree and the box MUST extract it to
    the prompt_loader/config.py HOME-sibling search path, with deploy.sh
    parity hard-fails on both sides (config#1687 pre-rehearsal review)."""
    sh = _SCRIPT.read_text()
    # Dispatcher side: source check hard-fails without prompts, then stages
    # ONE tarball (single-key GetObject on the spot; no ListBucket needed).
    assert 'RESEARCH_CONFIG_SRC="/home/ec2-user/alpha-engine-config/research"' in sh
    assert "research prompts not found" in sh
    assert "research-config.tgz" in sh
    # Box side: extract to search path #1 and verify prompts landed.
    assert "mkdir -p /home/ec2-user/alpha-engine-config/research" in sh
    assert "tar -xzf /tmp/research-config.tgz -C /home/ec2-user/alpha-engine-config/research" in sh
    assert "staged prompts missing after extract" in sh
