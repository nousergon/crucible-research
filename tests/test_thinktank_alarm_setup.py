"""Lock the think-tank failure alarm against drift (config#1579 P1;
cadence consolidation 2026-07-14 — alpha-engine-config-I2487 incident +
SOTA follow-up: the standalone EventBridge schedule is retired, the
Lambda is invoked ONLY by the Saturday weekly SF's ThinkTankCoverage
state, mode=sf_cover).

The alarm only works if two things stay in sync:

  1. ``infrastructure/setup-thinktank-alarm.sh`` targets the same
     FunctionName the SF state actually invokes
     (``alpha-engine-research-thinktank``).
  2. The alarm threshold (2/day) matches the SF state's Retry
     configuration — 1 initial invoke + 1 retry on States.Timeout/
     Lambda.Unknown = worst case 2 Errors datapoints for a
     fully-exhausted-retries failure. If the SF Retry's MaxAttempts
     changes, this threshold must change in lockstep.

If either drifts, a failed weekly run either never pages (threshold too
high) or pages on a self-healed single retry (threshold too low) — the
bug classes this pin exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "infrastructure" / "setup-thinktank-alarm.sh"
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_setup_script_exists():
    assert _SCRIPT.is_file()


def test_no_standalone_schedule_script():
    """The EventBridge rule + its setup script are retired — only the SF's
    ThinkTankCoverage state (mode=sf_cover) invokes this Lambda now."""
    assert not (_REPO_ROOT / "infrastructure" / "setup-thinktank-schedule.sh").is_file()


def test_alarm_targets_the_thinktank_function():
    text = _script_text()
    assert 'FUNCTION_THINKTANK="alpha-engine-research-thinktank"' in text
    assert "Name=FunctionName,Value=${FUNCTION_THINKTANK}" in text


def test_alarm_threshold_matches_sf_retry_worst_case():
    """Threshold 2 = initial invoke + the SF state's one Retry both raised
    (not 3 — that assumed EventBridge's two async retries, which no
    longer apply now the invoker is a synchronous SF Task)."""
    text = _script_text()
    assert re.search(r"--threshold 2\b", text)
    assert "--treat-missing-data notBreaching" in text
    assert re.search(r"--period 86400\b", text)


def test_alarm_uses_shared_sns_topic():
    text = _script_text()
    assert "alpha-engine-alerts" in text
    assert "alpha-engine-thinktank-run-failed" in text


def test_deploy_sh_ships_the_thinktank_target():
    """The alarm is only as meaningful as the function behind it: deploy.sh
    must define the thinktank target (image-share, 900s timeout — the SF
    state's TimeoutSeconds must match) and stage the private
    thinktank.yaml into the image alongside scoring.yaml/universe.yaml."""
    text = _DEPLOY_SH.read_text(encoding="utf-8")
    assert 'FUNCTION_THINKTANK="alpha-engine-research-thinktank"' in text
    assert re.search(
        r'_deploy_image_shared_lambda "\$FUNCTION_THINKTANK" "thinktank_handler" 900 1024',
        text,
    )
    assert "for yaml in scoring.yaml universe.yaml thinktank.yaml; do" in text
