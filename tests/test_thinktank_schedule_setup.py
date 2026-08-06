"""``setup-thinktank-schedule.sh`` is SUPERSEDED and must stay refusing
(alpha-engine-config-I5720).

This module used to lock the daily schedule and failure alarm against
drift (config#1579 P1) by pinning what the script provisions. The §47
spot cutover (alpha-engine-config-I5208) moved the daily Think Tank off
``alpha-engine-research-thinktank`` onto a spot box behind
``alpha-engine-thinktank-spot-dispatcher`` (nousergon-data), which
repointed the SAME EventBridge rule and deleted the two alarms this
script creates.

**Those pins therefore became actively wrong.** They asserted that the
script aims the live rule at the retired Lambda — the one thing that must
never happen again, because ``put-targets`` is an upsert and two repos now
aim the same rule at different functions, last writer wins. A test can pin
a contract into place long after the contract stopped being the one we
want; that is what these did, and it is why they are inverted here rather
than deleted.

What is pinned now:

  1. The script REFUSES — non-zero exit, before any AWS call.
  2. It names where the schedule actually lives, so the refusal is a
     redirect rather than a dead end.
  3. ``deploy.sh`` no longer builds the Lambda (alpha-engine-config-I5777,
     2026-08-04): the weekly SF's ``ThinkTankCoverage`` state was ALSO
     repointed at the spot dispatcher (config-I5758), so nothing invokes
     ``alpha-engine-research-thinktank`` any more — measured 2026-08-04,
     zero CloudWatch invocations since 2026-07-29, no resource-based
     policy, no deployed state machine names it. The function is retired,
     not merely its daily EventBridge path. ``lambda/thinktank_handler.py``
     is NOT retired — ``infrastructure/thinktank_box_runner.py`` still
     imports and runs it in-process on the spot box, so the orchestration
     code and its own tests (``test_thinktank_handler.py``) stay live.

The 14:30 cadence and the dispatch alarm now belong to the dispatcher's
own ``deploy.sh`` and are pinned in nousergon-data.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "infrastructure" / "setup-thinktank-schedule.sh"
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_setup_script_exists():
    assert _SCRIPT.is_file()


def test_script_refuses_to_run():
    """The operative pin. Running it would repoint the live rule back at the
    retired Lambda and re-create two alarms that read GREEN on zero
    invocations — the exact silence class the cutover closed."""
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, "a superseded provisioning script must not exit 0"
    assert "REFUSING TO RUN" in result.stderr
    # Non-zero alone is not the property under test: with `set -e`, a script
    # that ran on past the guard and then failed on a missing credential also
    # exits non-zero. What must be true is that it STOPPED — the provisioning
    # banner, which prints immediately before `aws events put-rule`, never
    # appears.
    assert "rule=" not in result.stdout, (
        f"the script proceeded past the guard — it printed the provisioning banner: {result.stdout!r}"
    )


def test_refusal_happens_before_any_aws_call():
    """The guard must sit above the first `aws` invocation, or a partial
    provision lands before the refusal is printed."""
    text = _script_text()
    guard_at = text.index("REFUSING TO RUN")
    first_aws = text.index("\naws ")
    assert guard_at < first_aws


def test_refusal_names_where_the_schedule_actually_lives():
    """A refusal that does not redirect is a dead end."""
    text = _script_text()
    assert "thinktank-spot-dispatcher/deploy.sh --cutover" in text
    assert "alpha-engine-config-I5208" in text


def test_the_deleted_alarms_are_not_resurrected_elsewhere():
    """The cutover DELETED these two alarms because both carry
    ``--treat-missing-data notBreaching`` and would sit green on zero
    invocations. They may only appear inside the refusing script."""
    infra = _REPO_ROOT / "infrastructure"
    for name in (
        "alpha-engine-thinktank-daily-run-failed",
        "alpha-engine-thinktank-daily-run-failed-timeout",
    ):
        for path in sorted(infra.glob("*.sh")):
            if path == _SCRIPT:
                continue
            assert name not in path.read_text(encoding="utf-8"), f"{path.name} references the deleted alarm {name}"


def test_deploy_sh_no_longer_ships_the_thinktank_lambda_target():
    """Inverted 2026-08-04 (alpha-engine-config-I5777): this used to assert
    deploy.sh SHIPS the thinktank Lambda target, back when only the daily
    schedule was superseded and the function still had a live invoker via
    the weekly SF's mode=gap_fill. That invoker is gone too (config-I5758
    repointed ThinkTankCoverage at the spot dispatcher) — the function
    itself is retired, so deploy.sh must NOT publish code to it any more.
    The private thinktank.yaml config stays staged into the shared image
    (thinktank/ core logic is still read on the spot box), so that
    assertion is unchanged."""
    text = _DEPLOY_SH.read_text(encoding="utf-8")
    assert "FUNCTION_THINKTANK" not in text, (
        "deploy.sh still declares FUNCTION_THINKTANK — the thinktank Lambda "
        "deploy target should have been removed by alpha-engine-config-I5777"
    )
    assert "deploy_thinktank" not in text
    assert "for yaml in scoring.yaml universe.yaml thinktank.yaml; do" in text
