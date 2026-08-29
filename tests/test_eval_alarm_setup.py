"""Lock the eval-alarm setup script against producer-metric drift (L4578e).

If a producer renames its metric constant, the alarm in
infrastructure/setup_eval_alarms.sh silently stops covering it. This
test fails the build when the script's metric names diverge from the
constants the producers actually emit.
"""

from __future__ import annotations

from pathlib import Path

from evals.control_bands import BREACH_COUNT_METRIC_NAME
from evals.rolling_mean import DERIVED_FLOOR_METRIC_NAME

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "infrastructure"
    / "setup_eval_alarms.sh"
)


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_setup_script_exists():
    assert _SCRIPT.is_file()


def test_control_breach_metric_name_matches_producer():
    assert BREACH_COUNT_METRIC_NAME in _script_text(), (
        f"setup_eval_alarms.sh does not reference the control-breach "
        f"metric {BREACH_COUNT_METRIC_NAME!r} — the alarm is orphaned "
        f"from its producer (evals/control_bands.py). Update the script."
    )


def test_floor_metric_name_matches_producer():
    assert DERIVED_FLOOR_METRIC_NAME in _script_text(), (
        f"setup_eval_alarms.sh does not reference the quality-floor "
        f"metric {DERIVED_FLOOR_METRIC_NAME!r} — the alarm is orphaned "
        f"from its producer (evals/rolling_mean.py). Update the script."
    )


def test_alarms_use_the_shared_sns_topic():
    text = _script_text()
    assert "alpha-engine-alerts" in text
    assert "alpha-engine-eval-control-breach" in text
    assert "alpha-engine-eval-quality-regression" in text


# ── alpha-engine-config-I9321 ────────────────────────────────────────────


def _floor_alarm_command() -> str:
    """The single `put-metric-alarm` line declaring the quality-floor alarm."""
    for line in _script_text().splitlines():
        if "alpha-engine-eval-quality-regression" in line and line.startswith("aws "):
            return line
    raise AssertionError(
        "no `aws cloudwatch put-metric-alarm` line for "
        "alpha-engine-eval-quality-regression found in the setup script"
    )


def test_floor_alarm_treats_missing_data_as_breaching():
    """A floor that stops publishing must not read as healthy.

    Measured 2026-08-29: the alarm was created `--treat-missing-data ignore`,
    which RETAINS the last state when data stops. `AlphaEngine/Eval/
    agent_quality_score` then went to zero live streams and the floor stopped
    publishing after 2026-08-20 — and the alarm simply held the state it
    already had. `ignore` makes a blind alarm and a breaching alarm
    indistinguishable on every surface (`principles.md` §2.7).
    """
    cmd = _floor_alarm_command()
    assert "--treat-missing-data breaching" in cmd
    assert "--treat-missing-data ignore" not in cmd


def test_floor_alarm_period_matches_the_weekly_emission_cadence():
    """The period and the missing-data policy have to move together.

    The floor is emitted once per weekly Step Function run, but the alarm was
    created with `--period 86400`. Six of every seven evaluation windows were
    therefore empty by construction — harmless under `ignore`, but under
    `breaching` it would flap the alarm every single week on a metric that is
    behaving perfectly, and a detector that cries wolf weekly gets muted.
    """
    cmd = _floor_alarm_command()
    assert "--period 604800" in cmd
    assert "--period 86400" not in cmd


def test_alarm_reconcile_runs_on_deploy():
    """The script must have a caller (`pull-request-policy.md` §4.2 form 1).

    Grepped 2026-08-29: `setup_eval_alarms.sh` had been infra-as-code since
    the L4578e arc and NOTHING had ever invoked it, in any repo. Its two alarm
    declarations were live AWS state that happened to match a file, with no
    mechanism holding them together — so an edit to this script changed
    nothing until a human remembered to run it, which is the same
    merged-but-not-deployed class the workflow's own comments describe twice.

    Without this wiring the I9321 alarm change would be a "run this after
    merging" instruction, which a PR body may not carry.
    """
    workflow = (
        Path(__file__).resolve().parent.parent
        / ".github" / "workflows" / "deploy.yml"
    ).read_text(encoding="utf-8")
    assert "infrastructure/setup_eval_alarms.sh" in workflow
