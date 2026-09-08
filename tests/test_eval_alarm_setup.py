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


# ── alpha-engine-config-I10166 ───────────────────────────────────────────
#
# The same two defects I9321 fixed on the floor alarm above were left in
# place on the control-breach alarm thirty lines below it in the same
# script, under a comment claiming it was "matching the floor alarm's
# cadence" — a sentence already false when written. These tests hold the
# pair together so the next fix cannot land on one alarm only.


def _control_breach_alarm_command() -> str:
    """The single `put-metric-alarm` line declaring the control-breach alarm."""
    for line in _script_text().splitlines():
        if "alpha-engine-eval-control-breach" in line and line.startswith("aws "):
            return line
    raise AssertionError(
        "no `aws cloudwatch put-metric-alarm` line for "
        "alpha-engine-eval-control-breach found in the setup script"
    )


def _control_bands_deadman_command() -> str:
    """The `put-metric-alarm` line declaring the absence deadman."""
    for line in _script_text().splitlines():
        if (
            "alpha-engine-eval-control-bands-no-datapoint" in line
            and line.startswith("aws ")
        ):
            return line
    raise AssertionError(
        "no `aws cloudwatch put-metric-alarm` line for "
        "alpha-engine-eval-control-bands-no-datapoint found in the setup "
        "script — the ceiling alarm carries notBreaching, so nothing owns "
        "the absence of the control-band evaluation (alpha-engine-config-I8118)"
    )


def test_control_breach_alarm_does_not_treat_missing_data_as_ignore():
    """`ignore` holds the last state FOREVER when the producer stops.

    Measured 2026-09-08: the alarm entered ALARM at 2026-08-30 13:19 PDT,
    then received no datapoint at all for the next 5.6 days and sat red
    for 8.8 days — indistinguishable, on every surface, from a producer
    that had died. `ignore` is strictly worse than `notBreaching` here:
    `notBreaching` at least resolves absence to OK, while `ignore` can
    latch either state, green OR red, indefinitely. `principles.md` §2.7.
    """
    cmd = _control_breach_alarm_command()
    assert "--treat-missing-data ignore" not in cmd
    assert "--treat-missing-data notBreaching" in cmd


def test_the_ceiling_alarm_does_not_also_own_absence():
    """`alpha-engine-config-I8118`: one latched state, one meaning.

    This alarm's comparison is an UPPER bound (`>= 1`), so it fires on a
    condition its metric REPORTS. `breaching` here would make the same
    latched state mean both "a combo went out of control" and "the
    control bands were never evaluated", and an operator reading the
    alarm surface alone could not tell which held. The floor alarm above
    is a `LessThan*` alarm, where absence and breach both mean "not
    proven good" — which is why `breaching` is right there and wrong
    here. `nous-ergon-ops/tests/test_cloudwatch_absence_is_not_a_breach.py`
    enforces the same rule on the codified JSON.
    """
    cmd = _control_breach_alarm_command()
    assert "--comparison-operator GreaterThanOrEqualToThreshold" in cmd
    assert "--treat-missing-data breaching" not in cmd


def test_absence_is_owned_by_a_deadman_on_the_same_emitter():
    """The other half of the I8118 split — either alone is a defect.

    Setting the ceiling alarm to `notBreaching` without this would delete
    the absence detector entirely, which `principles.md` §2.7 forbids
    just as firmly as latching red on silence.
    """
    cmd = _control_bands_deadman_command()
    assert "--treat-missing-data breaching" in cmd
    assert "--comparison-operator LessThanThreshold" in cmd
    assert "--statistic SampleCount" in cmd, (
        "the presence probe must count datapoints, not read their value — "
        "a value statistic makes 'emitted 0' and 'emitted nothing' the "
        "same number again, one layer down"
    )
    assert BREACH_COUNT_METRIC_NAME in cmd, (
        "a deadman on a DIFFERENT emitter proves nothing about this one"
    )


def test_the_deadman_gives_the_weekly_emitter_grace():
    """A window equal to the emission interval has no margin.

    The largest gap measured on this stream is 6 days (2026-08-30 to
    2026-09-05), so 7 daily periods with `DatapointsToAlarm` equal to
    `EvaluationPeriods` leaves ~1 day of grace and requires a full run of
    empty days before it pages.
    """
    cmd = _control_bands_deadman_command()
    assert "--period 86400" in cmd
    assert "--evaluation-periods 7" in cmd
    assert "--datapoints-to-alarm 7" in cmd


def test_control_breach_alarm_period_matches_the_weekly_emission_cadence():
    """A 1-day period on a producer that runs once a week.

    `agent_quality_score_control_breach_count` is emitted by the weekly
    `EvalRollingMean` state: measured 25 emissions between 2026-07-30 and
    2026-09-05, six of them on 2026-08-30 alone and six-day gaps
    elsewhere. Six of every seven daily windows are empty by
    construction, so `--period 86400` cannot carry `breaching` — the
    period and the missing-data policy move together, exactly as they did
    for the floor alarm (I9321). Every COMPLETE week since 2026-07-27
    carries at least one datapoint, so 604800 is the window the producer
    actually fills.
    """
    cmd = _control_breach_alarm_command()
    assert "--period 604800" in cmd
    assert "--period 86400" not in cmd


def test_both_eval_alarms_agree_on_absence_policy():
    """Class fix, not an instance fix.

    The two eval alarms share one producer Lambda and one cadence. A
    future edit that moves one off `breaching`/604800 without the other
    re-creates the split this test was written for.
    """
    floor = _floor_alarm_command()
    breach = _control_breach_alarm_command()
    for cmd in (floor, breach):
        assert "--period 604800" in cmd
    # The floor is a LessThan* alarm: absence and breach both mean "not
    # proven good", so it owns absence itself. The breach alarm is a
    # ceiling and hands absence to its deadman (I8118).
    assert "--treat-missing-data breaching" in floor
    assert "--treat-missing-data notBreaching" in breach
    assert "--treat-missing-data breaching" in _control_bands_deadman_command()
