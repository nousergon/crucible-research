"""Chokepoint tests for infrastructure/setup-eventbridge.sh — content-vs-uniqueness
guards for the script that is the SOLE source of truth for this repo's
EventBridge rules and Lambda invoke permissions.

Origin: ROADMAP L302 P0 retrospective on PR #317's content-vs-uniqueness
CI gap (alpha-engine-data) — the same meta-pattern (tests pin WHAT was
put, not HOW MANY were put) applies to this script. Without these
checks, a future PR could:

  * Add a duplicate `put-targets` call for the same rule (the exact
    failure mode that caused the 2026-05-26 trading-day miss in
    alpha-engine-data), shipping two parallel Lambda invocations per
    cron firing.
  * Add a duplicate `put-rule` call with the same name, silently
    overwriting the prior call's schedule + state config.
  * Add an EventBridge target with `Id != "1"`, creating a second
    target alongside the canonical one when the script re-runs.

These tests don't replace the IAM-drift / live-AWS verification, but
they catch the script-side regression at PR time, before any operator
re-runs the script against live AWS.

Sibling pattern: alpha-engine-data PR #322's
`TestCFNTargetUniqueness` + `TestDeployScriptsHaveNoEventBridgeWrites`
in `tests/test_deploy_step_function_eventbridge_input.py`. There the
canonical source is CFN; here the canonical source is this script (no
CFN-managed EB rules in this repo). The shape of the chokepoint
flips accordingly: instead of "scripts forbidden from writing EB",
the test is "this script must declare each rule exactly once with
exactly one target."
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "setup-eventbridge.sh"

# The 3 EventBridge rules this script is the sole source of truth for.
# Order doesn't matter; the set is what's pinned.
_EXPECTED_RULES = frozenset(
    {
        "alpha-research-weekly",
        "alpha-research-daily",
        "alpha-research-alerts",
    }
)

# Old rule names the script explicitly cleans up via remove-targets/delete-rule
# in its preamble. These MUST NOT appear in put-rule/put-targets calls —
# they were retired and re-introducing them would create orphaned EB targets.
_RETIRED_RULES = frozenset(
    {
        "alpha-research-pdt",
        "alpha-research-pst",
        "alpha-research-sunday",
    }
)


def _script_text() -> str:
    assert _SCRIPT.exists(), f"setup-eventbridge.sh missing at {_SCRIPT}"
    return _SCRIPT.read_text()


def _count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def test_setup_eventbridge_script_exists():
    assert _SCRIPT.exists(), (
        f"infrastructure/setup-eventbridge.sh missing at {_SCRIPT}. "
        "This script is the sole source of truth for EventBridge rules "
        "in this repo; CI cannot validate its uniqueness invariants "
        "without it."
    )


@pytest.mark.parametrize("rule_name", sorted(_EXPECTED_RULES))
def test_each_expected_rule_has_exactly_one_put_rule_call(rule_name: str):
    """Each expected EB rule name appears in `aws events put-rule --name <X>`
    EXACTLY ONCE in the script.

    Closes the duplicate-put-rule recurrence class: re-running the script
    would silently overwrite the prior put-rule, which is by-design
    idempotent. But a future PR that accidentally introduces a second
    put-rule for the same name (e.g., copy-paste between weekly + daily
    blocks) would silently let the second call clobber the first's
    schedule / state. The script-time count assertion catches that at
    PR time, before any operator runs it against live AWS.
    """
    text = _script_text()
    # Match: `--name "$VAR"` where VAR resolves to the rule_name, OR
    # `--name "<literal>"` if the script ever stops using variables.
    # The script today uses RULE_WEEKLY / RULE_DAILY / RULE_ALERTS vars,
    # which expand to the expected names — we assert on the literal
    # var-name spelling that maps to each rule.
    var_for_rule = {
        "alpha-research-weekly": "RULE_WEEKLY",
        "alpha-research-daily": "RULE_DAILY",
        "alpha-research-alerts": "RULE_ALERTS",
    }[rule_name]
    pattern = rf'aws events put-rule\s*\\?\s*\n?\s*--name "\${var_for_rule}"'
    n = _count_matches(pattern, text)
    assert n == 1, (
        f"Expected EXACTLY 1 `aws events put-rule --name \"${var_for_rule}\"` "
        f"call in setup-eventbridge.sh for rule {rule_name!r}; found {n}. "
        "Duplicate put-rule calls would silently overwrite the prior "
        "schedule/state config."
    )


@pytest.mark.parametrize("rule_name", sorted(_EXPECTED_RULES))
def test_each_expected_rule_has_exactly_one_put_targets_call(rule_name: str):
    """Each expected EB rule has `aws events put-targets --rule <X>` called
    EXACTLY ONCE in the script. Sibling to the put-rule uniqueness test.

    This is the chokepoint that prevents the 2026-05-26-class recurrence
    in this repo (alpha-engine-data #322 closed it for that repo's
    CFN-canonical path; this closes it for alpha-engine-research's
    script-canonical path).
    """
    var_for_rule = {
        "alpha-research-weekly": "RULE_WEEKLY",
        "alpha-research-daily": "RULE_DAILY",
        "alpha-research-alerts": "RULE_ALERTS",
    }[rule_name]
    text = _script_text()
    pattern = rf'aws events put-targets\s*\\?\s*\n?\s*--rule "\${var_for_rule}"'
    n = _count_matches(pattern, text)
    assert n == 1, (
        f"Expected EXACTLY 1 `aws events put-targets --rule \"${var_for_rule}\"` "
        f"call in setup-eventbridge.sh for rule {rule_name!r}; found {n}. "
        "Duplicate put-targets calls would fan a single cron firing to "
        "multiple Lambda invocations (the 2026-05-26 trading-day-miss "
        "failure mode in alpha-engine-data)."
    )


@pytest.mark.parametrize("rule_name", sorted(_EXPECTED_RULES))
def test_each_put_targets_call_declares_exactly_one_target(rule_name: str):
    """Each `put-targets` call in the script declares a SINGLE target with
    `Id:"1"`. EventBridge dispatches a rule trigger to every target on the
    rule, so the targets array's length IS the fan-out factor.

    A future PR that adds a second target object to the JSON array
    (e.g., to send the same event to a debug Lambda) would silently
    double the production Lambda invocations. The test pins exactly one
    target per put-targets call.
    """
    var_for_rule = {
        "alpha-research-weekly": "RULE_WEEKLY",
        "alpha-research-daily": "RULE_DAILY",
        "alpha-research-alerts": "RULE_ALERTS",
    }[rule_name]
    text = _script_text()
    # Find the put-targets call for this rule and extract the --targets
    # arg's JSON-shaped value (everything up to the next `--region`).
    # The script today uses single-line shell-quoted JSON; assert there
    # is exactly one `"Id":` token inside.
    block_pattern = (
        rf'aws events put-targets\s*\\\s*\n'
        rf'\s*--rule "\${var_for_rule}"\s*\\\s*\n'
        rf'\s*--targets \'(.+?)\'\s*\\\s*\n'
    )
    match = re.search(block_pattern, text, flags=re.DOTALL)
    assert match is not None, (
        f"Could not locate `aws events put-targets --rule \"${var_for_rule}\"` "
        f"block for {rule_name!r} in setup-eventbridge.sh. The script's "
        "formatting may have drifted; update this test if the structure "
        "changed deliberately."
    )
    targets_json = match.group(1)
    id_count = targets_json.count('"Id":')
    assert id_count == 1, (
        f"Expected EXACTLY 1 target (one `\"Id\":` key) in the put-targets "
        f"JSON for rule {rule_name!r}; found {id_count}. EventBridge "
        "fans a cron trigger to every target; multiple targets per rule "
        "cause silent duplicate Lambda invocations."
    )
    # The canonical target Id in this script is "1" — pin it so future
    # PRs can't introduce "1a" / "primary" / etc. that would create a
    # second target alongside re-runs of the original Id="1".
    assert '"Id":"1"' in targets_json, (
        f"Expected target `Id` to be the canonical \"1\" in put-targets "
        f"JSON for rule {rule_name!r}; got {targets_json!r}. The canonical "
        "Id is the dedup key for idempotent re-runs of this script."
    )


@pytest.mark.parametrize("retired_rule", sorted(_RETIRED_RULES))
def test_retired_rules_not_reintroduced_in_put_rule(retired_rule: str):
    """The script's preamble explicitly cleans up retired rule names via
    `aws events remove-targets` + `delete-rule`. The chokepoint asserts
    those names never appear in `put-rule` / `put-targets` calls below
    the cleanup block — if a future PR re-introduces one, the cleanup
    loop would delete it on every run and the put would re-create it,
    flapping the production EB state on every operator script run.
    """
    text = _script_text()
    # The retired-rule names appear literally in the cleanup loop's `for
    # old in ...` line; assert they appear NOWHERE ELSE in the script.
    # Count total occurrences, subtract the cleanup-loop line.
    total = text.count(retired_rule)
    # The cleanup loop has all 3 retired rules in one `for old in` line,
    # so each retired rule appears exactly once there. Anywhere above
    # that one occurrence would be a regression.
    assert total <= 1, (
        f"Retired EventBridge rule {retired_rule!r} appears {total} times "
        f"in setup-eventbridge.sh; expected at most 1 (cleanup-loop only). "
        "Re-introducing it in put-rule/put-targets would flap production "
        "state on every script run because the cleanup loop deletes it "
        "right before the put."
    )


def test_no_unexpected_rule_names_in_put_rule_calls():
    """Inverse of the per-expected-rule test: assert that the ONLY rule
    names appearing in put-rule calls are the 3 in _EXPECTED_RULES. Closes
    the gap where a future PR adds a fourth rule without updating this
    test — the fourth rule would lack the per-rule put-targets / Id
    uniqueness coverage.
    """
    text = _script_text()
    # Find all `--name "$<VAR>"` arguments that appear in put-rule calls.
    # The script's variable convention is RULE_WEEKLY / RULE_DAILY /
    # RULE_ALERTS — assert no other RULE_<X> variable name appears in a
    # put-rule call.
    pattern = r'aws events put-rule[\s\S]*?--name "\$(RULE_[A-Z_]+)"'
    found_vars = set(re.findall(pattern, text))
    expected_vars = {"RULE_WEEKLY", "RULE_DAILY", "RULE_ALERTS"}
    unexpected = found_vars - expected_vars
    assert not unexpected, (
        f"Unexpected RULE_* variable(s) in put-rule calls: {sorted(unexpected)}. "
        f"If you added a new EventBridge rule, extend _EXPECTED_RULES in "
        "this test file to include it and add the corresponding "
        "var_for_rule entries in the parametrized tests."
    )
    missing = expected_vars - found_vars
    assert not missing, (
        f"Expected RULE_* variable(s) missing from put-rule calls: "
        f"{sorted(missing)}. The script should declare all 3 canonical "
        "rules (weekly + daily + alerts) — if one was removed deliberately, "
        "update _EXPECTED_RULES to match."
    )
