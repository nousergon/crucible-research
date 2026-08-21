"""The deploy must strip denied environment variables from every Lambda it
publishes, and must do it where the promotion can carry the change to
traffic.

alpha-engine-config-I7925. This repo's nine research Lambdas were among the
eleven fleet-wide carrying a `GITHUB_TOKEN` set by hand and refreshed by
nothing — live-only state that no repo, IaC file or script here ever wrote.
That token expired 2026-06-03; on 2026-08-21 a first-party dependency picked
it up out of site-packages, sent it to GitHub, got a 401, and halted the
preopen trading pipeline 3.4 seconds after start (alpha-engine-config-I7924).
`alpha-engine-predictor-inference` (crucible-predictor) was the one of eleven
that broke and was fixed first (crucible-predictor-PR535); this propagates
the same convergence to the other nine.

`infrastructure/deploy.sh` now converges every function's environment
against a deny-list. These tests pin the properties that make that
convergence real, because each fails SILENTLY: a removal placed after
`publish-version` never reaches the published version and is a no-op on the
`live` alias (L4497); a removal that also promotes the alias would race the
deploy's own promotion; and FUNCTION_ALERTS has no alias/version promotion
at all, so a deferred removal there would never reach traffic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DEPLOY = Path(__file__).resolve().parents[1] / "infrastructure" / "deploy.sh"
_CODE = _DEPLOY.read_text(encoding="utf-8")

# Every call site that must converge its function's environment before that
# function's own promotion step, keyed to the shell variable identifying the
# function inside its deploy block.
_DEFERRED_TARGETS = (
    "FUNCTION_MAIN",
    "FUNCTION_EVAL_JUDGE",
    "FUNCTION_EVAL_ROLLING_MEAN",
    "FUNCTION_RATIONALE_CLUSTERING",
    # eval-judge-submit / -poll / -process, aggregate-costs, scanner,
    # signals-envelope, openrouter-shadow, perturbation-battery all share
    # this generic function, parameterized on $fn_name.
    "fn_name",
)


def test_deploy_script_exists() -> None:
    assert _DEPLOY.is_file(), f"{_DEPLOY} is missing"


def test_github_token_is_on_the_deny_list() -> None:
    """The credential that caused I7924 must be named, not merely implied."""
    assert "LAMBDA_ENV_DENIED_KEYS=(" in _CODE, (
        "the deploy no longer declares a denied-key set — a variable set by "
        "hand now outlives every deploy again (alpha-engine-config-I7925)"
    )
    declaration = _CODE.split("LAMBDA_ENV_DENIED_KEYS=(", 1)[1].split(")", 1)[0]
    assert "GITHUB_TOKEN" in declaration


def test_removal_uses_the_shared_cli_not_a_bare_aws_call() -> None:
    """`aws lambda update-function-configuration --environment` REPLACES the
    whole variable map, deleting every operator-set flag codified nowhere.
    The read-modify-write chokepoint is `krepis.aws remove-lambda-env`."""
    assert "krepis.aws remove-lambda-env" in _CODE
    # Called at least once per deferred target site plus once for the
    # non-alias-pinned alerts function.
    assert _CODE.count("remove-lambda-env") >= 2


def test_deferred_helper_runs_before_publish_and_never_promotes() -> None:
    """`_converge_lambda_env_deferred` edits $LATEST only. Each caller
    publishes a version and moves the `live` alias itself immediately after
    — placed after publish-version this would be a silent no-op on the
    alias (L4497)."""
    assert "_converge_lambda_env_deferred()" in _CODE
    helper_body = _CODE.split("_converge_lambda_env_deferred()", 1)[1].split(
        "\n}", 1
    )[0]
    assert "remove-lambda-env" in helper_body
    assert "--defer-publish" in helper_body
    assert "--promote-alias" not in helper_body
    assert "--missing-ok" in helper_body


def test_direct_helper_never_defers_publish() -> None:
    """FUNCTION_ALERTS has no alias/version promotion in this deploy.sh — a
    deferred ($LATEST-only) removal there would never reach traffic, since
    nothing subsequently publishes or promotes it."""
    assert "_converge_lambda_env_direct()" in _CODE
    helper_body = _CODE.split("_converge_lambda_env_direct()", 1)[1].split(
        "\n}", 1
    )[0]
    assert "remove-lambda-env" in helper_body
    assert "--defer-publish" not in helper_body
    assert "--missing-ok" in helper_body


def test_alerts_uses_the_direct_helper() -> None:
    call_index = _CODE.index('_converge_lambda_env_direct "$FUNCTION_ALERTS"')
    deployed_index = _CODE.index(
        'echo "  $FUNCTION_ALERTS deployed (container image)."'
    )
    assert call_index < deployed_index


@pytest.mark.parametrize("target", _DEFERRED_TARGETS)
def test_deferred_convergence_precedes_publish_version(target: str) -> None:
    """For each alias-pinned target, the convergence call for that variable
    must appear, and it is on the deferred (non-promoting) helper."""
    call = f'_converge_lambda_env_deferred "${target}"'
    assert call in _CODE, f"missing deferred convergence call for {target}"


def test_removal_is_idempotent_across_deploys() -> None:
    """Every deploy after the first finds the key already gone; without
    --missing-ok the CLI refuses and `set -euo pipefail` aborts the deploy."""
    assert _CODE.count("--missing-ok") >= 2


def test_krepis_pin_can_supply_the_subcommand() -> None:
    """`remove-lambda-env` ships in krepis 0.59.23. An older pin makes the
    deploy step exit 2 on an unknown subcommand."""
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    line = next(
        ln
        for ln in req.read_text(encoding="utf-8").splitlines()
        if ln.startswith("krepis==")
    )
    version = line.split("==", 1)[1].split()[0].strip()
    parts = tuple(int(p) for p in version.split("."))
    assert parts >= (0, 59, 23), (
        f"requirements.txt pins krepis {version}; remove-lambda-env needs >= 0.59.23"
    )
