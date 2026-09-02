"""The deploy must strip denied environment variables from every Lambda it
publishes, and must do it where the promotion can carry the change to
traffic.

alpha-engine-config-I7925. This repo's nine research Lambdas were among the
eleven fleet-wide carrying a `GITHUB_TOKEN` set by hand and refreshed by
nothing — live-only state that no repo, IaC file or script here ever wrote.
The environment carried a STALE COPY of the credential — set from an
older SSM parameter version and never re-derived on deploy — that GitHub
rejected while the SSM parameter's own value remained valid the whole
time (alpha-engine-config-I7968 tracks the mis-attribution). On
2026-08-21 a first-party dependency picked
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

import re
from pathlib import Path

import pytest

_DEPLOY = Path(__file__).resolve().parents[1] / "infrastructure" / "deploy.sh"
_CODE = _DEPLOY.read_text(encoding="utf-8")

# Every call site that must converge its function's environment before that
# function's own promotion step, keyed to the shell variable identifying the
# function inside its deploy block.
_DEFERRED_TARGETS = (
    "FUNCTION_MAIN",
    "FUNCTION_EVAL_ROLLING_MEAN",
    "FUNCTION_RATIONALE_CLUSTERING",
    # eval-judge-submit, aggregate-costs, scanner,
    # signals-envelope, openrouter-shadow all share this generic function,
    # parameterized on $fn_name. eval-judge and perturbation-battery were
    # removed (alpha-engine-config-I9756, 2026-09-01): both Lambdas were
    # deleted from AWS and deploy.sh no longer defines those targets.
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


def test_direct_helper_asserts_the_function_is_unaliased() -> None:
    """The claim this helper CAN make, and must, or the deploy dies.

    krepis enumerates aliases to decide whether a $LATEST edit reaches
    traffic, and this deploy role holds no `lambda:ListAliases`. 0.59.24 skips
    that enumeration only under `--defer-publish` — precisely the claim this
    helper cannot make, since nothing here publishes or promotes afterwards.
    So the alerts convergence raised and took the whole deploy with it, twice,
    with krepis 0.59.24 installed (runs 32512239555 and 32514368875,
    alpha-engine-config-I8037).

    `--no-alias` (krepis>=0.59.25) is the assertion that IS true of
    FUNCTION_ALERTS: it serves traffic from $LATEST and has no alias, so alias
    state cannot change the outcome. It belongs on THIS helper only — the
    deferred one makes a different claim about a function that does have an
    alias, and a caller asserting neither still enumerates and still refuses.
    """
    helper_body = _CODE.split("_converge_lambda_env_direct()", 1)[1].split(
        "\n}", 1
    )[0]
    assert "--no-alias" in helper_body, (
        "the direct helper does not assert --no-alias; krepis will enumerate "
        "aliases and the deploy role cannot, so the deploy fails"
    )
    deferred_body = _CODE.split("_converge_lambda_env_deferred()", 1)[1].split(
        "\n}", 1
    )[0]
    assert "--no-alias" not in deferred_body, (
        "the deferred helper handles ALIAS-PINNED functions — asserting they "
        "have no alias would be false, and would suppress the pin check that "
        "protects them"
    )


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


def test_krepis_pin_does_not_need_the_deploy_role_to_list_aliases() -> None:
    """krepis 0.59.23's `remove_lambda_environment_keys` enumerated Lambda
    aliases unconditionally, including under `--defer-publish` — which every
    deferred call site in this deploy.sh passes. The deploy roles here do
    not hold `lambda:ListAliases`. The failure lands after the image is
    pushed and $LATEST is updated, and before `publish-version` and the
    alias move: a PARTIAL deploy, with the `live` alias serving a stale
    image while main has moved on — the SHA drift the preopen
    `DeployDriftGate` halts on (alpha-engine-config-I8030, mirroring
    crucible-predictor's fix for I7925/deploy run 32509752554).

    krepis 0.59.24 skips the enumeration under `defer_publish` (krepis#176).
    An older pin reintroduces the partial deploy, so the floor is pinned
    here rather than left to memory.
    """
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    line = next(
        ln
        for ln in req.read_text(encoding="utf-8").splitlines()
        if ln.startswith("krepis==")
    )
    version = line.split("==", 1)[1].split()[0].strip()
    parts = tuple(int(p) for p in version.split("."))
    assert parts >= (0, 59, 25), (
        f"requirements.txt pins krepis {version}; the deferred sites need "
        f">= 0.59.24 and the alerts site's `--no-alias` needs >= 0.59.25, or "
        f"the deploy fails on lambda:ListAliases — as a PARTIAL deploy at the "
        f"deferred sites (alpha-engine-config-I8030) and as an outright red "
        f"deploy at the alerts site (alpha-engine-config-I8037)"
    )


def test_the_deploy_workflow_installs_a_krepis_that_can_run_the_cli() -> None:
    """The workflow's own `pip install krepis>=X` is the interpreter that runs
    `python3 -m krepis.aws remove-lambda-env` in the deploy steps, so ITS
    resolved version is what decides whether the convergence works —
    `requirements.txt` governs the Lambda images, not the runner.

    That floor read `>=0.7.0` while the step needed 0.59.25. It resolved a new
    enough krepis by luck, because a bare `>=` with no ceiling installs the
    latest. A floor that states nothing about what the step needs is the same
    shape as alpha-engine-config-I7989, where a stale generator floor let two
    routers break on one merge (alpha-engine-config-I8037).
    """
    wf = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"
    body = wf.read_text(encoding="utf-8")
    m = re.search(r'pip install "krepis>=(\d+)\.(\d+)\.(\d+)"', body)
    assert m, "the deploy workflow does not pin a krepis floor for the CLI"
    parts = tuple(int(g) for g in m.groups())
    assert parts >= (0, 59, 25), (
        f"deploy.yml installs krepis>={'.'.join(str(x) for x in parts)}; the "
        "alerts convergence passes --no-alias, which needs >= 0.59.25 or the "
        "step exits 2 on an unknown flag (alpha-engine-config-I8037)"
    )
