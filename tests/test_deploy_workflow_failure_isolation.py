"""Shape guard: one Lambda's deploy failure must not skip the others.

``.github/workflows/deploy.yml`` deploys every research Lambda as a
sequential step of ONE job. With the implicit ``if: success()``, a failure
in any step skipped every step after it: on 2026-10-12 (run 38290200265,
attempt 1) the scanner step hit a transient AWS AccessDenied, and
signals-envelope, openrouter-shadow, its EventBridge schedule and alerts were
all skipped, leaving the Lambda deploy drift check red until a manual re-run.

Contract pinned here:

1. Every ``bash infrastructure/deploy.sh <target>`` step other than ``main``
   carries an explicit ``if:`` containing ``!cancelled()``, so a sibling's
   failure does not skip it but a cancelled run still stops.
2. Ordering dependencies are preserved, not dropped:
   - image-share targets (every target except ``main`` and ``alerts``) deploy
     the ``$ECR_REPO:latest`` image the main step pushes and stamp this
     commit's SHA on it, so they run only when ``deploy_main`` succeeded;
   - the openrouter-shadow schedule runs only when the openrouter-shadow
     function deployed.
3. Failure still surfaces: no step sets ``continue-on-error``, and
   ``notify-main-failure`` still needs ``deploy`` on ``failure()``.
4. Every ``steps.<id>`` a condition references is an earlier step's id, so a
   renamed id cannot silently turn a gate into "always skipped".
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_YML = _REPO_ROOT / ".github" / "workflows" / "deploy.yml"
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"

_INVOKE_RE = re.compile(r"bash\s+infrastructure/deploy\.sh\s+([A-Za-z0-9_]+)")
_STEP_REF_RE = re.compile(r"steps\.([A-Za-z0-9_-]+)\.")

# Targets that build and push their OWN image rather than deploying the main
# runner's ``$ECR_REPO:latest``. Everything else is image-share and depends on
# the main step having pushed this commit's image.
_OWN_IMAGE_TARGETS = {"main", "alerts"}

_MAIN_GATE = "steps.deploy_main.outcome == 'success'"


def _deploy_steps() -> list[dict]:
    wf = yaml.safe_load(_DEPLOY_YML.read_text())
    return wf["jobs"]["deploy"]["steps"]


def _target(step: dict) -> str | None:
    m = _INVOKE_RE.search(step.get("run") or "")
    return m.group(1) if m else None


def _norm(cond: object) -> str:
    return re.sub(r"\s+", " ", str(cond or ""))


def _targets_by_step() -> dict[str, dict]:
    out = {}
    for step in _deploy_steps():
        t = _target(step)
        if t:
            out[t] = step
    return out


def test_parses_the_known_targets() -> None:
    targets = _targets_by_step()
    for expected in ("main", "scanner", "signals_envelope", "openrouter_shadow", "alerts"):
        assert expected in targets, f"deploy.yml no longer invokes {expected!r}; parser broken?"


def test_alerts_really_builds_its_own_image() -> None:
    """The exemption from the main-step gate is only correct while the alerts
    target builds its own image."""
    text = _DEPLOY_SH.read_text()
    body = text.split("build_and_deploy_alerts() {", 1)[1].split("\n}\n", 1)[0]
    assert "docker build" in body and "ECR_REPO_ALERTS" in body


def test_every_non_main_deploy_step_survives_a_sibling_failure() -> None:
    for target, step in _targets_by_step().items():
        if target == "main":
            continue
        cond = _norm(step.get("if"))
        assert "!cancelled()" in cond, (
            f"deploy step for {target!r} has if: {cond!r}; without `!cancelled()` "
            "an earlier Lambda's failure skips it (run 38290200265)."
        )


def test_image_share_steps_are_gated_on_the_main_image_push() -> None:
    for target, step in _targets_by_step().items():
        if target in _OWN_IMAGE_TARGETS:
            continue
        assert _MAIN_GATE in _norm(step.get("if")), (
            f"image-share target {target!r} must run only when deploy_main "
            "succeeded, or it stamps this SHA onto a stale :latest image."
        )


def test_openrouter_schedule_needs_the_function() -> None:
    steps = [s for s in _deploy_steps() if "setup-openrouter-shadow-schedule.sh" in (s.get("run") or "")]
    assert len(steps) == 1
    cond = _norm(steps[0].get("if"))
    assert "!cancelled()" in cond
    assert "steps.deploy_openrouter_shadow.outcome == 'success'" in cond


def test_failures_still_fail_the_job_and_notify() -> None:
    for step in _deploy_steps():
        assert not step.get("continue-on-error"), (
            f"step {step.get('name')!r} sets continue-on-error; a failed Lambda "
            "deploy must still fail the job."
        )
    wf = yaml.safe_load(_DEPLOY_YML.read_text())
    notify = wf["jobs"]["notify-main-failure"]
    needs = notify["needs"]
    assert "deploy" in (needs if isinstance(needs, list) else [needs])
    assert "failure()" in _norm(notify.get("if"))


def test_every_step_reference_names_an_earlier_step() -> None:
    seen: set[str] = set()
    for step in _deploy_steps():
        for ref in _STEP_REF_RE.findall(_norm(step.get("if"))):
            assert ref in seen, (
                f"step {step.get('name')!r} gates on steps.{ref}, which is not an "
                "earlier step id; the gate would always be skipped."
            )
        if "id" in step:
            seen.add(step["id"])
