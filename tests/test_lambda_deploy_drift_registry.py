"""CI guard: the drift detector's function registry may not fall behind
``infrastructure/deploy.sh`` (alpha-engine-config-I7840).

The detector cannot enumerate live functions — ``lambda:ListFunctions`` is an
account-level action the ``github-actions-lambda-deploy`` role does not hold,
and granting it is an IAM change no merge applies. So the one drift direction
that actually happens — a new Lambda added to ``deploy.sh``, never registered
here, and therefore never watched — is closed offline, here.

That is the same failure mode ``test_deploy_targets_invoked_by_workflow.py``
exists for one layer up: ``eval_rolling_mean`` had a deploy.sh target that no
workflow step invoked and rotted for 5+ weeks. A function the detector does
not know about is worse than that, because the detector would report full
coverage while watching a subset.

Also pinned: every registered function declares a criticality with a written
reason. An unreasoned entry is how "critical" quietly becomes the default
nobody re-examines, and severity is the whole product here.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"

_spec = importlib.util.spec_from_file_location(
    "lambda_deploy_drift", _REPO_ROOT / "infrastructure" / "lambda_deploy_drift.py",
)
ldd = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Register before exec: dataclasses resolves annotations through
# sys.modules[cls.__module__], which is absent for a module loaded by
# spec alone (AttributeError on 3.14).
sys.modules[_spec.name] = ldd
_spec.loader.exec_module(ldd)

_FUNCTION_VAR_RE = re.compile(
    r'^FUNCTION_[A-Z_]+="(alpha-engine-research-[a-z0-9-]+)"$', re.M,
)


def _deploy_sh_functions() -> set[str]:
    return set(_FUNCTION_VAR_RE.findall(_DEPLOY_SH.read_text()))


def test_every_deploy_sh_function_is_registered():
    missing = _deploy_sh_functions() - set(ldd.REGISTRY)
    assert not missing, (
        f"deploy.sh publishes {sorted(missing)} but lambda_deploy_drift.REGISTRY "
        "does not list them, so the drift detector would report full coverage "
        "while watching a subset. Add each with a criticality and a reason."
    )


def test_every_registered_function_is_either_deployed_here_or_declared_retired():
    known = _deploy_sh_functions()
    for name, spec in ldd.REGISTRY.items():
        if name in known:
            continue
        assert spec.criticality == ldd.RETIRED, (
            f"{name} is registered as {spec.criticality} but deploy.sh has no "
            "FUNCTION_* variable for it. Either it is retired (say so, with "
            "the evidence) or it is a live function nothing deploys — which is "
            "the merged-but-not-deployed defect itself, not a registry typo."
        )


def test_every_registry_entry_carries_a_written_reason():
    for name, spec in ldd.REGISTRY.items():
        assert spec.criticality in (ldd.CRITICAL, ldd.OBSERVE, ldd.RETIRED), name
        assert len(spec.reason.strip()) >= 40, (
            f"{name}: criticality decides whether drift pages or whispers. "
            "It needs a reason a later reader can disagree with."
        )


def test_deploy_sh_stamps_every_function_it_publishes():
    """Coverage of the stamp itself, checked structurally.

    The stamp is only useful if EVERY published function carries it: a
    function that silently misses ``_apply_deploy_stamp_env`` reports
    ``stamp_absent`` forever, which is an unmeasured hole wearing the
    appearance of a monitored function.
    """
    text = _DEPLOY_SH.read_text()
    stamped_vars = set(re.findall(
        r'_apply_deploy_stamp_env "\$(FUNCTION_[A-Z_]+|fn_name|fn)"', text,
    ))
    assert "fn_name" in stamped_vars, (
        "_deploy_image_shared_lambda is the generic publisher for the "
        "image-share Lambdas; if it does not stamp, most functions do not."
    )
    for var in ("FUNCTION_MAIN", "FUNCTION_ALERTS"):
        assert var in stamped_vars, f"{var} is published without a deploy stamp"


def test_the_stamp_is_applied_before_every_publish_version():
    """A published Lambda version snapshots its environment, and the ``live``
    alias points at a published version. Stamping AFTER ``publish-version``
    puts the stamp on ``$LATEST`` only — visible to a careless reader, absent
    from the code that actually runs. Same defect class as config#2766, which
    is why deploy.sh verifies the alias at all.

    Asserted positionally: between any two ``publish-version`` calls there must
    be a stamp call, so no publisher can be added without one.
    """
    text = _DEPLOY_SH.read_text()
    publishes = [m.start() for m in re.finditer(r"aws lambda publish-version", text)]
    stamps = [m.start() for m in re.finditer(r'_apply_deploy_stamp_env "', text)]
    assert publishes and stamps
    prev = 0
    for pub in publishes:
        assert any(prev < s_ < pub for s_ in stamps), (
            f"publish-version at offset {pub} has no _apply_deploy_stamp_env "
            "before it in the same deploy function — that version would be "
            "published without a stamp and the live alias would serve it."
        )
        prev = pub
