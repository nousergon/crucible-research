"""Lock the retired eval-alarm setup script against producer-metric drift
and against re-arming imperative alarm authorship (L4578e, alpha-engine-config-I10182).

`infrastructure/setup_eval_alarms.sh` used to call `aws cloudwatch
put-metric-alarm` directly, alongside `nous-ergon-ops` codifying the SAME
three alarms as JSON and applying them on merge — two independent authorship
paths, whichever repo merged last silently won (alpha-engine-config-I10182).
The script is now a pointer stub: it creates nothing, and this module's job
changed with it.

Two things survive the retirement:

1. **The metric-name guard.** If a producer renames its metric constant, the
   alarm nous-ergon-ops codified silently stops covering it — that was always
   this file's reason to exist, and the check does not require a live
   `put-metric-alarm` command line to make it: the stub documents the exact
   metric name each codified alarm watches, and the test below fails if the
   producer constant and the documented name diverge. (This module
   deliberately does not shallow-clone the PRIVATE nous-ergon-ops repo to
   compare against the live JSON — a public repo's CI holding a credential
   for a private repo is a bigger surface than this guard needs, and
   nous-ergon-ops's own `test_cloudwatch_absence_is_not_a_breach.py` /
   `check-drift.py` already enforce the JSON tree's correctness on that
   side.)
2. **The imperative-authorship guard.** `put-metric-alarm` must never
   reappear in this script, or deploy.yml must never call it again — the
   fleet-wide backstop for every OTHER repo doing this is
   `nous-ergon-ops/infrastructure/cloudwatch/check_no_foreign_alarm_authors.py`
   (alpha-engine-config-I10182), but this repo still gets its own local
   guard, same shape as `nousergon-data`'s and `crucible-dashboard`'s.
"""

from __future__ import annotations

import re
from pathlib import Path

from evals.control_bands import BREACH_COUNT_METRIC_NAME
from evals.rolling_mean import DERIVED_FLOOR_METRIC_NAME

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "infrastructure" / "setup_eval_alarms.sh"
_DEPLOY_WORKFLOW = _ROOT / ".github" / "workflows" / "deploy.yml"

#: Built from parts so this file's OWN prose (which discusses the verb at
#: length) never trips the mutator check below on itself — same technique
#: nous-ergon-ops's fleet-wide guard uses for the identical reason.
_MUTATOR = "put-metric-" + "alarm"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_setup_script_exists():
    assert _SCRIPT.is_file()


def test_control_breach_metric_name_matches_producer():
    assert BREACH_COUNT_METRIC_NAME in _script_text(), (
        f"setup_eval_alarms.sh no longer documents the control-breach metric "
        f"{BREACH_COUNT_METRIC_NAME!r} — the codified nous-ergon-ops alarm is "
        f"orphaned from its producer (evals/control_bands.py). Update the "
        f"pointer stub's documented metric name alongside the codified JSON."
    )


def test_floor_metric_name_matches_producer():
    assert DERIVED_FLOOR_METRIC_NAME in _script_text(), (
        f"setup_eval_alarms.sh no longer documents the quality-floor metric "
        f"{DERIVED_FLOOR_METRIC_NAME!r} — the codified nous-ergon-ops alarm "
        f"is orphaned from its producer (evals/rolling_mean.py). Update the "
        f"pointer stub's documented metric name alongside the codified JSON."
    )


def test_alarm_names_are_still_documented():
    text = _script_text()
    assert "alpha-engine-eval-control-breach" in text
    assert "alpha-engine-eval-quality-regression" in text
    assert "alpha-engine-eval-control-bands-no-datapoint" in text


# ── alpha-engine-config-I10182 — the retirement itself ──────────────────────


def test_setup_script_is_a_pointer_not_an_applier():
    """The retired script must still exist, and must still say where to go.

    Deleting it outright was the alternative. Keeping it as a pointer is
    deliberate, same as crucible-dashboard's install-host-alarms.sh
    (alpha-engine-config-I8035): a stale muscle-memory invocation should be a
    no-op that tells you where the definitions went, not `command not found`.
    """
    body = _script_text()
    assert "no longer creates alarms" in body
    assert "nous-ergon-ops/infrastructure/cloudwatch/alarms" in body
    assert "apply.py --prefix alpha-engine-eval-" in body, (
        "the stub must name the exact command an operator runs to apply a "
        "change immediately, or the pointer only says 'not here'"
    )


def test_setup_script_contains_no_imperative_alarm_call():
    """The class this whole retirement exists to close.

    A comment mentioning the verb is fine — an executable line calling it is
    the defect. Mirrors nous-ergon-ops's own tokenize/regex-based guards:
    a non-comment line matching the verb is a finding, a `#`-prefixed one is
    not.
    """
    hits = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(_script_text().splitlines(), 1)
        if not line.strip().startswith("#") and _MUTATOR in line
    ]
    assert not hits, (
        "setup_eval_alarms.sh calls the CloudWatch alarm-creation verb "
        "again:\n  " + "\n  ".join(hits) + "\n\nSince alpha-engine-config-"
        "I10182 this script creates nothing — the definitions and applier "
        "live in the PRIVATE nous-ergon-ops repo."
    )


def test_deploy_workflow_no_longer_runs_the_alarm_reconcile():
    """The other half of the retirement — a step that still called this
    script would still author alarms even with the script itself as a
    no-op-until-changed pointer, and a future edit to the stub could re-arm
    it silently if nothing here caught the caller too."""
    wf = _DEPLOY_WORKFLOW.read_text(encoding="utf-8")
    _comment = re.compile(r"^\s*#")
    for n, line in enumerate(wf.splitlines(), 1):
        if _comment.match(line):
            continue
        assert "setup_eval_alarms.sh" not in line, (
            f"deploy.yml:{n} still invokes the retired alarm script: "
            f"{line.strip()}"
        )
