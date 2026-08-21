"""The retirement guard (alpha-engine-config-I7827 deliverable 4).

``champion-challenger-policy.md`` §6: *"Retired code is deleted, not left
dormant. A disabled-but-present arm with live config advertising it is worse
than no arm: it reads as capability while doing nothing, and a future change
can silently reactivate it."*

Three times now a retired component of this repo has been read as live —
``crucible-research-PR670`` rewired the retired graph to implement a ruling and
changed nothing; ``alpha-engine-config-I7808`` (a bypassed scanner arm read as
live for four weeks); the ``tech_score`` gate mistaken for the funnel three
times. The specific instance is worth less than the class, so this guard is
written against the REGISTRY, not against one module name.

It makes two assertions, and the second is the one the obvious shape misses.

**Assertion 1 — import reachability.** No module named in any
``ProducerSpec.retired_modules`` may exist in the tree, or be reachable from
any ``lambda/*.py`` handler's transitive first-party import graph.

**Assertion 2 — payload reachability.** Every payload shape a DEPLOY or a
SCHEDULED TRIGGER sends must reach a producer whose ``kind`` is not
``"retired"``, and any payload outside that declared set must fail LOUD.

Assertion 2 exists because assertion 1 would have stayed green through the
exact defect that prompted it: on 2026-08-20 the deploy canary was invoking a
LIVE function (``alpha-engine-research-runner:live``) with a payload
(``{"dry_run_llm": true}``) that reached RETIRED code. The retired module was
reachable by *payload*, not by any import the guard would have flagged as
wrong, because the handler legitimately imports many things. The deploy's own
safety check was smoke-testing dead code and NOT smoke-testing the live
producers — so a deploy that broke ``producers/`` passed the canary green.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from producers.registry import (  # noqa: E402
    RESEARCH_PRODUCERS,
    buildable_challenger_producers,
)

_LAMBDA_DIR = _REPO_ROOT / "lambda"
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"


def _retired_modules() -> set[str]:
    out: set[str] = set()
    for spec in RESEARCH_PRODUCERS.values():
        if spec.kind == "retired":
            out.update(spec.retired_modules)
    return out


def _module_path(module: str) -> Path:
    rel = module.replace(".", "/")
    return _REPO_ROOT / f"{rel}.py"


def _first_party_imports(path: Path) -> set[str]:
    """Every first-party module name imported by ``path`` (any depth in the
    file: module-level, function-level, or inside a ``try``)."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — not a retired top-level module
                continue
            if node.module:
                found.add(node.module)
                for alias in node.names:
                    found.add(f"{node.module}.{alias.name}")
    return {m for m in found if _is_first_party(m)}


def _is_first_party(module: str) -> bool:
    return _module_path(module).exists() or (_REPO_ROOT / module.replace(".", "/")).is_dir()


def _transitive_first_party_imports(entry: Path) -> set[str]:
    seen: set[str] = set()
    queue = list(_first_party_imports(entry))
    while queue:
        mod = queue.pop()
        if mod in seen:
            continue
        seen.add(mod)
        p = _module_path(mod)
        if p.exists():
            queue.extend(_first_party_imports(p))
    return seen


# ── Assertion 1: import reachability ────────────────────────────────────────


def test_a_retired_producers_modules_are_deleted_from_the_tree():
    """§6 deletes the CODE. The registry row stays forever as the historical
    record — that is the point of ``retired_date`` — but the module must go."""
    still_present = sorted(m for m in _retired_modules() if _module_path(m).exists())
    assert not still_present, (
        f"retired producer module(s) still present in the tree: {still_present}. "
        "champion-challenger-policy.md §6 requires retired code to be DELETED, "
        "not left dormant — a dormant module reads as capability while doing "
        "nothing, and a future change can silently reactivate it."
    )


def test_a_no_lambda_handler_import_graph_reaches_a_retired_producer():
    retired = _retired_modules()
    assert retired, (
        "no retired producer declares `retired_modules` — the guard would pass "
        "vacuously. Every kind=='retired' spec must record the modules that "
        "implemented it (producers/registry.py)."
    )
    offenders: dict[str, list[str]] = {}
    handlers = sorted(_LAMBDA_DIR.glob("*.py"))
    assert handlers, "no lambda handlers found — the guard would pass vacuously"
    for handler in handlers:
        reached = _transitive_first_party_imports(handler)
        hits = sorted(retired & reached)
        if hits:
            offenders[handler.name] = hits
    assert not offenders, (
        f"Lambda handler(s) reach RETIRED producer code: {offenders}. "
        "A retired arm on a live boot path is the defect this guard exists "
        "for (alpha-engine-config-I7827)."
    )


# ── Assertion 2: payload reachability ───────────────────────────────────────
#
# The declared set of payload shapes a deploy or a scheduled trigger sends to
# `alpha-engine-research-runner:live`. The deploy canary's payload is READ from
# deploy.sh rather than restated, so changing it there without revisiting this
# guard fails here instead of silently narrowing the coverage.
#
# The weekly SF's ChallengerShadow state sends {"mode":"challengers_only",
# "date": <execution date>} (verified against the deployed ASL 2026-08-20);
# its SelfTest state sends {"mode":"self_test"}. EventBridge
# `alpha-research-weekly` / `alpha-research-daily` are DISABLED and are
# deliberately NOT in this set — their payloads ({"weekly_run": true} / {})
# are covered by the must-fail-loud case below.

_SF_PAYLOADS = [
    {"mode": "challengers_only", "date": "2026-08-15"},
    {"mode": "self_test", "date": "2026-08-15", "dry_run": True},
]


def _deploy_canary_payload() -> dict:
    text = _DEPLOY_SH.read_text()
    m = re.search(r"invoke-canary\b[^\n]*\n(?:\s*--[^\n]*\n)*?\s*--payload\s+'([^']+)'", text)
    assert m, (
        "could not find the deploy canary's --payload in infrastructure/deploy.sh — "
        "this guard reads the deployed payload from source rather than restating it"
    )
    return json.loads(m.group(1))


def _import_handler():
    spec = importlib.util.spec_from_file_location(
        "research_handler_retirement_guard", _LAMBDA_DIR / "handler.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_b_the_deploy_canary_payload_reaches_only_live_producers():
    payload = _deploy_canary_payload()
    mod = _import_handler()
    recorded: dict = {}

    real_specs = buildable_challenger_producers()
    assert real_specs, "no buildable challenger producers registered"
    assert not [s for s in real_specs if s.kind == "retired"], (
        "a retired spec is in the buildable set — producers/registry.py"
    )

    def _record():
        recorded["specs"] = [s.name for s in buildable_challenger_producers()]
        return {"producers": recorded["specs"], "modules": [], "payload_keys": ["signals"]}

    with patch.object(mod, "_ensure_init"), \
         patch("nousergon_lib.dates.resolve_trading_day", return_value="2026-08-14"), \
         patch("archive.manager.ArchiveManager", return_value=MagicMock()), \
         patch("dry_run.install_dry_run_stubs", return_value=MagicMock()), \
         patch("producers.boot_check.run_live_producer_boot_check", side_effect=_record):
        res = mod.handler(payload, None)

    assert res["status"] == "OK"
    assert recorded.get("specs"), (
        f"the deploy canary payload {payload} did not reach the live producer "
        "boot check. THIS is the defect alpha-engine-config-I7827 found: a "
        "live function invoked with a payload that reached retired code, which "
        "an import-graph guard alone stays green on."
    )
    retired_names = {s.name for s in RESEARCH_PRODUCERS.values() if s.kind == "retired"}
    assert not (set(recorded["specs"]) & retired_names)


def test_b_the_scheduled_sf_payloads_reach_only_live_producers():
    mod = _import_handler()

    # ChallengerShadow → producers.runner.run_challengers over the buildable set.
    seen: dict = {}

    def _fake_run_challengers(archive, run_date, **kw):
        seen["specs"] = [s.name for s in buildable_challenger_producers()]
        return {"written": {n: f"k/{n}" for n in seen["specs"]}, "errors": {}}

    am = MagicMock()
    am.s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps({"date": "2026-08-15"}).encode())
    }
    am.load_population.return_value = []

    with patch.object(mod, "_ensure_init"), \
         patch("nousergon_lib.dates.resolve_trading_day", return_value="2026-08-15"), \
         patch("archive.manager.ArchiveManager", return_value=am), \
         patch("producers.runner.run_challengers", side_effect=_fake_run_challengers):
        res = mod.handler(_SF_PAYLOADS[0], None)

    assert res["status"] == "OK"
    retired_names = {s.name for s in RESEARCH_PRODUCERS.values() if s.kind == "retired"}
    assert seen.get("specs"), "the SF's ChallengerShadow payload built no producer"
    assert not (set(seen["specs"]) & retired_names), (
        f"the weekly SF's ChallengerShadow payload builds a RETIRED arm: "
        f"{sorted(set(seen['specs']) & retired_names)}"
    )

    # SelfTest → a verdict stage, not a producer. It must still not reach one.
    with patch.object(mod, "_ensure_init"), \
         patch("nousergon_lib.dates.resolve_trading_day", return_value="2026-08-15"), \
         patch("scoring.self_test.run_self_test", return_value={"verdict": "PASS"}):
        res = mod.handler(_SF_PAYLOADS[1], None)
    assert res["status"] == "OK"
    assert res["mode"] == "self_test"


@pytest.mark.parametrize(
    "payload",
    [
        {"weekly_run": True},                      # DISABLED EventBridge alpha-research-weekly
        {},                                        # DISABLED EventBridge alpha-research-daily
        {"force": True},                           # the old manual-override shape
        {"weekly_run": True, "skip_dry_run_gate": True},  # the deleted weekly_box_runner shape
    ],
)
def test_b_a_payload_for_the_retired_stage_fails_loud(payload):
    """A trigger pointed at the deleted champion pass must RAISE, not return
    SKIPPED. A silent skip is indistinguishable from a run that had nothing to
    do, and that ambiguity is how this component was read as live three times.
    """
    mod = _import_handler()
    with patch.object(mod, "_ensure_init"):
        with pytest.raises(mod.RetiredResearchPathError):
            mod.handler(payload, None)
