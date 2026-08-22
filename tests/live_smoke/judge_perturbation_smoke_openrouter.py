"""Live synthetic-perturbation judge smoke — OpenRouter shadow tier
(config#2575 item 6, EXPERIMENTS 2026-05-29 harness).

Sibling of ``judge_perturbation_smoke.py`` (sync/primary judge path) —
SAME battery, SAME caught-rate threshold, SAME scorecard renderer
(``evals.perturbation.format_scorecard`` / ``run_perturbation_battery``),
just pointed at the OpenRouter shadow-judge tier via
``evals.perturbation.openrouter_judge`` (-> ``evals.judge.
evaluate_artifact_openrouter``) instead of the sync-primary
``_default_judge`` default.

alpha-engine-config-I6559 (2026-08-19, crucible-research#666) moved
``evaluate_artifact_openrouter`` off direct OpenRouter onto the krepis
model router (``evals.judge.JUDGE_MODEL_GROUP``), same as its sync-primary
sibling — see that function's docstring. This script reads NO provider
API key (alpha-engine-config-I7880 removed the stale OPENROUTER_API_KEY
gate this docstring used to describe): the router resolves the credential
by name, and a router-unreachable environment is signalled by
``run_perturbation_battery`` raising a ``ValueError`` prefixed with
``_ROUTER_UNREACHABLE_MARKER`` (see ``judge_perturbation_smoke.py`` for
the shared shape), caught below as a SKIP rather than a FAIL — this
script has no claim to make about judge quality if it never placed a
call.

Runs:
  * Manually / ad-hoc today (config#2575's implementation pass) — this
    tier is shadow-only and not yet wired into a CI gate the way the
    sync-primary smoke is a per-PR gate, since there is no
    OpenRouter-judge-touching PR class yet to gate. A follow-up (see the
    config#2575 PR description) should add this to CI once the shadow
    tier has a production trigger (a Lambda/SF invocation of
    ``evals.openrouter_shadow.run_shadow_judge_over_date``) worth
    protecting — at which point it inherits the same router-edge CI
    reachability question tracked at alpha-engine-config-I7853.
  * Locally: .venv/bin/python
    tests/live_smoke/judge_perturbation_smoke_openrouter.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")

# Same curated high-signal subset as the sync-primary smoke — apples-to-
# apples comparison against its already-validated performance on the
# exact same corruption set (config#2575 item 6's explicit "4/4
# corruption-catch parity" acceptance bar).
_SUBSET_NAMES = {
    "strip_numerical_grounding",
    "break_ranking_coherence",
    "strip_citation_grounding",
    "flatten_reasoning_depth",
    "strip_input_groundedness",
    "vacuous_moat",
    "contradict_stance",
    "unearned_material_change",
    "break_anchor_fidelity",
}

_MIN_CAUGHT_RATE = 0.75

# Same marker krepis.router.resolve_group_spec raises when no entry in
# the group is reachable from the declared execution context — see
# judge_perturbation_smoke.py's identical constant for the full note.
_ROUTER_UNREACHABLE_MARKER = "No model in group "


def main() -> int:
    try:
        from evals.judge_models import OPENROUTER_SHADOW
        from evals.perturbation import (
            CORRUPTIONS,
            format_scorecard,
            openrouter_judge,
            run_perturbation_battery,
        )
    except ImportError as exc:
        print(f"judge_perturbation_smoke_openrouter: import failed — {exc}", file=sys.stderr)
        return 1

    subset = [c for c in CORRUPTIONS if c.name in _SUBSET_NAMES]
    if not subset:
        print("judge_perturbation_smoke_openrouter: empty subset — check _SUBSET_NAMES",
              file=sys.stderr)
        return 1

    try:
        # api_key intentionally omitted: production callers of the
        # router-resolved judge core leave it None so krepis resolves the
        # credential by name (evals/judge.py::_call_openrouter_judge_llm
        # docstring — passing a real key here would OVERRIDE that
        # resolution, not merely gate on presence).
        report = run_perturbation_battery(
            corruptions=subset,
            judge_model=OPENROUTER_SHADOW.logical_key,
            judge_fn=openrouter_judge,
        )
    except ValueError as exc:
        # Narrow, named swallow (fail-loud default, ~/Development/CLAUDE.md):
        # (a) failure mode swallowed: krepis.router found no reachable
        #     entry for the judge model group from this environment — an
        #     infrastructure-provisioning gap, not a signal about judge
        #     quality. Every other ValueError still falls through to the
        #     loud FAIL below.
        # (b) why the primary deliverable survives: this smoke's job is to
        #     catch a REGRESSED shadow judge; an environment that cannot
        #     place any call has made no claim about it at all.
        # (c) recording surface: this stderr line, on every affected run.
        if str(exc).startswith(_ROUTER_UNREACHABLE_MARKER):
            print(
                f"judge_perturbation_smoke_openrouter: SKIP (0 judged) — "
                f"router resolution found no reachable model from this "
                f"environment's execution context ({exc}). Not a judge "
                f"regression; the battery never placed a call.",
                file=sys.stderr,
            )
            return 0
        print(f"judge_perturbation_smoke_openrouter: battery raised — {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface loudly, fail the gate
        print(f"judge_perturbation_smoke_openrouter: battery raised — {exc}", file=sys.stderr)
        return 1

    print(format_scorecard(report))

    rate = report["caught_rate"]
    if rate < _MIN_CAUGHT_RATE:
        print(
            f"\njudge_perturbation_smoke_openrouter: FAIL — caught_rate "
            f"{rate:.2f} < {_MIN_CAUGHT_RATE:.2f}. The OpenRouter shadow "
            f"judge is NOT reliably detecting degraded reasoning — it "
            f"must NOT be promoted (config#2575 item 7 stays blocked; "
            f"shadow lane continues collecting data).",
            file=sys.stderr,
        )
        return 1

    print(f"\njudge_perturbation_smoke_openrouter: PASS — caught_rate {rate:.2f} "
          f">= {_MIN_CAUGHT_RATE:.2f} "
          f"({report['n_caught']}/{report['n']} corruptions caught). "
          f"Validation criterion for config#2575 item 6 is MET for this "
          f"run — see the PR/issue thread for whether promotion (item 7) "
          f"is authorized to proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
