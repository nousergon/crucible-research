"""Live synthetic-perturbation judge smoke — OpenRouter shadow tier
(config#2575 item 6, EXPERIMENTS 2026-05-29 harness).

Sibling of ``judge_perturbation_smoke.py`` (Haiku/Anthropic path) — SAME
battery, SAME caught-rate threshold, SAME scorecard renderer
(``evals.perturbation.format_scorecard`` / ``run_perturbation_battery``),
just pointed at the OpenRouter shadow-judge tier via
``evals.perturbation.openrouter_judge`` instead of the Anthropic
``_default_judge`` default. This is the validation gate config#2575's
binding constraint requires before ANY OpenRouter judge verdict is
consumed by anything downstream (escalation routing, RationaleClustering,
ReplayConcordance, Director) — see ``evals/judge_models.py``'s
``SHADOW_LOGICAL_KEYS`` and ``OPENROUTER_SHADOW`` registry entry.

Requires ``OPENROUTER_API_KEY`` (not ``ANTHROPIC_API_KEY`` — this smoke
validates the OTHER judge tier). Env-only secret resolution, same
posture as the Haiku smoke (no SSM-capable IAM role in the CI runner
this is designed for).

Runs:
  * Manually / ad-hoc today (config#2575's implementation pass) — this
    tier is shadow-only and not yet wired into a CI gate the way the
    Haiku smoke is a per-PR gate, since there is no OpenRouter-judge-
    touching PR class yet to gate. A follow-up (see the config#2575 PR
    description) should add this to CI once the shadow tier has a
    production trigger (a Lambda/SF invocation of
    ``evals.openrouter_shadow.run_shadow_judge_over_date``) worth
    protecting.
  * Locally: OPENROUTER_API_KEY=... .venv/bin/python
    tests/live_smoke/judge_perturbation_smoke_openrouter.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")

# Same curated high-signal subset as the Haiku smoke — apples-to-apples
# comparison against Haiku's already-validated performance on the exact
# same corruption set (config#2575 item 6's explicit "4/4 corruption-catch
# parity with Haiku's already-validated performance" acceptance bar).
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


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "judge_perturbation_smoke_openrouter: OPENROUTER_API_KEY not "
            "set; skipping. (Expected when the OpenRouter key isn't "
            "provisioned for this runner; not a failure.)",
            file=sys.stderr,
        )
        return 0

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
        report = run_perturbation_battery(
            corruptions=subset,
            api_key=api_key,
            judge_model=OPENROUTER_SHADOW.logical_key,
            judge_fn=openrouter_judge,
        )
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
