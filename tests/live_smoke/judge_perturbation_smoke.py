"""Live synthetic-perturbation judge smoke (ROADMAP L480, 2026-05-29).

The CI gate for judge VALIDATION. Runs a small battery of deterministic
corruptions through the REAL judge LLM and asserts the judge catches them
— scoring the corrupted output's targeted rubric dimension below the
known-good reference. This is the one part of the perturbation validator
that needs a live model (the corruption logic + scoring math are covered
offline by tests/test_judge_perturbation.py).

Why a live gate: a mocked test can never tell you the judge prompt still
*works* — only that the harness around it does. If someone weakens a
rubric prompt (or the judge model regresses / the API changes) such that
the judge stops distinguishing good reasoning from broken reasoning, this
smoke goes red. It validates the judge on its actual construct (process
quality), NOT on stock outcomes.

Runs:
  * CI on PRs touching evals/judge.py, evals/perturbation.py, or the
    rubric prompts — gated on OPENROUTER_API_KEY (alpha-engine-config-I2997,
    2026-07-19: evaluate_artifact migrated off direct Anthropic to
    OpenRouter/DeepSeek V4 Flash) + the alpha-engine-config checkout (the
    rubric prompts are gitignored). Forks without the secret get a clean
    skip (exit 0), not a failure.
  * Locally: .venv/bin/python tests/live_smoke/judge_perturbation_smoke.py

Tolerant by design: LLM scores vary run-to-run, so the gate is a
caught-RATE threshold over a curated high-signal subset, not a per-case
exact assertion. A regressed/insensitive judge still fails it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make repo importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Env-only secret resolution — this workflow has no SSM-capable IAM role.
os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")

# Curated high-signal subset (bounds cost: 1 reference judging per rubric
# + one per corruption ≈ 10 judge LLM calls). These corruptions are
# unambiguous degradations a competent judge must catch.
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

# Tolerant threshold for LLM variance. A healthy judge catches all four
# clear corruptions; require >= 3/4 so one stochastic miss doesn't flake
# the gate, while a broadly insensitive judge (catches <= 2) fails.
_MIN_CAUGHT_RATE = 0.75


def main() -> int:
    # alpha-engine-config-I2997 (2026-07-19): evaluate_artifact migrated
    # off direct Anthropic to OpenRouter/DeepSeek V4 Flash — this smoke
    # exercises that same sync judge path via _default_judge, so it now
    # needs an OpenRouter key, not an Anthropic one.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "judge_perturbation_smoke: OPENROUTER_API_KEY not set; skipping. "
            "(Expected on fork PRs without the secret; not a failure.)",
            file=sys.stderr,
        )
        return 0

    try:
        from evals.perturbation import (
            CORRUPTIONS,
            format_scorecard,
            run_perturbation_battery,
        )
    except ImportError as exc:
        print(f"judge_perturbation_smoke: import failed — {exc}", file=sys.stderr)
        return 1

    subset = [c for c in CORRUPTIONS if c.name in _SUBSET_NAMES]
    if not subset:
        print("judge_perturbation_smoke: empty subset — check _SUBSET_NAMES",
              file=sys.stderr)
        return 1

    try:
        report = run_perturbation_battery(corruptions=subset, api_key=api_key)
    except Exception as exc:  # noqa: BLE001 — surface loudly, fail the gate
        print(f"judge_perturbation_smoke: battery raised — {exc}", file=sys.stderr)
        return 1

    print(format_scorecard(report))

    rate = report["caught_rate"]
    if rate < _MIN_CAUGHT_RATE:
        print(
            f"\njudge_perturbation_smoke: FAIL — caught_rate {rate:.2f} "
            f"< {_MIN_CAUGHT_RATE:.2f}. The judge is not reliably detecting "
            f"degraded reasoning (sensitivity/specificity regression).",
            file=sys.stderr,
        )
        return 1

    print(f"\njudge_perturbation_smoke: PASS — caught_rate {rate:.2f} "
          f">= {_MIN_CAUGHT_RATE:.2f} "
          f"({report['n_caught']}/{report['n']} corruptions caught).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
