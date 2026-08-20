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

alpha-engine-config-I6559 (2026-08-19, crucible-research#666) moved the
judge call this smoke exercises off direct OpenRouter and onto the krepis
model router (``evals.judge.JUDGE_MODEL_GROUP``), per the no-direct-
OpenRouter ruling (alpha-engine-config-I6367). The GitHub-hosted CI runner
this workflow runs on has never been provisioned with a router-edge
credential (model-router-policy.md §3.4a — reaching the router off-host
needs `KREPIS_LITELLM_PROXY_URL` + `KREPIS_ROUTER_CREDENTIAL_SECRET`,
neither of which this workflow sets; tracked at
alpha-engine-config-I7853), so today's CI environment structurally
cannot reach any model in the group. That is an infrastructure-provisioning
gap, not a judge regression, so `main()` treats "router resolution found
nothing reachable" as a distinct outcome from the same FAIL path as a real
caught-rate regression (see `_ROUTER_UNREACHABLE_MARKER` below) — but it
is NOT rendered as a bare ``exit 0`` (principles.md §2.7: "a component
emitting nothing is not healthy, it is unobserved; no data is never
rendered as green"). ``main()`` still exits 0 for this job's own
success/failure (a provisioning gap must not hard-block every PR touching
judge code), but it writes ``outcome=skip_router_unreachable`` and
``judged_count=0`` to ``$GITHUB_OUTPUT`` and a summary to
``$GITHUB_STEP_SUMMARY``; the workflow (see
``judge-perturbation-smoke.yml``) reads those and posts a SEPARATE
``neutral``-concluding GitHub check run naming the zero count and
alpha-engine-config-I7853, so a reviewer sees "0 judged, not validated"
rather than a plain green tick indistinguishable from a real pass. Once
the credential is wired, ``resolve_group_spec`` stops raising, this whole
branch stops firing, ``judged_count`` becomes >0, and the neutral check
run stops being posted — the carve-out expires on its own trigger rather
than needing separate cleanup.

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

# krepis.router.resolve_group_spec (called via evals.judge._default_judge)
# raises a bare ValueError with this exact prefix when no entry in the
# group — including the litellm_proxy route itself — is reachable from the
# declared execution context. It is not a distinguishable exception class
# (see krepis/src/krepis/router.py::resolve_group_structured), so the
# message text is the only signal available to tell "the environment
# cannot reach any model" apart from "the judge is broken". Matched as a
# prefix, not a substring, so a wording change that drops the prefix fails
# loud (NameError/AttributeError on the missing marker) rather than
# silently stopping matching.
_ROUTER_UNREACHABLE_MARKER = "No model in group "


def _emit_github_signals(*, outcome: str, judged_count: int, title: str, summary: str) -> None:
    """Write machine-readable + human-readable coverage signals for CI.

    principles.md §2.7: "a component emitting nothing is not healthy, it is
    unobserved; no data is never rendered as green." Every exit path from
    main() calls this exactly once (including the two paths that already
    predate this function — OPENROUTER_API_KEY missing, and the eventual
    PASS/FAIL), so GITHUB_OUTPUT/GITHUB_STEP_SUMMARY always carry an
    outcome and a judged_count, not only the paths a first pass happened to
    remember. A local run (no GITHUB_OUTPUT/GITHUB_STEP_SUMMARY env vars)
    is a silent no-op here — these are CI-only signals.
    """
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"outcome={outcome}\n")
            fh.write(f"judged_count={judged_count}\n")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"### {title}\n\n{summary}\n\n**judged_count:** {judged_count}\n")


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
        _emit_github_signals(
            outcome="skip_no_key",
            judged_count=0,
            title="judge_perturbation_smoke: SKIP (no OPENROUTER_API_KEY)",
            summary=(
                "OPENROUTER_API_KEY was not set — expected on fork PRs, "
                "which never receive secrets. 0 perturbations judged."
            ),
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
        _emit_github_signals(
            outcome="fail_import",
            judged_count=0,
            title="judge_perturbation_smoke: FAIL (import error)",
            summary=f"`evals.perturbation` import failed: {exc}",
        )
        return 1

    subset = [c for c in CORRUPTIONS if c.name in _SUBSET_NAMES]
    if not subset:
        print("judge_perturbation_smoke: empty subset — check _SUBSET_NAMES",
              file=sys.stderr)
        _emit_github_signals(
            outcome="fail_empty_subset",
            judged_count=0,
            title="judge_perturbation_smoke: FAIL (empty subset)",
            summary="_SUBSET_NAMES matched zero entries in CORRUPTIONS.",
        )
        return 1

    try:
        report = run_perturbation_battery(corruptions=subset, api_key=api_key)
    except ValueError as exc:
        # Narrow, named swallow (fail-loud default, ~/Development/CLAUDE.md):
        # (a) failure mode swallowed: krepis.router found no reachable
        #     entry for the judge model group from this CI runner's
        #     execution context — a provisioning gap (missing router-edge
        #     credential wiring, alpha-engine-config-I7853), not a signal
        #     about judge quality. Every other ValueError still falls
        #     through to the loud FAIL below.
        # (b) why the primary deliverable survives: this smoke's job is to
        #     catch a REGRESSED judge; an environment that cannot place any
        #     call has made no claim about the judge at all, so treating it
        #     as a pass-by-default FAIL trains reviewers to ignore a check
        #     that is permanently red for a reason unrelated to their PR.
        # (c) recording surface: this stderr line on every affected run,
        #     PLUS a distinct `neutral`-concluding check run the workflow
        #     posts from the judged_count=0 signal below (principles.md
        #     §2.7 — a bare exit 0 here would render this identically to a
        #     real pass, which is the failure mode this whole carve-out
        #     exists to avoid, not just relocate).
        if str(exc).startswith(_ROUTER_UNREACHABLE_MARKER):
            print(
                f"judge_perturbation_smoke: SKIP (0 judged) — router "
                f"resolution found no reachable model from this CI "
                f"runner's execution context ({exc}). This is an "
                f"infrastructure-provisioning gap (missing router-edge "
                f"credential wiring for this consumer, "
                f"alpha-engine-config-I7853), not a judge regression — "
                f"the battery never placed a call, so it has nothing to "
                f"say about judge quality. This run's own exit code is 0 "
                f"so a provisioning gap does not hard-block every PR "
                f"touching judge code, but it is NOT a validated pass: "
                f"see the separate neutral check run and "
                f"alpha-engine-config-I7853.",
                file=sys.stderr,
            )
            _emit_github_signals(
                outcome="skip_router_unreachable",
                judged_count=0,
                title=(
                    "judge_perturbation_smoke: 0 perturbations judged — "
                    "router unreachable from CI"
                ),
                summary=(
                    f"krepis.router found no reachable model for the judge model "
                    f"group (`low`) from this CI runner. This is a known, tracked "
                    f"infrastructure-provisioning gap "
                    f"(**alpha-engine-config-I7853**) — the CI runner has "
                    f"never been given a router-edge credential "
                    f"(`KREPIS_LITELLM_PROXY_URL` / "
                    f"`KREPIS_ROUTER_CREDENTIAL_SECRET`). **0 of "
                    f"{len(subset)} perturbations were judged.** This run "
                    f"validates nothing about judge quality; do not read "
                    f"it as a pass.\n\n"
                    f"Resolver error: `{exc}`"
                ),
            )
            return 0
        print(f"judge_perturbation_smoke: battery raised — {exc}", file=sys.stderr)
        _emit_github_signals(
            outcome="fail_other_valueerror",
            judged_count=0,
            title="judge_perturbation_smoke: FAIL (battery raised)",
            summary=f"`run_perturbation_battery` raised: {exc}",
        )
        return 1
    except Exception as exc:  # noqa: BLE001 — surface loudly, fail the gate
        print(f"judge_perturbation_smoke: battery raised — {exc}", file=sys.stderr)
        _emit_github_signals(
            outcome="fail_battery_exception",
            judged_count=0,
            title="judge_perturbation_smoke: FAIL (battery raised)",
            summary=f"`run_perturbation_battery` raised: {exc}",
        )
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
        _emit_github_signals(
            outcome="fail_caught_rate",
            judged_count=report["n"],
            title="judge_perturbation_smoke: FAIL (caught_rate below threshold)",
            summary=(
                f"caught_rate {rate:.2f} < {_MIN_CAUGHT_RATE:.2f} "
                f"({report['n_caught']}/{report['n']} corruptions caught)."
            ),
        )
        return 1

    print(f"\njudge_perturbation_smoke: PASS — caught_rate {rate:.2f} "
          f">= {_MIN_CAUGHT_RATE:.2f} "
          f"({report['n_caught']}/{report['n']} corruptions caught).")
    _emit_github_signals(
        outcome="pass",
        judged_count=report["n"],
        title="judge_perturbation_smoke: PASS",
        summary=(
            f"caught_rate {rate:.2f} >= {_MIN_CAUGHT_RATE:.2f} "
            f"({report['n_caught']}/{report['n']} corruptions caught)."
        ),
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
