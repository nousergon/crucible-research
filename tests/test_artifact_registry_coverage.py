"""
Artifact-registry coverage CI guard.

Phase 4 PR 5 of the artifact-freshness-monitor arc (plan doc:
``~/Development/alpha-engine-docs/private/artifact-freshness-monitor-260527.md``).
Mirrors ``alpha-engine-data/tests/test_artifact_registry_coverage.py``
(PR 4, merged 2026-05-27); the cascade closes producer-side coverage
of the registry across all 4 producing repos (ae-data, this repo,
ae-predictor, ae-backtester).

**What this catches.** A new ``s3.put_object(...)`` or
``s3.upload_file(...)`` site in ae-research production code that
hasn't been registered in
``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml`` (or
explicitly grandfathered). Forces operator attention at every new
producer addition — the silent absence-of-artifact bug class
(e.g., 2026-05-17→27 pit_parity.json, 2026-05-23 missing
signals.json) can't slip past PR review without an explicit
register-or-grandfather decision.

**Design choice — per-file count rather than per-key-template
extraction.** Statically extracting key templates from f-string
``put_object(Key=...)`` calls is fragile (keys are often constructed
from surrounding context). Per-file count is stable across refactors
and sufficient to force operator review. See the ae-data PR 4 commit
message for the full rationale.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-file PUT-site counts. Pinning enforces operator attention on
# every new producer addition. When a file gains/loses a PUT site:
#   1. Decide whether the new artifact is load-bearing.
#   2. Register it in alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml
#      (or add the prefix to grandfathered_paths with a one-line reason).
#   3. Bump the count here.
# Captured 2026-05-27.
EXPECTED_PER_FILE_PUT_COUNTS: dict[str, int] = {
    "archive/manager.py": 5,
    "data/fetchers/analyst_fetcher.py": 1,
    "data/fetchers/insider_fetcher.py": 1,
    "data/fetchers/revision_fetcher.py": 1,
    # 2 PUT sites: write_candidates_artifact (live candidates/) +
    # write_shadow_candidates_artifact (candidates_shadow/{spec}/ — the
    # config#1221 OBSERVE substrate; registered in ARTIFACT_REGISTRY.yaml as
    # scanner_candidates_shadow_momentum_sleeve at WATCH severity).
    "data/scanner_orchestrator.py": 2,
    # κ calibration report (ROADMAP L480): kappa.json + kappa.md + the two
    # latest/ pointers. Prefix decision_artifacts/_calibration/_report/ is
    # grandfathered in ARTIFACT_REGISTRY.yaml — operator-gated, consumer
    # graceful-degrades, so not yet a load-bearing freshness SLA.
    "evals/calibration_kappa.py": 4,
    # Control-band breach entries (L4578(e)) — writes to the SAME
    # changelog/entries/ corpus as rolling_mean.py (already registered;
    # observability, not a load-bearing freshness SLA), so no new
    # ARTIFACT_REGISTRY row, just this per-file PUT pin.
    "evals/control_bands.py": 1,
    "evals/eval_manifest.py": 1,
    # Two PUT sites (config#793 canonical eval_artifacts swap):
    #   1. the dated forensic artifact at decision_artifacts/_eval/{judge_run_id}_...json
    #      (the source of truth — pre-existing pin).
    #   2. the best-effort latest.json operator-UX sidecar mirror at
    #      decision_artifacts/_eval/latest.json (eval_latest_key), a rebuildable
    #      single-fetch pointer the lib's load_latest_eval_artifact reader resolves.
    # The sidecar write is best-effort (failure is logged, does NOT fail the
    # artifact write) and rebuildable from the dated key, so it is NOT a
    # load-bearing freshness-SLA artifact with a daily consumer — per-file PUT
    # pin only, no new ARTIFACT_REGISTRY freshness row (same rationale as the
    # cost/capture streams above; the decision_artifacts/_eval/ prefix is
    # grandfathered in ARTIFACT_REGISTRY.yaml).
    "evals/judge.py": 2,
    "evals/last_week_scorecard.py": 2,
    "evals/orchestrator.py": 2,
    "evals/rationale_clustering.py": 1,
    "evals/rolling_mean.py": 1,
    # Saturday-SF team-accuracy producer (config#1422) — single fixed-key
    # overwrite (config/team_accuracy.json). Registered in
    # ARTIFACT_REGISTRY.yaml as config_team_accuracy (warning severity;
    # consumer load_team_accuracy gracefully falls back to static
    # allocation when absent).
    "evals/team_accuracy.py": 1,
    # Two PUT sites: (1) per-call cost-raw JSONL (_cost_raw/), (2) the
    # SFT-lossless capture JSONL (_sft_raw/, config#1134). Both are gated on
    # ALPHA_ENGINE_DECISION_CAPTURE_ENABLED and are accumulation streams
    # (cost telemetry / Phase-3 distillation training data), NOT load-bearing
    # freshness-SLA artifacts with a daily consumer — so no ARTIFACT_REGISTRY
    # row, just this per-file PUT pin.
    "graph/llm_cost_tracker.py": 2,
    # Single PUT site: dated data_manifest/{module}/{date}.json. Health
    # enrichment writes moved to nousergon_lib.health (config#1727 Phase C).
    "data_manifest.py": 1,
    "local/sync_db.py": 1,
    "scoring/factor_scoring.py": 2,
    # Full-universe scoreboard (scanner/universe/{date}/universe.json + latest).
    # SECONDARY observability for the dashboard's filterable ~900-name universe
    # board — built fail-soft off archive_writer (a write failure WARNs, never
    # fails the research run; signals.json is the primary deliverable). The
    # dashboard consumer graceful-degrades when the artifact is absent, so
    # absence is NOT a silent failure and needs no daily freshness-SLA alarm.
    # Per-file PUT pin only; ARTIFACT_REGISTRY row deferred until the producer
    # has run once (register-with-or-after-producer). Single PUT site (loop over
    # dated + latest keys in write_universe_board_to_s3).
    "scoring/universe_board.py": 1,
    # Per-stock attractiveness HISTORY parquet
    # (scanner/universe/history/attractiveness_history.parquet) + the weekly
    # TRAJECTORY signal (scanner/universe/trajectory/{date}.json + latest).
    # SECONDARY observability built fail-soft off archive_writer (write failure
    # WARNs, never fails the run; signals.json is primary). OBSERVE-MODE signal;
    # the dashboard consumer graceful-degrades when absent → absence is NOT a
    # silent failure. Per-file PUT pin only; ARTIFACT_REGISTRY rows deferred
    # until first Saturday production (register-with-or-after-producer —
    # config#1393). One PUT site each.
    "scoring/attractiveness_history.py": 1,
    "scoring/attractiveness_trajectory.py": 1,
    "scripts/aggregate_costs.py": 1,
    # Distillation SFT-corpus stats artifact
    # (decision_artifacts/distillation/corpus_stats/{date}.json + latest.json).
    # SECONDARY observability built fail-soft as a non-fatal post-step of the
    # research run (WARNs, never fails the run; signals.json is primary). Reads
    # the whole _sft_raw corpus and rewrites a rebuildable summary; the console
    # Distillation-Corpus panel consumer graceful-degrades when absent → absence
    # is NOT a silent failure and needs no daily freshness-SLA alarm. Per-file
    # PUT pin only; ARTIFACT_REGISTRY row deferred until first Saturday
    # production (register-with-or-after-producer — config#1544). One PUT site
    # (loop over dated + latest keys in compute_corpus_stats).
    "scripts/corpus_stats.py": 1,
    # Champion/challenger leaderboard scorer (config#1221 scanner + config#1223
    # producer; ONE shared engine, ARCHITECTURE §37). Single PUT site
    # (_write_leaderboard) used by both build_scanner_leaderboard →
    # scanner/leaderboard/{date}.json and build_producer_leaderboard →
    # research/producer_leaderboard/{date}.json. OBSERVE-ONLY + fail-soft +
    # cohort-gated — never read by live trading; the consumer (operator review /
    # the cutover gates in OBSERVATION_REGISTRY) graceful-degrades when the
    # artifact is absent or ships n_dates=0, so absence is NOT a silent failure
    # and needs no daily freshness-SLA alarm. Per-file PUT pin only, no
    # ARTIFACT_REGISTRY row (same rationale as build_agent_quality + the shadow
    # substrates this scores).
    "scoring/leaderboard_producers.py": 1,
    # Report-card agent_quality.json (config#1149). Weekly report-card input;
    # the evaluator consumer (crucible-evaluator#59) graceful-degrades each
    # component to a visible N/A-MISSING-INPUT when absent — so absence is NOT
    # a silent failure and needs no daily freshness-SLA alarm. Per-file PUT pin
    # only, no ARTIFACT_REGISTRY row (same rationale as the cost/capture streams).
    "scripts/build_agent_quality.py": 1,
    "scripts/backfill_calibrator_v1_context.py": 2,
    "scripts/backfill_eval_option_b.py": 1,
    "scripts/backfill_orphan_theses.py": 2,
    "scripts/run_judge_cross_validation.py": 1,
    # Think-tank SFT capture flush (config#1579) — writes to the SAME
    # decision_artifacts/_sft_raw/ accumulation stream as
    # graph/llm_cost_tracker.py (producer tag crucible_thinktank), gated on
    # ALPHA_ENGINE_DECISION_CAPTURE_ENABLED. Distillation training data, NOT a
    # load-bearing freshness-SLA artifact — per-file PUT pin only (same
    # rationale as the cost/capture streams above). One PUT site (flush_sft).
    "thinktank/client.py": 1,
    # The think tank's single write chokepoint: ALL thinktank/ namespace
    # artifacts (coverage ledger, theses, themes, events, run manifests, month
    # cost ledger) go through ThinktankStore._put — which is exactly what makes
    # the namespace boundary auditable. OBSERVE-phase producer with NO live
    # consumer yet (admission gate config#1579 P2 / restructure decision
    # config#1580); consumers-to-be graceful-degrade on absence. Per-file PUT
    # pin only; ARTIFACT_REGISTRY rows deferred until the artifacts become
    # load-bearing (register-with-or-after-producer precedent, config#1393).
    "thinktank/storage.py": 1,
}


_SCAN_EXEMPT_PREFIXES: tuple[str, ...] = (
    "tests/",
    "infrastructure/lambdas/",
    ".claude/",
    ".venv/",
    "build/",
)


def _enumerate_put_sites() -> dict[str, int]:
    """Return ``{relative_path: count}`` of production files with PUT sites."""
    result = subprocess.run(
        [
            "git", "grep", "-l", "-E",
            r"(put_object|upload_file)\(",
            "--", "*.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = [
        line for line in result.stdout.splitlines()
        if line and not any(line.startswith(p) for p in _SCAN_EXEMPT_PREFIXES)
    ]
    counts: dict[str, int] = {}
    for rel in files:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        counts[rel] = len(re.findall(r"(?:put_object|upload_file)\(", text))
    return counts


def test_every_producer_file_is_pinned():
    actual = _enumerate_put_sites()
    unpinned = sorted(set(actual.keys()) - set(EXPECTED_PER_FILE_PUT_COUNTS.keys()))
    assert not unpinned, (
        "New producer file(s) with S3 PUT sites detected but not pinned:\n"
        + "\n".join(f"  - {f} ({actual[f]} PUT call(s))" for f in unpinned)
        + "\n\nResolution:\n"
        "  1. Register the new artifact(s) in alpha-engine-config/"
        "private-docs/ARTIFACT_REGISTRY.yaml (or add the prefix to "
        "grandfathered_paths with a one-line reason).\n"
        "  2. Add the file(s) to EXPECTED_PER_FILE_PUT_COUNTS in "
        "tests/test_artifact_registry_coverage.py with the per-file count.\n"
        "  3. Re-run this test."
    )


def test_every_pinned_file_still_exists():
    actual = _enumerate_put_sites()
    stale = sorted(set(EXPECTED_PER_FILE_PUT_COUNTS.keys()) - set(actual.keys()))
    assert not stale, (
        "Pinned file(s) no longer have PUT sites (or no longer exist):\n"
        + "\n".join(f"  - {f}" for f in stale)
        + "\n\nResolution: remove the file from EXPECTED_PER_FILE_PUT_COUNTS. "
        "If the artifact was retired, also retire its row in "
        "alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml."
    )


def test_pinned_counts_match_actual():
    actual = _enumerate_put_sites()
    deltas = []
    for path, expected_count in sorted(EXPECTED_PER_FILE_PUT_COUNTS.items()):
        actual_count = actual.get(path, 0)
        if actual_count != expected_count:
            deltas.append(f"  - {path}: expected={expected_count}, actual={actual_count}")
    assert not deltas, (
        "PUT-site count drift detected:\n"
        + "\n".join(deltas)
        + "\n\nResolution: for each delta, either (a) the PUT count changed "
        "legitimately — register the new artifact in alpha-engine-config/"
        "private-docs/ARTIFACT_REGISTRY.yaml (or grandfather), then bump "
        "the pinned count; or (b) the change was inadvertent — revert."
    )
