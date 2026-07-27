"""Refresh ``tests/fixtures/llm_outputs/`` from S3-captured decision artifacts.

PR 3 of the typed-state workstream wires every agent output to
``s3://alpha-engine-research/decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json``.
This script downloads the latest captured artifacts for each agent and
overwrites the corresponding fixture, keeping the fixture corpus in
sync with real Anthropic output shapes.

Usage from repo root:
    python scripts/refresh_llm_fixtures.py --since 2026-04-30
    python scripts/refresh_llm_fixtures.py --date 2026-05-02
    python scripts/refresh_llm_fixtures.py --dry-run  # preview, no writes

Reviewing the diff after refresh is the canonical drift-detection
moment: structural changes (new fields, missing fields, type shifts)
signal that the schema or prompt is drifting and needs attention.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import boto3

_BUCKET = "alpha-engine-research"
_PREFIX = "decision_artifacts"
_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "llm_outputs"

# Map captured agent_id (S3 prefix segment) → fixture filename.
# The captured artifact is the LLM-extraction shape; for state-level
# schemas (SectorTeamOutput, InvestmentThesis, CIODecision) we curate
# manually since those are post-aggregation, not directly captured.
_AGENT_TO_FIXTURE: dict[str, str] = {
    "macro_economist": "macro_economist_raw_output.json",
    "macro_critic": "macro_critic_output.json",
    "quant_analyst": "quant_analyst_output.json",
    "qual_analyst": "qual_analyst_output.json",
    "peer_review_quant_addition": "quant_acceptance_verdict.json",
    "peer_review_joint_finalization": "joint_finalization_output.json",
    "held_thesis_update": "held_thesis_update_llm_output.json",
    "ic_cio": "cio_raw_output.json",
}


def _list_artifacts_for_date(s3, date_str: str, agent_id: str) -> list[str]:
    """List S3 keys for a given date + agent. Returns full keys."""
    y, m, d = date_str.split("-")
    prefix = f"{_PREFIX}/{y}/{m}/{d}/{agent_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])
    return keys


def _latest_artifact_since(s3, since_str: str, agent_id: str) -> str | None:
    """Walk dates from ``since_str`` forward to today; return the key of
    the most recent artifact for ``agent_id`` or None if none found."""
    since = datetime.strptime(since_str, "%Y-%m-%d").date()
    today = date.today()
    cur = today
    # Walk backwards from today to since (most-recent-first).
    while cur >= since:
        keys = _list_artifacts_for_date(s3, cur.isoformat(), agent_id)
        if keys:
            return sorted(keys)[-1]  # latest run_id wins
        cur = date.fromordinal(cur.toordinal() - 1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since", default=None,
        help="Walk dates backwards from today until this date "
             "(YYYY-MM-DD). Default: 7 days ago.",
    )
    parser.add_argument(
        "--date", default=None,
        help="Refresh from artifacts on this single date (YYYY-MM-DD). "
             "Overrides --since.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be refreshed without writing.",
    )
    args = parser.parse_args()

    if args.date:
        since = args.date
    elif args.since:
        since = args.since
    else:
        # Default: 7 days back
        since = date.fromordinal(date.today().toordinal() - 7).isoformat()

    s3 = boto3.client("s3")
    n_refreshed = 0
    n_missing = 0

    for agent_id, fixture_name in _AGENT_TO_FIXTURE.items():
        if args.date:
            keys = _list_artifacts_for_date(s3, args.date, agent_id)
            key = sorted(keys)[-1] if keys else None
        else:
            key = _latest_artifact_since(s3, since, agent_id)

        if key is None:
            print(f"[skip] {agent_id}: no artifact found since {since}")
            n_missing += 1
            continue

        body = s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read()
        artifact = json.loads(body)
        # Decision-artifact wraps the raw output in metadata. The agent's
        # actual output is at ``artifact["agent_output"]`` per the
        # DecisionArtifact schema in alpha-engine-lib.
        agent_output = artifact.get("agent_output")
        if agent_output is None:
            print(f"[skip] {agent_id}: artifact has no agent_output key "
                  f"(key={key})")
            n_missing += 1
            continue

        target = _FIXTURE_DIR / fixture_name
        if args.dry_run:
            print(f"[dry-run] would write {target} ← s3://{_BUCKET}/{key}")
        else:
            with target.open("w") as f:
                json.dump(agent_output, f, indent=2, sort_keys=False)
                f.write("\n")
            print(f"[ok] {fixture_name} ← {key}")
        n_refreshed += 1

    print(f"\nRefreshed {n_refreshed}/{len(_AGENT_TO_FIXTURE)} fixtures "
          f"({n_missing} skipped)")
    if args.dry_run:
        print("(dry-run — no files written)")
    print(
        "\nReview ``git diff tests/fixtures/llm_outputs/`` before "
        "committing. Structural changes are a drift signal."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
