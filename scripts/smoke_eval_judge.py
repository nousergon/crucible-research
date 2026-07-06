"""Real-LLM smoke test for the LLM-as-judge pipeline.

Downloads captured decision artifacts from S3, runs the judge module
against them, and prints rubric scores + per-dimension reasoning to
stdout for manual cross-check (ROADMAP §1627: "cross-validate the LLM
judge with manual rating on a small sample to confirm the LLM judge
correlates with human judgment").

Calibration tool, not a wired pipeline step. Does NOT persist eval
artifacts to S3 — the smoke is for inspecting whether scores are
plausible and whether the judge's reasoning is grounded in the
actual artifact content.

Usage from repo root:
    python scripts/smoke_eval_judge.py --date 2026-05-02
    python scripts/smoke_eval_judge.py --date 2026-05-02 --agents macro_economist
    python scripts/smoke_eval_judge.py --date 2026-05-02 --judge-model claude-sonnet-4-6

Default agent set is ``macro_economist,ic_cio`` because the Sat 5/2
SF ran *before* PR #87 (capture split), so the per-team artifacts on
that date are still in the old combined ``sector_team:*`` shape and
do not match the ``sector_quant`` / ``sector_qual`` /
``sector_peer_review`` rubrics. macro + cio artifact shapes are
stable across the split. After Sat 5/9 SF runs with the new capture
format, ``--agents`` can be widened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Repo root on sys.path so ``from evals.judge import ...`` resolves
# when invoked as ``python scripts/smoke_eval_judge.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env before importing config-dependent modules so ANTHROPIC_API_KEY
# is visible to ChatAnthropic at call time.
load_dotenv(override=True)

from nousergon_lib.decision_capture import DecisionArtifact  # noqa: E402

from evals.judge import evaluate_artifact, resolve_rubric_for_agent  # noqa: E402


_BUCKET = "alpha-engine-research"
_DEFAULT_AGENTS = ("macro_economist", "ic_cio")


def _build_s3_prefix(date_str: str, agent_id: str) -> str:
    """``decision_artifacts/{YYYY}/{MM}/{DD}/{agent_id}/`` — partition layout
    matches what ``alpha_engine_lib.decision_capture`` writes."""
    y, m, d = date_str.split("-")
    return f"decision_artifacts/{y}/{m}/{d}/{agent_id}/"


def _list_artifact_keys(s3, prefix: str) -> list[str]:
    resp = s3.list_objects_v2(Bucket=_BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]


def _load_artifact(s3, key: str) -> DecisionArtifact:
    raw = s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read()
    return DecisionArtifact(**json.loads(raw))


def _print_eval(artifact: DecisionArtifact, eval_artifact, key: str) -> None:
    print(f"\n=== eval: {artifact.agent_id} (run_id={artifact.run_id}) ===")
    print(f"  source: s3://{_BUCKET}/{key}")
    print(
        f"  rubric: {eval_artifact.rubric_id} v{eval_artifact.rubric_version}  "
        f"judge: {eval_artifact.judge_model}"
    )
    width = max((len(d.dimension) for d in eval_artifact.dimension_scores), default=0)
    for dim in eval_artifact.dimension_scores:
        print(f"  {dim.dimension.ljust(width)}  {dim.score}  — {dim.reasoning}")
    print(f"  overall: {eval_artifact.overall_reasoning}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", required=True,
        help="S3 partition date (YYYY-MM-DD) of captured artifacts to score.",
    )
    parser.add_argument(
        "--agents", default=",".join(_DEFAULT_AGENTS),
        help=(
            "Comma-separated agent_ids to smoke. Default: macro_economist,ic_cio "
            "(stable across the PR #87 capture split). Add sector_quant:* / "
            "sector_qual:* / sector_peer_review:* once Sat 5/9 corpus is captured."
        ),
    )
    parser.add_argument(
        "--judge-model", default="claude-haiku-4-5",
        help="Judge LLM. Haiku for cost-tier smoke; Sonnet to spot-check nuance.",
    )
    args = parser.parse_args()

    requested_agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    s3 = boto3.client("s3")

    total_evaluated = 0
    total_skipped = 0
    for agent_id in requested_agents:
        rubric = resolve_rubric_for_agent(agent_id)
        if rubric is None:
            print(f"[skip] no rubric mapped for agent_id={agent_id!r}", file=sys.stderr)
            total_skipped += 1
            continue

        prefix = _build_s3_prefix(args.date, agent_id)
        keys = _list_artifact_keys(s3, prefix)
        if not keys:
            print(f"[skip] no artifacts under s3://{_BUCKET}/{prefix}", file=sys.stderr)
            total_skipped += 1
            continue

        for key in keys:
            artifact = _load_artifact(s3, key)
            eval_artifact = evaluate_artifact(
                artifact,
                judge_model=args.judge_model,
                judged_artifact_s3_key=key,
            )
            _print_eval(artifact, eval_artifact, key)
            total_evaluated += 1

    print(
        f"\n[summary] evaluated={total_evaluated} skipped={total_skipped} "
        f"judge_model={args.judge_model} date={args.date}",
        file=sys.stderr,
    )
    return 0 if total_evaluated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
