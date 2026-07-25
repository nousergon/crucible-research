"""OpenRouter shadow-judge runner (config#2575 items 4-5).

Runs the OpenRouter shadow-judge tier (``evals.judge.evaluate_artifact_openrouter``,
item 2/3) over a date partition's captured artifacts and persists its
verdicts alongside the existing Haiku/Sonnet evals — same S3 prefix
(``evals.judge.DEFAULT_EVAL_PREFIX``), same CloudWatch namespace
(``evals.metrics.DEFAULT_NAMESPACE``), distinguished only by the
``judge_model="openrouter-shadow"`` dimension, exactly the way Haiku and
Sonnet already coexist. Co-location (not an isolated ``judge_only``-style
prefix) is deliberate: the agreement metric this module also computes
(:func:`compute_shadow_agreement`) needs the shadow scores queryable
alongside the primary judges' scores for the same artifacts.

**Shadow-only, no decision authority (config#2575 binding constraint,
carried forward from config#1676/#1675).** This module:

* Does NOT feed the OpenRouter shadow judge's scores into
  ``should_escalate_to_sonnet`` or any escalation-routing decision.
* Does NOT wire into the production Saturday Batches-API SF/Lambda
  chain (Submit/Poll/Process) — that is deploy-risk-bearing
  infrastructure (new SF states, IAM grants, Lambda image changes) that
  is explicitly OUT of this pass's scope; see the config#2575 PR
  description for the follow-up issue this defers to. This module is a
  standalone, synchronously-invocable runner (ad-hoc / cron-invoked /
  manually-triggered) that can be wired into the SF chain later without
  changing its call contract.
* Is READ by nothing else in this codebase's decision paths — enforced
  structurally by ``evals.judge_models.SHADOW_LOGICAL_KEYS``, which every
  escalation/consumption call site should check before trusting a
  ``judge_model`` value (see ``evals.orchestrator.should_escalate_to_sonnet``
  and its docstring note).

Agreement metric (item 5): :func:`compute_shadow_agreement` compares the
shadow judge's per-dimension scores against the PRIMARY judge (Haiku, the
one that runs on every artifact — mirrors the existing
Haiku-vs-Sonnet-on-escalated-subset agreement convention in
``evals/cross_validation.py``) for every (judged_agent_id, run_id) pair
both judges scored, reusing the exact same quadratic-weighted-kappa +
exact/±1/MAD statistics ``evals.cross_validation.summarize_agreement``
already computes for the human-vs-judge case — same math, different
rater pair.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3

from evals.cross_validation import DimensionAgreement, RatingPair, summarize_agreement
from evals.judge import (
    DEFAULT_EVAL_PREFIX,
    evaluate_artifact_openrouter,
    persist_eval_artifact,
    resolve_rubric_for_agent,
)
from evals.judge_models import HAIKU, OPENROUTER_SHADOW
from evals.metrics import DEFAULT_NAMESPACE, emit_eval_metric
from evals.orchestrator import _load_capture_artifact, list_capture_keys
from graph.state_schemas import RubricEvalArtifact

logger = logging.getLogger(__name__)

_BUCKET_DEFAULT = "alpha-engine-research"


def run_shadow_judge_over_date(
    *,
    date: str,
    bucket: str = _BUCKET_DEFAULT,
    judge_model: str = OPENROUTER_SHADOW.logical_key,
    s3_client: Any | None = None,
    cloudwatch_client: Any | None = None,
    emit_metrics: bool = True,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Score every captured artifact under ``date`` with the OpenRouter
    shadow-judge tier and persist the verdicts. Returns a summary dict
    mirroring ``evals.orchestrator.evaluate_corpus``'s shape (same field
    names where applicable) for consistency with existing SF-result
    inspection tooling.

    Per-artifact errors (load / eval / persist) are logged + accumulated
    in ``failed`` and the run continues — same "eval is observability,
    not a gate" posture as the primary orchestrator, doubly true for a
    tier that has NO consumers yet.

    ``judge_model`` defaults to the registered shadow logical key
    (``OPENROUTER_SHADOW.logical_key``) — MUST resolve to a key in
    ``evals.judge_models.SHADOW_LOGICAL_KEYS`` or this function raises,
    since running this path under a non-shadow logical key would
    silently grant an unvalidated judge tier's scores the same S3/metric
    identity as an authoritative one.
    """
    from evals.judge_models import SHADOW_LOGICAL_KEYS

    if judge_model not in SHADOW_LOGICAL_KEYS:
        raise ValueError(
            f"run_shadow_judge_over_date refuses judge_model={judge_model!r} "
            f"— not in evals.judge_models.SHADOW_LOGICAL_KEYS "
            f"({sorted(SHADOW_LOGICAL_KEYS)}). This runner is shadow-only "
            f"by construction; an authoritative judge_model must go "
            f"through the primary evals.orchestrator path instead."
        )

    s3 = s3_client or boto3.client("s3")
    cw = cloudwatch_client or (boto3.client("cloudwatch") if emit_metrics else None)

    capture_keys = list_capture_keys(s3, date=date, bucket=bucket)

    from evals.judge import _new_judge_run_id

    judge_run_id = _new_judge_run_id()

    evaluated = 0
    skipped_unmapped = 0
    skipped_empty_or_degenerate = 0
    metric_emission_failures = 0
    failed: list[dict[str, str]] = []
    persisted_keys: list[str] = []

    logger.info(
        "[openrouter_shadow] start date=%s bucket=%s capture_keys=%d "
        "judge_model=%s (SHADOW — no decision authority, config#2575)",
        date, bucket, len(capture_keys), judge_model,
    )

    for key in capture_keys:
        try:
            artifact = _load_capture_artifact(s3, key=key, bucket=bucket)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[openrouter_shadow] load failed for %s", key)
            failed.append({"key": key, "agent_id": "<unknown>", "stage": "load", "error": str(exc)})
            continue

        if resolve_rubric_for_agent(artifact.agent_id) is None:
            skipped_unmapped += 1
            continue

        try:
            eval_artifact = evaluate_artifact_openrouter(
                artifact,
                judge_run_id=judge_run_id,
                judge_model=judge_model,
                api_key=api_key,
                judged_artifact_s3_key=key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[openrouter_shadow] eval failed for agent_id=%s key=%s",
                artifact.agent_id, key,
            )
            failed.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "eval_openrouter_shadow", "error": str(exc),
            })
            continue

        if eval_artifact.judge_skip_reason is not None:
            skipped_empty_or_degenerate += 1
            # Skip-marker evals are still persisted (mirrors the primary
            # orchestrator) so the corpus accounting stays consistent —
            # a skip is a recorded fact, not a silent gap.

        try:
            persisted_key = persist_eval_artifact(
                eval_artifact, s3_client=s3, bucket=bucket, prefix=DEFAULT_EVAL_PREFIX,
            )
            persisted_keys.append(persisted_key)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[openrouter_shadow] persist failed for agent_id=%s key=%s",
                artifact.agent_id, key,
            )
            failed.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "persist", "error": str(exc),
            })
            continue

        evaluated += 1

        if emit_metrics:
            try:
                emit_eval_metric(
                    eval_artifact, namespace=DEFAULT_NAMESPACE, cloudwatch_client=cw,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[openrouter_shadow] cloudwatch emit failed for agent_id=%s",
                    artifact.agent_id,
                )
                metric_emission_failures += 1

    summary = {
        "date": date,
        "judge_model": judge_model,
        "judge_run_id": judge_run_id,
        "capture_keys_seen": len(capture_keys),
        "evaluated": evaluated,
        "skipped_unmapped": skipped_unmapped,
        "skipped_empty_or_degenerate": skipped_empty_or_degenerate,
        "failed": failed,
        "metric_emission_failures": metric_emission_failures,
        "persisted_keys": persisted_keys,
        "shadow_only": True,
    }
    logger.info(
        "[openrouter_shadow] done date=%s evaluated=%d failed=%d "
        "skipped_unmapped=%d skipped_empty_or_degenerate=%d",
        date, evaluated, len(failed), skipped_unmapped, skipped_empty_or_degenerate,
    )
    return summary


# ── Agreement metric (item 5) ──────────────────────────────────────────


def _load_eval_artifacts_for_run(
    s3: Any, *, date: str, bucket: str, judge_models: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, RubricEvalArtifact]]:
    """Load every persisted eval artifact under ``date``'s capture
    partition whose ``judge_model`` is in ``judge_models``, grouped by
    ``(judged_agent_id, run_id)`` → ``{judge_model: RubricEvalArtifact}``.

    Reads via the ``_eval_by_capture/{date}/manifest.json`` index (same
    manifest ``load_already_judged_keys`` uses) rather than re-listing
    the full ``_eval/`` prefix, so this stays cheap even as the eval
    corpus grows across weeks.
    """
    manifest_key = f"decision_artifacts/_eval_by_capture/{date}/manifest.json"
    try:
        raw = s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
        manifest = json.loads(raw)
    except Exception:  # noqa: BLE001 — absent/unreadable manifest = nothing to pair
        logger.warning(
            "[openrouter_shadow] no readable eval manifest at %s — "
            "agreement metric has nothing to pair for date=%s",
            manifest_key, date,
        )
        return {}

    grouped: dict[tuple[str, str], dict[str, RubricEvalArtifact]] = {}
    for entry in manifest.get("entries", manifest.get("evals", [])) or []:
        # ``eval_s3_key`` is the manifest's actual field name — see
        # ``evals.eval_manifest._make_manifest_entry``. Also accept the
        # generic ``key`` name defensively in case a hand-built/legacy
        # manifest uses it.
        eval_key = entry.get("eval_s3_key") or entry.get("key")
        if not eval_key:
            continue
        try:
            raw_art = s3.get_object(Bucket=bucket, Key=eval_key)["Body"].read()
            art = RubricEvalArtifact.model_validate_json(raw_art)
        except Exception:  # noqa: BLE001 — one unreadable artifact must not sink the pairing
            logger.warning(
                "[openrouter_shadow] unreadable eval artifact %s — skipped for agreement",
                eval_key,
            )
            continue
        if art.judge_model not in judge_models:
            continue
        pair_key = (art.judged_agent_id, art.run_id)
        grouped.setdefault(pair_key, {})[art.judge_model] = art
    return grouped


def compute_shadow_agreement(
    *,
    date: str,
    bucket: str = _BUCKET_DEFAULT,
    primary_judge_model: str = HAIKU.logical_key,
    shadow_judge_model: str = OPENROUTER_SHADOW.logical_key,
    s3_client: Any | None = None,
) -> list[DimensionAgreement]:
    """Agreement between the OpenRouter shadow judge and the PRIMARY
    judge (Haiku by default — the tier that runs on every artifact,
    giving the largest paired-sample count) for every artifact both
    scored under ``date``'s capture partition.

    Reuses ``evals.cross_validation.summarize_agreement`` — the exact
    same quadratic-weighted-kappa / exact-match / ±1-tolerance / MAD
    statistics already computed for the human-vs-judge case in that
    module, applied here to a judge-vs-judge rater pair instead. This is
    the "agreement/disagreement metric across all active judge pairs"
    config#2575 item 5 asks for, extending the existing
    Haiku-vs-Sonnet-on-escalated-subset convention with a
    Sonnet-vs-OpenRouter-shadow (via Haiku as the common primary) pair.

    Returns an EMPTY list (not an error) when no artifact was scored by
    both judges under ``date`` — the shadow tier may not have run yet,
    or may not have overlapped with the primary judge's run for this
    partition. An empty result is a legitimate "not enough shadow data
    yet" signal, not a failure.
    """
    s3 = s3_client or boto3.client("s3")
    grouped = _load_eval_artifacts_for_run(
        s3, date=date, bucket=bucket,
        judge_models=(primary_judge_model, shadow_judge_model),
    )

    pairs: list[RatingPair] = []
    for (agent_id, run_id), by_model in grouped.items():
        primary = by_model.get(primary_judge_model)
        shadow = by_model.get(shadow_judge_model)
        if primary is None or shadow is None:
            continue
        if primary.judge_skip_reason is not None or shadow.judge_skip_reason is not None:
            continue  # skip-marker evals carry no dimension scores to compare

        shadow_by_dim = {d.dimension: d.score for d in shadow.dimension_scores}
        for dim in primary.dimension_scores:
            shadow_score = shadow_by_dim.get(dim.dimension)
            if shadow_score is None:
                continue
            # RatingPair is named for its original human-vs-judge use
            # (evals/cross_validation.py) — reused here for a judge-vs-judge
            # pair: ``human_score`` carries the PRIMARY judge's (Haiku's)
            # score, ``judge_score`` carries the shadow judge's score. Same
            # dataclass, same downstream stats (summarize_agreement doesn't
            # care which rater is "human").
            pairs.append(
                RatingPair(
                    artifact_nn=f"{agent_id}:{run_id}",
                    agent_id=agent_id,
                    rubric_family=primary.rubric_id,
                    run_id=run_id,
                    dimension=dim.dimension,
                    human_score=dim.score,
                    judge_score=shadow_score,
                    judge_model=shadow_judge_model,
                )
            )

    if not pairs:
        logger.info(
            "[openrouter_shadow] compute_shadow_agreement: no overlapping "
            "(%s, %s) pairs for date=%s — shadow data not yet accrued or "
            "no overlap this partition",
            primary_judge_model, shadow_judge_model, date,
        )
        return []

    return summarize_agreement(pairs)
