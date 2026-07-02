"""LLM-as-judge orchestrator (PR 3b of ROADMAP P3.1, Phase 2 P1).

Fans the judge module out over every captured DecisionArtifact for a
given date partition, applies two-tier sampling (Haiku default + Sonnet
escalation), and persists results to S3.

Two-tier sampling logic (per ROADMAP §1626):

  1. **Haiku for cost on every weekly run.** Every artifact whose
     ``agent_id`` resolves to a rubric is scored with Haiku.

  2. **Sonnet for nuance on a sampled subset.** A Sonnet pass also
     runs for any artifact when *either* of these holds:
       - ``force_sonnet_pass`` was passed in by the caller (used by
         the Saturday SF every 4th run — the run-frequency cadence is
         a SF concern, not a Lambda concern, so this flag is the
         contract surface).
       - The Haiku eval flagged a dimension score below
         ``haiku_escalate_threshold`` (default 3) — Haiku itself said
         the artifact has a concerning gap; Sonnet's nuance is worth
         the cost to confirm or refute.

Per-artifact escalation (rather than batch-level "if any artifact's
Haiku score < 3, re-run all of them with Sonnet") is the deliberate
choice — only the borderline ones get the expensive pass, which keeps
weekly judging cost bounded while preserving diagnostic depth where
it matters.

Eval is observability, NOT a gate. Errors during evaluation of any
single artifact are logged loudly and accumulated in the result
dict's ``failed`` list — the run continues so other artifacts still
get scored. Callers (the Lambda handler) decide whether a non-empty
``failed`` warrants alarming.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import boto3

from alpha_engine_lib.decision_capture import DecisionArtifact

from agents.prompt_loader import load_prompt
from evals.judge import (
    DEFAULT_EVAL_PREFIX,
    DEFAULT_MAX_TOKENS,
    build_batch_request,
    build_eval_s3_key,
    decode_custom_id,
    encode_custom_id,
    evaluate_artifact,
    parse_batch_message,
    persist_eval_artifact,
    resolve_rubric_for_agent,
    _make_skip_eval_artifact,
)
from evals.judge_models import request_model_for
from evals.metrics import DEFAULT_NAMESPACE, emit_eval_metric
from graph.state_schemas import RubricEvalArtifact

logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────


DEFAULT_HAIKU_MODEL = "claude-haiku-4-5"
"""Cost-tier judge — runs on every artifact every weekly run."""

DEFAULT_SONNET_MODEL = "claude-sonnet-4-6"
"""Nuance-tier judge — runs on the sampled subset (force_sonnet_pass
or Haiku-flagged borderline)."""

DEFAULT_HAIKU_ESCALATE_THRESHOLD = 3
"""Any Haiku dimension score strictly below this value escalates the
artifact to a Sonnet pass. 3 is the rubric midpoint — below 3 means
Haiku flagged a real problem, not just an average dimension."""

JUDGE_ONLY_EVAL_PREFIX = "decision_artifacts/_eval_judge_only/"
"""S3 path prefix for ``judge_only`` test-track outputs (PR 4e).
Isolated from the prod prefix so test runs don't pollute the
rolling-mean window or the dashboard's quality-trend page."""

JUDGE_ONLY_CW_NAMESPACE = "AlphaEngine/EvalJudgeOnly"
"""CloudWatch metric namespace for ``judge_only`` test-track emissions
(PR 4e). Distinct namespace keeps test datapoints out of the prod
``AlphaEngine/Eval`` stream the alarm + rolling-mean Lambda read."""

_BUCKET_DEFAULT = "alpha-engine-research"


# ── Sampling decision ─────────────────────────────────────────────────────


def should_escalate_to_sonnet(
    haiku_eval: RubricEvalArtifact,
    *,
    threshold: int = DEFAULT_HAIKU_ESCALATE_THRESHOLD,
) -> bool:
    """Per-artifact escalation: True iff any Haiku dimension score is
    below ``threshold``."""
    return any(d.score < threshold for d in haiku_eval.dimension_scores)


# ── Capture-corpus listing ────────────────────────────────────────────────


def expand_lookback_dates(date: str, lookback_days: int) -> list[str]:
    """The ``lookback_days`` TRADING days strictly before ``date``
    (newest first) — the Submit handler's ``capture_lookback_days``
    expansion.

    Trading days, not calendar days (Brian, 2026-07-02): daily
    producers (thinktank, config#1579) partition their captures by
    ``trading_day`` per the fleet date convention, so weekend/holiday
    runs land in the LAST trading day's partition (a Friday partition
    accrues Fri+Sat+Sun runs' outputs — expected). Enumerating trading
    days therefore covers every capture. The boundary case — a weekend
    run writing into Friday's partition AFTER Saturday's batch already
    ran — is handled by the already-judged dedup in
    ``build_batch_plan``, which re-enumerates that partition the NEXT
    week (a 6-trading-day lookback from Saturday reaches the prior
    Friday) and skips only what was already judged."""
    from datetime import date as _date

    from nousergon_lib import trading_calendar as _tc

    y, m, d = (int(x) for x in date.split("-"))
    cur = _date(y, m, d)
    out: list[str] = []
    for _ in range(lookback_days):
        cur = _tc.previous_trading_day(cur)
        out.append(str(cur))
    return out


def _build_capture_prefix(date: str) -> str:
    """``decision_artifacts/{Y}/{M}/{D}/`` — partition layout that
    ``alpha_engine_lib.decision_capture`` writes to."""
    y, m, d = date.split("-")
    return f"decision_artifacts/{y}/{m}/{d}/"


def list_capture_keys(
    s3: Any,
    *,
    date: str,
    bucket: str,
    agent_id_prefixes: list[str] | None = None,
) -> list[str]:
    """Enumerate every captured artifact key under the date partition.

    Excludes the ``_eval/`` subtree — those are eval artifacts (output
    of this very orchestrator), not capture artifacts that need scoring.
    Excludes any keys not ending in ``.json`` (defensive — the partition
    should only contain captures, but a stray prefix shouldn't crash
    the run).

    ``agent_id_prefixes`` (optional) filters to keys whose agent segment
    (``decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json``) starts
    with one of the given prefixes — the seam that lets a second Submit
    invocation judge one artifact family (e.g. ``thinktank_``) without
    re-judging everything else in the partition (config#1579 P2).
    """
    prefix = _build_capture_prefix(date)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/_eval/" in key or not key.endswith(".json"):
                continue
            if agent_id_prefixes is not None:
                parts = key.split("/")
                agent_seg = parts[-2] if len(parts) >= 2 else ""
                if not any(agent_seg.startswith(p) for p in agent_id_prefixes):
                    continue
            keys.append(key)
    return keys


def load_already_judged_keys(
    s3: Any, *, dates: list[str], bucket: str
) -> set[str]:
    """Capture keys already scored, per the ``_eval_by_capture``
    manifests (which index ACTUAL written evals — a batch that failed
    before persisting evals leaves no manifest entries, so its
    artifacts correctly re-enter the next plan; the manifests'
    ~24h eventual consistency is far inside the weekly cadence).
    A missing/unreadable manifest yields no dedup for that date — the
    failure mode is a harmless duplicate eval, never a silent skip."""
    judged: set[str] = set()
    for d in dates:
        key = f"decision_artifacts/_eval_by_capture/{d}/manifest.json"
        try:
            raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception:  # noqa: BLE001 — absent manifest = nothing judged yet
            continue
        try:
            manifest = json.loads(raw)
        except Exception:  # noqa: BLE001 — unreadable manifest = no dedup (dup evals, not skips)
            logger.warning("[batch_plan] unreadable eval manifest %s — no dedup for %s", key, d)
            continue
        for entry in manifest.get("entries", manifest.get("evals", [])) or []:
            jk = entry.get("judged_artifact_s3_key")
            if jk:
                judged.add(jk)
    return judged


def _load_capture_artifact(
    s3: Any, *, key: str, bucket: str,
) -> DecisionArtifact:
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return DecisionArtifact(**json.loads(raw))


# ── Orchestration ─────────────────────────────────────────────────────────


def evaluate_corpus(
    *,
    date: str,
    bucket: str = _BUCKET_DEFAULT,
    haiku_model: str = DEFAULT_HAIKU_MODEL,
    sonnet_model: str = DEFAULT_SONNET_MODEL,
    force_sonnet_pass: bool = False,
    haiku_escalate_threshold: int = DEFAULT_HAIKU_ESCALATE_THRESHOLD,
    dry_run: bool = False,
    judge_only: bool = False,
    s3_client: Optional[Any] = None,
    cloudwatch_client: Optional[Any] = None,
    emit_metrics: bool = True,
) -> dict[str, Any]:
    """Score every captured artifact under ``date`` per the two-tier
    sampling policy. Returns a summary dict suitable for SF inspection.

    Hard-fails on listing errors (bucket missing, S3 unreachable). Per
    artifact: a load / eval / persist error is logged + appended to
    ``failed`` and the run continues with the next artifact. Eval is
    observability — one rubric or LLM hiccup must not silently halt
    every other agent's eval.

    CloudWatch metric emission (PR 4a, ROADMAP §1634): each persisted
    eval also pushes one ``AlphaEngine/Eval/agent_quality_score``
    datapoint per rubric dimension. Metric write failures are
    observability OF observability — they're caught + counted in
    ``summary['metric_emission_failures']`` but never halt the run.
    Set ``emit_metrics=False`` to disable in tests / local replay.

    PR 4e test-track flags:

    * ``dry_run=True`` — list artifacts + resolve rubrics + render
      rubric prompts, but do NOT call Anthropic, do NOT persist eval
      artifacts, do NOT emit metrics. Returns ``would_evaluate`` in
      the summary so operators can confirm what WOULD have run.
      Cost: $0.

    * ``judge_only=True`` — real LLM calls and full pipeline, but
      writes eval artifacts under ``decision_artifacts/_eval_judge_only/``
      and emits CloudWatch metrics under ``AlphaEngine/EvalJudgeOnly``
      so test runs don't pollute prod observability. Cost: real
      judge-LLM calls (~$0.005-$0.05 per run vs ~$2-5 for a full
      Research re-run that this avoids).

    The two flags compose: ``dry_run=True, judge_only=True`` is the
    cheapest end-to-end smoke (lists + renders prompts against prod
    captures, no LLM, no writes anywhere).
    """
    s3 = s3_client or boto3.client("s3")
    cw = cloudwatch_client or (
        boto3.client("cloudwatch") if (emit_metrics and not dry_run) else None
    )
    eval_prefix = JUDGE_ONLY_EVAL_PREFIX if judge_only else DEFAULT_EVAL_PREFIX
    cw_namespace = JUDGE_ONLY_CW_NAMESPACE if judge_only else DEFAULT_NAMESPACE

    capture_keys = list_capture_keys(s3, date=date, bucket=bucket)

    # One judge_run_id per evaluate_corpus invocation — all artifacts
    # emitted by this run cluster under _eval/{date}/{judge_run_id}/.
    from evals.judge import _new_judge_run_id

    judge_run_id = _new_judge_run_id()

    haiku_evaluated = 0
    sonnet_evaluated = 0
    skipped_unmapped = 0
    skipped_empty_input = 0
    skipped_degenerate_input = 0
    metric_emission_failures = 0
    failed: list[dict[str, str]] = []
    persisted_keys: list[str] = []
    would_evaluate: list[dict[str, str]] = []

    def _try_emit(eval_artifact: RubricEvalArtifact) -> None:
        nonlocal metric_emission_failures
        if not emit_metrics or dry_run:
            return
        try:
            emit_eval_metric(
                eval_artifact,
                namespace=cw_namespace,
                cloudwatch_client=cw,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[eval_orchestrator] cloudwatch emit failed for "
                "agent_id=%s judge=%s",
                eval_artifact.judged_agent_id, eval_artifact.judge_model,
            )
            metric_emission_failures += 1

    logger.info(
        "[eval_orchestrator] start date=%s bucket=%s capture_keys=%d "
        "haiku_model=%s sonnet_model=%s force_sonnet=%s threshold=%d",
        date, bucket, len(capture_keys), haiku_model, sonnet_model,
        force_sonnet_pass, haiku_escalate_threshold,
    )

    for key in capture_keys:
        try:
            artifact = _load_capture_artifact(s3, key=key, bucket=bucket)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[eval_orchestrator] load failed for %s", key)
            failed.append({"key": key, "agent_id": "<unknown>", "stage": "load", "error": str(exc)})
            continue

        rubric = resolve_rubric_for_agent(artifact.agent_id)
        if rubric is None:
            skipped_unmapped += 1
            continue

        # dry_run short-circuits all LLM calls, persists, and metric
        # writes — operator inspects ``would_evaluate`` to confirm
        # the Lambda would touch the right corpus before paying for
        # real Haiku/Sonnet calls.
        if dry_run:
            would_evaluate.append({
                "key": key,
                "agent_id": artifact.agent_id,
                "rubric": rubric,
            })
            continue

        # Haiku tier — every mapped artifact every run.
        try:
            haiku_eval = evaluate_artifact(
                artifact, judge_run_id=judge_run_id,
                judge_model=haiku_model, judged_artifact_s3_key=key,
            )
            haiku_persisted_key = persist_eval_artifact(
                haiku_eval, s3_client=s3, bucket=bucket, prefix=eval_prefix,
            )
            haiku_evaluated += 1
            persisted_keys.append(haiku_persisted_key)
            _try_emit(haiku_eval)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[eval_orchestrator] haiku eval failed for %s (%s)",
                key, artifact.agent_id,
            )
            failed.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "haiku", "error": str(exc),
            })
            # Skip the Sonnet escalation if Haiku itself failed —
            # there's no haiku_eval to inspect for the threshold gate.
            continue

        # Empty-input structural skip — judge.evaluate_artifact already
        # short-circuited (no LLM call), persisted a skip-marker eval
        # with empty dimension_scores + judge_skip_reason set. Don't
        # escalate to Sonnet (nothing to evaluate); count separately
        # for ops visibility. Split between the two skip families:
        # ``precluded_by_empty_upstream`` (agent never ran) vs
        # ``degenerate_input`` (agent ran but inputs were degenerate;
        # added 2026-05-13).
        if haiku_eval.judge_skip_reason is not None:
            if haiku_eval.judge_skip_reason == "degenerate_input":
                skipped_degenerate_input += 1
            else:
                skipped_empty_input += 1
            continue

        # Sonnet tier — sampled subset.
        escalate = force_sonnet_pass or should_escalate_to_sonnet(
            haiku_eval, threshold=haiku_escalate_threshold,
        )
        if not escalate:
            continue

        try:
            sonnet_eval = evaluate_artifact(
                artifact, judge_run_id=judge_run_id,
                judge_model=sonnet_model, judged_artifact_s3_key=key,
            )
            sonnet_persisted_key = persist_eval_artifact(
                sonnet_eval, s3_client=s3, bucket=bucket, prefix=eval_prefix,
            )
            sonnet_evaluated += 1
            persisted_keys.append(sonnet_persisted_key)
            _try_emit(sonnet_eval)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[eval_orchestrator] sonnet eval failed for %s (%s)",
                key, artifact.agent_id,
            )
            failed.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "sonnet", "error": str(exc),
            })

    logger.info(
        "[eval_orchestrator] done date=%s haiku=%d sonnet=%d "
        "skipped_unmapped=%d skipped_empty_input=%d "
        "skipped_degenerate_input=%d failed=%d "
        "metric_emission_failures=%d",
        date, haiku_evaluated, sonnet_evaluated, skipped_unmapped,
        skipped_empty_input, skipped_degenerate_input,
        len(failed), metric_emission_failures,
    )

    return {
        "date": date,
        "capture_keys_total": len(capture_keys),
        "haiku_evaluated": haiku_evaluated,
        "sonnet_evaluated": sonnet_evaluated,
        "skipped_unmapped": skipped_unmapped,
        "skipped_empty_input": skipped_empty_input,
        "skipped_degenerate_input": skipped_degenerate_input,
        "metric_emission_failures": metric_emission_failures,
        "failed": failed,
        "persisted_keys": persisted_keys,
        "haiku_model": haiku_model,
        "sonnet_model": sonnet_model,
        "force_sonnet_pass": force_sonnet_pass,
        "dry_run": dry_run,
        "judge_only": judge_only,
        "eval_prefix": eval_prefix,
        "cw_namespace": cw_namespace,
        "would_evaluate": would_evaluate,
    }


# ── Batches API path ──────────────────────────────────────────────────────
#
# Replaces the sequential ``evaluate_corpus`` loop above for the
# Saturday SF run with three discrete phases (Submit, Poll, Process)
# wired together by SF Wait+Choice. The Anthropic Message Batches API
# gives a 50% cost discount on every batched message and decouples
# submission from result pickup, structurally bypassing the Lambda
# 15-min timeout class. ROADMAP P1 §1642.
#
# Plan-manifest design: Submit writes a small JSON manifest to S3
# describing the (capture_key, agent_id, run_id, custom_id, judge_model)
# tuples it submitted, plus first-Saturday flags + eval_prefix +
# cw_namespace. Process loads the manifest and joins each batch result
# to its capture metadata so the eval artifact can be persisted under
# the same (date, agent_id, run_id) path the sync path would have used.
# Decoding from the custom_id alone would lose the original (un-
# sanitized) agent_id, so the manifest is the canonical source of truth
# for agent_id and capture S3 key on the way out.


BATCH_PLAN_PREFIX = "decision_artifacts/_eval_batch_plans/"
"""S3 prefix where Submit writes the per-run plan manifest. Process
reads from here using the batch_id as the filename stem."""


def _build_batch_plan_key(*, date: str, batch_id: str) -> str:
    """``decision_artifacts/_eval_batch_plans/{date}/{batch_id}.json``.
    Date partition mirrors the eval prefix so an operator inspecting
    one Saturday's run can find both the plan manifest and the
    persisted eval artifacts under the same date hierarchy."""
    return f"{BATCH_PLAN_PREFIX}{date}/{batch_id}.json"


def build_batch_plan(
    *,
    date: str,
    bucket: str = _BUCKET_DEFAULT,
    haiku_model: str = DEFAULT_HAIKU_MODEL,
    sonnet_model: str = DEFAULT_SONNET_MODEL,
    force_sonnet_pass: bool = False,
    judge_only: bool = False,
    s3_client: Optional[Any] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extra_dates: Optional[list[str]] = None,
    agent_id_prefixes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """List captures, resolve rubrics, and build the (request_payload,
    plan_entries) pair for the Submit Lambda.

    For ``force_sonnet_pass=True`` (first-Saturday-of-month cadence),
    every mapped artifact gets BOTH a Haiku entry AND a Sonnet entry
    in the same batch — calibration insurance per ROADMAP §1626.
    For weekly cadence (force_sonnet_pass=False), only Haiku entries
    are added; the Process Lambda runs Sonnet escalations
    synchronously after Haiku results arrive (small set, well under
    the Process Lambda's 15-min budget).

    Empty-input captures are persisted client-side immediately with
    a skip-marker artifact (no batch slot consumed). Unmapped agents
    are counted and dropped. The result dict carries the entries to
    submit + counts for both client-side bookkeeping and the
    plan-manifest write that Process will read.
    """
    s3 = s3_client or boto3.client("s3")
    eval_prefix = JUDGE_ONLY_EVAL_PREFIX if judge_only else DEFAULT_EVAL_PREFIX
    cw_namespace = JUDGE_ONLY_CW_NAMESPACE if judge_only else DEFAULT_NAMESPACE

    # ``date`` plus optional ``extra_dates`` — the weekly graph writes only
    # on Saturday, but daily producers (thinktank) land in weekday
    # partitions; a caller judging that family passes the week's dates
    # (with agent_id_prefixes to avoid re-judging already-judged families).
    all_dates = [date] + [d for d in (extra_dates or []) if d != date]
    capture_keys: list[str] = []
    for _d in all_dates:
        capture_keys.extend(
            list_capture_keys(
                s3, date=_d, bucket=bucket, agent_id_prefixes=agent_id_prefixes
            )
        )
    # Already-judged dedup (Brian, 2026-07-02): a multi-date lookback
    # re-enumerates partitions that earlier batches partially judged
    # (e.g. weekend thinktank runs writing into Friday's partition after
    # Saturday's batch ran). Skip anything an ACTUAL eval already scored.
    skipped_already_judged = 0
    if len(all_dates) > 1:
        judged_keys = load_already_judged_keys(s3, dates=all_dates, bucket=bucket)
        if judged_keys:
            before = len(capture_keys)
            capture_keys = [k for k in capture_keys if k not in judged_keys]
            skipped_already_judged = before - len(capture_keys)
            if skipped_already_judged:
                logger.info(
                    "[batch_plan] skipped %d already-judged captures (dedup)",
                    skipped_already_judged,
                )
    requests: list[dict[str, Any]] = []
    plan_entries: list[dict[str, Any]] = []
    client_side_skips: list[dict[str, Any]] = []
    skipped_unmapped = 0

    for key in capture_keys:
        try:
            artifact = _load_capture_artifact(s3, key=key, bucket=bucket)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[batch_plan] load failed for %s — recording as failed",
                key,
            )
            client_side_skips.append({
                "key": key,
                "agent_id": "<unknown>",
                "stage": "load",
                "error": str(exc),
            })
            continue

        rubric = resolve_rubric_for_agent(artifact.agent_id)
        if rubric is None:
            skipped_unmapped += 1
            continue

        if not artifact.agent_output:
            # Skip-marker eval — persisted client-side at plan-build
            # time so we don't burn a batch slot on a no-op call. The
            # eval artifact itself is written here against haiku_model
            # to match the prior sync-path path-shape; first-Saturday
            # also writes one against sonnet_model below.
            client_side_skips.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "empty_input_skip",
            })
            continue

        # Input-sufficiency gate (added 2026-05-13, ROADMAP P0). Same
        # short-circuit shape as empty-input above, different rationale:
        # the agent ran + produced an output, but its inputs were
        # degenerate (per-rubric definition in evals/judge._is_degenerate_input).
        # Scoring the structurally-complete-but-fabricated output would
        # emit a misleading high score into the agent_quality_score CW
        # stream. Route through client-side skip path so no batch slot
        # is spent on a call we'd intentionally throw away.
        from evals.judge import _is_degenerate_input
        if _is_degenerate_input(artifact):
            client_side_skips.append({
                "key": key, "agent_id": artifact.agent_id,
                "stage": "degenerate_input_skip",
            })
            continue

        models_for_artifact = (
            [haiku_model, sonnet_model] if force_sonnet_pass
            else [haiku_model]
        )
        for jm in models_for_artifact:
            cid = encode_custom_id(
                judged_agent_id=artifact.agent_id,
                run_id=artifact.run_id,
                judge_model=jm,
            )
            request_payload = build_batch_request(
                artifact, judge_model=jm, custom_id=cid,
                max_tokens=max_tokens,
            )
            requests.append(request_payload)
            plan_entries.append({
                "custom_id": cid,
                "capture_s3_key": key,
                "agent_id": artifact.agent_id,
                "run_id": artifact.run_id,
                "judge_model": jm,
                "rubric_id": rubric,
            })

    # Mint ONE judge_run_id for this whole batch invocation. Threaded
    # to every RubricEvalArtifact emitted by this batch (skip-markers
    # via _persist_client_side_skips, Haiku-pass + Sonnet-escalation
    # via process_batch_results) so all artifacts cluster under a
    # single _eval/{date}/{judge_run_id}/ directory. Persisted on the
    # plan manifest so Process Lambda inherits the same UUID across
    # the SF state boundary.
    from evals.judge import _new_judge_run_id

    judge_run_id = _new_judge_run_id()

    return {
        "date": date,
        "bucket": bucket,
        "eval_prefix": eval_prefix,
        "cw_namespace": cw_namespace,
        "haiku_model": haiku_model,
        "sonnet_model": sonnet_model,
        "force_sonnet_pass": force_sonnet_pass,
        "judge_only": judge_only,
        "max_tokens": max_tokens,
        "judge_run_id": judge_run_id,
        "capture_keys_total": len(capture_keys),
        "skipped_unmapped": skipped_unmapped,
        "skipped_already_judged": skipped_already_judged,
        "client_side_skips": client_side_skips,
        "plan_entries": plan_entries,
        "requests": requests,
    }


def _persist_client_side_skips(
    plan: dict[str, Any],
    *,
    s3: Any,
    bucket: str,
) -> tuple[int, list[str], list[dict[str, str]]]:
    """Write skip-marker eval artifacts for empty-input captures the
    plan flagged. Returns ``(count, persisted_keys, failed)`` so the
    Submit Lambda can roll the counts into its returned summary
    without a second iteration in the SF state output."""
    persisted: list[str] = []
    failed: list[dict[str, str]] = []
    skipped_empty_input = 0
    eval_prefix = plan["eval_prefix"]
    haiku_model = plan["haiku_model"]
    sonnet_model = plan["sonnet_model"]
    force_sonnet_pass = plan["force_sonnet_pass"]
    judge_run_id = plan["judge_run_id"]

    skipped_degenerate_input = 0
    for skip in plan["client_side_skips"]:
        stage = skip.get("stage")
        if stage not in ("empty_input_skip", "degenerate_input_skip"):
            # ``load`` failures stay in ``failed`` — propagated by the
            # caller into the SF result.
            failed.append(skip)
            continue
        # Map stage → judge_skip_reason. Same skip-eval emit shape;
        # different reason recorded so operators can distinguish
        # "agent never ran" from "agent ran but inputs were degenerate"
        # in the persisted eval payload + SF result counters.
        skip_reason = (
            "precluded_by_empty_upstream"
            if stage == "empty_input_skip"
            else "degenerate_input"
        )
        agent_id = skip["agent_id"]
        capture_key = skip["key"]
        try:
            artifact = _load_capture_artifact(
                s3, key=capture_key, bucket=bucket,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[batch_skip_persist] re-load failed for %s",
                capture_key,
            )
            failed.append({
                "key": capture_key,
                "agent_id": agent_id,
                "stage": "skip_persist_load",
                "error": str(exc),
            })
            continue
        rubric_name = resolve_rubric_for_agent(agent_id)
        if rubric_name is None:
            # Should not happen — plan-build only enqueues skips for
            # mapped agents — but guard for completeness.
            continue
        loaded_prompt = load_prompt(rubric_name)

        models = (
            [haiku_model, sonnet_model] if force_sonnet_pass
            else [haiku_model]
        )
        for jm in models:
            skip_artifact = _make_skip_eval_artifact(
                artifact,
                rubric_name=rubric_name,
                rubric_version=loaded_prompt.version,
                judge_model=jm,
                judge_run_id=judge_run_id,
                judged_artifact_s3_key=capture_key,
                skip_reason=skip_reason,
            )
            try:
                persisted_key = persist_eval_artifact(
                    skip_artifact,
                    s3_client=s3,
                    bucket=bucket,
                    prefix=eval_prefix,
                )
                persisted.append(persisted_key)
                if skip_reason == "degenerate_input":
                    skipped_degenerate_input += 1
                else:
                    skipped_empty_input += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[batch_skip_persist] persist failed for "
                    "agent_id=%s judge=%s",
                    agent_id, jm,
                )
                failed.append({
                    "key": capture_key,
                    "agent_id": agent_id,
                    "stage": "skip_persist_write",
                    "error": str(exc),
                })

    return skipped_empty_input, skipped_degenerate_input, persisted, failed


def submit_batch(
    plan: dict[str, Any],
    *,
    anthropic_client: Any,
    s3_client: Optional[Any] = None,
) -> dict[str, Any]:
    """Submit the plan as one Anthropic message batch and persist the
    plan manifest to S3 keyed by the returned batch_id.

    Idempotency: not retried on submit failure here — the Submit
    Lambda's SF state has its own Retry/Catch posture. Each
    successful submission consumes Anthropic-side request slots, so
    we don't auto-retry to avoid double-billing.

    Empty-plan handling: if ``plan['requests']`` is empty (typical for
    a Saturday with no captures, or one where every artifact was
    empty-input), we skip the API call entirely and return a synthetic
    ``processing_status='empty'`` so the SF can short-circuit the
    Poll loop and run Process directly against the empty manifest.
    """
    s3 = s3_client or boto3.client("s3")
    bucket = plan["bucket"]
    date = plan["date"]
    requests = plan["requests"]

    if not requests:
        # No work to submit. Skip the API call but still write the
        # manifest so Process can pick up client-side skips uniformly.
        logger.info(
            "[batch_submit] empty plan for date=%s — skipping API call",
            date,
        )
        synthetic_batch_id = f"empty-{date}"
        plan_key = _build_batch_plan_key(date=date, batch_id=synthetic_batch_id)
        s3.put_object(
            Bucket=bucket,
            Key=plan_key,
            Body=json.dumps(plan, default=str, indent=2).encode("utf-8"),
        )
        return {
            "batch_id": synthetic_batch_id,
            "processing_status": "ended_empty",
            "plan_s3_key": plan_key,
            "request_count": 0,
        }

    logger.info(
        "[batch_submit] submitting %d requests for date=%s "
        "(force_sonnet_pass=%s)",
        len(requests), date, plan["force_sonnet_pass"],
    )
    batch = anthropic_client.messages.batches.create(requests=requests)
    batch_id = batch.id if hasattr(batch, "id") else batch["id"]

    plan_key = _build_batch_plan_key(date=date, batch_id=batch_id)
    s3.put_object(
        Bucket=bucket,
        Key=plan_key,
        Body=json.dumps(plan, default=str, indent=2).encode("utf-8"),
    )

    return {
        "batch_id": batch_id,
        "processing_status": "in_progress",
        "plan_s3_key": plan_key,
        "request_count": len(requests),
    }


def poll_batch(
    *,
    batch_id: str,
    anthropic_client: Any,
) -> dict[str, Any]:
    """Retrieve the batch's current ``processing_status`` and request
    counts. Used by the Poll Lambda — the SF Choice that follows
    inspects ``processing_status`` to decide between looping back to
    Wait or routing forward to Process.

    Returns ``{"processing_status": str, "request_counts": {...},
    "ended_at": str | None}``. ``ended`` is the terminal status; any
    other value (``in_progress``, ``canceling``) means keep polling.
    """
    if batch_id.startswith("empty-"):
        # Synthetic batch from an empty-plan run — already terminal.
        return {
            "processing_status": "ended",
            "request_counts": {
                "processing": 0, "succeeded": 0, "errored": 0,
                "canceled": 0, "expired": 0,
            },
            "ended_at": None,
        }
    batch = anthropic_client.messages.batches.retrieve(batch_id)
    # ``mode='json'`` coerces datetime → ISO-8601 string so the Lambda
    # response marshaller doesn't blow up. Plain ``model_dump()`` returns
    # Python ``datetime`` objects on ``created_at`` / ``ended_at`` /
    # ``expires_at``, which Lambda's JSON marshaller cannot serialize —
    # surfaced 2026-05-07 against a real Anthropic batch retrieval after
    # the unit tests (MagicMock-stubbed, no Pydantic) missed it.
    if hasattr(batch, "model_dump"):
        batch_dict = batch.model_dump(mode="json")
    elif hasattr(batch, "to_dict"):
        batch_dict = batch.to_dict()
    elif isinstance(batch, dict):
        batch_dict = batch
    else:
        ended_at = getattr(batch, "ended_at", None)
        batch_dict = {
            "processing_status": batch.processing_status,
            "request_counts": batch.request_counts,
            "ended_at": ended_at.isoformat() if hasattr(ended_at, "isoformat") else ended_at,
        }
    return {
        "processing_status": batch_dict.get("processing_status"),
        "request_counts": batch_dict.get("request_counts", {}),
        "ended_at": batch_dict.get("ended_at"),
    }


def process_batch_results(
    *,
    batch_id: str,
    plan_s3_key: str,
    bucket: str = _BUCKET_DEFAULT,
    anthropic_client: Any = None,
    s3_client: Optional[Any] = None,
    cloudwatch_client: Optional[Any] = None,
    emit_metrics: bool = True,
    haiku_escalate_threshold: int = DEFAULT_HAIKU_ESCALATE_THRESHOLD,
) -> dict[str, Any]:
    """Stream the completed batch's results, parse + persist + emit
    each, then run any Sonnet escalations synchronously.

    Drives the Process Lambda after the SF Poll loop has confirmed
    ``processing_status='ended'``. Streams results using the SDK's
    ``messages.batches.results(batch_id)`` iterator so the
    Process Lambda's memory footprint stays bounded by individual
    result size, not full-corpus size.

    Sonnet-escalation tail (weekly cadence only): once Haiku results
    are persisted, any artifact whose Haiku score has a dimension
    below ``haiku_escalate_threshold`` is re-evaluated synchronously
    via ``evaluate_artifact``. Typical cardinality is small (1-3
    artifacts/run) so the synchronous tail fits trivially in the
    Process Lambda's 15-min budget — we don't pay batch latency for
    the escalation path. First-Saturday-of-month cadence
    (``force_sonnet_pass=True`` in the plan) already submitted Sonnet
    requests in the batch; this tail is a no-op in that case.

    Returns a summary dict mirroring the legacy ``evaluate_corpus``
    return shape so dashboards / alarms / SF result inspectors keep
    working unchanged.
    """
    s3 = s3_client or boto3.client("s3")
    cw = cloudwatch_client or (
        boto3.client("cloudwatch") if emit_metrics else None
    )

    plan_raw = s3.get_object(Bucket=bucket, Key=plan_s3_key)["Body"].read()
    plan = json.loads(plan_raw)
    eval_prefix = plan["eval_prefix"]
    cw_namespace = plan["cw_namespace"]
    date = plan["date"]
    force_sonnet_pass = plan["force_sonnet_pass"]
    sonnet_model = plan["sonnet_model"]
    haiku_model = plan["haiku_model"]
    # Inherit the batch's judge_run_id from the persisted plan manifest.
    # All artifacts emitted by THIS Process invocation (Haiku-pass +
    # Sonnet-escalation tail) share this UUID so they cluster under
    # _eval/{date}/{judge_run_id}/ alongside the skip-markers Submit
    # already wrote. Fall back to a fresh UUID for legacy plan
    # manifests written pre-Option-B (replay safety only).
    from evals.judge import _new_judge_run_id

    judge_run_id = plan.get("judge_run_id") or _new_judge_run_id()
    plan_entries_by_cid = {e["custom_id"]: e for e in plan["plan_entries"]}

    haiku_evaluated = 0
    sonnet_evaluated = 0
    failed: list[dict[str, str]] = list(plan.get("client_side_skips", []))
    # Strip the empty_input_skip entries from `failed` — those are
    # successes (skip-marker eval persisted in Submit) not failures.
    # Preserve any `load` failures from the plan stage as failures.
    failed = [f for f in failed if f.get("stage") != "empty_input_skip"]
    metric_emission_failures = 0
    persisted_keys: list[str] = []
    haiku_evals_by_agent_run: dict[tuple[str, str], RubricEvalArtifact] = {}

    def _try_emit(eval_artifact: RubricEvalArtifact) -> None:
        nonlocal metric_emission_failures
        if not emit_metrics:
            return
        try:
            emit_eval_metric(
                eval_artifact, namespace=cw_namespace, cloudwatch_client=cw,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[batch_process] cloudwatch emit failed for "
                "agent_id=%s judge=%s",
                eval_artifact.judged_agent_id, eval_artifact.judge_model,
            )
            metric_emission_failures += 1

    # Stream batch results unless the synthetic empty-batch sentinel.
    if not batch_id.startswith("empty-") and plan["requests"]:
        for result in anthropic_client.messages.batches.results(batch_id):
            # SDK returns objects; coerce to a stable dict-or-attr access
            # via the helper closure to keep the parsing site agnostic
            # to SDK minor version drift.
            cid = (
                result["custom_id"] if isinstance(result, dict)
                else result.custom_id
            )
            entry = plan_entries_by_cid.get(cid)
            if entry is None:
                # Unknown custom_id — defensive; should not happen if
                # encode/decode is canonical. Record as failure with
                # decoded best-effort metadata.
                try:
                    decoded_agent, decoded_run, decoded_model = decode_custom_id(cid)
                except ValueError:
                    decoded_agent, decoded_run, decoded_model = (
                        "<unknown>", "<unknown>", "<unknown>",
                    )
                failed.append({
                    "key": "<unknown>",
                    "agent_id": decoded_agent,
                    "stage": "process_unknown_custom_id",
                    "error": f"custom_id={cid!r} not in plan_entries",
                })
                continue

            result_payload = (
                result["result"] if isinstance(result, dict) else result.result
            )
            result_type = (
                result_payload["type"] if isinstance(result_payload, dict)
                else result_payload.type
            )

            if result_type != "succeeded":
                # ``errored`` / ``expired`` / ``canceled`` — Anthropic
                # docs guarantee no charge for these. Record the failure
                # so the Saturday SF can alarm on PARTIAL.
                err_obj = (
                    result_payload.get("error")
                    if isinstance(result_payload, dict)
                    else getattr(result_payload, "error", None)
                )
                failed.append({
                    "key": entry["capture_s3_key"],
                    "agent_id": entry["agent_id"],
                    "stage": f"batch_{result_type}",
                    "error": str(err_obj),
                })
                continue

            try:
                message_payload = (
                    result_payload["message"]
                    if isinstance(result_payload, dict)
                    else result_payload.message
                )
                llm_output = parse_batch_message(message_payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[batch_process] parse failed for custom_id=%s "
                    "agent_id=%s judge=%s",
                    cid, entry["agent_id"], entry["judge_model"],
                )
                failed.append({
                    "key": entry["capture_s3_key"],
                    "agent_id": entry["agent_id"],
                    "stage": "batch_parse",
                    "error": str(exc),
                })
                continue

            # Resolved model Anthropic actually ran (batch message 'model'
            # field) — the re-anchor trigger for L4578(a). Defensive .get
            # so a shape change leaves it None rather than crashing.
            resolved_model = (
                message_payload.get("model")
                if isinstance(message_payload, dict)
                else getattr(message_payload, "model", None)
            )

            # Look up rubric_version cheaply — load_prompt is cached.
            loaded_prompt = load_prompt(entry["rubric_id"])
            eval_artifact = RubricEvalArtifact(
                run_id=entry["run_id"],
                judge_run_id=judge_run_id,
                timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                judged_agent_id=entry["agent_id"],
                judged_artifact_s3_key=entry["capture_s3_key"],
                rubric_id=entry["rubric_id"],
                rubric_version=loaded_prompt.version,
                judge_model=entry["judge_model"],
                judge_request_model=request_model_for(entry["judge_model"]),
                judge_resolved_model=resolved_model,
                dimension_scores=llm_output.dimension_scores,
                overall_reasoning=llm_output.overall_reasoning,
            )
            try:
                pkey = persist_eval_artifact(
                    eval_artifact, s3_client=s3, bucket=bucket,
                    prefix=eval_prefix,
                )
                persisted_keys.append(pkey)
                if entry["judge_model"] == haiku_model:
                    haiku_evaluated += 1
                    haiku_evals_by_agent_run[
                        (entry["agent_id"], entry["run_id"])
                    ] = eval_artifact
                elif entry["judge_model"] == sonnet_model:
                    sonnet_evaluated += 1
                _try_emit(eval_artifact)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[batch_process] persist failed for "
                    "agent_id=%s judge=%s",
                    entry["agent_id"], entry["judge_model"],
                )
                failed.append({
                    "key": entry["capture_s3_key"],
                    "agent_id": entry["agent_id"],
                    "stage": "batch_persist",
                    "error": str(exc),
                })

    # Sonnet-escalation tail (weekly cadence only). First-Saturday
    # already submitted Sonnet via the batch so we skip the tail in
    # that path — every artifact already has a Sonnet eval.
    if not force_sonnet_pass:
        for entry in plan["plan_entries"]:
            if entry["judge_model"] != haiku_model:
                continue
            haiku_eval = haiku_evals_by_agent_run.get(
                (entry["agent_id"], entry["run_id"])
            )
            if haiku_eval is None:
                continue
            if not should_escalate_to_sonnet(
                haiku_eval, threshold=haiku_escalate_threshold,
            ):
                continue

            try:
                artifact = _load_capture_artifact(
                    s3, key=entry["capture_s3_key"], bucket=bucket,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[batch_process_escalation] load failed for %s",
                    entry["capture_s3_key"],
                )
                failed.append({
                    "key": entry["capture_s3_key"],
                    "agent_id": entry["agent_id"],
                    "stage": "escalation_load",
                    "error": str(exc),
                })
                continue

            try:
                sonnet_eval = evaluate_artifact(
                    artifact, judge_run_id=judge_run_id,
                    judge_model=sonnet_model,
                    judged_artifact_s3_key=entry["capture_s3_key"],
                )
                pkey = persist_eval_artifact(
                    sonnet_eval, s3_client=s3, bucket=bucket,
                    prefix=eval_prefix,
                )
                persisted_keys.append(pkey)
                sonnet_evaluated += 1
                _try_emit(sonnet_eval)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[batch_process_escalation] sonnet eval failed for %s",
                    entry["capture_s3_key"],
                )
                failed.append({
                    "key": entry["capture_s3_key"],
                    "agent_id": entry["agent_id"],
                    "stage": "escalation_sonnet",
                    "error": str(exc),
                })

    # Empty-input skips already had their skip-marker artifacts
    # written in Submit; count them here for the SF result so the
    # rolling-mean alarm logic sees the same number it always has.
    skipped_empty_input = sum(
        1 for s in plan.get("client_side_skips", [])
        if s.get("stage") == "empty_input_skip"
    )
    # First-Saturday writes one skip per (haiku, sonnet) — count both.
    if force_sonnet_pass:
        skipped_empty_input *= 2

    logger.info(
        "[batch_process] done batch_id=%s date=%s haiku=%d sonnet=%d "
        "skipped_unmapped=%d skipped_empty_input=%d failed=%d "
        "metric_emission_failures=%d",
        batch_id, date, haiku_evaluated, sonnet_evaluated,
        plan.get("skipped_unmapped", 0), skipped_empty_input, len(failed),
        metric_emission_failures,
    )

    return {
        "date": date,
        "batch_id": batch_id,
        "plan_s3_key": plan_s3_key,
        "capture_keys_total": plan.get("capture_keys_total", 0),
        "haiku_evaluated": haiku_evaluated,
        "sonnet_evaluated": sonnet_evaluated,
        "skipped_unmapped": plan.get("skipped_unmapped", 0),
        "skipped_empty_input": skipped_empty_input,
        "metric_emission_failures": metric_emission_failures,
        "failed": failed,
        "persisted_keys": persisted_keys,
        "haiku_model": haiku_model,
        "sonnet_model": sonnet_model,
        "force_sonnet_pass": force_sonnet_pass,
        "judge_only": plan.get("judge_only", False),
        "eval_prefix": eval_prefix,
        "cw_namespace": cw_namespace,
    }
