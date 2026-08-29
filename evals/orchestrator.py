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
import time
from datetime import UTC, datetime
from typing import Any

import boto3
from nousergon_lib.decision_capture import DecisionArtifact

from agents.prompt_loader import load_prompt
from evals.judge import (
    DEFAULT_EVAL_PREFIX,
    DEFAULT_MAX_TOKENS,
    _make_skip_eval_artifact,
    build_batch_request,
    decode_custom_id,
    encode_custom_id,
    evaluate_artifact,
    parse_batch_message,
    persist_eval_artifact,
    resolve_rubric_for_agent,
)
from evals.judge_batch_transport import (
    BatchCapabilityUnavailable,
    batch_client_for_route,
    build_degradation_record,
    is_sync_batch_id,
    persist_degradation_record,
    resolve_batch_transport,
    sync_batch_id,
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


# ── Deadline budget (alpha-engine-config-I6920 class) ─────────────────────
#
# EvalJudgeProcess was killed at its 900s wall in 9 of 28 observed real
# invocations (32%), all nine inside the parse-retry tail below. Measured
# 2026-07-26: each synchronous re-judge took 45-105s of wall clock against
# a cap sized on the comment "40 × ≲8s ≪ the Process Lambda's 15-min
# ceiling". 40 × 75s is 3000s — 3.3× the wall — so the cap could never
# bind, and nothing in this module consulted the clock.
#
# Killed at the wall means the manifest build never runs, the summary is
# never returned, and the SF sees States.Timeout with no cause. The evals
# persisted before the kill are on S3 but nothing says how many of the
# corpus they represent.
#
# Same shape as crucible-backtester's replay/batch.py (config#6920) rather
# than a new invention — the two are the same primitive and the end state
# is one copy in nousergon-lib. Lifting it needs a lib tag plus lockstep
# pin bumps across three repos, which does not belong in a fix for a live
# pipeline failure; tracked on I6920 as the third adoption.

PROCESS_WRITE_RESERVE_S = 60.0
"""Time held back from every loop for the write tail — the
``_eval_by_capture`` manifest build (an S3 LIST + PUT per capture date)
and the summary return. Without a reserve a loop consumes the whole
budget and the invocation ends with nothing said about what it covered,
which is the failure this whole block exists to remove. Larger than the
concordance's 45s because the manifest build lists S3 prefixes rather
than emitting a handful of CloudWatch datapoints."""

PROCESS_LLM_ITEM_FLOOR_S = 30.0
"""Floor for the "does another synchronous judge call fit?" estimate
before this phase has completed an item. The judge client's own per-call
ceiling is 180s (``evals/judge.py::_call_openrouter_judge_llm``) with 3
attempts, so a single item's true worst case is 540s; 30s is the floor on
the ESTIMATE, not a claim about the worst case — after the first item the
observed p90 governs."""

PROCESS_STREAM_ITEM_FLOOR_S = 3.0
"""Floor for the batch-result stream, whose per-item work is a parse plus
an S3 PUT plus a CloudWatch emit — hundreds of milliseconds, not tens of
seconds. Kept separate from the LLM floor on purpose: a phase must
estimate from its OWN items. Sharing one latency sample would let the
stream's sub-second items license a 100s judge call."""


def _next_item_affordable(
    remaining_s, latencies_ms: list[int], *, floor_s: float,
) -> tuple[bool, float]:
    """Can another item of this phase finish before the deadline?

    Estimates from the observed p90 of THIS phase's own item latencies —
    the workload measures itself rather than trusting a literal — plus the
    write reserve. Returns ``(affordable, needed_seconds)``.

    ``remaining_s`` is a zero-arg callable returning seconds left, or
    ``None`` for callers with no deadline (the CLI, spot runs, tests),
    which are always affordable and behave exactly as before.

    Mirrors ``replay.batch._next_item_affordable`` in crucible-backtester.
    """
    if remaining_s is None:
        return True, 0.0
    if latencies_ms:
        ordered = sorted(latencies_ms)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))] / 1000.0
        estimate = max(floor_s, p90)
    else:
        estimate = floor_s
    needed = estimate + PROCESS_WRITE_RESERVE_S
    return remaining_s() >= needed, needed


# ── Sampling decision ─────────────────────────────────────────────────────


def should_escalate_to_sonnet(
    haiku_eval: RubricEvalArtifact,
    *,
    threshold: int = DEFAULT_HAIKU_ESCALATE_THRESHOLD,
) -> bool:
    """Per-artifact escalation: True iff any Haiku dimension score is
    below ``threshold``.

    Callers MUST only pass a ``haiku_eval`` whose ``judge_model`` is an
    AUTHORITATIVE tier — never a shadow-only judge_model (see
    ``evals.judge_models.SHADOW_LOGICAL_KEYS``, config#2575). A shadow
    judge's scores get no escalation authority until its perturbation-
    suite validation passes and an explicit promotion event lifts the
    exclusion; this function does not itself gate on that (it has no
    judge_model to check — ``RubricEvalArtifact.judge_model`` is a field
    on the artifact, not a parameter here), so the discipline lives in
    every call site never feeding it a shadow artifact in the first
    place. ``evals.openrouter_shadow`` deliberately never calls this
    function.
    """
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
        except Exception as e:  # noqa: BLE001 — absent manifest = nothing judged yet
            logger.debug("[batch_plan] no eval manifest at %s: %s", key, e)
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
    s3_client: Any | None = None,
    cloudwatch_client: Any | None = None,
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
    s3_client: Any | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extra_dates: list[str] | None = None,
    agent_id_prefixes: list[str] | None = None,
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
        # Anomaly guard: dedup returned zero skips despite a multi-date
        # lookback with actual captures. On a retry (the 2nd+ batch plan
        # for the same corpus), this means the _eval_by_capture manifests
        # are missing or stale — the exact failure mode from config#4776
        # (index not written since 2026-07-17 → 295-artifact corpus
        # re-judged ~10 times). On a first run, zero skips is normal.
        # The warning gives operators diagnostic breadcrumbs in CW Logs
        # without falsely alarming (the caller decides what's a retry).
        if skipped_already_judged == 0 and capture_keys:
            logger.warning(
                "[batch_plan] already-judged dedup returned zero skips for "
                "%d capture keys across %d dates — the _eval_by_capture "
                "manifests may be stale or missing (expected on the first "
                "run; anomalous on a retry of a previously-judged corpus)",
                len(capture_keys), len(all_dates),
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


def _submit_via_sync_rung(
    plan: dict[str, Any],
    *,
    s3: Any,
    unavailable: BatchCapabilityUnavailable,
) -> dict[str, Any]:
    """Second rung of the judge degradation ladder (alpha-engine-config-I9263).

    No batch route is available, so the plan is not submitted anywhere: it is
    persisted under a ``sync-{date}`` batch id and ``process_batch_results``
    judges every entry through the router-addressed synchronous judge.

    Three things happen here and all three are load-bearing:

    1. **The plan manifest is still written**, under the synthetic id, so
       Process reads its input from exactly the same place on either rung and
       the client-side skip markers Submit already persisted still apply.
    2. **A durable degradation record is written to S3.** Unguarded on purpose
       — see ``persist_degradation_record``. If the evidence cannot be written,
       the stage fails rather than degrading unobserved.
    3. **``degraded`` and the record ride out on the return value**, so the
       Step Function stage output carries the fact without a reader having to
       go looking in S3 or CloudWatch Logs for it.
    """
    bucket = plan["bucket"]
    date = plan["date"]
    batch_id = sync_batch_id(date)
    plan_key = _build_batch_plan_key(date=date, batch_id=batch_id)
    s3.put_object(
        Bucket=bucket,
        Key=plan_key,
        Body=json.dumps(plan, default=str, indent=2).encode("utf-8"),
    )
    record = build_degradation_record(
        date=date,
        reason=unavailable.reason,
        group=unavailable.group,
        capability=unavailable.capability,
        exec_context=unavailable.exec_context,
        request_count=len(plan["requests"]),
        plan_s3_key=plan_key,
    )
    degradation_key = persist_degradation_record(
        record, bucket=bucket, s3_client=s3,
    )
    return {
        "batch_id": batch_id,
        "processing_status": "ended_sync",
        "plan_s3_key": plan_key,
        "request_count": len(plan["requests"]),
        "degraded": True,
        "degradation_s3_key": degradation_key,
        "degradation": record,
    }


def submit_batch(
    plan: dict[str, Any],
    *,
    batch_client: Any = None,
    s3_client: Any | None = None,
    exec_context: str | None = None,
) -> dict[str, Any]:
    """Submit the plan through the batch rung, or degrade to the sync rung.

    **alpha-engine-config-I9263 (Brian ruling 2026-08-29: "I will not fund
    the anthropic account, at this point we shouldn\'t be using the anthropic
    api at all").** This function no longer receives a provider SDK client
    built at the Lambda call site. It asks the router whether the judge\'s
    model group can serve the ``batches`` CAPABILITY from this execution
    context (``evals.judge_batch_transport.resolve_batch_transport``) and
    takes one of two declared rungs:

    * **Batch rung** — a router-resolved batch-capable route. ``batch_client``
      is an explicit override (test seam, and the wiring point for a future
      krepis batch client); when it is ``None`` the router is asked.
    * **Sync rung** — when no batch route is available, the plan manifest is
      written under a ``sync-{date}`` batch id, a DURABLE degradation record
      is persisted beside it, and ``degraded: True`` plus the record are
      returned on the result so the Step Function stage output carries them.
      ``process_batch_results`` then judges every plan entry through
      ``evals.judge.evaluate_artifact``, which is already fully
      router-addressed (alpha-engine-config-I6559).

    The sync rung costs ~2x per judged artifact — the batch rung bills at half
    rate (``_BATCH_PRICE_MULTIPLIER``) and the sync rung at the standard rate.
    That is why the degradation is recorded rather than absorbed: a transport
    swap that doubles the bill and changes the series it produced must be
    legible from a durable artifact, never inferred from a log line.

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

    if batch_client is None:
        try:
            spec, route = resolve_batch_transport(exec_context=exec_context)
        except BatchCapabilityUnavailable as exc:
            return _submit_via_sync_rung(plan, s3=s3, unavailable=exc)
        # A batch route resolved but nothing can drive it — a fleet defect,
        # NOT a degradation. Falling through to the sync rung here would spend
        # 2x while hiding a broken batch route behind a "degraded" label.
        batch_client = batch_client_for_route(spec, route)

    logger.info(
        "[batch_submit] submitting %d requests for date=%s "
        "(force_sonnet_pass=%s)",
        len(requests), date, plan["force_sonnet_pass"],
    )
    batch = batch_client.messages.batches.create(requests=requests)
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
    batch_client: Any = None,
) -> dict[str, Any]:
    """Retrieve the batch's current ``processing_status`` and request
    counts. Used by the Poll Lambda — the SF Choice that follows
    inspects ``processing_status`` to decide between looping back to
    Wait or routing forward to Process.

    Returns ``{"processing_status": str, "request_counts": {...},
    "ended_at": str | None}``. ``ended`` is the terminal status; any
    other value (``in_progress``, ``canceling``) means keep polling.
    """
    if batch_id.startswith("empty-") or is_sync_batch_id(batch_id):
        # Synthetic batch id — an empty-plan run, or a run that took the
        # SYNCHRONOUS rung (alpha-engine-config-I9263). Neither has a
        # provider-side batch to poll, so both are already terminal and the SF
        # Choice routes straight to Process. Request counts are zero here by
        # construction: on the sync rung the work has not happened yet, and
        # reporting a non-zero count for work Process still has to do would
        # make a not-yet-run pass look finished.
        return {
            "processing_status": "ended",
            "request_counts": {
                "processing": 0, "succeeded": 0, "errored": 0,
                "canceled": 0, "expired": 0,
            },
            "ended_at": None,
        }
    batch = batch_client.messages.batches.retrieve(batch_id)
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


#: Anthropic's Message Batches API bills every batched message at half the
#: standard rate. `krepis.cost.record_llm_call` prices from the standard rate
#: card, which is correct for every OTHER call site in the fleet, so the
#: halving is applied here rather than in the shared library — a batch
#: discount is a property of this API, not of pricing.
_BATCH_PRICE_MULTIPLIER = 0.5

#: The call site this spend belongs to, matching the `evaljudge-batch` row in
#: `alpha-engine-config/private-docs/LLM_CALLSITE_REGISTRY.yaml`. Inventing a
#: new string here would attribute the spend to nothing, which is the defect
#: this emitter exists to close.
_BATCH_CALLSITE_ID = "evaljudge-batch"


def _resolve_cost_sink():
    """The environment-declared cost sink, or ``None``.

    A one-line seam over ``krepis.cost_sink.default_sink_from_env`` so this
    module's tests do not have to reach into the library to patch it, and
    so the sink stays THE SAME object every routed call site in this
    process already uses rather than a second one constructed here.
    """
    from krepis.cost_sink import default_sink_from_env

    return default_sink_from_env()


def _emit_batch_cost_record(message_payload: Any, *, entry: dict) -> None:
    """Record one batched judge message's tokens and cost.

    **Why this call site emits its own record when no other one does.**
    `model-router-policy` §4 carves the Anthropic Batches API out of the
    router: it has no router path, so there is no shared client to emit
    at, and the rule that cost records are written once at the router
    client cannot reach it. Everything else in the weekly pipeline goes
    through `krepis.llm.LLMClient`, which emits from the environment as of
    krepis 0.57.0. This is the one exception, and it is the exception
    *because* the carve-out exists — not because a stage-by-stage retrofit
    was the design (alpha-engine-config-I7179).

    Emission is best-effort by the same trade
    `LLMClient._emit_cost_record` makes, and for the same reason:

    (a) Failure mode swallowed: cost-record construction or the sink write
        failed — an unpriced model card, an S3 error, an SDK shape change.
    (b) The primary deliverable survives because this runs after the batch
        message has already been decoded and is about to be persisted.
        Raising here would convert a telemetry fault into a lost eval.
    (c) Recording surface, three layers: this ERROR log; the coverage
        check in `scripts/aggregate_costs.py`, which fails when
        `evaljudge-batch` is absent from `_cost_raw/{date}/` on a day
        EvalJudgeProcess ran; and the ARTIFACT_REGISTRY freshness row on
        the prefix itself. The middle layer is the one this issue added,
        and it is the one that catches SUSTAINED loss when logs go unread.
    """
    try:
        from krepis.cost import record_llm_call

        # The SAME environment-resolved sink every routed call site uses, not
        # a second one constructed here. Two consequences, both wanted: in
        # the Lambda it writes to exactly the prefix `deploy.sh` declared,
        # and under test — where no destination is declared — it is None and
        # nothing is buffered, so a unit run cannot write fake rows into the
        # production partition. That has happened: 2026-05-13, ~$1014 of
        # fake-agent rows, dashboard trend inflated 700x.
        sink = _resolve_cost_sink()
        if sink is None:
            return

        record = record_llm_call(
            message_payload,
            provider="anthropic",
            extra_fields={
                "callsite_id": _BATCH_CALLSITE_ID,
                "run_id": entry.get("run_id"),
                "agent_id": entry.get("agent_id"),
                "judge_model": entry.get("judge_model"),
                "billing_mode": "anthropic_batch",
            },
        )
        # Tokens stay verbatim; only the derived dollars are discounted, so a
        # later reprice against a corrected card still starts from the real
        # token counts (observability-policy §3.2: tokens are the persisted
        # quantity, dollars are derived).
        if isinstance(record.get("cost_usd"), (int, float)):
            record["cost_usd"] = record["cost_usd"] * _BATCH_PRICE_MULTIPLIER
            record["cost_source"] = f"{record.get('cost_source')}_batch50"
        sink(record)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[batch_process] cost emission failed for callsite_id=%s "
            "agent_id=%s: %s",
            _BATCH_CALLSITE_ID,
            entry.get("agent_id"),
            exc,
        )


def process_batch_results(
    *,
    batch_id: str,
    plan_s3_key: str,
    bucket: str = _BUCKET_DEFAULT,
    batch_client: Any = None,
    s3_client: Any | None = None,
    cloudwatch_client: Any | None = None,
    emit_metrics: bool = True,
    haiku_escalate_threshold: int = DEFAULT_HAIKU_ESCALATE_THRESHOLD,
    remaining_s=None,
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
    via ``evaluate_artifact``. First-Saturday-of-month cadence
    (``force_sonnet_pass=True`` in the plan) already submitted Sonnet
    requests in the batch; this tail is a no-op in that case.

    Its cardinality is DATA-dependent, not bounded by anything: every
    Haiku eval that flags a borderline dimension adds one synchronous
    judge call. A week where the judged agents degrade is exactly a week
    the tail grows, so the runtime rises with the thing it exists to
    measure. The docstring here used to claim "typical cardinality is
    small (1-3 artifacts/run) so the synchronous tail fits trivially" —
    an assumption, never a bound.

    **Deadline discipline (alpha-engine-config-I6920 class).** All three
    loops — the result stream, the parse-retry tail and the escalation
    tail — ask before each item whether it still fits in ``remaining_s``
    (see ``_next_item_affordable``). When it does not, that loop stops,
    the work already done is kept, the manifest build and summary still
    run inside ``PROCESS_WRITE_RESERVE_S``, and the summary carries
    ``complete=False`` plus ``budget_stopped``, ``budget_stopped_phases``
    and ``n_skipped_for_budget`` so a truncated pass can never be read as
    a full one (``sf-pipeline-policy.md`` §2.3a: a missing verdict
    propagates as UNKNOWN, never as pass).

    Args:
        remaining_s: zero-arg callable returning the seconds left before
            the caller's deadline, or ``None`` for no deadline. The
            Process Lambda passes ``context.get_remaining_time_in_millis``
            divided by 1000; the CLI, spot runs and tests pass nothing and
            behave exactly as they did before.

    Returns a summary dict mirroring the legacy ``evaluate_corpus``
    return shape so dashboards / alarms / SF result inspectors keep
    working unchanged, plus the four budget fields above.
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
    # Batch results whose tool output failed schema parse (malformed
    # stringified dimension_scores etc. — the 2026-07-03 weekly lost 12
    # evals this way, config#1650). Each gets ONE synchronous
    # evaluate_artifact retry after the stream (fresh decoder sample via
    # the sync path's own MAX_JUDGE_RETRIES) before being terminal-failed.
    parse_retry_queue: list[tuple[dict, str]] = []
    # Sync rung (alpha-engine-config-I9263): no provider batch was submitted,
    # so EVERY plan entry is judged here through ``evaluate_artifact``. Seeded
    # before the (skipped) stream so the deadline bookkeeping below sees the
    # real workload rather than an empty one.
    sync_fallback_queue: list[dict] = (
        list(plan["plan_entries"]) if is_sync_batch_id(batch_id) else []
    )
    sync_fallback_evaluated = 0
    # Strip the empty_input_skip entries from `failed` — those are
    # successes (skip-marker eval persisted in Submit) not failures.
    # Preserve any `load` failures from the plan stage as failures.
    failed = [f for f in failed if f.get("stage") != "empty_input_skip"]
    metric_emission_failures = 0
    persisted_keys: list[str] = []
    # Coverage ledger (alpha-engine-config-I9309). Every plan entry whose
    # judge eval was PERSISTED, by custom_id. A set, not a counter, because
    # the three rungs that can grade an entry — the batch stream, the sync
    # rung, and the parse-retry tail — are not mutually exclusive: a stream
    # item that failed to parse and was then recovered by the retry tail must
    # count once, and a counter incremented at each site would report 101%
    # coverage on exactly the week the retry tail earned its keep.
    #
    # The Sonnet-escalation tail deliberately does NOT record here: it emits a
    # SECOND eval for an entry the Haiku pass already graded, so counting it
    # would let escalations paper over an ungraded entry elsewhere. Coverage
    # is measured over plan entries, and each entry is graded exactly once.
    graded_custom_ids: set[str] = set()
    haiku_evals_by_agent_run: dict[tuple[str, str], RubricEvalArtifact] = {}

    # Deadline bookkeeping. Each phase keeps its OWN latency sample —
    # see PROCESS_STREAM_ITEM_FLOOR_S for why they may not be shared.
    budget_stopped_phases: list[str] = []
    n_skipped_for_budget: dict[str, int] = {}

    def _stop_for_budget(phase: str, skipped: int, needed: float) -> None:
        budget_stopped_phases.append(phase)
        n_skipped_for_budget[phase] = skipped
        logger.warning(
            "[batch_process] stopping %s early on budget: %d items not "
            "processed (needed ~%.0fs, %.0fs left). The manifest and "
            "summary still run; the summary is marked incomplete.",
            phase, skipped, needed,
            remaining_s() if remaining_s is not None else -1.0,
        )

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
    stream_latencies_ms: list[int] = []
    stream_processed = 0
    _item_t0: float | None = None
    if (
        not batch_id.startswith("empty-")
        and not is_sync_batch_id(batch_id)
        and plan["requests"]
    ):
        for result in batch_client.messages.batches.results(batch_id):
            # Close out the PREVIOUS item's latency here rather than at the
            # bottom of the loop: the body has several `continue` paths, and
            # a sample taken only on the fall-through path would measure the
            # cheap outcomes and miss the expensive ones.
            if _item_t0 is not None:
                stream_latencies_ms.append(int((time.time() - _item_t0) * 1000))
            affordable, needed = _next_item_affordable(
                remaining_s, stream_latencies_ms,
                floor_s=PROCESS_STREAM_ITEM_FLOOR_S,
            )
            if not affordable:
                # The stream is an iterator, so the residue is derived from
                # the plan's own request count rather than by draining it —
                # draining would spend the reserve we just protected.
                _stop_for_budget(
                    "batch_stream",
                    max(0, len(plan["requests"]) - stream_processed),
                    needed,
                )
                break
            _item_t0 = time.time()
            stream_processed += 1
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
                    decoded_agent, _decoded_run, _decoded_model = decode_custom_id(cid)
                except ValueError:
                    decoded_agent, _decoded_run, _decoded_model = (
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
                # Emitted BEFORE the parse, deliberately, and inside this
                # try because it never raises (see its own docstring).
                # Anthropic billed this message the moment it succeeded;
                # whether we can decode the answer is our problem, not the
                # invoice's. Emitting after the parse would drop the cost of
                # exactly the messages that go to the sync retry queue — the
                # expensive ones, since those are the ones paid for twice.
                _emit_batch_cost_record(message_payload, entry=entry)
                llm_output = parse_batch_message(message_payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[batch_process] parse failed for custom_id=%s "
                    "agent_id=%s judge=%s — queued for sync retry "
                    "(config#1650)",
                    cid, entry["agent_id"], entry["judge_model"],
                )
                parse_retry_queue.append((entry, str(exc)))
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
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
                graded_custom_ids.add(entry["custom_id"])
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

    # ── Sync rung (alpha-engine-config-I9263) ────────────────────────────
    # Every plan entry judged through ``evaluate_artifact`` — the SAME judge,
    # rubric and output shape the batch rung produced, over the router-resolved
    # `low` group (alpha-engine-config-I6559) rather than a provider batch API.
    #
    # No item cap. The parse-retry tail below carries `_PARSE_RETRY_CAP` as a
    # COST guard on an anomaly (a handful of malformed responses); here the
    # whole corpus is the expected workload, so a cap would silently thin the
    # eval set on a normal run. The deadline check bounds it instead, and a
    # truncated pass reports `complete=False` exactly as every other phase does.
    sync_latencies_ms: list[int] = []
    for _sync_i, entry in enumerate(sync_fallback_queue):
        affordable, needed = _next_item_affordable(
            remaining_s, sync_latencies_ms, floor_s=PROCESS_LLM_ITEM_FLOOR_S,
        )
        if not affordable:
            _stop_for_budget(
                "sync_fallback", len(sync_fallback_queue) - _sync_i, needed,
            )
            break
        _sync_t0 = time.time()
        try:
            artifact = _load_capture_artifact(
                s3, key=entry["capture_s3_key"], bucket=bucket,
            )
            sync_eval = evaluate_artifact(
                artifact, judge_run_id=judge_run_id,
                judge_model=entry["judge_model"],
                judged_artifact_s3_key=entry["capture_s3_key"],
            )
            pkey = persist_eval_artifact(
                sync_eval, s3_client=s3, bucket=bucket, prefix=eval_prefix,
            )
            persisted_keys.append(pkey)
            graded_custom_ids.add(entry["custom_id"])
            sync_fallback_evaluated += 1
            if entry["judge_model"] == haiku_model:
                haiku_evaluated += 1
                haiku_evals_by_agent_run[
                    (entry["agent_id"], entry["run_id"])
                ] = sync_eval
            elif entry["judge_model"] == sonnet_model:
                sonnet_evaluated += 1
            _try_emit(sync_eval)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[batch_process] sync-rung judge FAILED for agent_id=%s "
                "judge=%s", entry["agent_id"], entry["judge_model"],
            )
            failed.append({
                "key": entry["capture_s3_key"],
                "agent_id": entry["agent_id"],
                "stage": "sync_fallback_judge",
                "error": str(exc),
            })
        sync_latencies_ms.append(int((time.time() - _sync_t0) * 1000))

    # Sync retry tail for batch parse failures (config#1650). The batch
    # path had NO analog of the sync path's MAX_JUDGE_RETRIES: a judge
    # response with malformed stringified dimension_scores (Haiku's known
    # stochastic non-conformance — NOT token truncation; 7/3 failures were
    # stop_reason=tool_use at ~1k of the 10752 cap) was terminal, silently
    # thinning the eval corpus (12/85 lost on 2026-07-03, run=PARTIAL).
    # One synchronous evaluate_artifact per failed item re-rolls the
    # decoder via the sync path's own retry loop. Runs BEFORE the
    # escalation tail so a recovered borderline Haiku eval still
    # escalates to Sonnet. Overflow past the cap is terminal-failed loud.
    #
    # The cap is a COST guard, not a time budget. It was written as one —
    # "40 × ≲8s ≪ the Process Lambda's 15-min ceiling" — and that assumed
    # constant is what killed nine invocations. Measured 2026-07-26 on the
    # live function: 45-105s per retry, i.e. 40 × 75s = 3000s against a
    # 900s wall, so the cap could never bind on time. Time is now bounded
    # by the deadline check above, measured from this run's own retries.
    _PARSE_RETRY_CAP = 40
    parse_retry_recovered = 0
    retry_latencies_ms: list[int] = []
    for i, (entry, orig_err) in enumerate(parse_retry_queue):
        affordable, needed = _next_item_affordable(
            remaining_s, retry_latencies_ms, floor_s=PROCESS_LLM_ITEM_FLOOR_S,
        )
        if not affordable:
            _stop_for_budget(
                "parse_retry", len(parse_retry_queue) - i, needed,
            )
            break
        _retry_t0 = time.time()
        if i >= _PARSE_RETRY_CAP:
            failed.append({
                "key": entry["capture_s3_key"],
                "agent_id": entry["agent_id"],
                "stage": "batch_parse_retry_capped",
                "error": (
                    f"parse-retry cap {_PARSE_RETRY_CAP} exceeded "
                    f"({len(parse_retry_queue)} queued); original: {orig_err}"
                ),
            })
            continue
        try:
            artifact = _load_capture_artifact(
                s3, key=entry["capture_s3_key"], bucket=bucket,
            )
            retry_eval = evaluate_artifact(
                artifact, judge_run_id=judge_run_id,
                judge_model=entry["judge_model"],
                judged_artifact_s3_key=entry["capture_s3_key"],
            )
            pkey = persist_eval_artifact(
                retry_eval, s3_client=s3, bucket=bucket, prefix=eval_prefix,
            )
            persisted_keys.append(pkey)
            graded_custom_ids.add(entry["custom_id"])
            parse_retry_recovered += 1
            if entry["judge_model"] == haiku_model:
                haiku_evaluated += 1
                haiku_evals_by_agent_run[
                    (entry["agent_id"], entry["run_id"])
                ] = retry_eval
            elif entry["judge_model"] == sonnet_model:
                sonnet_evaluated += 1
            _try_emit(retry_eval)
            logger.info(
                "[batch_process] parse-retry recovered agent_id=%s judge=%s",
                entry["agent_id"], entry["judge_model"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[batch_process] parse-retry FAILED for agent_id=%s "
                "judge=%s (original parse error: %s)",
                entry["agent_id"], entry["judge_model"], orig_err,
            )
            failed.append({
                "key": entry["capture_s3_key"],
                "agent_id": entry["agent_id"],
                "stage": "batch_parse_retry",
                "error": f"retry: {exc}; original: {orig_err}",
            })
        retry_latencies_ms.append(int((time.time() - _retry_t0) * 1000))

    # Sonnet-escalation tail (weekly cadence only). First-Saturday
    # already submitted Sonnet via the batch so we skip the tail in
    # that path — every artifact already has a Sonnet eval.
    escalation_latencies_ms: list[int] = []
    if not force_sonnet_pass:
        # Select first, then spend. The eligibility test is pure in-memory
        # work over evals already held, so materialising the queue costs
        # nothing and makes the deadline residue an exact count rather than
        # "however many entries we had not looked at yet".
        escalation_queue = [
            entry for entry in plan["plan_entries"]
            if entry["judge_model"] == haiku_model
            and (
                _he := haiku_evals_by_agent_run.get(
                    (entry["agent_id"], entry["run_id"])
                )
            ) is not None
            and should_escalate_to_sonnet(
                _he, threshold=haiku_escalate_threshold,
            )
        ]
        logger.info(
            "[batch_process_escalation] %d of %d haiku evals flagged for "
            "sonnet escalation", len(escalation_queue),
            len(haiku_evals_by_agent_run),
        )
        _esc_t0: float | None = None
        for _esc_i, entry in enumerate(escalation_queue):
            # Close out the previous item here — the body's load-failure
            # path `continue`s, and a sample that misses it would under-
            # estimate the phase (same reasoning as the stream loop).
            if _esc_t0 is not None:
                escalation_latencies_ms.append(
                    int((time.time() - _esc_t0) * 1000)
                )
            affordable, needed = _next_item_affordable(
                remaining_s, escalation_latencies_ms,
                floor_s=PROCESS_LLM_ITEM_FLOOR_S,
            )
            if not affordable:
                _stop_for_budget(
                    "sonnet_escalation",
                    len(escalation_queue) - _esc_i, needed,
                )
                break
            _esc_t0 = time.time()

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

    budget_stopped = bool(budget_stopped_phases)

    # ── Coverage verdict (alpha-engine-config-I9309) ─────────────────────
    #
    # Brian's ruling 2026-08-29: "perhaps if the lambda times out then we
    # need to put the judge on a spot instance." The reasoning that governs
    # HOW it is built is his 2026-08-14 ruling about the Director — "director
    # should NEVER time out, if it times out it FAILS" — recorded with the
    # rationale that retries and larger ceilings are out of scope as fixes
    # BECAUSE they make a latency regression survivable instead of visible.
    #
    # Grading 10 of 85 artifacts and reporting `degraded` was precisely that
    # anti-pattern: an honest field on a stage that still returned success, so
    # a tenfold capacity shortfall cost the pipeline nothing and paged nobody.
    # These fields are the same measurement promoted to a VERDICT. The
    # transport rung stays honestly reported (`degraded_transport`) — that is
    # a legitimate degradation, recorded and priced. Coverage is now pass/fail:
    # `evals.judge_coverage.enforce_coverage` reads these fields and raises.
    #
    # Reported on every rung, including a clean one, because a field that
    # appears only when something is wrong is indistinguishable from a field
    # nobody emitted (principles.md §2.7: no data is never rendered as green).
    plan_entry_count = len(plan["plan_entries"])
    ungraded = [
        e for e in plan["plan_entries"]
        if e["custom_id"] not in graded_custom_ids
    ]
    logger.info(
        "[batch_process] done batch_id=%s date=%s haiku=%d sonnet=%d "
        "skipped_unmapped=%d skipped_empty_input=%d failed=%d "
        "parse_retry_recovered=%d metric_emission_failures=%d "
        "degraded_transport=%s sync_fallback_evaluated=%d "
        "complete=%s budget_stopped_phases=%s "
        "coverage=%d/%d ungraded=%d",
        batch_id, date, haiku_evaluated, sonnet_evaluated,
        plan.get("skipped_unmapped", 0), skipped_empty_input, len(failed),
        parse_retry_recovered, metric_emission_failures,
        is_sync_batch_id(batch_id), sync_fallback_evaluated,
        not budget_stopped, budget_stopped_phases or "-",
        len(graded_custom_ids), plan_entry_count, len(ungraded),
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
        "parse_retry_recovered": parse_retry_recovered,
        # Transport rung this pass actually ran on (alpha-engine-config-I9263).
        # `degraded_transport` is what a console or alarm reads to tell a
        # half-price batch pass from a full-price synchronous one; the count
        # says how much of the corpus the sync rung carried. Reported on BOTH
        # rungs — False/0 on the batch rung — because a field that only appears
        # when something is wrong is indistinguishable from a field nobody
        # emitted (principles.md §2.7: no data is never rendered as green).
        "degraded_transport": is_sync_batch_id(batch_id),
        "sync_fallback_evaluated": sync_fallback_evaluated,
        # sf-pipeline-policy.md §2.3a — a pass that covered less of the
        # corpus than it was asked to must say so, in a field a machine
        # reads. `complete` is False whenever any phase stopped on budget,
        # so a truncated eval sweep can never be read as a clean one.
        "complete": not budget_stopped,
        "budget_stopped": budget_stopped,
        "budget_stopped_phases": budget_stopped_phases,
        "n_skipped_for_budget": n_skipped_for_budget,
        # Coverage ledger — see the block above `plan_entry_count`.
        # `plan_entry_count` is the corpus this pass was ASKED to grade
        # (1:1 with `plan["requests"]`, client-side skips already excluded,
        # so it is never inflated by artifacts that were deliberately not
        # judged). `plan_entries_graded` is what it DID grade. Equality is
        # the pass condition; the shortfall list names the misses so a reader
        # does not have to diff two S3 listings to learn which agents lost
        # their eval this week.
        "plan_entry_count": plan_entry_count,
        "plan_entries_graded": len(graded_custom_ids),
        "plan_entries_ungraded": len(ungraded),
        "coverage_complete": not ungraded,
        "ungraded_entries": [
            {
                "custom_id": e["custom_id"],
                "agent_id": e["agent_id"],
                "run_id": e["run_id"],
                "judge_model": e["judge_model"],
                "capture_s3_key": e["capture_s3_key"],
            }
            for e in ungraded
        ],
        "failed": failed,
        "persisted_keys": persisted_keys,
        "haiku_model": haiku_model,
        "sonnet_model": sonnet_model,
        "force_sonnet_pass": force_sonnet_pass,
        "judge_only": plan.get("judge_only", False),
        "eval_prefix": eval_prefix,
        "cw_namespace": cw_namespace,
    }
