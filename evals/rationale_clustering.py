"""Cross-week rationale clustering for agent-justification.

Per ROADMAP P0 "Cross-week rationale clustering for agent-justification":

  An LLM agent that emits structurally identical rationales every week
  with only numerics swapped is a deterministic rule wearing an agent
  costume. The ``reasoning_complexity`` rubric dim scores per-call
  complexity but cannot detect this multi-week pattern. This module
  closes that gap.

Pipeline:

  1. Read ``decision_artifacts/{YYYY}/{MM}/{DD}/{agent_id}/*.json`` for
     a trailing N-week window (default 8 weeks ≈ 8 Saturdays).
  2. Extract per-agent rationale strings — different fields per agent_id
     family (sector_quant pulls per-pick ``quant_rationale``; sector_qual
     pulls ``bull_case``; sector_peer_review pulls per-pick ``rationale``;
     macro_economist pulls regime call text; ic_cio pulls per-decision
     ``rationale``; thesis_update:* pulls ``bull_case``).
  3. Cluster rationales using TF-IDF char n-grams + cosine-similarity
     greedy agglomerative merge. **Why TF-IDF char n-grams over semantic
     embeddings**: we want STRUCTURAL similarity ("same template, different
     numbers"), not SEMANTIC similarity ("both talk about momentum").
     A semantic embedding would say "yes, all sector_quant rationales are
     about quantitative analysis" and miss the template-detection signal.
     Char n-grams catch the skeleton — "P/E of {N} is below sector median
     of {M}" matches whether N=12, M=18 or N=25, M=30 — exactly the
     rule-in-LLM-costume pattern this module exists to detect.
  4. Compute per-agent ``rationale_template_concentration`` =
     (sum of top-3 cluster sizes) / total rationales. >70% indicates
     template-generation.
  5. Emit CloudWatch metric ``agent_rationale_template_concentration``
     dimensioned by ``judged_agent_id``.
  6. Persist per-agent analysis output to
     ``decision_artifacts/_analysis/{agent_id}/{YYYY-WW}.json``.

Composes with:

  - ``reasoning_complexity`` rubric dim (per-call complexity score).
  - LLM-as-judge rolling 4-week mean (``evals/rolling_mean.py``).
  - Counterfactual rule fit + cheap-model concordance (Model-Agnostic
    deliverable #7) — together these are the agent-justification triple.

Dependency note: this module uses pure-numpy clustering rather than
scikit-learn / HDBSCAN to avoid a +70MB image-size hit on the eval-judge
Lambda image. The pure-numpy implementation is sufficient for the
expected scale (~500 rationales × ~6 agents over 8 weeks). If the corpus
grows beyond ~5000 per-agent rationales, swap in scikit-learn's
``AgglomerativeClustering(metric='precomputed')`` — same algorithm,
better C-level inner loop.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import boto3

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_NAMESPACE = "AlphaEngine/Eval"
"""Same namespace as ``agent_quality_score`` — agent-justification
metrics share the dashboard with eval-quality metrics."""

DEFAULT_METRIC_NAME = "agent_rationale_template_concentration"
"""% of rationales falling into the top-3 clusters per agent. Range
[0, 1]. >0.7 indicates template-generation (deterministic-rule-shaped
behavior); <0.3 indicates broad rationale variety (genuine LLM
synthesis)."""

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_CAPTURE_PREFIX = "decision_artifacts"
DEFAULT_ANALYSIS_PREFIX = "decision_artifacts/_analysis"

DEFAULT_WINDOW_DAYS = 56
"""8-week (56-day) trailing window — matches ROADMAP wording. Captures
8 Saturdays of weekly research runs. Configurable via the
``window_days`` arg on ``compute_and_emit``."""

DEFAULT_SIMILARITY_THRESHOLD = 0.75
"""Cosine-similarity threshold above which two rationales are merged
into the same cluster. With digit-normalization (numbers → ``#``)
applied in ``_normalize``, same-template rationales land at ~0.95+
because the skeleton is identical post-normalization. Structurally
distinct rationales land near 0.01-0.10. 0.75 sits in the wide gap
between the two regimes — strict enough that two genuinely-different
templates that share a few common words don't merge, loose enough
that minor wording variation within a template (one extra adjective,
slight reordering) still merges. Calibrate against real corpus once
4+ weeks accumulated; v1 default."""

MIN_RATIONALES_FOR_CLUSTERING = 5
"""Below this count, clustering output is statistically meaningless —
emit metric as None (skip) rather than report a noisy value."""

CHAR_NGRAM_RANGE = (3, 5)
"""Character n-gram range for TF-IDF. 3-5 catches morphological
patterns ("the P/E", "ratio of") and short skeletal templates without
exploding feature dimensionality."""


# ── Per-agent rationale extraction ───────────────────────────────────────


def extract_rationales(agent_id: str, agent_output: dict[str, Any]) -> list[str]:
    """Pull rationale strings out of the agent_output dict. Different
    agents emit rationales under different field names — this function
    centralizes the per-agent mapping.

    Returns a list of rationale strings (one or many per artifact —
    e.g. sector_quant emits one per top-5 pick). Empty list when the
    agent_output has no rationales (skip-marker artifacts, parse
    failures, etc.).

    The agent_id may be plain (``"macro_economist"``) or namespaced
    (``"sector_quant:tech"``, ``"thesis_update:AAPL"``). The colon
    namespace separator is the existing capture convention.
    """
    if not agent_output or not isinstance(agent_output, dict):
        return []

    base_id = agent_id.split(":", 1)[0]

    # Field names below were validated against real captures from
    # 2026-05-03 (see extractor smoke run notes). Each branch lists the
    # actual top-level keys observed in agent_output.

    if base_id == "sector_quant":
        # ranked_picks[*].rationale — list of 5-10 picks per sector team.
        rationales: list[str] = []
        picks = agent_output.get("ranked_picks")
        if isinstance(picks, list):
            rationales.extend(
                str(p["rationale"]).strip()
                for p in picks
                if isinstance(p, dict) and p.get("rationale")
            )
        return [r for r in rationales if r]

    if base_id == "sector_qual":
        # assessments[*].bull_case — list of qual assessments per sector.
        rationales = []
        items = agent_output.get("assessments")
        if isinstance(items, list):
            rationales.extend(
                str(a["bull_case"]).strip()
                for a in items
                if isinstance(a, dict) and a.get("bull_case")
            )
        return [r for r in rationales if r]

    if base_id == "sector_peer_review":
        # Two-track: per-pick recommendations[*].peer_review_rationale +
        # the top-level cross-pick peer_review_rationale string. Both
        # carry distinct rationale signal — the former is per-ticker,
        # the latter is the team-level synthesis.
        rationales = []
        items = agent_output.get("recommendations")
        if isinstance(items, list):
            rationales.extend(
                str(d["peer_review_rationale"]).strip()
                for d in items
                if isinstance(d, dict) and d.get("peer_review_rationale")
            )
        team_rat = agent_output.get("peer_review_rationale")
        if isinstance(team_rat, str) and team_rat.strip():
            rationales.append(team_rat.strip())
        return [r for r in rationales if r]

    if base_id == "macro_economist":
        # macro_report is the canonical narrative field on real captures
        # (~2KB per call). Fall back to other candidates if a future
        # capture format renames it.
        for key in ("macro_report", "regime_rationale", "rationale", "summary"):
            v = agent_output.get(key)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
        return []

    if base_id == "ic_cio":
        # ic_decisions[*].rationale — per-candidate CIO decisions.
        decisions = agent_output.get("ic_decisions")
        if not isinstance(decisions, list):
            return []
        return [
            str(d["rationale"]).strip()
            for d in decisions
            if isinstance(d, dict) and d.get("rationale")
        ]

    if base_id == "thesis_update":
        # Held-stock thesis carries multiple narrative fields; treat each
        # as an independent rationale so per-field templating is
        # observable. bull_case + conviction_rationale + thesis_summary
        # + triggers_response are all author-emitted prose with distinct
        # purpose; clustering them separately catches "all four wear the
        # same template" patterns.
        rationales = []
        for key in (
            "bull_case",
            "conviction_rationale",
            "thesis_summary",
            "triggers_response",
        ):
            v = agent_output.get(key)
            if isinstance(v, str) and v.strip():
                rationales.append(v.strip())
        return rationales

    # Unknown agent_id family — return no rationales (silent skip is the
    # right behavior here; a new agent family should explicitly opt in).
    return []


# ── TF-IDF char n-gram vectorization (pure numpy) ────────────────────────


_TOKEN_RE = re.compile(r"\s+")
_DIGIT_RUN_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _normalize(text: str) -> str:
    """Normalize for STRUCTURAL similarity. Steps:

    1. Lowercase.
    2. Replace runs of digits (including decimals) with a single ``#``
       placeholder so "P/E of 12" and "P/E of 25" hash to the same
       skeleton. The numbers ARE the noise we're filtering — this
       module's whole point is detecting "same template, different
       numerics."
    3. Collapse whitespace.

    Punctuation is preserved — it's part of the skeleton (``P/E``,
    ``$N``, ``X%``). Multi-line rationales collapse to single-line
    skeletons via whitespace normalization.
    """
    lowered = text.lower()
    digit_normalized = _DIGIT_RUN_RE.sub("#", lowered)
    return _TOKEN_RE.sub(" ", digit_normalized).strip()


def _char_ngrams(text: str, n_min: int, n_max: int) -> list[str]:
    """Char n-grams over normalized text. Pad with spaces so word-edge
    n-grams ("P/E ", " ratio") stay distinct from mid-word ones."""
    s = f" {text} "
    grams: list[str] = []
    for n in range(n_min, n_max + 1):
        if len(s) < n:
            continue
        grams.extend(s[i : i + n] for i in range(len(s) - n + 1))
    return grams


def _build_tfidf_matrix(
    rationales: list[str],
) -> tuple[list[dict[str, float]], int]:
    """Build sparse TF-IDF vectors for ``rationales``. Returns a list of
    ``{ngram: tfidf_weight}`` dicts plus the feature dimensionality
    (vocab size). Sparse dict-of-floats is fine here — corpora are
    small enough (hundreds of rationales) that dense numpy matrices
    aren't worth the memory.
    """
    docs = [_normalize(r) for r in rationales]
    doc_grams = [_char_ngrams(d, *CHAR_NGRAM_RANGE) for d in docs]

    # Document frequency for each ngram.
    df: Counter[str] = Counter()
    for grams in doc_grams:
        df.update(set(grams))

    n_docs = len(docs)
    idf: dict[str, float] = {
        gram: math.log((n_docs + 1) / (count + 1)) + 1.0
        for gram, count in df.items()
    }

    vectors: list[dict[str, float]] = []
    for grams in doc_grams:
        if not grams:
            vectors.append({})
            continue
        tf = Counter(grams)
        # Sublinear TF: 1 + log(tf) — standard variant; reduces influence
        # of one ngram repeating many times in a single rationale.
        weighted = {
            gram: (1.0 + math.log(count)) * idf.get(gram, 0.0)
            for gram, count in tf.items()
        }
        # L2 normalize so cosine similarity is just the dot product.
        norm = math.sqrt(sum(w * w for w in weighted.values()))
        if norm > 0:
            weighted = {gram: w / norm for gram, w in weighted.items()}
        vectors.append(weighted)

    return vectors, len(idf)


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two L2-normalized sparse vectors.
    Iterate the smaller dict for speed."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(gram, 0.0) for gram, w in a.items())


# ── Greedy agglomerative clustering ──────────────────────────────────────


def cluster_rationales(
    rationales: list[str],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[list[int]]:
    """Greedy single-linkage clustering: each rationale joins the first
    existing cluster whose centroid (any member, single-linkage) has
    cosine similarity ≥ ``similarity_threshold``; otherwise starts a new
    cluster.

    Returns a list of clusters; each cluster is a list of indices into
    ``rationales``. Cluster order is creation order; sort by size at
    the consumer if needed.

    Single-linkage is the right choice for template detection: we want
    "rationale X and rationale Y share a structural skeleton" to chain
    transitively across a cluster. Average-linkage would dilute the
    skeleton signal as cluster size grows.

    Complexity: O(n²) similarity computations in the worst case. With
    n=500 rationales and ~hundreds of ngrams per vector this runs in
    a few seconds — acceptable for a weekly Lambda. If n grows past a
    few thousand, swap to scikit-learn's optimized impl.
    """
    if not rationales:
        return []

    vectors, _vocab_size = _build_tfidf_matrix(rationales)
    clusters: list[list[int]] = []

    for idx, vec in enumerate(vectors):
        joined = False
        for cluster in clusters:
            # Single-linkage: any cluster member above threshold means
            # this rationale joins.
            for member_idx in cluster:
                if _cosine_sim(vec, vectors[member_idx]) >= similarity_threshold:
                    cluster.append(idx)
                    joined = True
                    break
            if joined:
                break
        if not joined:
            clusters.append([idx])

    return clusters


def compute_concentration(
    clusters: list[list[int]],
    *,
    top_k: int = 3,
) -> float:
    """Return the fraction of rationales contained in the top-K clusters
    by size. Range [0, 1]. Returns 0 when clusters is empty."""
    if not clusters:
        return 0.0
    sizes = sorted((len(c) for c in clusters), reverse=True)
    total = sum(sizes)
    if total == 0:
        return 0.0
    return sum(sizes[:top_k]) / total


# ── S3 corpus reading ────────────────────────────────────────────────────


def _list_artifact_keys_in_window(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    end_date: datetime,
    window_days: int,
) -> list[str]:
    """List every captured-artifact key under
    ``{capture_prefix}/{Y}/{M}/{D}/`` for each day in the trailing
    ``window_days`` ending at ``end_date``. Excludes ``_eval/``,
    ``_eval_judge_only/``, ``_analysis/``, ``_cost/`` subtrees so we
    only ingest production captures.
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []

    # List per-day to keep the prefix tight (one listing per day rather
    # than one global listing across the whole bucket).
    for day_offset in range(window_days):
        day = end_date - timedelta(days=day_offset)
        prefix = (
            f"{capture_prefix}/{day.strftime('%Y')}/"
            f"{day.strftime('%m')}/{day.strftime('%d')}/"
        )
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                if "/_eval/" in key or "/_eval_judge_only/" in key:
                    continue
                if "/_analysis/" in key or "/_cost/" in key:
                    continue
                if "/_cost_raw/" in key:
                    continue
                keys.append(key)

    return keys


def _agent_id_from_key(key: str) -> Optional[str]:
    """Extract ``agent_id`` from an S3 key shaped
    ``decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json``.
    Returns None on unexpected layout (defensive)."""
    parts = key.split("/")
    # Find the last directory before the filename — that's the agent_id.
    if len(parts) < 2:
        return None
    return parts[-2]


def _load_artifact(s3: Any, *, bucket: str, key: str) -> dict[str, Any]:
    """Load and JSON-parse one captured artifact. Returns the raw dict
    rather than the typed ``DecisionArtifact`` model — we only need
    ``agent_id`` and ``agent_output``, and tolerating schema drift
    (additive-only fields) is preferable to hard-failing the whole
    weekly clustering run on one stale artifact."""
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(raw)


# ── Per-agent analysis output ────────────────────────────────────────────


def _build_per_agent_output(
    agent_id: str,
    rationales: list[str],
    clusters: list[list[int]],
    concentration: float,
    *,
    window_start: datetime,
    window_end: datetime,
    representatives_per_cluster: int = 3,
) -> dict[str, Any]:
    """Render the persisted per-agent analysis JSON. Includes cluster
    sizes + top-N representatives per cluster so dashboard consumers
    can show "what does this agent's #1 pattern look like?" without
    re-reading the source captures."""
    # Sort clusters by size descending so top_k clusters appear first.
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    cluster_summaries: list[dict[str, Any]] = []
    for cluster in sorted_clusters:
        reps = [
            rationales[idx][:500]  # truncate each rep to 500 chars for readability
            for idx in cluster[:representatives_per_cluster]
        ]
        cluster_summaries.append(
            {
                "size": len(cluster),
                "fraction": len(cluster) / len(rationales) if rationales else 0.0,
                "representatives": reps,
            }
        )

    return {
        "schema_version": 1,
        "agent_id": agent_id,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "n_rationales": len(rationales),
        "n_clusters": len(clusters),
        "top3_concentration": concentration,
        "clusters": cluster_summaries,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _persist_analysis(
    s3: Any,
    *,
    bucket: str,
    analysis_prefix: str,
    agent_id: str,
    end_date: datetime,
    payload: dict[str, Any],
) -> str:
    """Write per-agent analysis JSON to
    ``{analysis_prefix}/{agent_id}/{YYYY-WW}.json``. ISO week is the
    natural cadence (one analysis per Saturday SF run)."""
    iso_year, iso_week, _ = end_date.isocalendar()
    key = f"{analysis_prefix}/{agent_id}/{iso_year}-W{iso_week:02d}.json"
    body = json.dumps(payload, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return key


# ── CloudWatch metric emission ───────────────────────────────────────────


def _emit_concentration_metric(
    cw: Any,
    *,
    namespace: str,
    metric_name: str,
    agent_id: str,
    concentration: float,
    n_rationales: int,
    timestamp: datetime,
) -> None:
    """One datapoint per agent_id. Dimensioned by ``judged_agent_id``
    to match ``agent_quality_score`` so the dashboard can join on the
    same dim. ``n_rationales`` published as a separate metric so the
    operator can see when concentration is reported on a thin sample."""
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                "MetricName": metric_name,
                "Dimensions": [
                    {"Name": "judged_agent_id", "Value": agent_id},
                ],
                "Value": float(concentration),
                "Unit": "None",
                "Timestamp": timestamp,
            },
            {
                "MetricName": f"{metric_name}_n_rationales",
                "Dimensions": [
                    {"Name": "judged_agent_id", "Value": agent_id},
                ],
                "Value": float(n_rationales),
                "Unit": "Count",
                "Timestamp": timestamp,
            },
        ],
    )


# ── Top-level pipeline ───────────────────────────────────────────────────


def compute_and_emit(
    *,
    end_time: Optional[datetime] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    bucket: str = DEFAULT_BUCKET,
    capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
    analysis_prefix: str = DEFAULT_ANALYSIS_PREFIX,
    namespace: str = DEFAULT_NAMESPACE,
    metric_name: str = DEFAULT_METRIC_NAME,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    s3_client: Optional[Any] = None,
    cloudwatch_client: Optional[Any] = None,
    emit_metrics: bool = True,
) -> dict[str, Any]:
    """Read captured artifacts in the trailing window, cluster
    rationales per ``agent_id``, persist per-agent analysis JSON, and
    emit CloudWatch concentration metrics.

    Returns a summary dict suitable for SF inspection. Per-agent
    failures are logged + accumulated rather than raised, matching
    the eval orchestrator's "observability of observability" pattern.

    Agents with fewer than ``MIN_RATIONALES_FOR_CLUSTERING`` rationales
    in the window are skipped (concentration on a thin sample is
    statistically meaningless); the summary records them under
    ``skipped_thin_sample``.
    """
    s3 = s3_client or boto3.client("s3")
    cw = cloudwatch_client or (boto3.client("cloudwatch") if emit_metrics else None)
    end = end_time or datetime.now(timezone.utc)
    window_start = end - timedelta(days=window_days)

    keys = _list_artifact_keys_in_window(
        s3,
        bucket=bucket,
        capture_prefix=capture_prefix,
        end_date=end,
        window_days=window_days,
    )

    logger.info(
        "[rationale_clustering] discovered %d artifacts in window=[%s, %s]",
        len(keys), window_start.isoformat(), end.isoformat(),
    )

    # Group rationales by agent_id.
    by_agent: dict[str, list[str]] = defaultdict(list)
    load_failures: list[dict[str, str]] = []

    for key in keys:
        agent_id = _agent_id_from_key(key)
        if agent_id is None:
            continue
        try:
            artifact = _load_artifact(s3, bucket=bucket, key=key)
        except Exception as exc:  # noqa: BLE001
            load_failures.append({"key": key, "error": str(exc)})
            logger.warning(
                "[rationale_clustering] load failure key=%s err=%s",
                key, exc,
            )
            continue

        rationales = extract_rationales(
            artifact.get("agent_id", agent_id),
            artifact.get("agent_output") or {},
        )
        by_agent[agent_id].extend(rationales)

    # Cluster + emit per agent.
    per_agent_summary: list[dict[str, Any]] = []
    skipped_thin: list[dict[str, Any]] = []
    cluster_failures: list[dict[str, str]] = []

    for agent_id, rationales in sorted(by_agent.items()):
        if len(rationales) < MIN_RATIONALES_FOR_CLUSTERING:
            skipped_thin.append(
                {"agent_id": agent_id, "n_rationales": len(rationales)}
            )
            continue

        try:
            clusters = cluster_rationales(
                rationales, similarity_threshold=similarity_threshold,
            )
            concentration = compute_concentration(clusters, top_k=3)
        except Exception as exc:  # noqa: BLE001
            cluster_failures.append({"agent_id": agent_id, "error": str(exc)})
            logger.exception(
                "[rationale_clustering] cluster failure agent=%s",
                agent_id,
            )
            continue

        payload = _build_per_agent_output(
            agent_id, rationales, clusters, concentration,
            window_start=window_start, window_end=end,
        )

        try:
            analysis_key = _persist_analysis(
                s3,
                bucket=bucket,
                analysis_prefix=analysis_prefix,
                agent_id=agent_id,
                end_date=end,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            cluster_failures.append(
                {"agent_id": agent_id, "stage": "persist", "error": str(exc)}
            )
            logger.exception(
                "[rationale_clustering] persist failure agent=%s",
                agent_id,
            )
            continue

        if cw is not None:
            try:
                _emit_concentration_metric(
                    cw,
                    namespace=namespace,
                    metric_name=metric_name,
                    agent_id=agent_id,
                    concentration=concentration,
                    n_rationales=len(rationales),
                    timestamp=end,
                )
            except Exception as exc:  # noqa: BLE001
                # Metric emission is observability of observability —
                # don't fail the run if CloudWatch hiccups.
                cluster_failures.append(
                    {
                        "agent_id": agent_id,
                        "stage": "metric_emit",
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "[rationale_clustering] metric emission failed agent=%s err=%s",
                    agent_id, exc,
                )

        per_agent_summary.append(
            {
                "agent_id": agent_id,
                "n_rationales": len(rationales),
                "n_clusters": len(clusters),
                "top3_concentration": concentration,
                "analysis_key": analysis_key,
            }
        )

        logger.info(
            "[rationale_clustering] agent=%s n=%d clusters=%d top3_conc=%.3f",
            agent_id, len(rationales), len(clusters), concentration,
        )

    summary = {
        "window_start": window_start.isoformat(),
        "window_end": end.isoformat(),
        "artifacts_discovered": len(keys),
        "agents_analyzed": len(per_agent_summary),
        "agents_skipped_thin_sample": skipped_thin,
        "load_failures": load_failures,
        "cluster_failures": cluster_failures,
        "per_agent": per_agent_summary,
    }

    logger.info(
        "[rationale_clustering] done agents=%d skipped=%d load_fail=%d cluster_fail=%d",
        len(per_agent_summary),
        len(skipped_thin),
        len(load_failures),
        len(cluster_failures),
    )

    return summary
