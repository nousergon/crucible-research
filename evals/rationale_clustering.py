"""Cross-week rationale clustering for agent-justification — RETIRED
(alpha-engine-config-I8173).

Per ROADMAP P0 "Cross-week rationale clustering for agent-justification":

  An LLM agent that emits structurally identical rationales every week
  with only numerics swapped is a deterministic rule wearing an agent
  costume. The ``reasoning_complexity`` rubric dim scores per-call
  complexity but cannot detect this multi-week pattern. This module
  closed that gap for the six agent families the retired multi-agent
  research graph produced.

Retirement (alpha-engine-config-I8173, ruled 2026-08-22):

  This module used to (1) read
  ``decision_artifacts/{YYYY}/{MM}/{DD}/{agent_id}/*.json`` for a trailing
  N-week window, (2) extract + cluster rationales per agent family, and
  (3) persist ``decision_artifacts/_analysis/{agent_id}/{YYYY-WW}.json`` +
  emit the ``agent_rationale_template_concentration`` CloudWatch metric.

  All six agent families in ``RETIRED_AGENT_FAMILIES`` below were produced
  by exactly one writer — the retired multi-agent research graph's
  ``_capture_if_enabled`` capture hook — deleted in
  alpha-engine-config-I7827 / crucible-research-PR685 (the graph itself,
  alpha-engine-config-I7817). Last write across all six families:
  **2026-07-11**. Verified empty 2026-08-22:
  ``aws s3 ls s3://alpha-engine-research/decision_artifacts/2026/08/22/``
  returns nothing for any of the six families.

  Measured 2026-08-15 (the bug this module's freshness check was ADDED to
  catch, alpha-engine-config-I2638): every ``sector_quant`` capture was
  frozen at 2026-07-11, and this module still wrote a freshly-dated
  ``_analysis/sector_quant/2026-W33.json`` and stamped a CloudWatch
  datapoint at today's timestamp — a template-concentration figure
  computed over a dead corpus, presented as a claim about the current
  week. Widening the freshness tolerance would not have fixed this: the
  producer is gone permanently, not merely late. **Brian ruled: retire
  the stale-input check AND the live reader — do not re-publish the
  42-day-old captures, do not revive the producer.**

  ``compute_and_emit`` therefore no longer lists, reads, clusters,
  persists or emits anything for ``decision_artifacts/{date}/{agent}/``.
  It short-circuits and returns a ``status: "retired"`` summary naming
  the six families and the retiring issues — a future reader (of the SF
  summary, of this module, or of the S3 prefix) can tell "deliberately
  retired" from "producer broke" without re-deriving the history above.
  ``extract_rationales`` / ``cluster_rationales`` / ``compute_concentration``
  (the pure per-agent-parsing + TF-IDF clustering primitives) are left
  intact and unit-tested: per alpha-engine-config-I8173 deliverable 4, if
  a sweep ever finds one of these families alive again under a NEW path,
  this module resumes its captures rather than staying suppressed.

  Sweep result (I8173 deliverable 3): the fleet's only OTHER
  ``freshness.assert_upstream_fresh`` call sites are the two in
  ``thinktank/context.py`` (``regime_substrate``, ``news_aggregates``) —
  both verified live 2026-08-22 (``regime_substrate`` last wrote
  2026-08-22T17:17Z; ``news_aggregates`` was repointed at the live daily
  producer by alpha-engine-config-I8174/crucible-research-PR725). No
  other prefix orphaned by I7827/I7817 was left asserting freshness
  anywhere in ``crucible-research`` or ``crucible-dashboard``.

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

import logging
import math
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

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

RETIRED_AGENT_FAMILIES: frozenset[str] = frozenset(
    {
        "ic_cio",
        "macro_economist",
        "sector_peer_review",
        "sector_qual",
        "sector_quant",
        "thesis_update",
    }
)
"""Every ``decision_artifacts/{date}/{agent}`` family this module ever read
— the complete set ``extract_rationales`` had a branch for. The sole
producer for all six was the retired multi-agent research graph's
``_capture_if_enabled`` capture hook, deleted in
alpha-engine-config-I7827 / crucible-research-PR685 (the graph itself,
alpha-engine-config-I7817). Last write: 2026-07-11; verified empty
2026-08-22. This is the written rationale alpha-engine-config-I8173
deliverable 2 requires: naming the retiring change here, on the constant
``compute_and_emit`` cites in its retirement summary, so a future reader
sees an explicit RETIRED marker rather than an empty result
indistinguishable from "producer broke"."""

_RETIRED_REASON = (
    "producer deleted alpha-engine-config-I7827 / crucible-research-PR685 "
    "(the retired multi-agent research graph's decision-capture hook); "
    "agent retirement sweep alpha-engine-config-I7817; last write "
    "2026-07-11, verified empty 2026-08-22 "
    "(alpha-engine-config-I8173 — do not re-publish the stale captures, "
    "do not revive the producer)"
)

# Per-agent scope cap — bounds clustering wall-clock on agents with
# unusually large rationale corpora (e.g. a quant agent that ran 50
# tool-calls in one cycle generates ≥50 rationale strings per artifact,
# multiplied by ~56 days = up to ~3000 rationales for one agent before
# this cap). Mirrors the Counterfactual Lambda's
# DEFAULT_MAX_ARTIFACTS_PER_AGENT=500 precedent
# (alpha-engine-backtester #228, 2026-05-19). Closes 5/23-SF P0 (a) —
# the 2026-05-24 trading-day-fix recovery's RationaleClustering Lambda
# timed out at 600s (SF event 269), dropping the analysis artifact +
# leaving downstream consumers reading a stale corpus signature. Cap
# applied AFTER artifact load but BEFORE clustering (most expensive
# stage); truncation logged + reported in the summary so operators see
# when caps fire.
DEFAULT_MAX_RATIONALES_PER_AGENT = 500
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


# ── Top-level pipeline ───────────────────────────────────────────────────


def compute_and_emit(
    *,
    end_time: datetime | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    bucket: str = DEFAULT_BUCKET,
    capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
    analysis_prefix: str = DEFAULT_ANALYSIS_PREFIX,
    namespace: str = DEFAULT_NAMESPACE,
    metric_name: str = DEFAULT_METRIC_NAME,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    s3_client: Any | None = None,
    cloudwatch_client: Any | None = None,
    emit_metrics: bool = True,
    max_rationales_per_agent: int = DEFAULT_MAX_RATIONALES_PER_AGENT,
) -> dict[str, Any]:
    """RETIRED (alpha-engine-config-I8173) — see the module docstring.

    Used to read ``{capture_prefix}/{Y}/{M}/{D}/{agent_id}/*.json`` in the
    trailing window, cluster rationales per ``agent_id``, persist per-agent
    analysis JSON, and emit CloudWatch concentration metrics. Every agent
    family it ever read (``RETIRED_AGENT_FAMILIES``) lost its sole producer
    to alpha-engine-config-I7827 / crucible-research-PR685; the sweep for
    this issue found no other family alive under a new path
    (alpha-engine-config-I8173 deliverable 4). A consumer silently drawing
    conclusions from the resulting 42-day-frozen corpus is worse than one
    that does nothing, so this now performs **no S3 listing, no S3 read, no
    clustering, no S3 persist, and no CloudWatch emission** — ``bucket``,
    ``capture_prefix``, ``analysis_prefix``, ``namespace``, ``metric_name``,
    ``similarity_threshold``, ``s3_client``, ``cloudwatch_client``,
    ``emit_metrics`` and ``max_rationales_per_agent`` are accepted only to
    keep the call signature stable for ``lambda/rationale_clustering_handler.py``
    and are otherwise unused.

    Returns a summary dict shaped like the live pipeline's used to be
    (``load_failures`` / ``cluster_failures`` stay present and empty so the
    handler's ``has_failures`` check keeps working), plus an explicit
    ``status: "retired"`` and ``retired_agent_families`` / ``retired_reason``
    so a future reader of the SF summary — or a re-registration of this
    stage — can tell "deliberately retired" from "producer broke" without
    re-deriving the history in the module docstring.
    """
    end = end_time or datetime.now(UTC)
    window_start = end - timedelta(days=window_days)

    logger.info(
        "[rationale_clustering] RETIRED — no decision_artifacts read "
        "performed for families=%s (alpha-engine-config-I8173: producer "
        "retired by alpha-engine-config-I7827/crucible-research-PR685, "
        "agent retirement sweep alpha-engine-config-I7817)",
        sorted(RETIRED_AGENT_FAMILIES),
    )

    return {
        "status": "retired",
        "retired_agent_families": sorted(RETIRED_AGENT_FAMILIES),
        "retired_reason": _RETIRED_REASON,
        "window_start": window_start.isoformat(),
        "window_end": end.isoformat(),
        "artifacts_discovered": 0,
        "agents_analyzed": 0,
        "agents_skipped_thin_sample": [],
        "load_failures": [],
        "cluster_failures": [],
        "agents_truncated_by_scope_cap": [],
        "agents_key_capped": [],
        "artifacts_fetched": 0,
        "max_rationales_per_agent": max_rationales_per_agent,
        "corpus_freshness": None,
        "agents_stale_corpus": [],
        "per_agent": [],
    }
