"""
Score aggregator — live helpers used by the sector-team pipeline.

This module now hosts only the LIVE surface consumed by ``graph.research_graph``
and ``thesis.updater``:

  * ``_load_weights_from_s3`` / ``_get_weights`` — backtester-tuned scoring
    weights loader (contract-pinned by ``tests/test_tuned_config_consumer_contract``).
  * ``score_to_rating`` — composite score → BUY/HOLD/SELL.
  * ``assign_cross_sectional_ranks`` — attaches cross-sectional rank + percentile.

Removed 2026-06-14 (config#1060/#1061): the legacy ``compute_attractiveness_score``
+ ``aggregate_all`` cluster (and their boost/staleness/conviction helpers) had
ZERO live callers — the sector-team architecture scores via ``scoring.composite``
(``compute_composite_breakdown`` / ``compute_composite_score``). That cluster
carried a SECOND, divergent copy of the macro_shift formula
(``(sector_modifier - 1.0) / 0.30 × MACRO_MAX_SHIFT_POINTS``); deleting it
prevents the macro-overlay disable in ``scoring/composite.py`` from being
silently re-introduced via a dead-but-resurrectable path. The macro_shift
overlay enable knob lives in ``scoring.composite`` + ``config.MACRO_OVERLAY_ENABLED``.
"""

from __future__ import annotations

import json
import logging
import os

from config import (
    WEIGHT_QUANT,
    WEIGHT_QUAL,
    RATING_BUY_THRESHOLD,
    RATING_SELL_THRESHOLD,
    check_s3_pointer_staleness,
)

logger = logging.getLogger(__name__)

_weights_cache: dict | None = None
# Local cache persists last known optimal across Lambda cold-starts (via /tmp).
_WEIGHTS_CACHE_PATH = os.environ.get(
    "SCORING_WEIGHTS_CACHE", "/tmp/scoring_weights_cache.json"
)


def _load_weights_from_s3() -> dict | None:
    """
    Check S3 for backtester-updated scoring weights.

    Reads s3://{RESEARCH_BUCKET}/config/scoring_weights.json, written by the
    backtester's weight optimizer when it applies an update.

    Fallback chain: S3 → local cache file → None (hardcoded defaults).
    On successful S3 read, writes a local cache so the last known optimal
    weights survive transient S3 failures.
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    key = "config/scoring_weights.json"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        # config#2891: independent consumer-side staleness signal — WARN-only,
        # complements the central config_scoring_weights freshness-monitor row.
        check_s3_pointer_staleness(obj.get("LastModified"), key, logger=logger)
        data = json.loads(obj["Body"].read())
        weights = {
            "quant": float(data["quant"]),
            "qual": float(data["qual"]),
        } if "quant" in data and "qual" in data else {}
        if len(weights) == 2:
            logger.info(
                "Scoring weights loaded from S3 (updated %s, n=%s): %s",
                data.get("updated_at", "unknown"),
                data.get("n_samples", "?"),
                weights,
            )
            # Persist to local cache for fault tolerance
            try:
                with open(_WEIGHTS_CACHE_PATH, "w") as f:
                    json.dump(weights, f, indent=2)
            except Exception as e:
                logger.debug("Could not write scoring weights cache: %s", e)
            return weights
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            logger.warning("Could not read scoring weights from S3: %s", e)
    except Exception as e:
        logger.warning("Unexpected error loading S3 scoring weights: %s", e)

    # Fallback: last known optimal from local cache
    try:
        if os.path.exists(_WEIGHTS_CACHE_PATH):
            with open(_WEIGHTS_CACHE_PATH) as f:
                weights = json.load(f)
            if "quant" in weights and "qual" in weights:
                logger.info(
                    "Scoring weights loaded from local cache (last known optimal): %s",
                    weights,
                )
                return weights
    except Exception as e2:
        logger.debug("Could not read local scoring weights cache: %s", e2)

    return None


def _get_weights() -> tuple[float, float]:
    """
    Return current scoring weights (quant, qual), checking S3 override first.

    Result is cached for the lifetime of the Lambda instance (one S3 call
    per cold-start). Falls back to universe.yaml values if S3 file absent.
    """
    global _weights_cache
    if _weights_cache is None:
        s3_weights = _load_weights_from_s3()
        _weights_cache = s3_weights or {
            "quant": WEIGHT_QUANT,
            "qual": WEIGHT_QUAL,
        }
    return _weights_cache["quant"], _weights_cache["qual"]


def score_to_rating(score: float) -> str:
    """Convert numeric score to BUY / HOLD / SELL (§5.5)."""
    if score >= RATING_BUY_THRESHOLD:
        return "BUY"
    if score >= RATING_SELL_THRESHOLD:
        return "HOLD"
    return "SELL"


def assign_cross_sectional_ranks(results: dict[str, dict]) -> None:
    """Attach ``cross_sectional_rank`` + ``percentile`` to each result dict
    in-place.

    PR 1 of the rank-based portfolio-construction restructure. These
    additive fields are emitted on every per-ticker scoring dict so
    downstream consumers (executor, backtester signal_quality, dashboard)
    can migrate to rank-based selection without absolute thresholds.
    No behavior change in this PR — fields are observability-only until
    consumers wire to them.

    Semantics:

    - ``cross_sectional_rank``: 1-indexed integer where 1 = highest
      ``final_score`` in the run's scored population. Ties get the same
      rank via ``min`` semantics (e.g., scores [80, 75, 75, 60] →
      ranks [1, 2, 2, 4]). Stable across runs only if scores are
      deterministic — see typed-state arc 2026-04-30 for reducer
      ordering guarantees.
    - ``percentile``: float in ``[0.0, 1.0]`` where 1.0 = top. Computed
      as ``(n - rank) / max(n - 1, 1)`` so rank 1 → 1.0 and last → 0.0.
      Deterministic given ranks. Single-result population emits 1.0.

    Empty results: no-op. Tickers with non-finite ``final_score`` are
    sorted to the end (lowest rank, percentile 0.0) — defensive
    handling; the [0, 100] clamp upstream should make this unreachable
    in practice.

    Why ``final_score`` (not ``weighted_base``): emits the rank of the
    CURRENT pipeline output so downstream consumers see ranks that
    match the current rating/score they already read. The future
    restructure (PR 2+) may shift to ranking ``weighted_base`` so
    macro_shift can move to a sector-allocation layer rather than a
    per-stock score lever — but that's a behavior change, deferred.
    """
    n = len(results)
    if n == 0:
        return

    # Sort tickers by final_score descending; non-finite scores sink to
    # the bottom. ``-math.inf`` sentinel keeps the sort stable + deterministic.
    import math

    def _key(item: tuple[str, dict]) -> float:
        score = item[1].get("final_score")
        try:
            v = float(score) if score is not None else -math.inf
            return v if math.isfinite(v) else -math.inf
        except (TypeError, ValueError):
            return -math.inf

    sorted_items = sorted(results.items(), key=_key, reverse=True)

    # Assign ranks with ``min`` semantics for ties: scan in order and
    # bump the rank counter when the score strictly drops. First entry
    # gets rank 1; ties to its score also get rank 1.
    prev_score: float | None = None
    rank = 0
    seq = 0  # 1-indexed sequence position; rank only advances on score drop
    for ticker, result in sorted_items:
        seq += 1
        score = result.get("final_score")
        try:
            score_val: float | None = float(score) if score is not None else None
            # Coerce NaN/inf to None so the tie-detection branch below
            # (``score_val < prev_score``) doesn't return False on NaN
            # and silently bucket non-finite tickers into the prior
            # tie group.
            if score_val is not None and not math.isfinite(score_val):
                score_val = None
        except (TypeError, ValueError):
            score_val = None

        if rank == 0:
            rank = 1  # first entry
        elif prev_score is None or score_val is None:
            rank = seq  # non-finite handling: don't tie with valid scores
        elif score_val < prev_score:
            rank = seq  # advanced past prior tie group

        # Percentile: rank 1 → 1.0, rank n → 0.0. n=1 case returns 1.0.
        percentile = 1.0 - (rank - 1) / (n - 1) if n > 1 else 1.0

        result["cross_sectional_rank"] = rank
        result["percentile"] = round(percentile, 4)
        prev_score = score_val
