"""
Scanner orchestrator — single-call wrapper producing the
``candidates.json`` artifact contract for a run_date.

ROADMAP L1995 Phase 1 (plan doc:
~/Development/alpha-engine-docs/private/scanner-rag-resequence-260524.md).

Builds the artifact specified in §3 of the plan doc:

    {
      "run_date": ...,
      "scanner_version": ...,
      "generated_at": ...,
      "population_tickers": [...],  // current holdings + grandfathered
      "scanner_tickers": [...],     // quant-filtered top picks
      "agent_input_set": [...],     // union — what agents will evaluate
      "filters_applied": {...},
      "stats": {
        "universe_size": N,
        "post_scanner": M,
        "new_vs_prior_cycle": [...],
        "dropped_vs_prior_cycle": [...],
      },
    }

Phase 1 posture (observe-only): this module is invoked ONLY by
``lambda/scanner_handler.py`` and writes ``candidates.json`` to S3 for
parallel-observe comparison against Research Lambda's internal scanner
output. Research's internal scanner is unchanged in Phase 1 — the
divergence between this module and ``graph/research_graph.fetch_data_node``
is the load-bearing signal of the Phase 3 soak. Phase 5 (later) cuts
Research over to read this module's artifact + retires the internal
scanner.

Composition reuses the production scanner primitives via direct import:
``fetch_sp500_sp400_with_sectors``, ``run_quant_filter``,
``compute_technical_score``, the feature-store readers. The orchestration
layer is intentionally a thin wrapper so the underlying numerical
behavior matches Research's internal scanner byte-for-byte.

The "prior cycle" diff (new_vs_prior_cycle / dropped_vs_prior_cycle)
reads the prior week's ``signals/{prior_date}/signals.json`` via the
``signals/latest.json`` pointer; the prior cycle's scanner picks live in
``signals.json::universe`` (minus its ``population`` set). When no prior
signals.json exists (very first cycle), both diff fields are empty and
``stats.new_vs_prior_cycle_baseline_missing: true`` flags the cold-start
case loudly per [[feedback_no_silent_fails]].
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)


SCANNER_VERSION = "v1.0"
_DEFAULT_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
_CANDIDATES_PREFIX = "candidates"
_SIGNALS_LATEST_KEY = "signals/latest.json"


class ScannerOrchestratorError(RuntimeError):
    """Raised when the orchestrator cannot produce a valid candidates artifact."""


def _read_prior_signals_universe_tickers(
    s3_client: Any, bucket: str,
) -> tuple[list[str], list[str], str | None]:
    """Read the prior week's ``signals/latest.json`` pointer + the
    ``signals.json`` it points at, and return
    ``(prior_population, prior_scanner_picks, prior_run_date)`` where
    ``prior_scanner_picks`` is ``universe - population``.

    Returns ``([], [], None)`` on any S3 miss (no prior signals.json
    yet — cold-start case is flagged in the artifact stats). Does NOT
    raise on missing-pointer; a brand-new bucket should produce a
    valid artifact with empty diffs.
    """
    try:
        ptr_obj = s3_client.get_object(Bucket=bucket, Key=_SIGNALS_LATEST_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[scanner_orchestrator] signals/latest.json unreadable "
            "(cold-start case — diff will be empty): %s", exc,
        )
        return [], [], None

    try:
        pointer = json.loads(ptr_obj["Body"].read())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[scanner_orchestrator] signals/latest.json malformed JSON: %s",
            exc,
        )
        return [], [], None

    prior_key = pointer.get("s3_key") or pointer.get("key")
    prior_date = pointer.get("date") or pointer.get("run_date")
    if not prior_key:
        logger.warning(
            "[scanner_orchestrator] signals/latest.json has no s3_key "
            "or key field: %r", pointer,
        )
        return [], [], None

    try:
        body = s3_client.get_object(Bucket=bucket, Key=prior_key)
        prior_signals = json.loads(body["Body"].read())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[scanner_orchestrator] prior signals at s3://%s/%s unreadable: %s",
            bucket, prior_key, exc,
        )
        return [], [], None

    prior_population = list(prior_signals.get("population") or [])
    # universe in signals.json is a list of dicts; ticker key may be
    # at top level or nested. Normalize to a flat ticker list.
    universe_raw = prior_signals.get("universe") or []
    if universe_raw and isinstance(universe_raw[0], dict):
        prior_universe_tickers = [
            u.get("ticker") for u in universe_raw if u.get("ticker")
        ]
    else:
        prior_universe_tickers = list(universe_raw)

    # Scanner picks = universe - population (population is holdings; the
    # rest of universe is what the prior scanner promoted).
    pop_set = set(prior_population)
    prior_scanner_picks = [t for t in prior_universe_tickers if t not in pop_set]

    return prior_population, prior_scanner_picks, prior_date


def _build_technical_scores_from_feature_store(
    constituents: list[str], sector_map: dict[str, str],
) -> tuple[dict[str, dict], int]:
    """Load technical scores via the feature-store reader. Returns
    ``(technical_scores, n_enriched)``. Tickers missing from the feature
    store are absent from the returned dict — the caller's scanner pass
    will mark them ``no_data`` per its existing contract.

    Phase 1 deliberately skips the ArcticDB OHLCV fallback that
    ``fetch_data_node`` uses for feature-store misses. The feature store
    has covered the full universe since the 2026-04-14 cutover; if any
    ticker is missing, the divergence will surface in the Phase 3 soak
    and the fallback can be added then. Per
    [[feedback_no_silent_fails]] the miss is loudly logged.
    """
    from data.fetchers.feature_store_reader import (
        read_latest_features,
        read_latest_daily_closes,
    )
    from scoring.technical import compute_technical_score

    fs_features = read_latest_features() or {}
    if not fs_features:
        logger.error(
            "[scanner_orchestrator] feature store empty — scanner cannot "
            "compute technical scores. Saturday DataPhase1 must run before "
            "Scanner."
        )
        raise ScannerOrchestratorError(
            "feature store empty — upstream DataPhase1 did not produce "
            "feature_list.json or the file is unreadable"
        )

    daily_closes = read_latest_daily_closes() or {}

    technical_scores: dict[str, dict] = {}
    n_enriched = 0
    for ticker in constituents:
        fs_row = fs_features.get(ticker)
        if fs_row is None:
            continue
        indicators = {
            "rsi_14": fs_row.get("rsi_14", 50.0),
            "macd_cross": fs_row.get("macd_cross", 0.0),
            "macd_above_zero": bool(fs_row.get("macd_above_zero", False)),
            "macd_line_last": fs_row.get("macd_line_last", 0.0),
            "signal_line_last": 0.0,
            "current_price": daily_closes.get(ticker, 0.0),
            "ma50": None,
            "ma200": None,
            "price_vs_ma50": fs_row.get("price_vs_ma50"),
            "price_vs_ma200": fs_row.get("price_vs_ma200"),
            "momentum_20d": fs_row.get("momentum_20d"),
            "momentum_5d": fs_row.get("momentum_5d"),
            # Scanner's MIN_AVG_VOLUME gate expects raw shares; the bare
            # avg_volume_20d column is predictor-only normalized ratio.
            # avg_volume_20d_raw was added in alpha-engine-data Phase 1
            # of the schema audit. See alpha-engine-data/features/SCHEMA.md.
            "avg_volume_20d": fs_row.get("avg_volume_20d_raw"),
            "atr_14_pct": fs_row.get("atr_14_pct"),
            "dist_from_52w_high": fs_row.get("dist_from_52w_high"),
            "dist_from_52w_low": fs_row.get("dist_from_52w_low"),
        }
        ts = compute_technical_score(indicators, sector=sector_map.get(ticker))
        technical_scores[ticker] = {**indicators, "technical_score": ts}
        n_enriched += 1

    n_missing = len(constituents) - n_enriched
    if n_missing > 0:
        logger.warning(
            "[scanner_orchestrator] feature store missing %d / %d "
            "constituents — these will be tagged 'no_data' by the scanner "
            "and excluded from candidates",
            n_missing, len(constituents),
        )
    return technical_scores, n_enriched


def _resolved_scanner_params() -> dict:
    """Snapshot the scanner filter thresholds for the ``filters_applied``
    artifact field. Mirrors ``data.scanner.run_quant_filter``'s read of
    ``get_scanner_params()`` so the artifact records the EXACT parameters
    used this cycle (S3-configurable, auto-tuned by backtester)."""
    from config import (
        MIN_AVG_VOLUME, MIN_PRICE, MAX_ATR_PCT, get_scanner_params,
    )

    sp = get_scanner_params()
    return {
        "min_avg_volume": sp.get("min_avg_volume", MIN_AVG_VOLUME),
        "min_price": sp.get("min_price", MIN_PRICE),
        "max_atr_pct": sp.get("max_atr_pct", MAX_ATR_PCT),
        "tech_score_min": sp.get("tech_score_min", 60),
        "momentum_top_n": sp.get("momentum_top_n"),
        "min_combined_candidates": sp.get("min_combined_candidates"),
    }


def build_candidates_artifact(
    run_date: str,
    *,
    s3_client: Any | None = None,
    bucket: str = _DEFAULT_BUCKET,
    market_regime: str = "neutral",
) -> dict:
    """Build the candidates.json artifact dict for ``run_date``.

    Does NOT write to S3 — caller (Lambda handler) handles persistence.

    Raises ``ScannerOrchestratorError`` only when upstream data is
    structurally missing (constituents.json absent, feature store empty).
    Other failure modes (prior signals.json missing, single-ticker
    technical-score gaps) are loudly logged but produce a valid artifact
    with empty diffs or smaller candidate count.
    """
    from data.fetchers.price_fetcher import fetch_sp500_sp400_with_sectors
    from data.scanner import run_quant_filter

    s3 = s3_client if s3_client is not None else boto3.client("s3")

    # ── 1. Constituents (~903 S&P 500+400 tickers + sector map) ──────────
    constituents, sector_map = fetch_sp500_sp400_with_sectors()
    if len(constituents) < 800:
        raise ScannerOrchestratorError(
            f"constituents.json has {len(constituents)} tickers — "
            f"refusing to scan (expected >= 800 for S&P 500+400)"
        )

    # ── 2. Prior cycle: population + scanner picks for diff ──────────────
    prior_population, prior_scanner_picks, prior_run_date = (
        _read_prior_signals_universe_tickers(s3, bucket)
    )
    baseline_missing = prior_run_date is None

    # ── 3. Technical scores via feature store ─────────────────────────────
    technical_scores, n_enriched = _build_technical_scores_from_feature_store(
        constituents, sector_map,
    )

    # ── 4. Quant filter — same code Research uses internally ─────────────
    candidate_dicts = run_quant_filter(
        tickers=constituents,
        price_data={},
        technical_scores=technical_scores,
        market_regime=market_regime,
        sector_map=sector_map,
    )
    scanner_tickers = [c["ticker"] for c in candidate_dicts]

    # ── 5. Build artifact ─────────────────────────────────────────────────
    # Population = prior cycle's holdings list. Phase 1 reads it from the
    # prior signals.json::population; Phase 5 cutover will source it from
    # archive.manager.load_population directly. Empty list on cold-start.
    population_tickers = list(prior_population)
    pop_set = set(population_tickers)
    # agent_input_set = population ∪ top-50 scanner picks (the Research
    # Lambda's existing convention at research_graph.py:734).
    agent_input_set = list(
        dict.fromkeys(population_tickers + scanner_tickers[:50])
    )

    # Diff vs prior cycle's scanner picks (the operationally interesting
    # diff for RAGIngestion's new-corpus-fetch decisions). On
    # baseline_missing (cold-start), the diff is meaningless — every
    # ticker would land in "new", which is not the consumer's intended
    # signal. RAGIngestion on cold-start should ingest the FULL
    # agent_input_set, not "just the new ones," so we leave the diff
    # fields empty and the baseline_missing flag carries the cold-start
    # signal explicitly.
    if baseline_missing:
        new_vs_prior_cycle: list[str] = []
        dropped_vs_prior_cycle: list[str] = []
    else:
        prior_scanner_set = set(prior_scanner_picks)
        cur_scanner_set = set(scanner_tickers)
        new_vs_prior_cycle = sorted(cur_scanner_set - prior_scanner_set)
        dropped_vs_prior_cycle = sorted(prior_scanner_set - cur_scanner_set)

    artifact = {
        "run_date": run_date,
        "scanner_version": SCANNER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population_tickers": population_tickers,
        "scanner_tickers": scanner_tickers,
        "agent_input_set": agent_input_set,
        "filters_applied": _resolved_scanner_params(),
        "stats": {
            "universe_size": len(constituents),
            "post_scanner": len(scanner_tickers),
            "population_size": len(population_tickers),
            "agent_input_size": len(agent_input_set),
            "feature_store_enriched": n_enriched,
            "feature_store_missing": len(constituents) - n_enriched,
            "new_vs_prior_cycle": new_vs_prior_cycle,
            "dropped_vs_prior_cycle": dropped_vs_prior_cycle,
            "prior_run_date": prior_run_date,
            "baseline_missing": baseline_missing,
        },
    }
    return artifact


def write_candidates_artifact(
    artifact: dict,
    *,
    s3_client: Any | None = None,
    bucket: str = _DEFAULT_BUCKET,
) -> str:
    """Persist the artifact to
    ``s3://{bucket}/candidates/{run_date}/candidates.json`` and return
    the S3 key."""
    s3 = s3_client if s3_client is not None else boto3.client("s3")
    run_date = artifact["run_date"]
    key = f"{_CANDIDATES_PREFIX}/{run_date}/candidates.json"
    body = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=io.BytesIO(body).getvalue(),
        ContentType="application/json",
    )
    logger.info(
        "[scanner_orchestrator] wrote artifact: s3://%s/%s "
        "(scanner_tickers=%d population=%d agent_input=%d new=%d dropped=%d)",
        bucket, key,
        len(artifact["scanner_tickers"]),
        len(artifact["population_tickers"]),
        len(artifact["agent_input_set"]),
        len(artifact["stats"]["new_vs_prior_cycle"]),
        len(artifact["stats"]["dropped_vs_prior_cycle"]),
    )
    return key
