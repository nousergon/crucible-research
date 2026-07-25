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
      "scanner_eval_log": [...],    // per-ticker gate verdict (config#1458)
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
from datetime import UTC, datetime
from typing import Any

import boto3

logger = logging.getLogger(__name__)


SCANNER_VERSION = "v1.0"
_DEFAULT_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
_CANDIDATES_PREFIX = "candidates"
# Champion/challenger OBSERVE substrate (config#1221): challenger candidate-gen
# builds are emitted here, parallel to the live candidates/ path, never consumed
# by live trading until manually promoted.
_SHADOW_PREFIX = "candidates_shadow"
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
        read_latest_daily_closes,
        read_latest_features,
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


def _json_safe_scalar(value: Any) -> Any:
    """Cast a single eval-log value to a plain JSON-serializable scalar.

    ``data/scanner.py``'s ``_eval_log`` entries are built from feature-store
    dicts (already plain ``float``/``str``/``bool``/``None`` per
    ``feature_store_reader.read_latest_features``'s ``float(row[col])`` cast)
    or the OHLCV fallback (``compute_technical_indicators``, also
    pre-cast). This is a defensive belt-and-suspenders cast — NOT a fix for a
    known numpy leak — so a future change to either upstream source can't
    silently reintroduce a ``json.dumps`` failure in
    ``write_candidates_artifact``.
    """
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    # numpy scalar types (e.g. np.float64, np.int64) expose .item() to
    # convert to the equivalent plain Python scalar.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    return value


def _json_safe_eval_log(eval_log: list[dict]) -> list[dict]:
    """Return a copy of ``eval_log`` with every value cast JSON-safe (see
    :func:`_json_safe_scalar`) so ``write_candidates_artifact``'s
    ``json.dumps`` can never choke on a stray numpy/pandas scalar."""
    return [
        {k: _json_safe_scalar(v) for k, v in rec.items()}
        for rec in (eval_log or [])
    ]


def _resolved_scanner_params() -> dict:
    """Snapshot the scanner filter thresholds for the ``filters_applied``
    artifact field. Mirrors ``data.scanner.run_quant_filter``'s read of
    ``get_scanner_params()`` so the artifact records the EXACT parameters
    used this cycle (S3-configurable, auto-tuned by backtester)."""
    from config import (
        MAX_ATR_PCT,
        MIN_AVG_VOLUME,
        MIN_PRICE,
        get_scanner_params,
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

    # ``run_quant_filter`` stashes the per-ticker gate verdict (quant_filter_pass
    # / filter_fail_reason / scan_path / liquidity_pass / volatility_pass) on
    # its own ``_last_eval_log`` module attribute as a side effect — but that
    # stash is process-local. The Research Lambda runs `run_quant_filter` in a
    # SEPARATE process (or doesn't call it at all post-L1995-Phase5) and reads
    # only this artifact via ``am.load_candidates_json``, so the eval log must
    # be carried across that process boundary through ``candidates.json``
    # itself rather than relying on the module-attribute stash (config#1458 /
    # alpha-engine-config#1458 — the stash read in Research's archive_writer
    # was always empty by construction, producing quant_filter_pass=0 for
    # 100% of rows every cycle post-PR#344).
    eval_log = _json_safe_eval_log(
        getattr(run_quant_filter, "_last_eval_log", None) or []
    )

    # ── 5. Build artifact ─────────────────────────────────────────────────
    # Population = prior cycle's holdings list. Phase 1 reads it from the
    # prior signals.json::population; Phase 5 cutover will source it from
    # archive.manager.load_population directly. Empty list on cold-start.
    population_tickers = list(prior_population)
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
        "generated_at": datetime.now(UTC).isoformat(),
        "population_tickers": population_tickers,
        "scanner_tickers": scanner_tickers,
        "agent_input_set": agent_input_set,
        "scanner_eval_log": eval_log,
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


def build_shadow_candidate_artifacts(live_artifact: dict) -> dict[str, dict]:
    """Build the champion/challenger SHADOW candidate artifacts (config#1221).

    Must be called immediately after :func:`build_candidates_artifact` for the
    same cycle — it reads the per-ticker gate decisions that
    ``run_quant_filter`` stashed on ``_last_eval_log`` (so the hard gates are
    held constant across specs with zero gate duplication) and the
    cross-sectional ``*_zscore`` factor loadings. Fully fail-soft: any missing
    input (no eval log, no loadings) yields ``{}`` rather than raising — the
    shadow substrate must NEVER jeopardize the live candidates.json.
    """
    from data.fetchers.feature_store_reader import read_latest_factor_loadings
    from data.scanner import run_quant_filter
    from data.scanner_specs import build_shadow_artifacts

    eval_log = getattr(run_quant_filter, "_last_eval_log", None)
    if not eval_log:
        logger.warning(
            "[scanner_orchestrator] no scanner eval log available — skipping "
            "shadow candidate-gen specs (live artifact unaffected)",
        )
        return {}
    factor_loadings = read_latest_factor_loadings()
    if not factor_loadings:
        logger.warning(
            "[scanner_orchestrator] factor loadings unavailable — skipping "
            "shadow candidate-gen specs (live artifact unaffected)",
        )
        return {}
    params = _resolved_scanner_params()
    return build_shadow_artifacts(live_artifact, eval_log, factor_loadings, params)


def build_scanner_eval_rows_for_board(
    eval_log: list[dict],
    focus_lookup: dict[str, dict],
    run_date: str,
) -> list[dict]:
    """Project the candidates-artifact ``scanner_eval_log`` (already the
    AUTHORITATIVE per-ticker scanner verdict — ``run_quant_filter``'s
    ``_last_eval_log``, JSON-safe cast — with fields ``ticker, sector,
    tech_score, rsi_14, current_price, avg_volume_20d, price_vs_ma200,
    atr_pct, scan_path, quant_filter_pass, filter_fail_reason,
    liquidity_pass, volatility_pass``) into the ``scanner_evaluations``-row
    shape ``scoring.universe_board.build_universe_board`` consumes.

    This is the Scanner-path equivalent of
    ``graph.research_graph._build_scanner_eval_rows`` — same eval_log
    source, same row shape — minus the agent-only ``extra_override_tickers``
    union (there is no agent run in this Lambda, so every row comes
    straight from the scanner's own ~900-ticker universe pass).

    Adds ``eval_date`` and merges in the pure-quant focus-list audit fields
    from ``focus_lookup`` (see
    ``scoring.focus_list.build_pure_quant_focus_lookup``). Tickers absent
    from ``focus_lookup`` (factor-blend disabled, or the substrate wasn't
    available this cycle) keep their board-schema-tolerated null
    ``focus_score``/``focus_stance`` — never fabricated. Agent-only fields
    (``agent_override``/``override_team_id`` when an agent actually looked a
    ticker up outside its focus list) have no equivalent here and always
    degrade to ``0``/``None`` via the shared lookup builder.
    """
    rows = []
    for rec in eval_log:
        ticker = rec.get("ticker")
        if not ticker:
            continue
        row = {**rec, "eval_date": run_date}
        fl_entry = focus_lookup.get(ticker)
        if fl_entry is not None:
            row.update(fl_entry)
        rows.append(row)
    return rows


def write_universe_board_for_scanner_run(
    artifact: dict,
    *,
    market_regime: str = "neutral",
    s3_client: Any | None = None,
    bucket: str = _DEFAULT_BUCKET,
) -> str:
    """Standalone-Scanner path's universe-board write (alpha-engine-config-I2515,
    completing L1995 Phase 5's producer side — the Research graph's
    ``archive_writer`` has been the SOLE producer of ``scanner/universe/``
    until now).

    DUAL-WRITE TRANSITION STATE: the Research graph ALSO writes this board
    (``graph/research_graph.py::archive_writer`` →
    ``scoring.universe_board.compute_and_write_universe_board``) and will
    keep doing so until the SF cutover retires the graph's internal scanner
    (S3 contract safety — both producers coexist during the migration).
    Same-day overwrite by whichever producer runs LAST is expected and
    idempotent-ish; the two producers' rows can differ on agent-audit
    fields (``focus_*``/``agent_override``) since only the Research graph
    ever has a real agent run backing those fields. See alpha-engine-config-I2515.

    Sequencing (factor-profiles ordering, resolved): ``build_universe_board``
    reads the pillar substrate from ``factors/profiles/{run_date}/by_ticker.json``
    (``scoring.universe_board._read_factor_profiles``). That substrate is
    produced by ``scoring.factor_scoring.compute_and_write_factor_profiles``,
    which needs only ``run_date`` + ``sector_map`` and reads
    ``features/{run_date}/{technical,fundamental}.parquet`` — DataPhase1
    outputs already required upstream of the Scanner SF state (the same
    feature store this module's own ``build_candidates_artifact`` reads) —
    i.e. it has NO graph-only input. So this function produces it directly
    here rather than waiting on the Research graph's
    ``compute_factor_profiles_node``. This call is best-effort: on failure
    ``build_universe_board`` already WARN-degrades pillar/attractiveness
    fields to null (``scoring/universe_board.py::_read_factor_profiles``),
    so a hiccup here degrades board content, not availability.

    Best-effort secondary observability overall, mirroring the shadow-
    artifact / leaderboard blocks in ``lambda/scanner_handler.py``: this
    function's caller must wrap it in a fail-soft try/except — a board
    failure must NEVER fail the Scanner Lambda's primary deliverable
    (candidates.json, already written by the time the caller reaches this).

    Returns the dated S3 key written.
    """
    from data.fetchers.price_fetcher import fetch_sp500_sp400_with_sectors
    from scoring.factor_scoring import compute_and_write_factor_profiles
    from scoring.focus_list import build_pure_quant_focus_lookup
    from scoring.universe_board import compute_and_write_universe_board

    s3 = s3_client if s3_client is not None else boto3.client("s3")
    run_date = artifact["run_date"]

    # sector_map isn't threaded through the candidates artifact (it's a
    # ~900-entry map, out of scope for that contract) — re-derive it from
    # the same S3-backed constituents source build_candidates_artifact just
    # read. Single cheap S3 GET; constituents.json cannot change mid-invocation.
    _, sector_map = fetch_sp500_sp400_with_sectors()

    try:
        compute_and_write_factor_profiles(
            run_date=run_date, sector_map=sector_map, bucket=bucket,
        )
    except Exception as exc:  # noqa: BLE001 — board-support only, see docstring
        logger.warning(
            "[scanner_orchestrator] factor-profile production failed for "
            "board support (non-fatal — board pillars degrade to null via "
            "build_universe_board's own fail-soft read): %s", exc,
        )

    focus_lookup = build_pure_quant_focus_lookup(
        market_regime=market_regime, run_date=run_date, bucket=bucket,
    )
    # artifact["scanner_eval_log"] is already JSON-safe-cast by
    # build_candidates_artifact (_json_safe_eval_log) — no re-cast needed.
    scanner_evals = build_scanner_eval_rows_for_board(
        artifact.get("scanner_eval_log") or [], focus_lookup, run_date,
    )
    return compute_and_write_universe_board(
        run_date, scanner_evals, bucket=bucket, s3_client=s3,
    )


def write_shadow_candidates_artifact(
    artifact: dict,
    spec_name: str,
    *,
    s3_client: Any | None = None,
    bucket: str = _DEFAULT_BUCKET,
) -> str:
    """Persist a shadow spec artifact to
    ``s3://{bucket}/candidates_shadow/{spec_name}/{run_date}/candidates.json``
    and return the S3 key. Parallel to :func:`write_candidates_artifact` but on
    the isolated shadow prefix — never read by live trading."""
    s3 = s3_client if s3_client is not None else boto3.client("s3")
    run_date = artifact["run_date"]
    key = f"{_SHADOW_PREFIX}/{spec_name}/{run_date}/candidates.json"
    body = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=io.BytesIO(body).getvalue(),
        ContentType="application/json",
    )
    logger.info(
        "[scanner_orchestrator] wrote SHADOW artifact: s3://%s/%s "
        "(spec=%s scanner_tickers=%d)",
        bucket, key, spec_name, len(artifact["scanner_tickers"]),
    )
    return key
