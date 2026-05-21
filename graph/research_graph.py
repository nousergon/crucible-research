"""
Research Graph — Sector-Team Architecture with serial macro + LangGraph
Send() fan-out to sectors.

Topology (regime-v3 Stage C, 2026-05-14):
  fetch_data
  → load_regime_substrate_node (Stage C — S3 GET on regime/latest.json;
                                None when unavailable, gracefully)
  → macro_economist_node       (serial — substrate-as-strong-prior in the
                                ReAct prompt; remains FINAL regime authority.
                                Writes market_regime + sector_modifiers +
                                sector_ratings to state for sectors to read.)
  → compute_factor_profiles_node (un-orphan: writes factors/profiles/* from
                                features/{run_date}/* so compute_focus_list_node
                                + score_aggregator read a populated substrate
                                this run; graceful-degrade, never hard-fails)
  → compute_focus_list_node    (per-team regime-blended focus list; reads
                                factors/profiles from S3)
  → dispatch_sectors_and_exit  (Send: 6 sector teams + exit evaluator — parallel)
  → merge_results              (fan-in: team picks + exits → compute open slots)
  → score_aggregator           (composite scores for team recommendations)
  → cio_node                   (single Sonnet batch: evaluate all picks, select top N)
  → population_entry_handler
  → consolidator_node
  → archive_writer
  → email_sender_node
  → END

Why macro is serial (was: parallel with sectors + exit):
  Send() snapshots state at dispatch time. Pre-Stage-B sector teams
  received ``market_regime="neutral"`` regardless of what macro
  eventually classified — macro hadn't yet written to state when
  Send() captured it. Serializing macro upstream means sector teams
  see the real regime + sector_modifiers + sector_ratings in their
  ReAct context. Trade-off: +1-2 min Saturday SF wall-clock for
  structurally-guaranteed regime awareness across all sector LLM
  analyses.

Multi-agent patterns:
  - 6 sector teams: quant (ReAct) → qual (ReAct) → peer review → 2-3 recommendations
  - Macro economist: regime + sector ratings with reflection loop, runs SERIALLY before sectors
  - CIO: batch evaluation on 4 dimensions (team conviction, macro alignment, portfolio fit, catalyst specificity)
  - Thesis maintenance: triggered by material events only
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import ValidationError

from config import (
    CIO_MAX_NEW_ENTRANTS,
    CIO_MIN_NEW_ENTRANTS,
    POPULATION_CFG,
    RATING_BUY_THRESHOLD,
    RATING_SELL_THRESHOLD,
    SECTOR_COHERENCE_GATE_ENABLED,
    SECTOR_COHERENCE_UW_MIN_SCORE,
    FACTOR_BLEND_ENABLED,
    FACTOR_BLEND_WEIGHT,
    FACTOR_BLEND_REGIME_WEIGHTS,
    FACTOR_QUALITY_FLOOR_ENABLED,
    FACTOR_QUALITY_FLOOR_MIN_PERCENTILE,
    FACTOR_QUALITY_FLOOR_EXEMPT_SECTORS,
    FOCUS_LIST_DEFAULT_TEAM_SIZE,
    FOCUS_LIST_GATING_ENABLED,
    FOCUS_LIST_PER_TEAM_SIZE_OVERRIDES,
)
from agents.sector_teams.team_config import (
    ALL_TEAM_IDS,
    TEAM_SECTORS,
    SECTOR_TEAM_MAP,
    compute_team_slots,
    get_team_tickers,
)
from agents.sector_teams.sector_team import run_sector_team, SectorTeamContext
from agents.macro_agent import run_macro_agent_with_reflection
from agents.investment_committee.ic_cio import run_cio
from agents.prompt_loader import load_prompt
from data.population_selector import (
    compute_exits_and_open_slots,
    apply_ic_entries,
)
from scoring.composite import (
    compute_composite_breakdown,
    compute_composite_score,
    compute_factor_subscore,
    normalize_conviction,
    score_to_rating,
)
from scoring.factor_scoring import (
    compute_and_write_factor_profiles,
    read_factor_profiles_from_s3,
)
from scoring.focus_list import (
    build_focus_list,
    compute_focus_scores,
    summarize_focus_list,
)
from archive.manager import ArchiveManager

from alpha_engine_lib.decision_capture import (
    DecisionCaptureWriteError,
    FullPromptContext,
    ModelMetadata,
    capture_decision,
)

from graph.reducers import take_last, merge_typed_dicts, reject_on_conflict
from graph.decision_capture_helpers import (
    build_cio_capture_payload,
    build_macro_economist_capture_payload,
    build_sector_peer_review_capture_payload,
    build_sector_qual_capture_payload,
    build_sector_quant_capture_payload,
    build_thesis_update_capture_payload,
    derive_run_id,
    is_decision_capture_enabled,
)
from graph.llm_cost_tracker import pop_metadata_for, track_llm_cost
from graph.state_schemas import (
    CIODecision,
    ExitEvent,
    InvestmentThesis,
    PopulationRotationEvent,
    SectorTeamOutput,
    ThesisUpdate,
)

logger = logging.getLogger(__name__)


# ── Decision-capture helper (gated on ALPHA_ENGINE_DECISION_CAPTURE_ENABLED) ──


# Fallback model names — used only when ``track_llm_cost`` did not run
# before the capture call (e.g. early development of a new agent that
# hasn't been wired through the cost tracker yet). Real token counts +
# cost USD come from the tracker via ``pop_metadata_for(agent_id)``.
_FALLBACK_AGENT_MODEL_NAMES: dict[str, str] = {
    "sector_team": "claude-haiku-4-5",  # per-stock LLM analysis
    "macro_economist": "claude-sonnet-4-6",  # synthesis / regime call
    "ic_cio": "claude-sonnet-4-6",  # cross-stock ranking
}


def _capture_if_enabled(
    *,
    state: ResearchState,
    agent_id: str,
    model_name_key: str,
    input_data_snapshot: dict,
    input_data_summary: str,
    agent_output: dict,
) -> None:
    """Write a DecisionArtifact to S3 if capture is enabled, hard-fail on
    S3 errors per ``feedback_no_silent_fails``. No-op when the env-var
    flag is off (default).

    Reads populated ``ModelMetadata`` + ``FullPromptContext`` from the
    cost tracker (via ``pop_metadata_for``) when ``track_llm_cost`` ran
    before this call. Falls back to a model-name-only stub for paths
    that haven't been wired yet — those captures still land on S3, just
    with token counts at 0 and cost_usd at 0 (the artifact carries enough
    metadata to be repriced later if the call is replayed under the
    replay harness).
    """
    if not is_decision_capture_enabled():
        return

    run_id = derive_run_id(state)

    # Try the populated metadata path first — this is the canonical
    # post-PR-2 source of truth. The tracker handles model-name resolution,
    # token aggregation across multi-call decisions (ReAct + peer review),
    # and cost recompute against the active price card.
    populated = pop_metadata_for(agent_id)
    if populated is not None:
        model_metadata, full_prompt_context = populated
    else:
        # Fallback path — agent not yet wired through the tracker. Logs
        # a warning so dropped capture wiring is visible in CloudWatch
        # and fixed forward; hard-failing here would mean any new agent
        # added without tracker wiring blocks captures system-wide.
        logger.warning(
            "[decision_capture] no cost-tracker metadata for agent_id=%s — "
            "wiring gap. Capturing with model_name-only ModelMetadata "
            "(0 tokens, $0 cost). Wire the call site through track_llm_cost "
            "to close this.",
            agent_id,
        )
        model_metadata = ModelMetadata(
            model_name=_FALLBACK_AGENT_MODEL_NAMES.get(
                model_name_key, "claude-haiku-4-5",
            ),
        )
        full_prompt_context = FullPromptContext(
            system_prompt=f"<see config/prompts/{model_name_key}*.txt at run time; "
                          f"call site not yet wired through track_llm_cost>",
            user_prompt=f"<rendered from input_data_snapshot at run time; "
                        f"call site not yet wired through track_llm_cost>",
        )

    try:
        capture_decision(
            run_id=run_id,
            agent_id=agent_id,
            model_metadata=model_metadata,
            full_prompt_context=full_prompt_context,
            input_data_snapshot=input_data_snapshot,
            input_data_summary=input_data_summary,
            agent_output=agent_output,
        )
    except DecisionCaptureWriteError:
        # Hard-fail per design — capture failures must be loud so the
        # corpus doesn't silently rot. Operators see the SF step fail and
        # investigate (typically IAM or bucket-existence). Re-raising
        # keeps Saturday SF green only when capture is actually working.
        logger.error(
            "[decision_capture] S3 write failed for agent=%s run_id=%s — "
            "raising to fail the run loud (per feedback_no_silent_fails). "
            "Disable via ALPHA_ENGINE_DECISION_CAPTURE_ENABLED=false if "
            "S3/IAM is broken and you need to recover the run.",
            agent_id, run_id,
        )
        raise


# ── Schema validation helper (warn-mode → hard-fail toggle) ──────────────────


from strict_mode import is_strict_validation_enabled as _strict_validation_enabled
# Backward-compat: ``_strict_validation_enabled`` is the previous name in
# this file's API surface. Re-exported via the alias above so existing
# call sites and tests continue to work; new sites should import from
# ``strict_mode`` directly.


def _validate(
    model_cls,
    payload,
    *,
    context: str,
    strict: bool | None = None,
) -> None:
    """Validate ``payload`` against ``model_cls``.

    If ``strict`` is True (or ``STRICT_VALIDATION=true`` in env), raise
    ``RuntimeError`` on validation failure. Otherwise log a warning and
    continue (the warn-mode posture this helper started as).

    The state value itself is unchanged — this helper validates and
    discards the model. Storing the typed model would change runtime
    semantics; the annotation in ``ResearchState`` documents the intended
    shape, but LangGraph's TypedDict reducers see the raw dict either way.

    Renamed from ``_warn_validate`` in PR 2.1 (2026-04-30) to support the
    strict-flip without changing the call signature at the 5 invocation
    sites.
    """
    if strict is None:
        strict = _strict_validation_enabled()
    try:
        model_cls.model_validate(payload)
    except ValidationError as e:
        if strict:
            raise RuntimeError(
                f"[schema-fail:{context}] {model_cls.__name__} validation: {e}"
            ) from e
        logger.warning(
            "[schema-warn:%s] %s schema violation: %s",
            context, model_cls.__name__, e,
        )


# Backward-compatibility alias — to be removed after PR 2.x sequence
# completes. New call sites should use ``_validate``.
_warn_validate = _validate


# ── State Schema ──────────────────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    # ── Core run info ────────────────────────────────────────────────────────
    run_date: Annotated[str, take_last]
    run_time: Annotated[str, take_last]
    archive_manager: Annotated[Any, take_last]
    is_early_close: Annotated[bool, take_last]

    # ── Data (loaded in fetch_data; single-writer, no parallel update) ───────
    price_data: Annotated[dict[str, Any], take_last]
    technical_scores: Annotated[dict[str, dict], take_last]
    scanner_universe: Annotated[list[str], take_last]
    sector_map: Annotated[dict[str, str], take_last]
    macro_data: Annotated[dict, take_last]
    current_population: Annotated[list[dict], take_last]
    population_tickers: Annotated[list[str], take_last]
    prior_theses: Annotated[dict[str, ThesisUpdate], take_last]
    prior_sector_ratings: Annotated[dict[str, dict], take_last]
    predictions: Annotated[dict[str, dict], take_last]
    news_data_by_ticker: Annotated[dict[str, dict], take_last]
    analyst_data_by_ticker: Annotated[dict[str, dict], take_last]
    insider_data_by_ticker: Annotated[dict[str, dict], take_last]
    # Wave 1 PR F: institutional substrate per-ticker dict
    # (news_aggregates + insider_transactions + analyst_revisions
    # joined). Populated only when INSTITUTIONAL_SUBSTRATE_ENABLED=true
    # and the corresponding S3 parquets exist; otherwise empty.
    substrate_by_ticker: Annotated[dict[str, dict], take_last]

    # ── Prior context (memory) ─────────────────────────────────────────────
    prior_macro_report: Annotated[str, take_last]
    prior_macro_snapshots: Annotated[list[dict], take_last]

    # ── Regime substrate (Stage C, 2026-05-14) ─────────────────────────────
    # Quantitative regime substrate produced upstream by the Saturday SF
    # ``RegimeSubstrate`` Lambda (alpha-engine-predictor-regime-substrate).
    # Carries HMM posteriors + composite intensity_z + BOCPD change_signal
    # + guardrail flags + raw macro features. Loaded by
    # ``load_regime_substrate_node`` between fetch_data and macro
    # economist; consumed by macro_economist_node as a strong prior in
    # the agent's ReAct prompt. ``None`` is graceful — macro agent falls
    # back to its prior LLM + post-LLM-guardrail behavior. The macro
    # agent remains the FINAL regime authority either way.
    regime_substrate: Annotated[Optional[dict], take_last]
    episodic_memories: Annotated[dict[str, list], take_last]
    semantic_memories: Annotated[dict[str, list], take_last]

    # ── Sector team outputs (Send fan-out, disjoint team_id keyspace) ───────
    # reject_on_conflict: each Send branch owns a distinct team_id; an overlap
    # would indicate a graph-wiring bug, not a legitimate merge. Replaces the
    # legacy ``_merge_dicts`` (PR #50, 2026-04-29) which was last-write-wins
    # but happened to be safe under disjoint-key invariant.
    sector_team_outputs: Annotated[dict[str, SectorTeamOutput], reject_on_conflict]

    # ── Macro output ─────────────────────────────────────────────────────────
    macro_report: Annotated[str, take_last]
    sector_modifiers: Annotated[dict[str, float], take_last]
    sector_ratings: Annotated[dict[str, dict], take_last]
    market_regime: Annotated[str, take_last]

    # ── Focus list (PR 4 of scanner-placement arc, 260514 plan) ─────────────
    # Per-team regime-blended factor-composite focus list, computed by
    # compute_focus_list_node AFTER macro_economist_node has written
    # market_regime to state. Entries serialized as FocusListEntry.to_dict()
    # so the TypedDict surface stays primitive-only. Used as the quant
    # analyst's primary ranked input when FOCUS_LIST_GATING_ENABLED, and
    # projected onto scanner_evaluations rows by archive_writer for audit.
    focus_list_by_team: Annotated[dict[str, list[dict]], take_last]

    # ── Factor-profile substrate (un-orphan arc) ────────────────────────────
    # Observability-only delta written by compute_factor_profiles_node
    # (spliced macro → compute_focus_list_node). The factor profiles
    # themselves are NOT threaded through state — the consumers
    # (compute_focus_list_node, score_aggregator) read them from
    # s3://.../factors/profiles/latest.json by design; only this small
    # written?/key pair flows for trace/audit. Both default False/"" when
    # production failed (graceful-degrade — consumers skip the blend).
    factor_profiles_written: Annotated[bool, take_last]
    factor_profiles_s3_key: Annotated[str, take_last]

    # ── Exit evaluator output ────────────────────────────────────────────────
    remaining_population: Annotated[list[dict], take_last]
    exits: Annotated[list[ExitEvent], take_last]
    open_slots: Annotated[int, take_last]

    # ── CIO output ───────────────────────────────────────────────────────────
    ic_decisions: Annotated[list[CIODecision], take_last]
    advanced_tickers: Annotated[list[str], take_last]
    entry_theses: Annotated[dict[str, ThesisUpdate], take_last]

    # ── Final population ─────────────────────────────────────────────────────
    new_population: Annotated[list[dict], take_last]
    population_rotation_events: Annotated[list[PopulationRotationEvent], take_last]

    # ── Email ────────────────────────────────────────────────────────────────
    consolidated_report: Annotated[str, take_last]
    email_sent: Annotated[bool, take_last]

    # ── Team slot allocation ─────────────────────────────────────────────────
    team_slot_allocation: Annotated[dict[str, int], take_last]

    # ── Investment theses (computed by score_aggregator) ─────────────────────
    investment_theses: Annotated[dict[str, InvestmentThesis], take_last]

    # ── Dispatch metadata (for Send()) ───────────────────────────────────────
    team_id: Annotated[str, take_last]  # set by Send() per team


# ── Pre-fetch helpers ─────────────────────────────────────────────────────────


def _pre_fetch_held_enrichment(
    population_tickers: list[str],
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Pre-fetch news + analyst data for the held population.

    Returns ``(news_data_by_ticker, analyst_data_by_ticker,
    insider_data_by_ticker)``. The held-stock ``thesis_update`` path
    reads directly from these maps (no agent ReAct loop between
    producer and consumer), so this pre-fetch is the only writer.

    Bug-fix history:

    * **News (pre-2026-05-13):** the call site used
      ``fetch_all_news([ticker])`` — wrong function. ``fetch_all_news``
      takes a single ``str`` and returns a flat
      ``{"yahoo": [...], "edgar_8k": [...]}`` dict; the caller passed a
      list AND treated the return as a batch dict keyed by ticker.
      ``articles.get(ticker, [])`` always returned ``[]`` (no ``ticker``
      key in the flat dict). Silent population-wide news drought.
      Fixed: switched to ``fetch_news_batch(tickers)`` which IS keyed
      by ticker and combines yahoo + edgar into a single ``articles``
      list. ROADMAP P0 surfaced by L83 spot-check 2026-05-13.

    * **Analyst (pre-2026-05-13):** ``analyst_data_by_ticker`` was
      initialized empty and never populated. ``fetch_analyst_consensus``
      exists in ``data/fetchers/analyst_fetcher.py`` but was only
      called from sector_team agent tools. ``thesis_update`` read from
      ``ctx.analyst_data_by_ticker.get(ticker)`` which always returned
      ``None``. Fixed: added the analyst pre-fetch loop. ROADMAP P0
      surfaced by L83 spot-check 2026-05-13.

    Insider data is plumbed but not yet wired — explicit follow-up.
    """
    from data.fetchers.analyst_fetcher import fetch_analyst_consensus
    from data.fetchers.news_fetcher import fetch_news_batch

    news_data_by_ticker: dict[str, dict] = {}
    analyst_data_by_ticker: dict[str, dict] = {}
    insider_data_by_ticker: dict[str, dict] = {}

    try:
        news_batch = fetch_news_batch(population_tickers)
    except Exception as e:
        logger.warning("[fetch_data] news batch fetch failed: %s", e)
        news_batch = {}
    for ticker in population_tickers:
        news = news_batch.get(ticker) or {}
        articles = list(news.get("yahoo") or []) + list(news.get("edgar_8k") or [])
        news_data_by_ticker[ticker] = {
            "articles": articles,
            "article_count": len(articles),
        }

    for ticker in population_tickers:
        try:
            analyst_data_by_ticker[ticker] = fetch_analyst_consensus(ticker)
        except Exception as e:
            logger.debug(
                "[fetch_data] analyst fetch failed for %s: %s", ticker, e
            )

    return news_data_by_ticker, analyst_data_by_ticker, insider_data_by_ticker


def _read_institutional_substrate(
    tickers: list[str], *, run_date: str,
) -> dict[str, dict]:
    """Read the producer-side structured aggregates (Wave 1 PR F).

    Returns a per-ticker dict shape (not the SubstrateSnapshot dataclass)
    so it composes naturally with the existing legacy maps when joined
    into ``ctx.substrate_by_ticker``.

    Reader gates: empty maps when alpha-engine-data hasn't produced the
    parquets yet — agents see None/0 for missing fields. Any read
    exception is caught at the caller; this helper raises on
    configuration errors (missing date format, etc.) so they surface
    loud.
    """
    from datetime import date as _date
    from data.substrate.reader import read_substrate_for_population

    import boto3
    s3 = boto3.client("s3")
    try:
        as_of = _date.fromisoformat(run_date[:10])
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"INSTITUTIONAL_SUBSTRATE_ENABLED=true but run_date "
            f"is not ISO-formatted: {run_date!r}"
        ) from e

    snapshots = read_substrate_for_population(
        tickers, as_of_date=as_of, s3_client=s3,
    )
    out: dict[str, dict] = {}
    for ticker, snap in snapshots.items():
        out[ticker] = {
            # News
            "news_n_articles": snap.news_n_articles,
            "news_n_articles_trusted_weighted": snap.news_n_articles_trusted_weighted,
            "news_n_articles_by_source": snap.news_n_articles_by_source,
            "news_lm_sentiment_mean": snap.news_lm_sentiment_mean,
            "news_lm_sentiment_trusted_mean": snap.news_lm_sentiment_trusted_mean,
            "news_lm_uncertainty_words_total": snap.news_lm_uncertainty_words_total,
            "news_event_count": snap.news_event_count,
            "news_event_severity_max": snap.news_event_severity_max,
            "news_event_categories": list(snap.news_event_categories),
            "news_top_event_descriptions": snap.news_top_event_descriptions,
            # Insider
            "insider_n_transactions_90d": snap.insider_n_transactions_90d,
            "insider_n_buys_90d": snap.insider_n_buys_90d,
            "insider_n_sells_90d": snap.insider_n_sells_90d,
            "insider_net_dollar_flow_90d": snap.insider_net_dollar_flow_90d,
            "insider_distinct_insiders_90d": snap.insider_distinct_insiders_90d,
            "insider_max_single_transaction_usd": snap.insider_max_single_transaction_usd,
            # Analyst revisions
            "analyst_mean_target_current": snap.analyst_mean_target_current,
            "analyst_mean_target_delta_30d": snap.analyst_mean_target_delta_30d,
            "analyst_mean_target_pct_change_30d": snap.analyst_mean_target_pct_change_30d,
            "analyst_num_analysts_current": snap.analyst_num_analysts_current,
            "analyst_num_analysts_delta_30d": snap.analyst_num_analysts_delta_30d,
            "analyst_consensus_rating": snap.analyst_consensus_rating,
            "analyst_rating_changed_30d": snap.analyst_rating_changed_30d,
            # Convenience flags
            "has_news_signal": snap.has_news_signal,
            "has_insider_signal": snap.has_insider_signal,
            "has_analyst_signal": snap.has_analyst_signal,
        }
    return out


# ── Node Functions ────────────────────────────────────────────────────────────

def fetch_data(state: ResearchState) -> dict:
    """Load all shared data needed by sector teams, macro, and exit evaluator."""
    from data.fetchers.price_fetcher import (
        fetch_price_data, fetch_sp500_sp400_with_sectors, compute_technical_indicators,
    )
    from data.fetchers.macro_fetcher import fetch_macro_data, compute_market_breadth
    from scoring.technical import compute_technical_score

    run_date = state["run_date"]
    am: ArchiveManager = state["archive_manager"]

    logger.info("[fetch_data] starting for %s", run_date)

    # RAG availability check (early, so we know before agents start)
    rag_available = False
    try:
        from alpha_engine_lib.rag import is_available as _rag_is_available
        rag_available = _rag_is_available()
        logger.info("[fetch_data] RAG database: %s", "available" if rag_available else "UNAVAILABLE")
        # Reset per-run RAG stats
        from agents.sector_teams.qual_tools import reset_rag_stats
        reset_rag_stats()
    except Exception as e:
        logger.warning("[fetch_data] RAG availability check failed: %s", e)

    # Load S&P 900 universe
    scanner_universe, wikipedia_sector_map = fetch_sp500_sp400_with_sectors()
    logger.info("[fetch_data] %d tickers in S&P 900 universe", len(scanner_universe))

    # Load current population
    current_population = am.load_population()
    population_tickers = [p["ticker"] for p in current_population]

    # Build sector map
    sector_map = dict(wikipedia_sector_map)
    for p in current_population:
        sector_map.setdefault(p["ticker"], p.get("sector", "Unknown"))

    all_tickers = list(set(population_tickers + scanner_universe))

    # ── Feature store first: load pre-computed features for ~900 tickers ─────
    # The predictor's daily inference writes technical + interaction features for
    # the full universe. This eliminates the 3-month yfinance bulk fetch for all
    # tickers that have feature store coverage.
    technical_scores = {}
    _fs_features = {}
    _fs_enriched = 0
    try:
        from data.fetchers.feature_store_reader import read_latest_features
        _fs_features = read_latest_features() or {}
        if _fs_features:
            for ticker, fs_row in _fs_features.items():
                indicators = {
                    "rsi_14": fs_row.get("rsi_14", 50.0),
                    "macd_cross": fs_row.get("macd_cross", 0.0),
                    "macd_above_zero": bool(fs_row.get("macd_above_zero", False)),
                    "macd_line_last": fs_row.get("macd_line_last", 0.0),
                    "signal_line_last": 0.0,
                    "current_price": 0.0,  # filled below from daily_closes or yfinance
                    "ma50": None,
                    "ma200": None,
                    "price_vs_ma50": fs_row.get("price_vs_ma50"),
                    "price_vs_ma200": fs_row.get("price_vs_ma200"),
                    "momentum_20d": fs_row.get("momentum_20d"),
                    "momentum_5d": fs_row.get("momentum_5d"),
                    "avg_volume_20d": fs_row.get("avg_volume_20d"),
                    "atr_14_pct": fs_row.get("atr_14_pct"),
                    "dist_from_52w_high": fs_row.get("dist_from_52w_high"),
                    "dist_from_52w_low": fs_row.get("dist_from_52w_low"),
                }
                ts = compute_technical_score(indicators, sector=sector_map.get(ticker))
                technical_scores[ticker] = {**indicators, "technical_score": ts}
                _fs_enriched += 1
            logger.info("[fetch_data] feature store: %d tickers loaded (skipping yfinance for these)", _fs_enriched)
    except Exception as e:
        logger.debug("[fetch_data] feature store not available: %s", e)

    # ── Load current prices for feature store tickers from daily_closes ──────
    # daily_closes is a single parquet per trading day (~100KB), much cheaper than
    # ~900 yfinance batch calls. Provides current_price for scanner liquidity filter.
    _price_filled = 0
    try:
        from data.fetchers.feature_store_reader import read_latest_daily_closes
        daily_closes = read_latest_daily_closes()
        if daily_closes:
            for ticker in technical_scores:
                if ticker in daily_closes:
                    technical_scores[ticker]["current_price"] = daily_closes[ticker]
                    _price_filled += 1
            logger.info("[fetch_data] daily_closes filled current_price for %d tickers", _price_filled)
    except Exception as e:
        logger.debug("[fetch_data] daily_closes not available: %s", e)

    # ── ArcticDB OHLCV read: skip tickers already covered by feature store ───
    # Population tickers always fetched (agents need raw OHLCV for deep analysis).
    # Scanner universe tickers covered by feature store are SKIPPED — their
    # technical indicators are already populated above.
    _fs_covered = set(_fs_features.keys()) if _fs_features else set()
    ohlcv_tickers = [t for t in all_tickers if t not in _fs_covered or t in population_tickers]
    if len(ohlcv_tickers) < len(all_tickers):
        logger.info(
            "[fetch_data] ArcticDB: reading %d tickers (skipped %d from feature store)",
            len(ohlcv_tickers), len(all_tickers) - len(ohlcv_tickers),
        )
    price_data = fetch_price_data(ohlcv_tickers, period="3mo") if ohlcv_tickers else {}

    # Fill current_price from ArcticDB for tickers that had feature store data
    # but no daily_closes price
    for ticker in technical_scores:
        if technical_scores[ticker]["current_price"] == 0.0:
            if ticker in price_data and price_data[ticker] is not None and not price_data[ticker].empty:
                technical_scores[ticker]["current_price"] = float(price_data[ticker]["Close"].iloc[-1])

    # Fill in technical scores for tickers not covered by feature store
    for ticker, df in price_data.items():
        if ticker in technical_scores:
            continue
        if df is not None and len(df) >= 20:
            indicators = compute_technical_indicators(df)
            ts = compute_technical_score(indicators, sector=sector_map.get(ticker))
            technical_scores[ticker] = {**indicators, "technical_score": ts}

    # ── Macro data ───────────────────────────────────────────────────────────
    macro_data = fetch_macro_data()

    # Market breadth — compute from feature store if available (avoids needing
    # price_data for all ~900 tickers), fall back to price_data computation.
    if _fs_enriched >= 200:
        # Feature store has enough coverage for breadth computation
        above_50d, total_50d = 0, 0
        above_200d, total_200d = 0, 0
        advancers, decliners = 0, 0
        for ticker, ts in technical_scores.items():
            pv50 = ts.get("price_vs_ma50")
            pv200 = ts.get("price_vs_ma200")
            mom5d = ts.get("momentum_5d")
            if pv50 is not None:
                total_50d += 1
                if pv50 > 0:
                    above_50d += 1
            if pv200 is not None:
                total_200d += 1
                if pv200 > 0:
                    above_200d += 1
            if mom5d is not None:
                if mom5d > 0:
                    advancers += 1
                elif mom5d < 0:
                    decliners += 1
        breadth = {
            "pct_above_50d_ma": round(above_50d / total_50d * 100, 1) if total_50d > 0 else None,
            "pct_above_200d_ma": round(above_200d / total_200d * 100, 1) if total_200d > 0 else None,
            "advance_decline_ratio": round(advancers / max(decliners, 1), 2),
            "n_stocks": max(total_50d, total_200d),
        }
        logger.info("[fetch_data] breadth from feature store: %s", breadth)
    else:
        breadth = compute_market_breadth(price_data)
    macro_data.update(breadth)

    # Load prior theses from SQLite (most recent entry per population ticker)
    prior_theses = am.load_latest_theses(population_tickers)

    prior_sector_ratings = state.get("prior_sector_ratings", {})

    # Load prior macro report from S3
    prior_macro_report = ""
    try:
        prior_macro_data = am.load_prior_reports("macro_global", category="macro")
        prior_macro_report = prior_macro_data.get("news_report") or ""
        if prior_macro_report:
            logger.info("[fetch_data] loaded prior macro report (%d chars)", len(prior_macro_report))
    except Exception as e:
        logger.warning("[fetch_data] failed to load prior macro report: %s", e)

    # Load last 3 macro snapshots for structured context
    prior_macro_snapshots = []
    try:
        rows = am.db_conn.execute(
            "SELECT date, market_regime, vix, treasury_10yr, yield_curve_slope, "
            "sp500_30d_return, sector_modifiers, sector_ratings "
            "FROM macro_snapshots ORDER BY date DESC LIMIT 3"
        ).fetchall()
        for r in rows:
            prior_macro_snapshots.append({
                "date": r[0], "market_regime": r[1], "vix": r[2],
                "treasury_10yr": r[3], "yield_curve_slope": r[4],
                "sp500_30d_return": r[5], "sector_modifiers": r[6],
                "sector_ratings": r[7],
            })
    except Exception as e:
        logger.warning("[fetch_data] failed to load macro snapshots: %s", e)

    # Load episodic memories (Phase 2: lessons from failed signals)
    episodic_memories = {}
    try:
        all_sectors = list(set(sector_map.values()))
        episodic_memories = am.load_episodic_memories(
            tickers=population_tickers + scanner_universe[:50],
            sectors=all_sectors,
        )
        if episodic_memories:
            logger.info("[fetch_data] loaded episodic memories for %d tickers", len(episodic_memories))
    except Exception as e:
        logger.debug("[fetch_data] episodic memories not available: %s", e)

    # Load semantic memories (Phase 3: cross-agent observations)
    semantic_memories = {}
    try:
        all_sectors = list(set(sector_map.values()))
        semantic_memories = am.load_semantic_memories(sectors=all_sectors)
        if semantic_memories:
            logger.info("[fetch_data] loaded semantic memories for %d sectors", len(semantic_memories))
    except Exception as e:
        logger.debug("[fetch_data] semantic memories not available: %s", e)

    # Load predictions
    predictions = {}
    try:
        pred_json = am.load_predictions_json()
        if pred_json:
            predictions = pred_json.get("predictions", {})
    except Exception as e:
        logger.debug("[fetch_data] predictions not available: %s", e)

    # Pre-fetch news + analyst data for the held population. These feed
    # ``ctx.news_data_by_ticker`` / ``ctx.analyst_data_by_ticker`` which
    # the conditional held-stock ``thesis_update`` path reads directly
    # (sector_team.py:301). Sector team agents fetch their own data at
    # runtime via tools — this pre-fetch is specifically for thesis_update.
    #
    # Wave 1 PR F (data-revamp-260513.md): when the institutional
    # substrate is enabled, read pre-computed structured aggregates
    # from S3 parquet (written by alpha-engine-data PRs A.2/B/C) and
    # JOIN them onto the legacy per-ticker maps. Substrate fields
    # become a sibling map ``substrate_by_ticker`` so the thesis_update
    # snapshot capture can include them without breaking the legacy
    # news_data/analyst_data shape.
    #
    # Gated behind ``INSTITUTIONAL_SUBSTRATE_ENABLED=true`` (default
    # OFF until alpha-engine-data has soaked the producer pipelines).
    # When ON + parquets exist, the substrate snapshot enriches the
    # input context. When ON + parquets missing, all SubstrateSnapshot
    # fields default to empty — legacy maps still populate.
    news_data_by_ticker, analyst_data_by_ticker, insider_data_by_ticker = (
        _pre_fetch_held_enrichment(population_tickers)
    )
    substrate_by_ticker: dict[str, dict] = {}
    import os as _os
    if _os.environ.get("INSTITUTIONAL_SUBSTRATE_ENABLED", "").lower() == "true":
        try:
            substrate_by_ticker = _read_institutional_substrate(
                population_tickers, run_date=run_date,
            )
            logger.info(
                "[fetch_data] institutional substrate loaded for %d tickers "
                "(%d with news signal, %d with insider signal, "
                "%d with analyst signal)",
                len(substrate_by_ticker),
                sum(1 for s in substrate_by_ticker.values() if s.get("has_news_signal")),
                sum(1 for s in substrate_by_ticker.values() if s.get("has_insider_signal")),
                sum(1 for s in substrate_by_ticker.values() if s.get("has_analyst_signal")),
            )
        except Exception as e:
            logger.warning(
                "[fetch_data] institutional substrate read failed: %s — "
                "falling back to legacy enrichment only", e,
            )

    logger.info("[fetch_data] done — %d prices, %d tech scores, %d population",
                len(price_data), len(technical_scores), len(population_tickers))

    return {
        "scanner_universe": scanner_universe,
        "sector_map": sector_map,
        "price_data": price_data,
        "technical_scores": technical_scores,
        "macro_data": macro_data,
        "current_population": current_population,
        "population_tickers": population_tickers,
        "prior_theses": prior_theses,
        "prior_sector_ratings": prior_sector_ratings,
        "predictions": predictions,
        "news_data_by_ticker": news_data_by_ticker,
        "analyst_data_by_ticker": analyst_data_by_ticker,
        "insider_data_by_ticker": insider_data_by_ticker,
        "substrate_by_ticker": substrate_by_ticker,
        "prior_macro_report": prior_macro_report,
        "prior_macro_snapshots": prior_macro_snapshots,
        "episodic_memories": episodic_memories,
        "semantic_memories": semantic_memories,
    }


def load_regime_substrate_node(state: ResearchState) -> dict:
    """Load the quantitative regime substrate from S3 (regime-v3 Stage C).

    Resolves the canonical eval_artifacts sidecar pointer at
    ``regime/latest.json`` and reads the dated artifact. Returns
    ``{"regime_substrate": None}`` gracefully when:

    - The substrate has never been written (pre-deploy state).
    - The Saturday SF ``RegimeSubstrate`` state's non-blocking Catch
      tripped this week and no fresh artifact was produced.
    - The S3 read fails for transient reasons.

    A ``None`` regime_substrate flows through to ``macro_economist_node``
    and the macro agent falls back to its prior LLM + post-LLM-
    guardrail behavior — Stage C is observe-only at the substrate-
    influences-LLM layer; the macro agent's final regime call is still
    authoritative for downstream consumers either way.
    """
    am = state.get("archive_manager")
    if am is None:
        logger.warning(
            "[load_regime_substrate] no archive_manager in state — "
            "substrate cannot be loaded; macro agent will run without prior",
        )
        return {"regime_substrate": None}

    substrate = am.load_regime_substrate()
    if substrate is None:
        logger.info(
            "[load_regime_substrate] no substrate available; "
            "macro agent will run without quant prior (Stage A pre-deploy "
            "or non-blocking SF Catch tripped upstream)",
        )
        return {"regime_substrate": None}

    hmm_argmax = (substrate.get("hmm") or {}).get("argmax")
    intensity_z = (substrate.get("composite") or {}).get("intensity_z")
    change_signal = (substrate.get("bocpd") or {}).get("change_signal")
    logger.info(
        "[load_regime_substrate] loaded run_id=%s hmm_argmax=%s intensity_z=%+.2f change_signal=%s",
        substrate.get("run_id", "?"),
        hmm_argmax,
        intensity_z if isinstance(intensity_z, (int, float)) else 0.0,
        change_signal,
    )
    return {"regime_substrate": substrate}


def dispatch_sectors_and_exit(state: ResearchState) -> list:
    """
    Fan-out via Send(): launch 6 sector teams + exit evaluator in parallel.

    Macro economist no longer dispatched here — it runs SERIALLY upstream
    of this dispatcher per regime-v3 Stage B. The serialization ensures
    sector teams see the actual ``market_regime`` + ``sector_modifiers``
    + ``sector_ratings`` macro computed, rather than the default
    ``"neutral"`` they would receive if macro ran in parallel and
    finished after Send() snapshotted state. Pre-Stage-B, sector team
    prompts received ``{market_regime}=neutral`` regardless of what
    macro eventually classified — Send() captures state at dispatch.

    Each Send receives a subset of shared state including the macro-
    computed regime context.
    """
    sends = []

    # 6 sector teams — now see macro's actual outputs via state
    for team_id in ALL_TEAM_IDS:
        sends.append(Send("sector_team_node", {
            **state,
            "team_id": team_id,
        }))

    # Exit evaluator — parallel with sectors; independent of regime
    sends.append(Send("exit_evaluator_node", {
        **state,
    }))

    logger.info(
        "[dispatch] sending %d parallel tasks (6 teams + exits) — regime=%s",
        len(sends), state.get("market_regime", "neutral"),
    )
    return sends


def sector_team_node(state: ResearchState) -> dict:
    """Run a single sector team (dispatched via Send).

    Resumability: the Research Lambda runs the graph stateless (no
    checkpointer), so an SF re-invocation re-dispatches all 6 teams.
    Before doing ANY LLM/tool work for this team we check S3 for a
    persisted output for exactly this ``(run_date, team_id)`` — if a
    well-formed one exists (a prior invocation already completed this
    team for this run_date) we short-circuit and feed it straight into
    ``sector_team_outputs``, paying ZERO Haiku calls. That is the
    load-bearing invariant: a re-run must NEVER re-pay a team that
    already succeeded for this run_date. On success below we persist
    this team's output immediately (before any other team can fail and
    ERROR the run) so the next invocation can reuse it.
    """
    team_id = state.get("team_id", "unknown")
    run_date = state["run_date"]
    logger.info("[sector_team:%s] starting", team_id)

    # ── Resume short-circuit ──────────────────────────────────────────────
    # If a prior invocation of THIS run_date already completed this team,
    # reuse the persisted output and skip all LLM/tool work. Best-effort:
    # any failure / staleness in the load path returns None and we run
    # the team fresh (load_sector_team_run never raises).
    _am = state.get("archive_manager")
    if _am is not None:
        try:
            persisted = _am.load_sector_team_run(run_date, team_id)
        except Exception as e:  # pragma: no cover — defensive; loader is safe
            logger.warning(
                "[sector_team:%s] resume load raised unexpectedly (%s) — "
                "running team fresh", team_id, e,
            )
            persisted = None
        if persisted is not None:
            logger.info(
                "[sector_team:%s] RESUME — reusing persisted output "
                "(%d recommendations); zero Haiku calls this invocation",
                team_id,
                len(persisted.get("recommendations", []) or []),
            )
            return {"sector_team_outputs": {team_id: persisted}}

    # Stage D' Wire 1: extract intensity_z from the regime substrate
    # (loaded by load_regime_substrate_node upstream of macro). None
    # when substrate hasn't published yet — peer_review's gate degrades
    # to base threshold only.
    _substrate = state.get("regime_substrate") or {}
    _intensity_z = (_substrate.get("composite") or {}).get("intensity_z")

    ctx = SectorTeamContext(
        scanner_universe=state.get("scanner_universe", []),
        sector_map=state.get("sector_map", {}),
        price_data=state.get("price_data", {}),
        technical_scores=state.get("technical_scores", {}),
        market_regime=state.get("market_regime", "neutral"),
        prior_theses=state.get("prior_theses", {}),
        held_tickers=state.get("population_tickers", []),
        news_data_by_ticker=state.get("news_data_by_ticker", {}),
        analyst_data_by_ticker=state.get("analyst_data_by_ticker", {}),
        insider_data_by_ticker=state.get("insider_data_by_ticker", {}),
        prior_sector_ratings=state.get("prior_sector_ratings", {}),
        current_sector_ratings=state.get("sector_ratings", {}),
        run_date=state["run_date"],
        episodic_memories=state.get("episodic_memories", {}),
        semantic_memories=state.get("semantic_memories", {}),
        regime_intensity_z=_intensity_z,
        focus_list=state.get("focus_list_by_team", {}).get(team_id, []),
    )
    # Cost-telemetry scope spans the whole sector team's LLM activity:
    # quant ReAct + qual ReAct + peer_review (×2) + thesis updates. The
    # CostTelemetryCallback attached to each ChatAnthropic instance
    # accumulates token usage into this single frame; per-call rows are
    # flushed to a JSONL at scope exit (PR 3 cost-raw stream).
    with track_llm_cost(
        agent_id=f"sector_team:{team_id}",
        sector_team_id=team_id,
        node_name="sector_team_node",
        run_type="weekly_research",
        run_id=derive_run_id(state),
    ):
        result = run_sector_team(team_id, ctx)

    # Schema validation — strict-by-default (raises RuntimeError on
    # validation failure unless STRICT_VALIDATION=false).
    _validate(SectorTeamOutput, result, context=f"sector_team:{team_id}")

    # Decision-artifact capture (gated on ALPHA_ENGINE_DECISION_CAPTURE_ENABLED).
    # Per-sub-agent captures so LLM-as-judge eval can score quant + qual
    # independently. Cost telemetry stays aggregated under the outer
    # sector_team:{team_id} track_llm_cost scope above — pop_metadata_for
    # lookups for sector_quant / sector_qual hit the fallback stub today
    # (token counts at 0). A future PR can split the cost-tracker scopes
    # if per-sub-agent cost attribution becomes necessary.
    team_tickers = get_team_tickers(team_id, ctx.scanner_universe, ctx.sector_map)
    quant_output = result.get("quant_output", {}) or {}
    qual_output = result.get("qual_output", {}) or {}

    q_snapshot, q_summary = build_sector_quant_capture_payload(
        team_id, ctx, team_tickers=team_tickers,
    )
    _capture_if_enabled(
        state=state,
        agent_id=f"sector_quant:{team_id}",
        model_name_key="sector_team",
        input_data_snapshot=q_snapshot,
        input_data_summary=q_summary,
        agent_output=quant_output,
    )

    # Reproduce the same top5 slice run_sector_team handed to the qual
    # analyst (sector_team.py:96-109): drop picks without 'ticker', take
    # first 5. Capturing the actual hand-off avoids over-stating qual's
    # input.
    quant_picks = quant_output.get("ranked_picks", []) or []
    valid_picks = [
        p for p in quant_picks if isinstance(p, dict) and "ticker" in p
    ]
    quant_top5 = valid_picks[:5]
    ql_snapshot, ql_summary = build_sector_qual_capture_payload(
        team_id, ctx, quant_top5=quant_top5,
    )
    _capture_if_enabled(
        state=state,
        agent_id=f"sector_qual:{team_id}",
        model_name_key="sector_team",
        input_data_snapshot=ql_snapshot,
        input_data_summary=ql_summary,
        agent_output=qual_output,
    )

    # sector_peer_review capture — the synthesis call that produces the
    # 2-3 recommendations CIO sees. Reconstruct inputs from quant + qual
    # outputs to mirror what run_peer_review actually received.
    peer_output = result.get("peer_review_output", {}) or {}
    qual_assessments = qual_output.get("assessments", []) or []
    qual_additional = qual_output.get("additional_candidate")
    pr_snapshot, pr_summary = build_sector_peer_review_capture_payload(
        team_id, ctx,
        quant_top5=quant_top5,
        qual_assessments=qual_assessments,
        qual_additional_candidate=qual_additional,
    )
    _capture_if_enabled(
        state=state,
        agent_id=f"sector_peer_review:{team_id}",
        model_name_key="sector_team",
        input_data_snapshot=pr_snapshot,
        input_data_summary=pr_summary,
        agent_output=peer_output,
    )

    # held-stock thesis updates — one capture per ticker that actually
    # had triggers fire (i.e. an LLM call was made). The no-trigger path
    # in run_sector_team preserves the prior thesis without calling the
    # LLM, so we skip those — capturing them would pollute the eval
    # corpus with non-LLM artifacts.
    thesis_updates = result.get("thesis_updates", {}) or {}
    for ticker, updated in thesis_updates.items():
        if not isinstance(updated, dict):
            continue
        triggers = updated.get("triggers") or []
        if not triggers:
            continue
        tu_snapshot, tu_summary = build_thesis_update_capture_payload(
            team_id, ticker, ctx, triggers=list(triggers),
        )
        _capture_if_enabled(
            state=state,
            agent_id=f"thesis_update:{team_id}:{ticker}",
            model_name_key="sector_team",
            input_data_snapshot=tu_snapshot,
            input_data_summary=tu_summary,
            agent_output=updated,
        )

    # ── Persist on success (resumability) ─────────────────────────────────
    # Write this team's full output to S3 NOW — before any other team
    # can fail and ERROR the overall run. A team is persisted iff it has
    # no hard ``error`` (a partial/recursion-exhausted team is persisted
    # too: re-running it would just re-burn budget for the same empty
    # result, and the aggregator already tolerates partials). A team
    # that errored is NOT persisted so a re-run gets a fresh attempt at
    # it (the backoff / a TPM-window reset may let it succeed).
    if _am is not None and not result.get("error"):
        try:
            _am.save_sector_team_run(run_date, team_id, result)
        except Exception as e:  # pragma: no cover — saver is already safe
            logger.warning(
                "[sector_team:%s] persist raised unexpectedly (%s) — run "
                "continues; team just won't be resumable", team_id, e,
            )

    # Return partial state update — reject_on_conflict reducer merges team
    # outputs (each team_id is disjoint).
    return {
        "sector_team_outputs": {team_id: result},
    }


def macro_economist_node(state: ResearchState) -> dict:
    """Run the macro economist with reflection.

    ALL-AGENTS-STRICT (2026-05-16): extends #194's sector-team
    persist+resume pattern to macro. On an SF redrive (the Lambda runs
    the graph stateless), if this run_date's macro output was already
    persisted we short-circuit and reuse it with ZERO LLM calls — so a
    redrive triggered by a *different* agent's hard-fail does not
    re-pay the (Sonnet) macro + critic calls. Macro runs upstream of
    the sector dispatch (Stage B), so without this every redrive would
    re-run macro before even reaching the still-missing sector team.
    """
    logger.info("[macro] starting")

    run_date = state.get("run_date", "")
    _am = state.get("archive_manager")
    if _am is not None and run_date:
        try:
            _persisted = _am.load_agent_run(run_date, "macro")
        except Exception as e:  # pragma: no cover — loader is safe
            logger.warning(
                "[macro] resume load raised unexpectedly (%s) — "
                "running macro fresh", e,
            )
            _persisted = None
        if _persisted is not None:
            logger.info(
                "[macro] RESUME — reusing persisted output for %s "
                "(zero LLM calls this invocation)", run_date,
            )
            return _persisted

    macro_data = state.get("macro_data", {})
    prior_report = state.get("prior_macro_report", "")

    # Derive prior date from macro_snapshots
    prior_date = ""
    prior_snapshots = state.get("prior_macro_snapshots", [])
    if prior_snapshots:
        prior_date = prior_snapshots[0].get("date", "")

    if prior_report:
        logger.info("[macro] using prior report from %s (%d chars)", prior_date, len(prior_report))
    else:
        logger.info("[macro] no prior report — generating fresh")

    # Cost-telemetry scope wraps the macro-economist primary call + the
    # macro-critic reflection call as one logical decision. The PRIMARY
    # prompt (macro_agent) stamps prompt_id + prompt_version on
    # ModelMetadata; the critic prompt is a refinement and not the
    # canonical prompt for this decision.
    with track_llm_cost(
        agent_id="macro_economist",
        node_name="macro_economist_node",
        run_type="weekly_research",
        run_id=derive_run_id(state),
        prompt=load_prompt("macro_agent"),
    ):
        result = run_macro_agent_with_reflection(
            prior_report=prior_report,
            prior_date=prior_date,
            macro_data=macro_data,
            prior_snapshots=prior_snapshots,
            # Stage C: load_regime_substrate_node populates this state
            # field with the quant substrate (HMM + composite + BOCPD +
            # guardrails). None when not yet available — the macro
            # agent falls back to its prior LLM + post-LLM-guardrail
            # behavior. Macro agent remains the final regime authority.
            regime_substrate=state.get("regime_substrate"),
        )

    macro_state_update = {
        "macro_report": result.get("report_md", ""),
        "sector_modifiers": result.get("sector_modifiers", {}),
        "sector_ratings": result.get("sector_ratings", {}),
        "market_regime": result.get("market_regime", "neutral"),
    }

    # Decision-artifact capture (gated on ALPHA_ENGINE_DECISION_CAPTURE_ENABLED).
    snapshot, summary = build_macro_economist_capture_payload(state)
    _capture_if_enabled(
        state=state,
        agent_id="macro_economist",
        model_name_key="macro_economist",
        input_data_snapshot=snapshot,
        input_data_summary=summary,
        agent_output=macro_state_update,
    )

    # ── Persist on success (resumability) ─────────────────────────────────
    # We only reach here if the macro agent produced REAL output (a 429
    # past the deadline / a strict-mode parse failure would have raised
    # and hard-failed the run upstream). Persist so an SF redrive
    # triggered by a different agent's failure resumes macro with zero
    # LLM calls. Best-effort — save_agent_run never raises.
    if _am is not None and run_date:
        try:
            _am.save_agent_run(run_date, "macro", macro_state_update)
        except Exception as e:  # pragma: no cover — saver is already safe
            logger.warning(
                "[macro] persist raised unexpectedly (%s) — run "
                "continues; macro just won't be resumable", e,
            )

    return macro_state_update


def compute_factor_profiles_node(state: ResearchState) -> dict:
    """Produce the institutional factor-profile substrate for this run.

    Spliced between ``macro_economist_node`` and ``compute_focus_list_node``
    so it runs strictly AFTER ``fetch_data`` (which populates ``sector_map``
    + ``run_date`` — neither is mutated by load_regime_substrate_node or
    macro_economist_node) and strictly BEFORE both consumers
    (``compute_focus_list_node`` ~:1198 and ``score_aggregator`` ~:1322),
    each of which reads ``factors/profiles/latest.json`` from S3 via
    ``read_factor_profiles_from_s3()``. Writing the substrate here makes
    that read populated within the SAME run instead of the prior
    orphaned state (zero production callers → ``factors/`` empty in prod).

    Reads (produced upstream by alpha-engine-data DataPhase1):
      - s3://{bucket}/features/{run_date}/technical.parquet
      - s3://{bucket}/features/{run_date}/fundamental.parquet
    Writes:
      - s3://{bucket}/factors/profiles/{run_date}/by_ticker.json
      - s3://{bucket}/factors/profiles/latest.json (sidecar)

    Behavior-safety: ``config.FACTOR_BLEND_ENABLED`` and
    ``config.FOCUS_LIST_GATING_ENABLED`` both default False, so producing
    this substrate does NOT change scoring or agent behavior — it only
    lets the focus-list shadow audit populate ``scanner_evaluations.focus_*``
    and makes the factor substrate exist/ready. No flag is flipped here.

    Fail-loud (per feedback_no_silent_fails): this node HARD-FAILS the
    research run if it cannot produce the substrate. Graceful-degrade
    here would silently recreate the exact orphaned-producer class this
    wiring exists to fix (4+ days of inert focus-list / factor-blend that
    nobody noticed). ``features/{run_date}/*.parquet`` is produced by
    DataPhase1 UPSTREAM in the same Saturday SF, so its absence is already
    an incident (DataPhase1 should have failed) — raising here surfaces
    real breakage and cannot spuriously fail a healthy run. Any failure
    (missing run_date, missing/short feature parquets, S3 error, compute
    exception) raises → the Research SF state fails loudly and alerts.

    Returns a small observability delta (``factor_profiles_written`` +
    ``factor_profiles_s3_key``). Profiles are NOT threaded through state —
    the consumers read from S3 by design; that contract is preserved.
    """
    run_date = state.get("run_date", "")
    sector_map = state.get("sector_map", {})

    if not run_date:
        raise RuntimeError(
            "[compute_factor_profiles] no run_date in state — cannot "
            "produce the factor substrate. Hard-failing the research run "
            "(feedback_no_silent_fails) rather than letting focus-list + "
            "factor-blend silently degrade."
        )

    try:
        s3_key = compute_and_write_factor_profiles(
            run_date=run_date,
            sector_map=sector_map,
        )
        logger.info(
            "[compute_factor_profiles] wrote factor substrate for %s "
            "(%d sector-mapped tickers) → s3 key=%s",
            run_date, len(sector_map), s3_key,
        )
        return {"factor_profiles_written": True, "factor_profiles_s3_key": s3_key}
    except Exception as e:
        logger.error(
            "[compute_factor_profiles] factor-profile production FAILED "
            "for %s: %s — features/%s/*.parquet missing/short or an S3 "
            "error. features/ is produced by DataPhase1 upstream in this "
            "same SF, so this is a real incident, not a tolerable degrade. "
            "Hard-failing the research run (feedback_no_silent_fails) "
            "rather than silently recreating the orphaned-producer bug.",
            run_date, e, run_date,
        )
        raise


def compute_focus_list_node(state: ResearchState) -> dict:
    """Build the per-team regime-blended focus list from factor profiles.

    Runs AFTER ``macro_economist_node`` so ``state["market_regime"]`` carries
    the current cycle's regime (Stage B / PR #185 serializes macro upstream
    of sector dispatch — pre-Stage-B this was the prior week's regime).
    Writes ``focus_list_by_team`` to state for ``sector_team_node`` to read.

    Composes with:
      - Phase 1c (factor_scoring.py) — produces factors/profiles/latest.json
      - Phase 3 (compute_factor_subscore) — same regime-conditional blend
        formula (single source of truth — tuning factor_blend tunes focus)
      - PR 1 (scoring/focus_list.py) — top-N per team build logic
      - FOCUS_LIST_GATING_ENABLED — when true, the quant analyst's prompt
        uses this list as primary ranked input; tool calls on non-focus
        tickers are tagged agent_override=1 in the audit table

    Empty result on any of:
      - FACTOR_BLEND_ENABLED is False
      - ``factors/profiles/latest.json`` artifact missing
      - No tickers produced a contributing factor score
    sector_team_node + agent_override telemetry both degrade gracefully
    to "no focus list this cycle" semantics in these cases.
    """
    logger.info("[focus_list] starting")

    if not FACTOR_BLEND_ENABLED:
        logger.info(
            "[focus_list] factor blend disabled — focus_list_by_team empty"
        )
        return {"focus_list_by_team": {}}

    factor_profiles = read_factor_profiles_from_s3()
    if not factor_profiles:
        logger.warning(
            "[focus_list] factor profile artifact missing — focus_list_by_team "
            "empty (agents see full sector slice regardless of gating flag)"
        )
        return {"focus_list_by_team": {}}

    market_regime = state.get("market_regime", "neutral")
    focus_scores = compute_focus_scores(
        factor_profiles, market_regime, FACTOR_BLEND_REGIME_WEIGHTS,
    )
    if not focus_scores:
        logger.warning(
            "[focus_list] no factor scores computed for regime=%s — "
            "focus_list_by_team empty", market_regime,
        )
        return {"focus_list_by_team": {}}

    focus_list = build_focus_list(
        focus_scores,
        SECTOR_TEAM_MAP,
        per_team_size=FOCUS_LIST_PER_TEAM_SIZE_OVERRIDES,
        default_size=FOCUS_LIST_DEFAULT_TEAM_SIZE,
    )
    summary = summarize_focus_list(focus_list)
    logger.info(
        "[focus_list] regime=%s, gating_enabled=%s, summary=%s",
        market_regime, FOCUS_LIST_GATING_ENABLED, summary,
    )

    return {
        "focus_list_by_team": {
            team_id: [e.to_dict() for e in entries]
            for team_id, entries in focus_list.items()
        }
    }


def exit_evaluator_node(state: ResearchState) -> dict:
    """Determine exits from current population using prior theses."""
    logger.info("[exit_evaluator] starting")

    # Build investment_theses from prior_theses for score lookup
    investment_theses = {}
    for ticker, thesis in state.get("prior_theses", {}).items():
        investment_theses[ticker] = {
            "long_term_score": thesis.get("score", 50),
            **thesis,
        }

    # Pass scanner_universe as the constituents whitelist so grandfathered
    # non-S&P tickers get UNIVERSE_DROP exits before score/tenure logic.
    # scanner_universe is built in fetch_data_node and persisted in graph
    # state (line 83). Origin: 2026-04-20 — TSM + ASML surfaced as incumbent
    # outliers causing executor ArcticDB NoSuchVersionException downstream.
    scanner_universe = state.get("scanner_universe", [])
    remaining, exits, open_slots = compute_exits_and_open_slots(
        current_population=state.get("current_population", []),
        investment_theses=investment_theses,
        config=POPULATION_CFG,
        run_date=state.get("run_date"),
        constituents=set(scanner_universe) if scanner_universe else None,
    )

    # Schema validation on per-exit shape (strict-by-default).
    for ev in exits:
        _validate(ExitEvent, ev, context="exit_evaluator")

    return {
        "remaining_population": remaining,
        "exits": exits,
        "open_slots": open_slots,
    }


def merge_results_node(state: ResearchState) -> dict:
    """Fan-in: merge sector team outputs + macro + exits. Compute slot allocation."""
    logger.info("[merge] merging results")

    sector_ratings = state.get("sector_ratings", {})
    open_slots = state.get("open_slots", 0)

    team_slot_allocation = compute_team_slots(open_slots, sector_ratings)

    logger.info("[merge] %d open slots, allocation: %s", open_slots, team_slot_allocation)

    return {
        "team_slot_allocation": team_slot_allocation,
    }


def score_aggregator(state: ResearchState) -> dict:
    """Compute composite scores for all team recommendations.

    ALL-AGENTS-STRICT (Brian, 2026-05-16) — REVERTS #194's per-team
    degrade-and-continue isolation:

      "If the sector agents don't run, Research shouldn't complete
       until all sectors are run. ... We don't get anything from this
       process if the sectors, or any other agent for that matter,
       fail/don't run."

    So this node now HARD-FAILS the whole run (raises → handler returns
    status:ERROR → NO signals.json / email / DB write) if ANY sector
    team is:

      * **missing** — not present in ``sector_team_outputs`` at all
        (the expected set is ``ALL_TEAM_IDS``);
      * **failed** — carries an ``error`` (e.g. a 429 that survived the
        ~75-min deadline-bounded retry, or a held-thesis hard-raise);
      * **partial** — recursion-exhausted / produced a degraded result.

    What is KEPT from #194 (and composes with the directive): every
    team that *succeeds* is still persisted to S3 by ``sector_team_node``
    the moment it completes. So when a single team fails and this node
    hard-fails the run, an SF redrive reuses the persisted succeeded
    teams (zero Haiku calls) and the long retry only ever re-attempts
    the still-missing team(s) — which is exactly what makes a 60-90 min
    retry window affordable and bounded rather than re-paying the whole
    6-team fan-out every redrive.
    """
    logger.info("[score_aggregator] starting")

    team_outputs = state.get("sector_team_outputs", {})
    sector_modifiers = state.get("sector_modifiers", {})
    sector_map = state.get("sector_map", {})

    # ── All-agents-strict gate ────────────────────────────────────────────
    # The full expected set of sector teams. A team absent from
    # team_outputs never produced output at all (dispatch failed, the
    # node crashed before returning, or the reducer dropped it) — that
    # is just as fatal as a team that errored. Empty team_outputs
    # (e.g. the no-op exit-only test states) is left to the original
    # downstream behavior; the strict gate only fires once teams exist.
    expected_team_ids = set(ALL_TEAM_IDS)
    present_team_ids = set(team_outputs)
    missing_teams = (
        sorted(expected_team_ids - present_team_ids)
        if present_team_ids
        else []
    )

    failed_teams = {
        tid: out.get("error")
        for tid, out in team_outputs.items()
        if out.get("error")
    }
    partial_teams = {
        tid: out.get("partial_reasons", [])
        for tid, out in team_outputs.items()
        if out.get("partial")
    }

    if missing_teams or failed_teams or partial_teams:
        parts: list[str] = []
        if missing_teams:
            parts.append(f"missing (never produced output): {missing_teams}")
        if failed_teams:
            parts.append(
                "failed: "
                + "; ".join(
                    f"{tid}: {err}" for tid, err in failed_teams.items()
                )
            )
        if partial_teams:
            parts.append(f"partial: {partial_teams}")
        msg = (
            f"ALL-AGENTS-STRICT: {len(missing_teams)} missing + "
            f"{len(failed_teams)} failed + {len(partial_teams)} partial "
            f"sector team(s) — Research HARD-FAILS (no signals.json / "
            f"email / DB write). We get nothing from a process whose "
            f"agents didn't all run. Succeeded teams are persisted to "
            f"S3, so an SF redrive only re-attempts the still-missing "
            f"team(s) within the long 429-retry window. "
            + " | ".join(parts)
        )
        logger.error("[score_aggregator] %s", msg)
        raise RuntimeError(msg)

    investment_theses = {}
    market_regime = state.get("market_regime", "neutral")

    # Factor blend (Phase 3, 260513 plan): read the per-ticker factor profile
    # once at the top so the recommendation loop stays O(N) S3-call-free. None
    # is returned when no profile artifact exists yet — every per-ticker call
    # site degrades gracefully via the ``factor_subscore=None`` path in
    # ``compute_composite_score``.
    factor_profiles_by_ticker: dict[str, dict] | None = None
    factor_blend_applied_count = 0
    factor_blend_skipped_count = 0
    if FACTOR_BLEND_ENABLED:
        factor_profiles_by_ticker = read_factor_profiles_from_s3()
        if factor_profiles_by_ticker is None:
            logger.warning(
                "[score_aggregator] factor blend enabled but factors/profiles/"
                "latest.json could not be read — degrading to quant+qual-only "
                "scoring for this run"
            )
        else:
            logger.info(
                "[score_aggregator] factor blend enabled: %d ticker profiles "
                "loaded (regime=%s, weight=%.2f)",
                len(factor_profiles_by_ticker), market_regime, FACTOR_BLEND_WEIGHT,
            )

    # Phase 4 (attractiveness-pillars-260520 arc): pillar_assessments are
    # emitted by qual_analyst when PILLAR_EMIT_ENABLED=true and propagate
    # to score_aggregator via team_outputs[team_id]["qual_output"]
    # ["pillar_assessments"] — a {ticker: pillar_dict} map. We look the
    # dict up per team here and pass per-ticker into
    # compute_composite_breakdown below. When PILLAR_EMIT is off (or the
    # extraction failed), pillar_assessments_by_team[team_id] is an empty
    # dict; per-ticker lookup returns None and the breakdown gracefully
    # degrades to the pure-legacy path. At Phase 4 default weights
    # (pillar_weights all 0), the final_score is IDENTICAL to legacy
    # compute_composite_score by construction.
    pillar_assessments_by_team: dict[str, dict] = {
        tid: out.get("qual_output", {}).get("pillar_assessments", {}) or {}
        for tid, out in team_outputs.items()
    }
    pillar_assessment_applied_count = 0

    for team_id, output in team_outputs.items():
        # Score each recommendation
        for rec in output.get("recommendations", []):
            ticker = rec.get("ticker", "")
            sector = sector_map.get(ticker, "Unknown")
            modifier = sector_modifiers.get(sector, 1.0)

            factor_subscore_val: float | None = None
            factor_breakdown: dict = {"reason": "factor_blend_disabled"}
            ticker_factor_profile: dict | None = None
            if FACTOR_BLEND_ENABLED and factor_profiles_by_ticker is not None:
                ticker_factor_profile = factor_profiles_by_ticker.get(ticker)
                factor_subscore_val, factor_breakdown = compute_factor_subscore(
                    factor_profile=ticker_factor_profile,
                    market_regime=market_regime,
                    regime_weights=FACTOR_BLEND_REGIME_WEIGHTS,
                )
                if factor_subscore_val is not None:
                    factor_blend_applied_count += 1
                else:
                    factor_blend_skipped_count += 1

            ticker_pillar_assessment = (
                pillar_assessments_by_team.get(team_id, {}).get(ticker)
            )
            if ticker_pillar_assessment:
                pillar_assessment_applied_count += 1

            breakdown = compute_composite_breakdown(
                quant_score=rec.get("quant_score"),
                qual_score=rec.get("qual_score"),
                factor_subscore=factor_subscore_val,
                pillar_assessment=ticker_pillar_assessment,
                factor_profile=ticker_factor_profile,
                sector_modifier=modifier,
            )

            # Per-ticker raw quality_score (within-sector percentile) carried
            # through to _build_signals_payload's structural quality floor
            # (Phase 4 of factor substrate, 260513 plan). Replaces the
            # PR #177 narrative text-match adjustment retired in this PR.
            factor_quality_pct: float | None = None
            if FACTOR_BLEND_ENABLED and factor_profiles_by_ticker is not None:
                _profile = factor_profiles_by_ticker.get(ticker) or {}
                _qs = _profile.get("quality_score")
                if _qs is not None:
                    factor_quality_pct = float(_qs)

            investment_theses[ticker] = {
                "ticker": ticker,
                "sector": sector,
                "team_id": team_id,
                # Legacy fields (tolerant-reader compat — executor + CIO + dashboard
                # continue reading these unchanged). At Phase 4 default weights,
                # final_score is byte-equal to legacy compute_composite_score.
                "final_score": breakdown.final_score,
                "quant_score": rec.get("quant_score"),
                "qual_score": rec.get("qual_score"),
                "weighted_base": breakdown.weighted_base,
                "macro_shift": breakdown.macro_shift,
                "factor_subscore": breakdown.legacy_blend.factor_subscore,
                "factor_weight_applied": (
                    breakdown.legacy_blend.w_factor
                    if breakdown.legacy_blend.factor_subscore is not None else 0.0
                ),
                "factor_blend_breakdown": factor_breakdown,
                "factor_quality_score": factor_quality_pct,
                # Phase 4 — full pillar-decomposed breakdown for backtester
                # attribution + dashboard radar. Empty pillar_contributions
                # when PILLAR_EMIT is off / pillar extraction failed; CIO +
                # executor continue reading the legacy final_score above.
                "composite_breakdown": breakdown.model_dump(),
                "bull_case": rec.get("bull_case", ""),
                "bear_case": rec.get("bear_case", ""),
                "catalysts": rec.get("catalysts", []),
                "conviction": normalize_conviction(rec.get("conviction")),
                "quant_rationale": rec.get("quant_rationale", ""),
                "rating": score_to_rating(
                    breakdown.final_score,
                    buy_threshold=RATING_BUY_THRESHOLD,
                    sell_threshold=RATING_SELL_THRESHOLD,
                ),
                "score_failed": breakdown.score_failed,
            }

        # Merge thesis updates from held stocks
        for ticker, thesis in output.get("thesis_updates", {}).items():
            if ticker not in investment_theses:
                # thesis_updates come from the held-stock evaluation path,
                # which occasionally produces records missing ``final_score``
                # (first-time update, legacy archive entries predating the
                # current schema — CME/HSY/KR on 2026-04-20, 9 held tickers
                # on 2026-04-11). Previously we logged ERROR and skipped,
                # which silently dropped valid tickers whose sub-scores
                # were intact.
                #
                # Per feedback_no_silent_fails / feedback_no_unscoreable_labels,
                # recompute-or-hard-fail: if ``quant_score`` + ``qual_score``
                # are present, run them through the same
                # ``compute_composite_score`` path the recommendation loop
                # above uses. If BOTH sub-scores are also missing, raise —
                # the thesis is truly unscoreable and the upstream writer
                # must be fixed.
                if thesis.get("final_score") is None:
                    quant_score = thesis.get("quant_score")
                    qual_score = thesis.get("qual_score")
                    # sector_map is authoritative (loaded from constituents.json
                    # with 903/903 coverage). Held-stock thesis updates default
                    # `sector` to "Unknown" via the Pydantic schema when the LLM
                    # output omits it (state_schemas.py InvestmentThesis), and
                    # the literal string "Unknown" is truthy — `or` would short-
                    # circuit on it and never consult sector_map. Always prefer
                    # sector_map; thesis sector is the last-resort fallback.
                    sector = sector_map.get(ticker) or thesis.get("sector") or "Unknown"
                    modifier = sector_modifiers.get(sector, 1.0)

                    if quant_score is None and qual_score is None:
                        msg = (
                            f"thesis_update for {ticker} missing final_score "
                            f"AND both sub-scores (quant_score, qual_score). "
                            f"Thesis is unscoreable — upstream fix required "
                            f"(team evaluation path wrote an incomplete "
                            f"record). Refusing to drop silently per "
                            f"feedback_no_unscoreable_labels.md."
                        )
                        logger.error("[score_aggregator] %s", msg)
                        raise RuntimeError(msg)

                    recomputed = compute_composite_score(
                        quant_score=quant_score,
                        qual_score=qual_score,
                        sector_modifier=modifier,
                    )
                    logger.warning(
                        "[score_aggregator] thesis_update for %s missing "
                        "final_score — recomputed from sub-scores "
                        "(quant=%s, qual=%s, sector=%s, modifier=%.2f) → "
                        "final_score=%s. Upstream writer should populate "
                        "final_score at thesis-creation time.",
                        ticker, quant_score, qual_score, sector, modifier,
                        recomputed["final_score"],
                    )
                    thesis = dict(thesis)  # avoid mutating team output
                    thesis["final_score"] = recomputed["final_score"]
                    thesis.setdefault("weighted_base", recomputed["weighted_base"])
                    thesis.setdefault("macro_shift", recomputed["macro_shift"])
                    thesis.setdefault("score_failed", recomputed["score_failed"])
                    # Always overwrite sector with the resolved value above
                    # (sector_map first). The previous `if "sector" not in thesis`
                    # guard let an LLM-emitted "Unknown" survive into state and
                    # leak into signals.json, breaking sector attribution
                    # downstream (executor + EOD reconcile + dashboard).
                    thesis["sector"] = sector
                    if "rating" not in thesis:
                        thesis["rating"] = score_to_rating(
                            recomputed["final_score"],
                            buy_threshold=RATING_BUY_THRESHOLD,
                            sell_threshold=RATING_SELL_THRESHOLD,
                        )

                # Normalize conviction to storage format BEFORE spreading.
                # The recommendation path above (line ~752) calls
                # normalize_conviction explicitly; the held-stock thesis_updates
                # path was previously skipping it, so peer_review's raw
                # 'low'/'medium'/'high' agent output flowed unnormalized into
                # InvestmentThesis whose schema expects the storage format
                # (rising/stable/declining). 2026-04-30 warn-mode validation
                # surfaced 2 such violations (DVN, PODD); typed-state hard-fail
                # would crash on this same path. Single source of normalization:
                # both score_aggregator branches must run normalize_conviction.
                normalized_thesis = dict(thesis)
                if "conviction" in normalized_thesis:
                    normalized_thesis["conviction"] = normalize_conviction(
                        normalized_thesis["conviction"]
                    )
                investment_theses[ticker] = {
                    "ticker": ticker,
                    "team_id": team_id,
                    **normalized_thesis,
                }

    logger.info("[score_aggregator] scored %d tickers", len(investment_theses))
    if FACTOR_BLEND_ENABLED:
        logger.info(
            "[score_aggregator] factor_blend coverage: applied=%d skipped=%d "
            "(regime=%s, weight=%.2f)",
            factor_blend_applied_count,
            factor_blend_skipped_count,
            market_regime,
            FACTOR_BLEND_WEIGHT,
        )
    logger.info(
        "[score_aggregator] composite_breakdown emitted for %d tickers "
        "(pillar_assessment_applied=%d of %d) — Phase 4 of "
        "attractiveness-pillars-260520",
        len(investment_theses),
        pillar_assessment_applied_count,
        len(investment_theses),
    )

    # Schema validation on every produced thesis (strict-by-default).
    for ticker, thesis in investment_theses.items():
        _validate(InvestmentThesis, thesis, context=f"score_aggregator:{ticker}")

    return {"investment_theses": investment_theses}


def cio_node(state: ResearchState) -> dict:
    """Run CIO batch evaluation.

    ALL-AGENTS-STRICT (2026-05-16): extends #194's persist+resume to
    the CIO (the most expensive single call — a batch Sonnet
    evaluation). On an SF redrive after a different agent's hard-fail,
    if this run_date's CIO output was already persisted we reuse it
    with ZERO LLM calls.
    """
    logger.info("[cio] starting")

    _run_date = state.get("run_date", "")
    _am_resume = state.get("archive_manager")
    if _am_resume is not None and _run_date:
        try:
            _persisted = _am_resume.load_agent_run(_run_date, "cio")
        except Exception as e:  # pragma: no cover — loader is safe
            logger.warning(
                "[cio] resume load raised unexpectedly (%s) — running "
                "CIO fresh", e,
            )
            _persisted = None
        if _persisted is not None:
            logger.info(
                "[cio] RESUME — reusing persisted output for %s "
                "(zero LLM calls this invocation)", _run_date,
            )
            return _persisted

    # Collect all team recommendations as candidate list
    team_outputs = state.get("sector_team_outputs", {})
    candidates = []
    for team_id, output in team_outputs.items():
        for rec in output.get("recommendations", []):
            candidates.append({
                **rec,
                "team_id": team_id,
            })

    # Load prior IC decisions for portfolio continuity
    prior_ic = []
    try:
        am = state.get("archive_manager")
        if am and am.db_conn:
            rows = am.db_conn.execute(
                "SELECT ticker, thesis_type, rationale, conviction, score "
                "FROM thesis_history WHERE run_date = ("
                "  SELECT MAX(run_date) FROM thesis_history WHERE run_date < ?"
                ") ORDER BY conviction DESC",
                (state.get("run_date", ""),)
            ).fetchall()
            prior_ic = [
                {"ticker": r[0], "thesis_type": r[1], "rationale": r[2],
                 "conviction": r[3], "score": r[4]}
                for r in rows
            ]
            if prior_ic:
                logger.info("[cio] loaded %d prior IC decisions", len(prior_ic))
    except Exception as e:
        logger.debug("[cio] prior IC decisions not available: %s", e)

    # Cost-telemetry scope wraps the single CIO Anthropic call.
    with track_llm_cost(
        agent_id="ic_cio",
        node_name="cio_node",
        run_type="weekly_research",
        run_id=derive_run_id(state),
        prompt=load_prompt("ic_cio_evaluation"),
    ):
        cio_result = run_cio(
            candidates=candidates,
            macro_context={
                "market_regime": state.get("market_regime", "neutral"),
                "macro_report": state.get("macro_report", ""),
            },
            sector_ratings=state.get("sector_ratings", {}),
            current_population=state.get("remaining_population", []),
            open_slots=state.get("open_slots", 0),
            exits=state.get("exits", []),
            run_date=state.get("run_date", ""),
            prior_decisions=prior_ic,
            max_new_entrants=CIO_MAX_NEW_ENTRANTS,
            min_new_entrants=CIO_MIN_NEW_ENTRANTS,
        )

    # Schema validation on CIO output shapes (strict-by-default).
    for decision in cio_result.get("decisions", []):
        _validate(CIODecision, decision, context="cio")
    for ticker, thesis in cio_result.get("entry_theses", {}).items():
        _validate(ThesisUpdate, thesis, context=f"cio:entry_theses:{ticker}")

    cio_state_update = {
        "ic_decisions": cio_result.get("decisions", []),
        "advanced_tickers": cio_result.get("advanced_tickers", []),
        "entry_theses": cio_result.get("entry_theses", {}),
    }

    # Decision-artifact capture (gated on ALPHA_ENGINE_DECISION_CAPTURE_ENABLED).
    snapshot, summary = build_cio_capture_payload(
        state, candidates=candidates, prior_ic=prior_ic,
    )
    _capture_if_enabled(
        state=state,
        agent_id="ic_cio",
        model_name_key="ic_cio",
        input_data_snapshot=snapshot,
        input_data_summary=summary,
        agent_output=cio_state_update,
    )

    # ── Persist on success (resumability) ─────────────────────────────────
    # Reached only if the CIO produced REAL output (a 429 past the
    # deadline / strict-mode parse failure raised upstream). Persist so
    # an SF redrive resumes CIO with zero LLM calls. Best-effort.
    if _am_resume is not None and _run_date:
        try:
            _am_resume.save_agent_run(_run_date, "cio", cio_state_update)
        except Exception as e:  # pragma: no cover — saver is already safe
            logger.warning(
                "[cio] persist raised unexpectedly (%s) — run continues; "
                "CIO just won't be resumable", e,
            )

    return cio_state_update


def population_entry_handler(state: ResearchState) -> dict:
    """Place IC ADVANCE decisions into population."""
    logger.info("[entry_handler] starting")

    final_pop, entry_events = apply_ic_entries(
        remaining_population=state.get("remaining_population", []),
        ic_decisions=state.get("ic_decisions", []),
        entry_theses=state.get("entry_theses", {}),
        sector_map=state.get("sector_map", {}),
        run_date=state.get("run_date", ""),
    )

    all_events = state.get("exits", []) + entry_events

    return {
        "new_population": final_pop,
        "population_rotation_events": all_events,
    }


def consolidator(state: ResearchState) -> dict:
    """Build the weekly research email with 4 structured sections."""
    logger.info("[consolidator] starting")

    sections = []
    run_date = state.get("run_date", "")

    # ── Section 1: Macro Regime Summary ──────────────────────────────────────
    regime = state.get("market_regime", "neutral")
    sections.append(f"# Daily Research Brief — {run_date}\n")
    sections.append("---\n")
    sections.append("## a. MACRO REGIME SUMMARY\n")
    sections.append(f"**Current Regime: {regime.upper()}**\n")
    macro_report = state.get("macro_report", "")
    if macro_report:
        # Strip code fences that the macro agent sometimes includes
        import re
        macro_report = re.sub(r"```\w*\n?", "", macro_report).strip()
        sections.append(macro_report)
    sections.append("")

    # ── Section a.0: Regime Trend (quant substrate, ~8 weeks) ────────────────
    # Replaces the prior static one-shot HMM-posterior snapshot with a
    # compact trend over the most recent weekly regime artifacts so a
    # reader can see the direction of the continuous risk-on/off dial
    # rather than a single week's point estimate. Degrades gracefully
    # (missing / few artifacts → render what's available, never crash).
    regime_trend = _build_regime_trend(state.get("archive_manager"), n_weeks=8)
    if regime_trend:
        sections.append("---\n")
        sections.append("## a.0. REGIME TREND\n")
        sections.extend(regime_trend)
        sections.append("")

    # ── Section a.1: Risk Posture (regime indicator) ─────────────────────────
    # Surfaces the ATR-distribution of the agent population so a reader can
    # eyeball whether the agents have positioned toward higher-vol names
    # (the desired direction post-evaluator-revamp) or drifted back toward
    # defensives. Per evaluator-revamp-260506.md PR 7. This is a *snapshot* —
    # WoW comparison is gated on atr_pct propagating into signals.json
    # (tracked as a P3 ROADMAP follow-up).
    posture_lines = _build_risk_posture(state)
    if posture_lines:
        sections.append("---\n")
        sections.append("## a.1. RISK POSTURE\n")
        sections.extend(posture_lines)
        sections.append("")

    # ── Section 2: Sector Allocation ─────────────────────────────────────────
    sector_ratings = state.get("sector_ratings", {})
    if sector_ratings:
        sections.append("---\n")
        sections.append("## b. SECTOR ALLOCATION\n")
        sections.append("| Sector | Rating | Rationale |")
        sections.append("|--------|--------|-----------|")
        for sector in sorted(sector_ratings):
            sr = sector_ratings[sector]
            rating_raw = sr.get("rating", "market_weight")
            indicator = {"overweight": "\u25b2", "underweight": "\u25bc"}.get(rating_raw, "\u25cf")
            label = f"{indicator} {rating_raw.replace('_', ' ').upper()}"
            rationale = sr.get("rationale", "")
            sections.append(f"| {sector} | {label} | {rationale} |")
        sections.append("")

    # ── Section c: Universe Ratings (population-only, two clean axes) ─────────
    # Notable Developments was dropped (2026-05-16 redesign): exits are
    # already EXIT/Sell rows and CIO-advance lines are entry rationale, so
    # the section was redundant. Per-ticker development notes (exit reasons,
    # >2 ATR / news-spike flags, high-conviction calls) are now folded into
    # that ticker's Rationale cell via ``_build_notable_developments`` —
    # repurposed as a per-ticker note lookup (no longer renders a section).
    sections.append("---\n")
    sections.append("## c. UNIVERSE RATINGS\n")

    current_pop = state.get("current_population", [])
    new_pop = state.get("new_population", [])
    current_tickers = {p["ticker"] for p in current_pop} if current_pop else set()
    new_tickers = {p["ticker"] for p in new_pop} if new_pop else set()

    entrant_tickers = new_tickers - current_tickers
    exit_list = state.get("exits", [])
    exit_tickers = {e.get("ticker_out", "") for e in exit_list}

    theses = state.get("investment_theses", {})
    prior_theses = state.get("prior_theses", {})
    entry_theses = state.get("entry_theses", {})

    pop_lookup = {p["ticker"]: p for p in new_pop}

    # Per-ticker development notes (folded into Rationale cells). Repurposes
    # the old Notable Developments builder as a {ticker: [note, ...]} lookup
    # so every development stays attached to its stock.
    notes_by_ticker = _build_notable_developments(state)

    def _rating_to_recommendation(raw: str) -> str:
        """Map the raw rating value → title-case action.

        BUY → Buy, SELL (or exit) → Sell, anything else → Hold.
        """
        r = (raw or "").strip().upper()
        if r == "BUY":
            return "Buy"
        if r == "SELL":
            return "Sell"
        return "Hold"

    def _with_notes(ticker: str, rationale: str) -> str:
        """Append any per-ticker development notes to the rationale cell.

        Notes already substantively contained in the base rationale
        (common for exits, where the reason is both the base text and
        the folded "Exit: <reason>" note) are skipped to avoid an
        echoed cell.
        """
        notes = notes_by_ticker.get(ticker)
        if not notes:
            return rationale
        rationale_l = (rationale or "").lower()
        fresh = []
        for n in notes:
            # Strip the "<Label>: " prefix before the containment check
            # so "Exit: <reason>" dedupes against a base == "<reason>".
            body = n.split(": ", 1)[1] if ": " in n else n
            if body and body.lower() in rationale_l:
                continue
            fresh.append(n)
        if not fresh:
            return rationale
        note_str = " ".join(fresh)
        if not rationale:
            return note_str
        return f"{rationale} — {note_str}"

    # Population rows: (ticker, status, recommendation, score, rationale)
    # Status ∈ {New, Existing} (lifecycle only). Recommendation ∈
    # {Buy, Hold, Sell} (action, from the raw rating). An exit is an
    # Existing holding with Recommendation Sell.
    rows = []

    for p in new_pop:
        ticker = p["ticker"]
        thesis = theses.get(ticker, {})
        prior = prior_theses.get(ticker, {})
        pop_entry = pop_lookup.get(ticker, {})

        rating = thesis.get("rating") or prior.get("rating") or pop_entry.get("long_term_rating", "HOLD")
        score = thesis.get("final_score") or prior.get("score") or pop_entry.get("long_term_score", 0)
        recommendation = _rating_to_recommendation(rating)

        if ticker in entrant_tickers:
            status = "New"
            et = entry_theses.get(ticker, {})
            rationale = et.get("bull_case") or thesis.get("bull_case", "New entry")
        else:
            status = "Existing"
            rationale = (
                thesis.get("bull_case")
                or prior.get("thesis_summary")
                or "Continuing coverage — no material update"
            )

        rows.append((ticker, status, recommendation, score, _with_notes(ticker, rationale)))

    # Exited stocks — rotated out: Existing holding, Recommendation Sell.
    for e in exit_list:
        ticker = e.get("ticker_out", "?")
        score = e.get("score_out", 0)
        reason = e.get("reason", "Exited from population")
        rows.append((ticker, "Existing", "Sell", score, _with_notes(ticker, reason)))

    # Bench BUY-recs (rated BUY, no open population slot) — NOT in the main
    # table; rendered in their own subsection below.
    bench_buy_candidates = []
    for ticker, thesis in theses.items():
        if thesis.get("rating") == "BUY" and ticker not in new_tickers and ticker not in exit_tickers:
            score = thesis.get("final_score", 0)
            rationale = thesis.get("bull_case", "Buy recommendation — no open slot")
            bench_buy_candidates.append(
                (ticker, "Buy", score, _with_notes(ticker, rationale))
            )

    # Sort: New first, then Existing by Score desc, exits (Sell) last.
    def _row_sort_key(r):
        ticker, status, recommendation, score, _rationale = r
        if status == "New":
            bucket = 0
        elif recommendation == "Sell":
            bucket = 2
        else:
            bucket = 1
        return (bucket, -(score or 0))

    rows.sort(key=_row_sort_key)

    n_exits = len(exit_list)
    n_bench = len(bench_buy_candidates)
    sections.append(
        f"*{len(new_pop)} stocks in population | {len(entrant_tickers)} new | "
        f"{len(new_pop) - len(entrant_tickers)} existing | {n_exits} exited | "
        f"{n_bench} bench buy candidate{'s' if n_bench != 1 else ''}*\n"
    )
    sections.append("| Ticker | Status | Recommendation | Score (0–100) | Rationale |")
    sections.append("|--------|--------|----------------|---------------|-----------|")
    for ticker, status, recommendation, score, rationale in rows:
        score_str = f"{score:.0f}" if score else "—"
        sections.append(f"| {ticker} | {status} | {recommendation} | {score_str} | {rationale} |")
    sections.append("")
    sections.append(
        "*Score = composite of quant + qual sub-scores adjusted by the "
        "sector-macro modifier (0–100); drives population ranking.*"
    )
    sections.append("")

    # ── Section c.1: Buy Candidates (no slot) ────────────────────────────────
    # Bench BUY-recs that didn't get a portfolio slot. Separate small table;
    # omitted entirely when empty.
    if bench_buy_candidates:
        bench_buy_candidates.sort(key=lambda r: -(r[2] or 0))
        sections.append("---\n")
        sections.append("## c.1. BUY CANDIDATES (NO SLOT)\n")
        sections.append(
            "*Rated Buy but not currently held — no open population slot.*\n"
        )
        sections.append("| Ticker | Recommendation | Score | Rationale |")
        sections.append("|--------|----------------|-------|-----------|")
        for ticker, recommendation, score, rationale in bench_buy_candidates:
            score_str = f"{score:.0f}" if score else "—"
            sections.append(f"| {ticker} | {recommendation} | {score_str} | {rationale} |")
        sections.append("")

    # Footer
    sections.append("---\n")
    sections.append(f"*Brief generated: {run_date} | Portfolio: {len(new_pop)} stocks*")

    consolidated = "\n".join(sections)
    return {"consolidated_report": consolidated}


def _build_risk_posture(state: ResearchState) -> list[str]:
    """Build the Risk Posture section: ATR-distribution snapshot of the
    agent population.

    Reports avg / median / max ATR(%) across the new population plus the
    count of high-vol picks (top vol-quartile of the *fetched* universe,
    not just the population). The high-vol-quartile threshold is computed
    from ``technical_scores`` rather than hardcoded — adapts to whatever
    universe happened to be fetched this run.

    Returns an empty list when ``technical_scores`` is empty or no
    population picks have ATR data — graceful no-op so the consolidator
    skips the section. Per evaluator-revamp-260506.md PR 7.
    """
    new_pop = state.get("new_population", [])
    technical_scores = state.get("technical_scores", {})
    if not new_pop or not technical_scores:
        return []

    def _atr_for(ticker: str) -> float | None:
        ts = technical_scores.get(ticker, {})
        # Both keys appear depending on which feature pipeline ran.
        return ts.get("atr_pct") or ts.get("atr_14_pct")

    pop_atrs = [
        a for a in (_atr_for(p["ticker"]) for p in new_pop)
        if a is not None
    ]
    if not pop_atrs:
        return []

    universe_atrs = [
        a for a in (
            ts.get("atr_pct") or ts.get("atr_14_pct")
            for ts in technical_scores.values()
        )
        if a is not None
    ]

    pop_atrs_sorted = sorted(pop_atrs)
    avg_atr = sum(pop_atrs) / len(pop_atrs)
    median_atr = pop_atrs_sorted[len(pop_atrs_sorted) // 2]
    max_atr = max(pop_atrs)

    high_vol_threshold = None
    n_high_vol = None
    if len(universe_atrs) >= 4:
        u_sorted = sorted(universe_atrs)
        # Top-quartile threshold (75th percentile) of the fetched universe.
        high_vol_threshold = u_sorted[int(0.75 * len(u_sorted))]
        n_high_vol = sum(1 for a in pop_atrs if a >= high_vol_threshold)

    lines = [
        f"**Population: {len(new_pop)} stocks | ATR(%) — avg {avg_atr:.2f} | "
        f"median {median_atr:.2f} | max {max_atr:.2f}**",
        "",
    ]
    if high_vol_threshold is not None:
        lines.append(
            f"- **{n_high_vol}/{len(new_pop)}** picks in the top vol-quartile "
            f"of the fetched universe (ATR ≥ {high_vol_threshold:.2f}%)"
        )
    lines.append(
        "- _ATR-based vol proxy. Higher = more volatile names; intelligent risk-taking is the goal._"
    )
    lines.append(
        "- _WoW delta gated on atr_pct in signals.json (P3 follow-up)._"
    )
    return lines


def _build_notable_developments(state: ResearchState) -> dict[str, list[str]]:
    """Per-ticker development-note lookup.

    Repurposed 2026-05-16: was the source of the standalone "NOTABLE
    DEVELOPMENTS" section (dropped in the email redesign — its content
    was redundant with EXIT/Sell rows and entry rationale). Now returns
    a ``{ticker: [note, ...]}`` mapping so the consolidator can fold each
    development into that ticker's Universe-Ratings Rationale cell, keeping
    every development attached to its stock.

    Surfaces three classes of per-ticker note:
      - high-conviction sector-team recommendations (conviction ≥ 70),
      - exit reasons (e.g. ``min_rotation_floor`` / >2 ATR / news-spike),
      - CIO ADVANCE rationale.

    Note text is truncated to 200 chars and de-duplicated per ticker.
    """
    notes: dict[str, list[str]] = {}

    def _add(ticker: str, note: str) -> None:
        if not ticker or ticker == "?" or not note:
            return
        bucket = notes.setdefault(ticker, [])
        if note not in bucket:
            bucket.append(note)

    # High-conviction recommendations.
    # Option A 2026-04-30: agent-format conviction is int 0-100; ``high``
    # corresponds to ≥70 per the threshold scheme used by
    # ``normalize_conviction`` and the qual_analyst_user prompt.
    team_outputs = state.get("sector_team_outputs", {})
    for team_id, output in team_outputs.items():
        for rec in output.get("recommendations", []):
            ticker = rec.get("ticker", "?")
            bull = rec.get("bull_case", "")
            conviction = rec.get("conviction")
            if isinstance(conviction, (int, float)) and conviction >= 70 and bull:
                _add(ticker, f"High conviction ({team_id.title()}): {bull[:200]}")

    # Exits with reasons
    for e in state.get("exits", []):
        ticker = e.get("ticker_out", "?")
        reason = e.get("reason", "")
        if reason:
            _add(ticker, f"Exit: {reason[:200]}")

    # CIO advances
    for d in state.get("ic_decisions", []):
        if d.get("decision") == "ADVANCE":
            ticker = d.get("ticker", "?")
            rationale = d.get("rationale", "")
            if rationale:
                _add(ticker, f"CIO advance: {rationale[:200]}")

    return notes


def _build_regime_trend(archive_manager: Any, n_weeks: int = 8) -> list[str]:
    """Build the regime-trend block from the last ``n_weeks`` substrate
    artifacts.

    Replaces the prior static one-shot HMM-posterior snapshot. Loads the
    most recent weekly ``regime/{run_id}.json`` artifacts (oldest →
    newest) via ``ArchiveManager.list_regime_substrates`` and renders a
    compact per-week table plus a one-line summary of the continuous
    ``composite.intensity_z`` dial (current value + 8-week direction +
    any breached guardrail).

    Degrades gracefully — returns an empty list (consolidator skips the
    section) when there is no archive manager, and a single informational
    line when zero / one artifact is available. Never raises: a
    regime-trend lookup failure must not prevent the brief from
    generating.
    """
    if archive_manager is None:
        return []

    try:
        artifacts = archive_manager.list_regime_substrates(n_recent=n_weeks)
    except Exception as e:  # pragma: no cover — lib already degrades
        logger.debug("[regime_trend] list_regime_substrates failed: %s", e)
        return []

    if not artifacts:
        return ["_Regime substrate unavailable — no weekly artifacts found._"]

    def _date_of(a: dict) -> str:
        return a.get("trading_day") or a.get("calendar_date") or a.get("run_id", "?")

    def _argmax_of(a: dict) -> str:
        return (a.get("hmm") or {}).get("argmax") or a.get("hmm_argmax") or "?"

    def _iz_of(a: dict):
        comp = a.get("composite") or {}
        iz = comp.get("intensity_z")
        if iz is None:
            iz = a.get("composite_intensity_z")
        return iz

    def _change_of(a: dict):
        bocpd = a.get("bocpd") or {}
        cs = bocpd.get("change_signal")
        if cs is None:
            cs = a.get("regime_change_signal")
        return bool(cs)

    def _eff_of(a: dict) -> str:
        eff = a.get("effective_regime")
        if isinstance(eff, dict):
            return eff.get("effective_regime") or "—"
        # latest.json sidecar carries it as a bare string
        return eff if isinstance(eff, str) else "—"

    def _spy_dd_cell(a: dict) -> str:
        spy = ((a.get("drawdown") or {}).get("spy")) or {}
        dd = spy.get("drawdown")
        if not isinstance(dd, (int, float)):
            return "—"
        return f"{-dd * 100:.1f}% ({spy.get('tier', '?')})"

    def _fmt_iz(iz) -> str:
        return f"{iz:+.2f}" if isinstance(iz, (int, float)) else "—"

    lines: list[str] = [
        f"**Weekly regime substrate — last {len(artifacts)} run(s) "
        f"(oldest → newest):**",
        "",
        "| Date | HMM Regime | Intensity-z | BOCPD Change | "
        "SPY Drawdown | Effective |",
        "|------|------------|-------------|--------------|"
        "--------------|-----------|",
    ]
    for a in artifacts:
        lines.append(
            f"| {_date_of(a)} | {_argmax_of(a)} | {_fmt_iz(_iz_of(a))} | "
            f"{'yes' if _change_of(a) else 'no'} | {_spy_dd_cell(a)} | "
            f"{_eff_of(a)} |"
        )
    lines.append("")
    lines.append(
        "_HMM filter run-length (\"weeks in state\") is intentionally "
        "omitted — it is a label-stability diagnostic, not a "
        "market-duration statement; the continuous drawdown depth below "
        "is the market-grounded view._"
    )
    lines.append("")

    if len(artifacts) == 1:
        only = artifacts[0]
        iz = _iz_of(only)
        lines.append(
            f"_Single artifact only — no trend yet. Current intensity-z "
            f"{_fmt_iz(iz)} (HMM {_argmax_of(only)})._"
        )
        return lines

    latest = artifacts[-1]
    earliest = artifacts[0]
    iz_now = _iz_of(latest)
    iz_then = _iz_of(earliest)

    if isinstance(iz_now, (int, float)) and isinstance(iz_then, (int, float)):
        delta = iz_now - iz_then
        if delta > 0.10:
            direction = "rising (risk-off building)"
        elif delta < -0.10:
            direction = "falling (risk-on building)"
        else:
            direction = "flat"
        iz_summary = (
            f"current intensity-z {_fmt_iz(iz_now)} — "
            f"{direction} over {len(artifacts)} weeks "
            f"(Δ {_fmt_iz(delta)} vs {_date_of(earliest)})"
        )
    else:
        iz_summary = f"current intensity-z {_fmt_iz(iz_now)}"

    # Guardrail breach summary (latest artifact).
    guardrails = (latest.get("guardrails") or {})
    breached = [
        k for k, v in guardrails.items()
        if isinstance(v, bool) and v
    ]
    floor = guardrails.get("active_severity_floor")
    if breached:
        gr_summary = f"guardrail breached: {', '.join(sorted(breached))}"
    elif floor:
        gr_summary = f"active severity floor: {floor}"
    else:
        gr_summary = "no guardrail breached"

    # Continuous drawdown statement (latest artifact) — the
    # market-grounded reframe of the dropped "weeks in state" count.
    # Absent-key fallback: omit the clause entirely when the producer
    # wrote no drawdown block (pre-#176/#179) — no behavior change.
    dd = latest.get("drawdown") or {}
    spy = dd.get("spy") or {}
    spy_dd = spy.get("drawdown")
    dd_clause = ""
    if isinstance(spy_dd, (int, float)):
        parts = [
            f"SPY {-spy_dd * 100:.1f}% off trailing peak "
            f"(tier {spy.get('tier', '?')})"
        ]
        ex = dd.get("excess") or {}
        if ex.get("available"):
            nav_dd = ex.get("nav_drawdown")
            depth = ex.get("excess_depth")
            if isinstance(nav_dd, (int, float)):
                parts.append(f"book {-nav_dd * 100:.1f}% off NAV HWM")
            if isinstance(depth, (int, float)):
                parts.append(f"{depth * 100:.1f}pp deeper than market")
        else:
            parts.append("book NAV unavailable — SPY leg only")
        eff = _eff_of(latest)
        dd_clause = f" Drawdown leg: {'; '.join(parts)}; effective={eff}."

    lines.append(f"**Summary:** {iz_summary}; {gr_summary}.{dd_clause}")
    return lines


def _compute_focus_list_audit_lookup(
    *,
    market_regime: str,
    sector_map: dict[str, str],
    focus_list_by_team: dict[str, list[dict]] | None = None,
    override_tickers_by_team: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Project the focus list (from state) onto per-ticker audit fields.

    Returns ``{ticker: {focus_score, focus_stance, focus_team_id,
    focus_rank_in_team, focus_rank_in_sector, focus_list_passed,
    agent_override}}`` for every ticker the factor substrate has a profile
    for. ``focus_list_passed=1`` for top-N members. ``agent_override=1``
    when the quant agent looked up this non-focus ticker via
    @tool get_factor_profile during its team's run.

    PR 4 path: when ``focus_list_by_team`` is provided (computed in
    ``compute_focus_list_node`` and threaded through state), this is a
    pure projection — no S3 read, no recompute.

    Legacy path (PR 2 fallback): when ``focus_list_by_team`` is absent,
    fall back to computing the lookup here so the audit table still gets
    populated. PR 4 state plumbing makes the projection path authoritative;
    the fallback is defensive only.
    """
    override_tickers_by_team = override_tickers_by_team or {}

    # PR 4 path — pure projection
    if focus_list_by_team is not None:
        lookup: dict[str, dict] = {}
        all_overrides: set[str] = set()
        for team_overrides in override_tickers_by_team.values():
            all_overrides.update(team_overrides)

        for team_id, entries in focus_list_by_team.items():
            for e in entries:
                ticker = e["ticker"]
                lookup[ticker] = {
                    "focus_score": e["focus_score"],
                    "focus_stance": e["stance"],
                    "focus_team_id": team_id,
                    "focus_rank_in_team": e["rank_in_team"],
                    "focus_rank_in_sector": e["rank_in_sector"],
                    "focus_list_passed": 1,
                    "agent_override": 0,  # focus-list members aren't overrides
                }

        # Non-focus tickers the agent looked up via @tool get_factor_profile
        # surface here with empty focus fields + agent_override=1. The
        # dashboard reads (focus_list_passed=0 AND agent_override=1) as
        # "agent reached outside the curated set" — the precision /
        # recall / override-hit-rate audit primitives in §5.3 of the
        # scanner plan doc.
        for ticker in all_overrides:
            if ticker in lookup:
                continue  # in focus list — not an override by definition
            lookup[ticker] = {
                "focus_score": None,
                "focus_stance": None,
                "focus_team_id": None,
                "focus_rank_in_team": None,
                "focus_rank_in_sector": None,
                "focus_list_passed": 0,
                "agent_override": 1,
            }
        return lookup

    # ── Legacy fallback — focus_list_by_team absent from state ─────────
    if not FACTOR_BLEND_ENABLED:
        logger.info(
            "[focus_list] factor blend disabled — shadow logging skipped"
        )
        return {}

    factor_profiles = read_factor_profiles_from_s3()
    if not factor_profiles:
        logger.warning(
            "[focus_list] factor profile artifact missing — shadow logging "
            "skipped (factors/profiles/latest.json read returned None)"
        )
        return {}

    focus_scores = compute_focus_scores(
        factor_profiles, market_regime, FACTOR_BLEND_REGIME_WEIGHTS,
    )
    if not focus_scores:
        logger.warning(
            "[focus_list] no factor scores computed for regime=%s — shadow "
            "logging skipped",
            market_regime,
        )
        return {}

    focus_list = build_focus_list(focus_scores, SECTOR_TEAM_MAP)
    summary = summarize_focus_list(focus_list)
    logger.info(
        "[focus_list] regime=%s, %d teams, summary=%s",
        market_regime, len(focus_list), summary,
    )

    lookup_legacy: dict[str, dict] = {}
    passed_tickers: set[str] = set()
    for team_id, entries in focus_list.items():
        for e in entries:
            passed_tickers.add(e.ticker)
            lookup_legacy[e.ticker] = {
                "focus_score": e.focus_score,
                "focus_stance": e.stance,
                "focus_team_id": team_id,
                "focus_rank_in_team": e.rank_in_team,
                "focus_rank_in_sector": e.rank_in_sector,
                "focus_list_passed": 1,
                "agent_override": 0,
            }

    for ticker, entry in focus_scores.items():
        if ticker in passed_tickers:
            continue
        sector = entry.get("sector")
        team_id = SECTOR_TEAM_MAP.get(sector)
        if team_id is None:
            continue
        lookup_legacy[ticker] = {
            "focus_score": entry["focus_score"],
            "focus_stance": entry["stance"],
            "focus_team_id": team_id,
            "focus_rank_in_team": None,
            "focus_rank_in_sector": None,
            "focus_list_passed": 0,
            "agent_override": 0,
        }

    return lookup_legacy


def archive_writer(state: ResearchState) -> dict:
    """Write all data to S3 + SQLite.

    ── Stub-quarantine gate (2026-05-16, the dangerous-bug fix) ──────────
    This is the LAST line of defense before any promoted artifact is
    written. ``assert_no_stub_output`` raises ``StubQuarantineError``
    (→ handler returns status:ERROR, NO signals.json / email / DB
    write) if ANY agent output is synthetic stub text (the
    ``[DRY-RUN`` marker anywhere in the payload / report / theses /
    sector-team outputs) or a sector team is missing. It runs FIRST,
    before write_signals_json / upload_db, and because archive_writer
    precedes email_sender in the graph a raise here also prevents the
    email. A promoted artifact may ONLY be produced by a fully-real,
    all-agents-complete pass.
    """
    logger.info("[archive_writer] starting")
    am: ArchiveManager = state["archive_manager"]
    run_date = state["run_date"]

    from graph.stub_quarantine import assert_no_stub_output

    # Build the signals payload up front so the guard can scan exactly
    # what would be promoted. _build_signals_payload is pure (no I/O);
    # building it twice is cheap and keeps the guard authoritative over
    # the precise bytes that would land in signals.json.
    _candidate_signals_payload = _build_signals_payload(state)
    assert_no_stub_output(
        signals_payload=_candidate_signals_payload,
        consolidated_report=state.get("consolidated_report", "") or "",
        state=state,
    )
    # Bind once at the top so the scanner evaluations and team candidates
    # blocks below can reference it. Previously lines 945 and 975 referenced
    # a bare `team_outputs` that was never defined in this scope, causing
    # NameError and leaving team_candidates empty — which cascaded downstream
    # to the backtester evaluator on 2026-04-11.
    team_outputs = state.get("sector_team_outputs", {})

    # Save IC decisions
    for decision in state.get("ic_decisions", []):
        try:
            am.save_ic_decision(run_date, decision)
        except Exception as e:
            logger.warning("Failed to save IC decision for %s: %s", decision.get("ticker"), e)

    # Save stock archive entries
    for team_id, output in state.get("sector_team_outputs", {}).items():
        for rec in output.get("recommendations", []):
            ticker = rec.get("ticker", "")
            sector = state.get("sector_map", {}).get(ticker, "Unknown")
            try:
                am.save_stock_archive(ticker, sector, team_id, run_date)
            except Exception as e:
                logger.warning("Failed to save stock archive for %s: %s", ticker, e)

        # Save tool usage as analyst resources
        for tc in output.get("tool_calls", []):
            if tc.get("tool") and tc.get("ticker"):
                try:
                    am.save_analyst_resource(
                        ticker=tc["ticker"],
                        run_date=run_date,
                        agent=f"team:{team_id}",
                        resource_type=tc["tool"],
                    )
                except Exception as e:
                    logger.debug("[archive_writer] tool log failed: %s", e)

    # Save population — pass the canonical post-critic macro fields so
    # population/latest.json carries the same regime / sector_modifiers /
    # sector_ratings that signals/latest.json does. Without these, the
    # writer defaults (market_regime="neutral", sector_modifiers={}) and
    # population.json drifts from signals.json on every run — the cause
    # of the 2026-05-11 "Market Regime: NEUTRAL" morning-brief defect.
    # State fields are guaranteed populated by macro_economist_node before
    # this archive_writer node fires (see graph topology).
    new_pop = state.get("new_population", [])
    try:
        am.save_population(
            new_pop,
            run_date,
            market_regime=state.get("market_regime", "neutral"),
            sector_ratings=state.get("sector_ratings", {}),
            sector_modifiers=state.get("sector_modifiers", {}),
        )
    except Exception as e:
        logger.warning("Failed to save population: %s", e)

    # Save rotation events
    for event in state.get("population_rotation_events", []):
        try:
            am.log_rotation_event(event, run_date)
        except Exception as e:
            logger.warning("Failed to save rotation event: %s", e)

    # Persist per-ticker investment_thesis rows. This is what feeds
    # `prior_theses` on the next run via `archive.manager.load_prior_theses`.
    # Until this fix the writer existed but had zero callers, so every population
    # member added since 2026-03-16 was an orphan with no thesis. PR #42
    # (2026-04-22) hard-fails downstream when a held ticker's thesis_update
    # arrives with no scores — that hard-fail is the symptom; this is the cause.
    # Population and thesis must be written atomically so the invariant
    # `every population member has an investment_thesis` always holds.
    investment_theses = state.get("investment_theses", {})
    run_time = state.get("run_time") or run_date
    n_theses_written = 0
    for ticker, thesis in investment_theses.items():
        try:
            thesis_row = {**thesis, "ticker": ticker, "date": run_date}
            am.write_investment_thesis(thesis_row, run_time)
            n_theses_written += 1
        except Exception as e:
            logger.error("Failed to write investment_thesis for %s: %s", ticker, e)
    if investment_theses:
        am.commit()
        logger.info(
            "[archive_writer] wrote %d/%d investment_thesis rows",
            n_theses_written, len(investment_theses),
        )

    # Write signals.json (backward compatible).
    # Universe-membership check sourced from alpha_engine_lib.arcticdb so
    # producer (research preflight) and consumer (executor's
    # filter_buy_candidates_to_universe) compare ENTER tickers against the
    # same authoritative ArcticDB universe library. Soft-skip if ArcticDB
    # is unreachable — the executor's downstream filter is the second-line
    # defense and will surface the gap there.
    universe_symbols: set[str] | None = None
    try:
        from alpha_engine_lib.arcticdb import get_universe_symbols
        universe_symbols = get_universe_symbols(am.bucket)
    except Exception as e:
        logger.warning(
            "[signals_validation] could not load ArcticDB universe symbols: %s "
            "— skipping universe-membership check. Executor's downstream "
            "filter will still gate against the same source.",
            e,
        )

    # Per-team tool_call counts for the zero-tool-call gate. Walks
    # sector_team_outputs (top-level + nested quant_output/qual_output
    # tool_calls) so the count matches what alpha-engine-backtester#148
    # provenance metric records. Pydantic instances are dumped to dicts
    # before walking so the recursive walker handles both shapes.
    tool_call_counts_by_team: dict[str, int] = {}
    for team_id, output in (state.get("sector_team_outputs") or {}).items():
        if hasattr(output, "model_dump"):
            output = output.model_dump()
        if isinstance(output, dict):
            tool_call_counts_by_team[team_id] = _walk_tool_calls(output)

    try:
        # Reuse the payload the stub-quarantine guard already scanned at
        # the top of this node so the promoted bytes are exactly the
        # bytes the guard verified clean (no TOCTOU between guard +
        # write). _build_signals_payload is pure so this is identical.
        signals_payload = _candidate_signals_payload
        _validate_signals_payload(
            signals_payload,
            scanner_universe=universe_symbols,
            tool_call_counts_by_team=tool_call_counts_by_team,
            block_on_zero_tool_calls=False,  # soft-fail; flip after soak
        )
        am.write_signals_json(run_date, state.get("run_time", ""), signals_payload)
    except Exception as e:
        logger.error("Failed to write signals.json: %s", e)

    # Persist the consolidated morning brief alongside signals.json so
    # the dashboard's Research Briefing Archive page can read it. The
    # brief is the same body that goes out in the morning email
    # (`email_sender` node downstream) — emailing it without persisting
    # it leaves no audit trail and the archive page stales out, which
    # is what happened from 2026-03-16 through 2026-05-20.
    consolidated = state.get("consolidated_report", "") or ""
    if consolidated:
        try:
            am.save_consolidated_report(run_date, consolidated)
        except Exception as e:
            logger.error("Failed to save consolidated_report: %s", e)

    # Extract semantic memories from this run (Phase 3)
    try:
        from memory.semantic import extract_semantic_memories
        n_semantic = extract_semantic_memories(
            db_conn=am.db_conn,
            sector_team_outputs=state.get("sector_team_outputs", {}),
            macro_report=state.get("macro_report", ""),
            market_regime=state.get("market_regime", "neutral"),
            ic_decisions=state.get("ic_decisions", []),
            run_date=run_date,
        )
        if n_semantic:
            logger.info("[archive_writer] extracted %d semantic memories", n_semantic)
    except Exception as e:
        logger.debug("[archive_writer] semantic extraction skipped: %s", e)

    # ── Evaluation logging ──────────────────────────────────────────────────
    # Log all ~900 stocks with tech indicators for population baseline analysis.
    # These writes feed the backtester's weekly grading of scanner / sector
    # team / CIO components against universe_returns. Must not be silently
    # swallowed — a missed week leaves a permanent hole in grade history.
    scanner_universe = state.get("scanner_universe", [])
    technical_scores = state.get("technical_scores", {})
    sector_map = state.get("sector_map", {})
    market_regime = state.get("market_regime", "neutral")
    # Build set of tickers that any team picked (quant top-10 or recommended)
    team_picked_tickers: set[str] = set()
    for _tid, _out in team_outputs.items():
        for _rec in _out.get("recommendations", []):
            team_picked_tickers.add(_rec.get("ticker", ""))
        for _pick in _out.get("quant_output", {}).get("ranked_picks", []):
            if isinstance(_pick, dict):
                team_picked_tickers.add(_pick.get("ticker", ""))

    # ── Focus list audit (PR 4 of scanner-placement arc) ────────────────────
    # Project the focus list computed in compute_focus_list_node + the
    # per-team override_tickers (from @tool get_factor_profile invocations
    # on non-focus tickers) onto scanner_eval rows. The legacy recompute
    # path in _compute_focus_list_audit_lookup remains as a fallback when
    # focus_list_by_team is unexpectedly absent from state.
    # Plan doc: alpha-engine-docs/private/scanner-260514.md
    focus_list_by_team_state: dict[str, list[dict]] | None = state.get(
        "focus_list_by_team"
    )
    # Aggregate override_tickers from sector_team_outputs. Each team's
    # output dict (per sector_team.py) carries an "override_tickers" list
    # of tickers the quant agent looked up via @tool get_factor_profile
    # that were NOT in the team's focus list.
    override_tickers_by_team: dict[str, list[str]] = {}
    for _tid, _out in team_outputs.items():
        ot = _out.get("override_tickers") if isinstance(_out, dict) else None
        if ot:
            override_tickers_by_team[_tid] = list(ot)

    focus_lookup: dict[str, dict] = _compute_focus_list_audit_lookup(
        market_regime=market_regime,
        sector_map=sector_map,
        focus_list_by_team=focus_list_by_team_state,
        override_tickers_by_team=override_tickers_by_team,
    )

    # Build the scanner_universe row set, then UNION with any agent_override
    # tickers that fell outside the scanner_universe (the agent can look up
    # any ticker, not just ones in the current week's S&P 900 slice).
    universe_set = set(scanner_universe)
    extra_override_tickers = [
        t for t in focus_lookup.keys()
        if t not in universe_set and focus_lookup[t].get("agent_override") == 1
    ]

    scanner_evals = []
    for ticker in list(scanner_universe) + extra_override_tickers:
        ts = technical_scores.get(ticker, {})
        row = {
            "ticker": ticker,
            "eval_date": run_date,
            "sector": sector_map.get(ticker),
            "tech_score": ts.get("technical_score"),
            "rsi_14": ts.get("rsi_14"),
            "atr_pct": ts.get("atr_pct") or ts.get("atr_14_pct"),
            "price_vs_ma200": ts.get("price_vs_ma200"),
            "current_price": ts.get("current_price"),
            "avg_volume_20d": ts.get("avg_volume_20d"),
            "quant_filter_pass": 1 if ticker in team_picked_tickers else 0,
        }
        # Project focus-list audit fields. focus_lookup is empty when factor
        # profiles aren't readable — every row gets NULL focus_* fields +
        # focus_list_passed=0, which the dashboard reads as "shadow logging
        # didn't run this cycle" rather than "all tickers failed."
        fl_entry = focus_lookup.get(ticker)
        if fl_entry is not None:
            row.update(fl_entry)
        scanner_evals.append(row)

    am.write_scanner_evaluations(scanner_evals)
    if focus_lookup:
        n_passed = sum(1 for r in scanner_evals if r.get("focus_list_passed"))
        logger.info(
            "[archive_writer] logged %d scanner evaluations (focus list: %d passed)",
            len(scanner_evals), n_passed,
        )
    else:
        logger.info(
            "[archive_writer] logged %d scanner evaluations (focus list: shadow logging unavailable this run)",
            len(scanner_evals),
        )

    # Log quant top-10 per team + final recommendations.
    #
    # Per-sub-signal scores (rsi/macd/ma50/ma200/momentum) are computed
    # at write time from technical_scores using the committed
    # compute_technical_sub_scores() helper. Persisting them lets the
    # backtester's tech_weight_ablation optimizer (PR-C) re-rank under
    # alternate composite weights without re-running the research
    # pipeline. The market_regime input must match what the live
    # compute_technical_score call used so the RSI sub-score is
    # numerically consistent across producer + consumer.
    from scoring.technical import compute_technical_sub_scores
    team_candidate_records = []
    archive_writer_regime = state.get("market_regime", "neutral")
    for team_id, output in team_outputs.items():
        quant_picks = output.get("quant_output", {}).get("ranked_picks", [])
        recommended_tickers = {
            r.get("ticker", "") for r in output.get("recommendations", [])
        }
        for rank, pick in enumerate(quant_picks, 1):
            if not isinstance(pick, dict) or "ticker" not in pick:
                continue
            ticker = pick["ticker"]
            # Find qual score from recommendations if available
            qual_score = None
            for rec in output.get("recommendations", []):
                if rec.get("ticker") == ticker:
                    qual_score = rec.get("qual_score")
                    break
            # Sub-scores from cached indicators. None on each field if
            # technical_scores entry is missing (rare — would happen if
            # the quant agent emitted a pick for a ticker that didn't
            # pass scanner indicators), and the writer persists NULL.
            indicators = technical_scores.get(ticker)
            sub_scores: dict = {}
            if isinstance(indicators, dict):
                try:
                    sub_scores = compute_technical_sub_scores(
                        indicators, market_regime=archive_writer_regime,
                    )
                except Exception as e:
                    logger.warning(
                        "[archive_writer] sub-score computation failed for %s: %s",
                        ticker, e,
                    )
            team_candidate_records.append({
                "ticker": ticker,
                "eval_date": run_date,
                "team_id": team_id,
                "quant_rank": rank,
                "quant_score": pick.get("quant_score"),
                "qual_score": qual_score,
                "team_recommended": 1 if ticker in recommended_tickers else 0,
                "rsi_sub_score": sub_scores.get("rsi"),
                "macd_sub_score": sub_scores.get("macd"),
                "ma50_sub_score": sub_scores.get("ma50"),
                "ma200_sub_score": sub_scores.get("ma200"),
                "momentum_sub_score": sub_scores.get("momentum"),
            })
    am.write_team_candidates(team_candidate_records)
    logger.info("[archive_writer] logged %d team candidates", len(team_candidate_records))

    # Log all CIO decisions (ADVANCE/REJECT/DEADLOCK)
    cio_eval_records = []
    for decision in state.get("ic_decisions", []):
        ticker = decision.get("ticker", "")
        thesis = investment_theses.get(ticker, {})
        cio_eval_records.append({
            "ticker": ticker,
            "eval_date": run_date,
            "team_id": thesis.get("team_id"),
            "quant_score": thesis.get("quant_score"),
            "qual_score": thesis.get("qual_score"),
            "combined_score": thesis.get("weighted_base"),
            "macro_shift": thesis.get("macro_shift"),
            "final_score": thesis.get("final_score"),
            "cio_decision": decision.get("decision", "UNKNOWN"),
            "cio_conviction": decision.get("conviction"),
            "cio_rank": decision.get("rank"),
            "rationale": decision.get("rationale"),
            # Rule-tag attribution from prompt v1.3.0 + lib v0.7.0
            # CIORawDecision.rule_tags. None on legacy outputs (prompt
            # < v1.3.0); persisted as SQLite NULL so analytics can
            # distinguish "untagged legacy" from "no tags emitted."
            "rule_tags": decision.get("rule_tags"),
        })
    am.write_cio_evaluations(cio_eval_records)
    logger.info("[archive_writer] logged %d CIO evaluations", len(cio_eval_records))

    # Upload DB
    try:
        am.upload_db(run_date)
    except Exception as e:
        logger.warning("Failed to upload DB: %s", e)

    return {}


def email_sender(state: ResearchState) -> dict:
    """Send the morning email with properly rendered HTML."""
    from emailer.sender import send_email
    from emailer.formatter import format_email
    from config import EMAIL_RECIPIENTS, EMAIL_SENDER

    logger.info("[email_sender] starting")
    consolidated = state.get("consolidated_report", "")
    run_date = state.get("run_date", "")

    if consolidated:
        try:
            subject = f"Alpha Engine Research — {run_date}"
            html_body, plain_body = format_email(consolidated, run_date)
            send_email(
                subject=subject,
                html_body=html_body,
                plain_body=plain_body,
                recipients=EMAIL_RECIPIENTS,
                sender=EMAIL_SENDER,
            )
            return {"email_sent": True}
        except Exception as e:
            logger.error("Email send failed: %s", e)

    return {"email_sent": False}


def _walk_tool_calls(node: object) -> int:
    """Recursively count every ToolCall list found in an agent output dict.

    Sector teams nest tool_calls under ``quant_output.tool_calls`` and
    ``qual_output.tool_calls`` because the team is a sub-graph with
    quant + qual + peer_review sub-agents; macro_economist puts them at
    the top-level. Mirrors the walker shipped in alpha-engine-backtester
    #148 ``analysis/provenance_grounding.py`` so producer (this gate)
    and consumer (provenance metric) count the same trace.
    """
    count = 0
    if isinstance(node, dict):
        tcs = node.get("tool_calls")
        if isinstance(tcs, list):
            count += sum(1 for tc in tcs if isinstance(tc, dict))
        for k, v in node.items():
            if k == "tool_calls":
                continue
            count += _walk_tool_calls(v)
    elif isinstance(node, list):
        for item in node:
            count += _walk_tool_calls(item)
    return count


def _validate_signals_payload(
    payload: dict,
    scanner_universe: list[str] | set[str] | None = None,
    tool_call_counts_by_team: dict[str, int] | None = None,
    block_on_zero_tool_calls: bool = False,
) -> None:
    """Block emission of signals.json when any ENTER signal violates a
    structural correctness gate. Three checks:

    1. **Unresolved sector.** Surface for the 2026-05-04 EOG/NVT incident:
       the v1 signals.json had sector="Unknown" for tickers whose
       constituents sector_map hadn't loaded yet. The morning planner
       consumed v1, the order book persisted "Unknown", and the daemon's
       intraday fills wrote "Unknown" into trades.db.

    2. **Universe-membership drift.** A ticker that's no longer in the
       current S&P 500+400 scanner universe must not surface as a buy
       candidate. Held positions that exit the universe should still
       receive HOLD/EXIT signals (managed via the existing rating logic)
       — but never ENTER, which would let the executor add to a position
       outside the index.

    3. **Zero-tool-call producing agent.** Soft-fail by default. When
       ``tool_call_counts_by_team`` is provided, looks up the producing
       team for each ENTER signal. If the team's tool_call count is 0,
       the signal was emitted by an agent that didn't exercise its
       retrieval capability — a hallucination signal. With
       ``block_on_zero_tool_calls=True`` (post-soak hard-fail mode), the
       list raises; otherwise a WARNING is logged and emission proceeds.
       Composes with alpha-engine-backtester#148 provenance metrics
       (per-Saturday tracking) — same walker, same structural definition.

    On block: signals.json is not written, the existing try/except logs
    ERROR, and the executor falls back to prior trading day's signals via
    signal_reader.read_signals_with_fallback.
    """
    unresolved_sector: list[str] = []
    out_of_universe: list[str] = []
    zero_tool_call_signals: list[str] = []
    universe_set = set(scanner_universe) if scanner_universe else None
    for ticker, sig in (payload.get("signals") or {}).items():
        if sig.get("signal") != "ENTER":
            continue
        sector = sig.get("sector")
        if not sector or sector == "Unknown":
            unresolved_sector.append(ticker)
        if universe_set is not None and ticker not in universe_set:
            out_of_universe.append(ticker)
        if tool_call_counts_by_team is not None:
            team_id = sig.get("team_id")
            if team_id and tool_call_counts_by_team.get(team_id, 0) == 0:
                zero_tool_call_signals.append(ticker)

    # Soft-fail emission for the zero-tool-call check during the soak window.
    # Hard-fail flips the flag once ~4 weeks of observed metric show the
    # gate doesn't false-positive.
    if zero_tool_call_signals and not block_on_zero_tool_calls:
        logger.warning(
            "[signals_validation] SOFT-FAIL: %d ENTER signals from agents "
            "with zero tool_calls (hallucination signal): %s. signals.json "
            "still emitted; flip block_on_zero_tool_calls=True after soak.",
            len(zero_tool_call_signals), sorted(zero_tool_call_signals),
        )

    blocking_failures: list[str] = []
    if unresolved_sector:
        blocking_failures.append(
            f"{len(unresolved_sector)} ENTER signals with unresolved "
            f"sector: {sorted(unresolved_sector)}"
        )
    if out_of_universe:
        blocking_failures.append(
            f"{len(out_of_universe)} ENTER signals outside current S&P "
            f"900 scanner universe: {sorted(out_of_universe)}"
        )
    if zero_tool_call_signals and block_on_zero_tool_calls:
        blocking_failures.append(
            f"{len(zero_tool_call_signals)} ENTER signals from agents "
            f"with zero tool_calls: {sorted(zero_tool_call_signals)}"
        )
    if blocking_failures:
        raise RuntimeError(
            "[signals_validation] BLOCKED: "
            + "; ".join(blocking_failures)
            + ". signals.json not written; executor will fall back to "
            "prior trading day."
        )


def _build_signals_payload(state: ResearchState) -> dict:
    """Build backward-compatible signals.json payload.

    Includes both v2 keys (signals, population) and v1 keys (universe, buy_candidates)
    so the executor and predictor can read actionable signals.
    """
    theses = state.get("investment_theses", {})
    prior_theses = state.get("prior_theses", {})
    pop = state.get("new_population", [])
    pop_tickers = {p["ticker"] for p in pop}
    pop_lookup = {p["ticker"]: p for p in pop}
    sector_map = state.get("sector_map", {})
    sector_ratings = state.get("sector_ratings", {})
    entry_theses = state.get("entry_theses", {})

    # v2 signals dict (keyed by ticker)
    # Signal logic:
    #   ENTER  = BUY-rated AND in population (new entry or reaffirmed hold)
    #   HOLD   = held in population, not BUY-rated (maintain position)
    #   EXIT   = dropped from population (sell)
    #   Stocks not in population and not BUY-rated are excluded (irrelevant to executor)
    advanced_tickers = set(state.get("advanced_tickers", []))
    signals = {}

    # First: tickers with fresh theses from this run
    for ticker, thesis in theses.items():
        rating = thesis.get("rating", "HOLD")
        final_score = thesis.get("final_score")
        in_pop = ticker in pop_tickers

        # Safety gate: a BUY rating with no final_score is a broken thesis
        # (e.g., held-stock LLM update that dropped scoring fields). Downgrade
        # to HOLD so the executor does not attempt to ENTER on null score.
        # Root cause mitigation for the 2026-04-04 incident where
        # LNTH/KR/PR/HAL leaked through as ENTER with score=null.
        if rating == "BUY" and final_score is None:
            logger.warning(
                "[signals] %s has rating=BUY but final_score is None — "
                "downgrading to HOLD (broken thesis)",
                ticker,
            )
            rating = "HOLD"

        # Determine signal — CIO is the sole gate for new entrants.
        # Team recs that the CIO did not advance fall through and produce no
        # signal even if rated BUY. Reaffirmations of held BUY-rated names
        # remain unbounded.
        if rating == "BUY" and ticker in advanced_tickers:
            signal = "ENTER"  # CIO approved new entry
        elif rating == "BUY" and in_pop:
            signal = "ENTER"  # Reaffirm existing BUY position
        elif in_pop:
            signal = "HOLD"   # Held, not BUY-rated
        else:
            continue  # Not held, not CIO-advanced — drop

        # sector_map is authoritative (loaded from constituents.json with full
        # universe coverage). Held-stock thesis updates can leak sector="Unknown"
        # via the Pydantic default when the LLM omits the field (see
        # score_aggregator held-stock branch). Prefer sector_map; thesis sector
        # is the fallback. Mirrors the carry-over branch below.
        signals[ticker] = {
            "ticker": ticker,
            "score": thesis.get("final_score"),
            "rating": rating,
            "signal": signal,
            "conviction": normalize_conviction(thesis.get("conviction", "stable")),
            "thesis_summary": thesis.get("bull_case", ""),
            "sector": sector_map.get(ticker) or thesis.get("sector") or "Unknown",
            "team_id": thesis.get("team_id"),
            "quant_score": thesis.get("quant_score"),
            "qual_score": thesis.get("qual_score"),
            "factor_subscore": thesis.get("factor_subscore"),
            "factor_weight_applied": thesis.get("factor_weight_applied", 0.0),
            "factor_blend_breakdown": thesis.get("factor_blend_breakdown"),
            "factor_quality_score": thesis.get("factor_quality_score"),
            "sub_scores": {
                "quant": thesis.get("quant_score"),
                "qual": thesis.get("qual_score"),
            },
        }

    # Second: population tickers without fresh theses — carry over from prior week
    for p in pop:
        ticker = p["ticker"]
        if ticker not in signals:
            prior = prior_theses.get(ticker, {})
            sector = sector_map.get(ticker, p.get("sector", "Unknown"))
            prior_rating = prior.get("rating") or p.get("long_term_rating", "HOLD")
            carried_score = prior.get("score") or p.get("long_term_score")
            # Only emit ENTER if we have a score — unscored holdovers stay HOLD
            if prior_rating == "BUY" and carried_score is not None:
                carried_signal = "ENTER"
            else:
                carried_signal = "HOLD"
            signals[ticker] = {
                "ticker": ticker,
                "score": carried_score,
                "rating": prior_rating,
                "signal": carried_signal,
                "conviction": normalize_conviction(prior.get("conviction") or p.get("conviction", "stable")),
                "thesis_summary": prior.get("thesis_summary", ""),
                "sector": sector,
                "team_id": prior.get("team_id"),
                "quant_score": prior.get("quant_score"),
                "qual_score": prior.get("qual_score"),
                "sub_scores": {
                    "quant": prior.get("quant_score"),
                    "qual": prior.get("qual_score"),
                },
            }

    # Third: exited stocks — explicit EXIT signal so executor knows to sell
    for e in state.get("exits", []):
        ticker = e.get("ticker_out", "")
        if ticker and ticker not in signals:
            sector = sector_map.get(ticker, "Unknown")
            signals[ticker] = {
                "ticker": ticker,
                "score": e.get("score_out", 0),
                "rating": "SELL",
                "signal": "EXIT",
                "conviction": "declining",
                "thesis_summary": e.get("reason", "Exited from population"),
                "sector": sector,
                "team_id": None,
                "quant_score": None,
                "qual_score": None,
                "sub_scores": {"quant": None, "qual": None},
            }

    # v1-compatible universe list (executor reads this)
    universe = []
    for ticker, sig in signals.items():
        sector = sig["sector"]
        sr = sector_ratings.get(sector, {})
        pop_entry = pop_lookup.get(ticker, {})
        universe.append({
            "ticker": ticker,
            "signal": sig["signal"],
            "score": sig["score"],
            "rating": sig["rating"],
            "conviction": sig["conviction"],
            "price_target_upside": pop_entry.get("price_target_upside"),
            "sector_rating": sr.get("rating", "market_weight"),
            "sector": sector,
            "thesis_summary": sig["thesis_summary"],
            "factor_quality_score": sig.get("factor_quality_score"),
            "sub_scores": sig.get("sub_scores"),
        })

    # v1-compatible buy_candidates list (ENTER signals with enriched theses).
    #
    # Two structural gates run in sequence on every ENTER signal — both gate
    # only NEW buys, never HOLDs / EXITs:
    #   1. Macro-sector coherence gate (260513): block new buys in UW sectors
    #      below SECTOR_COHERENCE_UW_MIN_SCORE — forces structural alignment
    #      between the macro call and per-pick action.
    #   2. Factor quality floor (Phase 4, 260513 plan): block new buys whose
    #      within-sector quality_score percentile is below the floor — drops
    #      bottom-decile-quality names regardless of agent sentiment.
    #      Replaces the dormant Piotroski-lite scanner-side quality_floor.
    buy_candidates = []
    blocked_by_coherence_gate: list[dict] = []
    blocked_by_quality_floor: list[dict] = []
    for entry in universe:
        if entry["signal"] != "ENTER":
            continue
        if (
            SECTOR_COHERENCE_GATE_ENABLED
            and entry.get("sector_rating") == "underweight"
            and (entry.get("score") or 0) < SECTOR_COHERENCE_UW_MIN_SCORE
        ):
            blocked_by_coherence_gate.append({
                "ticker": entry["ticker"],
                "sector": entry["sector"],
                "score": entry["score"],
                "uw_min_score": SECTOR_COHERENCE_UW_MIN_SCORE,
            })
            continue
        # Factor quality floor: skip when the ticker has no factor profile
        # (graceful degrade — same pattern as the rest of the factor blend)
        # or its sector is in the exempt list (Financial / Real Estate /
        # Utilities by default, where the quality factor metrics do not apply).
        if (
            FACTOR_QUALITY_FLOOR_ENABLED
            and entry.get("factor_quality_score") is not None
            and entry["sector"] not in FACTOR_QUALITY_FLOOR_EXEMPT_SECTORS
            and entry["factor_quality_score"] < FACTOR_QUALITY_FLOOR_MIN_PERCENTILE
        ):
            blocked_by_quality_floor.append({
                "ticker": entry["ticker"],
                "sector": entry["sector"],
                "quality_pct": entry["factor_quality_score"],
                "floor": FACTOR_QUALITY_FLOOR_MIN_PERCENTILE,
            })
            continue
        candidate = dict(entry)
        et = entry_theses.get(entry["ticker"], {})
        if et:
            candidate["thesis_summary"] = et.get("bull_case", candidate["thesis_summary"])
            candidate["catalysts"] = et.get("catalysts", [])
        buy_candidates.append(candidate)
    if blocked_by_coherence_gate:
        logger.info(
            "macro_sector_coherence_gate blocked %d ENTER signal(s) "
            "from UNDERWEIGHT sectors (uw_min_score=%.1f): %s",
            len(blocked_by_coherence_gate),
            SECTOR_COHERENCE_UW_MIN_SCORE,
            [f"{b['ticker']}({b['sector']},{b['score']:.1f})" for b in blocked_by_coherence_gate],
        )
    if blocked_by_quality_floor:
        logger.info(
            "factor_quality_floor blocked %d ENTER signal(s) "
            "with quality_score below %.1f-percentile: %s",
            len(blocked_by_quality_floor),
            FACTOR_QUALITY_FLOOR_MIN_PERCENTILE,
            [f"{b['ticker']}({b['sector']},quality={b['quality_pct']:.1f})" for b in blocked_by_quality_floor],
        )

    return {
        "date": state.get("run_date", ""),
        "time": state.get("run_time", ""),
        "market_regime": state.get("market_regime", "neutral"),
        "sector_modifiers": state.get("sector_modifiers", {}),
        "sector_ratings": sector_ratings,
        "signals": signals,
        "population": [p["ticker"] for p in pop],
        "universe": universe,
        "buy_candidates": buy_candidates,
        "architecture_version": "sector_teams",
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Sector-team graph with serial macro upstream + Send() fan-out to sectors.

    Topology (regime-v3 Stage C, 2026-05-14):
      fetch_data
        → load_regime_substrate_node          (Stage C — S3 GET on
                                              regime/latest.json; graceful
                                              None when unavailable)
        → macro_economist_node                (serial — consumes substrate
                                              as strong prior, remains
                                              final regime authority)
        → compute_factor_profiles_node        (un-orphan: write
                                              factors/profiles/* from
                                              features/{run_date}/*;
                                              graceful-degrade, never
                                              hard-fails the run)
        → compute_focus_list_node             (reads factors/profiles
                                              from S3)
        → dispatch_sectors_and_exit           (Send: 6 teams + exit)
        → merge_results
        → score_aggregator → cio_node → population_entry_handler
        → consolidator → archive → email → END

    Why macro is serial (was: parallel with sectors + exit):
      Send() snapshots state at dispatch time. Before Stage B, sector
      teams received ``market_regime="neutral"`` regardless of what
      macro_economist eventually classified, because macro hadn't yet
      written to state when Send() captured it. Making macro serial
      upstream means sector teams see the real regime call +
      sector_modifiers + sector_ratings in their ReAct context. The
      4-class regime taxonomy (bull/neutral/caution/bear) now flows
      into every per-stock pick decision.

    Why load_regime_substrate_node is its own step (Stage C):
      Stage C adds a quantitative regime substrate (HMM + composite +
      BOCPD) as a strong prior into the macro agent's ReAct prompt.
      Reading it in a dedicated node makes the S3 fetch a discrete
      step visible in LangGraph traces; if the substrate is missing
      or malformed, the failure is isolated to this node (clearer
      logs) and ``regime_substrate=None`` flows to macro which falls
      back to its prior LLM + post-LLM-guardrail behavior.

    Trade-off: +1 macro-agent runtime (~1-2 min with reflection loop)
    + 1 S3 GET (~50ms) on Saturday SF wall-clock vs. structurally-
    guaranteed regime awareness across all sector LLM analyses +
    quantitative substrate informing the macro authority. Saturday is
    not latency-critical; the trade is correct.

    Macro fallback:
      run_macro_agent_with_reflection's LLM-failure path returns the
      default-neutral payload, so a macro crash degrades to "no regime
      tilt" rather than halting the graph. This was acceptable when
      macro ran in parallel; now structurally critical because the
      whole pipeline waits on it.
    """
    graph = StateGraph(ResearchState)

    # Nodes
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("load_regime_substrate_node", load_regime_substrate_node)
    graph.add_node("macro_economist_node", macro_economist_node)
    # Un-orphan arc: produce the institutional factor-profile substrate
    # AFTER fetch_data populated sector_map + run_date (macro does not
    # mutate either) and BEFORE both consumers (compute_focus_list_node
    # + score_aggregator) do their existing S3 read. Graceful-degrade
    # on any failure — never hard-fails the research run.
    graph.add_node("compute_factor_profiles_node", compute_factor_profiles_node)
    # PR 4 of scanner-placement arc: compute the per-team regime-blended
    # focus list AFTER macro has written market_regime to state, BEFORE
    # the sector team dispatch reads it.
    graph.add_node("compute_focus_list_node", compute_focus_list_node)
    graph.add_node("sector_team_node", sector_team_node)
    graph.add_node("exit_evaluator_node", exit_evaluator_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("score_aggregator", score_aggregator)
    graph.add_node("cio_node", cio_node)
    graph.add_node("population_entry_handler", population_entry_handler)
    graph.add_node("consolidator_node", consolidator)
    graph.add_node("archive_writer", archive_writer)
    graph.add_node("email_sender_node", email_sender)

    # Entry point
    graph.set_entry_point("fetch_data")

    # Serial: fetch_data → load_regime_substrate_node → macro_economist_node
    # → compute_focus_list_node. Substrate flows into the macro agent's
    # ReAct prompt as a strong prior; macro's regime feeds the focus list
    # blend; focus list lands in state before sector teams dispatch.
    graph.add_edge("fetch_data", "load_regime_substrate_node")
    graph.add_edge("load_regime_substrate_node", "macro_economist_node")
    # Splice compute_factor_profiles_node between macro and the focus
    # list: it needs sector_map + run_date (set in fetch_data, unchanged
    # by the substrate loader / macro) and must land the factor substrate
    # in S3 before compute_focus_list_node AND score_aggregator do their
    # existing read_factor_profiles_from_s3() this same run.
    graph.add_edge("macro_economist_node", "compute_factor_profiles_node")
    graph.add_edge("compute_factor_profiles_node", "compute_focus_list_node")

    # Fan-out AFTER focus list: dispatch to 6 sector teams + exit evaluator.
    graph.add_conditional_edges("compute_focus_list_node", dispatch_sectors_and_exit)

    # Fan-in: sector + exit Sends converge to merge_results.
    graph.add_edge("sector_team_node", "merge_results")
    graph.add_edge("exit_evaluator_node", "merge_results")

    # Sequential post-merge
    graph.add_edge("merge_results", "score_aggregator")
    graph.add_edge("score_aggregator", "cio_node")
    graph.add_edge("cio_node", "population_entry_handler")
    graph.add_edge("population_entry_handler", "consolidator_node")
    graph.add_edge("consolidator_node", "archive_writer")
    graph.add_edge("archive_writer", "email_sender_node")
    graph.add_edge("email_sender_node", END)

    return graph.compile()


def create_initial_state(
    run_date: str,
    archive_manager: ArchiveManager,
    is_early_close: bool = False,
) -> ResearchState:
    return ResearchState(
        run_date=run_date,
        run_time=datetime.now(timezone.utc).isoformat(),
        archive_manager=archive_manager,
        is_early_close=is_early_close,
        email_sent=False,
    )
