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
from typing import Annotated, Any, Callable, NamedTuple, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import ValidationError

from config import (
    CIO_MAX_NEW_ENTRANTS,
    CIO_MIN_NEW_ENTRANTS,
    CIO_FORCE_FILL_CONVICTION_FLOOR,
    CIO_NEW_ENTRANT_ALERT_FLOOR,
    CIO_DEBLENDED_ORCHESTRATION,
    ADAPTIVE_SLOT_ALLOCATION_ENABLED,
    CIO_CRITIC_ENABLED,
    POPULATION_CFG,
    RATING_BUY_THRESHOLD,
    RATING_SELL_THRESHOLD,
    SECTOR_COHERENCE_GATE_ENABLED,
    SECTOR_COHERENCE_UW_MIN_SCORE,
    FACTOR_BLEND_ENABLED,
    FACTOR_BLEND_WEIGHT,
    get_factor_blend_regime_weights,
    ATTRACTIVENESS_FEED_ENABLED,
    ATTRACTIVENESS_FEED_TOP_N,
    FACTOR_QUALITY_FLOOR_ENABLED,
    FACTOR_QUALITY_FLOOR_MIN_PERCENTILE,
    FACTOR_QUALITY_FLOOR_EXEMPT_SECTORS,
    FOCUS_LIST_DEFAULT_TEAM_SIZE,
    FOCUS_LIST_GATING_ENABLED,
    FOCUS_LIST_PER_TEAM_SIZE_OVERRIDES,
    PILLAR_COMPOSITE_WEIGHTS,
    PILLAR_COMPOSITE_WITHIN_PILLAR_QUAL_WEIGHT,
    PILLAR_COMPOSITE_LEGACY_BLEND,
    MACRO_OVERLAY_ENABLED,
    MACRO_MAX_SHIFT_POINTS,
    MACRO_MODIFIER_RANGE,
    NEUTRALIZATION_LIVE_ENABLED,
    NEUTRALIZATION_FACTORS,
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
from agents.investment_committee.ic_cio import (
    run_cio,
    run_cio_with_reflection,
    build_sector_neutral_quality_map,
    _PROMPT_DEFAULT,
    _PROMPT_DEBLENDED,
)
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
from archive.tool_usage_analysis import TEAM_RESOURCE_TICKER

from nousergon_lib.decision_capture import (
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
    ADVANCE_DECISIONS,
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
    "ic_cio_critic": "claude-haiku-4-5",  # cheap IC critic (config#927)
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

    # Provenance stamps for the run=code+data reproducibility contract (1a
    # schema add → 1b wire-in, #781). Both are best-effort and degrade to a
    # None/sentinel stamp rather than crashing the capture (feedback_no_
    # silent_fails — record the gap, don't drop the artifact):
    #   * data_snapshot_id — the ArcticDB price-data version threaded from
    #     fetch_data (``fetch_price_data(return_snapshot_id=True)``). Absent
    #     from state on resume paths that skip fetch_data → "unknown".
    #   * code_sha — the deployed image's git SHA, stamped at build time into
    #     ``ALPHA_ENGINE_CODE_SHA`` (GHA ``--build-arg GIT_SHA`` → Dockerfile
    #     ENV; manual ``deploy.sh`` stamps ``git rev-parse HEAD``). None when
    #     the env var is unset (local/dev) — the artifact still lands.
    data_snapshot_id = state.get("data_snapshot_id") or "unknown"
    code_sha = os.environ.get("ALPHA_ENGINE_CODE_SHA") or None

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
            code_sha=code_sha,
            data_snapshot_id=data_snapshot_id,
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
    # Run-level ArcticDB price-data snapshot version (``VersionedItem.version``)
    # surfaced by ``fetch_price_data(return_snapshot_id=True)``. Threaded into
    # every captured DecisionArtifact's ``data_snapshot_id`` so each research
    # decision records exactly which immutable price snapshot it was computed
    # on (reproducibility/provenance — L4567 sub-item 1b / #781). ``"unknown"``
    # when no versioned read occurred (full feature-store coverage, or a
    # non-versioned backend). Set once in fetch_data; never mutated downstream.
    data_snapshot_id: Annotated[str, take_last]
    technical_scores: Annotated[dict[str, dict], take_last]
    # Per-name Barra factor loadings (momentum_20d_zscore / return_60d_zscore /
    # beta_60d_zscore / size_zscore) read from the feature store's
    # factor_loading group in fetch_data. Consumed by the score-neutralization
    # OBSERVE shadow in archive_writer (config#1142). Empty when the loadings
    # group hasn't shipped a snapshot — the shadow is fail-soft on absence.
    factor_loadings: Annotated[dict[str, dict[str, float]], take_last]
    scanner_universe: Annotated[list[str], take_last]
    # Sector-team screening input (L1995 Phase 5 / L4464): the standalone
    # Scanner SF state's candidate set (candidates.json::scanner_tickers)
    # ∪ the held population. Replaces the raw-~900-by-sector handoff that
    # overran the Lambda recursion budget (each quant ReAct agent hit
    # recursion_limit on 92-217 tickers → 0 picks → retry storm → 900s
    # timeout). ``scanner_universe`` stays the FULL universe for the
    # exit_evaluator constituent whitelist.
    agent_input_set: Annotated[list[str], take_last]
    # The scanner gate's per-ticker verdict detail (quant_filter_pass /
    # filter_fail_reason / scan_path / liquidity_pass / volatility_pass),
    # sourced from candidates.json::scanner_eval_log in fetch_data
    # (cross-process safe: candidates.json is the S3 artifact the standalone
    # Scanner SF state writes, already read here via am.load_candidates_json)
    # rather than the in-process ``run_quant_filter._last_eval_log`` module
    # stash. That stash is empty in THIS Lambda — Research no longer calls
    # run_quant_filter itself post-L1995-Phase5; only the standalone Scanner
    # SF state does, in its own process (config#1458: the stash read in
    # archive_writer was always empty by construction, degrading
    # quant_filter_pass to 0 for 100% of rows every cycle). Empty list when
    # candidates.json carries none (absent field, or the dry-run-stub path).
    scanner_eval_log: Annotated[list, take_last]
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

    # Prior-cycle realized-outcomes scorecard (Phase 2.A.3 of the
    # research-feedback sidecar arc). Loaded by ``load_scorecard_node``
    # between regime substrate and macro economist; consumed by
    # ``macro_economist_node`` AND ``cio_node`` via the
    # ``prior_cycle_scorecard`` kwarg the agents accept. None / empty
    # is graceful — agents fall back to pre-Phase-2 behavior. Brian's
    # gitignored prompt-template edit gates whether the LLM actually
    # sees the scorecard text; until then the kwarg is silently unused
    # by ``str.format``. Mirrors the regime_substrate pattern above.
    prior_cycle_scorecard_text: Annotated[Optional[str], take_last]

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

    # ── Checkpoint-resume evidence (config#2263) ─────────────────────────────
    # Populated ONLY on the resume-hit branch of the three checkpoint-capable
    # nodes (sector_team_node, macro_economist_node, cio_node) — never on the
    # fresh-compute path. Keyed so the trajectory validator (evals/trajectory.py)
    # can tell, from final_state alone, whether this invocation used the S3
    # checkpoint short-circuit at all (a non-empty dict here means at least
    # one node was resumed rather than executed, which matters because a
    # resumed run can finish before its child spans land in LangSmith — see
    # the trajectory validator's final-state structural fallback).
    # ``merge_typed_dicts`` (last-write-wins, dict union) rather than
    # ``reject_on_conflict``: this is bookkeeping, not a correctness
    # invariant like sector_team_outputs' disjoint keyspace, so a duplicate
    # write here should never crash a run. sector_team_node's 6 concurrent
    # Send() branches each own a distinct key (``sector_team_node:{team_id}``)
    # so there is no legitimate overlap in practice either.
    checkpoint_resumed_nodes: Annotated[dict[str, bool], merge_typed_dicts]


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

class AgentInputSetResolution(NamedTuple):
    """Return shape of :func:`_resolve_agent_input_set`.

    ``scanner_eval_log`` rides along with ``agent_input_set`` because both
    are sourced from the SAME ``am.load_candidates_json(run_date)`` call —
    returning them together avoids a second S3 round-trip from ``fetch_data``.
    """
    agent_input_set: list[str]
    scanner_eval_log: list[dict]


def _resolve_agent_input_set(
    am: "ArchiveManager",
    run_date: str,
    scanner_universe: list[str],
    population_tickers: list[str],
) -> AgentInputSetResolution:
    """Resolve the sector-team screening input (L1995 Phase 5 / L4464).

    The standalone Scanner SF state (run upstream of Research) writes
    ``candidates/{run_date}/candidates.json`` (903 → ~60 quant-filtered via
    ``run_quant_filter`` — the same primitive Research used to embed inline).
    Feed the sector teams that pre-filtered set ∪ the held population instead
    of the raw ~900-by-sector slice that overran the Lambda recursion budget
    (each quant ReAct agent hit recursion_limit on 92-217 tickers → 0 picks →
    retry storm → 900s timeout). ``scanner_universe`` is retained by the
    caller for the exit_evaluator constituent whitelist.

    The held population is sourced from Research's own state
    (``population_tickers``), NOT ``candidates.json::population_tickers`` which
    is cold-start-empty (it depends on the prior signals.json).

    Fail-loud: a missing/empty ``candidates.json`` raises. The Scanner SF
    state runs unconditionally upstream of Research (L1995 Phase 3, post-#338),
    so absence is a real upstream failure — NOT a soft fallback to the raw
    ~900 universe (that path is exactly what overruns the budget, L4464).
    The ``ALPHA_ENGINE_DRY_RUN_STUB`` sentinel (set only by the stub/offline
    installers) relaxes this to a full-universe fallback for wiring validation;
    production never sets it.

    Also returns ``scanner_eval_log`` — ``candidates.json``'s per-ticker
    scanner gate verdict (config#1458), read here (rather than a second call
    to ``am.load_candidates_json``) since this function already loads the
    artifact. Empty on the dry-run-stub fallback path (no real candidates.json
    was read) or when the artifact predates this field.
    """
    import os as _os
    candidates = am.load_candidates_json(run_date)
    scanner_tickers = (candidates or {}).get("scanner_tickers") or []
    scanner_eval_log = (candidates or {}).get("scanner_eval_log") or []
    if not scanner_tickers:
        if _os.environ.get("ALPHA_ENGINE_DRY_RUN_STUB", "").lower() == "true":
            logger.warning(
                "[fetch_data] dry-run stub: no candidates.json for %s — "
                "falling back to full scanner_universe for wiring validation "
                "(NOT a real candidate selection)", run_date,
            )
            scanner_tickers = scanner_universe
            scanner_eval_log = []
        else:
            raise RuntimeError(
                f"[fetch_data] candidates.json missing or empty scanner_tickers "
                f"for run_date={run_date} (key candidates/{run_date}/candidates.json). "
                f"The standalone Scanner SF state must run + produce candidates "
                f"upstream of Research (L1995). Refusing to fall back to the raw "
                f"~900 universe (that path overruns the Lambda budget, L4464)."
            )
    agent_input_set = sorted(set(scanner_tickers) | set(population_tickers))
    logger.info(
        "[fetch_data] sector-team input set: %d tickers "
        "(scanner %d ∪ held population %d)",
        len(agent_input_set), len(scanner_tickers), len(population_tickers),
    )
    return AgentInputSetResolution(agent_input_set, scanner_eval_log)


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
        from nousergon_lib.rag import is_available as _rag_is_available
        rag_available = _rag_is_available()
        logger.info("[fetch_data] RAG database: %s", "available" if rag_available else "UNAVAILABLE")
        # Reset per-run RAG stats
        from agents.sector_teams.qual_tools import reset_rag_stats
        reset_rag_stats()
    except Exception as e:
        logger.warning("[fetch_data] RAG availability check failed: %s", e)

    # Reset the FMP 402 circuit breaker for this run (module-level state — see
    # data/fetchers/analyst_fetcher.py). Without this, a container reused
    # across Lambda invocations would keep an endpoint tripped from a prior
    # run's 402 forever, silently starving a run where the plan issue may
    # have since been fixed.
    from data.fetchers.analyst_fetcher import reset_fmp_402_breaker
    reset_fmp_402_breaker()

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

    # ── L1995 Phase 5 cutover: sector-team screening input ───────────────────
    agent_input_set, scanner_eval_log = _resolve_agent_input_set(
        am, run_date, scanner_universe, population_tickers,
    )

    # ── team_inputs ledger (v19): record the scanner→team input assignment ───
    # The per-team partition is otherwise computed in-memory inside each
    # sector_team_node and discarded. Persisting it here — once, deterministically,
    # for ALL teams, before fan-out — lets the decision-review console show the
    # complete input set per team (not just the names a team ended up ranking),
    # and survives SF resume (fetch_data always runs). Best-effort: a ledger
    # write failure must never sink the research run.
    try:
        _held = set(population_tickers)
        _team_inputs: list[dict] = []
        for _tid in ALL_TEAM_IDS:
            for _tkr in get_team_tickers(_tid, agent_input_set, sector_map):
                _team_inputs.append({
                    "ticker": _tkr,
                    "eval_date": run_date,
                    "team_id": _tid,
                    "source": "held_population" if _tkr in _held else "scanner",
                    "sector": sector_map.get(_tkr),
                })
        am.write_team_inputs(_team_inputs)
        am.commit()
        logger.info("[fetch_data] team_inputs ledger: %d assignments across %d teams",
                    len(_team_inputs), len(ALL_TEAM_IDS))
    except Exception as e:  # pragma: no cover — observability ledger, never fatal
        logger.warning("[fetch_data] team_inputs ledger write failed (non-fatal): %s", e)

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
                    # Scanner's absolute-liquidity gate compares against
                    # MIN_AVG_VOLUME (raw shares). The bare avg_volume_20d
                    # column is the predictor's normalized ratio (~1.0);
                    # avg_volume_20d_raw is the raw-shares column added in
                    # alpha-engine-data Phase 1 of the schema audit. See
                    # alpha-engine-data/features/SCHEMA.md.
                    "avg_volume_20d": fs_row.get("avg_volume_20d_raw"),
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

    # ── Barra factor loadings for the neutralization OBSERVE shadow ──────────
    # The *_zscore loadings live in the feature store's separate
    # ``factor_loading.parquet`` group (NOT technical.parquet), so they need
    # their own read. Consumed only by the config#1142 OBSERVE shadow in
    # archive_writer — fail-soft, empty dict when the group has no snapshot yet.
    factor_loadings: dict[str, dict[str, float]] = {}
    try:
        from data.fetchers.feature_store_reader import read_latest_factor_loadings
        factor_loadings = read_latest_factor_loadings() or {}
        if factor_loadings:
            logger.info("[fetch_data] factor loadings: %d tickers loaded", len(factor_loadings))
    except Exception as e:
        logger.debug("[fetch_data] factor loadings not available: %s", e)

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
    if ohlcv_tickers:
        # return_snapshot_id surfaces the ArcticDB ``VersionedItem.version``
        # the read resolved to → threaded into decision capture as the
        # run-level ``data_snapshot_id`` provenance stamp (L4567 1b / #781).
        price_data, data_snapshot_id = fetch_price_data(
            ohlcv_tickers, period="3mo", return_snapshot_id=True,
        )
    else:
        # No OHLCV read this run (full feature-store coverage) → no ArcticDB
        # version to stamp. Record the sentinel, never crash on absence.
        from data.fetchers.price_fetcher import DATA_SNAPSHOT_ID_UNKNOWN
        price_data, data_snapshot_id = {}, DATA_SNAPSHOT_ID_UNKNOWN
    logger.info("[fetch_data] data_snapshot_id=%s", data_snapshot_id)

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
            # compute_technical_indicators returns None for insufficient
            # history (df.empty or len(df) < 30). The >= 20 guard above admits
            # tickers with 20–29 rows, so honour the documented Optional[dict]
            # sentinel and skip them — same "insufficient data → not technically
            # scored" outcome as the <20-row tickers and as data/scanner.py /
            # local/time_scanner.py. Without this, a 20–29-row ticker makes
            # compute_technical_score(None) crash on indicators.get(...).
            if indicators is None:
                continue
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

    # Surface the FMP 402 circuit-breaker skip counts in the run summary log
    # (config#1821). The breaker itself already logs one WARN per endpoint
    # at trip time; this is the run-level counter so a known-dead endpoint
    # shows up in the summary rather than as a silent data hole.
    from data.fetchers.analyst_fetcher import fmp_402_skip_counts
    _402_skips = fmp_402_skip_counts()
    if _402_skips:
        logger.info("[fetch_data] FMP 402 circuit breaker skips: %s", _402_skips)

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
        "agent_input_set": agent_input_set,
        "scanner_eval_log": scanner_eval_log,
        "sector_map": sector_map,
        "price_data": price_data,
        "data_snapshot_id": data_snapshot_id,
        "technical_scores": technical_scores,
        "factor_loadings": factor_loadings,
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


def load_scorecard_node(state: ResearchState) -> dict:
    """Load the prior-cycle realized-outcomes scorecard from S3.

    Phase 2.A.3 of the research-feedback sidecar arc. Reads the
    canonical eval_artifacts sidecar at
    ``research/last_week_scorecard/latest.json`` and renders the
    prompt-ready text via ``format_scorecard_text``. Returns
    ``{"prior_cycle_scorecard_text": ""}`` gracefully when:

    - The producer (``_maybe_emit_scorecard`` in ``lambda/handler.py``)
      is flag-OFF (``RESEARCH_SCORECARD_ENABLED`` unset / false) —
      default state pre-soak.
    - The producer's flag-on state hasn't yet written any artifact
      (first Saturday SF after flip).
    - The S3 read or JSON parse fails transiently.

    Empty string flows through to ``macro_economist_node`` AND
    ``cio_node`` via the ``prior_cycle_scorecard`` kwarg; ``str.format``
    silently fills the ``{prior_cycle_scorecard}`` placeholder with
    "" (or ignores the kwarg entirely until Brian's gitignored
    template edit lands). Mirrors ``load_regime_substrate_node``'s
    graceful-degrade posture.
    """
    bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    try:
        import boto3
        from evals.last_week_scorecard import load_latest_scorecard_text
        text = load_latest_scorecard_text(
            s3_client=boto3.client("s3"),
            bucket=bucket,
        )
        if text:
            logger.info(
                "[load_scorecard] loaded prior-cycle scorecard (%d chars)",
                len(text),
            )
        else:
            logger.info(
                "[load_scorecard] no scorecard artifact found "
                "(producer flag off or first cycle) — agents will run "
                "without prior-cycle outcome data",
            )
        return {"prior_cycle_scorecard_text": text}
    except Exception as e:
        # Mirror load_regime_substrate_node's graceful posture — never
        # fail the research cycle on a missing observability artifact.
        logger.warning(
            "[load_scorecard] load failed: %s — agents will run without "
            "prior-cycle outcome data",
            e,
        )
        return {"prior_cycle_scorecard_text": ""}


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
            return {
                "sector_team_outputs": {team_id: persisted},
                "checkpoint_resumed_nodes": {f"sector_team_node:{team_id}": True},
            }

    # Stage D' Wire 1: extract intensity_z from the regime substrate
    # (loaded by load_regime_substrate_node upstream of macro). None
    # when substrate hasn't published yet — peer_review's gate degrades
    # to base threshold only.
    _substrate = state.get("regime_substrate") or {}
    _intensity_z = (_substrate.get("composite") or {}).get("intensity_z")

    ctx = SectorTeamContext(
        scanner_universe=state.get("scanner_universe", []),
        agent_input_set=state.get("agent_input_set", []),
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
        # Thread the run_id so the per-sub-agent track_llm_cost scopes
        # opened inside run_sector_team partition their cost-raw JSONL
        # under the same run_id the captures below use (derive_run_id).
        run_id=derive_run_id(state),
    )
    # Cost telemetry is scoped PER SUB-AGENT inside run_sector_team — one
    # track_llm_cost frame each for sector_quant / sector_qual /
    # sector_peer_review / thesis_update:{ticker}, keyed to match the
    # _capture_if_enabled agent_ids below. The CostTelemetryCallback on
    # each ChatAnthropic accumulates into whichever inner frame is active
    # (top of the frame stack), so pop_metadata_for(sector_quant:…) etc.
    # now returns real token counts + cost instead of the placeholder
    # fallback. The legacy single sector_team:{team_id} frame here mixed
    # all four sub-agents' tokens under one agent_id the captures never
    # read — config#1037's root cause.
    result = run_sector_team(team_id, ctx)

    # Schema validation — strict-by-default (raises RuntimeError on
    # validation failure unless STRICT_VALIDATION=false).
    _validate(SectorTeamOutput, result, context=f"sector_team:{team_id}")

    # Decision-artifact capture (gated on ALPHA_ENGINE_DECISION_CAPTURE_ENABLED).
    # Per-sub-agent captures so LLM-as-judge eval can score quant + qual
    # independently. Each agent_id below now matches a per-sub-agent
    # track_llm_cost scope opened inside run_sector_team (config#1037), so
    # pop_metadata_for(...) returns real token counts + cost + the stamped
    # prompt id/version instead of the placeholder fallback.
    team_tickers = get_team_tickers(team_id, ctx.agent_input_set, ctx.sector_map)
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
    # can fail and ERROR the overall run. A team is persisted iff it is
    # FULLY COMPLETE: no hard ``error`` AND not ``partial``. An errored or
    # partial (e.g. qual step-budget-exhausted) team is NOT persisted, so a
    # re-run gets a fresh attempt at it — the backoff / a TPM-window reset /
    # the workload-sized ReAct budget (config#1822) may let it succeed.
    # Persisting a partial team would poison every future rerun via the
    # resume short-circuit (``load_sector_team_run``). ``save_sector_team_run``
    # enforces the same guard defensively.
    if _am is not None and not result.get("error") and not result.get("partial"):
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
            return {
                **_persisted,
                "checkpoint_resumed_nodes": {"macro_economist_node": True},
            }

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
    ) as _macro_frame:
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
            # Phase 2.A.3: load_scorecard_node populates this state
            # field with the prior cycle's realized-outcomes scorecard
            # text. Empty string flows through when the producer's
            # flag is off or the artifact is missing; the agent's
            # template renders the placeholder as blank in that case.
            prior_cycle_scorecard=state.get("prior_cycle_scorecard_text"),
        )
        # config#1753: thread the actually-rendered primary-agent prompt
        # (what was handed to HumanMessage(...) inside run_macro_agent)
        # onto the frame so FullPromptContext.user_prompt captures the
        # substituted text instead of the raw LoadedPrompt template.
        _macro_frame.rendered_prompt = result.get("rendered_prompt")

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


def _load_metron_supplemental_sectors(run_date: str, bucket: str | None = None) -> dict[str, str]:
    """Metron-held/watchlisted tickers outside the S&P500+400 universe get their GICS
    sector from the sidecar alpha-engine-data writes alongside its supplemental
    factor-scoring snapshot (metron-ops#177). Optional and fail-soft — absent on any
    run where that producer found nothing to add, or hasn't shipped yet; never blocks
    ``compute_factor_profiles_node``."""
    import boto3

    bucket = bucket or os.environ.get("S3_BUCKET", "alpha-engine-research")
    key = f"features/metron_supplemental/{run_date}/sectors.json"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read()).get("sectors", {})
    except Exception as e:  # noqa: BLE001 - genuinely optional artifact, never raise
        logger.info("[compute_factor_profiles] no Metron supplemental sectors for %s (%s)", run_date, e)
        return {}


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

    # Additive only (never overrides an S&P500+400/population sector already in
    # sector_map) — metron-ops#177.
    supplemental_sectors = _load_metron_supplemental_sectors(run_date)
    sector_map = {**sector_map, **{t: s for t, s in supplemental_sectors.items() if t not in sector_map}}

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


def rank_candidates_by_attractiveness_node(state: ResearchState) -> dict:
    """CHAMPION-FEED CUT (config#1400 / ARCHITECTURE §43): re-select the
    sector-team candidate feed by ranking the scanned universe on the live
    6-pillar attractiveness composite, REPLACING the momentum-only tech_score
    gate's ~60 with the top-N attractiveness names.

    WHY: the live tech_score gate has 3.9% 21d recall (2026-06-29 e2e_lift) —
    a binary gate that amputates ~96% of eventual winners before the agents see
    them. Ranking the scanned universe by attractiveness (the measured +0.91%
    sector-neutral lift at matched N) feeds the agents a better-selected pool.
    Spliced AFTER ``compute_factor_profiles_node`` (which wrote the 6-pillar
    profiles this run) and BEFORE dispatch, so attractiveness is computable.

    Gated by ``ATTRACTIVENESS_FEED_ENABLED`` (default OFF — flipping ON is the
    reversible cut). The held population is ALWAYS retained (agents must still
    evaluate current holdings for HOLD/EXIT). FAIL-SAFE: any error returns ``{}``
    so the existing tech_score ``agent_input_set`` stands — the cut can never
    break the research run. The tech_score ``candidates.json`` remains the
    shadow baseline for the realized-alpha revert signal.
    """
    if not ATTRACTIVENESS_FEED_ENABLED:
        return {}
    run_date = state.get("run_date")
    try:
        from scoring.universe_board import (
            _read_factor_profiles,
            attractiveness_from_factor_profiles,
        )

        profiles = _read_factor_profiles(run_date, None, None) or {}
        if not profiles:
            raise RuntimeError(
                f"attractiveness feed: no factor profiles readable for {run_date}"
            )
        scores = attractiveness_from_factor_profiles(profiles)
        ranked = sorted(
            (t for t, v in scores.items() if v.get("attractiveness_score") is not None),
            key=lambda t: scores[t]["attractiveness_score"],
            reverse=True,
        )
        top_n = int(ATTRACTIVENESS_FEED_TOP_N)
        selected = ranked[:top_n]
        if not selected:
            raise RuntimeError("attractiveness feed: no ranked candidates produced")
        population_tickers = state.get("population_tickers") or []
        new_set = sorted(set(selected) | set(population_tickers))
        prior = state.get("agent_input_set") or []
        logger.info(
            "[attractiveness_feed] CHAMPION feed ON: %d tickers "
            "(top-%d attractiveness ∪ held population %d) — replacing tech_score "
            "feed of %d. Scored %d of %d profiled names.",
            len(new_set), top_n, len(population_tickers), len(prior),
            len(ranked), len(profiles),
        )
        return {"agent_input_set": new_set}
    except Exception as e:  # FAIL-SAFE: keep the existing tech_score feed
        logger.warning(
            "[attractiveness_feed] FAILED — falling back to existing tech_score "
            "candidate feed (the cut is inert this run): %s", e,
        )
        return {}


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
        factor_profiles, market_regime, get_factor_blend_regime_weights(),
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

    # Adaptive slot allocation (config#926): when enabled, weight each team's
    # eligible-pick count by its historical accuracy (loaded from the
    # backtester's team_accuracy artifact). Default OFF / artifact absent →
    # team_accuracy is None and compute_team_slots is byte-identical to static.
    team_accuracy = None
    if ADAPTIVE_SLOT_ALLOCATION_ENABLED:
        am = state.get("archive_manager")
        if am is not None:
            try:
                team_accuracy = am.load_team_accuracy()
            except Exception as e:  # pragma: no cover — loader is best-effort
                logger.warning("[merge] team_accuracy load failed: %s", e)
        logger.info("[merge] adaptive slots ON — team_accuracy=%s", team_accuracy)

    team_slot_allocation = compute_team_slots(
        open_slots, sector_ratings, team_accuracy=team_accuracy
    )

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
    pillar_coverage_skip_count = 0

    # PillarCoverageError raised by compute_composite_breakdown when config
    # has Σ pillar_weights > 0 but a ticker has no pillar inputs. Per-ticker
    # policy here is skip-with-WARN-and-counter — run completes with partial
    # signals; the existing pillar-distribution sanity check + Telegram
    # alert surface the aggregate. Whole-run hard-fail is the alternative
    # (matches all-agents-strict) but produces nothing-of-value when a
    # single ticker's pillar emission fails. Codified after the 2026-05-21
    # AQR cutover incident; closes the consumer-side silent-fail.
    from scoring.composite import PillarCoverageError

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
                    regime_weights=get_factor_blend_regime_weights(),
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

            try:
                breakdown = compute_composite_breakdown(
                    quant_score=rec.get("quant_score"),
                    qual_score=rec.get("qual_score"),
                    factor_subscore=factor_subscore_val,
                    pillar_assessment=ticker_pillar_assessment,
                    factor_profile=ticker_factor_profile,
                    sector_modifier=modifier,
                    pillar_weights=PILLAR_COMPOSITE_WEIGHTS,
                    within_pillar_qual_weight=PILLAR_COMPOSITE_WITHIN_PILLAR_QUAL_WEIGHT,
                    legacy_blend_weights=PILLAR_COMPOSITE_LEGACY_BLEND,
                    macro_overlay_enabled=MACRO_OVERLAY_ENABLED,
                    macro_max_shift_points=MACRO_MAX_SHIFT_POINTS,
                    macro_modifier_range=MACRO_MODIFIER_RANGE,
                )
            except PillarCoverageError as e:
                pillar_coverage_skip_count += 1
                logger.warning(
                    "[score_aggregator] PillarCoverageError for %s/%s — "
                    "skipping signal: %s",
                    team_id, ticker, e,
                )
                continue

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
                # Quality pillar's qualitative core — plumbed through 2026-05-22
                # so `_check_pillar_distribution_sanity` can run the
                # `moat_collapse` check (>95% primary_type=="none" means the
                # moat rubric has degraded). `None` when ticker_pillar_assessment
                # is absent (PILLAR_EMIT off / extraction skipped). Future
                # archive/universe/{TICKER}/moat_profile.json time-series
                # persistence reads the same field.
                "quality_moat": (
                    ticker_pillar_assessment.get("quality_moat")
                    if ticker_pillar_assessment else None
                ),
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
                        macro_overlay_enabled=MACRO_OVERLAY_ENABLED,
                        macro_max_shift_points=MACRO_MAX_SHIFT_POINTS,
                        macro_modifier_range=MACRO_MODIFIER_RANGE,
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
        "(pillar_assessment_applied=%d of %d, "
        "pillar_coverage_skipped=%d) — Phase 4 of "
        "attractiveness-pillars-260520",
        len(investment_theses),
        pillar_assessment_applied_count,
        len(investment_theses),
        pillar_coverage_skip_count,
    )
    # Emit a Telegram alert when coverage-skip count is non-zero — the
    # hardening Item 1 surface for the 2026-05-21 incident class. Best-
    # effort, secondary observability per the new CLAUDE.md fail-loud
    # rule's clause (i); the WARN logs per-ticker above + CW Logs alarm
    # path remain load-bearing.
    if pillar_coverage_skip_count > 0:
        try:
            from ops_alerts import publish_ops_alert

            publish_ops_alert(
                message=(
                    f"[score_aggregator] pillar coverage gap — skipped "
                    f"{pillar_coverage_skip_count} signal(s) due to "
                    f"PillarCoverageError (Σ pillar_weights > 0 but pillar "
                    f"inputs absent for these tickers). Run completed with "
                    f"partial signals; investigate qual_analyst pillar emission."
                ),
                severity="WARN",
                source="research:score_aggregator",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[score_aggregator] pillar-coverage-skip Telegram publish "
                "failed: %s (WARN log + CW Logs alarm remain the failure "
                "surface)", e,
            )

    # Pillar-distribution structural sanity check (first-cycle observation
    # gate for the 2026-05-21 AQR-prior cutover — pillar_weights are
    # AQR-seeded non-zero, so any structural collapse in the LLM's pillar
    # emission would materially distort live final_score with zero prior
    # observation data). Per [[feedback_no_silent_fails]] this fires a
    # Telegram alert via the canonical alpha_engine_lib.alerts CLI rather
    # than swallowing — operator needs to see structural pillar failure
    # within minutes of SF completion, not after a week of bad signals.
    #
    # Three structural-failure modes flagged:
    #   * collapsed_pillar: any pillar's std < 5 across all picks → LLM
    #     defaulting to a single value (e.g. all 50s) is not a usable signal.
    #   * low_coverage: <80% of picks have pillar_assessment present → parse
    #     failures higher than expected; AQR weights would zero-treat the
    #     missing pillar component and distort composites.
    #   * moat_collapse: >95% of MoatAssessments have primary_type=="none"
    #     → the moat rubric has degraded to "say none for everything", the
    #     Quality pillar's qualitative core isn't doing its job.
    _check_pillar_distribution_sanity(investment_theses)

    # Schema validation on every produced thesis (strict-by-default).
    for ticker, thesis in investment_theses.items():
        _validate(InvestmentThesis, thesis, context=f"score_aggregator:{ticker}")

    return {"investment_theses": investment_theses}


_PILLAR_NAMES = ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")


def _check_pillar_distribution_sanity(investment_theses: dict) -> None:
    """First-cycle observation gate for AQR-prior cutover (2026-05-21).

    Flags three structural-failure modes via Telegram alert. Best-effort —
    Telegram outage does NOT block research-Lambda completion; alert
    failures WARN-log per [[feedback_no_silent_fails]] secondary-observability
    rationale (the load-bearing surface is the WARN log + this function's
    return-via-raise pathway for the calling node, which there isn't here
    — sanity check is observation-only at Phase 4 cutover, not a hard gate).

    Skip when ``Σ PILLAR_COMPOSITE_WEIGHTS ≈ 0``: at zero pillar weights the
    composite reduces to legacy by construction (see
    ``scoring/composite.py::compute_composite_breakdown``), so empty
    ``pillar_contributions`` are harmless rather than a regression. Firing
    the check here produces false positives in two real situations:

    * **Phase-4-cutover-defaults state** (current live config 2026-05-21
      post-revert of config #260) — all pillar weights are 0; the composite
      uses only legacy components; empty pillar_contributions are by
      design, not a parse failure.
    * **Dry-run / smoke-test paths** (``local/run.py --dry-run``) — the
      ``_stub_run_qual_analyst`` stub does not synthesize ``pillar_assessments``,
      so theses always have empty pillar_contributions. The smoke run is
      not exercising the live LLM extraction path the check measures.

    The check is only meaningful when pillar weights are load-bearing.
    """
    import statistics

    pillar_weights_sum = sum(PILLAR_COMPOSITE_WEIGHTS.values())
    if pillar_weights_sum <= 1e-6:
        logger.debug(
            "[score_aggregator] pillar-distribution sanity skipped "
            "(Σ pillar_weights=%.6f ≈ 0; composite reduces to legacy "
            "so pillar coverage is not load-bearing)",
            pillar_weights_sum,
        )
        return

    n_total = len(investment_theses)
    if n_total == 0:
        return

    # Collect per-pillar scores from each thesis's composite_breakdown
    pillar_scores: dict[str, list[float]] = {p: [] for p in _PILLAR_NAMES}
    n_with_pillar = 0
    moat_types: list[str] = []
    for thesis in investment_theses.values():
        breakdown = thesis.get("composite_breakdown") or {}
        contribs = breakdown.get("pillar_contributions") or []
        if not contribs:
            continue
        n_with_pillar += 1
        for c in contribs:
            pillar = c.get("pillar")
            qual = c.get("qual_component")
            if pillar in pillar_scores and qual is not None:
                pillar_scores[pillar].append(float(qual))
        # Moat collapse check (plumbed 2026-05-22): pull from the
        # thesis's `quality_moat` field — populated by score_aggregator
        # from the qual_analyst's QualitativePillarAssessment. Absent
        # when PILLAR_EMIT is off / extraction skipped; counted only
        # when present so the threshold reflects the live signal, not
        # the legacy path's silence.
        moat = thesis.get("quality_moat") or {}
        primary_type = moat.get("primary_type")
        if primary_type:
            moat_types.append(primary_type)

    coverage_pct = (n_with_pillar / n_total) * 100.0

    alerts: list[str] = []

    if coverage_pct < 80.0:
        alerts.append(
            f"low_coverage: {n_with_pillar}/{n_total} picks "
            f"({coverage_pct:.1f}%) have populated pillar_contributions "
            f"— expected ≥80%. Parse failures higher than predicted."
        )

    for pillar in _PILLAR_NAMES:
        scores = pillar_scores[pillar]
        if len(scores) < 2:
            continue
        std = statistics.pstdev(scores)
        mean = statistics.mean(scores)
        if std < 5.0:
            alerts.append(
                f"collapsed_pillar: {pillar} std={std:.2f} (<5), "
                f"mean={mean:.1f}, n={len(scores)} — LLM rubric defaulted "
                f"to a single value; pillar not contributing real signal."
            )

    # Moat collapse check — fires when the qualitative core of the
    # Quality pillar has degraded to "no moat for everything." Requires
    # ≥5 sampled moat assessments to avoid spurious alerts on tiny pick
    # counts (early in the week or sector-team thin firings).
    if len(moat_types) >= 5:
        none_count = sum(1 for m in moat_types if m == "none")
        none_pct = (none_count / len(moat_types)) * 100.0
        if none_pct > 95.0:
            alerts.append(
                f"moat_collapse: {none_count}/{len(moat_types)} picks "
                f"({none_pct:.1f}%) have moat.primary_type=='none' "
                f"— the Quality pillar's qualitative core has degraded "
                f"to 'say none for everything'; rubric needs review."
            )

    if not alerts:
        logger.info(
            "[score_aggregator] pillar-distribution sanity OK "
            "(coverage=%.1f%%, n=%d)",
            coverage_pct, n_with_pillar,
        )
        return

    # Build a single composed message; emit as WARN here AND push to
    # Telegram via the canonical lib alerts CLI. Best-effort on the
    # Telegram side — research-Lambda completion is the load-bearing
    # signal, not the alert publish status. Secondary-observability
    # swallow per the new CLAUDE.md fail-loud rule's clause (i).
    msg = (
        "[score_aggregator] AQR-prior cutover sanity FAIL "
        f"({len(alerts)} issue(s)): " + " | ".join(alerts)
    )
    logger.warning(msg)
    # Dedup the publish on (run_date, coverage_bucket, n_failures). A real
    # cutover-bad day would otherwise spam the channel on every
    # research-Lambda invocation across the SF (canary + main + any
    # rebroadcasts) — N alerts for one incident, same defect class as
    # the 2026-05-21 alert-storm thread that motivated the v0.24.0 lib
    # dedup substrate. Lib v0.24.0's default 60-min window collapses
    # within-hour bursts; we omit the window override (60 min default
    # is right for an hourly research cadence — same incident a second
    # day re-fires intentionally).
    run_date = _resolve_run_date_for_dedup(investment_theses)
    coverage_bucket = f"{int(coverage_pct // 10) * 10}"  # 10-pct bucket
    dedup_key = (
        f"pillar_sanity_{run_date}_cov{coverage_bucket}_n{len(alerts)}"
    )
    try:
        from ops_alerts import publish_ops_alert

        publish_ops_alert(
            message=msg,
            severity="WARN",
            source="research:score_aggregator",
            dedup_key=dedup_key,
        )
    except Exception as e:  # noqa: BLE001
        # Secondary observability — alert-publish failure logs but does not
        # block research-Lambda completion. The WARN log above + the
        # SNS/CW-Logs alarm path are the load-bearing surfaces.
        logger.warning(
            "[score_aggregator] pillar-sanity Telegram publish failed: %s "
            "(WARN log + CW Logs alarm remain the failure surface)", e,
        )


def _resolve_run_date_for_dedup(investment_theses: dict) -> str:
    """Best-effort run_date for the pillar-sanity dedup key. Pulls from
    the first thesis's ``run_date`` metadata when present, otherwise
    falls back to today's UTC date. Same-day rebroadcast within the
    60-min default window collapses to one alert; a fresh next-day
    incident re-fires.
    """
    for thesis in investment_theses.values():
        rd = thesis.get("run_date")
        if rd:
            return str(rd)
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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
            return {
                **_persisted,
                "checkpoint_resumed_nodes": {"cio_node": True},
            }

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

    # De-blended CIO orchestration (L4564): when the flag is ON, compute the
    # deterministic sector-neutral stock-quality map (the rubric's per-sector
    # bias stripped via the trailing-sector baseline from research.db) so the
    # CIO ranks on apples-to-apples quality and weighs the sector tilt as a
    # SEPARATE lever. Default OFF → run_cio uses the legacy prompt + raw
    # scores, byte-identical to before (inert merge).
    _deblended = CIO_DEBLENDED_ORCHESTRATION
    _sector_neutral_quality = None
    if _deblended:
        _am = state.get("archive_manager")
        _db_conn = getattr(_am, "db_conn", None) if _am else None
        _sector_neutral_quality = build_sector_neutral_quality_map(
            candidates,
            state.get("investment_theses", {}),
            _db_conn,
            state.get("run_date", ""),
        )
        logger.info(
            "[cio] de-blend ON — sector-neutral quality computed for %d/%d "
            "candidates", len(_sector_neutral_quality), len(candidates),
        )
    _cio_prompt_name = _PROMPT_DEBLENDED if _deblended else _PROMPT_DEFAULT

    _macro_context = {
        "market_regime": state.get("market_regime", "neutral"),
        "macro_report": state.get("macro_report", ""),
    }
    _cio_call_kwargs = dict(
        candidates=candidates,
        macro_context=_macro_context,
        sector_ratings=state.get("sector_ratings", {}),
        current_population=state.get("remaining_population", []),
        open_slots=state.get("open_slots", 0),
        exits=state.get("exits", []),
        run_date=state.get("run_date", ""),
        prior_decisions=prior_ic,
        max_new_entrants=CIO_MAX_NEW_ENTRANTS,
        min_new_entrants=CIO_MIN_NEW_ENTRANTS,
        force_fill_conviction_floor=CIO_FORCE_FILL_CONVICTION_FLOOR,
        # Phase 2.A.3: scorecard text loaded upstream by
        # load_scorecard_node. Empty string when producer's flag
        # is off / artifact missing — CIO falls back to pre-Phase-2
        # behavior (no prior-cycle outcome data in its prompt).
        prior_cycle_scorecard=state.get("prior_cycle_scorecard_text"),
        deblended=_deblended,
        sector_neutral_quality=_sector_neutral_quality,
    )

    _cio_reflection_log = None
    # Cost-telemetry scope wraps the CIO Anthropic call(s). When the IC critic
    # (config#927) is enabled, run_cio_with_reflection may add one Haiku critic
    # call + one extra CIO call inside the same cost scope. Default OFF →
    # byte-identical to the prior single run_cio path (inert merge).
    with track_llm_cost(
        agent_id="ic_cio",
        node_name="cio_node",
        run_type="weekly_research",
        run_id=derive_run_id(state),
        prompt=load_prompt(_cio_prompt_name),
    ) as _cio_frame:
        if CIO_CRITIC_ENABLED:
            cio_result, _cio_reflection_log = run_cio_with_reflection(
                **_cio_call_kwargs
            )
            logger.info(
                "[cio] IC critic reflection: action=%s flagged=%s "
                "advanced %d→%d",
                _cio_reflection_log.get("critic_action"),
                _cio_reflection_log.get("flagged_tickers"),
                len(_cio_reflection_log.get("initial_advanced", [])),
                len(_cio_reflection_log.get("final_advanced", [])),
            )
        else:
            cio_result = run_cio(**_cio_call_kwargs)
        # config#1753: thread the actually-rendered CIO prompt (what was
        # handed to HumanMessage(...) inside run_cio) onto the frame so
        # FullPromptContext.user_prompt captures the substituted text
        # instead of the raw LoadedPrompt template.
        _cio_frame.rendered_prompt = cio_result.get("rendered_prompt")

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

    # ── New-entrant tripwire (L4532) ──────────────────────────────────────
    # A 0-add / below-floor week is DEFENSIBLE (the CIO correctly rejecting a
    # weak or saturated fresh slate) but must be VISIBLE, not silently inferred.
    # Emit the net-new count + breach flag to CloudWatch and log a loud WARN
    # with the why (fresh-slate max conviction vs the entrant bar). Best-effort
    # — observability must never fail the run (per feedback_no_silent_fails).
    try:
        from graph.agent_telemetry import emit_new_entrant_tripwire

        _held = {
            p.get("ticker")
            for p in (state.get("remaining_population") or [])
            if p.get("ticker")
        }
        _decisions = cio_result.get("decisions", [])
        _net_new = [t for t in cio_result.get("advanced_tickers", []) if t not in _held]
        _fresh = [d for d in _decisions if d.get("ticker") not in _held]
        _fresh_convs = [
            d.get("conviction") for d in _fresh
            if isinstance(d.get("conviction"), (int, float))
        ]
        _fresh_max = max(_fresh_convs) if _fresh_convs else None
        emit_new_entrant_tripwire(
            net_new_entrants=len(_net_new),
            alert_floor=CIO_NEW_ENTRANT_ALERT_FLOOR,
            fresh_slate_max_conviction=_fresh_max,
        )
        if len(_net_new) < CIO_NEW_ENTRANT_ALERT_FLOOR:
            logger.warning(
                "[cio] NEW-ENTRANT TRIPWIRE: %d net-new entrant(s) this week "
                "(floor=%d). Fresh slate: %d candidate(s), max conviction %s vs "
                "entrant bar %.0f. Defensible if the slate is genuinely weak "
                "(saturation) — flagged for review. net-new=%s",
                len(_net_new), CIO_NEW_ENTRANT_ALERT_FLOOR, len(_fresh),
                f"{_fresh_max:.0f}" if _fresh_max is not None else "n/a",
                CIO_FORCE_FILL_CONVICTION_FLOOR, _net_new,
            )
        else:
            logger.info(
                "[cio] new-entrant check OK: %d net-new entrant(s) (floor=%d)",
                len(_net_new), CIO_NEW_ENTRANT_ALERT_FLOOR,
            )
    except Exception as _e:  # pragma: no cover — telemetry must not fail the run
        logger.warning("[cio] new-entrant tripwire failed (non-fatal): %s", _e)

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

    final_pop, events = apply_ic_entries(
        remaining_population=state.get("remaining_population", []),
        ic_decisions=state.get("ic_decisions", []),
        entry_theses=state.get("entry_theses", {}),
        sector_map=state.get("sector_map", {}),
        run_date=state.get("run_date", ""),
        target_size=POPULATION_CFG.get("target_size", 25),
    )

    # L4534: replacement-aware swaps come back tagged FORCED_ROTATION. Route
    # them into the exits channel so _build_signals_payload (archive node, runs
    # after this) emits EXIT/sell signals for the swapped-out names — otherwise
    # the executor would never sell them (population/executor divergence).
    swap_exits = [e for e in events if e.get("type") == "FORCED_ROTATION"]
    entry_events = [e for e in events if e.get("type") != "FORCED_ROTATION"]
    all_exits = state.get("exits", []) + swap_exits

    return {
        "new_population": final_pop,
        "population_rotation_events": all_exits + entry_events,
        "exits": all_exits,
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

    # CIO advances (ADVANCE + ADVANCE_FORCED — see ADVANCE_DECISIONS)
    for d in state.get("ic_decisions", []):
        if d.get("decision") in ADVANCE_DECISIONS:
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
    agent_override, override_team_id}}`` for every ticker the factor
    substrate has a profile for. ``focus_list_passed=1`` for top-N members.
    ``agent_override=1`` when the quant agent looked up this non-focus ticker
    via @tool get_factor_profile during its team's run; ``override_team_id``
    then names WHICH team's agent reached out (config#750 per-team override
    attribution) — ``None`` for focus-list members and non-override rows.
    Because sector teams partition tickers by sector, an override ticker is
    overridden by at most one team; if the same ticker somehow appears in more
    than one team's override set, attribution is deterministic (sorted-first
    team wins).

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
        # Per-team override attribution (config#750): keep the OWNING team for
        # each override ticker instead of unioning across teams into an
        # anonymous set. Iterate teams in sorted order so that in the
        # (structurally impossible — sectors partition tickers) event the same
        # ticker appears in two teams' override sets, attribution is
        # deterministic (sorted-first team wins) rather than dict-order-dependent.
        override_team_by_ticker: dict[str, str] = {}
        for team_id in sorted(override_tickers_by_team):
            for ticker in override_tickers_by_team[team_id]:
                override_team_by_ticker.setdefault(ticker, team_id)

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
                    "override_team_id": None,
                }

        # Non-focus tickers the agent looked up via @tool get_factor_profile
        # surface here with empty focus fields + agent_override=1 and the
        # attributing team in override_team_id. The dashboard reads
        # (focus_list_passed=0 AND agent_override=1) as "agent reached outside
        # the curated set" — the precision / recall / override-hit-rate audit
        # primitives in §5.3 of the scanner plan doc — now split per team.
        for ticker, team_id in override_team_by_ticker.items():
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
                "override_team_id": team_id,
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
        factor_profiles, market_regime, get_factor_blend_regime_weights(),
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
                # Legacy recompute path has no per-team override telemetry
                # (overrides are only known from PR 4 state); attribution is
                # None here — the projection path above is authoritative.
                "override_team_id": None,
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
            "override_team_id": None,
        }

    return lookup_legacy


# ── Secondary-work deadline guard ───────────────────────────────────────────
# The research Lambda has a HARD 900s ceiling (15 min is the absolute Lambda
# maximum — it cannot be raised). Normal full-graph runs reach archive_writer
# in ~5-9 min, but a tail-latency-slow run can run right up to the wall:
# 2026-06-06 the sector teams alone ate ~13 of the 15 min, signals.json was
# persisted at ~14.9 min, and the Lambda was SIGKILL'd seconds later while
# extracting semantic memories (an unbounded LLM call). The Step Function saw
# a TIMEOUT and failed the whole Research branch — for a run whose primary
# deliverable (signals.json) had ALREADY landed, and worse, the kill landed
# BEFORE the "must not miss" scanner_eval grade logging that follows the
# extraction, leaving a permanent hole in grade history.
#
# This guard lets archive_writer skip the lowest-priority best-effort work
# (semantic-memory extraction) once a wall-clock budget is exhausted, so the
# remaining higher-priority logging + graph finalize complete inside 900s and
# the run returns OK with signals delivered instead of being killed mid-flight.
# Budget is measured from state["run_time"] (set at create_initial_state,
# ~Lambda start). Set RESEARCH_SECONDARY_DEADLINE_S=0 to disable.
_SECONDARY_DEADLINE_ENV = "RESEARCH_SECONDARY_DEADLINE_S"
_SECONDARY_DEADLINE_DEFAULT_S = 780.0  # 13 min — ~2 min headroom under the 900s ceiling


def _secondary_deadline_budget_s() -> float:
    raw = os.environ.get(_SECONDARY_DEADLINE_ENV, "")
    if not raw:
        return _SECONDARY_DEADLINE_DEFAULT_S
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[archive_writer] %s=%r is not a number; using default %.0fs",
            _SECONDARY_DEADLINE_ENV, raw, _SECONDARY_DEADLINE_DEFAULT_S,
        )
        return _SECONDARY_DEADLINE_DEFAULT_S


def _secondary_work_deadline_exhausted(state: ResearchState) -> tuple[bool, float]:
    """Return (exhausted, elapsed_s) for the secondary-work wall-clock budget.

    Fails OPEN (returns False) if disabled (budget<=0) or run_time is
    unparseable — i.e. when in doubt the secondary work still runs, matching
    the pre-guard behavior. Only a confidently-elapsed budget skips work.
    """
    budget = _secondary_deadline_budget_s()
    if budget <= 0:
        return (False, 0.0)
    run_time = state.get("run_time") or ""
    try:
        start = datetime.fromisoformat(run_time)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    except (TypeError, ValueError):
        return (False, 0.0)
    return (elapsed >= budget, elapsed)


def _run_secondary_within_budget(
    state: ResearchState, label: str, fn: Callable[[], Any],
) -> tuple[bool, Any]:
    """Run an UNBOUNDED best-effort observability task ``fn`` iff the
    secondary-work wall-clock budget is not yet exhausted; otherwise SKIP it
    (WARN) so the remaining Lambda budget is preserved for the must-not-miss
    grade-history logging + ``upload_db`` that finalize the run.

    Returns ``(ran, result)``. ``ran`` is False only when the task was
    deadline-skipped. ``fn`` is invoked at most once; because signals.json is
    already persisted by the time archive_writer reaches these tail tasks, any
    exception ``fn`` raises is swallowed to a WARN (best-effort observability
    must never fail a delivered-signals run) and returned as ``(True, None)``.

    This is the single chokepoint for archive_writer's UNBOUNDED tail: every
    tail task whose runtime has no internal bound (an LLM call, a full-universe
    ArcticDB price read + digest email) MUST route through here so a
    tail-latency-slow run degrades by SKIPPING observability rather than by a
    SIGKILL at the 900s Lambda ceiling — a SIGKILL that returns a spurious
    States.Timeout to the Step Function for a run whose signals.json already
    landed AND starves the ``upload_db`` finalize that follows (a permanent
    grade-history hole). Same rationale as the semantic-extraction gate above;
    see the 2026-06-06 (semantic extraction) and 2026-07-03 (attractiveness
    trajectory) SIGKILL incidents. NOTE: this SKIPS only best-effort,
    already-fail-soft observability — it is NOT a swallow of a real failure; the
    primary deliverables (signals.json + the must-not-miss grade history) still
    land, and a genuine research failure still hard-fails upstream.
    """
    hit, elapsed = _secondary_work_deadline_exhausted(state)
    if hit:
        logger.warning(
            "[archive_writer] %s SKIPPED — secondary-work deadline exhausted "
            "(%.0fs elapsed >= %.0fs budget). signals.json already persisted; "
            "preserving the remaining Lambda budget for the must-not-miss eval "
            "logging + upload_db so the run returns OK, not TIMEOUT.",
            label, elapsed, _secondary_deadline_budget_s(),
        )
        return (False, None)
    try:
        return (True, fn())
    except Exception as e:  # noqa: BLE001 — best-effort observability, never fatal
        logger.warning(
            "[archive_writer] %s FAILED (non-fatal, observe-mode — "
            "signals.json unaffected): %s", label, e,
        )
        return (True, None)


def _build_scanner_eval_rows(
    *,
    scanner_universe: list,
    extra_override_tickers: list,
    technical_scores: dict,
    sector_map: dict,
    scanner_eval_log: list,
    focus_lookup: dict,
    run_date: str,
) -> list:
    """Assemble the ``scanner_evaluations`` rows for the cycle (pure, so the
    join is unit-testable independently of the archive_writer graph node).

    ``scanner_eval_log`` is ``run_quant_filter._last_eval_log`` — the
    AUTHORITATIVE per-ticker scanner verdict. We join it (not agent team-picks)
    because ``scanner_evaluations`` records the SCANNER funnel (~900 → ~60):

      * ``quant_filter_pass`` is the scanner's recorded survival flag. The
        backtester's e2e_lift (``analysis/end_to_end.py``) grades scanner recall
        / lift off this column, so it MUST mean "survived the quant gate", not
        "an agent later picked it". Agent selection lives in the focus-list
        audit (``focus_list_passed`` / ``agent_override``) projected below.
      * ``filter_fail_reason`` / ``scan_path`` / ``liquidity_pass`` /
        ``volatility_pass`` are the scanner's recorded gate outcomes — the only
        place each dropped name's reason exists. Reconstructing rows without
        them left every failed name with a NULL reason (the dashboard's
        "(unspecified)" bucket) and uniform pass-flags.

    Names absent from the eval log (agent-override tickers outside the scanned
    universe; or a cycle where the stash was unavailable) degrade to
    ``quant_filter_pass=0`` + null reason — honest "not scanner-evaluated", with
    metrics still carried from ``technical_scores``.
    """
    eval_by_ticker = {
        e.get("ticker"): e for e in (scanner_eval_log or []) if e.get("ticker")
    }

    def _pref(elog: dict, key: str, *fallbacks):
        """Scanner eval-log value (authoritative) else the first non-None
        fallback (the state ``technical_scores`` slice) else None."""
        v = elog.get(key)
        if v is not None:
            return v
        for fb in fallbacks:
            if fb is not None:
                return fb
        return None

    rows = []
    for ticker in list(scanner_universe) + list(extra_override_tickers):
        ts = technical_scores.get(ticker, {}) or {}
        elog = eval_by_ticker.get(ticker, {})
        row = {
            "ticker": ticker,
            "eval_date": run_date,
            "sector": sector_map.get(ticker) or elog.get("sector"),
            "tech_score": _pref(elog, "tech_score", ts.get("technical_score")),
            "rsi_14": _pref(elog, "rsi_14", ts.get("rsi_14")),
            "atr_pct": _pref(elog, "atr_pct", ts.get("atr_pct"), ts.get("atr_14_pct")),
            "price_vs_ma200": _pref(elog, "price_vs_ma200", ts.get("price_vs_ma200")),
            "current_price": _pref(elog, "current_price", ts.get("current_price")),
            "avg_volume_20d": _pref(elog, "avg_volume_20d", ts.get("avg_volume_20d")),
            "scan_path": elog.get("scan_path"),
            "quant_filter_pass": int(elog.get("quant_filter_pass", 0) or 0),
            "filter_fail_reason": elog.get("filter_fail_reason"),
        }
        # Carry the scanner's recorded gate flags when present (the eval log
        # only stamps the flag relevant to where a name dropped); absent flags
        # fall to the schema's NOT NULL DEFAULT 1.
        if "liquidity_pass" in elog:
            row["liquidity_pass"] = int(elog.get("liquidity_pass") or 0)
        if "volatility_pass" in elog:
            row["volatility_pass"] = int(elog.get("volatility_pass") or 0)
        # Project focus-list audit fields. focus_lookup is empty when factor
        # profiles aren't readable — every row gets NULL focus_* fields +
        # focus_list_passed=0, which the dashboard reads as "shadow logging
        # didn't run this cycle" rather than "all tickers failed."
        fl_entry = focus_lookup.get(ticker)
        if fl_entry is not None:
            row.update(fl_entry)
        rows.append(row)
    return rows


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

        # Save tool usage as analyst resources. Tool calls are produced at the
        # *team* (sector) grain — the combined quant+qual ReAct log for the whole
        # sector team, not scoped to a single ticker (see
        # agents/sector_teams/sector_team.py:all_tool_calls). The previous guard
        # `tc.get("ticker")` was always False (extract_tool_calls only emits
        # {"tool", "input_summary"}), so analyst_resources was never populated.
        # Record one row per tool call at team grain with a sentinel ticker; the
        # team→sector mapping lives in `agent="team:{team_id}"`, which is what the
        # per-sector tool-usage analysis (config#925) aggregates on.
        for tc in output.get("tool_calls", []):
            tool = tc.get("tool")
            if not tool:
                continue
            try:
                am.save_analyst_resource(
                    ticker=tc.get("ticker") or TEAM_RESOURCE_TICKER,
                    run_date=run_date,
                    agent=f"team:{team_id}",
                    resource_type=tool,
                    resource_detail=str(tc.get("input_summary", ""))[:200],
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
    # `prior_theses` on the next run via `archive.manager.load_latest_theses`.
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

    # Per-ticker moat-profile time-series — ROADMAP L1650. Moats decay
    # slowly; the time derivative is the real signal, so we append a
    # snapshot per Saturday SF to ``archive/universe/{ticker}/moat_profile.json``
    # rather than overwriting. Only fires when `quality_moat` is present
    # on the thesis (PILLAR_EMIT on + qual analyst returned a moat
    # assessment); silently skipped otherwise so legacy / dry-run paths
    # don't write empty stubs. Best-effort per ticker — a single ticker's
    # S3 failure doesn't fail-out the archive_writer.
    n_moats_written = 0
    for ticker, thesis in investment_theses.items():
        moat = thesis.get("quality_moat")
        if not moat or not isinstance(moat, dict):
            continue
        try:
            am.save_moat_profile(ticker, run_date, moat)
            n_moats_written += 1
        except Exception as e:  # noqa: BLE001 — secondary observability
            logger.warning("Failed to save moat_profile for %s: %s", ticker, e)
    if n_moats_written:
        logger.info(
            "[archive_writer] wrote %d/%d moat_profile snapshots",
            n_moats_written, len(investment_theses),
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
        from nousergon_lib.arcticdb import get_universe_symbols
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

    # Reuse the payload the stub-quarantine guard already scanned at
    # the top of this node so the promoted bytes are exactly the
    # bytes the guard verified clean (no TOCTOU between guard +
    # write). _build_signals_payload is pure so this is identical.
    signals_payload = _candidate_signals_payload
    _validate_and_write_signals(
        am,
        run_date,
        state.get("run_time", ""),
        signals_payload,
        scanner_universe=universe_symbols,
        tool_call_counts_by_team=tool_call_counts_by_team,
    )

    # ── research_intel neutral artifact (config#1500 — Phase 0 of #1499) ──────
    # Publish the neutral, product-facing intel ONCE at derivation time to a
    # NEW sibling artifact (research_intel/{date}.json + latest.json). Derived
    # from the SAME already-computed run state as signals.json — recomputes
    # NOTHING (regime/sector_ratings/modifiers/narrative from macro,
    # attractiveness from score_aggregator, breadth from fetch_data). The edge
    # (tuned weights, prompts, ENTER/EXIT judgment) stays in signals.json and
    # crucible-research; only OUTPUT scores/labels/narratives cross this
    # boundary (see _build_research_intel_payload's allowlist).
    #
    # ADDITIVE + fail-soft: this runs AFTER signals.json is already persisted,
    # and a failure here is swallowed (WARN) so the neutral sibling can never
    # sink the primary signals deliverable. The producer contract test
    # (test_research_intel_producer_contract.py) + the nousergon_lib
    # research_intel schema gate the shape.
    try:
        research_intel_payload = _build_research_intel_payload(state)
        am.write_research_intel(run_date, research_intel_payload)
        logger.info(
            "[archive_writer] research_intel written → research_intel/%s.json "
            "(+latest.json): regime=%s, %d sectors, %d tickers",
            run_date,
            research_intel_payload.get("market_regime"),
            len(research_intel_payload.get("sector_ratings", {})),
            len(research_intel_payload.get("attractiveness", {})),
        )
    except Exception as e:  # noqa: BLE001 — neutral sibling, never fatal
        logger.warning(
            "[archive_writer] research_intel write FAILED (non-fatal — "
            "signals.json unaffected, executor path intact): %s", e,
        )

    # ── Score-neutralization OBSERVE shadow (config#1142) ────────────────────
    # UNCONDITIONAL observability hung off the primary path AFTER signals.json is
    # already persisted: residualize the live composite cross-section against the
    # Barra factor loadings and write decision_artifacts/_neutralization_shadow/
    # {run_date}.json. Pure observation — never alters live signals / population /
    # ENTER selection. Fail-soft (WARN + swallow): a shadow failure cannot fail
    # the run.
    #
    # LIVE cutover gate (NEUTRALIZATION_LIVE_ENABLED, DEFAULT OFF): when flipped
    # true (private-first, after the shadow validates against momentum_regime_ic
    # — config#1140) the neutralized scores would become the live composite
    # ranking. While OFF (the default) this branch is inert and the live
    # signals.json written above is byte-identical to today — the gate is wired
    # but does NOT touch the live path.
    # Captures the LIVE neutralized score per ticker when the cutover gate is on,
    # so archive_writer can persist it as the DUAL field on cio_evaluations
    # (config#1187). Empty when the gate is OFF — the column then stays NULL and
    # the forward-IC metric reads raw==neutralized (identity).
    _live_neutralized_scores: dict[str, float] = {}
    try:
        from scoring.neutralization_shadow import run_neutralization_shadow
        _shadow_artifact = run_neutralization_shadow(
            am,
            run_date,
            _candidate_signals_payload,
            state.get("factor_loadings"),
            factors=NEUTRALIZATION_FACTORS,
        )
        if NEUTRALIZATION_LIVE_ENABLED and _shadow_artifact:
            # LIVE cutover (gated off by default — see comment above). Apply the
            # neutralized score as the live composite ranking and re-persist.
            logger.warning(
                "[archive_writer] NEUTRALIZATION_LIVE_ENABLED — applying "
                "neutralized scores to live signals (config#1142 cutover)."
            )
            _live_signals = signals_payload.get("signals", {})
            for _t, _row in _shadow_artifact.get("tickers", {}).items():
                _neu = _row.get("neutralized_score")
                if _t in _live_signals and _neu is not None:
                    _live_signals[_t]["score"] = _neu
                    # Record the live neutralized score for DUAL persistence to
                    # research.db (config#1187) — keyed by ticker, joined into
                    # cio_eval_records below so the backtester can measure the
                    # LIVE neutralized score's realized forward efficacy.
                    _live_neutralized_scores[_t] = _neu
            am.write_signals_json(run_date, state.get("run_time", ""), signals_payload)
    except Exception as e:  # noqa: BLE001 — secondary observability, never fatal
        logger.warning("[archive_writer] neutralization shadow skipped: %s", e)

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

    # Extract semantic memories from this run (Phase 3) — lowest-priority,
    # best-effort, and an UNBOUNDED (LLM-call) tail task. Routed through the
    # SINGLE archive_writer secondary-work chokepoint (_run_secondary_within_budget)
    # so there is one gate implementation for the whole tail: on a tail-latency-slow
    # run it is deadline-SKIPPED rather than risking a SIGKILL here that would
    # (a) return TIMEOUT to the SF for a delivered-signals run and (b) starve the
    # "must not miss" scanner_eval logging that follows. Same rationale — and now
    # the same chokepoint — as the attractiveness-trajectory tail task below.
    from memory.semantic import extract_semantic_memories

    _sem_ran, n_semantic = _run_secondary_within_budget(
        state,
        "semantic extraction",
        lambda: extract_semantic_memories(
            db_conn=am.db_conn,
            sector_team_outputs=state.get("sector_team_outputs", {}),
            macro_report=state.get("macro_report", ""),
            market_regime=state.get("market_regime", "neutral"),
            ic_decisions=state.get("ic_decisions", []),
            run_date=run_date,
        ),
    )
    if _sem_ran and n_semantic:
        logger.info("[archive_writer] extracted %d semantic memories", n_semantic)

    # ── Evaluation logging ──────────────────────────────────────────────────
    # Log all ~900 stocks with tech indicators for population baseline analysis.
    # These writes feed the backtester's weekly grading of scanner / sector
    # team / CIO components against universe_returns. Must not be silently
    # swallowed — a missed week leaves a permanent hole in grade history.
    scanner_universe = state.get("scanner_universe", [])
    technical_scores = state.get("technical_scores", {})
    sector_map = state.get("sector_map", {})
    market_regime = state.get("market_regime", "neutral")

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

    # The AUTHORITATIVE per-ticker scanner verdict (quant_filter_pass +
    # filter_fail_reason + scan_path + the liquidity/volatility gate flags) is
    # computed by run_quant_filter and travels here via candidates.json ::
    # scanner_eval_log → ResearchState (set in fetch_data) — the only place
    # the funnel reason for each of the ~900 names exists. Join it in here
    # rather than reconstructing placeholder rows (which dropped fail_reason →
    # NULL and mis-set quant_filter_pass to agent team-picks).
    #
    # NOTE: this is deliberately NOT ``run_quant_filter._last_eval_log`` (a
    # process-local module-attribute stash). Research no longer calls
    # run_quant_filter in this process post-L1995-Phase5 — only the
    # standalone Scanner SF state does, in ITS OWN process — so that stash is
    # always empty here (config#1458: the prior version of this code read the
    # stash directly and silently degraded quant_filter_pass to 0 for 100% of
    # rows, every cycle, since PR#344 merged). ``scanner_orchestrator.
    # build_shadow_candidate_artifacts`` still reads the stash directly and
    # correctly — it runs in the SAME process as the run_quant_filter call
    # that populates it.
    scanner_eval_log = state.get("scanner_eval_log") or []
    if not scanner_eval_log:
        logger.warning(
            "[archive_writer] candidates.json carried no scanner_eval_log for "
            "this cycle — scanner_evaluations gate detail (filter_fail_reason "
            "/ scan_path / true quant_filter_pass) degrades to null/0 this cycle"
        )
    scanner_evals = _build_scanner_eval_rows(
        scanner_universe=scanner_universe,
        extra_override_tickers=extra_override_tickers,
        technical_scores=technical_scores,
        sector_map=sector_map,
        scanner_eval_log=scanner_eval_log,
        focus_lookup=focus_lookup,
        run_date=run_date,
    )

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

    # ── Universe scoreboard (full-universe dashboard artifact) ──────────────
    # Publish the ~900-name scoreboard (attractiveness + 6 pillar scores + raw
    # valuation/fundamental/technical metrics + sector + country + gate flags)
    # for the dashboard's filterable universe board. SECONDARY observability
    # hung off the research run's primary deliverable (signals.json): per the
    # no-silent-fails secondary-observability exception, fail SOFT here with a
    # WARN — a board-write failure must NOT fail the research run. The WARN is
    # the recording surface (the builder raises on genuinely broken inputs, so
    # this can't mask a real fault as success).
    try:
        from scoring.universe_board import compute_and_write_universe_board

        ub_key = compute_and_write_universe_board(run_date, scanner_evals)
        logger.info("[archive_writer] universe scoreboard written → %s", ub_key)
    except Exception as e:
        logger.warning(
            "[archive_writer] universe scoreboard write FAILED (non-fatal, "
            "dashboard visibility only — signals.json unaffected): %s", e,
        )

    # ── Attractiveness trajectory signal (orthogonalized factor-momentum) ────
    # Reads the attractiveness history (appended just above by the board write)
    # + full-universe ArcticDB prices → the weekly "rising attractiveness /
    # pre-repricing" signal + digest email. OBSERVE-MODE, SECONDARY
    # observability: fail SOFT (a signal failure must not fail the research
    # run); no-ops during warm-up.
    #
    # This is the SECOND unbounded best-effort item on the tail (the ArcticDB
    # price reads + digest email have no internal time bound). On 2026-07-03 it
    # was still running when the Lambda hit the 900s ceiling — for a run whose
    # signals.json had already landed — and the SIGKILL there ALSO starved the
    # must-not-miss upload_db that follows (the week's scanner/team/CIO grade
    # history, written to the local DB above, was never uploaded). So it is now
    # deadline-gated through the same chokepoint as semantic extraction: on a
    # tail-latency-slow run it is SKIPPED, keeping the higher-priority DB
    # logging + upload_db inside 900s so the run returns OK with signals + grade
    # history delivered, not TIMEOUT.
    from scoring.attractiveness_trajectory import compute_and_write_trajectory

    _tj_ran, tj_key = _run_secondary_within_budget(
        state,
        "attractiveness trajectory",
        lambda: compute_and_write_trajectory(run_date),
    )
    if _tj_ran:
        logger.info(
            "[archive_writer] attractiveness trajectory written → %s",
            tj_key or "(warm-up — skipped)",
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
            # DUAL field (config#1187): the LIVE #1142 neutralized composite
            # score for this ticker, persisted alongside the raw final_score so
            # the backtester can join the LIVE neutralized ranking to realized
            # forward 21d alpha. Populated only when NEUTRALIZATION_LIVE_ENABLED
            # rewrote this ticker's score above; NULL otherwise (gate off / no
            # exposures / name absent from the neutralized cross-section).
            "neutralized_final_score": _live_neutralized_scores.get(ticker),
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


# ── Slim briefing email (config#856 Phase 2.5) ───────────────────────────────
#
# Mirrors the crucible-executor EOD-email conversion (executor/eod_emailer.py,
# alpha-engine-config#856 "pull-for-state console page + push-on-transition
# emails only", shipped as crucible-executor#276 / crucible-dashboard#237):
# the full consolidated report is no longer rendered inline in the email.
# It's already persisted verbatim by archive_writer
# (ArchiveManager.save_consolidated_report — UNCHANGED by this conversion,
# still written to consolidated/{run_date}/morning.md) and rendered by the
# console's Research Briefing Archive page
# (alpha-engine-dashboard views/17_Research_Briefing_Archive.py). The email
# now carries only a compact at-a-glance summary + a deep-link there.
#
# Deep-link gap (flagged, not fixed here — this PR does not touch
# crucible-dashboard): UNLIKE the executor's "eod-report" slug — pinned via
# ``url_path=`` in the dashboard's app.py and guarded against drift by
# tests/test_eod_report_page.py (see executor/eod_emailer.py
# EOD_REPORT_SLUG) — the Research Briefing Archive page has NO pinned
# url_path and no per-run query-param handling at all:
#
#   1. It's rendered as an unpinned sub-view TAB ("Briefing Archive") lazily
#      hosted inside views/host_research_signals.py (itself also unpinned in
#      app.py's navigation table), selected via shared/view_host.py's
#      documented ``?tab=`` bookmark query param.
#   2. Streamlit's default (filename-derived) url_path for that host page is
#      "host_research_signals" — verified directly against the
#      ``page_icon_and_name()`` slug algorithm in the dashboard's pinned
#      ``streamlit>=1.40`` floor — but nothing in either repo guards that
#      string the way "eod-report" is guarded, so a file rename would break
#      this link silently.
#   3. The page itself reads no ``?date=``/``?run=`` query param — "latest
#      inline + prior ~2 weeks click-to-expand" is the whole UI (confirmed:
#      no ``st.query_params`` usage in that view at all). So this is
#      necessarily a link to the GENERAL archive (which shows today's
#      just-persisted brief as "latest" for a same-day open), NOT a
#      deep-link scoped to this specific run the way
#      ``…/eod-report?date=YYYY-MM-DD`` is.
#
# (This repo's own scoring/attractiveness_trajectory.py already hand-rolled a
# *different*, likely-stale link — ``f"{CONSOLE_BASE_URL}/Attractiveness_Trends"``
# — into the analogous host_universe_scanner.py host tab, ignoring the
# host/tab indirection entirely. That's the failure mode this link avoids by
# using the real filename-derived slug plus the documented ``?tab=``
# mechanism, but it's the same class of gap: a dashboard follow-up should
# pin a url_path for this surface — and ideally add ``?date=`` support —
# the way eod-report/director/model-zoo/analysis already do.)
RESEARCH_BRIEFING_SLUG = "host_research_signals"
RESEARCH_BRIEFING_TAB = "Briefing Archive"

_SLIM_EMAIL_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body {{ font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6;
          color: #222; max-width: 700px; margin: 0 auto; padding: 20px; }}
  h2   {{ font-size: 15px; border-bottom: 1px solid #999; padding-bottom: 4px; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .cta  {{ display: inline-block; margin: 14px 0; padding: 10px 18px;
           background: #0b5; color: #fff !important; text-decoration: none;
           border-radius: 4px; font-weight: bold; }}
  .foot {{ margin-top: 28px; font-size: 11px; color: #888;
           border-top: 1px solid #ccc; padding-top: 8px; }}
</style>
</head>
<body>
{body}
<div class="foot">Alpha Engine Research | {date}</div>
</body>
</html>
"""


def _research_briefing_url(console_base_url: str | None = None) -> str:
    """Deep-link to the console's Research Briefing Archive tab.

    General-archive link only — see the gap note above the
    ``RESEARCH_BRIEFING_SLUG`` constant; the page has no per-run scoping to
    deep-link into.
    """
    from urllib.parse import quote

    from krepis.console import console_url

    base_url = console_url(RESEARCH_BRIEFING_SLUG, base=console_base_url)
    return f"{base_url}?tab={quote(RESEARCH_BRIEFING_TAB)}"


def _build_slim_briefing_email(state: ResearchState) -> tuple[str, str]:
    """Build the slim morning-briefing email body: ``(html_body, plain_body)``.

    A compact summary (run date, regime, population headline counts) plus a
    deep-link to the console archive — NOT the full consolidated report.
    The full report is persisted separately and unconditionally by
    ``archive_writer`` via ``ArchiveManager.save_consolidated_report``
    (untouched by this function; see the module note above ``email_sender``).
    """
    run_date = state.get("run_date", "")
    regime = (state.get("market_regime") or "neutral").upper()
    new_pop = state.get("new_population", []) or []
    current_pop = state.get("current_population", []) or []
    exits = state.get("exits", []) or []

    current_tickers = {p["ticker"] for p in current_pop if "ticker" in p}
    new_tickers = {p["ticker"] for p in new_pop if "ticker" in p}
    n_entrants = len(new_tickers - current_tickers)
    n_pop = len(new_pop)
    n_exits = len(exits)

    url = _research_briefing_url()

    html_parts = [
        "<h2>Daily Research Brief</h2>",
        "<table>",
        "<tr><th>Metric</th><th>Value</th></tr>",
        f"<tr><td>Run date</td><td>{run_date}</td></tr>",
        f"<tr><td>Regime</td><td>{regime}</td></tr>",
        "<tr><td>Population</td><td>"
        f"{n_pop} stocks ({n_entrants} new, {n_pop - n_entrants} existing, "
        f"{n_exits} exited)</td></tr>",
        "</table>",
        f'<a class="cta" href="{url}">View full research briefing on the console →</a>',
        '<p style="font-size:11px;color:#888;">Sector allocation, per-ticker '
        "ratings/rationale, regime trend, and risk posture are on the "
        "console archive.</p>",
    ]
    html_body = _SLIM_EMAIL_HTML_TEMPLATE.format(
        body="\n".join(html_parts), date=run_date,
    )

    plain_body = "\n".join([
        f"Alpha Engine Research — {run_date}",
        "=" * 40,
        f"Regime:     {regime}",
        f"Population: {n_pop} stocks ({n_entrants} new, "
        f"{n_pop - n_entrants} existing, {n_exits} exited)",
        "",
        f"Full briefing: {url}",
        "",
    ])
    return html_body, plain_body


def email_sender(state: ResearchState) -> dict:
    """Send the slim morning-briefing email: summary + console deep-link.

    As of 2026-07-03 (alpha-engine-config#856 Phase 2.5) this is a SLIM
    link email — see the module note above ``RESEARCH_BRIEFING_SLUG`` for
    the full rationale and the per-run deep-link gap. The full consolidated
    report is unaffected: ``archive_writer`` still unconditionally persists
    it verbatim via ``ArchiveManager.save_consolidated_report`` before this
    node ever runs.
    """
    from emailer.sender import send_email
    from config import EMAIL_RECIPIENTS, EMAIL_SENDER

    logger.info("[email_sender] starting")
    consolidated = state.get("consolidated_report", "")
    run_date = state.get("run_date", "")

    if consolidated:
        try:
            subject = f"Alpha Engine Research — {run_date}"
            html_body, plain_body = _build_slim_briefing_email(state)
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


def _validate_and_write_signals(
    am,
    run_date: str,
    run_time: str,
    signals_payload: dict,
    *,
    scanner_universe=None,
    tool_call_counts_by_team: dict[str, int] | None = None,
) -> None:
    """Validate the candidate signals payload and persist signals.json.

    The single fail-loud seam for the PRIMARY producer artifact of the
    weekly cycle (Predictor, Executor, the backtester and the report card
    all depend on signals.json). Both the validation and the write live
    inside one try so an invalid payload is as fatal as a failed write —
    neither yields a usable signals.json.

    FAIL LOUD — do NOT swallow. Graceful-degrade on a producer/writer is
    forbidden (~/.claude/CLAUDE.md "Fail loud and fast"). Swallowing here
    produced the "ghost success" failure mode (crucible-research#312): the
    research SF task returned status=OK while signals.json was absent, so
    the run looked healthy but the whole downstream cycle silently ran on
    stale signals. Re-raising propagates to lambda/handler.py's outer
    except -> status="ERROR" -> the SF Research branch fails, AND the
    artifact-freshness monitor (research_signals) flags the absence.

    Extracted from ``archive_writer`` to give the contract a clean unit
    seam (config#1235); behavior is identical to the inline block.
    """
    try:
        _validate_signals_payload(
            signals_payload,
            scanner_universe=scanner_universe,
            tool_call_counts_by_team=tool_call_counts_by_team,
            block_on_zero_tool_calls=False,  # soft-fail; flip after soak
        )
        am.write_signals_json(run_date, run_time, signals_payload)
    except Exception as e:
        logger.error("Failed to write signals.json: %s", e)
        raise


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
            # Pick provenance (config#859 stance_source_provenance grader):
            # CIO-advanced new entrant vs reaffirmed held name. Derived from
            # already-available locals — cannot raise.
            "stance_source": "cio_entrant" if ticker in advanced_tickers else "reaffirmed_hold",
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
                # Pick provenance: population ticker with no fresh thesis this run.
                "stance_source": "carryover",
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
                "stance_source": "exit",
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
            # Propagate pick provenance to the v1 universe entries the
            # evaluator's stance_source_provenance grader reads (config#859).
            "stance_source": sig.get("stance_source"),
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


# Contract version stamp for the research_intel artifact. Mirrors
# nousergon_lib.contracts.SCHEMA_VERSIONS["research_intel"]; kept as a
# module constant so the producer stamps it without importing the lib at
# build time (the pure builder must stay import-light + I/O-free).
RESEARCH_INTEL_SCHEMA_VERSION = 1

# The strict market_breadth allowlist (config#1500). breadth is stored in
# state["macro_data"] (fetch_data does macro_data.update(breadth)); we lift
# ONLY these three neutral keys — never the raw macro_data (which also holds
# vix / treasury / yield-curve inputs that are not part of this contract).
_RESEARCH_INTEL_BREADTH_KEYS = (
    "pct_above_50d_ma",
    "pct_above_200d_ma",
    "advance_decline_ratio",
)


def _build_research_intel_payload(state: ResearchState) -> dict:
    """Build the neutral, product-facing ``research_intel`` artifact (config#1500).

    A NEW sibling to ``signals.json`` (NOT a replacement — signals.json is
    untouched). This is a PURE function (no I/O) that carves the neutral,
    derived OUTPUT out of the already-computed run state and republishes it
    under a STRICT allowlist. It recomputes NOTHING — every field is read
    from a node output that already ran this cycle:

      * ``market_regime`` / ``regime_narrative`` / ``sector_ratings`` /
        ``sector_modifiers`` — from ``macro_economist_node``
      * ``market_breadth`` — from ``fetch_data`` (lives in ``macro_data``)
      * per-ticker ``attractiveness`` (final composite score + neutral
        component breakdown) + generic ``thesis`` — from ``score_aggregator``
        (``investment_theses``)

    EDGE STAYS PRIVATE (config#1499 boundary): only OUTPUT scores / labels /
    narratives are published. The tuned pillar/blend WEIGHTS carried on
    ``composite_breakdown.legacy_blend`` (w_legacy_quant / w_legacy_qual /
    w_factor) and ``composite_breakdown.pillar_contributions[].pillar_weight``
    are deliberately DROPPED — the breakdown here carries component VALUES
    only. ``thesis`` carries the generic sector-team bull_case + sector, NOT
    the position/ENTER-EXIT judgment (that stays in signals.json). Prompts
    are gitignored and never touch this path.
    """
    macro_data = state.get("macro_data", {}) or {}
    market_breadth = {
        k: macro_data.get(k) for k in _RESEARCH_INTEL_BREADTH_KEYS
    }

    # sector_ratings: carry ONLY {rating, rationale} (the neutral macro call).
    # Any extra keys an upstream might attach are dropped by construction.
    sector_ratings_out: dict[str, dict] = {}
    for sector, entry in (state.get("sector_ratings", {}) or {}).items():
        if not isinstance(entry, dict):
            continue
        sector_ratings_out[sector] = {
            "rating": entry.get("rating", "market_weight"),
            "rationale": entry.get("rationale", ""),
        }

    # Per-ticker attractiveness + generic thesis, keyed by ticker, from the
    # already-scored investment_theses (score_aggregator output).
    attractiveness: dict[str, dict] = {}
    for ticker, thesis in (state.get("investment_theses", {}) or {}).items():
        breakdown = thesis.get("composite_breakdown") or {}
        legacy = breakdown.get("legacy_blend") or {}
        attractiveness[ticker] = {
            "ticker": ticker,
            # Final composite score (0-100) — the published attractiveness.
            "score": thesis.get("final_score"),
            "sector": thesis.get("sector"),
            # Neutral component-VALUE breakdown — NO weights (edge stays
            # private). Reads from the already-computed composite_breakdown;
            # falls back to the flat thesis fields when the breakdown dict is
            # absent (held-stock recompute path).
            "breakdown": {
                "quant_score": thesis.get("quant_score"),
                "qual_score": thesis.get("qual_score"),
                "factor_subscore": (
                    legacy.get("factor_subscore")
                    if legacy else thesis.get("factor_subscore")
                ),
                "weighted_base": breakdown.get("weighted_base")
                if breakdown else thesis.get("weighted_base"),
                "macro_shift": breakdown.get("macro_shift")
                if breakdown else thesis.get("macro_shift"),
            },
            # Generic sector-team narrative only — NOT a position judgment.
            "thesis": {
                "bull_case": thesis.get("bull_case", ""),
                "sector": thesis.get("sector"),
            },
        }

    return {
        "schema_version": RESEARCH_INTEL_SCHEMA_VERSION,
        "date": state.get("run_date", ""),
        "generated_at": state.get("run_time", ""),
        "market_regime": state.get("market_regime", "neutral"),
        "regime_narrative": state.get("macro_report", ""),
        "sector_ratings": sector_ratings_out,
        "sector_modifiers": state.get("sector_modifiers", {}),
        "market_breadth": market_breadth,
        "attractiveness": attractiveness,
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
      3-class regime taxonomy (bull/neutral/bear) now flows
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
    # Phase 2.A.3: load_scorecard_node sits between the regime substrate
    # loader and the macro economist — same shape and graceful-degrade
    # contract. Produces ``prior_cycle_scorecard_text`` for both
    # ``macro_economist_node`` AND ``cio_node``.
    graph.add_node("load_scorecard_node", load_scorecard_node)
    graph.add_node("macro_economist_node", macro_economist_node)
    # Un-orphan arc: produce the institutional factor-profile substrate
    # AFTER fetch_data populated sector_map + run_date (macro does not
    # mutate either) and BEFORE both consumers (compute_focus_list_node
    # + score_aggregator) do their existing S3 read. Graceful-degrade
    # on any failure — never hard-fails the research run.
    graph.add_node("compute_factor_profiles_node", compute_factor_profiles_node)
    graph.add_node(
        "rank_candidates_by_attractiveness_node",
        rank_candidates_by_attractiveness_node,
    )
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

    # Serial: fetch_data → load_regime_substrate_node → load_scorecard_node
    # → macro_economist_node → compute_focus_list_node. Substrate flows
    # into the macro agent's ReAct prompt as a strong prior; scorecard
    # text flows into both macro AND cio prompts as prior-cycle outcome
    # context; macro's regime feeds the focus list blend; focus list
    # lands in state before sector teams dispatch.
    graph.add_edge("fetch_data", "load_regime_substrate_node")
    graph.add_edge("load_regime_substrate_node", "load_scorecard_node")
    graph.add_edge("load_scorecard_node", "macro_economist_node")
    # Splice compute_factor_profiles_node between macro and the focus
    # list: it needs sector_map + run_date (set in fetch_data, unchanged
    # by the substrate loader / macro) and must land the factor substrate
    # in S3 before compute_focus_list_node AND score_aggregator do their
    # existing read_factor_profiles_from_s3() this same run.
    graph.add_edge("macro_economist_node", "compute_factor_profiles_node")
    # Splice the attractiveness champion-feed re-rank (config#1400) between the
    # factor substrate (just written) and the focus list / dispatch: it reads the
    # 6-pillar profiles and (when ATTRACTIVENESS_FEED_ENABLED) overwrites
    # agent_input_set with the top-N attractiveness selection. Default OFF → a
    # pass-through no-op, so the existing tech_score feed is unchanged.
    graph.add_edge(
        "compute_factor_profiles_node", "rank_candidates_by_attractiveness_node"
    )
    graph.add_edge(
        "rank_candidates_by_attractiveness_node", "compute_focus_list_node"
    )

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
