"""
Sector Team Orchestrator — wires quant + qual + peer review into one execution unit.

Each team:
  1. Quant screens sector → top 10
  2. Qual reviews quant's top 5 → qual scores + 0-1 additions
  3. Peer review → final 2-3 recommendations
  4. Thesis maintenance for held stocks with material triggers

All 6 teams run in parallel via LangGraph Send().
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config import (
    ANTHROPIC_API_KEY,
    MAX_TOKENS_STRATEGIC,
    PER_STOCK_MODEL,
    TEAM_PICKS_PER_RUN,
)
from agents.sector_teams.team_config import (
    TEAM_SECTORS, TEAM_SCREENING_PARAMS, get_team_tickers,
)
from agents.prompt_loader import load_prompt
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS,
    invoke_structured_with_validation_retry,
)
from agents.sector_teams.quant_analyst import run_quant_analyst_with_retry
from agents.sector_teams.qual_analyst import run_qual_analyst
from agents.sector_teams.peer_review import run_peer_review
from agents.sector_teams.material_triggers import check_material_triggers
from thesis.structured import build_structured_thesis, format_structured_thesis_for_prompt
# Per-sub-agent cost-tracker scopes. Each sub-agent call below opens its
# own ``track_llm_cost`` frame keyed by the SAME agent_id the paired
# ``_capture_if_enabled`` call in ``research_graph.sector_team_node`` uses
# (``sector_quant:{team_id}`` etc.), so the metadata stash + per-call
# JSONL stream attribute to the canonical split families rather than the
# legacy combined ``sector_team:{team_id}`` aggregate. PER_STOCK_MODEL is
# the fallback model name (every sector sub-agent runs on it) so cost
# recompute resolves even when the callback can't read model_name off the
# response shape.
from graph.llm_cost_tracker import track_llm_cost

log = logging.getLogger(__name__)


class QuarantinableThesisError(RuntimeError):
    """A held ticker's thesis update failed DETERMINISTICALLY (its LLM output
    could not be parsed into a valid thesis after the correction-feedback
    retries) — a per-ticker terminal failure, NOT a transient/infra one.

    Per the per-ticker quarantine contract (config#2247, Brian's 2026-07-11
    ruling amending all-agents-strict at the SCOPE level only): a ticker that
    raises this is QUARANTINED — omitted from signals.json with an explicit
    ``quarantined`` record, RED telemetry + Telegram page — and the run
    completes for the rest, subject to the run-level floor (> ``MAX_QUARANTINED
    _TICKERS`` quarantined tickers OR a whole failed team still hard-fails).

    Deliberately distinct from a bare ``RuntimeError`` / a rate-limit (429)
    error: a transient/TPM failure re-rolling can't fix must STILL hard-fail
    (so an SF redrive retries the whole team) — only a deterministic per-ticker
    content failure is quarantine-eligible. NO stale-thesis carry-forward
    (unchanged): a quarantined ticker becomes an explicit absence, never a
    silently-carried prior thesis.
    """


@dataclass
class SectorTeamContext:
    """Bundled context for a sector team run — avoids 17-parameter function signatures."""
    scanner_universe: list[str]
    # L1995 Phase 5 / L4464: the pre-filtered sector-team screening input
    # (standalone scanner candidate set ∪ held population). The quant/qual
    # ReAct agents screen THIS (~10-15/sector), not the full sector slice
    # of scanner_universe (92-217/sector) which overran the recursion
    # budget. scanner_universe is retained for non-screening consumers.
    agent_input_set: list[str]
    sector_map: dict[str, str]
    price_data: dict[str, Any]
    technical_scores: dict[str, dict]
    market_regime: str
    prior_theses: dict[str, dict]
    held_tickers: list[str]
    news_data_by_ticker: dict[str, Any]
    analyst_data_by_ticker: dict[str, Any]
    insider_data_by_ticker: dict[str, Any]
    prior_sector_ratings: dict[str, dict]
    current_sector_ratings: dict[str, dict]
    run_date: str
    api_key: str | None = None
    episodic_memories: dict[str, list] = field(default_factory=dict)
    semantic_memories: dict[str, list] = field(default_factory=dict)
    # Stage D' Wire 1 (regime-v3 2026-05-14): substrate intensity_z
    # threaded through for the regime-conditional pick gate in
    # peer_review. None when substrate hasn't published yet (Stage A
    # pre-deploy or non-blocking SF Catch tripped) — gate degrades
    # gracefully to base threshold only.
    regime_intensity_z: float | None = None
    # PR 4 of scanner-placement arc (260514) — this team's regime-blended
    # focus list (list of FocusListEntry.to_dict() entries) computed by
    # compute_focus_list_node. Empty list when the factor substrate is
    # unavailable this cycle. When FOCUS_LIST_GATING_ENABLED and non-empty,
    # the quant analyst's user prompt uses this in place of the full
    # sector ticker slice.
    focus_list: list[dict] = field(default_factory=list)
    # Mutable accumulator for agent_override tool-call telemetry. The
    # get_factor_profile tool appends to this list whenever the agent
    # looks up a ticker that's NOT in this team's focus_list. Shared by
    # reference into create_quant_tools' context for the @tool wrapper;
    # archive_writer aggregates per-team override_tickers from team
    # outputs and projects agent_override=1 onto scanner_evaluations.
    override_tickers: list[str] = field(default_factory=list)
    # Pipeline-invocation id for the cost-tracker per-call JSONL partition
    # key + per-run budget accumulator. Threaded from ``sector_team_node``
    # via ``derive_run_id(state)`` so the per-sub-agent ``track_llm_cost``
    # scopes opened below partition their cost-raw rows under the same
    # run_id the node uses for the paired decision-artifact captures. None
    # in unit tests that build a bare context — the scopes then skip the
    # JSONL flush (the in-process ``pop_metadata_for`` stash still works,
    # which is what the captures actually read).
    run_id: str | None = None


def run_sector_team(team_id: str, ctx: SectorTeamContext) -> dict:
    """
    Run the full sector team pipeline.

    Returns:
        {
            "team_id": str,
            "recommendations": list[dict],  # final 2-3 picks with quant+qual scores
            "thesis_updates": dict[str, dict],  # updated theses for held stocks
            "quant_output": dict,  # full quant analyst output
            "qual_output": dict,  # full qual analyst output
            "peer_review_output": dict,
            "tool_calls": list[dict],  # combined tool call log
        }
    """
    log.info("[team:%s] starting — %d universe, %d held",
             team_id, len(ctx.scanner_universe), len(ctx.held_tickers))

    # ── Step 1: Get sector tickers ────────────────────────────────────────────
    # L1995 Phase 5 / L4464: screen the pre-filtered candidate set ∪ held
    # population (agent_input_set), NOT the full sector slice of the raw
    # ~900 universe. ~10-15 tickers/sector converges inside the ReAct
    # recursion budget; 92-217 did not (recursion_limit → 0 picks → retry).
    sector_tickers = get_team_tickers(team_id, ctx.agent_input_set, ctx.sector_map)
    log.info("[team:%s] %d tickers in sector", team_id, len(sector_tickers))

    if not sector_tickers:
        log.warning("[team:%s] no tickers in sector — skipping", team_id)
        return _empty_result(team_id)

    # ── Step 2: Quant analyst screens sector ──────────────────────────────────
    # Uses the retry wrapper so the agent-gave-up failure mode (zero
    # picks despite running tools) gets one augmented-prompt retry
    # before falling through to the empty-result path. Recursion
    # exhaustion + exceptions are NOT retried — see
    # _should_retry_on_empty_picks() for the trigger contract.
    # Scope wraps the quant ReAct loop (multiple Anthropic calls incl. the
    # decoupled structured-output extraction + any empty-picks retry). The
    # user-prompt template stamps prompt_id/version onto ModelMetadata.
    # config#1753 audit note: this scope is intentionally NOT threaded
    # with ``rendered_prompt`` — a ReAct loop has no single canonical
    # rendered user-prompt string (multiple tool-calling turns, each with
    # its own message list); ``FullPromptContext.user_prompt`` falls back
    # to the raw ``LoadedPrompt.text`` template here, same as before this
    # fix. Only the single-shot batch-call sites (ic_cio, macro_economist,
    # eval_judge) have one rendered string to thread.
    with track_llm_cost(
        agent_id=f"sector_quant:{team_id}",
        sector_team_id=team_id,
        node_name="sector_team_node",
        run_type="weekly_research",
        run_id=ctx.run_id,
        model_name_fallback=PER_STOCK_MODEL,
        prompt=load_prompt("quant_analyst_user"),
    ):
        quant_output = run_quant_analyst_with_retry(
            team_id=team_id,
            sector_tickers=sector_tickers,
            market_regime=ctx.market_regime,
            price_data=ctx.price_data,
            technical_scores=ctx.technical_scores,
            run_date=ctx.run_date,
            api_key=ctx.api_key,
            focus_list=ctx.focus_list,
            override_tickers=ctx.override_tickers,
        )

    quant_picks = quant_output.get("ranked_picks", [])
    # Validate picks have required 'ticker' key — LLM output parsing can drop it
    valid_picks = [p for p in quant_picks if isinstance(p, dict) and "ticker" in p]
    if len(valid_picks) < len(quant_picks):
        log.warning(
            "[team:%s] quant produced %d picks but %d lack 'ticker' key — dropped",
            team_id, len(quant_picks), len(quant_picks) - len(valid_picks),
        )
    if not valid_picks:
        log.warning("[team:%s] quant produced no valid picks", team_id)
        return _empty_result(team_id, quant_output=quant_output)

    # ── Step 3: Qual analyst reviews top 5 ────────────────────────────────────
    top5 = valid_picks[:5]

    with track_llm_cost(
        agent_id=f"sector_qual:{team_id}",
        sector_team_id=team_id,
        node_name="sector_team_node",
        run_type="weekly_research",
        run_id=ctx.run_id,
        model_name_fallback=PER_STOCK_MODEL,
        prompt=load_prompt("qual_analyst_user"),
    ):
        qual_output = run_qual_analyst(
            team_id=team_id,
            quant_top5=top5,
            prior_theses=ctx.prior_theses,
            market_regime=ctx.market_regime,
            run_date=ctx.run_date,
            api_key=ctx.api_key,
            price_data=ctx.price_data,
            episodic_memories=ctx.episodic_memories,
            semantic_memories=ctx.semantic_memories,
        )

    # ── Step 4: Peer review → final 0-3 ──────────────────────────────────────
    # Stage D' Wire 1: peer_review applies a regime-conditional pick
    # gate after selection. Risk-off conditions (negative intensity_z
    # from the continuous regime substrate) raise the minimum composite-
    # score bar; teams now allowed to emit 0 picks when no candidate
    # clears the bar. Deeper risk-off → higher bar. (Pre-v0.42.0 the
    # docstring named the legacy 3-class projection
    # "bear/caution raise the bar"; the implementation has always read
    # the continuous intensity_z.)
    # Scope wraps both peer-review Anthropic passes (quant's review of
    # qual's addition + the joint finalization). The joint-selection
    # prompt — the synthesis call that produces the final picks CIO sees
    # — stamps prompt_id/version onto ModelMetadata.
    with track_llm_cost(
        agent_id=f"sector_peer_review:{team_id}",
        sector_team_id=team_id,
        node_name="sector_team_node",
        run_type="weekly_research",
        run_id=ctx.run_id,
        model_name_fallback=PER_STOCK_MODEL,
        prompt=load_prompt("peer_review_joint_selection"),
    ):
        peer_output = run_peer_review(
            team_id=team_id,
            quant_picks=top5,
            qual_assessments=qual_output.get("assessments", []),
            additional_candidate=qual_output.get("additional_candidate"),
            technical_scores=ctx.technical_scores,
            market_regime=ctx.market_regime,
            regime_intensity_z=ctx.regime_intensity_z,
            api_key=ctx.api_key,
        )

    # ── Step 5: Thesis maintenance for held stocks ────────────────────────────
    team_held = [t for t in ctx.held_tickers if ctx.sector_map.get(t, "") in
                 {s for s, tid in _sector_team_inverse().items() if tid == team_id}]

    # Check for sector regime change
    sector_regime_changed = _check_regime_change(
        team_id, ctx.prior_sector_ratings, ctx.current_sector_ratings
    )

    thesis_updates = {}
    # Per-ticker quarantine (config#2247): a held ticker whose thesis update
    # fails DETERMINISTICALLY is recorded here and omitted from thesis_updates
    # (explicit absence, no carry-forward) rather than raising and killing the
    # whole run. score_aggregator applies the run-level floor.
    quarantined: list[dict] = []
    for ticker in team_held:
        # Held tickers must always have a prior_thesis. The held-stock update
        # path (triggers branch below + no-trigger preservation branch) both
        # depend on prior_thesis carrying the score fields — the LLM is not
        # authoritative on scores for held updates. A held ticker without a
        # prior_thesis means archive_writer wrote `population` without writing
        # the corresponding `investment_thesis` row (the bug closed by the
        # 2026-04-25 atomic-thesis-write fix). Hard-fail loudly so the
        # invariant cannot silently regress.
        if ctx.prior_theses.get(ticker) is None:
            raise RuntimeError(
                f"Held ticker {ticker} has no prior_thesis in archive — "
                f"population/investment_thesis are out of sync. Either "
                f"archive_writer regressed the atomic-write invariant, or "
                f"a backfill was skipped after a schema migration. Refusing "
                f"to produce an unscoreable thesis_update per "
                f"feedback_no_unscoreable_labels.md."
            )

        triggers = check_material_triggers(
            ticker=ticker,
            news_data=ctx.news_data_by_ticker.get(ticker),
            price_data=ctx.price_data.get(ticker),
            analyst_data=ctx.analyst_data_by_ticker.get(ticker),
            insider_data=ctx.insider_data_by_ticker.get(ticker),
            prior_thesis=ctx.prior_theses.get(ticker),
            sector_regime_changed=sector_regime_changed,
            run_date=ctx.run_date,
        )

        if triggers:
            # Material event — update thesis via the per-stock model. One
            # cost-tracker scope per ticker, keyed to match the per-ticker
            # ``thesis_update:{team_id}:{ticker}`` capture in
            # sector_team_node — the no-trigger preservation branch fires
            # no LLM call and opens no scope (and is not captured).
            with track_llm_cost(
                agent_id=f"thesis_update:{team_id}:{ticker}",
                sector_team_id=team_id,
                node_name="sector_team_node",
                run_type="weekly_research",
                run_id=ctx.run_id,
                model_name_fallback=PER_STOCK_MODEL,
                prompt=load_prompt("sector_team_thesis_update"),
            ):
                try:
                    updated = _update_thesis_for_held_stock(
                        ticker, triggers, ctx.prior_theses.get(ticker),
                        ctx.news_data_by_ticker.get(ticker),
                        ctx.analyst_data_by_ticker.get(ticker),
                        ctx.run_date, team_id, ctx.api_key,
                    )
                except QuarantinableThesisError as exc:
                    # Deterministic per-ticker output failure (the CRUS case,
                    # config#2247). Quarantine and continue — omit from
                    # thesis_updates (NO stale-thesis carry-forward), record the
                    # explicit absence. Transient/429 failures are NOT
                    # QuarantinableThesisError and still propagate → hard-fail.
                    log.error(
                        "[team:%s] QUARANTINE %s — held-thesis update failed "
                        "deterministically (%s). Omitted from signals.json "
                        "(explicit-absence contract, config#2247); no "
                        "prior-thesis carry-forward. Run continues for the rest.",
                        team_id, ticker, exc,
                    )
                    quarantined.append({
                        "ticker": ticker,
                        "team_id": team_id,
                        "stage": "held_thesis_update",
                        "reason": str(exc),
                    })
                    continue
            thesis_updates[ticker] = updated
        else:
            # No material event — preserve prior thesis. Normalize the
            # conviction field at the boundary: archive may carry legacy
            # agent-format strings ("medium" etc.) from rows written before
            # the held-stock normalize_conviction fix (PR #56) or before
            # Option A (2026-04-30). The ThesisUpdate schema only accepts
            # int 0-100 or storage-format literals, so passing the raw
            # prior_thesis through would fail typed-state validation in
            # sector_team_node. Normalize once here to keep this path
            # schema-compliant; score_aggregator's recompute path also
            # normalizes downstream so this is double-cover, not a bypass.
            from scoring.composite import normalize_conviction
            prior = ctx.prior_theses[ticker]
            preserved = {
                **prior,
                "stale_days": prior.get("stale_days", 0) + 1,
                "triggers": [],
            }
            if "conviction" in preserved:
                preserved["conviction"] = normalize_conviction(
                    preserved["conviction"]
                )
            thesis_updates[ticker] = preserved

    # ── Combine tool call logs ────────────────────────────────────────────────
    all_tool_calls = (
        quant_output.get("tool_calls", []) +
        qual_output.get("tool_calls", []) +
        [{"phase": "peer_review", "rationale": peer_output.get("peer_review_rationale", "")}]
    )

    log.info("[team:%s] done — %d recommendations, %d thesis updates",
             team_id, len(peer_output.get("recommendations", [])), len(thesis_updates))

    # Propagate the first analyst error (if any) to the team level so the
    # score_aggregator can hard-fail loudly. Quant errors take precedence —
    # a broken quant stage guarantees a broken qual stage.
    team_error = quant_output.get("error") or qual_output.get("error")
    if team_error:
        team_error = f"[team:{team_id}] {team_error}"

    # Bubble the partial flag up too. A team is partial if either analyst
    # hit the recursion budget — score_aggregator treats partial teams as
    # WARN-and-include (zero recommendations contributed) rather than as
    # an error (which would crash the SF). Distinct from team_error so
    # genuine failures still hard-fail.
    team_partial = bool(quant_output.get("partial") or qual_output.get("partial"))
    partial_reasons = [
        f"quant:{quant_output.get('partial_reason')}"
        if quant_output.get("partial") else None,
        f"qual:{qual_output.get('partial_reason')}"
        if qual_output.get("partial") else None,
    ]
    partial_reasons = [r for r in partial_reasons if r is not None]

    return {
        "team_id": team_id,
        "recommendations": peer_output.get("recommendations", []),
        "thesis_updates": thesis_updates,
        "quant_output": quant_output,
        "qual_output": qual_output,
        "peer_review_output": peer_output,
        "tool_calls": all_tool_calls,
        "error": team_error,
        "partial": team_partial,
        "partial_reasons": partial_reasons,
        # Per-ticker quarantine records (config#2247). Distinct from
        # error/partial (which are TEAM-level and still hard-fail): these are
        # individual tickers deterministically omitted from signals.json.
        # score_aggregator aggregates across teams and applies the floor.
        "quarantined": quarantined,
        # PR 4 of scanner-placement arc — tickers the quant agent looked up
        # via @tool get_factor_profile that were NOT in this team's focus
        # list. archive_writer projects these onto scanner_evaluations
        # rows as agent_override=1 for the audit table.
        "override_tickers": list(ctx.override_tickers),
    }


def _empty_result(team_id: str, quant_output: dict | None = None,
                  error: str | None = None) -> dict:
    # If quant produced an error and no explicit error was passed, surface
    # it so the aggregator sees the failure rather than an empty team that
    # looks identical to "no sector tickers in universe".
    if error is None and quant_output is not None:
        error = quant_output.get("error")
    # Surface partial too so a quant team that hit recursion still flows
    # as partial-not-error through score_aggregator.
    partial = bool(quant_output.get("partial")) if quant_output else False
    partial_reasons = (
        [f"quant:{quant_output.get('partial_reason')}"]
        if partial and quant_output and quant_output.get("partial_reason") else []
    )
    return {
        "team_id": team_id,
        "recommendations": [],
        "thesis_updates": {},
        "quant_output": quant_output or {},
        "qual_output": {},
        "peer_review_output": {},
        "tool_calls": [],
        "error": error,
        "partial": partial,
        "partial_reasons": partial_reasons,
    }


def _sector_team_inverse() -> dict[str, str]:
    """Return {gics_sector: team_id} mapping."""
    from agents.sector_teams.team_config import SECTOR_TEAM_MAP
    return SECTOR_TEAM_MAP


def _check_regime_change(
    team_id: str,
    prior_ratings: dict,
    current_ratings: dict,
) -> bool:
    """Check if any of this team's sectors changed regime rating."""
    team_sectors = TEAM_SECTORS.get(team_id, [])
    for sector in team_sectors:
        prior = prior_ratings.get(sector, {}).get("rating", "market_weight")
        current = current_ratings.get(sector, {}).get("rating", "market_weight")
        if prior != current:
            return True
    return False


def _augment_news_summary_with_rag(
    *,
    ticker: str,
    triggers: list[str],
    base_news_summary: str,
) -> str:
    """Augment the headline-only news_summary with RAG-retrieved news +
    filings excerpts. Returns the combined string.

    Wave 1 PR E (data-revamp-260513.md): the thesis_update agent
    historically saw only 5 bullet headlines from the snapshot's
    news_data. With the news → RAG ingest pipeline live (data PR A.3
    #229), we can pull semantically-relevant excerpts for the ticker
    over the last 14 days — the LLM gets narrative depth, not just
    headlines.

    Gated behind ``THESIS_UPDATE_RAG_CONTEXT_ENABLED=true`` (default
    off) so production behavior is unchanged until parallel
    observation validates the augmented prompt's effect on output
    quality. Failure of either RAG retrieval call degrades gracefully
    to the base headline summary — never crashes the thesis update.

    Query strategy: build a per-trigger natural-language query so the
    retriever surfaces the most relevant context for what materially
    changed today.
    """
    try:
        from agents.sector_teams.rag_retrieval_tools import (
            search_filings_impl,
            search_news_impl,
        )
    except ImportError as e:
        log.warning(
            "[thesis_update:%s] RAG augment skipped — import error: %s",
            ticker, e,
        )
        return base_news_summary

    # Build a context-rich query from the material triggers. E.g.
    # ['price_move_gt_2atr', 'earnings_beat'] → 'price move earnings
    # beat'. Falls back to ticker name when triggers absent.
    trigger_terms = " ".join(t.replace("_", " ") for t in triggers)
    rag_query = (
        f"{ticker} {trigger_terms}".strip() if trigger_terms else ticker
    )

    pieces: list[str] = []
    if base_news_summary:
        pieces.append(base_news_summary)

    try:
        news_excerpts = search_news_impl(
            ticker, rag_query, days_back=14, top_k=5,
        )
        # search_news_impl returns a "No recent news found" string on
        # empty; skip those.
        if news_excerpts and not news_excerpts.startswith("No recent news"):
            pieces.append(
                "Recent news context (from RAG, last 14 days):\n"
                + news_excerpts
            )
    except Exception as e:
        log.warning(
            "[thesis_update:%s] RAG news augment failed: %s", ticker, e,
        )

    try:
        filings_excerpts = search_filings_impl(
            ticker, rag_query, days_back=90, top_k=3,
        )
        if filings_excerpts and not filings_excerpts.startswith("No filings"):
            pieces.append(
                "Recent filings context (from RAG, last 90 days):\n"
                + filings_excerpts
            )
    except Exception as e:
        log.warning(
            "[thesis_update:%s] RAG filings augment failed: %s", ticker, e,
        )

    return "\n\n".join(pieces) if pieces else base_news_summary


def _update_thesis_for_held_stock(
    ticker: str,
    triggers: list[str],
    prior_thesis: dict | None,
    news_data: dict | None,
    analyst_data: dict | None,
    run_date: str,
    team_id: str,
    api_key: str | None = None,
    *,
    temperature: float | None = None,
) -> dict:
    """Update thesis for a held stock with material triggers (single Haiku call).

    ``temperature`` is an optional passthrough to ``ChatAnthropic`` — the
    default ``None`` preserves production behavior exactly (the library's
    own default, unset). The scenario-replay harness
    (``scripts/replay_harness.py``, L4567 sub-item 2b / #781) passes an
    explicit value so a counterfactual can be run N times at temp>0 and
    produce an outcome *distribution* rather than one deterministic point.
    """
    # Defer-import to avoid module-init cycle (sector_team is imported by
    # research_graph during cost-tracker setup).
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    # Held-stock thesis update is single-ticker BUT narrative-rich: the
    # HeldThesisUpdateLLMOutput schema emits bull_case + bear_case (prose
    # paragraphs) + a `catalysts` list[str] + scores — synthesis-class
    # output, NOT a small accept/reject. It was mis-tiered onto the
    # per-stock budget (MAX_TOKENS_PER_STOCK=800) it shared with the
    # genuinely-tiny QuantAcceptanceVerdict call. On the 2026-06-27
    # Saturday SF run MDT's update overran 800: the structured-output
    # tool-call truncated mid-`<parameter name="catalysts">`, so
    # langchain captured a partial string where a list was required
    # (`catalysts: Input should be a valid list ... input_type=str`).
    # That is the 2026-05-03 qual_analyst truncation-bug class (PR
    # #100/#102) recurring at a different, under-budgeted call site, and
    # because 800 truncates DETERMINISTICALLY all three all-agents-strict
    # parse re-rolls failed identically and the run hard-failed.
    #
    # Root-cause fix: reclassify this site onto MAX_TOKENS_STRATEGIC —
    # the tier every other narrative-rich structured output already uses
    # (qual/quant analyst extraction, macro, ic_cio, evals.judge). The
    # MODEL is unchanged (still PER_STOCK_MODEL / Haiku); only the output
    # ceiling moves, so nothing about WHAT the thesis says changes — the
    # model simply finishes emitting the same output instead of being cut
    # off. Cost-neutral: Anthropic bills emitted tokens, not the cap.
    # The schema_max_tokens_audit row for this site is corrected in
    # lockstep so the static guard would now catch a regression to 800.
    # NOTE: keep default_request_timeout as a literal kwarg here (not folded
    # into a **kwargs dict) — tests/test_llm_request_timeout.py statically
    # AST-scans every ChatAnthropic(...) call site in agents/ for a literal
    # default_request_timeout/timeout keyword (config#687 regression guard);
    # a **kwargs-constructed call would be invisible to that walk.
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_STRATEGIC,
        max_retries=SECTOR_TEAM_LLM_MAX_RETRIES,
        default_request_timeout=SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS,
        callbacks=[get_cost_telemetry_callback()],
        # ChatAnthropic's own default is temperature=None (no override sent
        # to the API, server defaults to 1.0) — passing None through here
        # reproduces that exactly, so the harness-only ``temperature`` param
        # is a true no-op at its default and changes nothing for the live
        # production call path.
        temperature=temperature,
    )

    prior_text = ""
    if prior_thesis:
        prior_text = format_structured_thesis_for_prompt(prior_thesis)

    news_summary = ""
    if news_data:
        articles = news_data.get("articles", [])
        if articles:
            news_summary = "\n".join(
                f"- {a.get('headline', '')}" for a in articles[:5]
            )

    # Wave 1 PR E (data-revamp-260513.md): optional RAG-context injection.
    # When THESIS_UPDATE_RAG_CONTEXT_ENABLED=true (default OFF for
    # parallel-observation cutover), augment news_summary with RAG-
    # retrieved news + filings excerpts so the LLM has narrative depth
    # beyond the headline-only snapshot.
    #
    # Default OFF preserves current production behavior — flip ON in
    # Lambda env to A/B against the headline-only baseline.
    if os.environ.get("THESIS_UPDATE_RAG_CONTEXT_ENABLED", "").lower() == "true":
        news_summary = _augment_news_summary_with_rag(
            ticker=ticker,
            triggers=triggers,
            base_news_summary=news_summary,
        )

    # config#1821 Option B (2026-07-08): analyst consensus rating / price
    # target / upside were removed from fetch_analyst_consensus's returned
    # shape (the FMP endpoints that populated them 402'd for every ticker
    # on the current plan). Earnings surprises remain a live field.
    analyst_summary = ""
    if analyst_data:
        surprises = analyst_data.get("earnings_surprises") or []
        if surprises:
            latest = surprises[0]
            analyst_summary = (
                f"Latest earnings surprise ({latest.get('date', 'N/A')}): "
                f"{latest.get('surprise_pct', 'N/A')}%"
            )

    loaded_prompt = load_prompt("sector_team_thesis_update")
    prompt = loaded_prompt.format(
        team_title=team_id.title(),
        ticker=ticker,
        triggers_csv=", ".join(triggers),
        prior_text=prior_text or "No prior thesis available.",
        news_summary=news_summary or "No significant news.",
        analyst_summary=analyst_summary or "No analyst updates.",
    )

    # PR 2.3 Step E: held-stock LLM updates use HeldThesisUpdateLLMOutput,
    # which by design has NO score fields (final_score / quant_score /
    # qual_score / rating). The LLM cannot emit them because they're not in
    # the schema. This retires the strip-nulls workaround that existed to
    # defend against LNTH/LLY/PFE/VRTX/CME/JHG/COKE/HSY/KR's
    # `"final_score": null` overwrites in the 2026-04-11 run.
    #
    # ALL-AGENTS-STRICT rework (Brian, 2026-05-16) — REMOVES #193's
    # carry-forward-prior-thesis fallback:
    #
    #   "We don't get anything from this process if the sectors, or any
    #    other agent for that matter, fail/don't run."
    #
    #   A held-thesis update is one of the agents in scope. #193 made a
    #   failed held-thesis update silently carry the prior thesis
    #   forward so the run continued; the directive reverses this — a
    #   held-thesis update that still cannot produce REAL output after
    #   the long 429 retry window MUST fail the run, not silently ship
    #   a stale thesis dressed as a fresh one.
    #
    #   Resilience now lives in two complementary places:
    #     (1) 429s: ``invoke_with_rate_limit_retry`` retries persistently
    #         up to the ~75-min wall-clock deadline (long enough to ride
    #         out the org TPM window), then propagates.
    #     (2) Tool-XML schema leaks (the `catalysts` string-not-list
    #         failure, deterministic OR nondeterministic): the shared
    #         ``invoke_structured_with_validation_retry`` chokepoint re-prompts
    #         with the Pydantic ``ValidationError`` fed back as correction
    #         context (recovers deterministic leaks a bare re-roll cannot),
    #         then — if STILL malformed — RAISE. No prior-thesis carry-forward.
    #
    #   The raise propagates through ``_run_sector_team`` →
    #   ``run_sector_team`` and surfaces as the team's ``error``, which
    #   ``score_aggregator`` now hard-fails on (revert of #194's
    #   degrade-and-continue). The team is NOT persisted (errored teams
    #   are never persisted), so an SF redrive re-attempts only it.
    from graph.state_schemas import HeldThesisUpdateLLMOutput

    structured_llm = llm.with_structured_output(
        HeldThesisUpdateLLMOutput, include_raw=True,
    )
    # SOTA structured-output recovery: route through the shared
    # ``invoke_structured_with_validation_retry`` chokepoint every other
    # narrative-rich extraction site already uses (qual_analyst, macro,
    # evals.judge) instead of a bespoke bare re-roll.
    #
    # WHY (2026-07-11 CRUS Saturday hard-fail): the previous loop re-sent the
    # IDENTICAL prompt on every attempt with NO correction context. A
    # DETERMINISTIC tool-XML leak — the model emitting a literal
    # ``<parameter name="catalysts">`` tag into the field value, which
    # langchain captures as a ``str`` where ``catalysts: list[str]`` is
    # required — therefore re-rolls identically all 3 attempts and can never
    # recover, hard-failing the whole weekly run. (Distinct from a
    # ``max_tokens`` truncation, which the shared chokepoint's config#1294
    # ``raise_if_truncated`` guard now also covers here, and from a
    # NON-deterministic leak, which a bare re-roll happened to fix.)
    #
    # The chokepoint feeds the specific Pydantic ``ValidationError`` (plus the
    # model's own prior malformed output) back as correction context on each
    # retry — the industry-standard tool-use recovery that lets the model
    # correct the offending field (list-not-string) rather than repeat it. The
    # MODEL, prompt, and schema are unchanged, so nothing about WHAT the thesis
    # concludes changes — the model simply gets the standard correction retry.
    #
    # ALL-AGENTS-STRICT is preserved end-to-end: on terminal failure the
    # chokepoint returns with ``parsing_error`` set / ``parsed`` None and the
    # fail-loud ``raise`` below fires — NO prior-thesis carry-forward (that was
    # #193, removed 2026-05-16). A 429 still fails fast: the
    # ``invoke_with_rate_limit_retry`` wrapper INSIDE the chokepoint propagates
    # a post-deadline 429 unchanged (re-rolling can't fix an org TPM ceiling),
    # so it surfaces here uncaught.
    _MAX_PARSE_ATTEMPTS = 3  # total attempts = chokepoint max_retries (2) + 1
    extract_resp = invoke_structured_with_validation_retry(
        structured_llm,
        [HumanMessage(content=prompt)],
        label=f"thesis_update:{team_id}:{ticker}",
        ls_metadata=loaded_prompt.langsmith_metadata(),
        max_retries=_MAX_PARSE_ATTEMPTS - 1,
    )
    update: HeldThesisUpdateLLMOutput | None = extract_resp.get("parsed")
    parsing_error = extract_resp.get("parsing_error")
    if parsing_error is not None or update is None:
        # Still malformed after the correction-feedback retries.
        # ALL-AGENTS-STRICT: this agent did NOT produce real output, so the run
        # must fail. NO carry-forward of the prior thesis (that was #193, now
        # removed). Raise — surfaces as the team's hard error.
        last_error: Exception = parsing_error or ValueError(
            "structured-output returned no parsed model"
        )
        log.error(
            "[thesis_update:%s] still malformed after %d parse "
            "attempts (%s) — RAISING. Per the all-agents-strict "
            "directive a held-thesis update that cannot produce "
            "real output fails the whole run; the prior thesis is "
            "NOT carried forward.",
            ticker, _MAX_PARSE_ATTEMPTS, last_error,
        )
        raise QuarantinableThesisError(
            f"held-thesis update for {ticker} ({team_id}) failed "
            f"after {_MAX_PARSE_ATTEMPTS} attempts and cannot "
            f"produce real output — per-ticker quarantine "
            f"(no prior-thesis carry-forward): {last_error}"
        ) from last_error

    # Convert to dict + drop default-empty fields so they don't overwrite a
    # populated prior_thesis value with the default (e.g. an empty bull_case
    # shouldn't blank out a non-empty prior bull_case).
    llm_update_clean = {
        k: v for k, v in update.model_dump().items()
        if v not in (None, "", [])
    }
    if prior_thesis:
        result = {**prior_thesis, **llm_update_clean}
    else:
        result = llm_update_clean
        result["score_failed"] = True
    result["last_updated"] = run_date
    result["triggers"] = triggers
    result["stale_days"] = 0
    return result
