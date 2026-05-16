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
    MAX_TOKENS_PER_STOCK,
    PER_STOCK_MODEL,
    TEAM_PICKS_PER_RUN,
)
from agents.sector_teams.team_config import (
    TEAM_SECTORS, TEAM_SCREENING_PARAMS, get_team_tickers,
)
from agents.prompt_loader import load_prompt
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    _is_rate_limit_error,
    invoke_with_rate_limit_retry,
)
from agents.sector_teams.quant_analyst import run_quant_analyst_with_retry
from agents.sector_teams.qual_analyst import run_qual_analyst
from agents.sector_teams.peer_review import run_peer_review
from agents.sector_teams.material_triggers import check_material_triggers
from thesis.structured import build_structured_thesis, format_structured_thesis_for_prompt

log = logging.getLogger(__name__)


@dataclass
class SectorTeamContext:
    """Bundled context for a sector team run — avoids 17-parameter function signatures."""
    scanner_universe: list[str]
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
    sector_tickers = get_team_tickers(team_id, ctx.scanner_universe, ctx.sector_map)
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
    # gate after selection. Bear / caution regimes raise the minimum
    # composite-score bar; teams now allowed to emit 0 picks when no
    # candidate clears the bar. intensity_z (from regime substrate)
    # scales the bar — deeper risk-off → higher bar.
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
            # Material event — update thesis via Haiku
            updated = _update_thesis_for_held_stock(
                ticker, triggers, ctx.prior_theses.get(ticker),
                ctx.news_data_by_ticker.get(ticker),
                ctx.analyst_data_by_ticker.get(ticker),
                ctx.run_date, team_id, ctx.api_key,
            )
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
) -> dict:
    """Update thesis for a held stock with material triggers (single Haiku call)."""
    # Defer-import to avoid module-init cycle (sector_team is imported by
    # research_graph during cost-tracker setup).
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    # Held-stock thesis update is a single-ticker output (bull_case +
    # bear_case + score updates), fits the per-stock tier. Was hardcoded
    # at 500 before consolidation 2026-05-03; MAX_TOKENS_PER_STOCK=800
    # gives 60% more headroom for verbose triggers without altering
    # other call sites.
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_PER_STOCK,
        max_retries=SECTOR_TEAM_LLM_MAX_RETRIES,
        callbacks=[get_cost_telemetry_callback()],
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

    analyst_summary = ""
    if analyst_data:
        analyst_summary = (
            f"Consensus: {analyst_data.get('consensus_rating', 'N/A')}, "
            f"Target: ${analyst_data.get('mean_target', 'N/A')}, "
            f"Upside: {analyst_data.get('upside_pct', 'N/A')}%"
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
    #     (2) Transient tool-XML schema leaks (the 2026-05-16 `catalysts`
    #         string-not-list nondeterminism): a small bounded
    #         parse/validation retry (these recover on a re-roll), then
    #         — if STILL malformed — RAISE. No prior-thesis carry-forward.
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
    # Bounded parse/validation re-roll for the transient tool-XML leak
    # ONLY. This is NOT a 429 retry (429s are handled by the deadline-
    # bounded wrapper inside the thunk) and NOT a degrade path — after
    # the last attempt we RAISE.
    _MAX_PARSE_ATTEMPTS = 3
    last_error: Exception | None = None
    for attempt in range(1, _MAX_PARSE_ATTEMPTS + 1):
        try:
            extract_resp = invoke_with_rate_limit_retry(
                lambda: structured_llm.invoke(
                    [HumanMessage(content=prompt)],
                    config={"metadata": loaded_prompt.langsmith_metadata()},
                ),
                label=f"thesis_update:{team_id}:{ticker}",
            )
            update: HeldThesisUpdateLLMOutput | None = extract_resp.get("parsed")
            parsing_error = extract_resp.get("parsing_error")
            if parsing_error is not None or update is None:
                raise parsing_error or ValueError(
                    "structured-output returned no parsed model"
                )
            # Convert to dict + drop default-empty fields so they don't
            # overwrite a populated prior_thesis value with the default
            # (e.g. an empty bull_case shouldn't blank out a non-empty
            # prior bull_case).
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
            if attempt > 1:
                log.info(
                    "[thesis_update:%s] recovered on parse-retry "
                    "(attempt %d/%d)",
                    ticker, attempt, _MAX_PARSE_ATTEMPTS,
                )
            return result
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e):
                # The deadline-bounded wrapper already exhausted the
                # ~75-min 429 window. Re-rolling the prompt won't help
                # an org TPM ceiling — fail fast (the run hard-fails).
                log.error(
                    "[thesis_update:%s] org 429 persisted past the "
                    "retry deadline — failing the run per "
                    "all-agents-strict (no prior-thesis carry-forward)",
                    ticker,
                )
                raise
            if attempt < _MAX_PARSE_ATTEMPTS:
                log.warning(
                    "[thesis_update:%s] parse/validation attempt %d/%d "
                    "failed: %s — re-rolling (transient tool-XML leak)",
                    ticker, attempt, _MAX_PARSE_ATTEMPTS, e,
                )
                continue
            # Final attempt still malformed. ALL-AGENTS-STRICT: this
            # agent did NOT produce real output, so the run must fail.
            # NO carry-forward of the prior thesis (that was #193, now
            # removed). Raise — surfaces as the team's hard error.
            log.error(
                "[thesis_update:%s] still malformed after %d parse "
                "attempts (%s) — RAISING. Per the all-agents-strict "
                "directive a held-thesis update that cannot produce "
                "real output fails the whole run; the prior thesis is "
                "NOT carried forward.",
                ticker, _MAX_PARSE_ATTEMPTS, last_error,
            )
            raise RuntimeError(
                f"held-thesis update for {ticker} ({team_id}) failed "
                f"after {_MAX_PARSE_ATTEMPTS} attempts and cannot "
                f"produce real output — all-agents-strict hard-fail "
                f"(no prior-thesis carry-forward): {last_error}"
            ) from last_error

    # Unreachable — the loop either returns or raises.
    raise AssertionError(  # pragma: no cover
        f"held-thesis update for {ticker} exited retry loop without "
        f"returning or raising"
    )
