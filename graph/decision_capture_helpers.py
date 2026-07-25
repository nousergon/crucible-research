"""
Decision-artifact capture helpers — one ``build_capture_payload(...)`` per
producer node that returns ``(input_data_snapshot, summary)`` tuples.

The actual S3 write happens in ``alpha_engine_lib.decision_capture.capture_decision``.
This module exists so each producer node has a single place to declare
"what does the agent semantically depend on?" — which fields belong in
``input_data_snapshot`` for replay correctness, and what one-line summary
captures the decision context for at-a-glance reading.

Compatibility posture: the snapshots are JSON-serializable dicts (no
``pd.DataFrame``, no Pydantic models — those get ``.to_dict()`` /
``.model_dump()`` at the boundary). The 1MB cap in ``capture_decision``
truncates pathological payloads; steady-state agent inputs are well under
the cap per the Day-1 design doc § 4 PR 3 size estimates.

**Excluded by design** from per-node snapshots: ``price_data``
(``dict[str, pd.DataFrame]``, huge — agents use the already-derived
``technical_scores`` instead). Replay paths can reconstruct ``price_data``
from ArcticDB's universe library via ``run_date`` if needed.

**Field provenance:**
- ``ModelMetadata`` token counts + cost_usd now flow from
  ``graph.llm_cost_tracker`` — ``_capture_if_enabled`` reads populated
  metadata via ``pop_metadata_for(agent_id)``. Call sites not yet wired
  through ``track_llm_cost`` fall back to a model-name-only stub with
  zero tokens (logged loudly so the gap is visible).
- ``FullPromptContext.user_prompt`` is plumbed via ``LoadedPrompt`` at
  ``track_llm_cost`` enter; ``system_prompt`` is still a placeholder
  because the agent-specific system prompts live in gitignored
  ``agents/`` files and aren't loaded by name.

Workstream design: ``alpha-engine-docs/private/alpha-engine-research-typed-
state-capture-260429.md``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.sector_teams.sector_team import SectorTeamContext


# ── Feature flag ──────────────────────────────────────────────────────────


_DECISION_CAPTURE_ENV_VAR = "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"


def is_decision_capture_enabled() -> bool:
    """Check the env var fresh on each call (allows toggling in tests).

    Returns True iff ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` is set to
    ``"true"`` / ``"1"`` / ``"yes"`` (case-insensitive). Default off — the
    capture path stays dormant until production explicitly opts in via the
    Lambda env var (gated on IAM grant for ``s3:PutObject`` on the
    ``decision_artifacts/*`` prefix).
    """
    return os.environ.get(_DECISION_CAPTURE_ENV_VAR, "").lower() in (
        "true", "1", "yes",
    )


# ── Per-node snapshot builders ────────────────────────────────────────────


def build_sector_quant_capture_payload(
    team_id: str,
    ctx: SectorTeamContext,
    *,
    team_tickers: list[str],
) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for the sector quant analyst.

    Quant screens the full sector population (~50-200 tickers) by technical
    score + regime context. Captures only what the quant LLM call actually
    saw: sector tickers, regime, and the technical_scores slice for that
    team. News/analyst/insider data is qual-tier — not in this snapshot.
    """
    snapshot: dict[str, Any] = {
        "team_id": team_id,
        "run_date": ctx.run_date,
        "market_regime": ctx.market_regime,
        "scanner_universe_size": len(ctx.scanner_universe),
        "sector_tickers": list(team_tickers),
        "sector_tickers_count": len(team_tickers),
        # technical_scores filtered to team tickers — full dict is ~900
        # entries, the team only sees ~50-200.
        "technical_scores_team": {
            t: dict(ctx.technical_scores.get(t, {})) for t in team_tickers
        },
    }
    summary = (
        f"team_id={team_id}, run_date={ctx.run_date}, regime={ctx.market_regime}, "
        f"sector_tickers={len(team_tickers)}, "
        f"technical_scored={sum(1 for t in team_tickers if t in ctx.technical_scores)}"
    )
    return snapshot, summary


def build_sector_qual_capture_payload(
    team_id: str,
    ctx: SectorTeamContext,
    *,
    quant_top5: list[dict],
) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for the sector qual analyst.

    Qual reviews quant's top 5 picks. Captures only what the qual LLM call
    actually saw: the top5 hand-off, prior theses for those tickers, and
    news/analyst/insider data scoped to those tickers (qual's tools fetch
    by ticker — capturing the full universe would over-state inputs).
    """
    top5_tickers = [
        p["ticker"] for p in quant_top5
        if isinstance(p, dict) and "ticker" in p
    ]
    held_in_top5 = [t for t in ctx.held_tickers if t in set(top5_tickers)]
    prior_theses_for_top5 = {
        t: dict(ctx.prior_theses[t]) for t in top5_tickers if t in ctx.prior_theses
    }
    snapshot: dict[str, Any] = {
        "team_id": team_id,
        "run_date": ctx.run_date,
        "market_regime": ctx.market_regime,
        "quant_top5": list(quant_top5),
        "quant_top5_tickers": top5_tickers,
        "held_in_top5": held_in_top5,
        "prior_theses_for_top5": prior_theses_for_top5,
        "news_data_for_top5": {
            t: dict(ctx.news_data_by_ticker[t]) for t in top5_tickers
            if t in ctx.news_data_by_ticker
        },
        "analyst_data_for_top5": {
            t: dict(ctx.analyst_data_by_ticker[t]) for t in top5_tickers
            if t in ctx.analyst_data_by_ticker
        },
        "insider_data_for_top5": {
            t: dict(ctx.insider_data_by_ticker[t]) for t in top5_tickers
            if t in ctx.insider_data_by_ticker
        },
        "prior_sector_ratings": dict(ctx.prior_sector_ratings),
        "current_sector_ratings": dict(ctx.current_sector_ratings),
        "memories_summary": {
            "episodic_count": len(ctx.episodic_memories),
            "semantic_count": len(ctx.semantic_memories),
        },
    }
    summary = (
        f"team_id={team_id}, run_date={ctx.run_date}, regime={ctx.market_regime}, "
        f"top5={len(top5_tickers)}, held_in_top5={len(held_in_top5)}, "
        f"prior_theses_for_top5={len(prior_theses_for_top5)}"
    )
    return snapshot, summary


def build_sector_peer_review_capture_payload(
    team_id: str,
    ctx: SectorTeamContext,
    *,
    quant_top5: list[dict],
    qual_assessments: list[dict],
    qual_additional_candidate: dict | None,
) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for the sector peer-review call.

    Peer review is the synthesis layer between quant + qual that produces
    the 2-3 recommendations CIO sees. Captures only what the peer-review
    LLM saw: quant's top 5, qual's per-ticker assessments, qual's optional
    additional candidate, the technical_scores slice for those tickers
    (used when reviewing the addition), and regime.
    """
    top5_tickers = [
        p["ticker"] for p in quant_top5
        if isinstance(p, dict) and "ticker" in p
    ]
    addition_ticker = (
        qual_additional_candidate.get("ticker")
        if isinstance(qual_additional_candidate, dict) else None
    )
    review_tickers = list(top5_tickers)
    if addition_ticker and addition_ticker not in review_tickers:
        review_tickers.append(addition_ticker)
    snapshot: dict[str, Any] = {
        "team_id": team_id,
        "run_date": ctx.run_date,
        "market_regime": ctx.market_regime,
        "quant_top5": list(quant_top5),
        "qual_assessments": list(qual_assessments),
        "qual_additional_candidate": (
            dict(qual_additional_candidate)
            if isinstance(qual_additional_candidate, dict) else None
        ),
        "technical_scores_review_set": {
            t: dict(ctx.technical_scores.get(t, {})) for t in review_tickers
        },
    }
    summary = (
        f"team_id={team_id}, run_date={ctx.run_date}, regime={ctx.market_regime}, "
        f"quant_top5={len(top5_tickers)}, qual_assessments={len(qual_assessments)}, "
        f"addition={'yes' if addition_ticker else 'no'}"
    )
    return snapshot, summary


def build_thesis_update_capture_payload(
    team_id: str,
    ticker: str,
    ctx: SectorTeamContext,
    *,
    triggers: list[str],
) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for a held-stock thesis update.

    Thesis updates fire conditionally (one Haiku call per held ticker
    when ``check_material_triggers`` returns non-empty). Captures the
    inputs to ``_update_thesis_for_held_stock`` (sector_team.py:301):
    ticker, triggers list, prior thesis, and the news + analyst data
    surfaced in the prompt.
    """
    prior = ctx.prior_theses.get(ticker)
    news = ctx.news_data_by_ticker.get(ticker)
    analyst = ctx.analyst_data_by_ticker.get(ticker)
    snapshot: dict[str, Any] = {
        "team_id": team_id,
        "ticker": ticker,
        "run_date": ctx.run_date,
        "triggers": list(triggers),
        "prior_thesis": dict(prior) if isinstance(prior, dict) else None,
        "news_data": dict(news) if isinstance(news, dict) else None,
        "analyst_data": dict(analyst) if isinstance(analyst, dict) else None,
    }
    summary = (
        f"team_id={team_id}, ticker={ticker}, run_date={ctx.run_date}, "
        f"triggers={len(triggers)}, "
        f"prior_thesis={'yes' if prior else 'no'}, "
        f"news_articles={len((news or {}).get('articles', []))}"
    )
    return snapshot, summary


def build_macro_economist_capture_payload(state: dict) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for the macro_economist node.

    Stage C.2 T3 addition (2026-05-14): includes ``regime_substrate``
    from state when populated. The judge's
    ``regime_decision_process`` rubric dimension scores the macro
    agent's regime call against the substrate (when present), so the
    substrate must be in agent_input for the judge to see it. ``None``
    flows through cleanly — judge anchor handles substrate-absent
    case (score the dimension as 5, mirroring the
    no-prior-macro-report convention).
    """
    macro_data = state.get("macro_data", {})
    prior_macro_report = state.get("prior_macro_report", "")
    prior_snapshots = state.get("prior_macro_snapshots", []) or []
    prior_date = ""
    if prior_snapshots:
        prior_date = prior_snapshots[0].get("date", "") if isinstance(prior_snapshots[0], dict) else ""

    regime_substrate = state.get("regime_substrate")

    snapshot: dict[str, Any] = {
        "run_date": state.get("run_date"),
        "macro_data": dict(macro_data) if isinstance(macro_data, dict) else macro_data,
        "prior_macro_report": prior_macro_report,
        "prior_date": prior_date,
        "prior_snapshots_count": len(prior_snapshots),
        "regime_substrate": regime_substrate,
    }
    substrate_marker = "absent"
    if isinstance(regime_substrate, dict):
        hmm_argmax = (regime_substrate.get("hmm") or {}).get("argmax", "?")
        intensity_z = (regime_substrate.get("composite") or {}).get("intensity_z")
        substrate_marker = (
            f"present(argmax={hmm_argmax},"
            f"intensity_z={intensity_z:+.2f})"
            if isinstance(intensity_z, (int, float))
            else f"present(argmax={hmm_argmax})"
        )
    summary = (
        f"run_date={state.get('run_date')}, "
        f"macro_data_keys={len(macro_data) if isinstance(macro_data, dict) else 0}, "
        f"prior_report_chars={len(prior_macro_report)}, "
        f"prior_snapshots={len(prior_snapshots)}, "
        f"regime_substrate={substrate_marker}"
    )
    return snapshot, summary


def build_cio_capture_payload(
    state: dict,
    *,
    candidates: list[dict],
    prior_ic: list[dict],
) -> tuple[dict[str, Any], str]:
    """Build (input_data_snapshot, summary) for the cio_node."""
    macro_context = {
        "market_regime": state.get("market_regime", "neutral"),
        "macro_report": state.get("macro_report", ""),
    }
    snapshot: dict[str, Any] = {
        "run_date": state.get("run_date"),
        "candidates_count": len(candidates),
        "candidates": list(candidates),
        "macro_context": macro_context,
        "sector_ratings": dict(state.get("sector_ratings", {})),
        "remaining_population": list(state.get("remaining_population", [])),
        "remaining_population_count": len(state.get("remaining_population", [])),
        "open_slots": state.get("open_slots", 0),
        "exits": list(state.get("exits", [])),
        "prior_ic_decisions": list(prior_ic),
        "prior_ic_count": len(prior_ic),
    }
    summary = (
        f"run_date={state.get('run_date')}, candidates={len(candidates)}, "
        f"open_slots={state.get('open_slots', 0)}, "
        f"remaining_pop={len(state.get('remaining_population', []))}, "
        f"prior_ic={len(prior_ic)}, regime={state.get('market_regime', 'neutral')}"
    )
    return snapshot, summary


# ── Run ID extraction ─────────────────────────────────────────────────────


def derive_run_id(state: dict) -> str:
    """Pick a stable run_id for the current pipeline invocation.

    Today: falls back to ``run_date`` so artifacts from the same day cluster
    under one S3 key prefix per agent. When Lambda's ``aws_request_id``
    is plumbed through state (a follow-up), prefer that — it gives us
    per-invocation uniqueness so retries don't overwrite each other.
    """
    run_id = state.get("run_id")
    if run_id:
        return str(run_id)
    return str(state.get("run_date", "unknown"))
