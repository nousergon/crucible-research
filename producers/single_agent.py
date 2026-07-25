"""Single-agent research producer (config#1223 / M3 baseline).

ONE LLM call assesses every scanner candidate's qualitative attractiveness; the
quant score stays deterministic (technical composite) and the two combine via
the SAME ``compute_composite_score`` the champion uses. This replaces the
champion's 6-sector-team fan-out + macro economist + CIO with a single agent —
so a champion-vs-single-agent comparison isolates the value of the MULTI-AGENT
ORCHESTRATION specifically (vs the value of an LLM at all, which the no-agent
floor isolates). It is also the natural Phase-3 distillation target (config#1135).

Assembly reuses the live ``_build_signals_payload`` (no reimplementation) —
contract-identical to the champion; only the belief differs.

**alpha-engine-config-I2997 (2026-07-19): migrated ``assess_candidates`` off
direct Anthropic (``ChatAnthropic``/Sonnet-4-6) to the fleet-SOTA
``krepis.llm.LLMClient`` OpenRouter transport (DeepSeek V4 Pro — this
challenger's per-candidate qualitative reasoning is the "heavier reasoning"
tier per Brian's ruling, vs the lighter DeepSeek V4 Flash used for
mechanical/high-volume sites). This is dispatched WEEKLY by the Saturday SF's
ChallengerShadow state (``_run_challengers_only``), and is a SHADOW/
best-effort challenger — never blocks the champion.**

``ModelSpec.structured_outputs=False`` is REQUIRED, not a style choice: live-
verified 2026-07-19 that OpenRouter's strict ``response_format=json_schema``
mode is unreliable for DeepSeek-family models — across repeated live calls
against this exact ``RankingProducerOutput`` schema, the model intermittently
renamed the required ``ticker`` field (e.g. to ``symbol``/``candidate``),
which the strict/json_schema path took as ground truth and failed schema
validation on every attempt. ``structured_outputs=False`` (JSON-instruction +
tolerant extraction — the krepis/Think-Tank fallback path) round-tripped this
exact schema correctly on every live attempt tried. ``reasoning={"exclude":
True}`` mirrors the fleet's two other live DeepSeek V4 OpenRouter consumers
(morning-signal's ``fallback_llm``, crucible-research's own
``evals/judge_models.py::OPENROUTER_SHADOW``) — without it a reasoning-
capable OpenRouter model can burn its whole output budget on chain-of-thought
and return empty content (config#1659 / config#2575).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scoring.composite import compute_composite_score

logger = logging.getLogger(__name__)

_PROMPT_NAME = "ranking_producer"
DEFAULT_BUY_SCORE_THRESHOLD = 60.0
DEFAULT_MAX_NEW_ENTRANTS = 15


# ── Structured output (mirrors CIORawOutput) ─────────────────────────────────
class CandidateAssessment(BaseModel):
    """The single agent's qualitative read on one candidate. The quant score is
    NOT requested from the LLM — it stays deterministic (technical composite)."""

    model_config = ConfigDict(extra="allow")
    ticker: str
    qual_score: float = Field(ge=0, le=100)
    conviction: Literal["rising", "stable", "declining"] = "stable"
    brief_thesis: str = ""


class RankingProducerOutput(BaseModel):
    model_config = ConfigDict(extra="allow", validate_default=True)
    assessments: list[CandidateAssessment] = Field(
        default_factory=list, min_length=1,
        description="One qualitative assessment per scanner candidate.",
    )


def build_single_agent_signals(
    run_date: str,
    *,
    scanner_tickers: list[str],
    assessments: list[dict],
    technical_scores: dict[str, dict],
    population: list[dict],
    prior_theses: dict[str, dict],
    sector_map: dict[str, str],
    market_regime: str = "neutral",
    run_time: str = "",
    buy_score_threshold: float = DEFAULT_BUY_SCORE_THRESHOLD,
    max_new_entrants: int = DEFAULT_MAX_NEW_ENTRANTS,
) -> dict:
    """Build a conforming signals.json from the single agent's qual assessments
    + deterministic quant. Pure function (no I/O / no LLM) → unit-testable."""
    from graph.research_graph import _build_signals_payload

    pop_tickers = {p["ticker"] for p in population}
    assess_by_ticker = {a["ticker"]: a for a in assessments}

    theses: dict[str, dict] = {}
    for ticker in scanner_tickers:
        tech = technical_scores.get(ticker)
        if not tech:
            continue
        quant = tech.get("technical_score")
        a = assess_by_ticker.get(ticker)
        qual = a.get("qual_score") if a else None
        comp = compute_composite_score(
            quant_score=quant,
            qual_score=qual,            # the single agent's qualitative read
            sector_modifier=1.0,        # neutral; no macro agent
            macro_overlay_enabled=False,
        )
        final = comp.get("final_score")
        if final is None:
            continue
        rating = "BUY" if final >= buy_score_threshold else "HOLD"
        theses[ticker] = {
            "ticker": ticker,
            "rating": rating,
            "score": final,
            "final_score": final,
            "quant_score": quant,
            "qual_score": qual,
            "conviction": (a.get("conviction") if a else None) or "stable",
            "sector": sector_map.get(ticker, "Unknown"),
            "bull_case": (a.get("brief_thesis") if a else "") or "",
        }

    new_buys = sorted(
        (t for t, th in theses.items() if th["rating"] == "BUY" and t not in pop_tickers),
        key=lambda t: theses[t]["final_score"],
        reverse=True,
    )
    advanced_tickers = new_buys[:max_new_entrants]

    new_population = list(population) + [
        {
            "ticker": t,
            "sector": theses[t]["sector"],
            "long_term_rating": "BUY",
            "long_term_score": theses[t]["final_score"],
            "conviction": theses[t]["conviction"],
            "price_target_upside": None,
        }
        for t in advanced_tickers
    ]

    state: dict = {
        "investment_theses": theses,
        "prior_theses": prior_theses,
        "new_population": new_population,
        "sector_map": sector_map,
        "sector_ratings": {},
        "sector_modifiers": {},
        "entry_theses": {},
        "advanced_tickers": advanced_tickers,
        "exits": [],
        "run_date": run_date,
        "run_time": run_time,
        "market_regime": market_regime,
    }
    payload = _build_signals_payload(state)
    logger.info(
        "[single_agent] run_date=%s assessed=%d scored=%d buy_candidates=%d "
        "new_entrants=%d", run_date, len(assessments), len(theses),
        len(payload.get("buy_candidates", [])), len(advanced_tickers),
    )
    return payload


def _format_candidate_block(scanner_tickers: list[str], technical_scores: dict, sector_map: dict) -> str:
    """Per-candidate quant context the single agent reasons over."""
    lines = []
    for t in scanner_tickers:
        tech = technical_scores.get(t) or {}
        lines.append(
            f"{t} | sector={sector_map.get(t, 'Unknown')} | "
            f"tech_score={tech.get('technical_score')} | rsi_14={tech.get('rsi_14')} | "
            f"momentum_20d={tech.get('momentum_20d')} | price_vs_ma200={tech.get('price_vs_ma200')}"
        )
    return "\n".join(lines)


CHALLENGER_MODEL = "deepseek/deepseek-v4-pro"
"""OpenRouter/DeepSeek V4 Pro (alpha-engine-config-I2997, 2026-07-19). ID
verified two ways: (1) live against the OpenRouter models API
(`GET https://openrouter.ai/api/v1/models` lists `deepseek/deepseek-v4-pro`
— "DeepSeek: DeepSeek V4 Pro"); (2) the sibling `deepseek/deepseek-v4-flash`
ID cross-checked against two independent live fleet configs already running
it (morning-signal's SSM `fallback_llm`, this repo's own
`evals/judge_models.py::OPENROUTER_SHADOW`) confirms the `deepseek/deepseek-
v4-*` naming family is correct. Never hand-write an OpenRouter model ID — a
typo silently killed morning-signal on 2026-07-15."""

CHALLENGER_LLM_MAX_RETRIES = 3
"""SDK-level (openai client) retry count for transport/429 errors — the
OpenAI SDK's own bounded-backoff retry, replacing the Anthropic-specific
``invoke_anthropic_safe`` deadline-bounded 429 wrapper this call site used
pre-migration (that wrapper inspects Anthropic-shaped rate-limit errors
specifically and has no OpenRouter equivalent)."""


def assess_candidates(
    scanner_tickers: list[str],
    technical_scores: dict,
    sector_map: dict,
    *,
    api_key: str | None = None,
    client_factory=None,
) -> list[dict]:
    """The single LLM call: one DeepSeek V4 Pro (OpenRouter) invocation
    assesses every candidate.

    Structured output via ``krepis.llm.LLMClient.structured()`` — the
    fleet-SOTA multi-transport adapter (generalizes the Think-Tank-ratified
    pattern). Returns a list of assessment dicts. Raises on a persistent
    transport or parse failure (all-agents-strict — a challenger that
    silently degrades would pollute the leaderboard).

    ``client_factory`` is the krepis.llm.LLMClient test seam: a callable
    ``(spec, api_key) -> transport_client``. Production leaves it unset."""
    from krepis.llm import LLMClient
    from krepis.llm_config import ModelSpec

    from agents.prompt_loader import load_prompt
    from config import MAX_TOKENS_STRATEGIC, OPENROUTER_API_KEY

    loaded = load_prompt(_PROMPT_NAME)
    prompt = loaded.text + "\n\n## Candidates\n" + _format_candidate_block(
        scanner_tickers, technical_scores, sector_map
    )
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise RuntimeError(
            "single_agent challenger requires an OpenRouter API key: pass "
            "api_key= explicitly, or ensure config.OPENROUTER_API_KEY "
            "resolves (SSM parameter /alpha-engine/OPENROUTER_API_KEY, or "
            "the OPENROUTER_API_KEY environment variable as a fallback)."
        )
    spec = ModelSpec(
        provider="openrouter",
        model=CHALLENGER_MODEL,
        max_tokens=MAX_TOKENS_STRATEGIC,
        # REQUIRED — see module docstring (live-verified 2026-07-19: strict
        # response_format=json_schema is unreliable for DeepSeek-family
        # models on OpenRouter; the JSON-instruction + tolerant-extraction
        # fallback round-tripped this schema correctly every attempt).
        structured_outputs=False,
        reasoning={"exclude": True},
    )
    client = LLMClient(
        spec, api_key=key, client_factory=client_factory,
        max_retries=CHALLENGER_LLM_MAX_RETRIES,
    )
    result = client.structured(
        # Behavior parity with the pre-migration call (a single
        # HumanMessage carrying the whole rendered prompt, no system
        # turn) — the whole prompt goes into user_content unchanged;
        # system is empty rather than splitting/rewriting the prompt.
        system="",
        user_content=prompt,
        schema=RankingProducerOutput,
        schema_name="ranking_producer_output",
        attempts=2,
    )
    logger.info(
        "[single_agent] challenger llm_call model=%s resolved_model=%s "
        "input_tokens=%d output_tokens=%d provider_cost_usd=%s",
        CHALLENGER_MODEL, result.model, result.usage.input_tokens,
        result.usage.output_tokens, result.usage.provider_cost_usd,
    )
    raw: RankingProducerOutput = result.parsed
    return [a.model_dump() for a in raw.assessments]


def run_single_agent_producer(
    run_date: str,
    archive_manager,
    *,
    market_regime: str = "neutral",
    run_time: str = "",
    assess_fn: Callable | None = None,
    population: list[dict] | None = None,
) -> dict:
    """Integration entry: load the SAME scanner candidates the champion reads,
    make the single LLM assessment call, build the payload. ``assess_fn`` is
    injectable for tests; ``population`` may OVERRIDE the SQLite read (the SF
    post-step passes the snapshotted PRIOR population — clean selection
    comparison)."""
    from data.fetchers.price_fetcher import fetch_sp500_sp400_with_sectors
    from data.scanner_orchestrator import _build_technical_scores_from_feature_store

    cand = archive_manager.load_candidates_json(run_date) or {}
    scanner_tickers = cand.get("scanner_tickers", [])
    if population is None:
        population = archive_manager.load_population()
    pop_tickers = [p["ticker"] for p in population]
    prior_theses = archive_manager.load_latest_theses(
        list(dict.fromkeys(scanner_tickers + pop_tickers))
    )
    constituents, sector_map = fetch_sp500_sp400_with_sectors()
    technical_scores, _ = _build_technical_scores_from_feature_store(constituents, sector_map)

    assess = assess_fn or assess_candidates
    assessments = assess(scanner_tickers, technical_scores, sector_map)
    return build_single_agent_signals(
        run_date,
        scanner_tickers=scanner_tickers,
        assessments=assessments,
        technical_scores=technical_scores,
        population=population,
        prior_theses=prior_theses,
        sector_map=sector_map,
        market_regime=market_regime,
        run_time=run_time,
    )
