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

**alpha-engine-config-I6367 (2026-08-03): migrated ``assess_candidates`` off
a pinned ``ModelSpec(provider="openrouter", model="deepseek/deepseek-v4-pro")``
onto the ``high`` model group via ``krepis.router.resolve_group_spec()``.**
Brian's ruling that day is categorical: no agent may be directly linked to
OpenRouter. This call site is why it was made concrete — on 2026-08-02 the
OpenRouter account balance went negative and the pin had nowhere to go, so
every run raised ``ChallengerShadowGapError`` out of the daily Lambda and the
challenger shadow write was lost outright (measured 2026-08-03 22:50 UTC).
``high`` expresses the same capability tier the pin did: this challenger's
per-candidate qualitative reasoning is the "heavier reasoning" tier per
Brian's earlier ruling, vs the lighter tier used for mechanical/high-volume
sites. Dispatched WEEKLY by the Saturday SF's ChallengerShadow state
(``_run_challengers_only``); a SHADOW/best-effort challenger that never
blocks the champion.

(Prior hop, alpha-engine-config-I2997 / 2026-07-19: off direct Anthropic
``ChatAnthropic``/Sonnet-4-6 onto ``krepis.llm.LLMClient``.)

``structured_outputs=False`` is REQUIRED, not a style choice, and is passed
EXPLICITLY so a registry default cannot re-enable it: live-verified
2026-07-19 that strict ``response_format=json_schema`` is unreliable for
DeepSeek-family models — across repeated live calls against this exact
``RankingProducerOutput`` schema, the model intermittently renamed the
required ``ticker`` field (e.g. to ``symbol``/``candidate``), which the
strict path took as ground truth and failed validation on every attempt. The
JSON-instruction + tolerant-extraction path round-tripped this exact schema
correctly on every live attempt tried.

The former ``reasoning={"exclude": True}`` is no longer set here. It guarded
against a reasoning-capable model burning its whole output budget on
chain-of-thought and returning empty content (config#1659 / config#2575) —
but reasoning is a per-model param and the serving model is now a registry
decision, so the registry's ``params.reasoning`` for the entry that actually
serves is authoritative. Setting it from here would override a per-model fact
with a guess made for a different model.
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
        default_factory=list,
        min_length=1,
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
            qual_score=qual,  # the single agent's qualitative read
            sector_modifier=1.0,  # neutral; no macro agent
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
        "[single_agent] run_date=%s assessed=%d scored=%d buy_candidates=%d new_entrants=%d",
        run_date,
        len(assessments),
        len(theses),
        len(payload.get("buy_candidates", [])),
        len(advanced_tickers),
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


CHALLENGER_GROUP = "high"
"""Capability tier this challenger asks for — NOT a model, and not a provider.

Brian's ruling 2026-08-03 (alpha-engine-config-I6367): no agent may be
directly linked to OpenRouter. This call site previously pinned
``ModelSpec(provider="openrouter", model="deepseek/deepseek-v4-pro")`` and
died with it: on 2026-08-02 the OpenRouter account balance went negative and
every run raised ``ChallengerShadowGapError`` out of the daily Lambda, losing
the challenger shadow write entirely (measured 2026-08-03 22:50 UTC).

``high`` is the same capability tier the pin expressed — heavier per-candidate
qualitative reasoning, per Brian's earlier ruling that distinguishes this from
the lighter tier used for mechanical/high-volume sites. Its members and their
order are a registry decision (``LLM_MODEL_REGISTRY.yaml``), resolved above
this module. It asks for a tier and says where it runs; nothing else."""

CHALLENGER_EXEC_CONTEXT = "lambda"
"""Where this code runs — a fact, never a routing preference (R28/R29).

The registry declares ``lambda`` on NO model entry, deliberately: a Lambda has
no local egress proxy and no private-network peer, so the router is its only
path. That is what makes this call site **fail closed** rather than falling
through to a direct provider endpoint whose traffic is DLP-unscanned. Do not
"fix" a router outage here by widening what this context admits — the
Director's ``exclude_route="litellm_proxy"`` did exactly that and served
``glm-5.2`` at openrouter.ai unscanned while logging a healthy route
(alpha-engine-config-I6183)."""

CALLSITE_ID = "single-agent-quant"
"""Cost-attribution join key for this call site (krepis >= 0.23 requires it).

Every LLM call from here emits a cost row stamped with this literal; the
matching row in ``alpha-engine-config/private-docs/LLM_CALLSITE_REGISTRY.yaml``
carries the same value as its ``id``. The two are a lockstep pair — renaming
one without the other silently orphans this call site's spend from its
registry entry (the row still validates, the cost rows just never join). Slug
follows the registry's existing kebab-case convention (``thinktank-thesis``,
``evaljudge-sync``)."""

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
    cost_sink=None,
) -> list[dict]:
    """The single LLM call: one DeepSeek V4 Pro (OpenRouter) invocation
    assesses every candidate.

    Structured output via ``krepis.llm.LLMClient.structured()`` — the
    fleet-SOTA multi-transport adapter (generalizes the Think-Tank-ratified
    pattern). Returns a list of assessment dicts. Raises on a persistent
    transport or parse failure (all-agents-strict — a challenger that
    silently degrades would pollute the leaderboard).

    ``client_factory`` is the krepis.llm.LLMClient test seam: a callable
    ``(spec, api_key) -> transport_client``. Production leaves it unset.
    ``cost_sink`` is a test seam (any callable taking a record dict). When
    omitted, ``LLMClient`` resolves the PROCESS-DEFAULT sink from the
    environment — the one ``krepis.cost_sink.flush_default_sink()`` can reach
    at the end of a Lambda invocation (alpha-engine-config-I7423). This
    function must NOT construct its own; see the note at the call site."""
    from krepis.llm import LLMClient
    from krepis.router import resolve_group_spec

    from agents.prompt_loader import load_prompt
    from config import MAX_TOKENS_STRATEGIC, S3_BUCKET

    loaded = load_prompt(_PROMPT_NAME)
    prompt = (
        loaded.text + "\n\n## Candidates\n" + _format_candidate_block(scanner_tickers, technical_scores, sector_map)
    )
    # NO private sink (alpha-engine-config-I7423). `LLMClient` resolves
    # `krepis.cost_sink.default_sink_from_env()` when `cost_sink is None`, and
    # that PROCESS-DEFAULT sink is the one `flush_default_sink()` can reach.
    #
    # Constructing a private `S3JsonlCostSink` here shadowed it: the records
    # went into an instance nothing else held a reference to, whose only exit
    # path was `register_atexit=True` — and an AWS Lambda container is FROZEN
    # between invocations, not exited, so `atexit` never runs.
    #
    # Measured 2026-08-16 on weekly-SF execution `watch-rerun-2026-08-16-1`,
    # AFTER the handler-level flush shipped: `ChallengerShadow` made a real
    # DeepSeek call (`input_tokens=4296 output_tokens=7906`), wrote both
    # shadow signal sets, and `AggregateCosts` still reported
    # `1 stage(s) ran and emitted no cost record: single-agent-quant`. The
    # `cost sink: process default active` line never appeared in the log,
    # because the default sink was never constructed at all.
    #
    # `cost_sink` stays a parameter: it is the test seam, and an explicit
    # injection still wins.
    # The registry decides model, endpoint and credential; this module decides
    # only which capability tier it wants and where it is running. krepis
    # resolves the credential by NAME at call time, so no key is read here —
    # which is why `api_key` is now only a test seam.
    #
    # `structured_outputs=False` is REQUIRED, not a style choice, and is
    # passed explicitly so a registry default cannot re-enable it: live-
    # verified 2026-07-19 that strict `response_format=json_schema` is
    # unreliable for DeepSeek-family models against this exact
    # `RankingProducerOutput` schema — the model intermittently renamed the
    # required `ticker` field and the strict path took that as ground truth,
    # failing validation on every attempt. The JSON-instruction + tolerant-
    # extraction path round-tripped it correctly every time.
    spec, route = resolve_group_spec(
        CHALLENGER_GROUP,
        exec_context=CHALLENGER_EXEC_CONTEXT,
        # This call site builds an LLMClient on the openai transport. Asking
        # for the anthropic wire would let a fallback hand it a URL its
        # transport cannot speak.
        wire="openai",
        max_tokens=MAX_TOKENS_STRATEGIC,
        structured_outputs=False,
    )
    client = LLMClient(
        spec,
        api_key=api_key,
        client_factory=client_factory,
        max_retries=CHALLENGER_LLM_MAX_RETRIES,
        # REQUIRED since krepis 0.23 (krepis/llm.py::LLMClient.__init__ —
        # validated non-empty, raises TypeError when omitted). It is the join
        # key between this call's emitted cost row and its
        # LLM_CALLSITE_REGISTRY.yaml entry, so the literal MUST stay in sync
        # with that row's `id` (alpha-engine-config, id: single-agent-quant).
        callsite_id=CALLSITE_ID,
        cost_sink=cost_sink,
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
        "[single_agent] challenger llm_call group=%s route=%s requested=%s "
        "resolved_model=%s input_tokens=%d output_tokens=%d "
        "provider_cost_usd=%s",
        CHALLENGER_GROUP,
        route.get("route"),
        spec.model,
        result.model,
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.provider_cost_usd,
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
    prior_theses = archive_manager.load_latest_theses(list(dict.fromkeys(scanner_tickers + pop_tickers)))
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
