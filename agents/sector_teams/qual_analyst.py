"""
Qualitative Analyst Agent — LangGraph create_react_agent with LangChain tools.

Reviews the quant analyst's top 5 picks with qualitative data:
news, analyst reports, insider activity, SEC filings, prior theses.
Produces a single holistic qual_score (0-100) per stock.
May identify 0-1 additional candidates that quant missed.
"""

from __future__ import annotations

import logging
from typing import Optional

from alpha_engine_lib.pillars import QualitativePillarAssessment
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, ConfigDict, Field

from config import (
    ANTHROPIC_API_KEY,
    MAX_TOKENS_STRATEGIC,
    PER_STOCK_MODEL,
    PILLAR_EMIT_ENABLED,
    QUAL_MAX_ITERATIONS,
)
from agents.prompt_loader import load_prompt
from agents.sector_teams.qual_tools import create_qual_tools
from graph.llm_cost_tracker import get_cost_telemetry_callback
from strict_mode import is_strict_validation_enabled


class _QualPillarItem(BaseModel):
    """One ticker's pillar assessment — wraps the lib's
    QualitativePillarAssessment with a ticker key for batch emission.

    Research-internal transient shape used only as the wrapper for the
    second structured-output extraction call when PILLAR_EMIT_ENABLED is
    on. Not part of the lib schema surface — the lib emits
    QualitativePillarAssessment per-stock; this wrapper just keys by
    ticker for batch transport.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    pillar_assessment: QualitativePillarAssessment


class _QualPillarBatch(BaseModel):
    """Wrapper for the qual analyst's per-ticker pillar emission batch
    (Phase 2 of attractiveness-pillars-260520 arc).

    The qual ReAct loop reviews ~5 picks; this batch holds one
    ``_QualPillarItem`` per ticker. Empty list is permitted (e.g., the
    agent's reasoning didn't yield a clean pillar decomposition for any
    ticker — lax-mode degrades to empty rather than crashing).
    """

    model_config = ConfigDict(extra="allow")

    items: list[_QualPillarItem] = Field(default_factory=list)

log = logging.getLogger(__name__)

# +2 retained as defensive margin (was load-bearing for response_format's
# extra call inside the LangGraph subgraph; after the 2026-05-02 refactor
# the extraction is decoupled and the +2 is unused). See quant_analyst.py.
_QUAL_RECURSION_LIMIT = QUAL_MAX_ITERATIONS * 2 + 2


def run_qual_analyst(
    team_id: str,
    quant_top5: list[dict],
    prior_theses: dict[str, dict],
    market_regime: str,
    run_date: str,
    api_key: Optional[str] = None,
    price_data: Optional[dict] = None,
    episodic_memories: dict[str, list] | None = None,
    semantic_memories: dict[str, list] | None = None,
) -> dict:
    """
    Run the qual analyst ReAct agent.

    Returns:
        {
            "team_id": str,
            "assessments": list[dict],  # qual_score + bull/bear per stock
            "additional_candidate": dict | None,
            "tool_calls": list[dict],
            "iterations": int,
        }
    """
    # Strategic-tier max_tokens covers the structured-output extraction
    # call at the end of this function — QualAnalystOutput.assessments is
    # a list of ~5 QualAssessment entries × ~600 tokens each (bull_case +
    # bear_case + catalysts + qual_score + reasoning) plus tool-use JSON
    # envelope. ReAct-turn calls (which share this llm instance) are
    # individually small so the higher cap doesn't shift their token
    # cost (Anthropic bills emitted tokens, not the cap).
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_STRATEGIC,
        max_retries=SECTOR_TEAM_LLM_MAX_RETRIES,
        callbacks=[get_cost_telemetry_callback()],
    )

    tools = create_qual_tools({
        "prior_theses": prior_theses,
        "price_data": price_data or {},
        "episodic_memories": episodic_memories or {},
        "semantic_memories": semantic_memories or {},
    })

    # Wrap the system prompt in a SystemMessage with content-block
    # cache_control so the (tools + system) prefix caches across the
    # ~50 qual ReAct calls per weekly run (and across the inner ReAct
    # loop iterations for one ticker). Cache hit/miss surfaces in
    # ``graph.llm_cost_tracker`` telemetry — see quant_analyst.py for
    # the full rationale and the 4096-token Haiku-4-5 minimum check.
    system_prompt_text = _build_system_prompt(team_id, market_regime, len(quant_top5))
    system_prompt = SystemMessage(content=[{
        "type": "text",
        "text": system_prompt_text,
        "cache_control": {"type": "ephemeral"},
    }])

    # ReAct loop only (no response_format). Structured-output extraction
    # is decoupled and runs as ``with_structured_output(include_raw=True)``
    # after the loop ends — same convention as macro_agent.py /
    # peer_review.py / ic_cio.py. 2026-05-02 refactor rationale: a Haiku
    # `response_format=...` extraction inside ``create_react_agent``
    # occasionally returns markdown-fenced JSON text instead of using the
    # structured-output tool, crashing the SF with a ValidationError. The
    # decoupled call is constrained at the API boundary and honors the
    # strict-mode parsing-error contract (lax-mode falls back to empty
    # assessments; strict-mode raises).
    from graph.state_schemas import QualAnalystOutput
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    picks_text = "\n".join(
        f"  {i+1}. {p['ticker']} (quant_score={p.get('quant_score', '?')}): "
        f"{p.get('rationale', 'no rationale')}"
        for i, p in enumerate(quant_top5)
    )

    user_prompt = load_prompt("qual_analyst_user")
    user_message = user_prompt.format(
        run_date=run_date,
        market_regime=market_regime,
        picks_text=picks_text,
    )
    # System prompt's metadata anchors LangSmith trace attribution; the
    # user prompt's version + hash piggyback so a future drift in either
    # half of the prompt-pair is independently grep-able. Under
    # PILLAR_EMIT_ENABLED the system prompt loaded is
    # ``qual_analyst_system_pillars`` so metadata names IT, not the
    # legacy template (matches the prompt actually fed to the ReAct loop).
    _system_prompt_name = (
        "qual_analyst_system_pillars" if PILLAR_EMIT_ENABLED
        else "qual_analyst_system"
    )
    system_prompt_loaded = load_prompt(_system_prompt_name)
    _ls_metadata = {
        **system_prompt_loaded.langsmith_metadata(),
        "user_prompt_version": user_prompt.version,
        "user_prompt_hash": user_prompt.hash[:12],
    }

    log.info("[qual:%s] starting ReAct agent with %d picks", team_id, len(quant_top5))

    try:
        # Token usage from this ReAct loop's multiple Anthropic calls
        # accumulates into the active ``track_llm_cost`` frame opened
        # by the outer ``sector_team_node`` in research_graph.py.
        result = invoke_react_with_recovery(
            lambda: agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "recursion_limit": _QUAL_RECURSION_LIMIT,
                    "metadata": _ls_metadata,
                },
            ),
            label=f"qual:{team_id}:react",
        )

        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        final_text = _get_final_text(messages)

        # ── Decoupled structured-output extraction ──────────────────────
        # Mirrors quant_analyst + macro_agent / peer_review / ic_cio. The
        # extraction call is constrained at the API boundary so a Haiku
        # response with markdown-fenced text can no longer end up in a
        # Pydantic field as a raw string.
        if not final_text or not final_text.strip():
            raise RuntimeError(
                f"[qual:{team_id}] ReAct loop produced empty final_text — "
                f"nothing to extract assessments from. tool_calls={len(tool_calls)}"
            )
        structured_llm = llm.with_structured_output(
            QualAnalystOutput, include_raw=True,
        )
        extract_msg = HumanMessage(content=(
            "Extract the per-ticker assessments and any additional "
            "candidate from this analyst's answer into the structured "
            "schema. Use only what's in the text — do not invent picks. "
            "If the analyst produced no assessments, return an empty list.\n\n"
            f"--- ANALYST ANSWER ---\n{final_text}"
        ))
        extract_resp = invoke_structured_with_validation_retry(
            structured_llm,
            [extract_msg],
            label=f"qual:{team_id}:extract",
            ls_metadata=_ls_metadata,
        )
        parsed: QualAnalystOutput | None = extract_resp.get("parsed")
        parsing_error = extract_resp.get("parsing_error")
        if parsing_error is not None:
            msg = (
                f"[qual:{team_id}] structured-output parse failed: "
                f"{type(parsing_error).__name__}: {parsing_error}"
            )
            if is_strict_validation_enabled():
                raise RuntimeError(msg)
            log.warning("%s — falling back to empty assessments (lax mode)", msg)
            parsed = QualAnalystOutput()
        assert parsed is not None
        # Convert QualAssessment Pydantic models to dicts for downstream
        # peer_review consumption (which uses dict-access patterns).
        assessments = [a.model_dump() for a in parsed.assessments]
        additional_candidate = (
            parsed.additional_candidate.model_dump()
            if parsed.additional_candidate is not None
            else None
        )

        # ── Pillar-assessment emission (Phase 2 of pillars arc, gated) ──
        # When PILLAR_EMIT_ENABLED, a SECOND structured-output extraction
        # runs against the same final_text and produces a per-ticker
        # ``QualitativePillarAssessment`` (6-pillar decomposition). The
        # legacy QualAnalystOutput extraction above is untouched — pillar
        # emission is purely additive observability at this phase. Phase
        # 4 (composite scoring refactor) will consume it; Phase 2 just
        # ships the substrate behind the flag.
        pillar_assessments: dict[str, dict] = {}
        if PILLAR_EMIT_ENABLED:
            pillar_assessments = _extract_pillar_assessments(
                llm=llm,
                final_text=final_text,
                team_id=team_id,
                ls_metadata=_ls_metadata,
            )

        log.info(
            "[qual:%s] completed — %d assessments, %d tool calls, %d pillar",
            team_id, len(assessments), len(tool_calls), len(pillar_assessments),
        )

        return {
            "team_id": team_id,
            "assessments": assessments,
            "additional_candidate": additional_candidate,
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
            "error": None,
            "partial": False,
            "pillar_assessments": pillar_assessments,
            # Ground-truth reasoning for retrospective decision review
            # (L4567) — bounded; captured into the decision artifact +
            # carried in SectorTeamOutput state without bloat.
            "final_text": final_text[:_FINAL_TEXT_CAP],
            "transcript": _serialize_transcript(messages),
        }

    except GraphRecursionError as e:
        # Budget exhausted before the agent reached a stop condition.
        # Mirrors quant_analyst's policy: degraded-but-non-fatal — this
        # team contributes zero assessments but doesn't crash the SF.
        log.warning(
            "[qual:%s] recursion budget (%d transitions) exhausted before "
            "stop condition — accepting partial result (0 assessments). "
            "score_aggregator will proceed with this team excluded.",
            team_id, _QUAL_RECURSION_LIMIT,
        )
        return {
            "team_id": team_id,
            "assessments": [],
            "additional_candidate": None,
            "tool_calls": [],
            "iterations": _QUAL_RECURSION_LIMIT,
            "error": None,
            "partial": True,
            "partial_reason": "recursion_limit_exhausted",
            "pillar_assessments": {},
        }

    except Exception as e:
        # Record the error so downstream (score_aggregator) can hard-fail
        # instead of silently treating an exception the same as an LLM
        # legitimately producing zero assessments. Recursion budget
        # exhaustion is handled separately above as partial.
        log.error("[qual:%s] ReAct agent failed: %s", team_id, e)
        return {
            "team_id": team_id,
            "assessments": [],
            "additional_candidate": None,
            "tool_calls": [],
            "iterations": 0,
            "error": f"{type(e).__name__}: {e}",
            "partial": False,
            "pillar_assessments": {},
        }


def _build_system_prompt(team_id: str, market_regime: str, n_picks: int) -> str:
    """Load the qual analyst's system prompt — pillar-rubric variant when
    ``PILLAR_EMIT_ENABLED``, legacy otherwise.

    Both prompts accept the same ``{team_title}`` / ``{n_picks}`` /
    ``{market_regime}`` placeholders so the format() call is uniform.
    The pillar-rubric variant adds 6-pillar decomposition guidance +
    moat-archetype taxonomy so the agent's reasoning surfaces the
    decomposed signal cleanly for the second extraction call.
    """
    prompt_name = (
        "qual_analyst_system_pillars" if PILLAR_EMIT_ENABLED
        else "qual_analyst_system"
    )
    return load_prompt(prompt_name).format(
        team_title=team_id.title(),
        n_picks=n_picks,
        market_regime=market_regime,
    )


def _extract_pillar_assessments(
    llm,
    final_text: str,
    team_id: str,
    ls_metadata: dict,
) -> dict[str, dict]:
    """Run the second structured-output extraction for per-ticker pillar
    assessments (Phase 2 of attractiveness-pillars-260520).

    Mirrors the legacy ``QualAnalystOutput`` extraction pattern: bound at
    the API boundary via ``with_structured_output(include_raw=True)``,
    strict-mode raises on parse failure, lax-mode logs + returns an empty
    dict. The lib's ``QualitativePillarAssessment`` schema is used directly
    inside a thin research-internal wrapper that keys by ticker.

    Returns: ``{ticker: pillar_assessment_dict}`` where each value is the
    model_dump() of a QualitativePillarAssessment (per-pillar subscores +
    moat assessment + catalyst horizon modulation). Empty dict on lax-mode
    parse failure or when the agent's reasoning yields no clean pillar
    decomposition. Strict-mode raises ``RuntimeError`` on parse failure.
    """
    structured_llm = llm.with_structured_output(
        _QualPillarBatch, include_raw=True,
    )
    extract_msg = HumanMessage(content=(
        "Extract a per-ticker pillar decomposition from this analyst's "
        "answer. For EACH ticker the analyst assessed, produce a "
        "QualitativePillarAssessment with subscores on the 6 pillars "
        "(quality, value, momentum, growth, stewardship, defensiveness), "
        "a structured MoatAssessment for the Quality pillar, and a "
        "catalyst_horizon_modulation ∈ [-20, 20] for any near-term "
        "catalyst effect. Use ONLY what's in the text — do not invent "
        "pillar evidence the analyst did not surface. If the analyst's "
        "reasoning doesn't support a pillar score, score it 50 (neutral) "
        "with confidence 'low' and an evidence list explaining the "
        "absence. If no tickers were assessed, return an empty items "
        "list.\n\n"
        f"--- ANALYST ANSWER ---\n{final_text}"
    ))
    extract_resp = invoke_structured_with_validation_retry(
        structured_llm,
        [extract_msg],
        label=f"qual:{team_id}:extract-pillars",
        ls_metadata=ls_metadata,
    )
    parsed: _QualPillarBatch | None = extract_resp.get("parsed")
    parsing_error = extract_resp.get("parsing_error")
    if parsing_error is not None:
        # Pillar-hardening item 2 (2026-05-21 AQR cutover incident):
        # promoted from lax-mode-return-empty to ALWAYS-RAISE. Empty-dict
        # propagation under non-zero pillar_weights produces a degenerate
        # composite (per [[zero-legacy-weight-degenerates-on-pillar-emit-
        # failure]]); silent-fail class the new CLAUDE.md fail-loud rule
        # prohibits. Matches the all-agents-strict pattern shipped via
        # research #195 (2026-05-16) — all-or-nothing for pillar emission.
        # The consumer-side coverage guard in compute_composite_breakdown
        # is the per-ticker safety net for missing-input cases this raise
        # WOULDN'T have caught (e.g. factor_profile absent on a held
        # stock with no pillar_assessment).
        msg = (
            f"[qual:{team_id}] pillar-assessment parse failed: "
            f"{type(parsing_error).__name__}: {parsing_error}"
        )
        raise RuntimeError(msg)
    assert parsed is not None
    return {
        item.ticker: item.pillar_assessment.model_dump()
        for item in parsed.items
    }


from agents.langchain_utils import (
    extract_tool_calls as _extract_tool_calls,
    get_final_text as _get_final_text,
    serialize_transcript as _serialize_transcript,
)
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    invoke_react_with_recovery,
    invoke_structured_with_validation_retry,
)

# Cap the analyst's prose answer persisted into the decision artifact for
# retrospective "why" review (L4567); the transcript is bounded inside
# ``serialize_transcript``.
_FINAL_TEXT_CAP = 6_000
