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

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from config import ANTHROPIC_API_KEY, MAX_TOKENS_STRATEGIC, PER_STOCK_MODEL, QUAL_MAX_ITERATIONS
from agents.prompt_loader import load_prompt
from agents.sector_teams.qual_tools import create_qual_tools
from graph.llm_cost_tracker import get_cost_telemetry_callback
from strict_mode import is_strict_validation_enabled

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

    system_prompt = _build_system_prompt(team_id, market_regime, len(quant_top5))

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
    # half of the prompt-pair is independently grep-able.
    system_prompt_loaded = load_prompt("qual_analyst_system")
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
        result = invoke_with_rate_limit_retry(
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
        extract_resp = invoke_with_rate_limit_retry(
            lambda: structured_llm.invoke(
                [extract_msg],
                config={"metadata": _ls_metadata},
            ),
            label=f"qual:{team_id}:extract",
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

        log.info("[qual:%s] completed — %d assessments, %d tool calls",
                 team_id, len(assessments), len(tool_calls))

        return {
            "team_id": team_id,
            "assessments": assessments,
            "additional_candidate": additional_candidate,
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
            "error": None,
            "partial": False,
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
        }


def _build_system_prompt(team_id: str, market_regime: str, n_picks: int) -> str:
    return load_prompt("qual_analyst_system").format(
        team_title=team_id.title(),
        n_picks=n_picks,
        market_regime=market_regime,
    )


from agents.langchain_utils import (
    extract_tool_calls as _extract_tool_calls,
    get_final_text as _get_final_text,
)
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    invoke_with_rate_limit_retry,
)
