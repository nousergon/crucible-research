"""
Quant Analyst Agent — LangGraph create_react_agent with LangChain tools.

Each sector team's quant analyst autonomously screens its sector universe
using ReAct tool-calling. The agent decides its own screening strategy —
different sectors naturally use different tools and thresholds.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from config import ANTHROPIC_API_KEY, MAX_TOKENS_STRATEGIC, PER_STOCK_MODEL, QUANT_MAX_ITERATIONS
from agents.prompt_loader import load_prompt
from agents.sector_teams.quant_tools import create_quant_tools
from agents.sector_teams.team_config import TEAM_SCREENING_PARAMS, QUANT_TOP_N, MAX_TICKERS_IN_PROMPT
from graph.llm_cost_tracker import get_cost_telemetry_callback
from strict_mode import is_strict_validation_enabled

log = logging.getLogger(__name__)

# LangGraph state-transition budget. Each ReAct round = 1 LLM message + 1 tool
# response = 2 transitions, so QUANT_MAX_ITERATIONS rounds need 2× that.
# The +2 buffer is RETAINED defensively (was added 2026-05-02 to cover
# response_format's extra extraction call inside the subgraph). After the
# 2026-05-02 refactor that decouples the structured-output extraction from
# the ReAct loop, the +2 is no longer load-bearing — but the cost is one
# transition slot of headroom and removing it offers no benefit. Keep as
# defensive margin.
_QUANT_RECURSION_LIMIT = QUANT_MAX_ITERATIONS * 2 + 2


def _emit_retry_telemetry_safely(
    *, team_id: str, attempted: bool, succeeded: bool,
) -> None:
    """Best-effort emit one retry-event datapoint to AlphaEngine/Agents.

    Wrapped in broad except so a CW outage never bubbles up into the
    research pipeline. Mirrors the gate in
    ``graph.agent_telemetry.emit_agent_retry`` (which is itself
    best-effort), but the import is wrapped here too so a stale lib
    image at deploy time can't break the agent path.
    """
    try:
        from graph.agent_telemetry import emit_agent_retry

        emit_agent_retry(
            agent_id=f"sector_quant:{team_id}",
            attempted=attempted,
            succeeded=succeeded,
        )
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "[quant:%s] retry telemetry emission failed: %s", team_id, exc,
        )


def _render_focus_list_for_prompt(focus_list: list[dict]) -> str:
    """Render a per-team focus list as a compact prompt block.

    Table-shaped per scanner-260514.md §5.2:
        TICKER | sector | stance | focus_score | momentum_p | quality_p | value_p | low_vol_p

    Empty input → empty string (caller falls back to the full sector slice
    when this rendering is empty AND FOCUS_LIST_GATING_ENABLED is true).
    """
    if not focus_list:
        return ""
    header = (
        "TICKER | sector | stance | focus_score | "
        "momentum_p | quality_p | value_p | low_vol_p"
    )
    rows = []
    for e in focus_list:
        rows.append(
            "{ticker} | {sector} | {stance} | {focus_score:.1f} | "
            "{mom} | {qual} | {val} | {lv}".format(
                ticker=e.get("ticker", "?"),
                sector=e.get("sector", "?"),
                stance=e.get("stance", "?"),
                focus_score=float(e.get("focus_score") or 0.0),
                mom=_fmt_pct(e.get("momentum_score")),
                qual=_fmt_pct(e.get("quality_score")),
                val=_fmt_pct(e.get("value_score")),
                lv=_fmt_pct(e.get("low_vol_score")),
            )
        )
    return header + "\n" + "\n".join(rows)


def _fmt_pct(v) -> str:
    """Format a 0-100 percentile compactly. None → '?'."""
    if v is None:
        return "?"
    try:
        return f"{float(v):.0f}"
    except (TypeError, ValueError):
        return "?"


# Retry preamble injected into the user message on the second attempt
# when the first attempt produced zero picks despite running tools.
# Lives in code (not config) because it's a small fixed string used
# only when the retry path fires; flow-doctor surfaces retry events
# from the WARNING log lines below so observability is preserved.
# Kept literal-formattable: only one variable interpolated.
_QUANT_RETRY_PREAMBLE = (
    "RETRY NOTICE: Your previous attempt at this analysis produced ZERO "
    "ranked picks despite running {n_tool_calls} tool calls. The "
    "downstream pipeline (qualitative analyst, peer review, CIO) "
    "REQUIRES at least 3 ranked picks per sector — emitting an empty "
    "result silently degrades portfolio coverage and is not an "
    "acceptable outcome.\n\n"
    "On this retry, you MUST produce at least 3 ranked picks. If no "
    "setup meets your prior conviction threshold, lower the threshold "
    "and rank by relative strength among the available tickers. State "
    "explicitly when conviction is low rather than emitting empty. If "
    "data quality is the obstacle (tool calls returning sparse or "
    "stale data), produce your best ranking from what's available and "
    "flag the data-quality concern in the rationale.\n\n"
    "ORIGINAL ANALYSIS REQUEST FOLLOWS:\n\n"
)


def run_quant_analyst(
    team_id: str,
    sector_tickers: list[str],
    market_regime: str,
    price_data: dict,
    technical_scores: dict,
    run_date: str,
    api_key: Optional[str] = None,
    _retry_preamble: Optional[str] = None,
    focus_list: Optional[list[dict]] = None,
    override_tickers: Optional[list[str]] = None,
) -> dict:
    """
    Run the quant analyst ReAct agent for a sector team.

    Returns:
        {
            "team_id": str,
            "ranked_picks": list[dict],
            "tool_calls": list[dict],
            "iterations": int,
        }
    """
    team_params = TEAM_SCREENING_PARAMS.get(team_id, {})

    # Create LLM. Cost-telemetry callback aggregates token usage across
    # the ReAct loop's multiple Anthropic calls into the active
    # ``track_llm_cost`` frame. Strategic-tier max_tokens mirrors
    # qual_analyst — covers the structured-output extraction call's
    # list of ranked picks (5 picks × ~400 tokens each + envelope ≈
    # 2500 tokens typical, more at verbose end). Same risk class.
    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_STRATEGIC,
        max_retries=SECTOR_TEAM_LLM_MAX_RETRIES,
        callbacks=[get_cost_telemetry_callback()],
    )

    # Create tools with shared context. market_regime + factor_blend_regime_weights
    # are threaded for Phase 2's @tool get_factor_profile so it can render the
    # regime-blended focus score alongside the within-sector factor composites.
    # factor_profiles is read lazily inside the tool factory on cache miss.
    #
    # PR 4 of scanner-placement arc:
    #   - focus_list_tickers is the set of in-focus-list tickers for THIS team.
    #     get_factor_profile invocations on tickers OUTSIDE this set are
    #     appended to override_tickers (shared mutable list, propagated back
    #     up to sector_team's team_result so archive_writer can project
    #     agent_override=1 onto the audit table).
    from config import FACTOR_BLEND_REGIME_WEIGHTS
    focus_list_tickers = {
        e.get("ticker") for e in (focus_list or []) if e.get("ticker")
    }
    tools = create_quant_tools({
        "price_data": price_data,
        "technical_scores": technical_scores,
        "market_regime": market_regime,
        "factor_blend_regime_weights": FACTOR_BLEND_REGIME_WEIGHTS,
        "focus_list_tickers": focus_list_tickers,
        "override_tickers": override_tickers if override_tickers is not None else [],
    })

    # Build system prompt. Wrapped in a ``SystemMessage`` with a
    # content-block ``cache_control`` marker so the assembled prefix
    # (tools + system) caches across the ~3000 quant ReAct calls per
    # weekly run. The breakpoint is on the system block — caches both
    # ``tools`` and ``system`` together since the render order is
    # ``tools → system → messages``. Reuse model: same prefix across
    # every ticker in the sector, same prefix across all sectors when
    # market_regime is unchanged (regime is set once at pipeline start).
    # Requires the assembled prefix to clear Haiku-4-5's 4096-token
    # cache minimum — verified via scripts/measure_cache_prefixes.py
    # before merge. On miss, ``cache_creation_input_tokens`` stays 0 in
    # ``graph.llm_cost_tracker`` telemetry and the rollout is a no-op —
    # no functional regression.
    system_prompt_text = _build_system_prompt(team_id, team_params, market_regime, len(sector_tickers))
    system_prompt = SystemMessage(content=[{
        "type": "text",
        "text": system_prompt_text,
        "cache_control": {"type": "ephemeral"},
    }])

    # Create ReAct agent via LangGraph. The ReAct loop runs until the model
    # produces a final-text answer (no more tool calls). The structured-
    # output extraction is decoupled and runs as a separate
    # ``with_structured_output`` call after the loop — see "structured
    # extraction" block below. Mirrors the convention in macro_agent.py /
    # peer_review.py / ic_cio.py.
    #
    # 2026-05-02 refactor rationale: the prior ``response_format=
    # QuantAnalystOutput`` mechanism inside ``create_react_agent`` adds a
    # post-loop extraction call to the LangGraph subgraph. That call is
    # not constrained — Haiku occasionally returns markdown-fenced JSON
    # text instead of using the structured-output tool, which crashes the
    # SF with a Pydantic ``ValidationError`` (input_value is the entire
    # string-with-fences assigned to ``ranked_picks``). Decoupling lets us
    # drive ``with_structured_output`` directly with ``include_raw=True``
    # and the strict-mode parsing-error contract, which is the established
    # pattern across every other LLM-output site in this codebase.
    from graph.state_schemas import QuantAnalystOutput
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    # Build input message
    #
    # PR 4 of scanner-placement arc: when FOCUS_LIST_GATING_ENABLED and a
    # non-empty focus_list is available for this team, the agent receives
    # the regime-blended factor-composite focus list (top-N within-sector
    # names) as ticker_list AND a rendered focus_list_rendered block with
    # per-ticker scores + stance. The agent can still reach outside via
    # @tool get_factor_profile — those calls are tagged agent_override=1
    # by the tool wrapper for the audit table.
    #
    # Default-off (FOCUS_LIST_GATING_ENABLED=false): original full-sector-
    # slice contract preserved exactly. Same when gating is on but the
    # focus_list is empty (graceful degrade if the factor substrate is
    # unavailable this cycle).
    from config import FOCUS_LIST_GATING_ENABLED
    use_focus_gating = bool(FOCUS_LIST_GATING_ENABLED and focus_list)
    if use_focus_gating:
        focus_tickers_ordered = [e["ticker"] for e in focus_list if e.get("ticker")]
        ticker_list = ", ".join(focus_tickers_ordered)
        focus_list_rendered = _render_focus_list_for_prompt(focus_list)
        universe_size_for_prompt = len(focus_tickers_ordered)
        log.info(
            "[quant:%s] FOCUS_LIST_GATING_ENABLED: agent sees %d focus-list "
            "tickers (was %d full sector slice)",
            team_id, len(focus_tickers_ordered), len(sector_tickers),
        )
    else:
        ticker_list = ", ".join(sector_tickers[:MAX_TICKERS_IN_PROMPT])
        focus_list_rendered = ""
        universe_size_for_prompt = len(sector_tickers)

    user_prompt = load_prompt("quant_analyst_user")
    # focus_list_rendered is a new optional template var — alpha-engine-config's
    # sibling prompt PR will reference it. Until that lands str.format
    # silently ignores it (extra kwargs are not an error).
    user_message = user_prompt.format(
        run_date=run_date,
        market_regime=market_regime,
        universe_size=universe_size_for_prompt,
        ticker_list=ticker_list,
        quant_top_n=QUANT_TOP_N,
        focus_list_rendered=focus_list_rendered,
    )
    # On retry, prepend the retry preamble so the LLM sees the "previous
    # attempt produced zero picks" context before the original request.
    # The preamble is a code-side constant (_QUANT_RETRY_PREAMBLE) — not
    # versioned via the prompt registry because it's a fixed, bounded
    # string fired only on the empty-output retry path.
    if _retry_preamble:
        user_message = _retry_preamble + user_message
    # System prompt's metadata anchors LangSmith trace attribution; the
    # user prompt's version + hash piggyback so a future drift in either
    # half of the prompt-pair is independently grep-able.
    system_prompt_loaded = load_prompt("quant_analyst_system")
    _ls_metadata = {
        **system_prompt_loaded.langsmith_metadata(),
        "user_prompt_version": user_prompt.version,
        "user_prompt_hash": user_prompt.hash[:12],
    }

    log.info("[quant:%s] starting ReAct agent with %d tickers", team_id, len(sector_tickers))

    try:
        # Token usage from this ReAct loop's multiple Anthropic calls
        # accumulates into the active ``track_llm_cost`` frame opened
        # by the outer ``sector_team_node`` in research_graph.py.
        result = invoke_with_rate_limit_retry(
            lambda: agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "recursion_limit": _QUANT_RECURSION_LIMIT,
                    "metadata": _ls_metadata,
                },
            ),
            label=f"quant:{team_id}:react",
        )

        messages = result.get("messages", [])
        tool_calls = _extract_tool_calls(messages)
        final_text = _get_final_text(messages)

        # ── Decoupled structured-output extraction ──────────────────────
        # Drives ``with_structured_output(include_raw=True)`` directly so
        # the strict-mode parsing-error contract is honored — no markdown-
        # fence-text confusion possible because the extraction call is
        # constrained at the API boundary (Anthropic tool-use). Mirrors
        # macro_agent.py:184 / peer_review.py / ic_cio.py.
        if not final_text or not final_text.strip():
            raise RuntimeError(
                f"[quant:{team_id}] ReAct loop produced empty final_text — "
                f"nothing to extract structured picks from. tool_calls={len(tool_calls)}"
            )
        structured_llm = llm.with_structured_output(
            QuantAnalystOutput, include_raw=True,
        )
        extract_msg = HumanMessage(content=(
            "Extract the final ranked picks from this analyst's answer "
            "into the structured schema. Use only what's in the text — "
            "do not invent picks. If the analyst produced no picks, "
            "return an empty list.\n\n"
            f"--- ANALYST ANSWER ---\n{final_text}"
        ))
        extract_resp = invoke_with_rate_limit_retry(
            lambda: structured_llm.invoke(
                [extract_msg],
                config={"metadata": _ls_metadata},
            ),
            label=f"quant:{team_id}:extract",
        )
        parsed: QuantAnalystOutput | None = extract_resp.get("parsed")
        parsing_error = extract_resp.get("parsing_error")
        if parsing_error is not None:
            msg = (
                f"[quant:{team_id}] structured-output parse failed: "
                f"{type(parsing_error).__name__}: {parsing_error}"
            )
            if is_strict_validation_enabled():
                raise RuntimeError(msg)
            log.warning("%s — falling back to empty picks (lax mode)", msg)
            parsed = QuantAnalystOutput()
        assert parsed is not None
        # Convert QuantPick Pydantic models to dicts for downstream
        # consumers (peer_review, score_aggregator) that use dict-access.
        picks = [p.model_dump() for p in parsed.ranked_picks]

        log.info("[quant:%s] completed — %d picks, %d tool calls",
                 team_id, len(picks), len(tool_calls))

        # Diagnostic logging for the "no valid picks" case. 2-3 sector
        # teams have been returning zero picks per weekly run since at
        # least 2026-04-04 and we don't know whether it's (a) the LLM
        # producing no JSON, (b) _parse_picks_from_response failing to
        # extract it, (c) the ReAct agent hitting the recursion limit
        # before producing final text, or (d) all tool calls failing
        # and the LLM having no data to work with. Log enough context
        # to tell these apart on the next run.
        if not picks:
            last_tool = tool_calls[-1].get("tool") if tool_calls else "<none>"
            recursion_limit_hit = len(tool_calls) >= QUANT_MAX_ITERATIONS * 2 - 1
            text_tail = (final_text[-500:] if final_text else "<empty>").replace("\n", " ")
            log.warning(
                "[quant:%s] produced 0 picks — tool_calls=%d "
                "(recursion_limit_hit=%s) last_tool=%s "
                "final_text_tail=%r",
                team_id,
                len(tool_calls),
                recursion_limit_hit,
                last_tool,
                text_tail,
            )

        return {
            "team_id": team_id,
            "ranked_picks": picks,
            "tool_calls": tool_calls,
            "iterations": len(tool_calls),
            "error": None,
            "partial": False,
            # Ground-truth reasoning for retrospective decision review
            # (L4567). Captured into the decision artifact + carried in
            # SectorTeamOutput state; both bounded so this observability
            # payload never dominates. final_text is the analyst's prose
            # answer; transcript is the bounded ReAct message history.
            "final_text": final_text[:_FINAL_TEXT_CAP],
            "transcript": _serialize_transcript(messages),
        }

    except GraphRecursionError as e:
        # Budget exhausted before the agent reached a stop condition.
        # Treat as a degraded-but-non-fatal outcome: this team contributes
        # zero picks but doesn't crash the SF — score_aggregator will see
        # ``partial=True`` and accept the empty contribution with a WARN.
        # The +2 budget bump should already prevent the ``response_format``
        # extraction call from blowing the budget on its own; if we still
        # hit this, the agent legitimately needs more than 8 ReAct rounds
        # for this sector + run, which is a tunable observation, not a
        # crash-the-pipeline emergency.
        log.warning(
            "[quant:%s] recursion budget (%d transitions) exhausted before "
            "stop condition — accepting partial result (0 picks). "
            "score_aggregator will proceed with this team excluded.",
            team_id, _QUANT_RECURSION_LIMIT,
        )
        return {
            "team_id": team_id,
            "ranked_picks": [],
            "tool_calls": [],
            "iterations": _QUANT_RECURSION_LIMIT,
            "error": None,
            "partial": True,
            "partial_reason": "recursion_limit_exhausted",
        }

    except Exception as e:
        # Record the error so downstream (score_aggregator) can hard-fail
        # loudly instead of treating an exception as equivalent to the LLM
        # legitimately producing zero picks. Recursion budget exhaustion
        # is handled separately above as a partial outcome, not an error.
        log.error("[quant:%s] ReAct agent failed: %s", team_id, e)
        return {
            "team_id": team_id,
            "ranked_picks": [],
            "tool_calls": [],
            "iterations": 0,
            "error": f"{type(e).__name__}: {e}",
            "partial": False,
        }


def _should_retry_on_empty_picks(quant_output: dict) -> bool:
    """Detect the agent-gave-up failure mode that warrants a retry.

    Trigger conditions (all four must hold):

    1. ``ranked_picks`` is empty — the load-bearing symptom.
    2. ``error`` is None — exceptions go to a separate hard-fail path
       (retry won't recover from a missing API key or schema error).
    3. ``partial`` is False — recursion exhaustion already gave the
       agent its full budget; rerunning won't help.
    4. ``iterations > 0`` — the agent actually invoked tools. Zero
       iterations means the ReAct loop didn't run at all, and a retry
       with the same input would repeat the same failure.

    The combination is "the agent ran tools, finished cleanly, and
    chose to emit nothing." That's the give-up case from the 2026-05-04
    diagnosis: ``sector_quant:financials`` produced 22 tool calls and
    zero ranked picks. A retry with augmented prompting forces the
    agent to commit a ranking even at lower conviction.
    """
    return (
        len(quant_output.get("ranked_picks", []) or []) == 0
        and quant_output.get("error") is None
        and not quant_output.get("partial", False)
        and quant_output.get("iterations", 0) > 0
    )


def run_quant_analyst_with_retry(
    team_id: str,
    sector_tickers: list[str],
    market_regime: str,
    price_data: dict,
    technical_scores: dict,
    run_date: str,
    api_key: Optional[str] = None,
    focus_list: Optional[list[dict]] = None,
    override_tickers: Optional[list[str]] = None,
) -> dict:
    """Wrap ``run_quant_analyst`` with one-shot retry on empty picks.

    Empty ``ranked_picks`` despite ReAct iterations > 0 is the agent
    silently giving up — the 2026-05-04 diagnosis showed
    ``sector_quant:financials`` running 22 tool calls and emitting an
    empty result. Retry once with an augmented prompt that requires at
    least 3 picks. Recursion exhaustion and exceptions are NOT retried
    (more iterations won't help; rerunning won't fix a missing API key).

    Returns the same shape as ``run_quant_analyst`` plus three retry-
    observability fields:

      ``retry_attempted``: bool — true iff the second invocation fired.
      ``retry_succeeded``: bool — true iff the retry produced ≥1 pick.
      ``retry_first_iterations``: int | None — iteration count of the
                                  first attempt; populated only when
                                  retry fires.

    Cost: bounded — retry only fires on the give-up path (~4-5 sectors
    × Saturday × ~$0.10 retry = ~$0.50/week worst case). Latency: also
    bounded (sector teams run in parallel via Send(), so only the
    slowest retry blocks the SF).
    """
    first = run_quant_analyst(
        team_id=team_id,
        sector_tickers=sector_tickers,
        market_regime=market_regime,
        price_data=price_data,
        technical_scores=technical_scores,
        run_date=run_date,
        api_key=api_key,
        focus_list=focus_list,
        override_tickers=override_tickers,
    )

    if not _should_retry_on_empty_picks(first):
        first["retry_attempted"] = False
        first["retry_succeeded"] = False
        first["retry_first_iterations"] = None
        _emit_retry_telemetry_safely(
            team_id=team_id, attempted=False, succeeded=False,
        )
        return first

    # Retry path. The WARNING log line is flow-doctor-detectable so
    # the event surfaces in CW alarms without requiring an explicit
    # emit call (mirrors the existing 0-picks log pattern).
    log.warning(
        "[quant:%s] retry-on-empty firing — first attempt produced 0 picks "
        "after %d tool calls (error=None, partial=False). Retrying once "
        "with augmented prompt (must produce ≥3 picks).",
        team_id, first["iterations"],
    )

    retry_preamble = _QUANT_RETRY_PREAMBLE.format(
        n_tool_calls=first["iterations"],
    )
    second = run_quant_analyst(
        team_id=team_id,
        sector_tickers=sector_tickers,
        market_regime=market_regime,
        price_data=price_data,
        technical_scores=technical_scores,
        run_date=run_date,
        api_key=api_key,
        _retry_preamble=retry_preamble,
        focus_list=focus_list,
        override_tickers=override_tickers,
    )

    second_picks = len(second.get("ranked_picks", []) or [])
    second["retry_attempted"] = True
    second["retry_succeeded"] = second_picks > 0
    second["retry_first_iterations"] = first["iterations"]
    _emit_retry_telemetry_safely(
        team_id=team_id, attempted=True, succeeded=second["retry_succeeded"],
    )

    if second_picks == 0:
        # The retry also gave up — escalate via WARNING so flow-doctor
        # surfaces it. Downstream behavior is unchanged (the team
        # contributes 0 picks); the observability lets us track how
        # often retries fail to recover, which informs whether
        # workstream #3 (output_completeness rubric dimension) needs
        # to fire harder, or whether the system prompt itself needs
        # tightening.
        log.warning(
            "[quant:%s] retry produced 0 picks — agent failure persists "
            "across both attempts (first iterations=%d, second iterations=%d). "
            "Downstream qual loop will be bypassed for this sector. "
            "Investigate whether the sector universe has degenerate "
            "technical-score input, the prompt is too restrictive, or "
            "the LLM is regressing.",
            team_id, first["iterations"], second.get("iterations", 0),
        )
    else:
        log.info(
            "[quant:%s] retry SUCCEEDED — produced %d picks (first attempt: 0, "
            "second iterations=%d). Augmented prompt forced commitment.",
            team_id, second_picks, second.get("iterations", 0),
        )

    return second


def _build_system_prompt(
    team_id: str,
    team_params: dict,
    market_regime: str,
    universe_size: int,
) -> str:
    focus_metrics = team_params.get("focus_metrics", [])
    focus_str = ", ".join(focus_metrics) if focus_metrics else "standard technical and fundamental metrics"

    return load_prompt("quant_analyst_system").format(
        team_title=team_id.title(),
        universe_size=universe_size,
        quant_top_n=QUANT_TOP_N,
        focus_str=focus_str,
        market_regime=market_regime,
    )


from agents.langchain_utils import extract_tool_calls as _extract_tool_calls
from agents.langchain_utils import get_final_text as _get_final_text
from agents.langchain_utils import serialize_transcript as _serialize_transcript
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    invoke_with_rate_limit_retry,
)

# Cap the analyst's prose answer persisted into the decision artifact for
# retrospective "why" review (L4567). The transcript is bounded inside
# ``serialize_transcript``; this bounds the separate ``final_text`` field.
_FINAL_TEXT_CAP = 6_000
