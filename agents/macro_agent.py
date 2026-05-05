"""
Macro & Market Environment Agent (§4.3).

Single global instance — not per-stock.
Uses claude-sonnet-4-6 (strategic model) for nuanced economic interpretation.

Outputs per-sector macro modifiers (11 sectors), market_regime string,
and sector_ratings (overweight / market_weight / underweight + rationale).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from agents.prompt_loader import load_prompt
from config import STRATEGIC_MODEL, MAX_TOKENS_STRATEGIC, ANTHROPIC_API_KEY, ALL_SECTORS, REGIME_GUARDRAILS, PRIOR_REPORT_MAX_CHARS
from agents.token_guard import check_prompt_size
from graph.state_schemas import MacroCriticOutput, MacroEconomistRawOutput
from strict_mode import is_strict_validation_enabled

_PROMPT_TEMPLATE = load_prompt("macro_agent")

def _truncate_prior(text: str, max_chars: int | None = None) -> str:
    """Truncate prior report to last max_chars characters to manage prompt size."""
    if max_chars is None:
        max_chars = PRIOR_REPORT_MAX_CHARS
    if not text or len(text) <= max_chars:
        return text
    return "[...truncated...]\n" + text[-max_chars:]


_DEFAULT_SECTOR_MODIFIERS = {s: 1.0 for s in ALL_SECTORS}

_VALID_RATINGS = {"overweight", "market_weight", "underweight"}
_OW_THRESHOLD = 1.08   # modifier >= this → overweight
_UW_THRESHOLD = 0.92   # modifier <= this → underweight


def _find_json_block(text: str, key: str = '"market_regime"') -> tuple[int, int]:
    """
    Find the start and end indices of the JSON object containing `key`.
    Uses balanced-brace scanning — handles nested dicts correctly.
    Returns (start, end) inclusive, or (-1, -1) if not found.
    """
    key_pos = text.find(key)
    if key_pos == -1:
        return -1, -1
    brace_pos = text.rfind('{', 0, key_pos)
    if brace_pos == -1:
        return -1, -1
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[brace_pos:], brace_pos):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return brace_pos, i
    return -1, -1


def _derive_sector_ratings(sector_modifiers: dict[str, float]) -> dict[str, dict]:
    """Fallback: derive OW/MW/UW ratings from modifier values when agent JSON is missing."""
    ratings = {}
    for sector, mod in sector_modifiers.items():
        if mod >= _OW_THRESHOLD:
            rating = "overweight"
            rationale = f"Macro tailwind (modifier {mod:.2f})"
        elif mod <= _UW_THRESHOLD:
            rating = "underweight"
            rationale = f"Macro headwind (modifier {mod:.2f})"
        else:
            rating = "market_weight"
            rationale = f"Neutral macro backdrop (modifier {mod:.2f})"
        ratings[sector] = {"rating": rating, "rationale": rationale}
    return ratings


def _fmt(val, fmt=".1f", default="N/A") -> str:
    if val is None:
        return default
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return default


def run_macro_agent(
    prior_report: Optional[str],
    prior_date: str,
    macro_data: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run the Macro & Market Environment Agent.

    Returns dict with:
      report_md: str
      macro_json: dict (with market_regime and sector_modifiers)
      market_regime: str
      sector_modifiers: dict[str, float]
    """
    # Defer-import the cost-telemetry callback so this module's import
    # path stays leaf-side of graph/* (research_graph imports macro_agent).
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    llm = ChatAnthropic(
        model=STRATEGIC_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_STRATEGIC,
        callbacks=[get_cost_telemetry_callback()],
    )

    prior_text = _truncate_prior(prior_report) if prior_report else "NONE — initial report"

    # Breadth data — observability matters here. A missing or null breadth
    # means the upstream alpha-engine-data collector regressed and the
    # macro report will use N/A placeholders. Log loudly so silent
    # degradation shows up in CloudWatch.
    breadth = macro_data.get("breadth")
    if not breadth:
        logger.warning(
            "macro_agent[strategic]: breadth missing or null (key_present=%s, value_type=%s) — "
            "emitting N/A placeholders to LLM. Check alpha-engine-data macro collector.",
            "breadth" in macro_data,
            type(macro_data.get("breadth")).__name__,
        )
        breadth = {}

    prompt = _PROMPT_TEMPLATE.format(
        sector_list_text="\n".join(f"- {s}" for s in ALL_SECTORS),
        prior_date=prior_date,
        prior_report=prior_text,
        fed_funds=_fmt(macro_data.get("fed_funds_rate")),
        t2yr=_fmt(macro_data.get("treasury_2yr")),
        t10yr=_fmt(macro_data.get("treasury_10yr")),
        curve_slope=_fmt(macro_data.get("yield_curve_slope"), ".0f"),
        vix=_fmt(macro_data.get("vix")),
        spy_30d=_fmt(macro_data.get("sp500_30d_return")),
        qqq_30d=_fmt(macro_data.get("qqq_30d_return")),
        iwm_30d=_fmt(macro_data.get("iwm_30d_return")),
        oil=_fmt(macro_data.get("oil_wti"), ".2f"),
        gold=_fmt(macro_data.get("gold"), ".0f"),
        copper=_fmt(macro_data.get("copper"), ".2f"),
        cpi_yoy=_fmt(macro_data.get("cpi_yoy")),
        unemployment=_fmt(macro_data.get("unemployment")),
        consumer_sentiment=_fmt(macro_data.get("consumer_sentiment")),
        initial_claims=_fmt(macro_data.get("initial_claims"), ".0f"),
        hy_oas=_fmt(macro_data.get("hy_credit_spread_oas"), ".0f"),
        pct_above_50d=_fmt(breadth.get("pct_above_50d_ma")),
        pct_above_200d=_fmt(breadth.get("pct_above_200d_ma")),
        adv_dec_ratio=_fmt(breadth.get("advance_decline_ratio")),
        upcoming_releases="See FRED calendar for next CPI/FOMC dates.",
    )

    prompt = check_prompt_size(prompt, MAX_TOKENS_STRATEGIC, caller="macro_agent")

    # with_structured_output(include_raw=True): Anthropic tool-use populates the
    # typed Pydantic model; the raw text is kept alongside so the markdown
    # narrative (report_md) can be recovered if the LLM didn't fill the schema
    # field directly. include_raw=True captures parse failures as a
    # ``parsing_error`` rather than raising; the strict-mode gate below raises
    # explicitly to match the "no silent fallbacks" rule.
    structured_llm = llm.with_structured_output(
        MacroEconomistRawOutput, include_raw=True
    )
    response = structured_llm.invoke(
        [HumanMessage(content=prompt)],
        config={"metadata": _PROMPT_TEMPLATE.langsmith_metadata()},
    )

    raw_message = response.get("raw")
    parsed: MacroEconomistRawOutput | None = response.get("parsed")
    parsing_error = response.get("parsing_error")
    full_text = (
        raw_message.content
        if raw_message is not None and hasattr(raw_message, "content")
        else ""
    )

    if parsing_error is not None:
        msg = (
            f"[macro_agent] structured-output parse failed: "
            f"{type(parsing_error).__name__}: {parsing_error}"
        )
        if is_strict_validation_enabled():
            raise RuntimeError(msg)
        logger.warning("%s — falling back to defaults (lax mode)", msg)
        parsed = MacroEconomistRawOutput(
            market_regime="neutral",
            sector_modifiers=_DEFAULT_SECTOR_MODIFIERS.copy(),
        )

    # parsed is guaranteed non-None at this point (either the LLM populated it
    # or the lax-mode fallback above synthesized defaults).
    assert parsed is not None

    # report_md preference order: parsed.report_md (LLM filled the field) →
    # raw-slice fallback (LLM emitted prose-then-JSON like the prior contract).
    # The fallback retires once prompts are migrated to put markdown into the
    # schema field; until then, the include_raw path keeps narrative quality.
    report_md = (parsed.report_md or "").strip()
    if not report_md and full_text:
        _start, _end = _find_json_block(full_text)
        if _start != -1:
            report_md = (full_text[:_start] + full_text[_end + 1:]).strip()
        else:
            report_md = full_text.strip()

    # Sector ratings post-validation: per-sector validity check + derived
    # fallback when the LLM emits an invalid rating string. The schema only
    # types sector_ratings as ``dict[str, dict]``, so domain validation lives
    # in the agent.
    raw_ratings = dict(parsed.sector_ratings or {})
    validated_ratings = {}
    for sector in ALL_SECTORS:
        entry = raw_ratings.get(sector, {})
        rating = entry.get("rating", "") if isinstance(entry, dict) else ""
        rationale = entry.get("rationale", "") if isinstance(entry, dict) else ""
        if rating not in _VALID_RATINGS:
            mod = parsed.sector_modifiers.get(sector, 1.0)
            fallback = _derive_sector_ratings({sector: mod})[sector]
            validated_ratings[sector] = fallback
        else:
            validated_ratings[sector] = {"rating": rating, "rationale": rationale}

    # Build macro_json in the dict shape downstream consumers expect
    # (graph/research_graph.py merge_results, archive_writer, consolidator).
    macro_json = {
        "market_regime": parsed.market_regime,
        "sector_modifiers": dict(parsed.sector_modifiers),
        "sector_ratings": validated_ratings,
        "key_theme": parsed.key_theme,
        "material_changes": parsed.material_changes,
    }

    # Post-LLM regime validation — quantitative cross-check that may override
    # the LLM's regime call when guardrail invariants fire.
    validated_regime = _validate_regime(parsed.market_regime, macro_data)
    macro_json["market_regime"] = validated_regime

    return {
        "report_md": report_md,
        "macro_json": macro_json,
        "market_regime": validated_regime,
        "sector_modifiers": macro_json.get("sector_modifiers", _DEFAULT_SECTOR_MODIFIERS.copy()),
        "sector_ratings": macro_json.get("sector_ratings", _derive_sector_ratings(_DEFAULT_SECTOR_MODIFIERS)),
        "material_changes": bool(macro_json.get("material_changes", False)),
    }


# ── Regime severity ordering ────────────────────────────────────────────────
_REGIME_SEVERITY = {"bull": 0, "neutral": 1, "caution": 2, "bear": 3}


def _validate_regime(llm_regime: str, macro_data: dict) -> str:
    """
    Post-LLM quantitative guardrails for market_regime.

    Hard rules override LLM when quantitative thresholds are breached:
      - VIX > 30 AND SPY 30d < -10% → force 'bear'
      - VIX > 25 AND SPY 30d < -5% → force at least 'caution'
      - HY OAS > 500bps → force at least 'caution'

    Only escalates severity — never downgrades (e.g., won't override 'bear' to 'caution').
    """

    cfg = REGIME_GUARDRAILS
    if not cfg:
        return llm_regime

    vix = macro_data.get("vix")
    spy_30d = macro_data.get("sp500_30d_return")
    hy_oas = macro_data.get("hy_credit_spread_oas")

    current_severity = _REGIME_SEVERITY.get(llm_regime, 1)
    forced_regime = llm_regime

    # Hard override: extreme stress → bear
    bear_vix = cfg.get("bear_vix_threshold", 30)
    bear_spy = cfg.get("bear_spy_30d_threshold", -10.0)
    if vix is not None and spy_30d is not None:
        if vix > bear_vix and spy_30d < bear_spy:
            if _REGIME_SEVERITY.get("bear", 3) > current_severity:
                forced_regime = "bear"
                logger.warning(
                    "[regime_guardrail] OVERRIDE %s → bear: VIX=%.1f>%d AND SPY_30d=%.1f%%<%s%%",
                    llm_regime, vix, bear_vix, spy_30d, bear_spy,
                )
                return forced_regime

    # Soft override: elevated stress → at least caution
    caution_vix = cfg.get("caution_vix_threshold", 25)
    caution_spy = cfg.get("caution_spy_30d_threshold", -5.0)
    if vix is not None and spy_30d is not None:
        if vix > caution_vix and spy_30d < caution_spy:
            if _REGIME_SEVERITY.get("caution", 2) > current_severity:
                forced_regime = "caution"
                logger.warning(
                    "[regime_guardrail] OVERRIDE %s → caution: VIX=%.1f>%d AND SPY_30d=%.1f%%<%s%%",
                    llm_regime, vix, caution_vix, spy_30d, caution_spy,
                )

    # Credit stress override: HY OAS > threshold → at least caution
    hy_threshold = cfg.get("caution_hy_oas_threshold", 500)
    if hy_oas is not None and hy_oas > hy_threshold:
        if _REGIME_SEVERITY.get("caution", 2) > _REGIME_SEVERITY.get(forced_regime, 1):
            logger.warning(
                "[regime_guardrail] OVERRIDE %s → caution: HY_OAS=%.0fbps>%dbps",
                forced_regime, hy_oas, hy_threshold,
            )
            forced_regime = "caution"

    return forced_regime


# ── Phase 3: Macro Reflection Loop ───────────────────────────────────────────

_CRITIC_PROMPT = load_prompt("macro_agent_critic")


def run_macro_critic(
    initial_result: dict,
    macro_data: dict,
    api_key: str | None = None,
) -> dict:
    """
    Critique the macro agent's regime classification.

    Returns: {"action": "accept"|"revise", "critique": str, "suggested_regime": str|None}
    """
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    llm = ChatAnthropic(
        model=STRATEGIC_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=512,
        callbacks=[get_cost_telemetry_callback()],
    )
    macro_json = initial_result.get("macro_json", {})
    breadth = macro_data.get("breadth")
    if not breadth:
        logger.warning(
            "macro_agent[critic]: breadth missing or null (key_present=%s, value_type=%s) — "
            "emitting N/A placeholders to LLM. Check alpha-engine-data macro collector.",
            "breadth" in macro_data,
            type(macro_data.get("breadth")).__name__,
        )
        breadth = {}

    mods = macro_json.get("sector_modifiers", {})
    mod_lines = [f"  {s}: {v:.2f}" for s, v in sorted(mods.items())]

    prompt = _CRITIC_PROMPT.format(
        regime=macro_json.get("market_regime", "neutral"),
        key_theme=macro_json.get("key_theme", "N/A"),
        vix=_fmt(macro_data.get("vix")),
        spy_30d=_fmt(macro_data.get("sp500_30d_return")),
        curve_slope=_fmt(macro_data.get("yield_curve_slope"), ".0f"),
        consumer_sentiment=_fmt(macro_data.get("consumer_sentiment")),
        initial_claims=_fmt(macro_data.get("initial_claims"), ".0f"),
        hy_oas=_fmt(macro_data.get("hy_credit_spread_oas"), ".0f"),
        pct_above_50d=_fmt(breadth.get("pct_above_50d_ma")),
        pct_above_200d=_fmt(breadth.get("pct_above_200d_ma")),
        sector_modifiers_text="\n".join(mod_lines),
    )

    # Critic uses with_structured_output for typed extraction. Strict mode
    # raises on parse failure; lax mode keeps the silent-accept fallback
    # because the critic is an editorial gate — accepting the initial macro
    # classification is the conservative behavior on critic failure.
    structured_llm = llm.with_structured_output(MacroCriticOutput)
    try:
        verdict: MacroCriticOutput = structured_llm.invoke(
            [HumanMessage(content=prompt)],
            config={"metadata": _CRITIC_PROMPT.langsmith_metadata()},
        )
        result = {
            "action": verdict.action,
            "critique": verdict.critique,
        }
        if verdict.suggested_regime is not None:
            result["suggested_regime"] = verdict.suggested_regime
        logger.info(
            "[macro_critic] action=%s critique=%s",
            verdict.action, (verdict.critique or "")[:80],
        )
        return result
    except Exception as e:
        if is_strict_validation_enabled():
            raise
        logger.warning(
            "[macro_critic] LLM call failed: %s — accepting initial result", e
        )

    return {
        "action": "accept",
        "critique": "Critic unavailable — accepting initial classification.",
    }


def run_macro_agent_with_reflection(
    prior_report: Optional[str],
    prior_date: str,
    macro_data: dict,
    max_iterations: int = 2,
    api_key: Optional[str] = None,
    prior_snapshots: list[dict] | None = None,
) -> dict:
    """
    Run macro agent with self-critique reflection loop.

    1. Initial macro agent call
    2. Critic evaluates the result
    3. If critic says "revise" and iterations remain, re-run with critique as context
    4. Quantitative guardrails apply as final gate after reflection

    Returns the standard macro agent result dict plus a reflection_log field.
    """

    # Append structured prior snapshot context to prior_report
    enriched_prior = prior_report or ""
    if prior_snapshots:
        snapshot_lines = ["\n\nPRIOR MACRO SNAPSHOTS (recent history):"]
        for snap in prior_snapshots[:3]:
            snapshot_lines.append(
                f"  {snap.get('date', '?')}: regime={snap.get('market_regime', '?')}, "
                f"VIX={snap.get('vix', '?')}, 10Y={snap.get('treasury_10yr', '?')}, "
                f"curve={snap.get('yield_curve_slope', '?')}, "
                f"SP500_30d={snap.get('sp500_30d_return', '?')}"
            )
        enriched_prior += "\n".join(snapshot_lines)

    result = run_macro_agent(
        prior_report=enriched_prior,
        prior_date=prior_date,
        macro_data=macro_data,
        api_key=api_key,
    )
    initial_regime = result["market_regime"]

    reflection_log = {
        "initial_regime": initial_regime,
        "iterations": 1,
        "critic_action": "accept",
        "critique_text": "",
        "final_regime": initial_regime,
    }

    for iteration in range(1, max_iterations):
        critic_result = run_macro_critic(result, macro_data, api_key=api_key)
        reflection_log["critic_action"] = critic_result.get("action", "accept")
        reflection_log["critique_text"] = critic_result.get("critique", "")

        if critic_result.get("action") != "revise":
            logger.info("[macro_reflection] iteration %d: critic accepted regime=%s",
                         iteration, result["market_regime"])
            break

        logger.info("[macro_reflection] iteration %d: critic requests revision — %s",
                     iteration, critic_result.get("critique", "")[:80])

        critique_context = (
            f"\n\nCRITIC FEEDBACK (from prior iteration):\n{critic_result['critique']}\n"
            f"Suggested regime: {critic_result.get('suggested_regime', 'N/A')}\n"
            "Please reconsider your regime classification in light of this feedback."
        )
        augmented_prior = (prior_report or "") + critique_context

        result = run_macro_agent(
            prior_report=augmented_prior,
            prior_date=prior_date,
            macro_data=macro_data,
            api_key=api_key,
        )
        reflection_log["iterations"] = iteration + 1

    reflection_log["final_regime"] = result["market_regime"]
    result["reflection_log"] = reflection_log

    if reflection_log["iterations"] > 1:
        logger.info("[macro_reflection] %s → %s after %d iterations",
                     initial_regime, result["market_regime"], reflection_log["iterations"])

    return result
