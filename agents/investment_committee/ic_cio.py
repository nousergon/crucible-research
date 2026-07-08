"""
CIO Agent — evaluates all sector team recommendations in a single batch Sonnet call.

The CIO sees all candidates simultaneously and selects up to `max_new_entrants` for advancement (floor-enforced; `open_slots` is informational only — see `_compute_advance_bounds`).
Evaluates on 5 dimensions: risk/reward asymmetry (primary), team conviction, macro alignment, portfolio fit, catalyst specificity.
Writes entry theses for advanced stocks. All decisions (advance, reject, deadlock) saved.
"""

from __future__ import annotations

import logging
from typing import Optional

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict

from config import (
    STRATEGIC_MODEL,
    PER_STOCK_MODEL,
    MAX_TOKENS_STRATEGIC,
    ANTHROPIC_API_KEY,
)
from agents.prompt_loader import load_prompt
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS,
    invoke_with_rate_limit_retry,
)
from strict_mode import is_strict_validation_enabled

log = logging.getLogger(__name__)

# Prompt variants — the de-blended path (L4564) selects a distinct v1.5.0
# template so the flag flip is a clean A/B + instant rollback (flip OFF →
# the legacy prompt) with no shared-template KeyError risk.
_PROMPT_DEFAULT = "ic_cio_evaluation"
_PROMPT_DEBLENDED = "ic_cio_evaluation_deblended"

# IC critic (config#927) — a cheap Haiku reviewer that challenges the CIO's
# advance set before finalization, mirroring the macro-agent reflection loop
# (agents/macro_agent.run_macro_critic). Default OFF: the reflection wrapper is
# only invoked when CIO_CRITIC_ENABLED is set, so this is an inert merge.
_PROMPT_CRITIC = "ic_cio_critic"

# IC-critic max output budget. Small structured call (action + critique +
# flagged/drops/adds ticker lists); both MAX_TOKENS_* tiers are oversized for
# this narrow case. A named constant (not an inline literal) so the
# no-hardcoded-max_tokens lint stays satisfied while keeping the tight ceiling
# that makes a runaway critic output visible.
_CRITIC_MAX_TOKENS = 768


class CIOCriticOutput(BaseModel):
    """Reflection-loop critic output for the CIO selection.

    The critic accepts or asks the CIO to revise its advance set before
    finalization. ``revise`` triggers one more CIO call with the critique as
    context; ``accept`` ends the loop. ``flagged_tickers`` / ``suggested_drops``
    / ``suggested_adds`` are advisory — the re-run CIO is free to keep its slate.
    """

    model_config = ConfigDict(extra="allow")

    action: Literal["accept", "revise"]
    critique: str = ""
    flagged_tickers: list[str] = []
    suggested_drops: list[str] = []
    suggested_adds: list[str] = []


def _compute_advance_bounds(
    n_candidates: int,
    max_new_entrants: int,
    min_new_entrants: int,
) -> tuple[int, int]:
    """
    Compute (floor, cap) for new-entrant advances this week.

    Both bounds are clamped to n_candidates so we never demand advancing
    or permit advancing more candidates than exist:

        cap   = min(max_new_entrants, n_candidates)
        floor = min(min_new_entrants, n_candidates)

    Note: open_slots (population gap) is intentionally NOT a factor.
    The cap range [min, max] is decoupled from how empty the portfolio is —
    exits handle population-size pressure separately. Logged for ops in
    `run_cio` but not used in this calculation.

    Returns (0, 0) if no candidates exist or max_new_entrants <= 0.
    """
    if n_candidates <= 0 or max_new_entrants <= 0:
        return (0, 0)
    cap = min(max_new_entrants, n_candidates)
    floor = min(min_new_entrants, n_candidates)
    # Defensive: floor must never exceed cap (config bounds-check guards
    # this at startup, but belt-and-suspenders).
    floor = min(floor, cap)
    return (floor, cap)


def run_cio(
    candidates: list[dict],
    macro_context: dict,
    sector_ratings: dict,
    current_population: list[dict],
    open_slots: int,
    exits: list[dict],
    run_date: str,
    api_key: Optional[str] = None,
    prior_decisions: list[dict] | None = None,
    *,
    max_new_entrants: int = 10,
    min_new_entrants: int = 2,
    force_fill_conviction_floor: float = 60.0,
    prior_cycle_scorecard: Optional[str] = None,
    deblended: bool = False,
    sector_neutral_quality: dict[str, float] | None = None,
) -> dict:
    """
    Run the CIO evaluation in a single batch Sonnet call.

    Args:
        candidates: All team recommendations. Each has:
            ticker, team_id, quant_score, qual_score, bull_case, bear_case,
            catalysts, conviction, quant_rationale.
        macro_context: {market_regime, macro_report_summary, ...}
        sector_ratings: {sector: {rating, modifier, rationale}}
        current_population: Current held stocks (for portfolio fit analysis).
        open_slots: Number of population slots available (population gap).
        exits: Stocks being removed this week (for context).
        run_date: YYYY-MM-DD.
        max_new_entrants: Hard ceiling on new entrants per week.
        min_new_entrants: Floor on new entrants per week, when candidates are
            available. Forces CIO to advance at least this many even if the
            population gap is smaller — exits will rotate names out.

    Returns:
        {
            "decisions": list[dict],  # one per candidate with decision + rationale
            "advanced_tickers": list[str],
            "entry_theses": dict[str, dict],  # CIO-authored theses for advanced stocks
        }
    """
    if not candidates:
        log.info("[cio] no candidates to evaluate")
        return {"decisions": [], "advanced_tickers": [], "entry_theses": {}}

    # Held set — the floor/cap govern NET-NEW entrants (candidates not already
    # held), and the floor force-fill draws only from net-new names ≥ the
    # entrant bar. Re-affirmations of incumbents are unbounded.
    held_tickers = frozenset(
        p.get("ticker") for p in (current_population or []) if p.get("ticker")
    )

    floor, cap = _compute_advance_bounds(
        len(candidates), max_new_entrants, min_new_entrants,
    )
    log.info(
        "[cio] bounds: floor=%d cap=%d (n_candidates=%d, max_new=%d, "
        "min_new=%d, open_slots=%d [informational only])",
        floor, cap, len(candidates),
        max_new_entrants, min_new_entrants, open_slots,
    )

    if cap <= 0:
        log.info("[cio] cap=0 — rejecting all %d candidates", len(candidates))
        return {
            "decisions": [
                _reject_decision(c, "cap=0: no candidates or max_new_entrants<=0")
                for c in candidates
            ],
            "advanced_tickers": [],
            "entry_theses": {},
        }

    from graph.llm_cost_tracker import get_cost_telemetry_callback

    llm = ChatAnthropic(
        model=STRATEGIC_MODEL,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=MAX_TOKENS_STRATEGIC,
        max_retries=SECTOR_TEAM_LLM_MAX_RETRIES,
        default_request_timeout=SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS,
        callbacks=[get_cost_telemetry_callback()],
    )

    prompt_name = _PROMPT_DEBLENDED if deblended else _PROMPT_DEFAULT
    prompt = _build_cio_prompt(
        candidates, macro_context, sector_ratings,
        current_population, cap, exits, run_date,
        prior_decisions=prior_decisions,
        prior_cycle_scorecard=prior_cycle_scorecard,
        deblended=deblended,
        sector_neutral_quality=sector_neutral_quality,
    )

    # PR 2.3 Step E: flip CIO to with_structured_output. The LLM emits a
    # CIORawOutput (decisions list with ADVANCE | REJECT | NO_ADVANCE_DEADLOCK
    # literals — distinct from the post-processed CIODecision shape that adds
    # ADVANCE_FORCED / HOLD via _post_process_cio_decisions below). Strict
    # mode raises on parse error; lax-mode preserves the load-bearing fallback
    # to combined-score floor selection — every Saturday SF MUST yield some
    # advanced tickers for the executor to act on.
    from graph.state_schemas import CIORawOutput
    from strict_mode import is_strict_validation_enabled

    structured_llm = llm.with_structured_output(CIORawOutput)
    try:
        # ALL-AGENTS-STRICT (Brian, 2026-05-16): the CIO is one of the
        # agents in scope — "We don't get anything from this process if
        # ... any other agent for that matter, fail/don't run." Wrap
        # the single batch Sonnet call in the deadline-bounded (~75 min)
        # 429 retry so an org TPM ceiling is ridden out rather than
        # immediately failing the run; if the 429 STILL persists past
        # the deadline the wrapper re-raises and (strict mode default)
        # the run hard-fails — no synthetic/empty CIO substitute is
        # promoted. Non-429 errors propagate immediately as before.
        raw_output: CIORawOutput = invoke_with_rate_limit_retry(
            lambda: structured_llm.invoke(
                [HumanMessage(content=prompt)],
                config={
                    "metadata": load_prompt(
                        prompt_name
                    ).langsmith_metadata()
                },
            ),
            label="cio",
        )
        decisions_dicts = [d.model_dump() for d in raw_output.decisions]
        if not decisions_dicts:
            log.warning("[cio] structured response had empty decisions list")
            if is_strict_validation_enabled():
                raise RuntimeError(
                    "CIO structured response had empty decisions list"
                )
            return _fallback_selection(candidates, floor)
        # Per-candidate invariant: every input candidate must appear
        # exactly once in the decisions list (ADVANCE / REJECT /
        # NO_ADVANCE_DEADLOCK). Reconcile the LLM's decisions against
        # the candidate ticker SET rather than asserting a raw count.
        #
        # Why set reconciliation, not `len(decisions) == len(candidates)`:
        # the count check (added 2026-05-02 for the partial-list edge —
        # PR B stripped the inline JSON example and Sonnet emitted a
        # SHORT list) is brittle in both directions. 2026-05-17 Saturday
        # SF: Sonnet's structured-output batch returned 19 decisions for
        # 18 candidates (one stray extra/duplicate decision object) — a
        # benign LLM artifact the raw count check turned into a hard
        # strict-mode failure of the entire weekly run. The count check
        # is also too WEAK: 18 decisions for 18 candidates with one
        # ticker duplicated (so one real candidate silently missing)
        # passed it. Reconciling against the ticker set is strictly
        # stronger — it self-heals extraneous/duplicate noise and still
        # hard-fails the genuine partial-list regression the original
        # assertion protected against.
        decisions_dicts, recon = _reconcile_cio_decisions(
            decisions_dicts, candidates,
        )
        if recon["extraneous"] or recon["duplicate"]:
            log.warning(
                "[cio] reconciled CIO decisions vs candidate set: dropped "
                "%d extraneous %s, collapsed %d duplicated ticker(s) %s "
                "(conservative non-ADVANCE-wins); %d candidate decisions "
                "retained",
                len(recon["extraneous"]), recon["extraneous"],
                len(recon["duplicate"]), recon["duplicate"],
                len(decisions_dicts),
            )
        if recon["missing"]:
            msg = (
                f"CIO returned {recon['raw_count']} decisions for "
                f"{len(candidates)} candidates — {len(recon['missing'])} "
                f"candidate(s) missing a decision after reconciliation "
                f"(dropped {len(recon['extraneous'])} extraneous, "
                f"{len(recon['duplicate'])} duplicate): "
                f"missing={recon['missing']}. Every candidate must appear "
                f"exactly once in the decisions list."
            )
            log.warning("[cio] %s", msg)
            if is_strict_validation_enabled():
                raise RuntimeError(msg)
            # Lax mode: fall through to post-process, which tolerates the
            # still-missing tickers by treating them as REJECT.
        cio_result = _post_process_cio_decisions(
            decisions_dicts, candidates, floor, cap,
            held_tickers=held_tickers,
            force_fill_conviction_floor=force_fill_conviction_floor,
        )
        # config#1753: the actually-rendered CIO prompt (post
        # ``_build_cio_prompt(...)``) — what was handed to
        # ``HumanMessage(content=prompt)`` above, not the raw
        # ``LoadedPrompt`` template. Threaded back so the
        # ``research_graph.py`` call site's ``track_llm_cost`` scope can
        # stamp it onto ``FullPromptContext.user_prompt`` instead of
        # falling back to the unsubstituted template text.
        cio_result["rendered_prompt"] = prompt
        return cio_result
    except Exception as e:
        log.error("[cio] evaluation failed: %s", e)
        if is_strict_validation_enabled():
            raise
        # Lax fallback advances only `floor` (not `cap`). When the LLM signal is
        # unusable, be conservative — don't force max-advance on broken data.
        # No successful LLM call landed on this path (evaluation failed
        # before/at the API call), so there's no rendered prompt to thread —
        # the frame's user_prompt falls back to the raw LoadedPrompt.text.
        return _fallback_selection(candidates, floor)


def _summarize_advanced(cio_result: dict, candidates: list[dict]) -> str:
    """One line per advanced ticker for the critic prompt (ticker · conviction · thesis)."""
    by_ticker = {c.get("ticker"): c for c in candidates if c.get("ticker")}
    theses = cio_result.get("entry_theses", {})
    lines: list[str] = []
    for t in cio_result.get("advanced_tickers", []):
        cand = by_ticker.get(t, {})
        thesis = theses.get(t, {})
        conv = cand.get("conviction")
        summary = (
            thesis.get("thesis_summary")
            or thesis.get("rationale")
            or cand.get("bull_case")
            or ""
        )
        lines.append(
            f"  {t}: conviction={conv if conv is not None else 'n/a'} | "
            f"{str(summary)[:160]}"
        )
    return "\n".join(lines) if lines else "  (no tickers advanced)"


def run_cio_critic(
    cio_result: dict,
    candidates: list[dict],
    macro_context: dict,
    sector_ratings: dict,
    api_key: Optional[str] = None,
) -> dict:
    """Critique the CIO's advance set with a cheap Haiku reviewer (config#927).

    Mirrors ``agents.macro_agent.run_macro_critic``: structured Haiku call,
    deadline-bounded 429 retry, strict/lax editorial-accept fallback (accepting
    the CIO's slate is the conservative behavior when the critic is unavailable).

    Returns: ``{"action": "accept"|"revise", "critique": str,
    "flagged_tickers": list, "suggested_drops": list, "suggested_adds": list}``.
    """
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    llm = ChatAnthropic(
        model=PER_STOCK_MODEL,  # Claude Haiku 4.5 — the requested cheap critic
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=_CRITIC_MAX_TOKENS,
        max_retries=SECTOR_TEAM_LLM_MAX_RETRIES,
        default_request_timeout=SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS,
        callbacks=[get_cost_telemetry_callback()],
    )

    prompt_tmpl = load_prompt(_PROMPT_CRITIC)
    prompt = prompt_tmpl.format(
        market_regime=macro_context.get("market_regime", "neutral"),
        macro_report=str(macro_context.get("macro_report", ""))[:1200],
        n_candidates=len(candidates),
        n_advanced=len(cio_result.get("advanced_tickers", [])),
        advanced_summary=_summarize_advanced(cio_result, candidates),
        sector_ratings_text="\n".join(
            f"  {s}: {r.get('rating', 'n/a')}"
            for s, r in sorted((sector_ratings or {}).items())
        )
        or "  (none)",
    )

    structured_llm = llm.with_structured_output(CIOCriticOutput)
    try:
        verdict: CIOCriticOutput = invoke_with_rate_limit_retry(
            lambda: structured_llm.invoke(
                [HumanMessage(content=prompt)],
                config={"metadata": prompt_tmpl.langsmith_metadata()},
            ),
            label="cio_critic",
        )
        result = {
            "action": verdict.action,
            "critique": verdict.critique,
            "flagged_tickers": list(verdict.flagged_tickers or []),
            "suggested_drops": list(verdict.suggested_drops or []),
            "suggested_adds": list(verdict.suggested_adds or []),
        }
        log.info(
            "[cio_critic] action=%s flagged=%s critique=%s",
            verdict.action, result["flagged_tickers"],
            (verdict.critique or "")[:80],
        )
        return result
    except Exception as e:
        if is_strict_validation_enabled():
            raise
        log.warning("[cio_critic] LLM call failed: %s — accepting CIO slate", e)

    return {
        "action": "accept",
        "critique": "Critic unavailable — accepting CIO selection.",
        "flagged_tickers": [],
        "suggested_drops": [],
        "suggested_adds": [],
    }


def run_cio_with_reflection(
    candidates: list[dict],
    macro_context: dict,
    sector_ratings: dict,
    current_population: list[dict],
    open_slots: int,
    exits: list[dict],
    run_date: str,
    api_key: Optional[str] = None,
    prior_decisions: list[dict] | None = None,
    *,
    max_iterations: int = 2,
    **cio_kwargs,
) -> tuple[dict, dict]:
    """Run the CIO, then let a Haiku critic challenge the slate before finalizing.

    1. Initial ``run_cio`` call.
    2. ``run_cio_critic`` evaluates the advance set.
    3. If the critic says ``revise`` and iterations remain, re-run ``run_cio``
       once with the critique threaded into ``prior_decisions`` context.

    Returns ``(cio_result, reflection_log)``. ``reflection_log`` carries
    ``initial_advanced`` / ``final_advanced`` / ``critic_action`` /
    ``flagged_tickers`` / ``critique_text`` / ``iterations`` for telemetry. The
    CIO remains the sole gate — the critic only prompts a reconsideration; it
    never edits the slate directly.
    """
    result = run_cio(
        candidates=candidates,
        macro_context=macro_context,
        sector_ratings=sector_ratings,
        current_population=current_population,
        open_slots=open_slots,
        exits=exits,
        run_date=run_date,
        api_key=api_key,
        prior_decisions=prior_decisions,
        **cio_kwargs,
    )

    reflection_log = {
        "initial_advanced": list(result.get("advanced_tickers", [])),
        "iterations": 1,
        "critic_action": "accept",
        "flagged_tickers": [],
        "critique_text": "",
        "final_advanced": list(result.get("advanced_tickers", [])),
    }

    if not candidates:
        return result, reflection_log

    for iteration in range(1, max_iterations):
        critic = run_cio_critic(
            result, candidates, macro_context, sector_ratings, api_key=api_key
        )
        reflection_log["critic_action"] = critic.get("action", "accept")
        reflection_log["flagged_tickers"] = critic.get("flagged_tickers", [])
        reflection_log["critique_text"] = critic.get("critique", "")

        if critic.get("action") != "revise":
            log.info(
                "[cio_reflection] iteration %d: critic accepted slate (%d advanced)",
                iteration, len(result.get("advanced_tickers", [])),
            )
            break

        log.info(
            "[cio_reflection] iteration %d: critic requests revision — %s",
            iteration, critic.get("critique", "")[:80],
        )
        critique_note = {
            "ticker": "__CRITIC__",
            "thesis_type": "ic_critic_feedback",
            "rationale": (
                f"IC CRITIC FEEDBACK: {critic.get('critique', '')} "
                f"Flagged: {critic.get('flagged_tickers', [])}. "
                f"Suggested drops: {critic.get('suggested_drops', [])}. "
                f"Suggested adds: {critic.get('suggested_adds', [])}. "
                "Reconsider the advance set in light of this; keep names you can "
                "defend, drop names you cannot."
            ),
            "conviction": None,
            "score": None,
        }
        result = run_cio(
            candidates=candidates,
            macro_context=macro_context,
            sector_ratings=sector_ratings,
            current_population=current_population,
            open_slots=open_slots,
            exits=exits,
            run_date=run_date,
            api_key=api_key,
            prior_decisions=list(prior_decisions or []) + [critique_note],
            **cio_kwargs,
        )
        reflection_log["iterations"] = iteration + 1

    reflection_log["final_advanced"] = list(result.get("advanced_tickers", []))
    return result, reflection_log


def _format_prior_decisions(prior_decisions: list[dict] | None) -> str:
    """Format prior IC decisions for prompt injection. Returns empty string if none."""
    if not prior_decisions:
        return ""
    lines = ["PRIOR WEEK IC DECISIONS (for portfolio continuity):"]
    for d in prior_decisions[:10]:
        ticker = d.get("ticker", "?")
        action = d.get("thesis_type", "?")
        rationale = (d.get("rationale", "") or "")[:120]
        lines.append(f"  - {ticker}: {action} — {rationale}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_cio_prompt(
    candidates: list[dict],
    macro_context: dict,
    sector_ratings: dict,
    population: list[dict],
    open_slots: int,
    exits: list[dict],
    run_date: str,
    prior_decisions: list[dict] | None = None,
    prior_cycle_scorecard: str | None = None,
    deblended: bool = False,
    sector_neutral_quality: dict[str, float] | None = None,
) -> str:
    """Build the single batch CIO prompt.

    The ``prior_cycle_scorecard`` kwarg carries the rendered text from
    ``evals.last_week_scorecard.format_scorecard_text`` for the prior
    cycle's realized outcomes (per-sector hit rate, surprises,
    confirmations). When None / "" — the default and the pre-Brian's-
    gitignored-template-edit state — the kwarg is silently unused by
    ``str.format`` since the template has no ``{prior_cycle_scorecard}``
    placeholder yet. Mirrors the established
    ``agents/macro_agent.py::regime_substrate_block`` pattern.

    When ``deblended`` (L4564), each candidate line carries a SECTOR-NEUTRAL
    stock-quality rank (0–100, within this pool, derived from
    ``sector_neutral_quality``) and the v1.5.0 ``_PROMPT_DEBLENDED`` template
    is loaded — it ranks PRIMARILY on that neutral quality and weighs the
    sector tilt once, killing the rubric-bias double-count. OFF → byte-identical
    to the legacy path.
    """

    # De-blended: render a within-pool 0–100 rank of the sector-neutral quality
    # (a monotonic transform of the z-scores — Spearman-equivalent to what the
    # backtester attribution measures, but unifies z + cold-start-fallback onto
    # one interpretable scale for the LLM).
    quality_rank = (
        {t: v * 100.0 for t, v in _pct_rank(sector_neutral_quality).items()}
        if deblended and sector_neutral_quality
        else {}
    )

    # Format candidates
    cand_lines = []
    for i, c in enumerate(candidates, 1):
        team = c.get("team_id", "unknown")
        qs = c.get("quant_score", "?")
        qls = c.get("qual_score", "?")
        conv = c.get("conviction", "?")
        bull = (c.get("bull_case", "") or "")[:150]
        bear = (c.get("bear_case", "") or "")[:150]
        cats = ", ".join(c.get("catalysts", [])[:3]) if c.get("catalysts") else "none specified"

        if deblended:
            sq = quality_rank.get(c["ticker"])
            sq_str = f"{sq:.0f}/100" if sq is not None else "n/a"
            head = (
                f"  {i}. {c['ticker']} [{team}] — Sector-Neutral Quality: "
                f"{sq_str}; Quant: {qs}, Qual: {qls}, Conviction: {conv}"
            )
        else:
            head = (
                f"  {i}. {c['ticker']} [{team}] — Quant: {qs}, "
                f"Qual: {qls}, Conviction: {conv}"
            )
        cand_lines.append(
            f"{head}\n"
            f"     Bull: {bull}\n"
            f"     Bear: {bear}\n"
            f"     Catalysts: {cats}"
        )
    candidates_text = "\n".join(cand_lines)

    # Format current population by sector
    pop_by_sector = {}
    for p in population:
        sector = p.get("sector", "Unknown")
        pop_by_sector.setdefault(sector, []).append(p.get("ticker", ""))
    pop_text = "\n".join(f"  {s}: {', '.join(ts)}" for s, ts in sorted(pop_by_sector.items()))

    # Format exits
    exit_text = "\n".join(
        f"  - {e.get('ticker_out', e.get('ticker', '?'))}: {e.get('reason', 'unknown')}"
        for e in exits[:10]
    ) if exits else "  None"

    # Format sector ratings
    ratings_text = "\n".join(
        f"  {s}: {r.get('rating', 'market_weight')} (modifier: {r.get('modifier', 1.0):.2f})"
        for s, r in sorted(sector_ratings.items())
    )

    regime = macro_context.get("market_regime", "neutral")

    return load_prompt(_PROMPT_DEBLENDED if deblended else _PROMPT_DEFAULT).format(
        run_date=run_date,
        regime=regime,
        open_slots=open_slots,
        ratings_text=ratings_text,
        pop_text=pop_text,
        exit_text=exit_text,
        prior_decisions_block=_format_prior_decisions(prior_decisions),
        candidates_text=candidates_text,
        prior_cycle_scorecard=prior_cycle_scorecard or "",
    )


def _combined_score(c: dict) -> float:
    """Combined quant+qual score used for floor force-fill ranking."""
    qs = c.get("quant_score") or 0
    qls = c.get("qual_score") or 0
    return (qs + qls) / 2 if qls else qs


# ── De-blended CIO orchestration (L4564) ────────────────────────────────────
# Strip the rubric's persistent per-sector bias from the composite stock score
# in CODE so the CIO ranks on apples-to-apples quality and weighs the sector
# tilt SEPARATELY (instead of the raw, sector-biased score double-counting with
# the sector ratings it also receives). The construction mirrors the backtester
# instrument `analysis/end_to_end.py::_trailing_sector_neutral` (L4564 Phase A,
# measured IC +0.086 vs +0.042 pool-wide / -0.016 raw at n=89) so the live
# signal is exactly what the attribution recomputes — no separate persistence.


def _pct_rank(values: dict[str, float]) -> dict[str, float]:
    """Percentile rank in (0, 1] with pandas ``.rank(pct=True)`` semantics
    (average rank for ties / n). The cold-start / thin-sector fallback."""
    items = [(t, v) for t, v in values.items() if v is not None]
    n = len(items)
    if n == 0:
        return {}
    order = sorted(items, key=lambda kv: kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and order[j + 1][1] == order[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # 1-based ranks i..j averaged
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg_rank / n
        i = j + 1
    return ranks


def compute_sector_neutral_quality(
    candidate_quality: dict[str, tuple[str, float]],
    prior_sector_scores: dict[str, list[float]],
    *,
    k_min: int = 6,
) -> dict[str, float]:
    """Deterministic sector-neutral stock-quality per candidate (L4564).

    ``candidate_quality`` maps ticker → (sector, composite ``weighted_base``).
    ``prior_sector_scores`` maps sector → that sector's PRIOR-cycle scores from
    research.db. For each candidate::

        q = (weighted_base − μ_sector) / σ_sector

    where μ/σ are the trailing mean/std of the candidate's sector (the
    persistent rubric bias estimated from history — the current pool is only
    ~2–3 names/sector, too thin for a within-cycle within-sector z-score).
    Sectors with < ``k_min`` prior samples (cold start) fall back to the
    within-pool percentile rank, so every candidate gets a comparable value.
    Mirrors the backtester ``_trailing_sector_neutral``.

    Returns ``{ticker: neutral_score}`` (z-scores; fallback rows ∈ (0, 1]).
    """
    out: dict[str, float] = {}
    fallback: list[str] = []
    for ticker, (sector, q) in candidate_quality.items():
        prior = prior_sector_scores.get(sector) or []
        if len(prior) >= k_min:
            mu = sum(prior) / len(prior)
            var = sum((x - mu) ** 2 for x in prior) / (len(prior) - 1)
            sd = var ** 0.5
            if sd > 1e-9:
                out[ticker] = (q - mu) / sd
                continue
        fallback.append(ticker)
    if fallback:
        pool = _pct_rank({t: candidate_quality[t][1] for t in candidate_quality})
        for t in fallback:
            out[t] = pool.get(t, 0.5)
    return out


def load_prior_sector_scores(db_conn, run_date: str) -> dict[str, list[float]]:
    """Trailing per-sector composite scores from research.db (leak-free).

    Joins ``cio_evaluations.combined_score`` (= composite ``weighted_base``) to
    ``universe_returns.sector`` on (ticker, eval_date) for all eval_dates
    STRICTLY BEFORE ``run_date`` — the same sector source the backtester
    instrument uses. Returns ``{sector: [score, ...]}``; empty on any error
    (caller then ranks every candidate pool-wide, logged — graceful, never
    silently wrong)."""
    out: dict[str, list[float]] = {}
    if db_conn is None or not run_date:
        return out
    try:
        rows = db_conn.execute(
            "SELECT ur.sector, ce.combined_score "
            "FROM cio_evaluations ce "
            "JOIN universe_returns ur "
            "  ON ce.ticker = ur.ticker AND ce.eval_date = ur.eval_date "
            "WHERE ce.eval_date < ? AND ce.combined_score IS NOT NULL "
            "  AND ur.sector IS NOT NULL",
            (run_date,),
        ).fetchall()
    except Exception as e:  # noqa: BLE001 — graceful de-blend degradation, logged
        log.warning(
            "[cio] de-blend: prior sector-score load failed (%s) — every "
            "candidate falls back to pool-wide rank this cycle", e,
        )
        return out
    for sector, score in rows:
        if sector is not None and score is not None:
            out.setdefault(sector, []).append(float(score))
    return out


def build_sector_neutral_quality_map(
    candidates: list[dict],
    investment_theses: dict,
    db_conn,
    run_date: str,
    *,
    k_min: int = 6,
) -> dict[str, float]:
    """Orchestrate the live sector-neutral quality map for the CIO node.

    Sources each candidate's composite ``weighted_base`` + sector from
    ``investment_theses`` (computed upstream by ``score_aggregator``), loads the
    trailing per-sector baseline from research.db, and returns
    ``{ticker: neutral_score}``. Candidates missing a ``weighted_base``/sector
    are omitted (the prompt renderer falls back to no-rank for them)."""
    candidate_quality: dict[str, tuple[str, float]] = {}
    for c in candidates:
        ticker = c.get("ticker")
        thesis = (investment_theses or {}).get(ticker, {})
        wb = thesis.get("weighted_base")
        sector = thesis.get("sector") or c.get("sector")
        if ticker and wb is not None and sector:
            candidate_quality[ticker] = (sector, float(wb))
    if not candidate_quality:
        return {}
    prior = load_prior_sector_scores(db_conn, run_date)
    return compute_sector_neutral_quality(candidate_quality, prior, k_min=k_min)


def _reconcile_cio_decisions(
    decisions: list[dict], candidates: list[dict],
) -> tuple[list[dict], dict]:
    """Reconcile the LLM's raw decisions against the candidate ticker SET.

    Replaces the prior brittle ``len(decisions) == len(candidates)``
    assertion. Sonnet's structured-output batch occasionally emits a
    stray extra decision object (2026-05-17 SF: 19 decisions for 18
    candidates) or a hallucinated ticker not in the candidate set; the
    raw count check turned either benign artifact into a hard failure of
    the whole weekly run, while *missing* a duplicate that left a real
    candidate uncovered at an equal count.

    Reconciliation rules:

    * **Extraneous** — a decision whose ticker is not in the candidate
      set is dropped (the CIO can only rule on what it was given).
    * **Duplicate** — multiple decisions for the same candidate are
      collapsed to one with *conservative-wins* precedence: a
      non-ADVANCE decision beats an ADVANCE one, so a stray duplicate
      can never *upgrade* a candidate into advancement. Ties keep the
      first occurrence (the LLM's primary ordered judgment).
    * **Missing** — a candidate with no surviving decision is reported;
      the caller hard-fails in strict mode (this is the genuine
      partial-list regression the original assertion guarded).

    The returned decisions are emitted in candidate order with each
    decision's ``ticker`` normalised to the candidate's canonical
    spelling, so ``_post_process_cio_decisions`` exact-ticker matching
    stays deterministic even if the LLM altered casing/whitespace.

    Returns ``(reconciled_decisions, diagnostics)`` where diagnostics
    has keys ``raw_count``, ``extraneous``, ``duplicate``, ``missing``.
    """

    def _norm(t) -> str:
        return str(t or "").strip().upper()

    # Conservative-wins ranking: higher == more conservative (kept on a
    # duplicate clash). Unknown/HOLD treated as mid (never upgrades).
    _conservatism = {"ADVANCE": 0, "NO_ADVANCE_DEADLOCK": 1, "REJECT": 2}

    def _rank(dec: dict) -> int:
        return _conservatism.get(str(dec.get("decision") or "").upper(), 1)

    canonical: dict[str, str] = {}
    for c in candidates:
        nt = _norm(c.get("ticker"))
        if nt and nt not in canonical:
            canonical[nt] = c.get("ticker")

    chosen: dict[str, dict] = {}
    extraneous: list[str] = []
    duplicate: list[str] = []
    for d in decisions:
        nt = _norm(d.get("ticker"))
        if nt not in canonical:
            extraneous.append(d.get("ticker"))
            continue
        d = dict(d)
        d["ticker"] = canonical[nt]  # normalise to canonical spelling
        if nt not in chosen:
            chosen[nt] = d
        else:
            if nt not in duplicate:
                duplicate.append(nt)
            # Keep the more conservative of the two; tie → keep first.
            if _rank(d) > _rank(chosen[nt]):
                chosen[nt] = d

    reconciled = [chosen[_norm(c.get("ticker"))]
                  for c in candidates if _norm(c.get("ticker")) in chosen]
    missing = [c.get("ticker") for c in candidates
               if _norm(c.get("ticker")) not in chosen]

    return reconciled, {
        "raw_count": len(decisions),
        "extraneous": extraneous,
        "duplicate": duplicate,
        "missing": missing,
    }


def _decision_conviction(d: dict) -> float:
    """CIO-assigned conviction on a decision (the ~0-100 score the entrant
    bar is measured against). Missing → 0."""
    c = d.get("conviction")
    return float(c) if isinstance(c, (int, float)) else 0.0


def _stamp_candidate_context(decisions: list[dict], candidates: list[dict]) -> None:
    """Join sector + sub-scores from the source candidate onto each decision,
    in place (L4533). The LLM decision carries only ticker/decision/conviction
    — sector lives on the upstream team recommendation and was being dropped,
    leaving downstream consumers (the dashboard new-entrant panel, the
    underweight-sector tripwire context) unable to attribute a REJECTED fresh
    name to its sector. ``CIODecision`` is ``extra="allow"`` so these ride
    through validation. Only fills fields that are absent/None — never clobbers.
    """
    by_ticker = {c.get("ticker"): c for c in candidates if c.get("ticker")}
    for d in decisions:
        cand = by_ticker.get(d.get("ticker"))
        if not cand:
            continue
        for field in ("sector", "quant_score", "qual_score"):
            if d.get(field) is None and cand.get(field) is not None:
                d[field] = cand[field]


def _post_process_cio_decisions(
    decisions: list[dict],
    candidates: list[dict],
    floor: int,
    cap: int,
    held_tickers: frozenset[str] | set[str] = frozenset(),
    force_fill_conviction_floor: float = 0.0,
) -> dict:
    """Apply cap/floor/force-fill post-processing to a typed CIO decision list.

    The cap and floor govern **net-new entrants** (candidates NOT already in
    the held population) — NOT total advances. Re-affirmations of incumbents
    are unbounded (per the universe.yaml entrant-cap contract) and never
    truncated. This fixes the bug where a week that re-advanced N incumbents
    satisfied ``floor`` while admitting **zero** genuinely-new names — the
    "min_new_entrants" guarantee was silently measuring the wrong thing.

    Bounds enforcement (net-new only):
    - Truncate NEW advances at ``cap`` (lowest-conviction first) if the rubric
      advanced more new names than the ceiling. Incumbent re-advances are kept.
    - Force-fill toward ``floor`` from non-advanced NET-NEW candidates whose
      CIO conviction ≥ ``force_fill_conviction_floor`` (the entrant bar),
      ranked by conviction. **Quality-gated by design: never force a sub-bar
      name in** — when no fresh candidate clears the bar (saturation week),
      net-new stays below floor and the caller's tripwire fires. Forced
      promotions are tagged ``decision="ADVANCE_FORCED"`` (see
      ADVANCE_DECISIONS — every advance-consumer must honor both literals).
    """
    held = frozenset(held_tickers)

    # Extract rubric ADVANCE decisions + entry theses, partitioned by whether
    # the ticker is already held (incumbent re-affirmation vs net-new entrant).
    incumbent_adv: list[str] = []
    new_adv: list[dict] = []  # keep the decision so we can rank by conviction
    entry_theses: dict = {}
    for d in decisions:
        if d.get("decision") != "ADVANCE":
            continue
        ticker = d.get("ticker", "")
        if d.get("entry_thesis"):
            entry_theses[ticker] = d["entry_thesis"]
        if ticker in held:
            incumbent_adv.append(ticker)
        else:
            new_adv.append(d)

    rubric_new_count = len(new_adv)

    # Ceiling on NET-NEW entrants only — drop the lowest-conviction new
    # advances past the cap (incumbent re-affirmations are unbounded).
    new_adv.sort(key=_decision_conviction, reverse=True)
    truncated_new = new_adv[:cap]
    truncated_count = rubric_new_count - len(truncated_new)
    new_adv_tickers = [d.get("ticker", "") for d in truncated_new]

    # Floor: quality-gated force-fill of NET-NEW names if the rubric came up
    # short. Pool = non-advanced, not-held decisions clearing the entrant bar.
    forced_tickers: list[str] = []
    if len(new_adv_tickers) < floor:
        advanced_set = set(incumbent_adv) | set(new_adv_tickers)
        pool = [
            d for d in decisions
            if d.get("ticker") not in advanced_set
            and d.get("ticker") not in held
            and _decision_conviction(d) >= force_fill_conviction_floor
        ]
        pool.sort(key=_decision_conviction, reverse=True)
        # Don't exceed the cap even when force-filling.
        shortfall = min(floor - len(new_adv_tickers), cap - len(new_adv_tickers))
        for d in pool[:max(0, shortfall)]:
            ticker = d.get("ticker", "")
            new_adv_tickers.append(ticker)
            forced_tickers.append(ticker)
            d["decision"] = "ADVANCE_FORCED"
            prior_rationale = d.get("rationale", "") or ""
            d["rationale"] = (
                f"{prior_rationale} | Floor enforcement: rubric advanced "
                f"{rubric_new_count} net-new of {len(candidates)} candidates; "
                f"promoted (conviction {_decision_conviction(d):.0f} ≥ bar "
                f"{force_fill_conviction_floor:.0f}) to hit "
                f"min_new_entrants={floor}."
            ).strip(" |")

    advanced = incumbent_adv + new_adv_tickers
    net_new_count = len(new_adv_tickers)

    log.info(
        "[cio] %d advanced (%d incumbent re-affirm + %d net-new [%d rubric + "
        "%d forced]), %d new truncated, %d rejected, %d deadlocked out of %d "
        "candidates [floor=%d cap=%d, bar=%.0f, held=%d]",
        len(advanced),
        len(incumbent_adv),
        net_new_count,
        net_new_count - len(forced_tickers),
        len(forced_tickers),
        truncated_count,
        len([d for d in decisions if d.get("decision") == "REJECT"]),
        len([d for d in decisions if d.get("decision") == "NO_ADVANCE_DEADLOCK"]),
        len(decisions),
        floor, cap, force_fill_conviction_floor, len(held),
    )

    _stamp_candidate_context(decisions, candidates)

    return {
        "decisions": decisions,
        "advanced_tickers": advanced,
        "entry_theses": entry_theses,
        "net_new_entrants": net_new_count,
    }


def _fallback_selection(candidates: list[dict], floor: int) -> dict:
    """Fallback when the LLM signal is unusable.

    Advances exactly `floor` candidates by combined quant+qual score —
    NOT `cap`. When the LLM call fails, parsing breaks, or no decisions
    come back, we have no rubric output to truncate against. Be
    conservative: hit the floor and stop. Don't force max-advance on
    broken data.
    """
    scored = [(_combined_score(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    decisions = []
    advanced = []
    entry_theses: dict = {}
    for i, (score, c) in enumerate(scored):
        if i < floor:
            decisions.append({
                "ticker": c["ticker"],
                "decision": "ADVANCE",
                "rank": i + 1,
                "conviction": int(score),
                "rationale": (
                    "Fallback (LLM unusable): selected by combined score "
                    f"to hit floor={floor}"
                ),
                "entry_thesis": None,
            })
            advanced.append(c["ticker"])
        else:
            decisions.append({
                "ticker": c["ticker"],
                "decision": "REJECT",
                "rank": None,
                "conviction": int(score),
                "rationale": "Fallback: below floor cutoff",
                "entry_thesis": None,
            })

    _stamp_candidate_context(decisions, candidates)

    return {
        "decisions": decisions,
        "advanced_tickers": advanced,
        "entry_theses": entry_theses,
    }


def _reject_decision(candidate: dict, reason: str) -> dict:
    return {
        "ticker": candidate.get("ticker", ""),
        "decision": "REJECT",
        "rank": None,
        "conviction": 0,
        "rationale": reason,
        "entry_thesis": None,
        # L4533: carry sector + sub-scores so rejected names are attributable
        # downstream (panel sector column, underweight-sector tripwire context).
        "sector": candidate.get("sector"),
        "quant_score": candidate.get("quant_score"),
        "qual_score": candidate.get("qual_score"),
    }
