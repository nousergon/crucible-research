"""
CIO Agent — evaluates all sector team recommendations in a single batch Sonnet call.

The CIO sees all candidates simultaneously and selects up to `max_new_entrants` for advancement (floor-enforced; `open_slots` is informational only — see `_compute_advance_bounds`).
Evaluates on 5 dimensions: risk/reward asymmetry (primary), team conviction, macro alignment, portfolio fit, catalyst specificity.
Writes entry theses for advanced stocks. All decisions (advance, reject, deadlock) saved.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from config import STRATEGIC_MODEL, MAX_TOKENS_STRATEGIC, ANTHROPIC_API_KEY
from agents.prompt_loader import load_prompt
from agents.langchain_utils import (
    SECTOR_TEAM_LLM_MAX_RETRIES,
    invoke_with_rate_limit_retry,
)

log = logging.getLogger(__name__)


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
        callbacks=[get_cost_telemetry_callback()],
    )

    prompt = _build_cio_prompt(
        candidates, macro_context, sector_ratings,
        current_population, cap, exits, run_date,
        prior_decisions=prior_decisions,
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
                        "ic_cio_evaluation"
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
        return _post_process_cio_decisions(decisions_dicts, candidates, floor, cap)
    except Exception as e:
        log.error("[cio] evaluation failed: %s", e)
        if is_strict_validation_enabled():
            raise
        # Lax fallback advances only `floor` (not `cap`). When the LLM signal is
        # unusable, be conservative — don't force max-advance on broken data.
        return _fallback_selection(candidates, floor)


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
    """

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

        cand_lines.append(
            f"  {i}. {c['ticker']} [{team}] — Quant: {qs}, Qual: {qls}, Conviction: {conv}\n"
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

    return load_prompt("ic_cio_evaluation").format(
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


def _post_process_cio_decisions(
    decisions: list[dict],
    candidates: list[dict],
    floor: int,
    cap: int,
) -> dict:
    """Apply cap/floor/force-fill post-processing to a typed CIO decision list.

    PR 2.3 Step E split: regex/JSON parsing was retired (the LLM call now
    uses ``with_structured_output(CIORawOutput)`` and we receive a typed
    list directly). This function preserves the existing post-processing
    logic — bounds enforcement, ADVANCE_FORCED synthesis, audit trail
    annotation — operating on a ``list[dict]`` decisions input regardless
    of how the upstream parsing happened.

    Bounds enforcement:
    - Truncate at `cap` if the rubric advanced more than the ceiling.
    - Force-fill to `floor` from REJECT/DEADLOCK candidates (ranked by
      combined quant+qual score) when the rubric advanced fewer than the
      floor. Forced promotions are tagged `decision="ADVANCE_FORCED"` so
      the audit trail distinguishes them from rubric-driven advances.
    """
    # Extract advanced tickers + theses from rubric ADVANCE decisions
    advanced = []
    entry_theses = {}
    for d in decisions:
        if d.get("decision") == "ADVANCE":
            ticker = d.get("ticker", "")
            advanced.append(ticker)
            if d.get("entry_thesis"):
                entry_theses[ticker] = d["entry_thesis"]

    rubric_advanced_count = len(advanced)

    # Ceiling: truncate if rubric exceeded cap
    advanced = advanced[:cap]
    truncated_count = rubric_advanced_count - len(advanced)

    # Floor: force-fill from non-advanced candidates if rubric came up short.
    # The team-level Quant+Qual+Peer Review already validated all candidates;
    # the CIO rubric is a secondary editorial gate. When the rubric is too
    # strict in a given week, fall back to the best of the team-validated
    # pool ranked by combined quant+qual score.
    forced_tickers: list[str] = []
    if len(advanced) < floor:
        advanced_set = set(advanced)
        not_advanced = [
            c for c in candidates if c.get("ticker") not in advanced_set
        ]
        not_advanced.sort(key=_combined_score, reverse=True)
        shortfall = floor - len(advanced)
        for c in not_advanced[:shortfall]:
            ticker = c["ticker"]
            advanced.append(ticker)
            forced_tickers.append(ticker)
        # Mutate matching decision entries so the audit trail reflects the
        # forced promotion. Add synthetic entries for any forced ticker
        # missing from the LLM's decisions list.
        existing_decision_tickers = {d.get("ticker") for d in decisions}
        for ticker in forced_tickers:
            matched = False
            for d in decisions:
                if d.get("ticker") == ticker:
                    d["decision"] = "ADVANCE_FORCED"
                    prior_rationale = d.get("rationale", "") or ""
                    d["rationale"] = (
                        f"{prior_rationale} | Floor enforcement: rubric "
                        f"advanced {rubric_advanced_count} of {len(candidates)}; "
                        f"promoted to hit min_new_entrants={floor}."
                    ).strip(" |")
                    matched = True
                    break
            if not matched and ticker not in existing_decision_tickers:
                cand = next(
                    (c for c in candidates if c.get("ticker") == ticker), None,
                )
                decisions.append({
                    "ticker": ticker,
                    "decision": "ADVANCE_FORCED",
                    "rank": None,
                    "conviction": int(_combined_score(cand or {})),
                    "rationale": (
                        f"Floor enforcement: rubric advanced "
                        f"{rubric_advanced_count} of {len(candidates)}; "
                        f"promoted to hit min_new_entrants={floor}."
                    ),
                    "entry_thesis": None,
                })

    log.info(
        "[cio] %d advanced (%d rubric + %d forced), %d truncated, "
        "%d rejected, %d deadlocked out of %d candidates "
        "[floor=%d cap=%d]",
        len(advanced),
        rubric_advanced_count - truncated_count,
        len(forced_tickers),
        truncated_count,
        len([d for d in decisions if d.get("decision") == "REJECT"]),
        len([d for d in decisions if d.get("decision") == "NO_ADVANCE_DEADLOCK"]),
        len(decisions),
        floor, cap,
    )

    return {
        "decisions": decisions,
        "advanced_tickers": advanced,
        "entry_theses": entry_theses,
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
    }
