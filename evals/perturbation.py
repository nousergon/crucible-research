"""Synthetic-perturbation judge validator (ROADMAP L480, 2026-05-29).

Validates the LLM-as-judge on its ACTUAL construct — *process quality* —
without any human labels. Method: take a known-good, shape-realistic
agent output (the "reference"), apply a DETERMINISTIC corruption that
targets exactly one rubric dimension, run the judge on both, and assert
the judge (a) does not score the corrupted version higher overall and
(b) DROPS the targeted dimension. Because we authored the corruption the
ground-truth ordering is known by construction — zero annotation.

This tests **sensitivity** (does the judge notice degradation at all?)
and **dimension-specificity** (does the *right* dimension move?). Those
catch the failure modes that matter for an observability judge: a
rubber-stamp judge (no sensitivity), a halo-effect judge (no
specificity), and a verbosity-biased judge (the pad-worse-but-longer
probe).

This is explicitly NOT outcome-IC. It never touches stock returns.
Outcome (realized alpha) is a separate axis — a firewalled *system*
diagnostic — because reasoning quality and 21d return are only weakly
correlated, so validating (let alone tuning) the judge on outcomes would
Goodhart it from a process-quality assessor into a luck-predictor.

Honest limits: validates ordinal sensitivity + dimension targeting, NOT
absolute-scale calibration (that needs a human anchor, deliberately
deprioritized). And it only exercises the corruptions we authored —
necessary, not sufficient.

Design: the corruption functions are pure + deterministic, so the
harness logic is unit-tested in regular (mocked, no-key) CI; only the
``reference > corrupted`` assertion needs the live judge, which runs in a
paths-filtered live smoke (``tests/live_smoke/judge_perturbation_smoke.py``)
and — Phase B — a weekly scorecard.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Callable, Optional

from nousergon_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)

from evals.judge import DEFAULT_JUDGE_MODEL, evaluate_artifact

logger = logging.getLogger(__name__)

# Weekly sensitivity scorecard (Phase B, config#752). Mirrors
# calibration_kappa's report layout: research owns the render, the backtester
# evaluator email embeds ``latest/sensitivity.md`` verbatim. Same bucket env
# override as calibration_kappa so a single deploy config drives both.
_RESEARCH_BUCKET = os.environ.get("CHANGELOG_BUCKET", "alpha-engine-research")
_REPORT_PREFIX = "decision_artifacts/_perturbation/_report/"


# ── Reference fixtures ─────────────────────────────────────────────────────
#
# Synthetic-but-shape-realistic known-good agent outputs. NOT real
# production outputs (those are proprietary) — hand-authored to exercise
# each rubric dimension positively: specific numbers, multi-step
# reasoning, score-consistent rankings, complete coverage. A well-behaved
# judge should score these HIGH; the corruptions below break one
# dimension at a time and the judge should notice.

_QUANT_REFERENCE: dict[str, Any] = {
    "ranked_picks": [
        {
            "ticker": "AAPL",
            "quant_score": 82,
            "rationale": (
                "RSI-14 at 58 (neutral-bullish, not overbought); 20d/50d MA "
                "crossover confirmed 6 sessions ago; relative strength vs XLK "
                "+4.2% over 21d; avg daily volume 58M shares supports liquidity."
            ),
            "key_metrics": {"rsi_14": 58, "rs_vs_sector_21d": 0.042,
                            "ma_cross": "20>50", "avg_vol_20d": 58_000_000},
        },
        {
            "ticker": "MSFT",
            "quant_score": 74,
            "rationale": (
                "RSI-14 at 54; price holding above 50d MA; relative strength "
                "vs XLK +1.8% over 21d; volume steady at 24M. Slightly weaker "
                "momentum than AAPL hence the lower score."
            ),
            "key_metrics": {"rsi_14": 54, "rs_vs_sector_21d": 0.018,
                            "ma_cross": "above_50d", "avg_vol_20d": 24_000_000},
        },
        {
            "ticker": "NVDA",
            "quant_score": 61,
            "rationale": (
                "RSI-14 at 71 (approaching overbought); strong 21d RS of +9.1% "
                "but extension risk caps the score; volume elevated at 41M."
            ),
            "key_metrics": {"rsi_14": 71, "rs_vs_sector_21d": 0.091,
                            "ma_cross": "20>50", "avg_vol_20d": 41_000_000},
        },
    ],
}

_QUANT_INPUT_SNAPSHOT: dict[str, Any] = {
    "team_id": "technology",
    "run_date": "2026-05-09",
    "market_regime": "neutral",
    "sector_tickers": ["AAPL", "MSFT", "NVDA"],
    "technical_scores_team": {
        "AAPL": {"rsi_14": 58, "technical_score": 82},
        "MSFT": {"rsi_14": 54, "technical_score": 74},
        "NVDA": {"rsi_14": 71, "technical_score": 61},
    },
}

_QUAL_REFERENCE: dict[str, Any] = {
    "assessments": [
        {
            "ticker": "AAPL",
            "qual_score": 78,
            "bull_case": (
                "Services revenue grew 14% YoY to $24B last quarter, lifting "
                "gross margin to 46% because services carry ~70% margin vs ~36% "
                "on hardware; this mix shift compounds as the installed base of "
                "2.2B devices monetizes, so EPS can grow faster than revenue."
            ),
            "bear_case": (
                "China iPhone units fell 9% YoY and Greater China is 17% of "
                "revenue, so a prolonged share-loss to Huawei would offset much "
                "of the services tailwind; regulatory pressure on App Store fees "
                "is a second, slower drag on the highest-margin line."
            ),
            "catalysts": ["WWDC AI roadmap (June)", "Q3 services print"],
            "risks": ["China share loss", "App Store fee regulation"],
            "conviction": 72,
        },
        {
            "ticker": "MSFT",
            "qual_score": 71,
            "bull_case": (
                "Azure grew 30% YoY with AI services contributing ~7 points of "
                "that growth; Copilot attach on the 400M M365 commercial seats "
                "is early (<5%) so there is a long monetization runway as price "
                "moves from $30/seat into the base."
            ),
            "bear_case": (
                "Capex guided to $50B+ to fund AI capacity compresses near-term "
                "free cash flow, and if Copilot attach stalls below ~10% the "
                "ROIC on that capex disappoints versus the multiple."
            ),
            "catalysts": ["Copilot attach disclosure", "Azure AI revenue split"],
            "risks": ["AI capex ROIC", "Copilot adoption stall"],
            "conviction": 66,
        },
    ],
}

_QUAL_INPUT_SNAPSHOT: dict[str, Any] = {
    "team_id": "technology",
    "run_date": "2026-05-09",
    "market_regime": "neutral",
    # `sector_tickers` / `sector_population` are the non-degenerate
    # signal the judge's _is_degenerate_input check requires for
    # sector_qual — without them the judge short-circuits to
    # `degenerate_input` and emits no dimension scores.
    "sector_tickers": ["AAPL", "MSFT", "NVDA"],
    "sector_population": ["AAPL", "MSFT", "NVDA"],
    "quant_top_picks": ["AAPL", "MSFT", "NVDA"],
}


# Thinktank thesis (company-level) — mirrors CompanyThesisLLM
# (thinktank/schemas.py) + the input_data_snapshot shape built in
# thinktank/analyst.py::build_thesis. Bullish-but-honest reference: cites
# specific board metrics/filings/news, names a real moat mechanism with its
# erosion risk, reconciles valuation with the provided pillars, orders risks
# by materiality, engages the provided macro/sector themes, and stance/
# conviction follow from the body — a well-behaved judge should score all
# six eval_rubric_thinktank_thesis dimensions HIGH.
_THINKTANK_THESIS_REFERENCE: dict[str, Any] = {
    "business_summary": (
        "Cloud infrastructure + AI-accelerated compute leasing; unit economics "
        "driven by fleet utilization (currently 78% per the board row) and "
        "per-GPU-hour realized pricing."
    ),
    "moat": (
        "Switching costs from multi-year reserved-capacity contracts plus a "
        "scale advantage in power-constrained datacenter siting (the tech_score "
        "77 in the board row reflects this build-out lead); erosion risk is "
        "hyperscaler in-house silicon narrowing the leasing-vs-build cost gap "
        "over a 3-5yr horizon."
    ),
    "filings_review": (
        "Latest 10-Q capex guide raised to $2.1B (from $1.6B) to fund GPU fleet "
        "expansion; management flagged a one-quarter lag between capex and "
        "revenue recognition as new capacity ramps utilization."
    ),
    "news_sentiment": (
        "News aggregate shows LM sentiment +0.34 over the trailing week on 4 "
        "events, driven by a hyperscaler capacity-partnership announcement; no "
        "negative-severity events recorded."
    ),
    "valuation": (
        "Trading at 14x forward EV/EBITDA vs the sector's 18x median (per the "
        "board row's value pillar); the discount is explained by the recent "
        "capex step-up depressing near-term FCF, not by a growth-quality gap — "
        "the quality pillar score of 81 doesn't support a structural discount."
    ),
    "market_dynamics": (
        "The house macro theme's 'AI capex supercycle, mid-innings' view directly "
        "supports continued fleet demand; the sector theme's utilization-rate "
        "watch item is the one to track here specifically because this name's "
        "78% utilization is the lever on the bull case."
    ),
    "risks": [
        "Hyperscaler capacity partnership (the same one lifting sentiment this "
        "week) could be renegotiated or in-sourced at renewal in 18 months",
        "Power-siting constraints could cap fleet growth below the capex guide",
        "General AI capex slowdown macro risk",
    ],
    "catalysts": [
        "Q3 utilization print (board row's current 78% vs guide)",
        "Reserved-capacity contract renewal disclosure",
    ],
    "stance": "attractive",
    "conviction": 74,
    "summary": (
        "Bullish on continued fleet utilization gains supporting a valuation "
        "re-rate toward sector median; primary risk is customer concentration "
        "at contract renewal, which the current catalysts calendar should "
        "resolve within two quarters."
    ),
}

_THINKTANK_THESIS_INPUT_SNAPSHOT: dict[str, Any] = {
    "ticker": "NVDA",
    "update_reason": "event",
    "event_context": "Hyperscaler capacity-partnership announcement (2026-07-01)",
    "board_row": {
        "ticker": "NVDA",
        "sector": "technology",
        "industry": "semiconductors",
        "attractiveness_score": 79,
        "pillars": {"value": 68, "quality": 81, "growth": 84},
        "focus_score": 72,
        "tech_score": 77,
        "metrics": {"ev_ebitda_fwd": 14.0, "sector_median_ev_ebitda_fwd": 18.0,
                    "utilization_rate": 0.78, "capex_guide_usd_b": 2.1},
        "tradeability": {"avg_daily_volume": 41_000_000},
    },
    "weekly_signal": {"stance": "overweight", "rationale": "fleet utilization trend"},
    "news_aggregate": {"lm_sentiment_7d": 0.34, "event_count_7d": 4, "severity_max": "low"},
    "filings_excerpts": [
        "Capex guidance raised to $2.1B from $1.6B to fund GPU fleet expansion.",
        "Management: one-quarter lag between capex deployment and revenue recognition.",
    ],
    "macro_theme": "AI capex supercycle, mid-innings — demand still outstripping supply.",
    "sector_theme": "Semiconductors: watch fleet utilization rates as the near-term signal.",
    "prior_thesis": None,
}

# Thinktank theme (macro/sector keeper) — mirrors ThemeThesisLLM +
# thinktank/themes.py::_store_new's input_data_snapshot, RECONCILE mode
# (exercises anchor_fidelity most directly: reconcile must re-anchor to the
# weekly view and honestly surface any divergence). Churn-disciplined:
# material_change=False with a routine reconcile that finds no divergence.
_THINKTANK_THEME_REFERENCE: dict[str, Any] = {
    "narrative": (
        "Reconciling to the new weekly sector report: semiconductors remain "
        "overweight on AI capex demand. The intraweek utilization-rate watch "
        "item is confirmed by this week's report rather than contradicted, so "
        "no divergence to flag."
    ),
    "stance": "overweight",
    "drivers": ["AI capex demand", "fleet utilization trend confirmed by weekly report"],
    "watch_items": ["Q3 utilization prints across covered names", "hyperscaler capex guidance revisions"],
    "material_change": False,
    "change_summary": "",
}

_THINKTANK_THEME_INPUT_SNAPSHOT: dict[str, Any] = {
    "kind": "sector",
    "key": "technology",
    "update_reason": "reconcile",
    "market_regime": "risk-on",
    "weekly_anchor_date": "2026-06-29",
    "prior_theme": {
        "narrative": "Semiconductors overweight on AI capex demand; watching fleet utilization.",
        "stance": "overweight",
        "drivers": ["AI capex demand"],
        "watch_items": ["Q3 utilization prints across covered names"],
        "material_change": True,
        "change_summary": "Upgraded to overweight on capex-cycle acceleration.",
    },
}


REFERENCE_FIXTURES: dict[str, dict[str, Any]] = {
    "eval_rubric_sector_quant": {
        "agent_id": "sector_quant:technology",
        "agent_output": _QUANT_REFERENCE,
        "input_data_snapshot": _QUANT_INPUT_SNAPSHOT,
    },
    "eval_rubric_sector_qual": {
        "agent_id": "sector_qual:technology",
        "agent_output": _QUAL_REFERENCE,
        "input_data_snapshot": _QUAL_INPUT_SNAPSHOT,
    },
    "eval_rubric_thinktank_thesis": {
        "agent_id": "thinktank_thesis",
        "agent_output": _THINKTANK_THESIS_REFERENCE,
        "input_data_snapshot": _THINKTANK_THESIS_INPUT_SNAPSHOT,
    },
    "eval_rubric_thinktank_theme": {
        "agent_id": "thinktank_theme",
        "agent_output": _THINKTANK_THEME_REFERENCE,
        "input_data_snapshot": _THINKTANK_THEME_INPUT_SNAPSHOT,
    },
}


# ── Deterministic corruptions ──────────────────────────────────────────────
#
# Each takes a deep-copyable agent_output dict and returns a corrupted
# copy that degrades exactly ONE rubric dimension. Pure + deterministic
# so they are unit-tested without any LLM call.

_NUM_RE = re.compile(r"\d")


def _strip_numerical_grounding(out: dict) -> dict:
    """Quant: remove every concrete number — empty key_metrics, scrub
    digits from rationales. Targets `numerical_grounding`."""
    for p in out.get("ranked_picks", []):
        p["key_metrics"] = {}
        p["rationale"] = "Strong technical setup; momentum looks favorable here."
    return out


def _break_ranking_coherence(out: dict) -> dict:
    """Quant: reassign scores ASCENDING down the existing list order so
    the pick listed first (and whose rationale describes it as strongest)
    now carries the LOWEST quant_score — list rank, score, and rationale
    all contradict each other. Tickers + rationales untouched, so the
    incoherence is purely score-vs-rank-vs-narrative. Targets
    `ranking_coherence`."""
    picks = out.get("ranked_picks", [])
    n = len(picks)
    # e.g. 3 picks → [60, 72, 84]; first-listed gets the worst score.
    for i, p in enumerate(picks):
        p["quant_score"] = 60 + i * 12
    return out


def _flatten_signal_calibration(out: dict) -> dict:
    """Quant: collapse all quant_scores to an identical value so there is
    no differentiation/gradient across picks. Targets `signal_calibration`."""
    for p in out.get("ranked_picks", []):
        p["quant_score"] = 75
    return out


def _gut_output_completeness(out: dict) -> dict:
    """Quant: drop to a single pick with an empty rationale — inadequate
    coverage for the team's contract. Targets `output_completeness`."""
    picks = out.get("ranked_picks", [])
    if picks:
        first = picks[0]
        first["rationale"] = ""
        first["key_metrics"] = {}
        out["ranked_picks"] = [first]
    return out


def _strip_citation_grounding(out: dict) -> dict:
    """Qual: replace fact-grounded bull/bear with generic vague claims.
    Targets `citation_grounding`."""
    for a in out.get("assessments", []):
        a["bull_case"] = "The company is well run and has good prospects."
        a["bear_case"] = "There are some risks and the macro could be a headwind."
    return out


def _flatten_reasoning_depth(out: dict) -> dict:
    """Qual: collapse multi-step cause→effect chains to single-clause
    assertions. Targets `reasoning_depth`."""
    for a in out.get("assessments", []):
        a["bull_case"] = "Revenue is growing."
        a["bear_case"] = "Competition exists."
    return out


def _misalign_evidence(out: dict) -> dict:
    """Qual: set a very bullish qual_score while the bear_case dominates a
    thin bull_case — score no longer reflects the evidence balance.
    Targets `evidence_alignment`."""
    for a in out.get("assessments", []):
        a["qual_score"] = 94
        a["conviction"] = 95
        a["bull_case"] = "Probably fine."
        # bear_case left as the substantive, fact-heavy original.
    return out


def _verbosity_pad(out: dict) -> dict:
    """Cross-cutting: take the numerical-grounding corruption and PAD each
    rationale with long filler so the (worse) output is LONGER than the
    reference. A verbosity-biased judge would reward the length; a good
    judge still scores `numerical_grounding` down. Targets
    `numerical_grounding` via the verbosity-bias failure mode."""
    out = _strip_numerical_grounding(out)
    filler = (
        " It is worth emphasizing, broadly speaking, that this name remains a "
        "high-quality franchise with a durable position and many attractive "
        "qualities that investors have long appreciated across cycles and "
        "regimes, all things considered, on balance, generally."
    )
    for p in out.get("ranked_picks", []):
        p["rationale"] = (p.get("rationale", "") + filler * 3)
    return out


def _strip_input_groundedness(out: dict) -> dict:
    """Thinktank thesis: delete every citation of the provided inputs (board
    metrics, filings, news, weekly signal) — replace the input-referencing
    sections with generic boilerplate that could apply to any name. Stance/
    conviction/risks/catalysts untouched so only groundedness degrades.
    Targets `input_groundedness`."""
    out["business_summary"] = "The company operates in its industry and generates revenue."
    out["filings_review"] = "Recent filings were reviewed; nothing unusual stood out."
    out["news_sentiment"] = "News flow has been generally neutral to positive recently."
    out["valuation"] = "The stock trades at a reasonable valuation relative to peers."
    out["market_dynamics"] = "Current market conditions are broadly supportive."
    return out


def _vacuous_moat(out: dict) -> dict:
    """Thinktank thesis: replace the moat assessment with marketing language —
    no advantage mechanism, no erosion risk. Targets `moat_and_business_quality`."""
    out["moat"] = "Strong brand, great products, and a loyal customer base."
    return out


def _contradict_stance(out: dict) -> dict:
    """Thinktank thesis: leave the (bullish) body untouched but flip stance to
    `avoid` with conviction that doesn't follow from the text — stance no
    longer tracks the evidence. Targets `stance_consistency`."""
    out["stance"] = "avoid"
    out["conviction"] = 88
    return out


def _unearned_material_change(out: dict) -> dict:
    """Thinktank theme: flag material_change=True for a reconcile that
    actually just restates the prior view — the churn-discipline contract
    (a new version must EARN material_change) is violated. Targets
    `churn_discipline`."""
    out["material_change"] = True
    out["change_summary"] = "Reaffirming the prior view; no meaningful shift."
    return out


def _break_anchor_fidelity(out: dict) -> dict:
    """Thinktank theme: for a reconcile, silently rewrite the anchored view
    (flip stance) with no divergence acknowledgment in the narrative or
    change_summary — ignores the weekly anchor entirely. Targets
    `anchor_fidelity`."""
    out["stance"] = "underweight"
    out["narrative"] = (
        "Semiconductors are now underweight given near-term demand softness."
    )
    out["change_summary"] = ""
    return out


@dataclass(frozen=True)
class Corruption:
    name: str
    rubric: str
    target_dimension: str
    fn: Callable[[dict], dict]


CORRUPTIONS: list[Corruption] = [
    Corruption("strip_numerical_grounding", "eval_rubric_sector_quant",
               "numerical_grounding", _strip_numerical_grounding),
    Corruption("break_ranking_coherence", "eval_rubric_sector_quant",
               "ranking_coherence", _break_ranking_coherence),
    Corruption("flatten_signal_calibration", "eval_rubric_sector_quant",
               "signal_calibration", _flatten_signal_calibration),
    Corruption("gut_output_completeness", "eval_rubric_sector_quant",
               "output_completeness", _gut_output_completeness),
    Corruption("verbosity_pad_numerical", "eval_rubric_sector_quant",
               "numerical_grounding", _verbosity_pad),
    Corruption("strip_citation_grounding", "eval_rubric_sector_qual",
               "citation_grounding", _strip_citation_grounding),
    Corruption("flatten_reasoning_depth", "eval_rubric_sector_qual",
               "reasoning_depth", _flatten_reasoning_depth),
    Corruption("misalign_evidence", "eval_rubric_sector_qual",
               "evidence_alignment", _misalign_evidence),
    Corruption("strip_input_groundedness", "eval_rubric_thinktank_thesis",
               "input_groundedness", _strip_input_groundedness),
    Corruption("vacuous_moat", "eval_rubric_thinktank_thesis",
               "moat_and_business_quality", _vacuous_moat),
    Corruption("contradict_stance", "eval_rubric_thinktank_thesis",
               "stance_consistency", _contradict_stance),
    Corruption("unearned_material_change", "eval_rubric_thinktank_theme",
               "churn_discipline", _unearned_material_change),
    Corruption("break_anchor_fidelity", "eval_rubric_thinktank_theme",
               "anchor_fidelity", _break_anchor_fidelity),
]


# ── Battery runner ─────────────────────────────────────────────────────────


def build_artifact(agent_id: str, agent_output: dict,
                   input_data_snapshot: dict) -> DecisionArtifact:
    """Wrap an agent_output in a minimal judgeable DecisionArtifact.

    `evaluate_artifact` reads only `agent_id` (rubric resolution),
    `input_data_snapshot`, and `agent_output` — the rest is metadata.
    """
    return DecisionArtifact(
        run_id="perturbation-probe",
        timestamp="2026-05-09T00:00:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="synthetic-reference"),
        full_prompt_context=FullPromptContext(
            system_prompt="<perturbation fixture>",
            user_prompt="<perturbation fixture>",
        ),
        input_data_snapshot=input_data_snapshot,
        input_data_summary="perturbation reference fixture",
        agent_output=agent_output,
    )


def _default_judge(artifact: DecisionArtifact, *, judge_model: str,
                   api_key: Optional[str]) -> dict[str, int]:
    """Live-judge adapter: score an artifact → {dimension: score}."""
    ev = evaluate_artifact(artifact, judge_model=judge_model, api_key=api_key)
    return {d.dimension: d.score for d in ev.dimension_scores}


def run_perturbation_battery(
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    corruptions: Optional[list[Corruption]] = None,
    min_drop: int = 1,
    judge_fn: Optional[Callable[..., dict[str, int]]] = None,
) -> dict[str, Any]:
    """Run the perturbation battery and return a sensitivity report.

    For each corruption: judge the (cached) reference and the corrupted
    variant, then check the TARGETED dimension dropped by >= ``min_drop``.
    ``judge_fn`` is injectable so the harness logic is unit-testable
    without a live LLM; defaults to the live judge.
    """
    corruptions = corruptions if corruptions is not None else CORRUPTIONS
    judge_fn = judge_fn or _default_judge

    ref_cache: dict[str, dict[str, int]] = {}
    cases: list[dict[str, Any]] = []

    for c in corruptions:
        fix = REFERENCE_FIXTURES[c.rubric]
        agent_id = fix["agent_id"]
        snapshot = fix["input_data_snapshot"]

        if c.rubric not in ref_cache:
            ref_art = build_artifact(agent_id, copy.deepcopy(fix["agent_output"]), snapshot)
            ref_cache[c.rubric] = judge_fn(ref_art, judge_model=judge_model, api_key=api_key)
        ref_scores = ref_cache[c.rubric]

        corrupted_output = c.fn(copy.deepcopy(fix["agent_output"]))
        cor_art = build_artifact(agent_id, corrupted_output, snapshot)
        cor_scores = judge_fn(cor_art, judge_model=judge_model, api_key=api_key)

        ref_t = ref_scores.get(c.target_dimension)
        cor_t = cor_scores.get(c.target_dimension)
        drop = (ref_t - cor_t) if (ref_t is not None and cor_t is not None) else None
        caught = drop is not None and drop >= min_drop

        cases.append({
            "name": c.name,
            "rubric": c.rubric,
            "target_dimension": c.target_dimension,
            "ref_score": ref_t,
            "corrupted_score": cor_t,
            "drop": drop,
            "caught": caught,
            "ref_mean": round(mean(ref_scores.values()), 3) if ref_scores else None,
            "corrupted_mean": round(mean(cor_scores.values()), 3) if cor_scores else None,
        })

    n = len(cases)
    n_caught = sum(1 for x in cases if x["caught"])
    return {
        "judge_model": judge_model,
        "n": n,
        "n_caught": n_caught,
        "caught_rate": round(n_caught / n, 3) if n else 0.0,
        "cases": cases,
    }


def format_scorecard(report: dict[str, Any]) -> str:
    """One-glance markdown scorecard (weekly email / smoke output)."""
    lines = [
        "## Judge sensitivity (synthetic perturbation)",
        "",
        f"- Judge model: `{report['judge_model']}`",
        f"- Corruptions caught: **{report['n_caught']}/{report['n']}** "
        f"(targeted dimension dropped ≥1)",
        "",
        "| corruption | rubric | targeted dim | ref→corrupted | caught |",
        "|---|---|---|---|---|",
    ]
    for c in report["cases"]:
        rubric_short = c["rubric"].replace("eval_rubric_", "")
        arrow = f"{c['ref_score']}→{c['corrupted_score']}"
        lines.append(
            f"| {c['name']} | {rubric_short} | {c['target_dimension']} | "
            f"{arrow} | {'✅' if c['caught'] else '⚠️ MISSED'} |"
        )
    return "\n".join(lines)


def emit_perturbation_report(
    *,
    bucket: Optional[str] = None,
    s3_client: Any = None,
    report_date: Optional[str] = None,
    report: Optional[dict[str, Any]] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    judge_fn: Optional[Callable[..., dict[str, int]]] = None,
) -> dict[str, Any]:
    """Run (or accept a precomputed) perturbation battery and write the weekly
    sensitivity scorecard to
    ``_perturbation/_report/{date}/sensitivity.{json,md}`` plus stable
    ``latest/`` pointers — Phase B of config#752.

    This is the between-PR drift catcher: Phase A's paths-filtered CI gate
    catches per-PR judge regressions, but silent model/API drift between PRs
    only shows up when the battery is re-run on a cadence. Mirrors
    :func:`calibration_kappa.emit_calibration_report`: research owns the
    render (:func:`format_scorecard`), the backtester evaluator email embeds
    ``latest/sensitivity.md`` verbatim so it never has to date-walk.

    ``report`` may be injected precomputed (tests / re-emit); otherwise the
    battery is run live via :func:`run_perturbation_battery` (needs Anthropic
    access — hence a handler on the eval-judge image, not the no-LLM
    rolling-mean Lambda). ``judge_fn`` is injectable for hermetic tests.
    Written on EVERY run so the email always has a current artifact.
    """
    import boto3  # deferred — keep the module import-light for CI/unit callers

    bkt = bucket or _RESEARCH_BUCKET
    client = s3_client or boto3.client("s3")
    now = datetime.now(timezone.utc).isoformat()
    report_date = report_date or now[:10]

    if report is None:
        report = run_perturbation_battery(
            judge_model=judge_model, api_key=api_key, judge_fn=judge_fn,
        )
    report = {**report, "generated_at": now, "report_date": report_date}

    json_key = f"{_REPORT_PREFIX}{report_date}/sensitivity.json"
    md_key = f"{_REPORT_PREFIX}{report_date}/sensitivity.md"
    latest_json_key = f"{_REPORT_PREFIX}latest/sensitivity.json"
    latest_md_key = f"{_REPORT_PREFIX}latest/sensitivity.md"

    body = json.dumps(report, indent=2, default=str).encode("utf-8")
    md = format_scorecard(report).encode("utf-8")
    for key, payload, ctype in (
        (json_key, body, "application/json"),
        (md_key, md, "text/markdown"),
        (latest_json_key, body, "application/json"),
        (latest_md_key, md, "text/markdown"),
    ):
        client.put_object(Bucket=bkt, Key=key, Body=payload, ContentType=ctype)

    report["report_keys"] = [json_key, md_key, latest_json_key, latest_md_key]
    logger.info(
        "[perturbation] caught=%d/%d model=%s → s3://%s/%s",
        report["n_caught"], report["n"], report["judge_model"], bkt, json_key,
    )
    return report
