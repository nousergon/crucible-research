"""Pre-flight token-count audit for the prompt-caching rollout.

Counts the assembled cacheable-prefix for each LLM call site to confirm
whether the prefix clears the model-specific cache minimum before any
``cache_control`` marker is added. A prefix below the minimum silently
won't engage caching — no error, just ``cache_creation_input_tokens: 0``
forever — so this audit must run before merging any caching changes.

Model cache minimums come from ``krepis.cache_minimums`` (>= 0.20.0) --
they are NOT monotonic across generations and tier does not predict them
(Opus 5 is 512, Opus 4.7 is 2048, Haiku 4.5 is 4096 -- the highest, on the
tier where mechanical work routes). Never infer one; look it up.

Call sites measured:

* sector_quant_analyst  (Haiku-4-5, system + 6 tools) — caching SHIPPED
  in this PR via ``SystemMessage`` content-block ``cache_control``.

* sector_qual_analyst   (Haiku-4-5, system + 8 tools) — caching SHIPPED
  in this PR via the same pattern.

* peer_review (selection / rationale / addition)
* macro_agent (analyst / critic)
* eval_judge  (per-rubric)
  All three measured as "if we hoisted the stable rubric portion into a
  ``SystemMessage``, would it clear the threshold?". These are Phase 4
  follow-ups — they need ``alpha-engine-config`` prompt-template splits
  (stable rubric + volatile interpolated data) before caching can engage.

Usage::

    python scripts/measure_cache_prefixes.py [--json OUTFILE]

Outputs a markdown table to stdout. Optional ``--json`` writes the same
data to a JSON file for the PR body / ROADMAP follow-up sizing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Repo root on sys.path so ``from config import ...`` and
# ``from agents.prompt_loader import ...`` resolve when invoked as
# ``python scripts/measure_cache_prefixes.py``. Mirrors the pattern in
# scripts/smoke_eval_judge.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import anthropic  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402


class _KrepisCacheMinimums:
    """Mapping-shaped view over ``krepis.cache_minimums``.

    Keeps the existing ``_CACHE_MIN_BY_MODEL[model]`` call sites unchanged
    while removing the hand-maintained table behind them. Raises rather than
    returning a default for an unknown model -- this script's whole purpose
    is deciding whether a prefix clears the threshold, and guessing the
    threshold would defeat it.
    """

    def __getitem__(self, model: str) -> int:
        from krepis.cache_minimums import require_cache_minimum

        return require_cache_minimum(model)

    def get(self, model: str, default: int | None = None) -> int | None:
        from krepis.cache_minimums import cache_minimum

        value = cache_minimum(model)
        return default if value is None else value


# ── Model-specific cache minimums (tokens) ────────────────────────────────
# Sourced from ``krepis.cache_minimums`` (krepis >= 0.20.0), which ships the
# published per-model values as package data.
#
# This WAS a hand-maintained dict here, on the reasoning that a fixed,
# documented API constraint shouldn't drag in a config-repo dependency. The
# reasoning was sound -- this repo is public and the fleet registry is
# private -- but the copy drifted, and a 2026-07-27 audit found two of its
# four entries wrong: opus-4-7 as 4096 (actual 2048) and sonnet-4-6 as 2048
# (actual 1024).
#
# BOTH ERRED HIGH, which is the direction that costs silently: this script
# then reports "prefix does not clear the minimum" for prefixes that would
# in fact cache, and caching is declined with nothing to notice afterwards
# -- no marker, no error, no zero-hit-rate signal. The three Phase-4
# deferrals below (peer_review, macro_agent, eval_judge) are all Sonnet-4-6
# and were judged against the inflated 2048.
#
# krepis is a public, pip-installable package and already a dependency, so
# it satisfies the original constraint without the drift surface.
_CACHE_MIN_BY_MODEL = _KrepisCacheMinimums()


@dataclass
class PrefixMeasurement:
    """One call site's measured prefix tokens."""

    call_site: str
    model: str
    minimum_for_cache: int
    input_tokens: int
    n_tools: int
    system_chars: int
    user_chars: int
    clears_cache_minimum: bool
    notes: str = ""


def _format_lc_tools_for_anthropic(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain ``@tool`` objects to Anthropic tool-schema dicts.

    ``count_tokens`` accepts the same ``tools`` shape as ``messages.create``;
    we round-trip through ``convert_to_openai_tool`` and reshape because
    LangChain doesn't expose a direct Anthropic-format converter on the
    public utility surface.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        oa = convert_to_openai_tool(t)
        fn = oa["function"]
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object"}),
            }
        )
    return out


def _count(
    client: anthropic.Anthropic,
    *,
    model: str,
    system_text: str,
    user_text: str,
    tool_specs: list[dict[str, Any]],
) -> int:
    """Call ``messages.count_tokens`` against the assembled prefix."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_text or "x"}],
    }
    if system_text:
        kwargs["system"] = [{"type": "text", "text": system_text}]
    if tool_specs:
        kwargs["tools"] = tool_specs
    resp = client.messages.count_tokens(**kwargs)
    return int(resp.input_tokens)


# ── Per-call-site measurement functions ──────────────────────────────────


def measure_sector_quant(client: anthropic.Anthropic) -> PrefixMeasurement:
    """Quant analyst: system prompt + 6 quant tools, Haiku-4-5.

    Cacheable prefix = ``tools`` + ``system`` (LangGraph
    ``create_react_agent`` renders the system message first, before any
    user message). User content is the per-ticker volatile half — not
    part of the cacheable prefix.
    """
    from agents.prompt_loader import load_prompt
    from agents.sector_teams.quant_tools import create_quant_tools
    from agents.sector_teams.team_config import QUANT_TOP_N

    system_text = load_prompt("quant_analyst_system").format(
        team_title="Technology",
        universe_size=84,
        quant_top_n=QUANT_TOP_N,
        focus_str="momentum, value, quality",
        market_regime="neutral",
    )
    tools = create_quant_tools(
        {
            "price_data": {},
            "technical_scores": {},
            "market_regime": "neutral",
            "factor_blend_regime_weights": {},
            "focus_list_tickers": set(),
            "override_tickers": [],
        }
    )
    tool_specs = _format_lc_tools_for_anthropic(tools)
    model = "claude-haiku-4-5"
    tokens = _count(
        client,
        model=model,
        system_text=system_text,
        user_text="placeholder user message",
        tool_specs=tool_specs,
    )
    return PrefixMeasurement(
        call_site="sector_quant_analyst",
        model=model,
        minimum_for_cache=_CACHE_MIN_BY_MODEL[model],
        input_tokens=tokens,
        n_tools=len(tools),
        system_chars=len(system_text),
        user_chars=0,
        clears_cache_minimum=tokens >= _CACHE_MIN_BY_MODEL[model],
        notes="Caching SHIPPED in this PR via SystemMessage cache_control.",
    )


def measure_sector_qual(client: anthropic.Anthropic) -> PrefixMeasurement:
    """Qual analyst: system prompt + 8 qual tools, Haiku-4-5.

    Tests the PILLAR_EMIT_ENABLED variant when the flag is on (the larger
    of the two system prompts); falls back to the legacy variant.
    """
    from agents.prompt_loader import load_prompt
    from agents.sector_teams.qual_tools import create_qual_tools

    try:
        from config import PILLAR_EMIT_ENABLED
    except ImportError:
        PILLAR_EMIT_ENABLED = False

    prompt_name = "qual_analyst_system_pillars" if PILLAR_EMIT_ENABLED else "qual_analyst_system"
    system_text = load_prompt(prompt_name).format(
        team_title="Technology",
        n_picks=5,
        market_regime="neutral",
    )
    tools = create_qual_tools(
        {
            "prior_theses": {},
            "price_data": {},
            "episodic_memories": {},
            "semantic_memories": {},
        }
    )
    tool_specs = _format_lc_tools_for_anthropic(tools)
    model = "claude-haiku-4-5"
    tokens = _count(
        client,
        model=model,
        system_text=system_text,
        user_text="placeholder user message",
        tool_specs=tool_specs,
    )
    return PrefixMeasurement(
        call_site=f"sector_qual_analyst ({prompt_name})",
        model=model,
        minimum_for_cache=_CACHE_MIN_BY_MODEL[model],
        input_tokens=tokens,
        n_tools=len(tools),
        system_chars=len(system_text),
        user_chars=0,
        clears_cache_minimum=tokens >= _CACHE_MIN_BY_MODEL[model],
        notes="Caching SHIPPED in this PR via SystemMessage cache_control.",
    )


def measure_peer_review_selection(
    client: anthropic.Anthropic,
) -> PrefixMeasurement:
    """Joint-selection rubric. Currently rendered into a single
    ``HumanMessage`` — Phase 4 must split rubric (stable) from candidates
    (volatile) in ``alpha-engine-config``.

    Measures the WHOLE rendered prompt as an upper bound on the
    cacheable portion. The actual Phase 4 cacheable prefix will be
    smaller (rubric scaffolding minus interpolated candidates_text), so
    if this measurement falls below the threshold the call site is not
    a viable caching target even after the split.
    """
    from agents.prompt_loader import load_prompt
    from config import TEAM_PICKS_PER_RUN

    rendered = load_prompt("peer_review_joint_selection").format(
        team_title="Technology",
        market_regime="neutral",
        candidates_text="(volatile per-call — measurement uses placeholder)",
        team_picks_per_run=TEAM_PICKS_PER_RUN,
    )
    model = "claude-haiku-4-5"
    tokens = _count(
        client,
        model=model,
        system_text="",
        user_text=rendered,
        tool_specs=[],
    )
    return PrefixMeasurement(
        call_site="peer_review_joint_selection (Phase 4)",
        model=model,
        minimum_for_cache=_CACHE_MIN_BY_MODEL[model],
        input_tokens=tokens,
        n_tools=0,
        system_chars=0,
        user_chars=len(rendered),
        clears_cache_minimum=tokens >= _CACHE_MIN_BY_MODEL[model],
        notes=(
            "Upper bound: whole rendered prompt. Phase 4 cacheable prefix "
            "is smaller after splitting stable rubric from volatile candidates."
        ),
    )


def measure_macro_analyst(client: anthropic.Anthropic) -> PrefixMeasurement:
    """Macro economist analyst pass. Sonnet-4-6 (2048-token minimum)."""
    from agents.prompt_loader import load_prompt
    from config import ALL_SECTORS

    rendered = load_prompt("macro_agent").format(
        sector_list_text="\n".join(f"- {s}" for s in ALL_SECTORS),
        prior_date="2026-05-18",
        prior_report="(volatile per-call — placeholder)",
        regime_substrate="(volatile per-call — placeholder)",
        fed_funds="4.5",
        t2yr="4.8",
        t10yr="4.4",
        curve_slope="-40",
        vix="18",
        spy_30d="2.0",
        qqq_30d="3.0",
        iwm_30d="1.0",
        oil="78.0",
        gold="2400",
        copper="4.5",
        cpi_yoy="3.0",
        unemployment="4.0",
        consumer_sentiment="65",
        initial_claims="220000",
        hy_oas="350",
        pct_above_50d="55",
        pct_above_200d="60",
        adv_dec_ratio="1.1",
        upcoming_releases="(volatile)",
    )
    model = "claude-sonnet-4-6"
    tokens = _count(
        client,
        model=model,
        system_text="",
        user_text=rendered,
        tool_specs=[],
    )
    return PrefixMeasurement(
        call_site="macro_agent_analyst (Phase 4)",
        model=model,
        minimum_for_cache=_CACHE_MIN_BY_MODEL[model],
        input_tokens=tokens,
        n_tools=0,
        system_chars=0,
        user_chars=len(rendered),
        clears_cache_minimum=tokens >= _CACHE_MIN_BY_MODEL[model],
        notes=(
            "Upper bound: whole rendered prompt. Phase 4 cacheable prefix "
            "is smaller after splitting stable rubric from volatile macro data."
        ),
    )


def measure_eval_judge(
    client: anthropic.Anthropic,
    rubric_name: str,
) -> PrefixMeasurement:
    """One eval-judge rubric. Default judge model is Haiku-4-5."""
    from agents.prompt_loader import load_prompt

    rendered = load_prompt(rubric_name).format(
        agent_input="(volatile per-artifact — placeholder)",
        agent_output="(volatile per-artifact — placeholder)",
    )
    model = "claude-haiku-4-5"
    tokens = _count(
        client,
        model=model,
        system_text="",
        user_text=rendered,
        tool_specs=[],
    )
    return PrefixMeasurement(
        call_site=f"eval_judge / {rubric_name} (Phase 4)",
        model=model,
        minimum_for_cache=_CACHE_MIN_BY_MODEL[model],
        input_tokens=tokens,
        n_tools=0,
        system_chars=0,
        user_chars=len(rendered),
        clears_cache_minimum=tokens >= _CACHE_MIN_BY_MODEL[model],
        notes=(
            "Upper bound: whole rendered rubric. Phase 4 cacheable prefix "
            "is smaller after splitting stable rubric from volatile artifact data."
        ),
    )


# ── Driver ───────────────────────────────────────────────────────────────


_EVAL_RUBRICS = [
    "eval_rubric_sector_quant",
    "eval_rubric_sector_qual",
    "eval_rubric_sector_peer_review",
    "eval_rubric_macro_economist",
    "eval_rubric_ic_cio",
    "eval_rubric_thesis_update",
]


def _try_measure(label: str, fn) -> PrefixMeasurement | None:
    """Wrap a measurement so a missing prompt file doesn't kill the whole
    audit — config repo may not have every rubric checked out locally."""
    try:
        return fn()
    except FileNotFoundError as e:
        print(f"  [skip] {label}: prompt file not found ({e})", file=sys.stderr)
        return None
    except Exception as e:
        print(
            f"  [error] {label}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional JSON output path for PR-body / ROADMAP sizing.",
    )
    args = parser.parse_args()

    # API key from the standard SSM-backed loader (alpha-engine-lib).
    from config import ANTHROPIC_API_KEY

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    results: list[PrefixMeasurement] = []

    # In-scope-for-this-PR call sites.
    for label, fn in [
        ("sector_quant_analyst", lambda: measure_sector_quant(client)),
        ("sector_qual_analyst", lambda: measure_sector_qual(client)),
    ]:
        m = _try_measure(label, fn)
        if m is not None:
            results.append(m)

    # Phase 4 (deferred) — measured now for sizing.
    for label, fn in [
        ("peer_review_joint_selection", lambda: measure_peer_review_selection(client)),
        ("macro_agent_analyst", lambda: measure_macro_analyst(client)),
    ]:
        m = _try_measure(label, fn)
        if m is not None:
            results.append(m)

    for rubric_name in _EVAL_RUBRICS:
        m = _try_measure(
            f"eval_judge:{rubric_name}",
            lambda rn=rubric_name: measure_eval_judge(client, rn),
        )
        if m is not None:
            results.append(m)

    # Markdown report
    print()
    print("| Call site | Model | Tokens | Min | Clears? | Tools | Notes |")
    print("|---|---|---:|---:|:---:|---:|---|")
    for r in results:
        status = "✓" if r.clears_cache_minimum else "✗"
        print(
            f"| {r.call_site} | {r.model} | {r.input_tokens} | "
            f"{r.minimum_for_cache} | {status} | {r.n_tools} | {r.notes} |"
        )
    print()

    n_clear = sum(1 for r in results if r.clears_cache_minimum)
    n_below = len(results) - n_clear
    print(
        f"Summary: {n_clear}/{len(results)} prefixes clear the cache minimum; "
        f"{n_below} fall below and would silently NOT engage caching.",
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nJSON written to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
