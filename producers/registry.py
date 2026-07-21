"""Research producer spec registry (config#1221 / ARCHITECTURE §37).

Champion = the live agentic LangGraph producer (authoritative; emitted by the
research graph, ``build`` is None here). Challengers carry a ``build`` callable
``(run_date, archive_manager, **ctx) -> signals_payload`` that produces a
conforming signals.json from the SAME scanner candidate set (scanner held
constant across producers — a clean selection-only comparison).

A spec may instead be ``kind="retired"`` once it is no longer wired into the
live pipeline — this makes liveness a queryable fact (``retired_date``)
instead of a stale ``description`` string a downstream reader (e.g. the
evaluator/backtester's e2e_lift aggregation) has to re-derive by inference.
See ``agentic_sector_teams`` below (config-I2993).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from producers.no_agent import run_no_agent_producer
from producers.single_agent import run_single_agent_producer


@dataclass(frozen=True)
class ProducerSpec:
    name: str
    kind: str  # "champion" | "challenger" | "retired"
    version: str
    description: str
    build: Callable | None = None  # None for the champion (live agentic graph)
    # ISO date (str) a spec stopped being the live producer, or None while
    # still champion/challenger. Additive field (config-I2993) — defaults to
    # None so every pre-existing spec is unaffected.
    retired_date: str | None = None


RESEARCH_PRODUCERS: dict[str, ProducerSpec] = {
    "agentic_sector_teams": ProducerSpec(
        name="agentic_sector_teams",
        kind="retired",
        version="v1",
        description="RETIRED six-team + macro economist + CIO LangGraph "
        "orchestration — the live ne-weekly-freshness-pipeline SF no longer "
        "invokes this graph (config#1580 ruling); SignalsEnvelope "
        "(config-I2515 Phase B) is the live signals.json producer now.",
        build=None,
        # Derivation (config-I2993): research.db team_candidates/cio_evaluations
        # MAX date is trading_day 2026-07-10, i.e. the graph's last production
        # was the Saturday 2026-07-11 weekly cycle (calendar_date; weekly runs
        # tag trading_day = last closed trading day per the dual-track date
        # convention). retired_date is the first calendar date after that last
        # production — mirrors the backtester's day-after-last-use
        # ``cutover_date`` convention (e.g. neutralization_live_forward_ic,
        # cutover 2026-06-22).
        retired_date="2026-07-12",
    ),
    "no_agent_quant": ProducerSpec(
        name="no_agent_quant",
        kind="challenger",
        version="v1",
        description="pure-quant floor: scanner candidates scored by the technical "
        "composite, deterministic top-N ENTER gate, no LLM (config#1221)",
        build=run_no_agent_producer,
    ),
    "single_agent_quant": ProducerSpec(
        name="single_agent_quant",
        kind="challenger",
        version="v1",
        description="single-agent: ONE Sonnet call assesses qual for all scanner "
        "candidates; deterministic quant + composite; no multi-agent fan-out, no "
        "macro/CIO (config#1223 / M3 baseline)",
        build=run_single_agent_producer,
    ),
}


def challenger_producers() -> list[ProducerSpec]:
    return [p for p in RESEARCH_PRODUCERS.values() if p.kind == "challenger"]


def champion_producer() -> ProducerSpec | None:
    """The live ``kind=="champion"`` producer, or ``None`` when no spec is
    currently registered as champion (config-I2993: retiring a spec does not
    auto-promote a successor — registering the live producer as a new champion
    spec is tracked separately). Callers MUST treat ``None`` as a legitimate,
    non-error state, not an invariant violation."""
    return next((p for p in RESEARCH_PRODUCERS.values() if p.kind == "champion"), None)
