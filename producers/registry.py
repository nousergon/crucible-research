"""Research producer spec registry (config#1221 / ARCHITECTURE §37).

Champion = the live agentic LangGraph producer (authoritative; emitted by the
research graph, ``build`` is None here). Challengers carry a ``build`` callable
``(run_date, archive_manager, **ctx) -> signals_payload`` that produces a
conforming signals.json from the SAME scanner candidate set (scanner held
constant across producers — a clean selection-only comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from producers.no_agent import run_no_agent_producer
from producers.single_agent import run_single_agent_producer


@dataclass(frozen=True)
class ProducerSpec:
    name: str
    kind: str  # "champion" | "challenger"
    version: str
    description: str
    build: Callable | None = None  # None for the champion (live agentic graph)


RESEARCH_PRODUCERS: dict[str, ProducerSpec] = {
    "agentic_sector_teams": ProducerSpec(
        name="agentic_sector_teams",
        kind="champion",
        version="v1",
        description="live LangGraph 6-sector-team + macro economist + CIO",
        build=None,
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
