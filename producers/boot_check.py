"""Deterministic boot check for the LIVE research producer path.

This is what the deploy canary exercises (``infrastructure/deploy.sh`` →
``{"dry_run_llm": true}`` → ``lambda/handler.py``). It replaces the boot
validation that ran ``graph.research_graph.build_graph()`` until
alpha-engine-config-I7827.

**Why it was replaced.** The graph was retired on 2026-07-12
(``producers/registry.py``: ``agentic_sector_teams``, ``kind="retired"``) and
the weekly SF stopped invoking it long before that, so the deploy's own safety
check was smoke-testing a code path nothing in production runs — and was NOT
smoke-testing the path that does. A deploy that broke ``producers/`` passed the
canary green. Measured 2026-08-20, alpha-engine-config-I7827 deliverable 2b.

**What it exercises.** The two things a deploy can break in the live path:

1. *Import/wiring* — every producer the live ``ChallengerShadow`` stage will
   build (derived from ``producers/registry.py``, not hardcoded, so a
   newly-registered arm is covered the day it lands) plus ``producers.runner``
   itself. This is the 2026-05-06 RAG-import class of bug, moved onto the live
   modules.
2. *Payload assembly* — ``scoring.signals_payload._build_signals_payload``,
   the chokepoint EVERY producer routes its signals.json through, driven by a
   synthetic in-memory state. Exercises the five ``config.py`` gate constants
   it reads, which a config edit can break without touching any producer.

**What it must never do**, and does not: call an LLM, write to S3, touch the
DB, send email, or read the wall clock. Every input below is a literal in this
file; nothing here performs I/O. The handler additionally runs this under
``dry_run.install_dry_run_stubs`` so even an accidental future I/O call is
suppressed — belt and braces, matching the guarantee the ``dry_run_llm`` flag
has carried since the 2026-05-04 misfire (a canary that produced a real
``signals.json`` and research email outside the Saturday cadence).
"""

from __future__ import annotations

import importlib
import logging

from producers.registry import buildable_challenger_producers

logger = logging.getLogger(__name__)


class LiveProducerBootCheckError(RuntimeError):
    """The live producer path failed its deterministic boot validation."""


# Synthetic, in-memory only. Ticker names are deliberately non-tradeable
# sentinels so a stray write (there is none) could never be mistaken for a
# real signal set.
_SMOKE_TICKERS = ("__CANARY_A", "__CANARY_B")
_SMOKE_SECTOR_MAP = {"__CANARY_A": "Technology", "__CANARY_B": "Healthcare"}


def _smoke_state() -> dict:
    """A minimal, fully synthetic ``ResearchState``-shaped dict.

    Shaped like what the live producers hand ``_build_signals_payload`` (see
    ``producers/no_agent.py::build_no_agent_signals``), but it is a FIXTURE,
    not a copy of producer logic — it deliberately does not re-implement any
    scoring so it cannot drift into a second, silently-diverging
    implementation of the thing it is validating.
    """
    theses = {
        "__CANARY_A": {
            "ticker": "__CANARY_A",
            "rating": "BUY",
            "score": 75.0,
            "final_score": 75.0,
            "quant_score": 75.0,
            "qual_score": None,
            "conviction": "rising",
            "sector": "Technology",
            "bull_case": "",
        },
        "__CANARY_B": {
            "ticker": "__CANARY_B",
            "rating": "HOLD",
            "score": 40.0,
            "final_score": 40.0,
            "quant_score": 40.0,
            "qual_score": None,
            "conviction": "stable",
            "sector": "Healthcare",
            "bull_case": "",
        },
    }
    return {
        "investment_theses": theses,
        "prior_theses": {},
        "new_population": [
            {
                "ticker": "__CANARY_A",
                "sector": "Technology",
                "long_term_rating": "BUY",
                "long_term_score": 75.0,
                "conviction": "rising",
                "price_target_upside": None,
            }
        ],
        "sector_map": dict(_SMOKE_SECTOR_MAP),
        "sector_ratings": {},
        "sector_modifiers": {},
        "entry_theses": {},
        "advanced_tickers": ["__CANARY_A"],
        "exits": [],
        "quarantined": [],
        "run_date": "1970-01-01",
        "run_time": "1970-01-01T00:00:00+00:00",
        "market_regime": "neutral",
    }


def run_live_producer_boot_check() -> dict:
    """Validate the live producer path. Raise on any defect; return evidence.

    Returns a dict naming exactly what was exercised, so the canary's log line
    and the deploy's stdout say which producers a green result covers — a
    check that reports only "OK" cannot be told apart from a check that
    silently covered nothing (``principles.md`` §2.7).
    """
    specs = buildable_challenger_producers()
    if not specs:
        raise LiveProducerBootCheckError(
            "no buildable challenger producer is registered — the live "
            "ChallengerShadow stage would emit nothing. See "
            "producers/registry.py::RESEARCH_PRODUCERS."
        )

    retired = sorted(s.name for s in specs if s.kind == "retired")
    if retired:
        raise LiveProducerBootCheckError(
            f"retired producer(s) {retired} are on the live build path — "
            "champion-challenger-policy.md §6: a retired arm's code is "
            "deleted, not left reachable (alpha-engine-config-I7827)."
        )

    modules: list[str] = []
    for spec in specs:
        if not callable(spec.build):
            raise LiveProducerBootCheckError(
                f"producer {spec.name!r} is in the buildable set but its "
                "`build` is not callable — producers/registry.py"
            )
        module_name = spec.build.__module__
        importlib.import_module(module_name)
        modules.append(module_name)

    # The stage entry point itself, and the shadow-gap doctrine it enforces.
    importlib.import_module("producers.runner")

    from scoring.signals_payload import _build_signals_payload

    payload = _build_signals_payload(_smoke_state())
    for key in ("signals", "population", "universe", "buy_candidates"):
        if key not in payload:
            raise LiveProducerBootCheckError(
                f"signals payload assembly produced no {key!r} key — "
                "scoring/signals_payload.py or the config gate constants it "
                f"reads are broken (got keys: {sorted(payload)})"
            )

    result = {
        "producers": sorted(s.name for s in specs),
        "modules": sorted(set(modules)),
        "payload_keys": sorted(payload),
    }
    logger.info(
        "live producer boot check OK: producers=%s modules=%s",
        result["producers"],
        result["modules"],
    )
    return result
