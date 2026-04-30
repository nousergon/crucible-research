"""
Dry-run support for the research Lambda.

Installs LLM-only function-level stubs plus side-effect suppression
(archive_writer, email_sender, archive instance methods) so the entire
LangGraph pipeline can be exercised end-to-end without paying for
Anthropic tokens or polluting prod artifacts.

Returned ``restore()`` callable MUST be invoked after the stub-pass.
Lambda containers are reused across invocations, and module-level
patches persist on warm starts; failing to restore would silently
contaminate the subsequent real pass.

Used by:
- ``lambda/handler.py`` auto-gate — default behavior, runs a stub pass
  before every real pass and halts on stub-pass failure.
- ``local/offline_stubs.py`` — re-exports the agent stubs so the
  ``local/run.py --stub-llm`` path keeps working.

Decision-artifact capture is force-disabled during stub-pass via the
``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` env var (saved + restored).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── LLM agent stubs ──────────────────────────────────────────────────────


def _stub_run_macro_agent_with_reflection(
    prior_report,
    prior_date,
    macro_data,
    max_iterations=2,
    api_key=None,
    **kwargs,
):
    logger.info("[dry-run] stub run_macro_agent_with_reflection")
    from config import ALL_SECTORS
    return {
        "report_md": "[DRY-RUN STUB] Macro environment is neutral. Fed on hold, yields stable.",
        "macro_json": {
            "market_regime": "neutral",
            "sector_modifiers": {s: 1.0 for s in ALL_SECTORS},
            "sector_ratings": {
                s: {"rating": "market_weight", "rationale": "Synthetic data"}
                for s in ALL_SECTORS
            },
        },
        "market_regime": "neutral",
        "sector_modifiers": {s: 1.0 for s in ALL_SECTORS},
        "sector_ratings": {
            s: {"rating": "market_weight", "rationale": "Synthetic data"}
            for s in ALL_SECTORS
        },
    }


def _stub_run_macro_agent(prior_report, prior_date, macro_data, api_key=None, **kwargs):
    return _stub_run_macro_agent_with_reflection(prior_report, prior_date, macro_data)


def _stub_run_quant_analyst(
    team_id,
    sector_tickers,
    market_regime,
    price_data,
    technical_scores,
    run_date,
    api_key=None,
    **kwargs,
):
    logger.info(
        "[dry-run] stub run_quant_analyst: %s (%d tickers)",
        team_id,
        len(sector_tickers),
    )
    rng = random.Random(hash(team_id))
    picks = []
    for t in sector_tickers[:10]:
        picks.append(
            {
                "ticker": t,
                "quant_score": rng.randint(45, 85),
                "rationale": f"[DRY-RUN] Synthetic quant score for {t}",
                "key_metrics": {
                    "rsi": rng.randint(30, 70),
                    "momentum_20d": round(rng.uniform(-5, 10), 1),
                },
            }
        )
    return {
        "team_id": team_id,
        "ranked_picks": picks,
        "tool_calls": [],
        "iterations": 0,
    }


def _stub_run_qual_analyst(
    team_id,
    quant_top5,
    prior_theses,
    market_regime,
    run_date,
    api_key=None,
    price_data=None,
    **kwargs,
):
    logger.info(
        "[dry-run] stub run_qual_analyst: %s (%d picks)", team_id, len(quant_top5)
    )
    rng = random.Random(hash(team_id) + 10)
    assessments = []
    for pick in quant_top5:
        assessments.append(
            {
                "ticker": pick.get("ticker", ""),
                "qual_score": rng.randint(45, 85),
                "bull_case": "[DRY-RUN] Strong fundamentals",
                "bear_case": "[DRY-RUN] Valuation risk",
                "catalysts": ["Earnings", "Product launch"],
                "risks": ["Competition", "Macro headwinds"],
            }
        )
    return {
        "team_id": team_id,
        "assessments": assessments,
        "additional_candidate": None,
        "tool_calls": [],
    }


def _stub_run_peer_review(
    team_id,
    quant_picks,
    qual_assessments,
    additional_candidate=None,
    technical_scores=None,
    market_regime="neutral",
    api_key=None,
    **kwargs,
):
    logger.info("[dry-run] stub run_peer_review: %s", team_id)
    recs = []
    for qa in (qual_assessments or [])[:3]:
        recs.append(
            {
                "ticker": qa.get("ticker", ""),
                "quant_score": 65,
                "qual_score": qa.get("qual_score", 60),
                "combined_score": 63,
                "bull_case": qa.get("bull_case", ""),
                "bear_case": qa.get("bear_case", ""),
                "catalysts": qa.get("catalysts", []),
                "conviction": "medium",
                "quant_rationale": "[DRY-RUN] Synthetic peer review",
                "team_id": team_id,
            }
        )
    return {"recommendations": recs, "peer_review_rationale": "[DRY-RUN] Synthetic review"}


def _stub_run_sector_team(team_id, ctx, **kwargs):
    """Mirror real sector-team flow including held-stock thesis-update path.

    The held-stock loop in the real ``run_sector_team`` is exactly where
    score_aggregator hard-fails surface; a stub that returned
    ``thesis_updates: {}`` would skip that code path. This stub:
      - Runs the synthetic quant/qual/peer chain for recommendations.
      - Iterates ``team_held`` and produces a thesis_update per ticker by
        carrying forward ``prior_theses[ticker]`` (the no-LLM preservation
        branch of the real code).
    """
    logger.info("[dry-run] stub run_sector_team: %s", team_id)
    from agents.sector_teams.team_config import get_team_tickers
    from agents.sector_teams.sector_team import _sector_team_inverse

    sector_tickers = get_team_tickers(team_id, ctx.scanner_universe, ctx.sector_map)

    if sector_tickers:
        quant = _stub_run_quant_analyst(
            team_id,
            sector_tickers,
            ctx.market_regime,
            ctx.price_data,
            ctx.technical_scores,
            ctx.run_date,
        )
        qual = _stub_run_qual_analyst(
            team_id,
            quant["ranked_picks"][:5],
            ctx.prior_theses,
            ctx.market_regime,
            ctx.run_date,
        )
        peer = _stub_run_peer_review(
            team_id, quant["ranked_picks"][:5], qual["assessments"]
        )
        recommendations = peer["recommendations"]
    else:
        quant, qual, peer = {}, {}, {}
        recommendations = []

    team_sector_set = {
        s for s, tid in _sector_team_inverse().items() if tid == team_id
    }
    team_held = [
        t for t in ctx.held_tickers if ctx.sector_map.get(t, "") in team_sector_set
    ]
    thesis_updates = {}
    for ticker in team_held:
        if ctx.prior_theses.get(ticker) is None:
            raise RuntimeError(
                f"Held ticker {ticker} has no prior_thesis in archive — "
                f"population/investment_thesis are out of sync."
            )
        thesis_updates[ticker] = {
            **ctx.prior_theses[ticker],
            "stale_days": ctx.prior_theses[ticker].get("stale_days", 0) + 1,
            "triggers": [],
            "last_updated": ctx.run_date,
        }

    return {
        "team_id": team_id,
        "recommendations": recommendations,
        "thesis_updates": thesis_updates,
        "quant_output": quant,
        "qual_output": qual,
        "peer_review_output": peer,
        "tool_calls": [],
    }


def _stub_run_cio(
    candidates,
    macro_context,
    sector_ratings,
    current_population,
    open_slots,
    exits,
    run_date,
    api_key=None,
    **kwargs,
):
    logger.info(
        "[dry-run] stub run_cio: %d candidates, %d open slots",
        len(candidates),
        open_slots,
    )
    decisions = []
    advanced = []
    entry_theses = {}
    for i, c in enumerate(candidates[:open_slots]):
        ticker = c.get("ticker", f"UNK{i}")
        decisions.append(
            {
                "ticker": ticker,
                "decision": "ADVANCE",
                "rationale": "[DRY-RUN] Synthetic CIO advance decision",
                "scores": {
                    "conviction": 3,
                    "macro_alignment": 3,
                    "portfolio_fit": 3,
                    "catalyst": 3,
                },
            }
        )
        advanced.append(ticker)
        entry_theses[ticker] = {
            "bull_case": "[DRY-RUN] Synthetic bull case",
            "bear_case": "[DRY-RUN] Synthetic bear case",
            "catalysts": ["Earnings"],
            "risks": ["Valuation"],
            "conviction": "medium",
            "conviction_rationale": "Synthetic",
            "score": c.get("combined_score", 60),
        }
    for c in candidates[open_slots:]:
        decisions.append(
            {
                "ticker": c.get("ticker", ""),
                "decision": "REJECT",
                "rationale": "[DRY-RUN] No open slots remaining",
            }
        )
    return {
        "decisions": decisions,
        "advanced_tickers": advanced,
        "entry_theses": entry_theses,
    }


# ── Side-effect suppressors ──────────────────────────────────────────────


def _noop_archive_writer(state):
    """No-op archive writer for stub-pass — leaves state unchanged."""
    logger.info("[dry-run] stub archive_writer: no-op")
    return {}


def _noop_email_sender(state):
    logger.info("[dry-run] stub email_sender: no-op")
    return {"email_sent": False}


# ── Stub installer with restore ──────────────────────────────────────────

# Module-level patches: replace the function attribute on its source module.
_AGENT_PATCHES: list[tuple[str, str, Any]] = [
    ("agents.macro_agent", "run_macro_agent_with_reflection", _stub_run_macro_agent_with_reflection),
    ("agents.macro_agent", "run_macro_agent", _stub_run_macro_agent),
    ("agents.sector_teams.quant_analyst", "run_quant_analyst", _stub_run_quant_analyst),
    ("agents.sector_teams.qual_analyst", "run_qual_analyst", _stub_run_qual_analyst),
    ("agents.sector_teams.peer_review", "run_peer_review", _stub_run_peer_review),
    ("agents.sector_teams.sector_team", "run_sector_team", _stub_run_sector_team),
    ("agents.investment_committee.ic_cio", "run_cio", _stub_run_cio),
]

# Late-bound name patches: the graph module imports these names from
# agent modules into its own namespace, so setattr on the source module
# alone leaves the graph's bound reference pointing at the original.
# These targets only take effect if the graph module is already in
# sys.modules — handler imports build_graph + create_initial_state
# before invoking the gate, so it always is.
_GRAPH_NAME_PATCHES: list[tuple[str, str, Any]] = [
    ("graph.research_graph", "run_macro_agent_with_reflection", _stub_run_macro_agent_with_reflection),
    ("graph.research_graph", "run_sector_team", _stub_run_sector_team),
    ("graph.research_graph", "run_cio", _stub_run_cio),
    ("graph.research_graph", "archive_writer", _noop_archive_writer),
    ("graph.research_graph", "email_sender", _noop_email_sender),
]


def install_dry_run_stubs(archive_manager: Any | None = None) -> Callable[[], None]:
    """Install LLM stubs + side-effect suppressors. Return a restore() callable.

    Args:
        archive_manager: optional ArchiveManager instance whose ``upload_db``
            and ``write_signals_json`` methods will be patched to no-ops for
            the stub-pass. Pass the same instance the real-pass will use; it
            gets fully restored before returning to the caller.

    Returns:
        ``restore`` — callable taking no args. MUST be invoked after the
        stub-pass to undo every patch. Failing to restore leaves
        contaminated module state on warm Lambda containers.

    CRITICAL CONTRACT — build_graph AFTER install:
        ``archive_writer`` and ``email_sender`` are wired via
        ``graph.add_node(name, fn)`` in ``build_graph()``. That captures the
        function reference at build time. If you call ``build_graph()`` BEFORE
        ``install_dry_run_stubs()``, the resulting graph holds the REAL
        archive_writer + email_sender references, and patching the module
        global afterwards has NO effect on the bound nodes — the real S3
        writes + email send will fire.

        LLM agent functions (run_macro_agent_with_reflection, run_sector_team,
        run_cio) are late-bound through wrapper nodes that resolve them from
        the module's globals at call time. Those pick up patches regardless
        of build order — but you should not rely on this asymmetry. Always
        ``build_graph()`` AFTER ``install_dry_run_stubs()``.

        Order:
            restore = install_dry_run_stubs(archive)
            try:
                graph = build_graph()             # captures stubbed bindings
                state = create_initial_state(...)
                graph.invoke(state)
            finally:
                restore()
    """
    import importlib
    import os
    import sys

    saved_originals: list[tuple[Any, str, Any]] = []
    saved_env: dict[str, str | None] = {}

    saved_env["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] = os.environ.get(
        "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"
    )
    os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] = "false"

    # Suppress LangSmith tracing during stub-pass so synthetic runs do not
    # land in the prod LangSmith project. Restored on exit so the real
    # pass traces normally. Local/run.py disables these for the whole
    # process; here we scope to the stub-pass only.
    for env_key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        saved_env[env_key] = os.environ.get(env_key)
        os.environ[env_key] = "false"

    for mod_path, attr, stub in _AGENT_PATCHES:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            logger.warning("[dry-run] could not import %s: %s", mod_path, e)
            continue
        if not hasattr(mod, attr):
            logger.warning("[dry-run] %s has no attribute %s", mod_path, attr)
            continue
        saved_originals.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, stub)

    for mod_path, attr, stub in _GRAPH_NAME_PATCHES:
        mod = sys.modules.get(mod_path)
        if mod is None:
            logger.warning(
                "[dry-run] %s not in sys.modules — late-bound patch skipped "
                "(import the module before installing stubs)",
                mod_path,
            )
            continue
        if not hasattr(mod, attr):
            logger.warning("[dry-run] %s has no attribute %s", mod_path, attr)
            continue
        saved_originals.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, stub)

    if archive_manager is not None:
        for method_name in ("upload_db", "write_signals_json"):
            if hasattr(archive_manager, method_name):
                saved_originals.append(
                    (archive_manager, method_name, getattr(archive_manager, method_name))
                )
                setattr(archive_manager, method_name, lambda *a, **kw: None)

    logger.info("[dry-run] stubs installed: %d patches saved", len(saved_originals))

    def restore() -> None:
        for target, attr, original in reversed(saved_originals):
            try:
                setattr(target, attr, original)
            except Exception as e:
                logger.warning(
                    "[dry-run] restore failed for %r.%s: %s", target, attr, e
                )
        for key, val in saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        logger.info("[dry-run] restored %d originals", len(saved_originals))

    return restore
