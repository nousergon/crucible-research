"""
Offline stubs — monkey-patches all external calls (LLM, data APIs, S3, email)
so the research pipeline can run end-to-end without any network access.

Usage:
    from local.offline_stubs import install_offline_stubs
    install_offline_stubs()   # call BEFORE importing graph modules

Generates synthetic but structurally valid data so every graph node
receives the dict shapes it expects.

NOTE 2026-04-30: The LLM-only agent stub functions (``_stub_run_*``) are
duplicated in ``/dry_run.py`` at repo root, which is the Lambda-importable
copy. Keep them in sync. Single-source consolidation is tracked as a
follow-up item; this PR keeps two copies to avoid scope creep on the
Lambda dry-run gate landing.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Synthetic price data ─────────────────────────────────────────────────────

_SAMPLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH",
    "JNJ", "PG", "HD", "MA", "XOM", "ABBV", "LLY", "COST", "MRK", "PEP",
    "CVX", "KO", "AVGO", "TMO", "WMT", "MCD", "CSCO", "ACN", "ABT", "CRM",
]

_SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication Services",
    "AMZN": "Consumer Discretionary", "NVDA": "Technology", "META": "Communication Services",
    "TSLA": "Consumer Discretionary", "JPM": "Financial", "V": "Financial", "UNH": "Healthcare",
    "JNJ": "Healthcare", "PG": "Consumer Staples", "HD": "Consumer Discretionary",
    "MA": "Financial", "XOM": "Energy", "ABBV": "Healthcare", "LLY": "Healthcare",
    "COST": "Consumer Staples", "MRK": "Healthcare", "PEP": "Consumer Staples",
    "CVX": "Energy", "KO": "Consumer Staples", "AVGO": "Technology", "TMO": "Healthcare",
    "WMT": "Consumer Staples", "MCD": "Consumer Discretionary", "CSCO": "Technology",
    "ACN": "Technology", "ABT": "Healthcare", "CRM": "Technology",
}


def _synthetic_ohlcv(ticker: str, days: int = 252) -> pd.DataFrame:
    """Generate a synthetic 1-year daily OHLCV DataFrame."""
    rng = np.random.default_rng(hash(ticker) % (2**31))
    dates = pd.bdate_range(end=datetime.now(), periods=days)
    base = rng.uniform(50, 500)
    returns = rng.normal(0.0005, 0.015, size=days)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.02, size=days))
    low = close * (1 - rng.uniform(0, 0.02, size=days))
    opn = low + (high - low) * rng.uniform(0.3, 0.7, size=days)
    volume = rng.integers(500_000, 20_000_000, size=days)
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)


# ── Stub functions ───────────────────────────────────────────────────────────

def _stub_fetch_price_data(tickers, period="1y"):
    logger.info("[offline] stub fetch_price_data: %d tickers", len(tickers))
    return {t: _synthetic_ohlcv(t) for t in tickers}


def _stub_fetch_sp500_sp400():
    logger.info("[offline] stub fetch_sp500_sp400_with_sectors: %d tickers", len(_SAMPLE_TICKERS))
    return list(_SAMPLE_TICKERS), dict(_SECTOR_MAP)


def _stub_fetch_short_interest(tickers):
    logger.info("[offline] stub fetch_short_interest: %d tickers", len(tickers))
    rng = random.Random(42)
    return {t: {"short_pct_float": rng.uniform(1, 15), "short_ratio": rng.uniform(1, 5)} for t in tickers}


def _stub_fetch_all_news(ticker, hours=48):
    logger.info("[offline] stub fetch_all_news: %s", ticker)
    return {
        "yahoo": [
            {"headline": f"[STUB] {ticker} reports strong quarterly results",
             "source": "Yahoo Finance", "published": datetime.now().isoformat(),
             "url": "https://example.com", "excerpt": f"Synthetic news article for {ticker} dry run."},
            {"headline": f"[STUB] Analysts upgrade {ticker} outlook",
             "source": "Yahoo Finance", "published": datetime.now().isoformat(),
             "url": "https://example.com", "excerpt": f"Synthetic upgrade article for {ticker}."},
        ],
        "edgar_8k": [],
    }


def _stub_fetch_analyst_consensus(ticker, current_price=None):
    logger.info("[offline] stub fetch_analyst_consensus: %s", ticker)
    rng = random.Random(hash(ticker))
    price = current_price or rng.uniform(50, 500)
    target = price * rng.uniform(0.9, 1.3)
    return {
        "ticker": ticker,
        "consensus_rating": rng.choice(["Strong Buy", "Buy", "Hold"]),
        "num_analysts": rng.randint(5, 30),
        "mean_target": round(target, 2),
        "current_price": round(price, 2),
        "upside_pct": round((target / price - 1) * 100, 1),
        "rating_changes": "None recent",
        "earnings_surprise": "+2.5%",
    }


def _stub_fetch_macro_data():
    logger.info("[offline] stub fetch_macro_data")
    return {
        "fed_funds": 5.25, "t2yr": 4.60, "t10yr": 4.20,
        "curve_slope": -40, "vix": 16.5,
        "spy_30d": 2.1, "qqq_30d": 3.4, "iwm_30d": 1.2,
        "oil": 78.50, "gold": 2350.0, "copper": 4.25,
        "cpi_yoy": 3.1, "unemployment": 3.9,
        "consumer_sentiment": 67.8, "initial_claims": 215,
        "hy_oas": 340,
        "upcoming_releases": "CPI (next week), FOMC (in 2 weeks)",
    }


def _stub_compute_market_breadth(price_data):
    return {"pct_above_50d": 62.5, "pct_above_200d": 58.0, "adv_dec_ratio": 1.3}


def _stub_fetch_revisions(tickers, reference_date=None):
    logger.info("[offline] stub fetch_revisions: %d tickers", len(tickers))
    return {}


def _stub_fetch_options_signals(tickers, reference_date=None):
    logger.info("[offline] stub fetch_options_signals: %d tickers", len(tickers))
    return {}


def _stub_cache_options_to_s3(data, date):
    pass


def _stub_fetch_insider_activity(tickers, lookback_days=90, reference_date=None):
    logger.info("[offline] stub fetch_insider_activity: %d tickers", len(tickers))
    return {}


def _stub_cache_insider_to_s3(data, date):
    pass


def _stub_fetch_institutional_accumulation(tickers):
    logger.info("[offline] stub fetch_institutional_accumulation: %d tickers", len(tickers))
    return {}


# ── LLM agent stubs ─────────────────────────────────────────────────────────

def _stub_run_macro_agent_with_reflection(prior_report, prior_date, macro_data,
                                          max_iterations=2, api_key=None, **kwargs):
    logger.info("[offline] stub run_macro_agent_with_reflection")
    from config import ALL_SECTORS
    return {
        "report_md": "[OFFLINE STUB] Macro environment is neutral. Fed on hold, yields stable.",
        "macro_json": {
            "market_regime": "neutral",
            "sector_modifiers": {s: 1.0 for s in ALL_SECTORS},
            "sector_ratings": {s: {"rating": "market_weight", "rationale": "Synthetic data"} for s in ALL_SECTORS},
        },
        "market_regime": "neutral",
        "sector_modifiers": {s: 1.0 for s in ALL_SECTORS},
        "sector_ratings": {s: {"rating": "market_weight", "rationale": "Synthetic data"} for s in ALL_SECTORS},
    }


def _stub_run_macro_agent(prior_report, prior_date, macro_data, api_key=None, **kwargs):
    return _stub_run_macro_agent_with_reflection(prior_report, prior_date, macro_data)


# ── Sector team / CIO stubs ──────────────────────────────────────────────

def _stub_run_quant_analyst(team_id, sector_tickers, market_regime, price_data,
                            technical_scores, run_date, api_key=None, **kwargs):
    logger.info("[offline] stub run_quant_analyst: %s (%d tickers)", team_id, len(sector_tickers))
    rng = random.Random(hash(team_id))
    picks = []
    for i, t in enumerate(sector_tickers[:10]):
        picks.append({
            "ticker": t,
            "quant_score": rng.randint(45, 85),
            "rationale": f"[OFFLINE] Synthetic quant score for {t}",
            "key_metrics": {"rsi": rng.randint(30, 70), "momentum_20d": round(rng.uniform(-5, 10), 1)},
        })
    return {"team_id": team_id, "ranked_picks": picks, "tool_calls": [], "iterations": 0}


def _stub_run_qual_analyst(team_id, quant_top5, prior_theses, market_regime,
                           run_date, api_key=None, price_data=None, **kwargs):
    logger.info("[offline] stub run_qual_analyst: %s (%d picks)", team_id, len(quant_top5))
    rng = random.Random(hash(team_id) + 10)
    assessments = []
    for pick in quant_top5:
        assessments.append({
            "ticker": pick.get("ticker", ""),
            "qual_score": rng.randint(45, 85),
            "bull_case": "[OFFLINE] Strong fundamentals",
            "bear_case": "[OFFLINE] Valuation risk",
            "catalysts": ["Earnings", "Product launch"],
            "risks": ["Competition", "Macro headwinds"],
        })
    return {"team_id": team_id, "assessments": assessments, "additional_candidate": None, "tool_calls": []}


def _stub_run_peer_review(team_id, quant_picks, qual_assessments,
                          additional_candidate=None, technical_scores=None,
                          market_regime="neutral", api_key=None, **kwargs):
    logger.info("[offline] stub run_peer_review: %s", team_id)
    recs = []
    for qa in (qual_assessments or [])[:3]:
        recs.append({
            "ticker": qa.get("ticker", ""),
            "quant_score": 65,
            "qual_score": qa.get("qual_score", 60),
            "combined_score": 63,
            "bull_case": qa.get("bull_case", ""),
            "bear_case": qa.get("bear_case", ""),
            "catalysts": qa.get("catalysts", []),
            "conviction": "medium",
            "quant_rationale": "[OFFLINE] Synthetic peer review",
            "team_id": team_id,
        })
    return {"recommendations": recs, "peer_review_rationale": "[OFFLINE] Synthetic review"}


def _stub_run_sector_team(team_id, ctx, **kwargs):
    """Stub sector team — mirrors the real held-stock thesis update path.

    The real ``run_sector_team`` runs quant + qual + peer_review (LLM) for
    sector picks AND iterates ``team_held`` to produce ``thesis_updates``.
    The held loop is exactly where score_aggregator hard-fails surface.
    A stub that returns ``thesis_updates: {}`` would skip that code path
    entirely — exactly the path we need to exercise in dry-run debugging.

    This stub:
      - Runs the (synthetic) quant/qual/peer chain for recommendations.
      - Iterates ``team_held`` and produces a thesis_update per ticker by
        carrying forward ``prior_theses[ticker]`` (the no-LLM preservation
        branch of real code) — this exercises score_aggregator with the
        real prior_thesis shape.
    """
    logger.info("[offline] stub run_sector_team: %s", team_id)
    from agents.sector_teams.team_config import get_team_tickers
    from agents.sector_teams.sector_team import _sector_team_inverse
    scanner_universe = ctx.scanner_universe
    sector_map = ctx.sector_map
    price_data = ctx.price_data
    technical_scores = ctx.technical_scores
    market_regime = ctx.market_regime
    prior_theses = ctx.prior_theses
    held_tickers = ctx.held_tickers
    run_date = ctx.run_date
    sector_tickers = get_team_tickers(team_id, scanner_universe, sector_map)

    # Recommendations path (synthetic, since LLM is stubbed)
    if sector_tickers:
        quant = _stub_run_quant_analyst(team_id, sector_tickers, market_regime, price_data, technical_scores, run_date)
        qual = _stub_run_qual_analyst(team_id, quant["ranked_picks"][:5], prior_theses, market_regime, run_date)
        peer = _stub_run_peer_review(team_id, quant["ranked_picks"][:5], qual["assessments"])
        recommendations = peer["recommendations"]
    else:
        quant, qual, peer = {}, {}, {}
        recommendations = []

    # Held-stock thesis_updates — mirror the real no-trigger preservation
    # branch in agents.sector_teams.sector_team.run_sector_team. This is
    # the load-bearing path for score_aggregator validation.
    team_sector_set = {s for s, tid in _sector_team_inverse().items() if tid == team_id}
    team_held = [t for t in held_tickers if sector_map.get(t, "") in team_sector_set]
    thesis_updates = {}
    for ticker in team_held:
        if prior_theses.get(ticker) is None:
            # Same upstream guard as the real sector_team — fail loudly so
            # population/investment_thesis sync drift surfaces in dry-run.
            raise RuntimeError(
                f"Held ticker {ticker} has no prior_thesis in archive — "
                f"population/investment_thesis are out of sync."
            )
        thesis_updates[ticker] = {
            **prior_theses[ticker],
            "stale_days": prior_theses[ticker].get("stale_days", 0) + 1,
            "triggers": [],
            "last_updated": run_date,
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


def _stub_run_cio(candidates, macro_context, sector_ratings, current_population,
                  open_slots, exits, run_date, api_key=None, **kwargs):
    logger.info("[offline] stub run_cio: %d candidates, %d open slots", len(candidates), open_slots)
    decisions = []
    advanced = []
    entry_theses = {}
    for i, c in enumerate(candidates[:open_slots]):
        ticker = c.get("ticker", f"UNK{i}")
        decisions.append({
            "ticker": ticker,
            "decision": "ADVANCE",
            "rationale": "[OFFLINE] Synthetic CIO advance decision",
            "scores": {"conviction": 3, "macro_alignment": 3, "portfolio_fit": 3, "catalyst": 3},
        })
        advanced.append(ticker)
        entry_theses[ticker] = {
            "bull_case": "[OFFLINE] Synthetic bull case",
            "bear_case": "[OFFLINE] Synthetic bear case",
            "catalysts": ["Earnings"],
            "risks": ["Valuation"],
            "conviction": "medium",
            "conviction_rationale": "Synthetic",
            "score": c.get("combined_score", 60),
        }
    for c in candidates[open_slots:]:
        decisions.append({
            "ticker": c.get("ticker", ""),
            "decision": "REJECT",
            "rationale": "[OFFLINE] No open slots remaining",
        })
    return {"decisions": decisions, "advanced_tickers": advanced, "entry_theses": entry_theses}


# ── S3 / archive stubs ──────────────────────────────────────────────────────

def _stub_download_db(self):
    """Skip S3 download — use local DB or create empty."""
    import sqlite3
    logger.info("[offline] stub download_db — using local DB at %s", self.local_db_path)
    self.db_conn = sqlite3.connect(self.local_db_path)
    self.db_conn.row_factory = sqlite3.Row
    self._ensure_schema()
    return self.db_conn


def _stub_upload_db(self, run_date):
    logger.info("[offline] stub upload_db — skipping S3 upload")


def _stub_load_predictions_json(self):
    logger.info("[offline] stub load_predictions_json — returning empty")
    return {}


def _stub_send_email(**kwargs):
    logger.info("[offline] stub send_email — printing subject only")
    print(f"  [OFFLINE] Email would be sent: {kwargs.get('subject', '(no subject)')}")
    return True


# ── Installer ────────────────────────────────────────────────────────────────

_patches = []


def install_offline_stubs():
    """
    Monkey-patch all external call sites so the pipeline runs fully offline.
    Call this BEFORE importing graph modules.
    """
    logger.info("[offline] Installing offline stubs — no API/LLM calls will be made")

    # Force-disable decision-artifact capture for offline runs even if the
    # env var leaked in from the dev shell. Capture would otherwise hit
    # real S3 (no boto3 stub here) and either pollute the production
    # corpus with synthetic data (creds available) or hard-fail (no creds).
    # Both are wrong for offline mode; this guard makes offline always
    # capture-off regardless of shell state.
    import os as _os
    if _os.environ.get("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "").lower() in (
        "true", "1", "yes",
    ):
        logger.warning(
            "[offline] ALPHA_ENGINE_DECISION_CAPTURE_ENABLED was set in the "
            "shell environment — overriding to 'false' for offline run safety "
            "(would otherwise hit real S3 with synthetic data)."
        )
    _os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] = "false"

    targets = [
        # Data fetchers
        ("data.fetchers.price_fetcher.fetch_price_data", _stub_fetch_price_data),
        ("data.fetchers.price_fetcher.fetch_sp500_sp400_with_sectors", _stub_fetch_sp500_sp400),
        ("data.fetchers.price_fetcher.fetch_short_interest", _stub_fetch_short_interest),
        ("data.fetchers.news_fetcher.fetch_all_news", _stub_fetch_all_news),
        ("data.fetchers.analyst_fetcher.fetch_analyst_consensus", _stub_fetch_analyst_consensus),
        ("data.fetchers.macro_fetcher.fetch_macro_data", _stub_fetch_macro_data),
        ("data.fetchers.macro_fetcher.compute_market_breadth", _stub_compute_market_breadth),
        ("data.fetchers.revision_fetcher.fetch_revisions", _stub_fetch_revisions),
        ("data.fetchers.options_fetcher.fetch_options_signals", _stub_fetch_options_signals),
        ("data.fetchers.options_fetcher.cache_options_to_s3", _stub_cache_options_to_s3),
        ("data.fetchers.insider_fetcher.fetch_insider_activity", _stub_fetch_insider_activity),
        ("data.fetchers.insider_fetcher.cache_insider_to_s3", _stub_cache_insider_to_s3),

        # LLM agents
        ("agents.macro_agent.run_macro_agent_with_reflection", _stub_run_macro_agent_with_reflection),
        ("agents.macro_agent.run_macro_agent", _stub_run_macro_agent),

        # Sector teams + CIO
        ("agents.sector_teams.quant_analyst.run_quant_analyst", _stub_run_quant_analyst),
        ("agents.sector_teams.qual_analyst.run_qual_analyst", _stub_run_qual_analyst),
        ("agents.sector_teams.peer_review.run_peer_review", _stub_run_peer_review),
        ("agents.sector_teams.sector_team.run_sector_team", _stub_run_sector_team),
        ("agents.investment_committee.ic_cio.run_cio", _stub_run_cio),

        # Email
        ("emailer.sender.send_email", _stub_send_email),
    ]

    # Archive manager methods — patched on the class
    from archive.manager import ArchiveManager
    ArchiveManager._orig_download_db = ArchiveManager.download_db
    ArchiveManager._orig_upload_db = ArchiveManager.upload_db
    ArchiveManager._orig_load_predictions = ArchiveManager.load_predictions_json
    ArchiveManager.download_db = _stub_download_db
    ArchiveManager.upload_db = _stub_upload_db
    ArchiveManager.load_predictions_json = _stub_load_predictions_json

    # Also stub the S3 write methods on ArchiveManager
    for method_name in ("write_signals_json", "write_consolidated_report",
                        "upload_population_json"):
        if hasattr(ArchiveManager, method_name):
            original = getattr(ArchiveManager, method_name)
            setattr(ArchiveManager, f"_orig_{method_name}", original)
            def _make_stub(name):
                def _stub(self, *args, **kwargs):
                    logger.info("[offline] stub %s — skipping S3 write", name)
                return _stub
            setattr(ArchiveManager, method_name, _make_stub(method_name))

    # S3 config override — return defaults (no S3 call)
    import config as cfg_mod
    cfg_mod._research_params_cache = dict(cfg_mod._RP_DEFAULTS)

    # Patch institutional fetcher if importable
    try:
        import data.fetchers.institutional_fetcher as inst_mod
        inst_mod.fetch_institutional_accumulation = _stub_fetch_institutional_accumulation
    except ImportError:
        pass

    # Patch yfinance.download globally to prevent any stray yf calls
    try:
        import yfinance as yf
        def _stub_yf_download(*args, **kwargs):
            logger.info("[offline] stub yf.download — returning empty DataFrame")
            return pd.DataFrame()
        yf.download = _stub_yf_download
    except ImportError:
        pass

    # Apply function patches via direct module attribute replacement.
    # We import each module and replace the function attribute so that
    # callers importing from the module get the stub.
    import importlib
    for target, stub in targets:
        parts = target.rsplit(".", 1)
        mod_path, func_name = parts[0], parts[1]
        try:
            mod = importlib.import_module(mod_path)
            setattr(mod, func_name, stub)
        except (ImportError, AttributeError) as e:
            logger.warning("[offline] could not patch %s: %s", target, e)

    print("OFFLINE MODE: all API/LLM/S3/email calls stubbed with synthetic data")


def install_llm_only_stubs():
    """
    Stub ONLY the LLM-using agents — keep real data fetchers, real archive,
    real S3 reads. This is the dry-run mode for debugging Research bugs that
    aren't LLM-related: data-shape mismatches, score_aggregator hard-fails,
    archive_writer regressions, signals.json structure, etc.

    Decision-artifact capture is force-disabled by setting
    ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED=false`` (overriding any shell
    value). Stub-llm runs use real S3 in other paths but capture's hard-fail
    posture would block local debugging if IAM isn't right, AND a successful
    write would pollute the prod corpus with stub agent outputs. Force-off
    is safer.

    Compared to ``install_offline_stubs``: real APIs (FMP, FRED, yfinance,
    EDGAR), real research.db download, real population — but every Anthropic
    LLM call is replaced with a synthetic response. Costs $0 in tokens to
    run end-to-end.

    Caveat: LLM stubs return narrative-shaped placeholder text. They DO NOT
    emit score fields (matching the post-2026-04-25 prompt convention that
    held-stock updates are narrative-only). Sector-team picks come from the
    quant analyst stub which uses real technical_scores to rank — so the
    quant output still reflects real signal data.
    """
    logger.info(
        "[stub-llm] Installing LLM-only stubs — real data + real archive, "
        "stubbed agent calls"
    )

    # Force-disable decision-artifact capture (see install_offline_stubs
    # for rationale).
    import os as _os
    if _os.environ.get("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "").lower() in (
        "true", "1", "yes",
    ):
        logger.warning(
            "[stub-llm] ALPHA_ENGINE_DECISION_CAPTURE_ENABLED was set — "
            "overriding to 'false' for stub-llm run safety."
        )
    _os.environ["ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"] = "false"

    targets = [
        # LLM agents only — leave data fetchers, archive, S3 untouched.
        ("agents.macro_agent.run_macro_agent_with_reflection",
         _stub_run_macro_agent_with_reflection),
        ("agents.macro_agent.run_macro_agent", _stub_run_macro_agent),
        ("agents.sector_teams.quant_analyst.run_quant_analyst",
         _stub_run_quant_analyst),
        ("agents.sector_teams.qual_analyst.run_qual_analyst",
         _stub_run_qual_analyst),
        ("agents.sector_teams.peer_review.run_peer_review",
         _stub_run_peer_review),
        ("agents.sector_teams.sector_team.run_sector_team",
         _stub_run_sector_team),
        ("agents.investment_committee.ic_cio.run_cio", _stub_run_cio),
    ]

    import importlib
    for target, stub in targets:
        parts = target.rsplit(".", 1)
        mod_path, func_name = parts[0], parts[1]
        try:
            mod = importlib.import_module(mod_path)
            setattr(mod, func_name, stub)
        except (ImportError, AttributeError) as e:
            logger.warning("[stub-llm] could not patch %s: %s", target, e)

    print("STUB-LLM MODE: real data + real archive, agent LLM calls stubbed")


def patch_graph_modules_llm_only():
    """
    Patch graph module local name bindings AFTER they've been imported,
    LLM agents only. Companion to ``install_llm_only_stubs``.
    """
    import sys
    _graph_patches = {
        "run_macro_agent_with_reflection": _stub_run_macro_agent_with_reflection,
        "run_sector_team": _stub_run_sector_team,
        "run_cio": _stub_run_cio,
    }
    for mod_name in ("graph.research_graph",):
        mod = sys.modules.get(mod_name)
        if mod:
            for attr, stub in _graph_patches.items():
                if hasattr(mod, attr):
                    setattr(mod, attr, stub)


def patch_graph_modules():
    """
    Patch graph module local name bindings AFTER they've been imported.
    Call this after `from graph.research_graph import ...` in the runner.
    """
    import sys
    _graph_patches = {
        # V1 data fetchers
        "fetch_price_data": _stub_fetch_price_data,
        "fetch_sp500_sp400_with_sectors": _stub_fetch_sp500_sp400,
        "fetch_short_interest": _stub_fetch_short_interest,
        "fetch_all_news": _stub_fetch_all_news,
        "fetch_analyst_consensus": _stub_fetch_analyst_consensus,
        "fetch_macro_data": _stub_fetch_macro_data,
        "compute_market_breadth": _stub_compute_market_breadth,
        "fetch_revisions": _stub_fetch_revisions,
        "fetch_options_signals": _stub_fetch_options_signals,
        "cache_options_to_s3": _stub_cache_options_to_s3,
        "fetch_insider_activity": _stub_fetch_insider_activity,
        "cache_insider_to_s3": _stub_cache_insider_to_s3,
        # LLM agents
        "run_macro_agent_with_reflection": _stub_run_macro_agent_with_reflection,
        "send_email": _stub_send_email,
        # Sector teams + CIO
        "run_sector_team": _stub_run_sector_team,
        "run_cio": _stub_run_cio,
    }
    for mod_name in ("graph.research_graph",):
        mod = sys.modules.get(mod_name)
        if mod:
            for attr, stub in _graph_patches.items():
                if hasattr(mod, attr):
                    setattr(mod, attr, stub)
