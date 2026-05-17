"""
Offline stubs — monkey-patches all external calls (LLM, data APIs, S3, email)
so the research pipeline can run end-to-end without any network access.

Usage:
    from local.offline_stubs import install_offline_stubs
    install_offline_stubs()   # call BEFORE importing graph modules

Generates synthetic but structurally valid data so every graph node
receives the dict shapes it expects.

The 7 LLM-agent stub functions are imported from ``/dry_run.py`` at
repo root — the Lambda-importable single source. This file owns the
data-API + S3 + email stubs (which the Lambda doesn't need) plus the
``install_*`` orchestration. Output strings carry ``[DRY-RUN]``
markers (from dry_run.py) rather than ``[OFFLINE]``; cosmetic
difference, no functional change.
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


def _stub_fetch_insider_activity(tickers, lookback_days=90, reference_date=None):
    logger.info("[offline] stub fetch_insider_activity: %d tickers", len(tickers))
    return {}


def _stub_cache_insider_to_s3(data, date):
    pass


def _stub_fetch_institutional_accumulation(tickers):
    logger.info("[offline] stub fetch_institutional_accumulation: %d tickers", len(tickers))
    return {}


# ── LLM agent stubs (single-source — imported from /dry_run.py) ─────────────
#
# Both this file (local/offline_stubs.py for `local/run.py --offline`
# and `--stub-llm`) and the Lambda handler's auto-gate (via
# /dry_run.py::install_dry_run_stubs) need the same 7 synthetic agent
# functions. dry_run.py is the canonical home (Lambda-importable);
# this file imports them so there's exactly one definition to maintain.
#
# The imports surface the same 7 names this module historically defined,
# so install_offline_stubs / install_llm_only_stubs / patch_graph_modules
# below continue to reference them with no signature change.

from dry_run import (
    _stub_run_macro_agent_with_reflection,
    _stub_run_macro_agent,
    _stub_run_quant_analyst,
    _stub_run_qual_analyst,
    _stub_run_peer_review,
    _stub_run_sector_team,
    _stub_run_cio,
)


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
        ("data.fetchers.news_fetcher.fetch_all_news", _stub_fetch_all_news),
        ("data.fetchers.analyst_fetcher.fetch_analyst_consensus", _stub_fetch_analyst_consensus),
        ("data.fetchers.macro_fetcher.fetch_macro_data", _stub_fetch_macro_data),
        ("data.fetchers.macro_fetcher.compute_market_breadth", _stub_compute_market_breadth),
        ("data.fetchers.revision_fetcher.fetch_revisions", _stub_fetch_revisions),
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
        "fetch_all_news": _stub_fetch_all_news,
        "fetch_analyst_consensus": _stub_fetch_analyst_consensus,
        "fetch_macro_data": _stub_fetch_macro_data,
        "compute_market_breadth": _stub_compute_market_breadth,
        "fetch_revisions": _stub_fetch_revisions,
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
