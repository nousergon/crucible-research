"""
Central configuration reader — mirrors universe.yaml into typed Python constants.
All other modules import from here rather than reading YAML directly.

Research params (signal boosts, thresholds) support S3 override via
config/research_params.json, auto-tuned by the backtester weekly.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional
import yaml

from alpha_engine_lib.secrets import get_secret

def _find_config(filename: str, subdir: str = "research") -> Path:
    """Locate real config yaml across local, CI, and Lambda environments.

    `.sample.yaml` files in ./config/ are documentation for open-source viewers
    ONLY — never loaded by runtime. Real config lives in the private
    alpha-engine-config repo and is staged into the Lambda image at build
    time via deploy.sh (see infrastructure/deploy.sh:130+). Hard-fail if
    none of the known locations has the file.

    Search order:
      1. ~/alpha-engine-config/<subdir>/<file>      (local dev with sibling clone)
      2. <repo>/../alpha-engine-config/<subdir>/<file>  (local dev with repo parent)
      3. $GITHUB_WORKSPACE/alpha-engine-config/<subdir>/<file>  (CI checkout)
      4. <repo>/config/<file>                       (Lambda image: deploy.sh
         stages config repo yaml into this directory, subdir-flattened)
    """
    ws = os.environ.get("GITHUB_WORKSPACE")
    search = [
        Path.home() / "alpha-engine-config" / subdir / filename,
        Path(__file__).parent.parent / "alpha-engine-config" / subdir / filename,
    ]
    if ws:
        search.append(Path(ws) / "alpha-engine-config" / subdir / filename)
    # Lambda image: deploy.sh flattens <subdir>/<file> → config/<file>
    search.append(Path(__file__).parent / "config" / filename)
    found = next((p for p in search if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"Could not locate {subdir}/{filename} in alpha-engine-config. "
            f"Searched: {[str(p) for p in search]}. "
            "Checkout the config repo at ~/alpha-engine-config (local) or "
            "$GITHUB_WORKSPACE/alpha-engine-config (CI). On Lambda the "
            "config is staged into config/ by deploy.sh; if this is firing "
            "in Lambda the image was built without the staging step."
        )
    return found

_CONFIG_PATH = _find_config("universe.yaml")
_SCORING_CFG_PATH = _find_config("scoring.yaml")

_logger = logging.getLogger(__name__)


def _load() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_scoring() -> dict:
    with open(_SCORING_CFG_PATH) as f:
        return yaml.safe_load(f) or {}


_cfg = _load()
_scoring_cfg = _load_scoring()

# Technical scoring parameters (weights, thresholds, ma anchors) live in
# scoring.yaml under `technical:`. Loaded once at import; consumed by
# scoring/technical.py.
TECHNICAL_CFG: dict = _scoring_cfg.get("technical", {})

# ── Population (replaces static universe) ────────────────────────────────────
# All stocks are derived from S&P 900 scanner — no hardcoded starting stocks.
# UNIVERSE / UNIVERSE_TICKERS / SECTOR_MAP are loaded dynamically from
# population/latest.json (S3) or SQLite at run time.
# The static list below is kept empty — graph.research_graph loads the active
# population from the archive manager at startup.
POPULATION_CFG: dict = _cfg.get("population", {})
UNIVERSE: list[dict] = _cfg.get("universe", [])  # backward compat (empty after migration)
UNIVERSE_TICKERS: list[str] = [s["ticker"] for s in UNIVERSE]
SECTOR_MAP: dict[str, str] = {s["ticker"]: s["sector"] for s in UNIVERSE}

# ── Scoring ───────────────────────────────────────────────────────────────────
SCORING_WEIGHTS: dict[str, float] = _cfg["scoring_weights"]
# Horizon separation: Research uses quant + qual only (6–12 month fundamental).
# Technical analysis is handled by Predictor (GBM) and Executor (ATR/time exits).
# Hard-fail if legacy keys linger — silent fallback is how the NULL-archive
# bug survived for weeks. Config repo must ship quant/qual keys.
if "news" in SCORING_WEIGHTS or "research" in SCORING_WEIGHTS:
    raise ValueError(
        f"scoring_weights uses deprecated keys (news/research) in {_CONFIG_PATH}. "
        "Rename to quant/qual — see alpha-engine-config."
    )
WEIGHT_QUANT: float = SCORING_WEIGHTS["quant"]
WEIGHT_QUAL: float = SCORING_WEIGHTS["qual"]

RATING_BUY_THRESHOLD: float = _cfg["rating_thresholds"]["buy"]
RATING_SELL_THRESHOLD: float = _cfg["rating_thresholds"]["sell"]

# ── Macro-sector coherence gate (2026-05-13) ─────────────────────────────────
# Blocks NEW buy_candidates in UNDERWEIGHT sectors when composite < min score.
# Pure structural discipline — don't fight the macro call you just made.
# Loaded from scoring.yaml `aggregator.macro_sector_coherence_gate`.
_AGGREGATOR_CFG: dict = _scoring_cfg.get("aggregator", {})
_COHERENCE_GATE_CFG: dict = _AGGREGATOR_CFG.get("macro_sector_coherence_gate", {})
SECTOR_COHERENCE_GATE_ENABLED: bool = bool(_COHERENCE_GATE_CFG.get("enabled", False))
SECTOR_COHERENCE_UW_MIN_SCORE: float = float(_COHERENCE_GATE_CFG.get("uw_min_score", 80.0))

# ── Scanner ───────────────────────────────────────────────────────────────────
SCANNER_CFG: dict = _cfg["scanner"]
CANDIDATE_COUNT: int = SCANNER_CFG["candidate_count"]
CANDIDATE_UNIVERSE: str = SCANNER_CFG["candidate_universe"]
MIN_AVG_VOLUME: int = SCANNER_CFG["min_avg_volume"]
MIN_PRICE: float = SCANNER_CFG["min_price"]
ROTATION_TIERS: list[dict] = SCANNER_CFG["rotation_tiers"]
WEAK_PICK_SCORE_THRESHOLD: float = SCANNER_CFG["weak_pick_score_threshold"]
WEAK_PICK_CONSECUTIVE_RUNS: int = SCANNER_CFG["weak_pick_consecutive_runs"]
EMERGENCY_ROTATION_NEW_SCORE: float = SCANNER_CFG["emergency_rotation_new_score"]
DEEP_VALUE_PATH_ENABLED: bool = SCANNER_CFG["deep_value_path"]
DEEP_VALUE_MAX_RSI: float = SCANNER_CFG["deep_value_max_rsi"]
DEEP_VALUE_MIN_CONSENSUS: str = SCANNER_CFG["deep_value_min_consensus"]
DEEP_VALUE_MAX_CANDIDATES: int = SCANNER_CFG["deep_value_max_candidates"]
MAX_ATR_PCT: float = SCANNER_CFG.get("max_atr_pct", 8.0)
DEEP_VALUE_MAX_ATR_PCT: float = SCANNER_CFG.get("deep_value_max_atr_pct", 12.0)
MAX_DEBT_TO_EQUITY: float = SCANNER_CFG.get("max_debt_to_equity", 3.0)
MIN_CURRENT_RATIO: float = SCANNER_CFG.get("min_current_ratio", 0.5)
BALANCE_SHEET_EXEMPT_SECTORS: list[str] = SCANNER_CFG.get("balance_sheet_exempt_sectors", ["Financial", "Real Estate"])

# ── Archive ───────────────────────────────────────────────────────────────────
ARCHIVE_CFG: dict = _cfg["archive"]

# ── Alerts ────────────────────────────────────────────────────────────────────
ALERTS_CFG: dict = _cfg["alerts"]
ALERTS_ENABLED: bool = ALERTS_CFG["enabled"]
PRICE_MOVE_THRESHOLD_PCT: float = ALERTS_CFG["price_move_threshold_pct"]
ALERT_COOLDOWN_MINUTES: int = ALERTS_CFG["cooldown_minutes"]

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_CFG: dict = _cfg["email"]
EMAIL_RECIPIENTS: list[str] = EMAIL_CFG["recipients"]
EMAIL_SENDER: str = EMAIL_CFG["sender"]

# ── Schedule ──────────────────────────────────────────────────────────────────
SCHEDULE_CFG: dict = _cfg["schedule"]
HOLIDAY_CALENDAR: str = SCHEDULE_CFG["holiday_calendar"]

# ── Predictor ─────────────────────────────────────────────────────────────────
_pred_cfg: dict = _cfg.get("predictor", {})
PREDICTOR_PREDICTIONS_KEY: str = _pred_cfg.get("s3_predictions_key", "predictor/predictions/latest.json")
# Minimum GBM prediction_confidence required to apply the confirmation gate veto.
# Below this threshold the prediction is treated as low-conviction and ignored.
MIN_PREDICTION_CONFIDENCE: float = float(_pred_cfg.get("min_confidence", 0.60))

# ── CIO ───────────────────────────────────────────────────────────────────────
# Weekly entrant cap applied to new investments (does not affect reaffirmations
# of held BUY-rated names).
_cio_cfg: dict = _cfg.get("cio", {})
CIO_MAX_NEW_ENTRANTS: int = int(_cio_cfg.get("max_new_entrants", 10))
CIO_MIN_NEW_ENTRANTS: int = int(_cio_cfg.get("min_new_entrants", 2))
if CIO_MIN_NEW_ENTRANTS < 0 or CIO_MAX_NEW_ENTRANTS < CIO_MIN_NEW_ENTRANTS:
    raise ValueError(
        f"Invalid cio config: min_new_entrants={CIO_MIN_NEW_ENTRANTS}, "
        f"max_new_entrants={CIO_MAX_NEW_ENTRANTS} (must be 0 <= min <= max)"
    )

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_CFG: dict = _cfg["llm"]
PER_STOCK_MODEL: str = LLM_CFG["per_stock_model"]
STRATEGIC_MODEL: str = LLM_CFG["strategic_model"]
MAX_TOKENS_PER_STOCK: int = LLM_CFG["max_tokens_per_stock"]
MAX_TOKENS_STRATEGIC: int = LLM_CFG["max_tokens_strategic"]
CONCURRENT_AGENTS: int = LLM_CFG["concurrent_agents"]

# ── AWS / Environment ─────────────────────────────────────────────────────────
S3_BUCKET: str = os.environ.get("S3_BUCKET", "alpha-engine-research")
AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
ANTHROPIC_API_KEY: str = get_secret("ANTHROPIC_API_KEY", required=False, default="") or ""
FMP_API_KEY: str = get_secret("FMP_API_KEY", required=False, default="") or ""
FRED_API_KEY: str = get_secret("FRED_API_KEY", required=False, default="") or ""

# ── Thesis management ───────────────────────────────────────────────────────
_thesis_cfg: dict = _cfg.get("thesis", {})
FORCED_REFRESH_DAYS: int = int(_thesis_cfg.get("forced_refresh_days", 10))
PRIOR_REPORT_MAX_CHARS: int = int(_thesis_cfg.get("prior_report_max_chars", 2000))

# ── Regime guardrails ───────────────────────────────────────────────────────
REGIME_GUARDRAILS: dict = _cfg.get("regime_guardrails", {})

# ── Scoring / staleness ───────────────────────────────────────────────────────
STALENESS_THRESHOLD_DAYS: int = 5       # flag if score unchanged >= this many trading days
MATERIAL_SCORE_CHANGE_MIN: float = 3.0  # minimum point change to reset last_material_change_date

# ── Performance tracker (BUY signal accuracy feedback loop) ──────────────────
_perf_cfg: dict = _cfg.get("performance_tracker", {})
RECALIBRATION_THRESHOLD: float = float(_perf_cfg.get("recalibration_threshold", 0.55))
RECALIBRATION_LOOKBACK_DAYS: int = int(_perf_cfg.get("recalibration_lookback_days", 60))

# ── All tracked tickers in a run (universe + up to 3 candidates) ──────────────
ALL_SECTORS: list[str] = [
    "Technology", "Healthcare", "Financial", "Consumer Discretionary",
    "Consumer Staples", "Energy", "Industrials", "Materials",
    "Real Estate", "Utilities", "Communication Services",
]

# ── Sector Teams ──────────────────────────────────────────────────────────────
SECTOR_TEAMS_CFG: dict = _cfg.get("sector_teams", {})
IC_CFG: dict = _cfg.get("investment_committee", {})
QUANT_MAX_ITERATIONS: int = SECTOR_TEAMS_CFG.get("quant_max_iterations", 8)
QUAL_MAX_ITERATIONS: int = SECTOR_TEAMS_CFG.get("qual_max_iterations", 8)
TEAM_PICKS_PER_RUN: int = SECTOR_TEAMS_CFG.get("picks_per_run", 3)

# ── Research params (signal boosts + thresholds) ─────────────────────────────
# Defaults from universe.yaml; overridden at cold-start by S3
# config/research_params.json (auto-tuned by backtester weekly).
#
# S3 override chain: S3 → local cache (/tmp) → universe.yaml defaults.
# Same pattern as scoring_weights — one S3 call per cold-start.

_RESEARCH_PARAMS_S3_KEY = "config/research_params.json"
_RESEARCH_PARAMS_CACHE_PATH = os.environ.get(
    "RESEARCH_PARAMS_CACHE", "/tmp/research_params_cache.json"
)

# YAML defaults
_rp_yaml: dict = _cfg.get("research_params", {})

_RP_DEFAULTS: dict = {
    # ATR computation
    "atr_period": int(_rp_yaml.get("atr_period", 20)),
    # Short interest
    "short_interest_buy_threshold_pct": float(_rp_yaml.get("short_interest_buy_threshold_pct", 20)),
    "short_interest_high_threshold_pct": float(_rp_yaml.get("short_interest_high_threshold_pct", 40)),
    "short_interest_buy_boost": float(_rp_yaml.get("short_interest_buy_boost", 2.0)),
    "short_interest_high_boost": float(_rp_yaml.get("short_interest_high_boost", 4.0)),
    # 13F Institutional accumulation
    "institutional_min_funds": int(_rp_yaml.get("institutional_min_funds", 3)),
    "institutional_boost": float(_rp_yaml.get("institutional_boost", 3.0)),
    # Consistency check
    "consistency_bullish_dominance": float(_rp_yaml.get("consistency_bullish_dominance", 0.7)),
    "consistency_bearish_dominance": float(_rp_yaml.get("consistency_bearish_dominance", 0.3)),
    "consistency_low_score": float(_rp_yaml.get("consistency_low_score", 40)),
    "consistency_high_score": float(_rp_yaml.get("consistency_high_score", 70)),
    "consistency_divergence_threshold": float(_rp_yaml.get("consistency_divergence_threshold", 30)),
    # Aggregate boost cap
    "max_aggregate_boost": float(_rp_yaml.get("max_aggregate_boost", 10.0)),
    # O10: PEAD
    "pead_window_min_days": int(_rp_yaml.get("pead_window_min_days", 1)),
    "pead_window_max_days": int(_rp_yaml.get("pead_window_max_days", 20)),
    "pead_strong_threshold_pct": float(_rp_yaml.get("pead_strong_threshold_pct", 5.0)),
    "pead_strong_boost": float(_rp_yaml.get("pead_strong_boost", 5.0)),
    "pead_modest_boost": float(_rp_yaml.get("pead_modest_boost", 2.5)),
    "pead_strong_miss_boost": float(_rp_yaml.get("pead_strong_miss_boost", -5.0)),
    "pead_modest_miss_boost": float(_rp_yaml.get("pead_modest_miss_boost", -2.5)),
    # O11: EPS revisions
    "revision_strong_streak": int(_rp_yaml.get("revision_strong_streak", 3)),
    "revision_strong_boost": float(_rp_yaml.get("revision_strong_boost", 3.0)),
    "revision_modest_boost": float(_rp_yaml.get("revision_modest_boost", 1.5)),
    "revision_strong_negative_boost": float(_rp_yaml.get("revision_strong_negative_boost", -3.0)),
    "revision_modest_negative_boost": float(_rp_yaml.get("revision_modest_negative_boost", -1.5)),
    # O12: Options positioning
    "options_high_pc_ratio": float(_rp_yaml.get("options_high_pc_ratio", 1.5)),
    "options_high_pc_adj": float(_rp_yaml.get("options_high_pc_adj", -3.0)),
    "options_low_pc_ratio": float(_rp_yaml.get("options_low_pc_ratio", 0.5)),
    "options_low_pc_adj": float(_rp_yaml.get("options_low_pc_adj", 2.0)),
    "options_low_iv_rank": float(_rp_yaml.get("options_low_iv_rank", 20)),
    "options_low_iv_adj": float(_rp_yaml.get("options_low_iv_adj", 1.0)),
    # O13: Insider cluster buying
    "insider_cluster_boost": float(_rp_yaml.get("insider_cluster_boost", 5.0)),
    "insider_min_unique_buyers": int(_rp_yaml.get("insider_min_unique_buyers", 2)),
    "insider_unique_buyers_boost": float(_rp_yaml.get("insider_unique_buyers_boost", 2.5)),
    "insider_net_sentiment_threshold": float(_rp_yaml.get("insider_net_sentiment_threshold", -0.5)),
    "insider_net_sentiment_cap": float(_rp_yaml.get("insider_net_sentiment_cap", -2.0)),
}

# Module-level cache: populated once per cold-start by get_research_params().
_research_params_cache: Optional[dict] = None


def _load_research_params_from_s3() -> Optional[dict]:
    """
    Check S3 for backtester-updated research params.

    Reads s3://{bucket}/config/research_params.json, written by the
    backtester's research_optimizer when it applies an update.

    Fallback chain: S3 → local cache file → None (YAML defaults).
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        bucket = os.environ.get("RESEARCH_BUCKET", S3_BUCKET)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=_RESEARCH_PARAMS_S3_KEY)
        data = json.loads(obj["Body"].read())

        # Only load known keys, skip metadata like updated_at
        params = {k: data[k] for k in _RP_DEFAULTS if k in data}
        if params:
            _logger.info(
                "Research params loaded from S3 (updated %s): %s",
                data.get("updated_at", "unknown"), params,
            )
            try:
                with open(_RESEARCH_PARAMS_CACHE_PATH, "w") as f:
                    json.dump(params, f, indent=2)
            except Exception as e:
                _logger.warning("Could not write research params cache: %s", e)
            return params
    except Exception as e:
        if "NoSuchKey" not in str(e):
            _logger.warning("Could not read research params from S3: %s", e)

    # Fallback: local cache (M8 fix: with age check)
    try:
        if os.path.exists(_RESEARCH_PARAMS_CACHE_PATH):
            import time
            cache_age_hours = (time.time() - os.path.getmtime(_RESEARCH_PARAMS_CACHE_PATH)) / 3600
            if cache_age_hours > 168:  # 7 days
                _logger.warning("Research params cache is %.0fh old (>7d) — using YAML defaults instead", cache_age_hours)
                return None
            if cache_age_hours > 24:
                _logger.warning("Research params cache is %.0fh old — may be stale", cache_age_hours)
            with open(_RESEARCH_PARAMS_CACHE_PATH) as f:
                params = json.load(f)
            if params:
                _logger.info("Research params loaded from local cache (age: %.0fh): %s", cache_age_hours, params)
                return params
    except Exception as e:
        _logger.warning("Could not read research params cache: %s", e)

    return None


def get_research_params() -> dict:
    """
    Return current research params, checking S3 override first.

    Cached for Lambda lifetime (one S3 call per cold-start).
    Falls back to universe.yaml values if S3 file absent.
    """
    global _research_params_cache
    if _research_params_cache is None:
        s3_params = _load_research_params_from_s3()
        _research_params_cache = {**_RP_DEFAULTS, **(s3_params or {})}
    return _research_params_cache


# Convenience accessors — these call get_research_params() lazily, so they
# always reflect S3 overrides once the first access triggers the load.

def rp(key: str):
    """Get a single research param by key."""
    return get_research_params()[key]


# ── Scanner params (Phase 4a: auto-tuned by backtester) ─────────────────────
# Same pattern as research params: S3 → local cache → YAML defaults.

_SCANNER_PARAMS_S3_KEY = "config/scanner_params.json"
_SCANNER_PARAMS_CACHE_PATH = os.environ.get(
    "SCANNER_PARAMS_CACHE", "/tmp/scanner_params_cache.json"
)

_SP_DEFAULTS: dict = {
    "tech_score_min": int(SCANNER_CFG.get("tech_score_min", 60)),
    "max_atr_pct": float(SCANNER_CFG.get("max_atr_pct", MAX_ATR_PCT)),
    "min_avg_volume": int(MIN_AVG_VOLUME),
    "min_price": float(MIN_PRICE),
    "momentum_top_n": int(SCANNER_CFG.get("momentum_top_n", 60)),
    "momentum_ma200_floor_pct": float(SCANNER_CFG.get("momentum_ma200_floor_pct", -15)),
    "min_combined_candidates": int(SCANNER_CFG.get("min_combined_candidates", 3)),
    # Rotation parameters
    "rotation_default_required_delta": float(SCANNER_CFG.get("rotation_default_required_delta", 3.0)),
    "rotation_all_weak_score": float(SCANNER_CFG.get("rotation_all_weak_score", 55)),
    "rotation_weak_pick_min_tenure_days": int(SCANNER_CFG.get("rotation_weak_pick_min_tenure_days", 10)),
    "rotation_weak_pick_min_challenger_score": float(SCANNER_CFG.get("rotation_weak_pick_min_challenger_score", 65)),
}

_scanner_params_cache: Optional[dict] = None


def get_scanner_params() -> dict:
    """
    Return scanner filter params, checking S3 override first.

    Cached for Lambda lifetime. Falls back to universe.yaml values.
    """
    global _scanner_params_cache
    if _scanner_params_cache is not None:
        return _scanner_params_cache

    try:
        import boto3
        bucket = os.environ.get("RESEARCH_BUCKET", S3_BUCKET)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=_SCANNER_PARAMS_S3_KEY)
        data = json.loads(obj["Body"].read())
        params = {k: data[k] for k in _SP_DEFAULTS if k in data}
        if params:
            _logger.info("Scanner params from S3 (updated %s): %s",
                         data.get("updated_at", "unknown"), params)
            try:
                with open(_SCANNER_PARAMS_CACHE_PATH, "w") as f:
                    json.dump(params, f, indent=2)
            except Exception:
                pass
            _scanner_params_cache = {**_SP_DEFAULTS, **params}
            return _scanner_params_cache
    except Exception as e:
        if "NoSuchKey" not in str(e):
            _logger.warning("Could not read scanner params from S3: %s", e)

    # Fallback: local cache
    try:
        if os.path.exists(_SCANNER_PARAMS_CACHE_PATH):
            with open(_SCANNER_PARAMS_CACHE_PATH) as f:
                params = json.load(f)
            if params:
                _scanner_params_cache = {**_SP_DEFAULTS, **params}
                return _scanner_params_cache
    except Exception:
        pass

    _scanner_params_cache = _SP_DEFAULTS.copy()
    return _scanner_params_cache
