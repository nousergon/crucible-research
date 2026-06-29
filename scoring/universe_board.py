"""
Universe scoreboard — the full ~900-name S&P 500+400 scanner universe with
per-stock attractiveness + factor + raw-metric data, published as ONE typed S3
artifact for the dashboard's filterable universe board.

Motivation. The scanner evaluates ~903 tickers every Saturday and the factor
substrate (``scoring/factor_scoring.py`` → ``factors/profiles/{date}/``) already
computes 6 pillar scores for ALL of them — but downstream only the ~25-60 that
survive the quant filter + agent pipeline get a composite attractiveness score
or appear in ``signals.json``. The rejected ~840 retain factor + technical data
in the ``scanner_evaluations`` SQLite table, invisible outside the console DB.
This module lifts that full-universe data into a product contract so the
dashboard can show, sort, and FILTER all ~900 names by attractiveness, by each
pillar, by raw valuation/fundamental/technical metric ranges, by sector, by
country, and by scanner gate status.

Attractiveness method (schema_version 2 — SOTA / institutional). The 6 pillars
are already sector-neutral WITHIN-SECTOR percentile ranks (``factor_scoring.py``);
the institutional defect was only the FINAL blend — a plain equal-weight mean of
six bounded percentiles concentrates toward 50 (CLT) and erases cross-sectional
dispersion. The Grinold-Kahn fix (this module):

  z_{i,p} = clip((pillar_{i,p} - mean_p) / sd_p, -3, +3)   # cross-sectional z, winsorized
  blend_i = Σ_{p∈avail} w_p·z_{i,p} / Σ_{p∈avail} w_p       # coverage-renormalized weights
  attractiveness_i = 100 · cross_sectional_percentile(blend_i)   # rank-renorm → full 0-100 range

``w_p`` defaults to equal (1/6) — the competent public baseline — and is
overridden by the private, backtester-tuned ``config/factor_attractiveness_weights.json``
when present (mirrors ``config/scoring_weights.json``; the IC-tuning is the
Phase-2 private edge). Winsorization is near-no-op on bounded-percentile inputs
(kept as institutional convention + future-proofing); the dispersion restoration
comes from the terminal percentile rank. Per stock we emit ``attractiveness_raw``
(the signed blend) and ``pillar_contributions`` (the additive ``w_p·z_p / Σw``
terms that sum to the blend) so the board explains WHY a name ranks where it does.

Output (versioned — consumers pin on ``schema_version``):
  ``s3://{bucket}/scanner/universe/{run_date}/universe.json``
  ``s3://{bucket}/scanner/universe/latest.json`` (sidecar)

Schema::

  {
    "schema_version": 2,
    "as_of": "YYYY-MM-DD",
    "universe_count": int,
    "attractiveness_method": "sector_neutral_zscore_percentile",
    "pillars": ["quality", "value", "momentum", "growth", "stewardship", "defensiveness"],
    "pillar_weights": {quality: float, ...},   # normalized to sum 1.0 (equal default)
    "gate_config": {                           # the resolved scanner thresholds this cycle (null if unresolvable)
      "min_avg_volume", "min_price", "tech_score_min", "max_atr_pct",
      "momentum_ma200_floor_pct", "momentum_top_n",
      "deep_value_path_enabled", "deep_value_max_rsi", "deep_value_max_atr_pct", "deep_value_max_candidates"
    },
    "stocks": [
      {
        "ticker": "AAPL",
        "sector": "Information Technology",   # GICS, from factor profile / sector_map
        "country": "United States",            # domicile, from universe_classification (null if uncovered)
        "industry": "Consumer Electronics",    # null if uncovered
        "attractiveness_score": 0-100 | null,  # cross-sectional percentile of the weighted z-blend
        "attractiveness_raw": float | null,    # the signed z-blend (institutional dispersion preserved)
        "pillars": {quality, value, momentum, growth, stewardship, defensiveness},  # 0-100 | null each
        "pillar_contributions": {quality: float, ...},  # additive w_p·z_p/Σw terms (sum = attractiveness_raw)
        "pillar_coverage": {quality: int, ...},   # # raw factors that contributed per pillar
        "focus_score": 0-100 | null,           # scanner's 4-factor regime-blended subscore
        "focus_stance": "momentum" | ... | null,
        "tech_score": 0-100 | null,            # scanner's pure-technical attractiveness
        "gate": {"quant_filter_pass": 0|1, "filter_fail_reason": str | null},
        "gate_stage": str,                     # terminal funnel stage: passed|liquidity|volatility|below_thresholds|rank_cutoff|no_data
        "gate_trace": [                        # ordered per-gate value-vs-threshold trace (transparency)
          {"stage": "liquidity", "value": float|null, "threshold": float|null, "op": ">=", "pass": bool|null}, ...
        ],
        "metrics": {  # DISPLAY-ready raw units (see _DISPLAY_METRICS denorm contract)
          "current_price", "market_cap", "avg_volume",
          "pe", "pb", "fcf_yield", "dividend_yield", "debt_to_equity", "current_ratio", "payout_ratio",
          "roe", "gross_margin", "revenue_growth_3y", "eps_growth_3y",
          "rsi_14", "momentum_20d", "return_60d", "return_120d",
          "realized_vol_20d", "atr_pct", "dist_from_52w_high", "price_vs_ma200", "beta"
        }
      },
      ...
    ]
  }

Failure posture. This is a SECONDARY observability artifact hung off the
research run's primary deliverable (``signals.json``). Per the no-silent-fails
exception for secondary observability, the archive_writer call site fail-SOFTs
with a WARN log (the board is dashboard visibility, not a trading-path
contract) — a board-write failure must NOT fail the research run. The builder
itself raises on genuinely broken inputs so the caller's WARN records a real
fault rather than silently emitting an empty board.
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

UNIVERSE_BOARD_SCHEMA_VERSION = 2

# Winsorization clip for the per-pillar cross-sectional z-scores (institutional
# convention; near-no-op on already-percentile inputs — see module docstring).
_ZSCORE_CLIP = 3.0

# Pillar → factor-profile field. Single source of truth lives in
# scoring/composite.py::_PILLAR_TO_FACTOR_KEY; imported lazily in the builder so
# this module has no import-time dependency on composite's config loading.
_PILLAR_ORDER = ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")

# DISPLAY metric contract: (feature_store_column, output_field, denorm_multiplier).
#
# The feature store (alpha-engine-data ``features/{date}/*.parquet``) stores
# several valuation ratios PREDICTOR-NORMALIZED (e.g. pe_ratio = trailing P/E ÷
# 30) — see ``features/SCHEMA.md`` §Fundamental, whose normalizations are pinned
# by that repo's ``test_schema_contract.py``. ``denorm_multiplier`` recovers the
# human-readable raw value for display (multiplier 1.0 = already clean units).
# This map MIRRORS the documented SCHEMA.md contract; if a future PR changes a
# normalization in the data repo, ``test_universe_board.py`` and the data repo's
# schema-contract test are the paired guards.
_FUNDAMENTAL_METRICS: tuple[tuple[str, str, float], ...] = (
    ("pe_ratio", "pe", 30.0),                 # PE = pe_ratio × 30
    ("pb_ratio", "pb", 5.0),                  # PB = pb_ratio × 5
    ("debt_to_equity", "debt_to_equity", 2.0),  # D/E = col × 2
    ("current_ratio", "current_ratio", 3.0),  # CR = col × 3
    ("fcf_yield", "fcf_yield", 1.0),          # decimal pct — clean
    ("dividend_yield", "dividend_yield", 1.0),
    ("payout_ratio", "payout_ratio", 1.0),
    ("roe", "roe", 1.0),                      # decimal pct — clean
    ("gross_margin", "gross_margin", 1.0),    # 0–1 fraction — clean
    ("revenue_growth_3y", "revenue_growth_3y", 1.0),  # CAGR — clean
    ("eps_growth_3y", "eps_growth_3y", 1.0),
    ("market_cap_raw", "market_cap", 1.0),    # raw dollars — clean
)
_TECHNICAL_METRICS: tuple[tuple[str, str, float], ...] = (
    ("rsi_14", "rsi_14", 1.0),                # 0–100 — clean
    ("momentum_20d", "momentum_20d", 1.0),    # decimal return — clean
    ("return_60d", "return_60d", 1.0),
    ("return_120d", "return_120d", 1.0),
    ("realized_vol_20d", "realized_vol_20d", 1.0),  # annualized decimal — clean
    ("atr_14_pct", "atr_pct", 1.0),           # decimal pct — clean
    ("dist_from_52w_high", "dist_from_52w_high", 1.0),
    ("price_vs_ma200", "price_vs_ma200", 1.0),
    ("beta_60d", "beta", 1.0),                # dimensionless — clean
    ("avg_volume_20d_raw", "avg_volume", 1.0),  # raw shares — clean
)

DEFAULT_BUCKET = "alpha-engine-research"


def _bucket(bucket: str | None) -> str:
    return bucket or os.environ.get("S3_BUCKET", DEFAULT_BUCKET)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    """Population mean + std of a non-empty list (std == 0.0 when n < 2 or all
    values identical — the z-score helper treats that as 'no dispersion' → z=0)."""
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / n  # population variance
    return mean, var ** 0.5


def _avg_rank_pct(values: dict[str, float]) -> dict[str, float]:
    """Cross-sectional percentile (0-100) via average-rank — matches pandas
    ``rank(pct=True) * 100`` (ties share their mean rank). Empty → {}."""
    import bisect

    if not values:
        return {}
    arr = sorted(values.values())
    n = len(arr)
    out: dict[str, float] = {}
    for k, x in values.items():
        lo = bisect.bisect_left(arr, x)
        hi = bisect.bisect_right(arr, x)
        avg_rank = (lo + 1 + hi) / 2.0  # 1-indexed average rank
        out[k] = round(avg_rank / n * 100, 2)
    return out


def _zscore(value: float, mean: float, std: float) -> float:
    """Winsorized cross-sectional z-score. std == 0 → 0.0 (no dispersion)."""
    if std <= 0:
        return 0.0
    z = (value - mean) / std
    return max(-_ZSCORE_CLIP, min(_ZSCORE_CLIP, z))


def _num(v: Any, multiplier: float = 1.0) -> Optional[float]:
    """Coerce to a finite float (applying ``multiplier``) or None. NaN / inf /
    non-numeric → None (a coverage gap, never a fabricated value)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return round(f * multiplier, 6)


def _equal_weights() -> dict[str, float]:
    return {p: 1.0 / len(_PILLAR_ORDER) for p in _PILLAR_ORDER}


def _load_pillar_weights(bucket: str | None, s3_client: Any) -> dict[str, float]:
    """Pillar weights for the z-blend, NORMALIZED to sum 1.0.

    Reads the optional private ``config/factor_attractiveness_weights.json``
    (backtester/attribution-tuned — the Phase-2 edge) and falls back to EQUAL
    weights (the competent public baseline) when the file is absent, malformed,
    or sums to ≤0. Accepts either ``{"weights": {pillar: w}}`` or a flat
    ``{pillar: w}`` map. Negative / non-numeric weights coerce to 0.
    """
    try:
        s3 = _client(s3_client)
        obj = s3.get_object(Bucket=_bucket(bucket), Key="config/factor_attractiveness_weights.json")
        data = json.loads(obj["Body"].read())
        raw = data.get("weights", data) if isinstance(data, dict) else {}
        parsed: dict[str, float] = {}
        for p in _PILLAR_ORDER:
            v = _num(raw.get(p))
            parsed[p] = v if (v is not None and v > 0) else 0.0
        total = sum(parsed.values())
        if total <= 0:
            return _equal_weights()
        normalized = {p: round(w / total, 6) for p, w in parsed.items()}
        logger.info("[universe_board] pillar weights from config/factor_attractiveness_weights.json: %s", normalized)
        return normalized
    except Exception:
        logger.info("[universe_board] no tuned pillar weights — using equal-weight baseline")
        return _equal_weights()


def _resolve_gate_config() -> Optional[dict]:
    """Resolve the scanner gate thresholds USED THIS CYCLE from the same config
    source the scanner reads (``get_scanner_params()`` S3-tuned overrides +
    module constants in ``config.py``). Single source of truth — no cross-module
    state threading. Fail-soft to None (gate trace thresholds degrade to null)."""
    try:
        from config import (
            get_scanner_params,
            DEEP_VALUE_PATH_ENABLED,
            DEEP_VALUE_MAX_RSI,
            DEEP_VALUE_MAX_ATR_PCT,
            DEEP_VALUE_MAX_CANDIDATES,
        )

        sp = get_scanner_params()
        return {
            "min_avg_volume": _num(sp.get("min_avg_volume")),
            "min_price": _num(sp.get("min_price")),
            "tech_score_min": _num(sp.get("tech_score_min")),
            "max_atr_pct": _num(sp.get("max_atr_pct")),
            "momentum_ma200_floor_pct": _num(sp.get("momentum_ma200_floor_pct")),
            "momentum_top_n": _num(sp.get("momentum_top_n")),
            "deep_value_path_enabled": bool(DEEP_VALUE_PATH_ENABLED),
            "deep_value_max_rsi": _num(DEEP_VALUE_MAX_RSI),
            "deep_value_max_atr_pct": _num(DEEP_VALUE_MAX_ATR_PCT),
            "deep_value_max_candidates": _num(DEEP_VALUE_MAX_CANDIDATES),
        }
    except Exception:
        logger.warning("[universe_board] could not resolve gate_config — gate thresholds will be null")
        return None


def _gate_trace(row: dict, gate_config: Optional[dict]) -> tuple[list[dict], str]:
    """Reconstruct the per-stock funnel trace (each gate: value vs threshold,
    pass/fail) + the terminal ``gate_stage``, from the recorded scanner-eval
    values + the resolved ``gate_config``. Transparent and deterministic — the
    same thresholds + comparisons the scanner used, in funnel order.

    The terminal ``gate_stage`` is reconciled with the AUTHORITATIVE recorded
    outcome (``quant_filter_pass`` / ``filter_fail_reason``): the trace explains
    the value-vs-threshold comparisons, but the scanner's recorded verdict wins
    (a name can clear the synthesized gates yet be dropped at the rank cutoff,
    or be back-filled in past the thresholds — see scanner.py fallback fill).
    """
    gc = gate_config or {}
    avg_vol = _num(row.get("avg_volume_20d"))
    price = _num(row.get("current_price"))
    atr_pct = _num(row.get("atr_pct"))
    tech_score = _num(row.get("tech_score"))
    price_vs_ma200 = _num(row.get("price_vs_ma200"))
    passed = int(row.get("quant_filter_pass", 0) or 0) == 1
    fail_reason = row.get("filter_fail_reason")

    def _cmp(value, threshold, op):
        if value is None or threshold is None:
            return None
        return value >= threshold if op == ">=" else value <= threshold

    min_vol = gc.get("min_avg_volume")
    max_atr = gc.get("max_atr_pct")
    tsm = gc.get("tech_score_min")
    trace = [
        {"stage": "liquidity", "metric": "avg_volume_20d", "value": avg_vol,
         "threshold": min_vol, "op": ">=", "pass": _cmp(avg_vol, min_vol, ">=")},
        {"stage": "volatility", "metric": "atr_pct", "value": atr_pct,
         "threshold": max_atr, "op": "<=", "pass": _cmp(atr_pct, max_atr, "<=")},
        {"stage": "tech_score", "metric": "tech_score", "value": tech_score,
         "threshold": tsm, "op": ">=", "pass": _cmp(tech_score, tsm, ">=")},
    ]
    # min_price floor rides with liquidity — surface as a secondary check only
    # when a floor is configured (the common config has min_price == 0).
    min_price = gc.get("min_price")
    if min_price is not None and min_price > 0:
        trace.insert(1, {"stage": "price_floor", "metric": "current_price", "value": price,
                         "threshold": min_price, "op": ">=", "pass": _cmp(price, min_price, ">=")})

    # Terminal stage: the recorded verdict is authoritative.
    if passed:
        stage = "passed"
    elif fail_reason in ("no_data", "no_tech_indicators"):
        stage = "no_data"
    elif fail_reason == "liquidity":
        stage = "liquidity"
    elif fail_reason in ("volatility_momentum", "volatility_deep_value"):
        stage = "volatility"
    elif fail_reason == "below_thresholds":
        stage = "below_thresholds"
    elif fail_reason == "rank_cutoff":
        stage = "rank_cutoff"
    elif fail_reason:
        stage = str(fail_reason)
    else:
        # No recorded reason (graph rows don't carry filter_fail_reason) — infer
        # the first synthesized gate that fails; default to rank_cutoff when all
        # value-gates pass but the name still didn't make the basket.
        stage = "passed"
        for g in trace:
            if g["pass"] is False:
                stage = g["stage"]
                break
        else:
            stage = "rank_cutoff"
    return trace, stage


def build_universe_board(
    run_date: str,
    scanner_evals: list[dict],
    *,
    factor_profiles: dict | None = None,
    classification: dict | None = None,
    technical_df: "Any" = None,
    fundamental_df: "Any" = None,
    pillar_weights: dict | None = None,
    gate_config: dict | None = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> dict:
    """Assemble the full-universe scoreboard payload.

    Args:
        run_date: YYYY-MM-DD run stamp.
        scanner_evals: the per-ticker scanner-evaluation rows built in
            ``archive_writer`` (ticker, sector, tech_score, current_price,
            focus_score/stance, quant_filter_pass, filter_fail_reason, …). This
            is the authoritative universe membership + gate source.
        factor_profiles: ``{ticker: {sector, *_score, *_n}}`` (factors/profiles).
            Read from S3 when None.
        classification: ``{ticker: {sector, country, industry}}`` from the
            universe_classification artifact. Read from S3 when None; an empty
            map degrades country/industry to null (fail-soft, never guessed).
        technical_df / fundamental_df: feature-store parquets. Read from S3 when
            None.
        pillar_weights: the 6 z-blend weights. Loaded from the tuned config
            (equal-weight fallback) when None; normalized to sum 1.0 either way.
        gate_config: resolved scanner thresholds. Resolved from config when None;
            an explicit ``{}`` / None just degrades the gate trace to null
            thresholds (still emits the per-stock funnel order).
        bucket / s3_client: S3 wiring (defaults: env bucket + a fresh boto3
            client) — injectable for tests.

    Returns the board dict (also the unit under the producer contract test).

    Raises when ``scanner_evals`` is empty (a research run that produced no
    universe is a real fault — the caller's WARN records it).
    """
    if not scanner_evals:
        raise ValueError(
            "universe_board: scanner_evals is empty — the research run produced "
            "no universe rows; refusing to emit an empty board (no-silent-fails)."
        )

    if factor_profiles is None:
        factor_profiles = _read_factor_profiles(run_date, bucket, s3_client) or {}
    if classification is None:
        classification = _read_classification(bucket, s3_client) or {}
    if technical_df is None:
        technical_df = _read_parquet("technical", run_date, bucket, s3_client)
    if fundamental_df is None:
        fundamental_df = _read_parquet("fundamental", run_date, bucket, s3_client)
    if pillar_weights is None:
        pillar_weights = _load_pillar_weights(bucket, s3_client)
    if gate_config is None:
        gate_config = _resolve_gate_config()

    # Normalize weights to sum 1.0 (uniform for tuned-file, equal-default, and
    # injected-raw test inputs). Per-stock blends renormalize over AVAILABLE
    # pillars, so only the ratios matter for the math; this fixes the displayed
    # top-level pillar_weights.
    _wt_total = sum(max(0.0, (_num(w) or 0.0)) for w in pillar_weights.values()) or 1.0
    pillar_weights = {
        p: round(max(0.0, (_num(pillar_weights.get(p)) or 0.0)) / _wt_total, 6)
        for p in _PILLAR_ORDER
    }

    from scoring.composite import _PILLAR_TO_FACTOR_KEY

    tech_by_ticker = _index_parquet(technical_df)
    fund_by_ticker = _index_parquet(fundamental_df)

    # ── Pass 1: per-stock base records + collect per-pillar cross-section ─────
    records: list[tuple[dict, dict]] = []   # (stock, pillar_scores)
    pillar_values: dict[str, dict[str, float]] = {p: {} for p in _PILLAR_ORDER}
    for row in scanner_evals:
        ticker = row.get("ticker")
        if not ticker:
            continue
        profile = factor_profiles.get(ticker, {})
        cls = classification.get(ticker, {})
        tech = tech_by_ticker.get(ticker, {})
        fund = fund_by_ticker.get(ticker, {})

        pillar_scores = {
            pillar: _num(profile.get(_PILLAR_TO_FACTOR_KEY[pillar]))
            for pillar in _PILLAR_ORDER
        }
        for p, v in pillar_scores.items():
            if v is not None:
                pillar_values[p][ticker] = v
        pillar_coverage = {
            pillar: int(profile[f"{_PILLAR_TO_FACTOR_KEY[pillar][:-6]}_n"])
            for pillar in _PILLAR_ORDER
            if f"{_PILLAR_TO_FACTOR_KEY[pillar][:-6]}_n" in profile
        }

        metrics: dict[str, Optional[float]] = {
            "current_price": _num(row.get("current_price")),
        }
        for col, field, mult in _FUNDAMENTAL_METRICS:
            metrics[field] = _num(fund.get(col), mult)
        for col, field, mult in _TECHNICAL_METRICS:
            # tech_score row carries some technicals too; prefer the parquet
            # (full set), fall back to the scanner-eval row for the few it has.
            val = tech.get(col)
            if val is None and col in ("rsi_14",):
                val = row.get(col)
            metrics[field] = _num(val, mult)

        trace, gate_stage = _gate_trace(row, gate_config)

        records.append(({
            "ticker": ticker,
            "sector": profile.get("sector") or row.get("sector") or cls.get("sector"),
            "country": cls.get("country"),
            "industry": cls.get("industry"),
            "attractiveness_score": None,   # filled in pass 2 (cross-sectional percentile)
            "attractiveness_raw": None,
            "pillars": pillar_scores,
            "pillar_contributions": {},
            "pillar_coverage": pillar_coverage,
            "focus_score": _num(row.get("focus_score")),
            "focus_stance": row.get("focus_stance"),
            "tech_score": _num(row.get("tech_score")),
            "gate": {
                "quant_filter_pass": int(row.get("quant_filter_pass", 0) or 0),
                "filter_fail_reason": row.get("filter_fail_reason"),
            },
            "gate_stage": gate_stage,
            "gate_trace": trace,
            "metrics": metrics,
        }, pillar_scores))

    # ── Cross-sectional pillar stats (mean/std over names that HAVE each pillar) ─
    pillar_stats = {
        p: _mean_std(list(vals.values()))
        for p, vals in pillar_values.items()
        if vals
    }

    # ── Pass 2: winsorized z-blend + additive per-pillar contributions ───────
    blends: dict[str, float] = {}
    for stock, pillar_scores in records:
        contribs: dict[str, tuple[float, float]] = {}
        num = 0.0
        wsum = 0.0
        for p in _PILLAR_ORDER:
            v = pillar_scores[p]
            w = pillar_weights.get(p, 0.0)
            if v is None or w <= 0 or p not in pillar_stats:
                continue
            mean, std = pillar_stats[p]
            z = _zscore(v, mean, std)
            num += w * z
            wsum += w
            contribs[p] = (w, z)
        if wsum > 0:
            blend = num / wsum
            blends[stock["ticker"]] = blend
            stock["attractiveness_raw"] = round(blend, 4)
            stock["pillar_contributions"] = {
                p: round(w * z / wsum, 4) for p, (w, z) in contribs.items()
            }

    # ── Terminal cross-sectional percentile → 0-100 (restores dispersion) ────
    pct = _avg_rank_pct(blends)
    for stock, _ in records:
        stock["attractiveness_score"] = pct.get(stock["ticker"])

    stocks = [s for s, _ in records]
    stocks.sort(
        key=lambda s: (s["attractiveness_score"] is None, -(s["attractiveness_score"] or 0))
    )

    return {
        "schema_version": UNIVERSE_BOARD_SCHEMA_VERSION,
        "as_of": run_date,
        "universe_count": len(stocks),
        "attractiveness_method": "sector_neutral_zscore_percentile",
        "pillars": list(_PILLAR_ORDER),
        "pillar_weights": pillar_weights,
        "gate_config": gate_config,
        "stocks": stocks,
    }


# ── S3 I/O ──────────────────────────────────────────────────────────────────

def _client(s3_client: Any):
    if s3_client is not None:
        return s3_client
    import boto3
    return boto3.client("s3")


def _read_factor_profiles(run_date: str, bucket: str | None, s3_client: Any) -> dict | None:
    """Read ``factors/profiles/{run_date}/by_ticker.json`` (written earlier in
    the same run), falling back to the ``latest.json`` sidecar."""
    s3 = _client(s3_client)
    b = _bucket(bucket)
    for key in (f"factors/profiles/{run_date}/by_ticker.json", "factors/profiles/latest.json"):
        try:
            obj = s3.get_object(Bucket=b, Key=key)
            return json.loads(obj["Body"].read())
        except Exception:
            continue
    logger.warning("[universe_board] no factor profiles readable for %s — pillars will be null", run_date)
    return None


def _read_classification(bucket: str | None, s3_client: Any) -> dict | None:
    """Read ``market_data/universe_classification/latest.json`` → ``{ticker:
    {sector, country, industry}}``. Domicile is near-static so latest is fine.
    Absent artifact (pre-first-production) degrades country/industry to null."""
    s3 = _client(s3_client)
    try:
        obj = s3.get_object(Bucket=_bucket(bucket), Key="market_data/universe_classification/latest.json")
        return json.loads(obj["Body"].read()).get("data", {})
    except Exception:
        logger.warning("[universe_board] universe_classification not readable — country/industry will be null")
        return None


def _read_parquet(name: str, run_date: str, bucket: str | None, s3_client: Any):
    """Read ``features/{run_date}/{name}.parquet`` → DataFrame, or None on miss
    (board still builds; that metric group degrades to null)."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return None
    s3 = _client(s3_client)
    try:
        obj = s3.get_object(Bucket=_bucket(bucket), Key=f"features/{run_date}/{name}.parquet")
        return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
    except Exception:
        logger.warning("[universe_board] features/%s/%s.parquet not readable — those metrics null", run_date, name)
        return None


def _index_parquet(df: "Any") -> dict[str, dict]:
    """``{ticker: {col: value}}`` from a feature-store DataFrame (empty when
    df is None / has no ``ticker`` column)."""
    if df is None or not hasattr(df, "columns") or "ticker" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for record in df.to_dict(orient="records"):
        t = record.get("ticker")
        if t:
            out[str(t)] = record
    return out


def write_universe_board_to_s3(board: dict, run_date: str, *, bucket: str | None = None, s3_client: Any = None) -> str:
    """Write the board to the dated key + the ``latest.json`` sidecar. Returns
    the dated key."""
    s3 = _client(s3_client)
    b = _bucket(bucket)
    body = json.dumps(board, separators=(",", ":"), default=str).encode("utf-8")
    dated_key = f"scanner/universe/{run_date}/universe.json"
    for key in (dated_key, "scanner/universe/latest.json"):
        s3.put_object(Bucket=b, Key=key, Body=body, ContentType="application/json")
    logger.info(
        "[universe_board] wrote %d stocks → s3://%s/%s (+latest)",
        board.get("universe_count", 0), b, dated_key,
    )
    return dated_key


def compute_and_write_universe_board(
    run_date: str,
    scanner_evals: list[dict],
    *,
    bucket: str | None = None,
    s3_client: Any = None,
) -> str:
    """archive_writer entry point — build from S3-resident inputs + the in-memory
    scanner_evals and write the artifact. Returns the dated S3 key."""
    board = build_universe_board(run_date, scanner_evals, bucket=bucket, s3_client=s3_client)
    return write_universe_board_to_s3(board, run_date, bucket=bucket, s3_client=s3_client)
