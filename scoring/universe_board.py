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

Attractiveness method (schema_version 3 — SOTA / institutional). The 6 pillars
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
    "schema_version": 3,
    "as_of": "YYYY-MM-DD",
    "universe_count": int,
    "attractiveness_method": "sector_neutral_zscore_percentile",
    "tradeability_method": "sqrt_impact_almgren_chriss_round_trip",  # INDEPENDENT of attractiveness (§43)
    "tradeability_reference_notional_usd": float,                     # reference single-name trade size
    "pillars": ["quality", "value", "momentum", "growth", "stewardship", "defensiveness"],
    "pillar_weights": {quality: float, ...},   # normalized to sum 1.0 (equal default)
    "fundamental_distinctness": {pe: int, pb: int, ..., eps_growth_3y: int},  # per-field
      # cross-sectional DISTINCT count over non-null values (alpha-engine-config-
      # I8255) — non-null is not the same as informative; see
      # _assert_fundamental_distinctness_floor for the enforced subset + threshold.
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
        "tradeability": {                      # INDEPENDENT √-impact cost score — NEVER blended into attractiveness (§43)
          "expected_cost_bps": float | null,   # round-trip cost at the reference notional (half_spread + c·σ·√(Q/ADV) + commission)
          "tradeability_score": 0-100 | null,  # cross-sectional percentile (higher = cheaper to access)
          "adv_usd": float | null,             # avg 20d dollar volume (price × shares); null = coverage gap
          "reference_notional_usd": float
        },
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
fault rather than silently emitting an empty board. This now also covers a
POPULATED-but-WRONG board: when the caller passes ``scanner_tickers`` (the
authoritative ``candidates.json::scanner_tickers`` list), the builder verifies
the board's own ``gate_stage == "passed"`` set agrees exactly and raises on
mismatch (alpha-engine-config#4820 — a run whose gate input degraded silently
emitted a board reporting zero, or an undercount, of gate-passers with no
error anywhere).
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any

from nousergon_lib.quant.attractiveness import (
    DEFAULT_PILLAR_WEIGHTS,
    compute_cross_sectional_attractiveness,
    normalize_pillar_weights,
)
from nousergon_lib.quant.attractiveness import (
    PILLAR_ORDER as _PILLAR_ORDER,
)
from nousergon_lib.quant.attractiveness import (
    attractiveness_from_factor_profiles as _attractiveness_from_factor_profiles,
)
from nousergon_lib.quant.transaction_cost import (
    TransactionCostModel,
    tradeability_percentiles,
)

logger = logging.getLogger(__name__)

UNIVERSE_BOARD_SCHEMA_VERSION = 3

# Cap on the MEMBER list inside ``attractiveness_coverage``. The count itself is
# never capped — see ``_attractiveness_coverage``. 50 is well above any expected
# exclusion count on a healthy ~900-name board (the measured steady state is 0),
# so hitting it is itself a signal rather than routine truncation.
_COVERAGE_MEMBER_CAP = 50

# Known-bad gate block (alpha-engine-config#4820). These two dated boards were
# written BEFORE this module enforced the gate-passed-count producer contract
# (``_assert_gate_passed_matches_scanner_tickers``) and are frozen with an
# unreliable ``gate`` / ``gate_stage`` block:
#
#   2026-07-02: scanner_eval_log was missing/empty upstream — EVERY record's
#     gate_stage is wrong (0 stocks read gate_stage=="passed" vs the normal 60
#     in candidates.json::scanner_tickers that week).
#   2026-06-26: scanner_eval_log was present but incomplete — an undercount
#     (46 of the 60 candidates.json::scanner_tickers read gate_stage=="passed").
#
# NOT backfilled: the original 2026-07-02/2026-06-26 pipeline state (feature-
# store snapshot, scanner gate thresholds in effect that cycle) cannot be
# faithfully re-run, and reconstructing per-ticker gate verdicts from today's
# thresholds would silently swap one wrong-looking-right board for another —
# worse than a documented gap (see ``scripts/backfill_universe_membership.py``,
# which already routes around this same gap by reading scanner_tickers +
# the attractiveness history instead of the gate block, for precedent). Every
# OTHER field on these two boards (attractiveness, pillars, tradeability,
# metrics) is unaffected — only ``gate`` / ``gate_stage`` / ``gate_trace`` are
# untrustworthy. A consumer must skip/flag ``gate_stage`` for these dates
# rather than trust it at face value.
KNOWN_BAD_GATE_RUN_DATES: frozenset[str] = frozenset({"2026-07-02", "2026-06-26"})

# alpha-engine-config-I8255 — backfill decision for boards written BEFORE this
# module's fundamental-distinctness floor existed (``_assert_fundamental_
# distinctness_floor`` below). Measured 2026-08-24, AWS_PROFILE=ne-admin, from
# scanner/universe/{date}/universe.json::stocks[].metrics: five fundamental
# fields (fcf_yield, gross_margin, roe, revenue_growth_3y, eps_growth_3y) were
# saturated placeholders — as low as 1 DISTINCT value across ~899 covered
# names — on every board dated 2026-08-03 through 2026-08-19 inclusive, and
# genuinely populated (738-889 distinct) from 2026-08-21 on (the repair
# reached the live board between the 08-19 and 08-20 scanner runs; #4820's
# gate-passed contract and #7982's all-null-metrics collapse are separate,
# unrelated defects on the same window — see their own docstrings above/
# below). DECISION: ANNOTATE, not recompute. The feature-store snapshot that
# fed factor_scoring.py on any of these dates is gone; recomputing today's
# fundamentals against those historical dates would not reconstruct what the
# board actually ranked on — it would silently swap one wrong-looking-right
# board for another, exactly the reasoning already applied to
# KNOWN_BAD_GATE_RUN_DATES above. A consumer reading universe_board data for
# a date in this window (e.g. a backfill/audit script, NOT the live pipeline
# — no such consumer exists in this repo today) should treat the quality/
# growth/stewardship pillars as unreliable for that date via
# ``is_degenerate_fundamentals_run_date``. This module cannot itself flag
# the downstream cuts-leaderboard artifact (``research/cuts_leaderboard/
# {date}.json`` — deliverable 2 of alpha-engine-config-I8255, out of scope
# for this change: it lands in ``scoring/leaderboard_producers.py``, which
# has open PRs against it as of this writing) — that remains open.
DEGENERATE_FUNDAMENTALS_RUN_DATE_WINDOW: tuple[str, str] = ("2026-08-03", "2026-08-19")


def is_degenerate_fundamentals_run_date(run_date: str) -> bool:
    """True when ``run_date`` (YYYY-MM-DD) falls inside the measured
    saturated-placeholder window (alpha-engine-config-I8255). ISO date
    strings compare lexicographically, so a plain range check is exact."""
    start, end = DEGENERATE_FUNDAMENTALS_RUN_DATE_WINDOW
    return start <= run_date <= end

# Reference single-name trade size (USD) for the per-name tradeability estimate —
# a representative position on the paper book. ``expected_cost_bps`` is the
# ROUND-TRIP cost to enter+exit this notional; ``tradeability_score`` is its
# cross-sectional percentile (higher = cheaper to access). Overridable via the
# optional ``transaction_cost`` config block (which also tunes the cost model).
_DEFAULT_REFERENCE_NOTIONAL_USD = 100_000.0
_TRADEABILITY_METHOD = "sqrt_impact_almgren_chriss_round_trip"

# Winsorization clip lives in nousergon_lib.quant.attractiveness (re-exported via
# compute_cross_sectional_attractiveness). Kept here only for doc cross-refs.
_ZSCORE_CLIP = 3.0

# Pillar → factor-profile field. Single source of truth lives in
# scoring/composite.py::_PILLAR_TO_FACTOR_KEY; imported lazily in the builder so
# this module has no import-time dependency on composite's config loading.
# _PILLAR_ORDER imported from nousergon_lib.quant.attractiveness above.

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
    ("pe_ratio", "pe", 30.0),  # PE = pe_ratio × 30
    ("pb_ratio", "pb", 5.0),  # PB = pb_ratio × 5
    ("debt_to_equity", "debt_to_equity", 2.0),  # D/E = col × 2
    ("current_ratio", "current_ratio", 3.0),  # CR = col × 3
    ("fcf_yield", "fcf_yield", 1.0),  # decimal pct — clean
    ("dividend_yield", "dividend_yield", 1.0),
    ("payout_ratio", "payout_ratio", 1.0),
    ("roe", "roe", 1.0),  # decimal pct — clean
    ("gross_margin", "gross_margin", 1.0),  # 0–1 fraction — clean
    ("revenue_growth_3y", "revenue_growth_3y", 1.0),  # CAGR — clean
    ("eps_growth_3y", "eps_growth_3y", 1.0),
    ("market_cap_raw", "market_cap", 1.0),  # raw dollars — clean
)
_TECHNICAL_METRICS: tuple[tuple[str, str, float], ...] = (
    ("rsi_14", "rsi_14", 1.0),  # 0–100 — clean
    ("momentum_20d", "momentum_20d", 1.0),  # decimal return — clean
    ("return_60d", "return_60d", 1.0),
    ("return_120d", "return_120d", 1.0),
    ("realized_vol_20d", "realized_vol_20d", 1.0),  # annualized decimal — clean
    ("atr_14_pct", "atr_pct", 1.0),  # decimal pct — clean
    ("dist_from_52w_high", "dist_from_52w_high", 1.0),
    ("price_vs_ma200", "price_vs_ma200", 1.0),
    ("beta_60d", "beta", 1.0),  # dimensionless — clean
    ("avg_volume_20d_raw", "avg_volume", 1.0),  # raw shares — clean
)

DEFAULT_BUCKET = "alpha-engine-research"


def _bucket(bucket: str | None) -> str:
    return bucket or os.environ.get("S3_BUCKET", DEFAULT_BUCKET)


def _equal_weights() -> dict[str, float]:
    return dict(DEFAULT_PILLAR_WEIGHTS)


def _reference_notional(tradeability_config: dict | None) -> float:
    return float((tradeability_config or {}).get("reference_notional_usd", _DEFAULT_REFERENCE_NOTIONAL_USD))


def compute_tradeability(
    metrics_by_ticker: dict[str, dict],
    *,
    tradeability_config: dict | None = None,
) -> dict[str, dict]:
    """Per-name TRADEABILITY — an INDEPENDENT artifact, NEVER blended into the
    attractiveness composite (ARCHITECTURE §43): attractiveness forecasts forward
    return; tradeability measures the cost to ACCESS it. They are computed,
    stored and displayed independently and meet only at the decision layer via
    net-alpha. This lifts the ONE shared √-impact engine
    (``nousergon_lib.quant.transaction_cost``, §15) so the live score and the
    backtester's net-alpha read a single cost definition.

    For each name with price + ADV coverage, ``expected_cost_bps`` is the
    ROUND-TRIP cost — ``half_spread + impact_coef·(σ/ref_σ)·√(Q/ADV) + commission``,
    doubled for enter+exit — at the reference notional, where ADV$ = ``current_price
    × avg_volume`` (20d shares) and σ = ``realized_vol_20d``. σ is scaled to the
    cross-sectional MEDIAN σ so the median-volatility name reproduces the
    calibrated cost and more/less volatile names cost proportionally more/less
    (true Almgren-Chriss form, parameter-free reference). ``tradeability_score``
    is the 0-100 cross-sectional percentile (higher = cheaper).

    A name lacking price/ADV coverage gets ``None`` for both fields — an honest
    coverage gap, never a fabricated 'cheapest' rank (we do NOT fall the missing
    name back to the spread+commission floor, which would mis-rank it as maximally
    tradeable on absent data).
    """
    import statistics

    model = TransactionCostModel.from_config({"transaction_cost": tradeability_config} if tradeability_config else None)
    ref_notional = _reference_notional(tradeability_config)

    adv_usd: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for ticker, metrics in metrics_by_ticker.items():
        price = metrics.get("current_price")
        vol_shares = metrics.get("avg_volume")
        vol_pct = metrics.get("realized_vol_20d")
        if price is not None and vol_shares is not None and price > 0 and vol_shares > 0:
            adv_usd[ticker] = price * vol_shares
        if vol_pct is not None and vol_pct > 0:
            sigma[ticker] = vol_pct
    ref_sigma = statistics.median(sigma.values()) if sigma else None

    cost_bps: dict[str, float | None] = {}
    for ticker in metrics_by_ticker:
        adv = adv_usd.get(ticker)
        if adv is None:  # no ADV coverage → no honest cost estimate (gap, not floor)
            cost_bps[ticker] = None
            continue
        cost_bps[ticker] = round(
            model.round_trip_bps(ref_notional, adv, sigma=sigma.get(ticker), ref_sigma=ref_sigma),
            4,
        )

    scores = tradeability_percentiles(cost_bps)
    return {
        ticker: {
            "expected_cost_bps": cost_bps[ticker],
            "tradeability_score": scores[ticker],
            "adv_usd": round(adv_usd[ticker], 2) if ticker in adv_usd else None,
            "reference_notional_usd": ref_notional,
        }
        for ticker in metrics_by_ticker
    }


def _num(v: Any, multiplier: float = 1.0) -> float | None:
    """Coerce to a finite float (applying ``multiplier``) or None. NaN / inf /
    non-numeric → None (a coverage gap, never a fabricated value)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return round(f * multiplier, 6)


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
        normalized = normalize_pillar_weights(parsed)
        logger.info("[universe_board] pillar weights from config/factor_attractiveness_weights.json: %s", normalized)
        return normalized
    except Exception:
        logger.info("[universe_board] no tuned pillar weights — using equal-weight baseline")
        return _equal_weights()


def _resolve_gate_config() -> dict | None:
    """Resolve the scanner gate thresholds USED THIS CYCLE from the same config
    source the scanner reads (``get_scanner_params()`` S3-tuned overrides +
    module constants in ``config.py``). Single source of truth — no cross-module
    state threading. Fail-soft to None (gate trace thresholds degrade to null)."""
    try:
        from config import (
            DEEP_VALUE_MAX_ATR_PCT,
            DEEP_VALUE_MAX_CANDIDATES,
            DEEP_VALUE_MAX_RSI,
            DEEP_VALUE_PATH_ENABLED,
            get_scanner_params,
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


def _gate_trace(row: dict, gate_config: dict | None) -> tuple[list[dict], str]:
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
        {
            "stage": "liquidity",
            "metric": "avg_volume_20d",
            "value": avg_vol,
            "threshold": min_vol,
            "op": ">=",
            "pass": _cmp(avg_vol, min_vol, ">="),
        },
        {
            "stage": "volatility",
            "metric": "atr_pct",
            "value": atr_pct,
            "threshold": max_atr,
            "op": "<=",
            "pass": _cmp(atr_pct, max_atr, "<="),
        },
        {
            "stage": "tech_score",
            "metric": "tech_score",
            "value": tech_score,
            "threshold": tsm,
            "op": ">=",
            "pass": _cmp(tech_score, tsm, ">="),
        },
    ]
    # min_price floor rides with liquidity — surface as a secondary check only
    # when a floor is configured (the common config has min_price == 0).
    min_price = gc.get("min_price")
    if min_price is not None and min_price > 0:
        trace.insert(
            1,
            {
                "stage": "price_floor",
                "metric": "current_price",
                "value": price,
                "threshold": min_price,
                "op": ">=",
                "pass": _cmp(price, min_price, ">="),
            },
        )

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


def _attractiveness_coverage(stocks: list[dict]) -> dict:
    """How many names the attractiveness ranking EXCLUDED, and which ones.

    Emitted on every board UNCONDITIONALLY — a healthy run publishes an explicit
    zero rather than nothing. A component that says nothing is unobserved, not
    healthy (``observability-policy.md`` §3.4: records rejected WITH reason), and
    an exclusion nobody can see is the same failure as the fabricated ``0.0``
    this replaced (config-I7272, ``principles.md`` §2.7).

    ``n_excluded_undefined`` counts names carrying ``attractiveness_score: None``
    — every pillar leg undefined, so there is no measured position to rank them
    at. They are absent from the ranking, NOT ranked last: the board's sort keeps
    them after the scored names purely so the artifact has a stable order, and
    ``attractiveness_rank`` is never assigned to them (see
    ``universe_membership._rank_table``, which is fed a None-filtered dict).

    ``excluded_tickers`` is capped at ``_COVERAGE_MEMBER_CAP`` because the board
    is a dashboard artifact read on every page load; ``n_excluded_undefined`` is
    always the true, uncapped count, and ``excluded_truncated`` says so in-band
    rather than letting a reader mistake the cap for the total.

    STOPGAP, deliberately (config-I7272): ``nousergon_lib.quant.attractiveness``
    now publishes this same report itself, via
    ``compute_cross_sectional_attractiveness_with_coverage``. This repo still
    pins ``nousergon-lib@v0.124.3``, which predates that API, so the count is
    derived here from the scores the pinned lib already returns — an identical
    derivation over the identical input, not a second definition of the metric.
    The pin-bump follow-up switches this to the library's own report so exactly
    one implementation survives (``shared-code-policy.md``); until then the fork
    is one function with this note on it.
    """
    excluded = sorted(
        str(s["ticker"]) for s in stocks if s.get("attractiveness_score") is None
    )
    return {
        "n_tickers": len(stocks),
        "n_scored": len(stocks) - len(excluded),
        "n_excluded_undefined": len(excluded),
        "excluded_tickers": excluded[:_COVERAGE_MEMBER_CAP],
        "excluded_truncated": len(excluded) > _COVERAGE_MEMBER_CAP,
    }


# Required fields for the producer-side coverage floor below
# (alpha-engine-config-I7982): the three metrics `compute_tradeability`
# consumes directly (`avg_volume`, `realized_vol_20d`) plus one more
# feature-parquet-sourced technical (`atr_pct`) as a canary for the whole
# `_TECHNICAL_METRICS`/`_FUNDAMENTAL_METRICS` mapping class — the 08-17
# regression took 21 of 23 published fields to zero together, so one
# representative from each source parquet is sufficient to catch the class
# without hard-coding all 23 names into the floor.
_REQUIRED_METRIC_COVERAGE_FIELDS: tuple[str, ...] = ("avg_volume", "realized_vol_20d", "atr_pct")


def _assert_metric_coverage_floor(stocks: list[dict], run_date: str) -> None:
    """Refuse to publish a board where a non-trivial universe has ZERO
    non-null coverage on a metric the tradeability/optimizer path depends on
    (alpha-engine-config-I7982).

    Measured live 2026-08-17..08-20: ``features/{date}/{technical,fundamental}
    .parquet`` reads failed inside the deployed Lambda (pyarrow import
    failure — fixed in the same PR, see requirements.txt), so
    ``_TECHNICAL_METRICS``/``_FUNDAMENTAL_METRICS`` populated every one of 21
    fields as null for all 903 names, `compute_tradeability` derived
    ``adv_usd``/``expected_cost_bps`` as null for every name, and the board
    still wrote successfully with `attractiveness_score` intact — nothing
    downstream raised. `current_price` and `rsi_14` are exempt from this
    floor: both have their own fallback to the scanner-eval row directly
    (see `build_universe_board`'s technical-metrics loop) and stayed near-
    full coverage through the incident, so they cannot detect this class.

    A small universe (< 20 names — an early/degraded scanner run) is exempt:
    zero coverage there is plausibly a genuine data gap for that population,
    not this failure mode, and raising on it would turn a real edge case
    into a false-positive board outage on top of a real one.
    """
    if len(stocks) < 20:
        return
    zero_covered = [
        field
        for field in _REQUIRED_METRIC_COVERAGE_FIELDS
        if not any(s["metrics"].get(field) is not None for s in stocks)
    ]
    if not zero_covered:
        return
    raise ValueError(
        f"universe_board: metric coverage floor breached for {run_date} — "
        f"{len(stocks)} names published with ZERO non-null coverage on "
        f"{zero_covered} (of {_REQUIRED_METRIC_COVERAGE_FIELDS}). This is the "
        "signature of a feature-parquet read failure (alpha-engine-config-"
        "I7982) — refusing to publish a board whose tradeability/optimizer-"
        "facing metrics are silently all-null (no-silent-fails)."
    )



# Fundamental-distinctness floor (alpha-engine-config-I8255, deliverable 1).
#
# ``pillar_coverage`` reads 897-903 of 903 on a healthy board AND on a
# degenerate one — non-null is not the same as informative. Measured
# 2026-08-24 (see DEGENERATE_FUNDAMENTALS_RUN_DATE_WINDOW above for detail),
# distinct cross-sectional values on these five fields collapsed to 1-68
# for at least 18 days while genuinely populated boards carry 738-889:
#
#   field               2026-08-03/14 (broken)   2026-08-21 (healthy)
#   fcf_yield                    1                    889
#   gross_margin                 2                    741
#   roe                        14-18                  773
#   revenue_growth_3y            68                    795
#   eps_growth_3y               41-42                  738
#
# `pe`, `pb`, `debt_to_equity`, `current_ratio` were genuinely populated
# throughout (~760-796 distinct on every measured date, including the
# broken window) and are deliberately EXCLUDED from this floor — flooring
# them too would risk tripping on a legitimately narrow but real
# cross-section, which this module must not do (a guard that fires on
# healthy data gets silenced, which is worse than no guard).
#
# THRESHOLD: 100. The worst-case broken value measured across all five
# fields is 68 (revenue_growth_3y); the best-case healthy value is 738
# (eps_growth_3y). 100 sits just above the broken ceiling with a wide
# margin below the healthy floor (738), so it catches every measured
# degenerate shape (1, 2, 14, 18, 41, 42, 68) without coming within 6x of
# any measured healthy value — not tuned to the exact historical numbers,
# which would make the floor brittle to next quarter's coverage drifting a
# little either way.
_FUNDAMENTAL_DISTINCTNESS_FLOOR = 100
_REQUIRED_FUNDAMENTAL_DISTINCTNESS_FIELDS: tuple[str, ...] = (
    "fcf_yield",
    "gross_margin",
    "roe",
    "revenue_growth_3y",
    "eps_growth_3y",
)


def _fundamental_distinct_counts(stocks: list[dict]) -> dict[str, int]:
    """Per-field cross-sectional DISTINCT count over every published
    fundamental metric (alpha-engine-config-I8255 deliverable 1, first
    clause: "records a per-field cross-sectional distinct-count ... for
    every fundamental metric it publishes") — not just the five the
    producer floor enforces. Null values never count as a distinct value:
    an all-null field reads as 0 distinct, which is itself degenerate and
    already covered by ``_assert_metric_coverage_floor`` for its own three
    canary fields, and would also be caught here for any of these twelve
    if that class ever hit a fundamental column instead."""
    counts: dict[str, int] = {}
    for _col, field, _mult in _FUNDAMENTAL_METRICS:
        values = {s["metrics"].get(field) for s in stocks if s["metrics"].get(field) is not None}
        counts[field] = len(values)
    return counts


def _assert_fundamental_distinctness_floor(stocks: list[dict], run_date: str) -> None:
    """Refuse to publish a board where a required fundamental field is
    NON-null but SATURATED — the same non-informative-placeholder shape
    that let three of six attractiveness pillars ride near-constant for at
    least 18 days with `pillar_coverage` reading 897-903/903 the whole
    time (alpha-engine-config-I8255). Catches both measured degenerate
    shapes: a field pinned to one placeholder value (fcf_yield, 1 distinct)
    and a field all-null (0 distinct — the I7982 collapse shape, here
    applied to the fundamental fields I7982's own floor does not cover).

    Exemption is sized to the FLOOR itself, not the flat 20 used by
    ``_assert_metric_coverage_floor``: a distinct-count floor of
    ``_FUNDAMENTAL_DISTINCTNESS_FLOOR`` (100) cannot be met by construction
    on a universe smaller than that many names, healthy or not — a 20-name
    universe with 20 genuinely distinct ROE values would still read as
    degenerate under a flat 20-name cutoff. ``2 x`` the floor keeps a
    comfortable margin so a merely-small-but-real population (a scanner run
    with reduced coverage, still in the hundreds) is not itself flagged as
    saturated.
    """
    if len(stocks) < 2 * _FUNDAMENTAL_DISTINCTNESS_FLOOR:
        return
    counts = _fundamental_distinct_counts(stocks)
    degenerate = {
        field: counts.get(field, 0)
        for field in _REQUIRED_FUNDAMENTAL_DISTINCTNESS_FIELDS
        if counts.get(field, 0) < _FUNDAMENTAL_DISTINCTNESS_FLOOR
    }
    if not degenerate:
        return
    raise ValueError(
        f"universe_board: fundamental distinctness floor breached for {run_date} — "
        f"{len(stocks)} names published with fewer than {_FUNDAMENTAL_DISTINCTNESS_FLOOR} "
        f"distinct cross-sectional values on {degenerate} (of "
        f"{_REQUIRED_FUNDAMENTAL_DISTINCTNESS_FIELDS}). Non-null is not the same as "
        "informative — this is the signature of a saturated-placeholder fundamental "
        "input (alpha-engine-config-I8255): refusing to publish a board whose pillar "
        "ranking would ride a near-constant cross-section undetected (no-silent-fails)."
    )


def _assert_gate_passed_matches_scanner_tickers(stocks: list[dict], scanner_tickers: list[str], run_date: str) -> None:
    """Producer contract invariant (alpha-engine-config#4820): the board's own
    derived ``gate_stage == "passed"`` membership must equal the AUTHORITATIVE
    ``candidates.json::scanner_tickers`` set for the same run_date exactly.

    Both sides trace back to the SAME upstream fact — the scanner's quant-filter
    survivors — via independent paths (``run_quant_filter``'s recorded per-ticker
    verdict vs its own ``scanner_tickers`` list). They must agree. A mismatch is
    definitive evidence the gate input (``scanner_eval_log``) was missing, empty,
    or otherwise degraded for this run — exactly the historical failure mode
    where the board silently reported zero (2026-07-02) or an undercount
    (2026-06-26) of gate-passers while ``candidates.json`` carried the normal
    60. Raising here (rather than the prior WARN-and-emit-anyway posture) is
    the no-silent-fails fix: a producer must not publish an artifact whose gate
    block reads a false verdict.
    """
    passed = {s["ticker"] for s in stocks if s["gate_stage"] == "passed"}
    expected = set(scanner_tickers)
    if passed == expected:
        return
    missing = expected - passed  # scanner picked it; board didn't mark it passed
    extra = passed - expected  # board marked it passed; scanner didn't pick it
    raise ValueError(
        f"universe_board: gate-passed mismatch for {run_date} — "
        f"board gate_stage=='passed' count={len(passed)} vs "
        f"candidates.json::scanner_tickers count={len(expected)}. "
        f"Missing from board (n={len(missing)}): {sorted(missing)[:10]}"
        f"{'...' if len(missing) > 10 else ''}. "
        f"Unexpected on board (n={len(extra)}): {sorted(extra)[:10]}"
        f"{'...' if len(extra) > 10 else ''}. "
        "This means the gate input (scanner_eval_log) was missing/empty/"
        "malformed this run — refusing to publish a board with a false gate "
        "verdict (no-silent-fails, alpha-engine-config#4820)."
    )


def build_universe_board(
    run_date: str,
    scanner_evals: list[dict],
    *,
    factor_profiles: dict | None = None,
    classification: dict | None = None,
    technical_df: Any = None,
    fundamental_df: Any = None,
    pillar_weights: dict | None = None,
    gate_config: dict | None = None,
    tradeability_config: dict | None = None,
    scanner_tickers: list[str] | None = None,
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
        scanner_tickers: the AUTHORITATIVE ``candidates.json::scanner_tickers``
            list for this run_date — the scanner's own record of which names
            survived the quant filter, independent of ``scanner_evals``.
            When provided (production callers MUST provide it), the builder
            enforces the producer contract invariant: the set of tickers whose
            derived ``gate_stage == "passed"`` must equal this set exactly.
            A mismatch means the gate input (``scanner_eval_log``) was
            missing, empty, or otherwise malformed for this run — the
            historical failure mode (alpha-engine-config#4820) where a run
            silently emitted a board reporting zero (or an undercount of)
            gate-passers while ``candidates.json`` carried the normal count.
            ``None`` skips the check (unit tests exercising other aspects of
            the board without a full candidates.json fixture).
        bucket / s3_client: S3 wiring (defaults: env bucket + a fresh boto3
            client) — injectable for tests.

    Returns the board dict (also the unit under the producer contract test).

    Raises when ``scanner_evals`` is empty (a research run that produced no
    universe is a real fault — the caller's WARN records it), when
    ``scanner_tickers`` is provided and disagrees with the board's own
    ``gate_stage == "passed"`` set (no-silent-fails: a producer must not
    emit an artifact whose gate block silently reads a false verdict), or
    when a universe of >= 20 names publishes ZERO non-null coverage on any of
    ``avg_volume``/``realized_vol_20d``/``atr_pct`` (alpha-engine-config-I7982
    — the signature of a feature-parquet read failure taking the whole
    technical/fundamental mapping class to null; see
    ``_assert_metric_coverage_floor``), or when a universe of >= 20 names
    publishes fewer than 100 distinct cross-sectional values on any of
    ``fcf_yield``/``gross_margin``/``roe``/``revenue_growth_3y``/
    ``eps_growth_3y`` (alpha-engine-config-I8255 — the signature of a
    saturated-placeholder fundamental input; non-null but non-informative;
    see ``_assert_fundamental_distinctness_floor``).
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
    pillar_weights = {p: round(max(0.0, (_num(pillar_weights.get(p)) or 0.0)) / _wt_total, 6) for p in _PILLAR_ORDER}

    from scoring.composite import _PILLAR_TO_FACTOR_KEY

    tech_by_ticker = _index_parquet(technical_df)
    fund_by_ticker = _index_parquet(fundamental_df)

    # ── Pass 1: per-stock base records + collect per-pillar scores ───────────
    records: list[tuple[dict, dict]] = []  # (stock, pillar_scores)
    pillar_scores_by_ticker: dict[str, dict] = {}
    for row in scanner_evals:
        ticker = row.get("ticker")
        if not ticker:
            continue
        profile = factor_profiles.get(ticker, {})
        cls = classification.get(ticker, {})
        tech = tech_by_ticker.get(ticker, {})
        fund = fund_by_ticker.get(ticker, {})

        pillar_scores = {pillar: _num(profile.get(_PILLAR_TO_FACTOR_KEY[pillar])) for pillar in _PILLAR_ORDER}
        pillar_scores_by_ticker[ticker] = pillar_scores
        pillar_coverage = {
            pillar: int(profile[f"{_PILLAR_TO_FACTOR_KEY[pillar][:-6]}_n"])
            for pillar in _PILLAR_ORDER
            if f"{_PILLAR_TO_FACTOR_KEY[pillar][:-6]}_n" in profile
        }

        metrics: dict[str, float | None] = {
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

        records.append(
            (
                {
                    "ticker": ticker,
                    "sector": profile.get("sector") or row.get("sector") or cls.get("sector"),
                    "country": cls.get("country"),
                    "industry": cls.get("industry"),
                    "attractiveness_score": None,  # filled in pass 2 (cross-sectional percentile)
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
                },
                pillar_scores,
            )
        )

    # ── Attractiveness (SOTA z-blend → percentile) via the shared chokepoint ──
    # config-I7272: a name whose every pillar leg is UNDEFINED (no cross-sectional
    # dispersion to z-score against) comes back with attractiveness_score None and
    # is EXCLUDED from the ranking rather than scored — a fabricated 0.0 would have
    # competed against measured names, and a None sorted to one end would still
    # occupy a rank position, silently and systematically. The exclusion is counted
    # and published below: an exclusion nobody can see is the same failure as the
    # fabricated zero it replaced (principles.md §2.7).
    attractiveness = compute_cross_sectional_attractiveness(pillar_scores_by_ticker, pillar_weights)
    for stock, _ in records:
        a = attractiveness.get(stock["ticker"], {})
        stock["attractiveness_raw"] = a.get("attractiveness_raw")
        stock["attractiveness_score"] = a.get("attractiveness_score")
        stock["pillar_contributions"] = a.get("pillar_contributions", {})

    # ── Tradeability (INDEPENDENT √-impact cost score — computed separately and
    #    NEVER folded into the attractiveness blend above, ARCHITECTURE §43) ────
    metrics_by_ticker = {stock["ticker"]: stock["metrics"] for stock, _ in records}
    tradeability = compute_tradeability(metrics_by_ticker, tradeability_config=tradeability_config)
    for stock, _ in records:
        stock["tradeability"] = tradeability.get(stock["ticker"])

    stocks = [s for s, _ in records]
    stocks.sort(key=lambda s: (s["attractiveness_score"] is None, -(s["attractiveness_score"] or 0)))

    if scanner_tickers is not None:
        _assert_gate_passed_matches_scanner_tickers(stocks, scanner_tickers, run_date)

    _assert_metric_coverage_floor(stocks, run_date)

    fundamental_distinctness = _fundamental_distinct_counts(stocks)
    _assert_fundamental_distinctness_floor(stocks, run_date)

    coverage = _attractiveness_coverage(stocks)

    return {
        "schema_version": UNIVERSE_BOARD_SCHEMA_VERSION,
        "as_of": run_date,
        "universe_count": len(stocks),
        "attractiveness_coverage": coverage,
        "attractiveness_method": "sector_neutral_zscore_percentile",
        "tradeability_method": _TRADEABILITY_METHOD,
        "tradeability_reference_notional_usd": _reference_notional(tradeability_config),
        "pillars": list(_PILLAR_ORDER),
        "pillar_weights": pillar_weights,
        "gate_config": gate_config,
        # alpha-engine-config-I8255 deliverable 1: per-field cross-sectional
        # DISTINCT count for every published fundamental metric — emitted
        # unconditionally (healthy steady state is a real number in the
        # hundreds, never omitted), so a dashboard/audit consumer can see
        # spread degrade WITHOUT a producer failure once it drops but stays
        # above the enforcement floor. See _assert_fundamental_distinctness_floor.
        "fundamental_distinctness": fundamental_distinctness,
        "stocks": stocks,
    }


def attractiveness_from_factor_profiles(
    factor_profiles: dict,
    *,
    pillar_weights: dict | None = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> dict:
    """``{ticker: {attractiveness_raw, attractiveness_score, ...}}`` computed
    DIRECTLY from factor profiles via the SSOT ``compute_cross_sectional_attractiveness``.

    The candidate-feed path (config#1400 / ARCHITECTURE §43) needs attractiveness
    scores PRE-selection, before the full board exists — without the board's
    classification / feature-parquet reads. This mirrors ``build_universe_board``'s
    pillar mapping + weight normalization EXACTLY (same chokepoint), so the feed
    ranks on byte-identical numbers to the console board. Pure/read-only.
    """
    if pillar_weights is None:
        pillar_weights = _load_pillar_weights(bucket, s3_client)
    return _attractiveness_from_factor_profiles(
        factor_profiles,
        pillar_weights=pillar_weights,
    )


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
        except Exception as e:
            logger.debug("[universe_board] factor profile key %s unreadable: %s", key, e)
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
    except Exception as e:
        # alpha-engine-config-I7982: this except used to discard `e` entirely
        # (`except Exception:` with no `as e`, message printed no cause) — the
        # single reason a 4-day, 21-of-23-field, 0-of-903-names coverage
        # collapse (2026-08-17..08-20) produced only a content-free WARN and
        # nobody could diagnose it from the log alone. Root cause measured
        # live: the deployed scanner Lambda's `import pyarrow` was failing —
        # pyarrow was an UNDECLARED transitive dependency (pulled in only via
        # edgartools, itself unpinned to a specific pyarrow release), so a
        # routine `pip install` on a dependency bump could silently resolve a
        # pyarrow build incompatible with this image's numpy pin. Fixed at the
        # source by pinning pyarrow directly in requirements.txt (see that
        # file); this `exc_info` is the detection backstop for the next class
        # of read failure this same broad except would otherwise still hide.
        logger.warning(
            "[universe_board] features/%s/%s.parquet not readable — those metrics null: %s",
            run_date,
            name,
            e,
            exc_info=True,
        )
        return None


def _index_parquet(df: Any) -> dict[str, dict]:
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
        board.get("universe_count", 0),
        b,
        dated_key,
    )
    return dated_key


def compute_and_write_universe_board(
    run_date: str,
    scanner_evals: list[dict],
    *,
    scanner_tickers: list[str] | None = None,
    bucket: str | None = None,
    s3_client: Any = None,
) -> str:
    """archive_writer entry point — build from S3-resident inputs + the in-memory
    scanner_evals and write the artifact. Also appends today's attractiveness
    slice to the per-stock history time-series (fail-soft — a history failure
    must not mask the board write). Returns the dated S3 key.

    ``scanner_tickers``: forwarded to :func:`build_universe_board`'s producer
    contract check (alpha-engine-config#4820) — pass the authoritative
    ``candidates.json::scanner_tickers`` list so a degraded gate input raises
    instead of silently writing a board with a false gate verdict.
    """
    board = build_universe_board(
        run_date, scanner_evals, scanner_tickers=scanner_tickers, bucket=bucket, s3_client=s3_client
    )
    key = write_universe_board_to_s3(board, run_date, bucket=bucket, s3_client=s3_client)
    try:
        from scoring.attractiveness_history import append_history, extract_history_rows_from_board

        append_history(extract_history_rows_from_board(board), bucket=bucket, s3_client=s3_client)
    except Exception as e:  # secondary observability — never fail the board write
        logger.warning("[universe_board] attractiveness history append failed (non-fatal): %s", e)
    return key
