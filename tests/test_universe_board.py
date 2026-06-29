"""Producer contract test for scoring/universe_board.py.

Pins the full-universe scoreboard artifact that the dashboard's filterable
universe board consumes (crucible-dashboard). Locks:

  1. Artifact shape + schema_version (2) + every universe member present.
  2. The SOTA attractiveness method: per-pillar cross-sectional z-score
     (winsorized) → coverage-renormalized weighted blend → terminal
     cross-sectional percentile (0-100). Dispersion restored vs the old
     mean-of-percentiles; null when no pillar is available; pillar_contributions
     sum to attractiveness_raw; equal-weight default + tuned-weight override.
  3. The valuation-metric DENORMALIZATION contract (pe_ratio×30, pb_ratio×5,
     debt_to_equity×2, current_ratio×3).
  4. Country/industry join from universe_classification, fail-soft to null.
  5. Gate flags + gate_config passthrough + per-stock gate_trace/gate_stage
     (value-vs-threshold transparency), + sort order (most attractive first).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_board import (  # noqa: E402
    UNIVERSE_BOARD_SCHEMA_VERSION,
    build_universe_board,
)

# Explicit gate_config so the trace is deterministic (no AWS in tests). Mirrors
# the resolved scanner thresholds (config.get_scanner_params + constants).
_GATE_CONFIG = {
    "min_avg_volume": 500_000,
    "min_price": 0.0,
    "tech_score_min": 60,
    "max_atr_pct": 8.0,
    "momentum_ma200_floor_pct": -15.0,
    "momentum_top_n": 60,
    "deep_value_path_enabled": True,
    "deep_value_max_rsi": 30.0,
    "deep_value_max_atr_pct": 12.0,
    "deep_value_max_candidates": 10,
}


def _scanner_evals():
    return [
        # AAPL — passes the gate; row carries the gate-input values.
        {"ticker": "AAPL", "sector": "Information Technology", "tech_score": 72.0,
         "current_price": 195.0, "avg_volume_20d": 55_000_000.0, "atr_pct": 1.5,
         "price_vs_ma200": 0.10, "focus_score": 80.0, "focus_stance": "momentum",
         "quant_filter_pass": 1, "filter_fail_reason": None},
        # LIN — rejected by the liquidity gate (avg_vol below the floor).
        {"ticker": "LIN", "sector": "Materials", "tech_score": 40.0,
         "current_price": 460.0, "avg_volume_20d": 120_000.0, "atr_pct": 1.2,
         "price_vs_ma200": 0.03, "focus_score": 55.0, "focus_stance": "quality",
         "quant_filter_pass": 0, "filter_fail_reason": "liquidity"},
        # XYZ — no factor profile at all → all pillars null → attractiveness null.
        {"ticker": "XYZ", "sector": "Industrials", "tech_score": None,
         "current_price": 12.0, "focus_score": None, "focus_stance": None,
         "quant_filter_pass": 0, "filter_fail_reason": "no_data"},
    ]


def _factor_profiles():
    return {
        "AAPL": {"sector": "Information Technology", "quality_score": 90.0,
                 "value_score": 30.0, "momentum_score": 85.0, "low_vol_score": 60.0,
                 "growth_score": 80.0, "stewardship_score": 70.0,
                 "quality_n": 4, "momentum_n": 5},
        # LIN — only 4 of 6 pillars present (no growth/stewardship) → reallocation.
        "LIN": {"sector": "Materials", "quality_score": 50.0, "value_score": 40.0,
                "momentum_score": 30.0, "low_vol_score": 60.0},
    }


def _classification():
    return {
        "AAPL": {"sector": "Information Technology", "country": "United States",
                 "industry": "Consumer Electronics"},
        "LIN": {"sector": "Materials", "country": "Ireland",
                "industry": "Specialty Chemicals"},
    }


def _fundamental_df():
    return pd.DataFrame([
        # pe_ratio is PE/30; pb_ratio is PB/5; debt_to_equity is D/E÷2; current_ratio is CR÷3.
        {"ticker": "AAPL", "pe_ratio": 1.0, "pb_ratio": 8.0, "debt_to_equity": 0.75,
         "current_ratio": 0.4, "fcf_yield": 0.04, "roe": 1.5, "gross_margin": 0.44,
         "revenue_growth_3y": 0.08, "eps_growth_3y": 0.10, "market_cap_raw": 3.0e12,
         "dividend_yield": 0.005, "payout_ratio": 0.15},
        {"ticker": "LIN", "pe_ratio": 1.2, "pb_ratio": 1.0, "debt_to_equity": 0.5,
         "current_ratio": 0.3, "fcf_yield": 0.05, "roe": 0.16, "gross_margin": 0.45,
         "revenue_growth_3y": 0.06, "eps_growth_3y": 0.09, "market_cap_raw": 2.2e11,
         "dividend_yield": 0.013, "payout_ratio": 0.45},
    ])


def _technical_df():
    return pd.DataFrame([
        {"ticker": "AAPL", "rsi_14": 58.0, "momentum_20d": 0.03, "return_60d": 0.08,
         "return_120d": 0.12, "realized_vol_20d": 0.22, "atr_14_pct": 0.015,
         "dist_from_52w_high": -0.04, "price_vs_ma200": 0.10, "beta_60d": 1.2,
         "avg_volume_20d_raw": 55_000_000.0},
        {"ticker": "LIN", "rsi_14": 47.0, "momentum_20d": -0.01, "return_60d": 0.02,
         "return_120d": 0.05, "realized_vol_20d": 0.18, "atr_14_pct": 0.012,
         "dist_from_52w_high": -0.09, "price_vs_ma200": 0.03, "beta_60d": 0.9,
         "avg_volume_20d_raw": 3_000_000.0},
    ])


def _build(**overrides):
    kwargs = dict(
        factor_profiles=_factor_profiles(),
        classification=_classification(),
        technical_df=_technical_df(),
        fundamental_df=_fundamental_df(),
        gate_config=_GATE_CONFIG,
    )
    kwargs.update(overrides)
    return build_universe_board("2026-06-28", _scanner_evals(), **kwargs)


def _by_ticker(board):
    return {s["ticker"]: s for s in board["stocks"]}


# ── 1. Shape / membership ────────────────────────────────────────────────────

def test_artifact_shape_and_membership():
    board = _build()
    assert board["schema_version"] == UNIVERSE_BOARD_SCHEMA_VERSION == 2
    assert board["as_of"] == "2026-06-28"
    assert board["universe_count"] == 3  # ALL members, not just gate-passers
    assert board["attractiveness_method"] == "sector_neutral_zscore_percentile"
    assert {s["ticker"] for s in board["stocks"]} == {"AAPL", "LIN", "XYZ"}
    assert board["pillars"][0] == "quality"


# ── 2. SOTA attractiveness ───────────────────────────────────────────────────

def test_attractiveness_is_cross_sectional_percentile():
    b = _by_ticker(_build())
    # AAPL z-blend (0.1667) > LIN z-blend (-0.25) → percentiles 100 / 50.
    assert b["AAPL"]["attractiveness_score"] == 100.0
    assert b["LIN"]["attractiveness_score"] == 50.0
    # XYZ has no pillars → null score + null raw (never fabricated).
    assert b["XYZ"]["attractiveness_score"] is None
    assert b["XYZ"]["attractiveness_raw"] is None


def test_attractiveness_monotonic_in_blend():
    # Higher signed blend must map to >= percentile.
    b = _by_ticker(_build())
    assert b["AAPL"]["attractiveness_raw"] > b["LIN"]["attractiveness_raw"]
    assert b["AAPL"]["attractiveness_score"] >= b["LIN"]["attractiveness_score"]


def test_pillar_contributions_sum_to_raw():
    b = _by_ticker(_build())
    for tkr in ("AAPL", "LIN"):
        contribs = b[tkr]["pillar_contributions"]
        assert contribs, tkr
        assert round(sum(contribs.values()), 3) == round(b[tkr]["attractiveness_raw"], 3)
    # LIN only blends its 4 available pillars (coverage reallocation).
    assert set(b["LIN"]["pillar_contributions"]) == {"quality", "value", "momentum", "defensiveness"}
    assert b["LIN"]["pillars"]["growth"] is None


def test_pillar_weights_default_equal():
    board = _build()
    w = board["pillar_weights"]
    assert set(w) == {"quality", "value", "momentum", "growth", "stewardship", "defensiveness"}
    assert abs(sum(w.values()) - 1.0) < 1e-3  # 1/6 rounded to 6dp can't sum to exactly 1.0
    assert all(abs(v - 1 / 6) < 1e-4 for v in w.values())


def test_tuned_weights_override_normalizes_and_shifts_blend():
    # Raw (unnormalized) tuned weights — heavy on value/quality. The producer
    # must normalize to sum 1.0 and the blend must change vs equal-weight.
    raw = {"quality": 3.0, "value": 3.0, "momentum": 1.0,
           "growth": 1.0, "stewardship": 1.0, "defensiveness": 1.0}
    board = _build(pillar_weights=raw)
    w = board["pillar_weights"]
    assert round(sum(w.values()), 6) == 1.0
    assert w["quality"] == 0.3 and w["momentum"] == 0.1
    # AAPL value-z is negative; over-weighting value pulls its raw blend down
    # vs the equal-weight 0.1667.
    aapl = _by_ticker(board)["AAPL"]
    assert aapl["attractiveness_raw"] < 0.1667


def test_dispersion_restored_uses_full_range():
    # A constructed universe where the blend strictly orders names → the
    # terminal percentile must span ~0..100 (the old mean-of-percentiles
    # compressed toward 50; this is the institutional fix).
    n = 11
    evals = [{"ticker": f"T{i}", "sector": "X", "tech_score": float(i),
              "current_price": 10.0, "quant_filter_pass": 1,
              "filter_fail_reason": None} for i in range(n)]
    # Monotone pillar values per name so blends strictly increase with i.
    profiles = {f"T{i}": {"sector": "X", "quality_score": float(i * 10),
                          "value_score": float(i * 10), "momentum_score": float(i * 10),
                          "low_vol_score": float(i * 10), "growth_score": float(i * 10),
                          "stewardship_score": float(i * 10)} for i in range(n)}
    board = build_universe_board("2026-06-28", evals, factor_profiles=profiles,
                                 classification={}, technical_df=None,
                                 fundamental_df=None, gate_config=_GATE_CONFIG)
    scores = [s["attractiveness_score"] for s in board["stocks"]]
    assert max(scores) == 100.0
    assert min(scores) < 15.0  # bottom name lands near 0, not stuck near 50
    # Strictly ordered (most attractive first).
    assert scores == sorted(scores, reverse=True)


# ── 3. Denormalization contract ──────────────────────────────────────────────

def test_valuation_denormalization_contract():
    aapl = _by_ticker(_build())["AAPL"]["metrics"]
    assert aapl["pe"] == 30.0          # 1.0 × 30
    assert aapl["pb"] == 40.0          # 8.0 × 5
    assert aapl["debt_to_equity"] == 1.5   # 0.75 × 2
    assert aapl["current_ratio"] == 1.2    # 0.4 × 3
    assert aapl["fcf_yield"] == 0.04
    assert aapl["roe"] == 1.5
    assert aapl["market_cap"] == 3.0e12
    assert aapl["rsi_14"] == 58.0
    assert aapl["avg_volume"] == 55_000_000.0


# ── 4. Classification join ───────────────────────────────────────────────────

def test_country_industry_join():
    b = _by_ticker(_build())
    assert b["LIN"]["country"] == "Ireland"
    assert b["LIN"]["industry"] == "Specialty Chemicals"
    assert b["XYZ"]["country"] is None  # uncovered → null, never guessed


def test_missing_classification_degrades_to_null():
    board = _build(classification={})
    assert all(s["country"] is None for s in board["stocks"])


# ── 5. Gate transparency ─────────────────────────────────────────────────────

def test_gate_flags_and_config_passthrough():
    board = _build()
    assert board["gate_config"] == _GATE_CONFIG
    b = _by_ticker(board)
    assert b["AAPL"]["gate"]["quant_filter_pass"] == 1
    assert b["LIN"]["gate"] == {"quant_filter_pass": 0, "filter_fail_reason": "liquidity"}


def test_gate_trace_value_vs_threshold():
    b = _by_ticker(_build())
    # AAPL clears every value-gate; terminal stage = passed.
    aapl = b["AAPL"]
    assert aapl["gate_stage"] == "passed"
    liq = next(g for g in aapl["gate_trace"] if g["stage"] == "liquidity")
    assert liq["value"] == 55_000_000.0 and liq["threshold"] == 500_000 and liq["pass"] is True
    vol = next(g for g in aapl["gate_trace"] if g["stage"] == "volatility")
    assert vol["op"] == "<=" and vol["pass"] is True
    # LIN fails liquidity (avg_vol 120k < 500k floor) and that is its terminal stage.
    lin = b["LIN"]
    assert lin["gate_stage"] == "liquidity"
    lin_liq = next(g for g in lin["gate_trace"] if g["stage"] == "liquidity")
    assert lin_liq["value"] == 120_000.0 and lin_liq["pass"] is False
    # XYZ never had data.
    assert b["XYZ"]["gate_stage"] == "no_data"


def test_gate_trace_thresholds_null_when_config_absent():
    board = _build(gate_config={})
    aapl = _by_ticker(board)["AAPL"]
    liq = next(g for g in aapl["gate_trace"] if g["stage"] == "liquidity")
    assert liq["threshold"] is None and liq["pass"] is None  # value present, no threshold to compare


# ── 6. Sort order + guards ───────────────────────────────────────────────────

def test_sort_most_attractive_first_nulls_last():
    tickers = [s["ticker"] for s in _build()["stocks"]]
    assert tickers == ["AAPL", "LIN", "XYZ"]  # 100, 50, null


def test_empty_scanner_evals_raises():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        build_universe_board("2026-06-28", [], factor_profiles={}, classification={})
