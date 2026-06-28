"""Producer contract test for scoring/universe_board.py.

Pins the full-universe scoreboard artifact that the dashboard's filterable
universe board consumes (PR C, crucible-dashboard). Locks:

  1. Artifact shape + schema_version + every ~900 universe member present.
  2. Attractiveness = equal-weight mean of AVAILABLE pillar scores, with
     partial-coverage reallocation (a 4-of-6 name still scores) and null when
     no pillar is available.
  3. The valuation-metric DENORMALIZATION contract (pe_ratio×30, pb_ratio×5,
     debt_to_equity×2, current_ratio×3) that recovers display-raw values from
     the predictor-normalized feature store — the paired guard to
     alpha-engine-data's features/SCHEMA.md / test_schema_contract.py.
  4. Country/industry join from universe_classification, fail-soft to null.
  5. Gate flags + sort order (most attractive first, nulls last).
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


def _scanner_evals():
    return [
        # AAPL — passes the gate, full focus data.
        {"ticker": "AAPL", "sector": "Information Technology", "tech_score": 72.0,
         "current_price": 195.0, "focus_score": 80.0, "focus_stance": "momentum",
         "quant_filter_pass": 1, "filter_fail_reason": None},
        # LIN — Ireland-domiciled, rejected by the liquidity gate.
        {"ticker": "LIN", "sector": "Materials", "tech_score": 40.0,
         "current_price": 460.0, "focus_score": 55.0, "focus_stance": "quality",
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


def _build():
    return build_universe_board(
        "2026-06-28",
        _scanner_evals(),
        factor_profiles=_factor_profiles(),
        classification=_classification(),
        technical_df=_technical_df(),
        fundamental_df=_fundamental_df(),
    )


def test_artifact_shape_and_membership():
    board = _build()
    assert board["schema_version"] == UNIVERSE_BOARD_SCHEMA_VERSION
    assert board["as_of"] == "2026-06-28"
    assert board["universe_count"] == 3  # ALL members, not just gate-passers
    assert board["attractiveness_method"] == "equal_weight_available_pillars"
    assert {s["ticker"] for s in board["stocks"]} == {"AAPL", "LIN", "XYZ"}


def test_attractiveness_equal_weight_with_partial_coverage():
    board = {s["ticker"]: s for s in _build()["stocks"]}
    # AAPL: mean(90,30,85,80,70,60) = 69.17
    assert board["AAPL"]["attractiveness_score"] == 69.17
    # LIN: only 4 pillars present → mean(50,40,30,60) = 45.0 (reallocation)
    assert board["LIN"]["attractiveness_score"] == 45.0
    assert board["LIN"]["pillars"]["growth"] is None
    # XYZ: no profile → null
    assert board["XYZ"]["attractiveness_score"] is None


def test_valuation_denormalization_contract():
    aapl = {s["ticker"]: s for s in _build()["stocks"]}["AAPL"]["metrics"]
    assert aapl["pe"] == 30.0          # 1.0 × 30
    assert aapl["pb"] == 40.0          # 8.0 × 5
    assert aapl["debt_to_equity"] == 1.5   # 0.75 × 2
    assert aapl["current_ratio"] == 1.2    # 0.4 × 3
    # Clean-unit passthroughs.
    assert aapl["fcf_yield"] == 0.04
    assert aapl["roe"] == 1.5
    assert aapl["market_cap"] == 3.0e12
    # Technicals passthrough.
    assert aapl["rsi_14"] == 58.0
    assert aapl["avg_volume"] == 55_000_000.0


def test_country_industry_join_and_gate():
    board = {s["ticker"]: s for s in _build()["stocks"]}
    assert board["LIN"]["country"] == "Ireland"
    assert board["LIN"]["industry"] == "Specialty Chemicals"
    assert board["LIN"]["gate"] == {"quant_filter_pass": 0, "filter_fail_reason": "liquidity"}
    assert board["AAPL"]["gate"]["quant_filter_pass"] == 1
    # XYZ uncovered by classification → null, never guessed.
    assert board["XYZ"]["country"] is None


def test_sort_most_attractive_first_nulls_last():
    tickers = [s["ticker"] for s in _build()["stocks"]]
    assert tickers == ["AAPL", "LIN", "XYZ"]  # 69.17, 45.0, null


def test_missing_classification_degrades_to_null():
    board = build_universe_board(
        "2026-06-28", _scanner_evals(),
        factor_profiles=_factor_profiles(), classification={},
        technical_df=_technical_df(), fundamental_df=_fundamental_df(),
    )
    assert all(s["country"] is None for s in board["stocks"])


def test_empty_scanner_evals_raises():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        build_universe_board("2026-06-28", [], factor_profiles={}, classification={})
