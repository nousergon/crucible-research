"""Tests for the get_factor_profile @tool — factor-substrate Phase 2 wiring.

Plan doc: ``alpha-engine-docs/private/factor-substrate-260513.md`` Phase 2.
Scanner-placement arc dependency: PR 3 in ``alpha-engine-docs/private/scanner-260514.md``.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from agents.sector_teams.quant_tools import create_quant_tools


@pytest.fixture
def regime_weights():
    """Mirrors alpha-engine-config research/scoring.yaml aggregator.factor_blend."""
    return {
        "bull": {
            "momentum_score": 0.40,
            "quality_score": 0.30,
            "value_score": 0.20,
            "low_vol_score": -0.10,
        },
        "bear": {
            "low_vol_score": 0.40,
            "quality_score": 0.30,
            "momentum_score": -0.20,
            "value_score": 0.10,
        },
        "neutral": {
            "momentum_score": 0.25,
            "quality_score": 0.25,
            "value_score": 0.25,
            "low_vol_score": 0.25,
        },
    }


@pytest.fixture
def factor_profiles():
    return {
        "NVDA": {
            "sector": "Technology",
            "quality_score": 70.0,
            "momentum_score": 95.0,
            "value_score": 20.0,
            "low_vol_score": 25.0,
            "quality_n": 4,
            "momentum_n": 5,
            "value_n": 3,
            "low_vol_n": 3,
        },
        "JNJ": {
            "sector": "Health Care",
            "quality_score": 88.0,
            "momentum_score": 35.0,
            "value_score": 65.0,
            "low_vol_score": 82.0,
            "quality_n": 4,
            "momentum_n": 5,
            "value_n": 3,
            "low_vol_n": 3,
        },
    }


def _tools(factor_profiles, regime, regime_weights, price_data=None):
    """Helper: create the tool list with all dependencies in context."""
    if price_data is None:
        # All factor_profiles tickers in the price_data keys so
        # _validate_tickers accepts them.
        price_data = {t: pd.DataFrame() for t in factor_profiles}
    return create_quant_tools({
        "price_data": price_data,
        "technical_scores": {},
        "factor_profiles": factor_profiles,
        "market_regime": regime,
        "factor_blend_regime_weights": regime_weights,
    })


def _find_tool(tools, name):
    """Helper: pick a specific tool out of the returned list."""
    return next(t for t in tools if t.name == name)


# ── Tool presence + signature ───────────────────────────────────────────────


def test_get_factor_profile_in_tools_list(factor_profiles, regime_weights):
    tools = _tools(factor_profiles, "bull", regime_weights)
    names = [t.name for t in tools]
    assert "get_factor_profile" in names


def test_get_factor_profile_accepts_list_of_tickers(factor_profiles, regime_weights):
    """Pattern consistency with other quant tools — takes list[str]."""
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    raw = tool.invoke({"tickers": ["NVDA"]})
    result = json.loads(raw)
    assert "NVDA" in result


# ── Returned fields ──────────────────────────────────────────────────────────


def test_returns_all_four_composites(factor_profiles, regime_weights):
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    entry = result["NVDA"]
    assert entry["quality_score"] == 70.0
    assert entry["momentum_score"] == 95.0
    assert entry["value_score"] == 20.0
    assert entry["low_vol_score"] == 25.0


def test_returns_coverage_counts(factor_profiles, regime_weights):
    """_n columns surfaced so the agent can reason about data-completeness band."""
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    entry = result["NVDA"]
    assert entry["quality_n"] == 4
    assert entry["momentum_n"] == 5


def test_returns_sector(factor_profiles, regime_weights):
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["JNJ"]}))
    assert result["JNJ"]["sector"] == "Health Care"


def test_returns_stance_dominant_factor(factor_profiles, regime_weights):
    """Stance = dominant factor (highest within-sector composite)."""
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA", "JNJ"]}))
    # NVDA's momentum_score (95) is the highest → stance "momentum"
    assert result["NVDA"]["stance"] == "momentum"
    # JNJ's quality_score (88) is the highest → stance "quality"
    assert result["JNJ"]["stance"] == "quality"


def test_returns_regime_blended_focus_score(factor_profiles, regime_weights):
    """focus_score = compute_factor_subscore applied to the profile."""
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    entry = result["NVDA"]
    assert entry["focus_score"] is not None
    assert 0.0 <= entry["focus_score"] <= 100.0
    assert entry["regime"] == "bull"


def test_bull_regime_favors_momentum_heavy(factor_profiles, regime_weights):
    """Sanity check: NVDA (momentum=95) outscores JNJ (low_vol-defensive) in BULL."""
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA", "JNJ"]}))
    assert result["NVDA"]["focus_score"] > result["JNJ"]["focus_score"]


def test_bear_regime_favors_low_vol(factor_profiles, regime_weights):
    """Same tickers, BEAR regime → JNJ outscores NVDA."""
    tools = _tools(factor_profiles, "bear", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA", "JNJ"]}))
    assert result["JNJ"]["focus_score"] > result["NVDA"]["focus_score"]


def test_factor_blend_breakdown_surfaced(factor_profiles, regime_weights):
    """Breakdown returned for observability — agent can see per-factor contribution."""
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    assert "factor_blend_breakdown" in result["NVDA"]
    breakdown = result["NVDA"]["factor_blend_breakdown"]
    # BULL has 4 non-zero weights → all 4 factors contribute
    assert set(breakdown.keys()) == {
        "momentum_score", "quality_score", "value_score", "low_vol_score",
    }


# ── Graceful degrade ─────────────────────────────────────────────────────────


def test_missing_profile_returns_error_for_that_ticker(factor_profiles, regime_weights):
    """Ticker not in factor_profiles → per-ticker error (not crash).

    _validate_tickers writes errors as plain-string values (LLM-friendly) for
    rejected tickers and dict entries for resolved ones. Mirror that contract
    in the test rather than coercing both to dict.
    """
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    raw = tool.invoke({"tickers": ["NVDA", "MYSTERY"]})
    result = json.loads(raw)
    # MYSTERY fails ticker validation → string error keyed under the ticker
    assert "MYSTERY" in result
    assert "unknown ticker" in result["MYSTERY"].lower()
    # NVDA still resolves
    assert isinstance(result["NVDA"], dict)
    assert result["NVDA"]["quality_score"] == 70.0


def test_no_factor_profile_in_universe_path(regime_weights):
    """Ticker in price_data universe but missing from factor_profiles →
    per-ticker error explaining the missing artifact, not a crash."""
    factor_profiles = {}  # empty — no profiles fetched / available
    price_data = {"NVDA": pd.DataFrame()}
    tools = create_quant_tools({
        "price_data": price_data,
        "technical_scores": {},
        "factor_profiles": factor_profiles,
        "market_regime": "bull",
        "factor_blend_regime_weights": regime_weights,
    })
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    assert "NVDA" in result
    assert "error" in result["NVDA"]


def test_lazy_s3_read_when_factor_profiles_absent_from_context(regime_weights):
    """When context doesn't carry factor_profiles, the tool factory reads
    factors/profiles/latest.json once at create time."""
    profiles = {
        "NVDA": {
            "sector": "Technology", "quality_score": 70.0, "momentum_score": 90.0,
            "value_score": 20.0, "low_vol_score": 25.0,
        },
    }
    with patch(
        "agents.sector_teams.quant_tools.read_factor_profiles_from_s3",
        return_value=profiles,
    ) as mock_read:
        tools = create_quant_tools({
            "price_data": {"NVDA": pd.DataFrame()},
            "technical_scores": {},
            "market_regime": "bull",
            "factor_blend_regime_weights": regime_weights,
            # factor_profiles intentionally absent → lazy S3 fallback
        })
        # Read happens once at create-time (factory call), not per-invocation
        assert mock_read.call_count == 1
        tool = _find_tool(tools, "get_factor_profile")
        result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
        assert result["NVDA"]["quality_score"] == 70.0
        # No additional reads on tool invocation
        assert mock_read.call_count == 1


def test_no_regime_weights_returns_null_focus_score(factor_profiles):
    """factor_blend_regime_weights absent → composites still surfaced, focus_score=None."""
    tools = create_quant_tools({
        "price_data": {t: pd.DataFrame() for t in factor_profiles},
        "technical_scores": {},
        "factor_profiles": factor_profiles,
        "market_regime": "bull",
        # factor_blend_regime_weights intentionally absent
    })
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["NVDA"]}))
    assert result["NVDA"]["quality_score"] == 70.0
    assert result["NVDA"]["focus_score"] is None


def test_validates_unknown_ticker(factor_profiles, regime_weights):
    """Pattern-consistent with other tools: unknown ticker → validation error."""
    tools = _tools(factor_profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": ["CARRIER"]}))  # should be CARR
    assert "CARRIER" in result
    assert "unknown ticker" in str(result["CARRIER"]).lower()


def test_truncates_to_20_tickers(factor_profiles, regime_weights):
    """Tool caps at 20 tickers per invocation (cost/latency guard rail)."""
    # Build 25 tickers with profiles
    profiles = {
        f"T{i:02d}": {
            "sector": "Technology",
            "quality_score": 50.0, "momentum_score": 50.0,
            "value_score": 50.0, "low_vol_score": 50.0,
        }
        for i in range(25)
    }
    tools = _tools(profiles, "bull", regime_weights)
    tool = _find_tool(tools, "get_factor_profile")
    result = json.loads(tool.invoke({"tickers": [f"T{i:02d}" for i in range(25)]}))
    # 20 resolved, 5 silently dropped
    resolved = [k for k, v in result.items() if "quality_score" in v]
    assert len(resolved) == 20
