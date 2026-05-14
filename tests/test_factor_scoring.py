"""Tests for scoring/factor_scoring.py — Phase 1c of the institutional
factor substrate (260513 plan).

Validates the within-sector percentile ranking + composite combination
+ partial-data weight reallocation behavior. Production data integration
(reading parquets, writing S3) is exercised via the
compute_and_write_factor_profiles function with mocked S3.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scoring.factor_scoring import (
    _COMPOSITE_DEFS,
    _within_sector_pct_rank,
    compute_factor_composites,
    read_factor_profiles_from_s3,
    write_factor_profiles_to_s3,
)


# ── _within_sector_pct_rank ─────────────────────────────────────────────────


def test_pct_rank_basic_ascending():
    """Higher raw value → higher percentile rank (when invert=False)."""
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "sector": ["Tech"] * 4,
        "roe": [0.05, 0.10, 0.20, 0.30],
    })
    ranks = _within_sector_pct_rank(df, "roe", "sector", invert=False)
    assert ranks.tolist() == [25.0, 50.0, 75.0, 100.0]


def test_pct_rank_inverted():
    """Higher raw value → LOWER percentile rank (when invert=True, e.g. PE)."""
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "sector": ["Tech"] * 4,
        "pe_ratio": [10.0, 20.0, 30.0, 40.0],
    })
    ranks = _within_sector_pct_rank(df, "pe_ratio", "sector", invert=True)
    # invert: highest PE (40) → lowest score (0% inverted = lowest rank)
    assert ranks.iloc[3] < ranks.iloc[0]


def test_pct_rank_within_sector_isolated():
    """Each sector's ranks are computed independently."""
    df = pd.DataFrame({
        "ticker": ["LOW_T", "HI_T", "LOW_E", "HI_E"],
        "sector": ["Tech", "Tech", "Energy", "Energy"],
        "roe": [0.05, 0.50, 0.10, 0.40],
    })
    ranks = _within_sector_pct_rank(df, "roe", "sector", invert=False)
    # Within Tech: LOW_T(0.05) < HI_T(0.50) → ranks 50 and 100
    # Within Energy: LOW_E(0.10) < HI_E(0.40) → ranks 50 and 100
    # Cross-sector: LOW_E(0.10 ROE) doesn't outrank HI_T (0.50 ROE) by absolute
    # but they're in different sectors so their ranks are independent
    assert ranks.iloc[0] == 50.0   # LOW_T at 50% within Tech
    assert ranks.iloc[1] == 100.0  # HI_T at 100% within Tech
    assert ranks.iloc[2] == 50.0   # LOW_E at 50% within Energy
    assert ranks.iloc[3] == 100.0  # HI_E at 100% within Energy


def test_pct_rank_missing_column_yields_nan():
    """Asking for a non-existent column returns NaN per row."""
    df = pd.DataFrame({"ticker": ["A", "B"], "sector": ["Tech", "Tech"]})
    ranks = _within_sector_pct_rank(df, "nonexistent", "sector")
    assert ranks.isna().all()


def test_pct_rank_nan_propagates():
    """NaN raw values produce NaN ranks (consumers handle reallocation)."""
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "sector": ["Tech"] * 3,
        "roe": [0.10, float("nan"), 0.30],
    })
    ranks = _within_sector_pct_rank(df, "roe", "sector")
    assert pd.isna(ranks.iloc[1])
    assert pd.notna(ranks.iloc[0]) and pd.notna(ranks.iloc[2])


# ── compute_factor_composites ───────────────────────────────────────────────


def _make_test_dfs():
    """Build minimal technical + fundamental DataFrames for 4 tickers."""
    tickers = ["NVDA", "MSFT", "JNJ", "PFE"]
    technical = pd.DataFrame({
        "ticker": tickers,
        "date": ["2026-05-13"] * 4,
        "momentum_20d": [0.15, 0.05, -0.02, -0.08],
        "momentum_5d": [0.05, 0.02, -0.01, -0.03],
        "return_60d": [0.30, 0.10, -0.05, -0.10],
        "return_120d": [0.50, 0.20, -0.08, -0.15],
        "dist_from_52w_high": [-0.02, -0.10, -0.20, -0.35],
        "realized_vol_20d": [0.40, 0.20, 0.15, 0.10],
        "vol_ratio_10_60": [1.30, 1.05, 0.95, 0.80],
        "atr_14_pct": [3.5, 2.0, 1.5, 1.0],
    })
    fundamental = pd.DataFrame({
        "ticker": tickers,
        "date": ["2026-05-13"] * 4,
        "roe": [0.40, 0.35, 0.20, 0.05],
        "debt_to_equity": [0.30, 0.50, 1.20, 2.50],
        "gross_margin": [0.75, 0.65, 0.45, 0.30],
        "current_ratio": [4.0, 2.5, 1.5, 0.9],
        "pe_ratio": [50.0, 30.0, 18.0, 10.0],
        "pb_ratio": [40.0, 12.0, 4.0, 1.5],
        "fcf_yield": [0.02, 0.04, 0.06, 0.10],
    })
    return technical, fundamental


def test_compute_returns_one_row_per_ticker_with_all_composites():
    """All 4 composites populated for each ticker."""
    tech, fund = _make_test_dfs()
    sector_map = {"NVDA": "Tech", "MSFT": "Tech", "JNJ": "Healthcare", "PFE": "Healthcare"}
    out = compute_factor_composites(tech, fund, sector_map)

    assert len(out) == 4
    for col in ("quality_score", "momentum_score", "low_vol_score", "value_score"):
        assert col in out.columns
        assert out[col].notna().all(), f"{col} has NaN — partial data path broken"


def test_quality_composite_within_sector_separation():
    """NVDA + MSFT (Tech): both top-quality but ranked relative to each
    other; JNJ + PFE (Healthcare): JNJ better-quality than PFE."""
    tech, fund = _make_test_dfs()
    sector_map = {"NVDA": "Tech", "MSFT": "Tech", "JNJ": "Healthcare", "PFE": "Healthcare"}
    out = compute_factor_composites(tech, fund, sector_map)
    out = out.set_index("ticker")
    # Within Tech: NVDA has higher ROE+gross+current, lower D/E vs MSFT
    assert out.loc["NVDA", "quality_score"] > out.loc["MSFT", "quality_score"]
    # Within Healthcare: JNJ better than PFE
    assert out.loc["JNJ", "quality_score"] > out.loc["PFE", "quality_score"]
    # Cross-sector ranks are independent — NVDA vs JNJ comparison not meaningful


def test_low_vol_inverts_correctly():
    """Lowest-vol name has highest low_vol_score (inverted)."""
    tech, fund = _make_test_dfs()
    sector_map = {"NVDA": "Tech", "MSFT": "Tech", "JNJ": "Healthcare", "PFE": "Healthcare"}
    out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")
    # Within Tech: MSFT (0.20 vol) less volatile than NVDA (0.40) → MSFT higher low_vol_score
    assert out.loc["MSFT", "low_vol_score"] > out.loc["NVDA", "low_vol_score"]


def test_value_score_inverts_pe_pb():
    """Lower PE/PB (cheap) → higher value_score."""
    tech, fund = _make_test_dfs()
    sector_map = {"NVDA": "Tech", "MSFT": "Tech", "JNJ": "Healthcare", "PFE": "Healthcare"}
    out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")
    # MSFT cheaper than NVDA on PE+PB → higher value_score
    assert out.loc["MSFT", "value_score"] > out.loc["NVDA", "value_score"]


def test_partial_data_reallocates_weights():
    """A ticker missing one factor should still get a composite from
    the others, with the missing factor's weight reallocated pro-rata."""
    tickers = ["A", "B", "C"]
    tech = pd.DataFrame({
        "ticker": tickers,
        "date": ["2026-05-13"] * 3,
        "momentum_20d": [0.10, 0.05, 0.01],
        "return_60d": [0.20, 0.10, 0.02],
        "return_120d": [float("nan"), 0.20, 0.05],  # A missing return_120d
        "dist_from_52w_high": [-0.05, -0.10, -0.20],
        "momentum_5d": [0.02, 0.01, 0.0],
        "realized_vol_20d": [0.15, 0.20, 0.30],
        "vol_ratio_10_60": [1.0, 1.1, 1.2],
        "atr_14_pct": [1.5, 2.0, 3.0],
    })
    fund = pd.DataFrame({"ticker": tickers, "date": ["2026-05-13"] * 3})
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech"}
    out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")
    # All three should still get a momentum_score despite A missing return_120d
    assert out.loc["A", "momentum_score"] > 0
    # n column shows 4 (instead of 5) for A
    assert out.loc["A", "momentum_n"] == 4
    assert out.loc["B", "momentum_n"] == 5


def test_unknown_sector_assigned_to_unknown_group():
    """Tickers without a sector mapping default to 'Unknown' and rank together."""
    tech, fund = _make_test_dfs()
    sector_map = {"NVDA": "Tech"}  # only NVDA mapped
    out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")
    # MSFT, JNJ, PFE all in 'Unknown', ranked together
    unknowns = out[out["sector"] == "Unknown"]
    assert len(unknowns) == 3


# ── S3 IO ───────────────────────────────────────────────────────────────────


def test_write_serializes_payload_and_writes_dated_plus_latest():
    """Verifies the S3 write puts both {date}/by_ticker.json AND latest.json."""
    profiles = pd.DataFrame({
        "ticker": ["NVDA", "MSFT"],
        "sector": ["Tech", "Tech"],
        "quality_score": [85.0, 60.0],
        "momentum_score": [90.0, 50.0],
        "low_vol_score": [20.0, 70.0],
        "value_score": [10.0, 40.0],
        "quality_n": [4, 4],
        "momentum_n": [5, 5],
        "low_vol_n": [3, 3],
        "value_n": [3, 3],
    })

    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        key = write_factor_profiles_to_s3(profiles, "2026-05-13", bucket="test-bucket")

    assert key == "factors/profiles/2026-05-13/by_ticker.json"
    assert mock_s3.put_object.call_count == 2  # dated + latest
    keys_written = [c.kwargs["Key"] for c in mock_s3.put_object.call_args_list]
    assert "factors/profiles/2026-05-13/by_ticker.json" in keys_written
    assert "factors/profiles/latest.json" in keys_written

    # Validate the JSON shape
    body = mock_s3.put_object.call_args_list[0].kwargs["Body"]
    payload = json.loads(body)
    assert "NVDA" in payload
    assert payload["NVDA"]["quality_score"] == 85.0
    assert payload["NVDA"]["sector"] == "Tech"


def test_write_drops_nan_scores_from_payload():
    """Tickers with no factor data (all NaN) shouldn't have those keys in
    the JSON — consumers can detect missing factors as 'absent key'."""
    profiles = pd.DataFrame({
        "ticker": ["BROKEN"],
        "sector": ["Unknown"],
        "quality_score": [float("nan")],
        "momentum_score": [50.0],
        "low_vol_score": [float("nan")],
        "value_score": [float("nan")],
        "quality_n": [0],
        "momentum_n": [3],
        "low_vol_n": [0],
        "value_n": [0],
    })
    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        write_factor_profiles_to_s3(profiles, "2026-05-13", bucket="test-bucket")
    body = mock_s3.put_object.call_args_list[0].kwargs["Body"]
    payload = json.loads(body)
    assert "quality_score" not in payload["BROKEN"]
    assert "momentum_score" in payload["BROKEN"]


def test_read_returns_none_on_404():
    """No-such-key → returns None gracefully (consumers skip factor blend)."""
    from botocore.exceptions import ClientError
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
    )
    with patch("boto3.client", return_value=mock_s3):
        result = read_factor_profiles_from_s3(bucket="test-bucket")
    assert result is None


def test_read_returns_payload_on_success():
    """Happy path: payload returned as dict-of-dicts."""
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps({
            "NVDA": {"sector": "Tech", "quality_score": 85.0}
        }).encode())
    }
    with patch("boto3.client", return_value=mock_s3):
        result = read_factor_profiles_from_s3(bucket="test-bucket")
    assert result == {"NVDA": {"sector": "Tech", "quality_score": 85.0}}


# ── Composite weight integrity ──────────────────────────────────────────────


def test_each_composite_weights_sum_to_one():
    """Hard-pin: every composite definition's weights sum to 1.0 — guards
    against accidental drift on future composite-spec edits."""
    for composite, components in _COMPOSITE_DEFS.items():
        total = sum(weight for _, weight, _ in components)
        assert abs(total - 1.0) < 1e-9, f"{composite} weights sum to {total}, not 1.0"
