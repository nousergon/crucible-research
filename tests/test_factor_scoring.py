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
    _DERIVED_FACTOR_DEFS,
    _add_derived_factors,
    _within_sector_pct_rank,
    compute_and_write_factor_profiles,
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
        # Phase 3a fundamental fields backing the Phase 3b composites.
        # NVDA growth-leader on revenue + EPS; PFE the laggard. NVDA retains
        # (low payout) + reinvests; PFE high payout (dividend-heavy pharma).
        "revenue_growth_3y": [0.45, 0.18, 0.06, -0.02],
        "eps_growth_3y": [0.60, 0.20, 0.05, -0.10],
        "payout_ratio": [0.0, 0.30, 0.55, 0.85],
        "dividend_yield": [0.0, 0.008, 0.025, 0.045],
        "capex_growth_5y": [0.35, 0.12, 0.04, -0.05],
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


# ── Phase 3b of attractiveness-pillars-260520 — Growth + Stewardship
# composites. The 4 legacy composites continue to work identically (guarded
# by the existing tests above); these tests target the 2 new composites
# and the derived sustainable_growth_rate factor.


class TestPillarComposites:
    """Phase 3b composites built on top of Phase 3a fundamental fields."""

    def test_definitions_include_growth_and_stewardship(self):
        """The 6-composite contract: legacy 4 + growth + stewardship."""
        assert "growth_score" in _COMPOSITE_DEFS
        assert "stewardship_score" in _COMPOSITE_DEFS
        # Sanity: 6 total composites; existing 4 preserved.
        assert set(_COMPOSITE_DEFS.keys()) == {
            "quality_score",
            "momentum_score",
            "low_vol_score",
            "value_score",
            "growth_score",
            "stewardship_score",
        }

    def test_growth_composite_components(self):
        """Growth composite must reference the 4 expected raw factors,
        including the derived sustainable_growth_rate."""
        components = _COMPOSITE_DEFS["growth_score"]
        component_cols = {col for col, _, _ in components}
        assert component_cols == {
            "revenue_growth_3y",
            "eps_growth_3y",
            "sustainable_growth_rate",
            "capex_growth_5y",
        }

    def test_stewardship_composite_components(self):
        """Stewardship is thin-quant by design — 2 components only."""
        components = _COMPOSITE_DEFS["stewardship_score"]
        component_cols = {col for col, _, _ in components}
        assert component_cols == {"payout_ratio", "capex_growth_5y"}
        # payout_ratio MUST be inverted (low payout = better stewardship for
        # the cross-sectional rank); capex_growth_5y NOT inverted (sustained
        # reinvestment is good stewardship).
        for col, _, invert in components:
            if col == "payout_ratio":
                assert invert is True, "payout_ratio must be inverted"
            if col == "capex_growth_5y":
                assert invert is False

    def test_derived_factor_defs_documents_sustainable_growth_rate(self):
        """_DERIVED_FACTOR_DEFS is the audit trail for non-raw columns."""
        assert "sustainable_growth_rate" in _DERIVED_FACTOR_DEFS

    def test_add_derived_factors_computes_sustainable_growth_rate(self):
        """sustainable_growth_rate = roe × (1 - payout_ratio); NaN inputs
        propagate to NaN outputs."""
        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "roe": [0.30, 0.10, 0.20, float("nan")],
            "payout_ratio": [0.20, 0.50, float("nan"), 0.40],
        })
        out = _add_derived_factors(df)
        # A: 0.30 × (1 - 0.20) = 0.24
        # B: 0.10 × (1 - 0.50) = 0.05
        # C: NaN payout → NaN
        # D: NaN roe → NaN
        assert out.loc[0, "sustainable_growth_rate"] == pytest.approx(0.24)
        assert out.loc[1, "sustainable_growth_rate"] == pytest.approx(0.05)
        assert pd.isna(out.loc[2, "sustainable_growth_rate"])
        assert pd.isna(out.loc[3, "sustainable_growth_rate"])

    def test_add_derived_factors_handles_missing_columns(self):
        """Pre-Phase-3a-deploy: roe or payout_ratio missing → emit NaN
        sustainable_growth_rate column rather than crashing."""
        # Missing payout_ratio
        df_no_payout = pd.DataFrame({"ticker": ["A"], "roe": [0.20]})
        out = _add_derived_factors(df_no_payout)
        assert "sustainable_growth_rate" in out.columns
        assert pd.isna(out["sustainable_growth_rate"]).all()

        # Missing roe
        df_no_roe = pd.DataFrame({"ticker": ["A"], "payout_ratio": [0.20]})
        out2 = _add_derived_factors(df_no_roe)
        assert "sustainable_growth_rate" in out2.columns
        assert pd.isna(out2["sustainable_growth_rate"]).all()

    def test_compute_returns_all_six_composites_with_phase3a_columns(self):
        """When Phase 3a fields are present, all 6 composites populate."""
        tech, fund = _make_test_dfs()
        sector_map = {
            "NVDA": "Tech", "MSFT": "Tech",
            "JNJ": "Healthcare", "PFE": "Healthcare",
        }
        out = compute_factor_composites(tech, fund, sector_map)
        for col in (
            "quality_score", "momentum_score", "low_vol_score", "value_score",
            "growth_score", "stewardship_score",
        ):
            assert col in out.columns, f"missing {col}"
            assert out[col].notna().all(), f"{col} has NaN — partial-data path broken"
        # _n columns also present
        for col in (
            "quality_n", "momentum_n", "low_vol_n", "value_n",
            "growth_n", "stewardship_n",
        ):
            assert col in out.columns, f"missing {col}"

    def test_growth_score_ranks_higher_growth_higher(self):
        """Within Tech, NVDA (45% revenue 3y CAGR + 60% EPS 3y + 0% payout +
        35% capex growth) should score higher than MSFT (18% / 20% / 30% /
        12%) on growth."""
        tech, fund = _make_test_dfs()
        sector_map = {
            "NVDA": "Tech", "MSFT": "Tech",
            "JNJ": "Healthcare", "PFE": "Healthcare",
        }
        out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")
        assert out.loc["NVDA", "growth_score"] > out.loc["MSFT", "growth_score"]
        # Within Healthcare: JNJ > PFE on every growth component
        assert out.loc["JNJ", "growth_score"] > out.loc["PFE", "growth_score"]

    def test_stewardship_score_low_payout_high_capex_wins(self):
        """Within Tech, NVDA (0% payout, 35% capex growth) outranks MSFT
        (30% payout, 12% capex growth) on stewardship."""
        tech, fund = _make_test_dfs()
        sector_map = {
            "NVDA": "Tech", "MSFT": "Tech",
            "JNJ": "Healthcare", "PFE": "Healthcare",
        }
        out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")
        assert out.loc["NVDA", "stewardship_score"] > out.loc["MSFT", "stewardship_score"]
        # Within Healthcare: JNJ outranks PFE on both stewardship components
        assert out.loc["JNJ", "stewardship_score"] > out.loc["PFE", "stewardship_score"]

    def test_phase3a_columns_absent_yields_nan_composites_for_new_two(self):
        """Tolerant-reader: when Phase 3a fields aren't in fundamental.parquet
        (pre-merge or first SF firing after merge), growth_score +
        stewardship_score emit NaN; the legacy 4 composites continue to
        populate normally."""
        tickers = ["A", "B", "C"]
        tech = pd.DataFrame({
            "ticker": tickers,
            "date": ["2026-05-13"] * 3,
            "momentum_20d": [0.10, 0.05, 0.01],
            "momentum_5d": [0.02, 0.01, 0.0],
            "return_60d": [0.20, 0.10, 0.02],
            "return_120d": [0.40, 0.20, 0.05],
            "dist_from_52w_high": [-0.05, -0.10, -0.20],
            "realized_vol_20d": [0.15, 0.20, 0.30],
            "vol_ratio_10_60": [1.0, 1.1, 1.2],
            "atr_14_pct": [1.5, 2.0, 3.0],
        })
        # Legacy fundamentals only — no Phase 3a fields.
        fund = pd.DataFrame({
            "ticker": tickers,
            "date": ["2026-05-13"] * 3,
            "roe": [0.20, 0.15, 0.10],
            "debt_to_equity": [0.50, 0.80, 1.20],
            "gross_margin": [0.50, 0.40, 0.30],
            "current_ratio": [2.0, 1.5, 1.0],
            "pe_ratio": [25.0, 20.0, 15.0],
            "pb_ratio": [4.0, 3.0, 2.0],
            "fcf_yield": [0.04, 0.05, 0.06],
        })
        sector_map = {t: "Tech" for t in tickers}
        out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")

        # Legacy 4 composites populate normally — Phase 3b is non-breaking.
        for col in ("quality_score", "momentum_score", "low_vol_score", "value_score"):
            assert out[col].notna().all(), f"legacy {col} regressed under tolerant-reader path"
        # New 2 composites emit NaN (no underlying data).
        assert pd.isna(out["growth_score"]).all()
        assert pd.isna(out["stewardship_score"]).all()
        # _n columns reflect zero contributing factors.
        assert (out["growth_n"] == 0).all()
        assert (out["stewardship_n"] == 0).all()

    def test_phase3a_partial_columns_reallocates_within_growth(self):
        """If only some Phase 3a fields are present (e.g. revenue_growth_3y
        but no eps_growth_3y), growth_score still computes from the
        available components — same partial-coverage handling as the legacy
        composites."""
        tickers = ["A", "B", "C"]
        tech = pd.DataFrame({
            "ticker": tickers,
            "date": ["2026-05-13"] * 3,
            "momentum_20d": [0.10, 0.05, 0.01],
            "momentum_5d": [0.02, 0.01, 0.0],
            "return_60d": [0.20, 0.10, 0.02],
            "return_120d": [0.40, 0.20, 0.05],
            "dist_from_52w_high": [-0.05, -0.10, -0.20],
            "realized_vol_20d": [0.15, 0.20, 0.30],
            "vol_ratio_10_60": [1.0, 1.1, 1.2],
            "atr_14_pct": [1.5, 2.0, 3.0],
        })
        fund = pd.DataFrame({
            "ticker": tickers,
            "date": ["2026-05-13"] * 3,
            "roe": [0.20, 0.15, 0.10],
            "debt_to_equity": [0.50, 0.80, 1.20],
            "gross_margin": [0.50, 0.40, 0.30],
            "current_ratio": [2.0, 1.5, 1.0],
            "pe_ratio": [25.0, 20.0, 15.0],
            "pb_ratio": [4.0, 3.0, 2.0],
            "fcf_yield": [0.04, 0.05, 0.06],
            # Only revenue_growth_3y of the 5 Phase 3a fields is present.
            "revenue_growth_3y": [0.30, 0.15, 0.05],
        })
        sector_map = {t: "Tech" for t in tickers}
        out = compute_factor_composites(tech, fund, sector_map).set_index("ticker")
        # growth_score populates from the 1 available component (weight
        # reallocates 100% to revenue_growth_3y). sustainable_growth_rate
        # is also NaN because payout_ratio is missing.
        assert out["growth_score"].notna().all()
        assert (out["growth_n"] == 1).all()


# ── compute_and_write_factor_profiles: optional Metron supplemental source
# (metron-ops#177) ────────────────────────────────────────────────────────────

class TestComputeAndWriteFactorProfilesSupplemental:
    """A second, additive parquet source at
    features/metron_supplemental/{date}/{technical,fundamental}.parquet
    (written by alpha-engine-data for Metron-held tickers outside the
    S&P500+400 universe) gets concatenated in before composites are computed
    — when absent, behavior is unchanged (core parquets only)."""

    _CORE_TECHNICAL = pd.DataFrame({
        "ticker": ["AAPL"], "momentum_20d": [0.10], "return_60d": [0.20],
        "return_120d": [0.30], "dist_from_52w_high": [-0.05], "momentum_5d": [0.02],
        "realized_vol_20d": [0.15], "vol_ratio_10_60": [1.0], "atr_14_pct": [1.5],
    })
    _CORE_FUNDAMENTAL = pd.DataFrame({
        "ticker": ["AAPL"], "roe": [0.30], "debt_to_equity": [0.50],
        "gross_margin": [0.40], "current_ratio": [1.5],
        "pe_ratio": [25.0], "pb_ratio": [4.0], "fcf_yield": [0.04],
    })
    _SUPP_TECHNICAL = pd.DataFrame({
        "ticker": ["MARUY"], "momentum_20d": [0.05], "return_60d": [0.10],
        "return_120d": [0.15], "dist_from_52w_high": [-0.10], "momentum_5d": [0.01],
        "realized_vol_20d": [0.20], "vol_ratio_10_60": [1.1], "atr_14_pct": [2.0],
    })
    _SUPP_FUNDAMENTAL = pd.DataFrame({
        "ticker": ["MARUY"], "roe": [0.10], "debt_to_equity": [0.80],
        "gross_margin": [0.20], "current_ratio": [1.2],
        "pe_ratio": [10.0], "pb_ratio": [1.0], "fcf_yield": [0.08],
    })

    @staticmethod
    def _parquet_bytes(df: pd.DataFrame) -> bytes:
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        return buf.getvalue()

    def _mock_s3(self, *, supp_technical=None, supp_fundamental=None):
        s3 = MagicMock()
        parquets = {
            "features/2026-07-12/technical.parquet": self._CORE_TECHNICAL,
            "features/2026-07-12/fundamental.parquet": self._CORE_FUNDAMENTAL,
        }
        if supp_technical is not None:
            parquets["features/metron_supplemental/2026-07-12/technical.parquet"] = supp_technical
        if supp_fundamental is not None:
            parquets["features/metron_supplemental/2026-07-12/fundamental.parquet"] = supp_fundamental

        def _get(Bucket, Key):
            if Key not in parquets:
                raise Exception("NoSuchKey")
            body = MagicMock()
            body.read.return_value = self._parquet_bytes(parquets[Key])
            return {"Body": body}

        s3.get_object.side_effect = _get
        return s3

    def test_supplemental_absent_scores_only_core_universe(self):
        s3 = self._mock_s3()
        with patch("boto3.client", return_value=s3):
            compute_and_write_factor_profiles(
                run_date="2026-07-12", sector_map={"AAPL": "Information Technology"}, bucket="test-bucket",
            )
        payload = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
        assert set(payload.keys()) == {"AAPL"}

    def test_supplemental_present_adds_extra_ticker_to_profiles(self):
        s3 = self._mock_s3(supp_technical=self._SUPP_TECHNICAL, supp_fundamental=self._SUPP_FUNDAMENTAL)
        with patch("boto3.client", return_value=s3):
            compute_and_write_factor_profiles(
                run_date="2026-07-12",
                sector_map={"AAPL": "Information Technology", "MARUY": "Industrials"},
                bucket="test-bucket",
            )
        payload = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
        assert set(payload.keys()) == {"AAPL", "MARUY"}
        assert payload["MARUY"]["sector"] == "Industrials"

    def test_supplemental_technical_only_still_merges_via_outer_join(self):
        """A supplemental ticker with technical but no fundamental data still
        gets partial composites (momentum/low_vol) via the existing
        outer-join partial-coverage handling — same as any core ticker."""
        s3 = self._mock_s3(supp_technical=self._SUPP_TECHNICAL)
        with patch("boto3.client", return_value=s3):
            compute_and_write_factor_profiles(
                run_date="2026-07-12",
                sector_map={"AAPL": "Information Technology", "MARUY": "Industrials"},
                bucket="test-bucket",
            )
        payload = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
        assert "MARUY" in payload
        assert "momentum_score" in payload["MARUY"]
