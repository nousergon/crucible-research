"""Source selection + staleness for the scanner's feature reader.

The live champion path must never reach the WEEKLY S3 snapshot by accident,
and must never rank on a stale ArcticDB read without saying so.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _reload_with_source(value: str | None):
    """Re-import the reader with SCANNER_FEATURE_SOURCE set (module-level read)."""
    import os

    import data.fetchers.feature_store_reader as mod

    old = os.environ.get("SCANNER_FEATURE_SOURCE")
    if value is None:
        os.environ.pop("SCANNER_FEATURE_SOURCE", None)
    else:
        os.environ["SCANNER_FEATURE_SOURCE"] = value
    try:
        return importlib.reload(mod)
    finally:
        if old is None:
            os.environ.pop("SCANNER_FEATURE_SOURCE", None)
        else:
            os.environ["SCANNER_FEATURE_SOURCE"] = old


def _arctic_frame(last_date: str) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
    return pd.DataFrame({"rsi_14": [55.0], "momentum_20d": [0.03]}, index=idx)


def _batch_result(df):
    res = MagicMock()
    res.data = df
    return res


def test_ticker_list_selects_arcticdb_not_the_weekly_snapshot():
    mod = _reload_with_source(None)  # default
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(_arctic_frame("2026-07-29"))]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        out = mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())

    assert out == {"AAPL": {"rsi_14": 55.0, "momentum_20d": 0.03}}
    lib.read_batch.assert_called_once()


def test_no_ticker_list_falls_back_to_the_weekly_snapshot():
    """graph/research_graph.py discovers its universe by iterating the return
    value, so it structurally cannot use a per-symbol store. It must degrade
    to S3 rather than raise — but never silently.
    """
    mod = _reload_with_source(None)
    with (
        patch.object(mod, "_read_latest_features_s3", return_value={"AAPL": {}}) as s3_read,
        patch.object(mod, "_connect_universe_lib") as arctic,
    ):
        out = mod.read_latest_features()

    assert out == {"AAPL": {}}
    s3_read.assert_called_once()
    arctic.assert_not_called()


def test_env_override_forces_the_weekly_snapshot_even_with_tickers():
    """The rollback lever has to beat the ticker-list selector, or it isn't one."""
    mod = _reload_with_source("s3")
    with (
        patch.object(mod, "_read_latest_features_s3", return_value={"AAPL": {}}) as s3_read,
        patch.object(mod, "_connect_universe_lib") as arctic,
    ):
        mod.read_latest_features(["AAPL"])

    s3_read.assert_called_once()
    arctic.assert_not_called()
    _reload_with_source(None)


def test_stale_arcticdb_rows_raise_instead_of_ranking_the_universe():
    mod = _reload_with_source(None)
    lib = MagicMock()
    # Newest row is ~3 months old — the daily producer has stopped writing.
    lib.read_batch.return_value = [_batch_result(_arctic_frame("2026-05-01"))]

    with (
        patch.object(mod, "_connect_universe_lib", return_value=lib),
        pytest.raises(mod.FeatureStoreStalenessError, match="trading days behind"),
    ):
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())


def test_read_failure_rate_above_the_ceiling_raises():
    """A near-total read failure must not look like a small universe."""
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(None) for _ in range(10)]

    with (
        patch.object(mod, "_connect_universe_lib", return_value=lib),
        pytest.raises(mod.FeatureStoreStalenessError, match="read-failure rate"),
    ):
        mod.read_latest_features(
            [f"T{i}" for i in range(10)],
            ref_date=pd.Timestamp("2026-07-29").date(),
        )


def test_only_the_columns_the_scanner_consumes_are_projected():
    """A full-row read over ~900 symbols is the difference between a
    comfortable and a timed-out scanner Lambda.
    """
    mod = _reload_with_source(None)
    lib = MagicMock()
    lib.read_batch.return_value = [_batch_result(_arctic_frame("2026-07-29"))]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        mod.read_latest_features(["AAPL"], ref_date=pd.Timestamp("2026-07-29").date())

    requested = lib.read_batch.call_args[0][0][0]
    assert set(requested.columns) == set(mod._TECHNICAL_COLS)
    assert "avg_volume_20d_raw" in requested.columns, (
        "the scanner's MIN_AVG_VOLUME gate needs raw shares, not the normalized avg_volume_20d ratio"
    )


def test_factor_loadings_honour_the_same_source_selection():
    mod = _reload_with_source(None)
    lib = MagicMock()
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-29")])
    lib.read_batch.return_value = [_batch_result(pd.DataFrame({"momentum_20d_zscore": [1.2]}, index=idx))]

    with patch.object(mod, "_connect_universe_lib", return_value=lib):
        out = mod.read_latest_factor_loadings(
            tickers=["AAPL"],
            ref_date=pd.Timestamp("2026-07-29").date(),
        )

    assert out == {"AAPL": {"momentum_20d_zscore": 1.2}}
