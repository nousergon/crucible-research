"""
Unit tests for the ArcticDB ``data_snapshot_id`` surfacing in
``data.fetchers.price_fetcher.fetch_price_data`` (L4567 sub-item 1b / #781).

The fetcher must surface the ArcticDB ``VersionedItem.version`` of each read
as a run-level ``data_snapshot_id`` so decision capture can stamp exactly
which immutable price snapshot a decision was computed on. The unversioned /
missing-version path must record a ``"unknown"`` sentinel without crashing
(per ``feedback_no_silent_fails`` — visible, not silent).

Reads are stubbed at the ``_connect_arctic`` boundary with a fake library
whose ``read`` returns objects carrying a known ``.version`` (mirroring
ArcticDB's ``VersionedItem``), so the test needs no live ArcticDB/S3.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from data.fetchers import price_fetcher
from data.fetchers.price_fetcher import (
    DATA_SNAPSHOT_ID_UNKNOWN,
    fetch_price_data,
)


def _ohlcv_frame(rows: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=idx,
    )


class _FakeVersionedItem:
    """Stand-in for ArcticDB's ``VersionedItem`` — carries ``.data`` and
    ``.version`` (the per-symbol monotonic write generation)."""

    def __init__(self, data: pd.DataFrame, version):
        self.data = data
        self.version = version


class _FakeLib:
    """Fake ArcticDB library. ``versions`` maps ticker → version int (or
    ``_MISSING`` to return an object with no ``.version`` attribute, or
    ``_RAISE`` to simulate a per-ticker read failure)."""

    _MISSING = object()
    _RAISE = object()

    def __init__(self, versions: dict):
        self._versions = versions

    def read(self, ticker, date_range=None, columns=None):
        v = self._versions.get(ticker, 0)
        if v is self._RAISE:
            raise RuntimeError(f"simulated read failure for {ticker}")
        df = _ohlcv_frame()
        if v is self._MISSING:
            # An object with .data but NO .version (non-versioned backend).
            class _NoVersion:
                pass
            obj = _NoVersion()
            obj.data = df
            return obj
        return _FakeVersionedItem(df, v)


@pytest.fixture
def patched_arctic():
    """Patch ``_connect_arctic`` to return a fake lib supplied per-test."""
    def _install(versions: dict):
        return patch.object(
            price_fetcher, "_connect_arctic", return_value=_FakeLib(versions),
        )
    return _install


class TestSnapshotIdSurfacing:
    def test_surfaces_max_version_as_snapshot_id(self, patched_arctic):
        # Three tickers at versions 3, 7, 5 → run-level stamp = max = "7".
        with patched_arctic({"AAPL": 3, "MSFT": 7, "JPM": 5}):
            result, snap = fetch_price_data(
                ["AAPL", "MSFT", "JPM"], period="3mo", return_snapshot_id=True,
            )
        assert set(result) == {"AAPL", "MSFT", "JPM"}
        assert snap == "7"

    def test_single_ticker_version(self, patched_arctic):
        with patched_arctic({"AAPL": 42}):
            result, snap = fetch_price_data(
                ["AAPL"], period="3mo", return_snapshot_id=True,
            )
        assert snap == "42"
        assert "AAPL" in result

    def test_default_return_is_dict_only(self, patched_arctic):
        # Back-compat: without return_snapshot_id the contract is unchanged.
        with patched_arctic({"AAPL": 1}):
            result = fetch_price_data(["AAPL"], period="3mo")
        assert isinstance(result, dict)
        assert "AAPL" in result

    def test_empty_tickers_records_sentinel(self):
        result, snap = fetch_price_data([], return_snapshot_id=True)
        assert result == {}
        assert snap == DATA_SNAPSHOT_ID_UNKNOWN


class TestMissingVersionFallback:
    def test_unversioned_read_records_sentinel_no_crash(self, patched_arctic):
        # Read object exposes no ``.version`` → no version contributed →
        # run-level stamp is the "unknown" sentinel, and the read still
        # succeeds (frame returned), never crashes.
        with patched_arctic({"AAPL": _FakeLib._MISSING}):
            result, snap = fetch_price_data(
                ["AAPL"], period="3mo", return_snapshot_id=True,
            )
        assert "AAPL" in result
        assert snap == DATA_SNAPSHOT_ID_UNKNOWN

    def test_mixed_versioned_and_unversioned(self, patched_arctic):
        # One ticker versioned (9), one without a version → stamp = "9"
        # (the present version wins; the missing one just abstains).
        with patched_arctic({"AAPL": 9, "MSFT": _FakeLib._MISSING}):
            _, snap = fetch_price_data(
                ["AAPL", "MSFT"], period="3mo", return_snapshot_id=True,
            )
        assert snap == "9"

    def test_all_reads_fail_records_sentinel(self, patched_arctic):
        # Every ticker read raises → error rate 100% > threshold → PriceFetchError.
        with patched_arctic({"AAPL": _FakeLib._RAISE}):
            with pytest.raises(price_fetcher.PriceFetchError):
                fetch_price_data(
                    ["AAPL"], period="3mo", return_snapshot_id=True,
                )

    def test_bool_version_rejected(self, patched_arctic):
        # ``bool`` is an ``int`` subclass — must NOT be treated as a version.
        with patched_arctic({"AAPL": True}):
            _, snap = fetch_price_data(
                ["AAPL"], period="3mo", return_snapshot_id=True,
            )
        assert snap == DATA_SNAPSHOT_ID_UNKNOWN


class TestExtractVersionHelper:
    def test_extracts_int(self):
        assert price_fetcher._extract_arctic_version(
            _FakeVersionedItem(_ohlcv_frame(), 5)
        ) == 5

    def test_none_when_absent(self):
        assert price_fetcher._extract_arctic_version(object()) is None

    def test_none_for_bool(self):
        assert price_fetcher._extract_arctic_version(
            _FakeVersionedItem(_ohlcv_frame(), True)
        ) is None
