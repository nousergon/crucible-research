"""Machine-written ticker holds (alpha-engine-config-I13044).

nousergon-data opens ``data_quality/ticker_holds/{TICKER}.json`` when the
daily append writes an unexplained split-like bar (the CTVA 2026-10-01 class,
nous-ergon-ops-I2044). The scanner must keep an active one out of its
universe beside the declared holds, ignore a cleared one, and refuse to scan
when it cannot read them.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pytest

from data.scanner_orchestrator import ScannerOrchestratorError
from data.ticker_holds import DECLARED_HOLDS, TickerHold, machine_holds
from tests.test_ticker_holds import _build


class _FakeS3:
    def __init__(self, bodies: dict[str, dict]):
        self.store = {f"data_quality/ticker_holds/{t}.json": json.dumps(b).encode() for t, b in bodies.items()}

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.store[Key])}

    def get_paginator(self, _op):
        store = self.store

        class _P:
            def paginate(self, *, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(store) if k.startswith(Prefix)]}

        return _P()


def _hold(ticker, status="active", break_date="2026-10-01"):
    return {
        "ticker": ticker,
        "status": status,
        "break_date": break_date,
        "why": "pure-append close 12.57 vs 77.65 (ratio 0.1619) is split-like",
        "issue": "alpha-engine-config-I13044",
    }


def test_only_active_holds_on_or_before_the_run_date_apply():
    s3 = _FakeS3(
        {
            "CTVA": _hold("CTVA"),
            "OLD": _hold("OLD", status="cleared"),
            "LATER": _hold("LATER", break_date="2026-10-09"),
        }
    )
    holds = machine_holds(s3, "b", "2026-10-05")
    assert set(holds) == {"CTVA"}
    assert holds["CTVA"].valid_from == "2026-10-01"
    assert holds["CTVA"].valid_through is None
    assert holds["CTVA"].source == "s3://b/data_quality/ticker_holds/CTVA.json"


def _build_with(s3, universe, run_date, declared):
    from unittest.mock import patch

    with patch("data.ticker_holds.machine_holds", side_effect=lambda _s3, b, d: machine_holds(s3, b, d)):
        return _build(universe, run_date, declared)


@pytest.fixture
def universe():
    return ["CTVA", "XYZ"] + [f"T{i}" for i in range(898)]


def test_a_machine_held_ticker_never_reaches_the_quant_filter(universe):
    s3 = _FakeS3({"XYZ": _hold("XYZ")})
    artifact, scanned = _build_with(s3, universe, "2026-10-05", {})
    assert "XYZ" not in scanned and "CTVA" in scanned
    assert artifact["declared_holds"]["XYZ"]["valid_through"] is None
    assert artifact["declared_holds"]["XYZ"]["source"].endswith("ticker_holds/XYZ.json")


def test_a_cleared_machine_hold_scans_again(universe):
    s3 = _FakeS3({"XYZ": _hold("XYZ", status="cleared")})
    artifact, scanned = _build_with(s3, universe, "2026-10-05", {})
    assert "XYZ" in scanned
    assert artifact["declared_holds"] == {}


def test_a_declared_hold_wins_over_a_machine_hold_for_the_same_name(universe):
    s3 = _FakeS3({"CTVA": _hold("CTVA")})
    artifact, scanned = _build_with(s3, universe, "2026-10-05", dict(DECLARED_HOLDS))
    assert "CTVA" not in scanned
    assert artifact["declared_holds"]["CTVA"]["valid_through"] == "2026-11-11"
    assert artifact["declared_holds"]["CTVA"]["source"] == "declared"


def test_unreadable_holds_refuse_the_scan(universe):
    s3 = MagicMock()
    s3.get_paginator.side_effect = RuntimeError("AccessDenied")
    with pytest.raises(ScannerOrchestratorError, match="I13044"):
        _build_with(s3, universe, "2026-10-05", {})


def test_declared_holds_keep_their_shape():
    hold = TickerHold("2026-01-02", "2026-01-05", "why", "I0")
    assert hold.source == "declared"
