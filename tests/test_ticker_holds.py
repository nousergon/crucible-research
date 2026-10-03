"""Declared per-ticker holds (alpha-engine-config-I11806).

Pins that a held ticker never reaches ``run_quant_filter`` — and so is absent
from the eval log every rank table and cut is derived from — that the hold is
recorded on the artifact, that it expires on its own date, and that the CTVA
entry is the one the spin-off arc declared.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from data.ticker_holds import DECLARED_HOLDS, TickerHold, active_holds


def test_ctva_is_held_through_october():
    hold = DECLARED_HOLDS["CTVA"]
    assert hold.issue == "alpha-engine-config-I11806"
    assert "CTVA" in active_holds("2026-10-05")
    assert "CTVA" in active_holds("2026-10-31")
    assert "CTVA" not in active_holds("2026-11-01")
    assert "CTVA" not in active_holds("2026-09-30")


def test_bounds_are_inclusive_and_an_expired_hold_is_ignored():
    holds = {"X": TickerHold("2026-01-02", "2026-01-05", "why", "I0")}
    assert active_holds("2026-01-02", holds) == holds
    assert active_holds("2026-01-05", holds) == holds
    assert active_holds("2026-01-06", holds) == {}
    assert active_holds("2026-01-01", holds) == {}


def test_every_hold_has_a_reason_an_owner_and_an_ordered_window():
    for ticker, hold in DECLARED_HOLDS.items():
        assert hold.why and hold.issue, ticker
        assert hold.valid_from <= hold.valid_through, ticker


def _build(constituents, run_date, holds):
    from data.scanner_orchestrator import build_candidates_artifact

    captured: dict = {}

    def _quant(**kwargs):
        captured["tickers"] = list(kwargs["tickers"])
        return [{"ticker": t} for t in kwargs["tickers"][:60]]

    patches = [
        patch(
            "data.fetchers.price_fetcher.fetch_sp500_sp400_with_sectors",
            return_value=(constituents, dict.fromkeys(constituents, "Materials")),
        ),
        patch(
            "data.fetchers.feature_store_reader.read_latest_features",
            return_value={t: {"rsi_14": 55.0, "atr_14_pct": 0.02} for t in constituents},
        ),
        patch(
            "data.fetchers.feature_store_reader.read_latest_daily_closes",
            return_value=dict.fromkeys(constituents, 100.0),
        ),
        patch("scoring.technical.compute_technical_score", return_value=70.0),
        patch("data.scanner.run_quant_filter", side_effect=_quant),
        patch("data.fetchers.feature_store_reader.read_latest_factor_loadings", return_value=None),
        patch(
            "data.scanner_orchestrator._read_prior_signals_universe_tickers",
            return_value=([], [], None),
        ),
        patch(
            "data.scanner_orchestrator._read_prior_candidates_scanner_tickers",
            return_value=([], None),
        ),
        patch("data.ticker_holds.DECLARED_HOLDS", holds),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        artifact = build_candidates_artifact(
            run_date=run_date,
            s3_client=MagicMock(),
            bucket="test-bucket",
        )
    return artifact, captured["tickers"]


@pytest.fixture
def universe():
    return ["CTVA"] + [f"T{i}" for i in range(899)]


def test_a_held_ticker_never_reaches_the_quant_filter(universe):
    artifact, scanned = _build(universe, "2026-10-05", dict(DECLARED_HOLDS))
    assert "CTVA" not in scanned
    assert len(scanned) == 899
    assert "CTVA" not in artifact["scanner_tickers"]
    assert artifact["stats"]["universe_size"] == 899
    assert artifact["declared_holds"]["CTVA"]["issue"] == "alpha-engine-config-I11806"
    assert artifact["declared_holds"]["CTVA"]["valid_through"] == "2026-10-31"


def test_after_expiry_the_ticker_is_scanned_again(universe):
    artifact, scanned = _build(universe, "2026-11-02", dict(DECLARED_HOLDS))
    assert "CTVA" in scanned
    assert artifact["declared_holds"] == {}


def test_a_hold_on_a_non_member_records_nothing(universe):
    holds = {"ZZZZ": TickerHold("2026-10-01", "2026-10-31", "why", "I0")}
    artifact, scanned = _build(universe, "2026-10-05", holds)
    assert len(scanned) == 900
    assert artifact["declared_holds"] == {}


def test_the_constituents_floor_is_checked_before_holds_apply():
    """A hold is not a data loss: 800 members with one held still scan."""
    members = ["CTVA"] + [f"T{i}" for i in range(799)]
    artifact, scanned = _build(members, "2026-10-05", dict(DECLARED_HOLDS))
    assert len(scanned) == 799
