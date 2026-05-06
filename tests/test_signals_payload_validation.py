"""Preflight tests for signals.json sector validation (2026-05-06).

Surface for the 2026-05-04 EOG/NVT incident: research wrote the first pass
of signals.json with sector="Unknown" for tickers whose constituents
sector_map hadn't loaded yet, then re-ran 10 minutes later with correct
values. The morning planner consumed v1, the order book persisted
sector="Unknown", and the daemon's intraday fills wrote "Unknown" into
trades.db. The bad rows survived because no UPDATE path overwrites
trades.sector — eod_reconcile only enriches the in-memory positions snapshot.

The validator runs between _build_signals_payload and write_signals_json:
ENTER signals with sector="Unknown" (or empty/None) raise, the existing
try/except logs ERROR and skips the write, and the executor falls back to
the prior trading day's signals.json on the next morning planner run.
"""

from __future__ import annotations

import pytest


def _enter_signal(ticker: str, sector: str = "Energy") -> dict:
    return {
        "ticker": ticker,
        "signal": "ENTER",
        "score": 75.0,
        "rating": "BUY",
        "conviction": "stable",
        "thesis_summary": "",
        "sector": sector,
        "team_id": "energy",
        "quant_score": 75.0,
        "qual_score": 70.0,
        "sub_scores": {"quant": 75.0, "qual": 70.0},
    }


def test_clean_payload_passes():
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Energy"),
            "NVT": _enter_signal("NVT", sector="Industrials"),
        }
    }
    _validate_signals_payload(payload)


def test_unknown_sector_on_enter_raises():
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Unknown"),
        }
    }
    with pytest.raises(RuntimeError, match=r"\['EOG'\]"):
        _validate_signals_payload(payload)


def test_multiple_unknown_sectors_listed_in_message():
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Unknown"),
            "NVT": _enter_signal("NVT", sector="Unknown"),
            "CTAS": _enter_signal("CTAS", sector="Industrials"),
        }
    }
    with pytest.raises(RuntimeError) as exc_info:
        _validate_signals_payload(payload)
    msg = str(exc_info.value)
    assert "EOG" in msg and "NVT" in msg
    assert "CTAS" not in msg


def test_empty_sector_on_enter_raises():
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector=""),
        }
    }
    with pytest.raises(RuntimeError, match=r"\['EOG'\]"):
        _validate_signals_payload(payload)


def test_none_sector_on_enter_raises():
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": {**_enter_signal("EOG"), "sector": None},
        }
    }
    with pytest.raises(RuntimeError, match=r"\['EOG'\]"):
        _validate_signals_payload(payload)


def test_unknown_sector_on_hold_signal_does_not_raise():
    """HOLD signals don't propagate to the order book or trades.db. Only
    ENTER triggers the durable-record concern."""
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "META": {**_enter_signal("META"), "signal": "HOLD", "sector": "Unknown"},
        }
    }
    _validate_signals_payload(payload)


def test_unknown_sector_on_exit_signal_does_not_raise():
    """EXIT signals close existing positions and don't create new trade
    rows; the executor reuses the entry trade's sector for attribution."""
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "OLD": {**_enter_signal("OLD"), "signal": "EXIT", "sector": "Unknown"},
        }
    }
    _validate_signals_payload(payload)


def test_empty_payload_passes():
    from graph.research_graph import _validate_signals_payload

    _validate_signals_payload({})
    _validate_signals_payload({"signals": {}})


# ── Universe-membership drift gate ────────────────────────────────────────


def test_enter_ticker_outside_scanner_universe_raises():
    """A ticker that's no longer in the S&P 500+400 scanner universe must
    not surface as a buy candidate. Held positions get HOLD/EXIT signals
    via the existing rating logic; ENTER on a de-listed ticker would let
    the executor add to a position outside the index."""
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Energy"),
        }
    }
    universe = ["AAPL", "MSFT", "NVDA"]  # EOG dropped from scanner universe

    with pytest.raises(RuntimeError, match=r"outside current S&P 900.*EOG"):
        _validate_signals_payload(payload, scanner_universe=universe)


def test_enter_ticker_inside_universe_passes():
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Energy"),
        }
    }
    universe = ["EOG", "NVT"]
    _validate_signals_payload(payload, scanner_universe=universe)


def test_universe_check_skipped_when_universe_is_none():
    """Backward-compatible: passing scanner_universe=None disables the
    membership check (matches the original PR #126 contract)."""
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Energy"),
        }
    }
    _validate_signals_payload(payload)
    _validate_signals_payload(payload, scanner_universe=None)


def test_universe_check_skipped_for_hold_and_exit():
    """Held tickers that left the universe still need HOLD/EXIT signals to
    manage the existing position. The gate only blocks ENTER."""
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": {**_enter_signal("EOG"), "signal": "HOLD"},
            "NVT": {**_enter_signal("NVT"), "signal": "EXIT"},
        }
    }
    universe = ["AAPL"]  # EOG and NVT both out
    _validate_signals_payload(payload, scanner_universe=universe)


def test_combined_unresolved_sector_and_universe_drift_in_one_message():
    """When both checks fail, surface both in the same RuntimeError so
    operators see the full picture, not just the first one to trip."""
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Unknown"),
            "DELISTED": _enter_signal("DELISTED", sector="Energy"),
        }
    }
    universe = ["EOG", "AAPL"]  # EOG in universe but bad sector; DELISTED out

    with pytest.raises(RuntimeError) as exc_info:
        _validate_signals_payload(payload, scanner_universe=universe)
    msg = str(exc_info.value)
    assert "unresolved sector" in msg
    assert "EOG" in msg
    assert "outside current S&P 900" in msg
    assert "DELISTED" in msg


def test_universe_set_input_also_works():
    """Accept set or list for scanner_universe — defensively converted."""
    from graph.research_graph import _validate_signals_payload

    payload = {
        "signals": {
            "EOG": _enter_signal("EOG", sector="Energy"),
        }
    }
    _validate_signals_payload(payload, scanner_universe={"EOG", "NVT"})
