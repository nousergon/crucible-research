"""Regression tests for the held-stock thesis sector leak fix (2026-05-04).

Held-stock thesis_updates from the team qual analyst LLM path can default
``sector`` to "Unknown" via the Pydantic schema (state_schemas.py
InvestmentThesis) when the LLM omits the field. The literal string "Unknown"
is truthy, so a naive ``thesis.get("sector") or sector_map.get(...)`` short-
circuits on it and never consults the authoritative ``sector_map`` (loaded
from constituents.json with full universe coverage).

These tests pin: ``_build_signals_payload`` and the score_aggregator held-
stock recompute branch must always prefer ``sector_map[ticker]`` over an
LLM-emitted thesis sector.

Repro shape: 2026-05-04 EOG/VICI/UNP/NVT held BUY signals reached the
executor + dashboard with sector="Unknown" despite the constituents.json
sector_map carrying Energy/Industrials/Real Estate correctly.
"""

from __future__ import annotations


def _held_buy_thesis(ticker: str, sector: str = "Unknown") -> dict:
    """Mimic a held-stock thesis_update result that defaulted sector to
    "Unknown" via the Pydantic schema (LLM omitted the field)."""
    return {
        "ticker": ticker,
        "rating": "BUY",
        "final_score": 78.3,
        "quant_score": 70.0,
        "qual_score": None,
        "sector": sector,  # ← the bug surface
        "team_id": "defensives",
        "conviction": "stable",
        "bull_case": "",
    }


def test_signals_payload_prefers_sector_map_over_unknown_thesis_sector():
    """Authoritative sector_map must overwrite an LLM-emitted "Unknown"
    in held-stock thesis updates. Repros today's EOG/NVT/UNP/VICI bug."""
    from scoring.signals_payload import _build_signals_payload

    state = {
        "investment_theses": {
            "EOG": _held_buy_thesis("EOG", sector="Unknown"),
            "VICI": _held_buy_thesis("VICI", sector="Unknown"),
        },
        "prior_theses": {},
        "new_population": [{"ticker": "EOG"}, {"ticker": "VICI"}],
        "sector_map": {
            "EOG": "Energy",
            "VICI": "Real Estate",
        },
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],  # held reaffirm path
    }

    payload = _build_signals_payload(state)
    signals = payload["signals"]

    assert signals["EOG"]["sector"] == "Energy", (
        f"sector_map authoritative; got {signals['EOG']['sector']!r}"
    )
    assert signals["VICI"]["sector"] == "Real Estate", (
        f"sector_map authoritative; got {signals['VICI']['sector']!r}"
    )


def test_signals_payload_falls_back_to_thesis_sector_when_map_missing():
    """If sector_map doesn't carry a ticker (truly outside S&P universe),
    fall back to whatever the thesis claims rather than hard-coding."""
    from scoring.signals_payload import _build_signals_payload

    state = {
        "investment_theses": {
            "WEIRD": _held_buy_thesis("WEIRD", sector="Healthcare"),
        },
        "prior_theses": {},
        "new_population": [{"ticker": "WEIRD"}],
        "sector_map": {},  # missing
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    weird = payload["signals"]["WEIRD"]
    assert weird["sector"] == "Healthcare"


def test_signals_payload_unknown_when_both_map_and_thesis_missing():
    """Last resort: if neither sector_map nor thesis carries sector,
    the public default of "Unknown" is honest and matches the executor's
    fallback handling."""
    from scoring.signals_payload import _build_signals_payload

    state = {
        "investment_theses": {
            "EDGE": {
                "ticker": "EDGE",
                "rating": "BUY",
                "final_score": 70.0,
                "quant_score": 70.0,
                "qual_score": 65.0,
                "team_id": None,
                "conviction": "stable",
                "bull_case": "",
                # no sector key at all
            },
        },
        "prior_theses": {},
        "new_population": [{"ticker": "EDGE"}],
        "sector_map": {},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    edge = payload["signals"]["EDGE"]
    assert edge["sector"] == "Unknown"
