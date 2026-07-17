"""Consumer contract for the research_optimizer boost columns (config#1857).

crucible-research is the PRODUCER of ``short_interest_adj`` and
``institutional_boost`` on every ``universe`` / ``buy_candidates`` entry in
signals.json; the CONSUMER is
crucible-backtester ``optimizer/research_optimizer.py::compute_boost_correlations``,
which reads exactly those two column names with ``stock.get(col, 0.0)`` and
treats ``nonzero`` as "boost was applied that day". Before this shipped the
fields were never emitted, so the optimizer returned ``no_boost_data`` on every
run and ``config/research_params.json`` was never tuned (config#1857).

These tests pin the emit contract: the two fields are always present on both
lists, carry the applied-boost magnitude the config thresholds dictate, key off
the newest weekly artifact <= run_date (PIT-safe), and reader-gate to 0.0 on any
source failure so the signals write never breaks.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.boost_signals import (  # noqa: E402
    _latest_weekly_key,
    annotate_boost_signals,
    emit_boost_signals,
    institutional_boost_for,
    short_interest_adj_for,
)

_PARAMS = {
    "short_interest_buy_threshold_pct": 20,
    "short_interest_high_threshold_pct": 40,
    "short_interest_buy_boost": 2.0,
    "short_interest_high_boost": 4.0,
    "institutional_min_funds": 3,
    "institutional_boost": 3.0,
}

# The exact column names the backtester reads (mirror of research_optimizer
# boost_cols — cross-repo, so hard-coded per the test_scanner_consumer_contract
# precedent).
_BACKTESTER_BOOST_COLS = {"short_interest_adj", "institutional_boost"}


# ── pure short-interest math ─────────────────────────────────────────────────

def test_short_interest_below_buy_threshold_is_zero():
    assert short_interest_adj_for(19.9, _PARAMS) == 0.0


def test_short_interest_at_buy_threshold_earns_buy_boost():
    assert short_interest_adj_for(20.0, _PARAMS) == 2.0
    assert short_interest_adj_for(39.9, _PARAMS) == 2.0


def test_short_interest_at_high_threshold_earns_high_boost():
    assert short_interest_adj_for(40.0, _PARAMS) == 4.0
    assert short_interest_adj_for(75.0, _PARAMS) == 4.0


def test_short_interest_missing_or_unparseable_is_zero():
    assert short_interest_adj_for(None, _PARAMS) == 0.0
    assert short_interest_adj_for("n/a", _PARAMS) == 0.0


# ── pure institutional math ──────────────────────────────────────────────────

def test_institutional_no_record_or_no_signal_is_zero():
    assert institutional_boost_for(None, _PARAMS) == 0.0
    assert institutional_boost_for({}, _PARAMS) == 0.0
    assert institutional_boost_for({"n_funds_accumulating": 2}, _PARAMS) == 0.0


def test_institutional_signal_at_min_funds_earns_boost():
    assert institutional_boost_for({"n_funds_accumulating": 3}, _PARAMS) == 3.0
    assert institutional_boost_for({"n_funds_accumulating": 9}, _PARAMS) == 3.0


def test_institutional_min_funds_re_derived_from_count_honors_param_sweep():
    # A retuned min_funds (sweep) must be honored off n_funds without re-fetch.
    strict = {**_PARAMS, "institutional_min_funds": 5}
    assert institutional_boost_for({"n_funds_accumulating": 3}, strict) == 0.0
    assert institutional_boost_for({"n_funds_accumulating": 5}, strict) == 3.0


def test_institutional_falls_back_to_precomputed_signal_when_count_absent():
    assert institutional_boost_for({"accumulation_signal": True}, _PARAMS) == 3.0
    assert institutional_boost_for({"accumulation_signal": False}, _PARAMS) == 0.0


# ── annotation contract ──────────────────────────────────────────────────────

def _entries():
    return [
        {"ticker": "HISI", "signal": "ENTER"},   # high short interest
        {"ticker": "MISI", "signal": "HOLD"},    # buy-band short interest + inst
        {"ticker": "CLEAN", "signal": "ENTER"},  # neither
    ]


def test_annotation_adds_both_fields_with_correct_values():
    entries = _entries()
    si_map = {"HISI": {"short_pct_float": 55.0}, "MISI": {"short_pct_float": 25.0}}
    inst_map = {"MISI": {"n_funds_accumulating": 4}}
    annotate_boost_signals(entries, short_interest_map=si_map, institutional_map=inst_map, params=_PARAMS)
    by = {e["ticker"]: e for e in entries}
    assert by["HISI"]["short_interest_adj"] == 4.0
    assert by["HISI"]["institutional_boost"] == 0.0
    assert by["MISI"]["short_interest_adj"] == 2.0
    assert by["MISI"]["institutional_boost"] == 3.0
    assert by["CLEAN"]["short_interest_adj"] == 0.0
    assert by["CLEAN"]["institutional_boost"] == 0.0
    # every entry carries EXACTLY the backtester's column names
    for e in entries:
        assert _BACKTESTER_BOOST_COLS <= e.keys()


# ── emit orchestrator: gating, injection, reader-gate ────────────────────────

def _payload():
    return {
        "universe": [
            {"ticker": "HISI", "signal": "ENTER"},
            {"ticker": "MISI", "signal": "HOLD"},
        ],
        "buy_candidates": [{"ticker": "HISI", "signal": "ENTER"}],
    }


def test_emit_disabled_by_default_defaults_zero_without_io(monkeypatch):
    # No RESEARCH_BOOST_SIGNALS_ENABLED, no force → fields present + 0.0, no S3.
    monkeypatch.delenv("RESEARCH_BOOST_SIGNALS_ENABLED", raising=False)
    payload = _payload()
    emit_boost_signals(payload, run_date="2026-06-15")
    for lst in ("universe", "buy_candidates"):
        for e in payload[lst]:
            assert e["short_interest_adj"] == 0.0
            assert e["institutional_boost"] == 0.0


def test_emit_forced_with_injected_sources_populates_both_lists():
    payload = _payload()

    class _FakeS3:
        """Serves the newest weekly short_interest.json; no institutional artifact."""

        def get_paginator(self, _op):
            class _P:
                def paginate(self, **_):
                    return [{"Contents": [
                        {"Key": "market_data/weekly/2026-06-01/short_interest.json"},
                        {"Key": "market_data/weekly/2026-06-08/short_interest.json"},
                        {"Key": "market_data/weekly/2026-07-01/short_interest.json"},  # future — skipped
                        {"Key": "market_data/weekly/2026-06-08/daily_closes.json"},
                    ]}]
            return _P()

        def get_object(self, Bucket, Key):  # noqa: N803
            assert Key == "market_data/weekly/2026-06-08/short_interest.json", Key
            import io
            import json
            body = json.dumps({"data": {"HISI": {"short_pct_float": 55.0}}}).encode()
            return {"Body": io.BytesIO(body)}

    def _fake_fetcher(tickers, min_funds_for_signal=3):
        assert "MISI" in tickers
        return {"MISI": {"n_funds_accumulating": min_funds_for_signal + 1}}

    # inject the fetcher via read_institutional_map by monkeypatching import site:
    import scoring.boost_signals as bs
    orig = bs.read_institutional_map
    bs.read_institutional_map = lambda tickers, run_date, **kw: _fake_fetcher(
        tickers, kw["params"]["institutional_min_funds"]
    )
    try:
        emit_boost_signals(payload, run_date="2026-06-15", params=_PARAMS, s3_client=_FakeS3(), force=True)
    finally:
        bs.read_institutional_map = orig

    by = {e["ticker"]: e for e in payload["universe"]}
    assert by["HISI"]["short_interest_adj"] == 4.0
    assert by["MISI"]["institutional_boost"] == 3.0
    # buy_candidates annotated too (backtester reads both lists)
    assert payload["buy_candidates"][0]["short_interest_adj"] == 4.0


def test_emit_reader_gate_defaults_zero_when_source_raises():
    payload = _payload()

    class _BoomS3:
        def get_paginator(self, _op):
            raise RuntimeError("s3 down")

    emit_boost_signals(payload, run_date="2026-06-15", params=_PARAMS, s3_client=_BoomS3(), force=True)
    for e in payload["universe"]:
        assert e["short_interest_adj"] == 0.0
        assert e["institutional_boost"] == 0.0


# ── PIT-safe key selection ───────────────────────────────────────────────────

def test_latest_weekly_key_picks_newest_at_or_before_run_date():
    class _S3:
        def get_paginator(self, _op):
            class _P:
                def paginate(self, **_):
                    return [{"Contents": [
                        {"Key": "market_data/weekly/2026-05-30/short_interest.json"},
                        {"Key": "market_data/weekly/2026-06-06/short_interest.json"},
                        {"Key": "market_data/weekly/2026-06-20/short_interest.json"},  # future
                    ]}]
            return _P()

    key = _latest_weekly_key(_S3(), "b", "2026-06-15", "short_interest.json")
    assert key == "market_data/weekly/2026-06-06/short_interest.json"
