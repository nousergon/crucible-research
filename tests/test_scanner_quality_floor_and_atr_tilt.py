"""Tests for PR D (quality floor) + PR E (regime-conditional ATR tilt) in
data/scanner.py — 2026-05-13 fast-track intelligent-risk-taking arc.

These compose with the macro-shift / coherence-gate / narrative-penalty
arcs already shipped today. They operate on the population/scanner side
(upstream of agent scoring) rather than the per-pick scoring side.
"""

from unittest.mock import patch, MagicMock

from data.scanner import apply_quality_filter, apply_regime_atr_tilt


# ─────────────────────────────────────────────────────────────────────────
# PR D — apply_quality_filter
# ─────────────────────────────────────────────────────────────────────────


def _mock_yf_ticker(profit_margin, roe, raise_exc=False):
    """Build a mock yfinance Ticker whose .info returns the given values."""
    mock_ticker = MagicMock()
    if raise_exc:
        mock_ticker.info = property(lambda self: (_ for _ in ()).throw(RuntimeError("fetch failed")))
    else:
        mock_ticker.info = {"profitMargins": profit_margin, "returnOnEquity": roe}
    return mock_ticker


def _patch_yf(per_ticker_data: dict):
    """Patch yfinance.Ticker so each call returns mock data per ticker.

    per_ticker_data: {ticker: (profit_margin, roe)} or {ticker: "raise"}.
    """
    def _factory(ticker_arg, *args, **kwargs):
        data = per_ticker_data.get(ticker_arg, (0.10, 0.15))
        if data == "raise":
            raise RuntimeError(f"fetch failed for {ticker_arg}")
        pm, roe = data
        m = MagicMock()
        m.info = {"profitMargins": pm, "returnOnEquity": roe}
        return m
    return patch("yfinance.Ticker", side_effect=_factory)


def test_quality_floor_rejects_unprofitable_with_negative_roe():
    """A name with NEITHER profit margin NOR ROE positive is rejected."""
    candidates = [{"ticker": "JUNK", "sector": "Technology"}]
    with _patch_yf({"JUNK": (-0.05, -0.10)}):
        result = apply_quality_filter(
            candidates, sector_map={"JUNK": "Technology"}, exempt_sectors=[]
        )
    assert result == []


def test_quality_floor_keeps_positive_margin():
    """Pass if profitMargins > 0 (default lenient mode: EITHER/OR)."""
    candidates = [{"ticker": "MSFT", "sector": "Technology"}]
    with _patch_yf({"MSFT": (0.30, -0.05)}):  # positive margin, negative ROE
        result = apply_quality_filter(
            candidates, sector_map={"MSFT": "Technology"}, exempt_sectors=[]
        )
    assert len(result) == 1
    assert result[0]["ticker"] == "MSFT"


def test_quality_floor_keeps_positive_roe():
    """Pass if returnOnEquity > 0 (lenient mode: EITHER/OR)."""
    candidates = [{"ticker": "RKLB", "sector": "Industrials"}]
    with _patch_yf({"RKLB": (-0.02, 0.05)}):  # negative margin, positive ROE
        result = apply_quality_filter(
            candidates, sector_map={"RKLB": "Industrials"}, exempt_sectors=[]
        )
    assert len(result) == 1


def test_quality_floor_strict_mode_requires_both():
    """In strict mode (require_both=True), need positive margin AND positive ROE."""
    candidates = [{"ticker": "RKLB", "sector": "Industrials"}]
    with _patch_yf({"RKLB": (-0.02, 0.05)}):  # only ROE positive
        result = apply_quality_filter(
            candidates,
            sector_map={"RKLB": "Industrials"},
            require_both=True,
            exempt_sectors=[],
        )
    assert result == []  # rejected — strict mode demands both


def test_quality_floor_exempt_sectors_skip_check():
    """Financial / Real Estate / Utilities skip the gate."""
    candidates = [
        {"ticker": "JPM", "sector": "Financial"},
        {"ticker": "O", "sector": "Real Estate"},
    ]
    with _patch_yf({"JPM": (-0.10, -0.20), "O": (-0.05, -0.10)}):
        result = apply_quality_filter(
            candidates,
            sector_map={"JPM": "Financial", "O": "Real Estate"},
            exempt_sectors=["Financial", "Real Estate", "Utilities"],
        )
    assert len(result) == 2  # both kept despite negative metrics


def test_quality_floor_fail_closed_on_fetch_error():
    """A fetch error should cause the candidate to be REJECTED (fail-closed)."""
    candidates = [{"ticker": "BROKEN", "sector": "Technology"}]
    with _patch_yf({"BROKEN": "raise"}):
        result = apply_quality_filter(
            candidates, sector_map={"BROKEN": "Technology"}, exempt_sectors=[]
        )
    assert result == []


# ─────────────────────────────────────────────────────────────────────────
# PR E — apply_regime_atr_tilt
# ─────────────────────────────────────────────────────────────────────────


def _make_cand(ticker, sector, atr):
    return {"ticker": ticker, "sector": sector, "atr_pct": atr}


def test_atr_tilt_bull_drops_bottom_quartile_per_sector():
    """In BULL, drop the bottom-quartile-by-ATR within each sector."""
    candidates = [
        _make_cand("LO_TECH", "Technology", 0.5),
        _make_cand("MID_TECH", "Technology", 1.5),
        _make_cand("HI_TECH", "Technology", 3.0),
        _make_cand("VHI_TECH", "Technology", 5.0),
    ]
    result = apply_regime_atr_tilt(
        candidates, market_regime="bull", sector_map={c["ticker"]: c["sector"] for c in candidates},
        quartile_pct=25, min_sector_size=4,
    )
    tickers = {c["ticker"] for c in result}
    assert "LO_TECH" not in tickers   # bottom dropped
    assert "VHI_TECH" in tickers       # top kept


def test_atr_tilt_bear_drops_top_quartile_per_sector():
    """In BEAR, invert: drop top-quartile to defensive-tilt."""
    candidates = [
        _make_cand("LO_TECH", "Technology", 0.5),
        _make_cand("MID_TECH", "Technology", 1.5),
        _make_cand("HI_TECH", "Technology", 3.0),
        _make_cand("VHI_TECH", "Technology", 5.0),
    ]
    result = apply_regime_atr_tilt(
        candidates, market_regime="bear", sector_map={c["ticker"]: c["sector"] for c in candidates},
        quartile_pct=25, min_sector_size=4,
    )
    tickers = {c["ticker"] for c in result}
    assert "VHI_TECH" not in tickers   # top dropped in bear
    assert "LO_TECH" in tickers         # bottom kept


def test_atr_tilt_neutral_no_op():
    """NEUTRAL regime applies no tilt — pass-through."""
    candidates = [_make_cand(f"T{i}", "Technology", float(i)) for i in range(1, 5)]
    result = apply_regime_atr_tilt(
        candidates, market_regime="neutral", sector_map={c["ticker"]: c["sector"] for c in candidates},
    )
    assert len(result) == len(candidates)


def test_atr_tilt_skips_small_sectors():
    """Sectors with fewer than min_sector_size candidates skip the tilt."""
    candidates = [
        _make_cand("ONE", "Energy", 0.5),
        _make_cand("TWO", "Energy", 5.0),  # Energy has only 2 — skip
        _make_cand("LO_T", "Technology", 0.5),
        _make_cand("M_T", "Technology", 1.5),
        _make_cand("HI_T", "Technology", 3.0),
        _make_cand("VHI_T", "Technology", 5.0),
    ]
    result = apply_regime_atr_tilt(
        candidates, market_regime="bull",
        sector_map={c["ticker"]: c["sector"] for c in candidates},
        min_sector_size=4,
    )
    tickers = {c["ticker"] for c in result}
    # Both Energy candidates kept (sector too small to tilt)
    assert "ONE" in tickers and "TWO" in tickers
    # Tech tilt fired — LO_T (bottom) dropped
    assert "LO_T" not in tickers
    assert "VHI_T" in tickers


def test_atr_tilt_per_sector_isolation():
    """Within-sector tilt: each sector's quartiles computed independently —
    a high-ATR Tech name and a low-ATR Energy name aren't compared."""
    candidates = [
        _make_cand("LO_TECH", "Technology", 1.0),
        _make_cand("M1_TECH", "Technology", 2.0),
        _make_cand("M2_TECH", "Technology", 3.0),
        _make_cand("HI_TECH", "Technology", 4.0),
        _make_cand("LO_ENERGY", "Energy", 5.0),  # would be top in Tech but mid in Energy
        _make_cand("M1_ENERGY", "Energy", 7.0),
        _make_cand("M2_ENERGY", "Energy", 9.0),
        _make_cand("HI_ENERGY", "Energy", 12.0),
    ]
    result = apply_regime_atr_tilt(
        candidates, market_regime="bull",
        sector_map={c["ticker"]: c["sector"] for c in candidates},
        min_sector_size=4,
    )
    tickers = {c["ticker"] for c in result}
    # LO_ENERGY at 5.0 ATR survives (it's bottom in Energy but not in Tech)
    assert "LO_ENERGY" not in tickers   # actually LO_ENERGY IS bottom of Energy → dropped
    assert "HI_ENERGY" in tickers
    # LO_TECH at 1.0 ATR is bottom of Tech → dropped
    assert "LO_TECH" not in tickers
    assert "HI_TECH" in tickers


def test_atr_tilt_missing_atr_kept_as_is():
    """Candidates with missing/None atr_pct are kept (cannot rank → fail-open)."""
    candidates = [
        _make_cand("LO_TECH", "Technology", 1.0),
        _make_cand("M_TECH", "Technology", 2.0),
        _make_cand("HI_TECH", "Technology", 3.0),
        _make_cand("VHI_TECH", "Technology", 4.0),
        {"ticker": "NO_ATR_TECH", "sector": "Technology", "atr_pct": None},
    ]
    result = apply_regime_atr_tilt(
        candidates, market_regime="bull",
        sector_map={c["ticker"]: c["sector"] for c in candidates},
        min_sector_size=4,
    )
    tickers = {c["ticker"] for c in result}
    assert "NO_ATR_TECH" in tickers     # missing ATR kept


def test_atr_tilt_unknown_regime_no_op():
    """Unknown regime string ('caution', '') → no-op."""
    candidates = [_make_cand(f"T{i}", "Technology", float(i)) for i in range(1, 5)]
    for regime in ("caution", "", None):
        result = apply_regime_atr_tilt(
            candidates, market_regime=regime,
            sector_map={c["ticker"]: c["sector"] for c in candidates},
        )
        assert len(result) == len(candidates), f"regime={regime!r} should be no-op"


def test_atr_tilt_empty_candidates_no_op():
    """Empty candidate list → empty result, no crash."""
    result = apply_regime_atr_tilt([], market_regime="bull")
    assert result == []
