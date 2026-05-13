"""Tests for the regime-conditional narrative penalty (PR feat/narrative-regime-penalty, 2026-05-13).

In BULL regimes, defensive narratives ("oversold bounce", "dividend yield",
"extreme oversold") get penalized; growth narratives ("AI capex", "secular
growth", "breakout") get bonused. Inverted in BEAR regimes. NEUTRAL applies
no adjustment.

Pure text-match logic in scoring/composite.py:compute_narrative_regime_adjustment.
Wired into graph.research_graph.score_aggregator after compute_composite_score.
"""

from scoring.composite import compute_narrative_regime_adjustment


_DEFENSIVE = ["oversold bounce", "extreme oversold", "dividend yield"]
_GROWTH = ["secular growth", "ai capex", "breakout"]


def _adj(text, regime, **overrides):
    """Helper — call adjustment with sensible defaults for testing."""
    cfg = dict(
        bull_defensive_markers=_DEFENSIVE,
        bull_growth_markers=_GROWTH,
        bull_defensive_penalty=12.0,
        bull_growth_bonus=5.0,
        bear_defensive_bonus=8.0,
        bear_growth_penalty=8.0,
        max_marker_hits=3,
    )
    cfg.update(overrides)
    return compute_narrative_regime_adjustment(
        thesis_text=text, market_regime=regime, **cfg
    )


def test_bull_defensive_marker_yields_negative_adjustment():
    """A defensive marker in BULL → penalty (the HD/MCD case)."""
    adj, details = _adj("Extreme oversold typically bounces 8-15% over 2-4 weeks", "bull")
    assert adj < 0, f"Expected penalty, got {adj}"
    assert details["defensive_hits"] >= 1
    # MCD's score 61 - 12 = 49 → below BUY threshold of 55
    assert adj == -12.0


def test_bull_growth_marker_yields_positive_adjustment():
    """A growth marker in BULL → bonus."""
    adj, details = _adj("Secular growth in AI infrastructure", "bull")
    assert adj > 0
    assert details["growth_hits"] >= 1


def test_bear_defensive_marker_inverts_to_bonus():
    """In BEAR, defensive becomes bonus (not penalty)."""
    adj, details = _adj("Oversold bounce in a defensive name", "bear")
    assert adj > 0, f"BEAR defensive should bonus, got {adj}"
    assert details["regime"] == "bear"


def test_bear_growth_marker_inverts_to_penalty():
    """In BEAR, growth becomes penalty (not bonus)."""
    adj, details = _adj("AI capex thesis with secular growth", "bear")
    assert adj < 0


def test_neutral_regime_yields_zero_adjustment():
    """NEUTRAL regime → no adjustment regardless of markers."""
    adj, details = _adj("Oversold bounce with secular growth", "neutral")
    assert adj == 0.0


def test_no_thesis_yields_zero():
    """Empty/missing thesis → zero adjustment, no crash."""
    adj, _ = _adj("", "bull")
    assert adj == 0.0
    adj, _ = _adj(None, "bull")
    assert adj == 0.0


def test_no_regime_yields_zero():
    """Missing regime → zero adjustment, no crash."""
    adj, _ = _adj("Oversold bounce", None)
    assert adj == 0.0


def test_marker_hits_capped():
    """3+ marker hits should not exceed cap. Per-hit attenuation:
    full penalty on 1st, half on 2nd, third on 3rd, nothing past cap."""
    text = "oversold bounce, extreme oversold, dividend yield, mean reversion"
    adj, details = _adj(text, "bull", bull_defensive_markers=[
        "oversold bounce", "extreme oversold", "dividend yield", "mean reversion"
    ])
    # 4 hits, capped at 3. Scale: 1 + 1/2 + 1/3 = 1.833...
    expected_scale = 1.0 + 0.5 + (1.0 / 3.0)
    assert details["defensive_hits"] == 4
    assert details["defensive_capped"] == 3
    assert abs(adj - (-12.0 * expected_scale)) < 0.01


def test_mixed_defensive_and_growth_in_bull_nets():
    """Both defensive AND growth markers → net effect (penalty - bonus)."""
    text = "Oversold bounce but also secular growth"
    adj, details = _adj(text, "bull")
    # 1 defensive hit (-12) + 1 growth hit (+5) = -7
    assert details["defensive_hits"] == 1
    assert details["growth_hits"] == 1
    assert adj == -7.0


def test_case_insensitive_matching():
    """Matchers should be case-insensitive (canonical lowercase)."""
    adj_upper, _ = _adj("OVERSOLD BOUNCE", "bull")
    adj_mixed, _ = _adj("Oversold Bounce", "bull")
    adj_lower, _ = _adj("oversold bounce", "bull")
    assert adj_upper == adj_mixed == adj_lower


def test_unknown_regime_yields_zero():
    """Unknown regime string → no adjustment."""
    adj, details = _adj("Oversold bounce", "caution")
    assert adj == 0.0
    assert "neutral_or_unknown" in details["reason"]
