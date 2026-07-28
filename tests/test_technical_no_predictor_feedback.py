"""`tech_score` must be a function of price/volume indicators ONLY.

alpha-engine-config-I4983. A `predictor_enrichment` block used to adjust the
composite by the predictor's own ``p_up - p_down``. It never fired (nothing
populated those keys), but had it fired it would have closed a feedback loop:

    predictor output -> tech_score -> scanner gate cut -> predictor universe

These tests pin the removal at both layers — behaviour (passing predictor keys
changes nothing) and source (the block is gone, not merely unreachable). The
behavioural test is the one that would have caught the hazard: it fails against
the pre-I4983 implementation, because there the enrichment branch fires as soon
as a caller supplies the keys.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from scoring.technical import compute_technical_score

# Indicators shaped like price_fetcher.compute_technical_indicators() output.
# Chosen so the composite lands mid-range, leaving headroom in both directions —
# a score pinned at the 0/100 clamp would mask an adjustment.
_BASE_INDICATORS = {
    "rsi_14": 55.0,
    "macd_cross": 0.5,
    "macd_above_zero": True,
    "price_vs_ma50": 3.0,
    "price_vs_ma200": 8.0,
    "momentum_20d": 0.04,
}


def _score(**extra) -> float:
    return compute_technical_score({**_BASE_INDICATORS, **extra}, market_regime="neutral")


def test_predictor_keys_do_not_move_tech_score() -> None:
    """The load-bearing assertion: predictor output must not reach tech_score.

    Fails against the pre-I4983 implementation — there, a bullish high-confidence
    pair adds up to +10 points.
    """
    baseline = _score()

    bullish = _score(p_up=0.95, p_down=0.05, prediction_confidence=0.99)
    bearish = _score(p_up=0.05, p_down=0.95, prediction_confidence=0.99)

    assert bullish == baseline, (
        "tech_score moved when a bullish predictor signal was supplied "
        f"({baseline} -> {bullish}) — predictor output is feeding back into the "
        "scanner's own ranking (alpha-engine-config-I4983)"
    )
    assert bearish == baseline, (
        "tech_score moved when a bearish predictor signal was supplied "
        f"({baseline} -> {bearish}) — predictor output is feeding back into the "
        "scanner's own ranking (alpha-engine-config-I4983)"
    )


def test_predictor_keys_do_not_move_tech_score_below_legacy_confidence_gate() -> None:
    """Also pinned below the old 0.60/0.65 gate, so a reintroduction that merely
    lowered the threshold cannot slip through the test above."""
    baseline = _score()
    assert _score(p_up=0.9, p_down=0.1, prediction_confidence=0.10) == baseline


def test_enrichment_block_is_absent_from_source() -> None:
    """Source-level guard: the branch is removed, not left unreachable.

    Dead code guarding a live hazard is the failure mode here — an unreachable
    block becomes reachable the moment an unrelated change enriches the
    `indicators` dict.
    """
    # Strip comments and docstrings — the removal is documented in a comment
    # that necessarily names the thing removed, and a guard that trips on its
    # own rationale is noise. Only EXECUTABLE code is checked.
    src = inspect.getsource(compute_technical_score)
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for token in ("predictor_enrichment", "p_down", "prediction_confidence"):
        assert token not in code, (
            f"{token!r} is back in compute_technical_score's executable code — "
            "tech_score must not read predictor output "
            "(alpha-engine-config-I4983)"
        )


def test_sample_config_does_not_advertise_enrichment() -> None:
    """The config key must go with the code. Leaving it advertises a knob that
    does nothing, which is how the original block survived unnoticed."""
    sample = Path(__file__).resolve().parents[1] / "config" / "scoring.sample.yaml"
    text = sample.read_text(encoding="utf-8")
    active = [ln for ln in text.splitlines() if "predictor_enrichment" in ln and not ln.lstrip().startswith("#")]
    assert not active, f"config/scoring.sample.yaml still declares predictor_enrichment as an active key: {active}"
