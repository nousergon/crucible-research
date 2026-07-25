"""Tests for the PILLAR_EMIT vs pillar_weights coherence guard in config.py.

Defense-in-depth against the 5/21 AQR cutover incident class — non-zero
``pillar_weights`` with ``PILLAR_EMIT_ENABLED=false`` causes the composite
to collapse to 0 for every pick (composite reads pillar_assessments that
the qual analyst never emits). The guard raises at module load so a
misconfigured scoring.yaml fails before reaching the Saturday SF.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import _validate_pillar_emit_coherence

_FAKE_PATH = Path("/fake/scoring.yaml")


def test_both_off_passes():
    """Σ pillar_weights = 0 + PILLAR_EMIT=false is the pre-pillar baseline."""
    _validate_pillar_emit_coherence(
        pillar_weights_sum=0.0,
        pillar_emit_enabled=False,
        source_path=_FAKE_PATH,
    )


def test_emit_on_with_zero_weights_passes():
    """Σ pillar_weights = 0 + PILLAR_EMIT=true is harmless waste, not a bug.

    The 2nd extraction call fires but the composite ignores its output —
    no behavior change beyond the LLM cost. Allowed by design (e.g. soak
    a Saturday SF on PILLAR_EMIT to verify emission before flipping
    weights).
    """
    _validate_pillar_emit_coherence(
        pillar_weights_sum=0.0,
        pillar_emit_enabled=True,
        source_path=_FAKE_PATH,
    )


def test_emit_on_with_nonzero_weights_passes():
    """Σ pillar_weights = 1.0 + PILLAR_EMIT=true is the post-cutover steady state."""
    _validate_pillar_emit_coherence(
        pillar_weights_sum=1.0,
        pillar_emit_enabled=True,
        source_path=_FAKE_PATH,
    )


def test_nonzero_weights_with_emit_off_raises():
    """5/21-class misconfiguration: weights live, emission off → composite collapse.

    This is the exact preventive purpose of the guard. Mirrors the
    cutover-diagnostic-recipe.md root-cause candidate (a).
    """
    with pytest.raises(ValueError) as excinfo:
        _validate_pillar_emit_coherence(
            pillar_weights_sum=1.0,
            pillar_emit_enabled=False,
            source_path=_FAKE_PATH,
        )
    msg = str(excinfo.value)
    assert "pillar_weights sum to 1.000000" in msg
    assert "pillar_emit.enabled is false" in msg
    assert str(_FAKE_PATH) in msg


def test_epsilon_boundary_treats_near_zero_as_zero():
    """Numeric guard uses 1e-6 epsilon — float-arithmetic residual is not a misconfig."""
    _validate_pillar_emit_coherence(
        pillar_weights_sum=1e-9,
        pillar_emit_enabled=False,
        source_path=_FAKE_PATH,
    )


def test_partial_pillar_weight_still_raises():
    """Even a small but real pillar weight (e.g. 50/50 half-pillar-ramp) must

    require PILLAR_EMIT=true. The half-ramp config from
    `cutover-diagnostic-recipe.md` sets Σ pillar = 0.5 — this case is the
    one most likely to slip past an operator's eye.
    """
    with pytest.raises(ValueError, match="pillar_weights sum to 0.500000"):
        _validate_pillar_emit_coherence(
            pillar_weights_sum=0.5,
            pillar_emit_enabled=False,
            source_path=_FAKE_PATH,
        )
