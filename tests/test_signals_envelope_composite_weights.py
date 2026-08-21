"""alpha-engine-config-I7678 — the declared composite must be the one that runs.

Measured 2026-08-18 on ``s3://alpha-engine-research/signals/latest.json``
(producer ``signals_envelope``, run_date 2026-08-14): ``quant_score`` non-null
on 903/903, ``qual_score`` non-null on 0/903, ``score == quant_score`` on
903/903 — while the backtester's ``weight_optimizer`` reported a live vector of
``{quant: 0.5, qual: 0.5}``. The declared vector described a blend with no
producer since the 2026-07-12 research-graph retirement (config#1580).

These tests pin the three properties that keep that from recurring:

  1. the EFFECTIVE weights are derived per run and stamped on the envelope;
  2. an unarmed ``qual_score`` reappearing changes NO live score;
  3. an envelope whose scores no declared weight explains RAISES.
"""

from __future__ import annotations

import pytest

from scoring import signals_envelope as se


def _stock(ticker: str, score: float | None = 70.0, **extra) -> dict:
    return {
        "ticker": ticker,
        "sector": "Technology",
        "attractiveness_score": score,
        "pillars": {"quality": 60.0},
        **extra,
    }


def _board(stocks: list[dict]) -> dict:
    return {"stocks": stocks}


# ── 1. Effective weights are stamped and derived ────────────────────────────


def test_declared_vector_is_the_quant_only_composite_that_runs():
    """The declared weights ARE the running rule — not an aspirational blend."""
    assert se.DECLARED_COMPOSITE_WEIGHTS == {"quant": 1.0, "qual": 0.0}
    assert se.QUAL_COMPOSITE_ARMED is False


def test_envelope_stamps_declared_and_effective_weights_with_coverage():
    envelope = se.build_signals_envelope(
        "2026-08-21", _board([_stock("AAA"), _stock("BBB", 40.0)]), None,
    )
    cw = envelope["composite_weights"]
    assert cw["declared"] == {"quant": 1.0, "qual": 0.0}
    assert cw["effective"] == {"quant": 1.0, "qual": 0.0}
    assert cw["coverage"] == {"quant": 2, "qual": 0}
    assert cw["n"] == 2
    assert cw["qual_armed"] is False


def test_effective_weights_report_zero_coverage_for_an_absent_half():
    """The live 903/903 shape: qual coverage 0, effective qual weight 0.0."""
    entries = se.build_universe_entries([_stock(f"T{i}") for i in range(5)])
    cw = se.compute_composite_weights(entries)
    assert cw["coverage"]["qual"] == 0
    assert cw["effective"]["qual"] == 0.0
    assert cw["effective"]["quant"] == 1.0


def test_score_equals_quant_score_on_every_row():
    """The identity the live artifact exhibits on 903/903."""
    entries = se.build_universe_entries(
        [_stock("AAA", 91.5), _stock("BBB", 0.11), _stock("CCC", 100.0)]
    )
    assert entries
    for e in entries:
        assert e["score"] == e["quant_score"]
        assert e["sub_scores"] == {"quant": e["quant_score"], "qual": None}


# ── 2. The arming gate ──────────────────────────────────────────────────────


def test_unarmed_qual_score_on_the_board_changes_no_live_score():
    """A qual producer reappearing must not move a score without arming.

    This is the Closes-when assertion of I7678: the board carries a qual
    score that would drag a 90 down to 50 under the retired 50/50 blend, and
    every published score is unchanged.
    """
    board = _board([_stock("AAA", 90.0, qual_score=10.0)])
    envelope = se.build_signals_envelope("2026-08-21", board, None)
    entry = envelope["universe"][0]
    assert entry["score"] == 90.0
    assert entry["quant_score"] == 90.0
    assert entry["qual_score"] is None
    assert entry["sub_scores"] == {"quant": 90.0, "qual": None}
    cw = envelope["composite_weights"]
    assert cw["coverage"]["qual"] == 0
    assert cw["effective"]["qual"] == 0.0
    assert cw["qual_armed"] is False


def test_arming_alone_still_does_not_reweight_without_a_declared_weight(monkeypatch):
    """Arming ADMITS the sub-score; it does not silently give it a weight.

    Two explicit steps are required, and this pins the first one on its own:
    the sub-score becomes visible (coverage 1, published qual_score) while the
    composite still resolves to pure quant because the declared qual weight is
    still 0.0.
    """
    monkeypatch.setattr(se, "QUAL_COMPOSITE_ARMED", True)
    board = _board([_stock("AAA", 90.0, qual_score=10.0)])
    envelope = se.build_signals_envelope("2026-08-21", board, None)
    entry = envelope["universe"][0]
    assert entry["qual_score"] == 10.0
    assert entry["score"] == 90.0
    cw = envelope["composite_weights"]
    assert cw["qual_armed"] is True
    assert cw["coverage"]["qual"] == 1
    # Declared weight still 0.0 → effective still quant-only.
    assert cw["effective"] == {"quant": 1.0, "qual": 0.0}


def test_both_steps_together_produce_the_blend_and_say_so(monkeypatch):
    """Guard-fails-without-the-fix check: with BOTH steps taken the blend is
    live AND the stamped effective vector reports it — so the transparency
    property survives a future restoration rather than only describing today.
    """
    monkeypatch.setattr(se, "QUAL_COMPOSITE_ARMED", True)
    monkeypatch.setattr(
        se, "DECLARED_COMPOSITE_WEIGHTS", {"quant": 0.5, "qual": 0.5}
    )
    board = _board([_stock("AAA", 90.0, qual_score=10.0)])
    envelope = se.build_signals_envelope("2026-08-21", board, None)
    entry = envelope["universe"][0]
    assert entry["score"] == pytest.approx(50.0)
    cw = envelope["composite_weights"]
    assert cw["declared"] == {"quant": 0.5, "qual": 0.5}
    assert cw["effective"] == {"quant": 0.5, "qual": 0.5}
    assert cw["coverage"] == {"quant": 1, "qual": 1}


def test_a_half_with_a_weight_but_no_coverage_renormalises_rather_than_halving(
    monkeypatch,
):
    """The failure mode worth ruling out first: a null half must never be
    treated as a zero score. Declared 50/50 with qual absent yields the quant
    score untouched, and the stamped effective vector says 1.0/0.0 — the exact
    divergence that went unrecorded for nine weeks."""
    monkeypatch.setattr(
        se, "DECLARED_COMPOSITE_WEIGHTS", {"quant": 0.5, "qual": 0.5}
    )
    board = _board([_stock("AAA", 90.0)])
    envelope = se.build_signals_envelope("2026-08-21", board, None)
    assert envelope["universe"][0]["score"] == 90.0
    cw = envelope["composite_weights"]
    assert cw["declared"] == {"quant": 0.5, "qual": 0.5}
    assert cw["effective"] == {"quant": 1.0, "qual": 0.0}


# ── 3. Fail loud when nothing explains the scores ───────────────────────────


def test_no_populated_weighted_subscore_raises(monkeypatch):
    monkeypatch.setattr(
        se, "DECLARED_COMPOSITE_WEIGHTS", {"quant": 0.0, "qual": 1.0}
    )
    with pytest.raises(ValueError, match="no weighted sub-score is populated"):
        se.build_signals_envelope("2026-08-21", _board([_stock("AAA", 90.0)]), None)


def test_empty_entry_list_does_not_raise():
    """``build_signals_envelope`` already refuses an empty board upstream;
    the weights helper itself must stay total for an empty list."""
    cw = se.compute_composite_weights([])
    assert cw["n"] == 0
    assert cw["coverage"] == {"quant": 0, "qual": 0}
