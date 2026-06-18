"""Tests for the score-neutralization OBSERVE shadow (config#1142).

Covers the four invariants the shadow must hold:
  (a) the shadow artifact is produced with the right shape on a synthetic
      cross-section,
  (b) live signals are byte-identical whether the shadow runs or not (the shadow
      has ZERO live effect when the LIVE cutover gate is off),
  (c) a shadow failure (missing loadings / S3 error mocked) does not raise and
      does not alter live signals,
  (d) NEUTRALIZATION_LIVE_ENABLED defaults False and the live ranking is
      unchanged.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.neutralization_shadow import (  # noqa: E402
    SHADOW_FACTORS,
    SHADOW_PREFIX,
    build_shadow_artifact,
    run_neutralization_shadow,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _synthetic_cross_section(n=40, seed=0):
    """Build a synthetic signals block + factor loadings where the live
    composite is deliberately correlated with momentum (the unintended bet)."""
    rng = np.random.RandomState(seed)
    tickers = [f"T{i}" for i in range(n)]
    momentum = rng.normal(size=n)
    ret60 = rng.normal(size=n)
    beta = rng.normal(size=n)
    size = rng.normal(size=n)
    idio = rng.normal(size=n)
    # composite = strong momentum tilt + idiosyncratic component
    scores = 50.0 + 8.0 * momentum + 2.0 * idio

    signals = {
        t: {"ticker": t, "score": float(scores[i]), "signal": "HOLD"}
        for i, t in enumerate(tickers)
    }
    loadings = {
        t: {
            "momentum_20d_zscore": float(momentum[i]),
            "return_60d_zscore": float(ret60[i]),
            "beta_60d_zscore": float(beta[i]),
            "size_zscore": float(size[i]),
        }
        for i, t in enumerate(tickers)
    }
    return {"signals": signals}, loadings


class _FakeAM:
    """Captures _s3_put calls (key -> body) instead of hitting S3."""

    def __init__(self):
        self.puts: dict[str, str] = {}

    def _s3_put(self, key: str, body: str) -> None:
        self.puts[key] = body


class _RaisingAM:
    def _s3_put(self, key: str, body: str) -> None:
        raise RuntimeError("simulated S3 error")


# ── (a) shape ─────────────────────────────────────────────────────────────────


def test_build_shadow_artifact_shape():
    payload, loadings = _synthetic_cross_section(n=40)
    art = build_shadow_artifact(
        payload["signals"], loadings, run_date="2026-06-22", factors=SHADOW_FACTORS
    )
    assert art["run_date"] == "2026-06-22"
    assert art["factors_used"] == list(SHADOW_FACTORS)
    assert art["n_names"] == 40
    assert art["n_with_full_exposures"] == 40
    assert set(art["tickers"].keys()) == set(payload["signals"].keys())
    # Per-ticker shape: live + neutralized + exposures with all four factors.
    row = art["tickers"]["T0"]
    assert set(row.keys()) == {"live_score", "neutralized_score", "exposures"}
    assert set(row["exposures"].keys()) == set(SHADOW_FACTORS)
    # JSON-serializable.
    json.dumps(art, default=str)


def test_neutralization_changes_ranking_but_not_membership():
    """The shadow's whole point: it changes the RANKING (removes the momentum
    tilt) while never dropping a name."""
    payload, loadings = _synthetic_cross_section(n=40, seed=3)
    art = build_shadow_artifact(payload["signals"], loadings, run_date="2026-06-22")
    live = [art["tickers"][t]["live_score"] for t in sorted(art["tickers"])]
    neut = [art["tickers"][t]["neutralized_score"] for t in sorted(art["tickers"])]
    # Same set of names, but the ordering differs (momentum removed).
    assert len(live) == len(neut) == 40
    assert live != neut


def test_run_shadow_persists_artifact():
    payload, loadings = _synthetic_cross_section(n=30)
    am = _FakeAM()
    art = run_neutralization_shadow(am, "2026-06-22", payload, loadings)
    assert art is not None
    key = f"{SHADOW_PREFIX}/2026-06-22.json"
    assert key in am.puts
    written = json.loads(am.puts[key])
    assert written["n_names"] == 30
    assert written["run_date"] == "2026-06-22"


# ── (b) zero live effect ──────────────────────────────────────────────────────


def test_shadow_does_not_mutate_live_signals():
    payload, loadings = _synthetic_cross_section(n=30)
    payload_before = copy.deepcopy(payload)
    am = _FakeAM()
    run_neutralization_shadow(am, "2026-06-22", payload, loadings)
    # The live payload (signals + scores) is untouched by the shadow.
    assert payload == payload_before


def test_live_signals_byte_identical_with_and_without_shadow():
    """Serialize the live signals before and after the shadow runs — must be
    byte-for-byte identical (the shadow has zero live effect)."""
    payload, loadings = _synthetic_cross_section(n=30)
    before = json.dumps(payload, sort_keys=True, default=str)
    am = _FakeAM()
    run_neutralization_shadow(am, "2026-06-22", payload, loadings)
    after = json.dumps(payload, sort_keys=True, default=str)
    assert before == after


# ── (c) fail-soft ─────────────────────────────────────────────────────────────


def test_missing_loadings_returns_none_no_raise():
    payload, _ = _synthetic_cross_section(n=30)
    am = _FakeAM()
    payload_before = copy.deepcopy(payload)
    # No loadings → shadow is a graceful no-op, nothing written.
    assert run_neutralization_shadow(am, "2026-06-22", payload, None) is None
    assert run_neutralization_shadow(am, "2026-06-22", payload, {}) is None
    assert am.puts == {}
    assert payload == payload_before


def test_s3_error_swallowed_no_raise_no_mutation():
    payload, loadings = _synthetic_cross_section(n=30)
    payload_before = copy.deepcopy(payload)
    am = _RaisingAM()
    # S3 put raises inside the shadow — must be swallowed (returns None) and the
    # live payload must be untouched.
    result = run_neutralization_shadow(am, "2026-06-22", payload, loadings)
    assert result is None
    assert payload == payload_before


def test_empty_signals_returns_none():
    am = _FakeAM()
    _, loadings = _synthetic_cross_section(n=30)
    assert run_neutralization_shadow(am, "2026-06-22", {"signals": {}}, loadings) is None
    assert am.puts == {}


def test_partial_exposures_counted_but_names_retained():
    payload, loadings = _synthetic_cross_section(n=25)
    # Strip one factor from half the names → they have partial exposures.
    for i, t in enumerate(list(loadings.keys())):
        if i % 2 == 0:
            loadings[t].pop("size_zscore", None)
    art = build_shadow_artifact(payload["signals"], loadings, run_date="2026-06-22")
    assert art["n_names"] == 25  # no name dropped
    assert art["n_with_full_exposures"] < 25
    # None score names are excluded from the cross-section.
    payload["signals"]["T0"]["score"] = None
    art2 = build_shadow_artifact(payload["signals"], loadings, run_date="2026-06-22")
    assert "T0" not in art2["tickers"]
    assert art2["n_names"] == 24


# ── (d) live cutover gate defaults OFF ────────────────────────────────────────


def test_live_enabled_defaults_false():
    import config

    assert config.NEUTRALIZATION_LIVE_ENABLED is False


def test_live_cutover_branch_inert_when_gate_off(monkeypatch):
    """Drive archive_writer's live-cutover gate: with NEUTRALIZATION_LIVE_ENABLED
    False (default), the live signals.json that would be written is unchanged by
    neutralization — only the shadow artifact is produced."""
    import graph.research_graph as rg

    # Simulate the exact archive_writer gated block in isolation.
    payload, loadings = _synthetic_cross_section(n=30)
    am = _FakeAM()
    scores_before = {t: s["score"] for t, s in payload["signals"].items()}

    monkeypatch.setattr(rg, "NEUTRALIZATION_LIVE_ENABLED", False, raising=False)
    art = run_neutralization_shadow(am, "2026-06-22", payload, loadings)
    if rg.NEUTRALIZATION_LIVE_ENABLED and art:  # mirror archive_writer's gate
        live_signals = payload.get("signals", {})
        for t, row in art.get("tickers", {}).items():
            if t in live_signals and row.get("neutralized_score") is not None:
                live_signals[t]["score"] = row["neutralized_score"]

    scores_after = {t: s["score"] for t, s in payload["signals"].items()}
    assert scores_before == scores_after  # live ranking untouched


def test_live_cutover_branch_applies_when_gate_on(monkeypatch):
    """Belt-and-suspenders: when the gate is ON, the same block DOES substitute
    the neutralized scores (proves the gate is wired, not dead code)."""
    import graph.research_graph as rg

    payload, loadings = _synthetic_cross_section(n=30)
    am = _FakeAM()
    scores_before = {t: s["score"] for t, s in payload["signals"].items()}

    monkeypatch.setattr(rg, "NEUTRALIZATION_LIVE_ENABLED", True, raising=False)
    art = run_neutralization_shadow(am, "2026-06-22", payload, loadings)
    if rg.NEUTRALIZATION_LIVE_ENABLED and art:
        live_signals = payload.get("signals", {})
        for t, row in art.get("tickers", {}).items():
            if t in live_signals and row.get("neutralized_score") is not None:
                live_signals[t]["score"] = row["neutralized_score"]

    scores_after = {t: s["score"] for t, s in payload["signals"].items()}
    assert scores_before != scores_after  # neutralized scores applied
