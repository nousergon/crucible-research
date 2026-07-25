"""Tests for the attractiveness champion candidate feed (config#1400 / ARCHITECTURE §43).

Two layers:
  1. ``attractiveness_from_factor_profiles`` — the SSOT helper that scores the
     scanned universe directly from factor profiles (validated byte-identical to
     the live ``build_universe_board`` against production data).
  2. ``rank_candidates_by_attractiveness_node`` — the graph node that, when
     enabled, overwrites ``agent_input_set`` with the top-N attractiveness
     selection ∪ held population; default-off no-op; fail-safe on any error.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_board import attractiveness_from_factor_profiles  # noqa: E402


def _profiles(n: int = 20) -> dict:
    """Synthetic 6-pillar profiles: lower index = more attractive (every pillar
    higher), so the deterministic top-N is T0, T1, …"""
    pillars = ("quality_score", "value_score", "momentum_score",
               "growth_score", "stewardship_score", "low_vol_score")
    out = {}
    for i in range(n):
        v = float(100 - i * (100.0 / n))   # T0 highest … T{n-1} lowest
        out[f"T{i}"] = {"sector": "Tech", **dict.fromkeys(pillars, v)}
    return out


# ── Layer 1: the SSOT helper ─────────────────────────────────────────────────

def test_helper_ranks_winners_high():
    scores = attractiveness_from_factor_profiles(_profiles(20), pillar_weights=None)
    assert scores, "helper returned no scores"
    ranked = sorted(
        (t for t, v in scores.items() if v.get("attractiveness_score") is not None),
        key=lambda t: scores[t]["attractiveness_score"], reverse=True,
    )
    # T0 is best on every pillar → must rank first; T19 last.
    assert ranked[0] == "T0", ranked[:5]
    assert ranked[-1] == "T19", ranked[-5:]


def test_helper_is_deterministic():
    a = attractiveness_from_factor_profiles(_profiles(15), pillar_weights=None)
    b = attractiveness_from_factor_profiles(_profiles(15), pillar_weights=None)
    assert {t: a[t]["attractiveness_score"] for t in a} == {
        t: b[t]["attractiveness_score"] for t in b
    }


def test_helper_skips_non_dict_profiles():
    profiles = _profiles(5)
    profiles["JUNK"] = None  # malformed row must be skipped, not crash
    scores = attractiveness_from_factor_profiles(profiles, pillar_weights=None)
    assert "JUNK" not in scores
    assert len(scores) == 5


# ── Layer 2: the graph node ──────────────────────────────────────────────────
# Importing the node pulls graph.research_graph (langgraph + the pinned lib);
# skip ONLY the node tests if unavailable locally (CI installs
# nousergon-lib@pin and runs them). The helper tests above always run.
try:
    import graph.research_graph as rg  # noqa: E402
    _RG_IMPORT_ERR = None
except Exception as _e:  # pragma: no cover - env-dependent
    rg = None
    _RG_IMPORT_ERR = _e

needs_rg = pytest.mark.skipif(
    rg is None,
    reason=f"graph.research_graph needs the pinned nousergon-lib (present in CI): {_RG_IMPORT_ERR}",
)


def _patch(monkeypatch, enabled: bool, top_n: int, profiles: dict):
    monkeypatch.setattr(rg, "ATTRACTIVENESS_FEED_ENABLED", enabled, raising=False)
    monkeypatch.setattr(rg, "ATTRACTIVENESS_FEED_TOP_N", top_n, raising=False)
    import scoring.universe_board as ub
    monkeypatch.setattr(ub, "_read_factor_profiles", lambda *a, **k: profiles)


@needs_rg
def test_node_disabled_is_noop(monkeypatch):
    _patch(monkeypatch, enabled=False, top_n=10, profiles=_profiles(30))
    out = rg.rank_candidates_by_attractiveness_node(
        {"run_date": "2026-06-26", "agent_input_set": ["X"], "population_tickers": []}
    )
    assert out == {}  # disabled → existing feed stands


@needs_rg
def test_node_enabled_reranks_to_topN(monkeypatch):
    _patch(monkeypatch, enabled=True, top_n=10, profiles=_profiles(30))
    out = rg.rank_candidates_by_attractiveness_node(
        {"run_date": "2026-06-26", "agent_input_set": ["OLD1", "OLD2"], "population_tickers": []}
    )
    sel = set(out["agent_input_set"])
    assert sel == {f"T{i}" for i in range(10)}, sel  # top-10 by attractiveness
    assert "OLD1" not in sel  # tech_score feed replaced


@needs_rg
def test_node_always_retains_population(monkeypatch):
    _patch(monkeypatch, enabled=True, top_n=5, profiles=_profiles(30))
    out = rg.rank_candidates_by_attractiveness_node(
        {"run_date": "2026-06-26", "agent_input_set": [], "population_tickers": ["HELD1", "HELD2"]}
    )
    sel = set(out["agent_input_set"])
    assert {"HELD1", "HELD2"} <= sel, "held population must be retained for HOLD/EXIT"
    assert {f"T{i}" for i in range(5)} <= sel


@needs_rg
def test_node_failsafe_on_empty_profiles(monkeypatch):
    _patch(monkeypatch, enabled=True, top_n=10, profiles={})  # no profiles → must not break the run
    out = rg.rank_candidates_by_attractiveness_node(
        {"run_date": "2026-06-26", "agent_input_set": ["KEEP"], "population_tickers": []}
    )
    assert out == {}  # fail-safe: existing feed stands
