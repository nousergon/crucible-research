"""L1995 Phase 5 / L4464 — Research feeds the sector teams a ~60-name
pre-filtered candidate set (∪ held population) instead of the raw
~900-by-sector slice.

Root cause these pin: the sector-team quant ReAct agents were handed
92-217 tickers/sector with ~9-10 reasoning iterations, hit the recursion
limit, produced 0 picks, and triggered a retry storm that overran the
900s Lambda ceiling. Screening the ~60-name candidate set (~10/sector)
converges on the first attempt.

**The SOURCE of that set moved on 2026-08-20** (`alpha-engine-config-I7823`),
and these tests moved with it. It was `candidates/{run_date}/candidates.json`;
it is now the CHAMPION CUT of `universe_membership/latest.json`, resolved by
`scoring.universe_membership.resolve_feed_cut`. Two reasons, both measured:
`candidates.json` is a candidate-generation EXPERIMENT's artifact, and a
cutover inside it silently replaced the researched set with a disjoint one for
four weeks (`alpha-engine-config-I7808`); and under the weekly scanner cadence
it does not exist on four mornings in five, where `latest.json` always does.

The size and union invariants below are unchanged — they are properties of the
feed, not of where it came from.
"""

from __future__ import annotations

import json

import pytest

# ── _resolve_agent_input_set ────────────────────────────────────────────────

class _FakeAM:
    """Minimal ArchiveManager stand-in exposing only load_candidates_json.

    Still used: `candidates.json` remains the source of `scanner_eval_log`,
    which the archive writer needs and the membership artifact does not carry.
    It is no longer the source of the FEED.
    """

    def __init__(self, candidates: dict | None):
        self._candidates = candidates

    def load_candidates_json(self, run_date: str) -> dict | None:
        self.last_run_date = run_date
        return self._candidates


@pytest.fixture
def feed(monkeypatch):
    """Control what `resolve_feed_cut` returns, without S3.

    Patched at `graph.research_graph`'s import site is NOT possible — the
    import is function-local by design (circular-import avoidance), so the
    patch goes on the owning module.
    """
    import scoring.universe_membership as um

    state: dict = {"tickers": ["ACM", "INGR", "TTEK"], "raises": None}

    def _fake(**kwargs):
        if state["raises"] is not None:
            raise state["raises"]
        return list(state["tickers"]), {
            "cut": "attractiveness_top_60",
            "run_date": "2026-05-30",
            "cut_effective_date": "2026-05-30",
            "basis": "attractiveness_rank",
            "size": len(state["tickers"]),
        }

    monkeypatch.setattr(um, "resolve_feed_cut", _fake)
    return state


def _resolve(am, run_date, universe, population):
    """Returns just the agent_input_set list (the tests in this file predate
    the ``scanner_eval_log`` field on the ``AgentInputSetResolution``
    NamedTuple return — config#1458). See
    ``test_resolve_agent_input_set_also_returns_scanner_eval_log`` below for
    the eval-log half of the return shape."""
    from graph.research_graph import _resolve_agent_input_set
    return _resolve_agent_input_set(am, run_date, universe, population).agent_input_set


def test_union_of_feed_cut_and_population(feed):
    am = _FakeAM({})
    out = _resolve(am, "2026-05-30", ["ACM", "INGR", "TTEK", "ZZZ", "QQQ"],
                   population=["AAPL", "MSFT"])
    assert set(out) == {"ACM", "INGR", "TTEK", "AAPL", "MSFT"}


def test_held_population_always_retained_even_if_not_in_the_cut(feed):
    """Holdings must never drop out of coverage — they are unioned in even
    when the cut did not surface them this cycle."""
    feed["tickers"] = ["ACM", "INGR"]
    am = _FakeAM({})
    out = _resolve(am, "2026-05-30", ["ACM", "INGR"], population=["AAPL", "JNJ"])
    assert {"AAPL", "JNJ"}.issubset(set(out))


def test_input_set_is_far_smaller_than_full_universe(feed):
    """The whole point: feed ~60, not ~900. Guards against a regression that
    re-points screening at the raw universe."""
    universe = [f"T{i}" for i in range(903)]
    feed["tickers"] = [f"T{i}" for i in range(60)]
    am = _FakeAM({})
    out = _resolve(am, "2026-05-30", universe, population=["T0", "T1"])
    assert len(out) <= 65  # ~60 cut ∪ a couple held — never ~900


def test_missing_membership_hard_fails_without_sentinel(monkeypatch, feed):
    from scoring.universe_membership import UniverseMembershipError

    monkeypatch.delenv("ALPHA_ENGINE_DRY_RUN_STUB", raising=False)
    feed["raises"] = UniverseMembershipError("no universe_membership/latest.json")
    am = _FakeAM({})
    with pytest.raises(UniverseMembershipError, match="latest.json"):
        _resolve(am, "2026-05-30", ["ACM", "INGR"], population=["AAPL"])


def test_empty_champion_cut_hard_fails_without_sentinel(monkeypatch, feed):
    """An empty feed is indistinguishable from 'the scanner selected nobody',
    so it raises rather than quietly screening only the held population."""
    from scoring.universe_membership import UniverseMembershipError

    monkeypatch.delenv("ALPHA_ENGINE_DRY_RUN_STUB", raising=False)
    feed["raises"] = UniverseMembershipError("carries no tickers under the champion cut")
    am = _FakeAM({})
    with pytest.raises(UniverseMembershipError, match="champion cut"):
        _resolve(am, "2026-05-30", ["ACM", "INGR"], population=["AAPL"])


def test_dry_run_sentinel_falls_back_to_full_universe(monkeypatch, feed):
    """Stub/offline wiring validation tolerates an unresolvable feed — falls
    back to scanner_universe (NOT a real selection). Prod never sets the
    sentinel."""
    from scoring.universe_membership import UniverseMembershipError

    monkeypatch.setenv("ALPHA_ENGINE_DRY_RUN_STUB", "true")
    feed["raises"] = UniverseMembershipError("absent")
    am = _FakeAM(None)
    out = _resolve(am, "2026-05-30", ["ACM", "INGR", "TTEK"], population=["AAPL"])
    assert set(out) == {"ACM", "INGR", "TTEK", "AAPL"}


# ── scanner_eval_log passthrough (config#1458) ───────────────────────────────
#
# Root cause: candidates.json is built in a SEPARATE process (the standalone
# Scanner SF state / lambda/scanner_handler.py) from the one that reads it
# (this Research Lambda). run_quant_filter._last_eval_log is a module-level
# stash local to whichever process called run_quant_filter — so reading it
# here would always be empty. The eval log must instead ride through
# candidates.json itself (the artifact that already crosses the process
# boundary) and then through ResearchState, exactly like agent_input_set.

def test_resolve_agent_input_set_also_returns_scanner_eval_log(feed):
    """The eval log still rides through candidates.json — that artifact is no
    longer the FEED, but it is still the only carrier of the per-ticker gate
    verdict across the Scanner/Research process boundary."""
    from graph.research_graph import _resolve_agent_input_set

    feed["tickers"] = ["ACM", "INGR"]

    eval_log = [
        {"ticker": "ACM", "quant_filter_pass": 1, "scan_path": "momentum"},
        {"ticker": "ZZZ", "quant_filter_pass": 0, "filter_fail_reason": "liquidity"},
    ]
    am = _FakeAM({"scanner_eval_log": eval_log})
    result = _resolve_agent_input_set(
        am, "2026-05-30", ["ACM", "INGR", "ZZZ"], ["AAPL"],
    )
    assert set(result.agent_input_set) == {"ACM", "INGR", "AAPL"}
    assert result.scanner_eval_log == eval_log


def test_resolve_agent_input_set_scanner_eval_log_defaults_empty_when_absent(feed):
    """candidates.json predating this field (or produced with an empty
    eval log) must degrade to [] rather than raising — same fail-soft
    posture as the archive_writer WARN path that consumes this."""
    from graph.research_graph import _resolve_agent_input_set

    am = _FakeAM({})  # no scanner_eval_log key
    result = _resolve_agent_input_set(am, "2026-05-30", ["ACM"], [])
    assert result.scanner_eval_log == []


def test_resolve_agent_input_set_scanner_eval_log_empty_on_dry_run_fallback(monkeypatch, feed):
    """The dry-run-stub full-universe fallback doesn't read a real
    candidates.json, so it must not fabricate an eval log either."""
    from graph.research_graph import _resolve_agent_input_set
    from scoring.universe_membership import UniverseMembershipError

    monkeypatch.setenv("ALPHA_ENGINE_DRY_RUN_STUB", "true")
    feed["raises"] = UniverseMembershipError("absent")
    am = _FakeAM(None)
    result = _resolve_agent_input_set(am, "2026-05-30", ["ACM", "INGR"], ["AAPL"])
    assert result.scanner_eval_log == []


# ── ArchiveManager.load_candidates_json ─────────────────────────────────────

def test_load_candidates_json_reads_correct_key_and_parses():
    from archive.manager import ArchiveManager

    artifact = {"scanner_tickers": ["ACM", "INGR"], "run_date": "2026-05-30"}

    class _AM:
        def _s3_get(self, key):
            self.key = key
            return json.dumps(artifact)

    am = _AM()
    out = ArchiveManager.load_candidates_json(am, "2026-05-30")
    assert am.key == "candidates/2026-05-30/candidates.json"
    assert out == artifact


def test_load_candidates_json_returns_none_when_absent():
    from archive.manager import ArchiveManager

    class _AM:
        def _s3_get(self, key):
            return None

    assert ArchiveManager.load_candidates_json(_AM(), "2026-05-30") is None


# ── sector-team screening reads agent_input_set, not scanner_universe ────────

def test_sector_team_screens_agent_input_set_not_full_universe():
    """The real screening input (sector_team.run_sector_team → get_team_tickers)
    must read ctx.agent_input_set, so a 900-name scanner_universe with a
    60-name agent_input_set screens the 60, not the 900.

    Static source grep (patch-immune; mirrors the repo's other contract
    tests) — guards against a revert to ctx.scanner_universe.
    """
    from pathlib import Path

    from agents.sector_teams.sector_team import SectorTeamContext

    src = (Path(__file__).resolve().parent.parent
           / "agents" / "sector_teams" / "sector_team.py").read_text()
    assert "get_team_tickers(team_id, ctx.agent_input_set" in src, (
        "run_sector_team must screen ctx.agent_input_set (the pre-filtered "
        "candidate set), NOT ctx.scanner_universe (the raw ~900 universe)."
    )
    assert "get_team_tickers(team_id, ctx.scanner_universe" not in src, (
        "the raw-~900 screening handoff (ctx.scanner_universe) must be retired."
    )
    # And SectorTeamContext carries the field.
    assert "agent_input_set" in SectorTeamContext.__dataclass_fields__
