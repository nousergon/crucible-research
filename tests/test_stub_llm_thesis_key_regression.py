"""Regression lock: the ``--stub-llm`` offline path HARD-FAILS on a held
thesis that is missing the required ``final_score`` key.

Closes alpha-engine-config#889.

Context — the bug class this pins (the LNTH/MA "load_latest_theses key
bug"):
    ``archive.manager.load_latest_theses`` loads each held ticker's most
    recent thesis from research.db and is responsible for populating the
    required scoring keys (``final_score`` + the ``quant_score`` /
    ``qual_score`` sub-scores that let ``score_aggregator`` recompute it).
    A historical regression (config PR #46, reverting #42's guard) let
    ``load_latest_theses`` emit a prior thesis with ``final_score`` absent
    *and* no sub-scores. ``score_aggregator`` used to log-and-skip such a
    record, silently DROPPING a held ticker from ``investment_theses`` —
    a data-shape bug that produced a quietly-wrong signals.json with no
    error. The fix (config #42, re-asserted) makes ``score_aggregator``
    HARD-FAIL (``RuntimeError``) on a truly-unscoreable held thesis rather
    than drop it.

What #889 asks us to prove (and what this file locks in):
    alpha-engine-research PR #47 added the ``--stub-llm`` mode
    (``local/run.py --stub-llm`` → ``local.offline_stubs.
    install_llm_only_stubs``): real data + real archive, but every
    Anthropic LLM agent call is replaced with a synthetic stub.  The
    mode was built but never validated to actually CATCH the key bug.
    The risk is that stubbing the LLM agents BYPASSES the same
    ``score_aggregator`` validation the production path flows through —
    which would make ``--stub-llm`` useless as a guardrail for this bug
    class.

    The ``--stub-llm`` held-stock path is ``dry_run._stub_run_sector_team``
    (the function ``install_llm_only_stubs`` swaps in for
    ``run_sector_team``). It carries ``prior_theses[ticker]`` —
    i.e. exactly what ``load_latest_theses`` returned — forward into
    ``thesis_updates`` verbatim. So if ``load_latest_theses`` ever
    re-introduces the missing-key bug, the broken record reaches
    ``score_aggregator`` THROUGH the stub path unchanged, and the run
    must hard-fail.

These tests drive the REAL ``--stub-llm`` machinery (no hand-built
team output): ``install_llm_only_stubs()`` is invoked, the genuine
``_stub_run_sector_team`` produces the ``thesis_updates``, and the real
``score_aggregator`` scores them. They assert:

  1. A held thesis missing ``final_score`` AND both sub-scores
     (the unscoreable LNTH/MA shape) HARD-FAILS the run via the stub
     path — the validation is NOT bypassed in ``--stub-llm`` mode.
  2. The hard-fail comes from the score-validation layer (message names
     the ticker + ``final_score`` + the ``unscoreable`` guardrail), so
     this is the intended catch, not an incidental crash.
  3. ``install_llm_only_stubs`` genuinely repoints ``run_sector_team`` at
     the stub — proving the path under test IS the ``--stub-llm`` path.
  4. The complementary safe case: a held thesis carrying recoverable
     sub-scores (but no ``final_score``) flows through the SAME stub path
     and is recomputed (not dropped, not crashed) — so the hard-fail is
     specific to the genuinely-unscoreable shape, not stub mode per se.

NOTE ON RESTORE: ``install_llm_only_stubs`` permanently rebinds module
globals (it has no restore hook). To avoid the documented order-dependent
module-global leak in this suite, the ``stub_llm_installed`` fixture
snapshots and restores every patched attribute around each test.
"""

from __future__ import annotations

import importlib

import pytest

from agents.sector_teams.team_config import ALL_TEAM_IDS

# The (module_path, attr) set that install_llm_only_stubs rebinds — kept
# in lockstep with local/offline_stubs.py::install_llm_only_stubs so the
# fixture can fully restore them after each test.
_STUBBED_TARGETS = [
    ("agents.macro_agent", "run_macro_agent_with_reflection"),
    ("agents.macro_agent", "run_macro_agent"),
    ("agents.sector_teams.quant_analyst", "run_quant_analyst"),
    ("agents.sector_teams.qual_analyst", "run_qual_analyst"),
    ("agents.sector_teams.peer_review", "run_peer_review"),
    ("agents.sector_teams.sector_team", "run_sector_team"),
    ("agents.investment_committee.ic_cio", "run_cio"),
]
# patch_graph_modules_llm_only also rebinds late-bound names on the graph
# module; snapshot those too.
_GRAPH_REBOUND = [
    ("graph.research_graph", "run_macro_agent_with_reflection"),
    ("graph.research_graph", "run_sector_team"),
    ("graph.research_graph", "run_cio"),
]


def _snapshot(targets):
    snap = {}
    for mod_path, attr in targets:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if hasattr(mod, attr):
            snap[(mod_path, attr)] = getattr(mod, attr)
    return snap


def _restore(snap):
    for (mod_path, attr), original in snap.items():
        mod = importlib.import_module(mod_path)
        setattr(mod, attr, original)


@pytest.fixture
def stub_llm_installed():
    """Install the real ``--stub-llm`` stubs (install_llm_only_stubs +
    patch_graph_modules_llm_only), yield, then restore every rebound
    global so the process-wide patch doesn't bleed into sibling tests."""
    # Import the graph module first so patch_graph_modules_llm_only finds
    # it in sys.modules (it is a no-op for modules not yet imported).
    import graph.research_graph  # noqa: F401

    snap = _snapshot(_STUBBED_TARGETS + _GRAPH_REBOUND)
    from local.offline_stubs import (
        install_llm_only_stubs,
        patch_graph_modules_llm_only,
    )
    install_llm_only_stubs()
    patch_graph_modules_llm_only()
    try:
        yield
    finally:
        _restore(snap)


def _all_clean_team_outputs() -> dict:
    """A full, clean ALL_TEAM_IDS set so the all-agents-strict gate in
    score_aggregator passes — letting us isolate the thesis-key
    validation that runs after it."""
    return {
        tid: {
            "team_id": tid,
            "recommendations": [],
            "thesis_updates": {},
            "error": None,
            "partial": False,
            "partial_reasons": [],
        }
        for tid in ALL_TEAM_IDS
    }


def _held_ctx(prior_thesis: dict, ticker: str = "MA",
              sector: str = "Financials"):
    """Build a minimal SectorTeamContext with exactly one held ticker
    whose prior thesis is ``prior_thesis`` — the shape
    ``load_latest_theses`` returns for a held position."""
    from agents.sector_teams.sector_team import SectorTeamContext

    return SectorTeamContext(
        scanner_universe=[],
        agent_input_set=[],
        sector_map={ticker: sector},
        price_data={},
        technical_scores={},
        market_regime="neutral",
        prior_theses={ticker: prior_thesis},
        held_tickers=[ticker],
        news_data_by_ticker={},
        analyst_data_by_ticker={},
        insider_data_by_ticker={},
        prior_sector_ratings={},
        current_sector_ratings={},
        run_date="2026-06-26",
    )


def _run_stub_path(prior_thesis: dict, team_id: str = "financials",
                   ticker: str = "MA", sector: str = "Financials") -> dict:
    """Drive the genuine ``--stub-llm`` held-stock path end to end and
    return the score_aggregator state ready to score.

    Uses the run_sector_team that install_llm_only_stubs installed (the
    real ``_stub_run_sector_team``) — NOT a hand-built team output — so
    the test exercises the actual offline path, including the verbatim
    carry-forward of ``prior_theses[ticker]`` into ``thesis_updates``.
    """
    from agents.sector_teams import sector_team as st

    # Sanity: we must be running the stub, not the real LLM agent.
    assert st.run_sector_team.__name__ == "_stub_run_sector_team", (
        "install_llm_only_stubs did not repoint run_sector_team at the "
        "stub — the test would not be exercising the --stub-llm path"
    )

    team_output = st.run_sector_team(team_id, _held_ctx(prior_thesis, ticker, sector))

    team_outputs = _all_clean_team_outputs()
    team_outputs[team_id] = team_output
    return {
        "sector_team_outputs": team_outputs,
        "sector_modifiers": {sector: 1.0},
        "sector_map": {ticker: sector},
    }


# ── The bug-class lock ────────────────────────────────────────────────────


class TestStubLlmCatchesMissingThesisKey:
    def test_install_llm_only_stubs_repoints_sector_team(self, stub_llm_installed):
        """Guard: the ``--stub-llm`` installer actually swaps the LLM
        sector-team agent for the synthetic stub. If this ever stops
        being true, the rest of this file would silently stop testing
        the offline path."""
        from agents.sector_teams import sector_team as st

        assert st.run_sector_team.__name__ == "_stub_run_sector_team"

    def test_missing_final_score_and_subscores_hard_fails_via_stub_path(
        self, stub_llm_installed
    ):
        """THE regression. A held thesis missing ``final_score`` AND both
        sub-scores — the exact LNTH/MA ``load_latest_theses`` key-bug
        shape — must HARD-FAIL when it flows through the ``--stub-llm``
        path. A silent pass here would mean stub mode bypasses the
        score validation and could never have caught the data-shape bug."""
        broken_prior = {
            "ticker": "MA",
            "sector": "Financials",
            "rating": "HOLD",
            "conviction": "stable",
            "thesis_summary": "carried-forward held thesis",
            # final_score, quant_score, qual_score ALL absent — the bug.
        }
        import graph.research_graph as rg

        state = _run_stub_path(broken_prior)

        with pytest.raises(RuntimeError) as exc:
            rg.score_aggregator(state)

        msg = str(exc.value)
        # The failure must come from the thesis-key validation, not an
        # incidental crash — name the ticker + the missing key + the
        # feedback-driven guardrail so this is provably the intended catch.
        assert "MA" in msg
        assert "final_score" in msg
        assert "unscoreable" in msg.lower()

    def test_stub_path_carries_broken_prior_thesis_forward_unchanged(
        self, stub_llm_installed
    ):
        """Pin the leak mechanism: the stub sector-team carries the
        broken ``prior_theses`` entry into ``thesis_updates`` verbatim
        (it does NOT invent a final_score), so the bug actually REACHES
        score_aggregator's validation rather than being masked by the
        stub."""
        from agents.sector_teams import sector_team as st

        broken_prior = {
            "ticker": "MA",
            "sector": "Financials",
            "rating": "HOLD",
            "conviction": "stable",
        }
        out = st.run_sector_team("financials", _held_ctx(broken_prior))

        assert "MA" in out["thesis_updates"], (
            "stub sector-team dropped the held ticker before score "
            "validation — the bug would never reach the guard"
        )
        carried = out["thesis_updates"]["MA"]
        assert "final_score" not in carried
        assert "quant_score" not in carried
        assert "qual_score" not in carried

    def test_recoverable_subscores_are_recomputed_not_dropped_via_stub_path(
        self, stub_llm_installed
    ):
        """Complement: a held thesis missing ``final_score`` but carrying
        recoverable sub-scores flows through the SAME ``--stub-llm`` path
        and is RECOMPUTED (CME/HSY/KR shape) — neither dropped nor
        hard-failed. Confirms the hard-fail above is specific to the
        genuinely-unscoreable record, not a side effect of stub mode."""
        import graph.research_graph as rg

        recoverable_prior = {
            "ticker": "MA",
            "sector": "Financials",
            "rating": "HOLD",
            "conviction": "stable",
            "quant_score": 70,
            "qual_score": 80,
            # final_score absent but sub-scores present → recompute path.
        }
        state = _run_stub_path(recoverable_prior)
        out = rg.score_aggregator(state)

        assert "MA" in out["investment_theses"], (
            "recoverable held ticker was silently dropped by the stub "
            "path — recompute branch didn't fire"
        )
        # 0.5 * 70 + 0.5 * 80 + 0 macro shift = 75.0
        assert out["investment_theses"]["MA"]["final_score"] == 75.0
