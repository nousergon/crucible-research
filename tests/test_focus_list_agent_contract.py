"""Tests for the agent-contract change behind FOCUS_LIST_GATING_ENABLED (PR 4
of the scanner-placement arc, ``alpha-engine-docs/private/scanner-260514.md``).

Covers:
  - compute_focus_list_node (uses current state["market_regime"] from serial macro)
  - _render_focus_list_for_prompt (prompt-block rendering)
  - agent_override telemetry on @tool get_factor_profile
  - _compute_focus_list_audit_lookup PR 4 projection path (vs legacy fallback)
  - SectorTeamContext focus_list / override_tickers threading

Composes with:
  - Stage B (PR #185) — macro_economist_node runs serial upstream of dispatch
  - Stage D' Wire 1 (PR #188) — regime_intensity_z + pick gate (orthogonal,
    additive — both arcs add SectorTeamContext fields independently)
"""

from unittest.mock import patch

import pandas as pd
import pytest

# ── compute_focus_list_node ─────────────────────────────────────────────────


class TestComputeFocusListNode:
    @pytest.fixture
    def profiles(self):
        return {
            "NVDA": {
                "sector": "Technology", "quality_score": 70.0,
                "momentum_score": 95.0, "value_score": 20.0, "low_vol_score": 25.0,
            },
            "MSFT": {
                "sector": "Technology", "quality_score": 90.0,
                "momentum_score": 60.0, "value_score": 40.0, "low_vol_score": 55.0,
            },
        }

    def test_empty_when_factor_blend_disabled(self, profiles):
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", False):
            result = rg.compute_focus_list_node({"market_regime": "bull"})
        assert result == {"focus_list_by_team": {}}

    def test_empty_when_factor_artifact_missing(self):
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3", return_value=None):
            result = rg.compute_focus_list_node({"market_regime": "bull"})
        assert result == {"focus_list_by_team": {}}

    def test_uses_current_state_market_regime(self, profiles):
        """Post-Stage-B: regime is set by serial macro upstream — focus list
        consumes it from state, not from prior_macro_snapshots."""
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3", return_value=profiles):
            bull = rg.compute_focus_list_node({"market_regime": "bull"})
            bear = rg.compute_focus_list_node({"market_regime": "bear"})
        bull_tech = bull["focus_list_by_team"].get("technology", [])
        bear_tech = bear["focus_list_by_team"].get("technology", [])
        # BULL favors momentum-heavy NVDA; BEAR favors quality/low-vol MSFT
        assert bull_tech and bull_tech[0]["ticker"] == "NVDA"
        assert bear_tech and bear_tech[0]["ticker"] == "MSFT"

    def test_returns_serialized_dicts(self, profiles):
        """State surface stays primitive-only — entries are dicts, not
        FocusListEntry instances."""
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3", return_value=profiles):
            result = rg.compute_focus_list_node({"market_regime": "bull"})
        for entries in result["focus_list_by_team"].values():
            for e in entries:
                assert isinstance(e, dict)
                assert "ticker" in e and "focus_score" in e and "stance" in e

    def test_neutral_default_when_no_regime(self, profiles):
        """Missing market_regime in state → defaults to neutral."""
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", True), \
             patch.object(rg, "read_factor_profiles_from_s3", return_value=profiles):
            result = rg.compute_focus_list_node({})
        tickers = {
            e["ticker"]
            for entries in result["focus_list_by_team"].values()
            for e in entries
        }
        assert tickers == {"NVDA", "MSFT"}


# ── _render_focus_list_for_prompt ───────────────────────────────────────────


class TestRenderFocusListForPrompt:
    def test_empty_input_returns_empty_string(self):
        from agents.sector_teams.quant_analyst import _render_focus_list_for_prompt
        assert _render_focus_list_for_prompt([]) == ""

    def test_header_and_row_present(self):
        from agents.sector_teams.quant_analyst import _render_focus_list_for_prompt
        out = _render_focus_list_for_prompt([{
            "ticker": "NVDA", "sector": "Technology", "stance": "momentum",
            "focus_score": 78.5,
            "momentum_score": 95.0, "quality_score": 70.0,
            "value_score": 20.0, "low_vol_score": 25.0,
        }])
        assert "TICKER" in out
        assert "momentum_p" in out
        assert "NVDA" in out
        assert "momentum" in out
        assert "78.5" in out  # focus_score rendered with 1 decimal

    def test_missing_factor_renders_question_mark(self):
        from agents.sector_teams.quant_analyst import _render_focus_list_for_prompt
        out = _render_focus_list_for_prompt([{
            "ticker": "X", "sector": "Technology", "stance": "momentum",
            "focus_score": 50.0,
            "momentum_score": 70.0, "quality_score": None,
            "value_score": None, "low_vol_score": 30.0,
        }])
        assert "?" in out


# ── @tool get_factor_profile agent_override telemetry ───────────────────────


class TestAgentOverrideTelemetry:
    @pytest.fixture
    def regime_weights(self):
        return {
            "bull": {
                "momentum_score": 0.40, "quality_score": 0.30,
                "value_score": 0.20, "low_vol_score": -0.10,
            },
        }

    @pytest.fixture
    def profiles(self):
        return {
            "INFOCUS": {
                "sector": "Technology", "quality_score": 70.0,
                "momentum_score": 80.0, "value_score": 30.0, "low_vol_score": 40.0,
            },
            "OUTFOCUS": {
                "sector": "Technology", "quality_score": 50.0,
                "momentum_score": 30.0, "value_score": 25.0, "low_vol_score": 60.0,
            },
        }

    def _tools(self, profiles, regime_weights, focus_list_tickers, override_tickers):
        from agents.sector_teams.quant_tools import create_quant_tools
        price_data = {t: pd.DataFrame() for t in profiles}
        return create_quant_tools({
            "price_data": price_data,
            "technical_scores": {},
            "factor_profiles": profiles,
            "market_regime": "bull",
            "factor_blend_regime_weights": regime_weights,
            "focus_list_tickers": focus_list_tickers,
            "override_tickers": override_tickers,
        })

    def _get_tool(self, tools, name):
        return next(t for t in tools if t.name == name)

    def test_in_focus_ticker_not_recorded_as_override(self, profiles, regime_weights):
        override_tickers = []
        tools = self._tools(
            profiles, regime_weights,
            focus_list_tickers={"INFOCUS"}, override_tickers=override_tickers,
        )
        self._get_tool(tools, "get_factor_profile").invoke({"tickers": ["INFOCUS"]})
        assert override_tickers == []

    def test_out_of_focus_ticker_recorded_as_override(self, profiles, regime_weights):
        override_tickers = []
        tools = self._tools(
            profiles, regime_weights,
            focus_list_tickers={"INFOCUS"}, override_tickers=override_tickers,
        )
        self._get_tool(tools, "get_factor_profile").invoke({"tickers": ["OUTFOCUS"]})
        assert override_tickers == ["OUTFOCUS"]

    def test_empty_focus_list_skips_override_tagging(self, profiles, regime_weights):
        """No focus list this cycle → no override tagging (nothing to override)."""
        override_tickers = []
        tools = self._tools(
            profiles, regime_weights,
            focus_list_tickers=set(), override_tickers=override_tickers,
        )
        self._get_tool(tools, "get_factor_profile").invoke({
            "tickers": ["INFOCUS", "OUTFOCUS"]
        })
        assert override_tickers == []

    def test_repeated_lookups_all_recorded(self, profiles, regime_weights):
        """List, not set — repeated lookups of the same non-focus ticker are
        all appended. Useful signal about how often the agent revisits."""
        override_tickers = []
        tools = self._tools(
            profiles, regime_weights,
            focus_list_tickers={"INFOCUS"}, override_tickers=override_tickers,
        )
        tool = self._get_tool(tools, "get_factor_profile")
        tool.invoke({"tickers": ["OUTFOCUS"]})
        tool.invoke({"tickers": ["OUTFOCUS"]})
        assert override_tickers.count("OUTFOCUS") == 2


# ── _compute_focus_list_audit_lookup PR 4 path ──────────────────────────────


class TestAuditLookupPR4Path:
    @pytest.fixture
    def focus_list_state(self):
        return {
            "technology": [
                {
                    "ticker": "NVDA", "sector": "Technology", "team_id": "technology",
                    "focus_score": 85.0, "stance": "momentum",
                    "rank_in_sector": 1, "rank_in_team": 1,
                    "quality_score": 70.0, "momentum_score": 95.0,
                    "value_score": 20.0, "low_vol_score": 25.0,
                    "factor_blend_breakdown": {},
                },
                {
                    "ticker": "MSFT", "sector": "Technology", "team_id": "technology",
                    "focus_score": 72.0, "stance": "quality",
                    "rank_in_sector": 2, "rank_in_team": 2,
                    "quality_score": 90.0, "momentum_score": 60.0,
                    "value_score": 40.0, "low_vol_score": 55.0,
                    "factor_blend_breakdown": {},
                },
            ],
        }

    def test_pure_projection_no_s3_read(self, focus_list_state):
        """PR 4 path: when focus_list_by_team is provided, no S3 read."""
        from graph import research_graph as rg
        with patch.object(
            rg, "read_factor_profiles_from_s3",
            side_effect=AssertionError("should not be called in PR 4 path"),
        ):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull",
                sector_map={"NVDA": "Technology", "MSFT": "Technology"},
                focus_list_by_team=focus_list_state,
                override_tickers_by_team={},
            )
        assert "NVDA" in result
        assert result["NVDA"]["focus_list_passed"] == 1
        assert result["NVDA"]["agent_override"] == 0
        assert result["NVDA"]["focus_score"] == 85.0
        assert result["NVDA"]["focus_team_id"] == "technology"
        assert result["NVDA"]["focus_rank_in_team"] == 1

    def test_override_ticker_outside_focus_list_flagged(self, focus_list_state):
        from graph import research_graph as rg
        result = rg._compute_focus_list_audit_lookup(
            market_regime="bull",
            sector_map={"NVDA": "Technology", "MSFT": "Technology"},
            focus_list_by_team=focus_list_state,
            override_tickers_by_team={"technology": ["TSLA", "AMD"]},
        )
        # Focus list members untouched
        assert result["NVDA"]["agent_override"] == 0
        # Non-focus tickers surfaced with agent_override=1 + null focus fields
        assert "TSLA" in result and result["TSLA"]["agent_override"] == 1
        assert result["TSLA"]["focus_list_passed"] == 0
        assert result["TSLA"]["focus_score"] is None
        assert "AMD" in result and result["AMD"]["agent_override"] == 1

    def test_override_ticker_in_focus_list_not_double_flagged(self, focus_list_state):
        """Edge case: team B looks up a ticker in team A's focus list — it
        stays as a focus list member (focus_list takes precedence)."""
        from graph import research_graph as rg
        result = rg._compute_focus_list_audit_lookup(
            market_regime="bull",
            sector_map={"NVDA": "Technology"},
            focus_list_by_team=focus_list_state,
            override_tickers_by_team={"technology": ["NVDA"]},
        )
        assert result["NVDA"]["focus_list_passed"] == 1
        assert result["NVDA"]["agent_override"] == 0

    def test_empty_focus_list_returns_empty(self):
        from graph import research_graph as rg
        result = rg._compute_focus_list_audit_lookup(
            market_regime="bull", sector_map={},
            focus_list_by_team={},
            override_tickers_by_team={},
        )
        assert result == {}

    # ── config#750: per-team override attribution ────────────────────────

    def test_override_team_id_attributes_override_to_its_team(
        self, focus_list_state
    ):
        """config#750: each override ticker carries override_team_id naming the
        team whose quant agent reached outside its focus list — not a NULL
        anonymous group."""
        from graph import research_graph as rg
        result = rg._compute_focus_list_audit_lookup(
            market_regime="bull",
            sector_map={"NVDA": "Technology", "MSFT": "Technology"},
            focus_list_by_team=focus_list_state,
            override_tickers_by_team={
                "technology": ["TSLA"],
                "energy": ["XOM"],
            },
        )
        assert result["TSLA"]["agent_override"] == 1
        assert result["TSLA"]["override_team_id"] == "technology"
        assert result["XOM"]["agent_override"] == 1
        assert result["XOM"]["override_team_id"] == "energy"

    def test_override_team_id_none_for_focus_members(self, focus_list_state):
        """Focus-list members are not overrides → override_team_id is None."""
        from graph import research_graph as rg
        result = rg._compute_focus_list_audit_lookup(
            market_regime="bull",
            sector_map={"NVDA": "Technology", "MSFT": "Technology"},
            focus_list_by_team=focus_list_state,
            override_tickers_by_team={"technology": ["TSLA"]},
        )
        assert result["NVDA"]["override_team_id"] is None
        assert result["MSFT"]["override_team_id"] is None

    def test_override_attribution_deterministic_across_teams(
        self, focus_list_state
    ):
        """Defensive: if the same ticker appears in two teams' override sets
        (structurally impossible — sectors partition tickers — but guarded),
        attribution is deterministic (sorted-first team wins), never
        dict-order-dependent."""
        from graph import research_graph as rg
        result = rg._compute_focus_list_audit_lookup(
            market_regime="bull",
            sector_map={"NVDA": "Technology", "MSFT": "Technology"},
            focus_list_by_team=focus_list_state,
            override_tickers_by_team={
                "materials": ["DUP"],
                "energy": ["DUP"],
            },
        )
        # sorted(["materials", "energy"]) == ["energy", "materials"] → energy wins
        assert result["DUP"]["override_team_id"] == "energy"

    def test_all_lookup_entries_carry_override_team_id_key(
        self, focus_list_state
    ):
        """Every projected row exposes override_team_id so the writer's
        e.get('override_team_id') never silently drops attribution."""
        from graph import research_graph as rg
        result = rg._compute_focus_list_audit_lookup(
            market_regime="bull",
            sector_map={"NVDA": "Technology"},
            focus_list_by_team=focus_list_state,
            override_tickers_by_team={"technology": ["TSLA"]},
        )
        for entry in result.values():
            assert "override_team_id" in entry

    def test_legacy_fallback_when_state_absent(self):
        """When focus_list_by_team is None, fall back to recompute path —
        but only if FACTOR_BLEND_ENABLED. Otherwise return empty."""
        from graph import research_graph as rg
        with patch.object(rg, "FACTOR_BLEND_ENABLED", False):
            result = rg._compute_focus_list_audit_lookup(
                market_regime="bull", sector_map={},
                focus_list_by_team=None,
                override_tickers_by_team=None,
            )
        assert result == {}


# ── Config loader ────────────────────────────────────────────────────────────


class TestConfigLoader:
    def test_focus_list_gating_default_off(self):
        from config import FOCUS_LIST_GATING_ENABLED
        assert isinstance(FOCUS_LIST_GATING_ENABLED, bool)

    def test_focus_list_default_team_size_constant(self):
        from config import FOCUS_LIST_DEFAULT_TEAM_SIZE
        assert isinstance(FOCUS_LIST_DEFAULT_TEAM_SIZE, int)
        assert 5 <= FOCUS_LIST_DEFAULT_TEAM_SIZE <= 20

    def test_focus_list_per_team_overrides_is_dict(self):
        from config import FOCUS_LIST_PER_TEAM_SIZE_OVERRIDES
        assert isinstance(FOCUS_LIST_PER_TEAM_SIZE_OVERRIDES, dict)


# ── SectorTeamContext threading ─────────────────────────────────────────────


class TestSectorTeamContextThreading:
    def test_focus_list_default_empty_list(self):
        from agents.sector_teams.sector_team import SectorTeamContext
        ctx = SectorTeamContext(
            scanner_universe=[], agent_input_set=[], sector_map={}, price_data={},
            technical_scores={}, market_regime="neutral", prior_theses={},
            held_tickers=[], news_data_by_ticker={}, analyst_data_by_ticker={},
            insider_data_by_ticker={}, prior_sector_ratings={},
            current_sector_ratings={}, run_date="2026-05-17",
        )
        assert ctx.focus_list == []
        assert ctx.override_tickers == []

    def test_focus_list_accepted_as_param(self):
        from agents.sector_teams.sector_team import SectorTeamContext
        sample = [{"ticker": "NVDA", "focus_score": 85.0, "stance": "momentum"}]
        ctx = SectorTeamContext(
            scanner_universe=[], agent_input_set=[], sector_map={}, price_data={},
            technical_scores={}, market_regime="bull", prior_theses={},
            held_tickers=[], news_data_by_ticker={}, analyst_data_by_ticker={},
            insider_data_by_ticker={}, prior_sector_ratings={},
            current_sector_ratings={}, run_date="2026-05-17",
            focus_list=sample,
        )
        assert ctx.focus_list == sample

    def test_override_tickers_shared_by_reference(self):
        """Shared mutable list — the tool appends from inside the ReAct loop."""
        from agents.sector_teams.sector_team import SectorTeamContext
        shared = []
        ctx = SectorTeamContext(
            scanner_universe=[], agent_input_set=[], sector_map={}, price_data={},
            technical_scores={}, market_regime="bull", prior_theses={},
            held_tickers=[], news_data_by_ticker={}, analyst_data_by_ticker={},
            insider_data_by_ticker={}, prior_sector_ratings={},
            current_sector_ratings={}, run_date="2026-05-17",
            override_tickers=shared,
        )
        ctx.override_tickers.append("FOO")
        assert shared == ["FOO"]

    def test_regime_intensity_z_field_coexists(self):
        """Stage D' Wire 1's regime_intensity_z and PR 4's focus_list /
        override_tickers all coexist on SectorTeamContext."""
        from agents.sector_teams.sector_team import SectorTeamContext
        ctx = SectorTeamContext(
            scanner_universe=[], agent_input_set=[], sector_map={}, price_data={},
            technical_scores={}, market_regime="bear", prior_theses={},
            held_tickers=[], news_data_by_ticker={}, analyst_data_by_ticker={},
            insider_data_by_ticker={}, prior_sector_ratings={},
            current_sector_ratings={}, run_date="2026-05-17",
            regime_intensity_z=-1.5,
            focus_list=[{"ticker": "X", "focus_score": 50}],
            override_tickers=["Y"],
        )
        assert ctx.regime_intensity_z == -1.5
        assert ctx.focus_list == [{"ticker": "X", "focus_score": 50}]
        assert ctx.override_tickers == ["Y"]
