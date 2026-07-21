"""Regression coverage for config#1037: the per-sub-agent cost-tracker
wiring inside ``run_sector_team``.

Before #1037, ``sector_team_node`` opened ONE ``track_llm_cost`` frame
keyed ``sector_team:{team_id}`` spanning the whole team, while the paired
``_capture_if_enabled`` calls used the split agent_ids
(``sector_quant:{team_id}`` / ``sector_qual:{team_id}`` /
``sector_peer_review:{team_id}`` / ``thesis_update:{team_id}:{ticker}``).
``pop_metadata_for`` therefore missed for every sub-agent and the captures
fell back to the placeholder stub (0 tokens, $0 cost, placeholder prompt),
which made replay-concordance meaningless for those families.

The fix pushes per-sub-agent ``track_llm_cost`` scopes INTO
``run_sector_team`` keyed by exactly those split agent_ids. These tests
simulate each sub-agent's Anthropic calls (via the cost-tracker callback
firing into the active frame) and assert the metadata stash now lands
under the split agent_ids with real token counts + recomputed cost.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sector_team_mod():
    """Force-reload the sector_team module + its sub-agent deps.

    Other suites (e.g. test_dry_run.py's sentinel pattern) can leave
    MagicMocks bound for ``run_quant_analyst_with_retry`` /
    ``run_sector_team`` under some pytest orders; reloading guarantees we
    patch + exercise the REAL ``run_sector_team`` with the per-sub-agent
    track_llm_cost scopes under test. Mirrors test_sector_team_recursion
    _partial.py::fresh_modules."""
    from agents.sector_teams import peer_review, qual_analyst, quant_analyst, sector_team
    importlib.reload(quant_analyst)
    importlib.reload(qual_analyst)
    importlib.reload(peer_review)
    importlib.reload(sector_team)
    return sector_team


# ── Pricing + tracker-state fixtures (self-contained; mirrors
#    test_llm_cost_tracker.py so this file can run in isolation) ────────────


@pytest.fixture
def fake_price_table_yaml(tmp_path: Path) -> Path:
    yaml_path = tmp_path / "model_pricing.yaml"
    yaml_path.write_text(
        "cards:\n"
        "  - model_name: claude-haiku-4-5\n"
        "    effective_from: 2026-01-01\n"
        "    input_per_1m: 1.0\n"
        "    output_per_1m: 5.0\n"
        "    cache_read_per_1m: 0.1\n"
        "    cache_create_per_1m: 1.25\n"
    )
    return yaml_path


@pytest.fixture(autouse=True)
def reset_tracker_state():
    from graph import llm_cost_tracker

    llm_cost_tracker._reset_price_table_for_tests()
    llm_cost_tracker._frame_stack.set([])
    llm_cost_tracker._completed_metadata.set({})
    llm_cost_tracker._pending_sft_inputs.set({})
    yield
    llm_cost_tracker._reset_price_table_for_tests()


@pytest.fixture
def patched_pricing_path(monkeypatch, fake_price_table_yaml):
    from graph import llm_cost_tracker
    monkeypatch.setattr(
        llm_cost_tracker, "_resolve_pricing_path",
        lambda: fake_price_table_yaml,
    )


def _make_modern_response(*, input_tokens: int, output_tokens: int) -> MagicMock:
    """Minimal modern langchain-anthropic LLMResult shape the
    CostTelemetryCallback can read token counts off."""
    message = MagicMock()
    message.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": 0, "cache_creation": 0},
    }
    message.response_metadata = {"model_name": "claude-haiku-4-5"}
    generation = MagicMock()
    generation.message = message
    result = MagicMock()
    result.generations = [[generation]]
    result.llm_output = None
    return result


def _fire_llm_call(input_tokens: int, output_tokens: int) -> None:
    """Simulate one Anthropic call landing in the active track_llm_cost
    frame — exactly what get_cost_telemetry_callback() does in prod when
    a ChatAnthropic response comes back inside a sub-agent call."""
    from graph.llm_cost_tracker import get_cost_telemetry_callback

    get_cost_telemetry_callback().on_llm_end(
        _make_modern_response(input_tokens=input_tokens, output_tokens=output_tokens)
    )


def _build_ctx(st, team_id: str = "technology", run_id: str = "2026-06-24"):
    SectorTeamContext = st.SectorTeamContext

    # One held ticker that WILL fire a material trigger (sector regime
    # change) so the thesis_update LLM branch runs.
    return SectorTeamContext(
        scanner_universe=["AAPL", "MSFT"],
        agent_input_set=["AAPL", "MSFT", "NVDA"],
        sector_map={"AAPL": "technology", "MSFT": "technology", "NVDA": "technology"},
        price_data={"AAPL": {}, "MSFT": {}, "NVDA": {}},
        technical_scores={},
        market_regime="neutral",
        prior_theses={"NVDA": {"conviction": 50, "final_score": 60.0}},
        held_tickers=["NVDA"],
        news_data_by_ticker={},
        analyst_data_by_ticker={},
        insider_data_by_ticker={},
        prior_sector_ratings={"technology": {"rating": "overweight"}},
        current_sector_ratings={"technology": {"rating": "underweight"}},
        run_date="2026-06-24",
        run_id=run_id,
    )


def _patch_subagents(monkeypatch, st, *, with_thesis: bool = True):
    """Patch the four sub-agent entrypoints so each simulates its Anthropic
    call(s) firing into whatever track_llm_cost frame is active, then
    returns a minimal well-formed output. Token counts are distinct per
    sub-agent so the assertions can prove attribution (no cross-leak)."""

    def _quant(**kwargs):
        _fire_llm_call(1000, 100)
        return {"team_id": kwargs["team_id"], "ranked_picks": [
            {"ticker": "AAPL", "quant_score": 70},
            {"ticker": "MSFT", "quant_score": 65},
        ], "tool_calls": [], "iterations": 3}

    def _qual(**kwargs):
        _fire_llm_call(2000, 200)
        return {"team_id": kwargs["team_id"],
                "assessments": [{"ticker": "AAPL", "qual_score": 72}],
                "additional_candidate": None, "tool_calls": []}

    def _peer(**kwargs):
        _fire_llm_call(3000, 300)
        return {"recommendations": [{"ticker": "AAPL"}],
                "peer_review_rationale": "ok", "tool_calls": []}

    def _thesis(ticker, triggers, prior_thesis, *a, **k):
        _fire_llm_call(4000, 400)
        return {"ticker": ticker, "triggers": list(triggers),
                "bull_case": "b", "bear_case": "b",
                "conviction": prior_thesis.get("conviction", 50)}

    monkeypatch.setattr(st, "run_quant_analyst_with_retry", _quant)
    monkeypatch.setattr(st, "run_qual_analyst", _qual)
    monkeypatch.setattr(st, "run_peer_review", _peer)
    monkeypatch.setattr(st, "_update_thesis_for_held_stock", _thesis)
    # Force the thesis-update LLM branch (or suppress it) deterministically.
    monkeypatch.setattr(
        st, "check_material_triggers",
        lambda **kw: (["sector_regime_change"] if with_thesis else []),
    )
    # Decouple from the real team_config GICS mapping: return a fixed
    # non-empty ticker slice + map our held NVDA into the team under test
    # so the held-stock loop reaches it.
    monkeypatch.setattr(
        st, "get_team_tickers",
        lambda team_id, agent_input_set, sector_map: ["AAPL", "MSFT"],
    )
    monkeypatch.setattr(
        st, "_sector_team_inverse",
        lambda: {"technology": "technology", "healthcare": "healthcare"},
    )


# ── The fix: per-sub-agent metadata stash lands under split agent_ids ──────


class TestPerSubAgentCostTracking:
    def test_split_agent_ids_carry_real_token_counts(
        self, patched_pricing_path, monkeypatch, sector_team_mod
    ):
        """After run_sector_team, pop_metadata_for(sector_quant:…) etc.
        each return REAL (non-zero) token counts — the config#1037 fix.
        Pre-fix these all missed (only sector_team:{team_id} was stashed)
        and the captures fell back to 0-token placeholders."""
        from graph.llm_cost_tracker import pop_metadata_for

        _patch_subagents(monkeypatch, sector_team_mod, with_thesis=True)
        ctx = _build_ctx(sector_team_mod, team_id="technology")

        sector_team_mod.run_sector_team("technology", ctx)

        quant = pop_metadata_for("sector_quant:technology")
        qual = pop_metadata_for("sector_qual:technology")
        peer = pop_metadata_for("sector_peer_review:technology")
        thesis = pop_metadata_for("thesis_update:technology:NVDA")

        assert quant is not None and quant[0].input_tokens == 1000
        assert quant[0].output_tokens == 100
        assert qual is not None and qual[0].input_tokens == 2000
        assert peer is not None and peer[0].input_tokens == 3000
        assert thesis is not None and thesis[0].input_tokens == 4000
        # Cost recomputed off the test price card, not left at 0.
        assert quant[0].cost_usd > 0
        assert thesis[0].cost_usd > 0

    def test_legacy_combined_agent_id_no_longer_stashed(
        self, patched_pricing_path, monkeypatch, sector_team_mod
    ):
        """The legacy combined sector_team:{team_id} aggregate frame is
        gone — its metadata is no longer stashed (it was the agent_id no
        capture ever read)."""
        from graph.llm_cost_tracker import pop_metadata_for

        _patch_subagents(monkeypatch, sector_team_mod, with_thesis=False)
        ctx = _build_ctx(sector_team_mod, team_id="healthcare")

        sector_team_mod.run_sector_team("healthcare", ctx)

        assert pop_metadata_for("sector_team:healthcare") is None
        # And the split ids that DID run are present.
        assert pop_metadata_for("sector_quant:healthcare") is not None

    def test_no_token_cross_leak_between_subagents(
        self, patched_pricing_path, monkeypatch, sector_team_mod
    ):
        """Each sub-agent's frame accumulates ONLY its own call (top of the
        frame stack), so tokens don't leak across the four scopes."""
        from graph.llm_cost_tracker import pop_metadata_for

        _patch_subagents(monkeypatch, sector_team_mod, with_thesis=True)
        ctx = _build_ctx(sector_team_mod, team_id="technology")

        sector_team_mod.run_sector_team("technology", ctx)

        # quant = 1000/100 exactly — not 1000+2000+3000+4000.
        assert pop_metadata_for("sector_quant:technology")[0].input_tokens == 1000
        assert pop_metadata_for("sector_qual:technology")[0].input_tokens == 2000

    def test_no_trigger_held_stock_opens_no_thesis_scope(
        self, patched_pricing_path, monkeypatch, sector_team_mod
    ):
        """The no-material-trigger preservation branch fires no LLM call —
        no thesis_update scope is opened, so no placeholder pollutes the
        cost stream (mirrors the capture which also skips it)."""
        from graph.llm_cost_tracker import pop_metadata_for

        _patch_subagents(monkeypatch, sector_team_mod, with_thesis=False)
        ctx = _build_ctx(sector_team_mod, team_id="technology")

        sector_team_mod.run_sector_team("technology", ctx)

        assert pop_metadata_for("thesis_update:technology:NVDA") is None
