"""Regression tests for the 2026-04-04 unscored-BUY leak.

Pins two related fixes that prevent held-stock LLM thesis updates from
emitting ENTER signals with score=null:

1. ``_update_thesis_for_held_stock`` must merge the LLM JSON over
   ``prior_thesis`` so scoring fields (final_score, quant_score,
   qual_score, rating, sector, team_id) are preserved.
2. ``_build_signals_payload`` must downgrade BUY → HOLD when
   ``final_score`` is None, as a last line of defense if a broken
   thesis slips past step 1.

Root cause: the held-stock thesis update prompt only revises narrative
fields, but the old code replaced the entire thesis dict with the
LLM's partial JSON, stripping scores. Downstream code then emitted
BUY-rated ENTER signals with null scores (LNTH, KR, PR, HAL in
signals/2026-04-04/signals.json).
"""

from unittest.mock import MagicMock, patch


def _fake_llm_factory(response_content: str) -> MagicMock:
    """Build a MagicMock that mimics ChatAnthropic's invoke() response.

    Legacy factory — used by tests against agents that haven't migrated
    to ``with_structured_output``. New tests should use
    ``_fake_structured_llm_factory`` below.
    """
    fake_response = MagicMock()
    fake_response.content = response_content
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_response
    return fake_llm


def _fake_structured_llm_factory(parsed_model) -> MagicMock:
    """Build a MagicMock that mimics ``llm.with_structured_output(Schema, include_raw=True).invoke()``.

    The chain is: ``llm.with_structured_output(Schema, include_raw=True)``
    returns a wrapped LLM whose ``.invoke(...)`` returns a dict with
    ``parsed`` / ``raw`` / ``parsing_error`` keys (the include_raw
    contract used across quant_analyst.py / ic_cio.py and, since the
    2026-05-16 per-thesis-isolation fix, ``_update_thesis_for_held_stock``).
    """
    fake_llm = MagicMock()
    structured = MagicMock()
    fake_llm.with_structured_output.return_value = structured
    structured.invoke.return_value = {
        "raw": MagicMock(content="..."),
        "parsed": parsed_model,
        "parsing_error": None,
    }
    return fake_llm


def test_held_stock_thesis_update_preserves_prior_scores():
    """LLM thesis updates for held stocks must merge over prior_thesis.

    PR 2.3 Step E refactored ``_update_thesis_for_held_stock`` to use
    ``with_structured_output(HeldThesisUpdateLLMOutput)``. The new
    schema has NO score fields by design — the LLM cannot emit
    final_score / quant_score / qual_score / rating, which makes
    overwrite-with-null structurally impossible (the previous
    strip-nulls workaround is retired). This test pins the new
    invariant: prior scores survive the merge.
    """
    from agents.sector_teams import sector_team
    from graph.state_schemas import HeldThesisUpdateLLMOutput

    prior = {
        "ticker": "LNTH",
        "sector": "Healthcare",
        "team_id": "healthcare",
        "final_score": 45.0,
        "quant_score": 50.0,
        "qual_score": 40.0,
        "rating": "HOLD",
        "conviction": "stable",
        "bull_case": "old bull",
        "bear_case": "old bear",
    }

    fake_llm = _fake_structured_llm_factory(
        HeldThesisUpdateLLMOutput(
            bull_case="new bull narrative",
            bear_case="new bear narrative",
            conviction=25,  # agent format: int 0-100 (Option A)
        )
    )

    with patch.object(sector_team, "ChatAnthropic", return_value=fake_llm), \
         patch.object(sector_team, "load_prompt") as mock_load, \
         patch.object(sector_team, "format_structured_thesis_for_prompt", return_value=""):
        mock_load.return_value.format.return_value = "prompt"
        result = sector_team._update_thesis_for_held_stock(
            ticker="LNTH",
            triggers=[],
            prior_thesis=prior,
            news_data=None,
            analyst_data=None,
            run_date="2026-04-04",
            team_id="healthcare",
            api_key="test-key",
        )

    # Scores must be preserved from prior — schema makes overwrite impossible
    assert result["final_score"] == 45.0
    assert result["quant_score"] == 50.0
    assert result["qual_score"] == 40.0
    assert result["rating"] == "HOLD"
    assert result["sector"] == "Healthcare"
    assert result["team_id"] == "healthcare"

    # Narrative fields must be updated from LLM
    assert result["bull_case"] == "new bull narrative"
    assert result["bear_case"] == "new bear narrative"
    assert result["conviction"] == 25

    # Run metadata must be set
    assert result["last_updated"] == "2026-04-04"
    assert result["stale_days"] == 0


def test_held_stock_llm_schema_has_no_score_fields():
    """Structural guarantee: HeldThesisUpdateLLMOutput cannot contain
    score fields by design.

    Replaces the pre-2026-04-30 ``test_held_stock_thesis_update_strips_null_llm_fields``
    test. The original test verified that the `_update_thesis_for_held_stock`
    code stripped `final_score: null` (and similar) from the LLM's JSON
    output before merging into prior_thesis. PR 2.3 Step E migrated to
    ``with_structured_output(HeldThesisUpdateLLMOutput)``: the schema
    intentionally omits all score fields, so the LLM cannot emit them
    AT ALL. The strip-nulls workaround is structurally retired —
    confirmed by enumerating the schema's fields.

    This test pins that structural invariant. If anyone ever adds
    ``final_score`` (or sibling) to ``HeldThesisUpdateLLMOutput``,
    they'll fail this test before re-introducing the 2026-04-04
    leak class.
    """
    from graph.state_schemas import HeldThesisUpdateLLMOutput

    fields = HeldThesisUpdateLLMOutput.model_fields.keys()
    forbidden = {"final_score", "quant_score", "qual_score", "rating"}
    leaked = forbidden & set(fields)
    assert not leaked, (
        f"HeldThesisUpdateLLMOutput must NOT have score fields (regression "
        f"of 2026-04-04 LNTH/LLY/PFE/etc. leak). Found: {leaked}"
    )


def test_held_stock_thesis_update_no_prior_marks_score_failed():
    """If the LLM succeeds but there's no prior thesis, mark score_failed."""
    from agents.sector_teams import sector_team
    from graph.state_schemas import HeldThesisUpdateLLMOutput

    fake_llm = _fake_structured_llm_factory(
        HeldThesisUpdateLLMOutput(bull_case="bull", bear_case="bear")
    )

    with patch.object(sector_team, "ChatAnthropic", return_value=fake_llm), \
         patch.object(sector_team, "load_prompt") as mock_load, \
         patch.object(sector_team, "format_structured_thesis_for_prompt", return_value=""):
        mock_load.return_value.format.return_value = "prompt"
        result = sector_team._update_thesis_for_held_stock(
            ticker="NEW",
            triggers=[],
            prior_thesis=None,
            news_data=None,
            analyst_data=None,
            run_date="2026-04-04",
            team_id="healthcare",
            api_key="test-key",
        )

    assert result.get("score_failed") is True
    assert "final_score" not in result


def test_build_signals_downgrades_buy_when_final_score_none():
    """If a thesis has rating=BUY but final_score=None, emit HOLD not ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "LNTH": {
                "ticker": "LNTH",
                "rating": "BUY",
                "final_score": None,  # ← broken thesis
                "quant_score": None,
                "qual_score": None,
                "sector": "Healthcare",
                "team_id": "healthcare",
                "conviction": "declining",
                "bull_case": "",
            }
        },
        "prior_theses": {},
        "new_population": [{"ticker": "LNTH"}],  # held
        "sector_map": {"LNTH": "Healthcare"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    # Payload has multiple keys; the modern signals dict is keyed by ticker
    signals = payload.get("signals", {})
    lnth = signals.get("LNTH")
    assert lnth is not None, f"LNTH missing from signals: {signals}"
    # Must be HOLD, not ENTER — the safety gate downgrades unscored BUYs
    assert lnth["signal"] == "HOLD", (
        f"Unscored BUY should have been downgraded to HOLD, got: {lnth}"
    )


def test_build_signals_allows_scored_buy_through():
    """A BUY with a real final_score must still be emitted as ENTER."""
    from graph.research_graph import _build_signals_payload

    state = {
        "investment_theses": {
            "NVDA": {
                "ticker": "NVDA",
                "rating": "BUY",
                "final_score": 82.5,  # ← valid
                "quant_score": 85.0,
                "qual_score": 80.0,
                "sector": "Technology",
                "team_id": "technology",
                "conviction": "rising",
                "bull_case": "AI demand",
            }
        },
        "prior_theses": {},
        "new_population": [{"ticker": "NVDA"}],  # held
        "sector_map": {"NVDA": "Technology"},
        "sector_ratings": {},
        "entry_theses": {},
        "advanced_tickers": [],
    }

    payload = _build_signals_payload(state)
    signals = payload.get("signals", {})
    nvda = signals.get("NVDA")
    assert nvda is not None
    assert nvda["signal"] == "ENTER"
    assert nvda["score"] == 82.5
