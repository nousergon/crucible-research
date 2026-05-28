"""
Round-trip + validation tests for ``graph.state_schemas``.

These models are not yet wired into ``ResearchState`` — this commit ships them
standalone so they can be reviewed + tested against fixtures of real agent
outputs before the integration commit lands. The ``extra="allow"`` posture
on every model means construction from a real agent dict (which carries
fields not enumerated here) does NOT reject; PR 2 flips ``extra="forbid"``.

Workstream: typed LangGraph state + Pydantic agent outputs + decision-artifact
capture (alpha-engine-research-typed-state-capture-260429.md).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graph.state_schemas import (
    CIODecision,
    CIOOutput,
    ExitEvaluatorOutput,
    ExitEvent,
    InvestmentThesis,
    MacroEconomistOutput,
    PopulationRotationEvent,
    SectorRecommendation,
    SectorTeamOutput,
    ThesisUpdate,
    ToolCall,
)


# ── ToolCall ──────────────────────────────────────────────────────────────


class TestToolCall:
    def test_minimal_construction(self):
        tc = ToolCall(tool="quant_indicators")
        assert tc.tool == "quant_indicators"
        assert tc.ticker is None
        assert tc.args == {}
        assert tc.result_summary is None

    def test_full_construction(self):
        tc = ToolCall(
            tool="qual_news_search",
            ticker="AAPL",
            args={"hours": 48},
            result_summary="3 articles retrieved",
        )
        assert tc.ticker == "AAPL"
        assert tc.args == {"hours": 48}

    def test_extra_fields_allowed(self):
        tc = ToolCall(tool="x", undocumented_field="value")
        # extra fields preserved on dump
        assert tc.model_dump()["undocumented_field"] == "value"

    def test_tool_optional_for_peer_review_entries(self):
        # 2026-04-30: peer_review appends a phase-tracking entry to
        # tool_calls without a `tool` name (it's an orchestration step,
        # not a tool invocation). Schema must accept tool=None.
        tc = ToolCall(args={"phase": "peer_review"})
        assert tc.tool is None
        assert tc.args["phase"] == "peer_review"


# ── SectorRecommendation ──────────────────────────────────────────────────


class TestSectorRecommendation:
    def test_minimal_valid(self):
        r = SectorRecommendation(ticker="AAPL", quant_score=70.0, qual_score=65.0)
        # Option A 2026-04-30: agent-format conviction is int 0-100 with no
        # default — None means "agent did not emit one".
        assert r.conviction is None
        assert r.bull_case == ""

    def test_score_clamping(self):
        with pytest.raises(ValueError):
            SectorRecommendation(ticker="AAPL", quant_score=120, qual_score=50)
        with pytest.raises(ValueError):
            SectorRecommendation(ticker="AAPL", quant_score=50, qual_score=-1)

    def test_qual_score_optional(self):
        # 2026-04-30: peer_review can produce a recommendation when the
        # qual analyst returned 0 assessments (qual_score legitimately
        # absent). Schema must accept qual_score=None.
        r = SectorRecommendation(ticker="AAPL", quant_score=70.0, qual_score=None)
        assert r.qual_score is None
        # And must still clamp when a value IS provided.
        with pytest.raises(ValueError):
            SectorRecommendation(ticker="AAPL", quant_score=70.0, qual_score=120)

    def test_conviction_int_range_enforced(self):
        # Option A 2026-04-30: agent-format conviction is int 0-100; bounds
        # enforced. Strings (formerly high/medium/low) are no longer valid.
        with pytest.raises(ValueError):
            SectorRecommendation(
                ticker="AAPL", quant_score=70, qual_score=70, conviction=120
            )
        with pytest.raises(ValueError):
            SectorRecommendation(
                ticker="AAPL", quant_score=70, qual_score=70, conviction=-1
            )

    def test_conviction_string_rejected(self):
        # Post-Option-A: legacy agent-format strings are NOT accepted at the
        # SectorRecommendation boundary. Producers must emit int.
        with pytest.raises(ValueError):
            SectorRecommendation(
                ticker="AAPL", quant_score=70, qual_score=70, conviction="high"
            )

    def test_round_trip(self):
        original = SectorRecommendation(
            ticker="NVDA",
            quant_score=85.0,
            qual_score=78.0,
            bull_case="AI tailwind",
            bear_case="Valuation",
            catalysts=["Earnings", "GTC"],
            conviction=85,
        )
        roundtripped = SectorRecommendation.model_validate(original.model_dump())
        assert roundtripped == original


# ── ThesisUpdate ──────────────────────────────────────────────────────────


class TestThesisUpdate:
    def test_all_scores_none_allowed(self):
        # Permitted by design — score_aggregator's recompute path or
        # hard-fail handles the missing-score case.
        t = ThesisUpdate(ticker="CME")
        assert t.final_score is None
        assert t.quant_score is None
        assert t.qual_score is None

    def test_partial_scores_round_trip(self):
        t = ThesisUpdate(ticker="HSY", quant_score=60.0, qual_score=55.0)
        d = t.model_dump()
        assert d["final_score"] is None
        assert d["quant_score"] == 60.0
        assert d["qual_score"] == 55.0

    def test_score_range_enforced(self):
        with pytest.raises(ValueError):
            ThesisUpdate(ticker="AAPL", final_score=110)


# ── SectorTeamOutput ──────────────────────────────────────────────────────


class TestSectorTeamOutput:
    def test_empty_team(self):
        sto = SectorTeamOutput(team_id="technology")
        assert sto.recommendations == []
        assert sto.thesis_updates == {}
        assert sto.tool_calls == []
        assert sto.error is None

    def test_with_recommendations_and_updates(self):
        sto = SectorTeamOutput(
            team_id="financials",
            recommendations=[
                SectorRecommendation(ticker="JPM", quant_score=70, qual_score=65),
                SectorRecommendation(ticker="V", quant_score=72, qual_score=70),
            ],
            thesis_updates={
                "MA": ThesisUpdate(
                    ticker="MA", final_score=68.0, quant_score=70.0, qual_score=66.0
                )
            },
            tool_calls=[ToolCall(tool="quant_indicators", ticker="JPM")],
        )
        assert len(sto.recommendations) == 2
        assert "MA" in sto.thesis_updates

    def test_extra_stub_fields_preserved(self):
        # The offline stub returns quant_output / qual_output / peer_review_output
        # as extra fields. They must round-trip through the model without rejection.
        sto = SectorTeamOutput(
            team_id="technology",
            recommendations=[],
            thesis_updates={},
            tool_calls=[],
            quant_output={"ranked_picks": []},
            qual_output={"assessments": []},
            peer_review_output={"recommendations": []},
        )
        d = sto.model_dump()
        assert "quant_output" in d
        assert "qual_output" in d
        assert "peer_review_output" in d

    def test_construction_from_dict_payload(self):
        # Mirrors the shape the offline stub at local/offline_stubs.py:297-305
        # actually returns. Must validate cleanly.
        payload = {
            "team_id": "technology",
            "recommendations": [
                {"ticker": "AAPL", "quant_score": 65.0, "qual_score": 70.0,
                 "bull_case": "x", "bear_case": "y", "catalysts": [],
                 "conviction": 60, "team_id": "technology"},
            ],
            "thesis_updates": {},
            "quant_output": {},
            "qual_output": {},
            "peer_review_output": {},
            "tool_calls": [],
        }
        sto = SectorTeamOutput.model_validate(payload)
        assert sto.recommendations[0].ticker == "AAPL"


# ── MacroEconomistOutput ──────────────────────────────────────────────────


class TestMacroEconomistOutput:
    def test_minimal(self):
        m = MacroEconomistOutput()
        assert m.macro_report == ""
        assert m.market_regime == "neutral"
        assert m.sector_modifiers == {}

    def test_modifier_clamp_in_range(self):
        m = MacroEconomistOutput(
            sector_modifiers={"Technology": 1.20, "Energy": 0.85, "Healthcare": 1.0}
        )
        assert m.sector_modifiers["Technology"] == 1.20

    def test_modifier_clamp_above_range(self):
        with pytest.raises(ValueError, match=r"sector_modifiers"):
            MacroEconomistOutput(sector_modifiers={"Technology": 1.50})

    def test_modifier_clamp_below_range(self):
        with pytest.raises(ValueError, match=r"sector_modifiers"):
            MacroEconomistOutput(sector_modifiers={"Energy": 0.50})

    def test_modifier_boundaries_inclusive(self):
        # 0.70 and 1.30 are explicitly the inclusive boundaries
        m = MacroEconomistOutput(
            sector_modifiers={"Low": 0.70, "High": 1.30, "Mid": 1.0}
        )
        assert m.sector_modifiers["Low"] == 0.70
        assert m.sector_modifiers["High"] == 1.30

    def test_regime_literal_enforced(self):
        with pytest.raises(ValueError):
            MacroEconomistOutput(market_regime="euphoric")

    def test_regime_all_valid_values(self):
        # 3-class Ang-Bekaert taxonomy (v0.42.0 / 2026-05-28). Legacy
        # "caution" retired per caution-regime-retirement-260528.md.
        for regime in ("bull", "neutral", "bear"):
            m = MacroEconomistOutput(market_regime=regime)
            assert m.market_regime == regime

    def test_regime_legacy_caution_rejected(self):
        # "caution" was retired in v0.42.0. Raw LLM emissions are coerced
        # to "neutral" by macro_agent._validate_regime upstream of this
        # schema; the schema itself enforces the 3-class invariant.
        with pytest.raises(ValueError):
            MacroEconomistOutput(market_regime="caution")


# ── ExitEvent + ExitEvaluatorOutput ───────────────────────────────────────


class TestExitEvent:
    def test_minimal(self):
        e = ExitEvent(ticker_out="MA")
        assert e.reason == ""
        assert e.score_out == 0.0

    def test_with_reason_and_score(self):
        e = ExitEvent(ticker_out="MA", reason="min_rotation_floor", score_out=70.0)
        assert e.score_out == 70.0


class TestExitEvaluatorOutput:
    def test_minimal(self):
        eo = ExitEvaluatorOutput()
        assert eo.exits == []
        assert eo.open_slots == 0

    def test_open_slots_non_negative(self):
        with pytest.raises(ValueError):
            ExitEvaluatorOutput(open_slots=-1)


# ── CIODecision + CIOOutput ───────────────────────────────────────────────


class TestCIODecision:
    def test_minimal(self):
        d = CIODecision(ticker="JPM")
        assert d.thesis_type is None

    def test_thesis_type_literal(self):
        with pytest.raises(ValueError):
            CIODecision(ticker="JPM", thesis_type="MAYBE")

    def test_full(self):
        d = CIODecision(
            ticker="JPM",
            thesis_type="ADVANCE",
            rationale="Strong",
            conviction=78,
            score=78.0,
        )
        assert d.thesis_type == "ADVANCE"
        assert d.conviction == 78

    def test_conviction_int_range_enforced(self):
        # Path Y: conviction is a 0-100 score; bounds enforced.
        with pytest.raises(ValueError):
            CIODecision(ticker="JPM", conviction=120)
        with pytest.raises(ValueError):
            CIODecision(ticker="JPM", conviction=-1)

    def test_conviction_optional(self):
        d = CIODecision(ticker="JPM")
        assert d.conviction is None


class TestCIOOutput:
    def test_minimal(self):
        o = CIOOutput()
        assert o.ic_decisions == []
        assert o.advanced_tickers == []
        assert o.entry_theses == {}


# ── InvestmentThesis ──────────────────────────────────────────────────────


class TestInvestmentThesis:
    def test_minimal_required(self):
        t = InvestmentThesis(ticker="AAPL", final_score=70.0, rating="BUY")
        assert t.sector is None
        assert t.team_id == ""
        # Storage format from normalize_conviction (executor-compatible)
        assert t.conviction == "stable"

    def test_full(self):
        t = InvestmentThesis(
            ticker="NVDA",
            sector="Technology",
            team_id="technology",
            final_score=82.0,
            quant_score=80.0,
            qual_score=85.0,
            weighted_base=82.5,
            macro_shift=-0.5,
            bull_case="AI",
            bear_case="Valuation",
            catalysts=["Earnings"],
            conviction="rising",  # storage format
            quant_rationale="...",
            rating="BUY",
            score_failed=False,
        )
        assert t.weighted_base == 82.5

    def test_agent_format_conviction_rejected(self):
        # InvestmentThesis is post-normalize_conviction storage; agent format
        # must NOT be accepted at this boundary (use ThesisUpdate for the
        # union-format variant if needed). Post-Option-A the agent format is
        # int 0-100, but it still must NOT be the value stored here.
        with pytest.raises(ValueError):
            InvestmentThesis(
                ticker="AAPL", final_score=70.0, rating="BUY",
                conviction=72,
            )
        with pytest.raises(ValueError):
            InvestmentThesis(
                ticker="AAPL", final_score=70.0, rating="BUY",
                conviction="high",
            )

    def test_rating_literal(self):
        with pytest.raises(ValueError):
            InvestmentThesis(ticker="AAPL", final_score=70.0, rating="STRONG_BUY")

    def test_round_trip_with_extras(self):
        # Real score_aggregator output may carry a "date" or "team_id"
        # field added by archive_writer's row construction; extras allowed.
        payload = {
            "ticker": "JPM",
            "sector": "Financials",
            "team_id": "financials",
            "final_score": 68.0,
            "quant_score": 65.0,
            "qual_score": 70.0,
            "weighted_base": 67.5,
            "macro_shift": 0.5,
            "bull_case": "...",
            "bear_case": "...",
            "catalysts": [],
            "conviction": "stable",  # storage format from normalize_conviction
            "quant_rationale": "",
            "rating": "BUY",
            "score_failed": False,
            "date": "2026-04-25",  # extra
        }
        t = InvestmentThesis.model_validate(payload)
        assert t.model_dump()["date"] == "2026-04-25"


# ── PopulationRotationEvent ───────────────────────────────────────────────


class TestPopulationRotationEvent:
    def test_minimal(self):
        e = PopulationRotationEvent()
        assert e.event_type is None
        assert e.reason == ""

    def test_entry_event_shape(self):
        e = PopulationRotationEvent(event_type="entry", ticker_in="JPM", reason="advance")
        assert e.event_type == "entry"
        assert e.ticker_in == "JPM"

    def test_exit_event_shape(self):
        e = PopulationRotationEvent(event_type="exit", ticker_out="MA", reason="floor")
        assert e.event_type == "exit"
        assert e.ticker_out == "MA"

    def test_event_type_literal(self):
        with pytest.raises(ValueError):
            PopulationRotationEvent(event_type="rotation")


# ── PR 2 LLM-extraction schemas ──────────────────────────────────────────


class TestMacroEconomistRawOutput:
    def test_minimal(self):
        from graph.state_schemas import MacroEconomistRawOutput
        m = MacroEconomistRawOutput(report_md="...", market_regime="neutral")
        assert m.market_regime == "neutral"
        assert m.sector_modifiers == {}
        assert m.key_theme == ""

    def test_clamp_modifiers_in_range(self):
        from graph.state_schemas import MacroEconomistRawOutput
        m = MacroEconomistRawOutput(
            sector_modifiers={"Technology": 1.10, "Healthcare": 0.95}
        )
        assert m.sector_modifiers["Technology"] == 1.10

    def test_clamp_modifiers_out_of_range_raises(self):
        from graph.state_schemas import MacroEconomistRawOutput
        with pytest.raises(ValueError, match=r"outside \[0.70, 1.30\]"):
            MacroEconomistRawOutput(sector_modifiers={"Technology": 1.50})

    def test_extra_fields_preserved(self):
        from graph.state_schemas import MacroEconomistRawOutput
        m = MacroEconomistRawOutput(
            report_md="x", market_regime="bull", undocumented="kept"
        )
        assert m.model_dump()["undocumented"] == "kept"


class TestMacroCriticOutput:
    def test_accept(self):
        from graph.state_schemas import MacroCriticOutput
        c = MacroCriticOutput(action="accept")
        assert c.action == "accept"
        assert c.suggested_regime is None

    def test_revise_with_suggested_regime(self):
        from graph.state_schemas import MacroCriticOutput
        c = MacroCriticOutput(
            action="revise",
            critique="too aggressive",
            suggested_regime="neutral",
        )
        assert c.suggested_regime == "neutral"

    def test_revise_legacy_caution_rejected(self):
        # 3-class invariant (v0.42.0 — caution-regime-retirement-260528.md).
        from graph.state_schemas import MacroCriticOutput
        with pytest.raises(ValueError):
            MacroCriticOutput(
                action="revise",
                critique="elevated stress",
                suggested_regime="caution",
            )

    def test_action_literal_enforced(self):
        from graph.state_schemas import MacroCriticOutput
        with pytest.raises(ValueError):
            MacroCriticOutput(action="maybe")


class TestQuantAnalystOutput:
    def test_minimal(self):
        from graph.state_schemas import QuantAnalystOutput
        q = QuantAnalystOutput()
        assert q.ranked_picks == []

    def test_picks_score_clamping(self):
        from graph.state_schemas import QuantPick
        with pytest.raises(ValueError):
            QuantPick(ticker="AAPL", quant_score=120)
        with pytest.raises(ValueError):
            QuantPick(ticker="AAPL", quant_score=-1)

    def test_full_construction(self):
        from graph.state_schemas import QuantAnalystOutput, QuantPick
        q = QuantAnalystOutput(
            ranked_picks=[
                QuantPick(ticker="NVDA", quant_score=85, rationale="..."),
                QuantPick(ticker="AAPL", quant_score=72, rationale="..."),
            ]
        )
        assert len(q.ranked_picks) == 2
        assert q.ranked_picks[0].ticker == "NVDA"


class TestQualAnalystOutput:
    def test_minimal(self):
        from graph.state_schemas import QualAnalystOutput
        q = QualAnalystOutput()
        assert q.assessments == []
        assert q.additional_candidate is None

    def test_qual_score_optional(self):
        """Mirror PR #59 fix — qual_score=None must be valid."""
        from graph.state_schemas import QualAssessment
        a = QualAssessment(ticker="AAPL", qual_score=None)
        assert a.qual_score is None

    def test_conviction_int_range(self):
        # Option A 2026-04-30: qual analyst emits int 0-100.
        from graph.state_schemas import QualAssessment
        a = QualAssessment(ticker="AAPL", conviction=72)
        assert a.conviction == 72
        with pytest.raises(ValueError):
            QualAssessment(ticker="AAPL", conviction=120)

    def test_conviction_string_rejected(self):
        from graph.state_schemas import QualAssessment
        with pytest.raises(ValueError):
            QualAssessment(ticker="AAPL", conviction="high")


class TestQuantAcceptanceVerdict:
    def test_accept(self):
        from graph.state_schemas import QuantAcceptanceVerdict
        v = QuantAcceptanceVerdict(accept=True, reason="strong technicals")
        assert v.accept is True

    def test_reject(self):
        from graph.state_schemas import QuantAcceptanceVerdict
        v = QuantAcceptanceVerdict(accept=False)
        assert v.accept is False
        assert v.reason == ""


class TestJointFinalizationOutput:
    def test_minimal(self):
        from graph.state_schemas import JointFinalizationOutput
        j = JointFinalizationOutput()
        assert j.selected_decisions == []
        assert j.team_rationale == ""

    def test_full(self):
        from graph.state_schemas import (
            JointFinalizationDecision, JointFinalizationOutput,
        )
        j = JointFinalizationOutput(
            selected_decisions=[
                JointFinalizationDecision(
                    ticker="AAPL", rationale="Strongest R/R, services catalyst.",
                ),
                JointFinalizationDecision(
                    ticker="MSFT", rationale="Steady core, AI cap-ex thesis.",
                ),
                JointFinalizationDecision(
                    ticker="NVDA", rationale="Asymmetric breakout setup.",
                ),
            ],
            team_rationale="Sector overweight justified, asymmetry mix balanced.",
        )
        assert len(j.selected_decisions) == 3
        assert j.selected_decisions[0].ticker == "AAPL"
        assert "R/R" in j.selected_decisions[0].rationale

    def test_decision_requires_ticker(self):
        from graph.state_schemas import JointFinalizationDecision
        with pytest.raises(ValidationError):
            JointFinalizationDecision()  # ticker is required

    def test_decision_default_rationale_empty(self):
        from graph.state_schemas import JointFinalizationDecision
        d = JointFinalizationDecision(ticker="AAPL")
        assert d.rationale == ""


class TestHeldThesisUpdateLLMOutput:
    def test_minimal(self):
        from graph.state_schemas import HeldThesisUpdateLLMOutput
        h = HeldThesisUpdateLLMOutput()
        assert h.bull_case == ""
        assert h.conviction is None

    def test_no_score_fields_in_schema(self):
        """The schema MUST NOT enumerate score fields — that's the whole
        point. The held-stock LLM update path must not emit
        final_score/quant_score/qual_score, and the schema enforces this
        by simply not having those fields. extra='allow' means an LLM
        that ignores the schema and emits them anyway preserves them, but
        downstream consumers should ignore extras."""
        from graph.state_schemas import HeldThesisUpdateLLMOutput
        fields = HeldThesisUpdateLLMOutput.model_fields.keys()
        assert "final_score" not in fields
        assert "quant_score" not in fields
        assert "qual_score" not in fields

    def test_conviction_int_format(self):
        """Held-stock LLM emits agent format — int 0-100 post-Option-A
        (2026-04-30). Storage format strings are rejected — they belong on
        the InvestmentThesis side, post-normalize_conviction."""
        from graph.state_schemas import HeldThesisUpdateLLMOutput
        h = HeldThesisUpdateLLMOutput(conviction=80)
        assert h.conviction == 80
        with pytest.raises(ValueError):
            HeldThesisUpdateLLMOutput(conviction="rising")
        with pytest.raises(ValueError):
            HeldThesisUpdateLLMOutput(conviction="high")
        with pytest.raises(ValueError):
            HeldThesisUpdateLLMOutput(conviction=120)


class TestCIORawOutput:
    def test_empty_decisions_rejected(self):
        """``min_length=1`` on ``decisions`` (added 2026-05-02 after the
        post-PR-D validation invoke caught Sonnet emitting ``[]``)
        rejects an empty list at the schema layer. The constraint is
        propagated to the LLM via the structured-output tool schema
        description AND validated by the SDK parser. Previously the
        empty-list case only surfaced as a downstream
        ``CIO structured response had empty decisions list`` raise
        inside ``run_cio``; the schema-level rejection moves the
        failure to the call boundary with a clearer Pydantic error."""
        from graph.state_schemas import CIORawOutput
        with pytest.raises(ValueError) as exc_info:
            CIORawOutput()
        # Pydantic surfaces the constraint by name in the error.
        assert "decisions" in str(exc_info.value).lower()

    def test_single_decision_accepted(self):
        from graph.state_schemas import CIORawDecision, CIORawOutput
        c = CIORawOutput(decisions=[
            CIORawDecision(ticker="NVDA", decision="ADVANCE"),
        ])
        assert len(c.decisions) == 1

    def test_decision_literal_includes_no_advance_deadlock(self):
        from graph.state_schemas import CIORawDecision
        # All three valid LLM-emitted values
        for d in ["ADVANCE", "REJECT", "NO_ADVANCE_DEADLOCK"]:
            CIORawDecision(ticker="AAPL", decision=d)

    def test_decision_literal_rejects_advance_forced(self):
        """ADVANCE_FORCED is synthesized by post-processing; LLM must NOT
        emit it directly. Schema rejects it."""
        from graph.state_schemas import CIORawDecision
        with pytest.raises(ValueError):
            CIORawDecision(ticker="AAPL", decision="ADVANCE_FORCED")

    def test_decision_literal_rejects_hold(self):
        """HOLD is what post-processing maps REJECT to for held tickers;
        LLM does not emit it directly."""
        from graph.state_schemas import CIORawDecision
        with pytest.raises(ValueError):
            CIORawDecision(ticker="AAPL", decision="HOLD")

    def test_conviction_int_range(self):
        from graph.state_schemas import CIORawDecision
        c = CIORawDecision(ticker="JPM", decision="ADVANCE", conviction=78)
        assert c.conviction == 78
        with pytest.raises(ValueError):
            CIORawDecision(ticker="JPM", decision="ADVANCE", conviction=120)

    def test_full_construction(self):
        from graph.state_schemas import (
            CIORawDecision,
            CIORawOutput,
            HeldThesisUpdateLLMOutput,
        )
        c = CIORawOutput(
            decisions=[
                CIORawDecision(
                    ticker="NVDA",
                    decision="ADVANCE",
                    rank=1,
                    conviction=88,
                    rationale="strong",
                    entry_thesis=HeldThesisUpdateLLMOutput(
                        bull_case="AI", conviction=85
                    ),
                ),
                CIORawDecision(
                    ticker="JPM",
                    decision="REJECT",
                    rank=2,
                    rationale="rate cycle",
                ),
            ]
        )
        assert len(c.decisions) == 2
        assert c.decisions[0].entry_thesis.conviction == 85


class TestJointFinalizationOutputStringDefense:
    """Pins the 2026-05-03 fix for an observed Sonnet failure mode:
    ``selected_decisions`` returned as a JSON-encoded string instead
    of a structured array. First surfaced in SF
    ``eval-pipeline-validation-2-20260503-130145``.
    """

    def test_actual_list_passes_through_unchanged(self):
        from graph.state_schemas import (
            JointFinalizationDecision,
            JointFinalizationOutput,
        )
        out = JointFinalizationOutput(
            selected_decisions=[
                JointFinalizationDecision(ticker="AAPL", rationale="r/r 2.5"),
                JointFinalizationDecision(ticker="MSFT", rationale="r/r 2.0"),
            ],
            team_rationale="balanced asymmetry mix",
        )
        assert len(out.selected_decisions) == 2
        assert out.selected_decisions[0].ticker == "AAPL"

    def test_json_string_of_list_is_parsed_and_logged(self, caplog):
        """The exact failure shape Sonnet produced today: a string
        whose contents are valid JSON for the expected list."""
        import json
        import logging
        from graph.state_schemas import JointFinalizationOutput

        payload = json.dumps([
            {"ticker": "AAPL", "rationale": "asymmetric upside"},
            {"ticker": "MSFT", "rationale": "earnings catalyst"},
        ])

        with caplog.at_level(logging.WARNING):
            out = JointFinalizationOutput(
                selected_decisions=payload,  # type: ignore[arg-type]
                team_rationale="ok",
            )

        # Pydantic still constructed the structured form correctly.
        assert len(out.selected_decisions) == 2
        assert out.selected_decisions[0].ticker == "AAPL"
        assert out.selected_decisions[1].rationale == "earnings catalyst"
        # The drift event was logged loudly so flow-doctor / CW alarms
        # can pick it up — the validator parse-and-continued rather
        # than silently rescuing.
        assert any(
            "schema-vs-LLM drift" in rec.message
            or "JSON-string" in rec.message
            for rec in caplog.records
        )

    def test_invalid_json_string_raises_normal_pydantic_error(self):
        """If the string isn't valid JSON-list, fall through to the
        normal Pydantic list-type error — failure mode stays loud."""
        import pytest
        from pydantic import ValidationError
        from graph.state_schemas import JointFinalizationOutput

        with pytest.raises(ValidationError, match="list_type|valid list"):
            JointFinalizationOutput(
                selected_decisions="not even close to valid json",  # type: ignore[arg-type]
                team_rationale="ok",
            )

    def test_json_string_of_dict_raises_normal_pydantic_error(self):
        """A JSON-string of a dict (not a list) is NOT what we want
        to silently rescue — list-type error fires normally."""
        import pytest
        from pydantic import ValidationError
        from graph.state_schemas import JointFinalizationOutput

        with pytest.raises(ValidationError, match="list_type|valid list"):
            JointFinalizationOutput(
                selected_decisions='{"ticker": "AAPL"}',  # type: ignore[arg-type]
                team_rationale="ok",
            )

    def test_field_description_present(self):
        """The field description is what Anthropic's tool-use spec
        carries to the LLM — explicitly says 'array, NOT a JSON-encoded
        string'. Pin so future schema edits don't drop it."""
        from graph.state_schemas import JointFinalizationOutput
        field_info = JointFinalizationOutput.model_fields["selected_decisions"]
        assert "structured array" in (field_info.description or "")
        assert "NOT" in (field_info.description or "")

    def test_truncated_jsonlist_string_still_raises_loud(self):
        """Replay of the exact 2026-05-03 SF run-3 failure shape: Sonnet
        truncated mid-rationale at the 800-token cap so the string ends
        with no closing `]`. The mode='before' validator's json.loads()
        rightly rejects incomplete JSON (silently truncating-and-completing
        would mask real malformed responses), so the failure stays loud
        as a Pydantic list_type ValidationError. Pin so the validator's
        fall-through behavior can't regress to a silent rescue."""
        import pytest
        from pydantic import ValidationError
        from graph.state_schemas import JointFinalizationOutput

        truncated = (
            '[\n  {\n    "ticker": "S",\n    "rationale": "something high-'
            'confidence names.'  # <-- truncated mid-string, no closing
        )
        with pytest.raises(ValidationError, match="list_type|valid list"):
            JointFinalizationOutput(
                selected_decisions=truncated,  # type: ignore[arg-type]
                team_rationale="ok",
            )

    def test_jsonlist_with_extra_close_brace_raises_loud(self):
        """Replay of the 2026-05-03 SF run-4 qual_analyst-style failure
        shape (different schema, same class): malformed JSON with
        spurious extra closing braces. Should still raise loud."""
        import pytest
        from pydantic import ValidationError
        from graph.state_schemas import JointFinalizationOutput

        malformed = '[\n  {\n    "ticker": "X",\n    "rationale": null\n}\n}\n'
        with pytest.raises(ValidationError, match="list_type|valid list"):
            JointFinalizationOutput(
                selected_decisions=malformed,  # type: ignore[arg-type]
                team_rationale="ok",
            )
