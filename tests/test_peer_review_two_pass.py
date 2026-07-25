"""Unit tests for peer_review._joint_finalization two-pass flow.

The two-pass design replaces the prior single-pass call after Haiku's
rationale verbosity drift truncated structured output mid-emission
(2026-05-03 + 2026-05-06 incidents). Validates the new shape end-to-end:

- Pass 1 returns selected_tickers + team_rationale → drives slate.
- Pass 2 called once per ticker, returns per-ticker rationale.
- Combined output preserves the existing `{picks, rationale}` contract
  so downstream consumers (sector_team merge, decision capture) don't
  need to change.

Lax-mode fallback (combined-score sort) is exercised when Pass 1 raises;
per-ticker Pass 2 hiccups don't take down the slate.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.sector_teams.peer_review import _joint_finalization
from graph.state_schemas import JointFinalizationDecision, JointSelectionOutput


class _FakeLLM:
    """Minimal stand-in for ChatAnthropic used by `_joint_finalization`.

    `_joint_finalization` rebinds via `ChatAnthropic(model=llm.model,
    anthropic_api_key=llm.anthropic_api_key, max_tokens=...,
    callbacks=llm.callbacks)`, so we just need these attributes. The
    rebound LLM gets patched at the class level so its
    `with_structured_output(...).invoke(...)` is what the test controls.
    """

    model = "claude-haiku-4-5"
    anthropic_api_key = "test-key"
    callbacks = []


@pytest.fixture
def candidates():
    return [
        {
            "ticker": "NVDA",
            "quant_score": 78,
            "qual_score": 72,
            "conviction": 80,
            "rr_ratio": 2.4,
            "bull_case": "AI infrastructure leadership; data-center inflection.",
            "bear_case": "Cyclical inventory risk.",
        },
        {
            "ticker": "PLTR",
            "quant_score": 65,
            "qual_score": 70,
            "conviction": 75,
            "rr_ratio": 3.1,
            "bull_case": "Government contract pipeline + commercial AIP.",
            "bear_case": "High multiple compression risk.",
        },
        {
            "ticker": "RKLB",
            "quant_score": 60,
            "qual_score": 68,
            "conviction": 70,
            "rr_ratio": 2.8,
            "bull_case": "Neutron launch milestone catalyst.",
            "bear_case": "Cash burn.",
        },
        {
            "ticker": "KO",
            "quant_score": 55,
            "qual_score": 60,
            "conviction": 50,
            "rr_ratio": 1.0,
            "bull_case": "Defensive cashflow.",
            "bear_case": "Volume softness.",
        },
    ]


def _patch_two_pass(selection: JointSelectionOutput, rationales: dict[str, str]):
    """Patch ChatAnthropic.with_structured_output(...).invoke(...) so
    the two passes route to the right fake outputs based on the schema
    being bound."""

    def fake_with_structured_output(self, schema, *args, **kwargs):
        class _Bound:
            def invoke(_self, messages, config=None):
                if schema is JointSelectionOutput:
                    return selection
                if schema is JointFinalizationDecision:
                    # The Pass 2 prompt template embeds the ticker — pull
                    # it from the message text so each call returns the
                    # right rationale.
                    text = messages[0].content if messages else ""
                    for ticker in rationales:
                        if f'"{ticker}"' in text or f" {ticker}" in text or f"={ticker}" in text:
                            return JointFinalizationDecision(
                                ticker=ticker, rationale=rationales[ticker]
                            )
                    # Fallback if the message-shape changes — shouldn't
                    # happen in practice but keeps the test robust.
                    first = next(iter(rationales))
                    return JointFinalizationDecision(
                        ticker=first, rationale=rationales[first]
                    )
                raise AssertionError(f"Unexpected schema bound: {schema}")

        return _Bound()

    return patch(
        "agents.sector_teams.peer_review.ChatAnthropic.with_structured_output",
        autospec=True,
        side_effect=fake_with_structured_output,
    )


def test_two_pass_happy_path(candidates):
    """Pass 1 picks 3 tickers + team rationale; Pass 2 fills per-ticker
    rationale for each. Final dict matches the legacy `{picks, rationale}`
    contract so downstream callers see no shape change."""

    selection = JointSelectionOutput(
        selected_tickers=["NVDA", "PLTR", "RKLB"],
        team_rationale="Asymmetric high-R/R slate, AI-infrastructure tilt.",
    )
    rationales = {
        "NVDA": "R/R 2.4 with quant 78 / qual 72 — AI inflection catalyst.",
        "PLTR": "R/R 3.1 strongest in slate; commercial AIP momentum.",
        "RKLB": "R/R 2.8 + Neutron milestone catalyst.",
    }

    with _patch_two_pass(selection, rationales):
        result = _joint_finalization(_FakeLLM(), "tech", candidates, "neutral")

    assert {p["ticker"] for p in result["picks"]} == {"NVDA", "PLTR", "RKLB"}
    assert result["rationale"] == selection.team_rationale
    for pick in result["picks"]:
        assert pick["peer_review_rationale"] == rationales[pick["ticker"]]


def test_pass1_failure_falls_back_to_combined_score(candidates, monkeypatch):
    """If Pass 1 raises (and STRICT_VALIDATION is off), the slate falls
    back to combined quant+qual score. Critical invariant: every team
    must produce picks for the downstream merge."""

    monkeypatch.setenv("STRICT_VALIDATION", "false")

    def fake_with_structured_output(self, schema, *args, **kwargs):
        class _Bound:
            def invoke(_self, messages, config=None):
                if schema is JointSelectionOutput:
                    raise RuntimeError("simulated Pass 1 LLM failure")
                raise AssertionError("Pass 2 should not run when Pass 1 fails")

        return _Bound()

    with patch(
        "agents.sector_teams.peer_review.ChatAnthropic.with_structured_output",
        autospec=True,
        side_effect=fake_with_structured_output,
    ):
        result = _joint_finalization(_FakeLLM(), "tech", candidates, "neutral")

    # Top-3 by combined score in fixtures: NVDA (75), PLTR (67.5), RKLB (64)
    assert {p["ticker"] for p in result["picks"]} == {"NVDA", "PLTR", "RKLB"}
    assert "Fallback" in result["rationale"]


def test_pass2_failure_keeps_pick_with_empty_rationale(candidates, monkeypatch):
    """Per-ticker Pass 2 hiccup must NOT poison the slate. Pick stays;
    rationale is empty. Lax mode only — strict mode re-raises."""

    monkeypatch.setenv("STRICT_VALIDATION", "false")

    selection = JointSelectionOutput(
        selected_tickers=["NVDA", "PLTR"],
        team_rationale="Tight 2-pick slate.",
    )

    def fake_with_structured_output(self, schema, *args, **kwargs):
        class _Bound:
            def invoke(_self, messages, config=None):
                if schema is JointSelectionOutput:
                    return selection
                if schema is JointFinalizationDecision:
                    text = messages[0].content if messages else ""
                    if "NVDA" in text:
                        return JointFinalizationDecision(
                            ticker="NVDA", rationale="AI inflection."
                        )
                    if "PLTR" in text:
                        raise RuntimeError("simulated Pass 2 hiccup for PLTR")
                raise AssertionError(f"Unexpected schema: {schema}")

        return _Bound()

    with patch(
        "agents.sector_teams.peer_review.ChatAnthropic.with_structured_output",
        autospec=True,
        side_effect=fake_with_structured_output,
    ):
        result = _joint_finalization(_FakeLLM(), "tech", candidates, "neutral")

    by_ticker = {p["ticker"]: p for p in result["picks"]}
    assert "NVDA" in by_ticker and "PLTR" in by_ticker
    assert by_ticker["NVDA"]["peer_review_rationale"] == "AI inflection."
    assert by_ticker["PLTR"]["peer_review_rationale"] == ""


def test_hallucinated_ticker_dropped(candidates, monkeypatch):
    """If Pass 1 returns a ticker absent from the candidate set
    (LLM hallucination), it's silently dropped — slate clamps to whatever
    survives the candidate-set guard."""

    monkeypatch.setenv("STRICT_VALIDATION", "false")

    selection = JointSelectionOutput(
        selected_tickers=["NVDA", "FAKE_TICKER", "PLTR"],
        team_rationale="x",
    )
    rationales = {
        "NVDA": "real",
        "PLTR": "real",
    }

    with _patch_two_pass(selection, rationales):
        result = _joint_finalization(_FakeLLM(), "tech", candidates, "neutral")

    tickers = {p["ticker"] for p in result["picks"]}
    assert tickers == {"NVDA", "PLTR"}
    assert "FAKE_TICKER" not in tickers
