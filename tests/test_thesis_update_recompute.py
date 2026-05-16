"""Regression tests for score_aggregator's thesis_update recompute path.

Context (ROADMAP P1, 2026-04-22):
    ``thesis_updates`` from the held-stock evaluation path occasionally
    arrive with ``final_score=None``. Prior behavior was to log ERROR +
    skip, which silently dropped valid tickers from ``investment_theses``
    (CME/HSY/KR on 2026-04-20; 9 held tickers on 2026-04-11).

Correct posture (feedback_no_silent_fails / feedback_no_unscoreable_labels):
    - If ``quant_score`` + ``qual_score`` are present, RECOMPUTE the
      composite — the recommendation loop uses the same function.
    - If BOTH sub-scores are also missing, HARD-FAIL — the thesis is
      truly unscoreable and the upstream writer must be fixed.

These tests lock both behaviors.
"""

from __future__ import annotations

import logging

import pytest

from agents.sector_teams.team_config import ALL_TEAM_IDS
from graph.research_graph import score_aggregator


def _state(team_outputs: dict, sector_modifiers: dict | None = None,
           sector_map: dict | None = None) -> dict:
    # ALL-AGENTS-STRICT (Brian, 2026-05-16): score_aggregator now
    # hard-fails if ANY of ALL_TEAM_IDS is missing/failed/partial
    # (revert of #194 isolation). These tests exercise the
    # thesis_update RECOMPUTE / conviction-normalization path that runs
    # *after* the strict gate, so they only need the gate to pass —
    # pad every absent team with a clean, empty, no-error stub. This
    # contributes zero recommendations / theses, so each test's
    # investment_theses assertions are unchanged; it just lets the
    # strict gate through to the path under test. (Pre-existing tests
    # adapted to the new contract — see PR body.)
    padded = dict(team_outputs)
    for tid in ALL_TEAM_IDS:
        if tid not in padded:
            padded[tid] = {
                "team_id": tid,
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
                "partial": False,
                "partial_reasons": [],
            }
    return {
        "sector_team_outputs": padded,
        "sector_modifiers": sector_modifiers or {},
        "sector_map": sector_map or {},
    }


class TestThesisUpdateRecompute:
    def test_missing_final_score_with_sub_scores_is_recomputed(self, caplog):
        """CME/HSY/KR-class: thesis has quant + qual but no final_score.
        Aggregator must recompute and include the ticker.
        """
        state = _state(
            team_outputs={
                "financials": {
                    "recommendations": [],
                    "thesis_updates": {
                        "CME": {
                            "ticker": "CME",
                            "sector": "Financials",
                            "quant_score": 70,
                            "qual_score": 80,
                            # final_score INTENTIONALLY absent
                        },
                    },
                },
            },
            sector_modifiers={"Financials": 1.0},
        )

        with caplog.at_level(logging.WARNING):
            out = score_aggregator(state)

        assert "CME" in out["investment_theses"], (
            "Ticker with recoverable sub-scores was dropped — recompute path "
            "didn't fire. Silent-drop regression."
        )
        cme = out["investment_theses"]["CME"]
        # 0.5 × 70 + 0.5 × 80 + 0 macro shift = 75.0
        assert cme["final_score"] == 75.0
        # Recompute should emit a WARN so upstream is visible.
        assert any("missing final_score — recomputed" in r.message
                   for r in caplog.records), (
            "Recompute path must emit a WARNING pointing at upstream — "
            "silent recomputation would hide the real bug."
        )

    def test_missing_final_score_with_only_quant_score_is_recomputed(self):
        """If only quant_score is present, composite uses it at full weight."""
        state = _state(
            team_outputs={
                "tech": {
                    "recommendations": [],
                    "thesis_updates": {
                        "MSFT": {
                            "ticker": "MSFT",
                            "sector": "Technology",
                            "quant_score": 65,
                            # qual_score + final_score absent
                        },
                    },
                },
            },
            sector_modifiers={"Technology": 1.1},
        )
        out = score_aggregator(state)
        assert "MSFT" in out["investment_theses"]
        # weighted_base = 65 (quant at full weight when qual is None),
        # macro_shift = (1.1-1.0)/0.30 * 10 = 3.33. final = 68.3.
        assert out["investment_theses"]["MSFT"]["final_score"] == pytest.approx(68.3, abs=0.1)

    def test_missing_all_scores_hard_fails(self):
        """If BOTH sub-scores AND final_score are absent, thesis is truly
        unscoreable — raise rather than drop silently.
        """
        state = _state(
            team_outputs={
                "financials": {
                    "recommendations": [],
                    "thesis_updates": {
                        "BROKEN": {
                            "ticker": "BROKEN",
                            "sector": "Financials",
                            # quant_score, qual_score, final_score ALL absent
                        },
                    },
                },
            },
            sector_modifiers={"Financials": 1.0},
        )
        with pytest.raises(RuntimeError, match="BROKEN"):
            score_aggregator(state)

    def test_hard_fail_message_mentions_unscoreable_labels(self):
        """Hard-fail message must point at the feedback file so operators
        know this is a feedback-driven guardrail, not a random error.
        """
        state = _state(
            team_outputs={
                "x": {
                    "recommendations": [],
                    "thesis_updates": {"BROKEN": {"ticker": "BROKEN"}},
                },
            },
        )
        with pytest.raises(RuntimeError) as exc:
            score_aggregator(state)
        assert "unscoreable" in str(exc.value).lower()

    def test_present_final_score_is_passed_through_unchanged(self):
        """Happy path: thesis has final_score already. No recompute,
        no extra WARN.

        Post-Step-F (2026-04-30) the InvestmentThesis schema is enforced
        strictly by default — fixture must include the required ``rating``
        field. The recompute path injects rating from score_to_rating
        when missing, but the no-recompute branch (final_score present)
        passes through whatever the thesis carried.
        """
        state = _state(
            team_outputs={
                "tech": {
                    "recommendations": [],
                    "thesis_updates": {
                        "AAPL": {
                            "ticker": "AAPL",
                            "sector": "Technology",
                            "final_score": 82.5,
                            "quant_score": 80,
                            "qual_score": 85,
                            "rating": "BUY",
                        },
                    },
                },
            },
        )
        out = score_aggregator(state)
        assert out["investment_theses"]["AAPL"]["final_score"] == 82.5


class TestHeldStockConvictionNormalization:
    """Regression for the 2026-04-30 score_aggregator skipped-normalization
    bug. Held-stock thesis_updates carrying agent-format conviction
    previously bypassed normalize_conviction and flowed into
    InvestmentThesis with values that fail the StoredConvictionLiteral
    schema (rising/stable/declining). Fix in graph/research_graph.py:826
    normalizes the held-stock branch the same way the recommendation branch
    does. Updated 2026-04-30 (Option A) to use int 0-100 agent format.
    """

    def test_held_stock_low_conviction_normalized_to_declining(self):
        state = _state(
            team_outputs={
                "energy": {
                    "recommendations": [],
                    "thesis_updates": {
                        "DVN": {
                            "ticker": "DVN",
                            "sector": "Energy",
                            "final_score": 45.0,
                            "quant_score": 50,
                            "qual_score": 40,
                            "rating": "HOLD",
                            "conviction": 25,  # int < 40 → declining
                        },
                    },
                },
            },
        )
        out = score_aggregator(state)
        assert out["investment_theses"]["DVN"]["conviction"] == "declining"

    def test_held_stock_medium_conviction_normalized_to_stable(self):
        state = _state(
            team_outputs={
                "healthcare": {
                    "recommendations": [],
                    "thesis_updates": {
                        "PODD": {
                            "ticker": "PODD",
                            "sector": "Healthcare",
                            "final_score": 60.0,
                            "quant_score": 60,
                            "qual_score": 60,
                            "rating": "HOLD",
                            "conviction": 55,  # int 40-69 → stable
                        },
                    },
                },
            },
        )
        out = score_aggregator(state)
        assert out["investment_theses"]["PODD"]["conviction"] == "stable"

    def test_held_stock_high_conviction_normalized_to_rising(self):
        state = _state(
            team_outputs={
                "tech": {
                    "recommendations": [],
                    "thesis_updates": {
                        "NVDA": {
                            "ticker": "NVDA",
                            "sector": "Technology",
                            "final_score": 80.0,
                            "quant_score": 78,
                            "qual_score": 82,
                            "rating": "BUY",
                            "conviction": 85,  # int >= 70 → rising
                        },
                    },
                },
            },
        )
        out = score_aggregator(state)
        assert out["investment_theses"]["NVDA"]["conviction"] == "rising"

    def test_held_stock_already_storage_format_passes_through(self):
        """Idempotency: if upstream already normalized, normalize_conviction
        is a pass-through."""
        state = _state(
            team_outputs={
                "tech": {
                    "recommendations": [],
                    "thesis_updates": {
                        "AAPL": {
                            "ticker": "AAPL",
                            "sector": "Technology",
                            "final_score": 75.0,
                            "quant_score": 75,
                            "qual_score": 75,
                            "rating": "BUY",
                            "conviction": "rising",
                        },
                    },
                },
            },
        )
        out = score_aggregator(state)
        assert out["investment_theses"]["AAPL"]["conviction"] == "rising"

    def test_legacy_agent_string_normalized_at_aggregator(self):
        """Option A 2026-04-30 transition: research.db rows written before
        PR #56 may carry agent-format strings ('high'/'medium'/'low') that
        the StoredConvictionLiteral schema would reject. score_aggregator's
        normalize_conviction call must flatten them to storage format
        instead of failing typed-state validation."""
        state = _state(
            team_outputs={
                "industrials": {
                    "recommendations": [],
                    "thesis_updates": {
                        "UNP": {
                            "ticker": "UNP",
                            "sector": "Industrials",
                            "final_score": 60.0,
                            "quant_score": 60,
                            "qual_score": 60,
                            "rating": "HOLD",
                            "conviction": "medium",  # legacy string
                        },
                    },
                },
            },
        )
        out = score_aggregator(state)
        # "medium" is no longer a known agent-format string post-Option-A,
        # so normalize_conviction maps it to the safe default "stable".
        assert out["investment_theses"]["UNP"]["conviction"] == "stable"
