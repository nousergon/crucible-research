"""Tests for the eval-judge input-sufficiency gate.

ROADMAP P0 (2026-05-13, surfaced by L83 spot-check substrate): the
eval-judge had no input-sufficiency gate, so it scored degenerate
runs (e.g. thesis_update with empty prior + zero news + null analyst)
at high confidence — masking upstream substrate gaps with an inflated
quality signal in the CW metric stream.

These tests pin the per-rubric definitions of "degenerate" and verify
the skip-eval emit path produces the right ``judge_skip_reason``.
"""

from __future__ import annotations

from nousergon_lib.decision_capture import (
    DecisionArtifact,
    FullPromptContext,
    ModelMetadata,
)

from evals.judge import _is_degenerate_input


def _artifact(
    agent_id: str,
    snap: dict | None = None,
    output: dict | None = None,
) -> DecisionArtifact:
    return DecisionArtifact(
        run_id="r1",
        timestamp="2026-05-09T22:30:00.000Z",
        agent_id=agent_id,
        model_metadata=ModelMetadata(model_name="claude-haiku-4-5"),
        full_prompt_context=FullPromptContext(
            system_prompt="<prompt>",
            user_prompt="<rendered>",
        ),
        input_data_snapshot=snap or {},
        input_data_summary="summary",
        agent_output=output if output is not None else {"out": "ok"},
    )


# ── thesis_update ───────────────────────────────────────────────────────


class TestThesisUpdateDegenerate:
    def test_empty_prior_empty_news_null_analyst_is_degenerate(self):
        """Exactly the L83 finding: thesis_update fired with no
        substantive inputs."""
        snap = {
            "prior_thesis": {"thesis_summary": "", "score": 52.0},
            "news_data": {"articles": [], "article_count": 0},
            "analyst_data": None,
        }
        assert _is_degenerate_input(_artifact("thesis_update:tech:MCK", snap))

    def test_empty_prior_empty_news_empty_analyst_dict_is_degenerate(self):
        """``fetch_analyst_consensus`` returns a skeleton dict (config#1821
        Option B, 2026-07-08: only ticker/current_price/earnings_surprises
        remain in the shape) when FMP budget is exhausted — should still
        be degenerate."""
        snap = {
            "prior_thesis": {"thesis_summary": ""},
            "news_data": {"articles": []},
            "analyst_data": {
                "ticker": "MCK",
                "current_price": None,
                "earnings_surprises": [],
            },
        }
        assert _is_degenerate_input(_artifact("thesis_update:tech:MCK", snap))

    def test_prior_with_thesis_summary_is_not_degenerate(self):
        """If the agent has a real prior summary to update against, the
        run is substantive even without news/analyst — that's the
        partial-input case the agent SHOULD handle."""
        snap = {
            "prior_thesis": {
                "thesis_summary": "AMAT is a defensive semis supplier...",
            },
            "news_data": {"articles": []},
            "analyst_data": None,
        }
        assert not _is_degenerate_input(_artifact("thesis_update:tech:AMAT", snap))

    def test_news_articles_present_is_not_degenerate(self):
        snap = {
            "prior_thesis": {"thesis_summary": ""},
            "news_data": {"articles": [{"headline": "Big news"}]},
            "analyst_data": None,
        }
        assert not _is_degenerate_input(_artifact("thesis_update:tech:X", snap))

    def test_substantive_analyst_data_is_not_degenerate(self):
        """config#1821 Option B (2026-07-08): earnings_surprises is the
        only analyst_data field left that counts as substantive —
        consensus_rating/mean_target/num_analysts/rating_changes were
        removed from fetch_analyst_consensus's returned shape."""
        snap = {
            "prior_thesis": {"thesis_summary": ""},
            "news_data": {"articles": []},
            "analyst_data": {
                "earnings_surprises": [
                    {"date": "2026-06-15", "actual": 1.2, "estimated": 1.1, "surprise_pct": 9.1},
                ],
            },
        }
        assert not _is_degenerate_input(_artifact("thesis_update:tech:X", snap))

    def test_missing_prior_thesis_key_treated_as_empty(self):
        """Tolerate snapshot shapes that omit the prior_thesis key
        entirely (older capture format) — still degenerate when nothing
        else is substantive."""
        snap = {
            "news_data": {"articles": []},
            "analyst_data": None,
        }
        assert _is_degenerate_input(_artifact("thesis_update:tech:X", snap))


# ── sector_quant + sector_qual ──────────────────────────────────────────


class TestSectorTeamDegenerate:
    def test_sector_quant_empty_tickers_is_degenerate(self):
        snap = {"sector_tickers": [], "sector_tickers_count": 0}
        assert _is_degenerate_input(_artifact("sector_quant:tech", snap))

    def test_sector_quant_with_tickers_is_not_degenerate(self):
        snap = {
            "sector_tickers": ["AAPL", "MSFT"],
            "sector_tickers_count": 2,
            "technical_scores_team": {},
        }
        # Has tickers → not degenerate even if scores are empty
        assert not _is_degenerate_input(_artifact("sector_quant:tech", snap))

    def test_sector_quant_count_only_is_not_degenerate(self):
        """Older snapshot format with count but no list — still
        non-degenerate as long as count > 0."""
        snap = {"sector_tickers_count": 217}
        assert not _is_degenerate_input(_artifact("sector_quant:tech", snap))

    def test_sector_qual_empty_population_is_degenerate(self):
        snap = {
            "sector_tickers": [],
            "sector_population": [],
            "sector_tickers_count": 0,
        }
        assert _is_degenerate_input(_artifact("sector_qual:tech", snap))

    def test_sector_qual_with_population_is_not_degenerate(self):
        snap = {"sector_population": ["AAPL"], "sector_tickers_count": 1}
        assert not _is_degenerate_input(_artifact("sector_qual:tech", snap))


# ── sector_peer_review ──────────────────────────────────────────────────


class TestSectorPeerReviewDegenerate:
    def test_no_quant_no_qual_picks_is_degenerate(self):
        snap = {"quant_picks": [], "qual_picks": []}
        assert _is_degenerate_input(_artifact("sector_peer_review:tech", snap))

    def test_quant_picks_only_is_not_degenerate(self):
        snap = {"quant_picks": [{"ticker": "AAPL"}], "qual_picks": []}
        assert not _is_degenerate_input(_artifact("sector_peer_review:tech", snap))

    def test_qual_picks_only_is_not_degenerate(self):
        snap = {"quant_picks": [], "qual_picks": [{"ticker": "AAPL"}]}
        assert not _is_degenerate_input(_artifact("sector_peer_review:tech", snap))


# ── macro + ic_cio: never degenerate ───────────────────────────────────


class TestMacroAndCioNeverDegenerate:
    def test_macro_economist_always_substantive(self):
        # Even with an empty snapshot, macro is never degenerate by
        # design — explicit pass-through.
        assert not _is_degenerate_input(_artifact("macro_economist", {}))

    def test_ic_cio_always_substantive(self):
        assert not _is_degenerate_input(_artifact("ic_cio", {}))


# ── Unknown agent types: fall through to normal path ───────────────────


class TestUnknownAgentTypeFallsThrough:
    def test_unknown_agent_id_returns_false(self):
        """An unknown agent_id shouldn't trip the gate — the existing
        ``resolve_rubric_for_agent`` path handles that case earlier."""
        assert not _is_degenerate_input(_artifact("unknown_xyz", {}))


# ── Sync path emits the skip artifact + skips the LLM call ─────────────
#
# alpha-engine-config-I9330 (2026-08-29): the end-to-end
# degenerate-input-short-circuit tests that lived here drove
# ``evaluate_artifact`` with ``thesis_update:*`` agent_ids. Since
# ``resolve_rubric_for_agent`` retired that family (and
# sector_quant/sector_qual/sector_peer_review/macro_economist/ic_cio
# alongside it), ``evaluate_artifact`` now raises "No rubric mapped"
# for all of them BEFORE reaching ``_is_degenerate_input`` — the
# per-family branches above are exercised only by this file's UNIT
# tests (which call ``_is_degenerate_input`` directly, bypassing
# rubric resolution) and are otherwise unreachable in production.
# Neither live family (thinktank_thesis / thinktank_theme) has a
# degenerate-input definition, so there is currently no live agent
# for which this end-to-end path can be exercised. Follow-up filed to
# retire ``_is_degenerate_input``'s now-dead branches (and this file's
# now-parallel-dead per-family unit tests above) as the same class of
# defect I9330 already retired from ``resolve_rubric_for_agent`` —
# alpha-engine-config-I9382.
