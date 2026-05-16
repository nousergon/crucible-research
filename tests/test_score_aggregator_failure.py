"""Tests for score_aggregator failure / isolation behavior.

Two layered contracts:

1. Original (PR #25): an exception in a team's ReAct agent must not be
   silently swallowed — a crashed team is distinguishable from a team
   that legitimately produced no picks (the ``error`` marker).

2. Per-team isolation (2026-05-16 multi-team-429 fix): a team that
   *still* fails after 429 backoff is tolerated exactly like a partial
   team. It contributes zero recommendations but does NOT nuke the run
   when other teams produced usable picks. The run ERRORs ONLY when
   every team is failed-or-partial AND not a single recommendation
   survives ("nothing for CIO to rank"). Every team that *succeeded*
   has already been persisted to S3 by ``sector_team_node`` before any
   other team could fail, so an SF re-invocation reuses the completed
   teams and only re-executes the failed ones.
"""

from __future__ import annotations

import pytest

from graph.research_graph import score_aggregator


def _state(team_outputs: dict) -> dict:
    return {
        "sector_team_outputs": team_outputs,
        "sector_modifiers": {},
        "sector_map": {},
    }


class TestScoreAggregatorHardFail:
    """The run ERRORs only when there is genuinely nothing for CIO to
    rank: every team failed-or-partial AND zero recommendations survive.
    """

    def test_raises_when_only_team_errored_with_no_picks(self):
        # Single team, errored, no picks anywhere → nothing to rank.
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": "RecursionError: exceeded recursion_limit",
            },
        })
        with pytest.raises(RuntimeError, match="technology"):
            score_aggregator(state)

    def test_raises_with_all_failed_teams_listed(self):
        # Every team errored, no picks → nothing to rank; both named.
        state = _state({
            "healthcare": {"recommendations": [], "thesis_updates": {},
                           "error": "APIError: 529"},
            "defensives": {"recommendations": [], "thesis_updates": {},
                           "error": "JSONDecodeError: malformed"},
        })
        with pytest.raises(RuntimeError) as exc:
            score_aggregator(state)
        msg = str(exc.value)
        assert "healthcare" in msg
        assert "defensives" in msg

    def test_passes_through_when_no_errors(self):
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
            },
        })
        # Should not raise — error=None means the team legitimately had no picks.
        result = score_aggregator(state)
        assert result == {"investment_theses": {}}

    def test_passes_through_when_error_key_absent(self):
        # Backward compat: team_outputs written before this change lack
        # the `error` key entirely — aggregator should not raise.
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {}},
        })
        result = score_aggregator(state)
        assert result == {"investment_theses": {}}


class TestScoreAggregatorPartialTolerance:
    """2026-05-02: sector teams that hit recursion_limit must NOT crash the
    SF. They return ``{partial: True, error: None}``; aggregator logs WARN
    and proceeds. Distinct from real errors which still hard-fail.
    """

    def test_partial_team_does_not_raise(self, caplog):
        import logging
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
                "partial": True,
                "partial_reasons": ["quant:recursion_limit_exhausted"],
            },
            "healthcare": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
            },
        })
        with caplog.at_level(logging.WARNING, logger="research"):
            result = score_aggregator(state)
        assert result == {"investment_theses": {}}
        assert any(
            "partial" in r.message and "technology" in r.message
            for r in caplog.records
        ), f"Expected WARN naming the partial team; got: {[r.message for r in caplog.records]}"

    def test_only_team_failed_with_no_picks_raises(self):
        """A single team that errored (e.g. a 429 that survived backoff)
        with no other team and no picks anywhere → nothing for CIO to
        rank → raise. (Isolation only saves the run when *other* teams
        produced usable picks — see TestScoreAggregatorIsolation.)
        """
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": "APIError: 529",
                "partial": True,
                "partial_reasons": ["quant:recursion_limit_exhausted"],
            },
        })
        with pytest.raises(RuntimeError, match="technology"):
            score_aggregator(state)

    def test_all_teams_partial_raises(self):
        """If every team is partial, the CIO has nothing to rank — that's
        a system-wide failure even though no single team errored. Hard-fail
        so operators investigate the systemic cause."""
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["quant:recursion_limit_exhausted"]},
            "healthcare": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["quant:recursion_limit_exhausted"]},
        })
        with pytest.raises(
            RuntimeError,
            match="sector teams degraded with zero usable recommendations",
        ):
            score_aggregator(state)

    def test_mixed_partial_and_full_teams_advances(self):
        """Most realistic scenario: 1-2 teams partial, rest fine. SF
        advances; CIO ranks what it has."""
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["qual:recursion_limit_exhausted"]},
            "consumer": {"recommendations": [], "thesis_updates": {},
                         "error": None, "partial": False},
            "healthcare": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": False},
        })
        # Should not raise.
        result = score_aggregator(state)
        assert result == {"investment_theses": {}}


class TestScoreAggregatorIsolation:
    """Per-team isolation — a team that still fails after 429 backoff
    must NOT abort the run when other teams produced usable picks. This
    is the load-bearing behavior change of the 2026-05-16 resilience
    fix: re-runs reuse the persisted succeeded teams (already written to
    S3 by sector_team_node) and only re-execute the failed ones.
    """

    def test_one_team_failed_others_have_picks_does_not_raise(self, caplog):
        import logging
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": (
                    "RateLimitError: 429 — org rate limit of 450,000 "
                    "input tokens/min, claude-haiku-4-5"
                ),
            },
            "healthcare": {
                "recommendations": [
                    {"ticker": "LLY", "quant_score": 70, "qual_score": 65}
                ],
                "thesis_updates": {},
                "error": None,
            },
        })
        with caplog.at_level(logging.WARNING, logger="research"):
            result = score_aggregator(state)
        # Did NOT raise; the surviving team's pick was scored.
        assert "LLY" in result["investment_theses"]
        # The failed team is named in a WARN, not raised.
        assert any(
            "technology" in r.message and "FAILED" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_2026_05_16_multi_team_429_does_not_abort_when_picks_survive(self):
        """Regression: the exact 2026-05-16 shape — defensives /
        financials / technology all 429 — must NOT abort the whole run
        when ≥1 other team produced usable picks. Pre-fix this raised
        ``RuntimeError("sector team(s) failed: ...")`` and the Lambda
        returned status:ERROR, discarding every successful team.
        """
        msg_429 = (
            "RateLimitError 429 — org rate limit of 450,000 input "
            "tokens/min, claude-haiku-4-5"
        )
        state = _state({
            "defensives": {"recommendations": [], "thesis_updates": {},
                           "error": msg_429},
            "financials": {"recommendations": [], "thesis_updates": {},
                           "error": msg_429},
            "technology": {"recommendations": [], "thesis_updates": {},
                           "error": msg_429},
            "healthcare": {"recommendations": [
                    {"ticker": "LLY", "quant_score": 70, "qual_score": 65}
                ],
                           "thesis_updates": {}, "error": None},
            "industrials": {
                "recommendations": [
                    {"ticker": "CAT", "quant_score": 60, "qual_score": 58}
                ],
                "thesis_updates": {}, "error": None},
            "consumer": {
                "recommendations": [
                    {"ticker": "COST", "quant_score": 72, "qual_score": 68}
                ],
                "thesis_updates": {}, "error": None},
        })
        # Must NOT raise — 3 teams 429'd but 3 produced usable picks.
        result = score_aggregator(state)
        theses = result["investment_theses"]
        # All 3 surviving teams' picks were scored (not discarded).
        assert {"LLY", "CAT", "COST"}.issubset(set(theses))

    def test_all_failed_no_picks_still_raises(self):
        """Isolation does not mask a total wipeout — every team 429'd,
        zero picks survive → nothing for CIO to rank → ERROR.
        """
        msg_429 = "RateLimitError 429 — org rate limit"
        state = _state({
            "defensives": {"recommendations": [], "thesis_updates": {},
                           "error": msg_429},
            "financials": {"recommendations": [], "thesis_updates": {},
                           "error": msg_429},
        })
        with pytest.raises(
            RuntimeError,
            match="sector teams degraded with zero usable recommendations",
        ):
            score_aggregator(state)
