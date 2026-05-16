"""score_aggregator all-agents-strict hard-fail contract.

CONTRACT CHANGE (Brian, 2026-05-16) — this file's #194-era classes
(TestScoreAggregatorPartialTolerance / TestScoreAggregatorIsolation,
which pinned "a failed/partial team is tolerated; the run only ERRORs
when nothing survives for CIO to rank") are INTENTIONALLY REPLACED, not
a regression to preserve:

  "If the sector agents don't run, Research shouldn't complete until
   all sectors are run. ... We don't get anything from this process if
   the sectors, or any other agent for that matter, fail/don't run."

New contract: score_aggregator raises (-> handler status:ERROR, NO
signals.json / email / DB write) if ANY sector team is missing
(absent from ALL_TEAM_IDS coverage), failed (carries ``error``), or
partial. Surviving picks from other teams do NOT save the run.

What is KEPT (composes with the directive): teams that succeed are
still persisted to S3 by sector_team_node, so an SF redrive reuses
them and only re-attempts the still-missing team within the long
429-retry window. That persistence/resume behavior is exercised in
test_sector_team_persist_backoff.py.

Pre-existing tests rewritten:
  - test_passes_through_when_no_errors / _when_error_key_absent: a
    SINGLE team with no error used to pass; now it fails because the
    other 5 of ALL_TEAM_IDS are missing. Rewritten to supply a full
    6-team clean set for the pass-through assertion.
  - TestScoreAggregatorPartialTolerance.* : partial no longer
    tolerated -> all rewritten to assert RAISE.
  - TestScoreAggregatorIsolation.* : isolation removed -> the
    multi-team-429 regression now asserts the run RAISES even when
    other teams produced picks (the exact behavior the directive
    mandates).
"""

from __future__ import annotations

import pytest

from agents.sector_teams.team_config import ALL_TEAM_IDS
from graph.research_graph import score_aggregator


def _state(team_outputs: dict) -> dict:
    return {
        "sector_team_outputs": team_outputs,
        "sector_modifiers": {},
        "sector_map": {},
    }


def _clean(team_id: str, recs=None) -> dict:
    return {
        "team_id": team_id,
        "recommendations": recs if recs is not None else [],
        "thesis_updates": {},
        "error": None,
        "partial": False,
        "partial_reasons": [],
    }


def _all_clean(extra=None) -> dict:
    """A full, complete, all-clean 6-team set (the only shape that
    passes the all-agents-strict gate)."""
    out = {tid: _clean(tid) for tid in ALL_TEAM_IDS}
    if extra:
        out.update(extra)
    return out


class TestAllAgentsStrictHardFail:
    def test_single_errored_team_raises_and_names_it(self):
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": "RecursionError: exceeded recursion_limit",
            },
        })
        with pytest.raises(RuntimeError, match="technology"):
            score_aggregator(state)

    def test_all_errored_teams_listed(self):
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

    def test_missing_teams_hard_fail(self):
        """REWRITTEN from test_passes_through_when_no_errors: a single
        clean team is NO LONGER a pass — the other 5 of ALL_TEAM_IDS
        are missing, which is fatal under all-agents-strict."""
        state = _state({"technology": _clean("technology")})
        with pytest.raises(RuntimeError, match="missing"):
            score_aggregator(state)

    def test_missing_teams_hard_fail_error_key_absent(self):
        """REWRITTEN from test_passes_through_when_error_key_absent."""
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {}},
        })
        with pytest.raises(RuntimeError, match="missing"):
            score_aggregator(state)

    def test_full_clean_set_passes_through(self):
        """The ONLY pass shape: every ALL_TEAM_IDS team present, none
        errored, none partial. (Recommendations may be empty — a clean
        zero-pick team under the regime-conditional gate is valid.)"""
        result = score_aggregator(_state(_all_clean()))
        assert result == {"investment_theses": {}}


class TestPartialNoLongerTolerated:
    """REWRITTEN TestScoreAggregatorPartialTolerance. #194/#2026-05-02
    tolerated recursion-exhausted partial teams; the directive reverses
    that — a partial team did not produce real output, so the run
    hard-fails."""

    def test_partial_team_raises_even_with_other_clean_teams(self):
        state = _state(_all_clean({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
                "partial": True,
                "partial_reasons": ["quant:recursion_limit_exhausted"],
            },
        }))
        with pytest.raises(RuntimeError, match="partial"):
            score_aggregator(state)

    def test_single_partial_team_raises(self):
        state = _state({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": None,
                "partial": True,
                "partial_reasons": ["quant:recursion_limit_exhausted"],
            },
        })
        with pytest.raises(RuntimeError):
            score_aggregator(state)

    def test_all_partial_raises(self):
        state = _state({
            "technology": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["quant:recursion_limit_exhausted"]},
            "healthcare": {"recommendations": [], "thesis_updates": {},
                           "error": None, "partial": True,
                           "partial_reasons": ["quant:recursion_limit_exhausted"]},
        })
        with pytest.raises(RuntimeError, match="ALL-AGENTS-STRICT"):
            score_aggregator(state)

    def test_partial_with_picks_still_raises(self):
        """The most pointed reversal: a partial team that even produced
        picks still hard-fails (it did not COMPLETE — the directive is
        about agents running, not just yielding something)."""
        state = _state(_all_clean({
            "technology": {
                "recommendations": [{"ticker": "NVDA", "quant_score": 70,
                                     "qual_score": 66}],
                "thesis_updates": {},
                "error": None, "partial": True,
                "partial_reasons": ["qual:recursion_limit_exhausted"],
            },
        }))
        with pytest.raises(RuntimeError, match="partial"):
            score_aggregator(state)


class TestNoIsolation:
    """REWRITTEN TestScoreAggregatorIsolation. #194 isolation REMOVED:
    a failed team aborts the run even when other teams produced usable
    picks (the exact behavior Brian's directive mandates — we get
    nothing from a process whose agents didn't all run)."""

    def test_one_team_failed_others_have_picks_RAISES(self):
        state = _state(_all_clean({
            "technology": {
                "recommendations": [],
                "thesis_updates": {},
                "error": (
                    "RateLimitError: 429 — org rate limit of 450,000 "
                    "input tokens/min, claude-haiku-4-5"
                ),
            },
            "healthcare": _clean(
                "healthcare",
                recs=[{"ticker": "LLY", "quant_score": 70,
                       "qual_score": 65}],
            ),
        }))
        with pytest.raises(RuntimeError, match="technology"):
            score_aggregator(state)

    def test_2026_05_16_multi_team_429_now_hard_fails(self):
        """The exact 2026-05-16 shape — defensives / financials /
        technology all 429 while healthcare/industrials/consumer
        produced picks. #194 made this NOT abort; the all-agents-strict
        directive makes it abort (no signals.json / email). This is the
        single most important behavior reversal of the rework."""
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
            "healthcare": _clean("healthcare", recs=[
                {"ticker": "LLY", "quant_score": 70, "qual_score": 65}]),
            "industrials": _clean("industrials", recs=[
                {"ticker": "CAT", "quant_score": 60, "qual_score": 58}]),
            "consumer": _clean("consumer", recs=[
                {"ticker": "COST", "quant_score": 72, "qual_score": 68}]),
        })
        with pytest.raises(RuntimeError) as exc:
            score_aggregator(state)
        m = str(exc.value)
        assert "ALL-AGENTS-STRICT" in m
        assert "defensives" in m and "financials" in m and "technology" in m

    def test_all_failed_no_picks_still_raises(self):
        msg_429 = "RateLimitError 429 — org rate limit"
        state = _state({
            "defensives": {"recommendations": [], "thesis_updates": {},
                           "error": msg_429},
            "financials": {"recommendations": [], "thesis_updates": {},
                           "error": msg_429},
        })
        with pytest.raises(RuntimeError, match="ALL-AGENTS-STRICT"):
            score_aggregator(state)
