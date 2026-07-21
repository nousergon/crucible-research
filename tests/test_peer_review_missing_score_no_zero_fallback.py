"""Regression tests for the L4525 / config#680 "Score 0.0 root" fix.

Root cause (diagnosed 2026-07-03): ``peer_review._merge_candidates`` defaulted
a missing ``quant_score`` to a literal ``0`` (``qp.get("quant_score", 0)``),
while the sibling ``qual_score`` field correctly stayed ``None`` when absent
(``qa.get("qual_score")``, no default). A quant pick that reaches
``_merge_candidates`` with ``ticker`` present but no ``quant_score`` (schema
drop / partial structured-output extraction — ``sector_team.py``'s
``valid_picks`` filter only checks for the ``ticker`` key, not
``quant_score``) therefore got scored as if the quant analyst had rated it
0/100 — a genuine worst-case rating — rather than "no quant score available".
That fabricated ``0`` then flowed unchanged through
``compute_composite_breakdown`` / ``compute_composite_score`` (which have no
way to distinguish "quant_score=0, genuinely the worst" from "quant_score=0,
silently defaulted") into a ``final_score`` that reads as legitimate and gets
persisted to ``signals.json``, then swept by the backtester's param_sweep
(``min_score`` 45-80) as a real, terrible-but-real signal — the mechanism
behind the "wall of Score 0.0" the input_quality gate (backtester #306) was
built to detect but not fix.

A parallel silent-zero existed in ``_joint_finalization``'s lax-mode
combined-score fallback sort (``qs = c.get("quant_score") or 0``), which
would also rank a missing-score candidate as if it had scored a genuine 0.

Fix: missing ``quant_score`` now stays ``None`` (mirrors ``qual_score``)
through ``_merge_candidates``, and the fallback sort reuses
``_candidate_composite_score`` (None-safe) so unscored candidates sink to the
bottom deterministically instead of being treated as tied-with-genuine-zero.

Legitimate zero scores (a real, computed 0/100) are NOT touched by this fix —
``scoring/composite.py``'s ``test_zero_scores`` (test_composite_edge_cases.py)
continues to pin that ``quant_score=0, qual_score=0`` correctly produces
``final_score == 0.0``.
"""

from __future__ import annotations

from unittest.mock import patch

from agents.sector_teams.peer_review import (
    _candidate_composite_score,
    _joint_finalization,
    _merge_candidates,
)
from graph.state_schemas import JointSelectionOutput


class _FakeLLM:
    model = "claude-haiku-4-5"
    anthropic_api_key = "test-key"
    callbacks = []


class TestMergeCandidatesNoSilentZero:
    """``_merge_candidates`` must not coerce a missing quant_score to 0."""

    def test_missing_quant_score_stays_none(self):
        quant_picks = [{"ticker": "NVDA", "rationale": "no score emitted"}]
        qual_assessments = [{"ticker": "NVDA", "qual_score": 70}]

        merged = _merge_candidates(quant_picks, qual_assessments, None, False)

        assert len(merged) == 1
        assert merged[0]["quant_score"] is None, (
            "missing quant_score must stay None, not silently become 0 "
            "(0 reads as a genuine worst-case rating downstream)"
        )
        assert merged[0]["qual_score"] == 70

    def test_present_quant_score_passes_through_unchanged(self):
        quant_picks = [{"ticker": "NVDA", "quant_score": 82, "rationale": "r"}]
        qual_assessments = [{"ticker": "NVDA", "qual_score": 70}]

        merged = _merge_candidates(quant_picks, qual_assessments, None, False)

        assert merged[0]["quant_score"] == 82

    def test_genuine_zero_quant_score_preserved_as_zero(self):
        """A REAL 0 (the quant analyst explicitly scored it 0) must still
        come through as 0 — this fix only targets the missing-key case,
        not legitimate worst-case scores."""
        quant_picks = [{"ticker": "BAD", "quant_score": 0, "rationale": "r"}]
        qual_assessments = [{"ticker": "BAD", "qual_score": 10}]

        merged = _merge_candidates(quant_picks, qual_assessments, None, False)

        assert merged[0]["quant_score"] == 0
        assert merged[0]["quant_score"] is not None

    def test_missing_quant_score_on_additional_candidate_stays_none(self):
        additional = {"ticker": "RKLB", "qual_score": 65}  # no quant_score
        merged = _merge_candidates([], [], additional, additional_accepted=True)

        assert len(merged) == 1
        assert merged[0]["quant_score"] is None

    def test_missing_qual_score_still_none_unaffected_by_fix(self):
        """Sanity: qual_score's pre-existing None-on-missing behavior is
        unchanged by this fix (it was already correct)."""
        quant_picks = [{"ticker": "NVDA", "quant_score": 82}]
        merged = _merge_candidates(quant_picks, [], None, False)

        assert merged[0]["qual_score"] is None


class TestCandidateCompositeScoreNoneSafety:
    def test_missing_quant_returns_none(self):
        assert _candidate_composite_score({"qual_score": 70}) is None

    def test_missing_qual_falls_back_to_quant_only(self):
        assert _candidate_composite_score({"quant_score": 60}) == 60.0

    def test_both_present_averages(self):
        assert _candidate_composite_score(
            {"quant_score": 80, "qual_score": 60}
        ) == 70.0

    def test_genuine_zero_quant_computes_real_zero_not_none(self):
        assert _candidate_composite_score(
            {"quant_score": 0, "qual_score": 0}
        ) == 0.0


class TestJointFinalizationFallbackSortNoSilentZero:
    """Pass-1-failure combined-score fallback must not bury a candidate
    that IS scored behind one that merely lacks a score, and must not
    rank a missing-score candidate as if it tied a genuine 0."""

    def _run_fallback(self, candidates):
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
            return _joint_finalization(_FakeLLM(), "tech", candidates, "neutral")

    def test_unscored_candidate_sinks_below_genuine_low_scorer(self, monkeypatch):
        """Before the fix, a candidate with NO quant_score (None) sorted as
        combined=0, tying (or beating alphabetically/stably) a candidate
        that genuinely scored low but nonzero. After the fix the unscored
        candidate must rank strictly below any candidate with a real score,
        however low."""
        monkeypatch.setenv("STRICT_VALIDATION", "false")
        candidates = [
            {"ticker": "UNSCORED", "quant_score": None, "qual_score": None,
             "bull_case": "", "bear_case": ""},
            {"ticker": "LOWSCORE", "quant_score": 5, "qual_score": 3,
             "bull_case": "", "bear_case": ""},
            {"ticker": "MIDSCORE", "quant_score": 55, "qual_score": 60,
             "bull_case": "", "bear_case": ""},
        ]

        result = self._run_fallback(candidates)
        ranked_tickers = [p["ticker"] for p in result["picks"]]

        assert ranked_tickers.index("MIDSCORE") < ranked_tickers.index("LOWSCORE")
        assert ranked_tickers.index("LOWSCORE") < ranked_tickers.index("UNSCORED"), (
            "an unscored candidate must sink below a genuinely low (but "
            "real) scorer, not tie/beat it via a silent 0 default"
        )

    def test_all_scored_fallback_matches_legacy_ordering(self, monkeypatch):
        """Regression guard: when every candidate has real scores, the
        fallback ordering is unchanged by the refactor."""
        monkeypatch.setenv("STRICT_VALIDATION", "false")
        candidates = [
            {"ticker": "NVDA", "quant_score": 78, "qual_score": 72,
             "bull_case": "", "bear_case": ""},
            {"ticker": "PLTR", "quant_score": 65, "qual_score": 70,
             "bull_case": "", "bear_case": ""},
            {"ticker": "RKLB", "quant_score": 60, "qual_score": 68,
             "bull_case": "", "bear_case": ""},
        ]

        result = self._run_fallback(candidates)
        ranked_tickers = [p["ticker"] for p in result["picks"]]

        # NVDA (75), PLTR (67.5), RKLB (64) — same ordering the pre-existing
        # test_pass1_failure_falls_back_to_combined_score pins.
        assert ranked_tickers == ["NVDA", "PLTR", "RKLB"]
