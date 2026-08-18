"""The scanner leaderboard's champion is ranked by the score it ranks on
(alpha-engine-config-I7645).

`candidates.json::scanner_tickers` is the quant filter's emission order, not a
ranking — measured against the live artifact, Spearman(list position,
tech_score) is +0.27 on 2026-08-18 and -0.04 on 2026-07-30 where a descending
rank would be -1.00. Every CHALLENGER list is a true ranking by construction
(`data/scanner_specs._rank_*` sorts by its own score and slices the top N), so
reading the champion's emission order as a ranking made the board asymmetric in
the champion's disfavour on two separate metrics: its `realized_rank_ic`, and
which 50 of its 60 names the count-matched top-50 selected.

RED on ed3fb223: `_load_scanner_specs` passed `scanner_tickers` straight in.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.leaderboard_producers import _champion_scanner_day  # noqa: E402

# Emission order is deliberately NOT tech_score order.
EMITTED = ["AAA", "BBB", "CCC"]
SCORES = {"AAA": 0.1, "BBB": 0.9, "CCC": 0.5}


def _doc(*, with_log: bool = True, drop: str | None = None) -> dict:
    log = [
        {"ticker": t, "tech_score": s}
        for t, s in SCORES.items()
        if t != drop
    ]
    doc: dict = {"scanner_tickers": list(EMITTED)}
    if with_log:
        doc["scanner_eval_log"] = log
    return doc


def test_champion_day_is_ordered_by_tech_score_not_emission_order():
    day = _champion_scanner_day(_doc())
    assert day.ranked == ["BBB", "CCC", "AAA"]
    assert day.rank_ordered is True


def test_champion_day_carries_the_scores_the_rank_ic_should_use():
    """With explicit scores the rank-IC uses the real signal spacing rather
    than reconstructing a monotone stand-in from list position."""
    day = _champion_scanner_day(_doc())
    assert day.scores == SCORES


def test_a_candidates_artifact_without_the_eval_log_is_not_passed_off_as_ranked():
    """The pre-2026-07 objects carry no `scanner_eval_log`. An unknown order
    must produce a MISSING rank-IC, never one over emission order."""
    day = _champion_scanner_day(_doc(with_log=False))
    assert day.ranked == EMITTED
    assert day.rank_ordered is False


def test_a_partially_scored_cut_is_not_partially_ranked():
    """Ranking 2 of 3 names would silently drop the unscored one to the tail
    and report the result as a ranking."""
    day = _champion_scanner_day(_doc(drop="CCC"))
    assert day.rank_ordered is False


def test_loader_gives_the_champion_a_ranked_day_end_to_end():
    """The behavioural assertion that is RED on ed3fb223: the loader itself,
    not only the helper this change extracted."""
    import json

    from scoring.leaderboard_producers import _load_scanner_specs

    class _Body:
        def __init__(self, b):
            self._b = b

        def read(self):
            return self._b

    class _S3:
        def get_object(self, Bucket, Key):  # noqa: N803
            if Key.startswith("candidates/"):
                return {"Body": _Body(json.dumps(_doc()).encode())}
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    champion, _ = _load_scanner_specs(_S3(), "b", ["2026-07-01"])
    assert champion.by_date["2026-07-01"].ranked == ["BBB", "CCC", "AAA"]
