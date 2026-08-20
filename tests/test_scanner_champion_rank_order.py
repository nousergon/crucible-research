"""The scanner leaderboard's champion is ranked by its OWN arm's order
(alpha-engine-config-I7645 → I7808).

`candidates.json::scanner_tickers` IS the champion's ranking: the live path
applies `SCANNER_SPECS[LIVE_CHAMPION].rank`, and every `ScannerSpec.rank` sorts
by its own score descending and slices the top N, so list position is the
champion's ranking by construction — the same property every challenger's
shadow list has (SCANNER_CONTRACT.md §4).

**This file previously asserted the opposite, and the reversal is the point.**
I7645 measured Spearman(list position, `tech_score`) at +0.27 on 2026-08-18 and
-0.04 on 2026-07-30, where a descending `tech_score` order would be -1.00, and
concluded the emission order was arbitrary. It was not arbitrary — it was the
momentum sleeve's ranking, which the live path had adopted on 2026-07-22 while
the spec register still named `tech_score` as the champion's signal. Both
measured dates fall after that cutover, so the correlation was showing that the
champion had stopped ranking on `tech_score`, not that its list was unordered.
Sorting the champion's cut by a signal it does not rank on made its
`realized_rank_ic` a correlation against a RIVAL ARM's ordering, on the board
that judges it — the same asymmetry I7645 set out to remove, pointed the other
way.

`tech_score` is now the ranking signal of the `tech_score_gate` challenger,
where it is scored on its own terms.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.leaderboard_producers import _champion_scanner_day  # noqa: E402

# The champion's own descending ranking. `tech_score` is seeded as its exact
# reverse, so any re-sort by the challenger's signal is visible immediately.
EMITTED = ["AAA", "BBB", "CCC"]
TECH_SCORES = {"AAA": 0.1, "BBB": 0.5, "CCC": 0.9}


def _doc(*, with_log: bool = True, tickers: list[str] | None = None) -> dict:
    doc: dict = {"scanner_tickers": list(EMITTED if tickers is None else tickers)}
    if with_log:
        doc["scanner_eval_log"] = [{"ticker": t, "tech_score": s} for t, s in TECH_SCORES.items()]
    return doc


def test_champion_day_keeps_its_own_order_and_ignores_tech_score():
    day = _champion_scanner_day(_doc())
    assert day.ranked == EMITTED
    assert day.rank_ordered is True


def test_champion_day_is_not_re_sorted_by_a_rival_arms_signal():
    """RED on the I7645 implementation, which returned the tech_score order."""
    day = _champion_scanner_day(_doc())
    assert day.ranked != ["CCC", "BBB", "AAA"]


def test_champion_day_carries_no_borrowed_scores():
    """`tech_score` belongs to the `tech_score_gate` arm. Attaching it to the
    champion's day would make the rank-IC use another arm's signal spacing."""
    day = _champion_scanner_day(_doc())
    assert day.scores is None


def test_an_artifact_without_the_eval_log_is_still_ranked():
    """The pre-2026-07 objects carry no `scanner_eval_log`. Their list is still
    the live cut in its producer's order, so the rank-IC is measurable — under
    I7645 these dates silently reported no rank-IC at all."""
    day = _champion_scanner_day(_doc(with_log=False))
    assert day.ranked == EMITTED
    assert day.rank_ordered is True


def test_an_empty_cut_is_unranked():
    """The only cut with no ranking is an empty one."""
    day = _champion_scanner_day(_doc(tickers=[]))
    assert day.ranked == []
    assert day.rank_ordered is False


def test_loader_gives_the_champion_its_own_ranked_day_end_to_end():
    """The behavioural assertion, at the loader rather than the helper."""
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
    assert champion.name == "tech_score_gate"
    assert champion.by_date["2026-07-01"].ranked == EMITTED
