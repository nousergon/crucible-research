"""Intake fills the declared coverage WINDOW first (nous-ergon-ops-I570).

``coverage_complete`` — the gate on this arm writing any leaderboard evidence —
is measured over the window (``FeedWindow.tickers``). Intake used to walk
today's rank order alone. On a cut's formation day that is the same thing,
because the cut is the head of the ranking; on a HELD cut (ranks refreshed
mid-week, cut carried forward) it is not, and intake spent its slots on
non-window names ranked between drifted window members.

Measured on the live ledger 2026-10-03 (cut formed 2026-09-23, ranks of
2026-09-25): 19 window members uncovered, 74 uncovered non-window names inside
``rank_ceiling``, and the window member at today's rank 122 queued behind ~70
of them at five names a day — so the arm could not complete its window before
the cut re-formed, and wrote no leaderboard evidence at all.
"""

from __future__ import annotations

import logging

from thinktank.challenger_selection import _write_shadow_signals
from thinktank.feed import build_feed_window
from thinktank.ledger import select_intake
from thinktank.schemas import ChallengerSelection, CoverageLedger, LedgerEntry


def _ranked(n: int) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _board(tickers: list[str]) -> dict:
    return {"stocks": [{"ticker": t, "sector": "Tech"} for t in tickers]}


def _held_window(cut: list[str], ranked: list[str]):
    """A cut formed earlier than the ranks it is served with."""
    return build_feed_window(
        cut,
        {t: i + 1 for i, t in enumerate(ranked)},
        {
            "cut": "attractiveness_top_10",
            "basis": "attractiveness_rank",
            "run_date": "2026-09-25",
            "cut_effective_date": "2026-09-23",
        },
    )


def _ledger(covered: list[str]) -> CoverageLedger:
    return CoverageLedger(
        entries={
            t: LedgerEntry(ticker=t, covered_since="2026-09-01", thesis_version=1, thesis_updated_on="2026-09-20")
            for t in covered
        }
    )


def test_held_cut_window_members_are_taken_before_higher_ranked_non_members():
    ranked = _ranked(40)
    # The cut was the head on its formation day; since then T008/T009 drifted
    # to ranks 21 and 31, and T010..T019 rose into today's top 10.
    cut = ranked[:8] + ["T020", "T030"]
    window = _held_window(cut, ranked)
    ledger = _ledger(ranked[:8])  # every cut member still in the head is covered

    new_rows, _ = select_intake(ledger, _board(ranked), window, daily_new_names=2, rank_ceiling=35)

    # Before: T008, T009 (today's ranks 9, 10) — neither in the window, so the
    # day's intake moved coverage_complete not at all.
    assert [r["ticker"] for r in new_rows] == ["T020", "T030"]
    assert [r["_attractiveness_rank"] for r in new_rows] == [21, 31]


def test_non_window_names_still_fill_the_remaining_slots_in_rank_order():
    ranked = _ranked(40)
    cut = ranked[:9] + ["T020"]
    window = _held_window(cut, ranked)
    new_rows, _ = select_intake(_ledger(ranked[:9]), _board(ranked), window, daily_new_names=3, rank_ceiling=35)
    assert [r["ticker"] for r in new_rows] == ["T020", "T009", "T010"]


def test_formation_day_order_is_unchanged():
    """On the day a cut forms it IS the head, so window-first is today's order."""
    ranked = _ranked(30)
    window = build_feed_window(
        ranked[:10],
        {t: i + 1 for i, t in enumerate(ranked)},
        {"cut": "c", "basis": "attractiveness_rank", "run_date": "2026-09-23", "cut_effective_date": "2026-09-23"},
    )
    new_rows, _ = select_intake(_ledger(ranked[:3]), _board(ranked), window, daily_new_names=4, rank_ceiling=30)
    assert [r["ticker"] for r in new_rows] == ["T003", "T004", "T005", "T006"]


def test_rank_ceiling_still_bounds_window_members_and_the_block_is_reported(caplog):
    """The ceiling reads today's rank, window member or not (Brian, 2026-08-27).
    A window member past it cannot be taken in, which means coverage_complete
    cannot be reached on this cut — that is stated, not silent."""
    ranked = _ranked(60)
    cut = ranked[:9] + ["T050"]
    window = _held_window(cut, ranked)
    with caplog.at_level(logging.WARNING, logger="thinktank.ledger"):
        new_rows, _ = select_intake(_ledger(ranked[:9]), _board(ranked), window, daily_new_names=1, rank_ceiling=40)
    assert [r["ticker"] for r in new_rows] == ["T009"]
    assert "T050@51" in caplog.text
    assert "cannot be taken in by intake" in caplog.text


def _selection(uncovered: int) -> ChallengerSelection:
    return ChallengerSelection(
        trading_day="2026-10-01",
        calendar_date="2026-10-02",
        run_id="r",
        mode="daily",
        board_date="2026-09-25",
        coverage_complete=uncovered == 0,
        uncovered_count=uncovered,
        selections=[],
    )


def test_skip_alert_names_the_window_width_and_the_live_shadow_key(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "thinktank.challenger_selection.publish_observe_alert",
        lambda **kw: sent.append(kw),
    )

    class _Store:
        def put_json(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("an incomplete selection must not write the shadow view")

    assert _write_shadow_signals(_Store(), _selection(19), window_size=60) is None
    (alert,) = sent
    assert "uncovered=19 of the 60-name coverage window" in alert["message"]
    assert "signals_shadow/thinktank_20/2026-10-01/signals.json NOT written" in alert["message"]
    assert "top-20" not in alert["message"]


def test_complete_selection_writes_the_shadow_view():
    written = {}

    class _Store:
        def put_json(self, key, payload):
            written[key] = payload

    key = _write_shadow_signals(_Store(), _selection(0), window_size=60)
    assert key == "signals_shadow/thinktank_20/2026-10-01/signals.json"
    assert key in written
