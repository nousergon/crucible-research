"""Coverage is a declared set with hysteresis, not a monotonic ratchet.

alpha-engine-config-I6648. ``products/thinktank.md`` §2.1 requires an enter
threshold and a **wider exit threshold** over the upstream ranking, so a name
at the boundary does not enter and leave on ranking noise.

The issue as filed said the implementation applied one threshold for both entry
and continued membership, producing unbounded boundary churn. Measured
2026-08-10, that was wrong in an interesting direction: **nothing ever
de-covered a name at all.** ``ledger.py`` had no removal path — no ``del``, no
``pop``, no rank filter — so ``rank_ceiling`` gated ENTRY only and the covered
set could only grow. Run ``b150c317eeef`` reports ``sweep_tickers: 178``
against ``coverage.rank_ceiling: 150``: 28 names the ceiling no longer admits,
kept because nothing removes them.

So the real §2.1 gap was a **ratchet**, and this is the exit half. Brian's
ruling 2026-08-10 (option a): a de-covered entry stays in ``entries``, marked,
rather than moving to a separate ``dropped`` map.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinktank.ledger import select_intake  # noqa: E402
from thinktank.schemas import CoverageLedger, LedgerEntry  # noqa: E402
from thinktank.settings import _parse_exit_rank  # noqa: E402


def _board(ranked: list[str]) -> dict:
    """A universe board whose order IS the attractiveness ranking."""
    return {
        "stocks": [
            {"ticker": t, "attractiveness_score": 1000.0 - i, "sector": "Tech"}
            for i, t in enumerate(ranked)
        ]
    }


def _ledger(covered: list[str]) -> CoverageLedger:
    return CoverageLedger(
        entries={
            t: LedgerEntry(
                ticker=t,
                covered_since="2026-01-01",
                thesis_version=1,
                thesis_updated_on="2026-08-09",
            )
            for t in covered
        }
    )


# ── The band ─────────────────────────────────────────────────────────────────


def test_a_name_oscillating_inside_the_band_enters_once_and_is_never_dropped():
    """The property the whole clause exists for. Rank 3 admits (enter=3);
    ranks 4 and 5 are inside the band (exit=5) and must not de-cover."""
    ledger = _ledger([])
    for ranking in (
        ["AAA", "BBB", "TGT", "CCC", "DDD", "EEE"],   # TGT rank 3 — enters
        ["AAA", "BBB", "CCC", "TGT", "DDD", "EEE"],   # rank 4 — inside band
        ["AAA", "BBB", "CCC", "DDD", "TGT", "EEE"],   # rank 5 — still inside
        ["AAA", "BBB", "TGT", "CCC", "DDD", "EEE"],   # back to 3
    ):
        new_rows, _ = select_intake(
            ledger, _board(ranking), daily_new_names=5, rank_ceiling=3,
            exit_rank=5, skip_stale_refill=True, trading_day="2026-08-10",
        )
        for row in new_rows:
            ledger.entries[row["ticker"]] = LedgerEntry(
                ticker=row["ticker"], covered_since="2026-08-10",
                thesis_version=1, thesis_updated_on="2026-08-10",
            )
    assert "TGT" in ledger.covered(), (
        "a name oscillating strictly inside the hysteresis band was de-covered "
        "— the band is doing nothing and this is a single cut again"
    )
    assert ledger.entries["TGT"].dropped_on is None


def test_a_name_past_the_exit_rank_is_dropped_exactly_once():
    ledger = _ledger(["TGT"])
    ranking = ["AAA", "BBB", "CCC", "DDD", "EEE", "TGT"]  # rank 6 > exit 5
    for _ in range(3):
        select_intake(
            ledger, _board(ranking), daily_new_names=5, rank_ceiling=3,
            exit_rank=5, skip_stale_refill=True, trading_day="2026-08-10",
        )
    entry = ledger.entries["TGT"]
    assert entry.covered is False
    assert entry.dropped_on == "2026-08-10"
    assert entry.attractiveness_rank_at_drop == 6
    assert "TGT" not in ledger.covered()
    assert "TGT" in ledger.dropped()


# ── What a drop must NOT destroy ─────────────────────────────────────────────


def test_a_dropped_name_keeps_its_entry_and_its_thesis_lineage():
    """§2.2 makes every thesis version immutable and forbids destroying the
    record of what was believed when a decision was made. That binds a
    de-covered name exactly as much as a covered one — and Brian's 2026-08-10
    ruling chose marking over removal precisely so the record survives."""
    ledger = _ledger(["TGT"])
    ledger.entries["TGT"].thesis_version = 7
    select_intake(
        ledger, _board(["A", "B", "C", "D", "E", "TGT"]), daily_new_names=5,
        rank_ceiling=3, exit_rank=5, skip_stale_refill=True,
        trading_day="2026-08-10",
    )
    assert "TGT" in ledger.entries, "the entry was REMOVED, not marked"
    assert ledger.entries["TGT"].thesis_version == 7
    assert ledger.entries["TGT"].covered_since == "2026-01-01"


def test_absence_from_todays_ranking_never_de_covers():
    """Absence means the scanner did not rank it today — a universe change, a
    data gap, a halted ticker. Inferring 'it fell past the exit rank' from
    silence is a failure this codebase has shipped before."""
    ledger = _ledger(["TGT"])
    select_intake(
        ledger, _board(["AAA", "BBB", "CCC"]), daily_new_names=5,
        rank_ceiling=3, exit_rank=5, skip_stale_refill=True,
        trading_day="2026-08-10",
    )
    assert ledger.entries["TGT"].covered is True
    assert "TGT" in ledger.covered()


def test_no_exit_rank_configured_is_the_pre_i6648_ratchet_not_a_default_drop():
    """An un-migrated config must keep working. It must NOT start de-covering
    names because a default appeared."""
    ledger = _ledger(["TGT"])
    select_intake(
        ledger, _board(["A", "B", "C", "D", "E", "F", "G", "TGT"]),
        daily_new_names=5, rank_ceiling=3, skip_stale_refill=True,
        trading_day="2026-08-10",
    )
    assert ledger.entries["TGT"].covered is True


# ── A dropped name must not come back through another door ───────────────────


def test_a_dropped_name_does_not_consume_a_staleness_refresh_slot():
    """Otherwise the ratchet returns through the back door: de-covered today,
    refreshed tomorrow because it is the stalest entry in the map."""
    ledger = _ledger(["TGT", "KEEP"])
    ledger.entries["TGT"].thesis_updated_on = "2020-01-01"   # by far the stalest
    ledger.entries["TGT"].covered = False
    ledger.entries["TGT"].dropped_on = "2026-08-09"
    _, refresh = select_intake(
        ledger, _board(["KEEP", "AAA", "BBB"]), daily_new_names=5,
        rank_ceiling=3, exit_rank=5, trading_day="2026-08-10",
    )
    assert "TGT" not in refresh, (
        "a de-covered name consumed a refresh slot — coverage grows back "
        "silently and the exit threshold means nothing"
    )


def test_a_dropped_name_does_not_breach_the_staleness_sla():
    ledger = _ledger(["TGT"])
    ledger.entries["TGT"].thesis_updated_on = "2020-01-01"
    ledger.entries["TGT"].covered = False
    _, refresh = select_intake(
        ledger, _board(["AAA"]), daily_new_names=0, rank_ceiling=3,
        exit_rank=5, stale_after_days=30, trading_day="2026-08-10",
    )
    assert "TGT" not in refresh


def test_covered_is_not_the_entry_count_after_a_drop():
    """The stated delta of Brian's ruling (a): `entries` keeps every name ever
    covered, so `len(entries)` stopped being a coverage count. Any consumer
    reading it as one is now wrong."""
    ledger = _ledger(["A", "B", "C"])
    ledger.entries["C"].covered = False
    assert len(ledger.entries) == 3
    assert len(ledger.covered()) == 2
    assert ledger.dropped() == {"C"}


# ── Config validation fails at load, not at first drop ───────────────────────


@pytest.mark.parametrize("exit_rank", [150, 149, 1])
def test_an_exit_rank_not_strictly_wider_is_rejected_at_load(exit_rank, monkeypatch):
    """An exit rank at or inside the enter rank de-covers every name it just
    admitted, every run — and the symptom would surface in the coverage ledger
    a long way from the config that caused it."""
    monkeypatch.delenv("THINKTANK_EXIT_RANK", raising=False)
    monkeypatch.delenv("THINKTANK_RANK_CEILING", raising=False)
    with pytest.raises(ValueError, match="STRICTLY greater"):
        _parse_exit_rank({"rank_ceiling": 150, "exit_rank": exit_rank})


def test_a_strictly_wider_exit_rank_parses(monkeypatch):
    monkeypatch.delenv("THINKTANK_EXIT_RANK", raising=False)
    monkeypatch.delenv("THINKTANK_RANK_CEILING", raising=False)
    assert _parse_exit_rank({"rank_ceiling": 150, "exit_rank": 200}) == 200
    assert _parse_exit_rank({"rank_ceiling": 150}) is None
