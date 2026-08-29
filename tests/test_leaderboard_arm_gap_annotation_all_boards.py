"""Per-arm measurement gaps must be named on EVERY leaderboard, not just the
cuts board (alpha-engine-config-I9275).

``_annotate_arm_measurement_gaps`` was written for
``alpha-engine-config-I7819`` and wired into ``build_cuts_leaderboard`` alone.
The other two boards built by the same helpers — ``build_scanner_leaderboard``
(the scanner spec slot: ``momentum_sleeve`` champion vs ``tech_score_gate`` /
``mom_12_1_sleeve``) and ``build_producer_leaderboard`` — never received it, so
they publish a block-level ``n_dates`` that is a UNION across arms with no
field saying which arms contributed nothing.

Measured live on ``s3://alpha-engine-research/scanner/leaderboard/2026-08-28.json``
(2026-08-29): the 21-session block read ``status: "ok"``, ``n_dates: 7`` while
BOTH challengers scored ``n_dates_scored: 0``. The board's own top-level number
described the champion alone, and no ``arms_total`` / ``arms_unmeasured`` field
existed to say so. Same date, ``research/cuts_leaderboard/2026-08-28.json``
carried ``arms_unmeasured`` naming its four unscored arms — the same defect,
visible on one board and invisible on the other two.

engagement-protocol-policy.md §5: the fix survives the class. Both remaining
call sites are wired here, and this file asserts both.

GUARD PROVENANCE (champion-challenger-policy.md §7.4): these assertions were
run against the pre-fix tree and observed RED — ``arms_total`` /
``arms_unmeasured`` are simply absent from every scanner and producer block.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from scoring.leaderboard_producers import (
    build_producer_leaderboard,
    build_scanner_leaderboard,
)

_BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _put_json(client, key, obj):
    client.put_object(Bucket=_BUCKET, Key=key, Body=json.dumps(obj).encode())


def _session_calendar(n: int, start: str = "2025-02-03") -> list[str]:
    import pandas as pd

    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=n)]


class _Panel:
    def __init__(self) -> None:
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update({t: float(c) for t, c in closes.items()})
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


def _panel_and_entries() -> tuple[_Panel, list[str]]:
    cal = _session_calendar(400)
    panel = _Panel()
    for i, d in enumerate(cal):
        panel.put(d, {"A": 100 + i, "B": 100 + 0.5 * i, "C": 100 - 0.2 * i, "SPY": 100 + 0.3 * i})
    return panel, [cal[250], cal[260]]


def _seed(client, entries: list[str]) -> None:
    """One arm per slot carries the whole cohort; its co-arms emit nothing.

    That is the live 2026-08-28 state — ``scanner/leaderboard/2026-08-28.json``
    read ``n_dates: 7`` off the champion while both challengers sat at zero —
    and it is the exact shape the annotation exists to make visible: a block
    that reads measured because ONE arm carries it.

    The scanner board's cohort dates come from ``candidates_shadow/`` prefixes,
    so one challenger is seeded to give the board a cohort at all; the other
    challenger and, at the long horizons, every arm, contribute nothing.
    """
    for d in entries:
        # producer slot: live signals + exactly one shadow producer
        _put_json(
            client,
            f"signals/{d}/signals.json",
            {"signals": {t: {"signal": "ENTER", "score": s} for t, s in [("A", 0.9), ("B", 0.5), ("C", 0.1)]}},
        )
        _put_json(
            client,
            f"signals_shadow/no_agent_quant/{d}/signals.json",
            {"signals": {t: {"signal": "ENTER", "score": s} for t, s in [("C", 0.9), ("B", 0.5), ("A", 0.1)]}},
        )
        # scanner slot: live candidates + exactly one shadow challenger
        _put_json(client, f"candidates/{d}/candidates.json", {"scanner_tickers": ["A", "B", "C"]})
        _put_json(
            client,
            f"candidates_shadow/mom_12_1_sleeve/{d}/candidates.json",
            {"scanner_tickers": ["C", "B", "A"]},
        )


@pytest.mark.parametrize(
    ("builder", "leaderboard_id"),
    [
        (build_scanner_leaderboard, "scanner"),
        (build_producer_leaderboard, "producer"),
    ],
)
def test_every_board_names_the_arms_it_scored_nothing_for(s3, builder, leaderboard_id):
    """Closes-when: every horizon block on the scanner and producer boards
    carries ``arms_total`` and ``arms_unmeasured``, exactly as the cuts board
    already does.

    PRE-FIX: RED — neither key exists on either board.
    """
    panel, entries = _panel_and_entries()
    _seed(s3, entries)
    res = builder(s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader())
    assert res["status"] == "ok", res
    lb = res["leaderboard"]
    assert lb["horizons"], "no horizon blocks were produced"
    for block in lb["horizons"]:
        assert "arms_total" in block, (
            f"{leaderboard_id} block {block['horizon_days']}d does not say how many "
            "arms it holds — a reader cannot tell a measured board from one arm "
            "carrying it (alpha-engine-config-I7819)"
        )
        assert "arms_unmeasured" in block, (
            f"{leaderboard_id} block {block['horizon_days']}d does not name the arms "
            "that contributed nothing to its n_dates"
        )
        assert block["arms_total"] == len(block["specs"])
        # every challenger scored nothing here, and the block must say so
        insufficient = sorted(
            row["name"] for row in block["specs"] if row.get("confidence") == "insufficient"
        )
        assert block["arms_unmeasured"] == insufficient
        # An arm that emitted NO cohort dates at all is a distinct,
        # horizon-independent state and is named separately: it is unscored at
        # every horizon for one reason, not three.
        assert "arms_no_cohort" in block
        assert set(block["arms_no_cohort"]) <= set(block["arms_unmeasured"])


def test_the_scanner_boards_21d_block_is_not_reported_measured_in_silence(s3):
    """The live 2026-08-28 rendering: block ok, n_dates > 0, and every
    challenger at zero. The block must be readable as such WITHOUT diffing
    every row's ``confidence`` by hand."""
    panel, entries = _panel_and_entries()
    _seed(s3, entries)
    res = build_scanner_leaderboard(
        s3, _BUCKET, "2026-08-17", top_n=2, closes_panel_loader=panel.loader()
    )
    block21 = next(b for b in res["leaderboard"]["horizons"] if b["horizon_days"] == 21)
    assert block21["status"] == "ok"
    assert block21["n_dates"] > 0
    assert block21["arms_unmeasured"], (
        "the champion alone carried this block; the unmeasured challengers must "
        "be named on it"
    )
    assert block21["arms_total"] > len(block21["arms_unmeasured"]), (
        "at least the champion scored — a board where NOTHING scored is a "
        "different state and must not render as this one"
    )
