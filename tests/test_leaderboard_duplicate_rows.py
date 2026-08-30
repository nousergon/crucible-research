"""One row per arm per horizon, enforced on the ARTIFACT — not only in a test.

alpha-engine-config-I8026 deliverable 3, and the reason it is a separate
deliverable from the merge fix that preceded it.

WHAT ACTUALLY HAPPENED. ``crucible-research#658`` (alpha-engine-config-I7645)
fixed the shallow-copy aliasing in ``score_multi_horizon`` that put every arm
after the first into the primary block twice. It merged 2026-08-18 20:26 UTC
with eleven tests, all green. The scanner Lambda's live version at that moment
was 339, published 2026-08-18 00:04 UTC — twenty hours EARLIER — and the next
publish was version 340 on 2026-08-19 20:15 UTC. So the 2026-08-19 05:58 UTC
run wrote ``research/cuts_leaderboard/2026-08-19.json`` from pre-#658 code, and
it carries ``attractiveness_top_20`` and ``scanner_gate_baseline_60`` twice, in
the 21-session block and at the top level.

Nothing noticed, and nothing could have:

* the unit tests exercise the CODE, and the code was correct from 08-18 20:26;
* ``cut_promotion``'s own duplicate check reads ``PROMOTABLE_CUTS`` in the
  126-session block only, and the duplicates were funnel STAGES in the
  21-session block — a horizon that engine is structurally forbidden to read.
  Run against the real 08-19 artifact on ``origin/main`` it returns
  ``hold / decision_horizon_immature / defect=None``: a clean, plausible,
  wrong reason;
* ``_write_leaderboard`` wrote whatever it was handed.

So the guard is placed on the artifact, at both ends: the producer refuses to
ship a duplicated board and writes a verdict saying why, and the consumer
refuses to decide on one whatever surface the duplicate is on.

RED ON ``origin/main`` at ``aa2b9c92``: every test in this file fails there —
``duplicate_arm_rows`` and ``LeaderboardIntegrityError`` do not exist, and
``decide_cut_champion`` on the real artifact returns ``decision_horizon_immature``
rather than ``board_defective`` (champion-challenger-policy.md §7.4).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.cut_promotion import (  # noqa: E402
    decide_cut_champion,
)
from scoring.leaderboard_producers import _write_leaderboard  # noqa: E402
from scoring.leaderboard_scoring import (  # noqa: E402
    LeaderboardIntegrityError,
    duplicate_arm_rows,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cuts_leaderboard_2026-08-19_structure.json"


@pytest.fixture
def live_defective_board() -> dict:
    """The 2026-08-19 artifact's exact structure. See the fixture's own note:
    metric VALUES are nulled (public repo, private-tier numbers); arm names,
    row counts, horizon blocks and cohort counts are verbatim, and the
    invariant under test reads only those."""
    return json.loads(FIXTURE.read_text())


class _RecordingS3:
    def __init__(self) -> None:
        self.puts: list[tuple[str, dict]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:  # noqa: N803
        self.puts.append((Key, json.loads(Body)))


# ── The detector, against the real artifact's structure ───────────────────────


def test_the_live_2026_08_19_board_is_reported_defective(live_defective_board):
    """The §7.4 proof. This is not a synthetic case: it is the object that sat
    in S3 for two days being read by a promotion engine that said nothing."""
    assert duplicate_arm_rows(live_defective_board) == [
        "21d:attractiveness_top_20x2",
        "21d:scanner_gate_baseline_60x2",
        "top_level:attractiveness_top_20x2",
        "top_level:scanner_gate_baseline_60x2",
    ]


def test_a_clean_board_is_reported_clean(live_defective_board):
    """A guard that cannot pass is as useless as one that cannot fail. Drop the
    duplicate rows from the real artifact and it must go quiet."""
    def dedupe(rows):
        seen, out = set(), []
        for r in rows:
            if r["name"] not in seen:
                seen.add(r["name"])
                out.append(r)
        return out

    board = dict(live_defective_board)
    board["specs"] = dedupe(board["specs"])
    board["horizons"] = [{**b, "specs": dedupe(b["specs"])} for b in board["horizons"]]
    assert duplicate_arm_rows(board) == []


def test_an_empty_or_shapeless_board_is_not_a_duplicate():
    """``unmeasurable`` boards carry ``specs: []`` and no ``horizons``. The
    detector must not turn one failure mode into another."""
    assert duplicate_arm_rows({"status": "unmeasurable", "specs": []}) == []
    assert duplicate_arm_rows({}) == []
    assert duplicate_arm_rows({"horizons": [{"horizon_days": 21, "specs": [{}, {}]}]}) == []


def test_a_duplicate_in_a_non_primary_horizon_is_still_caught():
    """The 126d block is the one ``cut_promotion`` decides from; the 252d block
    is the one it takes a veto from. A duplicate in either must not be reachable
    just because the primary block is clean."""
    board = {
        "specs": [{"name": "a"}],
        "horizons": [
            {"horizon_days": 21, "specs": [{"name": "a"}]},
            {"horizon_days": 252, "specs": [{"name": "a"}, {"name": "a"}]},
        ],
    }
    assert duplicate_arm_rows(board) == ["252d:ax2"]


# ── The producer choke point ──────────────────────────────────────────────────


def test_write_leaderboard_refuses_the_defective_board(live_defective_board):
    """No producer, present or future, can write a duplicated board — the check
    is on the single function all three call, not on any one of them."""
    s3 = _RecordingS3()
    with pytest.raises(LeaderboardIntegrityError) as exc:
        _write_leaderboard(s3, "bucket", "research/cuts_leaderboard/2026-08-19.json", live_defective_board)
    assert "attractiveness_top_20x2" in str(exc.value)
    assert s3.puts == []


def test_write_leaderboard_passes_a_clean_board():
    s3 = _RecordingS3()
    board = {"leaderboard_id": "cuts", "specs": [{"name": "a"}, {"name": "b"}]}
    assert _write_leaderboard(s3, "bucket", "k.json", board) == "k.json"
    assert [k for k, _ in s3.puts] == ["k.json"]


# ── The consumer ──────────────────────────────────────────────────────────────


def test_promotion_refuses_to_decide_on_the_live_defective_board(live_defective_board):
    """On ``origin/main`` this returned ``decision_horizon_immature`` with
    ``defect=None`` — a hold whose stated reason is true of the board and
    silent about the thing actually wrong with it. The duplicates are in the
    21d block, which this engine never decides from, so a per-arm check on the
    decision surface alone cannot see them.

    Still true after the alpha-engine-config-I8261 cutover, and now for a
    sharper reason: the board is only the corroborating VETO source, and an
    ABSENT veto is legitimately non-blocking (§5.1) while a CORRUPT one is not.
    A safety mechanism that may be reading someone else's numbers is worse than
    one that is switched off, so a duplicated board still disqualifies itself
    and the hold still names the defect — even with a weekly ledger that would
    otherwise support a decision."""
    decision = decide_cut_champion(
        ledger_rows=[],
        board=live_defective_board,
        champion_before="attractiveness_top_60",
        decided_on="2026-08-29",
    )
    assert decision.decision == "hold"
    assert decision.defect is not None
    assert "attractiveness_top_20x2" in decision.defect
    # v4: the board no longer selects the reason_code — only records the defect.
    assert decision.reason_code != "board_defective"
    assert decision.champion == "attractiveness_top_60"


def test_a_clean_immature_board_still_holds_on_immaturity_not_on_a_defect():
    """The whole-board check must not swallow the immature case — that is the
    expected state for months and it is not a defect (I7826)."""
    board = {
        "horizons": [
            {"horizon_days": 21, "status": "ok", "specs": [{"name": "attractiveness_top_60"}]},
            {
                "horizon_days": 126,
                "status": "immature",
                "reason": "cohort has not matured",
                "specs": [
                    {"name": "attractiveness_top_60", "n_dates_scored": 0},
                    {"name": "tech_score_top_60", "n_dates_scored": 0},
                ],
            },
        ],
    }
    decision = decide_cut_champion(
        ledger_rows=[], board=board,
        champion_before="attractiveness_top_60", decided_on="2026-08-29",
    )
    assert decision.reason_code != "board_defective"
    assert decision.defect is None
