"""The producer-shadow board is scored under the slot its ARMS declare
(alpha-engine-config-I11425).

``crucible-research-PR814`` registered ``LEADERBOARD_SLOTS["research"]``
(``primary_metric="information_ratio"``, ``per_arm_width=True``) and ``-PR819``
registered five arms declaring ``slot="research"``. Nothing connected them:
``build_producer_leaderboard`` hardcoded ``slot_spec("producer")``, so the
research arms were graded on SPY alpha at a single ``top_n=50``, the
information ratio was computed on every row and ranked on by nothing, and the
``attractiveness_60`` vs ``attractiveness_20`` depth experiment was confounded
by the very width-mixing the information ratio was introduced to price.

Found by RUNNING an arm, not by reading the code: ``scripts/run_experiment.py``
printed ``SLOT producer   primary metric: topn_alpha_vs_benchmark`` on an arm
whose register row says ``research``.

VERIFIED RED against pre-fix routing (champion-challenger-policy.md §7.4).
Restoring the single line ``slot = slot_spec("producer")`` in
``build_producer_leaderboard`` and re-running this file, verbatim:

    FAILED ...::TestBoardIsScoredUnderTheDeclaredSlot::test_board_records_the_slot_it_was_scored_under
      AssertionError: assert 'producer' == 'research'
    FAILED ...::TestBoardIsScoredUnderTheDeclaredSlot::test_arms_are_scored_at_their_own_declared_widths
      KeyError: 'per_arm_width'

``TestSlotResolution`` stays GREEN under that revert, deliberately: the resolver
is correct in isolation and always was. What was broken is that nothing CALLED
it — which is why the guard that matters is the one asserting the property of
the written board, not of the helper.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

_BUCKET = "alpha-engine-research"


# ── The slot resolver ──────────────────────────────────────────────────────


class TestSlotResolution:
    def test_live_arms_resolve_the_research_slot(self):
        """Every live arm declares ``slot="research"``, so the board is scored
        under the research contract — not the producer one."""
        from scoring.leaderboard_producers import _resolve_board_slot

        slot = _resolve_board_slot()
        assert slot.slot_id == "research"
        assert slot.primary_metric == "information_ratio"
        assert slot.per_arm_width is True

    def test_mixed_live_slots_raise_rather_than_picking_one(self, monkeypatch):
        """One board cannot serve two measurement contracts. Picking either
        silently would grade half its rows on the wrong primary — §2's
        "slots are separate axes" failure, rendered as a normal board."""
        import dataclasses

        from producers.registry import RESEARCH_PRODUCERS
        from scoring.leaderboard_producers import _resolve_board_slot

        mixed = dict(RESEARCH_PRODUCERS)
        victim = next(
            n for n, s in mixed.items()
            if s.kind == "challenger" and s.slot == "research"
        )
        mixed[victim] = dataclasses.replace(mixed[victim], slot="producer")
        monkeypatch.setattr(
            "producers.registry.RESEARCH_PRODUCERS", mixed, raising=True
        )
        with pytest.raises(ValueError, match="two slot measurement contracts"):
            _resolve_board_slot()

    def test_no_declared_slot_falls_back_to_producer(self, monkeypatch):
        """Back-compat, stated rather than implicit: a register with no arm
        declaring a slot is the pre-I11425 world and keeps the producer
        contract, so this change cannot silently re-grade a board it was never
        about."""
        import dataclasses

        from producers.registry import RESEARCH_PRODUCERS
        from scoring.leaderboard_producers import _resolve_board_slot

        stripped = {
            n: dataclasses.replace(s, slot=None)
            for n, s in RESEARCH_PRODUCERS.items()
        }
        monkeypatch.setattr(
            "producers.registry.RESEARCH_PRODUCERS", stripped, raising=True
        )
        assert _resolve_board_slot().slot_id == "producer"


# ── Width resolution ───────────────────────────────────────────────────────


class TestDeclaredWidths:
    def test_live_arm_width_comes_from_the_register_not_the_output(self):
        """The width is part of the RECIPE. Reading it off the emitted pick
        count would let a short day silently re-define the arm."""
        from scoring.leaderboard_producers import _declared_widths
        from scoring.leaderboard_scoring import SpecDay, SpecHistory

        arm = SpecHistory(name="attractiveness_60", kind="challenger")
        arm.by_date["2026-09-18"] = SpecDay(ranked=["A", "B"])  # a short day
        assert _declared_widths([arm]) == {"attractiveness_60": 60}

    def test_retired_arm_falls_back_to_its_own_natural_width(self):
        """A retired arm predates ``ProducerSpec.width``. Under
        ``per_arm_width`` every row is compared to the population it drew from
        rather than to another row, so the arm's own breadth is the honest
        one."""
        from scoring.leaderboard_producers import _declared_widths
        from scoring.leaderboard_scoring import SpecDay, SpecHistory

        arm = SpecHistory(name="scanner_predictor_direct", kind="retired")
        arm.by_date["2026-09-11"] = SpecDay(ranked=list("ABCDE"))
        arm.by_date["2026-09-18"] = SpecDay(ranked=list("ABC"))
        assert _declared_widths([arm]) == {"scanner_predictor_direct": 5}

    def test_arm_with_no_width_and_no_picks_is_omitted(self):
        """Omitted, not defaulted: the scorer skips it, which is the correct
        treatment of an arm that produced nothing. Inventing a width would make
        the board's breadth a fact nobody wrote down."""
        from scoring.leaderboard_producers import _declared_widths
        from scoring.leaderboard_scoring import SpecHistory

        arm = SpecHistory(name="never_registered_arm", kind="retired")
        assert _declared_widths([arm]) == {}


# ── The board itself (moto S3) ─────────────────────────────────────────────


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


class _Panel:
    def __init__(self):
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update(
            {t: float(c) for t, c in closes.items()}
        )
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


# 60 tickers, so attractiveness_60 can actually be scored at 60 and
# attractiveness_20 at 20 — the pair whose comparison the fix unconfounds.
_TICKERS = [f"T{i:02d}" for i in range(60)]
_ENTRY = "2026-06-01"
_AS_OF = "2026-08-04"


def _seed_two_widths(s3) -> _Panel:
    """``attractiveness_60`` (60 picks) and ``attractiveness_20`` (20 picks) on
    one cohort date, with a matured 21d-forward panel so both actually score."""
    for arm, n in (("attractiveness_60", 60), ("attractiveness_20", 20)):
        s3.put_object(
            Bucket=_BUCKET,
            Key=f"signals_shadow/{arm}/{_ENTRY}/signals.json",
            Body=json.dumps({
                "signals": {
                    t: {"signal": "ENTER", "score": float(100 - i)}
                    for i, t in enumerate(_TICKERS[:n])
                }
            }).encode(),
        )
    panel = _Panel().put(_ENTRY, {**dict.fromkeys(_TICKERS, 100.0), "SPY": 100.0})
    for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
        # A real ranking: the top names run up more than the tail, which is what
        # makes the 60-vs-20 depth comparison a comparison at all.
        panel.put(d, {
            **{t: 100.0 + (60 - i) * 0.5 for i, t in enumerate(_TICKERS)},
            "SPY": 105.0,
        })
    return panel


class TestBoardIsScoredUnderTheDeclaredSlot:
    def test_board_records_the_slot_it_was_scored_under(self, s3):
        """principles §1 transparency: a reader cannot tell a board ranked on
        information_ratio from one ranked on topn_alpha_vs_benchmark unless the
        artifact says which contract produced its numbers."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        panel = _seed_two_widths(s3)
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=panel.loader()
        )
        assert res["status"] == "ok", res
        board = json.loads(
            s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read()
        )
        assert board["slot"] == "research"
        assert board["primary_metric"] == "information_ratio"
        # The S3 surface `crucible-backtester` reads is unchanged — only the
        # contract behind the numbers moved.
        assert board["leaderboard_id"] == "producer"

    def test_arms_are_scored_at_their_own_declared_widths(self, s3):
        """The depth experiment is unreadable while both arms are truncated to
        one shared ``top_n``: ``per_arm_width=False`` with ``top_n=50`` leaves
        the truncation inert above every arm's width, so 60 and 20 are compared
        on raw means at different breadths."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        panel = _seed_two_widths(s3)
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=panel.loader()
        )
        board = json.loads(
            s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read()
        )
        assert board["per_arm_width"] is True
        # The slot-level top_n is meaningless on this board and must not be
        # readable as one.
        assert board["top_n"] is None
        assert board["widths"]["attractiveness_60"] == 60
        assert board["widths"]["attractiveness_20"] == 20
        rows = {r["name"]: r for r in board["specs"]}
        assert rows["attractiveness_60"]["top_n"] == 60
        assert rows["attractiveness_20"]["top_n"] == 20

    def test_the_primary_metric_is_present_on_every_scored_row(self, s3):
        """A primary nothing carries is a primary nothing can rank on — the
        state I11425 measured, inverted."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        panel = _seed_two_widths(s3)
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=panel.loader()
        )
        board = json.loads(
            s3.get_object(Bucket=_BUCKET, Key=res["key"])["Body"].read()
        )
        for row in board["specs"]:
            assert "information_ratio" in row, row["name"]
