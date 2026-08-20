"""The weekly scanner-cut promotion engine, and its refusal to decide on an
immature horizon (alpha-engine-config-I7826).

RED on origin/main at 249bf666: `scoring/cut_promotion.py` does not exist, so
nothing decides and nothing writes `config/scanner_cut_champion.json` — the
pointer and its reader shipped in crucible-research#670 with no producer.

The load-bearing test here is `test_leading_challenger_on_immature_horizon_does
_not_move_the_pointer`: a challenger that is winning, by a wide margin, on a
horizon that has not matured must NOT be promoted, and the artifact must say
so in words a reader can act on. That is the deliverable of I7826, not a caveat
on it (alpha-engine-config-I7580: a −0.264 IC at 21 days drove a live change
nine years of history inverted at 126–252 days).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.cut_promotion import (  # noqa: E402
    AUDIT_DATED_KEY,
    AUDIT_LATEST_KEY,
    CUT_PROMOTION_SLOT,
    REASON_BOARD_DEFECTIVE,
    REASON_BOARD_MISSING,
    REASON_CHAMPION_LEADS,
    REASON_COOLDOWN_ACTIVE,
    REASON_CORROBORATION_DISAGREES,
    REASON_HORIZON_IMMATURE,
    REASON_INSUFFICIENT_DATES,
    REASON_MARGIN_NOT_MET,
    REASON_PROMOTED,
    CutPromotionError,
    decide_cut_champion,
    run_cut_promotion,
)
from scoring.leaderboard_scoring import LONG_HORIZONS_DAYS, slot_spec  # noqa: E402
from scoring.universe_membership import (  # noqa: E402
    CUT_CHAMPION_POINTER_KEY,
    DEFAULT_CUT_CHAMPION,
    PROMOTABLE_CUTS,
    live_cut_champion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contracts" / "scanner_cut_champion.schema.json"

CHAMP = "attractiveness_top_60"
CHALLENGER = "tech_score_top_60"
DATE = "2026-08-22"
FLOOR = CUT_PROMOTION_SLOT.min_dates_for_inference


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _row(name: str, mean: float | None, n: int) -> dict:
    return {
        "name": name,
        "kind": "challenger",
        "realized_rank_ic": None,
        "topn_alpha_vs_champion": None,
        "topn_alpha_vs_benchmark": None,
        "topn_alpha_vs_population": (
            None if mean is None else {"mean": mean, "se": 0.001, "t_stat": 2.0, "n_dates": n}
        ),
        "n_dates_scored": n,
        "confidence": "ok" if n >= FLOOR else ("thin" if n else "insufficient"),
        "top_n": 60,
    }


def _board(
    *,
    h126_status: str = "ok",
    champ: tuple[float | None, int] = (0.01, 10),
    chal: tuple[float | None, int] = (0.02, 10),
    h252_status: str = "immature",
    h252_champ: tuple[float | None, int] | None = None,
    h252_chal: tuple[float | None, int] | None = None,
    include_arms: tuple[str, ...] = (CHAMP, CHALLENGER),
    duplicate: str | None = None,
) -> dict:
    """A cuts leaderboard in the CORRECTED I7819 output contract: one row per
    arm per horizon block, each row carrying its own real `n_dates_scored`.

    This engine codes against that contract and never against today's defective
    board (which scores 1 of 3 arms and duplicates rows) — the fix for that is
    alpha-engine-config-I7819, merging first. `duplicate` exercises the
    engine's refusal to interpret the pre-fix shape.
    """
    def block(h: int, status: str, c, ch) -> dict:
        specs = []
        for arm, ev in ((CHAMP, c), (CHALLENGER, ch)):
            if arm not in include_arms or ev is None:
                continue
            specs.append(_row(arm, ev[0], ev[1]))
            if duplicate == arm:
                specs.append(_row(arm, ev[0], ev[1]))
        return {
            "horizon_days": h,
            "status": status,
            "reason": None if status == "ok" else f"{h}d not matured",
            "n_dates": max((s["n_dates_scored"] for s in specs), default=0),
            "specs": specs,
        }

    return {
        "leaderboard_id": "cuts",
        "date": DATE,
        "per_arm_width": True,
        "min_dates_for_inference": FLOOR,
        "horizons_days": list(LONG_HORIZONS_DAYS),
        "horizons": [
            # The 21d block is deliberately populated with a LANDSLIDE for the
            # challenger. Every test below must ignore it.
            block(21, "ok", (0.001, 30), (0.500, 30)),
            block(126, h126_status, champ, chal),
            block(252, h252_status, h252_champ, h252_chal),
        ],
    }


# ── The slot registry ─────────────────────────────────────────────────────────


def test_decision_horizon_is_never_the_21_day_block():
    assert CUT_PROMOTION_SLOT.decision_horizon_days == 126
    assert 21 in CUT_PROMOTION_SLOT.forbidden_horizons_days
    assert CUT_PROMOTION_SLOT.decision_horizon_days not in CUT_PROMOTION_SLOT.forbidden_horizons_days
    assert CUT_PROMOTION_SLOT.corroborating_horizon_days == 252


def test_slot_registry_agrees_with_the_board_it_reads():
    """§10: a slot names its own metric, horizons and evidence floor — and the
    engine may never demand a different one from the board that supplies them."""
    cuts = slot_spec("cuts")
    assert CUT_PROMOTION_SLOT.primary_metric == cuts.primary_metric
    assert CUT_PROMOTION_SLOT.min_dates_for_inference == cuts.min_dates_for_inference
    for h in (CUT_PROMOTION_SLOT.decision_horizon_days, CUT_PROMOTION_SLOT.corroborating_horizon_days):
        assert h in LONG_HORIZONS_DAYS


def test_arms_are_exactly_the_promotable_cuts():
    assert CUT_PROMOTION_SLOT.arms == PROMOTABLE_CUTS
    assert CUT_PROMOTION_SLOT.default_champion == DEFAULT_CUT_CHAMPION


def test_hysteresis_is_implemented_not_waived():
    """champion-challenger-policy.md §5.2 — either a margin+cooldown, or a §9.3
    delta. This slot implements it, so both must be non-zero."""
    assert CUT_PROMOTION_SLOT.promotion_margin > 0
    assert CUT_PROMOTION_SLOT.cooldown_days > 0


# ── The deliverable: refusing an immature horizon ─────────────────────────────


def test_leading_challenger_on_immature_horizon_does_not_move_the_pointer():
    d = decide_cut_champion(
        board=_board(h126_status="immature", chal=(None, 0), champ=(None, 0)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.champion == CHAMP
    assert d.reason_code == REASON_HORIZON_IMMATURE
    assert "126" in d.reason
    assert "n_dates_scored=0" in d.reason
    # The 21d landslide is present on the board and must not appear anywhere in
    # the record's reasoning as a basis for action.
    assert d.horizon_days == 126


def test_a_thin_but_scored_horizon_is_still_refused():
    d = decide_cut_champion(
        board=_board(champ=(0.01, FLOOR - 1), chal=(0.90, FLOOR - 1)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_INSUFFICIENT_DATES
    assert f"min_dates_for_inference={FLOOR}" in d.reason
    assert d.arms[CHALLENGER].n_dates_scored == FLOOR - 1


def test_a_hold_still_reports_both_arms_n_dates():
    """§3: an arm that produced nothing is recorded as a miss, never omitted —
    and n_dates_scored is the number that says how far off a decision is."""
    d = decide_cut_champion(
        board=_board(h126_status="immature", champ=(None, 0), chal=(None, 0)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert set(d.arms) == set(PROMOTABLE_CUTS)
    for ev in d.arms.values():
        assert ev.n_dates_scored == 0


def test_missing_board_is_a_hold_not_a_silence():
    d = decide_cut_champion(board=None, champion_before=CHAMP, decided_on=DATE)
    assert d.decision == "hold"
    assert d.reason_code == REASON_BOARD_MISSING
    assert d.defect is None


def test_missing_arm_row_never_wins_by_default():
    d = decide_cut_champion(
        board=_board(include_arms=(CHALLENGER,), chal=(0.90, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.champion == CHAMP


# ── Structural defects: record, then fail loud ───────────────────────────────


def test_duplicate_rows_are_a_defect_and_hold():
    d = decide_cut_champion(
        board=_board(duplicate=CHALLENGER, chal=(0.9, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_BOARD_DEFECTIVE
    assert d.defect and CHALLENGER in d.defect


def test_a_board_with_no_horizons_list_is_a_defect():
    d = decide_cut_champion(
        board={"leaderboard_id": "cuts", "specs": []},
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.reason_code == REASON_BOARD_DEFECTIVE
    assert d.defect


# ── Hysteresis ────────────────────────────────────────────────────────────────


def test_lead_under_the_margin_is_not_a_promotion():
    margin = CUT_PROMOTION_SLOT.promotion_margin
    d = decide_cut_champion(
        board=_board(champ=(0.010, 30), chal=(0.010 + margin / 2, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_MARGIN_NOT_MET


def test_cooldown_blocks_an_otherwise_clean_promotion():
    d = decide_cut_champion(
        board=_board(champ=(0.010, 30), chal=(0.100, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
        last_promoted_on="2026-08-15",
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_COOLDOWN_ACTIVE
    assert d.champion == CHAMP


def test_incumbent_still_leading_is_a_hold_with_its_own_slug():
    d = decide_cut_champion(
        board=_board(champ=(0.100, 30), chal=(0.010, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_CHAMPION_LEADS


# ── The 252d veto ─────────────────────────────────────────────────────────────


def test_mature_long_horizon_vetoes_a_short_horizon_promotion():
    d = decide_cut_champion(
        board=_board(
            champ=(0.010, 30),
            chal=(0.100, 30),
            h252_status="ok",
            h252_champ=(0.200, 30),
            h252_chal=(0.010, 30),
        ),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_CORROBORATION_DISAGREES
    assert d.corroborating["leader"] == CHAMP


def test_an_immature_long_horizon_is_reported_never_counted_as_a_pass():
    """§5.1: you cannot gate on a statistic you did not measure — and an
    uncomputed gate reported as a pass is the defect the rule prevents."""
    d = decide_cut_champion(
        board=_board(champ=(0.010, 30), chal=(0.100, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.corroborating["leader"] is None
    assert d.corroborating["disagrees"] is False
    assert "no corroboration applied" in d.corroborating["note"]


def test_a_clean_promotion_moves_the_pointer():
    d = decide_cut_champion(
        board=_board(champ=(0.010, 30), chal=(0.100, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.champion == CHALLENGER
    assert d.reason_code == REASON_PROMOTED
    assert d.last_promoted_on == DATE


# ── The artifact: schema + producer/consumer contract ────────────────────────


class _S3:
    def __init__(self, objects: dict[str, dict] | None = None):
        self.objects = dict(objects or {})
        self.written: dict[str, dict] = {}
        self.write_order: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key in self.written:
            body = json.dumps(self.written[Key]).encode()
        elif Key in self.objects:
            body = json.dumps(self.objects[Key]).encode()
        else:
            raise RuntimeError(f"NoSuchKey: {Key}")
        return {"Body": _Body(body)}

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803
        self.written[Key] = json.loads(Body)
        self.write_order.append(Key)


class _Body:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


def test_run_writes_all_three_keys_dated_record_first():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(h126_status="immature",
                                                               champ=(None, 0), chal=(None, 0))})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert doc["decision"] == "hold"
    assert s3.write_order == [
        AUDIT_DATED_KEY.format(date=DATE),
        AUDIT_LATEST_KEY,
        CUT_CHAMPION_POINTER_KEY,
    ]
    for key in s3.write_order:
        assert s3.written[key]["decision"] == "hold"


def test_every_written_document_validates_against_the_frozen_schema():
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    for board, promoted in (
        (_board(h126_status="immature", champ=(None, 0), chal=(None, 0)), False),
        (_board(champ=(0.010, 30), chal=(0.100, 30)), True),
        (_board(champ=(0.010, FLOOR - 1), chal=(0.100, FLOOR - 1)), False),
    ):
        s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": board})
        doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
        jsonschema.validate(instance=doc, schema=schema)
        assert (doc["decision"] == "promote") is promoted


def test_the_pointer_this_engine_writes_is_the_one_the_feed_reads():
    """Producer/consumer contract test (M0). The consumer is
    universe_membership.live_cut_champion — the function the sector-team feed
    actually resolves through — not a re-read of the same JSON."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert doc["champion"] == CHALLENGER
    assert live_cut_champion(bucket="b", s3_client=s3) == CHALLENGER


def test_a_hold_leaves_the_consumer_on_the_standing_champion():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(h126_status="immature",
                                                               champ=(None, 0), chal=(None, 0))})
    run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert live_cut_champion(bucket="b", s3_client=s3) == DEFAULT_CUT_CHAMPION


def test_a_defective_board_is_recorded_before_the_engine_raises():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(duplicate=CHALLENGER,
                                                               chal=(0.9, 30))})
    with pytest.raises(CutPromotionError):
        run_cut_promotion(DATE, bucket="b", s3_client=s3)
    record = s3.written[AUDIT_DATED_KEY.format(date=DATE)]
    assert record["decision"] == "hold"
    assert record["defect"]
    jsonschema.validate(instance=record, schema=_schema())


def test_demotion_needs_the_same_margin_in_reverse_plus_the_cooldown():
    """§5.2 is symmetric: a promotion one week must not be undone the next.
    Also pins that `last_promoted_on` carries forward across the hold."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    first = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert first["decision"] == "promote"
    assert first["last_promoted_on"] == DATE
    later = "2026-08-29"
    # The incumbent is now tech_score_top_60; the board flips back hard.
    s3.objects[f"research/cuts_leaderboard/{later}.json"] = _board(
        champ=(0.200, 30), chal=(0.010, 30)
    )
    second = run_cut_promotion(later, bucket="b", s3_client=s3)
    assert second["reason_code"] == REASON_COOLDOWN_ACTIVE
    assert second["champion"] == CHALLENGER
    assert second["last_promoted_on"] == DATE
    assert live_cut_champion(bucket="b", s3_client=s3) == CHALLENGER


def test_this_week_the_engine_holds_and_says_why():
    """The concrete first-run outcome I7826 commissions: tech_score_top_60 was
    first emitted 2026-08-20, so the 126d block is immature and the engine must
    write a hold whose reason names the horizon and the shortfall."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(h126_status="immature",
                                                               champ=(None, 0), chal=(None, 0))})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert doc["decision"] == "hold"
    assert doc["champion"] == DEFAULT_CUT_CHAMPION
    assert doc["reason_code"] == REASON_HORIZON_IMMATURE
    assert "126" in doc["reason"]
    assert doc["arms"][CHALLENGER]["n_dates_scored"] == 0
    assert doc["arms"][CHAMP]["n_dates_scored"] == 0


def test_schema_is_referenced_by_the_producer_module():
    assert SCHEMA_PATH.exists()
    assert "scanner_cut_champion.schema.json" in (
        REPO_ROOT / "scoring" / "cut_promotion.py"
    ).read_text(encoding="utf-8")
