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
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nousergon_lib.trading_calendar import add_trading_days  # noqa: E402

from scoring.cut_promotion import (  # noqa: E402
    AUDIT_DATED_KEY,
    AUDIT_LATEST_KEY,
    CONTAMINATION_CAVEAT,
    CUT_PROMOTION_SLOT,
    FIRST_COHORT_DATE,
    REASON_BOARD_DEFECTIVE,
    REASON_BOARD_MISSING,
    REASON_CHAMPION_LEADS,
    REASON_COOLDOWN_ACTIVE,
    REASON_CORROBORATION_DISAGREES,
    REASON_HORIZON_IMMATURE,
    REASON_INSUFFICIENT_DATES,
    REASON_MARGIN_NOT_MET,
    REASON_NO_PROMOTABLE_CHALLENGER,
    REASON_PROMOTED,
    CutPromotionError,
    decide_cut_champion,
    reconcile_arms_with_leaderboard,
    run_cut_promotion,
)
from scoring.leaderboard_scoring import LONG_HORIZONS_DAYS, slot_spec  # noqa: E402
from scoring.universe_membership import (  # noqa: E402
    CUT_CHAMPION_POINTER_KEY,
    DEFAULT_CUT_CHAMPION,
    OBSERVE_ONLY_CUTS,
    PROMOTABLE_CUTS,
    SLOT_ARMS,
    live_cut_champion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contracts" / "scanner_cut_champion.schema.json"

CHAMP = "attractiveness_top_60"
CHALLENGER = "tech_score_top_60"

# The DECISION machinery is exercised against a two-arm slot, explicitly.
#
# Brian ruling 2026-08-21 (alpha-engine-config-I8060) made `tech_score_top_60`
# observe-only, so the LIVE registry has one promotable arm and the engine
# short-circuits with `no_promotable_challenger` before any comparison. That is
# correct, and it must not take the promotion logic out of test with it: the arm
# returns by a one-line registry edit, and the margin / cooldown / veto paths
# have to still be right on the day it does. So these tests state the two-arm
# slot they need instead of inheriting whatever the registry currently declares
# — the same reason the fixtures state their own boards.
TWO_ARM_SLOT = replace(
    CUT_PROMOTION_SLOT,
    arms=(CHAMP, CHALLENGER),
    observe_only_arms=tuple(
        a for a in CUT_PROMOTION_SLOT.observe_only_arms if a != CHALLENGER
    ),
)
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
    assert CUT_PROMOTION_SLOT.observe_only_arms == OBSERVE_ONLY_CUTS
    # Promotable and observe-only are disjoint, and every arm the board scores
    # is in exactly one of them — an arm in neither is unscored, an arm in both
    # is a contradiction about whether the pointer may name it.
    assert not set(PROMOTABLE_CUTS) & set(OBSERVE_ONLY_CUTS)
    assert set(SLOT_ARMS) == set(PROMOTABLE_CUTS) | set(OBSERVE_ONLY_CUTS)
    assert CUT_PROMOTION_SLOT.default_champion == DEFAULT_CUT_CHAMPION


def test_hysteresis_is_implemented_not_waived():
    """champion-challenger-policy.md §5.2 — either a margin+cooldown, or a §9.3
    delta. This slot implements it, so both must be non-zero."""
    assert CUT_PROMOTION_SLOT.promotion_margin > 0
    assert CUT_PROMOTION_SLOT.cooldown_days > 0


# ── The deliverable: refusing an immature horizon ─────────────────────────────


def test_leading_challenger_on_immature_horizon_does_not_move_the_pointer():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
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
        slot=TWO_ARM_SLOT,
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
        slot=TWO_ARM_SLOT,
        board=_board(h126_status="immature", champ=(None, 0), chal=(None, 0)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert set(d.arms) == set(TWO_ARM_SLOT.arms)
    for ev in d.arms.values():
        assert ev.n_dates_scored == 0


def test_missing_board_is_a_hold_not_a_silence():
    d = decide_cut_champion(board=None, champion_before=CHAMP, decided_on=DATE, slot=TWO_ARM_SLOT)
    assert d.decision == "hold"
    assert d.reason_code == REASON_BOARD_MISSING
    assert d.defect is None


def test_missing_arm_row_never_wins_by_default():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        board=_board(include_arms=(CHALLENGER,), chal=(0.90, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.champion == CHAMP


# ── Structural defects: record, then fail loud ───────────────────────────────


def test_duplicate_rows_are_a_defect_and_hold():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        board=_board(duplicate=CHALLENGER, chal=(0.9, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_BOARD_DEFECTIVE
    assert d.defect and CHALLENGER in d.defect


def test_a_board_with_no_horizons_list_is_a_defect():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
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
        slot=TWO_ARM_SLOT,
        board=_board(champ=(0.010, 30), chal=(0.010 + margin / 2, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_MARGIN_NOT_MET


def test_cooldown_blocks_an_otherwise_clean_promotion():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
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
        slot=TWO_ARM_SLOT,
        board=_board(champ=(0.100, 30), chal=(0.010, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_CHAMPION_LEADS


# ── The 252d veto ─────────────────────────────────────────────────────────────


def test_mature_long_horizon_vetoes_a_short_horizon_promotion():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
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
        slot=TWO_ARM_SLOT,
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
        slot=TWO_ARM_SLOT,
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
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
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
        doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
        jsonschema.validate(instance=doc, schema=schema)
        assert (doc["decision"] == "promote") is promoted


def test_the_pointer_this_engine_writes_is_the_one_the_feed_reads():
    """Producer/consumer contract test (M0). The consumer is
    universe_membership.live_cut_champion — the function the sector-team feed
    actually resolves through — not a re-read of the same JSON.

    `PROMOTABLE_CUTS` is patched to the two-arm set for the read side as well:
    the pointer validator is registry-global by design, so a promotion in a
    two-arm world is only readable in a two-arm world. That coupling is the
    point — see the test below it.
    """
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
    assert doc["champion"] == CHALLENGER
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        assert live_cut_champion(bucket="b", s3_client=s3) == CHALLENGER


def test_an_observe_only_arm_can_never_be_served_even_if_the_pointer_names_it():
    """The registry is the last line, not the engine (alpha-engine-config-I8060).

    Making `tech_score_top_60` observe-only removes it from PROMOTABLE_CUTS,
    and `live_cut_champion` RAISES on a pointer naming anything outside that
    set rather than quietly serving the default. So even a stale pointer, a
    hand-edited object, or an engine running an older registry cannot hand the
    sector teams an arm Brian has not cleared — the refusal does not depend on
    the promotion engine being correct.
    """
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
    with pytest.raises(Exception) as exc:
        live_cut_champion(bucket="b", s3_client=s3)
    assert CHALLENGER in str(exc.value)
    assert "attractiveness_top_60" in str(exc.value)


def test_a_hold_leaves_the_consumer_on_the_standing_champion():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(h126_status="immature",
                                                               champ=(None, 0), chal=(None, 0))})
    run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
    assert live_cut_champion(bucket="b", s3_client=s3) == DEFAULT_CUT_CHAMPION


def test_a_defective_board_is_recorded_before_the_engine_raises():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(duplicate=CHALLENGER,
                                                               chal=(0.9, 30))})
    with pytest.raises(CutPromotionError):
        run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
    record = s3.written[AUDIT_DATED_KEY.format(date=DATE)]
    assert record["decision"] == "hold"
    assert record["defect"]
    jsonschema.validate(instance=record, schema=_schema())


def test_demotion_needs_the_same_margin_in_reverse_plus_the_cooldown():
    """§5.2 is symmetric: a promotion one week must not be undone the next.
    Also pins that `last_promoted_on` carries forward across the hold."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    # The whole sequence runs in a two-arm world: `run_cut_promotion` resolves
    # `champion_before` through `live_cut_champion`, which validates against the
    # LIVE registry, so once week one promotes to an arm that registry does not
    # list, week two cannot even read the incumbent.
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        first = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
        assert first["decision"] == "promote"
        assert first["last_promoted_on"] == DATE
        later = "2026-08-29"
        # The incumbent is now tech_score_top_60; the board flips back hard.
        s3.objects[f"research/cuts_leaderboard/{later}.json"] = _board(
            champ=(0.200, 30), chal=(0.010, 30)
        )
        second = run_cut_promotion(later, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
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
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
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


# ── Observe-only: the slot with no promotable challenger (I8060) ─────────────


def test_the_live_registry_has_no_promotable_challenger_and_says_so():
    """Brian ruling 2026-08-21: `tech_score_top_60` is observe-only until it has
    weeks of measured performance.

    The engine must NAME that state. Falling through to the comparison path and
    reporting `champion_already_leads` would be a claim about evidence made
    where no comparison happened — two different things rendering identically,
    which is the §3 failure this codebase keeps paying for.
    """
    d = decide_cut_champion(
        board=_board(champ=(0.010, 30), chal=(0.100, 30)),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER
    assert d.champion == CHAMP
    # The observe-only arms are NAMED in the record, so a reader can tell
    # "measured and ineligible" from "not measured at all" without leaving it.
    for arm in OBSERVE_ONLY_CUTS:
        assert arm in d.reason


def test_the_no_challenger_hold_does_not_depend_on_the_board():
    """It is a fact about the REGISTRY. A missing or unmeasurable board must not
    mask it — otherwise the reason flips week to week for a state that has not
    changed."""
    for board in (None, {"status": "unmeasurable", "reason": "no cohort"}):
        d = decide_cut_champion(board=board, champion_before=CHAMP, decided_on=DATE)
        assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER, board


def test_the_record_is_still_written_every_cycle_while_holding():
    """A loop that is armed and waiting must not look like a dead one: the
    unconditional audit record is the whole reason this engine writes on a hold
    (champion-challenger-policy.md §3)."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert doc["reason_code"] == REASON_NO_PROMOTABLE_CHALLENGER
    assert f"config/apply_audit/scanner_cut_champion/{DATE}.json" in s3.written
    assert "config/apply_audit/scanner_cut_champion/latest.json" in s3.written
    assert live_cut_champion(bucket="b", s3_client=s3) == CHAMP


def test_the_observe_only_arms_are_still_scored():
    """Non-promotable must never quietly become non-measured (§3). The board's
    arm list is resolved from the registry's SLOT_ARMS, so an arm moved out of
    PROMOTABLE_CUTS keeps its row and keeps accumulating n_dates_scored."""
    import inspect

    import scoring.leaderboard_producers as lp

    src = inspect.getsource(lp._load_cut_specs)
    assert "OBSERVE_ONLY_CUTS" in src
    assert set(SLOT_ARMS) - set(PROMOTABLE_CUTS) == set(OBSERVE_ONLY_CUTS)


def test_restoring_an_arm_is_a_one_line_registry_edit():
    """The exit from this state must be cheap and obvious, or the hold becomes
    permanent by inertia. Nothing outside the registry names the arm set."""
    d = decide_cut_champion(board=None, champion_before=CHAMP, decided_on=DATE)
    assert "PROMOTABLE_CUTS" in d.reason


def test_the_live_no_challenger_record_validates_against_the_frozen_schema():
    """The record the LIVE registry produces every cycle must satisfy the frozen
    contract — not only the two-arm records the rest of this file exercises.

    RED before alpha-engine-config-I8060's schema edit twice over: `arms`
    required a `tech_score_top_60` key this record no longer carries, and
    `no_promotable_challenger` was not in the reason_code enum.
    """
    import jsonschema

    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert doc["reason_code"] == REASON_NO_PROMOTABLE_CHALLENGER
    jsonschema.validate(doc, _schema())


# ── alpha-engine-config-I8257: the record's arms are self-qualifying ─────────
#
# RED on the pre-fix cut_promotion.py (verified by stashing these three
# deliverables and re-running): `ArmEvidence` carried no `horizon_days` /
# `metric`, `to_document()` emitted no `excluded_horizons` / `decision_
# earliest_on` key, and `reconcile_arms_with_leaderboard` did not exist at
# all — every test below either KeyErrors, AttributeErrors on the missing
# name, or (for the schema tests) fails `jsonschema.validate` on the newly
# `required` fields.


def test_every_arm_carries_its_own_horizon_and_metric():
    """I8257 deliverable 1: n_dates_scored must be self-qualifying without a
    reader joining back to CUT_PROMOTION_SLOT.decision_horizon_days by hand —
    the exact join the misread in the issue skipped."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
    for arm in (CHAMP, CHALLENGER):
        assert doc["arms"][arm]["horizon_days"] == CUT_PROMOTION_SLOT.decision_horizon_days
        assert doc["arms"][arm]["metric"] == CUT_PROMOTION_SLOT.primary_metric


def test_a_hold_that_never_reads_the_board_still_stamps_horizon_and_metric():
    """Even the registry-only no_promotable_challenger hold — which never
    reads a row — must not leave n_dates_scored=0 unqualified."""
    d = decide_cut_champion(board=None, champion_before=CHAMP, decided_on=DATE)
    assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER
    for ev in d.arms.values():
        assert ev.horizon_days == CUT_PROMOTION_SLOT.decision_horizon_days
        assert ev.metric == CUT_PROMOTION_SLOT.primary_metric
        assert ev.present is False


def test_excluded_horizons_carries_the_21d_measurement_with_its_caveat():
    """I8257 deliverable 2: the forbidden 21d block IS measured every cycle
    (the fixture always populates it with a landslide for the challenger) —
    the record must carry that number, why it is not a decision input, and
    the contamination caveat, rather than making a reader open the
    leaderboard to learn any of the three."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
    excl = doc["excluded_horizons"]
    for arm in (CHAMP, CHALLENGER):
        entry = excl[arm]["21"]
        assert entry["horizon_days"] == 21
        assert entry["n_dates_scored"] == 30
        assert CONTAMINATION_CAVEAT in entry["excluded_reason"]
        assert "alpha-engine-config-I7580" in entry["excluded_reason"]
    # 126 — the DECISION horizon — is never in the excluded block: it is the
    # one horizon that IS a decision input.
    assert "126" not in excl[CHAMP]


def test_excluded_horizons_is_written_even_with_no_board():
    """Unconditional, same discipline as the rest of this record (§3): a
    missing board must not make the excluded-horizons block disappear."""
    d = decide_cut_champion(
        board=None, champion_before=CHAMP, decided_on=DATE, slot=TWO_ARM_SLOT
    )
    assert d.reason_code == REASON_BOARD_MISSING
    entry = d.excluded_horizons[CHAMP]["21"]
    assert entry["n_dates_scored"] == 0
    assert "No 21d block" in entry["excluded_reason"]


def test_decision_earliest_on_is_first_cohort_plus_decision_horizon_sessions():
    """I8257 deliverable 3: turns 'hold, no promotable challenger' from a
    status that reads as possibly-imminent into one a reader can price
    against a calendar."""
    expected = add_trading_days(
        FIRST_COHORT_DATE, CUT_PROMOTION_SLOT.decision_horizon_days
    ).isoformat()
    d = decide_cut_champion(board=None, champion_before=CHAMP, decided_on=DATE)
    assert d.decision_earliest_on == expected
    # ~6 months of trading sessions after 2026-08-20 lands in early 2027, not
    # next week — pins the concrete fact from the issue body.
    assert expected.startswith("2027-")


def test_decision_earliest_on_does_not_move_with_decided_on():
    """It is a property of the SLOT (FIRST_COHORT_DATE + decision_horizon_days),
    never of when a given cycle happens to run — otherwise a run today and a
    run next week would disagree about when the horizon matures."""
    d1 = decide_cut_champion(board=None, champion_before=CHAMP, decided_on=DATE)
    d2 = decide_cut_champion(board=None, champion_before=CHAMP, decided_on="2026-09-05")
    assert d1.decision_earliest_on == d2.decision_earliest_on


# ── I8257 deliverable 4: the reconciliation guard ─────────────────────────────


def test_reconcile_catches_an_arm_whose_record_disagrees_with_its_own_horizon():
    """Direct unit test on the guard. RED on pre-fix cut_promotion.py:
    `reconcile_arms_with_leaderboard` does not exist there at all."""
    board = _board(champ=(0.010, 30), chal=(0.100, 30))
    doc = {
        "arms": {
            CHAMP: {
                "n_dates_scored": 0,  # WRONG — the board reports 30 at 126d
                "horizon_days": 126,
                "present": True,
            }
        }
    }
    mismatches = reconcile_arms_with_leaderboard(doc, board)
    assert len(mismatches) == 1
    assert CHAMP in mismatches[0]
    assert "126" in mismatches[0]
    assert "30" in mismatches[0]


def test_reconcile_is_silent_when_the_record_agrees():
    board = _board(champ=(0.010, 30), chal=(0.100, 30))
    doc = {
        "arms": {
            CHAMP: {"n_dates_scored": 30, "horizon_days": 126, "present": True},
        }
    }
    assert reconcile_arms_with_leaderboard(doc, board) == []


def test_reconcile_ignores_arms_the_decision_never_read():
    """present=False means the decision path never consulted the board for
    this arm at all (board_missing, the registry-only no_promotable_challenger
    short-circuit) — nothing to reconcile, and this must not false-positive."""
    board = _board(champ=(0.010, 30), chal=(0.100, 30))
    doc = {
        "arms": {
            CHAMP: {"n_dates_scored": 0, "horizon_days": 126, "present": False},
        }
    }
    assert reconcile_arms_with_leaderboard(doc, board) == []


def test_run_cut_promotion_fails_when_the_written_record_disagrees_with_the_board():
    """End-to-end: a corrupted read path (simulated here — decide_cut_champion
    itself never produces this, by construction) must not reach S3 unchallenged.
    The record is still WRITTEN first (FAIL-LOUD §: record before raise), then
    the process fails loud rather than serving a self-contradicting artifact
    silently."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(champ=(0.010, 30), chal=(0.100, 30))})

    def _bad_evidence(row, *, horizon_days, metric):
        from scoring.cut_promotion import ArmEvidence

        return ArmEvidence(
            n_dates_scored=999,  # deliberately wrong vs. the board's real 30
            confidence="ok",
            topn_alpha_vs_population_mean=row.get(metric, {}).get("mean")
            if isinstance(row.get(metric), dict)
            else None,
            t_stat=None,
            present=True,
            horizon_days=horizon_days,
            metric=metric,
        )

    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms), patch(
        "scoring.cut_promotion._evidence", side_effect=_bad_evidence
    ):
        with pytest.raises(CutPromotionError, match="disagrees with the leaderboard"):
            run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)

    # Record-first-then-fail: the corrupted record is durable, not swallowed.
    assert f"config/apply_audit/scanner_cut_champion/{DATE}.json" in s3.written
    written = s3.written[f"config/apply_audit/scanner_cut_champion/{DATE}.json"]
    assert written["arms"][CHAMP]["n_dates_scored"] == 999


def test_the_schema_requires_excluded_horizons_and_decision_earliest_on():
    schema = _schema()
    assert "excluded_horizons" in schema["required"]
    assert "decision_earliest_on" in schema["required"]
    assert "horizon_days" in schema["properties"]["arms"]["additionalProperties"]["required"]
    assert "metric" in schema["properties"]["arms"]["additionalProperties"]["required"]
