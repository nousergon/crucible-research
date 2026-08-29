"""The scanner SPEC slot's promotion engine (alpha-engine-config-I9273).

The spec slot — ``data/scanner_specs.py::SCANNER_SPECS`` — had a leaderboard
that RANKED and nothing that DECIDED, so its champion moved only by a human
editing ``LIVE_CHAMPION``. That is exactly how the four-week vacuous-leaderboard
defect (alpha-engine-config-I7808) happened: a hand-moved pointer with no record
of the move.

RED VERIFICATION (champion-challenger-policy.md §7.4 — a guard is verified to
fail without the fix, or it is coverage that reads as coverage and is not).
Two independent runs, both recorded in the PR body:

1. Against ``origin/main`` (``git stash`` of the four changed source files),
   the whole module fails at collection: ``scoring/spec_promotion.py`` does not
   exist, ``data/scanner_specs.py`` exports no pointer resolver, and
   ``contracts/scanner_spec_champion.schema.json`` is absent. 0 passed.
2. Against MUTANTS of the post-change module, each applied ALONE and reverted.
   Nine were killed by the guard they target:

   * ``_arm_evidence`` skipping the ``n_paired == 0`` branch (an arm with no
     common cohort read as eligible) → ``test_an_empty_cohort_intersection_
     holds_rather_than_promoting`` red;
   * ``_arm_evidence`` skipping the ``n_scored == 0`` branch →
     ``test_every_arm_is_named_with_its_own_count_including_zeros`` red;
   * ``_arm_evidence`` skipping the thin-cohort branch →
     ``test_a_thin_intersection_holds_and_says_so_distinctly`` red;
   * ``_leader`` losing its ``best_mean <= 0.0`` guard →
     ``test_a_measured_loss_is_not_the_same_record_as_an_unmeasured_arm`` AND
     ``test_an_exact_tie_resolves_to_the_incumbent`` both red;
   * ``no_common_cohort`` collapsed into ``no_eligible_challenger`` →
     ``test_an_empty_cohort_intersection_holds_rather_than_promoting`` red;
   * the demotion branch forced to ``promote`` →
     ``test_demotion_is_symmetric_with_promotion`` red;
   * the margin skipped on the demotion branch →
     ``test_demotion_obeys_the_same_margin_and_cooldown`` red;
   * the cooldown skipped → ``test_promotion_obeys_the_margin_and_the_cooldown``
     red;
   * the champion-row check and the duplicate-row check each dropped →
     ``test_a_record_is_written_on_every_path`` red on that parameter.

   THREE mutants SURVIVED, and that is recorded rather than tidied away,
   because which guard actually defends which test is a fact about this suite:

   * ``_leader`` ranking on ``n_dates_scored`` instead of the paired mean
     survives ``test_a_seven_vs_zero_board_is_not_a_champion_win`` and
     ``test_a_measured_loss_is_not_the_same_record_as_an_unmeasured_arm``. The
     first is defended by the ELIGIBILITY gate (no challenger is eligible, so
     ``_leader`` is never reached); the second by the tie-break to the
     incumbent. ``_leader``'s ranking is defended by the mutant above it.
   * skipping the ``n_scored == 0`` branch survives
     ``test_a_seven_vs_zero_board_is_not_a_champion_win`` — the arm falls
     through to the ``n_paired == 0`` branch and is still ineligible, so the
     decision is unchanged. Two independent conditions have to fail together
     before a 7-vs-0 board could read as a champion win, which is the intent.

   A guard that survives every mutation of the thing it claims to check is
   blind; a mutant that survives because a DIFFERENT guard caught it is
   defence in depth, and the two are only distinguishable if both are written
   down.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.scanner_specs import (  # noqa: E402
    DEFAULT_SPEC_CHAMPION,
    PROMOTABLE_SPECS,
    SCANNER_SPECS,
    SPEC_CHAMPION_POINTER_KEY,
    ScannerSpecPointerError,
    assert_register_coherent,
    live_champion_name,
    live_champion_spec,
    register_for_champion,
)
from scoring.leaderboard_scoring import slot_spec  # noqa: E402
from scoring.spec_promotion import (  # noqa: E402
    AUDIT_DATED_KEY,
    AUDIT_LATEST_KEY,
    DECISION_DEMOTE,
    DECISION_HOLD,
    DECISION_PROMOTE,
    HOLD_REASON_CODES,
    INELIGIBLE_IS_CHAMPION,
    INELIGIBLE_NO_COMMON_COHORT,
    INELIGIBLE_NO_SCORED_COHORT,
    INELIGIBLE_REASON_CODES,
    INELIGIBLE_THIN_COHORT,
    PROMOTION_MARGIN_NOTE,
    REASON_BOARD_DEFECTIVE,
    REASON_CHAMPION_LEADS,
    REASON_CHAMPION_ROW_MISSING,
    REASON_COOLDOWN_ACTIVE,
    REASON_DEMOTED,
    REASON_HORIZON_MISSING,
    REASON_INSUFFICIENT_DATES,
    REASON_LEADERBOARD_MISSING,
    REASON_LEADERBOARD_UNMEASURABLE,
    REASON_MARGIN_NOT_MET,
    REASON_NO_COMMON_COHORT,
    REASON_NO_ELIGIBLE_CHALLENGER,
    REASON_NO_REGISTERED_CHALLENGER,
    REASON_PROMOTED,
    SPEC_PROMOTION_SLOT,
    SpecPromotionError,
    decide_spec_champion,
    decision_earliest_on,
    reconcile_arms_with_board,
    run_spec_promotion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contracts" / "scanner_spec_champion.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())

CHAMP = "momentum_sleeve"
TECH = "tech_score_gate"
MOM121 = "mom_12_1_sleeve"

HORIZON = SPEC_PROMOTION_SLOT.horizon_days
MIN_DATES = SPEC_PROMOTION_SLOT.min_dates_for_inference
MARGIN = SPEC_PROMOTION_SLOT.promotion_margin


# ── board fixtures ────────────────────────────────────────────────────────────


def _stats(mean, n, t=None):
    return {"mean": mean, "se": 0.001, "t_stat": t if t is not None else 3.0,
            "n_dates": n, "se_method": "clustered_iid"}


def _row(name, kind, *, n_scored, vs_pop=None, vs_champ=None):
    return {
        "name": name,
        "kind": kind,
        "realized_rank_ic": None,
        "topn_alpha_vs_champion": vs_champ,
        "topn_alpha_vs_benchmark": None,
        "topn_alpha_vs_population": vs_pop,
        "n_dates_scored": n_scored,
        "confidence": "ok" if n_scored >= MIN_DATES else ("thin" if n_scored else "insufficient"),
    }


def _board(rows, *, status="ok", n_dates=7, champion=CHAMP):
    block = {"horizon_days": HORIZON, "status": status, "reason": None,
             "n_dates": n_dates, "overlap_lags": 0, "observations_overlap": False,
             "cohort_spacing_days": 1, "specs": rows}
    return {
        "leaderboard_id": "scanner",
        "champion": champion,
        "horizon_days": HORIZON,
        "top_n": slot_spec("scanner").top_n,
        "n_dates": n_dates,
        "specs": rows,
        "horizons_days": list(slot_spec("scanner").horizons_days),
        "horizons": [block],
        "min_dates_for_inference": MIN_DATES,
    }


# The board measured live on scanner/leaderboard/2026-08-28.json, 21-session
# block: the champion scored on 7 dates, BOTH challengers on zero. The cohort
# intersection is therefore empty and "which spec should have been promoted on
# 2026-08-28" is unanswerable.
BOARD_2026_08_28 = _board([
    _row(CHAMP, "champion", n_scored=7, vs_pop=_stats(0.031348, 7, 11.41)),
    _row(TECH, "challenger", n_scored=0),
    _row(MOM121, "challenger", n_scored=0),
])


def _winning_board(arm=TECH, *, mean=None, n=None):
    return _board([
        _row(CHAMP, "champion", n_scored=10, vs_pop=_stats(0.03, 10)),
        _row(arm, "challenger", n_scored=n or MIN_DATES,
             vs_pop=_stats(0.05, n or MIN_DATES),
             vs_champ=_stats(MARGIN * 10 if mean is None else mean, n or MIN_DATES)),
        _row(MOM121 if arm != MOM121 else TECH, "challenger", n_scored=0),
    ])


# ── §10 the slot registry names all six per-slot facts ───────────────────────


def test_slot_registry_names_metric_horizon_benchmark_width_floor_and_hysteresis():
    s = SPEC_PROMOTION_SLOT
    assert s.primary_metric
    assert s.horizon_days > 0
    assert s.benchmark_ticker
    assert s.count_matched_width > 0
    assert s.min_dates_for_inference >= 1
    assert s.promotion_margin > 0
    assert s.cooldown_days > 0
    assert PROMOTION_MARGIN_NOTE


def test_the_spec_slot_does_not_reuse_the_cut_slots_registry():
    """champion-challenger-policy.md §2 — separate axes, never conflated."""
    src = (REPO_ROOT / "scoring" / "spec_promotion.py").read_text()
    assert "CUT_PROMOTION_SLOT" not in src
    assert "PROMOTABLE_CUTS" not in src
    assert SPEC_PROMOTION_SLOT.slot_id == "scanner_spec"


def test_every_registered_arm_is_promotion_eligible():
    """Brian's ruling 2026-08-29: an arm that is SCORED must be able to WIN.

    Eligibility is a declared property carrying a reason, never an absence
    from a list — so the declared-ineligible map is what a reader consults,
    and it is empty.
    """
    assert set(PROMOTABLE_SPECS) == set(SCANNER_SPECS)
    assert SPEC_PROMOTION_SLOT.declared_ineligible_arms == {}


def test_hold_reason_codes_are_a_closed_disjoint_taxonomy():
    assert len(set(HOLD_REASON_CODES)) == len(HOLD_REASON_CODES)
    assert REASON_PROMOTED not in HOLD_REASON_CODES
    assert REASON_DEMOTED not in HOLD_REASON_CODES
    assert not set(HOLD_REASON_CODES) & set(INELIGIBLE_REASON_CODES)
    enum = SCHEMA["properties"]["reason_code"]["enum"]
    assert set(enum) == set(HOLD_REASON_CODES) | {REASON_PROMOTED, REASON_DEMOTED}


# ── the measured board: 7-vs-0 is NOT a champion win ─────────────────────────


def test_a_seven_vs_zero_board_is_not_a_champion_win():
    d = decide_spec_champion(
        board=BOARD_2026_08_28, champion_before=CHAMP, decided_on="2026-08-28",
    )
    assert d.decision == DECISION_HOLD
    assert d.reason_code == REASON_NO_ELIGIBLE_CHALLENGER
    # NOT champion_already_leads — that is a claim about a comparison that
    # never happened.
    assert d.reason_code != REASON_CHAMPION_LEADS


def test_every_arm_is_named_with_its_own_count_including_zeros():
    doc = decide_spec_champion(
        board=BOARD_2026_08_28, champion_before=CHAMP, decided_on="2026-08-28",
    ).to_document()
    assert set(doc["arms"]) == set(SCANNER_SPECS)
    assert doc["arms"][CHAMP]["n_dates_scored"] == 7
    for arm in (TECH, MOM121):
        assert doc["arms"][arm]["n_dates_scored"] == 0
        # An arm with zero scored cohort dates cannot be promoted, and that is
        # recorded as an explicit property with a machine-readable slug — never
        # as an absence from the block.
        assert doc["arms"][arm]["eligible_for_promotion"] is False
        assert doc["arms"][arm]["ineligible_reason_code"] == INELIGIBLE_NO_SCORED_COHORT
        # No alpha-vs-champion is reported for an arm with no overlapping dates.
        assert doc["arms"][arm]["mean_paired_alpha_vs_champion"] is None
        assert doc["arms"][arm]["n_dates_paired"] == 0


def test_the_champion_is_eligible_but_never_promoted_to_itself():
    doc = decide_spec_champion(
        board=BOARD_2026_08_28, champion_before=CHAMP, decided_on="2026-08-28",
    ).to_document()
    assert doc["arms"][CHAMP]["is_champion"] is True
    assert doc["arms"][CHAMP]["ineligible_reason_code"] == INELIGIBLE_IS_CHAMPION
    assert doc["arms"][CHAMP]["mean_paired_alpha_vs_champion"] is None


# ── an empty cohort intersection is not a win ────────────────────────────────


def test_an_empty_cohort_intersection_holds_rather_than_promoting():
    """The challenger HAS scored dates; none of them overlap the champion's.

    ``topn_alpha_vs_champion`` is therefore null even though the arm looks
    measured, and a decision taken on ``topn_alpha_vs_population`` alone would
    compare one arm's month to another's quarter (§4, same cohort dates).
    """
    board = _board([
        _row(CHAMP, "champion", n_scored=7, vs_pop=_stats(0.01, 7)),
        _row(TECH, "challenger", n_scored=6, vs_pop=_stats(0.90, 6), vs_champ=None),
        _row(MOM121, "challenger", n_scored=0),
    ])
    d = decide_spec_champion(board=board, champion_before=CHAMP, decided_on="2026-08-28")
    assert d.decision == DECISION_HOLD
    assert d.reason_code == REASON_NO_COMMON_COHORT
    doc = d.to_document()
    assert doc["arms"][TECH]["ineligible_reason_code"] == INELIGIBLE_NO_COMMON_COHORT
    assert doc["arms"][TECH]["n_dates_paired"] == 0
    assert doc["arms"][TECH]["mean_paired_alpha_vs_champion"] is None
    # the 0.90 population number is still REPORTED — it is just not a comparison
    assert doc["arms"][TECH]["alpha_vs_population"]["mean"] == 0.90


def test_a_thin_intersection_holds_and_says_so_distinctly():
    board = _board([
        _row(CHAMP, "champion", n_scored=7, vs_pop=_stats(0.01, 7)),
        _row(TECH, "challenger", n_scored=6, vs_pop=_stats(0.9, 6),
             vs_champ=_stats(0.5, MIN_DATES - 1)),
        _row(MOM121, "challenger", n_scored=0),
    ])
    d = decide_spec_champion(board=board, champion_before=CHAMP, decided_on="2026-08-28")
    assert d.decision == DECISION_HOLD
    assert d.reason_code == REASON_INSUFFICIENT_DATES
    doc = d.to_document()
    assert doc["arms"][TECH]["ineligible_reason_code"] == INELIGIBLE_THIN_COHORT
    assert doc["arms"][TECH]["n_dates_paired"] == MIN_DATES - 1


def test_a_measured_loss_is_not_the_same_record_as_an_unmeasured_arm():
    board = _board([
        _row(CHAMP, "champion", n_scored=10, vs_pop=_stats(0.03, 10)),
        _row(TECH, "challenger", n_scored=10, vs_pop=_stats(0.01, 10),
             vs_champ=_stats(-0.02, 10)),
        _row(MOM121, "challenger", n_scored=0),
    ])
    d = decide_spec_champion(board=board, champion_before=CHAMP, decided_on="2026-08-28")
    assert d.decision == DECISION_HOLD
    assert d.reason_code == REASON_CHAMPION_LEADS
    doc = d.to_document()
    assert doc["arms"][TECH]["eligible_for_promotion"] is True
    assert doc["arms"][TECH]["ineligible_reason_code"] is None
    assert doc["arms"][TECH]["mean_paired_alpha_vs_champion"] == -0.02


# ── promotion ────────────────────────────────────────────────────────────────


def test_a_promotion_moves_the_pointer_and_the_live_ranking():
    s3 = _S3()
    doc = run_spec_promotion(
        "2026-09-30", bucket="b", s3_client=s3, leaderboard=_winning_board(TECH),
    )
    assert doc["decision"] == DECISION_PROMOTE
    assert doc["reason_code"] == REASON_PROMOTED
    assert doc["champion"] == TECH
    assert doc["champion_before"] == CHAMP
    assert SPEC_CHAMPION_POINTER_KEY in s3.written
    # the LIVE ranking follows the pointer, with no code edit
    assert live_champion_name(bucket="b", s3_client=s3) == TECH
    spec = live_champion_spec(bucket="b", s3_client=s3)
    assert spec.name == TECH
    assert spec.kind == "champion"
    assert spec.rank is SCANNER_SPECS[TECH].rank


def test_a_promotion_constructs_a_coherent_register_not_a_mutated_field():
    reg = register_for_champion(TECH)
    champions = [s for s in reg.values() if s.kind == "champion"]
    assert [c.name for c in champions] == [TECH]
    assert reg[CHAMP].kind == "challenger"
    assert set(reg) == set(SCANNER_SPECS)
    assert_register_coherent(reg, TECH)
    # the module-level register is untouched
    assert SCANNER_SPECS[CHAMP].kind == "champion"


def test_a_pointer_naming_an_unknown_arm_raises_rather_than_serving_the_default():
    s3 = _S3({SPEC_CHAMPION_POINTER_KEY: {"champion": "not_an_arm"}})
    with pytest.raises(ScannerSpecPointerError):
        live_champion_name(bucket="b", s3_client=s3)


def test_an_absent_pointer_serves_the_register_default():
    assert live_champion_name(bucket="b", s3_client=_S3()) == DEFAULT_SPEC_CHAMPION
    assert live_champion_spec(bucket="b", s3_client=_S3()) is SCANNER_SPECS[DEFAULT_SPEC_CHAMPION]


# ── demotion is symmetric ────────────────────────────────────────────────────


def test_demotion_is_symmetric_with_promotion():
    """The pointer sits on a promoted arm and the standing default retakes it.

    Same margin, opposite direction, plus the cooldown — champion-challenger-
    policy.md §5.2. It is recorded as its OWN decision value, not as a
    promotion of the default, because "the experiment was reversed" and "a new
    arm won" are different facts about the slot.
    """
    board = _board([
        _row(TECH, "champion", n_scored=10, vs_pop=_stats(0.01, 10)),
        _row(CHAMP, "challenger", n_scored=10, vs_pop=_stats(0.05, 10),
             vs_champ=_stats(MARGIN * 10, 10)),
        _row(MOM121, "challenger", n_scored=0),
    ], champion=TECH)
    d = decide_spec_champion(board=board, champion_before=TECH, decided_on="2026-09-30")
    assert d.decision == DECISION_DEMOTE
    assert d.reason_code == REASON_DEMOTED
    assert d.champion == DEFAULT_SPEC_CHAMPION == CHAMP


def test_demotion_obeys_the_same_margin_and_cooldown():
    def board(mean):
        return _board([
            _row(TECH, "champion", n_scored=10, vs_pop=_stats(0.01, 10)),
            _row(CHAMP, "challenger", n_scored=10, vs_pop=_stats(0.05, 10),
                 vs_champ=_stats(mean, 10)),
            _row(MOM121, "challenger", n_scored=0),
        ], champion=TECH)

    under = decide_spec_champion(
        board=board(MARGIN / 2), champion_before=TECH, decided_on="2026-09-30")
    assert under.decision == DECISION_HOLD
    assert under.reason_code == REASON_MARGIN_NOT_MET

    cooling = decide_spec_champion(
        board=board(MARGIN * 10), champion_before=TECH, decided_on="2026-09-30",
        last_promoted_on="2026-09-20")
    assert cooling.decision == DECISION_HOLD
    assert cooling.reason_code == REASON_COOLDOWN_ACTIVE


def test_promotion_obeys_the_margin_and_the_cooldown():
    under = decide_spec_champion(
        board=_winning_board(TECH, mean=MARGIN / 2), champion_before=CHAMP,
        decided_on="2026-09-30")
    assert under.reason_code == REASON_MARGIN_NOT_MET
    cooling = decide_spec_champion(
        board=_winning_board(TECH), champion_before=CHAMP, decided_on="2026-09-30",
        last_promoted_on="2026-09-20")
    assert cooling.reason_code == REASON_COOLDOWN_ACTIVE


def test_an_exact_tie_resolves_to_the_incumbent():
    d = decide_spec_champion(
        board=_winning_board(TECH, mean=0.0), champion_before=CHAMP,
        decided_on="2026-09-30")
    assert d.decision == DECISION_HOLD
    assert d.champion == CHAMP
    assert d.reason_code == REASON_CHAMPION_LEADS


# ── a record on EVERY path ───────────────────────────────────────────────────


class _Body:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


class _S3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.written: dict[str, dict] = {}
        self.write_order: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key in self.written:
            return {"Body": _Body(json.dumps(self.written[Key]).encode())}
        if Key in self.objects:
            return {"Body": _Body(json.dumps(self.objects[Key]).encode())}
        raise RuntimeError(f"NoSuchKey: {Key}")

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803
        self.written[Key] = json.loads(Body)
        self.write_order.append(Key)


_DUPES = _board([
    _row(CHAMP, "champion", n_scored=7, vs_pop=_stats(0.03, 7)),
    _row(CHAMP, "champion", n_scored=7, vs_pop=_stats(0.03, 7)),
    _row(TECH, "challenger", n_scored=0),
    _row(MOM121, "challenger", n_scored=0),
])

_UNMEASURABLE = _board([
    _row(CHAMP, "champion", n_scored=0),
    _row(TECH, "challenger", n_scored=0),
    _row(MOM121, "challenger", n_scored=0),
], status="unmeasurable", n_dates=0)

_NO_CHAMPION_ROW = _board([
    _row(TECH, "challenger", n_scored=6, vs_pop=_stats(0.05, 6)),
    _row(MOM121, "challenger", n_scored=6, vs_pop=_stats(0.05, 6)),
])


@pytest.mark.parametrize(
    ("board", "expected", "raises"),
    [
        (BOARD_2026_08_28, REASON_NO_ELIGIBLE_CHALLENGER, False),
        (None, REASON_LEADERBOARD_MISSING, False),
        (_UNMEASURABLE, REASON_LEADERBOARD_UNMEASURABLE, False),
        (_NO_CHAMPION_ROW, REASON_CHAMPION_ROW_MISSING, False),
        (_DUPES, REASON_BOARD_DEFECTIVE, True),
        (_winning_board(TECH), REASON_PROMOTED, False),
    ],
)
def test_a_record_is_written_on_every_path(board, expected, raises):
    s3 = _S3()
    if raises:
        with pytest.raises(SpecPromotionError):
            run_spec_promotion("2026-09-30", bucket="b", s3_client=s3, leaderboard=board)
    else:
        run_spec_promotion("2026-09-30", bucket="b", s3_client=s3, leaderboard=board)
    dated = AUDIT_DATED_KEY.format(date="2026-09-30")
    assert set(s3.written) == {dated, AUDIT_LATEST_KEY, SPEC_CHAMPION_POINTER_KEY}
    # the immutable dated record lands FIRST: dying mid-write must never leave a
    # moved pointer with no record of why
    assert s3.write_order[0] == dated
    doc = s3.written[dated]
    assert doc["reason_code"] == expected
    assert doc["reason"]
    assert set(doc["arms"]) == set(SCANNER_SPECS)
    jsonschema.validate(doc, SCHEMA)


def test_a_missing_horizon_block_is_its_own_reason_code():
    board = dict(_winning_board(TECH))
    board["horizons"] = []
    d = decide_spec_champion(board=board, champion_before=CHAMP, decided_on="2026-09-30")
    assert d.reason_code == REASON_HORIZON_MISSING


def test_a_slot_with_one_arm_says_so_rather_than_claiming_the_champion_led():
    slot = replace(SPEC_PROMOTION_SLOT, arms=(CHAMP,))
    d = decide_spec_champion(
        board=BOARD_2026_08_28, champion_before=CHAMP, decided_on="2026-08-28", slot=slot)
    assert d.reason_code == REASON_NO_REGISTERED_CHALLENGER


def test_the_record_reconciles_against_the_board_it_cites():
    doc = decide_spec_champion(
        board=_winning_board(TECH), champion_before=CHAMP, decided_on="2026-09-30",
    ).to_document()
    assert reconcile_arms_with_board(doc, _winning_board(TECH)) == []
    tampered = json.loads(json.dumps(doc))
    tampered["arms"][TECH]["n_dates_paired"] = 99
    assert reconcile_arms_with_board(tampered, _winning_board(TECH))


def test_decision_earliest_on_is_a_ceiling_on_the_forward_window():
    assert decision_earliest_on() > SPEC_PROMOTION_SLOT.first_cohort_date.isoformat()


# ── wiring ───────────────────────────────────────────────────────────────────


def test_the_engine_is_invoked_from_the_scanner_handler():
    src = (REPO_ROOT / "lambda" / "scanner_handler.py").read_text()
    assert "run_spec_promotion" in src
    assert src.index("build_scanner_leaderboard") < src.index("run_spec_promotion")


def test_the_contract_documents_the_engine():
    text = (REPO_ROOT / "SCANNER_CONTRACT.md").read_text()
    assert "config/scanner_spec_champion.json" in text
    assert "scoring/spec_promotion.py" in text
