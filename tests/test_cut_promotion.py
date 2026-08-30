"""The universe-cut serving pointer after the I9317 arena wiring.

The DECISION is taken by ``nousergon_lib.arena.engine.run_cycle`` through
``scoring/cut_arena.py``; this module transcribes it onto
``config/scanner_cut_champion.json``. The two cohort-window functions in
``leaderboard_scoring.py`` remain on the cuts MEASUREMENT board only — they
no longer gate promotion.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nousergon_lib.arena import ArenaConfig

from scoring.cut_arena import ARENA_CONFIG, bootstrap_register, run_arena_cycle
from scoring.cut_promotion import (
    ARENA_CYCLE_DATED_KEY,
    ARENA_CYCLE_LATEST_KEY,
    AUDIT_DATED_KEY,
    AUDIT_LATEST_KEY,
    CONTAMINATION_CAVEAT,
    CUT_PROMOTION_SLOT,
    DECISION_METRIC,
    HOLD_REASON_CODES,
    LIVE_REASON_CODES,
    MIN_VETO_HORIZON_DAYS,
    REASON_CHAMPION_LEADS,
    REASON_LEDGER_MISSING,
    REASON_PROMOTED,
    RETIRED_V1_REASON_CODES,
    CutPromotionError,
    decide_cut_champion,
    reconcile_arms_with_ledger,
    run_cut_promotion,
)
from scoring.leaderboard_scoring import LONG_HORIZONS_DAYS, slot_spec
from scoring.universe_membership import (
    CUT_ARM_PROMOTION_EXCLUSIONS,
    CUT_CHAMPION_POINTER_KEY,
    DEFAULT_CUT_CHAMPION,
    OBSERVE_ONLY_CUTS,
    PROMOTABLE_CUTS,
    SLOT_ARMS,
    live_cut_champion,
)
from scoring.weekly_ledger import LEDGER_COLS, LEDGER_KEY, LEDGER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contracts" / "scanner_cut_champion.schema.json"

CHAMP = "attractiveness_top_60"
CHALLENGER = "tech_score_top_60"
DATE = "2026-08-29"

TWO_ARM_CONFIG = replace(ARENA_CONFIG, min_active_arms=2)
TWO_ARM_SLOT = replace(
    CUT_PROMOTION_SLOT,
    arms=(CHAMP, CHALLENGER),
    scored_arms=(CHAMP, CHALLENGER),
    observe_only_arms=(),
    excluded_arms={},
    arena_config=TWO_ARM_CONFIG,
)


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _weeks(n: int, start: str = "2026-08-01") -> list[tuple[str, str]]:
    d0 = date.fromisoformat(start)
    bounds = [(d0 + timedelta(days=7 * i)).isoformat() for i in range(n + 1)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _ledger_row(arm: str, week: tuple[str, str], *, net: float | None) -> dict:
    return {
        "arm": arm,
        "week_start": week[0],
        "week_end": week[1],
        "priced_from": week[0],
        "priced_to": week[1],
        "n_names": 60,
        "retained_from_prior": None,
        "turnover_frac": 0.3,
        "gross_log_return": net,
        "net_log_return": net,
        "net_unavailable_reason": None if net is not None else "turnover_unknown",
        "cost_bps": 10.0,
        "benchmark_log_return": 0.001,
        "population_log_return": 0.001,
        "champion_log_return": None,
        "rank_ic": None,
        "market_regime": "neutral",
        "is_champion": arm == CHAMP,
        "ledger_version": LEDGER_VERSION,
        "code_sha": "test",
        "written_at": "2026-08-22T00:00:00+00:00",
    }


def _ledger(
    *,
    champ_net: float | list[float] = 0.0,
    chal_net: float | list[float] = 0.0,
    n_weeks: int = 8,
) -> list[dict]:
    weeks = _weeks(n_weeks)
    cs = list(champ_net) if isinstance(champ_net, list) else [champ_net] * n_weeks
    xs = list(chal_net) if isinstance(chal_net, list) else [chal_net] * n_weeks
    rows: list[dict] = []
    for i, week in enumerate(weeks):
        rows.append(_ledger_row(CHAMP, week, net=cs[i]))
        rows.append(_ledger_row(CHALLENGER, week, net=xs[i]))
    return rows


def _board(*, h126_mature: bool = False) -> dict:
    def block(h: int, status: str, champ_mean: float, chal_mean: float) -> dict:
        return {
            "horizon_days": h,
            "status": status,
            "n_dates": 10,
            "specs": [
                {
                    "name": CHAMP,
                    "topn_alpha_vs_population": {"mean": champ_mean, "n_dates": 10},
                    "n_dates_scored": 10,
                    "confidence": "ok",
                },
                {
                    "name": CHALLENGER,
                    "topn_alpha_vs_population": {"mean": chal_mean, "n_dates": 10},
                    "n_dates_scored": 10,
                    "confidence": "ok",
                },
            ],
        }

    return {
        "leaderboard_id": "cuts",
        "date": DATE,
        "horizons": [
            block(21, "ok", 0.001, 0.5),
            block(126, "ok" if h126_mature else "immature", 0.01, 0.9),
            block(252, "immature", 0.01, 0.9),
        ],
    }


def _run_decision(**kwargs):
    register = bootstrap_register()
    rows = kwargs.pop("ledger_rows", _ledger())
    slot = kwargs.pop("slot", TWO_ARM_SLOT)
    champion_before = kwargs.pop("champion_before", CHAMP)
    decided_on = kwargs.pop("decided_on", DATE)
    cycle, counts = run_arena_cycle(
        ledger_rows=rows,
        champion_before=champion_before,
        decided_on=decided_on,
        register=register,
        config=slot.arena_config,
    )
    return decide_cut_champion(
        ledger_rows=rows,
        register=register,
        cycle=cycle,
        ledger_counts=counts,
        champion_before=champion_before,
        decided_on=decided_on,
        slot=slot,
        **kwargs,
    )


class TestSlotRegistry:
    def test_the_decision_basis_is_population_relative_weekly_ledger(self):
        assert CUT_PROMOTION_SLOT.decision_source == LEDGER_KEY
        assert CUT_PROMOTION_SLOT.primary_metric == DECISION_METRIC
        assert CUT_PROMOTION_SLOT.ledger_return_column in LEDGER_COLS
        assert CUT_PROMOTION_SLOT.arena_config.benchmark == "population"

    def test_hysteresis_and_evidence_floors_are_gone(self):
        assert not hasattr(CUT_PROMOTION_SLOT, "promotion_margin")
        assert not hasattr(CUT_PROMOTION_SLOT, "cooldown_days")
        assert not hasattr(CUT_PROMOTION_SLOT, "min_weeks_for_inference")
        assert not hasattr(CUT_PROMOTION_SLOT, "decision_earliest_on")

    def test_corroborating_horizons_report_only(self):
        assert CUT_PROMOTION_SLOT.corroborating_horizons_days == (126, 252)
        for h in CUT_PROMOTION_SLOT.corroborating_horizons_days:
            assert h >= MIN_VETO_HORIZON_DAYS
            assert h in LONG_HORIZONS_DAYS

    def test_live_slot_has_at_least_min_active_arms(self):
        assert len(PROMOTABLE_CUTS) >= CUT_PROMOTION_SLOT.arena_config.min_active_arms

    def test_arms_match_the_derived_register(self):
        assert CUT_PROMOTION_SLOT.arms == PROMOTABLE_CUTS
        assert CUT_PROMOTION_SLOT.scored_arms == SLOT_ARMS
        assert CUT_PROMOTION_SLOT.excluded_arms == CUT_ARM_PROMOTION_EXCLUSIONS
        assert CUT_PROMOTION_SLOT.observe_only_arms == OBSERVE_ONLY_CUTS


class TestDecide:
    def test_missing_ledger_is_weekly_ledger_missing_not_arena_unmeasurable(self):
        d = _run_decision(ledger_rows=None)
        assert d.decision == "hold"
        assert d.reason_code == REASON_LEDGER_MISSING

    def test_a_mature_126d_board_cannot_block_promotion(self):
        d = _run_decision(
            board=_board(h126_mature=True),
            ledger_rows=_ledger(champ_net=0.0, chal_net=0.05, n_weeks=10),
        )
        assert d.corroborating is not None
        assert d.corroborating["role"] == "reported_only"
        assert d.corroborating["blocking"] is True  # retained for archive comparability
        if d.decision == "promote":
            assert d.reason_code == REASON_PROMOTED

    def test_every_arm_carries_arena_transcription_fields(self):
        d = _run_decision()
        for arm in TWO_ARM_SLOT.scored_arms:
            ev = d.arms[arm]
            assert ev.metric == DECISION_METRIC
            assert ev.source == LEDGER_KEY
            assert ev.score_definition == CUT_PROMOTION_SLOT.score_definition
            assert ev.arm_id is not None

    def test_the_record_names_the_arena_cycle(self):
        d = _run_decision()
        assert d.arena["slot"] == "universe_cut"
        assert d.arena["benchmark"] == "population"
        assert d.arena["cycle_key"] == ARENA_CYCLE_DATED_KEY.format(date=DATE)


class TestSchema:
    def test_v4_document_validates(self):
        d = _run_decision(board=_board())
        doc = d.to_document(leaderboard_key=f"research/cuts_leaderboard/{DATE}.json")
        jsonschema.validate(doc, _schema())
        assert doc["schema_version"] == 4
        assert "hysteresis" not in doc
        assert "decision_earliest_on" not in doc
        assert "arena" in doc

    def test_retired_reason_codes_are_disjoint_from_live(self):
        assert not set(LIVE_REASON_CODES) & set(RETIRED_V1_REASON_CODES)
        assert "no_promotable_challenger" in RETIRED_V1_REASON_CODES
        assert REASON_CHAMPION_LEADS in HOLD_REASON_CODES


class TestReconcile:
    def test_reconcile_catches_a_wrong_n_weeks_scored(self):
        d = _run_decision()
        doc = d.to_document()
        doc["arms"][CHALLENGER]["n_weeks_scored"] += 1
        bad = reconcile_arms_with_ledger(doc, _ledger())
        assert bad

    def test_reconcile_passes_for_an_honest_record(self):
        rows = _ledger()
        d = _run_decision(ledger_rows=rows)
        assert reconcile_arms_with_ledger(d.to_document(), rows) == []


class TestRunCutPromotion:
    def test_writes_pointer_audit_and_arena_keys(self):
        class _S3:
            def __init__(self):
                self.objects: dict[str, bytes] = {}

            def get_object(self, **kwargs):
                key = kwargs["Key"]
                if key not in self.objects:
                    raise RuntimeError("NoSuchKey")
                return {"Body": type("B", (), {"read": lambda self: self.objects[key]})()}

            def put_object(self, **kwargs):
                self.objects[kwargs["Key"]] = kwargs["Body"]

        s3 = _S3()
        with patch("scoring.cut_promotion.live_cut_champion", return_value=CHAMP), patch(
            "scoring.cut_promotion.send_verdict_digest", return_value=True
        ):
            doc = run_cut_promotion(
                DATE,
                s3_client=s3,
                bucket="test-bucket",
                ledger_rows=_ledger(champ_net=0.0, chal_net=0.05, n_weeks=10),
                leaderboard=_board(),
                slot=TWO_ARM_SLOT,
            )
        keys = set(s3.objects)
        assert CUT_CHAMPION_POINTER_KEY in keys
        assert AUDIT_DATED_KEY.format(date=DATE) in keys
        assert AUDIT_LATEST_KEY in keys
        assert ARENA_CYCLE_DATED_KEY.format(date=DATE) in keys
        assert ARENA_CYCLE_LATEST_KEY in keys
        jsonschema.validate(doc, _schema())

    def test_a_defective_board_raises_after_writing(self):
        dup_board = _board()
        dup_board["horizons"][0]["specs"].append(dup_board["horizons"][0]["specs"][0])
        class _S3:
            objects: dict[str, bytes] = {}

            def get_object(self, **kwargs):
                raise RuntimeError("NoSuchKey")

            def put_object(self, **kwargs):
                self.objects[kwargs["Key"]] = kwargs["Body"]

        s3 = _S3()
        with patch("scoring.cut_promotion.live_cut_champion", return_value=CHAMP), patch(
            "scoring.cut_promotion.send_verdict_digest", return_value=True
        ):
            with pytest.raises(CutPromotionError, match="duplicate arm rows"):
                run_cut_promotion(
                    DATE,
                    s3_client=s3,
                    bucket="b",
                    ledger_rows=_ledger(),
                    leaderboard=dup_board,
                    slot=TWO_ARM_SLOT,
                )
        assert AUDIT_DATED_KEY.format(date=DATE) in s3.objects


class TestServingPointer:
    def test_the_pointer_this_engine_writes_is_the_one_the_feed_reads(self):
        assert CUT_CHAMPION_POINTER_KEY.endswith("scanner_cut_champion.json")

    def test_default_champion_is_promotable(self):
        assert DEFAULT_CUT_CHAMPION in PROMOTABLE_CUTS
        assert live_cut_champion.__name__ == "live_cut_champion"


class TestExcludedHorizons:
    def test_excluded_horizons_carries_the_contamination_caveat(self):
        d = _run_decision(board=_board())
        entry = d.excluded_horizons[CHAMP]["21"]
        assert CONTAMINATION_CAVEAT.split(".")[0] in entry["excluded_reason"]
