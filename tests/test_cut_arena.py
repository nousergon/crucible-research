"""Guards for the universe-cut slot's arena wiring (alpha-engine-config-I9317).

The score ladder, longest-common-window pairing, confidence sequence, Copeland
ranking, pointer rule and retirement rule live in ``nousergon_lib.arena`` and
are tested there. This file guards the four things the engine cannot know:
the register, the population-relative series, serving preconditions, and the
``arena_cycle`` artifact shape.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from nousergon_lib.arena import ArenaConfig, ArmRegister
from nousergon_lib.contracts import validate as validate_contract

from scoring import cut_arena
from scoring.universe_membership import CUT_SLOT_ARM_PREFIXES, PROMOTABLE_CUTS, SLOT_ARMS
from scoring.weekly_ledger import LEDGER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTER_PATH = REPO_ROOT / "scoring" / "arena" / "universe_cut_register.json"

CHAMP = "attractiveness_top_60"
CHAL = "tech_score_top_60"


def _weeks(n: int, start: str = "2026-08-01") -> list[tuple[str, str]]:
    d0 = date.fromisoformat(start)
    bounds = [(d0 + timedelta(days=7 * i)).isoformat() for i in range(n + 1)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _row(arm: str, week: tuple[str, str], *, net: float | None, pop: float = 0.001) -> dict:
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
        "net_unavailable_reason": None,
        "cost_bps": 10.0,
        "benchmark_log_return": 0.001,
        "population_log_return": pop,
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
    chal_net: float | list[float] = 0.01,
    n_weeks: int = 6,
    arms: tuple[str, ...] = (CHAMP, CHAL),
) -> list[dict]:
    weeks = _weeks(n_weeks)
    champ_series = list(champ_net) if isinstance(champ_net, list) else [champ_net] * n_weeks
    chal_series = list(chal_net) if isinstance(chal_net, list) else [chal_net] * n_weeks
    rows: list[dict] = []
    for i, week in enumerate(weeks):
        if CHAMP in arms:
            rows.append(_row(CHAMP, week, net=float(champ_series[i])))
        if CHAL in arms:
            net = chal_series[i]
            rows.append(_row(CHAL, week, net=(float(net) if net is not None else None)))
    return rows


class TestArenaConfig:
    def test_the_engine_refuses_spy_for_this_slot_kind(self):
        with pytest.raises(ValueError, match="selection-stage slot"):
            ArenaConfig(slot="universe_cut", slot_kind="universe_cut", benchmark="SPY")

    def test_the_slot_grades_against_the_population(self):
        cfg = cut_arena.ARENA_CONFIG
        assert cfg.benchmark == "population"
        assert cfg.slot_kind == "universe_cut"

    def test_brians_2026_08_29_ruling_is_the_config(self):
        cfg = cut_arena.ARENA_CONFIG
        assert cfg.cap == 5
        assert cfg.grace_weeks == 4
        assert cfg.min_active_arms == 3
        assert cfg.retire_evidence == "point"
        assert cfg.retired_trailing_cycles == 8

    def test_min_paired_dates_is_well_formedness_not_an_evidence_bar(self):
        assert cut_arena.ARENA_CONFIG.min_paired_dates == 1

    def test_no_hysteresis_gates_on_the_live_slot(self):
        from scoring.cut_promotion import CUT_PROMOTION_SLOT

        assert not hasattr(CUT_PROMOTION_SLOT, "promotion_margin")
        assert not hasattr(CUT_PROMOTION_SLOT, "min_weeks_for_inference")
        assert not hasattr(CUT_PROMOTION_SLOT, "cooldown_days")


class TestRegister:
    def test_the_committed_register_covers_every_slot_arm(self):
        events = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))
        register = ArmRegister.from_dicts(events)
        names = {register.state(a).record.name for a in register.all_arms()}
        assert names == set(SLOT_ARMS)

    def test_bootstrap_matches_the_committed_fixture(self):
        assert cut_arena.bootstrap_register().to_dicts() == json.loads(
            REGISTER_PATH.read_text(encoding="utf-8")
        )

    def test_arm_created_on_covers_every_prefix(self):
        assert set(cut_arena.ARM_CREATED_ON) == set(CUT_SLOT_ARM_PREFIXES)

    def test_a_changed_recipe_produces_a_new_arm_id(self):
        assert cut_arena.arm_id_for(CHAMP) != cut_arena.arm_id_for(CHAL)


class TestSeriesFromLedger:
    def test_scores_are_population_relative_not_pre_differenced(self):
        rows = _ledger(champ_net=0.002, chal_net=0.010, n_weeks=1)
        series, counts = cut_arena.series_from_ledger(rows)
        arm_id = cut_arena.arm_id_for(CHAL)
        week = rows[1]["week_start"]
        assert series[arm_id].scores[week] == pytest.approx(0.010 - 0.001)

    def test_null_net_is_a_miss_not_a_zero(self):
        rows = [_row(CHAMP, _weeks(1)[0], net=0.0)]
        rows.append({**_row(CHAL, _weeks(1)[0], net=0.0), "net_log_return": None})
        series, counts = cut_arena.series_from_ledger(rows)
        assert CHAL in counts
        assert counts[CHAL]["dropped_null_column"] == 1
        assert not series[cut_arena.arm_id_for(CHAL)].scores


class TestSlotFloor:
    def test_import_refuses_fewer_than_min_active_promotable_arms(self):
        assert len(PROMOTABLE_CUTS) >= cut_arena.ARENA_CONFIG.min_active_arms

    def test_assert_slot_floor_raises_when_register_is_too_small(self):
        register = cut_arena.bootstrap_register()
        # Retire until only two active arms remain.
        active = list(register.active_arms())
        for arm_id in active[2:]:
            register = register.retire(arm_id, "2026-08-29", "test shrink")
        alerts: list[str] = []

        def _alert(msg, **kwargs):
            alerts.append(msg)

        with pytest.raises(cut_arena.SlotFloorBreached):
            cut_arena.assert_slot_floor(
                register,
                config=replace(cut_arena.ARENA_CONFIG, min_active_arms=3),
                alert=_alert,
            )
        assert alerts and "arm floor" in alerts[0].lower()


class TestRunArenaCycle:
    def test_pairwise_windows_are_per_pair_not_whole_cohort_intersection(self):
        """The two leaderboard windows (relaxed promotion + strict reported)
        are retired for DECISIONS onto arena longest-common-window pairing."""
        register = cut_arena.load_register(events=cut_arena.bootstrap_register().to_dicts())
        rows = _ledger(
            champ_net=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            chal_net=[0.01, 0.01, 0.01, 0.01, None, None],
            n_weeks=6,
        )
        cycle, _ = cut_arena.run_arena_cycle(
            ledger_rows=rows,
            champion_before=CHAMP,
            decided_on="2026-08-29",
            register=register,
        )
        chal_id = cut_arena.arm_id_for(CHAL)
        comparison = next(c for c in cycle.decision.comparisons if c.challenger == chal_id)
        assert comparison.window.n_dates == 4

    def test_cycle_document_validates_against_arena_cycle_contract(self):
        register = cut_arena.load_register(events=cut_arena.bootstrap_register().to_dicts())
        rows = _ledger(n_weeks=4)
        cycle, counts = cut_arena.run_arena_cycle(
            ledger_rows=rows,
            champion_before=CHAMP,
            decided_on="2026-08-29",
            register=register,
        )
        doc = cut_arena.cycle_document(
            cycle,
            counts=counts,
            register=register,
            ledger_present=True,
        )
        validate_contract("arena_cycle", doc)
        assert doc["slot_floor"]["breached"] is False
