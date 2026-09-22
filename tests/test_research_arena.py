"""The research slot emits an arena cycle every cycle
(alpha-engine-config-I11403, phase 4 of -I11393).

§11: *"Every slot emits one `arena_cycle` artifact per cycle... whatever the
outcome"* and *"a slot that emits nothing is not healthy, it is unobserved."*
Before this, `arena/research/` was absent from S3 and the standing detector
merged in `nous-ergon-ops-PR1386` returned `error/cycle_missing` against live
state — which was the §7.4 red-first evidence for this issue.

§10: a slot re-implementing §§3–6 is a defect, not a variation. The ladder, the
pairing, the confidence sequence, the Condorcet ranking and the cap-with-grace
retirement rule are all imported from `nousergon_lib.arena`. This file asserts
that the slot SUPPLIES its three facts correctly — which arms exist, what their
per-date score is, which may serve — and consumes the engine for the rest.

MEASURED LIVE 2026-09-22, the first end-to-end run against real S3:

    status unmeasurable | champion attractiveness_60 | moved False
    4 comparisons, each `unmeasurable` with a stated common-window reason
    5 retirement verdicts, all `retire: False`
    thinktank_20: 5 dates scored, 2 dropped as degraded
    the other four arms: 0 scored, `dropped_null_series` (no matured cohort yet)

That is the correct and expected first outcome for a slot whose ladders have
just restarted — and it is a RECORD, not a silence.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from scoring import research_arena as ra

_BUCKET = "alpha-engine-research"
_AS_OF = "2026-09-22"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _board(rows: list[dict], champion: str | None = None) -> dict:
    return {
        "leaderboard_id": "producer",
        "slot": "research",
        "primary_metric": "information_ratio",
        "champion": champion,
        "horizons": [{"horizon_days": 21, "specs": rows}],
        "specs": rows,
    }


def _row(name: str, by_date: dict | None, **kw) -> dict:
    row = {
        "name": name,
        "kind": "challenger",
        "topn_alpha_vs_population_by_date": by_date,
        "promotion_eligible": True,
        "ineligible_reason": None,
    }
    row.update(kw)
    return row


def _dates(n: int, start: int = 1) -> list[str]:
    return [f"2026-08-{d:02d}" for d in range(start, start + n)]


# ── The slot's declared contract (§10: the registry row CI can check) ────────


class TestTheSlotRegistryRow:
    def test_it_is_a_selection_slot_so_spy_is_refused(self):
        """The research slot's job is to beat the population it narrowed.
        Declaring the kind is what makes ArenaConfig refuse SPY — the 2026-08-17
        inversion, where SPY trailed the drawn-from population by 140bp at 21d."""
        from nousergon_lib.arena.engine import SELECTION_SLOT_KINDS

        assert ra.ARENA_SLOT_KIND in SELECTION_SLOT_KINDS
        assert ra.ARENA_CONFIG.benchmark == "population"

    def test_it_ranks_on_the_information_ratio(self):
        """The slot carries different widths on purpose, so a raw mean would
        select for concentration rather than skill."""
        from nousergon_lib.arena import STATISTIC_INFORMATION_RATIO

        assert ra.ARENA_CONFIG.promote_statistic == STATISTIC_INFORMATION_RATIO
        # Forced, not chosen: the sequence is a bound on a mean of bounded
        # differences and says nothing about a difference of two ratios.
        assert ra.ARENA_CONFIG.promote_evidence == "point"

    def test_brians_retirement_rulings_are_the_config(self):
        cfg = ra.ARENA_CONFIG
        assert (cfg.cap, cfg.grace_weeks, cfg.min_active_arms) == (5, 4, 3)
        assert cfg.retired_trailing_cycles == 8
        assert cfg.retire_evidence == "point"
        assert cfg.promote_min_weeks == 2

    def test_min_paired_dates_is_not_an_evidence_bar(self):
        """One paired date is the least from which any statistic can be formed.
        §5.0 abolished minimum-evidence floors fleetwide."""
        assert ra.ARENA_CONFIG.min_paired_dates == 1

    def test_the_committed_register_matches_the_live_registry(self):
        """A rotted genesis fixture would silently score four arms out of five.
        Regenerate with `python -m scoring.research_arena`."""
        assert ra.bootstrap_register().to_dicts() == json.loads(
            ra.BOOTSTRAP_REGISTER_PATH.read_text()
        )

    def test_every_live_arm_has_a_recovered_created_date(self):
        from producers.registry import research_slot_producers

        assert set(ra.ARM_CREATED_ON) == {s.name for s in research_slot_producers()}

    def test_the_arm_id_hashes_the_recipe(self):
        """§3.1. Two arms differing only in WIDTH must not share an id, or a
        20-wide arm could silently become 60-wide under one record."""
        assert ra.arm_id_for("attractiveness_60") != ra.arm_id_for("attractiveness_20")
        assert ra.arm_name_from_id(ra.arm_id_for("tech_score_20")) == "tech_score_20"

    def test_the_recipe_carries_what_changes_the_picks_and_nothing_else(self):
        """A spec carrying the description would re-hash the arm on a typo."""
        spec = ra.arm_spec("tech_score_20")
        assert set(spec) == {
            "slot", "name", "version", "prefilter_cut", "width", "stateful",
            "score_source", "supersedes",
        }


class TestTheSlotFloor:
    def test_the_genesis_register_clears_the_floor(self):
        ra.assert_slot_floor(ra.bootstrap_register())

    def test_a_cycle_dated_before_the_arms_existed_is_refused(self):
        """MEASURED 2026-09-22, by writing one. A cycle run for 2026-09-18 —
        four days before any arm of this slot was created — passed a floor
        checked with `as_of=None` (five arms live TODAY) and then decided on
        `active_arms: 0`, emitting `unservable` with champion `None`. It
        rendered as an ordinary red verdict rather than as the impossible input
        it was, and the standing detector read it as the slot refusing to
        serve.

        A slot cannot decide anything on a day none of its arms existed."""
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        with pytest.raises(ra.SlotFloorBreached, match="as of 2026-09-18"):
            ra.run_arena_cycle(
                board=board, champion_before=None, decided_on="2026-09-18",
                register=ra.bootstrap_register(),
            )

    def test_the_floor_is_point_in_time_not_present_tense(self):
        """The two counts that slipped past each other: the register's
        present-tense `active_arms()` and the count the ENGINE computes for the
        cycle's own date."""
        register = ra.bootstrap_register()
        assert len(register.active_arms()) == 5
        assert len(register.active_arms("2026-09-18")) == 0
        ra.assert_slot_floor(register)  # present tense: fine
        with pytest.raises(ra.SlotFloorBreached):
            ra.assert_slot_floor(register, as_of="2026-09-18")

    def test_a_register_below_the_floor_raises_rather_than_rendering_dull(self):
        """The 2026-08-21/28 defect by name: a decision loop that produced zero
        comparisons and rendered it as a routine hold, for two cycles."""
        # Retire down to two ACTIVE arms — two is the bare minimum for a
        # comparison to exist and the floor is three, which is the slack that
        # lets one arm miss a cycle and still leave a live comparison.
        register = ra.bootstrap_register()
        for arm in ("tech_score_20", "predictor_from_60", "thinktank_20"):
            register = register.retire(ra.arm_id_for(arm), _AS_OF, "test")
        assert len(register.active_arms()) == 2
        with pytest.raises(ra.SlotFloorBreached, match="below min_active_arms=3"):
            ra.assert_slot_floor(register)


# ── The per-date series the slot supplies ────────────────────────────────────


class TestSeriesFromBoard:
    def test_an_arm_with_no_row_is_a_series_of_misses_not_an_omission(self):
        """§3: silent absence and a genuine zero must never render
        identically."""
        board = _board([_row("tech_score_20", dict.fromkeys(_dates(3), 0.01))])
        series, counts = ra.series_from_board(board)
        missing = series[ra.arm_id_for("attractiveness_60")]
        assert missing.scores == {}
        assert counts["attractiveness_60"]["dropped_no_row"] == 1
        assert counts["attractiveness_60"]["scored"] == 0

    def test_a_degraded_date_is_dropped_and_counted_separately(self):
        """The board is the authority on its own input quality
        (alpha-engine-config-I11423). Scoring a date the board says is unusable
        would enter a known-bad observation as evidence."""
        dates = _dates(4)
        board = _board([
            _row(
                "tech_score_20",
                dict.fromkeys(dates, 0.02),
                degraded_input=[{
                    "reason": "mis_wired_predictor_pool", "dates": dates[:2],
                }],
            )
        ])
        series, counts = ra.series_from_board(board)
        arm = series[ra.arm_id_for("tech_score_20")]
        assert sorted(arm.scores) == dates[2:]
        assert counts["tech_score_20"]["dropped_degraded"] == 2
        assert counts["tech_score_20"]["scored"] == 2
        # And the dropped dates are MISSES, not absent — they were cohort dates.
        assert set(dates[:2]) <= arm.misses

    def test_a_null_series_is_counted_not_silently_empty(self):
        board = _board([_row("tech_score_20", None)])
        _series, counts = ra.series_from_board(board)
        assert counts["tech_score_20"]["dropped_null_series"] == 1

    def test_the_primary_block_is_the_boards_own_first_block(self):
        """A horizon named by this module and a horizon named by the board are
        two facts that can disagree."""
        board = _board([_row("tech_score_20", dict.fromkeys(_dates(2), 0.01))])
        board["horizons"].append({"horizon_days": 126, "specs": []})
        series, _ = ra.series_from_board(board)
        assert len(series[ra.arm_id_for("tech_score_20")].scores) == 2

    def test_a_horizon_the_board_never_produced_raises(self):
        """Scoring the slot on an absent horizon would silently score nothing."""
        board = _board([_row("tech_score_20", dict.fromkeys(_dates(2), 0.01))])
        with pytest.raises(ra.ResearchArenaError, match="carries no 999-session"):
            ra.series_from_board(board, horizon_days=999)


class TestServingPreconditions:
    def test_every_arm_gets_a_recorded_verdict_including_the_passes(self):
        """§5.3's gates are evaluated per cycle. An arm with no verdict is
        indistinguishable from an arm nobody checked."""
        pre = ra.preconditions_for(_board([]))
        assert set(pre) == set(ra.derived_arm_ids().values())
        for verdicts in pre.values():
            assert {v.name for v in verdicts} == {
                "promotion_eligible", "pinned_prefilter"
            }
            assert all(v.reason for v in verdicts)

    def test_every_live_arm_passes_the_pinned_prefilter_gate(self):
        """An arm serving from a pre-filter that is not the pinned one has its
        input population redefined by a pointer it does not control — the
        2026-09-18 defect that replaced 100% of a downstream arm's input
        without changing its spec hash."""
        pre = ra.preconditions_for(_board([]))
        for arm in ra.ARM_CREATED_ON:
            gate = next(
                v for v in pre[ra.arm_id_for(arm)] if v.name == "pinned_prefilter"
            )
            assert gate.passed, arm

    def test_an_ineligible_arm_carries_the_registers_reason(self):
        board = _board([
            _row(
                "tech_score_20", None,
                promotion_eligible=False, ineligible_reason="under audit",
            )
        ])
        gate = next(
            v for v in ra.preconditions_for(board)[ra.arm_id_for("tech_score_20")]
            if v.name == "promotion_eligible"
        )
        assert gate.passed is False
        assert gate.reason == "under audit"


# ── The cycle ────────────────────────────────────────────────────────────────


class TestTheCycle:
    def test_a_slot_with_no_matured_cohort_still_emits_a_record(self):
        """The measured first outcome. §11: a cycle is emitted whatever the
        outcome — `unmeasurable` with named reasons is a RESULT, not a
        silence."""
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        cycle, counts = ra.run_arena_cycle(
            board=board, champion_before=None, decided_on=_AS_OF,
            register=ra.bootstrap_register(),
        )
        assert cycle.decision.status == "unmeasurable"
        assert cycle.decision.moved is False
        # Every arm is accounted for, and every non-retirement is recorded.
        assert len(cycle.retirements) == len(ra.ARM_CREATED_ON)
        assert all(not v.retire for v in cycle.retirements)
        assert set(counts) == set(ra.derived_arm_ids())

    def test_the_incumbent_defaults_to_the_baseline_not_to_nothing(self):
        """With no incumbent the engine takes §9.1's bootstrap path and
        promotes the highest-Copeland arm — which, on an empty comparison set,
        is every arm tied at zero and resolves to whichever name sorts first.
        Measured 2026-09-22: `champion: attractiveness_20, moved: true` on ZERO
        comparisons. A record naming a winner chosen alphabetically explains a
        decision nobody made."""
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        cycle, _ = ra.run_arena_cycle(
            board=board,
            # The live pointer names an arm of the RETIRED producer slot.
            champion_before="scanner_predictor_direct",
            decided_on=_AS_OF,
            register=ra.bootstrap_register(),
        )
        assert cycle.decision.champion == ra.arm_id_for(ra.BASELINE_ARM)
        assert cycle.decision.status != "bootstrap"

    def test_the_baseline_is_the_arm_whose_register_row_says_so(self):
        from producers.registry import RESEARCH_PRODUCERS

        assert "BASELINE" in RESEARCH_PRODUCERS[ra.BASELINE_ARM].description

    def test_a_challenger_that_leads_on_ir_takes_the_pointer(self):
        """The slot's whole objective, end to end: a challenger with a steadier
        active-return series wins even where its raw mean does not dominate."""
        dates = _dates(20, start=1) + [f"2026-09-{d:02d}" for d in range(1, 11)]
        # The SAME mean on both legs (0.010), so nothing here is decided by a
        # raw mean — only by dispersion. A perfectly CONSTANT series would have
        # no ratio at all (None, never an infinity), so the steady arm varies
        # slightly, which is also what a real one does.
        steady = {d: (0.011 if i % 2 else 0.009) for i, d in enumerate(dates)}
        lumpy = {d: (0.040 if i % 2 else -0.020) for i, d in enumerate(dates)}
        board = _board([
            _row(ra.BASELINE_ARM, lumpy),
            _row("tech_score_20", steady),
        ])
        cycle, _ = ra.run_arena_cycle(
            board=board, champion_before=None, decided_on="2026-09-22",
            register=ra.bootstrap_register(),
        )
        assert cycle.decision.champion == ra.arm_id_for("tech_score_20")
        assert cycle.decision.moved is True
        assert "information_ratio" in cycle.decision.reason

    def test_an_arm_the_register_does_not_carry_is_not_scored(self):
        """An arm handed a series without a register row is scored by nothing
        the record can explain."""
        board = _board([_row(n, dict.fromkeys(_dates(3), 0.01)) for n in ra.ARM_CREATED_ON])
        register = ra.bootstrap_register()
        register = register.retire(
            ra.arm_id_for("tech_score_20"), _AS_OF, "test"
        )
        cycle, _ = ra.run_arena_cycle(
            board=board, champion_before=None, decided_on=_AS_OF, register=register,
        )
        # Retired, so still SCORED for its trailing window, but not active.
        assert ra.arm_id_for("tech_score_20") not in cycle.active_arms
        assert ra.arm_id_for("tech_score_20") in cycle.scored_arms


class TestTheArtifact:
    def test_the_cycle_document_conforms_to_the_arena_cycle_contract(self):
        """A non-conforming document on the key a consumer resolves from is
        worse than no document: a consumer cannot tell it apart from a good one
        until it parses it."""
        from nousergon_lib.contracts import conformance_errors

        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        register = ra.bootstrap_register()
        cycle, counts = ra.run_arena_cycle(
            board=board, champion_before=None, decided_on=_AS_OF, register=register,
        )
        doc = ra.cycle_document(
            cycle, counts=counts, register=register, board_present=True,
        )
        assert conformance_errors(ra.ARENA_CYCLE_CONTRACT, doc) == []

    def test_the_document_says_it_does_not_move_the_pointer(self):
        """The field that keeps the record honest about what it did NOT do. A
        reader must not mistake a written decision for a served one."""
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        register = ra.bootstrap_register()
        cycle, counts = ra.run_arena_cycle(
            board=board, champion_before=None, decided_on=_AS_OF, register=register,
        )
        doc = ra.cycle_document(
            cycle, counts=counts, register=register, board_present=True,
            serving_pointer="scanner_predictor_direct",
        )
        assert doc["serving"]["moves_pointer"] is False
        assert doc["serving"]["pointer_key"] == "config/producer_champion.json"
        assert doc["serving"]["pointer_champion"] == "scanner_predictor_direct"
        assert "I11438" in doc["serving"]["reason"]

    def test_the_document_states_which_ladders_restarted_and_why(self):
        """A short ladder is a declared fact here, not a gap
        (alpha-engine-config-I11422)."""
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        register = ra.bootstrap_register()
        cycle, counts = ra.run_arena_cycle(
            board=board, champion_before=None, decided_on=_AS_OF, register=register,
        )
        doc = ra.cycle_document(
            cycle, counts=counts, register=register, board_present=True,
        )
        # Reversed 2026-09-22 (alpha-engine-config-I11422): the two funnel arms
        # inherit the cuts board's record through `supersedes_cut`, so only the
        # two with no precedent on ANY surface remain.
        assert doc["history"]["inherited"] == {"thinktank_20": "thinktank_coverage"}
        assert doc["history"]["inherited_cut"] == {
            "attractiveness_60": "attractiveness_top_60",
            "attractiveness_20": "attractiveness_top_20",
        }
        assert set(doc["history"]["no_history_import"]) == {
            "tech_score_20", "predictor_from_60",
        }

    def test_the_slot_floor_is_on_the_artifact_never_absent(self):
        """"The floor was fine" and "nobody checked" must not render
        identically."""
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        register = ra.bootstrap_register()
        cycle, counts = ra.run_arena_cycle(
            board=board, champion_before=None, decided_on=_AS_OF, register=register,
        )
        doc = ra.cycle_document(
            cycle, counts=counts, register=register, board_present=True,
        )
        assert doc["slot_floor"]["breached"] is False
        assert doc["slot_floor"]["active_arms"] == len(ra.ARM_CREATED_ON)

    def test_a_non_conforming_document_is_refused_on_the_write_path(self, s3):
        with pytest.raises(ra.ResearchArenaError, match="does not conform"):
            ra.write_arena_cycle(
                {"slot": "research"}, ra.bootstrap_register(),
                decided_on=_AS_OF, bucket=_BUCKET, s3_client=s3,
            )

    def test_the_dated_record_and_the_mirror_and_the_register_are_all_written(self, s3):
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        out = ra.run_research_arena(s3, _BUCKET, _AS_OF, board=board)
        assert out["status"] == "ok"
        for key in (
            ra.ARENA_CYCLE_DATED_KEY.format(date=_AS_OF),
            ra.ARENA_CYCLE_LATEST_KEY,
            ra.ARENA_REGISTER_KEY,
        ):
            body = s3.get_object(Bucket=_BUCKET, Key=key)["Body"].read()
            assert json.loads(body), key

    def test_the_register_survives_a_round_trip_through_s3(self, s3):
        board = _board([_row(n, None) for n in ra.ARM_CREATED_ON])
        ra.run_research_arena(s3, _BUCKET, _AS_OF, board=board)
        reloaded = ra.load_register(bucket=_BUCKET, s3_client=s3)
        assert set(reloaded.all_arms()) == set(ra.derived_arm_ids().values())
