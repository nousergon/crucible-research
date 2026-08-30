"""Weekly promotion engine for the universe-cut slot — the SERVING POINTER,
transcribed from the shared arena engine (alpha-engine-config-I9317).

WHAT THIS DECIDES, AND WHERE THE DECISION NOW COMES FROM. The arms of the
universe-cut slot are count-matched at 60 (SCANNER_CONTRACT.md §1, Brian's
ruling 2026-08-20). Whichever holds the pointer feeds the sector teams:

    ``s3://{bucket}/config/scanner_cut_champion.json``

read by ``scoring/universe_membership.py::live_cut_champion`` /
``resolve_feed_cut``. Those two readers are UNCHANGED by this module's rewiring,
and the pointer plus its dated apply-audit mirror keep being written on every
evaluation. What changed is that **the decision inside them is no longer taken
here**. It is taken by ``nousergon_lib.arena.engine.run_cycle`` — the fleet's
single implementation of ``champion-challenger-policy.md`` §§3–6 — through
``scoring/cut_arena.py``, and this module transcribes it onto the pointer.
Re-implementing any of §§3–6 here would be a defect, not a variation (§10).

WHICH ARTIFACT IS AUTHORITATIVE, AND WHEN THE OTHER RETIRES.
``arena/universe_cut/{date}.json`` (mirrored to ``latest.json``) is the
AUTHORITATIVE decision record from this change forward: it carries the score
ladder per arm, every pairwise verdict WITH the common window it rests on, the
pointer decision with its anytime-valid bound, and every retirement verdict
including the ones that did not retire (policy §11).
``config/scanner_cut_champion.json`` is now a SERVING POINTER — the smallest
document that answers "which cut feeds the sector teams right now" — and
``config/apply_audit/scanner_cut_champion/`` its dated mirror. The pointer
retires when ``live_cut_champion`` / ``resolve_feed_cut`` resolve the champion
from the arena artifact instead; that is deliberately NOT done in this change,
because moving the serving path and the decision engine together would leave no
cycle in which the two could be compared.

WHAT WAS DELETED, NOT TRANSLATED (deliverable 4 of I9317)
---------------------------------------------------------
Three gates governed this slot's promotion path until 2026-08-29, and all three
are abolished fleetwide rather than retuned:

* ``min_weeks_for_inference = 5`` (paired weeks before an arm could be
  considered). §5.0: minimum-evidence floors control no error rate at all and
  deadlocked every promotion. The anytime-valid confidence sequence IS the
  evidence bar — very wide at week one, narrowing as evidence accrues — and it
  subsumes the floor naturally. ``ArenaConfig.min_paired_dates`` is a
  WELL-FORMEDNESS check (a window from which no statistic can be formed) and is
  explicitly not usable as an evidence bar.
* ``promotion_margin = 0.0002`` per week and ``cooldown_days = 28``. §5.2,
  Brian's ruling 2026-08-29: the pointer moves freely in BOTH directions with
  no margin and no cooldown. This is safe because the decision window is
  CUMULATIVE and therefore self-damping — a single bad week cannot flip a
  ranking computed over an arm's whole shared history. Any change to a TRAILING
  window re-opens that ruling; the two move together or not at all.
* ``decision_earliest_on`` / ``first_decidable_week``. They existed only to
  price the evidence floor against the NYSE calendar. With no floor they have
  nothing to say, so they are retired rather than left emitting a date nothing
  gates on.

The long forward horizons (126d, 252d) are NOT an evidence floor and are not
deleted, but they no longer VETO either. They are reported on every record and
gate nothing — see ``corroborating`` below and ``_corroboration``'s own note.
A forward-window horizon measures a hold the weekly re-cut guarantees never
happens, its maturity test is a minimum-cohort count of exactly the kind §5.0
abolishes, and §4.1 admits one decision basis per slot: the longest window the
two compared arms actually share. Keeping it as a reported field preserves the
number for a reader; keeping it as a gate would have reintroduced an abolished
floor under another name.

WHY A HOLD IS WRITTEN AND NOT OMITTED. champion-challenger-policy.md §3: silent
absence and a genuine outcome must never render identically. A pointer that
stops being written is indistinguishable from a pointer that decided to hold.
This engine writes on EVERY evaluation, whatever the outcome:

    ``arena/universe_cut/{date}.json``                        the decision record
    ``arena/universe_cut/latest.json``                        its mirror
    ``arena/universe_cut/register.json``                      the arm register
    ``config/apply_audit/scanner_cut_champion/{date}.json``   immutable, dated
    ``config/apply_audit/scanner_cut_champion/latest.json``   pointer mirror
    ``config/scanner_cut_champion.json``                      the live pointer

MEASURABILITY (principles.md §2.7). The numbers that say this is working are
``arena/universe_cut/latest.json``'s ``slot_floor.active_arms`` against
``slot_floor.min_active_arms``, and its ``decision.status``. Their ABSENCE — a
missing or stale ``latest.json`` — is a freshness-registry row, and a slot that
emits nothing is unobserved, never green. The arm floor additionally PAGES on
breach through ``cut_arena.assert_slot_floor``, because the 2026-08-21 and
2026-08-28 cycles wrote a well-formed ``no_promotable_challenger`` record while
the slot held exactly one arm and nothing was watching the count.

FAIL-LOUD (AGENTS.md). ``decide_cut_champion`` is pure and never swallows: an
input it cannot interpret is a DEFECT, and a defect still produces a written
``hold`` record carrying it — after which ``run_cut_promotion`` RAISES
``CutPromotionError``. Record first, then fail: a defect that also erases the
evidence of itself is the worse of the two failures. An ABSENT ledger and an
``unmeasurable`` cycle are not defects — they are the expected state until the
ledger carries paired weeks for more than one arm — and are plain holds.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from nousergon_lib.arena import ArenaConfig, ArenaCycle, ArmRegister

from scoring.cut_arena import (
    ARENA_CONFIG,
    ARENA_CYCLE_DATED_KEY,
    ARENA_CYCLE_LATEST_KEY,
    ARENA_REGISTER_KEY,
    ARENA_SLOT,
    SCORE_DEFINITION,
    apply_retirements,
    arm_id_for,
    arm_name_from_id,
    assert_slot_floor,
    cycle_document,
    load_register,
    run_arena_cycle,
    series_from_ledger,
    write_arena_cycle,
)
from scoring.leaderboard_scoring import (
    HORIZON_OK,
    LONG_HORIZONS_DAYS,
    duplicate_arm_rows,
    slot_spec,
)
from scoring.universe_membership import (
    CUT_ARM_PROMOTION_EXCLUSIONS,
    CUT_CHAMPION_POINTER_KEY,
    DEFAULT_CUT_CHAMPION,
    OBSERVE_ONLY_CUTS,
    PROMOTABLE_CUTS,
    SLOT_ARMS,
    _bucket,
    _client,
    live_cut_champion,
    promotion_ineligibility_from_rank_tables,
)
from scoring.verdict_digest import VERDICT_SLOTS, send_verdict_digest
from scoring.weekly_ledger import (
    LEDGER_COLS,
    LEDGER_KEY,
    LEDGER_VERSION,
    read_ledger,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4
"""v3 → v4 on the alpha-engine-config-I9317 arena wiring.

Not additive. Three required v3 blocks are GONE because the mechanisms they
described are abolished (champion-challenger-policy.md §5.0 and §5.2):
``hysteresis`` (promotion margin, cooldown, min_weeks_for_inference) and
``decision_earliest_on`` (the calendar projection of the evidence floor). A new
required ``arena`` block carries the decision's real provenance — the arena
slot, the cycle key the full record lives at, the engine's status, and the
``ArenaConfig`` the cycle ran under. Several v3 ``reason_code`` slugs describe
conditions that can no longer arise and are retired rather than re-minted.

A v3 reader handed a v4 record would find no ``hysteresis`` block and read a
promotion as though a 0.0002/week margin had been cleared, when no margin was
applied at all. That is exactly the silent redefinition a schema version
exists to prevent.

v2 → v3 on the I9272 / I9284 change; v1 → v2 on the I8261 cutover. Both are
recorded in this file's git history.
"""

PRODUCER = "crucible-research/scoring/cut_promotion.py"

AUDIT_DATED_KEY = "config/apply_audit/scanner_cut_champion/{date}.json"
AUDIT_LATEST_KEY = "config/apply_audit/scanner_cut_champion/latest.json"
CUTS_LEADERBOARD_KEY = "research/cuts_leaderboard/{date}.json"

DECISION_PROMOTE = "promote"
DECISION_HOLD = "hold"

DECISION_CADENCE_WEEKLY = "weekly"

# The delivery row for this slot (alpha-engine-config-I9278), resolved from the
# module that owns delivery rather than restated here. It used to be declared
# beside the keys it names, so that a key rename could not leave the email
# pointing at an object that no longer exists — but a SECOND slot arrived
# (scoring/spec_promotion.py) and two engines each holding their own row is how
# the pair drifts. The registry is now the single place both are declared, and
# `test_the_registry_row_names_the_keys_this_module_writes` asserts this row's
# keys against THIS module's constants, which is the property the old placement
# was buying.
VERDICT_SLOT = VERDICT_SLOTS["scanner_cut"]

# The name of the decision metric, carried on every record and on every arm.
# It is deliberately long: a reader who sees only this string must be able to
# tell WHAT is scored (net-of-cost log return), AGAINST WHAT (the population
# the arm selected from, never SPY — the 2026-08-17 140bp inversion), and AT
# WHAT CADENCE (weekly, non-overlapping). The pairing against another arm is
# the ENGINE's job and is therefore not in the name: under
# `nousergon_lib.arena` every pair is differenced on its OWN longest common
# window, so there is no single "vs champion" series any more.
DECISION_METRIC = "weekly_population_relative_net_log_return"

# The lowest forward horizon that may be REPORTED as a corroborating read.
# Below this the horizon does not match the scanner's ~1-year objective, which
# is the substance of alpha-engine-config-I7580. It no longer gates anything —
# see the module docstring — but a reported number from a horizon nobody would
# act on is noise, so the floor is kept on the reporting surface.
MIN_VETO_HORIZON_DAYS = 126

# Machine-readable outcome slugs. A prose ``reason`` is for the human reading
# the artifact; this is what a sweep, a console adapter or a test matches on.
# Never reuse a slug for a different condition — a renamed slug is a schema bump.
#
# The live set is now a TRANSCRIPTION of the arena engine's own statuses
# (`nousergon_lib.arena.engine.PointerDecision.status`) plus the two conditions
# that stop a cycle from being run at all. Minting a slug here that the engine
# cannot produce is how the two decision surfaces drift.
REASON_PROMOTED = "promoted"
REASON_CHAMPION_LEADS = "champion_already_leads"
REASON_LEDGER_MISSING = "weekly_ledger_missing"
REASON_ARENA_UNMEASURABLE = "arena_unmeasurable"
REASON_ARENA_UNSERVABLE = "arena_unservable"
REASON_ARENA_BOOTSTRAP = "arena_bootstrap"

HOLD_REASON_CODES: tuple[str, ...] = (
    REASON_CHAMPION_LEADS,
    REASON_LEDGER_MISSING,
    REASON_ARENA_UNMEASURABLE,
    REASON_ARENA_UNSERVABLE,
)
"""The outcomes on which the pointer does NOT move."""

LIVE_REASON_CODES: tuple[str, ...] = (
    REASON_PROMOTED,
    REASON_ARENA_BOOTSTRAP,
    *HOLD_REASON_CODES,
)
"""Every slug this module can emit. ``arena_bootstrap`` is not a hold — it
moves the pointer, on a slot that had no eligible incumbent to hold (§9.1) —
and it is not ``promoted`` either, because nothing was beaten to get there."""

#: Arena pointer status → the slug this record carries. Exhaustive over
#: ``PointerDecision.status``; an unmapped status RAISES rather than defaulting,
#: because a status the engine grew and this module silently folded into "hold"
#: is a decision surface quietly losing resolution.
ARENA_STATUS_TO_REASON: dict[str, str] = {
    "decided": REASON_PROMOTED,
    "held": REASON_CHAMPION_LEADS,
    "unmeasurable": REASON_ARENA_UNMEASURABLE,
    "unservable": REASON_ARENA_UNSERVABLE,
    "bootstrap": REASON_ARENA_BOOTSTRAP,
}

# Retired with the mechanisms they described. Kept as a NAMED set, not deleted
# outright, so a reader of an archived v1/v2/v3 record can resolve a slug this
# module no longer emits, and so a future author cannot re-mint one of these
# strings for a different condition. Asserted disjoint from the live set at
# import.
#
# The v3 retirements are the substance of alpha-engine-config-I9317:
# `insufficient_weeks` was the §5.0-abolished evidence floor, `margin_not_met`
# and `cooldown_active` the §5.2-abolished hysteresis,
# `corroborating_horizon_disagrees` a second decision basis on a forward
# horizon, `weekly_ledger_arm_missing` a slot-wide hold on ONE arm's absence
# (the arena records that arm's miss and compares the others), and
# `no_promotable_challenger` the 2026-08-21/28 defect itself — now structurally
# unreachable, because `cut_arena` refuses to import below
# `ArenaConfig.min_active_arms` promotable arms and pages if the live register
# ever falls under it.
RETIRED_REASON_CODES: tuple[str, ...] = (
    # v1, retired at the I8261 cutover
    "board_missing",
    "board_unmeasurable",
    "decision_horizon_immature",
    "decision_horizon_unmeasurable",
    "arm_row_missing",
    "arm_metric_missing",
    "insufficient_dates",
    # v3, retired at the I9317 arena wiring
    "board_defective",
    "insufficient_weeks",
    "margin_not_met",
    "cooldown_active",
    "corroborating_horizon_disagrees",
    "weekly_ledger_arm_missing",
    "no_promotable_challenger",
)

RETIRED_V1_REASON_CODES = RETIRED_REASON_CODES
"""Deprecated alias. The set stopped being v1-only at the I9317 wiring; the
name is kept for one release so an external reader is not broken by a rename
that carries no meaning."""


class CutPromotionError(RuntimeError):
    """The engine could not complete an honest evaluation.

    Raised only AFTER the hold record carrying the defect has been written, so
    the failure is never the reason the evidence of it is missing.
    """


@dataclass(frozen=True)
class CutPromotionSlot:
    """The slot registry row for the universe-cut decision.

    champion-challenger-policy.md §10: *"Every slot names, in its registry, the
    metric, benchmark, count-matching width, and its ``ArenaConfig``
    parameters — ``alpha``, ``diff_clip``, ``cap``, ``grace_weeks``,
    ``min_active_arms``, ``retired_trailing_cycles``, ``retire_evidence``. This
    document deliberately does not enumerate their values — they are per-slot
    facts that CI can check against code."* This is that row, and the
    ``ArenaConfig`` is carried by REFERENCE (:data:`arena_config`) rather than
    restated field by field, so the registry and the object the engine actually
    runs on cannot disagree.

    Deliberately NOT an extension of ``LEADERBOARD_SLOTS["cuts"]``: that spec
    describes a MEASUREMENT surface whose arms are the funnel's own stages and
    explicitly *not competing* (``per_arm_width=True``). This slot is a subset
    of that board's arms — the count-matched 60s — which ARE competing. §2
    forbids conflating the two; sharing one dataclass would have.
    """

    slot_id: str
    # The arms eligible to hold the feed. Resolved from PROMOTABLE_CUTS rather
    # than restated, so the writer and the reader (`live_cut_champion`) can
    # never disagree about who is eligible.
    arms: tuple[str, ...]
    default_champion: str

    # ── The decision basis ───────────────────────────────────────────────────
    # WHERE the per-date score is read. The DECISION is taken by
    # `nousergon_lib.arena.engine.run_cycle` over the series built from it; this
    # field names the evidence, not the rule.
    decision_source: str
    decision_cadence: str
    primary_metric: str
    # The ledger column the per-date score is built from, net of transaction
    # cost. Net, not gross: at weekly rebalance turnover is first-order and the
    # arms in this slot have wildly different churn (42% vs 76% week-over-week
    # retention, measured 2026-07-27, EXPERIMENTS.md). An arm that wins gross
    # and loses net is the classic trap.
    ledger_return_column: str
    # The arithmetic that turns two ledger columns into ONE population-relative
    # per-arm score, stated here so a reader of the registry row never has to
    # open the producer to learn what a ladder rung is denominated in.
    score_definition: str

    # ── The arena (policy §10) ───────────────────────────────────────────────
    arena_slot: str
    arena_config: ArenaConfig

    # ── Reported, and gating nothing ─────────────────────────────────────────
    # These were corroborating VETOES until alpha-engine-config-I9317. They are
    # now recorded on every decision record and consulted by nothing. See the
    # module docstring for why: a forward-window horizon measures a hold the
    # weekly re-cut guarantees never happens, its "mature" test is a
    # minimum-cohort count of exactly the kind §5.0 abolishes, and §4.1 admits
    # one decision basis per slot.
    corroborating_horizons_days: tuple[int, ...]
    corroborating_leaderboard_id: str
    corroborating_metric: str
    corroborating_min_dates: int

    # Horizons that are SCORED every cycle and are neither a decision input nor
    # a report worth acting on. Asserted disjoint from
    # `corroborating_horizons_days` at import.
    excluded_horizons_days: tuple[int, ...]

    # Arms that are SCORED every cycle but cannot hold the feed. Declared here
    # so a reader of this row can tell "measured and ineligible" from "not
    # measured at all" without going to another module (ARCHITECTURE §140).
    observe_only_arms: tuple[str, ...]
    # Every arm the ledger scores, promotable or not. Evidence is built for ALL
    # of these — champion-challenger-policy.md §3 makes measurement
    # unconditional — while only ``arms`` may win.
    scored_arms: tuple[str, ...]
    # Arm → the REASON it may not hold the feed, carried onto every record as
    # ``excluded_arms``. Empty under Brian's ruling 2026-08-29: every scored arm
    # of this slot is promotion-eligible. Kept as a mechanism so a future
    # carve-out is a stated property rather than an absence from a tuple.
    excluded_arms: Mapping[str, str]
    # §4 count-matching: every arm of the slot, promotable or not, is 60 by
    # construction. A registry/documentation field, never a runtime gate.
    count_matched_width: int


# First cohort date on repaired fundamentals (alpha-engine-config-I8255): the
# vendor-fundamentals cross-section this slot's evidence is scored on was
# degenerate before this date. It is the earliest admissible weekly observation
# and the ``created_date`` the champion arm is registered under in
# ``scoring/arena/universe_cut_register.json``. It is NOT an evidence floor and
# nothing gates on it — the ledger simply carries no earlier week.
FIRST_COHORT_DATE = date(2026, 8, 20)

# The forward horizons this slot SCORES but neither decides on nor reports as
# corroboration. Every date scored at 21d so far predates FIRST_COHORT_DATE, so
# it was measured on the pre-repair fundamentals cross-section
# (alpha-engine-config-I8255), and its reported t_stat used an iid standard
# error over overlapping windows that inflates |t| by roughly sqrt(lags+1)
# (alpha-engine-config-I8263, fixed in crucible-research-PR732).
# ``excluded_horizons`` carries this caveat WITH the number so a reader of the
# promotion record cannot mistake it for a clean read.
CONTAMINATION_CAVEAT = (
    "every date scored at this horizon so far predates the 2026-08-20 "
    "fundamentals repair and was measured on the degenerate pre-repair "
    "cross-section (alpha-engine-config-I8255); the reported t_stat also used "
    "an iid standard error over overlapping windows that inflates |t| by "
    "roughly sqrt(lags+1) (alpha-engine-config-I8263, crucible-research-PR732) "
    "— this is not a clean read on the arm"
)

CUT_PROMOTION_SLOT = CutPromotionSlot(
    slot_id="scanner_cut",
    arms=PROMOTABLE_CUTS,
    observe_only_arms=OBSERVE_ONLY_CUTS,
    scored_arms=SLOT_ARMS,
    excluded_arms=CUT_ARM_PROMOTION_EXCLUSIONS,
    default_champion=DEFAULT_CUT_CHAMPION,
    decision_source=LEDGER_KEY,
    decision_cadence=DECISION_CADENCE_WEEKLY,
    primary_metric=DECISION_METRIC,
    ledger_return_column="net_log_return",
    score_definition=SCORE_DEFINITION,
    arena_slot=ARENA_SLOT,
    arena_config=ARENA_CONFIG,
    corroborating_horizons_days=(126, 252),
    corroborating_leaderboard_id="cuts",
    corroborating_metric=slot_spec("cuts").primary_metric,
    corroborating_min_dates=slot_spec("cuts").min_dates_for_inference,
    excluded_horizons_days=(21,),
    count_matched_width=60,
)

# ── Import-time invariants ────────────────────────────────────────────────────
# Each is a way the engine could silently start deciding on the wrong evidence,
# so none is left to a test alone.
if CUT_PROMOTION_SLOT.decision_source != LEDGER_KEY:
    raise AssertionError(
        "the universe-cut per-date score must be read from the weekly ledger "
        f"and nothing else ({LEDGER_KEY}). A score sourced from a leaderboard "
        "horizon block measures a hold the weekly re-cut guarantees never "
        "happens (alpha-engine-config-I8261, Brian's ruling 2026-08-24)"
    )
if CUT_PROMOTION_SLOT.arena_config.slot != CUT_PROMOTION_SLOT.arena_slot:
    raise AssertionError(
        "the registry row's arena_slot and the ArenaConfig it carries disagree "
        "— the row would describe a slot the engine never runs"
    )
if CUT_PROMOTION_SLOT.arena_config.slot_kind != "universe_cut":
    raise AssertionError(
        "this slot must declare slot_kind='universe_cut'; that declaration is "
        "what makes ArenaConfig REFUSE a SPY benchmark for it (the 2026-08-17 "
        "140bp inversion, champion-challenger-policy.md §4)"
    )
if CUT_PROMOTION_SLOT.arena_config.benchmark != "population":
    raise AssertionError(
        "a selection-stage slot is graded against the POPULATION it selected "
        "from, never a market index"
    )
if len(CUT_PROMOTION_SLOT.arms) < CUT_PROMOTION_SLOT.arena_config.min_active_arms:
    raise AssertionError(
        f"the slot has {len(CUT_PROMOTION_SLOT.arms)} promotable arm(s) against "
        f"min_active_arms={CUT_PROMOTION_SLOT.arena_config.min_active_arms}. "
        "This is the 2026-08-21/2026-08-28 `no_promotable_challenger` defect: "
        "a slot with one arm produces ZERO comparisons and writes a hold that "
        "reads like a decision (alpha-engine-config-I9317)"
    )
if set(CUT_PROMOTION_SLOT.excluded_horizons_days) & set(
    CUT_PROMOTION_SLOT.corroborating_horizons_days
):
    raise AssertionError(
        "a horizon may not be BOTH structurally excluded and reported as "
        "corroboration — the 21-session block that produced "
        "alpha-engine-config-I7580 must not reach the record twice under two "
        "different labels"
    )
for _h in CUT_PROMOTION_SLOT.corroborating_horizons_days:
    if _h < MIN_VETO_HORIZON_DAYS:
        raise AssertionError(
            f"corroborating horizon {_h} is below MIN_VETO_HORIZON_DAYS="
            f"{MIN_VETO_HORIZON_DAYS} — a reported horizon must match the "
            "scanner's ~1-year objective (alpha-engine-config-I7580)"
        )
    if _h not in LONG_HORIZONS_DAYS:
        raise AssertionError(
            f"horizon {_h} is not scored by the cuts leaderboard "
            f"(LONG_HORIZONS_DAYS={LONG_HORIZONS_DAYS}) — the record would "
            "report a block that is never written"
        )
if CUT_PROMOTION_SLOT.corroborating_metric != slot_spec("cuts").primary_metric:
    raise AssertionError(
        "the corroborating read ranks on a metric the cuts board does not treat "
        "as primary — one of the two is wrong and it must not be resolved here"
    )
if CUT_PROMOTION_SLOT.ledger_return_column not in LEDGER_COLS:
    raise AssertionError(
        f"{CUT_PROMOTION_SLOT.ledger_return_column!r} is not a weekly-ledger "
        f"column ({LEDGER_COLS}) — the score would read a field that is never "
        "written"
    )
if CUT_PROMOTION_SLOT.ledger_return_column not in CUT_PROMOTION_SLOT.score_definition:
    raise AssertionError(
        "the registry row's score_definition does not mention the column it "
        "names as the decision column — the row would describe an arithmetic "
        "the producer does not perform"
    )
if not set(CUT_PROMOTION_SLOT.arms) <= set(CUT_PROMOTION_SLOT.scored_arms):
    raise AssertionError(
        "a promotable arm that the ledger does not score is a pointer target "
        "nothing can ever justify moving to — PROMOTABLE_CUTS must be a subset "
        "of SLOT_ARMS (alpha-engine-config-I9272)"
    )
if set(CUT_PROMOTION_SLOT.excluded_arms) != (
    set(CUT_PROMOTION_SLOT.scored_arms) - set(CUT_PROMOTION_SLOT.arms)
):
    raise AssertionError(
        "every scored-but-not-promotable arm must carry a REASON in "
        "excluded_arms, and every reason must name a real exclusion. A "
        "non-promotable arm expressible only as an absence from PROMOTABLE_CUTS "
        "is the defect alpha-engine-config-I9272 retired: no artifact can "
        "report it and no reader can tell it from an oversight."
    )
if CUT_PROMOTION_SLOT.default_champion not in CUT_PROMOTION_SLOT.arms:
    raise AssertionError(
        "the default champion is not promotable — live_cut_champion() would "
        "raise on the very pointer value it falls back to"
    )
if set(LIVE_REASON_CODES) & set(RETIRED_REASON_CODES):
    raise AssertionError(
        "a retired reason_code has been re-minted for a live condition — a "
        "slug means one thing forever or the decision series stops being "
        "readable across schema versions"
    )
if set(ARENA_STATUS_TO_REASON.values()) - set(LIVE_REASON_CODES):
    raise AssertionError(
        "every arena status must map to a slug this module declares, or a "
        "decision the engine took would be recorded under a slug no consumer "
        "knows"
    )


@dataclass
class ArmEvidence:
    """What the ARENA CYCLE says about one arm, transcribed onto the pointer.

    Every number here is READ OFF the cycle artifact — the ladder for the arm's
    own record, the incumbent comparison for its head-to-head, the pairwise
    ranking for its standing. None of it is recomputed: a second implementation
    of §§3–6 living on the pointer would be exactly the drift §10 forbids, and
    it is how a record comes to disagree with the artifact it cites.

    Every field is self-qualifying by construction
    (alpha-engine-config-I8257): ``metric``, ``cadence``, ``score_definition``
    and ``source`` travel with the numbers, so ``n_weeks_paired: 0`` reads as
    "0 paired weeks against the incumbent on the weekly ledger", never as an
    unqualified zero a reader has to join back to a registry to interpret.
    """

    present: bool = False
    is_champion: bool = False
    #: The arena arm id. Carries the spec hash, so a reader can tell a retuned
    #: recipe from the arm it replaced without diffing two records (§3.1).
    arm_id: str | None = None

    # ── The arm's own record (the ladder) ────────────────────────────────────
    n_weeks_scored: int = 0
    n_weeks_missed: int = 0
    ladder_weeks: int = 0
    #: The longest rung's mean — Brian's "the longest running score". Reported;
    #: the DECISION is taken on the window a PAIR shares, never on this (§4.1).
    mean_score: float | None = None
    first_week: str | None = None
    last_week: str | None = None

    # ── Why a ledger week did not become a score ─────────────────────────────
    # Kept as three counters rather than one, because they have three different
    # fixes: a stale ledger_version, a row with no number in the decision or
    # population column (the ledger's first week is entirely of this kind — an
    # uncomputable transaction cost), and a row priced over a different span
    # from the rest of the slot that week. Folding them together renders a
    # missing cut and an uncomputable cost identically.
    weeks_dropped_stale_version: int = 0
    weeks_dropped_null_decision_column: int = 0
    weeks_dropped_span_mismatch: int = 0

    # ── Eligibility, as a STATE (alpha-engine-config-I9272) ───────────────────
    # Whether this arm could hold the feed if it won, and if not, WHY. Never an
    # absence: an arm the engine will not promote appears on the record saying
    # so, because "measured and ineligible" and "not measured" are different
    # answers to the only question a reader of this artifact is asking (§3).
    eligible_for_promotion: bool = True
    ineligibility_reason: str | None = None
    retired_on: str | None = None

    # ── Head-to-head with the incumbent, on the window they SHARE ────────────
    n_weeks_paired: int = 0
    mean_paired_log_return: float | None = None
    paired_first_week: str | None = None
    paired_last_week: str | None = None
    #: The anytime-valid interval on the paired difference. ``supported`` is
    #: True only when the WHOLE interval sits above zero — "ahead on the point
    #: estimate" is not a promotion signal on the serving path (§5.0).
    confseq_lower: float | None = None
    confseq_upper: float | None = None
    confseq_supported: bool = False
    comparison_status: str | None = None
    comparison_reason: str | None = None

    # ── Standing in the pool (§6.2 Condorcet-style pairwise wins) ────────────
    pairwise_wins: int = 0
    pairwise_losses: int = 0
    pairwise_unmeasurable: int = 0

    metric: str = ""
    cadence: str = ""
    source: str = ""
    score_definition: str = ""


@dataclass
class CutPromotionDecision:
    """The full decision record — the document written to all three keys."""

    decision: str
    champion: str
    champion_before: str
    reason: str
    reason_code: str
    decided_on: str
    arms: dict[str, ArmEvidence] = field(default_factory=dict)
    last_promoted_on: str | None = None
    corroborating: dict[str, Any] | None = None
    defect: str | None = None
    excluded_horizons: dict[str, dict] = field(default_factory=dict)
    excluded_arms: dict[str, dict] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    arena: dict[str, Any] = field(default_factory=dict)

    def to_document(self, *, leaderboard_key: str | None = None) -> dict:
        slot = CUT_PROMOTION_SLOT
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "slot_id": slot.slot_id,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "decided_on": self.decided_on,
            "decision": self.decision,
            "champion": self.champion,
            "champion_before": self.champion_before,
            "reason": self.reason,
            "reason_code": self.reason_code,
            # The decision basis, stated on the record rather than implied by
            # the module that wrote it (principles.md §2.1 — reconstructable
            # from durable artifacts alone).
            "decision_metric": slot.primary_metric,
            "decision_cadence": slot.decision_cadence,
            "decision_source": slot.decision_source,
            "decision_column": slot.ledger_return_column,
            "score_definition": slot.score_definition,
            "last_promoted_on": self.last_promoted_on,
            "leaderboard_key": leaderboard_key,
            "arms": {name: asdict(ev) for name, ev in self.arms.items()},
            # WHERE the decision was actually taken, and under what parameters.
            # Required, not optional: a pointer that does not name the cycle it
            # was transcribed from is a claim with no evidence attached, and
            # this record is deliberately the SMALLER of the two artifacts now.
            "arena": self.arena,
            # Reported, gating nothing — see the module docstring.
            "corroborating": self.corroborating,
            "defect": self.defect,
            "excluded_horizons": self.excluded_horizons,
            # Every arm the ledger scores that may NOT hold the feed, with
            # the reason. Empty under Brian's ruling 2026-08-29; present as
            # a field on every record regardless, because a reader must be
            # able to tell "no arm is excluded" from "this record does not
            # say" (alpha-engine-config-I9272).
            "excluded_arms": self.excluded_arms,
            "scored_arms": list(slot.scored_arms),
            "promotable_arms": list(slot.arms),
            "ledger": self.ledger,
        }


# ── Reading the ledger ────────────────────────────────────────────────────────


def _clean(value: Any) -> Any:
    """pandas renders a missing float as NaN, and NaN is not None.

    A NaN reaching the decision would compare False against every threshold and
    propagate silently through a mean — a missing observation rendering as a
    number. Every ledger cell is normalised here, once, on the way in.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def ledger_rows_from_frame(frame: Any) -> list[dict]:
    """A ledger DataFrame → plain records, NaN normalised to None.

    Kept separate from :func:`read_weekly_ledger_rows` so the pure decision
    path can be handed records built by a test without pandas in the loop.
    """
    if frame is None:
        return []
    rows: list[dict] = []
    for record in frame.to_dict("records"):
        rows.append({k: _clean(v) for k, v in record.items()})
    return rows


def read_weekly_ledger_rows(
    *, bucket: str | None = None, s3_client: Any = None
) -> list[dict] | None:
    """The weekly ledger as records, or ``None`` when it has never been written.

    ``None``, not ``[]``. The ledger is not yet written by anything — the
    producer is being wired concurrently (alpha-engine-config-I8264) — so an
    absent store is the EXPECTED state today, and it must render as "no
    measurement exists" and never as an empty-but-healthy series
    (champion-challenger-policy.md §7.2). ``read_ledger`` already draws that
    distinction; this preserves it rather than collapsing it into a list.
    """
    frame = read_ledger(bucket=bucket, s3_client=s3_client)
    if frame is None:
        return None
    return ledger_rows_from_frame(frame)


def _rows_by_arm(
    rows: Sequence[Mapping[str, Any]], arm: str
) -> tuple[list[dict], int]:
    """This arm's rows at the CURRENT ledger version, and how many were dropped
    for carrying an older one.

    A row written under an earlier ``LEDGER_VERSION`` may define a column
    differently (that is the only thing the version is bumped for), so mixing
    versions in one mean silently averages two different quantities. Older rows
    sit ALONGSIDE the current ones by design; the decision reads the current
    ones and REPORTS the count it set aside, because a silently narrowed
    sample is indistinguishable from a short history.
    """
    keep: list[dict] = []
    stale = 0
    for row in rows:
        if row.get("arm") != arm:
            continue
        version = row.get("ledger_version")
        if version is not None and int(version) != LEDGER_VERSION:
            stale += 1
            continue
        keep.append(dict(row))
    keep.sort(key=lambda r: str(r.get("week_start") or ""))
    return keep, stale


# ── The vetoes ────────────────────────────────────────────────────────────────


def _block_for(board: dict | None, horizon: int) -> dict | None:
    for block in (board or {}).get("horizons") or []:
        if block.get("horizon_days") == horizon:
            return block
    return None


def _rows_for(block: dict | None, arm: str) -> list[dict]:
    return [r for r in (block or {}).get("specs") or [] if r.get("name") == arm]


def _board_number(row: dict, metric: str) -> tuple[float | None, float | None, int]:
    m = row.get(metric)
    mean = m.get("mean") if isinstance(m, dict) else None
    t_stat = m.get("t_stat") if isinstance(m, dict) else None
    return (
        (float(mean) if mean is not None else None),
        (float(t_stat) if t_stat is not None else None),
        int(row.get("n_dates_scored") or 0),
    )


def _veto_horizon(
    board: dict | None, slot: CutPromotionSlot, proposed: str, horizon: int
) -> dict:
    """What ONE long horizon says, when it is in a position to say anything.

    An absent, immature or thin block is recorded with ``mature: false`` and
    ``disagrees: false`` — an unmeasured veto is not a veto
    (champion-challenger-policy.md §5.1: you cannot gate on a statistic you did
    not measure, and an uncomputed gate reported as a PASS is the defect that
    rule prevents). At rollout this is the normal state for both horizons, and
    the record says so in words rather than leaving a reader to infer it from a
    null.
    """
    base = {
        "horizon_days": horizon,
        "metric": slot.corroborating_metric,
        "role": "reported_only",
        "leader": None,
        "disagrees": False,
        "mature": False,
    }
    block = _block_for(board, horizon)
    if board is None:
        return {
            **base,
            "status": "absent",
            "note": (
                "no cuts leaderboard was available this evaluation; the veto is "
                "UNMEASURED and therefore non-blocking (§5.1). It does not hold "
                "the decision: the decision does not come from this board."
            ),
        }
    if block is None:
        return {
            **base,
            "status": "absent",
            "note": (
                f"the cuts leaderboard carries no {horizon}d block; the veto is "
                "unmeasured and non-blocking (§5.1)."
            ),
        }
    if block.get("status") != HORIZON_OK:
        return {
            **base,
            "status": block.get("status"),
            "note": (
                f"the {horizon}d block reports status="
                f"{block.get('status')!r} ({block.get('reason')}); the veto is "
                "unmeasured and non-blocking (§5.1). This is the expected state "
                "until the cohort matures."
            ),
        }
    arms: dict[str, dict] = {}
    for arm in slot.arms:
        rows = _rows_for(block, arm)
        if len(rows) != 1:
            return {
                **base,
                "status": block.get("status"),
                "note": (
                    f"{arm} has {len(rows)} rows at {horizon}d, so no single "
                    "number is the arm's; the veto is unmeasured and "
                    "non-blocking (§5.1)."
                ),
            }
        mean, t_stat, n_dates = _board_number(rows[0], slot.corroborating_metric)
        arms[arm] = {
            "mean": mean,
            "t_stat": t_stat,
            "n_dates_scored": n_dates,
            "confidence": str(rows[0].get("confidence") or "insufficient"),
        }
    if any(
        a["mean"] is None or a["n_dates_scored"] < slot.corroborating_min_dates
        for a in arms.values()
    ):
        return {
            **base,
            "status": block.get("status"),
            "arms": arms,
            "note": (
                f"at least one arm is below the board's evidence floor "
                f"({slot.corroborating_min_dates} dates) at {horizon}d; the veto "
                "is unmeasured and non-blocking (§5.1)."
            ),
        }
    leader = max(arms.items(), key=lambda kv: (kv[1]["mean"], kv[0]))[0]
    return {
        **base,
        "status": block.get("status"),
        "mature": True,
        "leader": leader,
        "disagrees": leader != proposed,
        "arms": arms,
        "note": (
            f"mature at {horizon}d; it may BLOCK the weekly series' proposal and "
            "may never make one."
        ),
    }


def _corroboration(
    board: dict | None, slot: CutPromotionSlot, proposed: str
) -> dict:
    """Every long forward horizon, REPORTED. Gates nothing.

    Until alpha-engine-config-I9317 the ``blocking`` / ``blocked_by`` fields
    below refused a promotion the weekly series proposed. They are retained on
    the record — the numbers are real and a reader wants them — and no code
    path consults them. See the module docstring: a forward-window horizon
    measures a hold the weekly re-cut guarantees never happens, its ``mature``
    test is a minimum-cohort count of exactly the kind
    champion-challenger-policy.md §5.0 abolishes, and §4.1 admits one decision
    basis per slot — the longest window the two compared arms actually share.
    """
    horizons = {
        str(h): _veto_horizon(board, slot, proposed, h)
        for h in slot.corroborating_horizons_days
    }
    blocked_by = [
        int(h) for h, entry in horizons.items() if entry.get("disagrees")
    ]
    mature = [int(h) for h, entry in horizons.items() if entry.get("mature")]
    return {
        "role": "reported_only",
        "proposed": proposed,
        "horizons": horizons,
        "mature_horizons": sorted(mature),
        "blocking": bool(blocked_by),
        "blocked_by": sorted(blocked_by),
        "note": (
            "REPORTED ONLY since alpha-engine-config-I9317. 126 and 252 were "
            "corroborating vetoes under the pre-arena engine (Brian's ruling "
            "2026-08-24, alpha-engine-config-I8261); they now gate nothing. "
            "`blocking` and `blocked_by` are retained so an archived record "
            "stays comparable with its predecessors, and nothing reads them. "
            "The decision is the anytime-valid confidence sequence on the "
            "longest window the two compared arms share "
            "(champion-challenger-policy.md §4.1, §5.0)."
        ),
    }


def _excluded_horizons(board: dict | None, slot: CutPromotionSlot) -> dict[str, dict]:
    """Per arm, what the horizons that are NEITHER a decision input NOR a veto
    measured, and why they are excluded (alpha-engine-config-I8257 deliverable
    2, carried forward to the I8261 basis).

    Written unconditionally, including when there is no board at all — a reader
    must not have to open a second artifact to learn either that the excluded
    number exists or that it is contaminated. Keyed
    ``{arm: {str(horizon_days): {...}}}``.
    """
    out: dict[str, dict] = {}
    for arm in slot.arms:
        entries: dict[str, dict] = {}
        for horizon in slot.excluded_horizons_days:
            block = _block_for(board, horizon) if board else None
            base_reason = (
                f"excluded_horizon: {slot.slot_id} decides on the chained "
                f"weekly series ({slot.primary_metric}) and vetoes with "
                f"{list(slot.corroborating_horizons_days)}d — {horizon}d is "
                "neither. It may not propose a promotion and it may not block "
                "one: it is the block that produced "
                "alpha-engine-config-I7580 (structurally excluded, asserted at "
                "import)."
            )
            if block is None:
                entries[str(horizon)] = {
                    "horizon_days": horizon,
                    "n_dates_scored": 0,
                    "mean": None,
                    "t_stat": None,
                    "metric": slot.corroborating_metric,
                    "excluded_reason": (
                        f"{base_reason} No {horizon}d block available."
                    ),
                }
                continue
            rows = _rows_for(block, arm)
            if len(rows) != 1:
                entries[str(horizon)] = {
                    "horizon_days": horizon,
                    "n_dates_scored": 0,
                    "mean": None,
                    "t_stat": None,
                    "metric": slot.corroborating_metric,
                    "excluded_reason": (
                        f"{base_reason} {len(rows)} rows for {arm!r} at "
                        f"{horizon}d — no single row to report."
                    ),
                }
                continue
            mean, t_stat, n_dates = _board_number(rows[0], slot.corroborating_metric)
            reason = base_reason
            if n_dates:
                reason = f"{base_reason} {CONTAMINATION_CAVEAT}."
            entries[str(horizon)] = {
                "horizon_days": horizon,
                "n_dates_scored": n_dates,
                "mean": mean,
                "t_stat": t_stat,
                "metric": slot.corroborating_metric,
                "excluded_reason": reason,
            }
        out[arm] = entries
    return out


# ── The decision, pure ────────────────────────────────────────────────────────


def evaluate_cut_slot(
    *,
    ledger_rows: Sequence[Mapping[str, Any]] | None,
    champion_before: str,
    decided_on: str,
    register: ArmRegister,
    cycle_ineligible_arms: Mapping[str, str] | None = None,
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
) -> tuple[ArenaCycle, dict[str, dict[str, int]]]:
    """Run one arena cycle for this slot. Pure: no S3, no clock, no alerting.

    A thin forward to :func:`scoring.cut_arena.run_arena_cycle`, kept here so
    the slot has ONE named entry point a caller and a test can both reach
    without having to know which of the two modules owns which half.
    """
    return run_arena_cycle(
        ledger_rows=ledger_rows,
        champion_before=champion_before,
        decided_on=decided_on,
        register=register,
        cycle_ineligible_arms=cycle_ineligible_arms,
        config=slot.arena_config,
    )


def _arm_evidence_from_cycle(
    arm: str,
    *,
    cycle: ArenaCycle,
    register: ArmRegister,
    counts: Mapping[str, Mapping[str, int]],
    champion_before: str,
    cycle_ineligible: Mapping[str, str],
    slot: CutPromotionSlot,
) -> ArmEvidence:
    """One arm's row on the pointer record, READ OFF the arena cycle."""
    arm_id = arm_id_for(arm)
    ladders = {ladder.arm_id: ladder for ladder in cycle.ladders}
    comparisons = {c.challenger: c for c in cycle.decision.comparisons}
    standings = dict(cycle.ranking.standings) if cycle.ranking is not None else {}
    ladder = ladders.get(arm_id)
    longest = ladder.longest if ladder is not None else None
    arm_counts = dict(counts.get(arm) or {})
    scored_weeks = int(arm_counts.get("scored", 0))

    state = register.state(arm_id) if arm_id in register else None
    retired_on = state.retired_date if state is not None else None

    ev = ArmEvidence(
        present=bool(scored_weeks or (ladder is not None and ladder.total_dates)),
        is_champion=(arm == champion_before),
        arm_id=arm_id,
        n_weeks_scored=scored_weeks,
        n_weeks_missed=(ladder.total_misses if ladder else 0),
        ladder_weeks=(ladder.total_weeks if ladder else 0),
        mean_score=(longest.mean_score if longest else None),
        first_week=(longest.start_date if longest else None),
        last_week=(longest.end_date if longest else None),
        weeks_dropped_stale_version=int(arm_counts.get("dropped_stale_version", 0)),
        weeks_dropped_null_decision_column=int(arm_counts.get("dropped_null_column", 0)),
        weeks_dropped_span_mismatch=int(arm_counts.get("dropped_span_mismatch", 0)),
        retired_on=retired_on,
        metric=slot.primary_metric,
        cadence=slot.decision_cadence,
        source=slot.decision_source,
        score_definition=slot.score_definition,
    )

    standing = standings.get(arm_id)
    if standing is not None:
        ev.pairwise_wins = standing.wins
        ev.pairwise_losses = standing.losses
        ev.pairwise_unmeasurable = standing.unmeasurable

    if arm == champion_before:
        ev.comparison_status = "incumbent"
        ev.comparison_reason = (
            "the incumbent is not compared against itself; its own record is "
            "the ladder above, and every challenger's number is a difference "
            "against it on the window the pair shares"
        )
    else:
        comparison = comparisons.get(arm_id)
        if comparison is not None:
            ev.comparison_status = comparison.status
            ev.comparison_reason = comparison.reason
            ev.n_weeks_paired = comparison.window.n_dates
            ev.paired_first_week = comparison.window.start_date
            ev.paired_last_week = comparison.window.end_date
            if comparison.window.measurable:
                ev.mean_paired_log_return = comparison.window.mean_diff
            if comparison.bound is not None:
                ev.confseq_lower = comparison.bound.lower
                ev.confseq_upper = comparison.bound.upper
                ev.confseq_supported = comparison.bound.supported

    # Eligibility, in the order the engine applies it: the register's permanent
    # exclusion, then retirement, then this cycle's §5.3 serving preconditions.
    # Each is recorded with its reason, never as an absence
    # (alpha-engine-config-I9272).
    if arm not in slot.arms:
        ev.eligible_for_promotion = False
        ev.ineligibility_reason = slot.excluded_arms.get(arm) or (
            "not_promotable: absent from PROMOTABLE_CUTS"
        )
    elif retired_on is not None:
        ev.eligible_for_promotion = False
        ev.ineligibility_reason = (
            f"retired on {retired_on}: {state.retired_reason}. Still scored for "
            f"{slot.arena_config.retired_trailing_cycles} cycle(s) past "
            "retirement so 'we retired the wrong one' stays detectable "
            "(champion-challenger-policy.md §3)"
        )
    elif arm_id in cycle.decision.ineligible:
        failed = [p for p in cycle.decision.ineligible[arm_id] if not p.passed]
        ev.eligible_for_promotion = False
        ev.ineligibility_reason = "; ".join(f"{p.name}: {p.reason}" for p in failed)
    elif arm in cycle_ineligible:
        ev.eligible_for_promotion = False
        ev.ineligibility_reason = cycle_ineligible[arm]
    return ev


def decide_cut_champion(
    *,
    ledger_rows: Sequence[Mapping[str, Any]] | None,
    board: dict | None = None,
    champion_before: str,
    decided_on: str,
    last_promoted_on: str | None = None,
    cycle_ineligible_arms: Mapping[str, str] | None = None,
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
    register: ArmRegister | None = None,
    cycle: ArenaCycle | None = None,
    ledger_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> CutPromotionDecision:
    """Transcribe one arena cycle onto the serving-pointer record. Pure.

    ``cycle`` and ``ledger_counts`` let a caller that has ALREADY run the cycle
    — ``run_cut_promotion`` does, because it must also write the cycle
    artifact — hand it in rather than have it recomputed, which would be two
    decisions for one evaluation. Absent them the cycle is run here from
    ``register`` (defaulting to the committed genesis register), so the
    function stays a pure ``inputs → record`` for a test.

    ``ledger_rows`` is the per-date score evidence. ``board`` — a cuts
    leaderboard — is REPORTED and gates nothing; it is read for the
    corroborating block and the excluded-horizon report, and a defect in it is
    recorded and raised on by the caller without touching the pointer, because
    a corrupt artifact that feeds no part of the decision must not be allowed
    to fake one (see the module docstring).

    Every exit produces a record. There is no path that returns nothing, and no
    path that applies an evidence floor, a promotion margin or a cooldown —
    §5.0 and §5.2 abolish all three, and the anytime-valid confidence sequence
    inside the engine is the only bar a promotion clears.
    """
    cycle_ineligible = dict(cycle_ineligible_arms or {})
    excluded_arms = {
        arm: {"arm": arm, "reason": reason, "scored": True, "scope": "register"}
        for arm, reason in slot.excluded_arms.items()
    } | {
        arm: {"arm": arm, "reason": reason, "scored": True, "scope": "this_cycle"}
        for arm, reason in cycle_ineligible.items()
        if arm not in slot.excluded_arms
    }

    if cycle is None:
        register = register if register is not None else load_register(events=[])
        cycle, counts = evaluate_cut_slot(
            ledger_rows=ledger_rows,
            champion_before=champion_before,
            decided_on=decided_on,
            register=register,
            cycle_ineligible_arms=cycle_ineligible,
            slot=slot,
        )
    else:
        if register is None:
            raise CutPromotionError(
                "a caller supplying a pre-run arena cycle must also supply the "
                "register it ran against — retirement state and created_dates "
                "are read from it, and inferring them from the cycle would be a "
                "second source of truth for arm lifecycle"
            )
        counts = dict(ledger_counts or {})

    arms = {
        arm: _arm_evidence_from_cycle(
            arm,
            cycle=cycle,
            register=register,
            counts=counts,
            champion_before=champion_before,
            cycle_ineligible=cycle_ineligible,
            slot=slot,
        )
        for arm in slot.scored_arms
    }

    status = cycle.decision.status
    try:
        reason_code = ARENA_STATUS_TO_REASON[status]
    except KeyError as exc:  # pragma: no cover -- guarded at import
        raise CutPromotionError(
            f"the arena engine returned pointer status {status!r}, which this "
            "module has no slug for. A status folded into a generic hold is a "
            "decision surface silently losing resolution "
            "(champion-challenger-policy.md §11)"
        ) from exc

    champion = (
        arm_name_from_id(cycle.decision.champion)
        if cycle.decision.champion is not None
        else champion_before
    )
    moved = champion != champion_before
    if ledger_rows is None:
        # More specific than the engine's `unmeasurable`, and a different fix:
        # an absent store is not a slot that failed to produce a comparison, it
        # is a slot whose evidence was never written (§7.2 — absence and an
        # empty-but-healthy series must never render identically).
        reason_code = REASON_LEDGER_MISSING

    defect: str | None = None
    if board:
        board_dupes = duplicate_arm_rows(board)
        if board_dupes:
            # Recorded and raised on by the caller; NOT a hold. Until
            # alpha-engine-config-I9317 the board held a veto, so a board that
            # counted an arm twice could not be trusted to have counted the
            # others once and the safe answer was to hold. The board now feeds
            # nothing the pointer depends on, and holding a decision on the
            # corruption of an artifact that does not inform it would be a gate
            # with no mechanism behind it.
            defect = f"duplicate arm rows: {', '.join(board_dupes)}"

    ledger_meta = {
        "key": slot.decision_source,
        "ledger_version": LEDGER_VERSION,
        "present": ledger_rows is not None,
        "rows_read": (len(ledger_rows) if ledger_rows is not None else 0),
        "column": slot.ledger_return_column,
        "score_definition": slot.score_definition,
        "arms_present": sorted(a for a, ev in arms.items() if ev.present),
        "per_arm": {arm: dict(c) for arm, c in sorted(counts.items())},
    }

    arena_block = {
        "slot": cycle.slot,
        "slot_kind": cycle.slot_kind,
        "benchmark": cycle.benchmark,
        "as_of": cycle.as_of,
        "status": status,
        "moved": cycle.decision.moved,
        "engine_reason": cycle.decision.reason,
        "incumbent_arm_id": cycle.decision.incumbent,
        "champion_arm_id": cycle.decision.champion,
        "cycle_key": ARENA_CYCLE_DATED_KEY.format(date=decided_on),
        "latest_key": ARENA_CYCLE_LATEST_KEY,
        "register_key": ARENA_REGISTER_KEY,
        "active_arms": list(cycle.active_arms),
        "scored_arms": list(cycle.scored_arms),
        "config": {
            "alpha": slot.arena_config.alpha,
            "diff_clip": slot.arena_config.diff_clip,
            "variance_mode": slot.arena_config.variance_mode,
            "min_paired_dates": slot.arena_config.min_paired_dates,
            "cap": slot.arena_config.cap,
            "grace_weeks": slot.arena_config.grace_weeks,
            "min_active_arms": slot.arena_config.min_active_arms,
            "retired_trailing_cycles": slot.arena_config.retired_trailing_cycles,
            "retire_evidence": slot.arena_config.retire_evidence,
        },
        "retirements": [v.to_dict() for v in cycle.retirements],
        "note": (
            "the full record — every ladder rung, every pairwise verdict with "
            "the window it rests on, and the confidence-sequence bound behind "
            "this decision — is at cycle_key. This document is the SERVING "
            "POINTER (champion-challenger-policy.md §11)."
        ),
    }

    return CutPromotionDecision(
        decision=DECISION_PROMOTE if moved else DECISION_HOLD,
        champion=champion,
        champion_before=champion_before,
        reason=cycle.decision.reason,
        reason_code=(REASON_PROMOTED if moved else reason_code),
        decided_on=decided_on,
        arms=arms,
        last_promoted_on=(decided_on if moved else last_promoted_on),
        corroborating=(
            _corroboration(board, slot, champion) if champion else None
        ),
        defect=defect,
        excluded_horizons=_excluded_horizons(board, slot),
        excluded_arms=excluded_arms,
        ledger=ledger_meta,
        arena=arena_block,
    )


# ── I/O ───────────────────────────────────────────────────────────────────────


def _get_json(s3: Any, bucket: str, key: str) -> dict | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — absence is a legitimate state
        if "NoSuchKey" in str(exc) or "NoSuchBucket" in str(exc) or "404" in str(exc):
            return None
        raise
    return json.loads(body)


def reconcile_arms_with_ledger(
    doc: dict, ledger_rows: Sequence[Mapping[str, Any]] | None
) -> list[str]:
    """Every ``arms.<arm>.n_weeks_scored`` in a WRITTEN record must equal what
    the ledger it cites actually supports.

    Successor to the same guard against the v3 basis, which compared
    ``n_weeks_paired`` recomputed by a SECOND pairing implementation living in
    this module. That implementation is gone — the engine owns pairing — so the
    guard now rebuilds the arm series with the very function the decision used
    (:func:`scoring.cut_arena.series_from_ledger`) and checks the RECORD
    against it. The property is unchanged and it is the one this module was
    originally filed to protect: a record and its own cited evidence must not
    disagree. What changed is that the check can no longer drift from the
    decision by being a fork of it.

    ``n_weeks_scored`` rather than ``n_weeks_paired`` is the reconciled
    quantity because it is the one the ledger alone determines. A paired count
    depends on which arm is the incumbent, which is a property of the decision
    and not of the evidence.

    Returns the list of mismatches; empty means every arm reconciles.
    """
    if ledger_rows is None:
        return []
    series, _counts = series_from_ledger(ledger_rows)
    mismatches: list[str] = []
    for arm, ev in (doc.get("arms") or {}).items():
        arm_id = ev.get("arm_id")
        if arm_id is None or arm_id not in series:
            continue
        supported = len(series[arm_id].scores)
        if ev.get("n_weeks_scored") != supported:
            mismatches.append(
                f"{arm}: record reports n_weeks_scored={ev.get('n_weeks_scored')} "
                f"on {doc.get('score_definition')!r}, but the cited ledger "
                f"supports {supported} scored week(s)"
            )
    return mismatches


def read_cut_champion_record(*, bucket: str | None = None, s3_client: Any = None) -> dict | None:
    """The current pointer document, or ``None`` when it has never been written.

    Only ``last_promoted_on`` is consumed from it; the authoritative champion is
    taken from :func:`live_cut_champion`, which VALIDATES it. Reading the name
    from the raw document here would fork that validation.
    """
    return _get_json(_client(s3_client), _bucket(bucket), CUT_CHAMPION_POINTER_KEY)


_UNSET = object()


def run_cut_promotion(
    decided_on: str,
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    leaderboard: dict | None = None,
    ledger_rows: Any = _UNSET,
    membership: dict | None = None,
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
) -> dict:
    """Run the arena cycle, WRITE both artifacts unconditionally, deliver, then
    fail loud on any defect. Returns the pointer document.

    ``leaderboard`` lets the caller hand in the board it just built (the scanner
    handler does), so the reported corroboration reads the exact artifact this
    run produced rather than re-fetching a key that may not have landed.
    ``ledger_rows`` likewise; ``_UNSET`` (the default) reads the ledger from S3,
    and an explicit ``None`` means "no ledger". The two are distinguished on
    purpose — passing ``None`` to mean "go and read it" is how an absent store
    would come to look like a caller's choice.

    Raises :class:`CutPromotionError` AFTER writing when the record carries a
    defect — the defect is durable before the process is allowed to fail.
    """
    s3 = _client(s3_client)
    b = _bucket(bucket)

    champion_before = live_cut_champion(bucket=b, s3_client=s3)
    prior = read_cut_champion_record(bucket=b, s3_client=s3) or {}
    last_promoted_on = prior.get("last_promoted_on")

    key = CUTS_LEADERBOARD_KEY.format(date=decided_on)
    board = leaderboard if leaderboard is not None else _get_json(s3, b, key)
    rows = (
        read_weekly_ledger_rows(bucket=b, s3_client=s3)
        if ledger_rows is _UNSET
        else ledger_rows
    )

    # An arm whose basis carries no full-universe rank table cannot be promoted
    # to without breaking a consumer on the morning of the promotion
    # (alpha-engine-config-I7843). The scanner handler passes the membership it
    # just built; without one the engine cannot ESTABLISH servability and says
    # so on the record rather than assuming it (§5.1 — you cannot gate on a
    # statistic you did not measure, and an uncomputed gate reported as a PASS
    # is the defect that rule prevents).
    if membership is not None:
        cycle_ineligible = promotion_ineligibility_from_rank_tables(membership)
    else:
        cycle_ineligible = {}
        logger.info(
            "[cut_promotion] no membership passed for %s — rank-table servability "
            "is UNCHECKED this cycle and no arm is excluded on it",
            decided_on,
        )

    # The register is live state, seeded once from the committed genesis file.
    # `load_register` PAGES and raises if the slot has fallen below its arm
    # floor — the 2026-08-21/28 defect — before any decision is attempted.
    register = load_register(bucket=b, s3_client=s3)
    assert_slot_floor(
        register,
        config=slot.arena_config,
        context=(
            f"universe-cut evaluation for {decided_on} cannot run: the arena "
            "would produce zero comparisons."
        ),
    )

    cycle, counts = evaluate_cut_slot(
        ledger_rows=rows,
        champion_before=champion_before,
        decided_on=decided_on,
        register=register,
        cycle_ineligible_arms=cycle_ineligible,
        slot=slot,
    )

    # Retirement is applied to the register BEFORE either artifact is written,
    # so the register object both of them describe is the post-cycle one and a
    # reader is never handed a cycle whose verdicts contradict the register
    # shipped beside it. `evaluate_retirements` has already applied every §6.1
    # veto — champion, floor, grace — so nothing is re-checked here (§10: one
    # implementation, not two).
    register = apply_retirements(register, cycle, decided_on)

    # ── The AUTHORITATIVE record (champion-challenger-policy.md §11) ──────────
    # Written FIRST and unconditionally, whatever the outcome. A slot that
    # emits nothing is not healthy, it is unobserved.
    arena_doc = cycle_document(
        cycle,
        counts=counts,
        register=register,
        ledger_present=rows is not None,
        config=slot.arena_config,
    )
    write_arena_cycle(
        arena_doc, register, decided_on=decided_on, bucket=b, s3_client=s3
    )

    decision = decide_cut_champion(
        ledger_rows=rows,
        board=board,
        cycle_ineligible_arms=cycle_ineligible,
        champion_before=champion_before,
        decided_on=decided_on,
        last_promoted_on=last_promoted_on,
        slot=slot,
        register=register,
        cycle=cycle,
        ledger_counts=counts,
    )
    doc = decision.to_document(leaderboard_key=key if board is not None else None)

    # Write ORDER is load-bearing, mirroring write_universe_membership_to_s3:
    # the immutable dated record lands FIRST. If the process dies between
    # writes the surviving state is "the decision is recorded but the live
    # pointer still names last week's champion" — recoverable and safe. The
    # reverse leaves a moved feed with no record of why.
    payload = json.dumps(doc, indent=2, sort_keys=True).encode()
    for k in (
        AUDIT_DATED_KEY.format(date=decided_on),
        AUDIT_LATEST_KEY,
        CUT_CHAMPION_POINTER_KEY,
    ):
        s3.put_object(Bucket=b, Key=k, Body=payload, ContentType="application/json")

    logger.info(
        "[cut_promotion] metric cut_promotion_decision decision=%s champion=%s "
        "reason_code=%s metric=%s cadence=%s arena_status=%s %s",
        doc["decision"],
        doc["champion"],
        doc["reason_code"],
        doc["decision_metric"],
        doc["decision_cadence"],
        doc["arena"]["status"],
        " ".join(
            f"{a}_n_weeks_scored={doc['arms'][a]['n_weeks_scored']}"
            for a in slot.scored_arms
            if a in doc["arms"]
        ),
    )

    # ── Delivery (alpha-engine-config-I9278) ──────────────────────────────────
    # AFTER the writes, so the email can never be the reason a record is
    # missing; and BEFORE the defect raise, because a cycle that recorded a
    # DEFECT is precisely the cycle Brian most needs delivered, and a raise
    # above this line would send nothing. `send_verdict_digest` escalates its
    # OWN failure to an ops alert and returns False rather than raising, so a
    # notification can never red a promotion run.
    send_verdict_digest(doc, VERDICT_SLOT)

    if decision.defect:
        raise CutPromotionError(
            f"the cuts leaderboard for {decided_on} carries a DEFECT: "
            f"{decision.defect}. The pointer decision did not depend on it and "
            f"stands; the record was written to "
            f"{AUDIT_DATED_KEY.format(date=decided_on)} and the cycle to "
            f"{ARENA_CYCLE_DATED_KEY.format(date=decided_on)} before this raise."
        )

    # Reconciliation guard. Written AFTER the record lands, same discipline as
    # the defect raise above: the record itself is the evidence a future reader
    # needs, so it must survive even the failure that says something about it
    # disagreed.
    mismatches = reconcile_arms_with_ledger(doc, rows)
    if mismatches:
        raise CutPromotionError(
            f"the universe-cut record for {decided_on} disagrees with the "
            f"weekly ledger it cites ({slot.decision_source}): "
            f"{'; '.join(mismatches)}. The record was written to "
            f"{AUDIT_DATED_KEY.format(date=decided_on)} before this raise."
        )
    return doc
