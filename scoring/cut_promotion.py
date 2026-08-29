"""Weekly promotion engine for the scanner cut slot — deciding on the CHAINED
WEEKLY SERIES, with the long forward horizons demoted to vetoes
(alpha-engine-config-I8261, Brian's ruling 2026-08-24; originally I7826).

WHAT THIS DECIDES. The promotable arms of the scanner-cut slot are count-matched
at 60 (SCANNER_CONTRACT.md §1, Brian's ruling 2026-08-20). Whichever performs
better holds the sector-team feed. This module is the only writer of the pointer
that says which:

    ``s3://{bucket}/config/scanner_cut_champion.json``

read by ``scoring/universe_membership.py::live_cut_champion`` /
``resolve_feed_cut`` (crucible-research#670). The pointer, its reader and its
default already existed; this module supplies the DECISION.

WHAT CHANGED ON 2026-08-24, AND WHY (Brian's ruling on I8261)
-------------------------------------------------------------
Brian: *"shouldn't we just be tracking performance weekly?"* — and then, in
words: decide on the chained weekly series; the paired weekly difference vs
champion is the decision metric; keep 126 and 252 as corroborating vetoes once
mature, not as the decision basis; retire ``forbidden_horizons_days`` as moot.

The horizon axis — 21 vs 126 vs 252 — was the wrong axis. The cut is re-formed
WEEKLY, so:

1. **A forward-window horizon measures a hold that never happens.** A
   126-session forward return from a cohort date describes a six-month hold
   that the weekly re-cut guarantees is replaced ~25 times inside the window.
2. **Its observations overlap.** Weekly cohort dates against an N-session
   window share most of their span, so consecutive observations are strongly
   dependent. That is corrected now (alpha-engine-config-I8263) but the
   correction costs power a non-overlapping series never loses.
3. **The fleet's own promotion battery takes a RETURN SERIES.**
   champion-challenger-policy.md §5.1's PSR / DSR / CSCV-PBO are statistics of
   a return series. Overlapping cohort-date draws are not one, so those
   sub-gates were structurally uncomputable — correctly reported
   ``insufficient`` and non-blocking, i.e. this slot's statistical gate has
   been switched off, silently and legitimately. A weekly holding-period
   series is exactly the object they take.

So the decision metric is now the **paired weekly difference vs the champion**,
read from ``scoring/weekly_ledger.py``'s append-only ledger
(``research/cuts_weekly_ledger/ledger.parquet``), aggregated over the chained
series. PAIRED because the same-week champion leg cancels the common market
factor and collapses the variance of the difference — at ~52 observations a
year that is the difference between needing tens of observations and hundreds
(champion-challenger-policy.md §4).

WHAT REPLACED ``forbidden_horizons_days`` (I8261 requirement 4)
---------------------------------------------------------------
That field, and its import-time assertion, existed to stop a forward-window
horizon — specifically the 21-session block — from becoming the DECISION BASIS.
After this change no forward-window horizon is the decision basis at all: the
decision reads a different artifact. The old guard is therefore moot, and it is
retired rather than left standing as coverage it no longer provides.

The property it held is now held by three import-time invariants, asserted
below and named here so nothing is quietly dropped:

* ``decision_source`` MUST be ``weekly_ledger.LEDGER_KEY``. The decision cannot
  read a leaderboard horizon block because it does not read the leaderboard for
  evidence at all — a stronger statement than "one horizon is banned".
* ``excluded_horizons_days`` (today ``(21,)``) MUST be DISJOINT from
  ``corroborating_horizons_days``. The 21-session block cannot even VETO, which
  is strictly more restrictive than the retired rule: under the old design it
  was merely barred from proposing.
* every ``corroborating_horizons_days`` entry MUST be ≥
  ``MIN_VETO_HORIZON_DAYS`` (126) and scored by the cuts board. A veto horizon
  has to match the scanner's ~1-year objective, which is the substance of
  alpha-engine-config-I7580 — a −0.264 IC at 21 days drove a live change that
  nine years of history inverted at 126–252 days.

THE VETO IS ASYMMETRIC, AND AN IMMATURE VETO IS NOT A VETO
-----------------------------------------------------------
126 and 252 may BLOCK a promotion the weekly series proposes. Neither may ever
PROPOSE one — structurally, because the promotion path only consults them after
the weekly series has already named a leader. When a veto horizon is absent,
immature, or below the board's own evidence floor it is recorded with
``mature: false``, ``disagrees: false`` and a note saying so: you cannot gate on
a statistic you did not measure (champion-challenger-policy.md §5.1), and an
uncomputed gate reported as a PASS is the defect that rule prevents. At rollout
this is the normal state for both of them.

A DEFECTIVE board is treated differently from an ABSENT one, deliberately. An
absent veto is honestly unmeasured and non-blocking. A board reporting duplicate
arm rows is not unmeasured — it is UNRELIABLE, and a safety mechanism that may
be reading someone else's numbers is worse than one that is switched off. So a
defective board holds and then raises; a missing board does not hold at all.

WHY A HOLD IS WRITTEN AND NOT OMITTED. champion-challenger-policy.md §3: silent
absence and a genuine outcome must never render identically. A pointer that
stops being written is indistinguishable from a pointer that decided to hold,
and the fleet has paid for that confusion before — ``config/producer_champion``
carries a whole second artifact (``config/apply_audit/producer_champion/``,
written unconditionally) for exactly this reason (config#2054). This engine
writes on EVERY evaluation, promote or hold:

    ``config/apply_audit/scanner_cut_champion/{date}.json``   immutable, dated
    ``config/apply_audit/scanner_cut_champion/latest.json``   pointer mirror
    ``config/scanner_cut_champion.json``                      the live pointer

all three carrying the same v2 document (``contracts/scanner_cut_champion.schema.json``).

MEASURABILITY (principles.md §2.7). The number that says this is working is
``arms.<arm>.n_weeks_paired`` — completed, paired weekly observations against
the champion. It is 0 until the I8264 producer starts writing the ledger and
climbs one per week thereafter; when every promotable arm crosses
``min_weeks_for_inference`` the engine can decide, and until then every record
says so in ``reason`` and prices it against a calendar in
``decision_earliest_on``. Its ABSENCE is a missing/stale
``config/apply_audit/scanner_cut_champion/latest.json`` — the engine did not run
— which is a freshness-registry row; no data is never rendered as a promotion
and never as green.

HYSTERESIS (champion-challenger-policy.md §5.2) is IMPLEMENTED here, not waived,
and the cutover deliberately changes the BASIS and not the BAR. See
``promotion_margin`` below for the units conversion, which is carried on every
record so a reader never has to do it.

FAIL-LOUD (AGENTS.md). ``decide_cut_champion`` is pure and never swallows: an
input it cannot interpret is a DEFECT, and a defect still produces a written
``hold`` record carrying it — after which ``run_cut_promotion`` RAISES
``CutPromotionError``. Record first, then fail: a defect that also erases the
evidence of itself is the worse of the two failures. An ABSENT ledger, an
IMMATURE veto and a THIN weekly series are not defects — they are the expected
state for weeks — and are plain holds with no raise and no alert.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from nousergon_lib.trading_calendar import add_trading_days

from scoring.leaderboard_scoring import (
    HORIZON_OK,
    LONG_HORIZONS_DAYS,
    confidence_for,
    date_clustered_stats,
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
    chained_log_return,
    paired_weekly_differences,
    read_ledger,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
"""v2 → v3 on the I9272 / I9284 change (Brian's ruling 2026-08-29).

Not additive either. ``decision_earliest_on`` changed from a bare date STRING
to an object carrying ``provisional`` / ``counted_from`` / ``basis``, because
the v2 string published a floor its own ledger could not reach and no reader
could tell (alpha-engine-config-I9284). ``excluded_arms`` and per-arm
``eligible_for_promotion`` / ``ineligibility_reason`` were added, and the
``arms`` block widened from the PROMOTABLE arms to every SCORED arm. A v2
reader handed a v3 record would read the earliest-decision field as a string
and find an object.

v1 → v2 on the I8261 cutover.

Not an additive change: the decision BASIS moved from a leaderboard forward
horizon to the weekly ledger, so ``horizon_days`` / ``primary_metric`` /
``arms.*`` no longer mean what a v1 reader would take them to mean, and several
v1 ``reason_code`` slugs describe conditions that can no longer arise. Silently
redefining them under ``schema_version: 1`` is precisely how a multi-year
decision series becomes uninterpretable.
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

# The name of the decision metric, carried on every record and on every arm. It
# is deliberately long: a reader who sees only this string must be able to tell
# WHAT was differenced (net, i.e. after transaction cost), AGAINST WHAT (the
# serving champion), and AT WHAT CADENCE (weekly, non-overlapping).
DECISION_METRIC = "paired_weekly_net_log_return_vs_champion"

# The lowest forward horizon that may hold a VETO. Below this the horizon does
# not match the scanner's ~1-year objective, which is the substance of
# alpha-engine-config-I7580. This is one of the three invariants that replaced
# the retired ``forbidden_horizons_days`` assertion.
MIN_VETO_HORIZON_DAYS = 126

# Trading sessions in a scanner week. The ledger's weeks are bounded by
# consecutive cut effective dates, not by a fixed count — see
# ``weekly_ledger.holding_period`` — so this is used for ONE thing only:
# projecting ``decision_earliest_on`` onto a calendar. It is never used to
# measure a week that has already happened.
SESSIONS_PER_WEEK = 5

# Machine-readable outcome slugs. A prose ``reason`` is for the human reading
# the artifact; this is what a sweep, a console adapter or a test matches on.
# Never reuse a slug for a different condition — a renamed slug is a schema bump.
REASON_PROMOTED = "promoted"
REASON_CHAMPION_LEADS = "champion_already_leads"
REASON_NO_PROMOTABLE_CHALLENGER = "no_promotable_challenger"
REASON_LEDGER_MISSING = "weekly_ledger_missing"
REASON_LEDGER_ARM_MISSING = "weekly_ledger_arm_missing"
REASON_INSUFFICIENT_WEEKS = "insufficient_weeks"
REASON_MARGIN_NOT_MET = "margin_not_met"
REASON_COOLDOWN_ACTIVE = "cooldown_active"
REASON_CORROBORATION_DISAGREES = "corroborating_horizon_disagrees"
REASON_BOARD_DEFECTIVE = "board_defective"

HOLD_REASON_CODES: tuple[str, ...] = (
    REASON_CHAMPION_LEADS,
    REASON_NO_PROMOTABLE_CHALLENGER,
    REASON_LEDGER_MISSING,
    REASON_LEDGER_ARM_MISSING,
    REASON_INSUFFICIENT_WEEKS,
    REASON_MARGIN_NOT_MET,
    REASON_COOLDOWN_ACTIVE,
    REASON_CORROBORATION_DISAGREES,
    REASON_BOARD_DEFECTIVE,
)

# Retired with the v1 basis (alpha-engine-config-I8261). Kept as a NAMED set,
# not deleted outright, so a reader of an archived v1 record can resolve a slug
# this module no longer emits, and so a future author cannot re-mint one of
# these strings for a different condition. Asserted disjoint from the live set
# at import.
RETIRED_V1_REASON_CODES: tuple[str, ...] = (
    "board_missing",
    "board_unmeasurable",
    "decision_horizon_immature",
    "decision_horizon_unmeasurable",
    "arm_row_missing",
    "arm_metric_missing",
    "insufficient_dates",
)


class CutPromotionError(RuntimeError):
    """The engine could not complete an honest evaluation.

    Raised only AFTER the hold record carrying the defect has been written, so
    the failure is never the reason the evidence of it is missing.
    """


@dataclass(frozen=True)
class CutPromotionSlot:
    """The slot registry row for the scanner-cut promotion decision.

    champion-challenger-policy.md §10: *"Every slot names, in its registry, the
    metric, horizon, benchmark, count-matching width, and hysteresis margins it
    uses."* This is that row, and it is deliberately NOT an extension of
    ``LEADERBOARD_SLOTS["cuts"]``: that spec describes a MEASUREMENT surface
    whose arms are the funnel's own stages and explicitly *not competing*
    (``per_arm_width=True``). This slot is a subset of that board's arms — the
    count-matched 60s — which ARE competing. §2 forbids conflating the two;
    sharing one dataclass would have.
    """

    slot_id: str
    # The arms eligible to hold the feed. Resolved from PROMOTABLE_CUTS rather
    # than restated, so the writer and the reader (`live_cut_champion`) can
    # never disagree about who is eligible.
    arms: tuple[str, ...]
    default_champion: str

    # ── The decision basis (alpha-engine-config-I8261) ────────────────────────
    # WHERE the decision reads its evidence. This field, and the import-time
    # assertion pinning it to weekly_ledger.LEDGER_KEY, is the first of the
    # three invariants that replaced `forbidden_horizons_days`: no forward
    # horizon can be the decision basis because no leaderboard is.
    decision_source: str
    decision_cadence: str
    primary_metric: str
    # The ledger column differenced against the champion's SAME column in the
    # SAME week. Net, not gross: at weekly rebalance turnover is first-order and
    # the arms in this slot have wildly different churn (42% vs 76%
    # week-over-week retention, measured 2026-07-27, EXPERIMENTS.md). An arm
    # that wins gross and loses net is the classic trap, and deciding on gross
    # would be a worse answer than the forward returns it replaces because it
    # would look decisive.
    ledger_return_column: str
    # Evidence floor, in COMPLETED PAIRED WEEKS. Weeks, not cohort dates: the
    # observations abut rather than overlap, so each one is an independent
    # cluster in fact and not by assumption.
    min_weeks_for_inference: int

    # ── The vetoes (Brian's ruling: "corroborating vetoes once mature") ───────
    # A mature block at any of these horizons may BLOCK a promotion the weekly
    # series proposes. None of them may ever propose one — enforced
    # structurally, since they are consulted only after the weekly series has
    # named a leader. An immature one is recorded non-blocking (§5.1).
    corroborating_horizons_days: tuple[int, ...]
    corroborating_leaderboard_id: str
    corroborating_metric: str
    corroborating_min_dates: int

    # Horizons that are SCORED every cycle and are neither a decision input nor
    # a veto. Successor to `forbidden_horizons_days` and strictly stronger: the
    # old field barred 21d from PROPOSING, this one bars it from vetoing too.
    # Asserted disjoint from `corroborating_horizons_days` at import.
    excluded_horizons_days: tuple[int, ...]

    # ── Hysteresis (§5.2) ────────────────────────────────────────────────────
    # In the DECISION METRIC's own units: a mean weekly paired net log-return
    # difference vs the champion. See PROMOTION_MARGIN_NOTE for the conversion
    # from the retired 126-session units, which is carried on every record.
    promotion_margin: float
    cooldown_days: int

    # Arms that are SCORED every cycle but cannot hold the feed. Declared here
    # so a reader of this row can tell "measured and ineligible" from "not
    # measured at all" without going to another module (ARCHITECTURE §140).
    observe_only_arms: tuple[str, ...]
    # Every arm the ledger scores, promotable or not. Evidence is built for ALL
    # of these — champion-challenger-policy.md §3 makes measurement
    # unconditional — while only ``arms`` may win. Before
    # alpha-engine-config-I9272 the record carried evidence for the PROMOTABLE
    # arms only, so an excluded arm's numbers existed on the ledger and reached
    # no decision artifact: measured, and invisible where it mattered.
    scored_arms: tuple[str, ...]
    # Arm → the REASON it may not hold the feed, carried onto every record as
    # ``excluded_arms``. Empty under Brian's ruling 2026-08-29: every scored arm
    # of this slot is promotion-eligible. Kept as a mechanism so a future
    # carve-out is a stated property rather than an absence from a tuple.
    excluded_arms: Mapping[str, str]
    # §4 count-matching: every arm of the slot, promotable or not, is 60 by
    # construction.
    count_matched_width: int


# First cohort date on repaired fundamentals (alpha-engine-config-I8255): the
# vendor-fundamentals cross-section this slot's evidence is scored on was
# degenerate before this date. No weekly observation earlier than this is
# admissible evidence, so ``decision_earliest_on`` is derived from it rather
# than from whatever history a store happens to carry
# (alpha-engine-config-I8257).
FIRST_COHORT_DATE = date(2026, 8, 20)

# Retired hysteresis bar, kept as a literal so the conversion below is
# auditable rather than asserted. 0.005 was 50 bps of mean lift over a
# 126-session forward window.
LEGACY_MARGIN_PER_126_SESSIONS = 0.005
LEGACY_DECISION_HORIZON_DAYS = 126

PROMOTION_MARGIN_NOTE = (
    "Units: mean WEEKLY paired net log-return difference vs the champion "
    "(the decision metric's own units), NOT a lift over a forward window. "
    f"Derived to preserve the retired bar exactly: {LEGACY_MARGIN_PER_126_SESSIONS} "
    f"of mean lift over {LEGACY_DECISION_HORIZON_DAYS} sessions is "
    f"{LEGACY_DECISION_HORIZON_DAYS}/{SESSIONS_PER_WEEK} = "
    f"{LEGACY_DECISION_HORIZON_DAYS // SESSIONS_PER_WEEK}.2 weeks, so the "
    "same economic bar per unit of time is 0.005/25.2 ≈ 0.0002 per week "
    "(≈2 bps/week, ≈1.0%/yr at 52 weeks). The I8261 cutover deliberately "
    "changes the decision BASIS and not the BAR — importing a different bar "
    "under cover of a mechanism change would make the two effects "
    "indistinguishable afterwards. The margin is NOT a significance test and "
    "is not sized to the noise floor: champion-challenger-policy.md §5 "
    "rejects a publication-grade gate for an operational loop, and "
    "min_weeks_for_inference plus cooldown_days are what bound oscillation. "
    "The paired series' clustered mean/se/t_stat are recorded on every arm so "
    "a reader can see whether the margin was cleared with or without "
    "statistical support, without that being a gate."
)

# The forward horizons this slot SCORES but neither decides on nor vetoes with.
# Every date scored at 21d so far predates FIRST_COHORT_DATE, so it was measured
# on the pre-repair fundamentals cross-section (alpha-engine-config-I8255), and
# its reported t_stat used an iid standard error over overlapping windows that
# inflates |t| by roughly sqrt(lags+1) (alpha-engine-config-I8263, fixed in
# crucible-research-PR732). ``excluded_horizons`` carries this caveat WITH the
# number so a reader of the promotion record cannot mistake it for a clean read.
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
    min_weeks_for_inference=slot_spec("cuts").min_dates_for_inference,
    corroborating_horizons_days=(126, 252),
    corroborating_leaderboard_id="cuts",
    corroborating_metric=slot_spec("cuts").primary_metric,
    corroborating_min_dates=slot_spec("cuts").min_dates_for_inference,
    excluded_horizons_days=(21,),
    promotion_margin=0.0002,
    cooldown_days=28,
    count_matched_width=60,
)

# ── Import-time invariants ────────────────────────────────────────────────────
# Each is a way the engine could silently start deciding on the wrong evidence,
# so none is left to a test alone. The first three are the named successors to
# the retired ``forbidden_horizons_days`` assertion (see the module docstring).
if CUT_PROMOTION_SLOT.decision_source != LEDGER_KEY:
    raise AssertionError(
        "the scanner-cut decision must read the weekly ledger and nothing else "
        f"({LEDGER_KEY}). A decision sourced from a leaderboard horizon block "
        "measures a hold the weekly re-cut guarantees never happens "
        "(alpha-engine-config-I8261, Brian's ruling 2026-08-24)"
    )
if set(CUT_PROMOTION_SLOT.excluded_horizons_days) & set(
    CUT_PROMOTION_SLOT.corroborating_horizons_days
):
    raise AssertionError(
        "a horizon may not be BOTH structurally excluded and a corroborating "
        "veto — the 21-session block that produced alpha-engine-config-I7580 "
        "must not be able to block a promotion any more than propose one"
    )
for _h in CUT_PROMOTION_SLOT.corroborating_horizons_days:
    if _h < MIN_VETO_HORIZON_DAYS:
        raise AssertionError(
            f"corroborating horizon {_h} is below MIN_VETO_HORIZON_DAYS="
            f"{MIN_VETO_HORIZON_DAYS} — a veto horizon must match the scanner's "
            "~1-year objective (alpha-engine-config-I7580)"
        )
    if _h not in LONG_HORIZONS_DAYS:
        raise AssertionError(
            f"horizon {_h} is not scored by the cuts leaderboard "
            f"(LONG_HORIZONS_DAYS={LONG_HORIZONS_DAYS}) — the veto would read a "
            "block that is never written"
        )
if CUT_PROMOTION_SLOT.corroborating_metric != slot_spec("cuts").primary_metric:
    raise AssertionError(
        "the veto ranks on a metric the cuts board does not treat as primary — "
        "one of the two is wrong and it must not be resolved here"
    )
if CUT_PROMOTION_SLOT.ledger_return_column not in LEDGER_COLS:
    raise AssertionError(
        f"{CUT_PROMOTION_SLOT.ledger_return_column!r} is not a weekly-ledger "
        f"column ({LEDGER_COLS}) — the decision would read a field that is "
        "never written"
    )
if CUT_PROMOTION_SLOT.min_weeks_for_inference < 1:
    raise AssertionError("a decision needs at least one completed paired week")
if CUT_PROMOTION_SLOT.promotion_margin <= 0 or CUT_PROMOTION_SLOT.cooldown_days <= 0:
    raise AssertionError(
        "champion-challenger-policy.md §5.2 hysteresis is implemented for this "
        "slot, not waived under the §9.3 delta — both the margin and the "
        "cooldown must be positive"
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
if set(HOLD_REASON_CODES) & set(RETIRED_V1_REASON_CODES):
    raise AssertionError(
        "a retired v1 reason_code has been re-minted for a live condition — a "
        "slug means one thing forever or the decision series stops being "
        "readable across schema versions"
    )


@dataclass
class ArmEvidence:
    """What the WEEKLY LEDGER says about one arm, paired against the champion.

    Every field is self-qualifying by construction (alpha-engine-config-I8257,
    carried forward to the new basis): ``metric``, ``cadence`` and ``source``
    travel with the numbers, so ``n_weeks_paired: 0`` reads as "0 paired weeks
    on the weekly ledger", never as an unqualified zero a reader has to join
    back to a registry to interpret.
    """

    present: bool = False
    is_champion: bool = False
    # How many ledger rows this arm has at the current LEDGER_VERSION.
    n_weeks_scored: int = 0
    # How many of those pair against a champion row for the SAME week with both
    # legs carrying the decision column. This is the measurability surface.
    n_weeks_paired: int = 0
    weeks_dropped_unpaired: int = 0
    weeks_dropped_window_mismatch: int = 0
    weeks_dropped_stale_version: int = 0
    # Weeks whose two rows PAIRED — same span, both present — but where one leg
    # carried no number in the decision column. Split out from
    # ``weeks_dropped_unpaired`` by alpha-engine-config-I9284: a missing cut and
    # an uncomputable transaction cost are different faults and rendered
    # identically before this field existed.
    weeks_dropped_null_decision_column: int = 0
    # ── Eligibility, as a STATE (alpha-engine-config-I9272) ───────────────────
    # Whether this arm could hold the feed if it won, and if not, WHY. Never an
    # absence: an arm the engine will not promote appears on the record saying
    # so, because "measured and ineligible" and "not measured" are different
    # answers to the only question a reader of this artifact is asking
    # (champion-challenger-policy.md §3; ARCHITECTURE §140).
    eligible_for_promotion: bool = True
    ineligibility_reason: str | None = None
    # The decision number, and the chained read of the same series.
    mean_paired_log_return: float | None = None
    chained_paired_log_return: float | None = None
    se: float | None = None
    t_stat: float | None = None
    se_method: str | None = None
    first_week: str | None = None
    last_week: str | None = None
    confidence: str = "insufficient"
    # DIAGNOSTIC, never the decision input. `weekly_ledger.paired_weekly_
    # differences` differences this arm's chosen column against the
    # `champion_log_return` leg carried inside the arm's OWN row — and that leg
    # is a GROSS basket return (weekly_ledger.build_week_row computes it with
    # equal_weight_log_return and applies no cost). Differencing a NET arm
    # against a GROSS champion charges the challenger's transaction cost and
    # not the incumbent's, which is a systematic bias against the challenger in
    # a slot whose arms differ mainly in churn. The decision therefore joins
    # champion rows explicitly and differences net-against-net; this field
    # records what the embedded leg would have said, so the gap is visible
    # rather than argued about. Producer-side fix tracked with I8264.
    mean_vs_embedded_champion_leg: float | None = None
    n_weeks_vs_embedded_champion_leg: int = 0
    metric: str = ""
    cadence: str = ""
    source: str = ""


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
    decision_earliest_on: dict[str, Any] = field(default_factory=dict)
    excluded_arms: dict[str, dict] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)

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
            "last_promoted_on": self.last_promoted_on,
            "leaderboard_key": leaderboard_key,
            "arms": {name: asdict(ev) for name, ev in self.arms.items()},
            "hysteresis": {
                "promotion_margin": slot.promotion_margin,
                "promotion_margin_units": (
                    "mean weekly paired net log-return difference vs champion"
                ),
                "promotion_margin_note": PROMOTION_MARGIN_NOTE,
                "cooldown_days": slot.cooldown_days,
                "min_weeks_for_inference": slot.min_weeks_for_inference,
                "corroborating_horizons_days": list(slot.corroborating_horizons_days),
            },
            "corroborating": self.corroborating,
            "defect": self.defect,
            "excluded_horizons": self.excluded_horizons,
            "decision_earliest_on": self.decision_earliest_on,
            # Every arm the ledger scores that may NOT hold the feed, with
            # the reason. Empty under Brian's ruling 2026-08-29; present as
            # a field on every record regardless, because a reader must be
            # able to tell "no arm is excluded" from "this record does not
            # say" (alpha-engine-config-I9272).
            "excluded_arms": self.excluded_arms,
            "scored_arms": list(CUT_PROMOTION_SLOT.scored_arms),
            "promotable_arms": list(CUT_PROMOTION_SLOT.arms),
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


def _paired_series(
    arm_rows: Sequence[Mapping[str, Any]],
    champion_rows: Sequence[Mapping[str, Any]],
    *,
    column: str,
) -> tuple[list[float], list[str], int, int, int]:
    """``(differences, weeks, dropped_unpaired, dropped_window_mismatch,
    dropped_null_decision_column)``.

    The last count is split out of ``dropped_unpaired`` by
    alpha-engine-config-I9284. "The week did not pair" and "the week had a
    paired row that carried no NUMBER in the decision column" are different
    conditions with different fixes — the first is a missing cut, the second is
    an uncomputable transaction cost — and folding them into one counter made
    them render identically. The ledger's first week (2026-08-21 → 08-28) is
    entirely of the second kind: every arm but the champion carries
    ``net_log_return: null`` with ``net_unavailable_reason:
    turnover_unknown_so_cost_uncomputable``, because ``_arm_turnover`` reads
    ``turnover.per_cut`` off an artifact that predates those arms.

    NET against NET, both legs read from the two arms' own ledger rows and
    joined on ``week_start``. See ``ArmEvidence.mean_vs_embedded_champion_leg``
    for why the ledger's in-row ``champion_log_return`` is a diagnostic and not
    this.

    A week where either leg is missing the column is DROPPED and counted, never
    substituted with a zero: substituting would manufacture a week in which the
    arm exactly matched the champion.

    A week whose two rows disagree about the span they cover is also dropped and
    counted separately (champion-challenger-policy.md §4, same cohort dates).
    The comparison is over ``week_end`` AND the actually-priced boundaries
    ``priced_from`` / ``priced_to`` — added by alpha-engine-config-I8264, which
    established that the two can diverge whenever a cut date is not itself a
    session or a closing bar has not landed. Checking only the LABEL would let
    two arms priced over different spans difference against each other while
    both rows agreed about what they claimed to cover, which is the defect that
    field was introduced to make visible. Rows written before those columns
    existed carry ``None`` on both legs and still agree, so an older ledger
    reconciles rather than emptying itself.
    """
    champ_by_week = {str(r.get("week_start")): r for r in champion_rows}
    diffs: list[float] = []
    weeks: list[str] = []
    unpaired = 0
    mismatched = 0
    null_column = 0
    for row in arm_rows:
        week = str(row.get("week_start"))
        champ = champ_by_week.get(week)
        if champ is None:
            unpaired += 1
            continue
        if any(
            row.get(f) != champ.get(f)
            for f in ("week_end", "priced_from", "priced_to")
        ):
            mismatched += 1
            continue
        mine = row.get(column)
        theirs = champ.get(column)
        if mine is None or theirs is None:
            # Both rows exist and cover the same span — the week PAIRED. What
            # is missing is the number, so this is not an unpaired week
            # (alpha-engine-config-I9284).
            null_column += 1
            continue
        try:
            diffs.append(float(mine) - float(theirs))
        except (TypeError, ValueError):
            unpaired += 1
            continue
        weeks.append(week)
    return diffs, weeks, unpaired, mismatched, null_column


def _arm_evidence(
    *,
    arm: str,
    arm_rows: Sequence[Mapping[str, Any]],
    champion_rows: Sequence[Mapping[str, Any]],
    stale: int,
    slot: CutPromotionSlot,
    is_champion: bool,
) -> ArmEvidence:
    """One arm's paired weekly evidence. Pure."""
    diffs, weeks, unpaired, mismatched, null_column = _paired_series(
        arm_rows, champion_rows, column=slot.ledger_return_column
    )
    # `overlap_lags=0` is a CLAIM, and it is true here for the first time: the
    # ledger's weeks abut (each ends where the next begins), so "each date =
    # one independent cluster" — `date_clustered_stats`'s own contract, written
    # for exactly this shape — holds in fact rather than by assumption. The
    # same call on a forward-window series would need a HAC SE
    # (alpha-engine-config-I8263).
    stats = date_clustered_stats(diffs, overlap_lags=0) if diffs else None
    embedded = paired_weekly_differences(
        arm_rows, column=slot.ledger_return_column
    )
    return ArmEvidence(
        present=True,
        is_champion=is_champion,
        n_weeks_scored=len(arm_rows),
        n_weeks_paired=len(diffs),
        weeks_dropped_unpaired=unpaired,
        weeks_dropped_window_mismatch=mismatched,
        weeks_dropped_stale_version=stale,
        weeks_dropped_null_decision_column=null_column,
        eligible_for_promotion=(arm not in slot.excluded_arms),
        ineligibility_reason=slot.excluded_arms.get(arm),
        mean_paired_log_return=(stats or {}).get("mean"),
        chained_paired_log_return=chained_log_return(diffs) if diffs else None,
        se=(stats or {}).get("se"),
        t_stat=(stats or {}).get("t_stat"),
        se_method=(stats or {}).get("se_method"),
        first_week=(weeks[0] if weeks else None),
        last_week=(weeks[-1] if weeks else None),
        confidence=confidence_for(len(diffs), slot.min_weeks_for_inference),
        mean_vs_embedded_champion_leg=(
            sum(embedded) / len(embedded) if embedded else None
        ),
        n_weeks_vs_embedded_champion_leg=len(embedded),
        metric=slot.primary_metric,
        cadence=slot.decision_cadence,
        source=slot.decision_source,
    )


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
        "role": "veto_only",
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
    """Every veto horizon, and whether any of them blocks ``proposed``."""
    horizons = {
        str(h): _veto_horizon(board, slot, proposed, h)
        for h in slot.corroborating_horizons_days
    }
    blocked_by = [
        int(h) for h, entry in horizons.items() if entry.get("disagrees")
    ]
    mature = [int(h) for h, entry in horizons.items() if entry.get("mature")]
    return {
        "role": "veto_only",
        "proposed": proposed,
        "horizons": horizons,
        "mature_horizons": sorted(mature),
        "blocking": bool(blocked_by),
        "blocked_by": sorted(blocked_by),
        "note": (
            "126 and 252 are corroborating vetoes, not the decision basis "
            "(Brian's ruling 2026-08-24, alpha-engine-config-I8261). A mature "
            "block may refuse a promotion the weekly series proposes; none may "
            "propose one. An immature block is recorded non-blocking."
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


def first_decidable_week(
    ledger_rows: Sequence[Mapping[str, Any]] | None,
    *,
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
) -> str | None:
    """``week_start`` of the earliest week in which EVERY promotable arm carries
    a non-null decision column, or ``None`` when no such week exists yet.

    The counting origin for :func:`decision_earliest_on`
    (alpha-engine-config-I9284). A week in which a challenger's decision column
    is null contributes nothing to that challenger's paired series, so it can
    never count toward ``min_weeks_for_inference`` — and projecting the floor
    from :data:`FIRST_COHORT_DATE` regardless is how the record came to publish
    ``decision_earliest_on: 2026-09-25`` on evidence whose first week paired for
    nobody.
    """
    if not ledger_rows:
        return None
    column = slot.ledger_return_column
    by_week: dict[str, set[str]] = {}
    for row in ledger_rows:
        version = row.get("ledger_version")
        if version is not None and int(version) != LEDGER_VERSION:
            continue
        arm = row.get("arm")
        if arm not in slot.arms:
            continue
        if _clean(row.get(column)) is None:
            continue
        by_week.setdefault(str(row.get("week_start")), set()).add(str(arm))
    complete = [w for w, arms in by_week.items() if arms >= set(slot.arms)]
    return min(complete) if complete else None


def decision_earliest_on(
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
    *,
    ledger_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The earliest date the weekly series could carry ``min_weeks_for_inference``
    paired observations, and whether that date is PROVISIONAL.

    Derived from the WEEKLY series (alpha-engine-config-I8261 requirement 2),
    not from a forward horizon: ``min_weeks_for_inference`` weeks of
    ~``SESSIONS_PER_WEEK`` trading sessions each, projected onto the NYSE
    calendar so holidays are priced in. The v1 basis put this at 2027-02-22
    (126 sessions after the same start); the weekly basis puts it five months
    earlier.

    **What alpha-engine-config-I9284 changed.** This used to count from
    :data:`FIRST_COHORT_DATE` unconditionally, which published a floor the
    ledger could not reach: the ledger's first week (2026-08-21 → 08-28) carries
    a null decision column for every arm but the champion, so it contributes
    zero paired observations and is permanently unusable — ledger rows are
    immutable. The counting origin is now the first week in which every
    promotable arm actually carries a number (:func:`first_decidable_week`), and
    while no such week exists the answer is marked ``provisional: true`` with
    the basis named. ``decision_earliest_on`` is the field a reader uses to tell
    a working loop from a stuck one, so a date that silently slips is precisely
    the failure it exists to prevent.

    A ceiling, not a promise, in either mode: it says the evidence CANNOT exist
    before this date, never that it will exist on it. A missed weekly run pushes
    the real date out, and the ledger's own ``n_weeks_paired`` is what says
    where it actually stands.
    """
    observed = first_decidable_week(ledger_rows, slot=slot)
    origin = date.fromisoformat(observed) if observed else FIRST_COHORT_DATE
    projected = add_trading_days(
        origin, slot.min_weeks_for_inference * SESSIONS_PER_WEEK
    ).isoformat()
    return {
        "date": projected,
        "provisional": observed is None,
        "counted_from": origin.isoformat(),
        "basis": (
            "first ledger week in which every promotable arm carries a non-null "
            f"{slot.ledger_return_column}"
            if observed
            else (
                "PROVISIONAL — no ledger week yet carries a non-null "
                f"{slot.ledger_return_column} for every promotable arm, so this "
                f"counts from FIRST_COHORT_DATE ({FIRST_COHORT_DATE.isoformat()}) "
                "and will move OUT, never in, as the real first decidable week "
                "lands later (alpha-engine-config-I9284)"
            )
        ),
        "min_weeks_for_inference": slot.min_weeks_for_inference,
    }


def _leader(arms: dict[str, ArmEvidence]) -> str:
    """The arm with the highest mean paired difference. The champion's own
    paired difference is 0.0 by construction (it is differenced against
    itself), so a challenger must be strictly positive to lead — and a tie
    resolves to the incumbent, which is what hysteresis means before the margin
    is even consulted.

    The ``is None`` test is deliberate and load-bearing: the champion's own mean
    is EXACTLY 0.0, and ``x or -inf`` sends a falsy zero to negative infinity —
    which would rank the incumbent below every challenger, including one that is
    losing. Caught by ``test_an_exact_tie_resolves_to_the_incumbent`` before this
    shipped; an unmeasured arm (``None``) is what ranks last, not a flat one.
    """

    def rank(kv: tuple[str, ArmEvidence]) -> tuple[float, int, str]:
        mean = kv[1].mean_paired_log_return
        # Incumbency, not alphabetical order, breaks a tie. Ranking on the name
        # made 'tech_score_top_60' beat 'attractiveness_top_60' on an EXACT
        # draw, which moves the sector-team feed on no evidence at all — the
        # opposite of what hysteresis is for.
        return (
            float("-inf") if mean is None else float(mean),
            1 if kv[1].is_champion else 0,
            kv[0],
        )

    return max(arms.items(), key=rank)[0]


def decide_cut_champion(
    *,
    ledger_rows: Sequence[Mapping[str, Any]] | None,
    board: dict | None = None,
    champion_before: str,
    decided_on: str,
    last_promoted_on: str | None = None,
    cycle_ineligible_arms: Mapping[str, str] | None = None,
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
) -> CutPromotionDecision:
    """Decide, from an already-loaded weekly ledger. Pure: no S3, no clock.

    ``ledger_rows`` is the DECISION evidence. ``board`` — a cuts leaderboard —
    is consulted only for the long-horizon vetoes and the excluded-horizon
    report; its absence is not a hold, because an unmeasured veto is not a veto
    (§5.1).

    ``cycle_ineligible_arms`` is arm → reason for arms this CYCLE cannot promote
    to, as opposed to arms the REGISTER excludes permanently
    (``slot.excluded_arms``). Today its only producer is
    ``universe_membership.promotion_ineligibility_from_rank_tables``: an arm
    whose basis has no full-universe rank table would resolve a rank ceiling in
    a consumer, on the morning the promotion was made
    (alpha-engine-config-I7843). Both kinds land on the record as
    ``eligible_for_promotion: false`` with a reason — never as an absence
    (alpha-engine-config-I9272).

    Every exit produces a record. There is no path that returns nothing, and no
    path that promotes without every arm clearing ``min_weeks_for_inference``
    paired weeks on the decision column.
    """

    # Computed once, up front: neither depends on which branch below fires, and
    # both must be on EVERY record — including the earliest holds, where a
    # reader most needs to know this is not a stuck loop.
    earliest = decision_earliest_on(slot, ledger_rows=ledger_rows)
    cycle_ineligible = dict(cycle_ineligible_arms or {})
    excluded_arms = {
        arm: {"arm": arm, "reason": reason, "scored": True, "scope": "register"}
        for arm, reason in slot.excluded_arms.items()
    } | {
        arm: {"arm": arm, "reason": reason, "scored": True, "scope": "this_cycle"}
        for arm, reason in cycle_ineligible.items()
        if arm not in slot.excluded_arms
    }
    excluded = _excluded_horizons(board, slot)

    def _ledger_meta(rows, arms_seen=None, present=True) -> dict:
        return {
            "key": slot.decision_source,
            "ledger_version": LEDGER_VERSION,
            "present": present,
            "rows_read": (len(rows) if rows is not None else 0),
            "column": slot.ledger_return_column,
            "arms_present": sorted(arms_seen or []),
        }

    def hold(
        code: str,
        reason: str,
        *,
        arms=None,
        corroborating=None,
        defect=None,
        ledger=None,
    ):
        return CutPromotionDecision(
            decision=DECISION_HOLD,
            champion=champion_before,
            champion_before=champion_before,
            reason=reason,
            reason_code=code,
            decided_on=decided_on,
            arms=arms
            or {
                a: ArmEvidence(
                    metric=slot.primary_metric,
                    cadence=slot.decision_cadence,
                    source=slot.decision_source,
                    is_champion=(a == champion_before),
                    eligible_for_promotion=(
                        a not in slot.excluded_arms and a not in cycle_ineligible
                    ),
                    ineligibility_reason=(
                        slot.excluded_arms.get(a) or cycle_ineligible.get(a)
                    ),
                )
                for a in slot.scored_arms
            },
            last_promoted_on=last_promoted_on,
            corroborating=corroborating,
            defect=defect,
            excluded_horizons=excluded,
            decision_earliest_on=earliest,
            excluded_arms=excluded_arms,
            ledger=ledger if ledger is not None else _ledger_meta(ledger_rows or []),
        )

    # ── WHOLE-BOARD integrity, before anything else ────────────────────────────
    # The board is now only the VETO source, and a missing one is legitimately
    # non-blocking. A board reporting duplicate arm rows is a different thing:
    # it is not unmeasured, it is UNRELIABLE, and a safety mechanism that may be
    # reading someone else's numbers is worse than one that is switched off. So
    # a duplicate anywhere disqualifies the board and holds — the engine cannot
    # establish that the rows it DOES read came from the pass it thinks they
    # did (alpha-engine-config-I8026 deliverable 3; the 2026-08-18/19 artifacts
    # doubled two funnel stages in the 21d block alone).
    #
    # Ordered ahead of the registry check because a producer fault is true
    # whether or not a decision was available to take, and rendering it as the
    # quiet `no_promotable_challenger` is this module's own stated worse
    # failure: "a defect that also erases the evidence of itself".
    if board:
        board_dupes = duplicate_arm_rows(board)
        if board_dupes:
            return hold(
                REASON_BOARD_DEFECTIVE,
                f"cuts leaderboard {decided_on} reports duplicate arm rows "
                f"({', '.join(board_dupes)}) — including on surfaces this engine "
                "does not veto from. A board that counts any arm twice cannot be "
                "shown to have counted the others once, so the long-horizon veto "
                "cannot be trusted this cycle. An ABSENT veto is non-blocking; a "
                f"CORRUPT one is not. {champion_before!r} holds and the run fails "
                "loud.",
                defect=f"duplicate arm rows: {', '.join(board_dupes)}",
            )

    # ── The slot has no promotable challenger ─────────────────────────────────
    # Brian's ruling 2026-08-29 (alpha-engine-config-I9272) makes every scored
    # arm of this slot promotion-eligible, so this branch is UNREACHABLE with
    # today's register — five arms, no exclusions. It is kept, and kept loud,
    # because the condition it names is a real one: if a future exclusion ever
    # shrinks PROMOTABLE_CUTS back to a single arm, the engine must SAY there
    # was nothing to decide rather than fall through to a comparison path and
    # report `champion_already_leads`, which is a claim about evidence made
    # where no comparison happened. Two states that mean different things must
    # not render identically (champion-challenger-policy.md §3).
    #
    # And it now names the EXCLUSIONS WITH THEIR REASONS rather than a bare
    # list of observe-only arms, because the register no longer expresses
    # non-promotability as an absence.
    if len(slot.arms) < 2:
        excluded_note = (
            "; ".join(f"{a}: {r}" for a, r in sorted(slot.excluded_arms.items()))
            or "none — every scored arm is promotion-eligible, so this hold "
            "means the slot itself has only one arm"
        )
        return hold(
            REASON_NO_PROMOTABLE_CHALLENGER,
            f"the scanner-cut slot has one promotable arm ({champion_before!r}) "
            f"and no promotable challenger, so there is no decision to take on "
            f"{decided_on} — on any metric, at any cadence. Scored arms: "
            f"{', '.join(slot.scored_arms)}. Excluded from promotion, with "
            f"reasons: {excluded_note}. The weekly evidence accumulates either "
            "way; what is missing is a second arm allowed to win.",
        )

    # ── The weekly ledger ──────────────────────────────────────────────────────
    if ledger_rows is None:
        return hold(
            REASON_LEDGER_MISSING,
            f"no weekly ledger at {slot.decision_source} on {decided_on}, so the "
            f"decision metric ({slot.primary_metric}) has no observations. "
            f"{champion_before!r} holds. This is a hold, not a silence: the "
            "record you are reading is the proof the engine ran. The ledger's "
            "producer is being wired under alpha-engine-config-I8264; until it "
            "writes, an absent store is the EXPECTED state and must never "
            "render as an empty-but-healthy series "
            "(champion-challenger-policy.md §7.2).",
            ledger=_ledger_meta(None, present=False),
        )

    per_arm: dict[str, list[dict]] = {}
    stale_counts: dict[str, int] = {}
    for arm in slot.scored_arms:
        per_arm[arm], stale_counts[arm] = _rows_by_arm(ledger_rows, arm)
    ledger_meta = _ledger_meta(
        ledger_rows, arms_seen=[a for a in slot.scored_arms if per_arm[a]]
    )

    champion_rows = per_arm.get(champion_before) or []

    # ── The CHAMPION's own rows are the only slot-wide precondition ───────────
    # Every difference is taken against this leg, so without it there is no
    # comparison to make for anybody. A CHALLENGER with no rows is a different
    # thing entirely: it is that arm's own miss, recorded on that arm
    # (champion-challenger-policy.md §3, "a cycle where an arm produces no
    # output is recorded as a miss, not omitted"), and it must not stop the
    # arms that DID produce output from being compared.
    #
    # This is the alpha-engine-config-I9272 correction generalised. Holding the
    # whole slot on one arm's absence is the same defect as excluding an arm
    # from the register: in both cases a slot with real evidence renders as a
    # slot with nothing to say. With five arms rather than one, an all-or-
    # nothing precondition would make `attractiveness_hard3_top_60` — which
    # emitted zero names in the ledger's first week — able to freeze the
    # decision indefinitely on its own.
    if not champion_rows:
        stale_note = (
            f" {stale_counts[champion_before]} row(s) set aside at an older "
            "ledger_version."
            if stale_counts.get(champion_before)
            else ""
        )
        return hold(
            REASON_LEDGER_ARM_MISSING,
            f"the incumbent {champion_before!r} has no weekly-ledger row at "
            f"ledger_version={LEDGER_VERSION} on {decided_on}. Every arm's "
            "decision number is a difference against the champion's leg for the "
            "SAME week, so an absent incumbent leg is not a thin comparison, it "
            f"is no comparison at all. {champion_before!r} holds." + stale_note,
            ledger=ledger_meta,
        )

    arms: dict[str, ArmEvidence] = {}
    for arm in slot.scored_arms:
        if not per_arm[arm]:
            # Recorded as a MISS with its own reason, never omitted and never
            # silently rendered as a zero.
            arms[arm] = ArmEvidence(
                present=False,
                is_champion=(arm == champion_before),
                weeks_dropped_stale_version=stale_counts[arm],
                metric=slot.primary_metric,
                cadence=slot.decision_cadence,
                source=slot.decision_source,
                eligible_for_promotion=(
                    arm not in slot.excluded_arms and arm not in cycle_ineligible
                ),
                ineligibility_reason=(
                    slot.excluded_arms.get(arm) or cycle_ineligible.get(arm)
                ),
            )
            continue
        arms[arm] = _arm_evidence(
            arm=arm,
            arm_rows=per_arm[arm],
            champion_rows=champion_rows,
            stale=stale_counts[arm],
            slot=slot,
            is_champion=(arm == champion_before),
        )
        if arm in cycle_ineligible and arm not in slot.excluded_arms:
            arms[arm].eligible_for_promotion = False
            arms[arm].ineligibility_reason = cycle_ineligible[arm]

    # ── Maturity is a PER-ARM property, not a slot-wide gate ──────────────────
    # alpha-engine-config-I9284. The floor used to be applied to every arm at
    # once: one arm short of `min_weeks_for_inference` held the entire slot, on
    # any evidence, forever. With one promotable arm that was invisible. With
    # five it is a live deadlock — `attractiveness_hard3_top_60` first emitted
    # on 2026-08-28 and produced zero names in the ledger's first week, so an
    # all-arms floor would have made `decision_earliest_on` unreachable by
    # construction and every future hold would have blamed the calendar for a
    # register problem.
    #
    # An immature arm is now recorded ineligible FOR THIS CYCLE, with its count
    # and the floor it missed, and it keeps accruing. The slot decides as soon
    # as the incumbent and at least one eligible challenger are both mature —
    # which is the smallest set on which a promotion could honestly be taken.
    champion_ev = arms[champion_before]
    for name, ev in arms.items():
        # Only the PROMOTABLE arms carry an eligibility verdict at all. A
        # scored-but-excluded arm already carries its register reason and must
        # not have it overwritten by a maturity one — the register exclusion is
        # the binding fact and the thin series is downstream of it.
        if name not in slot.arms:
            ev.eligible_for_promotion = False
            ev.ineligibility_reason = ev.ineligibility_reason or (
                "not_promotable: absent from PROMOTABLE_CUTS"
            )
            continue
        if not ev.eligible_for_promotion:
            continue
        if ev.n_weeks_paired < slot.min_weeks_for_inference:
            ev.eligible_for_promotion = False
            ev.ineligibility_reason = (
                f"{REASON_INSUFFICIENT_WEEKS}: n_weeks_paired="
                f"{ev.n_weeks_paired} < min_weeks_for_inference="
                f"{slot.min_weeks_for_inference}"
                + (
                    f" ({ev.weeks_dropped_null_decision_column} week(s) paired "
                    f"but carried no {slot.ledger_return_column})"
                    if ev.weeks_dropped_null_decision_column
                    else ""
                )
            )

    contenders = {
        name: arms[name]
        for name in slot.arms
        if name in arms and arms[name].eligible_for_promotion and name != champion_before
    }
    if not contenders or not champion_ev.eligible_for_promotion:
        counts = ", ".join(
            f"{a}.n_weeks_paired={arms[a].n_weeks_paired}"
            for a in slot.arms
            if a in arms
        )
        blocked = "; ".join(
            f"{a}: {arms[a].ineligibility_reason}"
            for a in slot.arms
            if a in arms and arms[a].ineligibility_reason
        )
        return hold(
            REASON_INSUFFICIENT_WEEKS,
            f"no eligible challenger has {slot.min_weeks_for_inference} paired "
            f"weeks against the incumbent {champion_before!r} on {decided_on} "
            f"({counts}). Below that floor a mean of paired weekly differences "
            "is an anecdote, not an inference (alpha-engine-config-I7542). "
            f"{champion_before!r} holds. The evidence cannot exist before "
            f"{earliest['date']}"
            + (
                " — PROVISIONAL: no ledger week yet carries a number for every "
                "promotable arm, so that date counts from FIRST_COHORT_DATE and "
                "will move out, not in (alpha-engine-config-I9284)"
                if earliest["provisional"]
                else f" (counting from the first fully-priced week "
                f"{earliest['counted_from']})"
            )
            + ". A run of holds until then is the loop working, not stuck. "
            "Per-arm: " + (blocked or "no arm carries an ineligibility reason."),
            arms=arms,
            ledger=ledger_meta,
        )

    leader = _leader({champion_before: champion_ev, **contenders})
    if leader == champion_before:
        champ_note = (
            ", ".join(
                f"{a}={arms[a].mean_paired_log_return:+.6f}"
                for a in slot.scored_arms
                if a != champion_before
                and a in arms
                and arms[a].mean_paired_log_return is not None
            )
            or "no challenger measured"
        )
        return hold(
            REASON_CHAMPION_LEADS,
            f"no challenger has a positive mean weekly paired difference against "
            f"the incumbent {champion_before!r} over "
            f"{arms[champion_before].n_weeks_paired} weeks ({champ_note}). "
            "Nothing to promote.",
            arms=arms,
            ledger=ledger_meta,
        )

    # The leader's mean IS the margin: the series is already differenced against
    # the champion, so no second subtraction is needed and none is done. That is
    # the point of pairing — the common market factor is gone from the number
    # the margin is compared against.
    margin = float(arms[leader].mean_paired_log_return or 0.0)
    chained = arms[leader].chained_paired_log_return

    # The vetoes, evaluated BEFORE the margin so a disagreement is reported as a
    # disagreement rather than swallowed by a margin failure.
    corroborating = _corroboration(board, slot, leader)
    if corroborating["blocking"]:
        blocked = ", ".join(f"{h}d" for h in corroborating["blocked_by"])
        leaders = ", ".join(
            f"{h}d→{corroborating['horizons'][h]['leader']!r}"
            for h in sorted(corroborating["horizons"])
            if corroborating["horizons"][h].get("disagrees")
        )
        return hold(
            REASON_CORROBORATION_DISAGREES,
            f"{leader!r} leads the weekly series by {margin:+.6f}/week over "
            f"{arms[leader].n_weeks_paired} paired weeks, but the mature "
            f"{blocked} horizon puts a different arm ahead ({leaders}). A "
            "corroborating horizon holds a veto and not a vote: it cannot "
            "propose a promotion, and it can refuse one "
            f"(alpha-engine-config-I7580). {champion_before!r} holds.",
            arms=arms,
            corroborating=corroborating,
            ledger=ledger_meta,
        )

    if margin < slot.promotion_margin:
        return hold(
            REASON_MARGIN_NOT_MET,
            f"{leader!r} leads {champion_before!r} by {margin:+.6f} per week "
            f"(chained {chained:+.6f} over {arms[leader].n_weeks_paired} weeks), "
            f"under the promotion margin {slot.promotion_margin:+.6f} in the same "
            "units — a mean weekly paired net log-return difference "
            "(champion-challenger-policy.md §5.2 hysteresis). Leading is not "
            "enough; a feed that oscillates on noise makes the sector teams' "
            "work incomparable week to week.",
            arms=arms,
            corroborating=corroborating,
            ledger=ledger_meta,
        )

    if last_promoted_on:
        elapsed = (
            date.fromisoformat(decided_on) - date.fromisoformat(last_promoted_on)
        ).days
        if elapsed < slot.cooldown_days:
            return hold(
                REASON_COOLDOWN_ACTIVE,
                f"{leader!r} clears the margin ({margin:+.6f}/week) but the "
                f"pointer last moved {elapsed}d ago on {last_promoted_on}, inside "
                f"the {slot.cooldown_days}d cooldown (§5.2). Held; the challenger "
                "keeps accruing weekly evidence and is re-evaluated next cycle.",
                arms=arms,
                corroborating=corroborating,
                ledger=ledger_meta,
            )

    mature = corroborating["mature_horizons"]
    veto_note = (
        f"corroborating horizons {mature} mature and not contradicting"
        if mature
        else (
            "no corroborating horizon is mature yet, so none applied — an "
            "unmeasured veto is not a veto (§5.1) and is never counted as a pass"
        )
    )
    return CutPromotionDecision(
        decision=DECISION_PROMOTE,
        champion=leader,
        champion_before=champion_before,
        reason=(
            f"{leader!r} beats {champion_before!r} on {slot.primary_metric} by "
            f"{margin:+.6f} per week (chained {chained:+.6f} over "
            f"{arms[leader].n_weeks_paired} paired weeks, "
            f"{arms[leader].first_week}→{arms[leader].last_week}), at or above "
            f"the margin {slot.promotion_margin}; both the incumbent and the "
            f"winner cleared ≥{slot.min_weeks_for_inference} paired weeks "
            "(arms short of the floor are recorded ineligible for this cycle "
            "and keep accruing, alpha-engine-config-I9284); "
            f"{veto_note}; cooldown clear."
        ),
        reason_code=REASON_PROMOTED,
        decided_on=decided_on,
        arms=arms,
        last_promoted_on=decided_on,
        corroborating=corroborating,
        excluded_horizons=excluded,
        decision_earliest_on=earliest,
        excluded_arms=excluded_arms,
        ledger=ledger_meta,
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
    """Every ``arms.<arm>.n_weeks_paired`` in a WRITTEN record must equal what
    the ledger it cites actually supports.

    Successor to ``reconcile_arms_with_leaderboard``, which checked the same
    property against the v1 decision source (alpha-engine-config-I8257). The
    source moved; the property did not, and deleting the guard with the source
    would have removed the only thing that catches a record and its own cited
    evidence disagreeing — the misread this module was originally filed to fix.

    Returns the list of mismatches; empty means every arm reconciles.
    """
    if ledger_rows is None:
        return []
    champion = doc.get("champion_before")
    column = doc.get("decision_column") or CUT_PROMOTION_SLOT.ledger_return_column
    champion_rows, _ = _rows_by_arm(ledger_rows, str(champion))
    mismatches: list[str] = []
    for arm, ev in (doc.get("arms") or {}).items():
        if not ev.get("present"):
            # present=False means the decision path never read ledger rows for
            # this arm at all (ledger absent, the registry-only short-circuit,
            # a defective board). The record is not claiming n_weeks_paired
            # reflects the ledger, so there is nothing here to reconcile.
            continue
        arm_rows, _ = _rows_by_arm(ledger_rows, arm)
        diffs, _weeks, _unpaired, _mismatched, _null_col = _paired_series(
            arm_rows, champion_rows, column=column
        )
        if ev.get("n_weeks_paired") != len(diffs):
            mismatches.append(
                f"{arm}: record reports n_weeks_paired={ev.get('n_weeks_paired')} "
                f"on column {column!r}, but the cited ledger supports "
                f"{len(diffs)} paired week(s) against {champion!r}"
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
    """Decide and WRITE, unconditionally. Returns the written document.

    ``leaderboard`` lets the caller hand in the board it just built (the scanner
    handler does), so the vetoes read the exact artifact this run produced
    rather than re-fetching a key that may not have landed. ``ledger_rows``
    likewise; ``_UNSET`` (the default) reads the ledger from S3, and an explicit
    ``None`` means "no ledger", which is a hold. The two are distinguished on
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

    decision = decide_cut_champion(
        ledger_rows=rows,
        board=board,
        cycle_ineligible_arms=cycle_ineligible,
        champion_before=champion_before,
        decided_on=decided_on,
        last_promoted_on=last_promoted_on,
        slot=slot,
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
        "reason_code=%s metric=%s cadence=%s %s",
        doc["decision"],
        doc["champion"],
        doc["reason_code"],
        doc["decision_metric"],
        doc["decision_cadence"],
        " ".join(
            f"{a}_n_weeks_paired={doc['arms'][a]['n_weeks_paired']}"
            for a in slot.arms
            if a in doc["arms"]
        ),
    )

    # ── Delivery (alpha-engine-config-I9278) ──────────────────────────────────
    # AFTER the three writes, so the email can never be the reason a record is
    # missing; and BEFORE the defect raise, because a cycle that held on a
    # DEFECTIVE board is precisely the cycle Brian most needs delivered, and a
    # raise above this line would send nothing. `send_verdict_digest` escalates
    # its OWN failure to an ops alert and returns False rather than raising, so
    # a notification can never red a promotion run.
    send_verdict_digest(doc, VERDICT_SLOT)

    if decision.defect:
        raise CutPromotionError(
            f"scanner-cut promotion held on a DEFECTIVE board for {decided_on}: "
            f"{decision.defect}. The hold record was written to "
            f"{AUDIT_DATED_KEY.format(date=decided_on)} before this raise."
        )

    # Reconciliation guard. Written AFTER the record lands, same discipline as
    # the defect raise above: the record itself is the evidence a future reader
    # needs, so it must survive even the failure that says something about it
    # disagreed.
    mismatches = reconcile_arms_with_ledger(doc, rows)
    if mismatches:
        raise CutPromotionError(
            f"scanner-cut promotion record for {decided_on} disagrees with the "
            f"weekly ledger it cites ({slot.decision_source}): "
            f"{'; '.join(mismatches)}. The record was written to "
            f"{AUDIT_DATED_KEY.format(date=decided_on)} before this raise."
        )
    return doc
