"""Promotion engine for the scanner SPEC slot (alpha-engine-config-I9273).

WHAT THIS DECIDES. The scanner has TWO champion/challenger slots on two
different axes (champion-challenger-policy.md §2, which forbids conflating
them). The CUT slot decides which names the sector teams research and is
decided by ``scoring/cut_promotion.py``. This module owns the other one: the
SPEC slot — ``data/scanner_specs.py::SCANNER_SPECS`` — which decides **how
candidates are ranked** into ``candidates/{date}/candidates.json``. Nothing
decided it before this module existed. It had a leaderboard that RANKED and
nothing that ACTED, so its champion moved only by a human editing
``LIVE_CHAMPION`` — ``principles.md`` §2.3's automation gap in its plainest
form, detect → diagnose → *no act, no verify, no close*.

That is not a hypothetical cost. The 2026-07-22 ``config#1186`` cutover was
applied by hand and the register did not catch up until 2026-08-20: four weeks
in which the leaderboard scored an arm against itself and alerted daily
(alpha-engine-config-I7808). A hand-moved pointer is exactly how that happens,
because a hand edit leaves no artifact anyone can compare the live path
against.

BRIAN'S RULING 2026-08-29, verbatim: *"for the research arm, we should make all
arms promote eligible, including think tank"*. The principle, stated twice that
day, is that **an arm that is scored must be able to win**. So
``PROMOTABLE_SPECS`` is derived from ``SCANNER_SPECS`` itself — every
registered arm is eligible — and a genuine exclusion is a DECLARED property
carrying a reason (``DECLARED_INELIGIBLE_SPECS``, today empty), copied onto
every decision record. Never an absence from a list: a disposition that is only
an inference is not a state (ARCHITECTURE §140).

THE DECISION METRIC IS ALREADY PAIRED, AND THAT IS THE POINT
-------------------------------------------------------------
The evidence is ``scanner/leaderboard/{date}.json``'s primary-horizon block,
and the number decided on is each arm's ``topn_alpha_vs_champion`` — the
per-cohort-date difference between the arm's top-N realized return and the
CHAMPION's, clustered across dates. ``leaderboard_scoring._topn_alpha_metric``
admits a date only when BOTH sides have a realized top-N return, so this metric
carries the cohort intersection inside it by construction
(champion-challenger-policy.md §4, same cohort dates), and its ``n_dates`` IS
the size of that intersection.

``topn_alpha_vs_population`` — the scanner slot's primary board metric — is
RECORDED on every arm and is never the decision. It is measured on each arm's
OWN dates, so ranking two arms by it compares one arm's month to another's
quarter whenever their cohorts differ, which on this slot they always do: the
champion's cohort comes from the LIVE ``candidates/{date}/`` prefix and the
challengers' from ``candidates_shadow/{spec}/{date}/``, so the champion has
history the challengers structurally cannot (alpha-engine-config-I9274).

WHY 7-VS-0 IS NOT A CHAMPION WIN
---------------------------------
Measured on ``scanner/leaderboard/2026-08-28.json``, 21-session block:
``momentum_sleeve`` scored 7 dates at mean +0.031348 (t 11.41); BOTH
challengers scored **zero**. The cohort intersection is empty, so "which spec
should have been promoted on 2026-08-28" is **unanswerable** — and this engine
must SAY that rather than read a 7-vs-0 board as the incumbent winning. Three
separate conditions are therefore three separate ``reason_code`` slugs and are
never collapsed:

* ``no_eligible_challenger`` — no challenger has a scored cohort AT ALL. There
  was no comparison. (This is the 2026-08-28 board.)
* ``no_common_cohort`` — a challenger IS scored, but on no date the champion
  also scored. Measured and incomparable.
* ``champion_already_leads`` — a real paired comparison happened and the
  incumbent won it.

Reporting the first two as the third would be a claim about evidence made where
no comparison occurred, and it is the specific misread this engine exists to
prevent. The same distinction is carried PER ARM as
``eligible_for_promotion: false`` plus an ``ineligible_reason_code`` slug, so a
reader of one arm's block never has to infer why it could not win.

DEMOTION IS SYMMETRIC (champion-challenger-policy.md §5.2)
-----------------------------------------------------------
A promotion moves the pointer AWAY from ``DEFAULT_SPEC_CHAMPION`` when a
challenger leads by ``promotion_margin`` and the cooldown is clear. A
DEMOTION moves it BACK when the standing default leads the sitting champion by
the same margin, with the same cooldown — the same bar, in reverse. It is
recorded as its own ``decision`` value rather than as "a promotion of the
default", because *the experiment was reversed* and *a new arm won* are
different facts about the slot and a reader of the series must be able to count
them separately.

WHY A HOLD IS WRITTEN AND NOT OMITTED. champion-challenger-policy.md §3:
silent absence and a genuine outcome must never render identically. This engine
writes on EVERY evaluation — promote, demote or hold:

    ``config/apply_audit/scanner_spec_champion/{date}.json``   immutable, dated
    ``config/apply_audit/scanner_spec_champion/latest.json``   liveness proxy
    ``config/scanner_spec_champion.json``                      the live pointer

all three carrying the same v1 document
(``contracts/scanner_spec_champion.schema.json``).

MEASURABILITY (principles.md §2.7). The number that says this is working is
``arms.<arm>.n_dates_paired`` — cohort dates on which an arm and the champion
were BOTH scored. It is 0 for every challenger today and climbs as the shadow
prefixes accumulate matured cohort dates; when a challenger crosses
``min_dates_for_inference`` the engine can decide. Its ABSENCE is a missing or
stale ``config/apply_audit/scanner_spec_champion/latest.json`` — the engine did
not run — which is a freshness-registry row. No data is never rendered as a
promotion and never as green.

FAIL-LOUD (AGENTS.md). ``decide_spec_champion`` is pure and never swallows: an
input it cannot interpret is a DEFECT, and a defect still produces a written
``hold`` record carrying it — after which ``run_spec_promotion`` RAISES
``SpecPromotionError``. Record first, then fail: a defect that also erases the
evidence of itself is the worse of the two failures. An ABSENT board, an
UNMEASURABLE board and an unscored challenger are not defects — they are the
expected state today — and are plain holds with no raise.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from nousergon_lib.trading_calendar import add_trading_days

from data.scanner_specs import (
    DECLARED_INELIGIBLE_SPECS,
    DEFAULT_SPEC_CHAMPION,
    PROMOTABLE_SPECS,
    SPEC_CHAMPION_POINTER_KEY,
    live_champion_name,
    register_for_champion,
)
from scoring.leaderboard_scoring import (
    HORIZON_OK,
    confidence_for,
    duplicate_arm_rows,
    slot_spec,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

PRODUCER = "crucible-research/scoring/spec_promotion.py"

AUDIT_DATED_KEY = "config/apply_audit/scanner_spec_champion/{date}.json"
AUDIT_LATEST_KEY = "config/apply_audit/scanner_spec_champion/latest.json"
SCANNER_LEADERBOARD_KEY = "scanner/leaderboard/{date}.json"

DECISION_PROMOTE = "promote"
DECISION_DEMOTE = "demote"
DECISION_HOLD = "hold"

DECISION_CADENCE_PER_RUN = "per_scanner_run"

# The name of the decision metric, carried on every record and on every arm. It
# is deliberately long: a reader who sees only this string must be able to tell
# WHAT was measured (long-only top-N realized alpha), AGAINST WHAT (the serving
# champion, on the SAME cohort date), and OVER WHAT (a 21-session forward
# window). It is the board's own `topn_alpha_vs_champion` field, named in full
# so a record is interpretable without the board's schema beside it.
DECISION_METRIC = "topn_alpha_vs_champion_21d_date_clustered"

# ── Decision-level outcome slugs ──────────────────────────────────────────────
# A prose ``reason`` is for the human reading the artifact; these are what a
# sweep, a console adapter or a test matches on. A slug means ONE condition
# forever — re-minting one for a different condition is a schema bump, because
# it silently redefines every archived record that carries it.
REASON_PROMOTED = "promoted"
REASON_DEMOTED = "demoted"
REASON_CHAMPION_LEADS = "champion_already_leads"
REASON_NO_REGISTERED_CHALLENGER = "no_registered_challenger"
REASON_NO_ELIGIBLE_CHALLENGER = "no_eligible_challenger"
REASON_NO_COMMON_COHORT = "no_common_cohort"
REASON_INSUFFICIENT_DATES = "insufficient_dates"
REASON_LEADERBOARD_MISSING = "leaderboard_missing"
REASON_LEADERBOARD_UNMEASURABLE = "leaderboard_unmeasurable"
REASON_HORIZON_MISSING = "decision_horizon_missing"
REASON_CHAMPION_ROW_MISSING = "champion_row_missing"
REASON_MARGIN_NOT_MET = "margin_not_met"
REASON_COOLDOWN_ACTIVE = "cooldown_active"
REASON_BOARD_DEFECTIVE = "board_defective"

HOLD_REASON_CODES: tuple[str, ...] = (
    REASON_CHAMPION_LEADS,
    REASON_NO_REGISTERED_CHALLENGER,
    REASON_NO_ELIGIBLE_CHALLENGER,
    REASON_NO_COMMON_COHORT,
    REASON_INSUFFICIENT_DATES,
    REASON_LEADERBOARD_MISSING,
    REASON_LEADERBOARD_UNMEASURABLE,
    REASON_HORIZON_MISSING,
    REASON_CHAMPION_ROW_MISSING,
    REASON_MARGIN_NOT_MET,
    REASON_COOLDOWN_ACTIVE,
    REASON_BOARD_DEFECTIVE,
)

# ── Per-ARM ineligibility slugs ───────────────────────────────────────────────
# Why THIS arm could not be promoted THIS evaluation, recorded on the arm's own
# block. A separate closed taxonomy from the decision-level one and asserted
# disjoint from it at import: "the slot held because no challenger was scored"
# and "this arm was not scored" are different statements, and one slug meaning
# both would make the series unreadable.
INELIGIBLE_IS_CHAMPION = "is_serving_champion"
INELIGIBLE_NO_SCORED_COHORT = "no_scored_cohort"
INELIGIBLE_NO_COMMON_COHORT = "no_common_cohort_with_champion"
INELIGIBLE_THIN_COHORT = "paired_cohort_below_evidence_floor"
INELIGIBLE_DECLARED = "declared_ineligible"

INELIGIBLE_REASON_CODES: tuple[str, ...] = (
    INELIGIBLE_IS_CHAMPION,
    INELIGIBLE_NO_SCORED_COHORT,
    INELIGIBLE_NO_COMMON_COHORT,
    INELIGIBLE_THIN_COHORT,
    INELIGIBLE_DECLARED,
)


class SpecPromotionError(RuntimeError):
    """The engine could not complete an honest evaluation.

    Raised only AFTER the hold record carrying the defect has been written, so
    the failure is never the reason the evidence of it is missing.
    """


@dataclass(frozen=True)
class SpecPromotionSlot:
    """The slot registry row for the scanner-SPEC promotion decision.

    champion-challenger-policy.md §10: *"Every slot names, in its registry, the
    metric, horizon, benchmark, count-matching width, and hysteresis margins it
    uses."* All six are here.

    It is deliberately NOT an extension of ``cut_promotion.CutPromotionSlot``
    and shares none of its constants. §2: *"Slots are separate axes and must
    never be conflated."* The cut slot decides WHICH names the sector teams
    research and is decided on a weekly holding-period ledger; this slot decides
    HOW candidates are ranked and is decided on the scanner board's paired
    forward-window alpha. Sharing a dataclass would have made a future edit to
    one silently apply to the other.
    """

    slot_id: str
    # Resolved from PROMOTABLE_SPECS rather than restated, so the writer and the
    # reader (`live_champion_spec`) can never disagree about who is eligible —
    # and so Brian's 2026-08-29 ruling ("all arms promote eligible") holds by
    # construction rather than by two lists being kept in step.
    arms: tuple[str, ...]
    default_champion: str
    # Arms declared ineligible WITH THE REASON. Empty today. Carried on the slot
    # rather than looked up at a call site so a decision record can state both
    # halves without a reader joining back to another module.
    declared_ineligible_arms: dict[str, str]

    # ── The decision basis ────────────────────────────────────────────────────
    decision_source: str
    decision_cadence: str
    primary_metric: str
    # The board field the decision reads. Already differenced against the
    # champion on the cohort INTERSECTION, which is what makes it a fair
    # comparison (§4) and what makes an empty intersection detectable rather
    # than silently rendering as a champion win.
    board_metric_field: str
    # Recorded on every arm and never decided on: it is measured on each arm's
    # OWN dates, so it is not comparable across arms with different cohorts.
    reported_metric_field: str
    horizon_days: int
    benchmark_ticker: str
    # §4 count-matching: every arm is scored at this width on the board. The
    # arms are additionally count-matched at generation to the scanner's own
    # `momentum_top_n`, so no arm can win on breadth at either stage.
    count_matched_width: int
    generation_width_param: str
    # Evidence floor, in PAIRED cohort dates.
    min_dates_for_inference: int
    # First cohort date on which more than one arm was emitting a shadow
    # candidate set, so the intersection could be non-empty at all.
    first_cohort_date: date

    # ── Hysteresis (§5.2) ─────────────────────────────────────────────────────
    # In the DECISION METRIC's own units: a mean 21-session top-N alpha
    # difference vs the champion. Applied IDENTICALLY to promotion and to
    # demotion — the same bar in reverse.
    promotion_margin: float
    cooldown_days: int


# Measured 2026-08-29 against the live prefixes:
#   candidates_shadow/momentum_sleeve/   2026-08-11..08-20 (stopped when it
#                                        became champion; its cohort comes from
#                                        the LIVE candidates/ prefix thereafter)
#   candidates_shadow/tech_score_gate/   2026-08-21 and 2026-08-28 only
#   candidates_shadow/mom_12_1_sleeve/   2026-08-18..08-28
# 2026-08-21 is the first date on which BOTH surviving challengers wrote a
# shadow set, so it is the earliest date the slot's cohort intersection could be
# non-empty. No observation earlier than this is admissible evidence for a
# challenger-vs-challenger comparison, and `decision_earliest_on` is derived
# from it rather than from whatever history a prefix happens to carry.
FIRST_COHORT_DATE = date(2026, 8, 21)

# The economic bar, stated once and derived twice so it is auditable rather than
# asserted. The fleet's operational significance bar is ~1%/yr of mean lift
# (`cut_promotion.PROMOTION_MARGIN_NOTE` carries the same figure as 0.005 over
# 126 sessions). A 21-session window is 1/12 of a trading year, so the same bar
# per unit of time is 0.01/12 ≈ 0.00083 — and the cut slot's own 0.005-per-126
# scales to 0.005 × 21/126 ≈ 0.00083 as well. The two derivations agree, which
# is the reason to state both.
PROMOTION_MARGIN = 0.00083

PROMOTION_MARGIN_NOTE = (
    "Units: mean 21-session long-only top-N realized alpha DIFFERENCE vs the "
    "serving champion, date-clustered over the cohort dates both arms scored "
    "(the board's `topn_alpha_vs_champion`). Derived two ways that agree: the "
    "fleet's ~1%/yr operational bar over a 21-session window (1/12 of a "
    "trading year) is 0.01/12 = 0.00083; the scanner-CUT slot's retired "
    "0.005-per-126-sessions bar scaled to 21 sessions is 0.005 x 21/126 = "
    "0.00083. Applied IDENTICALLY to promotion and to demotion — champion-"
    "challenger-policy.md §5.2 requires the same margin in reverse plus the "
    "cooldown, and an asymmetric bar would make the pointer stickier in one "
    "direction than the evidence justifies. The margin is NOT a significance "
    "test and is not sized to the noise floor: §5 rejects a publication-grade "
    "gate for an operational loop, and min_dates_for_inference plus "
    "cooldown_days are what bound oscillation. Every arm's se/t_stat is "
    "recorded so a reader can see whether the margin was cleared with or "
    "without statistical support, without that being a gate."
)

_SCANNER_SLOT = slot_spec("scanner")

SPEC_PROMOTION_SLOT = SpecPromotionSlot(
    slot_id="scanner_spec",
    arms=PROMOTABLE_SPECS,
    default_champion=DEFAULT_SPEC_CHAMPION,
    declared_ineligible_arms=dict(DECLARED_INELIGIBLE_SPECS),
    decision_source=SCANNER_LEADERBOARD_KEY,
    decision_cadence=DECISION_CADENCE_PER_RUN,
    primary_metric=DECISION_METRIC,
    board_metric_field="topn_alpha_vs_champion",
    reported_metric_field=_SCANNER_SLOT.primary_metric,
    horizon_days=_SCANNER_SLOT.horizons_days[0],
    benchmark_ticker=_SCANNER_SLOT.benchmark_ticker or "SPY",
    count_matched_width=_SCANNER_SLOT.top_n,
    generation_width_param="momentum_top_n",
    min_dates_for_inference=_SCANNER_SLOT.min_dates_for_inference,
    first_cohort_date=FIRST_COHORT_DATE,
    promotion_margin=PROMOTION_MARGIN,
    cooldown_days=28,
)

# ── Import-time invariants ────────────────────────────────────────────────────
# Each is a way the engine could silently start deciding on the wrong evidence
# or rendering two conditions identically, so none is left to a test alone.
if SPEC_PROMOTION_SLOT.board_metric_field == SPEC_PROMOTION_SLOT.reported_metric_field:
    raise AssertionError(
        "the decision metric and the reported metric must be different fields — "
        "topn_alpha_vs_population is measured on each arm's OWN cohort dates and "
        "ranking two arms by it compares one arm's month to another's quarter "
        "(champion-challenger-policy.md §4). The decision reads the paired "
        "topn_alpha_vs_champion, which carries the cohort intersection inside it"
    )
if SPEC_PROMOTION_SLOT.horizon_days not in _SCANNER_SLOT.horizons_days:
    raise AssertionError(
        f"horizon {SPEC_PROMOTION_SLOT.horizon_days} is not scored by the scanner "
        f"leaderboard ({_SCANNER_SLOT.horizons_days}) — the decision would read a "
        "block that is never written"
    )
if SPEC_PROMOTION_SLOT.min_dates_for_inference < 1:
    raise AssertionError("a decision needs at least one paired cohort date")
if SPEC_PROMOTION_SLOT.promotion_margin <= 0 or SPEC_PROMOTION_SLOT.cooldown_days <= 0:
    raise AssertionError(
        "champion-challenger-policy.md §5.2 hysteresis is IMPLEMENTED for this "
        "slot, not waived under the §9.3 delta — both the margin and the cooldown "
        "must be positive, and both apply to demotion as well as promotion"
    )
if set(HOLD_REASON_CODES) & set(INELIGIBLE_REASON_CODES):
    raise AssertionError(
        "a decision-level slug and a per-arm slug share a string — 'the slot "
        "held because nothing was comparable' and 'this arm was not comparable' "
        "are different statements and must never render identically"
    )
if REASON_PROMOTED in HOLD_REASON_CODES or REASON_DEMOTED in HOLD_REASON_CODES:
    raise AssertionError("a hold slug cannot also name a pointer movement")
if set(SPEC_PROMOTION_SLOT.declared_ineligible_arms) - set(SPEC_PROMOTION_SLOT.arms):
    raise AssertionError(
        "an arm is declared ineligible that is not a registered arm — an "
        "exclusion rule that outlives its arm reads as a live rule and is not one"
    )


@dataclass
class ArmEvidence:
    """What the SCANNER BOARD says about one arm, paired against the champion.

    Every field is self-qualifying by construction: ``metric``, ``cadence`` and
    ``source`` travel with the numbers, so ``n_dates_paired: 0`` reads as "0
    cohort dates shared with the champion on the scanner leaderboard", never as
    an unqualified zero a reader has to join back to a registry to interpret.

    ``eligible_for_promotion`` plus ``ineligible_reason_code`` are the pair that
    make "measured and lost" distinguishable from "never measured" at ARM
    granularity — the distinction the 2026-08-28 board makes load-bearing.
    """

    present: bool = False
    is_champion: bool = False
    kind: str = ""
    eligible_for_promotion: bool = False
    ineligible_reason_code: str | None = None
    ineligible_reason: str | None = None
    # Cohort dates this arm scored at all, on its own prefix.
    n_dates_scored: int = 0
    # Cohort dates this arm AND the champion both scored. THE measurability
    # surface: it is the size of the intersection the decision is taken over,
    # and it is 0 for every challenger today.
    n_dates_paired: int = 0
    mean_paired_alpha_vs_champion: float | None = None
    se: float | None = None
    t_stat: float | None = None
    se_method: str | None = None
    # Reported, never decided on. See the module docstring.
    alpha_vs_population: dict | None = None
    confidence: str = "insufficient"
    metric: str = ""
    cadence: str = ""
    source: str = ""


@dataclass
class SpecPromotionDecision:
    """The full decision record — the document written to all three keys."""

    decision: str
    champion: str
    champion_before: str
    reason: str
    reason_code: str
    decided_on: str
    arms: dict[str, ArmEvidence] = field(default_factory=dict)
    last_promoted_on: str | None = None
    defect: str | None = None
    decision_earliest_on: str = ""
    board: dict[str, Any] = field(default_factory=dict)
    slot: SpecPromotionSlot = SPEC_PROMOTION_SLOT

    def to_document(self, *, leaderboard_key: str | None = None) -> dict:
        slot = self.slot
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
            "decision_field": slot.board_metric_field,
            "decision_horizon_days": slot.horizon_days,
            "reported_field": slot.reported_metric_field,
            "last_promoted_on": self.last_promoted_on,
            "leaderboard_key": leaderboard_key,
            "arms": {name: asdict(ev) for name, ev in self.arms.items()},
            "eligibility": {
                "promotable_arms": list(slot.arms),
                "declared_ineligible": dict(slot.declared_ineligible_arms),
                "note": (
                    "Every registered arm of SCANNER_SPECS is promotion-eligible "
                    "(Brian's ruling 2026-08-29: an arm that is scored must be "
                    "able to win). `declared_ineligible` is the only mechanism by "
                    "which an arm may be excluded, it carries the reason, and it "
                    "is empty. An arm that cannot be promoted THIS cycle carries "
                    "eligible_for_promotion: false and an ineligible_reason_code "
                    "on its own block — an evidence state, not a registry state, "
                    "and the two are never rendered as one."
                ),
            },
            "hysteresis": {
                "promotion_margin": slot.promotion_margin,
                "promotion_margin_units": (
                    "mean 21-session top-N realized alpha difference vs champion"
                ),
                "promotion_margin_note": PROMOTION_MARGIN_NOTE,
                "cooldown_days": slot.cooldown_days,
                "min_dates_for_inference": slot.min_dates_for_inference,
                "demotion_margin_is_symmetric": True,
                "count_matched_width": slot.count_matched_width,
                "generation_width_param": slot.generation_width_param,
                "benchmark_ticker": slot.benchmark_ticker,
            },
            "defect": self.defect,
            "decision_earliest_on": self.decision_earliest_on,
            "board": self.board,
        }


# ── Reading the board ─────────────────────────────────────────────────────────


def _decision_block(board: Mapping[str, Any] | None, horizon: int) -> dict | None:
    """The horizon block the decision reads.

    Read from ``horizons`` and never from the artifact's top level. The two
    carry the SAME rows for the primary horizon by design (score_multi_horizon
    spreads the primary block across the top level for continuity), but only the
    block carries ``status`` — and a block that is ``unmeasurable`` must not be
    decided on. Reading the top level would drop the one field that says so.
    """
    for block in (board or {}).get("horizons") or []:
        if isinstance(block, Mapping) and block.get("horizon_days") == horizon:
            return dict(block)
    return None


def _row_for(block: Mapping[str, Any] | None, arm: str) -> dict | None:
    rows = [
        r for r in (block or {}).get("specs") or []
        if isinstance(r, Mapping) and r.get("name") == arm
    ]
    # A duplicate is caught by `duplicate_arm_rows` before this is reached; if
    # one somehow arrives here, returning None is the honest answer — no single
    # row is the arm's — and it renders as an unmeasured arm rather than as a
    # number that might belong to another pass.
    return dict(rows[0]) if len(rows) == 1 else None


def _metric(row: Mapping[str, Any] | None, field_name: str) -> dict | None:
    value = (row or {}).get(field_name)
    return dict(value) if isinstance(value, Mapping) else None


def _paired_count(metric: Mapping[str, Any] | None) -> int:
    """How many cohort dates stood behind a paired metric.

    ``date_clustered_stats`` carries its own ``n_dates``, and for
    ``topn_alpha_vs_champion`` that count IS the champion/challenger cohort
    intersection — a date enters only when both sides have a realized top-N
    return. A null metric means the intersection was EMPTY, which this returns
    as 0 rather than leaving a caller to read ``None`` as a small number.

    ``cohort_intersection_dates`` is read first when present: a concurrent
    change (crucible-research ``feat/leaderboard-cohort-intersection``) adds it
    to every block as an explicit field. It is treated as optional on purpose —
    this module must be correct whether that change merges before or after it.
    """
    if not metric:
        return 0
    explicit = metric.get("cohort_intersection_dates")
    if isinstance(explicit, int):
        return max(0, explicit)
    return max(0, int(metric.get("n_dates") or 0))


def _arm_evidence(
    *,
    arm: str,
    block: Mapping[str, Any] | None,
    slot: SpecPromotionSlot,
    is_champion: bool,
) -> ArmEvidence:
    """One arm's paired evidence, and WHY it can or cannot be promoted. Pure."""
    declared = slot.declared_ineligible_arms.get(arm)
    row = _row_for(block, arm)
    paired = _metric(row, slot.board_metric_field)
    population = _metric(row, slot.reported_metric_field)
    n_scored = int((row or {}).get("n_dates_scored") or 0)
    n_paired = 0 if is_champion else _paired_count(paired)

    # The champion's own `topn_alpha_vs_champion` is null by construction
    # (`score_leaderboard` never differences an arm against itself), so its
    # paired mean is left None rather than filled with a 0.0 that would read as
    # a measured flat result.
    mean = None if is_champion else (paired or {}).get("mean")

    eligible = True
    code: str | None = None
    why: str | None = None
    if declared is not None:
        eligible, code, why = False, INELIGIBLE_DECLARED, declared
    elif is_champion:
        eligible, code = False, INELIGIBLE_IS_CHAMPION
        why = (
            "this arm is the serving champion; it cannot be promoted to a seat "
            "it already holds. It is the leg every other arm is differenced "
            "against, and it is DEMOTED by another arm clearing the margin."
        )
    elif n_scored == 0:
        eligible, code = False, INELIGIBLE_NO_SCORED_COHORT
        why = (
            f"the arm scored 0 cohort dates at {slot.horizon_days}d on "
            f"{slot.decision_source}, so it has never been measured. This is NOT "
            "the arm losing a comparison — no comparison was possible."
        )
    elif n_paired == 0:
        eligible, code = False, INELIGIBLE_NO_COMMON_COHORT
        why = (
            f"the arm scored {n_scored} cohort date(s) but NONE of them is a date "
            f"the champion also scored, so {slot.board_metric_field} is null and "
            "there is no common cohort to compare over "
            "(champion-challenger-policy.md §4). Its "
            f"{slot.reported_metric_field} is reported and is not a comparison: "
            "it is measured on this arm's own dates."
        )
    elif n_paired < slot.min_dates_for_inference:
        eligible, code = False, INELIGIBLE_THIN_COHORT
        why = (
            f"the arm shares {n_paired} cohort date(s) with the champion, below "
            f"the evidence floor of {slot.min_dates_for_inference} "
            "(alpha-engine-config-I7542). Below that a per-date mean is an "
            "anecdote, not an inference."
        )

    return ArmEvidence(
        present=row is not None,
        is_champion=is_champion,
        kind="champion" if is_champion else "challenger",
        eligible_for_promotion=eligible,
        ineligible_reason_code=code,
        ineligible_reason=why,
        n_dates_scored=n_scored,
        n_dates_paired=n_paired,
        mean_paired_alpha_vs_champion=(
            float(mean) if mean is not None and n_paired else None
        ),
        se=(paired or {}).get("se") if n_paired else None,
        t_stat=(paired or {}).get("t_stat") if n_paired else None,
        se_method=(paired or {}).get("se_method") if n_paired else None,
        alpha_vs_population=population,
        confidence=confidence_for(n_paired, slot.min_dates_for_inference),
        metric=slot.primary_metric,
        cadence=slot.decision_cadence,
        source=slot.decision_source,
    )


# ── The decision, pure ────────────────────────────────────────────────────────


def decision_earliest_on(slot: SpecPromotionSlot = SPEC_PROMOTION_SLOT) -> str:
    """The earliest date the slot could carry ``min_dates_for_inference`` PAIRED
    cohort dates.

    From :data:`FIRST_COHORT_DATE` — the first date on which more than one arm
    emitted a shadow candidate set — a cohort date matures only after
    ``horizon_days`` forward sessions, and the slot needs
    ``min_dates_for_inference`` distinct matured dates. Cohort dates land on
    every scanner run, so the k-th one is ``k-1`` sessions after the first,
    projected onto the NYSE calendar so holidays are priced in.

    A CEILING, not a promise: it says the evidence CANNOT exist before this
    date, never that it will exist on it. A gap in a shadow prefix — and
    ``tech_score_gate`` has one, having written only 2026-08-21 and 2026-08-28 —
    pushes the real date out, and ``arms.*.n_dates_paired`` is what says where
    it actually stands.
    """
    return add_trading_days(
        slot.first_cohort_date,
        slot.horizon_days + slot.min_dates_for_inference - 1,
    ).isoformat()


def _leader(arms: dict[str, ArmEvidence], champion_before: str) -> str:
    """The eligible challenger with the highest mean paired alpha vs champion.

    Only arms with ``eligible_for_promotion`` are ranked, so an arm with no
    common cohort can never lead on the strength of a number measured over
    dates the champion never saw. The incumbent is the baseline at 0.0 and a
    tie resolves to it — which is what hysteresis means before the margin is
    even consulted.

    ``is None`` rather than a falsy test on the mean, deliberately: an eligible
    arm's mean can legitimately be exactly 0.0, and ``x or -inf`` would rank a
    flat arm below a losing one.
    """

    def rank(kv: tuple[str, ArmEvidence]) -> tuple[float, int, str]:
        mean = kv[1].mean_paired_alpha_vs_champion
        return (
            float("-inf") if mean is None else float(mean),
            1 if kv[0] == champion_before else 0,
            kv[0],
        )

    ranked = {n: e for n, e in arms.items() if e.eligible_for_promotion}
    if not ranked:
        return champion_before
    best = max(ranked.items(), key=rank)[0]
    best_mean = arms[best].mean_paired_alpha_vs_champion
    # The champion's baseline is 0.0 by construction: every ranked number is
    # ALREADY a difference against it. So a challenger must be strictly positive
    # to lead, and a non-positive best means the incumbent leads.
    if best_mean is None or best_mean <= 0.0:
        return champion_before
    return best


def decide_spec_champion(
    *,
    board: Mapping[str, Any] | None,
    champion_before: str,
    decided_on: str,
    last_promoted_on: str | None = None,
    slot: SpecPromotionSlot = SPEC_PROMOTION_SLOT,
) -> SpecPromotionDecision:
    """Decide, from an already-loaded scanner leaderboard. Pure: no S3, no clock.

    Every exit produces a record. There is no path that returns nothing, and no
    path that moves the pointer without an arm clearing
    ``min_dates_for_inference`` PAIRED cohort dates and the promotion margin.
    """
    earliest = decision_earliest_on(slot)

    def _board_meta(present: bool, block: Mapping[str, Any] | None = None) -> dict:
        return {
            "key": slot.decision_source,
            "present": present,
            "horizon_days": slot.horizon_days,
            "status": (block or {}).get("status") if block is not None else None,
            "n_dates": int((block or {}).get("n_dates") or 0) if block else 0,
            "arms_present": sorted(
                str(r.get("name"))
                for r in (block or {}).get("specs") or []
                if isinstance(r, Mapping) and r.get("name")
            ),
            "cohort_intersection_note": (
                "arms.*.n_dates_paired is the champion/challenger cohort "
                "intersection, read from the paired metric's own n_dates. An "
                "empty intersection is a HOLD, never a champion win."
            ),
        }

    def _blank_arms() -> dict[str, ArmEvidence]:
        return {
            a: _arm_evidence(
                arm=a, block=None, slot=slot, is_champion=(a == champion_before)
            )
            for a in slot.arms
        }

    def hold(code, reason, *, arms=None, defect=None, board_meta=None):
        return SpecPromotionDecision(
            decision=DECISION_HOLD,
            champion=champion_before,
            champion_before=champion_before,
            reason=reason,
            reason_code=code,
            decided_on=decided_on,
            arms=arms if arms is not None else _blank_arms(),
            last_promoted_on=last_promoted_on,
            defect=defect,
            decision_earliest_on=earliest,
            board=board_meta if board_meta is not None else _board_meta(board is not None),
            slot=slot,
        )

    # ── WHOLE-BOARD integrity, before anything else ───────────────────────────
    # A board reporting duplicate arm rows is not unmeasured, it is UNRELIABLE:
    # the engine cannot establish that the rows it DOES read came from the pass
    # it thinks they did. Ordered ahead of everything because a producer fault is
    # true whether or not a decision was available to take, and rendering it as
    # the quiet `no_eligible_challenger` would be this module's own stated worse
    # failure — a defect that also erases the evidence of itself.
    if board:
        dupes = duplicate_arm_rows(board)
        if dupes:
            return hold(
                REASON_BOARD_DEFECTIVE,
                f"scanner leaderboard {decided_on} reports duplicate arm rows "
                f"({', '.join(dupes)}). A board that counts any arm twice cannot "
                "be shown to have counted the others once, so no number on it is "
                f"attributable this cycle. {champion_before!r} holds and the run "
                "fails loud.",
                defect=f"duplicate arm rows: {', '.join(dupes)}",
            )

    # ── The slot has no registered challenger ─────────────────────────────────
    # With one arm there is nothing to decide, on any metric — and the engine
    # must SAY so rather than fall through to a comparison path and report
    # `champion_already_leads`, a claim about evidence made where no comparison
    # happened (§3: two states that mean different things must not render
    # identically).
    if len(slot.arms) < 2:
        return hold(
            REASON_NO_REGISTERED_CHALLENGER,
            f"the scanner-spec slot registers one arm ({champion_before!r}) and "
            f"no challenger, so there is no decision to take on {decided_on} — on "
            "any metric, at any cadence. Registering a second arm in "
            "data/scanner_specs.py::SCANNER_SPECS is the only thing standing "
            "between this hold and a live decision: every registered arm is "
            "promotion-eligible by construction (Brian's ruling 2026-08-29).",
        )

    if board is None:
        return hold(
            REASON_LEADERBOARD_MISSING,
            f"no scanner leaderboard at "
            f"{slot.decision_source.format(date=decided_on)} on {decided_on}, so "
            f"the decision metric ({slot.primary_metric}) has no observations. "
            f"{champion_before!r} holds. This is a hold, not a silence: the record "
            "you are reading is the proof the engine ran "
            "(champion-challenger-policy.md §3).",
            board_meta=_board_meta(False),
        )

    block = _decision_block(board, slot.horizon_days)
    if block is None:
        return hold(
            REASON_HORIZON_MISSING,
            f"the scanner leaderboard for {decided_on} carries no "
            f"{slot.horizon_days}d block, so the decision horizon was not scored "
            f"this cycle. {champion_before!r} holds. The block is read from "
            "`horizons` and never from the artifact's top level, because only the "
            "block carries the `status` that says whether it is decidable.",
            board_meta=_board_meta(True),
        )

    if block.get("status") != HORIZON_OK:
        return hold(
            REASON_LEADERBOARD_UNMEASURABLE,
            f"the {slot.horizon_days}d block reports status="
            f"{block.get('status')!r} ({block.get('reason')}), so no number on it "
            f"is a result. {champion_before!r} holds. An unmeasurable board is a "
            "DECISION recorded honestly, never an empty success "
            "(champion-challenger-policy.md §7.2).",
            board_meta=_board_meta(True, block),
        )

    board_meta = _board_meta(True, block)

    if _row_for(block, champion_before) is None:
        return hold(
            REASON_CHAMPION_ROW_MISSING,
            f"the {slot.horizon_days}d block carries no single row for the serving "
            f"champion {champion_before!r} (rows present: "
            f"{board_meta['arms_present']}). Every arm is differenced against the "
            "champion, so without its row nothing on this board is a comparison — "
            "and a leaderboard whose champion is absent is a broken leaderboard, "
            "not a leaderboard with a vacancy (champion-challenger-policy.md §3). "
            f"{champion_before!r} holds.",
            board_meta=board_meta,
        )

    arms = {
        a: _arm_evidence(
            arm=a, block=block, slot=slot, is_champion=(a == champion_before)
        )
        for a in slot.arms
    }
    challengers = {n: e for n, e in arms.items() if n != champion_before}

    eligible = {n: e for n, e in challengers.items() if e.eligible_for_promotion}
    if not eligible:
        # Three genuinely different states, each with its own slug. Ordered
        # most-informative-first: an arm that got as far as a thin paired cohort
        # is a different distance from a decision than one that has never been
        # scored, and collapsing them would hide the loop making progress.
        if any(e.n_dates_paired for e in challengers.values()):
            code = REASON_INSUFFICIENT_DATES
            headline = (
                "every challenger's paired cohort is below the evidence floor of "
                f"{slot.min_dates_for_inference} dates"
            )
        elif any(e.n_dates_scored for e in challengers.values()):
            code = REASON_NO_COMMON_COHORT
            headline = (
                "no challenger shares a single scored cohort date with the "
                f"champion {champion_before!r}, so no comparison exists to take"
            )
        else:
            code = REASON_NO_ELIGIBLE_CHALLENGER
            headline = (
                "no challenger has been scored on any cohort date, so no "
                "comparison was possible"
            )
        detail = "; ".join(
            f"{n}: n_dates_scored={e.n_dates_scored} n_dates_paired="
            f"{e.n_dates_paired} ({e.ineligible_reason_code})"
            for n, e in challengers.items()
        )
        return hold(
            code,
            f"{headline} on {decided_on}. {detail}. {champion_before!r} holds — "
            "and this is NOT the champion winning: a hold taken where no paired "
            "comparison existed is unanswerable, not a verdict. The champion's "
            f"own {slot.reported_metric_field} is recorded on its block and is "
            "measured against the population it narrowed, not against any arm. "
            f"The evidence cannot exist before {earliest}.",
            arms=arms,
            board_meta=board_meta,
        )

    leader = _leader(arms, champion_before)
    if leader == champion_before:
        note = ", ".join(
            f"{n}={e.mean_paired_alpha_vs_champion:+.6f} over {e.n_dates_paired}d"
            for n, e in eligible.items()
            if e.mean_paired_alpha_vs_champion is not None
        )
        return hold(
            REASON_CHAMPION_LEADS,
            f"no eligible challenger has a positive mean {slot.board_metric_field} "
            f"against the incumbent {champion_before!r} ({note}). This IS a "
            "comparison and the incumbent won it — distinct from a hold taken "
            "where no comparison existed. Nothing to promote.",
            arms=arms,
            board_meta=board_meta,
        )

    margin = float(arms[leader].mean_paired_alpha_vs_champion or 0.0)

    # PROMOTE vs DEMOTE. The pointer movement is the same operation; the two
    # names describe two different facts about the slot and a reader of the
    # series must be able to count them separately. A move BACK to the standing
    # default is the experiment being reversed; a move to any other arm is a new
    # arm winning. Both clear the SAME margin and the SAME cooldown (§5.2).
    is_demotion = (
        leader == slot.default_champion and champion_before != slot.default_champion
    )
    direction = "demotion" if is_demotion else "promotion"

    if margin < slot.promotion_margin:
        return hold(
            REASON_MARGIN_NOT_MET,
            f"{leader!r} leads {champion_before!r} by {margin:+.6f} over "
            f"{arms[leader].n_dates_paired} paired cohort date(s), under the "
            f"{direction} margin {slot.promotion_margin:+.6f} in the same units "
            "(champion-challenger-policy.md §5.2 hysteresis — the same bar "
            "governs both directions). Leading is not enough; a ranking that "
            "oscillates on noise makes every downstream cohort incomparable "
            "week to week.",
            arms=arms,
            board_meta=board_meta,
        )

    if last_promoted_on:
        elapsed = (
            date.fromisoformat(decided_on) - date.fromisoformat(last_promoted_on)
        ).days
        if elapsed < slot.cooldown_days:
            return hold(
                REASON_COOLDOWN_ACTIVE,
                f"{leader!r} clears the {direction} margin ({margin:+.6f}) but the "
                f"pointer last moved {elapsed}d ago on {last_promoted_on}, inside "
                f"the {slot.cooldown_days}d cooldown (§5.2 — applied identically "
                "to promotion and demotion). Held; the arm keeps accruing paired "
                "cohort dates and is re-evaluated next run.",
                arms=arms,
                board_meta=board_meta,
            )

    return SpecPromotionDecision(
        decision=DECISION_DEMOTE if is_demotion else DECISION_PROMOTE,
        champion=leader,
        champion_before=champion_before,
        reason=(
            f"{leader!r} beats {champion_before!r} on {slot.primary_metric} by "
            f"{margin:+.6f} over {arms[leader].n_dates_paired} paired cohort "
            f"date(s) at or above the {direction} margin "
            f"{slot.promotion_margin}, count-matched at "
            f"{slot.count_matched_width}; cooldown clear."
            + (
                " This is a DEMOTION: the pointer returns to the standing default "
                f"{slot.default_champion!r}, so the record says the experiment was "
                "reversed rather than that a new arm won."
                if is_demotion
                else ""
            )
        ),
        reason_code=REASON_DEMOTED if is_demotion else REASON_PROMOTED,
        decided_on=decided_on,
        arms=arms,
        last_promoted_on=decided_on,
        decision_earliest_on=earliest,
        board=board_meta,
        slot=slot,
    )


# ── I/O ───────────────────────────────────────────────────────────────────────


def _bucket(bucket: str | None) -> str:
    import os

    return bucket or os.environ.get("S3_BUCKET") or "alpha-engine-research"


def _client(s3_client: Any):
    if s3_client is not None:
        return s3_client
    import boto3

    return boto3.client("s3")


def _get_json(s3: Any, bucket: str, key: str) -> dict | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — absence is a legitimate state
        if "NoSuchKey" in str(exc) or "NoSuchBucket" in str(exc) or "404" in str(exc):
            return None
        raise
    return json.loads(body)


def reconcile_arms_with_board(
    doc: Mapping[str, Any], board: Mapping[str, Any] | None
) -> list[str]:
    """Every ``arms.<arm>.n_dates_paired`` in a WRITTEN record must equal what
    the board it cites actually supports.

    Provenance true by construction (champion-challenger-policy.md §7.5): the
    record is the durable artifact a future reader reconstructs the decision
    from, so a record and its own cited evidence disagreeing is the misread this
    module was filed to fix, wearing the fix's clothes.

    Returns the list of mismatches; empty means every arm reconciles.
    """
    if board is None:
        return []
    horizon = int(doc.get("decision_horizon_days") or SPEC_PROMOTION_SLOT.horizon_days)
    field_name = str(doc.get("decision_field") or SPEC_PROMOTION_SLOT.board_metric_field)
    block = _decision_block(board, horizon)
    if block is None:
        return []
    champion = doc.get("champion_before")
    mismatches: list[str] = []
    for arm, ev in (doc.get("arms") or {}).items():
        if not ev.get("present") or arm == champion:
            continue
        supported = _paired_count(_metric(_row_for(block, arm), field_name))
        if ev.get("n_dates_paired") != supported:
            mismatches.append(
                f"{arm}: record reports n_dates_paired={ev.get('n_dates_paired')} "
                f"on {field_name!r}, but the cited board supports {supported} "
                f"paired cohort date(s) against {champion!r}"
            )
    return mismatches


def read_spec_champion_record(
    *, bucket: str | None = None, s3_client: Any = None
) -> dict | None:
    """The current pointer document, or ``None`` when it has never been written.

    Only ``last_promoted_on`` is consumed from it; the authoritative champion
    comes from ``data.scanner_specs.live_champion_name``, which VALIDATES it.
    Reading the name from the raw document here would fork that validation.
    """
    return _get_json(_client(s3_client), _bucket(bucket), SPEC_CHAMPION_POINTER_KEY)


def _deliver_verdict(doc: Mapping[str, Any]) -> None:
    """Hand the decision to the fleet's slot-parameterised weekly verdict digest.

    Guarded by design, not by accident. ``scoring/verdict_digest.py`` lands on a
    sibling branch (``fix/scanner-cut-all-arms-promotable``); until it merges
    this import fails and the decision is still fully recorded on all three S3
    keys, which is the durable surface. The digest is a DELIVERY channel, never
    the record — champion-challenger-policy.md §7.2's lesson in the other
    direction: a delivery path must not be the only place a verdict exists.
    """
    try:
        from scoring.verdict_digest import deliver_slot_verdict
    except ImportError:
        logger.info(
            "[spec_promotion] scoring/verdict_digest.py not present yet — the "
            "decision is recorded on %s and delivery is a no-op this run",
            AUDIT_LATEST_KEY,
        )
        return
    try:
        deliver_slot_verdict(doc)
    except Exception:  # noqa: BLE001 — delivery is secondary; the record is durable
        logger.exception(
            "[spec_promotion] verdict digest delivery failed for %s — the decision "
            "record is already durable on all three keys and is unaffected",
            doc.get("decided_on"),
        )


_UNSET = object()


def run_spec_promotion(
    decided_on: str,
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    leaderboard: Any = _UNSET,
    slot: SpecPromotionSlot = SPEC_PROMOTION_SLOT,
) -> dict:
    """Decide and WRITE, unconditionally. Returns the written document.

    ``leaderboard`` lets the caller hand in the board it just built (the scanner
    handler does), so the decision reads the exact artifact this run produced
    rather than re-fetching a key that may not have landed. ``_UNSET`` (the
    default) reads it from S3; an explicit ``None`` means "no board", which is a
    hold. The two are distinguished on purpose — passing ``None`` to mean "go
    and read it" is how an absent artifact would come to look like a caller's
    choice.

    Raises :class:`SpecPromotionError` AFTER writing when the record carries a
    defect, or when the written record disagrees with the board it cites.
    """
    s3 = _client(s3_client)
    b = _bucket(bucket)

    # strict=True: an engine that cannot establish which arm is currently live
    # must not write a decision claiming it displaced one.
    champion_before = live_champion_name(bucket=b, s3_client=s3, strict=True)
    prior = read_spec_champion_record(bucket=b, s3_client=s3) or {}
    last_promoted_on = prior.get("last_promoted_on")

    key = SCANNER_LEADERBOARD_KEY.format(date=decided_on)
    board = _get_json(s3, b, key) if leaderboard is _UNSET else leaderboard

    decision = decide_spec_champion(
        board=board,
        champion_before=champion_before,
        decided_on=decided_on,
        last_promoted_on=last_promoted_on,
        slot=slot,
    )
    doc = decision.to_document(leaderboard_key=key if board is not None else None)

    # A promotion must construct a COHERENT register, not mutate one field. This
    # is the same guarantee `assert_registry_coherent` gives at import, applied
    # to the state the pointer is about to name — checked BEFORE the pointer
    # moves, so an incoherent target is refused rather than served.
    if decision.champion != champion_before:
        register_for_champion(decision.champion)

    # Write ORDER is load-bearing: the immutable dated record lands FIRST. If the
    # process dies between writes the surviving state is "the decision is
    # recorded but the live pointer still names the previous champion" —
    # recoverable and safe. The reverse leaves a moved ranking with no record of
    # why, which is the hand-edit failure mode this engine exists to end.
    payload = json.dumps(doc, indent=2, sort_keys=True).encode()
    for k in (
        AUDIT_DATED_KEY.format(date=decided_on),
        AUDIT_LATEST_KEY,
        SPEC_CHAMPION_POINTER_KEY,
    ):
        s3.put_object(Bucket=b, Key=k, Body=payload, ContentType="application/json")

    logger.info(
        "[spec_promotion] metric spec_promotion_decision decision=%s champion=%s "
        "reason_code=%s metric=%s %s",
        doc["decision"],
        doc["champion"],
        doc["reason_code"],
        doc["decision_metric"],
        " ".join(
            f"{a}_n_dates_paired={doc['arms'][a]['n_dates_paired']}"
            for a in slot.arms
            if a in doc["arms"]
        ),
    )

    _deliver_verdict(doc)

    if decision.defect:
        raise SpecPromotionError(
            f"scanner-spec promotion held on a DEFECTIVE board for {decided_on}: "
            f"{decision.defect}. The hold record was written to "
            f"{AUDIT_DATED_KEY.format(date=decided_on)} before this raise."
        )

    mismatches = reconcile_arms_with_board(doc, board)
    if mismatches:
        raise SpecPromotionError(
            f"scanner-spec promotion record for {decided_on} disagrees with the "
            f"leaderboard it cites ({key}): {'; '.join(mismatches)}. The record "
            f"was written to {AUDIT_DATED_KEY.format(date=decided_on)} before "
            "this raise."
        )
    return doc
