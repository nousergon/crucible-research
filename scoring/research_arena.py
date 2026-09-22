"""The RESEARCH slot, wired onto the shared arena engine
(``nousergon_lib.arena``) — alpha-engine-config-I11403, phase 4 of -I11393.

WHAT THIS IS. ``champion-challenger-policy.md`` §10: *"A slot re-implementing
§§3–6 rather than consuming ``nousergon_lib.arena`` is a defect, not a
variation."* This module is the research slot's half of that contract. It
supplies the three things the shared engine cannot know — WHICH arms exist,
WHAT their per-date score is, and WHICH of them may serve — and consumes the
engine for everything else. The score ladder (§4.1), longest-common-window
pairing (§4), the anytime-valid confidence sequence (§5.0), pairwise-wins
ranking (§6.2) and the cap-with-grace retirement rule (§6.1) are IMPORTED.
Nothing in this file computes a mean of differences.

WHY THE SLOT EXISTS. Brian's ruling 2026-09-22 (-I11393) collapsed the
scanner_spec / universe_cut / producer split into ONE slot deciding which names
reach the predictor. Those were three independently-promoting pointers over
successive stages of one pipeline: each link was optimised separately, the
composition — the objective — was measured nowhere, and a promotion in one
silently redefined the input population of arms in another, which §3.1 forbids
by construction. An arm of THIS slot is an end-to-end recipe: a pinned
pre-filter, a ranking key, and a declared width.

THE PER-DATE SCORE, AND WHY IT IS ALREADY POPULATION-RELATIVE.
``topn_alpha_vs_population`` from the arm's own row on
``research/producer_leaderboard/{date}.json`` — the arm's top-N realized return
minus the return of the population it drew from, per cohort date. It is NOT
pre-differenced against the champion: the engine forms the paired difference
itself (``nousergon_lib.arena.window.pair_on_common_window``), and handing it a
champion-relative number would make the champion's own series identically zero
and every ladder rung meaningless. ``benchmark="population"`` is therefore TRUE
of the numbers rather than asserted over them, which is what ``ArenaConfig``
refuses to let a selection-stage slot fake (the 2026-08-17 140bp SPY
inversion).

WHY THE POINTER DECIDES ON THE INFORMATION RATIO, AND NOT THE MEAN.
This slot deliberately carries arms at DIFFERENT widths — ``attractiveness_60``
against ``attractiveness_20`` on the identical ranking is the "what does depth
cost?" experiment, and §4's count-matching rule permits it because width is the
thing under test rather than an uncontrolled confound. But a raw mean with
widths free selects for CONCENTRATION rather than skill: mean alpha per name
declines with depth whenever a ranking carries any signal, so the narrowest arm
wins a comparison it did not earn. ``promote_statistic="information_ratio"``
(nousergon-lib-PR444) prices that — IR ~= IC x sqrt(breadth) — so a narrow arm
must beat a wider one by enough to pay for the dispersion its concentration
bought. ``realized_rank_ic`` rides on every leaderboard row as the
width-independent skill attribution; the cycle record states which statistic
decided.

WHAT THIS DOES NOT DO, DELIBERATELY: IT DOES NOT MOVE THE SERVING POINTER.
The cycle is written; ``config/producer_champion.json`` is left exactly as the
existing promotion path leaves it. This is the same staging ``cut_arena`` took
and for the same stated reason: moving the serving path and the decision engine
in one change would leave NO cycle in which the two could be compared. The
first cycles of a slot whose ladders restart are also the least trustworthy
ones, and a pointer move on them is unrecoverable in a way a written record is
not. Which artifact the research champion should feed, and when the pointer's
readers move to it, is tracked as alpha-engine-config-I11438 — a decision with
a live serving consequence, so it is Brian's.

WHY THE LADDERS RESTART. Four of the five arms carry no inherited history and
say so in ``producers.registry.NO_HISTORY_IMPORT`` (-I11422); the fifth,
``thinktank_20``, inherits ``thinktank_coverage``'s verified 20 dates through
``ProducerSpec.supersedes`` (-PR820), which the leaderboard applies before this
module ever sees a series. So the restart is a property of the register, stated
there, and this module neither re-derives nor overrides it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nousergon_lib.arena import (
    STATISTIC_INFORMATION_RATIO,
    ArenaConfig,
    ArenaCycle,
    ArmRegister,
    ArmSeries,
    ServingPrecondition,
    run_cycle,
)
from nousergon_lib.contracts import conformance_errors

from producers.registry import (
    NO_HISTORY_IMPORT,
    PINNED_RESEARCH_PREFILTER,
    RESEARCH_PRODUCERS,
    RESEARCH_SLOT,
    research_slot_producers,
    retired_producers,
)

logger = logging.getLogger(__name__)

ARENA_SLOT = RESEARCH_SLOT
"""The arena slot name, and the arm-id namespace, so an arm id is
self-describing about which slot it belongs to."""

ARENA_SLOT_KIND = "selection_producer"
"""One of ``nousergon_lib.arena.engine.SELECTION_SLOT_KINDS``. Declaring it is
what makes ``ArenaConfig`` REFUSE a SPY benchmark for this slot — the research
slot's whole job is to beat the population it narrowed."""

ARENA_CYCLE_DATED_KEY = "arena/research/{date}.json"
ARENA_CYCLE_LATEST_KEY = "arena/research/latest.json"
ARENA_REGISTER_KEY = "arena/research/register.json"

BOOTSTRAP_REGISTER_PATH = Path(__file__).resolve().parent / "arena" / "research_register.json"
"""The committed genesis register — the durable backfill, not a per-cycle
recomputation.

Seeded into S3 on the first cycle and never read again, so the live register
can accumulate retirement events a repo file could not. It is a DERIVED
artifact of ``producers.registry`` plus :data:`ARM_CREATED_ON` — regenerate it
with ``python -m scoring.research_arena`` — and :func:`load_register`
re-derives every arm id and appends any the register does not carry, so a
rotted fixture cannot silently score four arms out of five."""

ARENA_CYCLE_CONTRACT = "arena_cycle"
PRODUCER = "crucible-research/scoring/research_arena.py"

SCORE_METRIC = "topn_alpha_vs_population"
SCORE_DEFINITION = (
    f"{SCORE_METRIC} per cohort date, from the arm's own row on "
    "research/producer_leaderboard/{date}.json — the arm's top-N realized "
    "return minus the return of the population it drew from"
)
"""Carried onto the artifact so a reader never has to join back to this module
to learn what a ladder rung is denominated in."""


class ResearchArenaError(RuntimeError):
    """The slot could not complete an honest arena cycle."""


class SlotFloorBreached(ResearchArenaError):
    """The slot holds fewer active arms than a comparison requires.

    Its own class because a caller that wants to distinguish "the loop is
    broken" from "the loop is short of arms to compare" must be able to without
    matching on a message string. The universe_cut slot produced exactly that
    condition twice (2026-08-21 and 2026-08-28) and rendered it as a routine
    hold.
    """


# ── The slot's ArenaConfig (policy §10 — the registry row CI can check) ───────
#
# UNITS for ``diff_clip``: one cohort date's paired difference between two arms'
# population-relative 21-session realized returns. The population leg is common
# to both arms in any pair and cancels in the difference, so the clip bounds the
# 21-session realized-return difference between two equal-weight baskets drawn
# from the SAME pinned 60-name pre-filter.
#
# 0.25 rather than the cuts slot's 0.05, and the difference is not a judgement
# call: that slot's ladder rung is ONE WEEK of log return, this one's is
# TWENTY-ONE SESSIONS of simple return — roughly a 4.5x longer horizon on a
# quantity that scales with sqrt(time), i.e. ~2.2x, applied to a clip that was
# itself set at ~5 sigma. 0.25 is ~5 sigma on a 21-session difference between
# two baskets drawn from one 60-name pool (sigma ~= 0.06 * sqrt(2*(1-0.9))).
#
# The asymmetry of the two errors decides the rest, exactly as it does there: a
# clip that BITES biases the estimate and would need a live cohort exceeding 25
# percentage points of 21-session divergence between two subsets of the same 60
# names; a clip that is merely LOOSE only widens the Robbins boundary and costs
# power, never validity. Re-examine against the observed range once the board
# carries >= 8 matured cohort dates for every live arm.
DIFF_CLIP = 0.25

ARENA_CONFIG = ArenaConfig(
    slot=ARENA_SLOT,
    slot_kind=ARENA_SLOT_KIND,
    benchmark="population",
    diff_clip=DIFF_CLIP,
    # Brian's rulings 2026-08-29, carried across to this slot: cap 5, four-week
    # grace, floor of 3 active arms, retire on a POINT-estimate pairwise loss.
    # The grace period is the evidence bar for retirement; the sequence is the
    # evidence bar for serving (§6.2).
    cap=5,
    grace_weeks=4,
    min_active_arms=3,
    retired_trailing_cycles=8,
    retire_evidence="point",
    # NOT an evidence bar. One paired date is the least from which any
    # statistic can be formed; the information ratio needs two, and returns
    # None below that rather than a number (§5.0).
    min_paired_dates=1,
    # Two paired weeks before an arm may take the pointer, matching the
    # universe_cut delta (-I10546) for the same reason: a paper account
    # re-deciding weekly makes a reversible false promotion cheaper than never
    # promoting at all.
    promote_min_weeks=2,
    # POINT, and it is forced rather than chosen: `promote_statistic=
    # information_ratio` REFUSES `anytime_valid`, because the sequence is a
    # bound on the mean of bounded per-date differences and says nothing about
    # a difference of two ratios (nousergon-lib-PR444). The sequence is still
    # computed and emitted on mean_diff for every comparison, so the evidence
    # it would have required stays on the record.
    promote_evidence="point",
    promote_statistic=STATISTIC_INFORMATION_RATIO,
)

# ── The arms ─────────────────────────────────────────────────────────────────

ARM_CREATED_ON: dict[str, str] = {
    # All five registered in one PR, the day Brian's ruling collapsed the three
    # slots into this one (alpha-engine-config-I11393, crucible-research-PR819).
    # A created_date is the date the RECIPE was registered in this slot, which
    # is why it is recovered rather than defaulted: defaulting a new arm to
    # today hands it a four-week grace window it never served.
    "attractiveness_60": "2026-09-22",
    "attractiveness_20": "2026-09-22",
    "tech_score_20": "2026-09-22",
    "predictor_from_60": "2026-09-22",
    # Registered the same day, but its RECORD continues thinktank_coverage's,
    # whose verified history begins 2026-07-16 (crucible-research-PR820). The
    # created_date is the SUCCESSOR's registration, not the predecessor's: the
    # inherited dates are evidence, and grace is about how long this arm has
    # existed to be judged, which the inheritance does not change.
    "thinktank_20": "2026-09-22",
}
"""Arm name → the date its RECIPE was registered in this slot.

Asserted TOTAL against the live register at import: adding an arm without an
honest creation date fails the import rather than silently back-dating it to
today."""


def _live_arm_names() -> list[str]:
    return sorted(spec.name for spec in research_slot_producers())


if set(ARM_CREATED_ON) != set(_live_arm_names()):
    raise AssertionError(
        "ARM_CREATED_ON must cover the live research slot exactly; missing "
        f"{sorted(set(_live_arm_names()) - set(ARM_CREATED_ON))}, unknown "
        f"{sorted(set(ARM_CREATED_ON) - set(_live_arm_names()))}. An arm "
        "without a recovered created_date cannot be aged against grace_weeks, "
        "and defaulting it to today would make every new arm permanently "
        "un-retirable for four weeks starting from whenever it was noticed."
    )


BASELINE_ARM = "attractiveness_60"
"""The arm every other arm in the slot must beat to justify its cost.

Its own register row states it: *"BASELINE: the pinned attractiveness_top_60,
ranked by the scanner's own attractiveness_score, taken whole. Every other arm
in the slot must beat 'just take the names the scanner already ranked' to
justify its cost."*

Named here because it is also the cycle's INCUMBENT whenever the serving
pointer names no arm of this slot — see :func:`run_arena_cycle` for why that
matters more than it looks.
"""

if BASELINE_ARM not in ARM_CREATED_ON:
    raise AssertionError(
        f"BASELINE_ARM={BASELINE_ARM!r} is not a live arm of the {RESEARCH_SLOT!r} "
        "slot. Every cycle compares against it when the serving pointer names no "
        "arm of this slot, so a stale name here would take the whole cycle down."
    )


def arm_spec(arm: str) -> dict[str, Any]:
    """The arm's RECIPE, as the register hashes it (§3.1).

    Every field that changes what the arm PICKS is here and nothing else is: a
    spec carrying a description would re-hash the arm on a typo, and a spec
    missing the width would let a 20-wide arm silently become 60-wide under one
    id — which is the immutability §3.1 exists to enforce.
    """
    spec = RESEARCH_PRODUCERS[arm]
    return {
        "slot": ARENA_SLOT,
        "name": arm,
        "version": spec.version,
        "prefilter_cut": spec.prefilter_cut,
        "width": spec.width,
        "stateful": bool(spec.stateful),
        "score_source": spec.score_source,
        "supersedes": spec.supersedes,
    }


def arm_id_for(arm: str) -> str:
    """``research:{arm}:{spec_hash}`` — the id HASHES the recipe (§3.1).

    Derived, never restated: changing any field of :func:`arm_spec` produces a
    NEW arm that starts a fresh series and cannot inherit the old one's record.
    That is the whole content of "an arm is an immutable recipe" — re-pointing
    `tech_score_20` at a different pre-filter, or widening it from 20 to 60,
    becomes a new arm automatically rather than because somebody remembered to
    say so.
    """
    from nousergon_lib.arena.arms import derive_arm_id

    return derive_arm_id(ARENA_SLOT, arm, arm_spec(arm))


def arm_name_from_id(arm_id: str) -> str:
    """The arm name inside an arm id. The inverse of :func:`arm_id_for`."""
    parts = arm_id.split(":")
    if len(parts) != 3 or parts[0] != ARENA_SLOT:
        raise ResearchArenaError(
            f"{arm_id!r} is not a {ARENA_SLOT} arm id "
            f"('{ARENA_SLOT}:<name>:<spec_hash>')"
        )
    return parts[1]


def derived_arm_ids(*, as_of: str | None = None) -> dict[str, str]:
    """``{arm: arm_id}`` for every arm this slot SCORES — live arms plus the
    retired ones still inside §3's trailing window.

    Retired arms are included because §3 says so: a retired arm's record exists
    so "we retired the wrong one" stays answerable, and an arm the register
    does not carry is scored by nothing.
    """
    names = set(_live_arm_names())
    names.update(
        spec.name for spec in retired_producers(as_of=as_of)
        if spec.slot == RESEARCH_SLOT
    )
    return {arm: arm_id_for(arm) for arm in sorted(names)}


#: The trading day this slot's genesis register was filed. Distinct from every
#: arm's ``created_date`` on purpose (alpha-engine-config-I10948): the recipe
#: dates are when the arms were REGISTERED, this is when the row was written,
#: and letting the event take its date from the recipe would make a backfilled
#: registration indistinguishable from one filed that day.
REGISTER_FILED_ON = "2026-09-22"


def bootstrap_register() -> ArmRegister:
    """The genesis register: every live arm, with its recovered created_date."""
    register = ArmRegister()
    for arm in _live_arm_names():
        register, _ = register.register(
            slot=ARENA_SLOT,
            name=arm,
            spec=arm_spec(arm),
            created_date=ARM_CREATED_ON[arm],
            filed_on=REGISTER_FILED_ON,
            notes=(
                "registered by alpha-engine-config-I11393; "
                + (
                    f"continues {RESEARCH_PRODUCERS[arm].supersedes}"
                    if RESEARCH_PRODUCERS[arm].supersedes
                    else "starts with no inherited history — see "
                    "producers.registry.NO_HISTORY_IMPORT for why"
                )
            ),
        )
    return register


def assert_slot_floor(
    register: ArmRegister,
    *,
    config: ArenaConfig = ARENA_CONFIG,
    as_of: str | None = None,
) -> None:
    """Raise when the slot holds fewer active arms than a comparison needs.

    Checked at IMPORT against the derived register and again on EVERY cycle
    against the live one. A comment cannot do that, and the condition it
    guards — a decision loop that produced zero comparisons and rendered it as
    a routine hold — ran for two cycles on the universe_cut slot before anyone
    noticed.

    ``as_of`` makes the count POINT-IN-TIME, and passing the cycle's own date
    is what makes this check bind on the thing the engine will actually see.
    MEASURED 2026-09-22: a cycle run for 2026-09-18 — four days before any arm
    of this slot was created — passed a floor checked with ``as_of=None`` (five
    arms live TODAY) and then decided on ``active_arms: 0``, emitting
    ``unservable`` with champion ``None``. A cycle dated before its arms
    existed is a fabrication, and it rendered as an ordinary red verdict rather
    than as the impossible input it was.
    """
    active = register.active_arms(as_of)
    if len(active) < config.min_active_arms:
        when = f" as of {as_of}" if as_of else ""
        raise SlotFloorBreached(
            f"the {ARENA_SLOT!r} slot holds {len(active)} active arm(s){when}, "
            f"below min_active_arms={config.min_active_arms}: "
            f"{sorted(arm_name_from_id(a) for a in active)}. A slot below the "
            "floor produces too few comparisons to decide anything, and a "
            "decision loop that cannot decide renders as a routine hold "
            "(alpha-engine-config-I9317). If this names a date before the arms "
            "were registered, the cycle itself is the error: a slot cannot "
            "decide anything on a day none of its arms existed."
        )


assert_slot_floor(bootstrap_register())


# ── The per-date series, from the leaderboard ────────────────────────────────


def series_from_board(
    board: Mapping[str, Any] | None,
    *,
    arm_ids: Mapping[str, str] | None = None,
    horizon_days: int | None = None,
) -> tuple[dict[str, ArmSeries], dict[str, dict[str, int]]]:
    """``({arm_id: ArmSeries}, {arm: counts})`` from a producer leaderboard.

    Every registered arm gets a series, INCLUDING one the board does not carry
    at all: an arm absent from a cohort date is a MISS, recorded as one, and an
    arm absent from every date is a series of misses rather than an omission
    (§3 — silent absence and a genuine zero must never render identically).

    Three conditions drop a date from an arm's SCORES and add it to its MISSES,
    counted separately because they have different fixes:

    ``dropped_no_row``        the board carries no row for this arm at all.
    ``dropped_null_series``   the row exists and carries no
                              ``topn_alpha_vs_population_by_date`` — the metric
                              needs genuine full-population returns and is null
                              when the caller supplied none.
    ``dropped_degraded``      the date is marked on the row's
                              ``degraded_input`` (alpha-engine-config-I11423):
                              the arm WAS ranked that day, on inputs the board
                              itself says are unusable. Scoring it would enter
                              a known-bad observation as evidence, which is
                              worse than the miss it becomes.

    NOT a raise: every one is a producer-side condition this reader can only
    report. It is recorded on the arm and surfaced on the cycle artifact; a
    reader that raised here would take the whole slot down for one arm's bad
    date, the all-or-nothing precondition -I9272 removed.
    """
    ids = dict(arm_ids or derived_arm_ids())
    counts: dict[str, dict[str, int]] = {
        arm: {
            "rows_seen": 0,
            "scored": 0,
            "dropped_no_row": 0,
            "dropped_null_series": 0,
            "dropped_degraded": 0,
        }
        for arm in ids
    }
    block = _primary_block(board, horizon_days)
    rows = {
        str(r.get("name")): r
        for r in (block.get("specs") or [])
        if r.get("name")
    }
    scores: dict[str, dict[str, float]] = {arm: {} for arm in ids}
    all_dates: set[str] = set()
    for arm in ids:
        row = rows.get(arm)
        if row is None:
            counts[arm]["dropped_no_row"] += 1
            continue
        counts[arm]["rows_seen"] += 1
        by_date = row.get("topn_alpha_vs_population_by_date")
        if not isinstance(by_date, Mapping) or not by_date:
            counts[arm]["dropped_null_series"] += 1
            continue
        bad = _degraded_dates(row)
        for date_str, value in by_date.items():
            all_dates.add(str(date_str))
            if str(date_str) in bad:
                counts[arm]["dropped_degraded"] += 1
                continue
            if value is None:
                counts[arm]["dropped_null_series"] += 1
                continue
            scores[arm][str(date_str)] = float(value)
            counts[arm]["scored"] += 1

    series = {
        arm_id: ArmSeries(
            arm_id=arm_id,
            scores=dict(scores[arm]),
            misses=frozenset(d for d in sorted(all_dates) if d not in scores[arm]),
        )
        for arm, arm_id in ids.items()
    }
    return series, counts


def _primary_block(board: Mapping[str, Any] | None, horizon_days: int | None) -> Mapping[str, Any]:
    """The horizon block the ladder is denominated in.

    The PRIMARY horizon by default — the artifact's own first block, which §3
    makes the continuous series — rather than a literal restated here. A
    horizon named by this module and a horizon named by the board are two facts
    that can disagree.
    """
    blocks = (board or {}).get("horizons") or []
    if not blocks:
        return {}
    if horizon_days is None:
        return blocks[0]
    for block in blocks:
        if int(block.get("horizon_days") or 0) == int(horizon_days):
            return block
    raise ResearchArenaError(
        f"the leaderboard carries no {horizon_days}-session block; it has "
        f"{[b.get('horizon_days') for b in blocks]}. Scoring the slot on a "
        "horizon the board did not produce would silently score nothing."
    )


def _degraded_dates(row: Mapping[str, Any]) -> set[str]:
    """Every cohort date this row declares unusable (-I11423).

    Read off the row rather than re-derived: the board is the authority on its
    own input quality, and a second derivation here is how the two come to
    disagree about which dates are evidence.
    """
    out: set[str] = set()
    for finding in row.get("degraded_input") or []:
        if isinstance(finding, Mapping):
            out.update(str(d) for d in finding.get("dates") or [])
    return out


def preconditions_for(
    board: Mapping[str, Any] | None,
    *,
    arm_ids: Mapping[str, str] | None = None,
) -> dict[str, tuple[ServingPrecondition, ...]]:
    """§5.3 serving preconditions, one verdict per arm — including the passes.

    Two gates today, and every arm receives a RECORDED verdict for each,
    passed or not: §5.3's gates are evaluated per cycle, and an arm with no
    verdict is indistinguishable from an arm nobody checked.

    ``promotion_eligible`` is the register's own answer, projected onto the
    board by -I9277 and read back from it here rather than re-derived.

    ``pinned_prefilter`` is this slot's structural gate: an arm serving from a
    pre-filter that is not the pinned one would have its input population
    redefined by a pointer it does not control — the 2026-09-18 defect that
    replaced 100% of a downstream arm's input without changing its spec hash.
    """
    ids = dict(arm_ids or derived_arm_ids())
    block = _primary_block(board, None)
    rows = {str(r.get("name")): r for r in (block.get("specs") or []) if r.get("name")}
    out: dict[str, tuple[ServingPrecondition, ...]] = {}
    for arm, arm_id in ids.items():
        row = rows.get(arm) or {}
        eligible = bool(row.get("promotion_eligible", True))
        reason = row.get("ineligible_reason")
        spec = RESEARCH_PRODUCERS.get(arm)
        pinned = spec is not None and spec.prefilter_cut == PINNED_RESEARCH_PREFILTER
        out[arm_id] = (
            ServingPrecondition(
                name="promotion_eligible",
                passed=eligible,
                reason=(
                    reason or "the register marks this arm ineligible to serve"
                ) if not eligible else "the register marks this arm eligible to serve",
            ),
            ServingPrecondition(
                name="pinned_prefilter",
                passed=pinned,
                reason=(
                    f"draws from the pinned {PINNED_RESEARCH_PREFILTER!r}"
                    if pinned
                    else "does not draw from the pinned pre-filter, so a pointer "
                    "this arm does not control could replace its input "
                    "population without changing its spec hash (§3.1)"
                ),
            ),
        )
    return out


# ── The register, as live state ──────────────────────────────────────────────


def _register_events_from_s3(s3: Any, bucket: str) -> list[dict] | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=ARENA_REGISTER_KEY)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — first-run absence is a state, not a fault
        text = str(exc)
        if "NoSuchKey" in text or "NoSuchBucket" in text or "404" in text:
            return None
        raise
    return json.loads(body)


def load_register(
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    filed_on: str | None = None,
) -> ArmRegister:
    """The live register — S3 state, seeded from the committed genesis file.

    ``events`` short-circuits the read for a pure caller. Two things happen on
    the way out and both are loud:

    * an arm the registry DERIVES but the register has never carried is
      appended with its recovered ``created_date``. A RETIRED arm is not
      re-registered — it is still in ``all_arms()`` — so retirement is never
      silently undone by the next cycle.
    * :func:`assert_slot_floor` runs against the result, so a register that has
      shrunk below the floor raises here rather than at the point somebody
      notices the artifact is dull.
    """
    stored = list(events) if events is not None else None
    if stored is None:
        if s3_client is None:
            raise ResearchArenaError(
                "load_register needs an s3_client or an explicit `events` list; "
                "constructing one here would make every caller's credentials "
                "and region this module's business"
            )
        stored = _register_events_from_s3(s3_client, bucket or "alpha-engine-research")
    register = bootstrap_register() if stored is None else ArmRegister.from_dicts(stored)

    known = set(register.all_arms())
    for arm, arm_id in sorted(derived_arm_ids().items()):
        if arm_id in known or arm not in ARM_CREATED_ON:
            continue
        register, _ = register.register(
            slot=ARENA_SLOT,
            name=arm,
            spec=arm_spec(arm),
            created_date=ARM_CREATED_ON[arm],
            # The day the row is FILED, which is this cycle — not the recipe's
            # date, or a late registration would be indistinguishable from one
            # made when the arm was written (alpha-engine-config-I10948).
            filed_on=filed_on or REGISTER_FILED_ON,
            notes="registered from producers.registry on the first cycle after its PR",
        )
    assert_slot_floor(register)
    return register


# ── The cycle ────────────────────────────────────────────────────────────────


def run_arena_cycle(
    *,
    board: Mapping[str, Any] | None,
    champion_before: str | None,
    decided_on: str,
    register: ArmRegister,
    config: ArenaConfig = ARENA_CONFIG,
    horizon_days: int | None = None,
) -> tuple[ArenaCycle, dict[str, dict[str, int]]]:
    """One arena cycle for this slot. PURE — no S3, no clock, no alerting.

    ``training=None`` and that is a slot FACT, not an omission: every live arm
    is a deterministic re-ranking of a pinned pre-filter with no fitted
    weights, so there is no fit for anyone to vouch for and
    ``TrainingIntegrityError`` cannot arise. If a future arm IS fitted —
    ``predictor_from_60`` reads a learned model's output but does not fit it —
    this call must start passing statuses; §3 treats an unasserted fit as a
    failed one.
    """
    # The floor, AS OF this cycle's own date. Checked here and not only in
    # `load_register` because the register's present-tense count and the count
    # the ENGINE sees for a past date are different numbers — see
    # `assert_slot_floor`'s docstring for the cycle that slipped between them.
    assert_slot_floor(register, config=config, as_of=decided_on)
    ids = derived_arm_ids(as_of=decided_on)
    series, counts = series_from_board(board, arm_ids=ids, horizon_days=horizon_days)
    # The register is the authority on which arms are scored — a retired arm
    # keeps a series for its §3 trailing window, and an arm the register does
    # not carry must not be handed one at all.
    registered = set(register.all_arms())
    series = {arm_id: s for arm_id, s in series.items() if arm_id in registered}
    for arm_id in register.scored_arms(decided_on, config.retired_trailing_cycles):
        if arm_id not in series:
            series[arm_id] = ArmSeries(arm_id=arm_id, scores={}, misses=frozenset())

    incumbent = arm_id_for(champion_before) if champion_before in ARM_CREATED_ON else None
    if incumbent is None:
        # The live pointer names an arm of the RETIRED producer slot
        # (scanner_predictor_direct, serving since 2026-07-13), which this
        # slot's register correctly does not carry — and will keep naming it
        # until -I11438 moves the serving path.
        #
        # The incumbent falls back to the BASELINE arm rather than to None, and
        # that is the difference between a decision and a coin toss. With no
        # incumbent the engine takes §9.1's bootstrap path and promotes the
        # highest-Copeland arm — which, on a slot whose ladders have just
        # restarted and whose comparison set is empty, is every arm tied at
        # zero and resolves to whichever name sorts first. Measured
        # 2026-09-22: that produced `champion: attractiveness_20, moved: true,
        # status: bootstrap` on ZERO comparisons. Nothing served on it, but a
        # record that names a winner chosen alphabetically is a record that
        # explains a decision nobody made.
        #
        # :data:`BASELINE_ARM` is not an arbitrary substitute. Its register row
        # says so in words: "Every other arm in the slot must beat 'just take
        # the names the scanner already ranked' to justify its cost." An arm
        # that cannot beat the baseline has not earned the pointer, which is
        # exactly what an incumbent is for.
        logger.info(
            "[research_arena] the serving pointer names %r, which is not an arm "
            "of the %r slot — this cycle runs against the baseline arm %r as "
            "incumbent. Expected until alpha-engine-config-I11438 moves the "
            "serving pointer.",
            champion_before, ARENA_SLOT, BASELINE_ARM,
        )
        incumbent = arm_id_for(BASELINE_ARM)
    if incumbent not in registered:
        raise ResearchArenaError(
            f"the incumbent {arm_name_from_id(incumbent)!r} resolves to arm id "
            f"{incumbent}, which the register does not carry. An arm the slot "
            "compares every challenger against, scored by nothing, is the §3 "
            "defect this register exists to make impossible."
        )

    cycle = run_cycle(
        config=config,
        as_of=decided_on,
        register=register,
        series_by_arm=series,
        incumbent=incumbent,
        preconditions=preconditions_for(board, arm_ids=ids),
        training=None,
    )
    return cycle, counts


def cycle_document(
    cycle: ArenaCycle,
    *,
    counts: Mapping[str, Mapping[str, int]],
    register: ArmRegister,
    board_present: bool,
    board_key: str | None = None,
    serving_pointer: str | None = None,
    config: ArenaConfig = ARENA_CONFIG,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """``ArenaCycle.to_dict()`` plus this slot's provenance and its floor probe.

    The contract object is emitted verbatim; everything added is additive and
    namespaced, so the document validates against ``arena_cycle`` unchanged.

    ``serving`` is the field that keeps this cycle HONEST about what it did
    not do: it names the pointer the slot's champion does NOT yet move, so a
    reader cannot mistake a written decision for a served one.
    """
    doc = cycle.to_dict()
    active = register.active_arms()
    doc["producer"] = PRODUCER
    doc["generated_at"] = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    doc["score_definition"] = SCORE_DEFINITION
    doc["arm_names"] = {arm_id: arm for arm, arm_id in derived_arm_ids().items()}
    doc["config"] = dict(config.to_dict())
    doc["slot_floor"] = {
        "min_active_arms": config.min_active_arms,
        "active_arms": len(active),
        "breached": len(active) < config.min_active_arms,
    }
    doc["board"] = {
        "key": board_key,
        "present": bool(board_present),
        "metric": SCORE_METRIC,
        "per_arm": {arm: dict(c) for arm, c in sorted(counts.items())},
    }
    doc["serving"] = {
        "moves_pointer": False,
        "pointer_key": "config/producer_champion.json",
        "pointer_champion": serving_pointer,
        "reason": (
            "OBSERVE-ONLY by design. Moving the serving path and the decision "
            "engine in one change would leave no cycle in which the two could "
            "be compared, and the first cycles of a slot whose ladders restart "
            "are the least trustworthy ones. Tracked as "
            "alpha-engine-config-I11438."
        ),
    }
    doc["history"] = {
        "inherited": {
            arm: spec.supersedes
            for arm, spec in RESEARCH_PRODUCERS.items()
            if spec.slot == ARENA_SLOT and spec.supersedes
        },
        "no_history_import": sorted(NO_HISTORY_IMPORT),
        "detail": (
            "an arm with no inherited history states WHY in "
            "producers.registry.NO_HISTORY_IMPORT (alpha-engine-config-I11422); "
            "a short ladder here is a declared fact, not a gap"
        ),
    }
    return doc


def write_arena_cycle(
    doc: Mapping[str, Any],
    register: ArmRegister,
    *,
    decided_on: str,
    bucket: str,
    s3_client: Any,
) -> dict[str, Any]:
    """Validate against the ``arena_cycle`` contract, then write — dated, latest, register.

    Validation is a HARD gate on the write path, not a warning: this is a
    cross-repo product contract (M0 discipline), and a non-conforming document
    on the key a consumer resolves from is worse than no document, because a
    consumer cannot tell it apart from a good one until it parses it.
    """
    errors = conformance_errors(ARENA_CYCLE_CONTRACT, dict(doc))
    if errors:
        raise ResearchArenaError(
            f"the research arena cycle for {decided_on} does not conform to the "
            f"{ARENA_CYCLE_CONTRACT} contract: {'; '.join(errors)}. Refusing to "
            "write a non-conforming artifact to a key a consumer resolves from."
        )
    payload = json.dumps(dict(doc), indent=2, sort_keys=True).encode()
    # The immutable dated record lands FIRST, then the mirror: a process that
    # dies mid-write leaves the record present and the pointer stale, which is
    # recoverable, rather than the reverse.
    for key in (ARENA_CYCLE_DATED_KEY.format(date=decided_on), ARENA_CYCLE_LATEST_KEY):
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=payload, ContentType="application/json"
        )
    s3_client.put_object(
        Bucket=bucket,
        Key=ARENA_REGISTER_KEY,
        Body=json.dumps(register.to_dicts(), indent=2, sort_keys=True).encode(),
        ContentType="application/json",
    )
    logger.info(
        "[research_arena] metric arena_cycle slot=%s as_of=%s status=%s "
        "champion=%s moved=%s active_arms=%d retirements=%d",
        doc.get("slot"),
        decided_on,
        (doc.get("decision") or {}).get("status"),
        (doc.get("decision") or {}).get("champion"),
        (doc.get("decision") or {}).get("moved"),
        len(doc.get("active_arms") or []),
        sum(1 for v in (doc.get("retirements") or []) if v.get("retire")),
    )
    return dict(doc)


def apply_retirements(
    register: ArmRegister, cycle: ArenaCycle, decided_on: str
) -> ArmRegister:
    """Append the cycle's retirement verdicts to the register.

    The engine decides; this only records. Every veto (the champion, the
    ``min_active_arms`` floor, the grace window) has already been applied
    inside ``evaluate_retirements``, and re-checking them here would be a
    second implementation of §6.1 — which §10 calls a defect. The
    non-retirements are on the artifact with their reasons either way, so a
    retirement list containing only retirements never happens.
    """
    for verdict in cycle.retirements:
        if verdict.retire:
            register = register.retire(verdict.arm_id, decided_on, verdict.reason)
    return register


def run_research_arena(
    s3_client: Any,
    bucket: str,
    decided_on: str,
    *,
    board: Mapping[str, Any] | None = None,
    board_key: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """The whole cycle, end to end: read, decide, record.

    OBSERVE-ONLY toward the serving pointer (see this module's docstring), but
    NOT fail-soft toward its own artifact: §11 requires a cycle every cycle,
    and a slot that writes nothing is unobserved rather than healthy. A caller
    that must not be sunk by this wraps the call, as
    ``lambda/eval_rolling_mean_handler.py`` does for every secondary producer.
    """
    if board is None:
        from scoring.leaderboard_producers import build_producer_leaderboard

        result = build_producer_leaderboard(s3_client, bucket, decided_on, write=False)
        board = result.get("leaderboard")
        board_key = board_key or result.get("key")

    register = load_register(bucket=bucket, s3_client=s3_client, filed_on=decided_on)
    champion_before = (board or {}).get("champion")
    cycle, counts = run_arena_cycle(
        board=board,
        champion_before=champion_before,
        decided_on=decided_on,
        register=register,
    )
    register = apply_retirements(register, cycle, decided_on)
    doc = cycle_document(
        cycle,
        counts=counts,
        register=register,
        board_present=bool(board),
        board_key=board_key,
        serving_pointer=champion_before,
    )
    if write:
        write_arena_cycle(
            doc, register, decided_on=decided_on, bucket=bucket, s3_client=s3_client
        )
    return {
        "status": "ok",
        "key": ARENA_CYCLE_DATED_KEY.format(date=decided_on) if write else None,
        "cycle": doc,
    }


if __name__ == "__main__":  # pragma: no cover -- regenerates the committed genesis file
    BOOTSTRAP_REGISTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_REGISTER_PATH.write_text(
        json.dumps(bootstrap_register().to_dicts(), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {BOOTSTRAP_REGISTER_PATH}")
