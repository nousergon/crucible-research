"""The universe-cut slot, wired onto the shared arena engine
(``nousergon_lib.arena``) — alpha-engine-config-I9317.

WHAT THIS IS. ``champion-challenger-policy.md`` §10: *"A slot re-implementing
§§3–6 rather than consuming ``nousergon_lib.arena`` is a defect, not a
variation."* This module is the universe-cut slot's half of that contract: it
supplies the three things the shared engine cannot know — WHICH arms exist,
WHAT their per-date score is, and WHICH of them may serve — and consumes the
engine for everything else. The score ladder (§4.1), longest-common-window
pairing (§4), the anytime-valid confidence sequence (§5.0), pairwise-wins
ranking (§6.2) and the cap-with-grace retirement rule (§6.1) are imported, not
written here. Nothing in this file computes a mean of differences.

THE DEFECT THIS SLOT PRODUCED, AND WHAT NOW PREVENTS IT. ``PROMOTABLE_CUTS``
held exactly ONE arm, so the evaluations of 2026-08-21 and 2026-08-28 produced
zero comparisons and wrote ``no_promotable_challenger`` both times — two cycles
of a decision loop that could not decide, rendered as a routine hold. Three
things now make that unreachable rather than unlikely:

1. :data:`ARENA_CONFIG`'s ``min_active_arms`` is a RETIREMENT floor the engine
   itself enforces — it will not retire the slot below three active arms.
2. :func:`assert_slot_floor` is checked at IMPORT against the derived register
   and again on EVERY cycle against the live one, and it PAGES (a critical ops
   alert) and then raises. A comment cannot do that.
3. The cycle artifact carries ``slot_floor.breached`` as a first-class field,
   so the condition is legible on the artifact and not only in a log line.

WHICH ARTIFACT IS AUTHORITATIVE, AND WHEN THE OTHER RETIRES.
``arena/universe_cut/{date}.json`` (mirrored to ``latest.json``) is the
AUTHORITATIVE record of the decision from this change forward: it carries the
full ladder per arm, every pairwise verdict with the window it rests on, the
pointer decision with its confidence-sequence bound, and every retirement
verdict including the non-retirements (policy §11).
``config/scanner_cut_champion.json`` and
``config/apply_audit/scanner_cut_champion/`` remain the SERVING pointer and its
audit mirror, and keep being written unchanged in shape-family, because
``universe_membership.live_cut_champion`` / ``resolve_feed_cut`` read them; but
the DECISION inside them is now transcribed from the arena cycle rather than
taken there. The pointer retires only when its two readers resolve the champion
from the arena artifact instead — tracked as a follow-up, not done here,
because moving the serving path and the decision engine in one change would
leave no cycle in which the two could be compared.

THE PER-DATE SCORE. ``ArmSeries.scores`` is population-relative by
construction: ``net_log_return - population_log_return``, both read from the
arm's OWN row in ``research/cuts_weekly_ledger/ledger.parquet``. It is NOT
pre-differenced against the champion — the engine forms the paired difference
itself (:func:`nousergon_lib.arena.window.pair_on_common_window`), and handing
it a champion-relative number would make the champion's own series identically
zero and every ladder rung meaningless. ``benchmark="population"`` is therefore
TRUE of the numbers rather than asserted over them, which is what
``ArenaConfig`` refuses to let a selection-stage slot fake (the 2026-08-17 140bp
SPY inversion).

WHAT WAS DELETED RATHER THAN TRANSLATED. The slot's previous decision path
carried ``min_weeks_for_inference`` (5 paired weeks), a 0.0002/week promotion
margin and a 28-day cooldown. §5.0 abolishes minimum-evidence floors fleetwide
and §5.2 abolishes hysteresis and cooldown; the confidence sequence is the
evidence bar and the cumulative window is what damps oscillation.
``ArenaConfig.min_paired_dates`` is a WELL-FORMEDNESS check — it refuses a
window from which no statistic can be formed at all — and is not used as an
evidence bar here or anywhere.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nousergon_lib.arena import (
    ArenaConfig,
    ArenaCycle,
    ArmRegister,
    ArmSeries,
    ServingPrecondition,
    run_cycle,
)
from nousergon_lib.contracts import conformance_errors

from scoring.universe_membership import (
    _ARM_BASIS,
    _ARM_PILLAR_WEIGHTS,
    ATTRACTIVENESS_CUT_PREFIX,
    ATTRACTIVENESS_FEED_TOP_N,
    CHALLENGER_CUT_PREFIX,
    CUT_SLOT_ARM_PREFIXES,
    HARD3_CUT_PREFIX,
    MOMZERO_CUT_PREFIX,
    PROMOTABLE_CUTS,
    SLOT_ARMS,
    TECH_SCORE_CUT_PREFIX,
    _bucket,
    _client,
)
from scoring.weekly_ledger import LEDGER_KEY, LEDGER_VERSION

logger = logging.getLogger(__name__)

__all__ = [
    "ARENA_CONFIG",
    "ARENA_CYCLE_DATED_KEY",
    "ARENA_CYCLE_LATEST_KEY",
    "ARENA_REGISTER_KEY",
    "ARM_CREATED_ON",
    "BOOTSTRAP_REGISTER_PATH",
    "CutArenaError",
    "SlotFloorBreached",
    "arm_id_for",
    "arm_name_from_id",
    "arm_spec",
    "assert_slot_floor",
    "bootstrap_register",
    "cycle_document",
    "derived_arm_ids",
    "load_register",
    "run_arena_cycle",
    "series_from_ledger",
    "write_arena_cycle",
]

ARENA_SLOT = "universe_cut"
"""The arena slot name. Also the arm-id namespace, so an arm id is
self-describing about which slot it belongs to."""

ARENA_SLOT_KIND = "universe_cut"
"""One of ``nousergon_lib.arena.engine.SELECTION_SLOT_KINDS``. Declaring it is
what makes ``ArenaConfig`` REFUSE a SPY benchmark for this slot."""

ARENA_CYCLE_DATED_KEY = "arena/universe_cut/{date}.json"
ARENA_CYCLE_LATEST_KEY = "arena/universe_cut/latest.json"
ARENA_REGISTER_KEY = "arena/universe_cut/register.json"

BOOTSTRAP_REGISTER_PATH = Path(__file__).resolve().parent / "arena" / "universe_cut_register.json"
"""The committed genesis register — the durable backfill, not a per-cycle
recomputation.

It is seeded into S3 on the first cycle and never read again after that, so the
live register can accumulate retirement events that a repo file could not. The
file is a DERIVED artifact of :data:`CUT_SLOT_ARM_PREFIXES` and
:data:`ARM_CREATED_ON` — regenerate it with ``python -m scoring.cut_arena`` —
and :func:`load_register` re-derives every arm id and RAISES on any arm the
register does not carry, so a rotted fixture fails loud instead of silently
scoring four arms out of five. Three test fixtures in this repo rotted this
session by restating the registry as a literal; nothing here restates it."""

ARENA_CYCLE_CONTRACT = "arena_cycle"
PRODUCER = "crucible-research/scoring/cut_arena.py"


class CutArenaError(RuntimeError):
    """The slot could not complete an honest arena cycle."""


class SlotFloorBreached(CutArenaError):
    """The slot holds fewer promotable arms than a comparison requires.

    Its own class because it is the 2026-08-21/28 defect by name, and a caller
    that wants to distinguish "the loop is broken" from "the loop is short of
    arms to compare" must be able to without matching on a message string.
    """


# ── The slot's ArenaConfig (policy §10 — the registry row CI can check) ───────
#
# ``diff_clip`` is the only value here that is not a policy default, so it is
# the only one that needs an argument. UNITS: one week's paired difference
# between two arms' population-relative net log returns. The population leg is
# common to both arms in any pair and cancels in the difference, so the clip is
# in fact a bound on the weekly NET log-return difference between two
# count-matched 60-name equal-weight baskets drawn from the same ~900-name
# scanned universe.
#
# MEASURED, from the ledger's only complete week (2026-08-21 → 2026-08-28, all
# five arms present, read 2026-08-29 from research/cuts_weekly_ledger/
# ledger.parquet): gross weekly log returns ran -0.014982 (tech_score_top_60) to
# +0.003215 (attractiveness_top_60) — a widest pairwise spread of 0.0182 — with
# the population leg at -0.008085. Net differs from gross only by the cost term,
# which is single-digit bps at this turnover.
#
# MODELLED, because one week is not a range: a 60-name equal-weight US-equity
# basket has a weekly log-return sigma around 0.03, and two such baskets drawn
# from the same universe correlate at roughly 0.95, so the DIFFERENCE has
# sigma ~= 0.03 * sqrt(2 * (1 - 0.95)) ~= 0.01.
#
# 0.05 is therefore ~2.7x the widest spread actually observed and ~5 sigma on
# the modelled difference. The asymmetry of the two errors decides the rest:
# a clip that BITES biases the estimate and would need a live week exceeding
# five percentage points of weekly log-return divergence between two 60-name
# cuts of one universe, which has never happened; a clip that is merely LOOSE
# only widens the Robbins boundary and costs power, never validity. Re-examine
# against the observed range once the ledger carries >= 8 complete weeks.
DIFF_CLIP = 0.05

ARENA_CONFIG = ArenaConfig(
    slot=ARENA_SLOT,
    slot_kind=ARENA_SLOT_KIND,
    benchmark="population",
    diff_clip=DIFF_CLIP,
    # Brian's rulings 2026-08-29: cap 5, four-week grace, floor of 3 active
    # arms, retire on a POINT-estimate pairwise loss (the grace period is the
    # evidence bar for retirement; the sequence is the evidence bar for
    # serving — §6.2).
    cap=5,
    grace_weeks=4,
    min_active_arms=3,
    retired_trailing_cycles=8,
    retire_evidence="point",
    # NOT an evidence bar. One paired week is the least from which any
    # statistic can be formed; the confidence sequence is what refuses to
    # promote on it (§5.0).
    min_paired_dates=1,
)

# ── The arms ─────────────────────────────────────────────────────────────────

ARM_CREATED_ON: dict[str, str] = {
    # The champion basis. Count-matched at 60 by the scanner output contract
    # (crucible-research#667, alpha-engine-config-I7809/I7823, Brian's ruling
    # 2026-08-20) — the date this became an ARM of a slot rather than the only
    # cut there was. It is also FIRST_COHORT_DATE: no weekly observation before
    # it is admissible evidence (alpha-engine-config-I8255).
    ATTRACTIVENESS_CUT_PREFIX: "2026-08-20",
    # Registered in the same PR, as the count-matched pair.
    TECH_SCORE_CUT_PREFIX: "2026-08-20",
    # crucible-research#642 — the 12-1 momentum-horizon challenger. Its own
    # shadow profile store begins 2026-08-18, so it has nothing earlier to be
    # reconstructed from; the recipe existed from the 17th.
    CHALLENGER_CUT_PREFIX: "2026-08-17",
    # crucible-research#645 — the zero-momentum-weight arm. Brian's ruling
    # 2026-08-21 (alpha-engine-config-I7988) moved the momentum-zero composite
    # here when equal pillar weights were restored to the champion; the RECIPE
    # dates from the 17th, which is what a created_date means.
    MOMZERO_CUT_PREFIX: "2026-08-17",
    # crucible-research#729, alpha-engine-config-I8256. First EMITTED
    # 2026-08-28 and produced zero names in the ledger's first week — recorded
    # as a miss on that arm, which is what §3 requires and is precisely why the
    # register carries a created_date rather than inferring one from the first
    # week an arm happened to score.
    HARD3_CUT_PREFIX: "2026-08-24",
}
"""Cut prefix → the date its RECIPE was registered in this slot, recovered from
this repository's history.

Keyed on the PREFIX rather than on the arm name so it cannot drift from
:data:`CUT_SLOT_ARM_PREFIXES`, and asserted total against it at import: adding a
prefix without an honest creation date fails the import rather than silently
back-dating a new arm to today and handing it a four-week grace window it never
served."""

if set(ARM_CREATED_ON) != set(CUT_SLOT_ARM_PREFIXES):
    raise AssertionError(
        "ARM_CREATED_ON must cover CUT_SLOT_ARM_PREFIXES exactly; missing "
        f"{sorted(set(CUT_SLOT_ARM_PREFIXES) - set(ARM_CREATED_ON))}, unknown "
        f"{sorted(set(ARM_CREATED_ON) - set(CUT_SLOT_ARM_PREFIXES))}. An arm "
        "without a recovered created_date cannot be aged against grace_weeks, "
        "and defaulting it to today would make every new arm permanently "
        "un-retirable for four weeks starting from whenever it was noticed."
    )


def _prefix_for(arm: str) -> str:
    """The registered prefix ``arm`` was built from. Raises on an unknown arm."""
    for prefix in CUT_SLOT_ARM_PREFIXES:
        if arm.startswith(prefix):
            return prefix
    raise CutArenaError(
        f"{arm!r} is not built from any prefix in CUT_SLOT_ARM_PREFIXES "
        f"({list(CUT_SLOT_ARM_PREFIXES)}). An arm the arena scores but the slot "
        "register does not name is the `thinktank_coverage` defect "
        "(champion-challenger-policy.md §3)."
    )


def arm_spec(arm: str) -> dict[str, Any]:
    """The immutable RECIPE for one cut arm, DERIVED from the live registry.

    The arm id encodes this dict's hash (``nousergon_lib.arena.arms``), so
    changing any of it produces a NEW arm that starts a fresh series and cannot
    inherit the old one's record (§3.1, Brian's ruling 2026-08-29). That is the
    point of deriving rather than restating: retuning
    ``_ARM_PILLAR_WEIGHTS`` for an arm is a recipe change, and it becomes a new
    arm automatically rather than because somebody remembered to say so.

    ``refit_cadence`` is part of the recipe (§3.1): this slot's arms are
    re-formed weekly from the current factor store and carry no fitted weights
    at all, which is also why the cycle is run with ``training=None`` — there is
    no fit to vouch for.
    """
    prefix = _prefix_for(arm)
    spec: dict[str, Any] = {
        "kind": "universe_cut",
        "prefix": prefix,
        "width": ATTRACTIVENESS_FEED_TOP_N,
        "basis": _ARM_BASIS.get(prefix, "attractiveness_rank"),
        "refit_cadence": "weekly_recut_no_fitted_weights",
    }
    weights = _ARM_PILLAR_WEIGHTS.get(prefix)
    if weights is not None:
        spec["pillar_weights"] = {k: float(v) for k, v in sorted(weights.items())}
    return spec


def arm_id_for(arm: str) -> str:
    """``universe_cut:{arm}:{spec_hash}`` for a live cut name."""
    from nousergon_lib.arena.arms import derive_arm_id

    return derive_arm_id(ARENA_SLOT, arm, arm_spec(arm))


def arm_name_from_id(arm_id: str) -> str:
    """The cut name inside an arm id. The inverse of :func:`arm_id_for`."""
    parts = arm_id.split(":")
    if len(parts) != 3 or parts[0] != ARENA_SLOT:
        raise CutArenaError(
            f"{arm_id!r} is not a {ARENA_SLOT} arm id ('{ARENA_SLOT}:<name>:<spec_hash>')"
        )
    return parts[1]


def derived_arm_ids() -> dict[str, str]:
    """Cut name → arm id, for every arm of the slot as the registry defines it."""
    return {arm: arm_id_for(arm) for arm in SLOT_ARMS}


def bootstrap_register() -> ArmRegister:
    """The genesis register: one registration event per cut RECIPE.

    ``bootstrap=True`` on every row, because none of these arms won a
    comparison to get here — they were the slot's roster when the arena was
    wired, and saying so on the record is the difference between a champion
    that was chosen and one that was merely there (policy §11's
    ``operator_bootstrap`` finding)."""
    register = ArmRegister()
    for arm in sorted(SLOT_ARMS, key=lambda a: (ARM_CREATED_ON[_prefix_for(a)], a)):
        register, _ = register.register(
            slot=ARENA_SLOT,
            name=arm,
            spec=arm_spec(arm),
            created_date=ARM_CREATED_ON[_prefix_for(arm)],
            bootstrap=True,
            notes=(
                "backfilled at the alpha-engine-config-I9317 arena wiring from "
                "CUT_SLOT_ARM_PREFIXES; created_date recovered from repository "
                "history"
            ),
        )
    return register


def assert_slot_floor(register: ArmRegister, *, config: ArenaConfig = ARENA_CONFIG,
                      alert: Any = None, context: str = "") -> None:
    """PAGE and raise when the slot holds fewer active arms than a comparison needs.

    Deliverable 5 of alpha-engine-config-I9317, and the reason it is a probe
    rather than a comment: on 2026-08-21 and 2026-08-28 this slot held ONE
    promotable arm, produced zero comparisons, and wrote
    ``no_promotable_challenger`` — a decision loop that could not decide,
    rendered indistinguishably from a routine hold. Nothing paged, because
    nothing was watching the arm COUNT.

    The alert is sent BEFORE the raise, deliberately: the raise reaches an
    operator only through whatever happens to be catching it, and this
    condition must reach one whether or not the caller is."""
    active = register.active_arms()
    if len(active) >= config.min_active_arms:
        return
    message = (
        f"*Universe-cut slot below its arm floor*\n"
        f"{len(active)} active arm(s) — {', '.join(active) or 'none'} — against "
        f"min_active_arms={config.min_active_arms}. A slot at one arm produces "
        f"ZERO comparisons and writes a hold that reads like a decision; that is "
        f"the 2026-08-21/2026-08-28 `no_promotable_challenger` defect "
        f"(champion-challenger-policy.md §6.1). "
        f"{context or 'The cut promotion cycle cannot honestly decide.'}"
    )
    if alert is None:  # pragma: no cover -- exercised via the injected alert
        from ops_alerts import publish_ops_alert as alert

    alert(
        message,
        severity="critical",
        source="crucible-research/scoring/cut_arena.py",
        dedup_key="universe-cut-slot-below-arm-floor",
    )
    raise SlotFloorBreached(message)


# The IMPORT-time half of the same probe. `PROMOTABLE_CUTS` is what
# `live_cut_champion` will accept and therefore what the slot can actually
# serve; if the register shrinks below the floor, this module refuses to load
# and the scanner run fails rather than emitting another well-formed record of
# a decision nobody could take.
if len(PROMOTABLE_CUTS) < ARENA_CONFIG.min_active_arms:
    raise AssertionError(
        f"the universe-cut slot has {len(PROMOTABLE_CUTS)} promotable arm(s) "
        f"({list(PROMOTABLE_CUTS)}) against min_active_arms="
        f"{ARENA_CONFIG.min_active_arms}. This is the 2026-08-21/2026-08-28 "
        "`no_promotable_challenger` defect at import: with fewer than two "
        "promotable arms the slot produces zero comparisons, and with fewer "
        "than three it cannot absorb one arm missing a cycle. Add an arm or "
        "clear an entry from CUT_ARM_PROMOTION_EXCLUSIONS "
        "(alpha-engine-config-I9317)."
    )


# ── The per-date score ───────────────────────────────────────────────────────

SCORE_COLUMN = "net_log_return"
POPULATION_COLUMN = "population_log_return"

SCORE_DEFINITION = f"{SCORE_COLUMN} - {POPULATION_COLUMN}"
"""Carried onto the artifact so a reader never has to join back to this module
to learn what a ladder rung is denominated in."""


def _span(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("week_end"), row.get("priced_from"), row.get("priced_to"))


def series_from_ledger(
    ledger_rows: Sequence[Mapping[str, Any]] | None,
    *,
    arm_ids: Mapping[str, str] | None = None,
) -> tuple[dict[str, ArmSeries], dict[str, dict[str, int]]]:
    """``({arm_id: ArmSeries}, {arm: counts})`` from the weekly ledger.

    Every registered arm gets a series, including one with no rows at all: an
    arm absent from a week is a MISS, recorded as one, and an arm absent from
    every week is a series of misses rather than an omission
    (champion-challenger-policy.md §3 — silent absence and a genuine zero must
    never render identically).

    Three conditions drop a week from an arm's SCORES and add it to that arm's
    MISSES, each counted separately because they have different fixes:

    ``dropped_stale_version``   the row was written under a different
                                ``LEDGER_VERSION``, so a column may not mean
                                what this reader thinks it means.
    ``dropped_null_column``     the row exists and covers the right span but
                                carries no number in ``net_log_return`` or
                                ``population_log_return``. The ledger's first
                                week is entirely of this kind for four of five
                                arms: turnover was unknown so the cost — and
                                therefore the net return — was uncomputable.
    ``dropped_span_mismatch``   the row's ``(week_end, priced_from, priced_to)``
                                disagrees with the span the rest of the slot
                                priced that week. Two arms priced over
                                different spans must not be differenced (§4);
                                checking the LABEL alone would let exactly that
                                through while both rows agreed about what they
                                claimed to cover.

    NOT a defect and NOT a raise: every one of these is a producer-side or
    calendar condition this reader can only report. It is recorded on the arm,
    surfaced on the cycle artifact, and left to the ledger's own producer
    guards — a reader that raised here would take the whole slot down for one
    arm's missing week, which is the all-or-nothing precondition
    alpha-engine-config-I9272 removed.
    """
    ids = dict(arm_ids or derived_arm_ids())
    rows = list(ledger_rows or [])

    counts: dict[str, dict[str, int]] = {
        arm: {
            "rows_seen": 0,
            "scored": 0,
            "dropped_stale_version": 0,
            "dropped_null_column": 0,
            "dropped_span_mismatch": 0,
        }
        for arm in ids
    }

    # The span every arm priced a given week over. Decided by majority across
    # the slot rather than by the champion's row, so a champion whose own
    # boundary resolved oddly cannot silently disqualify everybody else.
    spans: dict[str, dict[tuple[Any, Any, Any], int]] = {}
    current: list[Mapping[str, Any]] = []
    for row in rows:
        arm = row.get("arm")
        if arm not in ids:
            continue
        counts[str(arm)]["rows_seen"] += 1
        version = row.get("ledger_version")
        if version is not None and int(version) != LEDGER_VERSION:
            counts[str(arm)]["dropped_stale_version"] += 1
            continue
        current.append(row)
        spans.setdefault(str(row.get("week_start")), {})
        spans[str(row.get("week_start"))][_span(row)] = (
            spans[str(row.get("week_start"))].get(_span(row), 0) + 1
        )
    modal = {
        week: max(sorted(tally), key=lambda s: (tally[s], str(s)))
        for week, tally in spans.items()
    }

    all_weeks = sorted(modal)
    scores: dict[str, dict[str, float]] = {arm: {} for arm in ids}
    for row in current:
        arm = str(row.get("arm"))
        week = str(row.get("week_start"))
        if _span(row) != modal[week]:
            counts[arm]["dropped_span_mismatch"] += 1
            continue
        gross = row.get(SCORE_COLUMN)
        pop = row.get(POPULATION_COLUMN)
        if gross is None or pop is None:
            counts[arm]["dropped_null_column"] += 1
            continue
        scores[arm][week] = float(gross) - float(pop)
        counts[arm]["scored"] += 1

    series: dict[str, ArmSeries] = {}
    for arm, arm_id in ids.items():
        series[arm_id] = ArmSeries(
            arm_id=arm_id,
            scores=dict(scores[arm]),
            misses=frozenset(w for w in all_weeks if w not in scores[arm]),
        )
    return series, counts


def preconditions_for(
    cycle_ineligible_arms: Mapping[str, str] | None,
    *,
    arm_ids: Mapping[str, str] | None = None,
) -> dict[str, tuple[ServingPrecondition, ...]]:
    """§5.3 serving preconditions, one verdict per arm — including the passes.

    Today's only producer is
    ``universe_membership.promotion_ineligibility_from_rank_tables``: an arm
    whose basis carries no full-universe rank table would break a consumer
    resolving a rank ceiling on the morning of the promotion
    (alpha-engine-config-I7843). Every arm receives a recorded verdict, passed
    or not, because §5.3's gates are evaluated per cycle and an arm with no
    verdict is indistinguishable from an arm nobody checked.
    """
    ids = dict(arm_ids or derived_arm_ids())
    blocked = dict(cycle_ineligible_arms or {})
    out: dict[str, tuple[ServingPrecondition, ...]] = {}
    for arm, arm_id in ids.items():
        reason = blocked.get(arm)
        out[arm_id] = (
            ServingPrecondition(
                name="rank_table_servable",
                passed=reason is None,
                reason=reason or "a full-universe rank table exists for this arm's basis",
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


def load_register(*, bucket: str | None = None, s3_client: Any = None,
                  events: Sequence[Mapping[str, Any]] | None = None) -> ArmRegister:
    """The live register — S3 state, seeded from the committed genesis file.

    ``events`` short-circuits the read for a pure caller. Two things happen on
    the way out and both are loud:

    * an arm the registry DERIVES but the register has never carried is
      appended, with its recovered ``created_date``. §10 requires the registry
      row, the write path and the scoring wiring in one PR; the registry row
      for this slot is ``CUT_SLOT_ARM_PREFIXES`` plus :data:`ARM_CREATED_ON`,
      and a prefix added without a creation date has already failed the import
      above. A RETIRED arm is not re-registered — it is still in
      ``all_arms()`` — so retirement is not silently undone by the next cycle.
    * :func:`assert_slot_floor` runs against the result, so a register that has
      shrunk below the floor pages here rather than at the point somebody
      notices the artifact is dull.
    """
    stored = list(events) if events is not None else None
    if stored is None:
        s3 = _client(s3_client)
        stored = _register_events_from_s3(s3, _bucket(bucket))
    if stored is None:
        register = bootstrap_register()
    else:
        register = ArmRegister.from_dicts(stored)

    known = set(register.all_arms())
    for arm, arm_id in sorted(derived_arm_ids().items()):
        if arm_id in known:
            continue
        register, _ = register.register(
            slot=ARENA_SLOT,
            name=arm,
            spec=arm_spec(arm),
            created_date=ARM_CREATED_ON[_prefix_for(arm)],
            notes="registered from CUT_SLOT_ARM_PREFIXES on first cycle after its PR",
        )
    return register


# ── The cycle ────────────────────────────────────────────────────────────────


def run_arena_cycle(
    *,
    ledger_rows: Sequence[Mapping[str, Any]] | None,
    champion_before: str,
    decided_on: str,
    register: ArmRegister,
    cycle_ineligible_arms: Mapping[str, str] | None = None,
    config: ArenaConfig = ARENA_CONFIG,
) -> tuple[ArenaCycle, dict[str, dict[str, int]]]:
    """One arena cycle for this slot. PURE — no S3, no clock, no alerting.

    ``training=None`` and that is a slot FACT, not an omission: a universe cut
    is a deterministic re-ranking of the current factor store with no fitted
    weights, so there is no fit for anyone to vouch for and
    ``TrainingIntegrityError`` cannot arise here. A slot whose arms ARE fitted
    must pass statuses; §3 treats an unasserted fit as a failed one.
    """
    ids = {arm: arm_id_for(arm) for arm in SLOT_ARMS}
    series, counts = series_from_ledger(ledger_rows, arm_ids=ids)
    # The register is the authority on which arms are scored — a retired arm
    # keeps a series for its §3 trailing window and an arm the register does
    # not carry must not be handed one at all.
    registered = set(register.all_arms())
    series = {arm_id: s for arm_id, s in series.items() if arm_id in registered}
    for arm_id in register.scored_arms(decided_on, config.retired_trailing_cycles):
        if arm_id not in series:
            series[arm_id] = ArmSeries(arm_id=arm_id, scores={}, misses=frozenset())

    incumbent = ids.get(champion_before)
    if incumbent is not None and incumbent not in registered:
        raise CutArenaError(
            f"the serving champion {champion_before!r} resolves to arm id "
            f"{incumbent} which the register does not carry. An arm serving the "
            "sector-team feed without a register row is scored by nothing "
            "(champion-challenger-policy.md §3)."
        )

    cycle = run_cycle(
        config=config,
        as_of=decided_on,
        register=register,
        series_by_arm=series,
        incumbent=incumbent,
        preconditions=preconditions_for(cycle_ineligible_arms, arm_ids=ids),
        training=None,
    )
    return cycle, counts


def cycle_document(
    cycle: ArenaCycle,
    *,
    counts: Mapping[str, Mapping[str, int]],
    register: ArmRegister,
    ledger_present: bool,
    config: ArenaConfig = ARENA_CONFIG,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """``ArenaCycle.to_dict()`` plus this slot's provenance and its floor probe.

    The contract object is emitted verbatim; everything added is additive and
    namespaced, so the document validates against ``arena_cycle`` unchanged.
    ``slot_floor`` is the measurable condition deliverable 5 requires ON THE
    ARTIFACT: a reader — or a console adapter — can see the arm count against
    the floor without re-deriving it, and ``breached`` is never absent, so "the
    floor was fine" and "nobody checked" cannot render identically.
    """
    doc = cycle.to_dict()
    active = register.active_arms()
    doc["producer"] = PRODUCER
    doc["generated_at"] = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    doc["score_definition"] = SCORE_DEFINITION
    doc["arm_names"] = {arm_id_for(a): a for a in SLOT_ARMS}
    doc["config"] = {
        "alpha": config.alpha,
        "diff_clip": config.diff_clip,
        "variance_mode": config.variance_mode,
        "min_paired_dates": config.min_paired_dates,
        "cap": config.cap,
        "grace_weeks": config.grace_weeks,
        "min_active_arms": config.min_active_arms,
        "retired_trailing_cycles": config.retired_trailing_cycles,
        "retire_evidence": config.retire_evidence,
    }
    doc["slot_floor"] = {
        "min_active_arms": config.min_active_arms,
        "active_arms": len(active),
        "breached": len(active) < config.min_active_arms,
    }
    doc["ledger"] = {
        "key": LEDGER_KEY,
        "ledger_version": LEDGER_VERSION,
        "present": bool(ledger_present),
        "score_column": SCORE_COLUMN,
        "population_column": POPULATION_COLUMN,
        "per_arm": {arm: dict(c) for arm, c in sorted(counts.items())},
    }
    return doc


def write_arena_cycle(
    doc: Mapping[str, Any],
    register: ArmRegister,
    *,
    decided_on: str,
    bucket: str | None = None,
    s3_client: Any = None,
) -> dict[str, Any]:
    """Validate against the ``arena_cycle`` contract, then write — dated, latest, register.

    Validation is a HARD gate on the write path and not a warning: this is a
    cross-repo product contract (M0 discipline), and a non-conforming document
    written to the key a consumer resolves from is worse than no document,
    because a consumer cannot tell it apart from a good one until it parses it.
    """
    errors = conformance_errors(ARENA_CYCLE_CONTRACT, dict(doc))
    if errors:
        raise CutArenaError(
            f"the universe-cut arena cycle for {decided_on} does not conform to "
            f"the {ARENA_CYCLE_CONTRACT} contract: {'; '.join(errors)}. Refusing "
            "to write a non-conforming artifact to a key a consumer resolves from."
        )
    s3 = _client(s3_client)
    b = _bucket(bucket)
    payload = json.dumps(dict(doc), indent=2, sort_keys=True).encode()
    # The immutable dated record lands first, then the mirror — the same order
    # and the same reason as the pointer writes in cut_promotion: a process
    # that dies mid-write leaves the record present and the pointer stale,
    # which is recoverable, rather than the reverse.
    for key in (ARENA_CYCLE_DATED_KEY.format(date=decided_on), ARENA_CYCLE_LATEST_KEY):
        s3.put_object(Bucket=b, Key=key, Body=payload, ContentType="application/json")
    s3.put_object(
        Bucket=b,
        Key=ARENA_REGISTER_KEY,
        Body=json.dumps(register.to_dicts(), indent=2, sort_keys=True).encode(),
        ContentType="application/json",
    )
    logger.info(
        "[cut_arena] metric arena_cycle slot=%s as_of=%s status=%s champion=%s "
        "moved=%s active_arms=%d retirements=%d",
        doc.get("slot"),
        decided_on,
        (doc.get("decision") or {}).get("status"),
        (doc.get("decision") or {}).get("champion"),
        (doc.get("decision") or {}).get("moved"),
        len(doc.get("active_arms") or []),
        sum(1 for v in (doc.get("retirements") or []) if v.get("retire")),
    )
    return dict(doc)


def apply_retirements(register: ArmRegister, cycle: ArenaCycle, decided_on: str) -> ArmRegister:
    """Append the cycle's retirement verdicts to the register.

    The engine decides; this only records. Every veto (`champion`, the
    `min_active_arms` floor, the grace window) has already been applied inside
    ``evaluate_retirements`` — re-checking them here would be a second
    implementation of §6.1, which §10 calls a defect. The non-retirements are on
    the artifact with their reasons either way, so a retirement list containing
    only retirements never happens.
    """
    for verdict in cycle.retirements:
        if verdict.retire:
            register = register.retire(verdict.arm_id, decided_on, verdict.reason)
    return register


if __name__ == "__main__":  # pragma: no cover -- regenerates the committed genesis file
    BOOTSTRAP_REGISTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_REGISTER_PATH.write_text(
        json.dumps(bootstrap_register().to_dicts(), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {BOOTSTRAP_REGISTER_PATH}")
