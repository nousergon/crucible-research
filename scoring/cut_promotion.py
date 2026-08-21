"""Weekly promotion engine for the scanner cut pair — and its refusal to decide
on an immature horizon (alpha-engine-config-I7826, child of I7823 step 2).

WHAT THIS DECIDES. ``attractiveness_top_60`` and ``tech_score_top_60`` are two
arms of ONE slot, count-matched at 60 (SCANNER_CONTRACT.md §1, Brian's ruling
2026-08-20). Whichever performs better holds the sector-team feed. This module
is the only writer of the pointer that says which:

    ``s3://{bucket}/config/scanner_cut_champion.json``

read by ``scoring/universe_membership.py::live_cut_champion`` /
``resolve_feed_cut`` (crucible-research#670). The pointer, its reader and its
default already existed; this module supplies the DECISION.

WHY REFUSING IS THE DELIVERABLE, NOT A CAVEAT. ``tech_score_top_60`` was first
emitted 2026-08-20. The horizons matching the scanner's ~1-year objective are
126 and 252 sessions, so neither is measurable for months. The 21-session block
matures first and is exactly the horizon Brian's momentum-zero ruling removed
from the champion's ranking: on ``alpha-engine-config-I7580`` a −0.264 IC at 21
days drove a live change that nine years of history inverted (+3% to +6%/yr at
126–252 days). An engine that promotes on the first number to arrive reproduces
that error weekly, on a schedule, with a decision record that looks correct.

So the 21-session horizon is **structurally excluded** from this decision
(``forbidden_horizons_days``, asserted at import), and a decision horizon that
has not matured produces an explicit written ``hold`` — never a promotion, and
never a silence.

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

all three carrying the same v1 document (``contracts/scanner_cut_champion.schema.json``).
The dated record is the history series: promote→hold→promote is reconstructable
from S3 alone, without asking anyone what the engine was thinking.

MEASURABILITY (principles.md §2.7). The number that says this is working is
``arms.<arm>.n_dates_scored`` at the decision horizon, carried in every record.
It is 0 today and climbs as cohorts mature; when both arms cross
``min_dates_for_inference`` the engine can decide, and until then every record
says so in ``reason``. Its ABSENCE is a missing/stale
``config/apply_audit/scanner_cut_champion/latest.json`` — the engine did not run
— which is a freshness-registry row, filed as a follow-up (see the PR body); no
data is never rendered as a promotion and never as green.

HYSTERESIS (champion-challenger-policy.md §5.2) is IMPLEMENTED here, not waived.
The §9.3 winner-take-all delta is scoped to the selection-producer slot and its
argument does not transfer: there the arms are re-scored weekly on a 21-day
horizon, so a cooldown is a material fraction of the measurement window. Here
the decision horizon is 126 sessions and the pointer moves the SECTOR TEAMS'
research input — a 28-day cooldown is 22% of one horizon, and an oscillating
feed makes the qualitative work downstream incomparable week to week. Both a
margin and a cooldown are therefore cheap and are applied.

WHY NOT ``crucible-backtester/optimizer/champion_promotion.py`` (the issue asks
for a stated reason, policy-shared-code second-adoption). Three, in order:
(1) §2 — that engine serves the SELECTION-PRODUCER slot, and slots are separate
axes that must never be conflated; its ``VALID_CHAMPIONS``, its evidence source
(a backtester-internal counterfactual computed inside ``evaluate.py``) and its
policy (winner-take-all, no hysteresis, per the §9.3 delta) are all wrong for
this slot. (2) Both this engine's INPUT (``research/cuts_leaderboard/{date}.json``)
and its OUTPUT (the pointer ``universe_membership`` reads) live in this repo;
running the decision from the backtester would put a cross-repo hop on both
sides of a decision neither end owns. (3) It runs on the Saturday Evaluator, not
on the scanner run that produces the board. What is shared is the SHAPE — the
unconditional audit record, the machine-readable block reason, the pointer
write-order — and that shape is mirrored here deliberately.

FAIL-LOUD (AGENTS.md). ``decide_cut_champion`` is pure and never swallows: an
input it cannot interpret (a board missing its ``horizons`` list, duplicate rows
for one arm, a row naming an arm outside ``PROMOTABLE_CUTS``) is a DEFECT, and a
defect still produces a written ``hold`` record carrying it — after which
``run_cut_promotion`` RAISES ``CutPromotionError``. Record first, then fail: a
defect that also erases the evidence of itself is the worse of the two failures.
An IMMATURE or THIN horizon is not a defect — it is the expected state for
months — and is a plain hold with no raise and no alert.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from scoring.leaderboard_scoring import (
    HORIZON_OK,
    LONG_HORIZONS_DAYS,
    duplicate_arm_rows,
    slot_spec,
)
from scoring.universe_membership import (
    CUT_CHAMPION_POINTER_KEY,
    DEFAULT_CUT_CHAMPION,
    OBSERVE_ONLY_CUTS,
    PROMOTABLE_CUTS,
    _bucket,
    _client,
    live_cut_champion,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PRODUCER = "crucible-research/scoring/cut_promotion.py"

AUDIT_DATED_KEY = "config/apply_audit/scanner_cut_champion/{date}.json"
AUDIT_LATEST_KEY = "config/apply_audit/scanner_cut_champion/latest.json"
CUTS_LEADERBOARD_KEY = "research/cuts_leaderboard/{date}.json"

DECISION_PROMOTE = "promote"
DECISION_HOLD = "hold"

# Machine-readable hold reasons. A prose ``reason`` is for the human reading the
# artifact; this is what a sweep, a console adapter or a test matches on. Never
# reuse a slug for a different condition — a renamed slug is a schema bump.
REASON_PROMOTED = "promoted"
REASON_CHAMPION_LEADS = "champion_already_leads"
REASON_BOARD_MISSING = "board_missing"
REASON_BOARD_UNMEASURABLE = "board_unmeasurable"
REASON_HORIZON_IMMATURE = "decision_horizon_immature"
REASON_HORIZON_UNMEASURABLE = "decision_horizon_unmeasurable"
REASON_ARM_ROW_MISSING = "arm_row_missing"
REASON_NO_PROMOTABLE_CHALLENGER = "no_promotable_challenger"
REASON_ARM_METRIC_MISSING = "arm_metric_missing"
REASON_INSUFFICIENT_DATES = "insufficient_dates"
REASON_MARGIN_NOT_MET = "margin_not_met"
REASON_COOLDOWN_ACTIVE = "cooldown_active"
REASON_CORROBORATION_DISAGREES = "corroborating_horizon_disagrees"
REASON_BOARD_DEFECTIVE = "board_defective"

HOLD_REASON_CODES: tuple[str, ...] = (
    REASON_CHAMPION_LEADS,
    REASON_BOARD_MISSING,
    REASON_BOARD_UNMEASURABLE,
    REASON_HORIZON_IMMATURE,
    REASON_HORIZON_UNMEASURABLE,
    REASON_ARM_ROW_MISSING,
    REASON_NO_PROMOTABLE_CHALLENGER,
    REASON_ARM_METRIC_MISSING,
    REASON_INSUFFICIENT_DATES,
    REASON_MARGIN_NOT_MET,
    REASON_COOLDOWN_ACTIVE,
    REASON_CORROBORATION_DISAGREES,
    REASON_BOARD_DEFECTIVE,
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
    two count-matched 60s — which ARE competing. §2 forbids conflating the two;
    sharing one dataclass would have.
    """

    slot_id: str
    # The arms eligible to hold the feed. Resolved from PROMOTABLE_CUTS rather
    # than restated, so the writer and the reader (`live_cut_champion`) can
    # never disagree about who is eligible.
    arms: tuple[str, ...]
    default_champion: str
    # Where the evidence comes from, and what is read off it.
    leaderboard_id: str
    primary_metric: str
    # The horizon the decision is taken at. 126 sessions ≈ 6 months — the
    # SHORTER of the two horizons that match the scanner's ~1-year objective,
    # chosen over 252 because it matures first and 252 then serves as the
    # corroborating check below. Grading a ~1-year thesis on anything shorter
    # marks it on a horizon it was not built for (alpha-engine-config-I7540).
    decision_horizon_days: int
    # A mature 252-session block may VETO a promotion the 126 block proposes,
    # but never propose one on its own. Asymmetric on purpose: the failure
    # being defended against is a short horizon overturning a long one
    # (I7580), so the longer horizon gets a veto and not a vote.
    corroborating_horizon_days: int
    # Never a decision input, at any confidence, for any reason. Asserted at
    # import (below) and by tests/test_cut_promotion.py.
    forbidden_horizons_days: tuple[int, ...]
    # Evidence floor per arm, inherited from the board's own slot spec so the
    # engine can never demand less evidence than the board calls comparable.
    min_dates_for_inference: int
    # Hysteresis (§5.2). Margin is in the primary metric's own units — a mean
    # realized log-return lift over the horizon — so 0.005 is 50 bps of
    # 126-session lift the challenger must clear the incumbent by. Set at the
    # smallest value that is not noise at n≈5 date clusters rather than at a
    # significance test: §5 rejects a publication-grade gate for an operational
    # loop, and the cooldown already bounds oscillation.
    promotion_margin: float
    cooldown_days: int
    # Arms that are SCORED every cycle but cannot hold the feed. Declared here
    # so a reader of this row can tell "measured and ineligible" from "not
    # measured at all" without going to another module (ARCHITECTURE §140).
    observe_only_arms: tuple[str, ...]
    # §4 count-matching: every arm of the slot, promotable or not, is 60 by
    # construction.
    count_matched_width: int


CUT_PROMOTION_SLOT = CutPromotionSlot(
    slot_id="scanner_cut",
    arms=PROMOTABLE_CUTS,
    observe_only_arms=OBSERVE_ONLY_CUTS,
    default_champion=DEFAULT_CUT_CHAMPION,
    leaderboard_id="cuts",
    primary_metric="topn_alpha_vs_population",
    decision_horizon_days=126,
    corroborating_horizon_days=252,
    forbidden_horizons_days=(21,),
    min_dates_for_inference=slot_spec("cuts").min_dates_for_inference,
    promotion_margin=0.005,
    cooldown_days=28,
    count_matched_width=60,
)

# Import-time invariants. Each of these is a way the engine could silently start
# deciding on the wrong evidence, so none of them is left to a test alone.
if CUT_PROMOTION_SLOT.decision_horizon_days in CUT_PROMOTION_SLOT.forbidden_horizons_days:
    raise AssertionError(
        "the scanner-cut decision horizon may never be a forbidden horizon — "
        "alpha-engine-config-I7580 is what that produces"
    )
for _h in (
    CUT_PROMOTION_SLOT.decision_horizon_days,
    CUT_PROMOTION_SLOT.corroborating_horizon_days,
):
    if _h not in LONG_HORIZONS_DAYS:
        raise AssertionError(
            f"horizon {_h} is not scored by the cuts leaderboard "
            f"(LONG_HORIZONS_DAYS={LONG_HORIZONS_DAYS}) — the engine would read "
            "a block that is never written"
        )
if CUT_PROMOTION_SLOT.primary_metric != slot_spec("cuts").primary_metric:
    raise AssertionError(
        "the promotion engine ranks on a metric the cuts board does not treat "
        "as primary — one of the two is wrong and it must not be resolved here"
    )


@dataclass
class ArmEvidence:
    """What the board says about one arm at one horizon. Every field is copied
    from the board verbatim; nothing is derived, so a reader can join this back
    to ``research/cuts_leaderboard/{date}.json`` and see the same numbers."""

    n_dates_scored: int = 0
    confidence: str = "insufficient"
    topn_alpha_vs_population_mean: float | None = None
    t_stat: float | None = None
    present: bool = False


@dataclass
class CutPromotionDecision:
    """The full decision record — the document written to all three keys."""

    decision: str
    champion: str
    champion_before: str
    reason: str
    reason_code: str
    horizon_days: int
    decided_on: str
    arms: dict[str, ArmEvidence] = field(default_factory=dict)
    last_promoted_on: str | None = None
    corroborating: dict[str, Any] | None = None
    defect: str | None = None

    def to_document(self, *, leaderboard_key: str | None = None) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": PRODUCER,
            "slot_id": CUT_PROMOTION_SLOT.slot_id,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "decided_on": self.decided_on,
            "decision": self.decision,
            "champion": self.champion,
            "champion_before": self.champion_before,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "horizon_days": self.horizon_days,
            "primary_metric": CUT_PROMOTION_SLOT.primary_metric,
            "last_promoted_on": self.last_promoted_on,
            "leaderboard_key": leaderboard_key,
            "arms": {name: asdict(ev) for name, ev in self.arms.items()},
            "hysteresis": {
                "promotion_margin": CUT_PROMOTION_SLOT.promotion_margin,
                "cooldown_days": CUT_PROMOTION_SLOT.cooldown_days,
                "corroborating_horizon_days": CUT_PROMOTION_SLOT.corroborating_horizon_days,
                "min_dates_for_inference": CUT_PROMOTION_SLOT.min_dates_for_inference,
            },
            "corroborating": self.corroborating,
            "defect": self.defect,
        }


# ── The decision, pure ────────────────────────────────────────────────────────


def _block_for(board: dict, horizon: int) -> dict | None:
    for block in board.get("horizons") or []:
        if block.get("horizon_days") == horizon:
            return block
    return None


def _rows_for(block: dict, arm: str) -> list[dict]:
    return [r for r in block.get("specs") or [] if r.get("name") == arm]


def _evidence(row: dict) -> ArmEvidence:
    metric = row.get(CUT_PROMOTION_SLOT.primary_metric)
    mean = metric.get("mean") if isinstance(metric, dict) else None
    t_stat = metric.get("t_stat") if isinstance(metric, dict) else None
    return ArmEvidence(
        n_dates_scored=int(row.get("n_dates_scored") or 0),
        confidence=str(row.get("confidence") or "insufficient"),
        topn_alpha_vs_population_mean=(float(mean) if mean is not None else None),
        t_stat=(float(t_stat) if t_stat is not None else None),
        present=True,
    )


def _leader(arms: dict[str, ArmEvidence]) -> str:
    """The arm with the higher mean lift. Ties resolve to the incumbent by the
    caller (a tie never clears the margin), so this is only ever consulted when
    the two means differ."""
    return max(
        arms.items(),
        key=lambda kv: (kv[1].topn_alpha_vs_population_mean or float("-inf"), kv[0]),
    )[0]


def decide_cut_champion(
    *,
    board: dict | None,
    champion_before: str,
    decided_on: str,
    last_promoted_on: str | None = None,
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
) -> CutPromotionDecision:
    """Decide, from an already-loaded cuts leaderboard. Pure: no S3, no clock.

    Every exit produces a record. There is no path that returns nothing, and no
    path that promotes without both arms clearing ``min_dates_for_inference`` at
    ``decision_horizon_days``.
    """

    def hold(code: str, reason: str, *, arms=None, corroborating=None, defect=None):
        return CutPromotionDecision(
            decision=DECISION_HOLD,
            champion=champion_before,
            champion_before=champion_before,
            reason=reason,
            reason_code=code,
            horizon_days=slot.decision_horizon_days,
            decided_on=decided_on,
            arms=arms or {a: ArmEvidence() for a in slot.arms},
            last_promoted_on=last_promoted_on,
            corroborating=corroborating,
            defect=defect,
        )

    # ── WHOLE-BOARD integrity runs BEFORE the registry check ───────────────
    # A duplicate arm row is a PRODUCER fault, and a producer fault is true
    # whether or not a decision was available to take. Ordering this after the
    # `len(slot.arms) < 2` hold below made a defective board render as the quiet
    # `no_promotable_challenger` — a benign-looking verdict that fails no run and
    # names no defect — the moment alpha-engine-config-I8060 left the slot with a
    # single promotable arm. That is this module's own stated worse failure:
    # "a defect that also erases the evidence of itself".
    #
    # Guarded on `board` being present so it cannot mask the registry condition
    # when there is no artifact at all — which is what the ordering below was
    # protecting, and which still holds.
    if board:
        _early_dupes = duplicate_arm_rows(board)
        if _early_dupes:
            return hold(
                REASON_BOARD_DEFECTIVE,
                f"cuts leaderboard {decided_on} reports duplicate arm rows "
                f"({', '.join(_early_dupes)}) — including on surfaces this engine "
                "does not decide from. A board that counts any arm twice cannot be "
                f"shown to have counted the others once. {champion_before!r} holds "
                "and the run fails loud. Reported ahead of any registry-shape hold: "
                "a producer fault is true whether or not a decision was available.",
                defect=f"duplicate arm rows: {', '.join(_early_dupes)}",
            )

    # ── The slot has no promotable challenger (alpha-engine-config-I8060) ───
    # Brian ruling 2026-08-21: `tech_score_top_60` is observe-only until it has
    # weeks of measured performance. With one promotable arm there is nothing to
    # decide, and the engine must SAY so rather than fall through to the
    # comparison path and report `champion_already_leads` — which is a claim
    # about evidence, made where no comparison happened. Two states that mean
    # different things must not render identically (§3).
    #
    # This is checked before the board, deliberately: it is true of the REGISTRY
    # and does not depend on any evidence, so a missing board must not mask it.
    # The record is still written on every evaluation, so "the engine is armed
    # and waiting" stays visible instead of looking like a dead loop.
    if len(slot.arms) < 2:
        observe = ", ".join(slot.observe_only_arms) or "none"
        return hold(
            REASON_NO_PROMOTABLE_CHALLENGER,
            f"the scanner-cut slot has one promotable arm ({champion_before!r}) "
            f"and no promotable challenger, so there is no decision to take on "
            f"{decided_on}. Observe-only arms, scored every cycle and ineligible "
            f"to hold the feed: {observe}. Restoring one of them to "
            f"PROMOTABLE_CUTS is a one-line registry edit and is the ONLY thing "
            "standing between this hold and a live decision — the evidence is "
            "accumulating either way.",
        )

    if not board:
        return hold(
            REASON_BOARD_MISSING,
            f"no research/cuts_leaderboard for {decided_on} — the engine holds "
            f"{champion_before!r} rather than decide from no evidence. This is a "
            "hold, not a silence: the record you are reading is the proof the "
            "engine ran.",
        )

    status = board.get("status")
    if status and status != "ok":
        return hold(
            REASON_BOARD_UNMEASURABLE,
            f"cuts leaderboard for {decided_on} reports status={status!r} "
            f"({board.get('reason')}) — no arm is scorable, so {champion_before!r} holds.",
        )

    if "horizons" not in board:
        return hold(
            REASON_BOARD_DEFECTIVE,
            f"cuts leaderboard for {decided_on} carries no 'horizons' list — this "
            "is a pre-multi-horizon artifact shape and the engine will not read a "
            "top-level block whose horizon it cannot verify.",
            defect="cuts_leaderboard missing 'horizons'",
        )

    # WHOLE-BOARD integrity, before any row is read. The per-arm duplicate
    # check below covers `slot.arms` in the decision block only, which is
    # exactly the surface the live duplicates were NOT on: the 2026-08-18 and
    # 08-19 artifacts doubled `attractiveness_top_20` and
    # `scanner_gate_baseline_60` — two funnel STAGES — in the 21d block, a
    # horizon this engine structurally never reads. So a board known to be
    # defective produced a decision record that said nothing about it, for two
    # cycles (alpha-engine-config-I8026 deliverable 3).
    #
    # A duplicate anywhere is a producer fault of unknown shape, so it
    # disqualifies the board rather than only the rows it touched: the engine
    # cannot establish that the rows it DOES read came from the pass it thinks
    # they did.
    board_dupes = duplicate_arm_rows(board)
    if board_dupes:
        return hold(
            REASON_BOARD_DEFECTIVE,
            f"cuts leaderboard {decided_on} reports duplicate arm rows "
            f"({', '.join(board_dupes)}) — including on surfaces this engine "
            "does not decide from. A board that counts any arm twice cannot be "
            f"shown to have counted the others once. {champion_before!r} holds "
            "and the run fails loud.",
            defect=f"duplicate arm rows: {', '.join(board_dupes)}",
        )

    block = _block_for(board, slot.decision_horizon_days)
    if block is None:
        return hold(
            REASON_BOARD_DEFECTIVE,
            f"cuts leaderboard for {decided_on} has no {slot.decision_horizon_days}-session "
            f"block. The engine decides at that horizon and nowhere else; the "
            f"{slot.forbidden_horizons_days} block is structurally excluded "
            "(alpha-engine-config-I7580).",
            defect=f"no {slot.decision_horizon_days}d horizon block on the cuts leaderboard",
        )

    # Rows first, so an immature hold still reports each arm's real n_dates —
    # that count IS the measurability surface, and suppressing it while holding
    # would leave nobody able to tell how far off a decision is.
    arms: dict[str, ArmEvidence] = {}
    duplicated: list[str] = []
    missing: list[str] = []
    for arm in slot.arms:
        rows = _rows_for(block, arm)
        if len(rows) > 1:
            duplicated.append(f"{arm}×{len(rows)}")
            arms[arm] = ArmEvidence()
            continue
        if not rows:
            missing.append(arm)
            arms[arm] = ArmEvidence()
            continue
        arms[arm] = _evidence(rows[0])

    if duplicated:
        return hold(
            REASON_BOARD_DEFECTIVE,
            f"cuts leaderboard {decided_on} emits duplicate rows at "
            f"{slot.decision_horizon_days}d for {', '.join(duplicated)} — a board "
            "that reports one arm twice cannot say which number is the arm's "
            "(alpha-engine-config-I7631/I7819). Holding and failing loud.",
            arms=arms,
            defect=f"duplicate rows: {', '.join(duplicated)}",
        )

    block_status = block.get("status")
    if block_status != HORIZON_OK:
        code = (
            REASON_HORIZON_IMMATURE
            if block_status == "immature"
            else REASON_HORIZON_UNMEASURABLE
        )
        counts = ", ".join(f"{a}.n_dates_scored={arms[a].n_dates_scored}" for a in slot.arms)
        return hold(
            code,
            f"{slot.decision_horizon_days}d horizon reports status={block_status!r} "
            f"on {decided_on}: {block.get('reason')} ({counts}). "
            f"{champion_before!r} holds. Promoting on the "
            f"{slot.forbidden_horizons_days[0]}d block instead is the "
            "alpha-engine-config-I7580 error on a weekly schedule and is refused "
            "by construction, not by judgement.",
            arms=arms,
        )

    if missing:
        return hold(
            REASON_ARM_ROW_MISSING,
            f"{', '.join(missing)} has no row in the {slot.decision_horizon_days}d "
            f"block on {decided_on} — an arm that is not scored is not a "
            "challenger (champion-challenger-policy.md §3), and a comparison "
            f"against a missing row is not a comparison. {champion_before!r} holds.",
            arms=arms,
        )

    no_metric = [a for a in slot.arms if arms[a].topn_alpha_vs_population_mean is None]
    if no_metric:
        return hold(
            REASON_ARM_METRIC_MISSING,
            f"{', '.join(no_metric)} carries no {slot.primary_metric} at "
            f"{slot.decision_horizon_days}d on {decided_on} — a real number "
            f"against a null is not a comparison. {champion_before!r} holds.",
            arms=arms,
        )

    thin = [a for a in slot.arms if arms[a].n_dates_scored < slot.min_dates_for_inference]
    if thin:
        counts = ", ".join(f"{a}.n_dates_scored={arms[a].n_dates_scored}" for a in thin)
        return hold(
            REASON_INSUFFICIENT_DATES,
            f"{slot.decision_horizon_days}d horizon immature for inference: {counts} "
            f"< min_dates_for_inference={slot.min_dates_for_inference}. A per-date "
            "mean below that floor is an anecdote, not an inference "
            f"(alpha-engine-config-I7542). {champion_before!r} holds.",
            arms=arms,
        )

    leader = _leader(arms)
    champ_mean = arms[champion_before].topn_alpha_vs_population_mean
    if champ_mean is None:  # pragma: no cover — no_metric above already returned
        return hold(
            REASON_ARM_METRIC_MISSING,
            f"the incumbent {champion_before!r} has no {slot.primary_metric}.",
            arms=arms,
        )
    if leader == champion_before:
        return hold(
            REASON_CHAMPION_LEADS,
            f"the incumbent {champion_before!r} still leads at {slot.decision_horizon_days}d "
            f"({slot.primary_metric} mean {champ_mean:+.6f}). Nothing to promote.",
            arms=arms,
        )

    lead_mean = arms[leader].topn_alpha_vs_population_mean or 0.0
    margin = lead_mean - champ_mean

    # The long horizon's VETO, evaluated before the margin so a disagreement is
    # reported as a disagreement rather than swallowed by a margin failure.
    corroborating = _corroboration(board, slot, leader)
    if corroborating and corroborating.get("disagrees"):
        return hold(
            REASON_CORROBORATION_DISAGREES,
            f"{leader!r} leads at {slot.decision_horizon_days}d by {margin:+.6f}, but the "
            f"mature {slot.corroborating_horizon_days}d horizon puts "
            f"{corroborating.get('leader')!r} ahead. A shorter horizon does not "
            "overturn a longer one in this slot — that inversion is exactly "
            f"alpha-engine-config-I7580. {champion_before!r} holds.",
            arms=arms,
            corroborating=corroborating,
        )

    if margin < slot.promotion_margin:
        return hold(
            REASON_MARGIN_NOT_MET,
            f"{leader!r} leads {champion_before!r} at {slot.decision_horizon_days}d by "
            f"{margin:+.6f}, under the promotion margin {slot.promotion_margin:+.6f} "
            "(champion-challenger-policy.md §5.2 hysteresis). Leading is not enough; "
            "a feed that oscillates on noise makes the sector teams' work "
            "incomparable week to week.",
            arms=arms,
            corroborating=corroborating,
        )

    if last_promoted_on:
        elapsed = (date.fromisoformat(decided_on) - date.fromisoformat(last_promoted_on)).days
        if elapsed < slot.cooldown_days:
            return hold(
                REASON_COOLDOWN_ACTIVE,
                f"{leader!r} clears the margin ({margin:+.6f}) but the pointer last moved "
                f"{elapsed}d ago on {last_promoted_on}, inside the {slot.cooldown_days}d "
                "cooldown (§5.2). Held; the challenger keeps accruing evidence and "
                "is re-evaluated next cycle.",
                arms=arms,
                corroborating=corroborating,
            )

    return CutPromotionDecision(
        decision=DECISION_PROMOTE,
        champion=leader,
        champion_before=champion_before,
        reason=(
            f"{leader!r} beats {champion_before!r} at {slot.decision_horizon_days}d on "
            f"{slot.primary_metric} by {margin:+.6f} (≥ margin {slot.promotion_margin}), "
            f"both arms scored on ≥{slot.min_dates_for_inference} dates, "
            f"{slot.corroborating_horizon_days}d horizon not contradicting, cooldown clear."
        ),
        reason_code=REASON_PROMOTED,
        horizon_days=slot.decision_horizon_days,
        decided_on=decided_on,
        arms=arms,
        last_promoted_on=decided_on,
        corroborating=corroborating,
    )


def _corroboration(board: dict, slot: CutPromotionSlot, proposed: str) -> dict | None:
    """What the long horizon says, when it is in a position to say anything.

    Returns ``None`` when the corroborating block is absent, immature, or thin
    for either arm — an unmeasured veto is not a veto (§5.1: you cannot gate on
    a statistic you did not measure), and at rollout this is the normal state.
    """
    block = _block_for(board, slot.corroborating_horizon_days)
    if block is None or block.get("status") != HORIZON_OK:
        return {
            "horizon_days": slot.corroborating_horizon_days,
            "status": (block or {}).get("status", "absent"),
            "leader": None,
            "disagrees": False,
            "note": "not mature enough to veto; no corroboration applied",
        }
    arms: dict[str, ArmEvidence] = {}
    for arm in slot.arms:
        rows = _rows_for(block, arm)
        if len(rows) != 1:
            return {
                "horizon_days": slot.corroborating_horizon_days,
                "status": block.get("status"),
                "leader": None,
                "disagrees": False,
                "note": f"{arm} has {len(rows)} rows; no corroboration applied",
            }
        arms[arm] = _evidence(rows[0])
    if any(
        ev.topn_alpha_vs_population_mean is None
        or ev.n_dates_scored < slot.min_dates_for_inference
        for ev in arms.values()
    ):
        return {
            "horizon_days": slot.corroborating_horizon_days,
            "status": block.get("status"),
            "leader": None,
            "disagrees": False,
            "note": "below min_dates_for_inference; no corroboration applied",
        }
    leader = _leader(arms)
    return {
        "horizon_days": slot.corroborating_horizon_days,
        "status": block.get("status"),
        "leader": leader,
        "disagrees": leader != proposed,
        "arms": {name: asdict(ev) for name, ev in arms.items()},
        "note": "mature; vetoes a disagreeing promotion",
    }


# ── I/O ───────────────────────────────────────────────────────────────────────


def _get_json(s3: Any, bucket: str, key: str) -> dict | None:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 — absence is a legitimate state
        if "NoSuchKey" in str(exc) or "NoSuchBucket" in str(exc) or "404" in str(exc):
            return None
        raise
    return json.loads(body)


def read_cut_champion_record(*, bucket: str | None = None, s3_client: Any = None) -> dict | None:
    """The current pointer document, or ``None`` when it has never been written.

    Only ``last_promoted_on`` is consumed from it; the authoritative champion is
    taken from :func:`live_cut_champion`, which VALIDATES it. Reading the name
    from the raw document here would fork that validation.
    """
    return _get_json(_client(s3_client), _bucket(bucket), CUT_CHAMPION_POINTER_KEY)


def run_cut_promotion(
    decided_on: str,
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    leaderboard: dict | None = None,
    slot: CutPromotionSlot = CUT_PROMOTION_SLOT,
) -> dict:
    """Decide and WRITE, unconditionally. Returns the written document.

    ``leaderboard`` lets the caller hand in the board it just built (the scanner
    handler does), so the decision reads the exact artifact this run produced
    rather than re-fetching a key that may not have landed. Absent, the dated
    key for ``decided_on`` is read; an absent key is a ``hold``, never a silence.

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

    decision = decide_cut_champion(
        board=board,
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
        "reason_code=%s horizon=%s %s",
        doc["decision"],
        doc["champion"],
        doc["reason_code"],
        doc["horizon_days"],
        " ".join(
            f"{a}_n_dates_scored={doc['arms'][a]['n_dates_scored']}" for a in CUT_PROMOTION_SLOT.arms
        ),
    )

    if decision.defect:
        raise CutPromotionError(
            f"scanner-cut promotion held on a DEFECTIVE board for {decided_on}: "
            f"{decision.defect}. The hold record was written to "
            f"{AUDIT_DATED_KEY.format(date=decided_on)} before this raise."
        )
    return doc
