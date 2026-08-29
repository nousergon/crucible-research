"""The scanner-cut promotion engine after the I8261 cutover: it decides on the
CHAINED WEEKLY SERIES, and the long forward horizons are vetoes only.

Brian's ruling, 2026-08-24 (alpha-engine-config-I8261): decide on the paired
weekly difference vs champion; keep 126 and 252 as corroborating vetoes once
mature, not as the decision basis; retire ``forbidden_horizons_days`` as moot;
do NOT restore ``tech_score_top_60`` to ``PROMOTABLE_CUTS`` in the same change.

RED VERIFICATION (champion-challenger-policy.md §7.4 — a guard is verified to
fail without the fix, or it is coverage that reads as coverage and is not).
Two independent runs, both recorded in the PR body:

1. Against the pre-cutover module (``git checkout origin/main -- scoring/
   cut_promotion.py contracts/scanner_cut_champion.schema.json``), every test
   below that names the weekly basis fails — most on the missing symbol, and
   the behavioural ones (``test_a_forward_horizon_landslide_cannot_promote_
   without_the_ledger``, ``test_decision_earliest_on_is_the_weekly_floor``)
   on the assertion itself once the import is satisfied.
2. Against MUTANTS of the post-cutover module — a slot whose
   ``ledger_return_column`` is the gross leg, a ``_leader`` that ranks on the
   embedded champion leg, a ``decision_earliest_on`` that projects 126
   sessions, a veto that is allowed to propose. Each mutation was applied
   alone and the specific guard it targets went red while the rest stayed
   green. A guard that survives every mutation of the thing it claims to
   check is blind, and two such guards shipped elsewhere in this repo today
   before being rewritten.
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

from nousergon_lib.trading_calendar import add_trading_days  # noqa: E402

from scoring.cut_promotion import (  # noqa: E402
    AUDIT_DATED_KEY,
    AUDIT_LATEST_KEY,
    CONTAMINATION_CAVEAT,
    CUT_PROMOTION_SLOT,
    FIRST_COHORT_DATE,
    MIN_VETO_HORIZON_DAYS,
    PROMOTION_MARGIN_NOTE,
    REASON_BOARD_DEFECTIVE,
    REASON_CHAMPION_LEADS,
    REASON_COOLDOWN_ACTIVE,
    REASON_CORROBORATION_DISAGREES,
    REASON_INSUFFICIENT_WEEKS,
    REASON_LEDGER_ARM_MISSING,
    REASON_LEDGER_MISSING,
    REASON_MARGIN_NOT_MET,
    REASON_NO_PROMOTABLE_CHALLENGER,
    REASON_PROMOTED,
    RETIRED_V1_REASON_CODES,
    SESSIONS_PER_WEEK,
    ArmEvidence,
    CutPromotionError,
    decide_cut_champion,
    decision_earliest_on,
    reconcile_arms_with_ledger,
    run_cut_promotion,
)
from scoring.leaderboard_scoring import LONG_HORIZONS_DAYS, slot_spec  # noqa: E402
from scoring.universe_membership import (  # noqa: E402
    CUT_CHAMPION_POINTER_KEY,
    DEFAULT_CUT_CHAMPION,
    OBSERVE_ONLY_CUTS,
    PROMOTABLE_CUTS,
    SLOT_ARMS,
    live_cut_champion,
)
from scoring.weekly_ledger import LEDGER_COLS, LEDGER_KEY, LEDGER_VERSION  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contracts" / "scanner_cut_champion.schema.json"

CHAMP = "attractiveness_top_60"
CHALLENGER = "tech_score_top_60"

# The DECISION machinery is exercised against a two-arm slot, explicitly.
#
# Brian ruled on 2026-08-21 (alpha-engine-config-I8060) and again on 2026-08-24
# (I8261) that `tech_score_top_60` stays observe-only until it has a scored
# cohort, so the LIVE registry has one promotable arm and the engine
# short-circuits with `no_promotable_challenger` before any comparison. That is
# correct, and it must not take the promotion logic out of test with it: the arm
# returns by a one-line registry edit, and the margin / cooldown / veto paths
# have to still be right on the day it does. So these tests state the two-arm
# slot they need instead of inheriting whatever the registry currently declares.
TWO_ARM_SLOT = replace(
    CUT_PROMOTION_SLOT,
    arms=(CHAMP, CHALLENGER),
    observe_only_arms=tuple(
        a for a in CUT_PROMOTION_SLOT.observe_only_arms if a != CHALLENGER
    ),
)
DATE = "2026-08-22"
FLOOR = CUT_PROMOTION_SLOT.min_weeks_for_inference
MARGIN = CUT_PROMOTION_SLOT.promotion_margin


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# ── The weekly ledger fixture ─────────────────────────────────────────────────


def _weeks(n: int, start: str = "2026-06-04") -> list[tuple[str, str]]:
    """``n`` ABUTTING weekly holding periods — each week ends where the next
    begins, which is what makes the observations independent clusters in fact
    rather than by assumption."""
    d0 = date.fromisoformat(start)
    bounds = [(d0 + timedelta(days=7 * i)).isoformat() for i in range(n + 1)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _ledger_row(
    arm: str,
    week: tuple[str, str],
    *,
    net: float | None,
    gross: float | None = None,
    champion_leg: float | None = None,
    is_champion: bool = False,
    ledger_version: int = LEDGER_VERSION,
    week_end: str | None = None,
    priced_to: str | None = None,
) -> dict:
    """One weekly-ledger row in ``weekly_ledger.LEDGER_COLS`` shape.

    ``champion_leg`` populates the ledger's own in-row ``champion_log_return``
    — the GROSS leg. It is deliberately settable independently of the champion
    arm's own ``net_log_return`` so the tests can show that the decision does
    not read it.
    """
    return {
        "arm": arm,
        "week_start": week[0],
        "week_end": week_end or week[1],
        "priced_from": week[0],
        "priced_to": priced_to or week_end or week[1],
        "n_names": 60,
        "retained_from_prior": None,
        "turnover_frac": 0.3,
        "gross_log_return": (net if gross is None else gross),
        "net_log_return": net,
        "net_unavailable_reason": None if net is not None else "turnover_unknown",
        "cost_bps": 10.0,
        "benchmark_log_return": 0.001,
        "population_log_return": 0.001,
        "champion_log_return": champion_leg,
        "rank_ic": None,
        "market_regime": "neutral",
        "is_champion": is_champion,
        "ledger_version": ledger_version,
        "code_sha": "test",
        "written_at": "2026-08-22T00:00:00+00:00",
    }


def _ledger(
    *,
    champ_net: float | list[float | None] = 0.0,
    chal_net: float | list[float | None] = 0.0,
    n_weeks: int = 10,
    champion_leg: float | None = None,
    arms: tuple[str, ...] = (CHAMP, CHALLENGER),
    chal_week_end_shift: int = 0,
    chal_priced_to_shift: int = 0,
    chal_stale_versions: int = 0,
) -> list[dict]:
    weeks = _weeks(n_weeks)

    def series(v):
        return list(v) if isinstance(v, list) else [v] * n_weeks

    champ_series, chal_series = series(champ_net), series(chal_net)
    rows: list[dict] = []
    for i, week in enumerate(weeks):
        if CHAMP in arms:
            rows.append(
                _ledger_row(
                    CHAMP, week, net=champ_series[i], is_champion=True,
                    champion_leg=champion_leg,
                )
            )
        if CHALLENGER in arms:
            end = week[1]
            if chal_week_end_shift:
                end = (
                    date.fromisoformat(week[1]) + timedelta(days=chal_week_end_shift)
                ).isoformat()
            priced_to = None
            if chal_priced_to_shift:
                priced_to = (
                    date.fromisoformat(week[1])
                    + timedelta(days=chal_priced_to_shift)
                ).isoformat()
            rows.append(
                _ledger_row(
                    CHALLENGER, week, net=chal_series[i], champion_leg=champion_leg,
                    week_end=end, priced_to=priced_to,
                    ledger_version=(
                        LEDGER_VERSION - 1 if i < chal_stale_versions else LEDGER_VERSION
                    ),
                )
            )
    return rows


# ── The cuts leaderboard fixture (the VETO source only, post-I8261) ───────────


def _row(name: str, mean: float | None, n: int) -> dict:
    return {
        "name": name,
        "kind": "challenger",
        "realized_rank_ic": None,
        "topn_alpha_vs_champion": None,
        "topn_alpha_vs_benchmark": None,
        "topn_alpha_vs_population": (
            None if mean is None else {"mean": mean, "se": 0.001, "t_stat": 2.0, "n_dates": n}
        ),
        "n_dates_scored": n,
        "confidence": "ok" if n >= 5 else ("thin" if n else "insufficient"),
        "top_n": 60,
    }


def _board(
    *,
    h21: tuple[tuple[float | None, int], tuple[float | None, int]] = ((0.001, 30), (0.500, 30)),
    h126_status: str = "immature",
    h126: tuple[tuple[float | None, int] | None, tuple[float | None, int] | None] = (None, None),
    h252_status: str = "immature",
    h252: tuple[tuple[float | None, int] | None, tuple[float | None, int] | None] = (None, None),
    include_arms: tuple[str, ...] = (CHAMP, CHALLENGER),
    duplicate: str | None = None,
) -> dict:
    """A cuts leaderboard in the corrected I7819 output contract: one row per
    arm per horizon block. Post-I8261 this board is the VETO source and the
    excluded-horizon report — never the decision evidence.

    The 21d block defaults to a LANDSLIDE for the challenger. Every test below
    must ignore it: after this cutover it may neither propose a promotion nor
    block one.
    """

    def block(h: int, status: str, pair) -> dict:
        specs = []
        for arm, ev in zip((CHAMP, CHALLENGER), pair, strict=True):
            if arm not in include_arms or ev is None:
                continue
            specs.append(_row(arm, ev[0], ev[1]))
            if duplicate == arm:
                specs.append(_row(arm, ev[0], ev[1]))
        return {
            "horizon_days": h,
            "status": status,
            "reason": None if status == "ok" else f"{h}d not matured",
            "n_dates": max((s["n_dates_scored"] for s in specs), default=0),
            "specs": specs,
        }

    return {
        "leaderboard_id": "cuts",
        "date": DATE,
        "per_arm_width": True,
        "min_dates_for_inference": 5,
        "horizons_days": list(LONG_HORIZONS_DAYS),
        "horizons": [
            block(21, "ok", h21),
            block(126, h126_status, h126),
            block(252, h252_status, h252),
        ],
    }


# ── The slot registry ─────────────────────────────────────────────────────────


def test_the_decision_basis_is_the_weekly_ledger_and_nothing_else():
    """The FIRST of the three invariants that replaced `forbidden_horizons_days`.

    Stronger than the rule it retires: v1 banned ONE horizon from being the
    decision basis; this says no leaderboard horizon can be, because the
    decision does not read a leaderboard for evidence at all.
    """
    assert CUT_PROMOTION_SLOT.decision_source == LEDGER_KEY
    assert CUT_PROMOTION_SLOT.decision_cadence == "weekly"
    assert CUT_PROMOTION_SLOT.primary_metric == "paired_weekly_net_log_return_vs_champion"
    assert CUT_PROMOTION_SLOT.ledger_return_column in LEDGER_COLS
    # Net, not gross: this slot's arms differ mainly in churn (42% vs 76%
    # week-over-week retention, 2026-07-27), so a gross decision is the trap.
    assert CUT_PROMOTION_SLOT.ledger_return_column == "net_log_return"


def test_forbidden_horizons_days_is_retired_not_merely_renamed():
    """I8261 requirement 4. The old field and its import assertion are gone, and
    a reader must not be able to find a decision horizon on this slot either —
    a lingering `decision_horizon_days` would be a second, contradicting answer
    to 'what does this decide on'."""
    assert not hasattr(CUT_PROMOTION_SLOT, "forbidden_horizons_days")
    assert not hasattr(CUT_PROMOTION_SLOT, "decision_horizon_days")
    assert not hasattr(CUT_PROMOTION_SLOT, "corroborating_horizon_days")  # singular


def test_the_excluded_horizon_can_neither_propose_nor_veto():
    """The SECOND replacement invariant, and it is strictly stronger than what
    it replaces: 21d was barred from proposing, and is now barred from vetoing
    too."""
    assert 21 in CUT_PROMOTION_SLOT.excluded_horizons_days
    assert not (
        set(CUT_PROMOTION_SLOT.excluded_horizons_days)
        & set(CUT_PROMOTION_SLOT.corroborating_horizons_days)
    )


def test_every_veto_horizon_matches_the_scanner_objective():
    """The THIRD replacement invariant. A veto horizon below 126 sessions does
    not match the scanner's ~1-year objective — the substance of
    alpha-engine-config-I7580."""
    assert CUT_PROMOTION_SLOT.corroborating_horizons_days == (126, 252)
    for h in CUT_PROMOTION_SLOT.corroborating_horizons_days:
        assert h >= MIN_VETO_HORIZON_DAYS
        assert h in LONG_HORIZONS_DAYS


def test_the_veto_reads_the_metric_the_board_calls_primary():
    cuts = slot_spec("cuts")
    assert CUT_PROMOTION_SLOT.corroborating_metric == cuts.primary_metric
    assert CUT_PROMOTION_SLOT.corroborating_min_dates == cuts.min_dates_for_inference


def test_arms_are_exactly_the_promotable_cuts():
    assert CUT_PROMOTION_SLOT.arms == PROMOTABLE_CUTS
    assert CUT_PROMOTION_SLOT.observe_only_arms == OBSERVE_ONLY_CUTS
    assert not set(PROMOTABLE_CUTS) & set(OBSERVE_ONLY_CUTS)
    assert set(SLOT_ARMS) == set(PROMOTABLE_CUTS) | set(OBSERVE_ONLY_CUTS)
    assert CUT_PROMOTION_SLOT.default_champion == DEFAULT_CUT_CHAMPION


def test_tech_score_is_not_restored_to_promotable_by_this_change():
    """Brian, 2026-08-24 on I8261: NOT in the same change. It has 2 scored
    dates; arming an automatic pointer write before evidence exists re-creates
    the exact condition of the 2026-08-21 I8060 ruling."""
    assert PROMOTABLE_CUTS == (CHAMP,)
    assert CHALLENGER in OBSERVE_ONLY_CUTS


def test_hysteresis_is_implemented_not_waived():
    """champion-challenger-policy.md §5.2 — either a margin+cooldown, or a §9.3
    delta. This slot implements it, so both must be non-zero."""
    assert CUT_PROMOTION_SLOT.promotion_margin > 0
    assert CUT_PROMOTION_SLOT.cooldown_days > 0


def test_the_margin_is_stated_in_weekly_units_and_preserves_the_old_bar():
    """The cutover changes the BASIS, not the BAR. 0.005 of mean lift over 126
    sessions is 25.2 weeks, so the same economic bar per unit of time is
    0.005/25.2 ≈ 0.0002 per week — and a reader of any record can check that
    arithmetic without leaving the artifact."""
    assert CUT_PROMOTION_SLOT.promotion_margin == pytest.approx(0.0002)
    assert 0.005 / (126 / SESSIONS_PER_WEEK) == pytest.approx(
        CUT_PROMOTION_SLOT.promotion_margin, rel=0.02
    )
    assert "0.005" in PROMOTION_MARGIN_NOTE
    assert "per week" in PROMOTION_MARGIN_NOTE
    assert "not a significance test" in PROMOTION_MARGIN_NOTE.lower()


def test_no_retired_v1_slug_is_re_minted_for_a_live_condition():
    """A slug means one thing forever or the multi-year decision series stops
    being readable across schema versions."""
    live = {
        REASON_PROMOTED, REASON_CHAMPION_LEADS, REASON_NO_PROMOTABLE_CHALLENGER,
        REASON_LEDGER_MISSING, REASON_LEDGER_ARM_MISSING, REASON_INSUFFICIENT_WEEKS,
        REASON_MARGIN_NOT_MET, REASON_COOLDOWN_ACTIVE,
        REASON_CORROBORATION_DISAGREES, REASON_BOARD_DEFECTIVE,
    }
    assert not live & set(RETIRED_V1_REASON_CODES)
    assert "board_missing" in RETIRED_V1_REASON_CODES
    assert "insufficient_dates" in RETIRED_V1_REASON_CODES


# ── The deliverable: a forward horizon can no longer decide ──────────────────


def test_a_forward_horizon_landslide_cannot_promote_without_the_ledger():
    """THE load-bearing guard of this change.

    The board hands the challenger a landslide at BOTH the 21d block and the
    126d block that used to be the decision horizon — mature, well past the
    evidence floor, exactly the shape that promoted under v1. There is no
    weekly ledger. The engine must hold, and must say the LEDGER is what it is
    missing, not a horizon.
    """
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=None,
        board=_board(h126_status="ok", h126=((0.010, 30), (0.900, 30))),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.champion == CHAMP
    assert d.reason_code == REASON_LEDGER_MISSING
    assert LEDGER_KEY in d.reason
    assert d.ledger["present"] is False


def test_a_21d_landslide_never_proposes_a_promotion():
    """The 21d block is populated with a challenger landslide by the fixture and
    is structurally excluded. With the weekly series showing the champion ahead,
    the outcome is `champion_already_leads` — never a promotion."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.001),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_CHAMPION_LEADS


def test_a_21d_block_favouring_the_incumbent_cannot_veto():
    """The other half, and the half v1 never had to answer: 21d is excluded from
    the veto set too, so a mature 21d block disagreeing with the weekly series
    must not block a promotion."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.010),
        board=_board(h21=((0.900, 30), (0.001, 30))),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.champion == CHALLENGER


# ── The weekly series ────────────────────────────────────────────────────────


def test_a_thin_weekly_series_is_refused_and_prices_itself_against_a_calendar():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.900, n_weeks=FLOOR - 1),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_INSUFFICIENT_WEEKS
    assert f"min_weeks_for_inference={FLOOR}" in d.reason
    assert d.arms[CHALLENGER].n_weeks_paired == FLOOR - 1
    assert d.decision_earliest_on in d.reason


def test_a_hold_still_reports_every_arms_week_count():
    """§3: an arm that produced nothing is recorded as a miss, never omitted —
    and n_weeks_paired is the number that says how far off a decision is."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=None,
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert set(d.arms) == set(TWO_ARM_SLOT.arms)
    for ev in d.arms.values():
        assert ev.n_weeks_paired == 0
        assert ev.present is False


def test_an_arm_with_no_ledger_rows_never_wins_by_default():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(chal_net=0.900, arms=(CHALLENGER,)),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_LEDGER_ARM_MISSING
    assert d.champion == CHAMP


def test_the_decision_is_paired_net_against_net_not_the_gross_embedded_leg():
    """weekly_ledger's in-row `champion_log_return` is a GROSS basket return, so
    differencing a NET arm against it charges the challenger's transaction cost
    and not the incumbent's — a systematic bias against the challenger in a slot
    whose arms differ mainly in churn.

    Here the embedded leg is set well ABOVE both arms' net returns, so a
    decision reading it would see the challenger losing badly. The net-vs-net
    join sees the challenger ahead, and the record carries both so the gap is
    visible rather than argued about.
    """
    rows = _ledger(champ_net=0.000, chal_net=0.010, champion_leg=0.500)
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=rows,
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.champion == CHALLENGER
    assert d.arms[CHALLENGER].mean_paired_log_return == pytest.approx(0.010)
    # The diagnostic disagrees, loudly, and is on the record.
    assert d.arms[CHALLENGER].mean_vs_embedded_champion_leg == pytest.approx(-0.490)
    assert d.arms[CHALLENGER].n_weeks_vs_embedded_champion_leg == 10


def test_a_week_whose_two_legs_disagree_about_its_end_is_dropped_and_counted():
    """The holding period ends when the next cut lands (weekly_ledger.
    holding_period), so two arms reporting different ends for the same start are
    not describing the same window. Differencing them compares different spans
    (§4, same cohort dates) — they are dropped, counted, and the shortfall
    surfaces as an immature series rather than as a silently narrower one."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.900, chal_week_end_shift=1),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_INSUFFICIENT_WEEKS
    assert d.arms[CHALLENGER].n_weeks_paired == 0
    assert d.arms[CHALLENGER].weeks_dropped_window_mismatch == 10


def test_two_legs_priced_over_different_spans_are_dropped_even_when_labelled_alike():
    """alpha-engine-config-I8264 established that a week's LABEL and the sessions
    it was actually priced from can diverge — a holiday re-cut, or a closing bar
    that has not landed. Two arms agreeing about what they claim to cover while
    being priced over different spans is exactly what `priced_from`/`priced_to`
    were added to make visible, and differencing them compares different spans.
    A guard on `week_end` alone would pass this."""
    rows = _ledger(champ_net=0.000, chal_net=0.900, chal_priced_to_shift=1)
    # The LABELS agree on both legs; only the priced span differs. A guard on
    # week_end alone sees nothing wrong here.
    assert {r["week_end"] for r in rows if r["arm"] == CHAMP} == {
        r["week_end"] for r in rows if r["arm"] == CHALLENGER
    }
    assert {r["priced_to"] for r in rows if r["arm"] == CHAMP} != {
        r["priced_to"] for r in rows if r["arm"] == CHALLENGER
    }
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT, ledger_rows=rows, board=_board(),
        champion_before=CHAMP, decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_INSUFFICIENT_WEEKS
    assert d.arms[CHALLENGER].n_weeks_paired == 0
    assert d.arms[CHALLENGER].weeks_dropped_window_mismatch == 10


def test_a_ledger_written_before_the_priced_span_columns_still_reconciles():
    """Rows predating alpha-engine-config-I8264 carry no `priced_from`/
    `priced_to`. Both legs read None, both agree, and the series must not empty
    itself — a guard that rejects an older ledger wholesale is worse than the
    ambiguity it removes."""
    rows = _ledger(champ_net=0.000, chal_net=0.100)
    for r in rows:
        r.pop("priced_from", None)
        r.pop("priced_to", None)
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT, ledger_rows=rows, board=_board(),
        champion_before=CHAMP, decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.arms[CHALLENGER].n_weeks_paired == 10
    assert d.arms[CHALLENGER].weeks_dropped_window_mismatch == 0


def test_rows_at_an_older_ledger_version_are_set_aside_and_counted():
    """LEDGER_VERSION is bumped only when a column's MEANING changes, so mixing
    versions in one mean silently averages two different quantities. Older rows
    sit alongside by design; the decision reads the current ones and REPORTS
    what it set aside, because a silently narrowed sample is indistinguishable
    from a short history."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.900, chal_stale_versions=8),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.reason_code == REASON_INSUFFICIENT_WEEKS
    assert d.arms[CHALLENGER].n_weeks_scored == 2
    assert d.arms[CHALLENGER].weeks_dropped_stale_version == 8


def test_a_missing_week_makes_the_chained_read_none_never_a_shorter_span():
    """A gap in a compounding series is not a zero — it is a span that cannot be
    stated. The mean is still computable over the weeks that paired; the CHAINED
    number over a gapped span is not, and must not silently report a 9-week
    return under a 10-week label."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(
            champ_net=0.000, chal_net=[0.010] * 4 + [None] + [0.010] * 5
        ),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    ev = d.arms[CHALLENGER]
    assert ev.n_weeks_paired == 9
    assert ev.weeks_dropped_unpaired == 1
    # The dropped week is excluded from the series entirely, so what IS chained
    # is the 9 weeks that paired — and the record says it is 9.
    assert ev.chained_paired_log_return == pytest.approx(0.090)


def test_the_clustered_statistic_is_recorded_but_is_not_a_gate():
    """§5 rejects a publication-grade bar for an operational loop, so the t-stat
    never blocks. It is recorded because a reader must be able to see whether
    the margin was cleared with or without statistical support."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(
            champ_net=0.000, chal_net=[0.01, -0.02, 0.03, -0.01, 0.02, 0.05, -0.03, 0.04, 0.01, 0.00]
        ),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    ev = d.arms[CHALLENGER]
    assert ev.se is not None
    assert ev.t_stat is not None
    # Well under any conventional significance bar, and it still promotes: the
    # gate is the margin plus the cooldown, not the t-stat.
    assert abs(ev.t_stat) < 2.0
    assert d.decision == "promote"


# ── Hysteresis ────────────────────────────────────────────────────────────────


def test_lead_under_the_margin_is_not_a_promotion():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.010 + MARGIN / 2),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_MARGIN_NOT_MET
    assert "per week" in d.reason


def test_cooldown_blocks_an_otherwise_clean_promotion():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
        last_promoted_on="2026-08-15",
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_COOLDOWN_ACTIVE
    assert d.champion == CHAMP


def test_incumbent_still_leading_is_a_hold_with_its_own_slug():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.100, chal_net=0.010),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_CHAMPION_LEADS


def test_an_exact_tie_resolves_to_the_incumbent():
    """Hysteresis before the margin is even consulted: the champion's own paired
    difference is 0.0 by construction, so a challenger must be strictly positive
    to lead at all."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.010),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_CHAMPION_LEADS
    assert d.arms[CHAMP].mean_paired_log_return == pytest.approx(0.0)
    assert d.arms[CHAMP].is_champion is True


# ── The vetoes ────────────────────────────────────────────────────────────────


def test_a_mature_long_horizon_vetoes_a_weekly_promotion():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
        board=_board(h252_status="ok", h252=((0.200, 30), (0.010, 30))),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_CORROBORATION_DISAGREES
    assert d.corroborating["blocked_by"] == [252]
    assert d.corroborating["horizons"]["252"]["leader"] == CHAMP
    assert d.corroborating["horizons"]["252"]["role"] == "veto_only"


def test_the_126_horizon_is_now_a_veto_too_not_the_decision_basis():
    """Brian's ruling names BOTH 126 and 252 as corroborating vetoes. Under v1,
    126 was the decision horizon and could promote on its own."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
        board=_board(h126_status="ok", h126=((0.200, 30), (0.010, 30))),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.reason_code == REASON_CORROBORATION_DISAGREES
    assert d.corroborating["blocked_by"] == [126]


def test_a_mature_long_horizon_can_never_propose_a_promotion():
    """The asymmetry, stated as a test: 126 and 252 both hand the challenger a
    landslide, and the weekly series has the champion ahead. A veto that could
    vote would promote here."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.100, chal_net=0.000),
        board=_board(
            h126_status="ok", h126=((0.001, 30), (0.900, 30)),
            h252_status="ok", h252=((0.001, 30), (0.900, 30)),
        ),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.champion == CHAMP
    assert d.reason_code == REASON_CHAMPION_LEADS


def test_an_immature_veto_is_reported_never_counted_as_a_pass():
    """§5.1: you cannot gate on a statistic you did not measure — and an
    uncomputed gate reported as a pass is the defect the rule prevents. At
    rollout this is the normal state for BOTH veto horizons."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.corroborating["blocking"] is False
    assert d.corroborating["mature_horizons"] == []
    for h in ("126", "252"):
        entry = d.corroborating["horizons"][h]
        assert entry["mature"] is False
        assert entry["disagrees"] is False
        assert "non-blocking" in entry["note"]
    assert "no corroborating horizon is mature yet" in d.reason


def test_a_veto_below_the_boards_evidence_floor_is_non_blocking_and_says_so():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
        board=_board(h252_status="ok", h252=((0.900, 2), (0.001, 2))),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    entry = d.corroborating["horizons"]["252"]
    assert entry["mature"] is False
    assert "evidence floor" in entry["note"]


def test_a_missing_board_makes_every_veto_unmeasured_and_is_not_a_hold():
    """v1 held on `board_missing`. Post-cutover the board is only the veto
    source, and an unmeasured veto is non-blocking (§5.1) — so an absent board
    must not stop a decision the weekly ledger fully supports."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
        board=None,
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.corroborating["blocking"] is False
    assert d.corroborating["horizons"]["126"]["status"] == "absent"


def test_a_clean_promotion_moves_the_pointer():
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "promote"
    assert d.champion == CHALLENGER
    assert d.reason_code == REASON_PROMOTED
    assert d.last_promoted_on == DATE
    assert "paired weeks" in d.reason
    assert "chained" in d.reason


# ── Structural defects: record, then fail loud ───────────────────────────────


def test_a_corrupt_board_holds_even_though_a_missing_one_does_not():
    """The asymmetry a v1 reader will not expect, and it is deliberate. An
    ABSENT veto is honestly unmeasured and non-blocking. A board reporting
    duplicate arm rows is not unmeasured — it is UNRELIABLE, and a safety
    mechanism that may be reading someone else's numbers is worse than one that
    is switched off."""
    d = decide_cut_champion(
        slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
        board=_board(duplicate=CHALLENGER),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_BOARD_DEFECTIVE
    assert d.defect and CHALLENGER in d.defect


# ── The artifact: schema + producer/consumer contract ────────────────────────


class _S3:
    def __init__(self, objects: dict[str, dict] | None = None):
        self.objects = dict(objects or {})
        self.written: dict[str, dict] = {}
        self.write_order: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key in self.written:
            body = json.dumps(self.written[Key]).encode()
        elif Key in self.objects:
            body = json.dumps(self.objects[Key]).encode()
        else:
            raise RuntimeError(f"NoSuchKey: {Key}")
        return {"Body": _Body(body)}

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803
        self.written[Key] = json.loads(Body)
        self.write_order.append(Key)


class _Body:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


def test_run_writes_all_three_keys_dated_record_first():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    doc = run_cut_promotion(
        DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT, ledger_rows=None
    )
    assert doc["decision"] == "hold"
    assert s3.write_order == [
        AUDIT_DATED_KEY.format(date=DATE),
        AUDIT_LATEST_KEY,
        CUT_CHAMPION_POINTER_KEY,
    ]
    for key in s3.write_order:
        assert s3.written[key]["decision"] == "hold"


def test_an_absent_ledger_object_reads_as_absent_not_as_an_empty_series():
    """§7.2. `run_cut_promotion` with no `ledger_rows` argument goes to S3, and
    the fake store has no ledger object. That must render as present:false and
    a `weekly_ledger_missing` hold — never as a healthy zero-row series."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT)
    assert doc["reason_code"] == REASON_LEDGER_MISSING
    assert doc["ledger"]["present"] is False
    assert doc["ledger"]["key"] == LEDGER_KEY


def test_every_written_document_validates_against_the_frozen_schema():
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    cases = [
        (None, False),
        (_ledger(champ_net=0.010, chal_net=0.100), True),
        (_ledger(champ_net=0.010, chal_net=0.100, n_weeks=FLOOR - 1), False),
        (_ledger(champ_net=0.100, chal_net=0.010), False),
        (_ledger(champ_net=0.010, chal_net=0.010 + MARGIN / 2), False),
    ]
    for rows, promoted in cases:
        s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
        doc = run_cut_promotion(
            DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT, ledger_rows=rows
        )
        jsonschema.validate(instance=doc, schema=schema)
        assert (doc["decision"] == "promote") is promoted


def test_a_vetoed_record_validates_against_the_frozen_schema():
    s3 = _S3(
        {
            f"research/cuts_leaderboard/{DATE}.json": _board(
                h252_status="ok", h252=((0.200, 30), (0.010, 30))
            )
        }
    )
    doc = run_cut_promotion(
        DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
    )
    assert doc["reason_code"] == REASON_CORROBORATION_DISAGREES
    jsonschema.validate(instance=doc, schema=_schema())


def test_the_pointer_this_engine_writes_is_the_one_the_feed_reads():
    """Producer/consumer contract test (M0). The consumer is
    universe_membership.live_cut_champion — the function the sector-team feed
    actually resolves through — not a re-read of the same JSON."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    doc = run_cut_promotion(
        DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
    )
    assert doc["champion"] == CHALLENGER
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        assert live_cut_champion(bucket="b", s3_client=s3) == CHALLENGER


def test_an_observe_only_arm_can_never_be_served_even_if_the_pointer_names_it():
    """The registry is the last line, not the engine (alpha-engine-config-I8060,
    reaffirmed on I8261). Even a stale pointer, a hand-edited object, or an
    engine running an older registry cannot hand the sector teams an arm Brian
    has not cleared — the refusal does not depend on the promotion engine being
    correct."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    run_cut_promotion(
        DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
    )
    with pytest.raises(Exception) as exc:
        live_cut_champion(bucket="b", s3_client=s3)
    assert CHALLENGER in str(exc.value)
    assert CHAMP in str(exc.value)


def test_a_hold_leaves_the_consumer_on_the_standing_champion():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    run_cut_promotion(
        DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT, ledger_rows=None
    )
    assert live_cut_champion(bucket="b", s3_client=s3) == DEFAULT_CUT_CHAMPION


def test_a_defective_board_is_recorded_before_the_engine_raises():
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board(duplicate=CHALLENGER)})
    with pytest.raises(CutPromotionError):
        run_cut_promotion(
            DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
            ledger_rows=_ledger(champ_net=0.000, chal_net=0.100),
        )
    record = s3.written[AUDIT_DATED_KEY.format(date=DATE)]
    assert record["decision"] == "hold"
    assert record["defect"]
    jsonschema.validate(instance=record, schema=_schema())


def test_demotion_needs_the_same_margin_in_reverse_plus_the_cooldown():
    """§5.2 is symmetric: a promotion one week must not be undone the next.
    Also pins that `last_promoted_on` carries forward across the hold."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        first = run_cut_promotion(
            DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
            ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
        )
        assert first["decision"] == "promote"
        assert first["last_promoted_on"] == DATE
        later = "2026-08-29"
        s3.objects[f"research/cuts_leaderboard/{later}.json"] = _board()
        # The incumbent is now tech_score_top_60, so the ledger legs swap roles:
        # every difference is taken against IT this time.
        second = run_cut_promotion(
            later, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
            ledger_rows=_ledger(champ_net=0.200, chal_net=0.010),
        )
        assert second["reason_code"] == REASON_COOLDOWN_ACTIVE
        assert second["champion"] == CHALLENGER
        assert second["last_promoted_on"] == DATE
        assert live_cut_champion(bucket="b", s3_client=s3) == CHALLENGER


def test_schema_is_referenced_by_the_producer_module():
    assert SCHEMA_PATH.exists()
    assert "scanner_cut_champion.schema.json" in (
        REPO_ROOT / "scoring" / "cut_promotion.py"
    ).read_text(encoding="utf-8")


# ── Observe-only: the slot with no promotable challenger (I8060 / I8261) ─────


def test_the_live_registry_has_no_promotable_challenger_and_says_so():
    """Brian rulings 2026-08-21 and 2026-08-24: `tech_score_top_60` is
    observe-only until it has a scored cohort.

    The engine must NAME that state. Falling through to the comparison path and
    reporting `champion_already_leads` would be a claim about evidence made
    where no comparison happened — two different things rendering identically,
    which is the §3 failure this codebase keeps paying for.
    """
    d = decide_cut_champion(
        ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
        board=_board(),
        champion_before=CHAMP,
        decided_on=DATE,
    )
    assert d.decision == "hold"
    assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER
    assert d.champion == CHAMP
    for arm in OBSERVE_ONLY_CUTS:
        assert arm in d.reason


def test_the_no_challenger_hold_does_not_depend_on_the_ledger_or_the_board():
    """It is a fact about the REGISTRY. A missing ledger, a missing board, or a
    fully populated one must not change it — otherwise the reason flips week to
    week for a state that has not changed."""
    for rows in (None, _ledger(champ_net=0.010, chal_net=0.900)):
        for board in (None, _board(), {"status": "unmeasurable", "reason": "no cohort"}):
            d = decide_cut_champion(
                ledger_rows=rows, board=board, champion_before=CHAMP, decided_on=DATE
            )
            assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER, (rows is None, board)


def test_the_record_is_still_written_every_cycle_while_holding():
    """A loop that is armed and waiting must not look like a dead one: the
    unconditional audit record is the whole reason this engine writes on a hold
    (champion-challenger-policy.md §3)."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert doc["reason_code"] == REASON_NO_PROMOTABLE_CHALLENGER
    assert f"config/apply_audit/scanner_cut_champion/{DATE}.json" in s3.written
    assert "config/apply_audit/scanner_cut_champion/latest.json" in s3.written
    assert live_cut_champion(bucket="b", s3_client=s3) == CHAMP


def test_the_live_record_still_reports_hold_no_promotable_challenger():
    """The measured live state (s3://alpha-engine-research/config/
    scanner_cut_champion.json reports decision=hold, reason_code=
    no_promotable_challenger) must survive this cutover unchanged. The slot has
    ONE promotable arm, so there is no decision to take regardless of what it
    decides ON, and this change must not fabricate one."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3)
    assert doc["decision"] == "hold"
    assert doc["reason_code"] == REASON_NO_PROMOTABLE_CHALLENGER
    assert doc["champion"] == DEFAULT_CUT_CHAMPION
    jsonschema.validate(doc, _schema())


def test_the_observe_only_arms_are_still_scored():
    """Non-promotable must never quietly become non-measured (§3)."""
    import inspect

    import scoring.leaderboard_producers as lp

    src = inspect.getsource(lp._load_cut_specs)
    assert "OBSERVE_ONLY_CUTS" in src
    assert set(SLOT_ARMS) - set(PROMOTABLE_CUTS) == set(OBSERVE_ONLY_CUTS)


def test_restoring_an_arm_is_a_one_line_registry_edit():
    """The exit from this state must be cheap and obvious, or the hold becomes
    permanent by inertia. Nothing outside the registry names the arm set."""
    d = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE
    )
    assert "PROMOTABLE_CUTS" in d.reason


# ── The record is self-qualifying (I8257, carried onto the I8261 basis) ──────


def test_every_arm_carries_its_own_metric_cadence_and_source():
    """n_weeks_paired must be self-qualifying without a reader joining back to
    CUT_PROMOTION_SLOT by hand — the exact join the I8257 misread skipped, now
    on the new basis where the confusion would be 'weeks of what, against
    what'."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        doc = run_cut_promotion(
            DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
            ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
        )
    for arm in (CHAMP, CHALLENGER):
        assert doc["arms"][arm]["metric"] == CUT_PROMOTION_SLOT.primary_metric
        assert doc["arms"][arm]["cadence"] == "weekly"
        assert doc["arms"][arm]["source"] == LEDGER_KEY
    assert doc["decision_metric"] == CUT_PROMOTION_SLOT.primary_metric
    assert doc["decision_source"] == LEDGER_KEY
    assert doc["decision_cadence"] == "weekly"


def test_a_hold_that_never_reads_the_ledger_still_stamps_metric_and_source():
    """Even the registry-only no_promotable_challenger hold — which never reads
    a row — must not leave n_weeks_paired=0 unqualified."""
    d = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE
    )
    assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER
    for ev in d.arms.values():
        assert ev.metric == CUT_PROMOTION_SLOT.primary_metric
        assert ev.cadence == "weekly"
        assert ev.source == LEDGER_KEY
        assert ev.present is False


def test_excluded_horizons_carries_the_21d_measurement_with_its_caveat():
    """The excluded 21d block IS measured every cycle (the fixture always
    populates it with a landslide for the challenger) — the record must carry
    that number, why it is neither a decision input nor a veto, and the
    contamination caveat, rather than making a reader open the leaderboard to
    learn any of the three."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms):
        doc = run_cut_promotion(
            DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT,
            ledger_rows=_ledger(champ_net=0.010, chal_net=0.100),
        )
    excl = doc["excluded_horizons"]
    for arm in (CHAMP, CHALLENGER):
        entry = excl[arm]["21"]
        assert entry["horizon_days"] == 21
        assert entry["n_dates_scored"] == 30
        assert CONTAMINATION_CAVEAT in entry["excluded_reason"]
        assert "alpha-engine-config-I7580" in entry["excluded_reason"]
    # 126 and 252 are NOT in the excluded block — they are vetoes, and a reader
    # must be able to tell "excluded entirely" from "may block but not propose".
    assert "126" not in excl[CHAMP]
    assert "252" not in excl[CHAMP]


def test_excluded_horizons_is_written_even_with_no_board():
    """Unconditional, same discipline as the rest of this record (§3): a missing
    board must not make the excluded-horizons block disappear."""
    d = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE,
        slot=TWO_ARM_SLOT,
    )
    assert d.reason_code == REASON_LEDGER_MISSING
    entry = d.excluded_horizons[CHAMP]["21"]
    assert entry["n_dates_scored"] == 0
    assert "No 21d block available" in entry["excluded_reason"]


# ── decision_earliest_on, on the WEEKLY floor (I8261 requirement 2) ──────────


def test_decision_earliest_on_is_the_weekly_floor_not_a_forward_horizon():
    """The concrete number the ruling buys. FIRST_COHORT_DATE plus
    min_weeks_for_inference weekly holding periods — 2026-09-25 — against the
    v1 basis's FIRST_COHORT_DATE + 126 sessions = 2027-02-22."""
    expected = add_trading_days(
        FIRST_COHORT_DATE, CUT_PROMOTION_SLOT.min_weeks_for_inference * SESSIONS_PER_WEEK
    ).isoformat()
    assert expected == "2026-09-25"
    d = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE
    )
    assert d.decision_earliest_on == expected
    # The v1 answer, pinned so a regression to the forward-horizon projection is
    # a failing assertion and not a silently later date.
    v1 = add_trading_days(FIRST_COHORT_DATE, 126).isoformat()
    assert v1 == "2027-02-22"
    assert d.decision_earliest_on != v1
    assert d.decision_earliest_on.startswith("2026-09")


def test_decision_earliest_on_does_not_move_with_decided_on():
    """It is a property of the SLOT, never of when a given cycle happens to run
    — otherwise a run today and a run next week would disagree about when the
    evidence matures."""
    d1 = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE
    )
    d2 = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on="2026-09-05"
    )
    assert d1.decision_earliest_on == d2.decision_earliest_on == decision_earliest_on()


# ── The reconciliation guard, moved to the new source ────────────────────────


def test_reconcile_catches_a_record_that_disagrees_with_its_own_ledger():
    rows = _ledger(champ_net=0.010, chal_net=0.100)
    doc = {
        "champion_before": CHAMP,
        "decision_column": "net_log_return",
        "arms": {
            CHALLENGER: {"n_weeks_paired": 0, "present": True},  # the ledger has 10
        },
    }
    mismatches = reconcile_arms_with_ledger(doc, rows)
    assert len(mismatches) == 1
    assert CHALLENGER in mismatches[0]
    assert "10 paired week" in mismatches[0]


def test_reconcile_is_silent_when_the_record_agrees():
    rows = _ledger(champ_net=0.010, chal_net=0.100)
    doc = {
        "champion_before": CHAMP,
        "decision_column": "net_log_return",
        "arms": {CHALLENGER: {"n_weeks_paired": 10, "present": True}},
    }
    assert reconcile_arms_with_ledger(doc, rows) == []


def test_reconcile_ignores_arms_the_decision_never_read():
    """present=False means the decision path never consulted the ledger for this
    arm at all (ledger absent, the registry-only short-circuit) — nothing to
    reconcile, and this must not false-positive."""
    rows = _ledger(champ_net=0.010, chal_net=0.100)
    doc = {
        "champion_before": CHAMP,
        "decision_column": "net_log_return",
        "arms": {CHALLENGER: {"n_weeks_paired": 0, "present": False}},
    }
    assert reconcile_arms_with_ledger(doc, rows) == []


def test_run_cut_promotion_fails_when_the_written_record_disagrees_with_the_ledger():
    """End-to-end: a corrupted read path (simulated here — decide_cut_champion
    itself never produces this, by construction) must not reach S3 unchallenged.
    The record is still WRITTEN first, then the process fails loud rather than
    serving a self-contradicting artifact silently."""
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": _board()})
    rows = _ledger(champ_net=0.010, chal_net=0.100)
    real = __import__("scoring.cut_promotion", fromlist=["_arm_evidence"])._arm_evidence

    def _bad(**kwargs) -> ArmEvidence:
        ev = real(**kwargs)
        ev.n_weeks_paired = 999  # deliberately wrong vs the ledger's real 10
        return ev

    with patch("scoring.universe_membership.PROMOTABLE_CUTS", TWO_ARM_SLOT.arms), patch(
        "scoring.cut_promotion._arm_evidence", side_effect=_bad
    ), pytest.raises(CutPromotionError, match="disagrees with the weekly ledger"):
        run_cut_promotion(
            DATE, bucket="b", s3_client=s3, slot=TWO_ARM_SLOT, ledger_rows=rows
        )

    # Record-first-then-fail: the corrupted record is durable, not swallowed.
    written = s3.written[AUDIT_DATED_KEY.format(date=DATE)]
    assert written["arms"][CHAMP]["n_weeks_paired"] == 999


# ── The frozen contract ──────────────────────────────────────────────────────


def test_the_schema_is_v2_and_pins_the_new_decision_basis():
    schema = _schema()
    assert schema["properties"]["schema_version"]["const"] == 2
    for key in (
        "decision_metric", "decision_cadence", "decision_source", "decision_column",
        "excluded_horizons", "decision_earliest_on", "ledger", "corroborating",
    ):
        assert key in schema["required"], key
    assert schema["properties"]["decision_source"]["const"] == LEDGER_KEY
    assert schema["properties"]["decision_cadence"]["const"] == "weekly"
    # The v1 decision-horizon field is gone from the contract, not merely
    # unused: leaving `horizon_days: {const: 126}` would let a v2 record assert
    # a decision horizon it does not have.
    assert "horizon_days" not in schema["properties"]
    assert "primary_metric" not in schema["properties"]


def test_the_schema_reason_codes_match_the_module_and_exclude_the_retired_ones():
    enum = set(_schema()["properties"]["reason_code"]["enum"])
    assert enum == {REASON_PROMOTED, *(
        c for c in (
            REASON_CHAMPION_LEADS, REASON_NO_PROMOTABLE_CHALLENGER,
            REASON_LEDGER_MISSING, REASON_LEDGER_ARM_MISSING,
            REASON_INSUFFICIENT_WEEKS, REASON_MARGIN_NOT_MET,
            REASON_COOLDOWN_ACTIVE, REASON_CORROBORATION_DISAGREES,
            REASON_BOARD_DEFECTIVE,
        )
    )}
    assert not enum & set(RETIRED_V1_REASON_CODES)


def test_the_schema_requires_the_margin_units_on_every_record():
    """A bare number carried across the v1→v2 boundary would be read as the same
    bar in different clothes, which is exactly the confusion this cutover has to
    avoid creating."""
    hyst = _schema()["properties"]["hysteresis"]
    for key in ("promotion_margin_units", "promotion_margin_note", "min_weeks_for_inference"):
        assert key in hyst["required"], key
    assert "min_dates_for_inference" not in hyst["properties"]


# ── The registry-state hold must still REPORT the ledger it was handed
# (alpha-engine-config-I9276) ────────────────────────────────────────────────
#
# Measured live on config/apply_audit/scanner_cut_champion/2026-08-28.json:
#
#     "ledger": {"present": true, "rows_read": 5, "arms_present": []}
#     "arms": {"attractiveness_top_60": {"present": false,
#                                        "n_weeks_scored": 0,
#                                        "n_weeks_paired": 0, ...}}
#
# while research/cuts_weekly_ledger/ledger.parquet carried a scored row for
# attractiveness_top_60 over 2026-08-21..2026-08-28 (net_log_return 0.002985).
# The record denied evidence its own ledger block said it had read. Because
# `len(slot.arms) < 2` returned ahead of the ledger read, `hold()`'s all-zero
# default ArmEvidence went out unconditionally — and n_weeks_paired is the one
# number this slot publishes to say how far off a real decision is.


def test_the_registry_state_hold_reports_the_ledger_rows_it_was_given():
    """A hold taken on a REGISTRY state must still carry the MEASUREMENT
    (champion-challenger-policy.md §3: promotion changes which arm is
    consumed, never what is measured).

    PRE-FIX: RED — `present` is False, `n_weeks_scored` is 0 and
    `arms_present` is empty even though two ledger rows were handed in.
    """
    rows = _ledger(champ_net=0.010, chal_net=0.100)
    d = decide_cut_champion(
        ledger_rows=rows, board=None, champion_before=CHAMP, decided_on=DATE
    )
    assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER, (
        "the live registry has one promotable arm; this test is about what the "
        "hold REPORTS, not about which branch it takes"
    )
    champ = d.arms[CHAMP]
    assert champ.present is True, (
        "the champion has ledger rows — reporting present=false is the record "
        "contradicting its own ledger block"
    )
    assert champ.n_weeks_scored > 0
    assert champ.metric == CUT_PROMOTION_SLOT.primary_metric
    assert champ.source == LEDGER_KEY
    assert d.ledger["rows_read"] == len(rows)
    assert CHAMP in d.ledger["arms_present"]


def test_the_registry_state_hold_with_no_ledger_still_reports_absence_honestly():
    """The other half of the same claim: with no ledger there is nothing to
    report, and the record must say so rather than inventing a read."""
    d = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE
    )
    assert d.reason_code == REASON_NO_PROMOTABLE_CHALLENGER
    assert d.arms[CHAMP].present is False
    assert d.arms[CHAMP].n_weeks_scored == 0
    assert d.ledger["arms_present"] == []
