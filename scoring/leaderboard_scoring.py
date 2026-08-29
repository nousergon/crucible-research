"""Shared champion/challenger leaderboard SCORING engine
(config#1221 scanner + config#1223 producer — ONE engine, ARCHITECTURE §37
lift-to-chokepoint).

The scanner (config#1221) and producer (config#1223) champion/challenger
substrates already SHIP their shadow artifacts:

- scanner specs → ``candidates_shadow/{spec}/{date}/candidates.json``
  (PR #308; ``data/scanner_specs.py`` + ``data/scanner_orchestrator.py``)
- research producers → ``signals_shadow/{producer}/{date}/signals.json``
  (PRs #309/#310/#311; ``producers/registry.py`` + ``producers/runner.py``)

The MISSING piece both backlog issues call for is the SCORER that reads those
shadow artifacts, joins each spec's per-ticker picks to the realized forward
return, and scores every spec against the champion on the SAME two objectives
the cutover gates name (see OBSERVATION_REGISTRY.yaml):

1. **Cross-sectional realized rank-IC** — Spearman rank correlation between a
   spec's per-ticker ranking signal on date *d* and the realized forward return
   over the next *h* trading days, computed PER DATE and averaged across the
   observed cohort. Significance is **date-clustered**: each date is one
   independent cluster (weeks-as-N), so the t-stat is ``mean / SE`` of the
   per-date IC series — never the naive cross-sectional n that double-counts the
   within-week correlation the #1142 work flagged.
2. **Long-only top-N realized alpha vs the champion** — mean realized forward
   return of the spec's top-N picks minus the champion's top-N picks, per date,
   averaged, with the same date-clustered t-stat. This is the scanner's OWN
   long-only objective (config#1186 reconciliation) and the producer's
   selection objective.

DESIGN — this module is PURE and side-effect-free (no S3, no boto3, no clock):
it takes already-loaded shadow picks + an externally-resolved realized-return
map and returns a leaderboard dict. The two thin producers in
``scoring/leaderboard_producers.py`` do the I/O (read shadow artifacts, resolve
returns, write the JSON) and are fail-soft. This mirrors the off-hot-path,
reads-persisted-artifacts shape of ``scripts/build_agent_quality.py`` and keeps
the statistics unit-testable with zero AWS.

COHORT-GATED, by design: a date contributes to a metric only when its realized
forward return exists (the join is non-null) — so on a fresh date with no matured
21d outcome the leaderboard ships with ``n_dates=0`` and every metric ``None``
(an honest "not yet scorable", never a fabricated value). It scores meaningfully
only as forward cohorts mature; full closure of #1221/#1223 is the same
cohort-gate the OBSERVATION_REGISTRY rows already name (earliest_flip 2026-07-20).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

logger = logging.getLogger(__name__)

# Default forward-return horizon: 21 trading days (~1 month), the horizon both
# cutover gates name ("realized 21d outcomes"). This stays the PRIMARY horizon
# — the top-level block of every leaderboard artifact and the one every
# existing consumer reads — so that a promoted arm's series is continuous
# across this change (champion-challenger-policy.md §3: a promoted arm keeps
# its history; promotion does not reset a series, and neither may a scorer
# change).
DEFAULT_HORIZON_DAYS = 21

# Every horizon an arm is scored at, PRIMARY FIRST (alpha-engine-config-I7540).
#
# 21 alone could not answer the question the scanner exists to answer. The
# scanner's stated objective is to surface names attractive over ~1 year,
# comparable to sell-side coverage; grading every arm on a 21-session horizon
# marks a long-horizon thesis on a horizon ~12x shorter than the thing it is
# trying to do. An arm built for a long view loses that comparison whether or
# not its view is correct, and a momentum-tilted arm wins it whether or not the
# tilt serves the objective.
#
# 126 ≈ 6 months, 252 ≈ 1 year, in TRADING sessions (the panel's own calendar).
#
# This was historically blocked, not merely unbuilt: the leaderboards read
# ``staging/daily_closes/``, a prefix under a 7-day S3 expiry, so even 21
# sessions was structurally unservable (alpha-engine-config-I5195). The source
# is now the durable ArcticDB universe library, which carries NO lifecycle
# expiry rule (verified live 2026-08-17 against
# ``get-bucket-lifecycle-configuration`` on ``alpha-engine-research``: exactly
# two rules, ``expire-staging-after-7-days`` on ``staging/`` and
# ``feature-store-retention`` on ``features/`` — neither matches ``arcticdb/``)
# and ``nousergon_lib.arcticdb`` already defaults to a 730-day read window. So
# there is no storage cost and no new producer behind a 252-session horizon —
# only cohort maturity, which is a matter of TIME and must render as immature
# rather than as a zero (§7.2).
LONG_HORIZONS_DAYS: tuple[int, ...] = (21, 126, 252)

# Below this many scored dates a per-date mean is not an inference, it is an
# anecdote (alpha-engine-config-I7542). ``date_clustered_stats`` already
# returns ``se: None`` / ``t_stat: None`` at n=1 — but a mean rendered beside
# two nulls in the SAME shape as a real result reads as a result. 5 is the
# smallest n at which the date-clustered SE is computed over enough weeks-as-N
# clusters to carry any signal at all; it is a per-slot fact and lives in the
# slot registry below (champion-challenger-policy.md §10), never at a call
# site.
MIN_DATES_FOR_INFERENCE = 5

# Per-spec confidence vocabulary. Deliberately NOT shared with the
# leaderboard-level ``status`` vocabulary (``ok`` / ``unmeasurable``): machine
# health and experiment performance never share a grade vocabulary
# (champion-challenger-policy.md §8), and these three answer "how much evidence
# stands behind THIS row", not "did the measurement work".
CONFIDENCE_OK = "ok"
CONFIDENCE_THIN = "thin"
CONFIDENCE_INSUFFICIENT = "insufficient"

# Per-row CROSS-ARM comparability vocabulary (alpha-engine-config-I9274).
# A fourth vocabulary on purpose, and the reason is the defect: ``confidence``
# answers "how much evidence stands behind THIS row" and was being read as an
# answer to "can this row be compared against the champion's". Those are
# different questions with different remedies, and conflating them is what let
# ``scanner/leaderboard/2026-08-28.json`` publish the champion at
# ``topn_alpha_vs_population`` mean 0.031348, t 11.41, beside two challengers
# at ``confidence: insufficient`` — with nothing on the artifact saying the two
# sides share ZERO cohort dates and are therefore not comparable at all.
COMPARISON_OK = "ok"
COMPARISON_SELF = "self"                       # this row IS the champion
COMPARISON_NO_CHAMPION = "no_champion"         # the board registers no champion
COMPARISON_NO_COMMON_COHORT = "no_common_cohort"   # the intersection is empty
COMPARISON_WIDTH_MISMATCH = "width_mismatch"   # paired across widths measures breadth
COMPARISON_PARTIAL_COHORT = "partial_cohort"   # see `apply_cohort_intersection`

# Two ADDITIVE axes landed in the same week and they answer different
# questions — keep both, and do not collapse them:
#
#   comparison    (I9274) — can this row be compared AGAINST THE CHAMPION?
#                           A width mismatch or an empty intersection is a
#                           property of the PAIR.
#   measurability (I9307) — can this arm produce a comparable output AT ALL?
#                           A property of the ARM, true even with no champion.
#
# An arm can be `comparison: ok` and `measurability: unmeasurable` only in the
# degenerate case where nothing is comparable; the reverse — measurable but not
# comparable — is the common one and is exactly why they are separate.
# Per-spec MEASURABILITY vocabulary (alpha-engine-config-I9307). A THIRD axis,
# deliberately separate from both ``confidence`` (how much evidence stands
# behind this row) and the leaderboard-level ``status`` (did the measurement
# run at all), because the defect it exists to name is invisible on either:
#
#   confidence says  "this arm has little evidence"        -> keep watching
#   measurability says "this arm CANNOT produce evidence"  -> fix it
#
# The champion arm `scanner_predictor_direct` was scored from an artifact whose
# live producer is empty-by-contract. It could not score, ever. For seven weeks
# it rendered `thin` — a word that means "not enough yet", i.e. a state that
# resolves itself by waiting. Nothing waited it out because nothing was coming.
#
# champion-challenger-policy.md §7.2: an unmeasurable result must fail LOUD,
# never render as an empty success. §3: silent absence and a genuine zero must
# never render identically.
MEASURABILITY_MEASURED = "measured"
MEASURABILITY_UNMEASURABLE = "unmeasurable"

# An arm scoring nothing among the most recent ``COHORT_LAG_UNMEASURABLE_DATES``
# dates on which the cohort DID score is structurally behind, not thin. Three
# weekly cycles is deliberately short: the defect this catches ran for seven.
COHORT_LAG_UNMEASURABLE_DATES = 3


class LeaderboardIntegrityError(RuntimeError):
    """A leaderboard artifact violates a structural invariant every consumer
    assumes, so it must not be written or decided on.

    Distinct from ``unmeasurable`` on purpose (champion-challenger-policy.md
    §8): ``unmeasurable`` says the measurement could not be taken, this says
    the artifact does not mean what its shape claims.
    """


def duplicate_arm_rows(board: Mapping[str, Any]) -> list[str]:
    """Every ``(surface, arm)`` a board reports MORE THAN ONCE.

    One row per arm per horizon is the invariant every consumer of these
    boards reads on: ``cut_promotion._rows_for`` cannot say which of two rows
    is the arm's, the console pane renders an arm twice, and a mean-of-means
    over a doubled row silently reweights it.

    It is checked on the ARTIFACT rather than only in the producer's unit
    tests because that is where it was actually observed. The merge defect
    that created it was fixed in ``crucible-research#658``
    (alpha-engine-config-I7645) on 2026-08-18 20:26 UTC, and
    ``research/cuts_leaderboard/2026-08-19.json`` — written 2026-08-19 05:58
    UTC by scanner Lambda version 339, published 2026-08-18 00:04 UTC, i.e.
    20 hours BEFORE that merge — still carried ``attractiveness_top_20`` and
    ``scanner_gate_baseline_60`` twice in its 21-session block and at its top
    level. Every unit test passed on the merged code the whole time. A guard
    that lives only in the test suite cannot see a board produced by a
    deployment that predates it; one that reads the emitted artifact can
    (alpha-engine-config-I8026 deliverable 3).

    Returns descriptors like ``"21d:attractiveness_top_20x2"`` — empty when
    the board is clean. Pure: no I/O, no raising, so both the producer (which
    turns a hit into an ``unmeasurable`` verdict) and the consumer (which
    turns it into a written ``hold``) can call it before deciding what to do.
    """
    found: list[str] = []

    def _scan(surface: str, rows: Any) -> None:
        counts: dict[str, int] = {}
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            name = row.get("name")
            if name is None:
                continue
            counts[str(name)] = counts.get(str(name), 0) + 1
        found.extend(f"{surface}:{n}x{c}" for n, c in sorted(counts.items()) if c > 1)

    for block in board.get("horizons") or []:
        if isinstance(block, Mapping):
            _scan(f"{block.get('horizon_days')}d", block.get("specs"))
    _scan("top_level", board.get("specs"))
    return found


@dataclass(frozen=True)
class SlotMeasurementSpec:
    """The measurement contract for one champion/challenger SLOT.

    champion-challenger-policy.md §10: *"Every slot names, in its registry, the
    metric, horizon, benchmark, count-matching width, and hysteresis margins it
    uses. This document deliberately does not enumerate them — they are
    per-slot facts that CI can check against code."*

    This is that registry for the two leaderboard-scored slots. It is the SSoT
    for the horizons an arm is graded at and the evidence floor below which a
    row is not comparable; call sites read it rather than carrying literals.
    """

    slot_id: str
    # The slot's primary objective — the metric a promotion consumer ranks on.
    primary_metric: str
    # Every horizon scored, PRIMARY FIRST. The primary horizon is the
    # artifact's top-level block (§3 continuity).
    horizons_days: tuple[int, ...]
    benchmark_ticker: str | None
    # Count-matching width (§4): every arm in the slot is compared at this size.
    top_n: int
    # Evidence floor for a comparable row (§7.2 at arm granularity).
    min_dates_for_inference: int
    # When True, arms in this slot are scored at their OWN natural width and the
    # rows are NOT comparable to each other — only to the population
    # (alpha-engine-config-I7584). Legitimate exactly when the slot's arms are
    # not competing selection rules: the funnel's own stages differ in breadth
    # BY DEFINITION, so count-matching them would measure a truncation nobody
    # consumes. Default False — a slot must opt out of count-matching
    # explicitly, because silently mixed widths are the confound §4 exists to
    # prevent.
    per_arm_width: bool = False


LEADERBOARD_SLOTS: dict[str, SlotMeasurementSpec] = {
    # The scanner slot measures CANDIDATE GENERATION, where the definitionally
    # correct question is whether the cut beat the population it narrowed — not
    # whether it beat the market (alpha-engine-config-I7576). The SPY series is
    # still emitted on every row and its definition is unchanged; only which
    # metric a promotion consumer ranks on moves.
    "scanner": SlotMeasurementSpec(
        slot_id="scanner",
        primary_metric="topn_alpha_vs_population",
        horizons_days=LONG_HORIZONS_DAYS,
        benchmark_ticker="SPY",
        top_n=50,
        min_dates_for_inference=MIN_DATES_FOR_INFERENCE,
    ),
    # The producer slot deliberately keeps the SPY primary for now: it mixes
    # selection with sizing, so "beat the population you selected from" is not
    # the whole of its objective. Moving it is a separate argument and a
    # separate change, not a side effect of this one.
    # The funnel's own stages plus its downstream consumer's window. These are
    # NOT competing candidate-generation rules — attractiveness_top_20 is the
    # HEAD of attractiveness_top_60, so asking which "wins" against the other is
    # incoherent. Each is compared to the population it narrowed, which is what
    # makes differing widths legitimate here and only here.
    "cuts": SlotMeasurementSpec(
        slot_id="cuts",
        primary_metric="topn_alpha_vs_population",
        horizons_days=LONG_HORIZONS_DAYS,
        benchmark_ticker="SPY",
        top_n=0,  # unused; per_arm_width governs
        min_dates_for_inference=MIN_DATES_FOR_INFERENCE,
        per_arm_width=True,
    ),
    "producer": SlotMeasurementSpec(
        slot_id="producer",
        primary_metric="topn_alpha_vs_benchmark",
        horizons_days=LONG_HORIZONS_DAYS,
        benchmark_ticker="SPY",
        top_n=50,
        min_dates_for_inference=MIN_DATES_FOR_INFERENCE,
    ),
}


def slot_spec(slot_id: str) -> SlotMeasurementSpec:
    """The registered measurement contract for ``slot_id``. Raises on an
    unregistered slot — an arm scored under a slot with no registry row is the
    ``thinktank_coverage`` defect repeating (§10), and a silent default here
    would hide exactly that."""
    try:
        return LEADERBOARD_SLOTS[slot_id]
    except KeyError:
        raise KeyError(
            f"no slot registry row for {slot_id!r} — every scored slot must "
            f"name its metric, horizons, benchmark, count-matching width and "
            f"evidence floor in LEADERBOARD_SLOTS "
            f"(champion-challenger-policy.md §10). Known: "
            f"{sorted(LEADERBOARD_SLOTS)}"
        ) from None


def confidence_for(n_dates_scored: int | None, min_dates_for_inference: int) -> str:
    """How much evidence stands behind a spec row (alpha-engine-config-I7542).

    - ``insufficient`` — nothing scored. There is no result here at all.
    - ``thin`` — scored, but on fewer than ``min_dates_for_inference`` dates.
      The mean exists and is honest; it is NOT comparable against another arm's
      mean, and ``se``/``t_stat`` are null because they cannot be computed.
    - ``ok`` — enough date clusters for the clustered statistic to mean
      something.

    Why this is not covered by the guards that already exist: the producers
    handle the WHOLE-LEADERBOARD cases (``n_dates == 0`` structurally
    impossible → ``unmeasurable`` + alert; ``n_dates == 0`` overdue → escalate;
    ``n_dates == 0`` immature → silent and self-resolving, correctly). None of
    them says anything at ARM granularity. Live artifact
    ``research/producer_leaderboard/2026-08-14.json`` carried
    ``thinktank_coverage`` with ``n_dates_scored: 1``, ``se: null``,
    ``t_stat: null``, a mean of -0.107 — rendered in exactly the shape of the
    champion's real two-date row. Read by a human or an agent, that says "the
    Think Tank arm is losing badly". It says nothing of the kind.

    The row is NEVER suppressed on the strength of this status: §3 requires the
    miss to stay visible, and a hidden thin row is indistinguishable from an
    arm that was never scored. The defect is the rendering, not the row.
    """
    n = int(n_dates_scored or 0)
    if n <= 0:
        return CONFIDENCE_INSUFFICIENT
    if n < min_dates_for_inference:
        return CONFIDENCE_THIN
    return CONFIDENCE_OK


def measurability_for(
    spec: SpecHistory,
    dates_scored: Sequence[str],
    cohort_dates: Sequence[str],
) -> tuple[str, str | None]:
    """``(measurability, reason)`` for one arm (alpha-engine-config-I9307).

    Three ways an arm is UNMEASURABLE rather than merely thin:

    1. **The loader already knows.** ``spec.unmeasurable_reason`` is set — the
       arm's declared score source cannot carry its output, or its shadow
       prefix held nothing across the whole window. Most specific, so it wins.
    2. **It scored nothing at all** while a cohort exists. An arm with a cohort
       to score against and zero scored dates has not "not enough evidence yet";
       it produced no comparable output. ``insufficient`` alone said only the
       former.
    3. **It is structurally behind the cohort.** It scored no date among the
       most recent ``COHORT_LAG_UNMEASURABLE_DATES`` on which the cohort scored.
       This is the rung that would have caught the seven-week defect at week
       three: the champion's last scored date was 2026-07-17 while every other
       arm scored 08-14, 08-21 and 08-28, and its row still said ``thin``.

    Everything else is ``measured`` — including a genuinely thin arm, which is a
    legitimate result that resolves by waiting. Keeping those two apart is the
    whole point: one is a state, the other is a defect.
    """
    if spec.unmeasurable_reason:
        return MEASURABILITY_UNMEASURABLE, spec.unmeasurable_reason

    cohort = sorted(cohort_dates)
    if not cohort:
        # No cohort at all is a LEADERBOARD-level condition, already handled by
        # the producers' `unmeasurable` status + alert. Not an arm's fault.
        return MEASURABILITY_MEASURED, None

    scored = sorted(dates_scored)
    if not scored:
        return (
            MEASURABILITY_UNMEASURABLE,
            f"scored 0 of {len(cohort)} cohort date(s) — the arm produced no "
            "comparable output on any date the cohort covers, which is an "
            "absence, not thin evidence",
        )

    recent = cohort[-COHORT_LAG_UNMEASURABLE_DATES:]
    if not (set(scored) & set(recent)):
        return (
            MEASURABILITY_UNMEASURABLE,
            f"last scored {scored[-1]}, but scored none of the {len(recent)} "
            f"most recent cohort date(s) ({', '.join(recent)}) — the arm has "
            "stopped producing comparable output; its remaining dates are "
            "residue, not a thin-but-live record",
        )

    return MEASURABILITY_MEASURED, None


# ──────────────────────────────────────────────────────────────────────────
# Pure statistics — Spearman rank-IC + date-clustered significance.
# No scipy/numpy dependency: the leaderboard is a best-effort observe artifact
# that must import cleanly in the Lambda task layout with zero heavy wheels.
# ──────────────────────────────────────────────────────────────────────────

def _rankdata(values: Sequence[float]) -> list[float]:
    """Average-rank transform (ties share the mean of their rank span), matching
    ``scipy.stats.rankdata(method="average")`` — the standard Spearman input."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or None if undefined (n<2 or a zero-variance side)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx, my = fmean(xs), fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


def spearman_ic(signal: Sequence[float], realized: Sequence[float]) -> float | None:
    """Spearman rank correlation between a ranking ``signal`` and ``realized``
    forward returns (the cross-sectional rank-IC for ONE date). None when it is
    undefined (fewer than 2 paired names, or no rank variance on a side)."""
    if len(signal) != len(realized) or len(signal) < 2:
        return None
    return _pearson(_rankdata(signal), _rankdata(realized))


SE_IID = "iid"
SE_NEWEY_WEST = "newey-west"
SE_UNAVAILABLE = "unavailable"
"""How a metric block's standard error was produced.

Recorded ON the metric (``se_method``) rather than left to be inferred, because
an SE is only interpretable against the dependence assumption behind it and no
consumer can recover that from the number. The vocabulary is deliberately the
one ``crucible-evaluator``'s ``MetricRecord.ci_method`` already uses — two
vocabularies for one claim is how two repos drift into disagreeing about what a
t-stat means (alpha-engine-config-I8263).
"""


def overlap_lags_for(horizon_days: int, cohort_spacing_days: int | None) -> int:
    """How many neighbouring observations a forward window overlaps.

    A metric measured as "the return over the ``horizon_days`` sessions after
    each cohort date" produces observations whose windows OVERLAP whenever the
    cohort dates are spaced closer together than the horizon. With weekly cohort
    dates (~5 sessions) and a 21-session horizon, each window shares roughly 76%
    of its span with each of its four neighbours.

    That is not a small correction. ``sd / sqrt(n)`` assumes independence; under
    this much overlap the effective independent count is nearer ``n / (lags+1)``
    and the iid SE is understated by roughly ``sqrt(lags+1)``. Returning the
    Bartlett truncation lag here lets the caller ask for a HAC SE instead of
    silently publishing a t-stat that assumes something false.

    Returns 0 when the observations genuinely do not overlap — a weekly
    holding-period return measured on weekly cohort dates — in which case the
    iid SE is correct as written and no correction is applied.
    """
    if not cohort_spacing_days or cohort_spacing_days <= 0 or horizon_days <= 0:
        return 0
    return max(0, math.ceil(horizon_days / cohort_spacing_days) - 1)


def date_clustered_stats(
    per_date: Sequence[float], *, overlap_lags: int | None = None,
) -> dict | None:
    """Significance for a per-date metric series. Returns mean, standard error,
    t-stat, n_dates and the ``se_method`` that produced them — or None if empty.

    ``overlap_lags`` declares how many neighbouring observations each one
    overlaps (see :func:`overlap_lags_for`). ``None`` or ``0`` means the
    observations are independent and the clustered ``sd / sqrt(n)`` is correct —
    the original weeks-as-N contract, unchanged. A positive value switches to a
    Newey-West (Bartlett kernel) HAC standard error via the fleet's shared
    ``nousergon_lib.quant.stats.intervals.newey_west_se``.

    **A requested HAC SE is never silently downgraded to iid.** If the shared
    primitive cannot be imported or reports insufficient data, the block is
    emitted with ``se`` and ``t_stat`` NULL and ``se_method`` = ``unavailable``.
    Publishing the iid number instead is the exact defect this parameter exists
    to remove, and it would be indistinguishable from a correct one
    (alpha-engine-config-I8263).

    A 2-sided p-value is NOT returned (no scipy in the Lambda layout): the t-stat
    is the load-bearing surface the operator + the cutover gate read, and at the
    weeks-as-N counts here (n≈4–12) a normal-approx p would mislead. Significance
    is the t-stat vs the gate's threshold, exactly as the #1186 reconciliation
    reported (lift + clustered stat)."""
    vals = [float(v) for v in per_date]
    n = len(vals)
    if n == 0:
        return None
    mean = fmean(vals)
    if n == 1:
        return {"mean": round(mean, 6), "se": None, "t_stat": None, "n_dates": 1,
                "se_method": SE_UNAVAILABLE}

    lags = int(overlap_lags or 0)
    if lags > 0:
        se = _hac_se(vals, lags)
        if se is None:
            return {"mean": round(mean, 6), "se": None, "t_stat": None,
                    "n_dates": n, "se_method": SE_UNAVAILABLE,
                    "overlap_lags": lags}
        t_stat = (mean / se) if se > 0.0 else None
        return {
            "mean": round(mean, 6),
            "se": round(se, 6),
            "t_stat": (round(t_stat, 4) if t_stat is not None else None),
            "n_dates": n,
            "se_method": SE_NEWEY_WEST,
            "overlap_lags": lags,
        }

    var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample variance
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t_stat = (mean / se) if se > 0.0 else None
    return {
        "mean": round(mean, 6),
        "se": round(se, 6),
        "t_stat": (round(t_stat, 4) if t_stat is not None else None),
        "n_dates": n,
        "se_method": SE_IID,
    }


def _hac_se(vals: Sequence[float], lags: int) -> float | None:
    """Newey-West SE of the mean, or None when it cannot be computed.

    Imported lazily and inside the branch that needs it: this module is
    stdlib-only on the path every other metric takes, and the HAC primitive
    pulls numpy. None on ANY failure — the caller renders ``unavailable``
    rather than an iid substitute.
    """
    try:
        from nousergon_lib.quant.stats.intervals import newey_west_se
    except Exception as exc:  # noqa: BLE001 — recorded as `unavailable`, never as iid
        logger.warning(
            "[leaderboard] HAC standard error requested but "
            "nousergon_lib.quant.stats.intervals is unavailable (%s) — emitting "
            "se/t_stat NULL rather than an iid SE that assumes independence "
            "these overlapping observations do not have (I8263)", exc,
        )
        return None
    try:
        res = newey_west_se(list(vals), max_lags=lags)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[leaderboard] newey_west_se raised (%s) — se NULL", exc)
        return None
    if not isinstance(res, dict) or res.get("status") == "insufficient_data":
        return None
    se = res.get("se") if "se" in res else res.get("standard_error")
    try:
        se = float(se)
    except (TypeError, ValueError):
        return None
    return se if math.isfinite(se) and se >= 0.0 else None


# ──────────────────────────────────────────────────────────────────────────
# Spec abstraction — scanner specs and producer specs both reduce to, per date:
#   ranked: ordered ticker list (best→worst) carrying the ranking signal
#   scores: optional {ticker: float} for the cross-sectional rank-IC; when None
#           the rank order itself is the signal (descending = best), which is all
#           a count-matched top-N scanner spec exposes.
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpecDay:
    """One spec's picks for one date.

    ``rank_ordered`` declares whether ``ranked`` IS a ranking. It defaults to
    True — every producer that reads an ordered candidate list is unaffected —
    and exists because the membership artifact stores each cut's tickers
    ALPHABETICALLY (``universe_membership._top_n`` returns ``sorted(...)``: set
    semantics, with rank order carried separately in the ``ranks`` table). A
    day built from such a list has no order to correlate, and
    :func:`_rank_ic_metric` must then emit a MISSING rank-IC rather than a
    Spearman correlation against the alphabet, which is noise wearing the
    shape of a result (alpha-engine-config-I7631). Explicit ``scores`` override
    it: a scored day carries its own ranking regardless of list order.
    """

    ranked: list[str]
    scores: dict[str, float] | None = None
    rank_ordered: bool = True


@dataclass(frozen=True)
class SpecHistory:
    """A spec's picks across the cohort, keyed by date, plus its identity.

    ``held_dates`` maps each SCORED date to every calendar date that carried
    the same decision — ``{scored_date: [scored_date, ...carried dates]}``. A
    loader that writes one artifact per run while the DECISION is re-derived on
    a slower cadence (``universe_membership/{date}` is written every run; the
    cut is re-formed on ``cut_refresh_cadence``) must collapse the carried
    dates into the one they carry, or every horizon's ``n_dates_scored`` counts
    a held decision once per calendar prefix and overstates the evidence by
    roughly the hold length (alpha-engine-config-I8269).

    Default empty: a loader whose observations are one-per-decision by
    construction (every producer/scanner spec) leaves it alone and the
    consumer treats a missing entry as ``[date]`` — one calendar date, one
    decision. It is deliberately not a bare count: the dates themselves are
    what make "the cut was held for four sessions" legible on the artifact
    rather than inferred from the cadence config.
    """

    name: str
    kind: str  # "champion" | "challenger" | "retired"
    by_date: dict[str, SpecDay] = field(default_factory=dict)
    held_dates: dict[str, list[str]] = field(default_factory=dict)
    # Promotion eligibility, carried from the producer register
    # (``producers/registry.py``) so it lands on the artifact rather than
    # being re-derived by each consumer (alpha-engine-config-I9277). The
    # loader sets these; scoring only passes them through. Defaults keep every
    # non-producer board (scanner, cuts) unchanged.
    promotion_eligible: bool = True
    ineligible_reason: str | None = None
    # Set by the LOADER when it already knows the arm cannot produce a
    # comparable output — a declared score source that cannot carry the arm's
    # picks, or a shadow prefix with nothing in it across the whole window
    # (alpha-engine-config-I9307). ``None`` means "the loader saw no structural
    # obstacle"; scoring may still derive one from the cohort (see
    # :func:`measurability_for`). Non-None ALWAYS wins: a reason the loader can
    # name is more specific than one inferred from a date count.
    unmeasurable_reason: str | None = None


def _signal_for_ic(day: SpecDay) -> dict[str, float]:
    """Per-ticker ranking signal for the rank-IC. Uses explicit ``scores`` when
    present; else the descending rank position (top pick = highest signal) so a
    count-matched top-N scanner still yields a monotone within-pick signal."""
    if day.scores:
        return dict(day.scores)
    n = len(day.ranked)
    return {t: float(n - i) for i, t in enumerate(day.ranked)}


def _rank_ic_metric(
    spec: SpecHistory,
    realized: Mapping[str, Mapping[str, float]],
    *,
    overlap_lags: int | None = None,
) -> dict | None:
    """Date-clustered cross-sectional rank-IC: per date, Spearman(signal,
    realized) over the names with a realized return; cluster across dates."""
    per_date: list[float] = []
    for date_str, day in spec.by_date.items():
        ret = realized.get(date_str)
        if not ret:
            continue
        if not day.rank_ordered and not day.scores:
            # No ranking exists for this day — see SpecDay.rank_ordered. The
            # other metrics are unaffected (they read the picked SET), so the
            # row stays scored with rank_ic absent rather than being dropped.
            continue
        sig = _signal_for_ic(day)
        paired = [(s, ret[t]) for t, s in sig.items() if t in ret]
        if len(paired) < 2:
            continue
        ic = spearman_ic([p[0] for p in paired], [p[1] for p in paired])
        if ic is not None:
            per_date.append(ic)
    return date_clustered_stats(per_date, overlap_lags=overlap_lags)


def _top_n_return_by_date(
    day: SpecDay, ret: Mapping[str, float], top_n: int,
) -> float | None:
    """Equal-weight mean realized return of a spec's top-N picks (only names with
    a realized return count). None if no top-N name has a realized return."""
    picks = day.ranked[:top_n]
    rets = [ret[t] for t in picks if t in ret]
    return fmean(rets) if rets else None


def _topn_alpha_metric(
    spec: SpecHistory,
    champion: SpecHistory,
    realized: Mapping[str, Mapping[str, float]],
    top_n: int,
    *,
    overlap_lags: int | None = None,
) -> dict | None:
    """Date-clustered long-only top-N alpha = mean(spec top-N) − mean(champion
    top-N), per date, clustered across dates. A date contributes only when BOTH
    sides have a realized top-N return (a clean paired difference)."""
    per_date: list[float] = []
    for date_str, day in spec.by_date.items():
        ret = realized.get(date_str)
        champ_day = champion.by_date.get(date_str)
        if not ret or champ_day is None:
            continue
        spec_r = _top_n_return_by_date(day, ret, top_n)
        champ_r = _top_n_return_by_date(champ_day, ret, top_n)
        if spec_r is None or champ_r is None:
            continue
        per_date.append(spec_r - champ_r)
    return date_clustered_stats(per_date, overlap_lags=overlap_lags)


def _topn_alpha_vs_benchmark_metric(
    spec: SpecHistory,
    realized: Mapping[str, Mapping[str, float]],
    top_n: int,
    benchmark_ticker: str,
    *,
    overlap_lags: int | None = None,
) -> dict | None:
    """Date-clustered long-only top-N alpha = mean(spec top-N) − the
    ``benchmark_ticker``'s own realized return, per date, clustered across
    dates (alpha-engine-config-I2998: a champion-free, self-contained lift
    metric — comparable across specs with no live comparator dependency).
    ``realized`` already carries every ticker present in the same
    ``staging/daily_closes/{date}.parquet`` join ``_resolve_realized_returns``
    uses for every other ticker (SPY included — verified live 2026-07-20), so
    no separate benchmark fetch is needed here. A date contributes only when
    BOTH the spec's top-N return AND the benchmark's realized return are
    available for that date."""
    per_date: list[float] = []
    for date_str, day in spec.by_date.items():
        ret = realized.get(date_str)
        if not ret:
            continue
        bench_r = ret.get(benchmark_ticker)
        if bench_r is None:
            continue
        spec_r = _top_n_return_by_date(day, ret, top_n)
        if spec_r is None:
            continue
        per_date.append(spec_r - bench_r)
    return date_clustered_stats(per_date, overlap_lags=overlap_lags)


def population_return_from_panel(
    ret: Mapping[str, float], benchmark_ticker: str | None,
) -> float | None:
    """Equal-weight mean realized return over ``ret``, excluding the benchmark.

    **Only valid when ``ret`` spans the whole scored population.** Callers that
    narrow their closes read to the arms' own picks — which every leaderboard
    producer does, since an arm's other metrics only ever reference its picks —
    MUST NOT pass that narrowed map here. See
    :func:`_topn_alpha_vs_population_metric` for why that is a silent wrong
    answer rather than a missing one; this helper exists so a producer holding
    a genuine full-universe panel can build the input in one obvious place.

    The benchmark is excluded: it is an index proxy, not a name any arm could
    have picked, and the population must mean "the names I chose from".
    """
    rets = [r for t, r in ret.items() if t != benchmark_ticker]
    return fmean(rets) if rets else None


def _topn_alpha_vs_population_metric(
    spec: SpecHistory,
    realized: Mapping[str, Mapping[str, float]],
    top_n: int,
    population_returns: Mapping[str, float] | None,
    *,
    overlap_lags: int | None = None,
) -> dict | None:
    """Date-clustered long-only top-N excess over the scored population
    (alpha-engine-config-I7576): mean(spec top-N) − mean(all scored names),
    per date, clustered across dates.

    **This is the question a SELECTION stage is actually answering.** An arm's
    job at this stage is not to beat the market — it is to beat the universe it
    narrowed. Those give different verdicts, and in the current regime they give
    opposite ones: measured 2026-08-17 over 903 tracked names, SPY returned
    +0.73% over 21 sessions against the equal-weight population's +2.13%, and
    +1.95% vs +4.59% over 42. An arm returning +1.5% over 21 sessions therefore
    books ``topn_alpha_vs_benchmark`` of +0.77% (a win) while being −0.63%
    against the names it selected from (a loss). The gap widens with horizon,
    so the 126- and 252-session blocks are the more distorted ones, not less.

    ``topn_alpha_vs_benchmark`` is retained unchanged alongside it — "did this
    beat the market" is a real and different question, and closer to the
    executor's own objective. This adds a metric; it does not redefine one.

    ``population_returns`` is ``{date: equal-weight return of the whole scored
    population}`` and MUST be supplied by the caller. Returns ``None`` when it
    is absent — deliberately a missing metric rather than one derived from
    ``realized``.

    **Why it cannot be derived from ``realized``.** Every leaderboard producer
    narrows its closes read to ``_picked_symbols()`` — the union of the arms'
    own picks plus the benchmark — because an arm's other two metrics never
    reference anything else, and pulling ~900 tickers for three horizons is the
    Lambda's dominant cost. Taking the mean of that narrowed map would silently
    compute "excess over the other arms' picks", which is a plausible number,
    close to zero, and not the quantity this metric names. A metric that is
    absent is recoverable; one that is quietly the wrong baseline is the defect
    this whole issue was filed about, reintroduced one level down.

    A date contributes only when the spec's top-N return AND that date's
    population return both exist.
    """
    if not population_returns:
        return None
    per_date: list[float] = []
    for date_str, day in spec.by_date.items():
        ret = realized.get(date_str)
        pop_r = population_returns.get(date_str)
        if not ret or pop_r is None:
            continue
        spec_r = _top_n_return_by_date(day, ret, top_n)
        if spec_r is None:
            continue
        per_date.append(spec_r - pop_r)
    return date_clustered_stats(per_date, overlap_lags=overlap_lags)


def paired_alpha_vs_champion(
    spec: SpecHistory,
    champion: SpecHistory,
    realized: Mapping[str, Mapping[str, float]],
    top_n: int,
    *,
    overlap_lags: int | None = None,
) -> dict | None:
    """The PAIRED per-date difference between ``spec`` and the champion.

    Public because the cuts board cannot get this from
    :func:`score_leaderboard`: it scores each arm in its own single-arm pass
    (arms have different widths, and one call takes one ``top_n``), so every
    challenger pass sees ``champion=None`` and the paired metric is skipped.
    The result was ``topn_alpha_vs_champion: null`` on every arm on every
    board — the most powerful statistic the design makes available, never
    computed (alpha-engine-config-I8263).

    **Why it is worth more than any unpaired metric here.** ``spec`` and the
    champion pick from the SAME universe on the SAME date under the SAME
    market. Differencing them date by date cancels the common market factor, so
    the variance of the difference is a small fraction of the variance of
    either leg. At the observation counts this slot will ever have — a weekly
    cut is ~52 a year — that is the difference between needing tens of
    observations and needing hundreds. Comparing two independently-estimated
    means is not a substitute: it is comparable across count-matched arms, but
    it throws away the pairing the count-matched same-date design was built to
    create (champion-challenger-policy.md §4).

    Callers MUST hold the two arms at the same width. A paired difference
    between a 60-wide arm and a 20-wide champion measures breadth, not the
    selection rule, and the count-match guard exists upstream precisely so this
    function never has to guess.
    """
    return _topn_alpha_metric(
        spec, champion, realized, top_n, overlap_lags=overlap_lags,
    )


#: How many of the cohort's most recent observations an arm must have reached
#: to constrain the shared comparison window. An arm quieter than this is stale
#: rather than merely sparse: it is not contributing to the CURRENT question.
STALE_ARM_EXCLUSION_DATES = 5


def cohort_intersection(
    champion: SpecHistory | None,
    challengers: Sequence[SpecHistory],
    realized: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """The dates on which EVERY promotion-eligible arm produced output AND a
    realized return exists — the only cohort on which a promotion comparison
    is fair (champion-challenger-policy.md §4: "Arms are scored over the
    intersection of dates where all arms produced output, and the intersection
    is reported alongside the metric").

    The defect this closes (alpha-engine-config-I9279): on the live
    2026-08-28 board the four scored arms carried n_dates_scored of 2, 6, 6
    and 4 over a NINE-date cohort — every arm scored on its OWN available
    dates, so the numbers being ranked against each other described four
    different windows of the market. "Comparing an arm's good month to
    another's bad quarter is not a comparison."

    Only promotion-ELIGIBLE arms constrain the intersection. A retired arm is
    historical evidence that can never win (§3/§6), so letting its sparse
    history shrink the window every live arm is judged on would be the
    measurement paying rent for a row that cannot use it. An arm that produced
    NOTHING at all is likewise excluded from the constraint and reported
    separately as an absence — otherwise one dead arm empties the intersection
    and silently converts every comparison into a no-contest, which is the
    dead-arm failure mode wearing a statistical costume.
    """
    arms = [a for a in ([champion] if champion is not None else []) + list(challengers)
            if a.promotion_eligible]
    contributing = {
        a.name: {d for d in a.by_date if realized.get(d)} for a in arms
    }
    live = {n: c for n, c in contributing.items() if c}
    if not live:
        return []

    # An arm that STOPPED producing must not freeze the comparison window for
    # the arms that did not. Measured 2026-08-29: `thinktank_coverage` last
    # wrote a shadow on 2026-08-14 (it had deadlocked — see
    # alpha-engine-config-I9282), and its 4 remaining dates cut the shared
    # cohort from 18 dates to 4 for every other arm. Enforcing §4's
    # intersection naively would let one dead arm collapse every comparison to
    # a no-contest — the dead-arm failure mode wearing a statistical costume,
    # and precisely the "over-strict gate that never promotes" §5 rejects.
    #
    # So: an arm whose most recent contribution predates the cohort's most
    # recent date by more than STALE_ARM_EXCLUSION_DATES observations is
    # excluded from CONSTRAINING the intersection. It is still scored, still
    # on the artifact, and its staleness is recorded — §3's "a cycle where an
    # arm produces no output is recorded as a MISS, not omitted". What it
    # loses is only the power to shrink everyone else's window.
    ordered = sorted({d for c in live.values() for d in c})
    if len(ordered) > STALE_ARM_EXCLUSION_DATES:
        recent_floor = ordered[-STALE_ARM_EXCLUSION_DATES]
        current = {n: c for n, c in live.items() if max(c) >= recent_floor}
    else:
        current = live
    if not current:
        return []
    return sorted(set.intersection(*current.values()))

# ── TWO windows, deliberately, and neither may wear the other's name ─────────
#
# This module now carries two cohort-window functions, because the two callers
# are asking different questions and merging them would answer one with the
# other — the failure this whole arc has been about.
#
#   cohort_intersection(champion, challengers, realized)   [above, from #762]
#       The PROMOTION window. Filters to promotion-eligible arms and drops an
#       arm that has gone quiet (STALE_ARM_EXCLUSION_DATES) from CONSTRAINING
#       the window, so one dead arm cannot collapse every comparison into a
#       no-contest (alpha-engine-config-I9279/-I9282). Deliberately RELAXED,
#       and correct for deciding.
#
#   strict_cohort_intersection(arms, realized)             [below, from #760]
#       The REPORTED window. Every arm handed to it must have scored, with no
#       eligibility filter and no staleness relaxation. Deliberately STRICT,
#       and correct for reporting.
#
# The reported number takes the STRICT rule. Publishing the relaxed window as
# the block's stated intersection would claim more common ground than exists —
# a board asserting comparability it does not have, which is worse than the
# union it replaces because it looks like it was checked. A window may be
# relaxed to reach a DECISION; it may never be relaxed to make a NUMBER larger.
#
# WHY THIS MATTERS RIGHT NOW, MEASURED 2026-08-29: `scoring/spec_promotion.py`
# (merged, #761) reads `cohort_intersection_dates` and FALLS BACK TO `n_dates`
# — the UNION — when it is absent. Nothing on main writes that field, so the
# fallback is taken on every run today and the spec slot's evidence floor is
# currently reading the union as if it were the intersection: 7 where the truth
# is 0. `apply_cohort_intersection` below is what writes it.

def scored_dates_for(
    spec: SpecHistory, realized: Mapping[str, Mapping[str, float]],
) -> set[str]:
    """The dates this arm actually contributed at this horizon.

    Exactly the predicate behind the row's ``n_dates_scored``, lifted to one
    named function so the intersection below and the count on the row can never
    disagree — two independent spellings of "scored" is the shape that put a
    union count and a per-arm count on the same block reading as one number.
    """
    return {d for d in spec.by_date if realized.get(d)}


def strict_cohort_intersection(
    arms: Sequence[SpecHistory], realized: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """The dates on which EVERY arm in ``arms`` scored, sorted.

    champion-challenger-policy.md §4: *"arms are scored over the intersection
    of dates where all arms produced output, and the intersection is reported
    alongside the metric."* The boards reported the UNION and called it
    ``n_dates``, so a champion with a live cohort running back to July and a
    challenger registered last week rendered on one block as though they had
    been measured against each other.

    Empty ``arms`` yields ``[]`` — no arms is not "every date", and returning
    the union there would be the same conflation one level down.
    """
    if not arms:
        return []
    common: set[str] | None = None
    for arm in arms:
        dates = scored_dates_for(arm, realized)
        common = dates if common is None else (common & dates)
        if not common:
            return []
    return sorted(common or ())


def apply_cohort_intersection(
    block: dict,
    arms: Sequence[SpecHistory],
    realized: Mapping[str, Mapping[str, float]],
    *,
    champion: SpecHistory | None,
    top_n: int | Mapping[str, int],
    overlap_lags: int | None = None,
) -> list[str]:
    """Record the block's cohort intersection and narrow every CROSS-ARM figure
    to it, in place. Returns the intersection dates.

    Emitted on the block (alpha-engine-config-I9274):

    * ``cohort_intersection_dates`` — how many dates every arm scored,
    * ``cohort_intersection_first`` / ``_last`` — its span, ``None`` when empty,
    * ``cohort_union_dates`` — the existing ``n_dates``, restated under a name
      that says which of the two it is.

    ``n_dates`` is deliberately NOT renamed. ``crucible-dashboard``'s
    ``loaders/s3_loader.py`` and live ``gate:data`` predicates poll it, and a
    rename would break both to fix a rendering. The sibling field is added
    alongside; the union keeps its name and its meaning.

    ``topn_alpha_vs_champion`` is then recomputed over the intersection ONLY,
    and every row carries a ``comparison_status`` saying why it holds what it
    holds. Where the intersection is empty the figure is ``None`` and the
    status is ``no_common_cohort`` — a state distinct from
    ``confidence: insufficient``, which today carries both "this arm scored
    little" and "this arm cannot be compared" and lets a reader take the first
    for the second.

    The champion's OWN dates are never dropped: ``n_dates_scored``,
    ``realized_rank_ic``, ``topn_alpha_vs_benchmark`` and
    ``topn_alpha_vs_population`` all keep the arm's full honest history
    (champion-challenger-policy.md §3 — a promoted arm keeps its series). Only
    the cross-arm comparison narrows, because only the cross-arm comparison is
    the thing that needs common ground.

    ``top_n`` may be one width for the whole block, or a per-arm mapping (the
    cuts board scores each arm at its own width). A challenger whose width
    differs from the champion's keeps a null at ``width_mismatch``: a paired
    difference across widths measures breadth, not the selection rule.
    """
    rows = block.get("specs") or []
    inter = strict_cohort_intersection(arms, realized)
    block["cohort_intersection_dates"] = len(inter)
    block["cohort_intersection_first"] = inter[0] if inter else None
    block["cohort_intersection_last"] = inter[-1] if inter else None
    block["cohort_union_dates"] = block.get("n_dates")

    by_name = {a.name: a for a in arms}
    widths = top_n if isinstance(top_n, Mapping) else None

    def _width(name: str) -> int | None:
        if widths is None:
            return int(top_n)  # type: ignore[arg-type]
        w = widths.get(name)
        return int(w) if w else None

    champ_width = _width(champion.name) if champion is not None else None
    restricted = {d: realized[d] for d in inter if d in realized}

    for row in rows:
        name = row.get("name")
        if "error" in row:
            # The per-spec fail-soft row already says the scoring RAISED. It
            # must not also be given a comparison verdict it never earned.
            row["comparison_status"] = None
            continue
        if champion is not None and name == champion.name:
            row["topn_alpha_vs_champion"] = None
            row["comparison_status"] = COMPARISON_SELF
            continue
        if champion is None:
            row["topn_alpha_vs_champion"] = None
            row["comparison_status"] = COMPARISON_NO_CHAMPION
            continue
        arm = by_name.get(name)
        if arm is None:
            # A row with no arm behind it cannot be compared and must not be
            # left carrying a stale figure from a wider pass.
            row["topn_alpha_vs_champion"] = None
            row["comparison_status"] = COMPARISON_NO_COMMON_COHORT
            continue
        width = _width(name)
        if width is None or champ_width is None or width != champ_width:
            row["topn_alpha_vs_champion"] = None
            row["comparison_status"] = COMPARISON_WIDTH_MISMATCH
            continue
        if not inter:
            row["topn_alpha_vs_champion"] = None
            row["comparison_status"] = COMPARISON_NO_COMMON_COHORT
            continue
        paired = _topn_alpha_metric(
            arm, champion, restricted, width, overlap_lags=overlap_lags,
        )
        row["topn_alpha_vs_champion"] = paired
        n_paired = (paired or {}).get("n_dates") or 0
        if paired is None or n_paired != len(inter):
            # LOUD, never silent (§7.2). Every date in the intersection was
            # scored by both arms, so a paired difference should exist on each;
            # a shortfall means a date whose top-N picks had no realized return
            # on one side, and the figure is then NOT the block's stated
            # intersection. Say so on the row rather than publish a number
            # under a denominator that does not hold.
            logger.error(
                "[leaderboard] arm %r paired vs champion %r covered %s of the "
                "block's %s intersection dates at %sd — the figure is reported "
                "as partial_cohort rather than under the block's stated "
                "intersection (alpha-engine-config-I9274)",
                name, champion.name, n_paired, len(inter),
                block.get("horizon_days"),
            )
            row["comparison_status"] = COMPARISON_PARTIAL_COHORT
        else:
            row["comparison_status"] = COMPARISON_OK
    return inter

def score_leaderboard(
    champion: SpecHistory | None,
    challengers: Sequence[SpecHistory],
    realized: Mapping[str, Mapping[str, float]],
    *,
    top_n: int = 50,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    benchmark_ticker: str | None = "SPY",
    min_dates_for_inference: int = MIN_DATES_FOR_INFERENCE,
    population_returns: Mapping[str, float] | None = None,
    overlap_lags: int | None = None,
) -> dict:
    """Score the champion (if any) + every challenger on the cutover-gate
    objectives, joined to ``realized`` (``{date: {ticker: forward_return}}``).
    PURE — no I/O.

    ``champion`` is ``Optional`` (alpha-engine-config-I2998): no producer is
    currently registered ``kind=="champion"`` (config-I2993 retired
    ``agentic_sector_teams`` without a successor) — scoring must degrade to
    champion-free metrics rather than refuse to run. ``topn_alpha_vs_champion``
    is ``None`` for every spec whenever ``champion`` is ``None`` (nothing to
    compare against); ``realized_rank_ic`` and ``topn_alpha_vs_benchmark`` are
    unaffected, since neither needs a live comparator.

    Returns a leaderboard dict::

        {
          "champion": <name> | None,
          "horizon_days": 21,
          "top_n": 50,
          "benchmark_ticker": <str> | None,
          "n_dates": <#dates with ANY realized join>,
          "specs": [
            {"name", "kind",
             "realized_rank_ic": <clustered stats | None>,
             "topn_alpha_vs_champion": <clustered stats | None>,  # None for the champion, and whenever champion is None
             "topn_alpha_vs_benchmark": <clustered stats | None>,  # champion-free direct lift vs benchmark_ticker
             "topn_alpha_vs_population": <clustered stats | None>, # lift vs the scored population it narrowed (I7576)
             "n_dates_scored": <#dates this spec contributed>,
             "confidence": "ok" | "thin" | "insufficient"},   # alpha-engine-config-I7542
            ...
          ],
        }

    Per-spec fail-soft: a spec whose metric computation raises is logged at
    WARNING and emitted with null metrics — never sinks the others (no-silent-fails:
    the failure is recorded)."""
    dates_with_join = sorted(
        d for d in realized
        if (
            (champion is not None and d in champion.by_date)
            or any(d in c.by_date for c in challengers)
        )
        and realized.get(d)
    )

    # champion-challenger-policy.md §4 — the ONE cohort every promotion
    # comparison is made on, computed once and reported on the artifact.
    intersection = cohort_intersection(champion, challengers, realized)
    realized_intersection = {d: realized[d] for d in intersection if d in realized}

    spec_rows: list[dict] = []

    def _row(spec: SpecHistory, is_champion: bool) -> dict:
        try:
            rank_ic = _rank_ic_metric(spec, realized, overlap_lags=overlap_lags)
            alpha_vs_champion = None if (is_champion or champion is None) else _topn_alpha_metric(
                spec, champion, realized, top_n, overlap_lags=overlap_lags,
            )
            alpha_vs_benchmark = (
                _topn_alpha_vs_benchmark_metric(
                    spec, realized, top_n, benchmark_ticker, overlap_lags=overlap_lags,
                )
                if benchmark_ticker else None
            )
            # None unless the caller supplied genuine full-population returns —
            # never derived from the pick-narrowed `realized` map.
            alpha_vs_population = _topn_alpha_vs_population_metric(
                spec, realized, top_n, population_returns, overlap_lags=overlap_lags,
            )
            dates_scored = sorted(d for d in spec.by_date if realized.get(d))
            n_scored = len(dates_scored)
            # The SAME metric, restricted to the dates every promotion-eligible
            # arm shares (§4). This is the number a promotion gate must rank
            # on; ``topn_alpha_vs_benchmark`` above stays each arm's own-cohort
            # figure so the existing 21-day series is continuous (§3) and the
            # two are never conflated — they answer different questions and the
            # artifact names both.
            alpha_vs_benchmark_intersection = (
                _topn_alpha_vs_benchmark_metric(
                    spec, realized_intersection, top_n, benchmark_ticker,
                    overlap_lags=overlap_lags,
                )
                if (benchmark_ticker and realized_intersection) else None
            )
            n_intersection = len(
                [d for d in spec.by_date if realized_intersection.get(d)]
            )
            measurability, unmeasurable_reason = measurability_for(
                spec, dates_scored, dates_with_join,
            )
            return {
                "name": spec.name,
                "kind": spec.kind,
                "realized_rank_ic": rank_ic,
                "topn_alpha_vs_champion": alpha_vs_champion,
                "topn_alpha_vs_benchmark": alpha_vs_benchmark,
                "topn_alpha_vs_population": alpha_vs_population,
                "n_dates_scored": n_scored,
                # alpha-engine-config-I9277/I9279 — the cohort itself, not just
                # its size. Two arms reporting n_dates_scored: 6 over disjoint
                # weeks were indistinguishable on this artifact before.
                "dates_scored": dates_scored,
                "topn_alpha_vs_benchmark_intersection": alpha_vs_benchmark_intersection,
                "n_dates_in_intersection": n_intersection,
                # Projected from producers/registry.py so the promotion engine
                # resolves the arm set from THIS artifact and carries no
                # second, hand-maintained list (alpha-engine-config-I9277).
                "promotion_eligible": spec.promotion_eligible,
                "ineligible_reason": spec.ineligible_reason,
                # alpha-engine-config-I7542 — how much evidence stands behind
                # this row. Additive: every numeric field above is unchanged.
                "confidence": confidence_for(n_scored, min_dates_for_inference),
                # alpha-engine-config-I9307 — whether this arm CAN produce
                # comparable output at all. A separate axis from `confidence`
                # on purpose: `thin` is a state that resolves by waiting,
                # `unmeasurable` is a defect that does not.
                "measurability": measurability,
                "unmeasurable_reason": unmeasurable_reason,
            }
        except Exception as exc:  # noqa: BLE001 — observe artifact, per-spec isolation
            logger.warning(
                "[leaderboard] spec %s scoring failed (non-fatal, other specs "
                "unaffected): %s", spec.name, exc,
            )
            return {
                "name": spec.name,
                "kind": spec.kind,
                "realized_rank_ic": None,
                "topn_alpha_vs_champion": None,
                "topn_alpha_vs_benchmark": None,
                "topn_alpha_vs_population": None,
                "n_dates_scored": 0,
                "dates_scored": [],
                "topn_alpha_vs_benchmark_intersection": None,
                "n_dates_in_intersection": 0,
                "promotion_eligible": spec.promotion_eligible,
                "ineligible_reason": spec.ineligible_reason,
                "confidence": CONFIDENCE_INSUFFICIENT,
                # A spec whose scoring RAISED produced no comparable output for
                # a reason we can name — that is unmeasurable, never thin.
                "measurability": MEASURABILITY_UNMEASURABLE,
                "unmeasurable_reason": f"scoring raised: {exc}",
                "error": str(exc),
            }

    if champion is not None:
        spec_rows.append(_row(champion, is_champion=True))
    for ch in challengers:
        spec_rows.append(_row(ch, is_champion=False))

    out = {
        "champion": champion.name if champion is not None else None,
        "horizon_days": horizon_days,
        "top_n": top_n,
        "benchmark_ticker": benchmark_ticker,
        "n_dates": len(dates_with_join),
        # §4: the intersection is reported ALONGSIDE the metric, never left for
        # a reader to reconstruct from per-row date lists.
        "cohort_intersection": intersection,
        "n_dates_intersection": len(intersection),
        "specs": spec_rows,
        # alpha-engine-config-I9307 — the arms that CANNOT be measured, named
        # at board level so a consumer (and the producers' alert path) does not
        # have to walk `specs` to discover that the comparison is missing a
        # side. An empty list is the healthy state and is emitted every cycle:
        # an absent field would be unmeasured, not fine.
        "unmeasurable_arms": [
            {"name": r["name"], "kind": r["kind"], "reason": r.get("unmeasurable_reason")}
            for r in spec_rows
            if r.get("measurability") == MEASURABILITY_UNMEASURABLE
        ],
    }
    # The cohort INTERSECTION, and every cross-arm figure narrowed to it
    # (alpha-engine-config-I9274). Applied here rather than at each producer so
    # a direct caller of this function cannot get a board without it — the
    # scanner and producer boards score every arm in one pass and are covered
    # by this alone. The cuts board scores one arm per pass and RE-APPLIES it
    # over the merged arm set, where the whole block is finally in scope.
    apply_cohort_intersection(
        out,
        [a for a in (champion, *challengers) if a is not None],
        realized,
        champion=champion,
        top_n=top_n,
        overlap_lags=overlap_lags,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────
# Multi-horizon assembly (alpha-engine-config-I7540)
# ──────────────────────────────────────────────────────────────────────────

# Per-horizon block statuses. Distinct from the leaderboard-level status, and
# from the per-spec confidence, on purpose: three different questions.
HORIZON_OK = "ok"
HORIZON_IMMATURE = "immature"
HORIZON_UNMEASURABLE = "unmeasurable"


def score_multi_horizon(
    champion: SpecHistory | None,
    challengers: Sequence[SpecHistory],
    realized_by_horizon: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    top_n: int = 50,
    horizons_days: Sequence[int] = LONG_HORIZONS_DAYS,
    benchmark_ticker: str | None = "SPY",
    min_dates_for_inference: int = MIN_DATES_FOR_INFERENCE,
    horizon_notes: Mapping[int, tuple[str, str]] | None = None,
    population_by_horizon: Mapping[int, Mapping[str, float]] | None = None,
    cohort_spacing_days: int | None = None,
    cohort_spacing_by_horizon: Mapping[int, int | None] | None = None,
) -> dict:
    """Score every arm at EVERY horizon in ``horizons_days`` and assemble one
    leaderboard artifact. PURE — no I/O.

    ``realized_by_horizon`` is ``{horizon_days: {date: {ticker: forward_return}}}``
    — one realized-return map per horizon, all derived by the caller from ONE
    closes panel read.

    ``cohort_spacing_days`` is the fallback trading-day gap between cohort
    dates; ``cohort_spacing_by_horizon`` overrides it PER HORIZON and is what a
    caller should pass whenever different horizons score different subsets of
    the cohort (a long horizon scores only the matured, older dates). Each
    block records the spacing it used in ``cohort_spacing_days``, so no reader
    has to reconstruct why one block's ``overlap_lags`` differs from another's.

    ``horizon_notes`` optionally carries ``{horizon_days: (status, reason)}``
    for horizons the caller already knows could not be measured (e.g. the
    source cannot serve a 252-session lookforward at all). A horizon named
    there is emitted with that status and reason rather than ``ok``.

    CONTINUITY (champion-challenger-policy.md §3). ``horizons_days[0]`` is the
    PRIMARY horizon, and its block is spread across the TOP LEVEL of the
    returned dict exactly as ``score_leaderboard`` has always returned it —
    same keys, same values, same rounding. Every existing consumer
    (``crucible-backtester``'s ``champion_promotion._score_thinktank_coverage``
    reads ``specs[].topn_alpha_vs_benchmark.mean`` off the top level; the
    console leaderboard pane reads ``specs`` and ``n_dates``) therefore sees an
    unchanged 21-day series across this change. A scorer that silently
    re-based a promoted arm's history would break the same invariant a
    promotion is forbidden from breaking.

    The new ``horizons`` list is the CANONICAL surface: one block per horizon,
    each carrying its OWN ``n_dates`` and its own per-spec rows. The primary
    horizon appears in both places, by design — the duplication is the price of
    continuity and is cheap (an observe artifact of a few KB).

    Block shape::

        {"horizon_days": 252,
         "status": "ok" | "immature" | "unmeasurable",
         "reason": <str | None>,          # why, whenever status != "ok"
         "n_dates": <#dates with ANY realized join at this horizon>,
         "specs": [...]}                  # same row shape, incl. "confidence"

    A horizon with no matured cohort emits ``n_dates: 0`` and every spec row at
    ``confidence: "insufficient"`` — never a numeric result and never a zero.
    At rollout a 252-session horizon will be immature for every arm, because a
    252-session horizon needs ~252 trading days of shadow history before ANY
    date scores. That is honest immaturity by construction, and rendering it as
    such is part of this deliverable rather than a follow-up (§7.2).
    """
    if not horizons_days:
        raise ValueError("horizons_days must name at least the primary horizon")

    notes = dict(horizon_notes or {})
    blocks: list[dict] = []
    primary: dict | None = None

    for h in horizons_days:
        # Each horizon gets its OWN overlap correction, because overlap is a
        # property of (horizon, cohort spacing) and the horizons here span
        # 21..252 sessions against the same weekly cohort. Applying one lag
        # count across all three would under-correct the long horizons — which
        # are the MORE distorted ones, not less (alpha-engine-config-I8263).
        #
        # The SPACING may also differ per horizon, and on the live board it
        # does. A 21-session horizon scores only the cohort dates whose window
        # has MATURED — on 2026-08-21 that was 8 weekly dates spaced ~5
        # sessions — while the full enumerated cohort runs daily through
        # August, whose median gap is 1. Deriving the lag from the whole
        # calendar rather than from the dates a horizon actually used asks for
        # 20 lags on 8 observations: the HAC estimator then truncates to n-1
        # and returns a number built from almost no independent information
        # (measured 2026-08-24, alpha-engine-config-I8263 deliverable 4).
        # Overlap is a property of the observations that ENTERED the mean.
        lags = overlap_lags_for(
            h, (cohort_spacing_by_horizon or {}).get(h, cohort_spacing_days),
        )
        scored = score_leaderboard(
            champion,
            challengers,
            realized_by_horizon.get(h) or {},
            top_n=top_n,
            horizon_days=h,
            benchmark_ticker=benchmark_ticker,
            min_dates_for_inference=min_dates_for_inference,
            population_returns=(population_by_horizon or {}).get(h),
            overlap_lags=lags,
        )
        if primary is None:
            primary = scored
        status, reason = notes.get(h, (HORIZON_OK, None))
        blocks.append(
            {
                "horizon_days": h,
                "status": status,
                "reason": reason,
                "n_dates": scored["n_dates"],
                # The cohort intersection for THIS horizon (alpha-engine-config
                # -I9274). Per horizon, never per board: an arm can be inside
                # the 21d intersection and outside the 126d one, because a
                # longer horizon scores only the older, matured cohort dates.
                "cohort_union_dates": scored["cohort_union_dates"],
                "cohort_intersection_dates": scored["cohort_intersection_dates"],
                "cohort_intersection_first": scored["cohort_intersection_first"],
                "cohort_intersection_last": scored["cohort_intersection_last"],
                # Declared per block so a reader never has to infer whether this
                # horizon's observations were independent. `overlap_lags: 0`
                # means genuinely non-overlapping, not "not checked".
                "overlap_lags": lags,
                "observations_overlap": lags > 0,
                # The denominator behind `overlap_lags`, stated rather than
                # inferable: two blocks on one board can legitimately carry
                # different spacings (see the per-horizon comment above).
                "cohort_spacing_days": (
                    (cohort_spacing_by_horizon or {}).get(h, cohort_spacing_days)
                ),
                "specs": scored["specs"],
            }
        )

    assert primary is not None  # noqa: S101 — horizons_days is non-empty above
    out = dict(primary)
    # `dict(primary)` is SHALLOW, so `out["specs"]` would be the SAME list
    # object as `blocks[0]["specs"]`. A caller that appends to one surface then
    # appends to the other — which is exactly what a per-arm producer must do
    # (`build_cuts_leaderboard`) — writes twice into one list, and the primary
    # horizon's block silently double-counts every arm after the first.
    # Measured live on `research/cuts_leaderboard/2026-08-18.json`: 5 rows for
    # 3 arms (alpha-engine-config-I7631). The rows are shared dicts on purpose
    # — the two surfaces are the same rows — but the LISTS are not.
    out["specs"] = list(primary["specs"])
    out["horizons_days"] = list(horizons_days)
    out["min_dates_for_inference"] = min_dates_for_inference
    out["horizons"] = blocks
    return out
