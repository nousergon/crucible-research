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


def date_clustered_stats(per_date: Sequence[float]) -> dict | None:
    """Date-clustered significance for a per-date metric series (each date = one
    independent cluster; weeks-as-N). Returns mean, the clustered standard error
    ``sd / sqrt(n)``, a t-stat ``mean / SE``, and n_dates — or None if empty.

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
        return {"mean": round(mean, 6), "se": None, "t_stat": None, "n_dates": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample variance
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t_stat = (mean / se) if se > 0.0 else None
    return {
        "mean": round(mean, 6),
        "se": round(se, 6),
        "t_stat": (round(t_stat, 4) if t_stat is not None else None),
        "n_dates": n,
    }


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
    """A spec's picks across the cohort, keyed by date, plus its identity."""

    name: str
    kind: str  # "champion" | "challenger"
    by_date: dict[str, SpecDay] = field(default_factory=dict)


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
    return date_clustered_stats(per_date)


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
    return date_clustered_stats(per_date)


def _topn_alpha_vs_benchmark_metric(
    spec: SpecHistory,
    realized: Mapping[str, Mapping[str, float]],
    top_n: int,
    benchmark_ticker: str,
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
    return date_clustered_stats(per_date)


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
    return date_clustered_stats(per_date)


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

    spec_rows: list[dict] = []

    def _row(spec: SpecHistory, is_champion: bool) -> dict:
        try:
            rank_ic = _rank_ic_metric(spec, realized)
            alpha_vs_champion = None if (is_champion or champion is None) else _topn_alpha_metric(
                spec, champion, realized, top_n,
            )
            alpha_vs_benchmark = (
                _topn_alpha_vs_benchmark_metric(spec, realized, top_n, benchmark_ticker)
                if benchmark_ticker else None
            )
            # None unless the caller supplied genuine full-population returns —
            # never derived from the pick-narrowed `realized` map.
            alpha_vs_population = _topn_alpha_vs_population_metric(
                spec, realized, top_n, population_returns,
            )
            n_scored = sum(
                1 for d in spec.by_date if realized.get(d)
            )
            return {
                "name": spec.name,
                "kind": spec.kind,
                "realized_rank_ic": rank_ic,
                "topn_alpha_vs_champion": alpha_vs_champion,
                "topn_alpha_vs_benchmark": alpha_vs_benchmark,
                "topn_alpha_vs_population": alpha_vs_population,
                "n_dates_scored": n_scored,
                # alpha-engine-config-I7542 — how much evidence stands behind
                # this row. Additive: every numeric field above is unchanged.
                "confidence": confidence_for(n_scored, min_dates_for_inference),
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
                "confidence": CONFIDENCE_INSUFFICIENT,
                "error": str(exc),
            }

    if champion is not None:
        spec_rows.append(_row(champion, is_champion=True))
    for ch in challengers:
        spec_rows.append(_row(ch, is_champion=False))

    return {
        "champion": champion.name if champion is not None else None,
        "horizon_days": horizon_days,
        "top_n": top_n,
        "benchmark_ticker": benchmark_ticker,
        "n_dates": len(dates_with_join),
        "specs": spec_rows,
    }


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
) -> dict:
    """Score every arm at EVERY horizon in ``horizons_days`` and assemble one
    leaderboard artifact. PURE — no I/O.

    ``realized_by_horizon`` is ``{horizon_days: {date: {ticker: forward_return}}}``
    — one realized-return map per horizon, all derived by the caller from ONE
    closes panel read.

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
        scored = score_leaderboard(
            champion,
            challengers,
            realized_by_horizon.get(h) or {},
            top_n=top_n,
            horizon_days=h,
            benchmark_ticker=benchmark_ticker,
            min_dates_for_inference=min_dates_for_inference,
            population_returns=(population_by_horizon or {}).get(h),
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
