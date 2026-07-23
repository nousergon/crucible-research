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

logger = logging.getLogger(__name__)

# Default forward-return horizon: 21 trading days (~1 month), the horizon both
# cutover gates name ("realized 21d outcomes").
DEFAULT_HORIZON_DAYS = 21


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
    """One spec's picks for one date."""

    ranked: list[str]
    scores: dict[str, float] | None = None


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


def score_leaderboard(
    champion: SpecHistory | None,
    challengers: Sequence[SpecHistory],
    realized: Mapping[str, Mapping[str, float]],
    *,
    top_n: int = 50,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    benchmark_ticker: str | None = "SPY",
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
             "n_dates_scored": <#dates this spec contributed>},
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
            n_scored = sum(
                1 for d in spec.by_date if realized.get(d)
            )
            return {
                "name": spec.name,
                "kind": spec.kind,
                "realized_rank_ic": rank_ic,
                "topn_alpha_vs_champion": alpha_vs_champion,
                "topn_alpha_vs_benchmark": alpha_vs_benchmark,
                "n_dates_scored": n_scored,
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
                "n_dates_scored": 0,
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
