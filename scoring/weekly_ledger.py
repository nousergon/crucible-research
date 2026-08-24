"""weekly_ledger.py — the universe-cut slot's append-only performance record
(alpha-engine-config-I8261, Brian's ruling 2026-08-24, option (d)).

Brian, 2026-08-24: *"shouldn't we just be tracking performance weekly?"*

WHAT THIS MEASURES, AND WHY IT IS DIFFERENT FROM THE LEADERBOARD
---------------------------------------------------------------
The scanner re-forms its cuts **weekly**. So the return the system actually
earns from a cut is the return over the week it is HELD — formed on
``cut_effective_date``, held until the next cut replaces it. One week, one
observation. Chain them for a month, a quarter, a year.

``scoring/leaderboard_producers`` answers a different question: "what did the
names in this cut do over the N sessions after it was formed?" That is an
event study of the RANKING, and it is a legitimate diagnostic. It is not what
the slot earns, for two reasons:

1. **It measures a hold that never happens.** A 126-session forward return
   from a cohort date describes a six-month hold, and the weekly re-cut
   guarantees the cut is replaced 25 times inside that window.
2. **Its observations overlap.** Weekly cohort dates against a 21-session
   window share ~76% of their span, so consecutive observations are strongly
   dependent. That is corrected now (alpha-engine-config-I8263) but the
   correction costs power that a non-overlapping series never loses in the
   first place.

A weekly holding-period series has neither problem. Successive weeks ABUT
rather than overlap, so ``date_clustered_stats``'s "each date = one
independent cluster" contract — written for exactly this shape, and applied
until now to something else — becomes true rather than assumed.

THE LEDGER IS APPEND-ONLY AND IMMUTABLE, WHICH IS THE POINT
-----------------------------------------------------------
``build_cuts_leaderboard`` recomputes its entire cohort from scratch on every
run. A change to the scoring code therefore silently RESTATES every historical
number, and no reader can tell which code produced which row. For a series
meant to accumulate for years, that is the defect that matters most in the
long run — it is not detectable from the artifact, and every conclusion drawn
from a restated series is unfalsifiable after the fact.

So a ledger row is written ONCE, for one ``(arm, week)``, and never rewritten.
Each row carries ``ledger_version`` and ``code_sha`` so a reader can tell which
implementation produced it, and a later version's rows sit ALONGSIDE the
earlier ones rather than replacing them. :func:`append_week` refuses to
overwrite an existing key unless explicitly asked, and the refusal is a
recorded outcome (``skipped_immutable``), never a silent no-op.

WHY NET-OF-COST IS NOT OPTIONAL HERE
-------------------------------------
At weekly rebalance, turnover is first-order, and the arms in this slot have
very different churn — 42% week-over-week retention for the gate cut against
76% for the attractiveness cut, measured 2026-07-27 (``EXPERIMENTS.md``). An
arm that wins gross and loses net is the classic trap, and a weekly slot with
a 34-point retention spread between arms is built to fall into it. **A
gross-only weekly series would be a worse answer than the forward returns it
replaces, because it would look decisive.**

Cost comes from ``nousergon_lib.quant.transaction_cost`` — the fleet's one
square-root-impact engine — applied to the turnover the membership artifact
already records. Both inputs existed before this module; neither was used.

A row whose net return could not be computed carries ``net_log_return: None``
and a ``net_unavailable_reason``. It is never silently equal to gross: those
are different claims, and rendering them identically is the failure mode this
whole arc has been about.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

LEDGER_KEY = "research/cuts_weekly_ledger/ledger.parquet"
"""One object, not one per week.

A per-week key would make "read the whole series" an N-object listing whose
cost grows without bound, and the series is read in full on every use — that
is what a series IS. The append-only guarantee lives in
:func:`append_week`'s refusal to replace an existing ``(arm, week)`` row, not
in the object layout.
"""

LEDGER_VERSION = 1
"""Bumped when the MEANING of a column changes, never for an added column.

A reader comparing rows across versions needs to know whether
``net_log_return`` means the same thing in both. An added column is answerable
from the schema; a redefined one is not, and silently redefining one is how a
multi-year series becomes uninterpretable.
"""

LEDGER_COLS = [
    # Identity — the primary key is (arm, week_start).
    "arm", "week_start", "week_end",
    # What the arm held.
    "n_names", "retained_from_prior", "turnover_frac",
    # What it earned. Log-domain throughout, market-relative computed by the
    # consumer from the legs rather than baked in, so a reader can see both.
    "gross_log_return", "net_log_return", "net_unavailable_reason",
    "cost_bps",
    # What it is measured against. Three legs, three questions.
    "benchmark_log_return", "population_log_return", "champion_log_return",
    # The signal question, alongside the portfolio question.
    "rank_ic",
    # Conditioning variables — cheap to record now, unrecoverable later.
    "market_regime", "is_champion",
    # Provenance.
    "ledger_version", "code_sha", "written_at",
]

_DEFAULT_BUCKET = "alpha-engine-research"

# Reference notional for the cost model, in dollars. The square-root impact
# term needs a trade size; the slot has no position sizing of its own (it
# decides which names are ELIGIBLE, not how much of each is held), so a fixed
# reference makes the cost comparable ACROSS ARMS, which is the only
# comparison this ledger supports. It is deliberately not a portfolio NAV:
# tying the arms' cost to a live NAV would make an arm's historical record
# move when the account did.
REFERENCE_NOTIONAL_USD = 100_000.0


class WeeklyLedgerError(RuntimeError):
    """Raised when a ledger write cannot be completed correctly.

    Deliberately a hard error on the WRITE path (feedback_no_silent_fails):
    this artifact is the slot's durable record of what happened, and a partial
    or silently-skipped append is indistinguishable from a week in which
    nothing was measured. The CALLER decides whether to fail-soft — the
    scanner's own run must not go red for an observe-only ledger — but it must
    decide knowingly rather than by never being told.
    """


# ── The week's holding period ────────────────────────────────────────────────

def holding_period(
    cut_effective_date: str, next_cut_effective_date: str | None,
) -> tuple[str, str] | None:
    """``(week_start, week_end)`` for a cut, or None when the week is not over.

    The holding period ENDS when the next cut replaces this one — not a fixed
    five sessions later. The two diverge whenever a weekly run is missed, a
    holiday shortens the week, or a re-cut lands off-cadence, and using a fixed
    window would then attribute returns to an arm that had already been
    replaced, or drop returns it genuinely earned.

    Returns None when ``next_cut_effective_date`` is absent: the current week
    is still being held, its return is not yet realized, and writing a partial
    week would put a number in the series that a later run would want to
    change — which is precisely what the append-only guarantee forbids. An
    unfinished week is not a missing week; it is a week that has not happened
    yet, and the ledger simply does not carry it until it has.
    """
    if not cut_effective_date or not next_cut_effective_date:
        return None
    if next_cut_effective_date <= cut_effective_date:
        return None
    return (cut_effective_date, next_cut_effective_date)


# ── The measurement ──────────────────────────────────────────────────────────

def equal_weight_log_return(
    tickers: Sequence[str], closes_start: Mapping[str, float],
    closes_end: Mapping[str, float],
) -> tuple[float | None, int]:
    """Equal-weight log return over ``tickers``, and how many names contributed.

    Log domain and arithmetic mean of per-name log returns — the same
    convention every other alpha quantity in the fleet uses, so a ledger row
    and a leaderboard row are comparable without a units conversion.

    Returns ``(None, 0)`` when no name has both closes. A name missing from
    either side is DROPPED rather than treated as flat: a delisting or a data
    gap is not a 0% return, and averaging a fabricated zero into the basket
    would bias every arm toward the mean by an amount proportional to its own
    data quality.
    """
    import math

    rets: list[float] = []
    for t in tickers:
        p0 = closes_start.get(t)
        p1 = closes_end.get(t)
        if p0 is None or p1 is None:
            continue
        try:
            p0f, p1f = float(p0), float(p1)
        except (TypeError, ValueError):
            continue
        if p0f <= 0.0 or p1f <= 0.0:
            continue
        rets.append(math.log(p1f / p0f))
    if not rets:
        return None, 0
    return sum(rets) / len(rets), len(rets)


def turnover_cost_bps(
    turnover_frac: float | None,
    *,
    adv_dollar: float | None = None,
    notional: float = REFERENCE_NOTIONAL_USD,
    config: dict | None = None,
) -> float | None:
    """Round-trip cost in bps of rebalancing a ``turnover_frac`` share of the
    basket, or None when turnover is unknown.

    Turnover is the fraction of the basket REPLACED, and replacing a name is
    two sides — sell the old, buy the new — so the round-trip cost applies to
    the replaced share. An arm retaining 76% of its names pays this on 24% of
    the basket; one retaining 42% pays it on 58%. That ratio is the entire
    reason this column exists.

    None rather than 0.0 on unknown turnover. Zero is a real and different
    claim — an arm that changed nothing — and a slot whose arms differ mainly
    in churn must never render "no churn" and "churn unknown" identically.
    """
    if turnover_frac is None:
        return None
    try:
        frac = float(turnover_frac)
    except (TypeError, ValueError):
        return None
    if frac < 0.0:
        return None
    from nousergon_lib.quant.transaction_cost import TransactionCostModel

    model = TransactionCostModel.from_config(config)
    return model.round_trip_bps(notional, adv_dollar) * min(frac, 1.0)


def net_from_gross(
    gross_log_return: float | None, cost_bps: float | None,
) -> tuple[float | None, str | None]:
    """``(net_log_return, unavailable_reason)``.

    Cost is subtracted in the LOG domain as ``cost_bps / 10_000``. At the
    magnitudes involved (tens of bps) the log/simple distinction is far below
    the noise in a weekly equity return, and keeping one domain throughout is
    worth more than the third-decimal precision a conversion would buy.

    A missing cost yields ``(None, reason)`` — never gross. Returning gross
    under a "net" column name is the exact shape of defect this arc has spent
    its length on: a number that is plausible, close to right, and answers a
    different question than its name claims.
    """
    if gross_log_return is None:
        return None, "gross_return_unavailable"
    if cost_bps is None:
        return None, "turnover_unknown_so_cost_uncomputable"
    return gross_log_return - (float(cost_bps) / 10_000.0), None


# ── Row assembly ─────────────────────────────────────────────────────────────

def build_week_row(
    *,
    arm: str,
    week_start: str,
    week_end: str,
    tickers: Sequence[str],
    closes_start: Mapping[str, float],
    closes_end: Mapping[str, float],
    turnover_frac: float | None,
    retained_from_prior: int | None = None,
    benchmark_ticker: str = "SPY",
    population_tickers: Sequence[str] | None = None,
    champion_tickers: Sequence[str] | None = None,
    rank_ic: float | None = None,
    market_regime: str | None = None,
    is_champion: bool = False,
    adv_dollar: float | None = None,
    config: dict | None = None,
    code_sha: str | None = None,
    written_at: str,
) -> dict:
    """One immutable ledger row. PURE — no I/O, no clock.

    ``written_at`` is a required ARGUMENT rather than a ``utcnow()`` call, so
    the row is a deterministic function of its inputs and a replay produces a
    byte-identical row. A module that stamps its own clock cannot be tested for
    the idempotence its own append-only guarantee depends on.

    The three benchmark legs answer three different questions and none
    substitutes for another:

    ``benchmark_log_return``   SPY. "Did this beat the market?" — closest to
                               the executor's own objective.
    ``population_log_return``  the whole scanned universe, equal-weight. "Did
                               narrowing to 60 names beat NOT narrowing?" —
                               the selection-skill question, and the slot's
                               primary metric.
    ``champion_log_return``    the serving arm, same week. The PAIRED leg: a
                               consumer differences it per week, which cancels
                               the common market factor and collapses the
                               variance relative to either leg alone. At ~52
                               observations a year that is the difference
                               between needing tens of observations and
                               hundreds (champion-challenger-policy.md §4).

    The row carries the LEGS, not the differences. A difference is one
    subtraction a consumer can always do; a leg it was never given is gone for
    good, and a stored difference cannot be re-based when the champion changes.
    """
    gross, n_names = equal_weight_log_return(tickers, closes_start, closes_end)
    cost_bps = turnover_cost_bps(
        turnover_frac, adv_dollar=adv_dollar, config=config,
    )
    net, net_reason = net_from_gross(gross, cost_bps)

    bench, _ = equal_weight_log_return([benchmark_ticker], closes_start, closes_end)
    pop = None
    if population_tickers:
        pop, _ = equal_weight_log_return(population_tickers, closes_start, closes_end)
    champ = None
    if champion_tickers is not None and not is_champion:
        champ, _ = equal_weight_log_return(champion_tickers, closes_start, closes_end)

    return {
        "arm": arm,
        "week_start": week_start,
        "week_end": week_end,
        "n_names": n_names,
        "retained_from_prior": retained_from_prior,
        "turnover_frac": (float(turnover_frac) if turnover_frac is not None else None),
        "gross_log_return": gross,
        "net_log_return": net,
        "net_unavailable_reason": net_reason,
        "cost_bps": cost_bps,
        "benchmark_log_return": bench,
        "population_log_return": pop,
        "champion_log_return": champ,
        "rank_ic": rank_ic,
        "market_regime": market_regime,
        "is_champion": bool(is_champion),
        "ledger_version": LEDGER_VERSION,
        "code_sha": code_sha or os.environ.get("CODE_SHA") or "unknown",
        "written_at": written_at,
    }


# ── The store ────────────────────────────────────────────────────────────────

def _bucket(bucket: str | None) -> str:
    return bucket or os.environ.get("RESEARCH_BUCKET") or _DEFAULT_BUCKET


def _client(s3_client: Any):
    if s3_client is not None:
        return s3_client
    import boto3

    return boto3.client("s3")


def read_ledger(*, bucket: str | None = None, s3_client: Any = None):
    """The full ledger → DataFrame, or None when it has never been written.

    None, not an empty frame: "no ledger exists" and "the ledger is empty" are
    different states — the first is a first run, the second would be a write
    that produced nothing — and a consumer that cannot tell them apart will
    report a healthy empty series for a store that was never created
    (champion-challenger-policy.md §7.2).
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover — pandas is in the Lambda image
        return None
    s3 = _client(s3_client)
    try:
        obj = s3.get_object(Bucket=_bucket(bucket), Key=LEDGER_KEY)
    except Exception:
        return None
    return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")


def append_week(
    rows: Sequence[Mapping[str, Any]],
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    allow_restate: bool = False,
) -> dict:
    """Append ``rows`` to the ledger. Returns a per-row outcome report.

    **An existing ``(arm, week_start)`` is NOT overwritten.** That is the whole
    contract: the leaderboard recomputes its history on every run, so a scoring
    change silently restates numbers nobody can audit afterward. Here a week is
    written once and stands. A refused row is reported as ``skipped_immutable``
    — a recorded outcome, never a silent no-op, because "already written" and
    "failed to write" must not look the same to the caller.

    ``allow_restate`` exists for one legitimate case: a row written from
    inputs later found to be wrong (a contaminated close, a mis-stamped cut).
    Restating then is a deliberate, logged act. It is not a default, and it is
    not how a code change reaches the history — a new ``LEDGER_VERSION`` writes
    ALONGSIDE the old rows so both are visible.
    """
    import pandas as pd

    report = {"written": [], "skipped_immutable": [], "restated": []}
    if not rows:
        return report

    new = pd.DataFrame(list(rows)).reindex(columns=LEDGER_COLS)
    if new[["arm", "week_start"]].isna().any().any():
        raise WeeklyLedgerError(
            "a ledger row is missing its primary key (arm, week_start) — a row "
            "that cannot be identified cannot be protected from restatement, "
            "which is the only guarantee this store makes."
        )
    if new.duplicated(subset=["arm", "week_start"]).any():
        dupes = new[new.duplicated(subset=["arm", "week_start"], keep=False)]
        raise WeeklyLedgerError(
            "two rows in ONE append claim the same (arm, week_start): "
            f"{sorted(set(zip(dupes['arm'], dupes['week_start'], strict=True)))}. "
            "Last-write-wins inside a single batch would make which one landed "
            "depend on row order, and neither would be recorded as skipped."
        )

    existing = read_ledger(bucket=bucket, s3_client=s3_client)
    if existing is not None and not existing.empty:
        existing = existing.reindex(columns=LEDGER_COLS)
        have = set(zip(existing["arm"], existing["week_start"], strict=True))
        keep_new, dropped = [], []
        for row in new.to_dict("records"):
            key = (row["arm"], row["week_start"])
            if key not in have:
                keep_new.append(row)
                report["written"].append(key)
            elif allow_restate:
                keep_new.append(row)
                report["restated"].append(key)
                logger.warning(
                    "[weekly_ledger] RESTATING %s %s — an already-written week "
                    "is being replaced deliberately (allow_restate=True). This "
                    "breaks the append-only guarantee for this row by design; "
                    "it must never be the path a code change takes into the "
                    "history (alpha-engine-config-I8261).", *key,
                )
            else:
                dropped.append(key)
                report["skipped_immutable"].append(key)
        if dropped:
            logger.info(
                "[weekly_ledger] %d row(s) already written and left untouched: "
                "%s", len(dropped), sorted(dropped)[:8],
            )
        if not keep_new:
            return report
        restated = {(r["arm"], r["week_start"]) for r in keep_new}
        mask = [
            (a, w) not in restated
            for a, w in zip(existing["arm"], existing["week_start"], strict=True)
        ]
        combined = pd.concat(
            [existing[mask], pd.DataFrame(keep_new).reindex(columns=LEDGER_COLS)],
            ignore_index=True,
        )
    else:
        combined = new
        report["written"] = list(zip(new["arm"], new["week_start"], strict=True))

    combined = combined.sort_values(["week_start", "arm"]).reset_index(drop=True)
    buf = io.BytesIO()
    combined.to_parquet(buf, engine="pyarrow", index=False)
    _client(s3_client).put_object(
        Bucket=_bucket(bucket), Key=LEDGER_KEY,
        Body=buf.getvalue(), ContentType="application/octet-stream",
    )
    logger.info(
        "[weekly_ledger] %d row(s) written, %d skipped as already-written; "
        "ledger now %d rows → s3://%s/%s",
        len(report["written"]), len(report["skipped_immutable"]),
        len(combined), _bucket(bucket), LEDGER_KEY,
    )
    return report


# ── Reading the series back ──────────────────────────────────────────────────

def chained_log_return(weekly: Sequence[float | None]) -> float | None:
    """Sum of the weekly log returns — the compounded return over the span.

    Log returns ADD across time, which is the whole reason the ledger stores
    them: a six-month read is the sum of 26 weekly rows, computed by a reader,
    with no re-derivation from prices and no dependence on the scoring code
    that produced any individual week.

    A None week makes the whole span None rather than being skipped. Skipping
    would silently report a 25-week return under a 26-week label, and a gap in
    a compounding series is not a zero — it is a span that cannot be stated.
    """
    vals = list(weekly)
    if not vals or any(v is None for v in vals):
        return None
    return float(sum(float(v) for v in vals))


def paired_weekly_differences(
    rows: Sequence[Mapping[str, Any]], *, column: str = "net_log_return",
) -> list[float]:
    """Per-week ``arm − champion`` differences, for the rows of ONE arm.

    The paired series the slot decides on. Weeks where either leg is missing
    are dropped, because a difference needs both legs and substituting a zero
    for an absent champion leg would manufacture a week in which the arm
    exactly matched the champion.

    Defaults to the NET column. The gross difference is available by asking for
    it, and is the wrong default for a slot whose arms differ mainly in churn.
    """
    out: list[float] = []
    for row in rows:
        mine = row.get(column)
        theirs = row.get("champion_log_return")
        if mine is None or theirs is None:
            continue
        try:
            out.append(float(mine) - float(theirs))
        except (TypeError, ValueError):
            continue
    return out
