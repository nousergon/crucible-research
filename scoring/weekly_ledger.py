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
    # The sessions the return was actually priced from. Normally identical to
    # (week_start, week_end); they diverge when a cut date is not itself a
    # session (a holiday re-cut) or when the closing session's bar has not
    # landed in ArcticDB yet. Recorded rather than assumed, because a week
    # priced over a shorter span than its label claims is a wrong number
    # wearing a right name — the defect class this ledger exists to end.
    "priced_from", "priced_to",
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
    priced_from: str | None = None,
    priced_to: str | None = None,
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

    ``priced_from`` / ``priced_to`` default to the week's own labels and are
    overridden by the caller when the boundary had to be resolved to a nearby
    session. They are stored rather than left implicit so a reader can see the
    span a number was actually measured over instead of trusting the label —
    the same reason ``net_unavailable_reason`` exists.

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
        "priced_from": priced_from or week_start,
        "priced_to": priced_to or week_end,
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


# ── Wiring: turning a scanner run into one completed week ────────────────────
#
# alpha-engine-config-I8264. Everything above is pure or S3-only and testable
# without a pipeline; this section is the impure edge that runs inside the
# Scanner invocation, and it is deliberately the ONLY place the two are joined.
#
# The shape of the wiring follows from what a week IS. A week's return is
# realized at the moment the NEXT cut replaces the current one — which is
# exactly the moment this code runs. So a scanner run does not record the week
# it is starting; it records the week it just ENDED, whose basket is the PRIOR
# membership artifact and whose end date is the cut being written now. There is
# no other run at which that week is both complete and cheap to price.
#
# WHAT THIS DOES NOT DO: it never walks history. A run records at most the ONE
# week that just closed. Backfilling the ledger from archived membership
# artifacts would mix forward-accumulated weeks (immune to the 2026-08-20
# fundamentals restatement, alpha-engine-config-I8255) with reconstructed ones
# (not immune) inside a store whose entire value is that no row was ever
# restated. If a prior series is ever wanted it is a separate, explicitly
# flagged decision — alpha-engine-config-I8262 owns it — and it must be
# distinguishable in the artifact, not silently interleaved.

_MAX_BOUNDARY_SLIP_DAYS = 5
"""How far back a week boundary may be resolved to find a priced session.

A cut date is normally a session, but a holiday re-cut or an ArcticDB append
that has not yet landed leaves the exact date unpriced. Resolving to the last
session on-or-before the boundary is correct and is RECORDED (``priced_from`` /
``priced_to``); resolving arbitrarily far back is not — past about a week the
span stops being the week the row is labelled with. Beyond this, the week is
reported unmeasurable rather than priced over a span nobody asked for.
"""

_CLOSE_COLS = ("Close", "close")


def _closes_panel(
    symbols: Sequence[str], *, week_start: str, week_end: str,
    bucket: str | None = None, ohlcv_loader: Any = None,
) -> dict[str, dict[str, float]]:
    """``{date: {ticker: close}}`` over the week, from the DURABLE OHLCV store.

    ArcticDB via ``nousergon_lib.arcticdb.load_universe_ohlcv``, never
    ``staging/daily_closes/``: that prefix carries a 7-day expiry rule, and a
    ledger that reads it would silently stop being able to price any week the
    moment a run slipped (alpha-engine-config-I5195 — the same source cost the
    leaderboards a month of empty artifacts).

    The window is the WEEK plus a small margin, not a cohort span. A weekly
    holding period needs exactly two cross-sections, so a wide read here would
    be dead weight paid inside a Lambda that has already died on a timeout once
    (alpha-engine-config-I7841). ``end`` is pinned to the week's end so the read
    cannot drift with the wall clock either.
    """
    import pandas as pd
    from nousergon_lib.arcticdb import load_universe_ohlcv

    span = (pd.Timestamp(week_end) - pd.Timestamp(week_start)).days
    loader = ohlcv_loader or load_universe_ohlcv
    frames = loader(
        _bucket(bucket),
        symbols=sorted({str(s).upper() for s in symbols}),
        lookback_days=max(span, 1) + _MAX_BOUNDARY_SLIP_DAYS + 2,
        end=week_end,
        columns=[_CLOSE_COLS[0]],
    ) or {}
    panel: dict[str, dict[str, float]] = {}
    for ticker, df in frames.items():
        if df is None or getattr(df, "empty", True):
            continue
        # ArcticDB's universe library stores TITLE-case OHLCV. Accept either
        # case rather than assuming — a wrong-case filter returns an empty
        # frame SILENTLY, which is the I5195 failure shape exactly.
        col = next((c for c in _CLOSE_COLS if c in df.columns), None)
        if col is None:
            continue
        for ts, close in df[col].items():
            if pd.isna(close) or float(close) <= 0:
                continue
            panel.setdefault(ts.strftime("%Y-%m-%d"), {})[str(ticker).upper()] = float(close)
    return panel


def resolve_boundary(panel: Mapping[str, Mapping[str, float]], boundary: str) -> str | None:
    """The latest priced session on-or-before ``boundary``, within the slip.

    None when no session inside :data:`_MAX_BOUNDARY_SLIP_DAYS` is priced. None
    rather than the nearest available date: past the slip the row would carry a
    span materially shorter than its own label, and a wrong number under a right
    name is the failure this ledger was built against.
    """
    from datetime import date, timedelta

    # ISO dates compare lexicographically, so the whole resolution is string
    # comparison against one computed floor — no timestamp parsing per key.
    floor = (date.fromisoformat(boundary) - timedelta(days=_MAX_BOUNDARY_SLIP_DAYS)).isoformat()
    candidates = [d for d in panel if floor <= d <= boundary]
    return max(candidates) if candidates else None


def prior_membership(
    membership: Mapping[str, Any], *, bucket: str | None = None, s3_client: Any = None,
) -> dict | None:
    """The membership artifact whose cut the completed week was HELD in.

    Resolved from the current artifact's own ``turnover`` block — which names
    the prior write by ``prior_run_date`` + ``prior_generated_at`` — and read
    from the IMMUTABLE ``runs/`` copy keyed on that timestamp, never from the
    dated ``membership.json`` pointer. On a day with two scanner runs the dated
    pointer already holds the later one, so reading it would price the week
    against a basket that was never held for it: the clobber defect
    (alpha-engine-config-I6785) entering through the reader instead of the
    writer.

    Falls back to the dated pointer ONLY when its ``generated_at`` matches the
    one the turnover block named — same bytes, different key. Any mismatch
    returns None: an unrecoverable prior is an unmeasurable week, not a week to
    price against whatever happens to be there.
    """
    import json

    from scoring.universe_membership import run_stamp

    turnover = membership.get("turnover") or {}
    prior_run_date = turnover.get("prior_run_date")
    prior_generated_at = turnover.get("prior_generated_at")
    if not prior_run_date or not prior_generated_at:
        return None
    s3 = _client(s3_client)
    b = _bucket(bucket)
    stamp = run_stamp(prior_generated_at)
    for key in (
        f"universe_membership/{prior_run_date}/runs/{stamp}.json",
        f"universe_membership/{prior_run_date}/membership.json",
    ):
        try:
            doc = json.loads(s3.get_object(Bucket=b, Key=key)["Body"].read())
        except Exception as exc:  # noqa: BLE001 — absence is a legitimate state here
            logger.info("[weekly_ledger] prior membership not at %s: %s", key, exc)
            continue
        if doc.get("generated_at") == prior_generated_at:
            return doc
        logger.warning(
            "[weekly_ledger] %s carries generated_at=%s but the turnover block "
            "named %s — a later same-day run has replaced it, so it is NOT the "
            "basket the completed week was held in",
            key, doc.get("generated_at"), prior_generated_at,
        )
    return None


def _arm_turnover(prior: Mapping[str, Any], arm: str) -> tuple[float | None, int | None]:
    """``(turnover_frac, retained_from_prior)`` for ``arm`` AT FORMATION.

    Read from the week's OWN artifact — the churn recorded when its basket was
    formed — not recomputed here and not taken from the following cut. Two
    reasons, and both matter:

    * ``compute_turnover`` already writes this on every run
      (alpha-engine-config-I6785). A second derivation in this module would be
      exactly the multi-writer drift the membership artifact exists to end.
    * Charging the FORMATION rebalance bills each week's basket exactly once,
      in the week that holds it. Charging the closing rebalance instead would
      also bill once, but the two conventions must never be mixed inside one
      series, and formation is the one whose number lives in the same artifact
      as the basket it belongs to.

    None turnover (the prior artifact's own first run) yields a null cost and a
    null net with a stated reason — never a zero, which is the real and
    different claim that an arm changed nothing.
    """
    per_cut = ((prior.get("turnover") or {}).get("per_cut") or {}).get(arm)
    if not per_cut:
        return None, None
    retention = per_cut.get("retention_pct")
    if retention is None:
        return None, per_cut.get("retained")
    return max(0.0, 1.0 - float(retention) / 100.0), per_cut.get("retained")


def record_completed_week(
    membership: Mapping[str, Any],
    *,
    bucket: str | None = None,
    s3_client: Any = None,
    market_regime: str | None = None,
    written_at: str | None = None,
    code_sha: str | None = None,
    ohlcv_loader: Any = None,
) -> dict:
    """Record the week that ``membership``'s new cut just ENDED. One week, once.

    ``membership`` is the artifact this scanner run just wrote. Returns a status
    dict — never raises for a week that cannot be recorded, because the caller
    is on the live Scanner path — carrying one of:

    ``ok``            rows were appended (or refused as already-written; the
                      per-row ``report`` says which, and "already written" is a
                      recorded outcome, not a silent no-op).
    ``skipped``       there is nothing to record: no prior cut (first run), or
                      the cut was carried forward so no week closed. A legitimate
                      state with a stated ``reason``, distinct from a failure.
    ``unmeasurable``  a week DID close and could not be priced. This is the
                      status that must alarm (champion-challenger-policy.md
                      §7.2): an unmeasurable result rendering as an empty success
                      is the fleet's dominant bug class.

    Every arm of the slot gets a row, including arms the prior cut did not carry
    — those are written as MISSES (null gross, ``n_names`` 0) rather than
    omitted, per §3: a cycle where an arm produced no output is recorded, and
    silent absence must never render like a genuine zero.
    """
    from datetime import UTC, datetime

    from scoring.universe_membership import SLOT_ARMS, live_cut_champion

    week_end_date = membership.get("cut_effective_date")
    prior = prior_membership(membership, bucket=bucket, s3_client=s3_client)
    if prior is None:
        return {
            "status": "skipped",
            "reason": "no_recoverable_prior_cut",
            "detail": (
                "no prior membership artifact was named by this run's turnover "
                "block, or the one named could not be recovered — the first "
                "run of the series, or a clobbered prior"
            ),
        }
    week = holding_period(prior.get("cut_effective_date"), week_end_date)
    if week is None:
        return {
            "status": "skipped",
            "reason": "no_week_closed",
            "detail": (
                f"the cut effective on {prior.get('cut_effective_date')} is still "
                f"in force (this run's cut_effective_date is {week_end_date}) — "
                "an unfinished week is not a missing week"
            ),
        }
    week_start, week_end = week

    prior_cuts = prior.get("cuts") or {}
    population = sorted((prior.get("ranks") or {}).keys())
    champion = live_cut_champion(bucket=bucket, s3_client=s3_client)
    champion_tickers = list((prior_cuts.get(champion) or {}).get("tickers") or [])

    symbols = {"SPY", *population, *champion_tickers}
    for arm in SLOT_ARMS:
        symbols.update((prior_cuts.get(arm) or {}).get("tickers") or [])

    try:
        panel = _closes_panel(
            sorted(symbols), week_start=week_start, week_end=week_end,
            bucket=bucket, ohlcv_loader=ohlcv_loader,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as unmeasurable, never swallowed
        return {
            "status": "unmeasurable",
            "week": [week_start, week_end],
            "reason": "closes_read_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    priced_from = resolve_boundary(panel, week_start)
    priced_to = resolve_boundary(panel, week_end)
    if priced_from is None or priced_to is None or priced_from >= priced_to:
        return {
            "status": "unmeasurable",
            "week": [week_start, week_end],
            "reason": "week_boundaries_unpriced",
            "detail": (
                f"no priced session within {_MAX_BOUNDARY_SLIP_DAYS} days of both "
                f"boundaries (resolved start={priced_from}, end={priced_to}) over "
                f"{len(symbols)} symbols — the ArcticDB universe library is the "
                "source; check MorningArcticAppend"
            ),
        }
    closes_start = dict(panel.get(priced_from) or {})
    closes_end = dict(panel.get(priced_to) or {})

    stamped = written_at or datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for arm in SLOT_ARMS:
        tickers = list((prior_cuts.get(arm) or {}).get("tickers") or [])
        turnover_frac, retained = _arm_turnover(prior, arm)
        rows.append(
            build_week_row(
                arm=arm,
                week_start=week_start,
                week_end=week_end,
                tickers=tickers,
                closes_start=closes_start,
                closes_end=closes_end,
                turnover_frac=turnover_frac,
                retained_from_prior=retained,
                population_tickers=population,
                champion_tickers=champion_tickers,
                # Resolved from the live pointer, never a literal
                # (champion-challenger-policy.md §7.5). Read BEFORE the
                # promotion engine runs later in this same invocation, so it
                # names the arm that actually served the week being recorded.
                is_champion=(arm == champion),
                # The regime observed as the week CLOSED — this run. Recorded
                # for conditioning, not as a claim about the whole week.
                market_regime=market_regime,
                priced_from=priced_from,
                priced_to=priced_to,
                code_sha=code_sha,
                written_at=stamped,
            )
        )

    try:
        report = append_week(rows, bucket=bucket, s3_client=s3_client)
    except Exception as exc:  # noqa: BLE001 — the caller is the live Scanner path
        return {
            "status": "unmeasurable",
            "week": [week_start, week_end],
            "reason": "append_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    priced = [r["arm"] for r in rows if r["gross_log_return"] is not None]
    return {
        "status": "ok",
        "week": [week_start, week_end],
        "priced_span": [priced_from, priced_to],
        "champion": champion,
        "arms_priced": priced,
        "arms_missing": [r["arm"] for r in rows if r["gross_log_return"] is None],
        "report": {k: [list(key) for key in v] for k, v in report.items()},
        "key": LEDGER_KEY,
    }
