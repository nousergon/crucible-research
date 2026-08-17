"""Champion/challenger leaderboard PRODUCERS — the two thin observe-only,
fail-soft I/O wrappers around the shared scorer (``scoring/leaderboard_scoring``):

- ``build_scanner_leaderboard``  → ``scanner/leaderboard/{date}.json``      (config#1221)
- ``build_producer_leaderboard`` → ``research/producer_leaderboard/{date}.json`` (config#1223)

Both read the shadow artifacts the substrate already emits, resolve realized
forward (21d) returns per cohort date, call the ONE shared scorer, and write the
leaderboard JSON. They are OBSERVE-ONLY (never read by live trading) and FULLY
FAIL-SOFT: every public entry point wraps its body in try/except that LOGS and
RETURNS a status dict — it never raises into the live signal/eval path
(no-silent-fails: the failure is recorded, never swallowed silently).

Off-hot-path + backfillable like ``scripts/build_agent_quality.py``: they only
READ persisted S3 artifacts, so they run after the fact for any past date and
never perturb the live pipeline.

Realized-return join (cohort gate): for a cohort date *d* the forward return of
ticker *t* is ``close(t, d+H) / close(t, d) − 1`` where *H* = horizon trading
days, read from the DURABLE ArcticDB universe library via
``nousergon_lib.arcticdb.load_universe_ohlcv``. A date whose horizon has not
matured simply does not join — its metrics stay an honest ``None`` until the
cohort matures. The horizon-end date is the H-th session after *d* in the
panel's own calendar (trading days only — non-sessions are absent by
construction).

Source note (alpha-engine-config-I5195): this previously read
``staging/daily_closes/{date}.parquet``, a prefix under an
``expire-staging-after-7-days`` S3 lifecycle rule. A 21-session LOOKFORWARD
cannot be served by a 7-day-retention source, so the join could never resolve
and both leaderboards wrote ``n_dates: 0`` weekly for over a month while
looking perfectly healthy. Two defences now exist: the source is durable, and
``_assert_horizon_is_satisfiable`` checks the horizon-vs-coverage relationship
DIRECTLY rather than letting a downstream count of zero stand in for it.

Fail-soft boundary: these producers stay fail-soft toward the LIVE PATH (a
scoring failure must never red the Scanner or research Lambda), but they are
no longer silent. An unmeasurable leaderboard is written with
``status: "unmeasurable"`` and raises an observe alert — "we could not
measure" and "nothing has matured yet" are different states and must never
render identically again.

Vacuity guard (champion-challenger-policy.md §4, alpha-engine-config-I6429):
after specs are loaded, every challenger's resolved picked-ticker set is
compared against the champion's, per cohort date. An exact-set collision is
LOUD (``publish_observe_alert``, same mechanism as the ``unmeasurable``
alert) and recorded on the leaderboard as ``vacuous_membership_collisions``,
never blocking — a well-formed, green leaderboard comparing an arm to itself
must be visible, not silent. Previously this existed only as a fixture-based
unit test (``tests/test_universe_membership.py::
test_incumbent_arm_disagrees_with_the_champion``) that never ran against a
live cycle's actual resolved membership.

MULTI-HORIZON (alpha-engine-config-I7540). Every arm is now scored at 21, 126
AND 252 trading sessions — one block per horizon under ``horizons``, each with
its own ``n_dates`` and its own per-spec rows. 21 alone could not answer the
~1-year question the scanner exists to ask. The panel is read ONCE, sized to
the longest horizon, and every horizon's returns are derived from it. The 21-
session block remains the artifact's TOP LEVEL, unchanged in shape and value,
because a promoted arm keeps its history (champion-challenger-policy.md §3) and
every existing consumer reads that surface.

PER-ROW CONFIDENCE (alpha-engine-config-I7542). Every spec row carries
``confidence`` — ``insufficient`` / ``thin`` / ``ok`` — against the slot
registry's ``min_dates_for_inference``. A one-date mean with a null SE and a
null t-stat is not a result, and before this it rendered in the same shape as
one. Rows are never suppressed on the strength of that status: §3 requires the
miss to stay visible, and a hidden thin row is indistinguishable from an arm
that was never scored.

The long horizons make immaturity the DOMINANT rendering for months — a
252-session horizon needs ~252 trading days of shadow history before any date
scores. That is honest immaturity by construction: it renders as
``status: "immature"`` with a reason and every row ``insufficient``, never as a
numeric result and never as a zero (§7.2).
"""

from __future__ import annotations

import json
import logging
from datetime import date as _date
from typing import Any

import arcticdb as _arcticdb  # noqa: F401

# Imported at module top, BEFORE anything pulls in pandas/pyarrow, to prime
# arcticdb's bundled aws-c-common allocator — mirrors the identical guard in
# data/fetchers/price_fetcher.py. The lib chokepoint imports arcticdb lazily,
# which lets pyarrow's allocator load first and SEGFAULTS on the first
# get_library() call (reproduced locally 2026-07-28 while smoke-testing the
# I5195 closes read: a native abort, not a Python exception, so no traceback
# and no fail-soft path can catch it). Also fails loud at cold start if an
# image lacks arcticdb, rather than degrading silently at scoring time.
from nousergon_lib.trading_calendar import count_trading_days

from observe_alerts import publish_observe_alert
from scoring.leaderboard_scoring import (
    DEFAULT_HORIZON_DAYS,
    HORIZON_IMMATURE,
    HORIZON_OK,
    HORIZON_UNMEASURABLE,
    SpecDay,
    SpecHistory,
    score_multi_horizon,
    slot_spec,
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "alpha-engine-research"
# ArcticDB universe library stores title-case OHLCV columns.
_CLOSE_COL = "Close"

_SCANNER_OUTPUT = "scanner/leaderboard/{date}.json"
_PRODUCER_OUTPUT = "research/producer_leaderboard/{date}.json"

_CANDIDATES_SHADOW = "candidates_shadow/{spec}/{date}/candidates.json"
_CANDIDATES_LIVE = "candidates/{date}/candidates.json"
_SIGNALS_SHADOW = "signals_shadow/{producer}/{date}/signals.json"
_SIGNALS_LIVE = "signals/{date}/signals.json"
# SSoT for WHICH producer is champion, written by crucible-backtester's
# promotion engine (optimizer/champion_promotion.py). The research-side
# registry does NOT duplicate this fact — see _resolve_champion_name.
_CHAMPION_POINTER = "config/producer_champion.json"


# ── S3 read helpers (mirror build_agent_quality._get_json) ────────────────────


def _get_json(s3: Any, bucket: str, key: str) -> dict | None:
    from botocore.exceptions import ClientError

    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


class LeaderboardUnmeasurableError(RuntimeError):
    """The leaderboard cannot be computed at all — as distinct from "computed,
    and nothing has matured yet".

    These two states rendered IDENTICALLY before alpha-engine-config-I5195 (a
    well-formed artifact with ``n_dates: 0``), which is why a structurally
    impossible measurement ran for over a month unnoticed. They must never be
    conflated again: an immature cohort is a legitimate result, an unreadable
    source is a defect.
    """


def _assert_horizon_is_satisfiable(
    close_dates: list[str],
    entry_dates: list[str],
    horizon_days: int,
) -> None:
    """Raise unless the closes SOURCE is capable of supporting a
    ``horizon_days`` lookforward at all.

    THE I5195 INVARIANT: *a measurement's horizon must never exceed the
    retention of the data it reads.* The original source held ~7 days and the
    horizon was 21 sessions, so the join could never resolve — yet every failure
    mode was best-effort, so a structural impossibility rendered as an ordinary
    empty result for over a month.

    This checks the SOURCE's capability, deliberately NOT a particular cohort's
    maturity. The two must not be conflated in either direction:

      - **Structural (raises):** the calendar holds fewer than ``horizon_days``
        sessions in total, so NO cohort date can ever mature; or every cohort
        date falls before the calendar begins, so the source cannot see the
        cohort at all (a retention window that slid past it).
      - **Immature (does NOT raise):** the source is capable and the cohort is
        inside it, but this particular entry date's horizon has not elapsed
        yet. That is an honest ``None`` and a legitimate ``ok`` result — a
        freshly-entered cohort is the normal state every Monday, not a defect.

    An earlier formulation of this check counted only sessions AFTER the
    earliest cohort date, which fired on exactly that legitimate fresh-cohort
    case. Retained as a comment because it is the tempting wrong version.
    """
    if not close_dates:
        raise LeaderboardUnmeasurableError("closes calendar is empty")

    if len(close_dates) < horizon_days:
        raise LeaderboardUnmeasurableError(
            f"closes calendar holds {len(close_dates)} session(s) but the "
            f"scoring horizon is {horizon_days} sessions — no cohort date can "
            f"ever mature against this source (calendar spans "
            f"{close_dates[0]}..{close_dates[-1]}). The horizon exceeds the "
            "source's effective retention. See alpha-engine-config-I5195."
        )

    latest_entry = max(entry_dates)
    if latest_entry < close_dates[0]:
        raise LeaderboardUnmeasurableError(
            f"every cohort date (latest {latest_entry}) precedes the closes "
            f"calendar, which begins {close_dates[0]} — the source's retention "
            "window has slid past the entire cohort, so no entry close can be "
            "read. See alpha-engine-config-I5195."
        )


# ``_list_close_dates`` / ``_read_closes`` were REMOVED with
# alpha-engine-config-I5195. They paginated ``staging/daily_closes/`` — a
# prefix under an ``expire-staging-after-7-days`` S3 lifecycle rule — to
# serve a 21-session lookforward. Deleted rather than left unused: a
# dormant reader of an ephemeral prefix is an invitation to wire it back
# in, and the whole defect was that nothing made the retention/horizon
# mismatch visible at the call site.


def _horizon_date(close_dates: list[str], entry_date: str, horizon_days: int) -> str | None:
    """The trading date ``horizon_days`` sessions AFTER ``entry_date`` in the
    available daily_closes calendar, or None if the horizon hasn't matured."""
    after = [d for d in close_dates if d > entry_date]
    if len(after) < horizon_days:
        return None
    return after[horizon_days - 1]


def _closes_panel_from_arcticdb(
    bucket: str,
    entry_dates: list[str],
    horizon_days: int,
    symbols: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """``{date: {ticker: close}}`` over the window the cohort needs, read from
    the DURABLE ArcticDB universe library.

    Why not ``staging/daily_closes/`` (alpha-engine-config-I5195): that prefix
    carries an S3 lifecycle rule ``expire-staging-after-7-days``, while this
    scorer needs a ``horizon_days``-session LOOKFORWARD (21 by default). A
    21-session forward return cannot be computed from a 7-day-retention source
    — the horizon was 3x the retention, so the join could never resolve and the
    leaderboards wrote ``n_dates: 0`` every week for over a month without ever
    erroring. ArcticDB is the fleet's durable OHLCV store (written daily by
    MorningArcticAppend) and holds years, so the window is always covered.

    Reads via ``nousergon_lib.arcticdb.load_universe_ohlcv`` — the SSoT
    chokepoint for "read an OHLCV slice per ticker out of ArcticDB" — rather
    than reimplementing the read/dedup/normalize idiom here (§15: one shared
    quant engine).
    """
    import pandas as pd
    from nousergon_lib.arcticdb import load_universe_ohlcv

    earliest = min(entry_dates)
    # Cover earliest entry → today, plus slack so the horizon session after the
    # LAST entry date is inside the window AND the panel holds at least
    # ``horizon_days`` sessions in total (which is what
    # ``_assert_horizon_is_satisfiable`` checks).
    #
    # alpha-engine-config-I7540 widened this: ``horizon_days`` is now the
    # LONGEST horizon scored (252 sessions), not 21. Trading sessions run ~252
    # per 365 calendar days (~0.69), so H sessions span ~H/0.69 calendar days —
    # 252 sessions ≈ 366 calendar days. The 2x + 10 slack therefore yields ~514
    # calendar days of pad against the ~366 actually needed at the longest
    # horizon, and proportionally more at the shorter ones. Deliberately kept
    # multiplicative rather than pinned to a constant: a slack that does not
    # scale with the horizon is exactly the shape of the I5195 defect (a
    # measurement whose window silently could not serve its own horizon).
    #
    # ArcticDB carries no lifecycle expiry (verified live 2026-08-17: the
    # ``alpha-engine-research`` bucket has exactly two lifecycle rules, on
    # ``staging/`` and ``features/``, neither matching ``arcticdb/``), and
    # ``load_universe_ohlcv`` defaults to a 730-day window, so a ~570-day read
    # is comfortably inside what the source holds.
    span_days = (pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timestamp(earliest)).days
    lookback = max(span_days, 1) + horizon_days * 2 + 10

    # Narrow to the tickers the arms actually picked (plus the benchmark).
    # Reading the full ~900-symbol universe took >2min locally — far too
    # slow for the Lambda this runs in, and entirely wasted work since only
    # picked names contribute to any metric.
    frames = load_universe_ohlcv(
        bucket,
        symbols=sorted(symbols) if symbols else None,
        lookback_days=lookback,
        columns=[_CLOSE_COL],
    )
    panel: dict[str, dict[str, float]] = {}
    for ticker, df in (frames or {}).items():
        if df is None or df.empty:
            continue
        # ArcticDB's universe library stores TITLE-case OHLCV ("Close"), per
        # alpha-engine-data's builders/backfill.py + daily_append.py schema.
        # Accept either case rather than assuming: a wrong-case column filter
        # returns an empty frame SILENTLY, which is precisely the failure shape
        # I5195 is about (caught here in smoke-testing, not by any unit test).
        col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if col is None:
            continue
        for ts, close in df[col].items():
            if pd.isna(close) or close <= 0:
                continue
            panel.setdefault(ts.strftime("%Y-%m-%d"), {})[str(ticker)] = float(close)
    return panel


def _resolve_realized_returns(
    s3: Any,
    bucket: str,
    entry_dates: list[str],
    horizon_days: int,
    *,
    symbols: set[str] | None = None,
    closes_panel_loader: Any = None,
) -> dict[str, dict[str, float]]:
    """``{entry_date: {ticker: forward_return}}`` for every entry date whose
    horizon close has matured, as ``close_h / close_0 − 1``.

    Best-effort per DATE (a date whose horizon has genuinely not matured yet
    simply does not appear — that is an honest ``None``, not an error), but NOT
    best-effort about the SOURCE: an unreachable closes panel raises, because a
    silently-empty panel is indistinguishable from "no cohort has matured" and
    that ambiguity is exactly what hid I5195 for a month. The caller converts
    the raise into a loud `unmeasurable` verdict.
    """
    if not entry_dates:
        return {}
    panel = _load_closes_panel(bucket, entry_dates, horizon_days, symbols, closes_panel_loader)
    return _realized_from_panel(panel, entry_dates, horizon_days)


def _load_closes_panel(
    bucket: str,
    entry_dates: list[str],
    horizon_days: int,
    symbols: set[str] | None,
    closes_panel_loader: Any,
) -> dict[str, dict[str, float]]:
    """Load the closes panel once, wide enough for ``horizon_days``.

    Injectable so the source is a parameter, not a hard binding: tests supply a
    panel directly, and a backfill can point at an alternative store without
    editing this module. Production default is the durable ArcticDB read.

    An EMPTY panel raises rather than returning ``{}`` — a silently-empty panel
    is indistinguishable from "no cohort has matured", and that ambiguity is
    exactly what hid I5195 for a month.
    """
    loader = closes_panel_loader or _closes_panel_from_arcticdb
    panel = loader(bucket, entry_dates, horizon_days, symbols)
    if not panel:
        raise LeaderboardUnmeasurableError(
            "closes panel is empty — no realized returns can be computed for "
            f"any of {len(set(entry_dates))} cohort date(s). Check the ArcticDB "
            "universe library and MorningArcticAppend."
        )
    return panel


def _realized_from_panel(
    panel: dict[str, dict[str, float]],
    entry_dates: list[str],
    horizon_days: int,
) -> dict[str, dict[str, float]]:
    """``{entry_date: {ticker: forward_return}}`` at ONE horizon, from an
    already-loaded closes panel. Raises ``LeaderboardUnmeasurableError`` when
    the panel structurally cannot serve this horizon (§7.1)."""
    close_dates = sorted(panel)
    _assert_horizon_is_satisfiable(close_dates, entry_dates, horizon_days)

    realized: dict[str, dict[str, float]] = {}
    for entry in sorted(set(entry_dates)):
        h_date = _horizon_date(close_dates, entry, horizon_days)
        if h_date is None:
            continue
        c0 = panel.get(entry) or {}
        ch = panel.get(h_date) or {}
        if not c0 or not ch:
            continue
        fwd: dict[str, float] = {}
        for t, p0 in c0.items():
            ph = ch.get(t)
            if ph is not None and p0 > 0:
                fwd[t] = ph / p0 - 1.0
        if fwd:
            realized[entry] = fwd
    return realized


def _resolve_realized_returns_by_horizon(
    bucket: str,
    entry_dates: list[str],
    horizons_days: list[int],
    *,
    symbols: set[str] | None = None,
    closes_panel_loader: Any = None,
) -> tuple[dict[int, dict[str, dict[str, float]]], dict[int, tuple[str, str]]]:
    """Realized forward returns at EVERY horizon, from ONE closes-panel read
    (alpha-engine-config-I7540).

    Returns ``({horizon_days: realized}, {horizon_days: (status, reason)})`` —
    the second map naming only the horizons that could not be measured at all.

    The panel is loaded ONCE, sized to the LONGEST horizon, and every horizon's
    forward returns are derived from it. Three reads of an ArcticDB slice for
    three horizons would triple the Lambda's dominant cost for data it already
    holds in memory.

    FAILURE ISOLATION, deliberately asymmetric:

    - The PRIMARY horizon (``horizons_days[0]``, 21) raises as it always has.
      An unmeasurable primary is an unmeasurable leaderboard, and the caller
      turns that into the loud ``unmeasurable`` verdict (§7.2). Behaviour here
      is byte-unchanged from before this issue.
    - A LONGER horizon that the source cannot serve is recorded as an
      ``unmeasurable`` BLOCK with its reason and does not sink the artifact.
      A 252-session horizon the panel cannot span is a fact about that horizon;
      failing the whole leaderboard on it would take the working 21-day series
      down with it, which §3 forbids.
    """
    if not entry_dates:
        return {h: {} for h in horizons_days}, {}

    panel = _load_closes_panel(bucket, entry_dates, max(horizons_days), symbols, closes_panel_loader)

    by_horizon: dict[int, dict[str, dict[str, float]]] = {}
    notes: dict[int, tuple[str, str]] = {}
    primary = horizons_days[0]
    for h in horizons_days:
        try:
            by_horizon[h] = _realized_from_panel(panel, entry_dates, h)
        except LeaderboardUnmeasurableError as exc:
            if h == primary:
                raise
            logger.warning(
                "[leaderboard] %sd horizon is unmeasurable against this closes "
                "panel (the %sd primary is unaffected): %s",
                h, primary, exc,
            )
            by_horizon[h] = {}
            notes[h] = (HORIZON_UNMEASURABLE, str(exc))
    return by_horizon, notes


def _annotate_horizon_maturity(
    leaderboard_id: str,
    blocks: list[dict],
    dates: list[str],
    as_of: str,
    primary_horizon: int,
) -> None:
    """Give every zero-scored horizon block an explicit, honest status, in
    place (alpha-engine-config-I7540 + §7.2).

    A block that scored zero cohorts is NEVER left reading ``ok`` with
    ``n_dates: 0`` and a list of null metrics — that is the exact rendering
    that let I5195 run four weeks unnoticed, and at 126/252 sessions it would
    be the normal state for months, so the reader has no way to tell "not yet"
    from "broken" without being told.

    Two states, and the difference is bounded-ness:

    - ``immature`` — no cohort has aged past this horizon yet. Self-resolving,
      expected, and must NOT alert: a 252d block will sit here for the better
      part of a year and alerting on it every cycle manufactures exactly the
      fatigue that hides real findings.
    - ``unmeasurable`` — the oldest cohort IS old enough to have matured and
      still nothing scored. Immaturity is bounded; this is not. Alerts.

    The PRIMARY horizon is skipped here: its overdue escalation replaces the
    WHOLE artifact with an ``unmeasurable`` result in the caller, which is
    pre-existing behaviour this change does not touch.
    """
    for block in blocks:
        h = block["horizon_days"]
        if h == primary_horizon or block["status"] != HORIZON_OK or block["n_dates"]:
            continue
        overdue = _overdue_zero_cohort_reason(dates, h, as_of)
        if overdue is None:
            block["status"] = HORIZON_IMMATURE
            block["reason"] = (
                f"no cohort date has aged {h} trading sessions yet — the "
                f"{h}-session horizon is not yet measurable for any arm. This "
                "is immaturity, not a result: every spec row is reported "
                "confidence=insufficient rather than as a zero."
            )
            continue
        block["status"] = HORIZON_UNMEASURABLE
        block["reason"] = overdue
        logger.error(
            "[leaderboard] %s %sd horizon OVERDUE on %s: %s",
            leaderboard_id, h, as_of, overdue,
        )
        publish_observe_alert(
            message=(
                f"[leaderboard] {leaderboard_id} leaderboard: the {h}-session "
                f"horizon is UNMEASURABLE on {as_of}: {overdue}. The primary "
                f"{primary_horizon}-session block is unaffected; this horizon "
                "is reported status=unmeasurable so the gap is visible rather "
                "than silent (alpha-engine-config-I7540, §7.2)."
            ),
            source=f"research:{leaderboard_id}_leaderboard",
            dedup_key=f"{leaderboard_id}_leaderboard_horizon_overdue:{h}:{as_of}",
        )


# ── Shadow-artifact → SpecHistory loaders ─────────────────────────────────────


def _load_scanner_specs(s3: Any, bucket: str, dates: list[str]) -> tuple[SpecHistory, list[SpecHistory]]:
    """Champion (live ``candidates/{date}/candidates.json``) + every challenger
    (``candidates_shadow/{spec}/{date}/candidates.json``) as SpecHistories. A
    scanner spec exposes a count-matched ranked ticker list (``scanner_tickers``)
    and no per-ticker score — the rank order IS the signal."""
    from data.scanner_specs import SCANNER_SPECS, challenger_specs

    champ_spec = next(s for s in SCANNER_SPECS.values() if s.kind == "champion")
    champion = SpecHistory(name=champ_spec.name, kind="champion")
    for d in dates:
        doc = _get_json(s3, bucket, _CANDIDATES_LIVE.format(date=d))
        if doc and doc.get("scanner_tickers"):
            champion.by_date[d] = SpecDay(ranked=list(doc["scanner_tickers"]))

    challengers: list[SpecHistory] = []
    for spec in challenger_specs():
        hist = SpecHistory(name=spec.name, kind="challenger")
        for d in dates:
            doc = _get_json(s3, bucket, _CANDIDATES_SHADOW.format(spec=spec.name, date=d))
            if doc and doc.get("scanner_tickers"):
                hist.by_date[d] = SpecDay(ranked=list(doc["scanner_tickers"]))
        challengers.append(hist)
    return champion, challengers


def _enter_ranked_and_scores(signals_doc: dict) -> SpecDay:
    """Reduce a signals.json to a SpecDay: ENTER-rated tickers ranked by ``score``
    descending, with the per-ticker score carried for the rank-IC."""
    signals = signals_doc.get("signals") or {}
    rows = [
        (t, float(v["score"]))
        for t, v in signals.items()
        if isinstance(v, dict) and v.get("signal") == "ENTER" and v.get("score") is not None
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return SpecDay(ranked=[t for t, _ in rows], scores=dict(rows))


def _resolve_champion_name(s3: Any, bucket: str) -> str | None:
    """The live champion producer's name, from the SSoT pointer.

    alpha-engine-config-I5195. The champion's IDENTITY lives in
    ``config/producer_champion.json``, written by crucible-backtester's
    promotion engine — that is the artifact the promotion loop updates and the
    one the executor's arm dispatch follows. The research-side
    ``producers/registry.py`` separately carried a ``kind=="champion"`` spec,
    and after ``agentic_sector_teams`` was retired (config-I2993) nothing
    replaced it, so ``champion_producer()`` returned None and every producer
    leaderboard since has reported ``champion: null`` — while the live pointer
    had said ``scanner_predictor_direct`` since 2026-07-13.

    Two registries for one fact is the multi-writer drift class. Reading the
    pointer makes the leaderboard track the live champion by construction; the
    registry stays the source for CHALLENGER specs (which carry ``build``
    callables the pointer knows nothing about).

    Falls back to the registry's champion spec if the pointer is absent or
    malformed, so an S3 hiccup degrades to prior behaviour rather than
    dropping the champion arm entirely.
    """
    try:
        doc = _get_json(s3, bucket, _CHAMPION_POINTER)
    except Exception as exc:  # noqa: BLE001 — fall back, never fail the build
        logger.warning("[leaderboard] champion pointer unreadable (%s) — falling back to registry", exc)
        doc = None

    name = (doc or {}).get("champion")
    if isinstance(name, str) and name:
        return name

    from producers.registry import champion_producer

    spec = champion_producer()
    if spec is not None:
        logger.info("[leaderboard] champion pointer absent — using registry spec %s", spec.name)
        return spec.name
    return None


def _load_producer_specs(
    s3: Any,
    bucket: str,
    dates: list[str],
    *,
    as_of: str | None = None,
) -> tuple[SpecHistory | None, list[SpecHistory]]:
    """Champion (live ``signals/{date}/signals.json``) + every challenger
    (``signals_shadow/{producer}/{date}/signals.json``) + every in-window
    retired arm (same shadow path — config-I6427) as SpecHistories, each
    reduced to its ENTER picks ranked by score.

    The champion's NAME comes from the live pointer (see
    ``_resolve_champion_name``), not from the research registry — the promotion
    loop owns that fact. Champion is ``None`` only when neither the pointer nor
    the registry names one; callers must treat that as an honest "no champion
    to score", not an error.

    ``as_of`` is the cohort/run date (the leaderboard's ``date_str``), passed
    straight through to ``registry.retired_producers()`` to resolve the
    trailing window (champion-challenger-policy.md §3). Retired rows are
    returned in the SAME list as challengers — ``score_leaderboard`` treats
    that parameter as "every non-champion spec to score" and passes each
    ``SpecHistory.kind`` through to the artifact unchanged, so a retired arm
    is tagged ``"kind": "retired"`` there by construction, distinct from a
    live ``"kind": "challenger"`` row (§3: retired arms are scored but never
    promotion-eligible; promotion consumers filter on ``kind=="challenger"``)."""
    from producers.registry import challenger_producers, retired_producers

    champ_name = _resolve_champion_name(s3, bucket)
    champion: SpecHistory | None = None
    if champ_name is not None:
        champion = SpecHistory(name=champ_name, kind="champion")
        for d in dates:
            doc = _get_json(s3, bucket, _SIGNALS_LIVE.format(date=d))
            if doc:
                day = _enter_ranked_and_scores(doc)
                if day.ranked:
                    champion.by_date[d] = day

    challengers: list[SpecHistory] = []
    for spec in challenger_producers():
        hist = SpecHistory(name=spec.name, kind="challenger")
        for d in dates:
            doc = _get_json(s3, bucket, _SIGNALS_SHADOW.format(producer=spec.name, date=d))
            if doc:
                day = _enter_ranked_and_scores(doc)
                if day.ranked:
                    hist.by_date[d] = day
        challengers.append(hist)

    for spec in retired_producers(as_of=as_of):
        hist = SpecHistory(name=spec.name, kind="retired")
        for d in dates:
            doc = _get_json(s3, bucket, _SIGNALS_SHADOW.format(producer=spec.name, date=d))
            if doc:
                day = _enter_ranked_and_scores(doc)
                if day.ranked:
                    hist.by_date[d] = day
        challengers.append(hist)

    return champion, challengers


# ── Cohort-date discovery ─────────────────────────────────────────────────────


def _cohort_dates(s3: Any, bucket: str, prefix: str, depth: int) -> list[str]:
    """The set of cohort dates available under a shadow prefix. ``depth`` = the
    0-based position of the ``{date}`` path segment after ``prefix`` (scanner
    shadow: ``candidates_shadow/{spec}/{date}/...`` → date is segment 1; producer
    shadow: ``signals_shadow/{producer}/{date}/...`` → segment 1)."""
    dates: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            rest = obj["Key"][len(prefix) :].split("/")
            if len(rest) > depth:
                seg = rest[depth]
                if seg:
                    dates.add(seg)
    return sorted(dates)


# ── Public producers (fail-soft) ──────────────────────────────────────────────


def _write_leaderboard(s3: Any, bucket: str, key: str, leaderboard: dict) -> str:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(leaderboard, indent=2).encode(),
        ContentType="application/json",
    )
    logger.info("[leaderboard] wrote s3://%s/%s", bucket, key)
    return key


def _picked_symbols(
    champion: SpecHistory | None,
    challengers: list[SpecHistory],
    benchmark: str = "SPY",
) -> set[str]:
    """Every ticker any arm picked on any cohort date, plus the benchmark.

    The closes read is narrowed to these: an arm's metrics only ever reference
    its own picks, so pulling the full universe is pure cost.
    """
    out: set[str] = {benchmark}
    for hist in [h for h in [champion, *challengers] if h is not None]:
        for day in hist.by_date.values():
            out.update(day.ranked)
    return out


def _resolve_horizons(
    horizons_days: list[int] | None,
    primary_horizon_days: int,
    registered: tuple[int, ...],
) -> list[int]:
    """The horizons to score, PRIMARY FIRST (alpha-engine-config-I7540).

    Defaults to the slot registry's ``horizons_days``
    (champion-challenger-policy.md §10: the slot names its own horizons; this
    module does not carry them as literals). An explicit ``horizons_days``
    argument wins — a backfill scoring one horizon at a time is a legitimate
    caller.

    ``primary_horizon_days`` is always FIRST and always present, whatever the
    caller passed. §3 continuity is not a caller's option to drop: the primary
    horizon is the artifact's top-level block, and a call that omitted it would
    silently rewrite a promoted arm's series shape.
    """
    chosen = list(horizons_days) if horizons_days else list(registered)
    ordered = [primary_horizon_days] + [h for h in chosen if h != primary_horizon_days]
    return ordered


def _overdue_zero_cohort_reason(dates: list[str], horizon_days: int, as_of: str) -> str | None:
    """Reason a zero-scored leaderboard is a DEFECT rather than immaturity.

    Returns ``None`` while zero cohorts scored is still legitimately expected —
    no cohort has aged past the horizon yet. That state resolves itself and
    must NOT alert: it would fire every cycle for weeks on a healthy system,
    which is how alert fatigue gets manufactured.

    Returns a reason once the OLDEST cohort is old enough that it should have
    matured. Immaturity is bounded; this is not. That distinction is the whole
    point — alpha-engine-config-I5195's actual defect was four consecutive
    weeks of ``n_dates: 0`` that nobody noticed, and a bare "immature" reading
    renders that identically to a healthy first week.
    """
    if not dates:
        return (
            "no cohort dates found under the shadow prefix — no arm emitted "
            "shadow signals at all, so there is nothing to score"
        )
    oldest = min(dates)
    try:
        elapsed = count_trading_days(_date.fromisoformat(oldest), _date.fromisoformat(as_of))
    except Exception:  # noqa: BLE001 - calendar unavailable: do not invent a verdict
        # An unverifiable clock must not manufacture an alert (ARCHITECTURE
        # §132: unverified is not false). Stay silent and let the next cycle,
        # with a working calendar, decide.
        logger.warning(
            "[leaderboard] trading-day calendar unavailable — cannot tell "
            "immature from overdue for oldest cohort %s; not alerting",
            oldest,
        )
        return None
    if elapsed < horizon_days:
        return None
    return (
        f"scored 0 cohorts, but the oldest ({oldest}) is {elapsed} trading "
        f"days old against a {horizon_days}-day horizon — it should have "
        "matured. This is no longer immaturity; realized returns are not "
        "resolving for cohorts that ought to be measurable"
    )


def _vacuous_membership_collisions(
    champion: SpecHistory | None,
    challengers: list[SpecHistory],
) -> list[dict]:
    """Cohort dates where a challenger resolved to EXACTLY the champion's
    picked-ticker set — champion-challenger-policy.md §4: "if two arms
    resolve to the same membership, the comparison is worthless while every
    other assertion still passes. Assert that competing arms actually
    differ."

    Before this, a guard existed ONLY as a fixture-based unit test
    (``tests/test_universe_membership.py::
    test_incumbent_arm_disagrees_with_the_champion``) — it asserts static
    test fixtures never collide, and exercises nothing at runtime. Nothing on
    a live cycle ever compared the champion's ACTUAL resolved picks against a
    challenger's (alpha-engine-config#6429).

    Exact-set-equality, deliberately not a fuzzy/near-identical threshold: no
    fuzzy-match convention exists anywhere else in this codebase for a
    membership comparison of this kind, and policy §4's count-matching
    already holds every arm in a slot to the same width — a near-miss under
    count-matching is itself the finding, not noise to smooth over.

    Checked against every cohort date both specs cover, not only the current
    run date: a collision on any historical date is exactly as vacuous as one
    today, and shadow artifacts back-fill.

    Returns one entry per ``(challenger, date)`` collision — empty when every
    arm differs everywhere, which is the expected, healthy state.
    """
    if champion is None:
        return []
    collisions: list[dict] = []
    for ch in challengers:
        for d in sorted(set(champion.by_date) & set(ch.by_date)):
            champ_set = set(champion.by_date[d].ranked)
            ch_set = set(ch.by_date[d].ranked)
            if champ_set and champ_set == ch_set:
                collisions.append({"challenger": ch.name, "date": d, "n_tickers": len(champ_set)})
    return collisions


def _alert_vacuous_collisions(
    leaderboard_id: str,
    champion_name: str | None,
    collisions: list[dict],
) -> None:
    """Alarm on a vacuous champion/challenger comparison (policy §4). Mirrors
    the CC-7.2 ``unmeasurable`` alert shape (below): LOUD but never
    blocking — an alarm/status surface, not an exception on the live path.
    Fires once per build with every colliding arm+date named, rather than
    once per date, so a persistent collision does not spam one alert per
    cohort date."""
    if champion_name is None or not collisions:
        return
    by_challenger: dict[str, list[str]] = {}
    for c in collisions:
        by_challenger.setdefault(c["challenger"], []).append(c["date"])
    detail = "; ".join(
        f"{name} on {len(dates)} date(s) ({', '.join(sorted(dates))})" for name, dates in sorted(by_challenger.items())
    )
    logger.error(
        "[leaderboard] %s VACUOUS comparison: champion %r resolved identical membership to %s",
        leaderboard_id,
        champion_name,
        detail,
    )
    publish_observe_alert(
        message=(
            f"[leaderboard] {leaderboard_id} leaderboard: champion {champion_name!r} "
            f"resolved to IDENTICAL membership as {detail} — the comparison is vacuous "
            "on those cohort date(s) while every other assertion still passes "
            "(champion-challenger-policy.md §4). Verify the challenger spec actually "
            "differs from the champion."
        ),
        source=f"research:{leaderboard_id}_leaderboard",
        dedup_key=f"{leaderboard_id}_leaderboard_vacuous:{champion_name}:{'|'.join(sorted(by_challenger))}",
    )


def _unmeasurable_result(
    s3: Any,
    bucket: str,
    date_str: str,
    *,
    leaderboard_id: str,
    output_tmpl: str,
    horizon_days: int,
    reason: str,
    write: bool,
) -> dict:
    """Write a leaderboard whose verdict is an explicit ``unmeasurable``, and
    alert on it.

    alpha-engine-config-I5195. Before this, an unmeasurable leaderboard and a
    perfectly healthy one awaiting cohort maturity produced byte-comparable
    artifacts (``n_dates: 0``, all metrics null). The artifact is still written
    — downstream consumers depend on it existing — but it now SAYS it could not
    measure, and says why. Silence was the actual defect; fail-soft toward the
    live path is retained.
    """
    leaderboard = {
        "leaderboard_id": leaderboard_id,
        "date": date_str,
        "status": "unmeasurable",
        "unmeasurable_reason": reason,
        "horizon_days": horizon_days,
        "n_dates": 0,
        "specs": [],
    }
    key = _write_leaderboard(s3, bucket, output_tmpl.format(date=date_str), leaderboard) if write else None
    logger.error("[leaderboard] %s UNMEASURABLE on %s: %s", leaderboard_id, date_str, reason)
    publish_observe_alert(
        message=(
            f"[leaderboard] {leaderboard_id} leaderboard is UNMEASURABLE on {date_str}: "
            f"{reason} No champion/challenger arm can be scored until this is fixed — "
            "the artifact is written with status=unmeasurable so the gap is visible "
            "rather than silent (alpha-engine-config-I5195)."
        ),
        source=f"research:{leaderboard_id}_leaderboard",
        dedup_key=f"{leaderboard_id}_leaderboard_unmeasurable:{date_str}",
    )
    return {"status": "unmeasurable", "key": key, "leaderboard": leaderboard}


def build_scanner_leaderboard(
    s3: Any,
    bucket: str,
    date_str: str,
    *,
    top_n: int = 50,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    horizons_days: list[int] | None = None,
    write: bool = True,
    closes_panel_loader: Any = None,
) -> dict:
    """Score the scanner champion vs every challenger spec over all available
    cohort dates, write ``scanner/leaderboard/{date}.json`` (``date_str`` = the
    run date keying the output), return ``{"status", "key"?, "leaderboard"?}``.

    ``horizons_days`` defaults to the scanner slot's registered horizons
    (``LEADERBOARD_SLOTS["scanner"]``, currently 21/126/252 sessions —
    alpha-engine-config-I7540). ``horizon_days`` remains the PRIMARY horizon
    and stays the artifact's top-level block, so the existing 21-day series is
    continuous across this change.

    OBSERVE-ONLY + FAIL-SOFT: any failure logs + returns ``{"status": "error"}``;
    it NEVER raises into the caller (the scanner Lambda's live path)."""
    try:
        slot = slot_spec("scanner")
        horizons = _resolve_horizons(horizons_days, horizon_days, slot.horizons_days)
        dates = _cohort_dates(s3, bucket, "candidates_shadow/", depth=1)
        champion, challengers = _load_scanner_specs(s3, bucket, dates)
        vacuous = _vacuous_membership_collisions(champion, challengers)
        _alert_vacuous_collisions("scanner", champion.name if champion else None, vacuous)
        try:
            realized_by_horizon, horizon_notes = _resolve_realized_returns_by_horizon(
                bucket,
                dates,
                horizons,
                symbols=_picked_symbols(champion, challengers),
                closes_panel_loader=closes_panel_loader,
            )
        except LeaderboardUnmeasurableError as exc:
            return _unmeasurable_result(
                s3,
                bucket,
                date_str,
                leaderboard_id="scanner",
                output_tmpl=_SCANNER_OUTPUT,
                horizon_days=horizon_days,
                reason=str(exc),
                write=write,
            )
        leaderboard = score_multi_horizon(
            champion,
            challengers,
            realized_by_horizon,
            top_n=top_n,
            horizons_days=horizons,
            min_dates_for_inference=slot.min_dates_for_inference,
            horizon_notes=horizon_notes,
        )
        _annotate_horizon_maturity("scanner", leaderboard["horizons"], dates, date_str, horizons[0])
        leaderboard["leaderboard_id"] = "scanner"
        leaderboard["date"] = date_str
        leaderboard["vacuous_membership_collisions"] = vacuous
        # config-I5195: zero scored cohorts is EXPECTED while cohorts are
        # immature and a DEFECT once they should have matured. The two render
        # identically as n_dates=0, which is how the original defect survived
        # four weeks unnoticed. Immaturity stays status="ok" (self-resolving,
        # must not alert); overdue escalates to unmeasurable.
        overdue = _overdue_zero_cohort_reason(dates, horizon_days, date_str) if not leaderboard.get("n_dates") else None
        if overdue:
            return _unmeasurable_result(
                s3,
                bucket,
                date_str,
                leaderboard_id="scanner",
                output_tmpl=_SCANNER_OUTPUT,
                horizon_days=horizon_days,
                reason=overdue,
                write=write,
            )
        key = _write_leaderboard(s3, bucket, _SCANNER_OUTPUT.format(date=date_str), leaderboard) if write else None
        return {"status": "ok", "key": key, "leaderboard": leaderboard}
    except Exception as exc:  # noqa: BLE001 — observe-only, must never raise into live path
        logger.warning(
            "[leaderboard] scanner leaderboard build failed (non-fatal, observe-only): %s",
            exc,
        )
        # Fail-LOUD: scanner/leaderboard/ is OBSERVATION_REGISTRY always-on; a
        # build failure means the artifact is NOT written to S3 — the silent gap
        # the 2026-06-27 audit caught (config#1403). Surface it, never swallow.
        publish_observe_alert(
            message=(
                f"[leaderboard] scanner leaderboard build FAILED on {date_str} "
                f"(observe-only, non-fatal): {exc}. "
                f"{_SCANNER_OUTPUT.format(date=date_str)} NOT written to S3 — "
                f"no-silent-fails (config#1403)."
            ),
            source="research:scanner_leaderboard",
            dedup_key=f"scanner_leaderboard_build_error:{date_str}",
        )
        return {"status": "error", "error": str(exc)}


def build_producer_leaderboard(
    s3: Any,
    bucket: str,
    date_str: str,
    *,
    top_n: int = 50,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    horizons_days: list[int] | None = None,
    write: bool = True,
    closes_panel_loader: Any = None,
) -> dict:
    """Score the research producer champion vs every challenger producer over all
    available cohort dates, write ``research/producer_leaderboard/{date}.json``,
    return ``{"status", "key"?, "leaderboard"?}``.

    ``horizons_days`` defaults to the producer slot's registered horizons
    (``LEADERBOARD_SLOTS["producer"]``, currently 21/126/252 sessions —
    alpha-engine-config-I7540). ``horizon_days`` remains the PRIMARY horizon
    and the artifact's top-level block, so the existing 21-day series that
    ``crucible-backtester``'s champion-promotion gate reads is continuous
    across this change (champion-challenger-policy.md §3).

    OBSERVE-ONLY + FAIL-SOFT: never raises into the caller (the research Lambda)."""
    try:
        slot = slot_spec("producer")
        horizons = _resolve_horizons(horizons_days, horizon_days, slot.horizons_days)
        dates = _cohort_dates(s3, bucket, "signals_shadow/", depth=1)
        champion, challengers = _load_producer_specs(s3, bucket, dates, as_of=date_str)
        # Vacuity guard compares champion against LIVE challengers only — a
        # retired-but-in-window arm (config-I6427) is historical evidence,
        # never a promotion-eligible competitor, so its membership colliding
        # with the champion's is not the condition policy §4 warns about.
        live_challengers = [c for c in challengers if c.kind == "challenger"]
        vacuous = _vacuous_membership_collisions(champion, live_challengers)
        _alert_vacuous_collisions("producer", champion.name if champion else None, vacuous)
        if champion is None:
            # No kind=="champion" producer registered (config-I2993:
            # agentic_sector_teams retired 2026-07-12, no successor champion
            # spec registered yet). This is an honest, expected state post-
            # retirement, not a failure — WARN and PROCEED (alpha-engine
            # -config-I2998): score_leaderboard degrades gracefully to
            # champion-free metrics (realized_rank_ic, topn_alpha_vs_benchmark)
            # for every challenger rather than refusing to write the artifact
            # at all. A live champion_promotion.py consumer (crucible
            # -backtester) needs THIS leaderboard to keep existing even with
            # no registered champion — silently writing nothing here was
            # exactly the "both arms simultaneously no-contest" defect
            # I2998 was filed to close.
            logger.warning(
                '[leaderboard] no producer registered kind=="champion" in '
                "RESEARCH_PRODUCERS — scoring producer leaderboard for %s "
                "with champion-free metrics only (config-I2993/I2998).",
                date_str,
            )
        try:
            realized_by_horizon, horizon_notes = _resolve_realized_returns_by_horizon(
                bucket,
                dates,
                horizons,
                symbols=_picked_symbols(champion, challengers),
                closes_panel_loader=closes_panel_loader,
            )
        except LeaderboardUnmeasurableError as exc:
            return _unmeasurable_result(
                s3,
                bucket,
                date_str,
                leaderboard_id="producer",
                output_tmpl=_PRODUCER_OUTPUT,
                horizon_days=horizon_days,
                reason=str(exc),
                write=write,
            )
        leaderboard = score_multi_horizon(
            champion,
            challengers,
            realized_by_horizon,
            top_n=top_n,
            horizons_days=horizons,
            min_dates_for_inference=slot.min_dates_for_inference,
            horizon_notes=horizon_notes,
        )
        _annotate_horizon_maturity("producer", leaderboard["horizons"], dates, date_str, horizons[0])
        leaderboard["leaderboard_id"] = "producer"
        leaderboard["date"] = date_str
        leaderboard["vacuous_membership_collisions"] = vacuous
        # config-I5195: zero scored cohorts is EXPECTED while cohorts are
        # immature and a DEFECT once they should have matured. The two render
        # identically as n_dates=0, which is how the original defect survived
        # four weeks unnoticed. Immaturity stays status="ok" (self-resolving,
        # must not alert); overdue escalates to unmeasurable.
        overdue = _overdue_zero_cohort_reason(dates, horizon_days, date_str) if not leaderboard.get("n_dates") else None
        if overdue:
            return _unmeasurable_result(
                s3,
                bucket,
                date_str,
                leaderboard_id="producer",
                output_tmpl=_PRODUCER_OUTPUT,
                horizon_days=horizon_days,
                reason=overdue,
                write=write,
            )
        key = _write_leaderboard(s3, bucket, _PRODUCER_OUTPUT.format(date=date_str), leaderboard) if write else None
        return {"status": "ok", "key": key, "leaderboard": leaderboard}
    except Exception as exc:  # noqa: BLE001 — observe-only, must never raise into live path
        logger.warning(
            "[leaderboard] producer leaderboard build failed (non-fatal, observe-only): %s",
            exc,
        )
        # Fail-LOUD: research/producer_leaderboard/ is OBSERVATION_REGISTRY
        # always-on; a build failure means the artifact is NOT written to S3 —
        # the silent gap the 2026-06-27 audit caught (config#1403).
        publish_observe_alert(
            message=(
                f"[leaderboard] producer leaderboard build FAILED on {date_str} "
                f"(observe-only, non-fatal): {exc}. "
                f"{_PRODUCER_OUTPUT.format(date=date_str)} NOT written to S3 — "
                f"no-silent-fails (config#1403)."
            ),
            source="research:producer_leaderboard",
            dedup_key=f"producer_leaderboard_build_error:{date_str}",
        )
        return {"status": "error", "error": str(exc)}
