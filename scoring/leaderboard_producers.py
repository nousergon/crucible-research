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
"""

from __future__ import annotations

import json
import logging
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
from observe_alerts import publish_observe_alert
from scoring.leaderboard_scoring import (
    DEFAULT_HORIZON_DAYS,
    SpecDay,
    SpecHistory,
    score_leaderboard,
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
    # LAST entry date is inside the window. Trading days are ~0.7 of calendar
    # days; 2x plus a fixed pad is comfortably conservative and costs only a
    # slightly wider read.
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
    # Injectable so the source is a parameter, not a hard binding: tests supply
    # a panel directly, and a backfill can point at an alternative store
    # without editing this module. Production default is the durable ArcticDB
    # read.
    loader = closes_panel_loader or _closes_panel_from_arcticdb
    panel = loader(bucket, entry_dates, horizon_days, symbols)
    if not panel:
        raise LeaderboardUnmeasurableError(
            "closes panel is empty — no realized returns can be computed for "
            f"any of {len(set(entry_dates))} cohort date(s). Check the ArcticDB "
            "universe library and MorningArcticAppend."
        )

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


def _load_producer_specs(
    s3: Any,
    bucket: str,
    dates: list[str],
) -> tuple[SpecHistory | None, list[SpecHistory]]:
    """Champion (live ``signals/{date}/signals.json``) + every challenger
    (``signals_shadow/{producer}/{date}/signals.json``) as SpecHistories, each
    reduced to its ENTER picks ranked by score.

    Champion is ``None`` when no ``kind=="champion"`` producer is currently
    registered (config-I2993: ``agentic_sector_teams`` retired 2026-07-12,
    no successor champion spec registered yet — that registration is tracked
    separately). Callers must treat ``None`` as an honest "no champion to
    score", not an error."""
    from producers.registry import challenger_producers, champion_producer

    champ_spec = champion_producer()
    champion: SpecHistory | None = None
    if champ_spec is not None:
        champion = SpecHistory(name=champ_spec.name, kind="champion")
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
    write: bool = True,
    closes_panel_loader: Any = None,
) -> dict:
    """Score the scanner champion vs every challenger spec over all available
    cohort dates, write ``scanner/leaderboard/{date}.json`` (``date_str`` = the
    run date keying the output), return ``{"status", "key"?, "leaderboard"?}``.

    OBSERVE-ONLY + FAIL-SOFT: any failure logs + returns ``{"status": "error"}``;
    it NEVER raises into the caller (the scanner Lambda's live path)."""
    try:
        dates = _cohort_dates(s3, bucket, "candidates_shadow/", depth=1)
        champion, challengers = _load_scanner_specs(s3, bucket, dates)
        try:
            realized = _resolve_realized_returns(
                s3,
                bucket,
                dates,
                horizon_days,
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
        leaderboard = score_leaderboard(
            champion,
            challengers,
            realized,
            top_n=top_n,
            horizon_days=horizon_days,
        )
        leaderboard["leaderboard_id"] = "scanner"
        leaderboard["date"] = date_str
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
    write: bool = True,
    closes_panel_loader: Any = None,
) -> dict:
    """Score the research producer champion vs every challenger producer over all
    available cohort dates, write ``research/producer_leaderboard/{date}.json``,
    return ``{"status", "key"?, "leaderboard"?}``.

    OBSERVE-ONLY + FAIL-SOFT: never raises into the caller (the research Lambda)."""
    try:
        dates = _cohort_dates(s3, bucket, "signals_shadow/", depth=1)
        champion, challengers = _load_producer_specs(s3, bucket, dates)
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
            realized = _resolve_realized_returns(
                s3,
                bucket,
                dates,
                horizon_days,
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
        leaderboard = score_leaderboard(
            champion,
            challengers,
            realized,
            top_n=top_n,
            horizon_days=horizon_days,
        )
        leaderboard["leaderboard_id"] = "producer"
        leaderboard["date"] = date_str
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
