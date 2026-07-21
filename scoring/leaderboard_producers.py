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
days, read from alpha-engine-data's date-keyed ``staging/daily_closes/{date}.parquet``
(same source ``feature_store_reader`` uses). A date with no matured horizon close
simply does not join — its metrics stay an honest ``None`` until the cohort
matures. The horizon-end calendar date is resolved by listing the available
daily_closes dates and taking the H-th trading date on/after ``d+1`` (trading
days only — weekends/holidays are absent from the parquet listing).
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

from observe_alerts import publish_observe_alert
from scoring.leaderboard_scoring import (
    DEFAULT_HORIZON_DAYS,
    SpecDay,
    SpecHistory,
    score_leaderboard,
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "alpha-engine-research"
_CLOSES_PREFIX = "staging/daily_closes/"

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


def _list_close_dates(s3: Any, bucket: str) -> list[str]:
    """Sorted list of available ``staging/daily_closes/{date}.parquet`` dates
    (trading days only — the parquet exists per trading day)."""
    dates: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_CLOSES_PREFIX):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith(".parquet"):
                stem = key[len(_CLOSES_PREFIX):-len(".parquet")]
                if stem:
                    dates.append(stem)
    return sorted(dates)


def _read_closes(s3: Any, bucket: str, date_str: str) -> dict[str, float]:
    """``{ticker: close}`` for one date's daily_closes parquet (empty on any
    miss/parse error — best-effort)."""
    import pandas as pd

    key = f"{_CLOSES_PREFIX}{date_str}.parquet"
    doc = None
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        doc = obj["Body"].read()
    except Exception:  # noqa: BLE001 — absent date = no join, not an error
        return {}
    try:
        df = pd.read_parquet(io.BytesIO(doc), engine="pyarrow")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[leaderboard] daily_closes parse failed for %s: %s", date_str, exc)
        return {}
    if df.empty:
        return {}
    close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if close_col is None:
        return {}
    if "ticker" not in df.columns and df.index.name == "ticker":
        df = df.reset_index()
    if "ticker" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        t, c = row["ticker"], row[close_col]
        if t and pd.notna(c) and c > 0:
            out[str(t)] = float(c)
    return out


def _horizon_date(close_dates: list[str], entry_date: str, horizon_days: int) -> str | None:
    """The trading date ``horizon_days`` sessions AFTER ``entry_date`` in the
    available daily_closes calendar, or None if the horizon hasn't matured."""
    after = [d for d in close_dates if d > entry_date]
    if len(after) < horizon_days:
        return None
    return after[horizon_days - 1]


def _resolve_realized_returns(
    s3: Any,
    bucket: str,
    entry_dates: list[str],
    horizon_days: int,
) -> dict[str, dict[str, float]]:
    """``{entry_date: {ticker: forward_return}}`` for every entry date whose
    horizon close has matured. Reads each entry-date and its horizon-date closes
    from daily_closes and computes ``close_h / close_0 − 1``. Best-effort: a date
    with no matured horizon (or a missing parquet) simply does not appear."""
    close_dates = _list_close_dates(s3, bucket)
    if not close_dates:
        return {}
    realized: dict[str, dict[str, float]] = {}
    closes_cache: dict[str, dict[str, float]] = {}

    def _closes(d: str) -> dict[str, float]:
        if d not in closes_cache:
            closes_cache[d] = _read_closes(s3, bucket, d)
        return closes_cache[d]

    for entry in sorted(set(entry_dates)):
        h_date = _horizon_date(close_dates, entry, horizon_days)
        if h_date is None:
            continue
        c0 = _closes(entry)
        ch = _closes(h_date)
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
    s3: Any, bucket: str, dates: list[str],
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
            rest = obj["Key"][len(prefix):].split("/")
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


def build_scanner_leaderboard(
    s3: Any,
    bucket: str,
    date_str: str,
    *,
    top_n: int = 50,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    write: bool = True,
) -> dict:
    """Score the scanner champion vs every challenger spec over all available
    cohort dates, write ``scanner/leaderboard/{date}.json`` (``date_str`` = the
    run date keying the output), return ``{"status", "key"?, "leaderboard"?}``.

    OBSERVE-ONLY + FAIL-SOFT: any failure logs + returns ``{"status": "error"}``;
    it NEVER raises into the caller (the scanner Lambda's live path)."""
    try:
        dates = _cohort_dates(s3, bucket, "candidates_shadow/", depth=1)
        champion, challengers = _load_scanner_specs(s3, bucket, dates)
        realized = _resolve_realized_returns(s3, bucket, dates, horizon_days)
        leaderboard = score_leaderboard(
            champion, challengers, realized, top_n=top_n, horizon_days=horizon_days,
        )
        leaderboard["leaderboard_id"] = "scanner"
        leaderboard["date"] = date_str
        key = (
            _write_leaderboard(s3, bucket, _SCANNER_OUTPUT.format(date=date_str), leaderboard)
            if write else None
        )
        return {"status": "ok", "key": key, "leaderboard": leaderboard}
    except Exception as exc:  # noqa: BLE001 — observe-only, must never raise into live path
        logger.warning(
            "[leaderboard] scanner leaderboard build failed (non-fatal, observe-only): %s", exc,
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
            # retirement, not a failure — WARN and return a distinguishable
            # status rather than raising or fabricating a champion row.
            logger.warning(
                "[leaderboard] no producer registered kind==\"champion\" in "
                "RESEARCH_PRODUCERS — skipping producer leaderboard for %s "
                "(config-I2993).", date_str,
            )
            return {"status": "no_champion_registered"}
        realized = _resolve_realized_returns(s3, bucket, dates, horizon_days)
        leaderboard = score_leaderboard(
            champion, challengers, realized, top_n=top_n, horizon_days=horizon_days,
        )
        leaderboard["leaderboard_id"] = "producer"
        leaderboard["date"] = date_str
        key = (
            _write_leaderboard(s3, bucket, _PRODUCER_OUTPUT.format(date=date_str), leaderboard)
            if write else None
        )
        return {"status": "ok", "key": key, "leaderboard": leaderboard}
    except Exception as exc:  # noqa: BLE001 — observe-only, must never raise into live path
        logger.warning(
            "[leaderboard] producer leaderboard build failed (non-fatal, observe-only): %s", exc,
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
