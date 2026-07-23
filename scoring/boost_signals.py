"""Boost-signal emission for the research_optimizer measurement loop (config#1857).

The backtester's ``optimizer/research_optimizer.py::compute_boost_correlations``
reads two per-ticker fields from every ``universe`` / ``buy_candidates`` entry in
signals.json — ``short_interest_adj`` and ``institutional_boost`` — and
correlates them against the primary-horizon beat-SPY outcome to auto-tune
``config/research_params.json``. Until this module shipped, crucible-research
never emitted those fields, so ``compute_boost_correlations`` returned
``no_boost_data`` on every run, ``config/research_params_history/`` stayed empty,
and the optimizer was **structurally starved at birth** (config#1857). The
sample gate (``research_optimizer._MIN_SAMPLES = 200``) has long been satisfied
(376+ resolved samples); the only missing input was the emitted boost data this
module now produces.

It computes the *applied-boost magnitude* per ticker from the two data sources
the fleet already collects, using the live ``research_params`` thresholds:

  * ``short_interest_adj`` — from nousergon-data's weekly
    ``market_data/weekly/{date}/short_interest.json`` (``short_pct_float``),
    thresholded by ``short_interest_{buy,high}_threshold_pct`` into the
    corresponding ``short_interest_{buy,high}_boost``.
  * ``institutional_boost`` — from the 13F accumulation signal
    (``data.fetchers.institutional_fetcher.fetch_institutional_accumulation``,
    or a future cached ``institutional_accumulation.json`` weekly artifact),
    gated by ``institutional_min_funds`` into ``institutional_boost``.

DELIBERATELY EMIT-ONLY — this is the measurement layer, not a scoring change.
------------------------------------------------------------------------------
These fields are the *input* the optimizer correlates against outcomes; they are
NOT folded into the live attractiveness blend or ``final_score`` (universe_board
schema_version 3; ARCHITECTURE §43 keeps independent signal/cost scores out of
the ranking blend). Whether a *proven* boost should later rerank live candidates
is a separate, optimizer-informed decision that must wait on the correlation
evidence this module generates — it is NOT smuggled in here, and the live
buy_candidate ranking is byte-for-byte unchanged by this emission. ``nonzero`` in
an emitted field means "this ticker cleared the threshold and earns this boost",
exactly the ``stock.get(col, 0.0)`` / nonzero==active semantics
``compute_boost_correlations`` expects. The ``max_aggregate_boost`` cap is a
live-integration concern (summing boosts into a score) and is intentionally NOT
applied to the independent emitted columns.

Rollout is gated behind ``RESEARCH_BOOST_SIGNALS_ENABLED=true`` (mirrors the
``INSTITUTIONAL_SUBSTRATE_ENABLED`` producer-reader precedent) so the S3 read is
a deliberate, reversible production enablement and unit/offline runs never touch
the network.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

from alpha_engine_lib.secrets import get_secret

logger = logging.getLogger(__name__)

_MARKET_DATA_PREFIX = "market_data/"
_ENABLE_FLAG = "RESEARCH_BOOST_SIGNALS_ENABLED"


def emit_enabled() -> bool:
    """True when boost-signal emission is switched on for this run."""
    return os.environ.get(_ENABLE_FLAG, "").lower() == "true"


# ── Pure boost math (no I/O — the unit-testable core) ────────────────────────

def short_interest_adj_for(short_pct_float: float | None, params: dict) -> float:
    """Applied short-interest boost for one ticker.

    ``short_pct_float`` is a percent (e.g. 5.0 == 5% of float short), matching
    nousergon-data's ``collectors/short_interest.py`` output. Returns 0.0 when
    the value is missing/unparseable or below the buy threshold.
    """
    if short_pct_float is None:
        return 0.0
    try:
        si = float(short_pct_float)
    except (TypeError, ValueError):
        return 0.0
    high_thr = float(params.get("short_interest_high_threshold_pct", 40))
    buy_thr = float(params.get("short_interest_buy_threshold_pct", 20))
    if si >= high_thr:
        return float(params.get("short_interest_high_boost", 4.0))
    if si >= buy_thr:
        return float(params.get("short_interest_buy_boost", 2.0))
    return 0.0


def institutional_boost_for(inst: dict | None, params: dict) -> float:
    """Applied institutional-accumulation boost for one ticker.

    ``inst`` is one ``fetch_institutional_accumulation`` record
    (``n_funds_accumulating`` / ``accumulation_signal``). The ``min_funds``
    threshold is re-derived from ``n_funds_accumulating`` when present so a
    ``institutional_min_funds`` param sweep is honored without re-fetching;
    falls back to the fetcher's precomputed ``accumulation_signal`` otherwise.
    Returns 0.0 when there is no accumulation signal.
    """
    if not inst:
        return 0.0
    min_funds = int(params.get("institutional_min_funds", 3))
    n = inst.get("n_funds_accumulating")
    if n is not None:
        try:
            active = int(n) >= min_funds
        except (TypeError, ValueError):
            active = bool(inst.get("accumulation_signal"))
    else:
        active = bool(inst.get("accumulation_signal"))
    return float(params.get("institutional_boost", 3.0)) if active else 0.0


def annotate_boost_signals(
    entries: list[dict],
    *,
    short_interest_map: dict,
    institutional_map: dict,
    params: dict,
) -> None:
    """Attach ``short_interest_adj`` + ``institutional_boost`` to each entry (in place)."""
    for entry in entries:
        ticker = entry.get("ticker") or entry.get("symbol")
        si_rec = short_interest_map.get(ticker) if ticker else None
        si_pct = si_rec.get("short_pct_float") if isinstance(si_rec, dict) else None
        entry["short_interest_adj"] = short_interest_adj_for(si_pct, params)
        entry["institutional_boost"] = institutional_boost_for(
            institutional_map.get(ticker) if ticker else None, params
        )


# ── S3 / fetcher-backed source readers (I/O — production path) ───────────────

def _s3():
    import boto3
    return boto3.client("s3")


def _latest_weekly_key(s3_client, bucket: str, run_date: str, filename: str) -> str | None:
    """Newest ``market_data/weekly/{date}/{filename}`` key with date <= run_date.

    Short-interest / market-data artifacts are keyed by their own weekly
    collection date, which may lag the research ``run_date``; pick the most
    recent one at or before it (never a future artifact — PIT-safe).
    """
    prefix = f"{_MARKET_DATA_PREFIX}weekly/"
    target = (run_date or "")[:10]
    paginator = s3_client.get_paginator("list_objects_v2")
    best: tuple[str, str] | None = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key[len(prefix):].split("/")
            if len(parts) != 2 or parts[1] != filename:
                continue
            d = parts[0]
            if target and d > target:
                continue
            if best is None or d > best[0]:
                best = (d, key)
    return best[1] if best else None


def read_short_interest_map(
    run_date: str, *, s3_client=None, bucket: str | None = None
) -> dict[str, dict]:
    """Per-ticker short-interest records for the newest weekly artifact <= run_date.

    Reader-gated (config#1857): any read failure yields an empty map (every
    ticker then emits ``short_interest_adj=0.0``) rather than breaking the
    signals write.
    """
    bucket = bucket or os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    s3_client = s3_client or _s3()
    try:
        key = _latest_weekly_key(s3_client, bucket, run_date, "short_interest.json")
        if not key:
            logger.info(
                "[boost_signals] no short_interest.json under s3://%s/%sweekly/ <= %s",
                bucket, _MARKET_DATA_PREFIX, run_date,
            )
            return {}
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        out = dict(data.get("data") or {})
        logger.info("[boost_signals] short_interest: %d tickers from s3://%s/%s", len(out), bucket, key)
        return out
    except Exception as e:  # noqa: BLE001 — reader-gate, never break signals write
        logger.warning("[boost_signals] short_interest read failed (emitting 0.0): %s", e)
        return {}


def read_institutional_map(
    tickers: list[str],
    run_date: str,
    *,
    params: dict,
    s3_client=None,
    bucket: str | None = None,
    fetcher: Callable[..., dict] | None = None,
) -> dict[str, dict]:
    """Per-ticker 13F accumulation records.

    Resolution order (each stage reader-gated to empty on failure):
      1. cached weekly ``institutional_accumulation.json`` artifact (same
         producer pattern as short_interest — used automatically if a
         nousergon-data collector later emits it, keeping the expensive 13F
         fetch off the research critical path);
      2. live ``fetch_institutional_accumulation`` over the (small,
         already-screened) universe, only when ``EDGAR_IDENTITY`` is set;
      3. empty map → ``institutional_boost=0.0`` for all.
    """
    bucket = bucket or os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    s3_client = s3_client or _s3()
    # 1. cached artifact (cheap, PIT-safe, preferred)
    try:
        key = _latest_weekly_key(s3_client, bucket, run_date, "institutional_accumulation.json")
        if key:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            recs = dict(json.loads(obj["Body"].read()).get("data") or {})
            if recs:
                logger.info(
                    "[boost_signals] institutional_accumulation: %d tickers from s3://%s/%s",
                    len(recs), bucket, key,
                )
                return recs
    except Exception as e:  # noqa: BLE001
        logger.warning("[boost_signals] institutional artifact read failed: %s", e)
    # 2. live 13F fetch for the screened universe
    if fetcher is None:
        if not get_secret("EDGAR_IDENTITY", required=False):
            logger.info(
                "[boost_signals] no institutional artifact and EDGAR_IDENTITY unset "
                "— institutional_boost=0.0 this run"
            )
            return {}
        try:
            from data.fetchers.institutional_fetcher import (
                fetch_institutional_accumulation as fetcher,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[boost_signals] institutional fetcher import failed: %s", e)
            return {}
    try:
        min_funds = int(params.get("institutional_min_funds", 3))
        return dict(fetcher(list(tickers), min_funds_for_signal=min_funds) or {})
    except Exception as e:  # noqa: BLE001
        logger.warning("[boost_signals] live institutional fetch failed (emitting 0.0): %s", e)
        return {}


# ── Orchestrator called from the archive_writer node ─────────────────────────

def emit_boost_signals(
    signals_payload: dict,
    *,
    run_date: str,
    params: dict | None = None,
    s3_client=None,
    bucket: str | None = None,
    force: bool = False,
) -> None:
    """Attach the two boost fields to the payload's ``universe`` + ``buy_candidates``.

    Idempotent and never raises: any failure leaves the two fields present and
    0.0 so the backtester's fixed column read stays stable. No-op (0.0 defaults
    only) unless ``RESEARCH_BOOST_SIGNALS_ENABLED=true`` or ``force=True``
    (tests pass injected fakes + ``force``), so unit/offline runs never touch S3.
    """
    universe = signals_payload.get("universe") or []
    buy_candidates = signals_payload.get("buy_candidates") or []

    def _default_zero() -> None:
        for entries in (universe, buy_candidates):
            for entry in entries:
                entry.setdefault("short_interest_adj", 0.0)
                entry.setdefault("institutional_boost", 0.0)

    if not (force or emit_enabled()):
        _default_zero()
        return

    try:
        if params is None:
            from config import get_research_params
            params = get_research_params()
        tickers = sorted(
            {(e.get("ticker") or e.get("symbol")) for e in universe if (e.get("ticker") or e.get("symbol"))}
        )
        si_map = read_short_interest_map(run_date, s3_client=s3_client, bucket=bucket)
        inst_map = read_institutional_map(
            tickers, run_date, params=params, s3_client=s3_client, bucket=bucket
        )
        annotate_boost_signals(
            universe, short_interest_map=si_map, institutional_map=inst_map, params=params
        )
        annotate_boost_signals(
            buy_candidates, short_interest_map=si_map, institutional_map=inst_map, params=params
        )
        n_si = sum(1 for e in universe if e.get("short_interest_adj"))
        n_inst = sum(1 for e in universe if e.get("institutional_boost"))
        logger.info(
            "[boost_signals] emitted for %d universe tickers (short_interest nonzero=%d, institutional nonzero=%d)",
            len(universe), n_si, n_inst,
        )
    except Exception as e:  # noqa: BLE001 — emission must never break the signals write
        logger.warning("[boost_signals] emission skipped (signals still valid, boosts=0.0): %s", e)
        _default_zero()
