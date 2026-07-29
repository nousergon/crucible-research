"""
Feature store reader — reads pre-computed technical features + Barra factor
loadings for the scanner, providing richer indicators (MA200, 52-week
high/low) than the 3-month local price history can compute.

Two sources, selected by ``SCANNER_FEATURE_SOURCE``:

``arcticdb`` (DEFAULT)
    The ArcticDB ``universe`` library, rewritten EVERY TRADING DAY by
    alpha-engine-data's ``builders/daily_append.py`` (weekday MorningEnrich).

``s3``
    The dated ``features/{date}/`` snapshot, written ONLY by
    ``features/compute.py::compute_and_write`` — whose sole caller is
    ``weekly_collector.py``, i.e. Saturday DataPhase1. Retained as an
    explicit rollback lever, not as a silent fallback.

Why the default moved to ArcticDB (alpha-engine-config-I4983 follow-on):
on the S3 source a scanner run on any weekday recomputes ``tech_score``
from Saturday's frozen snapshot, so the incumbent ``scanner_top_20`` cut is
bit-identical Monday through Friday while ``current_price`` (read
separately from ``staging/daily_closes/``) keeps moving — the score pairs a
live close with an indicator up to 5 trading days stale.

This reader is NOT the only weekly chokepoint on the scanner's path. The
``attractiveness_top_20`` CHAMPION cut ranks on factor PROFILES, which
``scoring/factor_scoring.py::compute_and_write_factor_profiles`` computes
in-run from ``features/{run_date}/{technical,fundamental}.parquet`` by
EXACT dated key with no fallback — a weekday ``run_date`` simply misses.
Fixing this reader alone therefore makes the incumbent arm daily-fresh but
leaves the champion arm unbuildable off-Saturday. See the issue above.

The two surfaces are not different computations: ``compute_and_write`` and
``daily_append`` both call ``features.feature_engineer.compute_features``
and both apply ``features.cross_sectional.apply_factor_zscores``. The S3
snapshot is a weekly PROJECTION of the same per-ticker computation that
ArcticDB already carries daily, so this is a freshness change and not a
methodology change.

Staleness is counted in NYSE TRADING days and fails LOUD
(``FeatureStoreStalenessError``) — never silently degraded. See
``feedback_no_silent_fails`` and
``feedback_staleness_windows_trading_days_not_calendar_260717``.
"""

from __future__ import annotations

import io
import logging
import os

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("S3_BUCKET", "alpha-engine-research")
_FEATURE_PREFIX = "features/"

# Source selector. ArcticDB is the default; ``s3`` reverts to the weekly
# snapshot without a redeploy if the daily path misbehaves. Deliberately a
# real rollback lever rather than a dormant opt-in flag that never gets
# flipped.
_SOURCE = os.environ.get("SCANNER_FEATURE_SOURCE", "arcticdb").strip().lower()

# The 12 technical columns the scanner actually consumes
# (``data/scanner_orchestrator.py::_build_technical_scores_from_feature_store``
# builds its ``indicators`` dict from exactly these; ``signal_line_last``,
# ``ma50`` and ``ma200`` are hardcoded there and ``current_price`` comes from
# ``read_latest_daily_closes``). Kept as an explicit tuple so an ArcticDB
# read projects only what is needed — a full-row read over ~900 symbols is
# the difference between a comfortable and a timed-out scanner Lambda.
_TECHNICAL_COLS: tuple[str, ...] = (
    "rsi_14",
    "macd_cross",
    "macd_above_zero",
    "macd_line_last",
    "price_vs_ma50",
    "price_vs_ma200",
    "momentum_20d",
    "momentum_5d",
    "avg_volume_20d_raw",
    "atr_14_pct",
    "dist_from_52w_high",
    "dist_from_52w_low",
)

# Trailing window to read so the latest row is found even after a long
# weekend or holiday stretch. Wide enough to be robust, narrow enough that
# the projected read stays small.
_ARCTIC_LOOKBACK_DAYS = 14

# Staleness bound on the newest row, in NYSE TRADING days, mirroring the
# board-staleness bound in ``scoring/signals_envelope.py``. Saturday's
# weekly run legitimately sees Friday's close (1 trading day back).
_MAX_STALENESS_TRADING_DAYS = 3

# Per-ticker read-failure ceiling before the read is declared failed,
# mirroring ``data/fetchers/price_fetcher.py::_MAX_ERR_RATE``.
_MAX_ERR_RATE = 0.05


class FeatureStoreStalenessError(RuntimeError):
    """The newest available feature row is older than the staleness bound.

    Raised rather than returning degraded data: the scanner's
    ``universe_membership`` write is load-bearing (it fails the Scanner run
    by design, config-I4820) because the predictor resolves its daily
    universe from it. Ranking ~900 names on stale indicators and publishing
    the result as today's cut is precisely the silent-degradation this
    exception exists to prevent.
    """


def _connect_universe_lib():
    """Open the ArcticDB ``universe`` library.

    Mirrors ``data/fetchers/price_fetcher.py::_connect_arctic`` — same lib
    chokepoint, so S3 URI construction and ``get_library`` error wrapping
    stay in one place.
    """
    from nousergon_lib.arcticdb import open_universe_lib

    return open_universe_lib(_BUCKET)


def _assert_fresh(latest_ts, ref_date, *, source: str, what: str) -> None:
    """Raise ``FeatureStoreStalenessError`` if ``latest_ts`` is too old.

    Counted in NYSE trading days via ``nousergon_lib.trading_calendar``, so
    a normal weekend never reads as staleness and a genuine 3-day stall
    never hides behind one.
    """
    from nousergon_lib.trading_calendar import count_trading_days

    stale_td = count_trading_days(latest_ts.date(), ref_date)
    if stale_td > _MAX_STALENESS_TRADING_DAYS:
        raise FeatureStoreStalenessError(
            f"{what} newest row is {latest_ts.date()} — {stale_td} NYSE "
            f"trading days behind {ref_date} (bound "
            f"{_MAX_STALENESS_TRADING_DAYS}). Source={source}. The weekday "
            "producer (alpha-engine-data builders/daily_append.py, "
            "MorningEnrich) did not write, or wrote without the "
            "cross-sectional second pass."
        )


def _read_arctic_latest(
    tickers: list[str],
    columns: tuple[str, ...],
    ref_date,
    *,
    what: str,
) -> dict[str, dict]:
    """Newest row per ticker from the ArcticDB ``universe`` library.

    Returns ``{ticker: {column: float}}``. Uses ``read_batch`` rather than a
    per-ticker loop — at ~900 symbols the round-trip count, not the payload,
    dominates.
    """
    import numpy as np
    import pandas as pd
    from arcticdb.version_store.library import ReadRequest

    universe_lib = _connect_universe_lib()

    end_ts = pd.Timestamp(ref_date).normalize()
    start_ts = end_ts - pd.Timedelta(days=_ARCTIC_LOOKBACK_DAYS)

    results = universe_lib.read_batch(
        [ReadRequest(symbol=t, date_range=(start_ts, end_ts), columns=list(columns)) for t in tickers]
    )

    out: dict[str, dict] = {}
    n_err = 0
    newest = None
    for ticker, res in zip(tickers, results, strict=False):
        df = getattr(res, "data", None)
        if df is None or getattr(df, "empty", True):
            n_err += 1
            continue
        row = df.sort_index().iloc[-1]
        row_ts = df.sort_index().index[-1]
        if newest is None or row_ts > newest:
            newest = row_ts
        values = {
            c: float(row[c]) for c in columns if c in df.columns and pd.notna(row[c]) and np.isfinite(float(row[c]))
        }
        if values:
            out[ticker] = values

    err_rate = n_err / max(len(tickers), 1)
    if err_rate > _MAX_ERR_RATE:
        raise FeatureStoreStalenessError(
            f"{what} ArcticDB per-ticker read-failure rate {err_rate:.1%} "
            f"exceeds {_MAX_ERR_RATE:.0%} ({n_err} of {len(tickers)} symbols "
            "returned no row in the window)"
        )

    if newest is None:
        raise FeatureStoreStalenessError(
            f"{what} ArcticDB returned no rows at all for {len(tickers)} symbols in {start_ts.date()} → {end_ts.date()}"
        )

    _assert_fresh(newest, ref_date, source="arcticdb", what=what)

    logger.info(
        "[data_source=arcticdb] %s: %d/%d tickers, newest row %s (ref %s)",
        what,
        len(out),
        len(tickers),
        newest.date(),
        ref_date,
    )
    return out


def _use_arctic(tickers: list[str] | None, *, what: str) -> bool:
    """Whether this call reads ArcticDB (daily) or the S3 snapshot (weekly).

    ``tickers`` presence is the selector, with ``SCANNER_FEATURE_SOURCE=s3``
    as a hard override. ArcticDB is keyed per symbol and cannot enumerate a
    universe, so a caller that passes no ticker list structurally cannot use
    it — ``graph/research_graph.py::fetch_data_node`` is the one such
    caller, and it iterates the returned mapping to DISCOVER its universe.

    Raising there instead would break the ``challengers_only`` graph runner
    for no benefit; falling through to S3 silently would be worse, so the
    fallthrough is WARN-level and names the consequence. The live champion
    path (``scanner_orchestrator``) always passes its constituent list, so
    it can never reach the weekly surface by accident.
    """
    if _SOURCE == "s3":
        logger.warning(
            "%s: SCANNER_FEATURE_SOURCE=s3 — reading the WEEKLY snapshot. "
            "Rankings will not change between Saturday runs.",
            what,
        )
        return False
    if not tickers:
        logger.warning(
            "%s: no ticker list passed, falling back to the WEEKLY S3 "
            "snapshot (ArcticDB is keyed per symbol and cannot enumerate a "
            "universe). This caller's features are as of the last Saturday "
            "DataPhase1, not today.",
            what,
        )
        return False
    return True


def _today() -> object:
    """Reference date for staleness — the run's calendar date.

    Split out so tests can pin it without patching ``datetime`` globally.
    """
    from datetime import date

    return date.today()


def read_latest_features(
    tickers: list[str] | None = None,
    *,
    ref_date=None,
) -> dict[str, dict] | None:
    """Newest technical features per ticker, ``{ticker: {feature: value}}``.

    Source per ``SCANNER_FEATURE_SOURCE`` (default ``arcticdb``). ``tickers``
    is REQUIRED for the ArcticDB source — ArcticDB is keyed per symbol and
    cannot enumerate a universe the way an S3 parquet can. The scanner
    already holds the constituent list, so this is a pass-through, not a new
    lookup.

    Staleness raises ``FeatureStoreStalenessError``. An unreadable source
    still returns ``None`` on the S3 path, preserving the existing contract
    for the caller's explicit empty-store raise.
    """
    if _use_arctic(tickers, what="technical features"):
        return (
            _read_arctic_latest(
                tickers,
                _TECHNICAL_COLS,
                ref_date or _today(),
                what="technical features",
            )
            or None
        )
    return _read_latest_features_s3()


def _read_latest_features_s3() -> dict[str, dict] | None:
    """
    Read the most recent feature store snapshot from S3.

    Returns {ticker: {feature_name: value}} or None if unavailable.
    Only reads the 'technical' group (the features research needs).

    WEEKLY surface — written only by Saturday DataPhase1. Retained as the
    explicit ``SCANNER_FEATURE_SOURCE=s3`` rollback path.
    """
    try:
        import boto3
        import pandas as pd

        s3 = boto3.client("s3")

        # Find the latest date directory in features/
        response = s3.list_objects_v2(Bucket=_BUCKET, Prefix=_FEATURE_PREFIX, Delimiter="/")
        prefixes = response.get("CommonPrefixes", [])
        dates = []
        for p in prefixes:
            part = p["Prefix"].rstrip("/").split("/")[-1]
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                dates.append(part)

        if not dates:
            logger.debug("No feature store snapshots found in s3://%s/%s", _BUCKET, _FEATURE_PREFIX)
            return None

        latest_date = sorted(dates)[-1]
        logger.info("Feature store: reading snapshot from %s", latest_date)

        # Read technical group
        key = f"{_FEATURE_PREFIX}{latest_date}/technical.parquet"
        obj = s3.get_object(Bucket=_BUCKET, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, engine="pyarrow")

        if df.empty or "ticker" not in df.columns:
            return None

        # Convert to {ticker: {feature: value}} dict
        result = {}
        for _, row in df.iterrows():
            ticker = row["ticker"]
            features = {
                col: float(row[col]) for col in df.columns if col not in ("ticker", "date") and pd.notna(row[col])
            }
            # Reconstruct current_price from price_vs_ma50 if available
            # (feature store doesn't store raw price, but close is in OHLCV cache)
            result[ticker] = features

        logger.info("Feature store: loaded %d tickers from %s", len(result), latest_date)
        return result

    except Exception as e:
        logger.debug("Feature store read failed (non-blocking): %s", e)
        return None


def read_latest_factor_loadings(
    columns: tuple[str, ...] = (
        "momentum_20d_zscore",
        "return_60d_zscore",
        "beta_60d_zscore",
        "size_zscore",
    ),
    tickers: list[str] | None = None,
    *,
    ref_date=None,
) -> dict[str, dict[str, float]] | None:
    """Newest Barra factor loadings per ticker, ``{ticker: {factor: z}}``.

    Source per ``SCANNER_FEATURE_SOURCE`` (default ``arcticdb``), same
    contract as :func:`read_latest_features`.

    NOT fail-soft on the ArcticDB path, deliberately reversing this
    function's original posture. These loadings now drive the
    ``attractiveness_top_20`` CHAMPION cut that feeds the predictor
    (config-I4983); the old "research must never break because the loadings
    group hasn't shipped" rationale applied when they fed an OBSERVE-only
    neutralization shadow. Their daily producer
    (``daily_append.py``'s ``update_factor_loading_zscores_latest`` second
    pass) is best-effort and env-gated on the WRITE side, so the READ side
    is where a missed cross-sectional pass has to become visible — otherwise
    a silently skipped pass ranks the champion cut on stale loadings and
    publishes it as today's universe.
    """
    if _use_arctic(tickers, what="factor loadings"):
        return (
            _read_arctic_latest(
                tickers,
                tuple(columns),
                ref_date or _today(),
                what="factor loadings",
            )
            or None
        )
    return _read_latest_factor_loadings_s3(columns)


def _read_latest_factor_loadings_s3(
    columns: tuple[str, ...],
) -> dict[str, dict[str, float]] | None:
    """Read the most recent Barra factor-loading snapshot from S3.

    The factor loadings (``*_zscore`` columns, group ``factor_loading`` in
    alpha-engine-data's feature registry) are written to a SEPARATE parquet
    ``features/{date}/factor_loading.parquet`` — NOT the ``technical.parquet``
    group that :func:`read_latest_features` reads. They are the per-name
    exposures the score-neutralization OBSERVE shadow residualizes the
    composite against (config#1142).

    Returns ``{ticker: {factor_name: exposure}}`` for the requested ``columns``
    (only finite values included), or ``None`` if unavailable / the group's
    parquet is missing. Fully fail-soft — research must never break because the
    loadings group hasn't shipped a snapshot yet.
    """
    try:
        import boto3
        import pandas as pd

        s3 = boto3.client("s3")

        # Find the latest date directory in features/ (same discovery as
        # read_latest_features — the two groups share the date partition).
        response = s3.list_objects_v2(Bucket=_BUCKET, Prefix=_FEATURE_PREFIX, Delimiter="/")
        prefixes = response.get("CommonPrefixes", [])
        dates = []
        for p in prefixes:
            part = p["Prefix"].rstrip("/").split("/")[-1]
            if len(part) == 10 and part[4] == "-" and part[7] == "-":
                dates.append(part)

        if not dates:
            logger.debug("No feature store snapshots found in s3://%s/%s", _BUCKET, _FEATURE_PREFIX)
            return None

        latest_date = sorted(dates)[-1]
        key = f"{_FEATURE_PREFIX}{latest_date}/factor_loading.parquet"
        try:
            obj = s3.get_object(Bucket=_BUCKET, Key=key)
        except Exception as e:
            logger.debug("Factor-loading parquet not present at %s: %s", key, e)
            return None

        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, engine="pyarrow")

        if df.empty or "ticker" not in df.columns:
            return None

        present = [c for c in columns if c in df.columns]
        if not present:
            logger.debug(
                "Factor-loading parquet %s has none of the requested columns %s",
                key,
                columns,
            )
            return None

        result: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            ticker = row["ticker"]
            ex = {c: float(row[c]) for c in present if pd.notna(row[c])}
            if ex:
                result[ticker] = ex

        logger.info(
            "Factor loadings: loaded %d tickers from %s (%d/%d columns present)",
            len(result),
            latest_date,
            len(present),
            len(columns),
        )
        return result or None

    except Exception as e:
        logger.debug("Factor-loading read failed (non-blocking): %s", e)
        return None


def read_latest_daily_closes() -> dict[str, float] | None:
    """Read the most recent daily_closes parquet from S3.

    Returns {ticker: close_price} or None if unavailable.
    Much cheaper than yfinance batch fetch (~100KB single S3 read vs ~900 HTTP calls).

    Reads from ``staging/daily_closes/`` per the 2026-04-29 prefix migration
    in alpha-engine-data PR #112 (the parquet's role is intermediate state
    between API fetch and ArcticDB ingest, not authoritative storage).
    Hard-cutover with no fallback per ``feedback_no_silent_fails`` — if
    the staging prefix is empty/missing, the function returns ``None`` and
    callers must handle that explicitly (existing contract).
    """
    try:
        import boto3
        import pandas as pd

        s3 = boto3.client("s3")

        # Find the latest daily_closes file
        response = s3.list_objects_v2(
            Bucket=_BUCKET,
            Prefix="staging/daily_closes/",
            MaxKeys=100,
        )
        contents = response.get("Contents", [])
        if not contents:
            return None

        # Get the most recent parquet by key name (dates sort lexicographically)
        parquet_keys = [c["Key"] for c in contents if c["Key"].endswith(".parquet")]
        if not parquet_keys:
            return None

        latest_key = sorted(parquet_keys)[-1]
        logger.info("Daily closes: reading %s", latest_key)

        import io

        obj = s3.get_object(Bucket=_BUCKET, Key=latest_key)
        buf = io.BytesIO(obj["Body"].read())
        df = pd.read_parquet(buf, engine="pyarrow")

        if df.empty:
            return None

        # Schema: index=ticker or column=ticker, with 'close' or 'adj_close' column
        result = {}
        close_col = "close" if "close" in df.columns else "Close"
        ticker_col = None
        if "ticker" in df.columns:
            ticker_col = "ticker"
        elif df.index.name == "ticker":
            df = df.reset_index()
            ticker_col = "ticker"

        if ticker_col and close_col in df.columns:
            for _, row in df.iterrows():
                t = row[ticker_col]
                c = row[close_col]
                if t and pd.notna(c) and c > 0:
                    result[t] = float(c)

        logger.info("Daily closes: %d tickers with prices", len(result))
        return result if result else None

    except Exception as e:
        logger.debug("Daily closes read failed (non-blocking): %s", e)
        return None
