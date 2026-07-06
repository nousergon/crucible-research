"""
Research Lambda preflight: connectivity checks run at the top of each
handler invocation before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module composes them into two mode-specific sequences matching the
research Lambdas.

Modes:

- ``"weekly"`` — ``lambda/handler.py``, the weekly research pipeline.
  AWS_REGION + ANTHROPIC_API_KEY + S3 bucket reachable + ArcticDB ``universe``
  library reachable with SPY's last row populated. Phase 7c (2026-04-17) made
  ArcticDB the only price source for the weekly path, so an ArcticDB outage
  is now a hard failure rather than a degraded-mode scenario.
- ``"alerts"`` — ``lambda/alerts_handler.py``, the 30-minute intraday
  price alert Lambda. AWS_REGION + S3 bucket only; alerts do not call
  Anthropic and still read intraday bars from yfinance (ArcticDB is daily
  only — see ROADMAP "Intraday data store investigation").
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from nousergon_lib.preflight import BasePreflight

log = logging.getLogger(__name__)


class ResearchPreflight(BasePreflight):
    """Preflight checks for the two research Lambdas."""

    def __init__(self, bucket: str, mode: str):
        super().__init__(bucket)
        if mode not in ("weekly", "alerts"):
            raise ValueError(f"ResearchPreflight: unknown mode {mode!r}")
        self.mode = mode

    # Modules that ``lambda/handler.py`` imports lazily inside its hot
    # path. Without preflight verification they only fail at the WARN-
    # caught end-of-run path (or worse, silently degrade), AFTER all
    # LLM tokens have been spent. Eager import here surfaces a missing-
    # from-Docker-image module at the top of the invocation, before any
    # real work starts. Caught 2026-05-02 on a post-PR-D validation
    # invoke: ``scripts.aggregate_costs`` was imported by handler.py
    # (PR #81 wire-up) but the Dockerfile never copied ``scripts/``,
    # so every Lambda run logged a non-fatal WARN at the end.
    _DEFERRED_IMPORTS: tuple[tuple[str, str], ...] = (
        ("scripts.aggregate_costs", "aggregate_day"),
    )

    def _check_deferred_imports(self) -> None:
        """Verify every deferred-import module + symbol is resolvable.

        Failure surfaces at the top of the handler with a clear
        actionable error pointing at the Docker COPY contract — not as
        a silent end-of-run WARN. ``ImportError`` (module missing) and
        ``AttributeError`` (symbol renamed) are both treated as the
        same class of "deployment-side regression."
        """
        for module_path, attr in self._DEFERRED_IMPORTS:
            try:
                mod = __import__(module_path, fromlist=[attr])
                getattr(mod, attr)
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    f"Preflight: deferred import {module_path}.{attr} "
                    f"unresolvable: {type(exc).__name__}: {exc}. "
                    f"Check Dockerfile COPY lines + the module's "
                    f"__init__.py."
                ) from exc
        log.info(
            "preflight: %d deferred imports resolved",
            len(self._DEFERRED_IMPORTS),
        )

    def _check_arcticdb_universe(self) -> None:
        """Assert ArcticDB is reachable and SPY has fresh data.

        SPY is written by alpha-engine-data's weekly + daily collectors to the
        ``macro`` library (benchmarks/sector ETFs/macro series live there; the
        ``universe`` library holds the ~910 S&P 500+400 constituents). Its
        last-row date is the cleanest proxy for "DataPhase1 has run recently."

        Trading-day-aware via ``alpha_engine_lib.dates.is_fresh_in_trading_days``
        (lib v0.27.0). max_stale=5 trading days tolerates a research-only
        Saturday run after a holiday-shortened week without false-failing —
        tighter freshness is enforced by the predictor's daily inference,
        not the weekly research batch. Replaces the 7-calendar-day threshold
        that double-counted weekends/holidays as staleness.
        """
        from datetime import datetime, timezone
        from nousergon_lib.dates import (
            expected_last_close,
            is_fresh_in_trading_days,
            trading_days_stale,
        )
        import arcticdb as adb

        region = os.environ.get("AWS_REGION", "us-east-1")
        uri = f"s3s://s3.{region}.amazonaws.com:{self.bucket}?path_prefix=arcticdb&aws_auth=true"
        try:
            arctic = adb.Arctic(uri)
            macro = arctic.get_library("macro")
        except Exception as exc:
            raise RuntimeError(
                f"ArcticDB unreachable at {uri}: {exc}"
            ) from exc

        try:
            df = macro.read("SPY", columns=["Close"]).data
        except Exception as exc:
            raise RuntimeError(
                f"ArcticDB macro.SPY unreadable: {exc} — DataPhase1 did "
                f"not run or the macro library is broken."
            ) from exc

        if df is None or df.empty:
            raise RuntimeError(
                "ArcticDB macro.SPY has no rows — DataPhase1 has never written."
            )

        last_date = pd.Timestamp(df.index.max()).normalize().date()
        today_iso = datetime.now(timezone.utc).date().isoformat()
        if not is_fresh_in_trading_days(last_date, today_iso, max_stale=5):
            stale = trading_days_stale(last_date, today_iso)
            expected = expected_last_close(today_iso)
            raise RuntimeError(
                f"ArcticDB macro.SPY last_date={last_date} is "
                f"{stale} trading-day(s) behind the expected last close "
                f"{expected} as of {today_iso} (>5 trading-day(s) threshold) — "
                f"DataPhase1 has not refreshed recently."
            )
        log.info(
            "preflight: ArcticDB macro.SPY last_date=%s (within 5 trading-day(s) of today)",
            last_date,
        )

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        if self.mode == "weekly":
            # Without the Anthropic key the graph fails mid-invocation
            # with a less-actionable error; checking here surfaces the
            # misconfiguration before any S3 read or LLM call.
            self.check_env_vars("ANTHROPIC_API_KEY")
        self.check_s3_bucket()
        if self.mode == "weekly":
            self._check_deferred_imports()
            self._check_arcticdb_universe()
