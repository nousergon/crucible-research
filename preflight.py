"""
Research Lambda preflight: connectivity checks run at the top of each
handler invocation before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module composes them into two mode-specific sequences matching the
research Lambdas.

Modes:

- ``"weekly"`` — ``lambda/handler.py``, the weekly research pipeline.
  AWS_REGION + the krepis router env contract + both weekly model classes
  RESOLVING through the router + S3 bucket reachable + ArcticDB ``universe``
  library reachable with SPY's last row populated. Phase 7c (2026-04-17) made
  ArcticDB the only price source for the weekly path, so an ArcticDB outage
  is now a hard failure rather than a degraded-mode scenario.
- ``"alerts"`` — ``lambda/alerts_handler.py``, the 30-minute intraday
  price alert Lambda. AWS_REGION + S3 bucket only; alerts make no LLM call
  at all and still read intraday bars from yfinance (ArcticDB is daily
  only — see ROADMAP "Intraday data store investigation").

WHY THE WEEKLY CHECK IS NOT A CREDENTIAL CHECK (alpha-engine-config-I9302).
It used to be ``check_env_vars("ANTHROPIC_API_KEY")``. Direct Anthropic is
RETIRED (Brian's 2026-08-29 ruling: "we shouldn't be using the anthropic api
at all"), and every model call on this path now resolves through
``krepis.router.resolve_group_spec`` — which returns the model, the endpoint
AND the credential NAME from the registry. So the presence of a retired
vendor's key says nothing about whether this run can reach a model: it would
pass on a box with no router at all, and fail on a correctly configured one.
That is a preflight that is wrong in both directions.

What is checked instead is the thing the run actually needs: krepis' own
environment contract is declared, and BOTH model classes the weekly path uses
resolve from the execution context this process DECLARES. A resolution failure
here is the same failure the run would hit mid-invocation, surfaced before any
S3 read or LLM spend.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC

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
    #
    # `krepis.router` and `agents.langchain_utils` join this table rather than
    # becoming module-level imports (alpha-engine-config-I9302). Both are
    # weekly-only, and importing either at module level makes the WHOLE agent
    # graph transitively reachable from `Dockerfile.alerts`' entrypoint — the
    # packaging guard says so, correctly, and satisfying it by COPYing twelve
    # more packages into the alerts image would pay a real cost for a code path
    # alerts never executes.
    #
    # Deferring is safe here ONLY because this table exists: it resolves the
    # MODULE and the SYMBOL separately, which a bare guarded import cannot —
    # `try: from x import y except ImportError` gives a rename and a missing
    # COPY the same exception and lets the run go silently on
    # (alpha-engine-config-I9339). Both are resolved eagerly in weekly mode,
    # before any LLM spend.
    _DEFERRED_IMPORTS: tuple[tuple[str, str], ...] = (
        ("scripts.aggregate_costs", "aggregate_day"),
        ("krepis.router", "resolve_group_spec"),
        ("agents.langchain_utils", "_exec_context"),
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
            self._deferred(module_path, attr)
        log.info(
            "preflight: %d deferred imports resolved",
            len(self._DEFERRED_IMPORTS),
        )

    @staticmethod
    def _deferred(module_path: str, attr: str):
        """Resolve one deferred (module, symbol) pair, or raise saying WHICH failed.

        A missing MODULE (an un-COPY'd package) and a missing SYMBOL (a rename
        upstream) are different faults with different fixes, and a bare
        `try: from x import y except ImportError` gives them the same exception
        — that conflation is alpha-engine-config-I9339. Resolved separately so
        the error names the actual fault.
        """
        try:
            mod = __import__(module_path, fromlist=[attr])
        except ImportError as exc:
            raise RuntimeError(
                f"Preflight: deferred import {module_path}.{attr} "
                f"unresolvable — MODULE missing: {exc}. Check Dockerfile "
                f"COPY lines + the module's __init__.py."
            ) from exc
        try:
            return getattr(mod, attr)
        except AttributeError as exc:
            raise RuntimeError(
                f"Preflight: deferred import {module_path}.{attr} "
                f"unresolvable — module imported but SYMBOL absent: {exc}. "
                f"A rename upstream, not a packaging fault."
            ) from exc

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
        from datetime import datetime

        import arcticdb as adb
        from nousergon_lib.dates import (
            expected_last_close,
            is_fresh_in_trading_days,
            trading_days_stale,
        )

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
        today_iso = datetime.now(UTC).date().isoformat()
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

    def _check_router_resolves_weekly_classes(self) -> None:
        """Both weekly model classes must RESOLVE through the krepis router.

        Resolution is what proves the run can reach a model: it exercises the
        registry, the declared execution context, and the group's membership
        in one call, and it returns the credential name the call will actually
        Both symbols come from ``_DEFERRED_IMPORTS`` rather than module-level
        imports: they are weekly-only, and importing them at module level drags
        the whole agent graph into the alerts image (see that table's note).

        Raises rather than degrades: a weekly run that cannot resolve its model
        classes has nothing to fall back to, and a preflight that shrugs at
        that is worse than no preflight (`model-router-policy` §5 step 3).
        """
        import config  # noqa: PLC0415

        resolve_group_spec = self._deferred("krepis.router", "resolve_group_spec")
        exec_context = self._deferred("agents.langchain_utils", "_exec_context")()
        for model_class in (config.PER_STOCK_CLASS, config.STRATEGIC_CLASS):
            try:
                spec, _route = resolve_group_spec(
                    model_class, exec_context=exec_context, wire="openai"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"preflight: model class {model_class!r} does not resolve "
                    f"through the krepis router from exec_context="
                    f"{exec_context or '(undeclared)'!r} — the weekly run "
                    f"cannot reach a model. This is the same failure the run "
                    f"would hit mid-invocation, surfaced before any spend."
                ) from exc
            # The credential NAME is deliberately not logged. It is not a
            # secret, but CodeQL reads `api_key_env` as sensitive and it adds
            # nothing a failure would not already say — the resolver's own
            # error names the credential when resolution is what failed.
            log.info("preflight: %s resolves to %s", model_class, spec.model)

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        if self.mode == "weekly":
            # Without a reachable router the graph fails mid-invocation with a
            # less-actionable error; checking here surfaces the
            # misconfiguration before any S3 read or LLM spend.
            self.check_env_vars(
                "KREPIS_LITELLM_PROXY_URL", "KREPIS_ROUTER_CREDENTIAL_SECRET"
            )
            self._check_router_resolves_weekly_classes()
        self.check_s3_bucket()
        if self.mode == "weekly":
            self._check_deferred_imports()
            self._check_arcticdb_universe()
