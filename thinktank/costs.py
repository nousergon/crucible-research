"""Budget guard + monthly spend ledger.

The monthly cap is a HARD gate (hold-costs preference): a run refuses to
start once month-to-date spend reaches the limit. Limit resolution order:

1. ``THINKTANK_MONTHLY_BUDGET_USD`` env (explicit operator/test override)
2. SSM parameter (``thinktank.yaml: budget.ssm_param``)
3. ``thinktank.yaml: budget.monthly_usd_default``

An unreadable SSM parameter falls back to the YAML default WITH a WARN — the
guard fails CLOSED onto the conservative default cap, never open; the WARN +
the manifest's ``budget_month_limit_usd`` field are the recording surface.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from thinktank import COSTS_KEY_TMPL
from thinktank.schemas import MonthlyCostLedger
from thinktank.settings import ThinktankSettings
from thinktank.storage import ThinktankStore

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Month-to-date spend has reached the cap — the run must not start."""


class BudgetGuard:
    def __init__(
        self,
        store: ThinktankStore,
        settings: ThinktankSettings,
        *,
        ssm_client: Any | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._ssm = ssm_client

    # ── limit resolution ─────────────────────────────────────────────────────

    def limit_usd(self) -> float:
        env = os.environ.get("THINKTANK_MONTHLY_BUDGET_USD")
        if env:
            return float(env)
        try:
            ssm = self._ssm
            if ssm is None:
                import boto3

                ssm = boto3.client("ssm")
            resp = ssm.get_parameter(Name=self._settings.budget_ssm_param)
            return float(resp["Parameter"]["Value"])
        except Exception as exc:  # noqa: BLE001 — guard fails CLOSED onto the default
            logger.warning(
                "budget SSM param %s unreadable (%s) — enforcing YAML default $%.2f",
                self._settings.budget_ssm_param,
                exc,
                self._settings.monthly_budget_usd_default,
            )
            return self._settings.monthly_budget_usd_default

    # ── ledger ───────────────────────────────────────────────────────────────

    @staticmethod
    def month_key(calendar_date: str) -> str:
        return calendar_date[:7]  # YYYY-MM

    def load_ledger(self, month: str) -> MonthlyCostLedger:
        raw = self._store.get_json(COSTS_KEY_TMPL.format(month=month))
        if raw is None:
            return MonthlyCostLedger(month=month)
        return MonthlyCostLedger.model_validate(raw)

    def check(self, calendar_date: str) -> tuple[float, float]:
        """Raise if month-to-date spend >= limit. Returns (spent, limit)."""
        month = self.month_key(calendar_date)
        ledger = self.load_ledger(month)
        limit = self.limit_usd()
        if ledger.spent_usd >= limit:
            raise BudgetExceededError(
                f"thinktank month-to-date spend ${ledger.spent_usd:.2f} >= "
                f"cap ${limit:.2f} for {month} — refusing to run. Raise "
                f"{self._settings.budget_ssm_param} deliberately to continue."
            )
        return ledger.spent_usd, limit

    def record_run(
        self, calendar_date: str, *, run_id: str, trading_day: str, cost_usd: float
    ) -> MonthlyCostLedger:
        month = self.month_key(calendar_date)
        ledger = self.load_ledger(month)
        ledger.spent_usd = round(ledger.spent_usd + cost_usd, 6)
        ledger.updated_at = datetime.now(UTC).isoformat()
        ledger.runs.append(
            {"run_id": run_id, "trading_day": trading_day, "cost_usd": cost_usd}
        )
        self._store.put_json(COSTS_KEY_TMPL.format(month=month), ledger.model_dump())
        return ledger
