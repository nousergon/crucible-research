"""Versioned Pydantic schemas for every ``thinktank/`` S3 artifact.

M0 contract discipline: these are product contracts from birth. Fields are
only ever ADDED (never renamed/removed); ``schema_version`` bumps on any
shape change. ``tests/test_thinktank_schema_contract.py`` pins the frozen
field sets.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from thinktank import SCHEMA_VERSION


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION


# ── Company theses ────────────────────────────────────────────────────────────


class ThesisSection(BaseModel):
    """One narrative section of a company thesis."""

    model_config = ConfigDict(extra="forbid")

    title: str
    body: str


class CompanyThesisLLM(BaseModel):
    """The LLM-authored core of a thesis (what the model must return)."""

    model_config = ConfigDict(extra="forbid")

    business_summary: str = Field(description="What the company does; unit economics in brief.")
    moat: str = Field(description="Durable competitive advantage assessment.")
    filings_review: str = Field(description="Key takeaways from recent filings/8-Ks/earnings.")
    news_sentiment: str = Field(description="Recent news flow + sentiment read.")
    valuation: str = Field(description="Valuation frame vs sector/history.")
    market_dynamics: str = Field(description="How current market/regime context bears on the name.")
    risks: list[str] = Field(description="Key risks, most material first.")
    catalysts: list[str] = Field(description="Concrete upcoming catalysts, if any.")
    stance: Literal["attractive", "neutral", "avoid"]
    conviction: int = Field(ge=0, le=100)
    summary: str = Field(description="3-5 sentence executive summary of the thesis.")


class CompanyThesis(_Artifact):
    """Versioned per-ticker thesis — ``thinktank/theses/{ticker}/v{N}.json``."""

    ticker: str
    version: int = Field(ge=1)
    trading_day: str
    calendar_date: str
    update_reason: Literal["initial", "event", "staleness_refresh", "reconcile"]
    thesis: CompanyThesisLLM
    sector: str | None = None
    attractiveness_score: float | None = None
    attractiveness_rank: int | None = None
    macro_theme_version: int | None = None
    sector_theme_version: int | None = None
    sources_used: list[str] = Field(default_factory=list)
    event_context: str | None = None
    model: str = ""
    tier: str = ""
    prompt_version: str = ""
    cost_usd: float = 0.0


# ── Theme theses (macro + sector) ────────────────────────────────────────────


class ThemeThesisLLM(BaseModel):
    """LLM-authored core of a macro or sector theme."""

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(description="Current working view of this theme.")
    stance: str = Field(description="One-word/phrase stance (e.g. risk-on, overweight, cautious).")
    drivers: list[str] = Field(description="What is driving the current view.")
    watch_items: list[str] = Field(description="What would change the view; upcoming data/events.")
    material_change: bool = Field(
        description="True ONLY if today's inputs materially change the prior view."
    )
    change_summary: str = Field(
        default="",
        description="If material_change, what changed and why; else empty.",
    )


class ThemeThesis(_Artifact):
    """Versioned theme — ``thinktank/themes/{kind}/{key}/v{N}.json``.

    Lifecycle: seeded from the weekly SF artifacts, updated daily on material
    events only (churn discipline), reconciled to the weekly analysis when a
    new weekly run lands (the weekly view is the authoritative anchor for now).
    """

    kind: Literal["macro", "sector"]
    key: str  # "macro" or the sector name
    version: int = Field(ge=1)
    trading_day: str
    calendar_date: str
    update_reason: Literal["seed", "event", "reconcile"]
    theme: ThemeThesisLLM
    weekly_anchor_date: str | None = None  # signals.json date this theme is reconciled to
    divergence_from_weekly: str | None = None
    model: str = ""
    tier: str = ""
    prompt_version: str = ""
    cost_usd: float = 0.0


# ── Events sweep ─────────────────────────────────────────────────────────────


class TickerEventAssessment(BaseModel):
    """Per-ticker verdict from the daily events sweep (LLM output item)."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    action: Literal["none", "update_thesis"]
    severity: int = Field(ge=0, le=100)
    rationale: str = Field(description="One-two sentences; why this action.")


class SweepBatchLLM(BaseModel):
    """LLM output for one sweep chunk."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[TickerEventAssessment]
    macro_relevant: str = Field(
        default="",
        description="Any market-wide/macro-relevant development seen in this batch; else empty.",
    )


class EventRecord(_Artifact):
    """One line of ``thinktank/events/{trading_day}.jsonl``."""

    ticker: str
    trading_day: str
    action: Literal["none", "update_thesis"]
    severity: int
    rationale: str
    thesis_version_written: int | None = None


# ── Coverage ledger ──────────────────────────────────────────────────────────


class LedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    covered_since: str
    thesis_version: int
    thesis_updated_on: str  # trading_day of last thesis write
    last_sweep_on: str | None = None
    attractiveness_rank_at_entry: int | None = None
    sector: str | None = None


class CoverageLedger(_Artifact):
    """``thinktank/coverage_ledger.json`` — the think tank's core state."""

    updated_at: str = ""
    entries: dict[str, LedgerEntry] = Field(default_factory=dict)

    def covered(self) -> set[str]:
        return set(self.entries)


# ── Run manifest / cost ledger ───────────────────────────────────────────────


class TierUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class RunManifest(_Artifact):
    """``thinktank/runs/{trading_day}/manifest_{run_id}.json`` — one per run."""

    run_id: str
    mode: Literal["daily", "reconcile", "dry_run"]
    trading_day: str
    calendar_date: str
    started_at: str
    finished_at: str = ""
    names_added: list[str] = Field(default_factory=list)
    names_refreshed: list[str] = Field(default_factory=list)
    theses_written: int = 0
    sweep_tickers: int = 0
    events_flagged: int = 0
    event_updates_written: int = 0
    themes_reconciled: bool = False
    theme_updates_written: int = 0
    context_sources_present: dict[str, bool] = Field(default_factory=dict)
    usage_by_tier: dict[str, TierUsage] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    budget_month_spent_usd: float = 0.0
    budget_month_limit_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class MonthlyCostLedger(_Artifact):
    """``thinktank/costs/{YYYY-MM}.json`` — month-to-date spend, feeds the budget guard."""

    month: str
    spent_usd: float = 0.0
    updated_at: str = ""
    runs: list[dict] = Field(default_factory=list)  # {run_id, trading_day, cost_usd}
