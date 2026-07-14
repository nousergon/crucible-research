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
    # Independent 0-100 rating (Brian, 2026-07-02): the analyst's OWN
    # attractiveness call from its evidence review (filings, news/sentiment,
    # weekly research, macro/sector themes, raw metrics). Deliberately
    # independent of the scanner composite — the prompt WITHHOLDS the
    # attractiveness score / pillars so the model cannot anchor on them
    # (analyst._facts_board_row). Optional-with-default HERE so theses
    # stored before this field existed still parse (S3 add-only contract);
    # new generations are forced to emit it via CompanyThesisRatedLLM.
    rating: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Your independent attractiveness rating for this name, 0-100 "
            "(100 = most attractive), derived solely from your own review "
            "of the evidence provided. You are deliberately not shown the "
            "house quant composite — do not try to guess or match it."
        ),
    )
    rating_rationale: str = Field(
        default="",
        description=(
            "2-4 sentences: why this specific number — the evidence that "
            "drives it and what would move it up or down."
        ),
    )


class CompanyThesisRatedLLM(CompanyThesisLLM):
    """Response contract for NEW thesis generations — rating is REQUIRED.

    Storage keeps ``rating`` optional on :class:`CompanyThesisLLM` so
    pre-rating artifacts on S3 still parse; this subclass tightens the two
    fields to required for the LLM call, so an omitted rating fails
    validation and gets the client's bounded corrective retry instead of
    silently persisting ``None``.
    """

    rating: int = Field(
        ge=0,
        le=100,
        description=CompanyThesisLLM.model_fields["rating"].description,
    )
    rating_rationale: str = Field(
        description=CompanyThesisLLM.model_fields["rating_rationale"].description,
    )


class CompanyThesis(_Artifact):
    """Versioned per-ticker thesis — ``thinktank/theses/{ticker}/v{N}.json``."""

    ticker: str
    version: int = Field(ge=1)
    trading_day: str
    calendar_date: str
    update_reason: Literal[
        "initial", "event", "staleness_refresh", "reconcile", "operator_refresh"
    ]
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


# ── Ratings board (console/eval rollup) ──────────────────────────────────────


class RatingRow(BaseModel):
    """One covered name's current think-tank view, denormalized for consumers."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    sector: str | None = None
    rating: int | None = None  # None = thesis predates the rating field
    rating_rationale: str = ""
    stance: str = ""
    conviction: int | None = None
    summary: str = ""
    thesis_version: int = 0
    thesis_trading_day: str = ""
    update_reason: str = ""
    # Scanner composite AT THE TIME the thesis was written — metadata for
    # divergence display only; never shown to the model (see CompanyThesisLLM).
    attractiveness_score: float | None = None
    attractiveness_rank: int | None = None
    rating_minus_attractiveness: float | None = None


class RatingsBoard(_Artifact):
    """``thinktank/ratings/{trading_day}.json`` + ``latest.json`` — upserted
    every run from the theses written; the console/eval join surface (one
    read instead of N per-ticker thesis fetches)."""

    trading_day: str = ""
    updated_at: str = ""
    rows: dict[str, RatingRow] = Field(default_factory=dict)


# ── Challenger selection (champion/challenger leaderboard) ──────────────────


class ChallengerSelectionRow(BaseModel):
    """One name in Think Tank's challenger-arm selection.

    Sourced from the ratings board row for the ticker (independent rating +
    stance/conviction/thesis_version); ``attractiveness_rank`` rides along
    as metadata only — the ranking itself is by ``rating``, never by this.
    """

    model_config = ConfigDict(extra="forbid")

    ticker: str
    rating: int = Field(ge=0, le=100)
    stance: str
    conviction: int | None = None
    thesis_version: int = Field(ge=1)
    attractiveness_rank: int | None = None


class ChallengerSelection(_Artifact):
    """``thinktank/challenger_selection/{trading_day}.json`` + ``latest.json``
    — Think Tank's CHALLENGER-arm submission to the champion/challenger
    leaderboard (epic alpha-engine-config-I2515; champion = scanner→
    predictor direct, already live).

    Written at the tail of every non-dry ``run_daily`` (see
    ``thinktank.challenger_selection.write_challenger_selection``). ALWAYS
    emitted for observability, but ``coverage_complete`` is the validity
    flag downstream consumers must gate on — Brian's ruling (config#1580):
    the selection only counts once the ENTIRE current-scan top-N coverage
    window (``thinktank.run.GAP_FILL_TOP_N``) is covered. ``selections`` is
    ranked by Think Tank's OWN independent rating — never scanner
    attractiveness (independence is the point, see ``ratings.py``).

    ``board_date`` is the universe board's ``as_of`` at ranking time —
    carried for consumers to verify same-day-ness themselves; the daily
    cadence legitimately reads a stale (e.g. Saturday's) board all week, so
    this module never hard-fails on staleness (Brian, 2026-07-14, config#1580).
    """

    arm: Literal["thinktank_coverage"] = "thinktank_coverage"
    trading_day: str
    calendar_date: str
    run_id: str
    mode: Literal["daily", "gap_fill", "operator_refresh"]
    board_date: str | None = None
    coverage_complete: bool
    uncovered_count: int
    selections: list[ChallengerSelectionRow] = Field(default_factory=list)


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
    mode: Literal["daily", "reconcile", "dry_run", "operator_refresh", "gap_fill"]
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
    coverage_gap: dict | None = Field(
        default=None,
        description="Coverage gap vs scanner top-N: top60/top30 pct covered, "
        "uncovered counts. Emitted at end of every daily run.",
    )
    ratings_rows: int = 0
    challenger_selection_written: bool = False
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
