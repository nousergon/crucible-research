"""Versioned Pydantic schemas for every ``thinktank/`` S3 artifact.

M0 contract discipline: these are product contracts from birth. Fields are
only ever ADDED (never renamed/removed); ``schema_version`` bumps on any
shape change. ``tests/test_thinktank_schema_contract.py`` pins the frozen
field sets.
"""

from __future__ import annotations

from typing import Literal

from nousergon_lib.pillars import QualitativePillarAssessment
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
            "2-4 sentences: why this specific number — the evidence that drives it and what would move it up or down."
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
    update_reason: Literal["initial", "event", "staleness_refresh", "reconcile", "operator_refresh"]
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
    # config#2678: second, decoupled structured extraction over the SAME
    # scanner-blind evidence bundle (mirrors agents/sector_teams/qual_analyst.py's
    # pattern) — the qualitative 6-pillar/moat decomposition. None on
    # pre-port artifacts (add-only S3 contract); new generations always
    # populate it (thinktank.client.complete fails loud on parse failure,
    # no lax-mode empty path). Feeds thinktank.pillars.blend_rating — never
    # the scanner composite (thinktank/analyst.py::_facts_board_row still
    # withholds attractiveness_score/pillars from every prompt).
    pillar_assessment: QualitativePillarAssessment | None = None


# ── Ratings board (console/eval rollup) ──────────────────────────────────────


class RatingRow(BaseModel):
    """One covered name's current think-tank view, denormalized for consumers."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    sector: str | None = None
    # The OPERATIVE rating — what challenger_selection ranks by and the
    # leaderboard shadow view scores with. Since config#2678 this is the
    # analyst's raw rating blended with the pillar composite via
    # thinktank.pillars.blend_rating (still fully scanner-independent —
    # the pillar extraction is scanner-blind too); pre-2678 this equaled
    # raw_llm_rating exactly. None = thesis predates the rating field.
    rating: int | None = None
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
    # config#2678: the analyst's own rating BEFORE the pillar-composite
    # blend — audit/divergence display so the blend's effect is visible.
    # None only when ``rating`` itself is None (thesis predates rating).
    raw_llm_rating: int | None = None


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
    window (the cut ``universe_membership`` declares for Think Tank) is
    covered. ``selections`` is
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
    material_change: bool = Field(description="True ONLY if today's inputs materially change the prior view.")
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


class TriageDecisionLLM(BaseModel):
    """LLM output for one triage call (`products/thinktank.md` §2.4, T1).

    The sweep is wide and cheap, so its precision is low by design; this is the
    gate that decides whether a flagged event actually CHANGES the standing
    belief, before the expensive write tier is allowed to run. `escalate=False`
    is the expected common answer and is not a failure.
    """

    model_config = ConfigDict(extra="forbid")

    escalate: bool = Field(description="True only if the event changes the standing thesis's claim.")
    reason: str = Field(description="One-two sentences; why the belief does or does not move.")


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
    # ── Triage gate (alpha-engine-config-I6649, products/thinktank.md §2.4) ──
    # §2.3 is explicit that the NO decisions are the denominator: without them
    # the gate's precision is unmeasurable and a gate that has silently stopped
    # firing looks identical to a quiet week. So both verdicts are written, and
    # `triage_escalated` is None only for rows the gate never saw — i.e.
    # `action="none"`, which the sweep already rejected.
    triage_escalated: bool | None = None
    triage_reason: str | None = None


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
    # ── Hysteresis exit (alpha-engine-config-I6648, products/thinktank.md
    # §2.1). Coverage was a MONOTONIC RATCHET until 2026-08-10: nothing in
    # ledger.py removed an entry, so `rank_ceiling` gated ENTRY only and the
    # covered set could only grow — measured at 178 covered names against a
    # ceiling of 150. §2.1 wants a declared set with an enter threshold and a
    # WIDER exit threshold; this is the exit half.
    #
    # Brian's ruling 2026-08-10, option (a): the entry STAYS in `entries` and
    # is marked, rather than moving to a separate `dropped` map. `covered()`
    # already filters, so one more predicate is a smaller surface than a
    # second structure to keep consistent — and the drop record IS the audit
    # trail §2.1 wants for a de-covered name.
    #
    # The thesis history under `thinktank/theses/{ticker}/` is NEVER touched
    # by a drop. §2.2 makes every version immutable and forbids destroying the
    # record of what was believed when a decision was made; that binds a
    # de-covered name exactly as much as a covered one.
    covered: bool = True
    dropped_on: str | None = None
    attractiveness_rank_at_drop: int | None = None


class CoverageLedger(_Artifact):
    """``thinktank/coverage_ledger.json`` — the think tank's core state."""

    updated_at: str = ""
    entries: dict[str, LedgerEntry] = Field(default_factory=dict)

    def covered(self) -> set[str]:
        """Currently-covered tickers only.

        NOT ``set(self.entries)`` since alpha-engine-config-I6648: a
        de-covered name keeps its entry (and its whole thesis history) and is
        marked ``covered=False``. ``len(ledger.entries)`` therefore stopped
        being a coverage count — every consumer that wants one must call this.
        """
        return {t for t, e in self.entries.items() if e.covered}

    def dropped(self) -> set[str]:
        """Names covered at some point and de-covered since (I6648)."""
        return {t for t, e in self.entries.items() if not e.covered}


# ── Run manifest / cost ledger ───────────────────────────────────────────────


class TierUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    #: ``{rung: call_count}`` over ``krepis.llm``'s structured-output
    #: degradation ladder (``native`` / ``tool_emulation`` / ``prompt_only``,
    #: plus ``unknown`` when the transport reported none).
    #:
    #: krepis populates ``structured_output_rung`` on EVERY structured result,
    #: degraded or not, expressly so a degraded call is visible in the
    #: consumer's artifact — and this consumer read it nowhere
    #: (alpha-engine-config-I7658). Measured live 2026-08-18 against the `low`
    #: group: the deployment answers `400 This response_format type is
    #: unavailable now`, krepis descends native -> prompt_only and records the
    #: drop, and the run manifest said nothing. Every `sweep` and `triage`
    #: call in the daily run has been running one rung down since the ladder
    #: shipped, invisibly.
    #:
    #: Counts are published for the undegraded rung too: a tier that emits
    #: nothing here is unobserved, not healthy.
    structured_output_rungs: dict[str, int] = Field(default_factory=dict)


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
    # ── Triage gate counters (alpha-engine-config-I6649) ─────────────────────
    # events_flagged is the sweep's output; triage_yes + triage_no partition it
    # (modulo triage_errors). Their RATIO is the number the gate exists to move,
    # and it is unreadable from a single counter — which is why both are here
    # rather than only the escalations.
    triage_yes: int = 0
    triage_no: int = 0
    triage_errors: int = 0
    event_updates_written: int = 0
    themes_reconciled: bool = False
    theme_updates_written: int = 0
    context_sources_present: dict[str, bool] = Field(default_factory=dict)
    coverage_gap: dict | None = Field(
        default=None,
        description="Coverage gap vs the DECLARED coverage window: pct covered, "
        "uncovered count, plus the cut and basis the window resolved to. "
        "Emitted at end of every daily run.",
    )
    # ── Which contract this run read (alpha-engine-config-I7842) ─────────────
    # The window is resolved from `universe_membership/latest.json` through the
    # live champion pointer, so "which arm was Think Tank covering on date D"
    # has an artifact rather than being reconstructed from a deploy log. Before
    # this, Think Tank re-derived its own ranking and no run recorded what it
    # had ranked by — a champion cutover would have been invisible in the run
    # record as well as in the behaviour.
    feed_window: dict | None = Field(
        default=None,
        description="Provenance of the coverage window this run consumed: "
        "cut, declared_cut, basis, size, run_date, cut_effective_date, "
        "cut_refresh_cadence, rank_table_size, schema_version.",
    )
    ratings_rows: int = 0
    challenger_selection_written: bool = False
    # ── Challenger-selection POINTER lag (alpha-engine-config-I7232) ─────────
    # `challenger_selection/latest.json` is deliberately withheld on the abort
    # path (see `_terminal_writes`) — the dated key still lands, so the
    # directory keeps advancing daily while the pointer freezes, and to every
    # consumer that resolves the arm through the pointer a frozen pointer is
    # indistinguishable from a healthy one. Measured 2026-08-13: the pointer
    # was byte-identical to the 08-10 object while 08-11 and 08-12 were written
    # beside it.
    #
    # These two fields make the pointer's staleness a NUMBER published by the
    # run itself, readable without listing the dated keys next to it, and
    # published on healthy runs too — where it is 0, because the run just
    # advanced the pointer. `principles.md` §2.7: a component emitting nothing
    # is not healthy, it is unobserved, and "no data" is never rendered green.
    #
    # `None` on BOTH fields together means the pointer object does not exist at
    # all — a distinct state from "exists and is N days behind", which is why
    # the observed trading_day is carried rather than a lone lag integer with
    # an overloaded sentinel.
    challenger_selection_pointer_trading_day: str | None = None
    challenger_selection_pointer_lag_days: int | None = None
    usage_by_tier: dict[str, TierUsage] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    budget_month_spent_usd: float = 0.0
    budget_month_limit_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)
    # ── Deadline truncation (alpha-engine-config-I5208) ──────────────────────
    # A run that ran out of wall-clock still persists its terminal artifacts
    # (ratings board, challenger selection, shadow view). These fields make
    # that visible: a truncated run is PARTIAL, and a reader must be able to
    # tell partial coverage from complete coverage. Silence here would
    # reproduce the defect this was filed for.
    deadline_truncated: bool = False
    deadline_skipped_new: list[str] = Field(default_factory=list)
    deadline_skipped_refresh: list[str] = Field(default_factory=list)
    deadline_skipped_sweep: bool = False
    # ── Error truncation (same invariant, different cause) ───────────────────
    # The deadline fields above cover a run that ran out of TIME. A run killed
    # by an exception mid-loop lands in the identical state — completed work,
    # unwritten terminal artifacts — so it persists the same way and records
    # the cause here. Non-empty means the run RAISED: partial, and never to be
    # read as a healthy run.
    aborted_by_error: str = ""


class MonthlyCostLedger(_Artifact):
    """``thinktank/costs/{YYYY-MM}.json`` — month-to-date spend, feeds the budget guard."""

    month: str
    spent_usd: float = 0.0
    updated_at: str = ""
    runs: list[dict] = Field(default_factory=list)  # {run_id, trading_day, cost_usd}
