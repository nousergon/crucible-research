"""Schema contract pins for every ``thinktank/`` artifact (M0 discipline).

Fields are ADD-ONLY: removing/renaming any name below is a breaking contract
change and must instead bump ``schema_version`` with a coordinated consumer
migration. New fields are appended to the frozen sets when added.
"""

from __future__ import annotations

from thinktank.schemas import (
    ChallengerSelection,
    ChallengerSelectionRow,
    CompanyThesis,
    CompanyThesisLLM,
    CompanyThesisRatedLLM,
    CoverageLedger,
    EventRecord,
    LedgerEntry,
    MonthlyCostLedger,
    RatingRow,
    RatingsBoard,
    RunManifest,
    SweepBatchLLM,
    ThemeThesis,
    ThemeThesisLLM,
    TickerEventAssessment,
)

_FROZEN_FIELDS = {
    CompanyThesis: {
        "schema_version", "ticker", "version", "trading_day", "calendar_date",
        "update_reason", "thesis", "sector", "attractiveness_score",
        "attractiveness_rank", "macro_theme_version", "sector_theme_version",
        "sources_used", "event_context", "model", "tier", "prompt_version",
        "cost_usd", "pillar_assessment",
    },
    CompanyThesisLLM: {
        "business_summary", "moat", "filings_review", "news_sentiment",
        "valuation", "market_dynamics", "risks", "catalysts", "stance",
        "conviction", "summary", "rating", "rating_rationale",
    },
    # Response contract for new generations: same field set, rating REQUIRED.
    CompanyThesisRatedLLM: {
        "business_summary", "moat", "filings_review", "news_sentiment",
        "valuation", "market_dynamics", "risks", "catalysts", "stance",
        "conviction", "summary", "rating", "rating_rationale",
    },
    RatingRow: {
        "ticker", "sector", "rating", "rating_rationale", "stance",
        "conviction", "summary", "thesis_version", "thesis_trading_day",
        "update_reason", "attractiveness_score", "attractiveness_rank",
        "rating_minus_attractiveness", "raw_llm_rating",
    },
    RatingsBoard: {"schema_version", "trading_day", "updated_at", "rows"},
    ChallengerSelectionRow: {
        "ticker", "rating", "stance", "conviction", "thesis_version",
        "attractiveness_rank",
    },
    ChallengerSelection: {
        "schema_version", "arm", "trading_day", "calendar_date", "run_id",
        "mode", "board_date", "coverage_complete", "uncovered_count",
        "selections",
    },
    ThemeThesis: {
        "schema_version", "kind", "key", "version", "trading_day",
        "calendar_date", "update_reason", "theme", "weekly_anchor_date",
        "divergence_from_weekly", "model", "tier", "prompt_version", "cost_usd",
    },
    ThemeThesisLLM: {
        "narrative", "stance", "drivers", "watch_items", "material_change",
        "change_summary",
    },
    TickerEventAssessment: {"ticker", "action", "severity", "rationale"},
    SweepBatchLLM: {"assessments", "macro_relevant"},
    EventRecord: {
        "schema_version", "ticker", "trading_day", "action", "severity",
        "rationale", "thesis_version_written",
    },
    LedgerEntry: {
        "ticker", "covered_since", "thesis_version", "thesis_updated_on",
        "last_sweep_on", "attractiveness_rank_at_entry", "sector",
    },
    CoverageLedger: {"schema_version", "updated_at", "entries"},
    RunManifest: {
        "schema_version", "run_id", "mode", "trading_day", "calendar_date",
        "started_at", "finished_at", "names_added", "names_refreshed",
        "theses_written", "sweep_tickers", "events_flagged",
        "event_updates_written", "themes_reconciled", "theme_updates_written",
        "ratings_rows", "challenger_selection_written", "context_sources_present",
        "usage_by_tier", "total_cost_usd",
        "coverage_gap",
        "budget_month_spent_usd", "budget_month_limit_usd", "errors",
    },
    MonthlyCostLedger: {"schema_version", "month", "spent_usd", "updated_at", "runs"},
}


def test_frozen_field_sets_are_superset_stable():
    for model, frozen in _FROZEN_FIELDS.items():
        current = set(model.model_fields)
        missing = frozen - current
        assert not missing, (
            f"{model.__name__} removed/renamed contract fields {sorted(missing)} — "
            "thinktank artifact schemas are add-only; bump schema_version and "
            "migrate consumers instead."
        )
        added = current - frozen
        assert not added, (
            f"{model.__name__} gained fields {sorted(added)} — additions are fine, "
            "but append them to _FROZEN_FIELDS so the contract stays pinned."
        )


def test_schema_version_stamped_on_artifacts():
    # config#2678: bumped 1 -> 2 (CompanyThesis.pillar_assessment +
    # RatingRow.raw_llm_rating) — SCHEMA_VERSION is one shared constant
    # across every thinktank/ artifact (thinktank/__init__.py), not
    # per-model, so the bump shows up on all of them.
    for model in (CompanyThesis, ThemeThesis, EventRecord, CoverageLedger,
                  RunManifest, MonthlyCostLedger, ChallengerSelection):
        assert "schema_version" in model.model_fields
        assert model.model_fields["schema_version"].default == 2


def test_llm_outputs_forbid_extra_fields():
    # extra="forbid" is what makes the bounded-retry validation loop bite —
    # a model that hallucinates fields must FAIL validation, not pass silently.
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TickerEventAssessment.model_validate(
            {"ticker": "AAPL", "action": "none", "severity": 1,
             "rationale": "x", "hallucinated": True}
        )
