"""
Pydantic schemas for typed LangGraph state — agent outputs + computed
artifacts referenced from ``ResearchState``.

These schemas are the typed-state successor to the ``dict[str, Any]`` and
``dict[str, dict]`` annotations that lived in
``graph.research_graph.ResearchState``. They are now wired into
``ResearchState`` via ``Annotated[..., reducer]`` field types and used by
``_validate`` at every node boundary in strict-by-default mode.

**Compatibility posture:** most models still use
``model_config = ConfigDict(extra="allow")`` because agents emit fields
not enumerated here (e.g. ``quant_output``, ``qual_output``,
``peer_review_output`` from the sector-team stub). LLM-extraction schemas
wrapped via ``with_structured_output()`` at 8 sites enforce the stricter
contract on the LLM side; the state-schema ``extra="forbid"`` flip lands
incrementally as each agent's output contract soaks. ``RubricEvalArtifact``
already runs ``extra="forbid"``.

Numeric constraints (e.g. ``quant_score ∈ [0, 100]``, sector modifiers
∈ [0.70, 1.30]) ARE enforced even with ``extra="allow"`` — the validators
fire on the named fields regardless of the extras policy.

Workstream context: ``~/Development/alpha-engine-docs/private/alpha-engine-
research-typed-state-capture-260429.md`` (Day-1 design doc § 3).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# LLM-output schemas live in alpha-engine-lib so the replay harness in
# alpha-engine-backtester can validate against the canonical contract
# without a heavy cross-repo dep on this repo. Lifted 2026-05-05 (lib
# v0.4.0). Re-exported below so existing call sites that
# `from graph.state_schemas import QuantAnalystOutput` etc. keep working
# unchanged. State-machine objects (SectorTeamOutput, MacroEconomistOutput,
# CIOOutput, InvestmentThesis) stay in this module — they're research-
# internal state types, not LLM-output contracts.
from nousergon_lib.agent_schemas import (
    CIORawDecision,
    CIORawDecisionLiteral,
    CIORawOutput,
    HeldThesisUpdateLLMOutput,
    JointFinalizationDecision,
    JointFinalizationOutput,
    JointSelectionOutput,
    MacroCriticOutput,
    MacroEconomistRawOutput,
    QualAnalystOutput,
    QualAssessment,
    QuantAcceptanceVerdict,
    QuantAnalystOutput,
    QuantPick,
    RegimeLiteral,
    RubricDimensionScore,
    RubricEvalLLMOutput,
)

__all__ = [
    # Re-exported from alpha_engine_lib.agent_schemas for backward
    # compatibility with existing call sites.
    "CIORawDecision",
    "CIORawDecisionLiteral",
    "CIORawOutput",
    "HeldThesisUpdateLLMOutput",
    "JointFinalizationDecision",
    "JointFinalizationOutput",
    "JointSelectionOutput",
    "MacroCriticOutput",
    "MacroEconomistRawOutput",
    "QualAnalystOutput",
    "QualAssessment",
    "QuantAcceptanceVerdict",
    "QuantAnalystOutput",
    "QuantPick",
    "RegimeLiteral",
    "RubricDimensionScore",
    "RubricEvalLLMOutput",
    # Research-internal state-machine + storage types (defined below).
    "StoredConvictionLiteral",
    "ToolCall",
    "SectorRecommendation",
    "ThesisUpdate",
    "SectorTeamOutput",
    "MacroEconomistOutput",
    "ExitEvent",
    "ExitEvaluatorOutput",
    "CIODecision",
    "CIOOutput",
    "InvestmentThesis",
    "PopulationRotationEvent",
    "RubricEvalArtifact",
]


# ── Conviction enums (agent vs storage formats) ───────────────────────────
# Agent format (post-Option-A 2026-04-30): every agent emits int 0-100. The
# old Literal["high","medium","low"] alias is retired — see ROADMAP entry
# "Standardize agent-level conviction to int 0-100 (Option A)".
# Storage format: produced by ``scoring.composite.normalize_conviction()``
# and used by downstream executor + archive writes. Trend label, NOT
# strength — preserved for backward compatibility with existing rows in
# ``investment_thesis`` SQLite (cleanly renamed under Option B / Phase 4).
# ``ThesisUpdate.conviction`` is an int|StoredConvictionLiteral|None union
# because prior_theses (loaded from archive) carry storage format while
# cio entry_theses + agent-emitted theses carry the agent int format.
StoredConvictionLiteral = Literal["rising", "stable", "declining"]

# Canonical "this decision admits the ticker into the population" predicate.
# The CIO emits BOTH "ADVANCE" (rubric) and "ADVANCE_FORCED" (the
# min_new_entrants floor force-fill). Any consumer that matches only
# "ADVANCE" silently drops forced entrants — the bug class that hid the
# floor for weeks (apply_ic_entries, semantic memory, the report-note
# builder all filtered only "ADVANCE"). Match this set EVERYWHERE a
# decision is interpreted as an advance so the literal can never diverge
# again. Mirrors the dashboard-side ADVANCE_DECISIONS in
# alpha-engine-dashboard loaders/signal_loader.py.
ADVANCE_DECISIONS = frozenset({"ADVANCE", "ADVANCE_FORCED"})


# ── Atomic agent-output components ────────────────────────────────────────


class ToolCall(BaseModel):
    """One ReAct tool invocation log entry. Diagnostic; not load-bearing.

    ``tool`` is optional because peer-review-orchestration entries are
    appended to ``tool_calls`` to record the phase, but they have no
    underlying tool name (peer review is a synthesis step, not a tool
    invocation). 2026-04-30 warn-mode validation surfaced 3 such entries
    across healthcare/financials/defensives in a real Saturday SF run.
    """

    model_config = ConfigDict(extra="allow")

    tool: str | None = None
    ticker: str | None = None
    args: dict = Field(default_factory=dict)
    result_summary: str | None = None


class SectorRecommendation(BaseModel):
    """One BUY-candidate output from a sector team's quant→qual→peer chain.

    ``qual_score`` is optional because peer-review can produce
    recommendations even when the qual analyst returned zero assessments
    (logged as ``[qual:<team>] completed — 0 assessments, N tool calls``).
    2026-04-30 warn-mode validation surfaced 6 such entries (healthcare 3
    + defensives 3) in a real Saturday SF run.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    quant_score: float = Field(ge=0, le=100)
    qual_score: float | None = Field(default=None, ge=0, le=100)
    bull_case: str = ""
    bear_case: str = ""
    catalysts: list[str] = Field(default_factory=list)
    conviction: int | None = Field(default=None, ge=0, le=100)
    quant_rationale: str = ""


class ThesisUpdate(BaseModel):
    """
    Per-stock held-position thesis update.

    All score fields are nullable: the held-stock evaluation path
    occasionally produces records missing ``final_score`` (first-time
    update, legacy archive entries predating the current schema). The
    ``score_aggregator`` recompute-or-hard-fail path (alpha-engine-research
    PR #42, 2026-04-22) handles the partial-score case by recomputing
    ``final_score`` from sub-scores when both are present, hard-failing
    only when ALL three score fields are absent.
    """

    model_config = ConfigDict(extra="allow")

    # ``ticker`` is optional because ThesisUpdate values appear as the
    # value-half of a ``dict[str, ThesisUpdate]`` mapping where the key IS
    # the ticker (e.g. ``cio.entry_theses[ticker] = thesis_dict``). The
    # cio agent and held-stock thesis_update path both rely on this
    # convention. score_aggregator's investment_theses path repeats the
    # ticker in the value too, so callers can use either shape.
    ticker: str | None = None
    final_score: float | None = Field(default=None, ge=0, le=100)
    quant_score: float | None = Field(default=None, ge=0, le=100)
    qual_score: float | None = Field(default=None, ge=0, le=100)
    sector: str | None = None
    rating: Literal["BUY", "HOLD", "SELL"] | None = None
    # int-or-storage-string union: agent-emitted entries carry int 0-100
    # (post-Option-A); prior_theses loaded from archive carry the storage
    # trend label rising/stable/declining. ``normalize_conviction`` flattens
    # the union back to storage format at the score_aggregator boundary.
    conviction: int | StoredConvictionLiteral | None = None
    bull_case: str = ""
    bear_case: str = ""
    thesis_summary: str = ""


# ── Sector team output (one per Send fan-out branch) ──────────────────────


class SectorTeamOutput(BaseModel):
    """
    Wraps a single sector team's full output. Stored under
    ``state['sector_team_outputs'][team_id]`` after Send fan-out merges.
    """

    model_config = ConfigDict(extra="allow")

    team_id: str
    recommendations: list[SectorRecommendation] = Field(default_factory=list)
    thesis_updates: dict[str, ThesisUpdate] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    error: str | None = None


# ── Macro economist output ────────────────────────────────────────────────


REGIME_VALUES = ("bull", "neutral", "bear")
"""3-class Ang-Bekaert macro regime taxonomy (v0.42.0 / 2026-05-28).
Legacy 4-class "caution" retired per caution-regime-retirement-260528.md.
Portfolio-protective hysteresis (risk_on/caution/risk_off) is a separate
axis on the predictor drawdown leg; consumers compose via most-protective
override at decision time."""
class MacroEconomistOutput(BaseModel):
    """
    Stored as four separate keys in ``state`` (``macro_report``,
    ``sector_modifiers``, ``sector_ratings``, ``market_regime``) — this
    schema captures the contract those four fields must satisfy together.
    """

    model_config = ConfigDict(extra="allow")

    macro_report: str = ""
    sector_modifiers: dict[str, float] = Field(default_factory=dict)
    sector_ratings: dict[str, dict] = Field(default_factory=dict)
    market_regime: RegimeLiteral = "neutral"

    @field_validator("sector_modifiers")
    @classmethod
    def clamp_modifiers(cls, v: dict[str, float]) -> dict[str, float]:
        """Each per-sector modifier must lie in the macro-economist invariant
        range [0.70, 1.30] per the agent's prompt-level clamping rule."""
        for sector, m in v.items():
            if not (0.70 <= float(m) <= 1.30):
                raise ValueError(
                    f"sector_modifiers[{sector!r}]={m} outside [0.70, 1.30]"
                )
        return v


# ── Exit evaluator output ─────────────────────────────────────────────────


class ExitEvent(BaseModel):
    """One exit-from-population event."""

    model_config = ConfigDict(extra="allow")

    ticker_out: str
    reason: str = ""
    score_out: float = 0.0


class ExitEvaluatorOutput(BaseModel):
    """
    Stored as three separate keys in ``state`` (``remaining_population``,
    ``exits``, ``open_slots``) — this schema captures the contract.
    """

    model_config = ConfigDict(extra="allow")

    remaining_population: list[dict] = Field(default_factory=list)
    exits: list[ExitEvent] = Field(default_factory=list)
    open_slots: int = Field(default=0, ge=0)


# ── CIO output ────────────────────────────────────────────────────────────


class CIODecision(BaseModel):
    """One per-ticker CIO decision (ADVANCE / REJECT / HOLD).

    ``conviction`` is an integer score (0-100), aligned with what the CIO
    agent prompt actually emits — composite ranking scores like 78, 72, 25.
    2026-04-30 warn-mode validation surfaced 9 violations against the prior
    Literal['high','medium','low'] schema in a real Saturday SF run; every
    decision had a numeric conviction. Path Y of the conviction-semantics
    decision (versatility — int representation generalizes; downstream
    consumers can map to display levels via a level-helper). PR for
    producer-side alignment of SectorRecommendation/InvestmentThesis to
    int convention is the follow-up.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    thesis_type: Literal["ADVANCE", "REJECT", "HOLD"] | None = None
    rationale: str = ""
    conviction: int | None = Field(default=None, ge=0, le=100)
    score: float | None = Field(default=None, ge=0, le=100)


class CIOOutput(BaseModel):
    """
    Stored as three separate keys in ``state`` (``ic_decisions``,
    ``advanced_tickers``, ``entry_theses``).
    """

    model_config = ConfigDict(extra="allow")

    ic_decisions: list[CIODecision] = Field(default_factory=list)
    advanced_tickers: list[str] = Field(default_factory=list)
    entry_theses: dict[str, ThesisUpdate] = Field(default_factory=dict)


# ── Investment thesis (computed by score_aggregator) ──────────────────────


class InvestmentThesis(BaseModel):
    """
    Per-ticker investment thesis. Constructed by ``score_aggregator`` from
    sector_team recommendations + sector modifiers; written to S3 +
    research.db by ``archive_writer``; consumed by ``consolidator`` for
    the email body.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    # `None` is the schema-level marker for "LLM omitted this field" so the
    # downstream `or` chains in research_graph.py route to sector_map first.
    # Prior default of "Unknown" was a truthy string that short-circuited
    # those `or` chains and leaked through to signals.json + trades.db
    # (2026-05-04 EOG/NVT incident; root-cause fix in 39c379f).
    sector: str | None = None
    team_id: str = ""
    final_score: float = Field(ge=0, le=100)
    quant_score: float | None = Field(default=None, ge=0, le=100)
    qual_score: float | None = Field(default=None, ge=0, le=100)
    weighted_base: float = 0.0
    macro_shift: float = 0.0
    bull_case: str = ""
    bear_case: str = ""
    catalysts: list[str] = Field(default_factory=list)
    # InvestmentThesis is constructed by score_aggregator AFTER
    # ``normalize_conviction()`` runs, so the stored value uses the
    # executor-compatible enum (rising/stable/declining), NOT the
    # agent-input format (high/medium/low). Surfaced 2026-04-29 by
    # warn-mode validation against the original draft schema.
    conviction: StoredConvictionLiteral = "stable"
    quant_rationale: str = ""
    rating: Literal["BUY", "HOLD", "SELL"]
    score_failed: bool = False


# ── Population rotation event (entry/exit log entry) ──────────────────────


class PopulationRotationEvent(BaseModel):
    """
    One row of the population-rotation log. Mixes entry and exit shapes
    (different agents emit different field sets), so we leave the schema
    open and capture only the load-bearing fields.
    """

    model_config = ConfigDict(extra="allow")

    event_type: Literal["entry", "exit"] | None = None
    ticker: str | None = None
    ticker_in: str | None = None
    ticker_out: str | None = None
    reason: str = ""


# ── LLM-extraction schemas — lifted to alpha-engine-lib ──────────────────
#
# The typed shapes passed to ``llm.with_structured_output(Schema)`` at
# each agent boundary live in ``alpha_engine_lib.agent_schemas`` (lifted
# 2026-05-05, lib v0.4.0). They're re-exported at the top of this module
# so existing imports from ``graph.state_schemas`` keep working
# unchanged. The split is:
#
#   - State schemas (SectorTeamOutput, CIODecision, InvestmentThesis, ...)
#     describe what flows through ``ResearchState`` after agent post-
#     processing — they include fields downstream nodes compute or merge
#     in (e.g. ``score_aggregator`` populates ``rating`` from sub-scores).
#     These stay in this module.
#
#   - LLM-extraction schemas (QuantAnalystOutput, JointFinalizationOutput,
#     CIORawOutput, RubricEvalLLMOutput, etc.) describe the EXACT shape
#     the LLM must emit. They live in the shared lib so the replay
#     harness in alpha-engine-backtester can validate against the
#     canonical contract without a heavy cross-repo dep on this repo.
#
# RubricEvalArtifact (below) is the *persisted* eval form — wraps the
# (lib-side) RubricEvalLLMOutput with metadata that ties it back to the
# judged decision artifact + rubric prompt version + judge model. Stays
# here because it's the storage contract for ``decision_artifacts/_eval/``,
# not an LLM-output schema.


class RubricEvalArtifact(BaseModel):
    """One persisted eval result.

    Stored at the institutional production-grade path:
    ``decision_artifacts/_eval/{judge_run_date}/{judge_run_id}/
    {judged_agent_id}.{judged_run_id}.{judge_model}.json`` (Option B
    partition shipped 2026-05-08). Each judge batch invocation gets a
    fresh ``judge_run_id`` (UUID) so all artifacts emitted by one batch
    cluster under a single directory and are queryable as a group.
    Operator queries by capture-date use the manifest layer at
    ``_eval_by_capture/{capture_date}/manifest.json`` (PR 2).

    Wraps the LLM output with metadata that ties it back to the judged
    decision artifact, the rubric prompt version, and the judge model.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    run_id: str = Field(description="Pipeline-invocation identifier of the JUDGED artifact; ties this eval back to the agent's run.")
    judge_run_id: str = Field(description="UUID of the eval-judge batch invocation that produced this eval. Constant across all artifacts emitted by one batch; lets operators query 'what did this judge batch produce?' without scanning the corpus.")
    timestamp: str = Field(description="ISO-8601 emission time. Per-artifact wall-clock; used as the date-partition source when present.")
    judged_agent_id: str = Field(description="agent_id of the DecisionArtifact this eval is scoring (e.g. 'sector_quant:technology').")
    judged_artifact_s3_key: str | None = Field(
        default=None,
        description="Backref to the judged artifact's S3 key. Optional — None if eval was run on an in-memory artifact.",
    )
    rubric_id: str = Field(description="Rubric prompt name (e.g. 'eval_rubric_sector_quant').")
    rubric_version: str = Field(description="Rubric prompt version at eval time (semver from prompt frontmatter).")
    judge_model: str = Field(description="STABLE logical key of the judge LLM (e.g. 'claude-haiku-4-5'). Keyed on by the S3 path, the CloudWatch 'judge_model' dimension, and the custom_id tag — held constant across snapshot pins so the rolling-mean time series doesn't reset for a non-change. See evals/judge_models.py.")
    judge_request_model: str | None = Field(
        default=None,
        description=(
            "Exact model string sent to the Anthropic API (e.g. "
            "'claude-haiku-4-5-20251001'). Pinned to an immutable dated "
            "snapshot where Anthropic publishes one, else the alias. None "
            "on skip-marker artifacts (no LLM call) and on pre-L4578(a) "
            "records. See evals/judge_models.py."
        ),
    )
    judge_resolved_model: str | None = Field(
        default=None,
        description=(
            "Model string Anthropic RESOLVED the request to (response "
            "'model' field) — the authoritative record of what actually "
            "ran and the re-anchor trigger: a change here for a given "
            "judge_model signals a judge upgrade that breaks score "
            "comparability. None on skips and pre-L4578(a) records."
        ),
    )
    dimension_scores: list[RubricDimensionScore] = Field(
        default_factory=list,
        description=(
            "Per-dimension scores from the judge LLM. Empty list when "
            "``judge_skip_reason`` is set — the judge short-circuited "
            "before the LLM call because the captured ``agent_output`` "
            "was empty (e.g. sector_qual when upstream quant returned "
            "empty top5). Downstream consumers (CloudWatch metric "
            "emission, rolling-mean Lambda) already exclude empty "
            "lists from aggregation, so skipped records don't drag "
            "alarm thresholds toward the floor."
        ),
    )
    overall_reasoning: str = Field(
        default="",
        description=(
            "Cross-dimension summary from the judge LLM, or a short "
            "skip-reason sentence when ``judge_skip_reason`` is set."
        ),
    )
    judge_skip_reason: str | None = Field(
        default=None,
        description=(
            "Set when the judge short-circuited before invoking the LLM. "
            "Today's only value is ``'precluded_by_empty_upstream'`` — "
            "the captured ``agent_output`` was empty (None or {}) "
            "because the graph design bypassed the agent (e.g. "
            "sector_qual loop is skipped when ``quant_top5`` is empty). "
            "Distinct from the agent-ran-and-emitted-empty-output case "
            "(e.g. quant returning ``ranked_picks: []`` after running "
            "tools), which is a real agent failure to flag, not a "
            "structural skip — that lives in a separate retry+gate "
            "workstream. None when the judge actually ran the LLM."
        ),
    )
