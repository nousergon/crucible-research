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


REGIME_VALUES = ("bull", "neutral", "bear", "caution")
RegimeLiteral = Literal["bull", "neutral", "bear", "caution"]


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
    sector: str = "Unknown"
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


# ── LLM-extraction schemas (used by `with_structured_output()` at agent boundaries) ───────
#
# These are the typed shapes passed to ``llm.with_structured_output(Schema)``
# at each agent boundary. They are distinct from the state-shape schemas
# above:
#
#   - State schemas (SectorTeamOutput, CIODecision, InvestmentThesis, ...)
#     describe what flows through ``ResearchState`` after agent post-
#     processing — they include fields downstream nodes compute or merge
#     in (e.g. ``score_aggregator`` populates ``rating`` from sub-scores).
#
#   - LLM-extraction schemas (below) describe the EXACT shape the LLM
#     must emit. The agent code reads the typed model and may transform
#     it before writing to state. Keeping them separate avoids the
#     trap of asking the LLM to emit fields that downstream code is
#     supposed to compute.
#
# ``extra="allow"`` matches the state-schema convention so prompts that
# emit additional context (e.g. macro_economist's ``key_theme``) are
# preserved even when the schema doesn't enumerate them. PR 2 Step F may
# revisit this once the agent contracts have soaked.


class MacroEconomistRawOutput(BaseModel):
    """Wrapping schema for ``run_macro_agent`` output. The agent emits
    free-form prose (``report_md``) interleaved with a JSON block carrying
    structured fields; ``with_structured_output`` extracts both."""

    model_config = ConfigDict(extra="allow")

    report_md: str = ""
    market_regime: RegimeLiteral = "neutral"
    sector_modifiers: dict[str, float] = Field(default_factory=dict)
    sector_ratings: dict[str, dict] = Field(default_factory=dict)
    key_theme: str = ""
    material_changes: list[str] = Field(default_factory=list)

    @field_validator("sector_modifiers")
    @classmethod
    def clamp_modifiers(cls, v: dict[str, float]) -> dict[str, float]:
        """Mirror MacroEconomistOutput's clamp on the [0.70, 1.30] band."""
        for sector, m in v.items():
            if not (0.70 <= float(m) <= 1.30):
                raise ValueError(
                    f"sector_modifiers[{sector!r}]={m} outside [0.70, 1.30]"
                )
        return v


class MacroCriticOutput(BaseModel):
    """Reflection-loop critic output for the macro agent.

    The critic accepts or revises the macro_economist's draft. ``revise``
    triggers another macro_economist call; ``accept`` ends the loop.
    """

    model_config = ConfigDict(extra="allow")

    action: Literal["accept", "revise"]
    critique: str = ""
    suggested_regime: RegimeLiteral | None = None


class QuantPick(BaseModel):
    """One ranked candidate from the quant analyst's ReAct loop."""

    model_config = ConfigDict(extra="allow")

    ticker: str
    quant_score: float = Field(ge=0, le=100)
    rationale: str = ""
    key_metrics: dict = Field(default_factory=dict)


class QuantAnalystOutput(BaseModel):
    """Wrapper for the quant ReAct agent's structured response.

    LangGraph ``create_react_agent(response_format=...)`` runs an extra
    LLM call after the tool-loop terminates to extract this typed shape
    from the conversation."""

    model_config = ConfigDict(extra="allow")

    ranked_picks: list[QuantPick] = Field(default_factory=list)


class QualAssessment(BaseModel):
    """One per-ticker qualitative assessment from the qual analyst."""

    model_config = ConfigDict(extra="allow")

    ticker: str
    qual_score: float | None = Field(default=None, ge=0, le=100)
    bull_case: str = ""
    bear_case: str = ""
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    conviction: int | None = Field(default=None, ge=0, le=100)


class QualAnalystOutput(BaseModel):
    """Wrapper for the qual ReAct agent's structured response.

    ``additional_candidate`` is the qual-side proposal that the peer-review
    quant gate then accepts or rejects (see QuantAcceptanceVerdict).
    """

    model_config = ConfigDict(extra="allow")

    assessments: list[QualAssessment] = Field(default_factory=list)
    additional_candidate: QualAssessment | None = None


class QuantAcceptanceVerdict(BaseModel):
    """Side-LLM call: peer_review's quant analyst rules on whether to
    accept the qual analyst's added candidate."""

    model_config = ConfigDict(extra="allow")

    accept: bool
    reason: str = ""


class JointFinalizationDecision(BaseModel):
    """One per-ticker decision from peer_review's joint finalization.

    Per-ticker rationale enables LLM-as-judge eval to score the
    synthesis reasoning at decision granularity (rather than one
    rationale string covering all 2-3 picks). Composes with the
    LLM-as-judge workstream (ROADMAP Phase 2 P1).
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    rationale: str = Field(
        default="",
        description=(
            "Why this ticker was selected — name R/R reasoning, score "
            "asymmetry, catalyst. 1-2 sentences."
        ),
    )


class JointFinalizationOutput(BaseModel):
    """Side-LLM call: peer_review's joint quant+qual finalization, picks
    the team's 2-3 final recommendations from the merged candidate set.

    Per-ticker rationale lives on each ``selected_decisions`` entry;
    ``team_rationale`` carries cross-pick context (sector concentration,
    regime fit across the slate)."""

    model_config = ConfigDict(extra="allow")

    selected_decisions: list[JointFinalizationDecision] = Field(
        default_factory=list,
        description=(
            "Array of per-ticker selection decisions. Return one entry "
            "per selected ticker as a structured array (NOT a single "
            "JSON-encoded string). Each entry must be a JSON object "
            "with `ticker` and `rationale` fields."
        ),
    )
    team_rationale: str = Field(
        default="",
        description=(
            "Cross-pick rationale — sector concentration, regime fit "
            "across the slate, asymmetry mix. 1-2 sentences."
        ),
    )

    @field_validator("selected_decisions", mode="before")
    @classmethod
    def _parse_string_as_list(cls, v):
        """Defense for an observed Sonnet failure mode: the model
        occasionally returns ``selected_decisions`` as a JSON-encoded
        string instead of a structured array, even though the tool
        spec declares it as a list. First seen 2026-05-03 in SF
        ``eval-pipeline-validation-2`` where Sonnet returned
        ``'[\\n  {\\n    "ticker": "C..."\\n'`` — valid JSON inside a
        string wrapper.

        We log loudly (so flow-doctor surfaces the drift event in CW
        alarms) and parse-and-continue rather than hard-fail, because
        the downstream cost of a hard-fail is a wasted ~$5 Research
        run; the log entry preserves observability while the parse
        salvages the run. If the string isn't valid JSON list, fall
        through to the normal Pydantic list-type error so the failure
        mode stays loud.
        """
        if isinstance(v, str):
            import json
            import logging
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    logging.getLogger(__name__).warning(
                        "[joint_finalization_schema] LLM returned "
                        "selected_decisions as JSON-string of length %d "
                        "instead of a structured array; parsed-and-continued "
                        "(see schema-vs-LLM drift class).",
                        len(v),
                    )
                    return parsed
            except json.JSONDecodeError:
                pass
        return v


class HeldThesisUpdateLLMOutput(BaseModel):
    """LLM-extraction shape for ``_update_thesis_for_held_stock``.

    Intentionally narrative-only — NO score fields. The held-stock LLM
    update path must NOT overwrite prior_scores; the existing strip-nulls
    merge logic exists today specifically because the LLM occasionally
    emits ``final_score: null``. By omitting score fields from the schema
    entirely, the LLM cannot emit them, and the strip-nulls workaround
    becomes unnecessary.

    Field-level ``description`` strings are propagated by
    ``with_structured_output()`` into the tool-input schema the LLM sees,
    so the per-field length/count guidance previously inlined in the
    prompt body lives here now (audit finding F1, PR B 2026-05-02).
    """

    model_config = ConfigDict(extra="allow")

    bull_case: str = Field(default="", description="Bull case narrative — 1-2 sentences, ~200 chars.")
    bear_case: str = Field(default="", description="Bear case narrative — 1-2 sentences, ~200 chars.")
    catalysts: list[str] = Field(default_factory=list, description="Up to 5 catalysts.")
    risks: list[str] = Field(default_factory=list, description="Up to 5 risks.")
    conviction: int | None = Field(default=None, ge=0, le=100, description="Strength of view (0-100). ≥70 high, 40-69 moderate, <40 low.")
    conviction_rationale: str = Field(default="", description="Why this conviction level — ~100 chars.")
    thesis_summary: str = ""
    triggers_response: str = ""


# CIO emits the literal ``NO_ADVANCE_DEADLOCK`` for low-conviction picks
# that don't clear the floor; post-processing in ``_parse_cio_response``
# may synthesize ``ADVANCE_FORCED`` to fill below-floor open slots, but
# that synthesis happens AFTER the LLM extraction, so the raw schema
# only enumerates the three values the LLM is allowed to emit.
CIORawDecisionLiteral = Literal["ADVANCE", "REJECT", "NO_ADVANCE_DEADLOCK"]


class CIORawDecision(BaseModel):
    """One CIO decision as emitted by the LLM (pre-post-processing).

    Note ``decision`` (LLM-emitted) vs ``thesis_type`` (post-processed,
    used in ``CIODecision``): the LLM never emits ``HOLD`` directly —
    ``HOLD`` is what the post-processing maps ``REJECT`` to for held
    tickers in the current population. The two shapes are kept separate
    so each can describe its own contract precisely.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    decision: CIORawDecisionLiteral
    rank: int | None = Field(default=None, ge=0, description="1-based rank for ADVANCE picks; null for REJECT / NO_ADVANCE_DEADLOCK.")
    conviction: int | None = Field(default=None, ge=0, le=100, description="Strength of view (0-100).")
    rationale: str = Field(default="", description="Why this decision — name R/R reasoning (sub-scores, rr_ratio, catalyst).")
    entry_thesis: HeldThesisUpdateLLMOutput | None = Field(default=None, description="Required for ADVANCE; null for REJECT / NO_ADVANCE_DEADLOCK.")


class CIORawOutput(BaseModel):
    """Wrapper for the CIO agent's structured response. The list-of-
    decisions shape mirrors what ``_parse_cio_response`` consumes today
    via balanced-brace JSON extraction.

    ``min_length=1`` is propagated to the LLM via the structured-output
    tool schema description AND validated by the SDK parser. Caught
    2026-05-02: PR B's strip of the CIO prompt's inline JSON example
    let Sonnet emit ``decisions: []`` because the structural cue that
    "one entry per candidate" was lost. The prompt fix (config #21,
    explicit OUTPUT REQUIREMENT block) addresses the LLM-side cue;
    this constraint is the schema-side defense — empty list now
    surfaces as a parsing_error at the call boundary rather than as a
    later "empty decisions" raise inside ``run_cio``.
    """

    # ``validate_default=True`` ensures the ``min_length=1`` constraint
    # fires even when ``decisions`` falls back to ``default_factory=list``.
    # Pydantic v2 skips default validation by default; without this the
    # empty-list rejection only triggers when a caller explicitly passes
    # ``decisions=[]`` — defeating the schema-side defense.
    model_config = ConfigDict(extra="allow", validate_default=True)

    decisions: list[CIORawDecision] = Field(
        default_factory=list,
        min_length=1,
        description="One entry per input candidate. Never empty — every candidate must receive a decision (ADVANCE / REJECT / NO_ADVANCE_DEADLOCK).",
    )


# ── LLM-as-judge eval (PR 2 of P3.1 workstream) ───────────────────────────


class RubricDimensionScore(BaseModel):
    """One dimension's score from the eval judge.

    Score is integer 1-5 per the rubric anchors (see
    eval_rubric_*.txt prompts in alpha-engine-config). The ``reasoning``
    string carries the judge's per-dimension justification — used by
    the dashboard's quality-trend page to surface WHY scores dropped,
    not just THAT they dropped.
    """

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(description="Rubric dimension name (e.g. 'numerical_grounding', 'signal_calibration').")
    score: int = Field(ge=1, le=5, description="Integer score 1-5 per the rubric anchors.")
    reasoning: str = Field(description="1-2 sentence justification citing specific artifact content that drove the score.")


class RubricEvalLLMOutput(BaseModel):
    """LLM-extraction shape for the eval judge call.

    The judge LLM (Haiku or Sonnet) produces this against a rubric
    prompt + DecisionArtifact pair. Wrapped in ``RubricEvalArtifact``
    by ``evals.judge.evaluate_artifact`` before persisting to S3.
    """

    model_config = ConfigDict(extra="allow")

    dimension_scores: list[RubricDimensionScore] = Field(
        default_factory=list,
        min_length=1,
        description=(
            "Array of per-dimension score entries. Return one entry "
            "per rubric dimension as a structured array (NOT a single "
            "JSON-encoded string). Each entry must be a JSON object "
            "with `dimension`, `score`, and `reasoning` fields. Order "
            "matches the rubric prompt's dimension list."
        ),
    )
    overall_reasoning: str = Field(
        description="1-2 sentence cross-dimension summary — strongest signal + most concerning gap.",
    )

    @field_validator("dimension_scores", mode="before")
    @classmethod
    def _parse_string_as_list(cls, v):
        """Defense for an observed Haiku failure mode (first surfaced
        2026-05-03 in judge_only smoke against new-format Sat 5/3
        captures): the model occasionally returns ``dimension_scores``
        as a JSON-encoded string instead of a structured array, even
        though the tool spec declares it as a list. Same pattern PR
        #99 fixed for ``JointFinalizationOutput.selected_decisions``.

        We log loudly (so flow-doctor surfaces the drift event in CW
        alarms) and parse-and-continue rather than hard-fail, because
        the downstream cost of a hard-fail is a wasted ~$0.0001 judge
        call and a missing eval datapoint; the log entry preserves
        observability while the parse salvages the run. If the string
        isn't valid JSON list, fall through to the normal Pydantic
        list-type error so the failure mode stays loud.
        """
        if isinstance(v, str):
            import json
            import logging
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    logging.getLogger(__name__).warning(
                        "[rubric_eval_schema] LLM returned "
                        "dimension_scores as JSON-string of length %d "
                        "instead of a structured array; parsed-and-continued "
                        "(see schema-vs-LLM drift class).",
                        len(v),
                    )
                    return parsed
            except json.JSONDecodeError:
                pass
        return v


class RubricEvalArtifact(BaseModel):
    """One persisted eval result.

    Stored at ``decision_artifacts/_eval/{YYYY-MM-DD}/{judged_agent_id}/
    {run_id}.json`` (per ROADMAP §1630). Wraps the LLM output with
    metadata that ties it back to the judged decision artifact, the
    rubric prompt version, and the judge model.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str = Field(description="Pipeline-invocation identifier; ties this eval to the judged artifact's run_id.")
    timestamp: str = Field(description="ISO-8601 capture time (wall clock at the moment the wrapper writes to S3).")
    judged_agent_id: str = Field(description="agent_id of the DecisionArtifact this eval is scoring (e.g. 'sector_quant:technology').")
    judged_artifact_s3_key: str | None = Field(
        default=None,
        description="Backref to the judged artifact's S3 key. Optional — None if eval was run on an in-memory artifact.",
    )
    rubric_id: str = Field(description="Rubric prompt name (e.g. 'eval_rubric_sector_quant').")
    rubric_version: str = Field(description="Rubric prompt version at eval time (semver from prompt frontmatter).")
    judge_model: str = Field(description="Model name of the judge LLM (e.g. 'claude-haiku-4-5').")
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
