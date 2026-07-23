"""
LLM-as-judge evaluation pipeline.

Reads a captured ``DecisionArtifact``, looks up the matching rubric
prompt, sends ``(rubric, artifact_input, artifact_output)`` to a judge
LLM (Haiku-tier default; Sonnet-tier for the nuance-tier sampled subset —
see the alpha-engine-config-I2997 note below for what these tiers now
route to), and persists the structured eval result to S3.

Eval is observability, NOT a gate. Runs proceed regardless of eval
score; the eval corpus + dashboard surface quality regressions weeks
before they show up in alpha-vs-SPY.

Two execution paths share the rubric-rendering + parsing core:

* ``evaluate_artifact`` — synchronous single-artifact path. Used by
  ad-hoc replays, the judge_only test track, dry_run smoke, and the
  Sonnet-escalation tail in the batch Process Lambda.

* ``build_batch_request`` + ``parse_batch_message`` — Anthropic
  Message Batches API path. Used by the Saturday SF Submit/Poll/
  Process chain to fan a single batch over every (artifact × judge_model)
  pair, then stream and persist results. 50% cost discount per the
  Batches API contract; structurally bypasses the Lambda 15-min timeout
  class that nearly fired on the 2026-05-06 manual midweek SF run.
  **UNCHANGED by alpha-engine-config-I2997** — Brian's 2026-07-19 ruling
  keeps EvalJudge Batches on direct Anthropic as the sole deliberate
  exception to the fleet's off-direct-Anthropic migration (retains the
  50% batch discount).

Composes with:
- Decision-artifact capture (alpha_engine_lib.decision_capture).
- Rubric prompts in alpha-engine-config (eval_rubric_*.txt at
  version 1.0.0+, loaded via ``agents.prompt_loader.load_prompt``).
- Cost telemetry — the BATCH path still emits
  ``agent_id="eval_judge"`` telemetry via ``track_llm_cost``-shaped
  per-result usage from Anthropic's batch results stream. The SYNC
  path (``evaluate_artifact``) does NOT integrate with ``track_llm_cost``
  post-migration — see its docstring; this mirrors the pre-existing
  ``evaluate_artifact_openrouter`` shadow tier, which never had this
  integration either (config#2575).

**alpha-engine-config-I2997 (2026-07-19):** ``evaluate_artifact`` (the sync
path above) migrated off direct Anthropic (``ChatAnthropic``) to the
OpenRouter/DeepSeek transport that ``evaluate_artifact_openrouter`` already
used as its shadow tier (config#2575) — see both functions' docstrings for
the full rationale, including why the Haiku/Sonnet ``judge_model`` logical
keys are PRESERVED (S3 path / CloudWatch dimension / rolling-mean identity)
even though both tiers now physically call the SAME OpenRouter model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import boto3
from krepis.judge import JudgeToolCallLeakError, check_openai_tool_response_for_leak
from krepis.judge import ToolResultNotFoundError as _LibToolResultNotFoundError
from krepis.judge import build_structured_tool_spec as _lib_build_tool_spec
from krepis.judge import decode_custom_id as _lib_decode_custom_id
from krepis.judge import encode_custom_id as _lib_encode_custom_id
from krepis.judge import parse_batch_tool_result as _lib_parse_batch_tool_result
from krepis.judge import render_rubric as _lib_render_rubric
from krepis.llm_config import ModelSpec
from nousergon_lib.decision_capture import DecisionArtifact
from nousergon_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
from openai import OpenAI

from agents.prompt_loader import LoadedPrompt, load_prompt
from config import MAX_TOKENS_STRATEGIC, OPENROUTER_API_KEY, S3_BUCKET
from evals.judge_models import OPENROUTER_SHADOW, TAG_BY_LOGICAL, request_model_for
from graph.state_schemas import (
    RubricEvalArtifact,
    RubricEvalLLMOutput,
)

logger = logging.getLogger(__name__)


def _new_judge_run_id() -> str:
    """Mint a fresh judge_run_id for a judge batch invocation.

    Production paths generate one of these at the start of a batch and
    propagate it to every RubricEvalArtifact emitted by that batch.
    Solo / replay / smoke callers get a fresh id per call.

    Delegates to ``alpha_engine_lib.eval_artifacts.new_eval_run_id``
    (config#793 canonical-layout swap) — returns a ``YYMMDDHHMM``
    structured-timestamp string (sortable, human-readable) rather than
    the legacy UUIDv4. The timestamp encoding lets operators see when a
    batch ran straight from the S3 path listing, and lexicographic sort
    across the flat ``_eval/`` prefix yields chronological order without
    a date sub-partition.

    Same-minute collisions are by design (production cron cadence makes
    them effectively impossible); see the lib docstring. Tests inject
    explicit ``judge_run_id`` strings where determinism is needed.
    """
    return new_eval_run_id()


# ── Defaults ──────────────────────────────────────────────────────────────


DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
"""Default judge model — Haiku for cost on every weekly run.

This is the STABLE logical key, not the API request string: the actual
request is pinned to the dated snapshot via
``judge_models.request_model_for`` (L4578(a)), while this logical key
stays constant for the S3 path / CloudWatch dimension / custom_id tag so
a snapshot pin doesn't reset the rolling-mean time series. Sonnet
(``claude-sonnet-4-6``) is the nuance-tier judge on the sampled subset.
See ``evals/judge_models.py`` for the pin + re-anchor protocol."""

DEFAULT_MAX_TOKENS = MAX_TOKENS_STRATEGIC
"""Token cap for the judge response. Routes through the strategic-tier
constant per the consolidation in PR #102 (4 hardcoded literals
replaced; CI lint guard prevents drift). Synthesis-class output:
5-6 dimension entries × verbose reasoning + overall_reasoning, plus
tool-use envelope. Bumped from 1500 hardcoded on 2026-05-03 after
judge_only smoke against Sat 5/3 captures showed ~5/32 evals failed
with truncated/stringified dimension_scores at the prior 1500 cap."""

MAX_JUDGE_RETRIES = 3
"""Max attempts to parse a judge LLM response into ``RubricEvalLLMOutput``.

LLM tool-use is stochastically non-conformant — Haiku occasionally
produces malformed JSON-as-string for ``dimension_scores`` even with
the schema's Field description + ``mode='before'`` validator (PR
#104). Retrying gets a fresh decoder sample; ~20% per-call failure
→ ~0.8% after 3 attempts. Token cost is bounded: only failed attempts
retry, so the premium is paid only on the failing tail.

Caps at 3 to bound worst-case latency (each retry is a full Haiku
call ≈ 3-8s). Beyond 3 attempts the underlying issue is structural
(rubric prompt too dense, model regressed, etc.) and surfaces as a
loud failure for the operator to diagnose."""


# ── Agent → rubric mapping ────────────────────────────────────────────────


def resolve_rubric_for_agent(agent_id: str) -> str | None:
    """Return the rubric prompt name for ``agent_id``, or ``None`` if
    the agent type is intentionally unevaluated.

    Mapping mirrors the captured agent_id taxonomy (see
    research_graph.sector_team_node + cio_node + macro_economist_node):

      sector_quant:{team_id}        → eval_rubric_sector_quant
      sector_qual:{team_id}         → eval_rubric_sector_qual
      sector_peer_review:{team_id}  → eval_rubric_sector_peer_review
      macro_economist               → eval_rubric_macro_economist
      ic_cio                        → eval_rubric_ic_cio
      thesis_update:{team}:{ticker} → eval_rubric_thesis_update

    Unknown agent_ids return None so the caller can skip cleanly
    rather than crash on rubric lookup.

    The thesis_update rubric was added 2026-05-05 after confirming the
    held-stock update is alpha-load-bearing: executor's position_sizer
    reads conviction (0.7× multiplier on declining); eod_reconcile reads
    bull_case (EOD email rationale). Silent regression in this output
    directly costs alpha through wrong sizing on held positions, so the
    rubric makes the regression visible weeks before it shows up in
    alpha-vs-SPY.
    """
    if agent_id.startswith("sector_quant:"):
        return "eval_rubric_sector_quant"
    if agent_id.startswith("sector_qual:"):
        return "eval_rubric_sector_qual"
    if agent_id.startswith("sector_peer_review:"):
        return "eval_rubric_sector_peer_review"
    if agent_id.startswith("thesis_update:"):
        return "eval_rubric_thesis_update"
    if agent_id == "macro_economist":
        return "eval_rubric_macro_economist"
    if agent_id == "ic_cio":
        return "eval_rubric_ic_cio"
    # Think-tank family (config#1579 P2) — deliberately COARSE ids (not
    # per-ticker/per-theme) so the rolling-mean floor's >=3-samples-per-combo
    # gate is met; ticker/theme identity rides in run_id + the snapshot.
    if agent_id == "thinktank_thesis":
        return "eval_rubric_thinktank_thesis"
    if agent_id == "thinktank_theme":
        return "eval_rubric_thinktank_theme"
    return None


# ── Custom-id codec ───────────────────────────────────────────────────────
#
# The Anthropic Batches API returns results keyed by an opaque
# ``custom_id`` (1-64 chars, ``^[a-zA-Z0-9_-]{1,64}$``). The Submit
# Lambda encodes the (judged_agent_id, run_id, judge_model) tuple into
# the custom_id; the Process Lambda decodes it on the way out so the
# eval artifact can be persisted under the same path the sync path
# would have written. Encoding is round-trippable so we don't depend
# on the in-flight plan manifest for correctness — the manifest is a
# convenience for ops visibility, not a load-bearing dependency.
#
# config#1675 / config#2575 lift (2026-07-15): the codec MECHANICS now
# live in ``krepis.judge.encode_custom_id`` / ``decode_custom_id``
# (generalized ``subject_id`` naming since the lib is agent-pipeline
# agnostic). This module keeps the ``judged_agent_id``-named wrapper
# functions so existing call sites are unchanged, and keeps
# ``_CUSTOM_ID_PATTERN`` / ``_JUDGE_MODEL_TAG`` as module-level names
# other tests inspect directly.


_CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
"""Anthropic batch custom_id regex (per Message Batches API docs). Mirrors
``krepis.judge._CUSTOM_ID_PATTERN`` — kept here too since
``test_eval_judge_batch.py`` imports this name directly."""

_JUDGE_MODEL_TAG = TAG_BY_LOGICAL
"""Compact tags for the judge models, keyed by logical key. Sourced from
``judge_models.TAG_BY_LOGICAL`` so the tag map can't drift from the
registry. Keeps custom_id under the 64-char limit even when
judged_agent_id is long (e.g. ``thesis_update:technology:AAPL``)."""


def encode_custom_id(
    *, judged_agent_id: str, run_id: str, judge_model: str,
) -> str:
    """Encode (judged_agent_id, run_id, judge_model) → batch custom_id.

    Replaces ``:`` and ``/`` separators with ``-`` since the Anthropic
    custom_id charset only allows alphanumerics, ``-``, and ``_``.
    Truncates the agent_id segment if needed so the final string fits
    the 64-char ceiling. Round-trippable via ``decode_custom_id``.

    Delegates to ``krepis.judge.encode_custom_id`` (config#2575 lift).
    """
    return _lib_encode_custom_id(
        subject_id=judged_agent_id, run_id=run_id, judge_model=judge_model,
        tag_by_logical=_JUDGE_MODEL_TAG,
    )


def decode_custom_id(custom_id: str) -> tuple[str, str, str]:
    """Inverse of ``encode_custom_id``.

    Returns ``(judged_agent_id, run_id, judge_model)``. The agent_id
    reconstruction maps ``-`` back to ``:`` for the prefixed-team
    pattern (``sector_quant:tech``); other ``-`` characters in the
    original would be lost (acceptable — the batch plan manifest carries
    the canonical agent_id and the eval artifact stamps it from there).

    Raises ``ValueError`` if the custom_id doesn't match the expected
    triple-segment shape (defensive — should not happen in production
    since we control both sides of the codec).

    Delegates to ``krepis.judge.decode_custom_id`` (config#2575 lift).
    """
    return _lib_decode_custom_id(custom_id, tag_by_logical=_JUDGE_MODEL_TAG)


# ── Render + parse helpers ────────────────────────────────────────────────


def _render_rubric(
    artifact: DecisionArtifact, loaded_prompt: LoadedPrompt,
) -> str:
    """Render the rubric template against the artifact's payload.

    ``json.dumps(..., default=str)`` handles any stray types
    (datetimes, Decimals) that snuck into the captured snapshot.
    Shared by the sync and batch paths so rubric rendering is
    semantically identical regardless of which transport delivers
    the call.

    Delegates to ``krepis.judge.render_rubric`` (config#2575 lift) —
    ``loaded_prompt.format`` is a plain ``str.format`` wrapper
    (``agents/prompt_loader.py::LoadedPrompt.format``), so
    ``loaded_prompt.text`` is the equivalent plain-string template the
    lib function expects.
    """
    return _lib_render_rubric(
        loaded_prompt.text,
        agent_input=artifact.input_data_snapshot,
        agent_output=artifact.agent_output,
    )


def _make_skip_eval_artifact(
    artifact: DecisionArtifact,
    *,
    rubric_name: str,
    rubric_version: str,
    judge_model: str,
    judge_run_id: str,
    judged_artifact_s3_key: str | None,
    skip_reason: str = "precluded_by_empty_upstream",
    overall_reasoning: str | None = None,
) -> RubricEvalArtifact:
    """Build the skip-marker eval for an artifact that should not be
    scored by the rubric. Shared by the sync + batch paths.

    Two skip reasons supported:

    * ``precluded_by_empty_upstream`` (existing) — ``agent_output`` is
      empty because the graph bypassed the agent (typically sector_qual
      or sector_peer_review when upstream quant_top5 is empty).

    * ``degenerate_input`` (added 2026-05-13) — the agent ran and
      produced an output, but its inputs were degenerate (e.g.
      thesis_update fired on a held ticker with empty prior summary,
      zero news, null analyst data). Scoring "thesis_completeness=5"
      on such a run is misleading: the output IS complete-looking but
      it's fabrication from nothing, not a substantive update. See
      :func:`_is_degenerate_input` for per-rubric definitions.

    In both cases, short-circuiting BEFORE the LLM call means we pay
    no token cost and emit no spurious scores into the CW
    ``agent_quality_score`` metric stream that drives the
    rolling-4-week alarm.
    """
    default_reasoning = {
        "precluded_by_empty_upstream": (
            "Judge short-circuited: captured agent_output is empty "
            "(graph bypassed the agent — typically sector_qual or "
            "sector_peer_review when upstream quant_top5 is empty). "
            "No work to evaluate; not a quality regression."
        ),
        "degenerate_input": (
            "Judge short-circuited: input_data_snapshot is degenerate "
            "for this rubric (see evals/judge._is_degenerate_input for "
            "per-rubric definitions). Scoring the output would conflate "
            "structural completeness with substantive content — emitting "
            "a high score into the CW metric stream would mask the "
            "upstream substrate gap that produced the degenerate input."
        ),
    }
    return RubricEvalArtifact(
        run_id=artifact.run_id,
        judge_run_id=judge_run_id,
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        judged_agent_id=artifact.agent_id,
        judged_artifact_s3_key=judged_artifact_s3_key,
        rubric_id=rubric_name,
        rubric_version=rubric_version,
        judge_model=judge_model,
        dimension_scores=[],
        overall_reasoning=overall_reasoning or default_reasoning.get(
            skip_reason,
            f"Judge short-circuited: {skip_reason}.",
        ),
        judge_skip_reason=skip_reason,
    )


def _is_degenerate_input(artifact: DecisionArtifact) -> bool:
    """Return True if the artifact's ``input_data_snapshot`` is
    degenerate for its rubric type — i.e. the inputs are so empty that
    scoring the output measures nothing real.

    Per-rubric definitions (added 2026-05-13 alongside the L83
    spot-check P0 — substrate-side fix in
    ``graph.research_graph._pre_fetch_held_enrichment`` for thesis_update):

    * **thesis_update:** prior_thesis.thesis_summary empty AND
      news_data.articles empty AND analyst_data null. The agent has
      nothing to update against and is fabricating bull/bear cases on
      the trigger alone.

    * **sector_quant / sector_qual:** sector_tickers_count == 0 OR
      technical_scores_team empty (sector_quant) /
      sector_population empty (sector_qual). No candidates to screen.

    * **sector_peer_review:** quant_picks empty AND qual_picks empty.
      Nothing to merge.

    * **macro_economist / ic_cio:** never degenerate. macro always has
      a regime call; ic_cio always has a candidate slate of some
      shape — return False unconditionally. Explicit pass-through so
      a future rubric author can't accidentally widen the gate.

    Unknown agent types: return False (don't skip — fall through to
    the normal rubric path).
    """
    snap = artifact.input_data_snapshot or {}
    agent_id = artifact.agent_id

    if agent_id.startswith("thesis_update:"):
        prior = snap.get("prior_thesis") or {}
        if not isinstance(prior, dict):
            prior = {}
        prior_summary = (prior.get("thesis_summary") or "").strip()
        news = snap.get("news_data") or {}
        if not isinstance(news, dict):
            news = {}
        news_articles = news.get("articles") or []
        analyst = snap.get("analyst_data")
        # Treat a skeleton analyst dict (all None / all empty lists) as
        # degenerate — ``fetch_analyst_consensus`` returns the skeleton
        # when FMP is unavailable. ``bool(value)`` short-circuits both
        # ``None`` and empty containers; we want at least one field with
        # actual content.
        #
        # config#1821 Option B (2026-07-08): consensus_rating / mean_target
        # / num_analysts / rating_changes were removed from
        # fetch_analyst_consensus's returned shape (the FMP endpoints that
        # populated them 402'd for every ticker on the current plan).
        # earnings_surprises is the only field left to check.
        analyst_is_substantive = (
            isinstance(analyst, dict)
            and bool(analyst.get("earnings_surprises"))
        )
        return not prior_summary and not news_articles and not analyst_is_substantive

    if agent_id.startswith("sector_quant:"):
        # Degenerate iff sector is empty AND technical_scores are
        # empty. Use ``sector_tickers`` list length as the authoritative
        # signal — older snapshots may not have ``sector_tickers_count``.
        has_tickers = bool(
            snap.get("sector_tickers")
            or int(snap.get("sector_tickers_count") or 0) > 0
        )
        has_scores = bool(snap.get("technical_scores_team"))
        return not (has_tickers or has_scores)

    if agent_id.startswith("sector_qual:"):
        has_tickers = bool(
            snap.get("sector_tickers")
            or snap.get("sector_population")
            or int(snap.get("sector_tickers_count") or 0) > 0
        )
        return not has_tickers

    if agent_id.startswith("sector_peer_review:"):
        return (
            not (snap.get("quant_picks") or [])
            and not (snap.get("qual_picks") or [])
        )

    # macro_economist + ic_cio + anything else: never degenerate.
    return False


# ── Tool-use spec for the batch path ──────────────────────────────────────
#
# The sync path uses LangChain's ``with_structured_output(...)`` which
# automatically derives a tool spec from ``RubricEvalLLMOutput``. The
# batch path uses the raw Anthropic SDK (LangChain doesn't wrap the
# Batches endpoint), so we synthesize the same tool spec here from the
# Pydantic model's JSON schema. Pinning the spec via the live model
# means a schema bump (e.g. adding a new dimension field) automatically
# flows into the batch tool — no second source of truth.


_RUBRIC_TOOL_NAME = "RubricEvalLLMOutput"
"""Tool name in the batch tool-use spec. Mirrors the LangChain default
(class name) so any callsite that grep-greps for the tool name finds
both paths."""

_RUBRIC_TOOL_DESCRIPTION = (
    "Emit the rubric eval as a structured tool call. Each rubric "
    "dimension produces one entry in dimension_scores with an integer "
    "score and short reasoning. overall_reasoning is a 1-2 sentence "
    "cross-dimension summary."
)


def _build_rubric_tool_spec() -> dict[str, Any]:
    """Synthesize the Anthropic tool-use spec for the rubric eval call.

    Pinning the input_schema to ``RubricEvalLLMOutput.model_json_schema()``
    means the schema-bump path is single-source-of-truth: edit the
    Pydantic model and both transports pick it up.

    Delegates to ``krepis.judge.build_structured_tool_spec`` (config#2575
    lift) — schema-agnostic in the lib (accepts any Pydantic model), so
    this wrapper is the one place that pins it to ``RubricEvalLLMOutput``.
    """
    return _lib_build_tool_spec(
        RubricEvalLLMOutput,
        tool_name=_RUBRIC_TOOL_NAME,
        description=_RUBRIC_TOOL_DESCRIPTION,
    )


def build_batch_request(
    artifact: DecisionArtifact,
    *,
    judge_model: str,
    custom_id: str,
    max_tokens: int = MAX_TOKENS_STRATEGIC,
) -> dict[str, Any]:
    """Build one entry of the ``messages.batches.create`` ``requests``
    array for an artifact under a given judge model.

    Uses the same rubric resolution + rendering as the sync path. Tool
    use is set up so the LLM is forced to emit the eval via
    ``RubricEvalLLMOutput``'s schema (matches the sync
    ``with_structured_output`` call shape — same structured-output
    semantics, just transported via the Batches API).

    Delegates payload construction to
    ``alpha_engine_lib.anthropic_payload.build_batches_request_params``
    (L334 chokepoint — second consumer of the lib's anthropic_payload
    substrate after morning-signal). The chokepoint enforces the
    server-tool ⊥ assistant-prefill invariant on the embedded
    ``params`` dict so future RubricEval extensions that add a server
    tool (web_search etc.) can't reach Anthropic's HTTP 400.

    Raises ``ValueError`` if ``artifact.agent_id`` has no rubric
    mapped — callers must pre-filter via ``resolve_rubric_for_agent``.

    Empty-input short-circuit is handled by the orchestrator BEFORE
    this function is invoked (the skip artifact is persisted
    client-side without spending a batch slot).
    """
    from nousergon_lib.anthropic_payload import build_batches_request_params

    rubric_name = resolve_rubric_for_agent(artifact.agent_id)
    if rubric_name is None:
        raise ValueError(
            f"No rubric mapped for agent_id={artifact.agent_id!r}. "
            f"Pre-filter with resolve_rubric_for_agent() before building "
            f"a batch request."
        )

    loaded_prompt = load_prompt(rubric_name)
    rendered = _render_rubric(artifact, loaded_prompt)
    tool_spec = _build_rubric_tool_spec()

    # Force the model to call the rubric tool — equivalent to
    # ``with_structured_output(...)``'s ``tool_choice`` posture. Without
    # this the model can decide to emit prose, which would fall through
    # every parser in this module.
    #
    # ``judge_model`` is the logical key; pin it to the dated snapshot
    # for the actual API call (L4578(a)). The custom_id (built by the
    # caller) keeps the logical key so persistence/dimension stay stable.
    return build_batches_request_params(
        custom_id=custom_id,
        model=request_model_for(judge_model),
        max_tokens=max_tokens,
        user_content=rendered,
        tools=[tool_spec],
        tool_choice={"type": "tool", "name": _RUBRIC_TOOL_NAME},
    )


def parse_batch_message(
    message_payload: Any,
) -> RubricEvalLLMOutput:
    """Parse one batch result's ``message`` block into ``RubricEvalLLMOutput``.

    Accepts either an SDK Message object or its dict equivalent (the
    Process Lambda streams the raw dict to keep dependencies minimal).
    Locates the tool_use block named ``RubricEvalLLMOutput`` and
    validates its ``input`` against the Pydantic schema.

    Raises ``ValueError`` if no matching tool_use block is found, or
    ``pydantic.ValidationError`` if the input fails schema validation.
    Both are caught by the orchestrator and recorded in the run's
    ``failed`` list — the batch result is preserved on Anthropic's
    side (29-day retention) so the operator can re-pull and diagnose
    without re-paying for the call.

    Delegates to ``krepis.judge.parse_batch_tool_result`` (config#2575
    lift). Only the lib's ``ToolResultNotFoundError`` (tool never
    called) is re-raised with this module's Anthropic-retention-window
    detail — a ``pydantic.ValidationError`` (tool called, input failed
    schema validation) is a DIFFERENT failure mode and propagates
    unwrapped so callers/tests can still distinguish the two (both are
    ``ValueError`` subclasses, so catching bare ``ValueError`` here
    would incorrectly conflate them).
    """
    try:
        return _lib_parse_batch_tool_result(
            message_payload,
            tool_name=_RUBRIC_TOOL_NAME,
            schema=RubricEvalLLMOutput,
        )
    except _LibToolResultNotFoundError:
        raise ValueError(
            "No tool_use block named "
            f"{_RUBRIC_TOOL_NAME!r} found in batch result message; the "
            "judge LLM did not emit the rubric eval via the structured "
            "tool — inspect the raw batch result on Anthropic's side "
            "(retained 29 days)."
        ) from None


# ── Judge call ────────────────────────────────────────────────────────────


def evaluate_artifact(
    artifact: DecisionArtifact,
    *,
    judge_run_id: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    judged_artifact_s3_key: str | None = None,
    max_retries: int = MAX_JUDGE_RETRIES,
) -> RubricEvalArtifact:
    """Judge a single ``DecisionArtifact`` against its rubric — the sync
    primary path (Haiku/Sonnet ``judge_model`` tiers).

    Resolves the rubric for ``artifact.agent_id``, renders the rubric
    prompt with the artifact's ``input_data_snapshot`` + ``agent_output``,
    and invokes the judge LLM. Retries up to ``max_retries`` times on parse
    failures (LLM output is stochastically non-conformant — fresh decoder
    sample on retry typically succeeds). The returned ``RubricEvalArtifact``
    carries the dimension scores plus metadata (rubric_id+version,
    judge_model, judged_agent_id).

    **alpha-engine-config-I2997 (2026-07-19): migrated off direct Anthropic
    (``ChatAnthropic``) to the OpenRouter/DeepSeek transport, reusing the
    EXACT same tool-forced-structured-output + leak-guard + bounded-retry
    call core ``evaluate_artifact_openrouter`` already validated live
    (config#2575) — see ``_call_openrouter_judge_llm``.**

    ``judge_model`` (``"claude-haiku-4-5"`` / ``"claude-sonnet-4-6"``) is
    PRESERVED as the persisted logical key — it is the STABLE identity for
    the S3 eval-artifact path / CloudWatch dimension / rolling-mean time
    series (see ``evals/judge_models.py``'s docstring); changing it would
    reset those series for a non-semantic reason. Per Brian's ruling
    ("model per [evaluate_artifact_openrouter]'s existing default — it
    already uses DeepSeek, keep consistent"), BOTH the Haiku and Sonnet
    tiers now physically call the SAME OpenRouter model
    (``evals.judge_models.OPENROUTER_SHADOW.request_model`` —
    ``deepseek/deepseek-v4-flash``) rather than gaining a new bespoke
    Flash/Pro split; this collapses the two tiers' PHYSICAL distinction
    (their ``judge_model`` identity, S3 path, and CloudWatch dimension stay
    separate) — flagged prominently in the alpha-engine-config-I2997 PR
    body as a real behavior change worth Brian's explicit awareness. The
    ``judge_resolved_model``/re-anchor mechanism (see judge_models.py) is
    exactly the protocol this system already has for "same logical key,
    new backing model" — this is that mechanism engaging as designed, not
    a workaround.

    The ``request_model_for(judge_model)`` Anthropic-snapshot-pinning
    indirection is UNCHANGED and still used by the Batches path
    (``build_batch_request``) for the HAIKU/SONNET specs — this function no
    longer calls it; ``judge_request_model`` on the returned artifact now
    records the ACTUAL OpenRouter model string that was called.

    On every parse-failure attempt, the raw payload head is logged at
    WARNING so production failures are diagnosable without re-running the
    artifact.

    Cost telemetry: UNLIKE the pre-migration Anthropic path, this call does
    NOT integrate with ``track_llm_cost`` — the OpenRouter transport core
    is shared with ``evaluate_artifact_openrouter``, which never had that
    integration either (config#2575; it logs a plain INFO
    ``persisted-cost`` line instead). Per-call cost is therefore visible in
    CloudWatch Logs but NOT in the ``track_llm_cost`` S3 JSONL / dashboard
    LLM-cost surface for this specific call site — a known, flagged gap
    from the alpha-engine-config-I2997 migration (see its PR body for the
    tracked follow-up).

    Raises:
      - ``ValueError`` if no rubric is mapped for the artifact's
        agent_id — callers should pre-filter via ``resolve_rubric_for_agent``
        when iterating a mixed batch.
      - ``RuntimeError`` if all ``max_retries`` attempts fail (leak guard
        trip or schema validation failure) — the underlying issue is
        structural (model regression, rubric too dense, etc.) and surfaces
        as a loud failure for diagnosis.
    """
    rubric_name = resolve_rubric_for_agent(artifact.agent_id)
    if rubric_name is None:
        raise ValueError(
            f"No rubric mapped for agent_id={artifact.agent_id!r}. "
            f"Pre-filter with resolve_rubric_for_agent() if iterating "
            f"a mixed batch."
        )

    # Default judge_run_id when caller didn't pass one. Production paths
    # (orchestrator batch + escalation tail) pass an explicit batch-scoped
    # UUID so all artifacts emitted by one batch cluster under a single
    # judge_run_id directory. Solo callers (replay, ad-hoc smoke) get a
    # fresh UUID per call — works but loses batch cohesion since each
    # artifact lands at its own judge_run_id directory.
    judge_run_id = judge_run_id or _new_judge_run_id()

    # Empty-input short-circuit. When the captured ``agent_output`` is
    # falsy (None or {}), the agent never produced anything to evaluate
    # — the most common case is ``sector_qual:{team}`` whose loop is
    # bypassed by graph design when the upstream ``quant_top5`` is
    # empty (and ``sector_peer_review:{team}`` cascading off that).
    # Asking the judge to score "no rationale, no citations, no
    # synthesis" produces uniform 1/1/1/1 outputs that drag the
    # rolling-mean alarm threshold toward the floor without any real
    # quality regression. Short-circuit BEFORE the LLM call so we pay
    # no token cost and emit no spurious low scores. Distinct from the
    # agent-ran-and-emitted-empty-output case (e.g. quant returning
    # ``ranked_picks: []`` after iterating its tools) — that capture
    # has work to evaluate (tool_calls, iterations, the empty result
    # itself) and is the agent-failure signal we WANT the judge to
    # surface, not skip. We detect structural-skip via the broader
    # ``not agent_output`` check (catches None + {} but lets through
    # any non-empty payload).
    loaded_prompt = load_prompt(rubric_name)

    if not artifact.agent_output:
        return _make_skip_eval_artifact(
            artifact,
            rubric_name=rubric_name,
            rubric_version=loaded_prompt.version,
            judge_model=judge_model,
            judge_run_id=judge_run_id,
            judged_artifact_s3_key=judged_artifact_s3_key,
            skip_reason="precluded_by_empty_upstream",
        )

    # Input-sufficiency gate (added 2026-05-13, ROADMAP P0).
    # If the inputs are degenerate (per-rubric definition), scoring the
    # structurally-complete-but-fabricated output would emit a
    # misleading high score into the CW metric stream that drives the
    # quality-regression alarm. Skip BEFORE the LLM call.
    if _is_degenerate_input(artifact):
        logger.info(
            "[eval_judge] degenerate_input skip — agent_id=%s "
            "(see evals/judge._is_degenerate_input for per-rubric definition)",
            artifact.agent_id,
        )
        return _make_skip_eval_artifact(
            artifact,
            rubric_name=rubric_name,
            rubric_version=loaded_prompt.version,
            judge_model=judge_model,
            judge_run_id=judge_run_id,
            judged_artifact_s3_key=judged_artifact_s3_key,
            skip_reason="degenerate_input",
        )

    rendered = _render_rubric(artifact, loaded_prompt)

    # ``judge_model`` stays the stable logical key (persisted + dimension —
    # see docstring). The ACTUAL request model is the OpenRouter default
    # ``evaluate_artifact_openrouter`` already uses — deliberately the SAME
    # for both Haiku and Sonnet tiers per Brian's ruling (see docstring).
    request_model = OPENROUTER_SHADOW.request_model
    call_result = _call_openrouter_judge_llm(
        rendered,
        agent_id=artifact.agent_id,
        request_model=request_model,
        max_tokens=max_tokens,
        api_key=api_key,
        max_retries=max_retries,
        log_prefix="[eval_judge]",
    )

    logger.info(
        "[eval_judge] persisted-cost agent_id=%s judge_model=%s "
        "request_model=%s resolved_model=%s provider_cost_usd=%.6f",
        artifact.agent_id, judge_model, request_model,
        call_result.resolved_model, call_result.total_usd,
    )

    return RubricEvalArtifact(
        run_id=artifact.run_id,
        judge_run_id=judge_run_id,
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        judged_agent_id=artifact.agent_id,
        judged_artifact_s3_key=judged_artifact_s3_key,
        rubric_id=rubric_name,
        rubric_version=loaded_prompt.version,
        judge_model=judge_model,
        judge_request_model=request_model,
        judge_resolved_model=call_result.resolved_model,
        dimension_scores=call_result.llm_output.dimension_scores,
        overall_reasoning=call_result.llm_output.overall_reasoning,
    )


# ── OpenRouter judge transport core (config#2575 items 2-3; shared with the
#    sync primary path since alpha-engine-config-I2997, 2026-07-19) ───────
#
# Runs the SAME rubric/artifact through the SAME ``RubricEvalArtifact``
# output shape as the pre-migration Anthropic path (no bespoke third judge
# implementation, per config#2575's binding constraint carried forward
# from config#1676/#1675), but via a bare ``openai.OpenAI`` client pointed
# at OpenRouter with a FORCED tool call (``tool_choice``) instead of
# LangChain's ``ChatAnthropic`` — the Anthropic-specific
# ``with_structured_output`` + ``invoke_structured_with_validation_retry``
# chokepoint has no OpenRouter equivalent. (Not
# ``krepis.llm.LLMClient.structured()`` either: live-verified
# 2026-07-19 that OpenRouter's strict ``response_format=json_schema`` mode
# — that method's ``structured_outputs=True`` path — is unreliable for
# DeepSeek-family models, intermittently renaming/dropping required field
# names; forced tool-calling, as implemented here, is the mechanism this
# repo has already validated live for the judge's schema, config#2575.)
#
# ``evaluate_artifact_openrouter`` is SHADOW-only (see its own docstring);
# ``evaluate_artifact`` (the sync primary path) uses this SAME call core
# but is NOT shadow — it keeps full decision authority under its
# pre-existing Haiku/Sonnet ``judge_model`` logical keys.

MAX_OPENROUTER_JUDGE_RETRIES = 3
"""Same attempt budget as ``MAX_JUDGE_RETRIES`` (the pre-migration Anthropic
path's chokepoint) — kept as a distinct constant since the retry UNIT
differs: each attempt here is a fresh forced-tool-call request PLUS this
module's own leak-guard gate (``check_openai_tool_response_for_leak``),
not a langchain correction turn. Caps worst-case latency at 3 full model
calls."""


def _openrouter_judge_model_spec(*, request_model: str, max_tokens: int) -> ModelSpec:
    """Build the ``ModelSpec`` for an OpenRouter judge call.

    ``reasoning={"exclude": True}`` is NOT the default here — live
    validation (config#2575, 2026-07-18) confirmed a reasoning-capable
    OpenRouter model can burn its entire budget on chain-of-thought before
    ever emitting the forced tool call (``finish_reason="length"``, no
    ``tool_calls`` — see ``krepis.judge.check_openai_tool_response_for_leak``
    and its docstring for the live-reproduced failure shape). Excluding
    reasoning avoids paying for tokens that never reach the scored output
    and, per the same live check, reliably avoids the truncation failure
    mode for the pinned model. A future judge-tier model that specifically
    benefits from visible reasoning could override this — kept as an
    explicit, documented default rather than silently omitted.
    """
    return ModelSpec(
        provider="openrouter",
        model=request_model,
        max_tokens=max_tokens,
        reasoning={"exclude": True},
    )


@dataclass
class _OpenRouterJudgeCallResult:
    """Outcome of :func:`_call_openrouter_judge_llm` — a validated judge
    output plus provenance for the caller to persist onto its own
    ``RubricEvalArtifact``."""

    llm_output: RubricEvalLLMOutput
    resolved_model: str | None
    total_usd: float


def _call_openrouter_judge_llm(
    rendered: str,
    *,
    agent_id: str,
    request_model: str,
    max_tokens: int,
    api_key: str | None,
    max_retries: int,
    log_prefix: str,
) -> _OpenRouterJudgeCallResult:
    """Shared OpenRouter judge-call core: forced-tool-call request + leak
    guard + bounded retry loop. Used by BOTH ``evaluate_artifact`` (sync
    primary path, alpha-engine-config-I2997) and
    ``evaluate_artifact_openrouter`` (shadow tier, config#2575) — the only
    difference between the two callers is which ``request_model`` /
    ``judge_model`` identity they persist onto the result.

    Leak guard (config#2575 item 3): before accepting ANY OpenRouter
    response as a valid structured judge output, checks it against
    ``krepis.judge.check_openai_tool_response_for_leak`` — catches both
    the reasoning-budget-truncation and control-token-leak failure shapes
    documented on that function (both live-reproduced against a real
    OpenRouter call). A caught leak is logged at WARNING with a DISTINCT,
    grep-able marker (``leak_guard_triggered``) so a near-miss is
    diagnosable separately from an ordinary retry. A caught leak consumes
    a retry attempt (fresh decoder sample) rather than failing
    immediately, since — like ordinary schema non-conformance — a
    resample often recovers.

    Raises ``RuntimeError`` if all ``max_retries`` attempts fail (leak
    guard trip or schema validation failure).
    """
    spec = _openrouter_judge_model_spec(request_model=request_model, max_tokens=max_tokens)
    client = OpenAI(
        base_url=spec.resolved_base_url(),
        api_key=_resolve_openrouter_api_key(api_key),
        # Same 180s convention as thinktank/client.py's OpenAI construction.
        # max_retries=0: the SDK's own retry-on-transport-error would stack
        # underneath this function's own bounded attempt loop (max_retries
        # attempts, above) and silently blow past the "3 full model calls"
        # worst-case-latency bound documented on MAX_OPENROUTER_JUDGE_RETRIES.
        # Without an explicit timeout, a hung OpenRouter response blocks on
        # the SDK's 600s default — past the judge-perturbation-smoke
        # workflow's 8-minute job timeout — killing the job with zero
        # diagnostic output instead of a clear TimeoutError.
        timeout=180.0,
        max_retries=0,
    )

    tool_schema = _build_rubric_tool_spec()
    tool_name = tool_schema["name"]
    tools = [{
        "type": "function",
        "function": {
            "name": tool_schema["name"],
            "description": tool_schema["description"],
            "parameters": tool_schema["input_schema"],
        },
    }]

    last_error: BaseException | None = None
    llm_output: RubricEvalLLMOutput | None = None
    resolved_model: str | None = None
    total_usd = 0.0

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=spec.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a strict, evidence-grounded rubric judge."},
                    {"role": "user", "content": rendered},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": tool_name}},
                extra_body={"reasoning": spec.reasoning} if spec.reasoning else {},
            )
        except Exception as exc:  # noqa: BLE001 — transport error, bounded retry below
            last_error = exc
            logger.warning(
                "%s:%s attempt %d/%d transport error: %s",
                log_prefix, agent_id, attempt, max_retries, exc,
            )
            continue

        choice = resp.choices[0]
        resolved_model = getattr(resp, "model", None) or resolved_model
        usage = getattr(resp, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        if isinstance(cost, (int, float)):
            total_usd += float(cost)

        try:
            check_openai_tool_response_for_leak(choice, tool_name=tool_name)
        except JudgeToolCallLeakError as exc:
            last_error = exc
            # DISTINCT, grep-able marker — see docstring: a leak near-miss
            # must be diagnosable separately from ordinary schema-validation
            # retries, not folded into the same generic "attempt failed" line.
            logger.warning(
                "%s:%s leak_guard_triggered attempt=%d/%d "
                "reason=%s finish_reason=%s request_model=%s",
                log_prefix, agent_id, attempt, max_retries,
                exc.reason, exc.finish_reason, spec.model,
            )
            continue

        tool_calls = choice.message.tool_calls or []
        matching = next(
            (tc for tc in tool_calls if tc.function.name == tool_name), None,
        )
        if matching is None:
            last_error = ValueError(
                f"no {tool_name!r} tool call in OpenRouter response "
                f"(finish_reason={choice.finish_reason!r})"
            )
            logger.warning(
                "%s:%s attempt %d/%d: %s",
                log_prefix, agent_id, attempt, max_retries, last_error,
            )
            continue

        try:
            raw_args = json.loads(matching.function.arguments)
            llm_output = RubricEvalLLMOutput.model_validate(raw_args)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 — covers JSONDecodeError + ValidationError; bounded retry
            last_error = exc
            logger.warning(
                "%s:%s attempt %d/%d schema validation failed: %s",
                log_prefix, agent_id, attempt, max_retries, exc,
            )
            continue

    if llm_output is None:
        raise RuntimeError(
            f"{log_prefix} {max_retries} attempts failed for "
            f"agent_id={agent_id} request_model={request_model}. Last error: "
            f"{type(last_error).__name__ if last_error else 'Unknown'}: "
            f"{last_error}. Inspect the leak_guard_triggered / schema "
            f"validation WARNING logs above."
        )

    return _OpenRouterJudgeCallResult(
        llm_output=llm_output, resolved_model=resolved_model, total_usd=total_usd,
    )


def evaluate_artifact_openrouter(
    artifact: DecisionArtifact,
    *,
    judge_run_id: str | None = None,
    judge_model: str = OPENROUTER_SHADOW.logical_key,
    api_key: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    judged_artifact_s3_key: str | None = None,
    max_retries: int = MAX_OPENROUTER_JUDGE_RETRIES,
) -> RubricEvalArtifact:
    """Judge a single ``DecisionArtifact`` against its rubric via the
    OpenRouter shadow-judge tier.

    Mirrors ``evaluate_artifact``'s contract exactly (same rubric
    resolution, same empty-input / degenerate-input skip gates, same
    ``RubricEvalArtifact`` output shape, same S3 persistence/metric
    conventions downstream) — the only difference is the transport: a
    forced-tool-call OpenRouter request (via the shared
    ``_call_openrouter_judge_llm`` core, see its docstring) instead of
    LangChain's ``ChatAnthropic``.

    **Shadow-only, no decision authority** (config#2575 binding
    constraint) — see the module-level comment above this section and
    ``evals/judge_models.py::SHADOW_LOGICAL_KEYS``. Since
    alpha-engine-config-I2997 (2026-07-19), ``evaluate_artifact`` (the
    sync PRIMARY path) shares this same OpenRouter call core but is NOT
    shadow-tagged — this function remains the standalone, independently
    invocable shadow-tier entry point (``evals/openrouter_shadow.py``,
    the agreement-metric computation) unaffected by that migration.

    Raises:
      - ``ValueError`` if no rubric is mapped for the artifact's
        agent_id (same as ``evaluate_artifact``).
      - ``RuntimeError`` if all ``max_retries`` attempts fail (leak
        guard trip or schema validation failure) — same fail-loud
        contract as ``evaluate_artifact``.
    """
    rubric_name = resolve_rubric_for_agent(artifact.agent_id)
    if rubric_name is None:
        raise ValueError(
            f"No rubric mapped for agent_id={artifact.agent_id!r}. "
            f"Pre-filter with resolve_rubric_for_agent() if iterating "
            f"a mixed batch."
        )

    judge_run_id = judge_run_id or _new_judge_run_id()
    loaded_prompt = load_prompt(rubric_name)

    if not artifact.agent_output:
        return _make_skip_eval_artifact(
            artifact,
            rubric_name=rubric_name,
            rubric_version=loaded_prompt.version,
            judge_model=judge_model,
            judge_run_id=judge_run_id,
            judged_artifact_s3_key=judged_artifact_s3_key,
            skip_reason="precluded_by_empty_upstream",
        )

    if _is_degenerate_input(artifact):
        logger.info(
            "[eval_judge_openrouter] degenerate_input skip — agent_id=%s",
            artifact.agent_id,
        )
        return _make_skip_eval_artifact(
            artifact,
            rubric_name=rubric_name,
            rubric_version=loaded_prompt.version,
            judge_model=judge_model,
            judge_run_id=judge_run_id,
            judged_artifact_s3_key=judged_artifact_s3_key,
            skip_reason="degenerate_input",
        )

    rendered = _render_rubric(artifact, loaded_prompt)
    request_model = request_model_for(judge_model)
    call_result = _call_openrouter_judge_llm(
        rendered,
        agent_id=artifact.agent_id,
        request_model=request_model,
        max_tokens=max_tokens,
        api_key=api_key,
        max_retries=max_retries,
        log_prefix="[eval_judge_openrouter]",
    )

    logger.info(
        "[eval_judge_openrouter] persisted-cost agent_id=%s request_model=%s "
        "resolved_model=%s provider_cost_usd=%.6f (shadow-only, no "
        "decision authority — config#2575)",
        artifact.agent_id, request_model, call_result.resolved_model,
        call_result.total_usd,
    )

    return RubricEvalArtifact(
        run_id=artifact.run_id,
        judge_run_id=judge_run_id,
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        judged_agent_id=artifact.agent_id,
        judged_artifact_s3_key=judged_artifact_s3_key,
        rubric_id=rubric_name,
        rubric_version=loaded_prompt.version,
        judge_model=judge_model,
        judge_request_model=request_model,
        judge_resolved_model=call_result.resolved_model,
        dimension_scores=call_result.llm_output.dimension_scores,
        overall_reasoning=call_result.llm_output.overall_reasoning,
    )


def _resolve_openrouter_api_key(api_key: str | None) -> str:
    """Resolve the OpenRouter API key: explicit ``api_key`` arg wins,
    else ``config.OPENROUTER_API_KEY`` (SSM-first with env fallback via
    ``nousergon_lib.secrets.get_secret`` — the fleet's standard secret-
    resolution convention). Shared by ``evaluate_artifact`` (sync primary
    path, alpha-engine-config-I2997) and ``evaluate_artifact_openrouter``
    (shadow tier, config#2575) via ``_call_openrouter_judge_llm``. Raises
    loudly rather than letting the OpenAI SDK client construction fail
    with a less diagnosable error.
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise RuntimeError(
            "the OpenRouter judge call requires an OpenRouter API key: "
            "pass api_key= explicitly, or ensure config.OPENROUTER_API_KEY "
            "resolves (SSM parameter /alpha-engine/OPENROUTER_API_KEY, or "
            "the OPENROUTER_API_KEY environment variable as a fallback)."
        )
    return key


# ── Persistence ───────────────────────────────────────────────────────────


DEFAULT_EVAL_PREFIX = "decision_artifacts/_eval/"
"""Production eval-artifact prefix. ``judge_only`` mode swaps in
``decision_artifacts/_eval_judge_only/`` so isolated test runs don't
pollute the prod corpus the rolling-mean Lambda + dashboard read."""


_CAPTURE_DATE_RE = re.compile(
    r"^decision_artifacts/(\d{4})/(\d{2})/(\d{2})/",
)
"""Match the leading ``decision_artifacts/{Y}/{M}/{D}/`` segment of a
DecisionArtifact S3 key. The capture date lives in this prefix and is
the authoritative date for partitioning the eval artifact — judge
wall-clock is not (the judge can run hours/days after capture)."""


def _capture_date_from_s3_key(judged_artifact_s3_key: str | None) -> str | None:
    """Extract ``YYYY-MM-DD`` from a DecisionArtifact S3 key.

    Returns None when the key is None, doesn't match the canonical
    ``decision_artifacts/{Y}/{M}/{D}/`` shape, or the date components
    fail strict parse. None means "fall back to judge wall-clock" —
    matches pre-2026-05-08 behavior for in-memory / synthetic artifacts.
    """
    if not judged_artifact_s3_key:
        return None
    match = _CAPTURE_DATE_RE.match(judged_artifact_s3_key)
    if match is None:
        return None
    y, m, d = match.groups()
    try:
        # strict parse: rejects "2026-13-99" silently from passing through
        date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None
    return f"{y}-{m}-{d}"


def _eval_basename(
    *, judged_agent_id: str, run_id: str, judge_model: str,
) -> str:
    """Per-file basename for one eval artifact inside a judge batch.

    ``{judged_agent_id}.{run_id}.{judge_model}.json`` — the same triple
    that disambiguated artifacts in the legacy nested layout, now carried
    as the multi-file basename under the lib's flat ``{run_id}_{basename}``
    convention. The ``judge_model`` segment lets Haiku-tier and
    Sonnet-tier evals of the same judged artifact coexist without
    clobbering each other within one batch.
    """
    return f"{judged_agent_id}.{run_id}.{judge_model}.json"


def build_eval_s3_key(
    *,
    judged_agent_id: str,
    run_id: str,
    judge_run_id: str,
    judge_model: str,
    timestamp: datetime | None = None,  # noqa: ARG001 — see below
    prefix: str = DEFAULT_EVAL_PREFIX,
) -> str:
    """Build the canonical S3 key for an eval artifact.

    **Canonical ``alpha_engine_lib.eval_artifacts`` layout (config#793
    swap, supersedes the 2026-05-08 Option B nested partition):**

    Path shape (flat — no ``{date}/`` sub-partition)::

        {prefix}{judge_run_id}_{judged_agent_id}.{run_id}.{judge_model}.json

    Delegates the key format to
    ``alpha_engine_lib.eval_artifacts.eval_artifact_key`` (single source
    of truth — we do NOT hand-roll the format). The eval-judge pipeline
    is a *multi-file-per-run* consumer: one judge batch mints one
    ``judge_run_id`` (now a ``YYMMDDHHMM`` structured timestamp from
    ``new_eval_run_id``, formerly a UUID) and emits one artifact per
    (judged_agent_id, run_id, judge_model). The lib's
    ``{run_id}_{basename}`` form keeps every file from one batch grouped
    by the shared ``judge_run_id`` prefix in path listings — the
    flat-layout equivalent of the legacy nested
    ``{date}/{judge_run_id}/`` directory.

    Because the ``judge_run_id`` is a UTC timestamp, lexicographic sort
    across the flat ``_eval/`` prefix yields chronological order with no
    date partition needed. Operators query one batch's outputs via
    ``aws s3 ls _eval/ | grep {judge_run_id}`` (or
    ``--starting-token``); capture-date queries are still served by the
    manifest layer at ``_eval_by_capture/{capture_date}/manifest.json``.

    ``timestamp`` is accepted for backward-compatible call signatures but
    is no longer used to build the key — the date now lives inside the
    timestamp-encoded ``judge_run_id``. Legacy nested keys (produced
    before this swap) are still readable via :func:`build_legacy_eval_s3_key`
    and the tolerant manifest scanner.

    ``prefix`` lets ``judge_only`` mode redirect outputs to an isolated
    path so test runs don't pollute prod observability.
    """
    if not judge_run_id:
        raise ValueError(
            "build_eval_s3_key requires judge_run_id (canonical eval_artifacts "
            "layout). Generate one per judge batch invocation via "
            "_new_judge_run_id() and pass it to every RubricEvalArtifact "
            "construction in that batch."
        )
    basename = _eval_basename(
        judged_agent_id=judged_agent_id, run_id=run_id, judge_model=judge_model,
    )
    return eval_artifact_key(prefix, judge_run_id, basename=basename)


def build_legacy_eval_s3_key(
    *,
    judged_agent_id: str,
    run_id: str,
    judge_run_id: str,
    judge_model: str,
    timestamp: datetime | None = None,
    prefix: str = DEFAULT_EVAL_PREFIX,
) -> str:
    """Build the *legacy* nested Option B key (pre-config#793 swap).

    Path shape::

        {prefix}{judge_run_date}/{judge_run_id}/
          {judged_agent_id}.{run_id}.{judge_model}.json

    Retained for backward-compatibility readers and tests — months of
    historical eval artifacts already live at this layout and are NOT
    backfilled (see config#793 migration discipline). New writes use the
    canonical flat layout via :func:`build_eval_s3_key`.
    """
    if not judge_run_id:
        raise ValueError("build_legacy_eval_s3_key requires judge_run_id.")
    ts = timestamp or datetime.now(UTC)
    date_partition = ts.strftime("%Y-%m-%d")
    return (
        f"{prefix}{date_partition}/{judge_run_id}/"
        f"{judged_agent_id}.{run_id}.{judge_model}.json"
    )


def persist_eval_artifact(
    artifact: RubricEvalArtifact,
    *,
    s3_client: Any = None,
    bucket: str = S3_BUCKET,
    prefix: str = DEFAULT_EVAL_PREFIX,
    update_latest: bool = True,
) -> str:
    """Write an eval artifact to S3 and return the (dated) S3 key.

    Writes under the canonical flat ``alpha_engine_lib.eval_artifacts``
    layout (config#793) and, when ``update_latest`` is True, mirrors a
    ``latest.json`` operator-UX sidecar pointing at the just-written key
    (``eval_latest_key``). The dated key remains the forensic source of
    truth; the sidecar is a convenience pointer the lib's
    ``load_latest_eval_artifact`` reader resolves.

    Hard-fails on the primary artifact write (per
    ``feedback_no_silent_fails``). The sidecar mirror is best-effort:
    a sidecar write failure is logged but does NOT fail the artifact
    write — the dated artifact is the durable record, the sidecar a
    rebuildable pointer.

    ``prefix`` lets ``judge_only`` mode persist to an isolated path and
    is forwarded to ``build_eval_s3_key`` / ``eval_latest_key``.

    The ``s3_client`` parameter accepts an injected client for tests;
    production passes None and the helper builds the default client.
    """
    s3 = s3_client or boto3.client("s3")
    key = build_eval_s3_key(
        judged_agent_id=artifact.judged_agent_id,
        run_id=artifact.run_id,
        judge_run_id=artifact.judge_run_id,
        judge_model=artifact.judge_model,
        prefix=prefix,
    )
    body = artifact.model_dump_json(indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    logger.info(
        "[eval_judge] persisted eval for agent_id=%s rubric=%s judge=%s → %s",
        artifact.judged_agent_id, artifact.rubric_id,
        artifact.judge_model, key,
    )
    if update_latest:
        # Operator-UX sidecar mirror. Best-effort: the dated artifact is
        # the durable record; the sidecar is a rebuildable single-fetch
        # pointer consumed by alpha_engine_lib.load_latest_eval_artifact.
        sidecar_key = eval_latest_key(prefix)
        sidecar_body = json.dumps(
            {
                "artifact_key": key,
                "judge_run_id": artifact.judge_run_id,
                "judged_agent_id": artifact.judged_agent_id,
                "run_id": artifact.run_id,
                "judge_model": artifact.judge_model,
                "timestamp": artifact.timestamp,
            },
            indent=2,
        ).encode("utf-8")
        try:
            s3.put_object(Bucket=bucket, Key=sidecar_key, Body=sidecar_body)
        except Exception:  # noqa: BLE001
            logger.warning(
                "[eval_judge] latest sidecar mirror failed at s3://%s/%s — "
                "dated artifact %s is the durable record; sidecar is "
                "rebuildable",
                bucket, sidecar_key, key, exc_info=True,
            )
    return key
