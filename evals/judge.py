"""
LLM-as-judge evaluation pipeline.

Reads a captured ``DecisionArtifact``, looks up the matching rubric
prompt, sends ``(rubric, artifact_input, artifact_output)`` to a judge
LLM (Haiku default; Sonnet for nuance-tier sampled subset), and
persists the structured eval result to S3.

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

Composes with:
- Decision-artifact capture (alpha_engine_lib.decision_capture).
- Rubric prompts in alpha-engine-config (eval_rubric_*.txt at
  version 1.0.0+, loaded via ``agents.prompt_loader.load_prompt``).
- Cost telemetry — sync eval LLM calls are tagged
  ``agent_id="eval_judge"`` via ``track_llm_cost``. Batch results emit
  the same telemetry from the Process Lambda using the per-result usage
  block returned by Anthropic's batch results stream.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional


def _new_judge_run_id() -> str:
    """Mint a fresh UUID for a judge batch invocation.

    Production paths generate one of these at the start of a batch and
    propagate it to every RubricEvalArtifact emitted by that batch.
    Solo / replay / smoke callers get a fresh UUID per call.

    UUIDv4 chosen over UUIDv7 for readability — the path already
    encodes the date as the partition prefix, so embedding a timestamp
    in the UUID itself is redundant.
    """
    return str(uuid.uuid4())

import boto3
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from alpha_engine_lib.decision_capture import DecisionArtifact

from config import ANTHROPIC_API_KEY, MAX_TOKENS_STRATEGIC, S3_BUCKET
from agents.prompt_loader import LoadedPrompt, load_prompt
from graph.llm_cost_tracker import get_cost_telemetry_callback, track_llm_cost
from graph.state_schemas import (
    RubricEvalArtifact,
    RubricEvalLLMOutput,
)

logger = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────────


DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
"""Default judge model — Haiku for cost on every weekly run.

Sonnet (``claude-sonnet-4-6``) is used for the nuance-tier sampled
subset."""

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

_RETRY_BACKOFF_BASE_SEC = 0.2
"""Initial backoff between retry attempts. 200ms × 2^attempt =
200ms / 400ms / 800ms — short enough to not blow Lambda budgets,
long enough to ride out transient API hiccups."""


# ── Agent → rubric mapping ────────────────────────────────────────────────


def resolve_rubric_for_agent(agent_id: str) -> Optional[str]:
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


_CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
"""Anthropic batch custom_id regex (per Message Batches API docs)."""

_JUDGE_MODEL_TAG = {
    "claude-haiku-4-5": "h45",
    "claude-sonnet-4-6": "s46",
}
"""Compact tags for the two judge models. Keeps custom_id under the
64-char limit even when judged_agent_id is long
(e.g. ``thesis_update:technology:AAPL``)."""

_JUDGE_MODEL_TAG_REVERSE = {v: k for k, v in _JUDGE_MODEL_TAG.items()}


def encode_custom_id(
    *, judged_agent_id: str, run_id: str, judge_model: str,
) -> str:
    """Encode (judged_agent_id, run_id, judge_model) → batch custom_id.

    Replaces ``:`` and ``/`` separators with ``-`` since the Anthropic
    custom_id charset only allows alphanumerics, ``-``, and ``_``.
    Truncates the agent_id segment if needed so the final string fits
    the 64-char ceiling. Round-trippable via ``decode_custom_id``.
    """
    tag = _JUDGE_MODEL_TAG.get(judge_model)
    if tag is None:
        # Unknown judge model — fall back to a hash-stable suffix.
        tag = f"x{abs(hash(judge_model)) % 10_000:04d}"
    safe_agent = re.sub(r"[^a-zA-Z0-9_-]", "-", judged_agent_id)
    safe_run = re.sub(r"[^a-zA-Z0-9_-]", "-", run_id)
    # Reserve 4 chars for "__" separators + 3-char model tag.
    fixed_overhead = len(safe_run) + len(tag) + 4
    max_agent = max(8, 64 - fixed_overhead)
    if len(safe_agent) > max_agent:
        safe_agent = safe_agent[:max_agent]
    cid = f"{safe_agent}__{safe_run}__{tag}"
    if not _CUSTOM_ID_PATTERN.match(cid):
        # Last-ditch sanitize — strip anything that snuck through and
        # trim to the cap. The decode side just needs the model tag at
        # the tail; agent_id round-trip is best-effort once truncated.
        cid = re.sub(r"[^a-zA-Z0-9_-]", "-", cid)[:64]
    return cid


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
    """
    parts = custom_id.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"Cannot decode batch custom_id={custom_id!r}: expected "
            f"three '__'-separated segments, got {len(parts)}."
        )
    safe_agent, safe_run, tag = parts
    judge_model = _JUDGE_MODEL_TAG_REVERSE.get(tag, tag)
    return safe_agent, safe_run, judge_model


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
    """
    return loaded_prompt.format(
        agent_input=json.dumps(
            artifact.input_data_snapshot, indent=2, default=str,
        ),
        agent_output=json.dumps(
            artifact.agent_output, indent=2, default=str,
        ),
    )


def _make_skip_eval_artifact(
    artifact: DecisionArtifact,
    *,
    rubric_name: str,
    rubric_version: str,
    judge_model: str,
    judge_run_id: str,
    judged_artifact_s3_key: Optional[str],
) -> RubricEvalArtifact:
    """Build the skip-marker eval for an artifact whose ``agent_output``
    is empty (graph bypassed the agent — see comment in
    ``evaluate_artifact``). Shared by the sync + batch paths.

    Asking the judge to score "no rationale, no citations, no
    synthesis" produces uniform 1/1/1/1 outputs that drag the
    rolling-mean alarm threshold toward the floor without any real
    quality regression. Short-circuit BEFORE the LLM call so we pay
    no token cost and emit no spurious low scores.
    """
    return RubricEvalArtifact(
        run_id=artifact.run_id,
        judge_run_id=judge_run_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        judged_agent_id=artifact.agent_id,
        judged_artifact_s3_key=judged_artifact_s3_key,
        rubric_id=rubric_name,
        rubric_version=rubric_version,
        judge_model=judge_model,
        dimension_scores=[],
        overall_reasoning=(
            "Judge short-circuited: captured agent_output is empty "
            "(graph bypassed the agent — typically sector_qual or "
            "sector_peer_review when upstream quant_top5 is empty). "
            "No work to evaluate; not a quality regression."
        ),
        judge_skip_reason="precluded_by_empty_upstream",
    )


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
    """
    schema = RubricEvalLLMOutput.model_json_schema()
    return {
        "name": _RUBRIC_TOOL_NAME,
        "description": _RUBRIC_TOOL_DESCRIPTION,
        "input_schema": schema,
    }


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

    Raises ``ValueError`` if ``artifact.agent_id`` has no rubric
    mapped — callers must pre-filter via ``resolve_rubric_for_agent``.

    Empty-input short-circuit is handled by the orchestrator BEFORE
    this function is invoked (the skip artifact is persisted
    client-side without spending a batch slot).
    """
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

    return {
        "custom_id": custom_id,
        "params": {
            "model": judge_model,
            "max_tokens": max_tokens,
            "tools": [tool_spec],
            # Force the model to call the rubric tool — equivalent to
            # ``with_structured_output(...)``'s ``tool_choice`` posture.
            # Without this the model can decide to emit prose, which
            # would fall through every parser in this module.
            "tool_choice": {"type": "tool", "name": _RUBRIC_TOOL_NAME},
            "messages": [
                {"role": "user", "content": rendered},
            ],
            # ``metadata.user_id`` is reserved for end-user identification
            # in Anthropic's contract; we pass the rubric+version pair
            # via ``metadata`` for batch-side observability without
            # putting it on a schema-validated field.
        },
    }


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
    """
    content = (
        message_payload["content"]
        if isinstance(message_payload, dict)
        else message_payload.content
    )
    for block in content:
        block_type = (
            block.get("type") if isinstance(block, dict) else block.type
        )
        block_name = (
            block.get("name") if isinstance(block, dict)
            else getattr(block, "name", None)
        )
        if block_type == "tool_use" and block_name == _RUBRIC_TOOL_NAME:
            tool_input = (
                block["input"] if isinstance(block, dict) else block.input
            )
            return RubricEvalLLMOutput.model_validate(tool_input)
    raise ValueError(
        "No tool_use block named "
        f"{_RUBRIC_TOOL_NAME!r} found in batch result message; the "
        "judge LLM did not emit the rubric eval via the structured "
        "tool — inspect the raw batch result on Anthropic's side "
        "(retained 29 days)."
    )


# ── Judge call ────────────────────────────────────────────────────────────


def evaluate_artifact(
    artifact: DecisionArtifact,
    *,
    judge_run_id: Optional[str] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    judged_artifact_s3_key: Optional[str] = None,
    max_retries: int = MAX_JUDGE_RETRIES,
) -> RubricEvalArtifact:
    """Judge a single ``DecisionArtifact`` against its rubric.

    Resolves the rubric for ``artifact.agent_id``, renders the rubric
    prompt with the artifact's ``input_data_snapshot`` + ``agent_output``,
    and invokes the judge LLM via ``with_structured_output(include_raw=True)``.
    Retries up to ``max_retries`` times on parse failures (LLM tool-use
    is stochastically non-conformant — fresh decoder sample on retry
    typically succeeds). The returned ``RubricEvalArtifact`` carries
    the dimension scores plus metadata (rubric_id+version, judge_model,
    judged_agent_id).

    On every parse-failure attempt, the raw tool-use payload head is
    logged at WARNING so production failures are diagnosable without
    re-running the artifact.

    Cost telemetry: scoped under ``agent_id="eval_judge"`` so judging
    cost is tracked separately from the agents being judged. Retry
    attempts accumulate into the same cost frame.

    Raises:
      - ``ValueError`` if no rubric is mapped for the artifact's
        agent_id — callers should pre-filter via ``resolve_rubric_for_agent``
        when iterating a mixed batch.
      - ``RuntimeError`` if all ``max_retries`` parse attempts fail —
        the underlying issue is structural (model regression, rubric
        too dense, etc.) and surfaces as a loud failure for diagnosis.
    """
    import time

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
        )

    rendered = _render_rubric(artifact, loaded_prompt)

    llm = ChatAnthropic(
        model=judge_model,
        anthropic_api_key=api_key or ANTHROPIC_API_KEY,
        max_tokens=max_tokens,
        callbacks=[get_cost_telemetry_callback()],
    )
    # ``include_raw=True`` returns ``{"raw": AIMessage, "parsed":
    # RubricEvalLLMOutput | None, "parsing_error": Exception | None}``
    # so we can log the raw tool-use payload on parse failures and
    # retry the call rather than letting the parse error escape.
    structured_llm = llm.with_structured_output(
        RubricEvalLLMOutput, include_raw=True,
    )

    llm_output: Optional[RubricEvalLLMOutput] = None
    last_err: Optional[BaseException] = None

    with track_llm_cost(
        agent_id="eval_judge",
        node_name="eval_judge_node",
        run_type="weekly_research",
        prompt=loaded_prompt,
        model_name_fallback=judge_model,
        run_id=artifact.run_id,
    ):
        for attempt in range(max_retries):
            resp = structured_llm.invoke(
                [HumanMessage(content=rendered)],
                config={"metadata": loaded_prompt.langsmith_metadata()},
            )
            parsed = resp.get("parsed")
            parsing_error = resp.get("parsing_error")
            if parsed is not None and parsing_error is None:
                llm_output = parsed
                if attempt > 0:
                    logger.info(
                        "[eval_judge] parse succeeded on attempt %d/%d "
                        "for agent_id=%s",
                        attempt + 1, max_retries, artifact.agent_id,
                    )
                break

            last_err = parsing_error
            raw_head = str(resp.get("raw"))[:300] if resp.get("raw") else "(no raw)"
            logger.warning(
                "[eval_judge] parse attempt %d/%d failed for agent_id=%s "
                "judge=%s: %s: %s; raw head=%r",
                attempt + 1, max_retries, artifact.agent_id, judge_model,
                type(parsing_error).__name__ if parsing_error else "Unknown",
                str(parsing_error)[:200] if parsing_error else "(no error)",
                raw_head,
            )
            if attempt + 1 < max_retries:
                time.sleep(_RETRY_BACKOFF_BASE_SEC * (2 ** attempt))

    if llm_output is None:
        raise RuntimeError(
            f"[eval_judge] {max_retries} parse attempts failed for "
            f"agent_id={artifact.agent_id} judge={judge_model}. "
            f"Last error: {type(last_err).__name__ if last_err else 'Unknown'}: "
            f"{last_err}. Underlying issue is structural — inspect raw "
            f"tool-use payloads in the WARNING logs above."
        )

    return RubricEvalArtifact(
        run_id=artifact.run_id,
        judge_run_id=judge_run_id,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        judged_agent_id=artifact.agent_id,
        judged_artifact_s3_key=judged_artifact_s3_key,
        rubric_id=rubric_name,
        rubric_version=loaded_prompt.version,
        judge_model=judge_model,
        dimension_scores=llm_output.dimension_scores,
        overall_reasoning=llm_output.overall_reasoning,
    )


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


def build_eval_s3_key(
    *,
    judged_agent_id: str,
    run_id: str,
    judge_run_id: str,
    judge_model: str,
    timestamp: Optional[datetime] = None,
    prefix: str = DEFAULT_EVAL_PREFIX,
) -> str:
    """Build the canonical S3 key for an eval artifact.

    **Institutional production-grade partition (Option B, ROADMAP P1
    closure 2026-05-08):**

    Path shape:
      ``{prefix}{judge_run_date}/{judge_run_id}/
        {judged_agent_id}.{judged_run_id}.{judge_model}.json``

    The eval artifact is treated as a first-class entity owned by the
    eval-judge batch invocation that produced it (the canonical pattern
    in LangSmith / Langfuse / Helicone-class systems). Each batch gets
    a fresh ``judge_run_id`` (UUID) so all artifacts emitted by one
    batch cluster under a single directory and are queryable as a group
    via ``aws s3 ls _eval/{date}/{judge_run_id}/``.

    Capture-date queries ("show me all evals of artifacts captured on
    day X") are served by a separate manifest layer at
    ``_eval_by_capture/{capture_date}/manifest.json`` (PR 2 of the
    Option B arc). The judged artifact's S3 key remains as a
    foreign-key field on the artifact for direct lookup.

    The date partition is the artifact's emission timestamp (UTC).
    Within a single batch, all artifacts share the same ``judge_run_id``;
    if the batch crosses UTC midnight, the directory listing for that
    judge_run_id will straddle two date prefixes — but
    ``aws s3 ls --recursive --include "*{judge_run_id}*"`` still
    returns the full batch as a single logical group because the
    ``judge_run_id`` is in the path.

    The ``judge_model`` segment lets Haiku-tier and Sonnet-tier evals
    of the same judged artifact coexist without clobbering each other.

    ``prefix`` lets ``judge_only`` mode redirect outputs to an isolated
    path so test runs don't pollute prod observability. Must end in
    ``/``.
    """
    if not judge_run_id:
        raise ValueError(
            "build_eval_s3_key requires judge_run_id (Option B partition). "
            "Generate one UUID per judge batch invocation and pass it to "
            "every RubricEvalArtifact construction in that batch."
        )
    ts = timestamp or datetime.now(timezone.utc)
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
) -> str:
    """Write an eval artifact to S3 and return the S3 key.

    Uses the canonical ``decision_artifacts/_eval/...`` path by default.
    Hard-fails on S3 errors (per ``feedback_no_silent_fails``) — callers
    should handle the exception explicitly if running in best-effort
    mode.

    ``prefix`` lets ``judge_only`` mode persist to an isolated path.
    Must end in ``/`` and is forwarded to ``build_eval_s3_key``.

    The ``s3_client`` parameter accepts an injected client for tests;
    production passes None and the helper builds the default client.
    """
    s3 = s3_client or boto3.client("s3")
    # Partition by judge_run_id (Option B institutional pattern,
    # ROADMAP closure 2026-05-08). The judge_run_id is constant across
    # all artifacts emitted by one batch invocation — operators query
    # batch outputs as a single group via
    # ``aws s3 ls _eval/{date}/{judge_run_id}/``. Capture-date queries
    # are served by the manifest layer at _eval_by_capture/.
    artifact_ts = datetime.fromisoformat(artifact.timestamp.replace("Z", "+00:00"))
    key = build_eval_s3_key(
        judged_agent_id=artifact.judged_agent_id,
        run_id=artifact.run_id,
        judge_run_id=artifact.judge_run_id,
        judge_model=artifact.judge_model,
        timestamp=artifact_ts,
        prefix=prefix,
    )
    body = artifact.model_dump_json(indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    logger.info(
        "[eval_judge] persisted eval for agent_id=%s rubric=%s judge=%s → %s",
        artifact.judged_agent_id, artifact.rubric_id,
        artifact.judge_model, key,
    )
    return key
