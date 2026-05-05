"""
LLM-as-judge evaluation pipeline (workstream closed 2026-05-03).

Reads a captured ``DecisionArtifact``, looks up the matching rubric
prompt, sends ``(rubric, artifact_input, artifact_output)`` to a judge
LLM (Haiku default; Sonnet for nuance-tier sampled subset), and
persists the structured eval result to S3.

Eval is observability, NOT a gate. Runs proceed regardless of eval
score; the eval corpus + dashboard surface quality regressions weeks
before they show up in alpha-vs-SPY.

Composes with:
- Decision-artifact capture (alpha_engine_lib.decision_capture).
- Rubric prompts in alpha-engine-config (eval_rubric_*.txt at
  version 1.0.0+, loaded via ``agents.prompt_loader.load_prompt``).
- Cost telemetry — eval LLM calls are tagged ``agent_id="eval_judge"``
  via ``track_llm_cost`` so judging cost is observable + bounded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from alpha_engine_lib.decision_capture import DecisionArtifact

from config import ANTHROPIC_API_KEY, MAX_TOKENS_STRATEGIC, S3_BUCKET
from agents.prompt_loader import load_prompt
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
      thesis_update:{team}:{ticker} → None (deferred — narrower call,
                                       structured update not novel
                                       analysis; eval value lower)

    Unknown agent_ids return None so the caller can skip cleanly
    rather than crash on rubric lookup.
    """
    if agent_id.startswith("sector_quant:"):
        return "eval_rubric_sector_quant"
    if agent_id.startswith("sector_qual:"):
        return "eval_rubric_sector_qual"
    if agent_id.startswith("sector_peer_review:"):
        return "eval_rubric_sector_peer_review"
    if agent_id == "macro_economist":
        return "eval_rubric_macro_economist"
    if agent_id == "ic_cio":
        return "eval_rubric_ic_cio"
    return None


# ── Judge call ────────────────────────────────────────────────────────────


def evaluate_artifact(
    artifact: DecisionArtifact,
    *,
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
    if not artifact.agent_output:
        loaded_prompt = load_prompt(rubric_name)
        return RubricEvalArtifact(
            run_id=artifact.run_id,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            judged_agent_id=artifact.agent_id,
            judged_artifact_s3_key=judged_artifact_s3_key,
            rubric_id=rubric_name,
            rubric_version=loaded_prompt.version,
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

    loaded_prompt = load_prompt(rubric_name)

    # Render with the artifact's payload. ``json.dumps(..., default=str)``
    # handles any stray types (datetimes, Decimals) that snuck into the
    # captured snapshot.
    rendered = loaded_prompt.format(
        agent_input=json.dumps(
            artifact.input_data_snapshot, indent=2, default=str,
        ),
        agent_output=json.dumps(
            artifact.agent_output, indent=2, default=str,
        ),
    )

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


def build_eval_s3_key(
    *,
    judged_agent_id: str,
    run_id: str,
    judge_model: str,
    timestamp: Optional[datetime] = None,
    prefix: str = DEFAULT_EVAL_PREFIX,
) -> str:
    """Build the canonical S3 key for an eval artifact.

    Path shape:
      ``{prefix}{YYYY-MM-DD}/{judged_agent_id}/{run_id}.{judge_model}.json``

    The ``judge_model`` segment lets Haiku-tier and Sonnet-tier evals
    of the same artifact coexist without clobbering each other.

    The date partition is taken from ``timestamp`` (defaults to
    now-UTC) so multiple runs on the same calendar day cluster under
    one prefix. ``run_id`` is the filename stem so retries with the
    same run_id + judge_model idempotently overwrite.

    ``prefix`` lets ``judge_only`` mode redirect outputs to an isolated
    path so test runs don't pollute prod observability. Must end in
    ``/``.
    """
    ts = timestamp or datetime.now(timezone.utc)
    date_partition = ts.strftime("%Y-%m-%d")
    return (
        f"{prefix}{date_partition}/"
        f"{judged_agent_id}/{run_id}.{judge_model}.json"
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
    # Re-derive timestamp from the artifact's stamped ISO-8601 so the
    # partition matches what the artifact records — not "now at write
    # time" — keeping replay paths stable.
    artifact_ts = datetime.fromisoformat(artifact.timestamp.replace("Z", "+00:00"))
    key = build_eval_s3_key(
        judged_agent_id=artifact.judged_agent_id,
        run_id=artifact.run_id,
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
