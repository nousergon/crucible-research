"""
Shared utilities for sector-team agents.

Two concerns:

  1. Extracting tool calls / final text from LangGraph message histories
     (used by the quant + qual ReAct agents).
  2. A 429-aware, *deadline-bounded* persistent retry wrapper for every
     agent Haiku/Sonnet call (``invoke_with_rate_limit_retry``).

The 429 wrapper exists because the 6-team parallel ``Send()`` fan-out
in ``graph/research_graph.py`` bursts over the org's Haiku input-TPM
ceiling (450,000 tokens/min, claude-haiku-4-5). On the 2026-05-16
recovery run that surfaced as ``RateLimitError 429`` aborting
defensives/financials/technology.

ALL-AGENTS-STRICT rework (Brian, 2026-05-16) — supersedes the #194
"~6 attempt cap then degrade-and-continue" philosophy:

  "If the sector agents don't run, Research shouldn't complete until
   all sectors are run. We should have a long retry mechanism and after
   this long period if we still don't have all sectors it should fail.
   We don't get anything from this process if the sectors, or any other
   agent for that matter, fail/don't run."

So the wrapper is now an **overall wall-clock deadline** (default
``RATE_LIMIT_RETRY_DEADLINE_SECONDS`` = 75 min) of persistent 429
retry with capped exponential backoff between attempts — NOT a small
fixed attempt count. It honors the ``retry-after`` response header
when Anthropic sends one. Only 429 / rate-limit errors are retried;
every other exception propagates immediately and unchanged (then the
caller's hard-fail rule applies — a non-429 failure still fails the
run because we get nothing from a partial process).

The long window is affordable because every agent that already
succeeded for this ``run_date`` is persisted to S3 (sector teams via
``ArchiveManager.save_sector_team_run``; CIO/macro via the agent-run
persistence added in this rework). So the deadline is only ever spent
re-attempting the *still-missing* agents — both within a single
invocation and across a Step-Function redrive.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

# Constructor-level retry budget for every agent ChatAnthropic
# instance. langchain-anthropic defaults to 2, which is exhausted
# almost immediately under a sustained org-wide 450K-TPM 429. This is
# the inner SDK-level backoff; ``invoke_with_rate_limit_retry`` is the
# outer, retry-after-aware, deadline-bounded backoff around the whole
# ``.invoke()``.
SECTOR_TEAM_LLM_MAX_RETRIES = 8

# ── All-agents-strict deadline (Brian, 2026-05-16) ────────────────────────
# Overall wall-clock budget for persistent 429 retry of a SINGLE
# ``.invoke()``. This is the "long retry mechanism" the directive asks
# for: keep riding out the org TPM ceiling for up to this long, then —
# if the call still cannot produce real output — propagate the 429 so
# the caller's hard-fail rule turns the whole run into status:ERROR
# (no signals.json / no email / no DB upload). 75 min default; env
# override ``RATE_LIMIT_RETRY_DEADLINE_SECONDS`` (clamped to a sane
# 5 min .. 3 hr band so a typo can't make it unbounded or trivial).
#
# This is per-``.invoke()``, NOT per-run, but it is bounded in
# aggregate because every agent that already succeeded for this
# run_date is persisted (sector teams → save_sector_team_run; CIO /
# macro → save_agent_run) and short-circuited on resume with ZERO LLM
# calls — so a retry / SF redrive only ever spends the deadline on the
# agents still missing, not on the whole pipeline.
def _resolve_deadline_seconds() -> float:
    raw = os.environ.get("RATE_LIMIT_RETRY_DEADLINE_SECONDS")
    if raw is None:
        return 75.0 * 60.0
    try:
        secs = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "[rate_limit_retry] RATE_LIMIT_RETRY_DEADLINE_SECONDS=%r "
            "unparseable — using 75 min default", raw,
        )
        return 75.0 * 60.0
    # Clamp: never < 5 min (too short to ride a TPM window) and never
    # > 3 hr (a typo must not make the Lambda hang past its own timeout).
    return max(5.0 * 60.0, min(secs, 3.0 * 60.0 * 60.0))


# Module-level constant (the deadline the rework is built around).
# Resolved at import; tests monkeypatch this attribute directly.
RATE_LIMIT_RETRY_DEADLINE_SECONDS: float = _resolve_deadline_seconds()

# Backoff between 429 attempts. Capped so a single sleep can't blow
# past the deadline check granularity; the deadline (not an attempt
# count) is what bounds the loop.
_BACKOFF_BASE_SECONDS = 4.0
_BACKOFF_CAP_SECONDS = 60.0

_T = TypeVar("_T")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True iff ``exc`` is an Anthropic 429 / rate-limit error.

    Catches both the typed ``anthropic.RateLimitError`` and the generic
    ``APIStatusError`` with ``status_code == 429`` (langchain may wrap
    or re-raise either shape depending on where the limit trips). Falls
    back to a status-code / message sniff so a future SDK reshuffle
    doesn't silently turn 429s into hard failures again.
    """
    try:
        import anthropic

        if isinstance(exc, anthropic.RateLimitError):
            return True
        api_status = getattr(anthropic, "APIStatusError", None)
        if api_status is not None and isinstance(exc, api_status):
            return getattr(exc, "status_code", None) == 429
    except Exception:  # pragma: no cover — anthropic always importable here
        pass
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a ``retry-after`` hint (seconds) from a 429 response.

    Anthropic sends ``retry-after`` (integer seconds) on a 429. The
    value lives on ``exc.response.headers`` for ``APIStatusError``
    subclasses. Returns None when absent / unparseable so the caller
    falls back to exponential backoff.
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        secs = float(raw)
    except (TypeError, ValueError):
        return None
    # Defensive clamp — never sleep longer than the backoff cap even if
    # the server hands back an unreasonable hint.
    return max(0.0, min(secs, _BACKOFF_CAP_SECONDS))


def invoke_with_rate_limit_retry(
    fn: Callable[[], _T],
    *,
    label: str,
    deadline_seconds: float | None = None,
) -> _T:
    """Call ``fn()`` with 429-aware, deadline-bounded persistent retry.

    ``fn`` is a zero-arg thunk wrapping a single ``ChatAnthropic`` /
    structured-LLM / ReAct ``.invoke()``. On ``anthropic.RateLimitError``
    (or a 429 ``APIStatusError``) this honors the ``retry-after``
    response header when present, otherwise sleeps
    ``min(base * 2**attempt, cap)`` with jitter, and **keeps retrying
    until an overall wall-clock deadline** (``deadline_seconds``,
    default ``RATE_LIMIT_RETRY_DEADLINE_SECONDS`` ≈ 75 min) is reached
    — NOT a small fixed attempt count.

    This is the "long retry mechanism" of the all-agents-strict rework
    (Brian, 2026-05-16). When the deadline is exceeded the 429 is
    re-raised — the caller does NOT degrade-and-continue; the run
    hard-fails (status:ERROR, nothing promoted) because we get nothing
    from a process whose agents didn't all run.

    Any non-429 exception propagates immediately and unchanged — this
    wrapper deliberately does NOT swallow or retry schema errors,
    recursion exhaustion, missing-key errors, etc. Those keep flowing
    to the caller's hard-fail path (a non-429 failure still fails the
    run under the directive — a partial process produces nothing of
    value).
    """
    if deadline_seconds is None:
        # Read at call time (NOT default-arg bind) so a test / Lambda
        # env that monkeypatches the module constant takes effect.
        deadline_seconds = RATE_LIMIT_RETRY_DEADLINE_SECONDS
    start = time.monotonic()
    deadline = start + deadline_seconds
    last_exc: BaseException | None = None
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised below if not 429
            if not _is_rate_limit_error(exc):
                raise
            last_exc = exc
            now = time.monotonic()
            elapsed = now - start
            hint = _retry_after_seconds(exc)
            if hint is not None:
                delay = hint
            else:
                delay = min(
                    _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    _BACKOFF_CAP_SECONDS,
                )
            # Decorrelated jitter so parallel teams don't re-burst in
            # lockstep when the TPM window resets.
            delay += random.uniform(0.0, min(delay, 5.0))
            # Deadline check: if even a minimal sleep would land us past
            # the deadline, give up now and propagate the 429. The
            # caller's hard-fail rule turns this into status:ERROR.
            if now + delay >= deadline:
                log.error(
                    "[rate_limit_retry:%s] org 429 persisted past the "
                    "%.0f min deadline (%.0fs elapsed, %d attempts) — "
                    "propagating. Per the all-agents-strict directive "
                    "the caller HARD-FAILS the run (no signals.json / "
                    "email / DB write); already-succeeded agents stay "
                    "persisted so an SF redrive only re-attempts the "
                    "still-missing ones.",
                    label, deadline_seconds / 60.0, elapsed, attempt,
                )
                raise
            log.warning(
                "[rate_limit_retry:%s] Haiku/Sonnet 429 (org TPM "
                "ceiling) attempt %d — backing off %.1fs (%s); %.0fs of "
                "%.0fs deadline elapsed",
                label, attempt, delay,
                "retry-after header" if hint is not None
                else "exponential",
                elapsed, deadline_seconds,
            )
            time.sleep(delay)
    # Unreachable — the loop either returns or raises.
    assert last_exc is not None  # pragma: no cover
    raise last_exc  # pragma: no cover


# ── SOTA structured-output retry with validation feedback ─────────────────────


# Default retry budget for a single ``with_structured_output(...).invoke()``
# call. Industry-standard tool-use SOTA is "retry with validation error fed
# back as correction context" — surfaced in this codebase by the 2026-05-24
# Saturday SF healthcare-team failure where the LLM emitted 'medium_high'
# for a Pydantic ``Literal['low','medium','high']`` field. A single bad roll
# on a 6-agent fan-out should not hard-fail the cycle when the fix is to
# re-prompt with the schema violation as context.
STRUCTURED_OUTPUT_MAX_RETRIES = 2


def invoke_structured_with_validation_retry(
    structured_llm,
    messages: list,
    *,
    label: str,
    ls_metadata: dict | None = None,
    max_retries: int = STRUCTURED_OUTPUT_MAX_RETRIES,
) -> dict:
    """Invoke a structured-output handle, retrying on Pydantic
    ``ValidationError`` with the specific schema violation fed back to the
    model as correction context.

    The SOTA tool-use pattern for structured output: when the LLM emits a
    value that doesn't fit the schema (e.g., ``'medium_high'`` for a
    ``Literal['low','medium','high']`` field, or a string where a float
    was expected), don't hard-fail — re-prompt with the exact
    ``ValidationError`` so the model can correct the specific field on the
    retry. Anthropic + OpenAI tool-use docs both describe this pattern as
    the institutional default for production structured-output pipelines.

    Composes with ``invoke_with_rate_limit_retry`` (called inside each
    attempt) — the rate-limit retry handles 429 backoff; this outer
    retry handles schema-validation correction. Different failure classes,
    different cures; the wrappers stack cleanly.

    Args:
        structured_llm: a structured-output-bound LLM handle (typically
            produced via the langchain ``with_structured_output`` call at
            the consumer's bind site with ``include_raw=True`` so this
            wrapper can inspect ``parsing_error``).
        messages: list of input messages (typically a single ``HumanMessage``).
        label: log/metric label for retry traces (``f'qual:{team}:extract'``).
        ls_metadata: LangSmith metadata dict forwarded as
            ``config={'metadata': ls_metadata}``.
        max_retries: max retry attempts on validation error (default 2 →
            up to 3 total LLM calls per logical extraction).

    Returns:
        The final ``extract_resp`` dict — ``{'raw': AIMessage, 'parsed':
        Schema | None, 'parsing_error': Exception | None}``. On success
        ``parsed`` is populated and ``parsing_error`` is None. On terminal
        failure (all retries exhausted) ``parsing_error`` carries the LAST
        ``ValidationError`` and the caller's existing fail-loud branch
        (e.g., ``raise RuntimeError(...)``) fires as before.
    """
    from langchain_core.messages import HumanMessage

    current_messages = list(messages)
    ls_metadata = ls_metadata or {}
    final_resp: dict = {}

    for attempt in range(max_retries + 1):
        attempt_label = f"{label}:attempt={attempt + 1}/{max_retries + 1}"
        final_resp = invoke_with_rate_limit_retry(
            lambda: structured_llm.invoke(
                current_messages,
                config={"metadata": ls_metadata},
            ),
            label=attempt_label,
        )
        parsing_error = final_resp.get("parsing_error")
        if parsing_error is None:
            if attempt > 0:
                log.info(
                    "[%s] structured-output succeeded after %d validation-retry "
                    "attempt(s)",
                    label, attempt,
                )
            return final_resp

        # Parse failed; decide whether to retry.
        if attempt >= max_retries:
            log.warning(
                "[%s] structured-output failed after %d validation-retry "
                "attempt(s) — propagating last ValidationError: %s",
                label, max_retries, parsing_error,
            )
            return final_resp

        # Build a correction message that names the specific schema violation
        # so the LLM can fix the offending field directly. The raw AIMessage
        # from the failed attempt is included so the model sees its own prior
        # output in context (full conversation, not just a fresh prompt).
        correction = HumanMessage(content=(
            f"Your prior response failed schema validation:\n\n"
            f"{type(parsing_error).__name__}: {parsing_error}\n\n"
            f"Please re-submit your response with the schema corrections "
            f"applied. Use ONLY exact values specified in the schema — "
            f"for enum/Literal fields use the listed values verbatim "
            f"(no synonyms, no compound values like 'medium_high', no "
            f"rephrasings, no additions). For other typed fields match "
            f"the exact type (string vs number vs boolean vs list). "
            f"Preserve all the substantive content from your prior "
            f"response — only fix the schema violation."
        ))

        raw = final_resp.get("raw")
        if raw is not None:
            current_messages = list(messages) + [raw, correction]
        else:
            current_messages = list(messages) + [correction]

        log.info(
            "[%s] structured-output parse failed on attempt %d/%d "
            "(%s: %s) — re-prompting with schema-violation context",
            label, attempt + 1, max_retries + 1,
            type(parsing_error).__name__, parsing_error,
        )

    return final_resp  # pragma: no cover — loop always returns


def extract_tool_calls(messages: list) -> list[dict]:
    """Extract tool call records from LangGraph message history."""
    calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                calls.append({
                    "tool": tc.get("name", ""),
                    "input_summary": str(tc.get("args", {}))[:200],
                })
        elif hasattr(msg, "type") and msg.type == "tool":
            calls.append({
                "tool": getattr(msg, "name", "unknown"),
                "status": "executed",
            })
    return calls


def get_final_text(messages: list) -> str:
    """Get the last AI message text from a LangGraph message history."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
            if isinstance(msg.content, str):
                return msg.content
            elif isinstance(msg.content, list):
                texts = [
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in msg.content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                return "\n".join(texts)
    return ""
