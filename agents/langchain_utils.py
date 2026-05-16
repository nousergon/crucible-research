"""
Shared utilities for sector-team agents.

Two concerns:

  1. Extracting tool calls / final text from LangGraph message histories
     (used by the quant + qual ReAct agents).
  2. A 429-aware retry wrapper for every sector-team Haiku call
     (``invoke_with_rate_limit_retry``).

The 429 wrapper exists because the 6-team parallel ``Send()`` fan-out
in ``graph/research_graph.py`` bursts over the org's Haiku input-TPM
ceiling (450,000 tokens/min, claude-haiku-4-5). On the 2026-05-16
recovery run that surfaced as ``RateLimitError 429`` aborting
defensives/financials/technology. langchain's ChatAnthropic default
``max_retries`` is insufficient for a *sustained* org-level 429 (it
backs off a couple of times then gives up), so we (a) bump the
constructor ``max_retries`` at every call site and (b) wrap each
``.invoke()`` in this helper, which honors the ``retry-after`` response
header when Anthropic sends one. Only 429 / rate-limit errors are
retried here — every other exception propagates unchanged so the
existing strict-mode / partial / isolation contracts are preserved.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

# Constructor-level retry budget for every sector-team ChatAnthropic
# instance. langchain-anthropic defaults to 2, which is exhausted
# almost immediately under a sustained org-wide 450K-TPM 429. This is
# the inner SDK-level backoff; ``invoke_with_rate_limit_retry`` is the
# outer, retry-after-aware backoff around the whole ``.invoke()``.
SECTOR_TEAM_LLM_MAX_RETRIES = 8

# Outer-wrapper attempt cap. With exponential backoff (base 4s, cap 60s)
# 6 attempts spans up to ~3-4 min of org-429 wait per call — long enough
# to ride out a TPM-window reset without unbounded Lambda time burn.
_DEFAULT_MAX_ATTEMPTS = 6
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
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> _T:
    """Call ``fn()`` with 429-aware exponential backoff.

    ``fn`` is a zero-arg thunk wrapping a single ``ChatAnthropic`` /
    structured-LLM / ReAct ``.invoke()``. On ``anthropic.RateLimitError``
    (or a 429 ``APIStatusError``) this honors the ``retry-after``
    response header when present, otherwise sleeps
    ``min(base * 2**attempt, cap)`` with jitter, up to ``max_attempts``.

    Any non-429 exception propagates immediately and unchanged — this
    wrapper deliberately does NOT swallow or retry schema errors,
    recursion exhaustion, missing-key errors, etc. Those keep flowing
    to the existing strict-mode / partial / per-team-isolation paths.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised below if not 429
            if not _is_rate_limit_error(exc):
                raise
            last_exc = exc
            if attempt == max_attempts:
                log.error(
                    "[rate_limit_retry:%s] org 429 persisted after %d "
                    "attempts — propagating (team will be recorded as "
                    "failed; per-team isolation keeps the run alive)",
                    label, max_attempts,
                )
                raise
            hint = _retry_after_seconds(exc)
            if hint is not None:
                delay = hint
            else:
                delay = min(
                    _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    _BACKOFF_CAP_SECONDS,
                )
            # Decorrelated jitter so 6 parallel teams don't re-burst in
            # lockstep when the TPM window resets.
            delay += random.uniform(0.0, min(delay, 5.0))
            log.warning(
                "[rate_limit_retry:%s] Haiku 429 (org TPM ceiling) on "
                "attempt %d/%d — backing off %.1fs (%s)",
                label, attempt, max_attempts, delay,
                "retry-after header" if hint is not None
                else "exponential",
            )
            time.sleep(delay)
    # Unreachable — the loop either returns or raises.
    assert last_exc is not None
    raise last_exc


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
