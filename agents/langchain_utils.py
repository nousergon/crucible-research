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


# ── Per-call request timeout (config#687) ─────────────────────────────────
# Per-``.invoke()`` HTTP request ceiling on every agent ChatAnthropic
# instance. langchain-anthropic / the underlying SDK default to NO request
# timeout, so a single silently-stalled call (a hung connection that never
# 429s and never streams) can consume the entire sector-team budget — the
# root cause of the 2026-06-06 sector-team tail-latency blowout (#687).
#
# This bounds ONE HTTP request, distinct from the two retry layers above:
#   - ``max_retries`` (SECTOR_TEAM_LLM_MAX_RETRIES) = SDK-level attempt count
#   - ``RATE_LIMIT_RETRY_DEADLINE_SECONDS`` = outer 429-aware wall-clock
# A hung call that is neither retried (not a 429) nor progressing was
# previously bounded by neither; this is the missing per-request guard.
#
# 300 s default — generous vs the 5-9 min full-run norm for a single
# agent call, tight enough that one stuck call can't eat the ~15-min
# (now larger) budget. Env override ``SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS``
# (clamped 30 s .. 20 min so a typo can't disable or unbound it). A timed-out
# request raises, which ``invoke_with_rate_limit_retry`` treats as a
# non-429 error and propagates — turning a hang into a fast, visible failure
# instead of a deadline-consuming stall.
def _resolve_request_timeout_seconds() -> float:
    raw = os.environ.get("SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS")
    if raw is None:
        return 300.0
    try:
        secs = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "[llm_request_timeout] SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS=%r "
            "unparseable — using 300 s default", raw,
        )
        return 300.0
    return max(30.0, min(secs, 20.0 * 60.0))


# Resolved at import; tests monkeypatch this attribute directly.
SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS: float = _resolve_request_timeout_seconds()

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


# ── ReAct malformed-tool-history recovery (transient 400) ─────────────────────


# Anthropic rejects a message list in which a ``tool_use`` block is not
# immediately followed by its matching ``tool_result`` block (HTTP 400,
# ``invalid_request_error``: "messages.N: `tool_use` ids were found
# without `tool_result` blocks immediately after: toolu_…"). With the
# prebuilt ``create_react_agent`` loop this is produced sporadically by
# the MODEL (e.g. a ``max_tokens``-truncated tool_use emission), NOT by
# our message construction — surfaced 2026-06-13 hard-failing the
# consumer + healthcare sector teams (2 of 6) on the Saturday SF while the
# other four teams ran clean off the identical code path.
#
# Each ``agent.invoke`` starts a FRESH graph state (the input is just the
# user message), so re-invoking re-rolls the entire ReAct loop from a
# clean history and clears the transient malformation. This is the ReAct
# analogue of ``invoke_structured_with_validation_retry`` (added for the
# 2026-05-24 'medium_high' Literal single-bad-roll): a single bad roll on
# the 6-team fan-out must not hard-fail the whole all-agents-strict cycle.
# After ``max_retries`` fresh re-rolls the 400 propagates unchanged and
# the caller's hard-fail rule fires (status:ERROR) — a genuinely
# deterministic malformation still fails loud.
REACT_MALFORMED_HISTORY_MAX_RETRIES = 2


def _is_recoverable_tool_use_400(exc: BaseException) -> bool:
    """True iff ``exc`` is the recoverable Anthropic 400 about a
    ``tool_use`` block missing its following ``tool_result`` block.

    Matched narrowly (400 / invalid_request_error AND the specific
    dangling-tool_use phrasing) so it never swallows an unrelated 400.
    """
    status = getattr(exc, "status_code", None)
    low = str(exc).lower()
    is_400 = (
        status == 400
        or "error code: 400" in low
        or "invalid_request_error" in low
    )
    if not is_400:
        return False
    return (
        "tool_use" in low
        and "tool_result" in low
        and "were found without" in low
    )


def invoke_react_with_recovery(
    invoke_thunk: Callable[[], _T],
    *,
    label: str,
    max_retries: int = REACT_MALFORMED_HISTORY_MAX_RETRIES,
) -> _T:
    """Run a fresh ``create_react_agent`` ``.invoke()`` thunk with bounded
    retry on the recoverable malformed-tool-history 400.

    429 backoff is handled INSIDE each attempt — the thunk is wrapped in
    ``invoke_with_rate_limit_retry``, so a 429 keeps riding the long
    deadline as before. This outer layer only re-rolls the rare 400 that
    ``invoke_with_rate_limit_retry`` re-raises unchanged. Every other
    exception (incl. ``GraphRecursionError`` and a deadline-exhausted 429)
    propagates immediately to the caller's existing handlers.

    Args:
        invoke_thunk: zero-arg thunk wrapping a single ReAct
            ``agent.invoke({...}, config={...})`` call.
        label: log/metric label (``f'quant:{team}:react'``).
        max_retries: extra fresh re-rolls on the recoverable 400 (default
            2 → up to 3 total ReAct invocations).
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return invoke_with_rate_limit_retry(invoke_thunk, label=label)
        except BaseException as exc:  # noqa: BLE001 — re-raised below if terminal
            if attempt > max_retries or not _is_recoverable_tool_use_400(exc):
                raise
            log.warning(
                "[react_recovery:%s] malformed tool_use/tool_result history "
                "400 (attempt %d/%d) — re-rolling a fresh ReAct invocation "
                "from clean state: %s",
                label, attempt, max_retries + 1, exc,
            )


# ── Pre-send tool_use/tool_result pairing repair (structured-tool-use discipline) ──
#
# Anthropic's Messages API rejects a message list in which an assistant
# ``tool_use`` block is not immediately answered by a matching
# ``tool_result`` block (HTTP 400, ``invalid_request_error``: "messages.N:
# `tool_use` ids were found without `tool_result` blocks immediately
# after: toolu_…"). config#1065: on the 2026-06-13 Saturday run two of
# six sector teams (consumer, healthcare) hard-failed on exactly this 400
# while four teams ran clean off the identical code path — an INTERMITTENT
# malformation, surfaced sporadically by the prebuilt ``create_react_agent``
# loop (e.g. a ``max_tokens``-truncated / aborted tool_use emission that
# never received its ToolMessage answer before the next turn was assembled).
#
# ``invoke_react_with_recovery`` (below) is the OUTER safety net — it
# re-rolls the whole ReAct loop from a fresh graph state on this 400.
# This block is the INNER, structural fix per the SOTA structured-tool-use
# discipline (config#1065 fix-plan 1+2): a ``pre_model_hook`` that runs
# before EVERY LLM call inside the ReAct loop and REPAIRS the message list
# in place — dropping any orphan ``tool_use`` (an assistant turn whose
# tool_call ids are not all answered by a following ToolMessage) — so a
# malformed history can never be SENT to the API in the first place. The
# repaired view is fed via ``llm_input_messages`` (the create_react_agent
# pre_model_hook contract), which does NOT mutate the persisted graph
# state — it only sanitizes what the model sees on each turn.


def _ai_tool_call_ids(msg) -> list[str]:
    """tool_call ids emitted by an assistant message, in order.

    Handles the langchain ``AIMessage.tool_calls`` shape (list of dicts
    carrying ``id``). Returns [] for any non-assistant / no-tool-call
    message. Drops blank/None ids defensively (a tool_call with no id can
    never be paired and must be treated as an orphan-bearing turn)."""
    if getattr(msg, "type", None) != "ai":
        return []
    tcs = getattr(msg, "tool_calls", None) or []
    ids: list[str] = []
    for tc in tcs:
        # tool_calls dicts; be tolerant of object-shaped entries too.
        tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        ids.append(tid if tid else "")
    return ids


def _tool_result_id(msg) -> str | None:
    """The ``tool_call_id`` a ToolMessage answers, or None if not a tool msg."""
    if getattr(msg, "type", None) != "tool":
        return None
    tid = getattr(msg, "tool_call_id", None)
    return tid if tid else None


def find_orphan_tool_use_ids(messages: list) -> list[str]:
    """Return the tool_call ids that are emitted by an assistant message
    but never answered by a following ToolMessage (the malformed-history
    signature that produces the Anthropic 400).

    A ``tool_use`` id is satisfied iff some message AFTER its assistant
    turn is a ToolMessage with a matching ``tool_call_id``. Anthropic
    requires the answer to be in the *immediately following* message, but
    for repair purposes the looser "answered anywhere later" rule is
    correct: an id answered later is not an orphan (the create_react_agent
    loop always appends the ToolMessage right after), and an id answered
    nowhere is the orphan we must drop. A blank/None id (a tool_call the
    model emitted without an id) is always an orphan."""
    answered: set[str] = set()
    for msg in messages:
        rid = _tool_result_id(msg)
        if rid is not None:
            answered.add(rid)
    orphans: list[str] = []
    for msg in messages:
        for tid in _ai_tool_call_ids(msg):
            if not tid or tid not in answered:
                orphans.append(tid or "<missing-id>")
    return orphans


def validate_tool_use_pairing(messages: list) -> None:
    """Raise ``ValueError`` iff ``messages`` contains an orphan ``tool_use``.

    The pre-send invariant: every assistant ``tool_use`` id has a
    following ``tool_result`` (ToolMessage). Used by tests and as an
    optional assertion; the runtime path REPAIRS rather than raises (see
    ``repair_tool_use_pairing``)."""
    orphans = find_orphan_tool_use_ids(messages)
    if orphans:
        raise ValueError(
            "malformed message history: tool_use id(s) without a following "
            f"tool_result block: {orphans}"
        )


def repair_tool_use_pairing(messages: list) -> tuple[list, list[str]]:
    """Return ``(repaired_messages, dropped_ids)`` with every orphan
    ``tool_use`` turn removed so the list can never trigger the Anthropic
    "tool_use ids were found without tool_result blocks" 400.

    Repair rule (drop, never fabricate): an assistant message that emits
    ANY unanswered tool_call id is dropped WHOLE — we never synthesize a
    fake ``tool_result``, because a fabricated result would feed the model
    a hallucinated observation. We also drop any ToolMessage left dangling
    after its assistant turn is removed (a tool_result with no preceding
    tool_use is itself a 400). An assistant turn whose tool_calls are ALL
    answered is kept verbatim. Idempotent: a clean list returns unchanged
    with an empty ``dropped_ids``."""
    orphan_ids = set(find_orphan_tool_use_ids(messages))
    if not orphan_ids:
        return list(messages), []

    # Pass 1: drop assistant turns that carry an orphan tool_call, and
    # record which tool_call ids those turns DID legitimately request
    # (so the matching ToolMessages, now answering a removed turn, also go).
    dropped_ids: list[str] = []
    drop_answer_ids: set[str] = set()
    kept_after_pass1: list = []
    for msg in messages:
        ids = _ai_tool_call_ids(msg)
        if ids and any((not tid) or tid in orphan_ids for tid in ids):
            # This assistant turn is the orphan-bearing one — drop it whole.
            for tid in ids:
                dropped_ids.append(tid or "<missing-id>")
                if tid:
                    drop_answer_ids.add(tid)
            continue
        kept_after_pass1.append(msg)

    # Pass 2: drop ToolMessages that answered a now-removed assistant turn.
    repaired: list = []
    for msg in kept_after_pass1:
        rid = _tool_result_id(msg)
        if rid is not None and rid in drop_answer_ids:
            continue
        repaired.append(msg)

    return repaired, dropped_ids


def make_tool_use_repair_hook(*, label: str):
    """Build a ``create_react_agent`` ``pre_model_hook`` that drops orphan
    ``tool_use`` turns from the LLM-input view on every ReAct turn.

    The hook runs immediately before each LLM call. It reads the current
    ``messages`` from graph state, repairs the tool_use/tool_result pairing,
    and returns ``{"llm_input_messages": repaired}`` — the
    ``create_react_agent`` contract for "send THESE messages to the model
    this turn without rewriting the persisted state". So even if the loop's
    accumulated history ever holds an orphan ``tool_use`` (truncated /
    aborted tool emission), the model never SEES it and the 400 never fires.

    On a repair, emits a WARN naming the dropped id(s) so the event is
    flow-doctor-detectable (mirrors the existing retry WARN pattern) — a
    silent repair would hide a real upstream malformation. No-op (returns
    the messages unchanged) on a clean history."""

    def _hook(state: dict) -> dict:
        messages = state.get("messages", []) or []
        repaired, dropped = repair_tool_use_pairing(messages)
        if dropped:
            log.warning(
                "[react_repair:%s] dropped %d orphan tool_use turn(s) from the "
                "LLM-input view before send (ids=%s) — pre-send pairing repair "
                "prevented an Anthropic 'tool_use without tool_result' 400 "
                "(config#1065). Persisted graph state is unchanged.",
                label, len(dropped), dropped,
            )
        return {"llm_input_messages": repaired}

    return _hook


# ── Single send-time tool_use/tool_result pairing chokepoint (config#2255) ────
#
# The "no orphan ``tool_use`` may reach the Anthropic Messages API" invariant
# bit the fleet TWICE, ~1 month apart, at two DIFFERENT call sites — the
# prebuilt ``create_react_agent`` ReAct loop (config#1065, fixed with the
# ``pre_model_hook`` above) and the ``invoke_structured_with_validation_retry``
# structured-retry chokepoint (config#2245, fixed with a per-site belt). Both
# fixes were correct at their layer, but each was a PER-CALL-PATH application
# of the same invariant: any new or existing message-assembling send path that
# forgets to repair the pairing before ``.invoke()`` can reintroduce the exact
# same 400. The SOTA response to the second occurrence of a class is to LIFT
# the invariant to a single chokepoint rather than patch the next site.
#
# ``invoke_anthropic_safe`` is that chokepoint: it runs ``repair_tool_use_pairing``
# IMMEDIATELY before delegating to the underlying ``.invoke()`` (composed with
# ``invoke_with_rate_limit_retry`` for 429 backoff), so an orphan ``tool_use``
# turn is structurally incapable of reaching the API regardless of which send
# path assembled the history. Every Anthropic-backed multi-message ``.invoke()``
# in the repo routes through it; the ReAct ``pre_model_hook`` stays (it is
# contractually required by ``create_react_agent`` and shares the same
# ``repair_tool_use_pairing`` primitive). A CI guard
# (``tests/test_no_raw_anthropic_invoke.py``) proves a naive new call site that
# bypasses the chokepoint is caught.


def invoke_anthropic_safe(
    handle,
    messages: list,
    *,
    label: str,
    deadline_seconds: float | None = None,
    **invoke_kwargs,
):
    """Send ``messages`` through ``handle.invoke`` with the tool_use/tool_result
    pairing invariant enforced AT SEND TIME (config#2255).

    This is the SINGLE send-time chokepoint for every Anthropic-backed
    multi-message ``.invoke()`` in crucible-research. It:

      1. Runs ``repair_tool_use_pairing(messages)`` immediately before the
         send — dropping any PRE-EXISTING orphan ``tool_use`` turn (an
         assistant turn whose tool_call ids are not all answered by a
         following ``ToolMessage``) so the Anthropic Messages API can never
         reject the send with "``tool_use`` ids were found without
         ``tool_result`` blocks immediately after" (config#1065 / config#2245).
      2. Delegates to ``handle.invoke(repaired, **invoke_kwargs)`` wrapped in
         ``invoke_with_rate_limit_retry`` — so 429 backoff composes exactly as
         it did at the per-site call this replaces.

    Fail-loud is preserved, deliberately and narrowly:

      * The wrapper ONLY drops pre-existing orphan ``tool_use`` turns; it NEVER
        fabricates a ``tool_result`` (a synthetic result would feed the model a
        hallucinated observation — see ``repair_tool_use_pairing``). On a drop
        it WARNs naming the dropped id(s), so the event is flow-doctor-visible.
      * It does NOT inspect or swallow the RESULT: a structured-output parse
        failure still comes back in the ``parsing_error`` field and a genuine
        (non-429) exception still propagates unchanged to the caller's hard-fail
        branch — ``invoke_with_rate_limit_retry`` only retries 429s.

    This is a HOW-it-runs (plumbing) lift only: it changes nothing about WHAT
    any agent concludes — a clean history (the normal path) routes through
    byte-for-byte unchanged (``repair_tool_use_pairing`` is a no-op on a clean
    list).

    Args:
        handle: an Anthropic-backed LLM handle exposing ``.invoke(messages,
            **kw)`` — a ``ChatAnthropic`` or a ``with_structured_output(...)``
            handle (the ``include_raw=True`` dict-returning shape is passed
            through untouched).
        messages: the message list to send (a single ``HumanMessage`` on the
            normal path; a multi-turn ``[..., raw_tool_use, ToolMessage]``
            history on a correction re-send).
        label: log/metric label, forwarded to ``invoke_with_rate_limit_retry``.
        deadline_seconds: optional override of the 429-retry wall-clock deadline
            (e.g. a best-effort shadow challenger bounding itself tight).
        **invoke_kwargs: forwarded verbatim to ``handle.invoke`` (typically
            ``config={"metadata": ...}``).

    Returns:
        Whatever ``handle.invoke`` returns (an ``AIMessage`` for a bare LLM, or
        the ``{'raw', 'parsed', 'parsing_error'}`` dict for an
        ``include_raw=True`` structured handle).
    """
    repaired, dropped = repair_tool_use_pairing(messages)
    if dropped:
        log.warning(
            "[anthropic_safe:%s] dropped %d PRE-EXISTING orphan tool_use "
            "turn(s) (ids=%s) from the message list before send — the "
            "send-time pairing chokepoint prevented an Anthropic 'tool_use "
            "without tool_result' 400 (config#2255). No tool_result was "
            "fabricated; a genuine parse/validation failure still propagates.",
            label, len(dropped), dropped,
        )
    return invoke_with_rate_limit_retry(
        lambda: handle.invoke(repaired, **invoke_kwargs),
        label=label,
        deadline_seconds=deadline_seconds,
    )


# ── Runtime structured-output truncation detection (config#1294) ──────────────
#
# Root-cause guard for the truncation-bug class. When an Anthropic tool-call
# response hits the ``max_tokens`` ceiling MID-emission, the API returns
# ``stop_reason == "max_tokens"`` and langchain captures the PARTIAL tool
# parameter block — a half-written JSON object — as a raw string. With
# ``with_structured_output`` that surfaces downstream as a confusing Pydantic
# error far from the cause (e.g. ``catalysts: Input should be a valid list …
# input_type=str``), because the partial argument is handed to the schema as
# an un-parseable fragment rather than the list it was meant to become.
#
# Before config#1294 NOTHING in the repo inspected ``stop_reason`` at runtime;
# the only guard was a hand-estimated static budget table
# (``tests/test_schema_max_tokens_audit.py``) which missed a real incident.
# The SOTA fix is a RUNTIME check at this shared structured-output chokepoint:
# after EVERY ``with_structured_output(...).invoke()`` we read the response's
# ``stop_reason`` and, if it is ``max_tokens``, RAISE a clear, explicit error
# AT THE ROOT CAUSE — naming the call site / schema and the token budget — so
# the failure is diagnosed as "the model ran out of output tokens" instead of
# masquerading as a schema-shape bug.
#
# Truncation is NOT a transient roll and NOT fixable by re-prompting against
# the SAME budget, so this raises IMMEDIATELY (before the validation-retry
# loop burns attempts re-prompting a budget that will truncate again). It is a
# structural chokepoint guard, not a per-call-site patch: every agent / eval /
# producer that funnels through ``invoke_structured_with_validation_retry``
# inherits it. The cure is to raise the offending site's ``max_tokens`` (or
# shrink the schema), which the descriptive message points the operator to.


class StructuredOutputTruncationError(RuntimeError):
    """A structured-output call was truncated by the ``max_tokens`` ceiling.

    Raised at the shared structured-output chokepoint when an Anthropic
    response carries ``stop_reason == "max_tokens"`` (config#1294). Carries
    the call-site label, the schema name, and (when discoverable) the
    ``max_tokens`` budget so the root cause is unambiguous — distinct from a
    Pydantic ``ValidationError``, which is what this error PREVENTS the
    truncation from masquerading as.
    """


# stop_reason / finish_reason values that mean "output hit the token ceiling".
# Anthropic uses ``max_tokens``; the alias set guards against a langchain/SDK
# reshuffle that surfaces the OpenAI-style ``length`` finish_reason instead.
_TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "length"})


def _response_metadata_of(raw) -> dict:
    """Best-effort ``response_metadata`` mapping from a structured-output raw.

    ``with_structured_output(include_raw=True)`` puts the underlying
    ``AIMessage`` under the ``"raw"`` key. The truncation signal lives in
    ``AIMessage.response_metadata`` (a provider-controlled dict carrying
    ``"stop_reason"``). Returns {} for any shape that isn't a dict-bearing
    AIMessage so a metadata-less response can never crash the guard."""
    md = getattr(raw, "response_metadata", None)
    return md if isinstance(md, dict) else {}


def _is_truncated_response(raw) -> bool:
    """True iff the structured-output ``raw`` AIMessage was ``max_tokens``-truncated.

    Reads ``raw.response_metadata['stop_reason']`` (the langchain-anthropic
    access path confirmed in THIS codebase — see ``evals/judge.py`` and
    ``graph/llm_cost_tracker.py``) and also tolerates a ``finish_reason``
    alias. Case/whitespace-insensitive. False for any non-AIMessage or a
    response with no recognizable truncation stop reason."""
    md = _response_metadata_of(raw)
    for key in ("stop_reason", "finish_reason"):
        val = md.get(key)
        if isinstance(val, str) and val.strip().lower() in _TRUNCATION_STOP_REASONS:
            return True
    return False


def _max_tokens_of(raw) -> int | None:
    """Best-effort ``max_tokens`` budget the truncated call ran under.

    Not always present on the response metadata; returned for the error
    message when discoverable so the operator knows which budget to raise.
    Checks the request-echo shapes langchain/Anthropic may surface."""
    md = _response_metadata_of(raw)
    for key in ("max_tokens", "max_output_tokens"):
        val = md.get(key)
        if isinstance(val, int) and val > 0:
            return val
    return None


def raise_if_truncated(resp: dict, *, label: str, schema_name: str | None = None) -> None:
    """Raise ``StructuredOutputTruncationError`` iff ``resp`` was ``max_tokens``-truncated.

    The runtime truncation guard (config#1294). ``resp`` is the
    ``with_structured_output(include_raw=True)`` dict
    (``{'raw': AIMessage, 'parsed': …, 'parsing_error': …}``). When the raw
    response's ``stop_reason`` indicates the ``max_tokens`` ceiling was hit,
    this raises a clear, explicit error naming the call site, schema, and
    token budget — at the ROOT CAUSE — so the truncation is never allowed to
    surface downstream as a confusing Pydantic shape error. No-op on a
    non-truncated response."""
    raw = resp.get("raw") if isinstance(resp, dict) else None
    if not _is_truncated_response(raw):
        return
    budget = _max_tokens_of(raw)
    budget_str = f"{budget}" if budget is not None else "unknown (raise the call site's max_tokens)"
    schema_str = schema_name or "<unknown schema>"
    raise StructuredOutputTruncationError(
        f"[{label}] structured-output call was TRUNCATED by the max_tokens "
        f"ceiling (stop_reason='max_tokens'): the model ran out of output "
        f"tokens mid-tool-call, so only a PARTIAL parameter block was "
        f"emitted. schema={schema_str}, max_tokens={budget_str}. This is the "
        f"ROOT CAUSE — without this guard the partial argument surfaces "
        f"downstream as a confusing Pydantic shape error (e.g. 'Input should "
        f"be a valid list … input_type=str'). FIX: raise the max_tokens "
        f"budget for this call site (or shrink the schema / batch size); "
        f"re-prompting against the same budget will only truncate again."
    )


def _schema_name_of(structured_llm) -> str | None:
    """Best-effort schema name bound into a ``with_structured_output`` handle.

    The handle is a ``RunnableSequence``; the bound schema is not part of a
    stable public contract, so this is purely for a friendlier error message
    and returns None on any shape it can't introspect (the guard still fires
    with a ``<unknown schema>`` placeholder)."""
    for attr in ("name", "__name__"):
        val = getattr(structured_llm, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


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
    from langchain_core.messages import HumanMessage, ToolMessage

    current_messages = list(messages)
    ls_metadata = ls_metadata or {}
    final_resp: dict = {}
    schema_name = _schema_name_of(structured_llm)

    for attempt in range(max_retries + 1):
        attempt_label = f"{label}:attempt={attempt + 1}/{max_retries + 1}"
        # Route through the shared send-time pairing chokepoint (config#2255):
        # it repairs any orphan tool_use in ``current_messages`` immediately
        # before the send (closing the gap that the OLD per-site belt below the
        # loop left open on attempt 0 — the caller's own ``messages`` are now
        # repaired on the very first send too) and composes with the 429 retry.
        final_resp = invoke_anthropic_safe(
            structured_llm,
            current_messages,
            label=attempt_label,
            config={"metadata": ls_metadata},
        )
        # Runtime truncation guard (config#1294): a max_tokens-truncated
        # response yields a PARTIAL tool-call that would otherwise surface as
        # a confusing Pydantic shape error below. Detect it FIRST and raise a
        # clear root-cause error — re-prompting against the same budget would
        # only truncate again, so this raises immediately instead of burning
        # a validation-retry attempt.
        raise_if_truncated(final_resp, label=label, schema_name=schema_name)
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

        # Build a correction that names the specific schema violation so the
        # LLM can fix the offending field directly. The failed ``raw`` AIMessage
        # from the prior attempt is included so the model sees its own output in
        # context (full conversation, not just a fresh prompt).
        correction_text = (
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
        )

        raw = final_resp.get("raw")
        # ``with_structured_output`` is FORCED TOOL-USE: on a validation failure
        # the ``raw`` AIMessage carries a ``tool_use`` block. Anthropic requires
        # the VERY NEXT message to be a ``tool_result`` answering that tool_use —
        # appending a plain HumanMessage here orphans the tool_use and 400s with
        # "`tool_use` ids were found without `tool_result` blocks immediately
        # after" (config#2245: the 2026-07-11 Saturday Research failure, exposed
        # once crucible-research#402 routed the held-thesis update through this
        # chokepoint). The SOTA correction is to answer the failed tool_call with
        # a ToolMessage/tool_result carrying the violation — this keeps the
        # tool_use/tool_result pairing valid AND feeds the schema error back so
        # the model re-emits a corrected tool call.
        tool_call_ids = (
            [tid for tid in _ai_tool_call_ids(raw) if tid] if raw is not None else []
        )
        if raw is not None and tool_call_ids:
            # One tool_result per tool_use id in the failed turn (structured
            # output emits a single call, but pair them all defensively — every
            # tool_use MUST be answered or the 400 recurs).
            correction_msgs = [
                ToolMessage(content=correction_text, tool_call_id=tid)
                for tid in tool_call_ids
            ]
            current_messages = list(messages) + [raw, *correction_msgs]
        elif raw is not None:
            # Plain-text structured output (no tool_use block in ``raw``) — a
            # HumanMessage correction is already well-formed, no pairing needed.
            current_messages = list(messages) + [
                raw, HumanMessage(content=correction_text)
            ]
        else:
            current_messages = list(messages) + [HumanMessage(content=correction_text)]

        # NOTE: no per-site pairing belt here anymore — the next iteration's
        # send goes through ``invoke_anthropic_safe`` (config#2255), which runs
        # ``repair_tool_use_pairing`` at send time. Repairing here as well would
        # be redundant (and would only ever fire on a caller-supplied malformed
        # history, which the chokepoint now handles on attempt 0 too).

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


# ── Silent step-budget exhaustion detection (config#1822) ────────────────────
#
# ``langgraph.prebuilt.chat_agent_executor``'s internal ``call_model`` node
# tracks a managed ``remaining_steps`` channel derived from the
# ``recursion_limit`` passed at invoke time. When ``remaining_steps < 2``
# and the model still wants to call a tool, the executor does NOT raise
# ``GraphRecursionError`` — it swaps in a synthetic AIMessage with this
# EXACT literal content and returns the graph invocation normally (see
# ``_are_more_steps_needed`` / ``call_model`` in that module). That is a
# SEPARATE, earlier-firing guard than the graph-level recursion_limit crash
# that ``GraphRecursionError`` covers; ``quant_analyst.py`` /
# ``qual_analyst.py`` only ever caught the latter.
#
# Investigation (config#1822, 2026-07-03 defensives/financials/consumer
# qual + healthcare/industrials quant): the 7/3 weekly's "90-102 tool
# calls, 0 assessments" teams all terminated with this exact sentinel as
# ``final_text``. Because it arrives as a normal, non-empty AI message,
# the calling code's ``if not final_text: raise ...`` guard never fires
# either — the decoupled structured-output extraction runs against this
# boilerplate string, correctly finds no picks/assessments in it, and the
# team silently contributes zero output with ``error=None, partial=False``.
# That is exactly the "vanishes instead of surfacing" failure the issue
# calls out: score_aggregator's ALL-AGENTS-STRICT gate only inspects
# ``error`` and ``partial`` — this failure mode set neither.
LANGGRAPH_STEP_BUDGET_EXHAUSTED_SENTINEL = (
    "Sorry, need more steps to process this request."
)


def is_step_budget_exhausted_sentinel(final_text: str | None) -> bool:
    """True iff ``final_text`` IS (not merely contains) the prebuilt
    ReAct executor's silent step-budget-exhaustion bailout message.

    Exact-match (after stripping) rather than substring — this is a fixed
    literal owned by langgraph, not agent-authored prose that might
    legitimately reference running out of time/steps.
    """
    return bool(final_text) and final_text.strip() == (
        LANGGRAPH_STEP_BUDGET_EXHAUSTED_SENTINEL
    )


# ── Bounded transcript serialization (decision-review / L4567) ────────────────
#
# The ReAct agents (quant, qual) discard their full reasoning after the
# structured-output extraction — only the parsed picks survive. For
# retrospective "why did/didn't you pick X" review we persist the agent's
# actual reasoning into the captured decision artifact. This is the
# ground-truth "why" — strictly better than re-deriving it via a fresh
# replay, which would generate a NEW rationalization that may not match
# what actually drove the decision (the explicit rationale for capturing
# the transcript over LangGraph checkpointing — see ARCHITECTURE).
#
# Bounded HARD so it is safe to ride in the captured ``agent_output`` AND
# in the persisted ``SectorTeamOutput`` graph state without bloat: the
# agent's reasoning + tool-call args are kept; bulky tool-RESPONSE payloads
# (price/factor data blobs) are truncated, and the whole transcript is
# capped. This is observability data, not decision input — it must never
# dominate the state it rides in.

_TRANSCRIPT_MAX_TOTAL_CHARS = 8_000
_TRANSCRIPT_MAX_MSG_CHARS = 800
_TRANSCRIPT_MAX_TOOL_RESPONSE_CHARS = 300


def _message_text(msg) -> str:
    """Best-effort plain-text content of a LangChain message (str or
    list-of-content-blocks form)."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                # tool_use blocks are surfaced via tool_calls, not here
            else:
                parts.append(str(b))
        return "\n".join(p for p in parts if p)
    return str(content)


def serialize_transcript(
    messages: list,
    *,
    max_total_chars: int = _TRANSCRIPT_MAX_TOTAL_CHARS,
    max_msg_chars: int = _TRANSCRIPT_MAX_MSG_CHARS,
    max_tool_response_chars: int = _TRANSCRIPT_MAX_TOOL_RESPONSE_CHARS,
) -> list[dict]:
    """Compact, hard-bounded serialization of a ReAct message history.

    Returns a list of ``{role, content, [tool, tool_calls]}`` dicts
    preserving the agent's reasoning and its tool-call arguments, with
    bulky tool *responses* truncated and the overall size capped. When the
    cap is reached the remaining messages are dropped and a final
    ``{"_truncated": "<n> message(s) omitted (transcript size cap)"}``
    marker is appended so truncation is never silent."""
    out: list[dict] = []
    total = 0
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", None) or "unknown"
        is_tool = role == "tool"
        cap = max_tool_response_chars if is_tool else max_msg_chars
        text = _message_text(msg)
        if len(text) > cap:
            text = text[:cap] + "…[truncated]"
        entry: dict = {"role": role, "content": text}
        if is_tool:
            name = getattr(msg, "name", None)
            if name:
                entry["tool"] = name
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name", ""), "args": str(tc.get("args", {}))[:200]}
                for tc in tool_calls
            ]
        entry_size = len(text) + sum(len(str(v)) for v in entry.get("tool_calls", []))
        if total + entry_size > max_total_chars and out:
            out.append(
                {"_truncated": f"{len(messages) - i} message(s) omitted "
                               f"(transcript size cap)"}
            )
            break
        out.append(entry)
        total += entry_size
    return out
