"""
LLM cost telemetry tracker — wires Anthropic SDK token counts into the
``DecisionArtifact`` capture stream and emits a per-call JSONL cost stream.

This is the consumer side of the per-run LLM cost telemetry workstream
(ROADMAP P1). The price-table primitive lives in ``alpha_engine_lib.cost``
(v0.2.3+); the capture wrapper lives in ``alpha_engine_lib.decision_capture``.
This module is the glue: a LangChain callback handler that pulls
``input_tokens`` / ``output_tokens`` / ``cache_read_input_tokens`` /
``cache_creation_input_tokens`` from every ``ChatAnthropic`` response, plus
a context manager that scopes the accumulation to one logical agent
decision (one node — sector_team, macro_economist, or ic_cio).

**Two output streams** (PR 2 + PR 3):

1. **Aggregate per-node** — populated ``ModelMetadata`` (sum across all
   LLM calls in the scope) is stashed for ``_capture_if_enabled`` to
   embed in the existing ``DecisionArtifact`` at
   ``s3://alpha-engine-research/decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json``.
2. **Per-call JSONL stream (PR 3)** — every Anthropic API call gets one
   line in a JSONL flushed at scope exit to
   ``s3://alpha-engine-research/decision_artifacts/_cost_raw/{YYYY-MM-DD}/{run_id}/{agent_id}.jsonl``.
   PR 3's daily aggregator (``scripts/aggregate_costs.py``) reads these
   into a single ``_cost/{date}/cost.parquet`` for analytics.

Both streams gated on ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` — same
flag governs both since they're complementary views of the same data.

Why both per-node and per-call streams:

- The existing decision-capture surface writes ONE artifact per node
  boundary in ``research_graph.py``. A sector_team node fires up to four
  LLM calls (quant ReAct → qual ReAct → peer_review quant addition →
  peer_review joint finalization); summing them into ``ModelMetadata``
  yields the total cost attributed to that team's decision.
- Per-call granularity (one row per Anthropic API call) flows through
  the JSONL sink so the daily aggregator can recompute costs against
  any pricing table version without replaying agents.

Usage pattern (one ChatAnthropic instance, one agent decision)::

    from graph.llm_cost_tracker import (
        get_cost_telemetry_callback, track_llm_cost,
    )
    from agents.prompt_loader import load_prompt

    cb = get_cost_telemetry_callback()
    llm = ChatAnthropic(model=PER_STOCK_MODEL, callbacks=[cb], ...)

    user_prompt = load_prompt("cio_decision")
    with track_llm_cost(
        agent_id="ic_cio",
        node_name="cio_node",
        prompt=user_prompt,
        run_type="weekly_research",
    ):
        result = llm.with_structured_output(CIORawOutput).invoke([msg])

    # later in cio_node, _capture_if_enabled reads the populated metadata:
    metadata = pop_metadata_for("ic_cio")  # ModelMetadata, fully populated

**Hard-fail surface (per ``feedback_no_silent_fails``):**

- ``track_llm_cost`` enter/exit must be balanced — RuntimeError on stack
  underflow.
- The callback hard-fails if ``on_llm_end`` fires with no usage metadata
  on the response. Anthropic responses always carry usage; missing
  fields are an upstream change worth surfacing immediately.
- Price-table lookup tolerates unknown model (warn + leave cost_usd=0)
  so a missing yaml entry doesn't halt the SF on cost-telemetry; token
  counts are still captured and the aggregator can re-derive later if a
  card is added retroactively.
- **Run budget ceiling (PR 5)** — per-run cumulative cost is tracked in
  a ContextVar. At each ``track_llm_cost`` frame exit, the cumulative
  cost for the active ``run_id`` is compared against
  ``ALPHA_ENGINE_RUN_BUDGET_USD`` (default $100). When exceeded, the
  frame raises ``RunBudgetExceededError`` AFTER the in-flight call
  completes — runaway prompt loops kill the run before they bill us
  into the next decade. Disable for tests / smoke runs by setting
  ``ALPHA_ENGINE_RUN_BUDGET_USD=0`` (zero or negative disables the
  check; positive values enforce the ceiling).

Thread + async safety: state lives in ``ContextVar`` so it's per-task in
asyncio, per-thread otherwise. ``contextvars.copy_context()`` semantics
take care of LangGraph fan-out propagation.

Workstream design: ``alpha-engine-config/private-docs/ROADMAP.md`` line ~1708.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from alpha_engine_lib.cost import (
    PriceCardLookupError,
    PriceTable,
    PriceTableLoadError,
    ToolFeeTable,
    compute_cost,
    load_pricing,
    load_tool_fees,
    recompute_cost,
)
from alpha_engine_lib.decision_capture import (
    FullPromptContext,
    ModelMetadata,
)
from langchain_core.callbacks import BaseCallbackHandler

from agents.prompt_loader import LoadedPrompt
from config import _find_config

logger = logging.getLogger(__name__)


# ── Per-call JSONL sink (PR 3) ────────────────────────────────────────────


_COST_RAW_BUCKET = "alpha-engine-research"
_COST_RAW_PREFIX = "decision_artifacts/_cost_raw"
# Schema v2 (additive): adds ``web_search_requests`` + ``web_fetch_requests``
# per-call columns to capture Anthropic server-tool usage. Backwards-compat
# at the aggregator side — older v1 rows omit the new fields and the daily
# parquet treats missing as zero. Bump again whenever a column is added or
# renamed (per CLAUDE.md S3 contract safety).
_PER_CALL_SCHEMA_VERSION = 2
_DECISION_CAPTURE_ENV_VAR = "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"


class CostRawWriteError(RuntimeError):
    """Raised when the S3 write of the per-call JSONL sink fails.

    Per ``feedback_no_silent_fails``, the cost stream does not swallow
    S3 errors — every captured artifact must land or the run hard-fails.
    Mirror of ``DecisionCaptureWriteError`` for the cost-raw stream.
    """


class RunBudgetExceededError(RuntimeError):
    """Raised at frame exit when the per-run cumulative cost exceeds the
    configured ceiling.

    Surfaces the offending run_id, the cumulative cost, and the
    threshold so operators can map the failure back to the SF run that
    blew the budget. Per ``feedback_no_silent_fails`` — runaway prompt
    loops should kill the run, not silently bill us into the next
    decade. Disable for tests / smoke runs via
    ``ALPHA_ENGINE_RUN_BUDGET_USD=0`` (zero or negative disables).
    """

    def __init__(self, *, run_id: str, cumulative_cost_usd: float, ceiling_usd: float) -> None:
        self.run_id = run_id
        self.cumulative_cost_usd = cumulative_cost_usd
        self.ceiling_usd = ceiling_usd
        super().__init__(
            f"[cost_tracker] run budget exceeded: run_id={run_id!r} "
            f"cumulative_cost=${cumulative_cost_usd:.4f} > "
            f"ceiling=${ceiling_usd:.4f}. Set "
            f"ALPHA_ENGINE_RUN_BUDGET_USD=<higher_value> to raise the cap, "
            f"or =0 to disable. Investigate the offending agent before "
            f"raising the cap — a runaway prompt loop will keep growing.",
        )


def _is_capture_enabled() -> bool:
    """Same flag as decision_capture's gate; the two streams co-fire."""
    return os.environ.get(_DECISION_CAPTURE_ENV_VAR, "").lower() in (
        "true", "1", "yes",
    )


# ── Run budget ceiling (PR 5) ─────────────────────────────────────────────


_RUN_BUDGET_ENV_VAR = "ALPHA_ENGINE_RUN_BUDGET_USD"
_RUN_BUDGET_DEFAULT_USD = 100.0


def _resolve_run_budget_ceiling() -> float:
    """Read ``ALPHA_ENGINE_RUN_BUDGET_USD`` per-call (allows test toggling).

    Returns 0.0 (which disables enforcement) on parse failure rather than
    raising — a malformed env var shouldn't take down a Sat SF run; the
    parse-failure log is loud enough that the operator notices.

    Returns a positive float to enforce the ceiling; zero or negative
    disables enforcement entirely. Default $100 reflects the
    workstream's "runaway prompt loop should fire well before the
    monthly Anthropic bill" intent.
    """
    raw = os.environ.get(_RUN_BUDGET_ENV_VAR, "")
    if not raw:
        return _RUN_BUDGET_DEFAULT_USD
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[cost_tracker] ALPHA_ENGINE_RUN_BUDGET_USD=%r is not a number; "
            "disabling run-budget enforcement (set to a positive float to "
            "enable, 0 to explicitly disable)",
            raw,
        )
        return 0.0


# Per-run cumulative cost accumulator. Keyed by run_id; values in USD.
# Lives in a ContextVar so async + threaded runs (LangGraph Send fan-out)
# share the same accumulator across frames within a single pipeline
# invocation.
_run_cost_totals: contextvars.ContextVar[dict[str, float]] = contextvars.ContextVar(
    "alpha_engine_cost_tracker_run_totals", default={},
)


def _accumulate_run_cost(run_id: str, frame_cost_usd: float) -> float:
    """Add ``frame_cost_usd`` to the per-run accumulator and return the new total."""
    totals = dict(_run_cost_totals.get())
    totals[run_id] = totals.get(run_id, 0.0) + frame_cost_usd
    new_total = totals[run_id]
    _run_cost_totals.set(totals)
    return new_total


def _reset_run_cost_totals_for_tests() -> None:
    """Clear the per-run accumulator — exposed for tests that simulate
    multiple runs in one process. Not used in production code paths."""
    _run_cost_totals.set({})


def get_run_cost(run_id: str) -> float:
    """Return the cumulative cost for ``run_id`` so far in this pipeline
    invocation. Used by tests and, optionally, by diagnostic logging.
    Returns 0.0 if the run has no recorded cost (unknown run_id, or
    capture flag was off so cost wasn't computed)."""
    return _run_cost_totals.get().get(run_id, 0.0)


def _build_cost_raw_s3_key(*, capture_dt: datetime, run_id: str, agent_id: str) -> str:
    """Compute the JSONL S3 key for a frame's flushed cost stream.

    Format: ``decision_artifacts/_cost_raw/{YYYY-MM-DD}/{run_id}/{agent_id}.jsonl``

    Date-partitioned by capture date (UTC) to match the
    ``decision_artifacts/{Y}/{M}/{D}/...`` partition scheme. ``run_id``
    is the leaf-prior so all artifacts from one pipeline run can be
    discovered by listing the directory; the aggregator scans by date.
    """
    date_str = capture_dt.strftime("%Y-%m-%d")
    return f"{_COST_RAW_PREFIX}/{date_str}/{run_id}/{agent_id}.jsonl"


# ── Pricing table loader (cached at module level) ────────────────────────


_PRICING_FILENAME = "model_pricing.yaml"
_PRICING_SUBDIR = "cost"

# Anthropic snapshot suffix: -YYYYMMDD appended to the family name
# (e.g. "claude-haiku-4-5-20251001"). Pricing yaml keys cards by family
# only ("claude-haiku-4-5") because Anthropic publishes prices per family,
# not per snapshot. Strip the suffix before lookup so a runtime model
# pinned to a snapshot still matches its family card.
import re as _re

_SNAPSHOT_SUFFIX_RE = _re.compile(r"-\d{8}$")


def _normalize_model_for_pricing(model_name: str) -> str:
    """Drop Anthropic ``-YYYYMMDD`` snapshot suffix for price-card lookup.

    ``claude-haiku-4-5-20251001`` → ``claude-haiku-4-5``
    ``claude-sonnet-4-6``         → ``claude-sonnet-4-6`` (no-op)

    Origin: 2026-05-02 Saturday SF Research step halted with
    ``PriceCardLookupError: No price card for model
    'claude-haiku-4-5-20251001' active on 2026-05-02`` after the
    cost-telemetry primitive (alpha-engine-lib v0.2.4) shipped earlier
    today and ``recompute_cost`` started enforcing exact-match lookups.
    The runtime pin in ``config/universe.yaml`` (``per_stock_model:
    claude-haiku-4-5-20251001``) is a snapshot ID; the pricing yaml
    cards are keyed per family. Normalizing here is more robust than
    duplicating yaml entries because every future Anthropic snapshot
    inherits the family card without config maintenance.
    """
    return _SNAPSHOT_SUFFIX_RE.sub("", model_name)
_price_table: Optional[PriceTable] = None
_tool_fee_table: Optional[ToolFeeTable] = None


def _resolve_pricing_path() -> Path:
    """Locate ``cost/model_pricing.yaml`` via the same search order as
    other config files (sibling clone → parent → $GITHUB_WORKSPACE →
    Lambda-staged ``config/``). Uses ``subdir="cost"`` so the file lives
    alongside the existing ``research/`` and ``predictor/`` subtrees in
    the alpha-engine-config repo.
    """
    return _find_config(_PRICING_FILENAME, subdir=_PRICING_SUBDIR)


def _load_price_table() -> PriceTable:
    """Load + cache the price table on first use.

    Hard-fails on missing / malformed pricing yaml. The cached table is
    a module-level singleton — pricing changes require a Lambda redeploy
    or process restart, which matches the cadence at which Anthropic
    publishes new rates.
    """
    global _price_table
    if _price_table is None:
        path = _resolve_pricing_path()
        _price_table = load_pricing(path)
        logger.info(
            "[cost_tracker] loaded price table from %s (%d cards)",
            path, len(_price_table.cards),
        )
    return _price_table


def _load_tool_fee_table() -> Optional[ToolFeeTable]:
    """Load + cache the server-tool fee table from the same yaml.

    Returns ``None`` when the yaml has no ``tool_fees:`` section so the
    frame-exit + per-call pricing paths can short-circuit without
    raising. Once a section is present, hard-fails on malformed entries
    per ``load_tool_fees`` semantics — silent-zero on a real fee slice
    would bury cost regressions (per ``feedback_no_silent_fails``).
    """
    global _tool_fee_table
    if _tool_fee_table is None:
        path = _resolve_pricing_path()
        try:
            _tool_fee_table = load_tool_fees(path)
            logger.info(
                "[cost_tracker] loaded tool-fee table from %s (%d fees)",
                path, len(_tool_fee_table.fees),
            )
        except PriceTableLoadError as exc:
            logger.info(
                "[cost_tracker] no tool_fees: section in %s — server-tool "
                "request counts will flow through to JSONL but cost-USD "
                "for tool fees will be left at None (%s)",
                path, exc,
            )
            return None
    return _tool_fee_table


def _reset_price_table_for_tests() -> None:
    """Clear the cached price + tool-fee tables — exposed for tests that
    swap yaml fixtures between cases. Not used in production code paths.
    """
    global _price_table, _tool_fee_table
    _price_table = None
    _tool_fee_table = None


# ── Per-frame accumulator (one frame = one ``track_llm_cost`` scope) ─────


RunType = Literal["weekly_research", "morning", "EOD"]


@dataclass
class _Frame:
    """One agent decision's accumulating LLM-cost surface.

    Multiple LLM calls fire within a single ``track_llm_cost`` scope
    (especially for ReAct loops); this frame sums their token counts
    AND keeps a per-call buffer (``per_call_rows``) for the JSONL sink
    flushed at scope exit. The model_name is captured from the FIRST
    call's response and locked — mixed-model frames (e.g. agent retries
    on a different model) would corrupt cost recompute, so we surface
    that explicitly.
    """

    agent_id: str
    sector_team_id: Optional[str]
    node_name: Optional[str]
    run_type: Optional[RunType]
    prompt: Optional[LoadedPrompt]
    run_id: Optional[str] = None

    model_name: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    # Server-side tool request counts (Anthropic ``Message.usage.server_tool_use``).
    # Flat per-request fees, billed via ``ToolFee`` rather than the per-1M-token
    # rate on ``PriceCard``. Zero-defaulted; the frame-exit pricing path passes
    # a ``ToolFeeTable`` only when the cumulative requests > 0 — the wiring is
    # dormant on agents that don't bind Anthropic server tools.
    web_search_requests: int = 0
    web_fetch_requests: int = 0
    call_count: int = 0
    enter_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Per-call rows buffered in memory for the JSONL sink. One dict per
    # Anthropic API call; flushed in JSONL form at scope exit.
    per_call_rows: list[dict] = field(default_factory=list)


# Stack of frames — supports nested ``track_llm_cost`` (rare, but legal).
_frame_stack: contextvars.ContextVar[list[_Frame]] = contextvars.ContextVar(
    "alpha_engine_cost_tracker_frames", default=[],
)

# Populated metadata, keyed by agent_id — read by ``_capture_if_enabled``
# at the node boundary right after ``track_llm_cost`` exits.
_completed_metadata: contextvars.ContextVar[
    dict[str, tuple[ModelMetadata, FullPromptContext]]
] = contextvars.ContextVar(
    "alpha_engine_cost_tracker_completed", default={},
)


def _current_frame() -> Optional[_Frame]:
    stack = _frame_stack.get()
    return stack[-1] if stack else None


# ── LangChain callback handler ────────────────────────────────────────────


class CostTelemetryCallback(BaseCallbackHandler):
    """Sums tokens from every Anthropic LLM call into the active frame.

    LangChain fires ``on_llm_end(response, run_id, parent_run_id, **kwargs)``
    once per chat-model API call. For ReAct agents, this fires multiple
    times per ``agent.invoke()`` (once per ReAct iteration).

    The handler reads usage from the response in this priority order:

    1. ``response.generations[0][0].message.usage_metadata`` — the modern
       AIMessage shape exposed by langchain-core ≥0.2 / langchain-anthropic
       ≥0.2. Carries ``input_tokens``, ``output_tokens``, and the optional
       ``input_token_details`` sub-dict with ``cache_read`` + ``cache_creation``.
    2. ``response.llm_output["token_usage"]`` — legacy fallback shape
       from older langchain-anthropic. No cache fields available; logs
       a warning if hit because we lose cache visibility.

    No usage on the response → ``RuntimeError`` per ``feedback_no_silent_fails``.
    Anthropic always returns usage; a missing field signals an SDK shape
    change worth investigating.
    """

    def on_llm_end(self, response, **kwargs: Any) -> None:  # type: ignore[override]
        frame = _current_frame()
        if frame is None:
            # No active scope — call happened outside ``track_llm_cost``.
            # Could be a non-research code path or a misuse; log + skip.
            # We don't raise here because the callback is attached at
            # ChatAnthropic construction time and may outlive the scope.
            logger.debug(
                "[cost_tracker] on_llm_end fired with no active frame — "
                "call accumulation skipped",
            )
            return

        usage = self._extract_usage(response)
        frame.input_tokens += usage["input_tokens"]
        frame.output_tokens += usage["output_tokens"]
        frame.cache_read_tokens += usage["cache_read_tokens"]
        frame.cache_create_tokens += usage["cache_create_tokens"]
        frame.web_search_requests += usage["web_search_requests"]
        frame.web_fetch_requests += usage["web_fetch_requests"]
        frame.call_count += 1

        # Lock model_name on first call. ChatAnthropic doesn't expose
        # the configured model on the response message in every shape,
        # so we pull from the AIMessage's ``response_metadata.model_name``
        # if available, else from llm_output's ``model``.
        call_model = self._extract_model_name(response)
        if frame.model_name is None:
            frame.model_name = call_model

        # PR 3: per-call row for the JSONL sink. cost_usd is left at
        # None here — populated at flush time when the price table is
        # loaded once for all rows. Persisting tokens immutable +
        # cost-derived-at-flush matches the workstream's "tokens are
        # immutable, dollars are derived" rule.
        frame.per_call_rows.append({
            "schema_version": _PER_CALL_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_seq": frame.call_count,
            "model_name": call_model or frame.model_name,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cache_read_tokens": usage["cache_read_tokens"],
            "cache_create_tokens": usage["cache_create_tokens"],
            "web_search_requests": usage["web_search_requests"],
            "web_fetch_requests": usage["web_fetch_requests"],
        })

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        """Extract token counts + server-tool request counts from an LLMResult.

        Returns a dict with six fields populated. Hard-fails if the
        response carries no usage information at all.

        Server-tool request counts (``web_search_requests`` /
        ``web_fetch_requests``) come from the Anthropic raw-usage payload
        on ``message.response_metadata['usage']['server_tool_use']`` (the
        full SDK response that langchain-anthropic stashes alongside the
        normalized ``usage_metadata``). Zero-defaulted when absent — the
        modern-shape path is the one consumers exercise; agents that
        don't bind Anthropic server tools simply emit zero requests.
        """
        # Modern path: AIMessage.usage_metadata for token counts +
        # response_metadata['usage']['server_tool_use'] for tool requests.
        try:
            generations = response.generations
            if generations and generations[0]:
                message = getattr(generations[0][0], "message", None)
                if message is not None and getattr(message, "usage_metadata", None):
                    um = message.usage_metadata
                    details = um.get("input_token_details") or {}
                    tool_counts = CostTelemetryCallback._extract_server_tool_use(message)
                    return {
                        "input_tokens": int(um.get("input_tokens", 0)),
                        "output_tokens": int(um.get("output_tokens", 0)),
                        "cache_read_tokens": int(details.get("cache_read", 0)),
                        "cache_create_tokens": int(details.get("cache_creation", 0)),
                        "web_search_requests": tool_counts["web_search_requests"],
                        "web_fetch_requests": tool_counts["web_fetch_requests"],
                    }
        except (AttributeError, IndexError, KeyError, TypeError):
            pass

        # Legacy path: response.llm_output["token_usage"]. No cache or
        # server-tool fields available — both zero-defaulted.
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if token_usage:
            logger.warning(
                "[cost_tracker] response carried legacy token_usage shape — "
                "cache_read + cache_create + server_tool_use counts are not "
                "available; consider upgrading langchain-anthropic for cache "
                "+ tool-fee visibility",
            )
            return {
                "input_tokens": int(token_usage.get("input_tokens", 0)),
                "output_tokens": int(token_usage.get("output_tokens", 0)),
                "cache_read_tokens": 0,
                "cache_create_tokens": 0,
                "web_search_requests": 0,
                "web_fetch_requests": 0,
            }

        raise RuntimeError(
            "[cost_tracker] LLM response carried no usage metadata; "
            "Anthropic responses always include usage — this indicates an "
            "SDK shape change worth investigating. "
            "Per feedback_no_silent_fails, refusing to record a 0-token call."
        )

    @staticmethod
    def _extract_server_tool_use(message: Any) -> dict[str, int]:
        """Pull web_search / web_fetch request counts off the response.

        langchain-anthropic stashes the raw Anthropic SDK response on
        ``message.response_metadata`` (mapping shape: keys like
        ``"model_name"``, ``"usage"``, ``"stop_reason"``). The ``usage``
        sub-dict mirrors ``anthropic.types.Usage`` and includes
        ``server_tool_use`` when the model exercised server tools in the
        turn. Field names match the SDK's ``ServerToolUsage``.

        Returns zero-defaulted dict on every miss path (no
        ``response_metadata`` mapping, no ``usage`` sub-key, no
        ``server_tool_use`` sub-key, or non-int values). Server-tool
        usage is optional in the SDK; absence is not a failure mode.
        """
        try:
            md = getattr(message, "response_metadata", None) or {}
            raw_usage = md.get("usage") if isinstance(md, dict) else None
            stu = raw_usage.get("server_tool_use") if isinstance(raw_usage, dict) else None
            if not isinstance(stu, dict):
                return {"web_search_requests": 0, "web_fetch_requests": 0}
            return {
                "web_search_requests": int(stu.get("web_search_requests", 0) or 0),
                "web_fetch_requests": int(stu.get("web_fetch_requests", 0) or 0),
            }
        except (AttributeError, KeyError, TypeError, ValueError):
            return {"web_search_requests": 0, "web_fetch_requests": 0}

    @staticmethod
    def _extract_model_name(response: Any) -> Optional[str]:
        """Best-effort model-name extraction from the response.

        Returns None if the response shape doesn't expose it; the frame
        falls back to the ChatAnthropic instance's configured model
        (passed at ``track_llm_cost`` exit via ``recompute_cost``'s lookup).
        """
        try:
            generations = response.generations
            if generations and generations[0]:
                message = getattr(generations[0][0], "message", None)
                if message is not None:
                    md = getattr(message, "response_metadata", None) or {}
                    name = md.get("model_name") or md.get("model")
                    if name:
                        return str(name)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass

        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            return llm_output.get("model") or llm_output.get("model_name")
        return None


_callback_singleton: Optional[CostTelemetryCallback] = None


# ── Per-call JSONL flush ──────────────────────────────────────────────────


def _enrich_row_with_frame_dimensions(
    row: dict, frame: _Frame, *, table: Optional[PriceTable],
    tool_fee_table: Optional[ToolFeeTable] = None,
) -> dict:
    """Stamp frame-level dimensions onto a per-call row + compute cost_usd.

    Frame-level fields (run_id, agent_id, sector_team_id, node_name,
    run_type, prompt_*) are constant within a frame so they live on the
    frame, not on each row. We copy them onto each row at flush time so
    the JSONL stream is self-describing (the aggregator doesn't need
    out-of-band frame metadata to interpret a row).

    Cost is derived from token counts × the active price card at row
    timestamp. Hard-fails on lookup error per ``feedback_no_silent_fails``
    only when the price table is loadable; if pricing yaml is missing or
    malformed (caught at frame exit), we set cost_usd=None and the
    aggregator can re-derive later.
    """
    enriched = dict(row)
    enriched["run_id"] = frame.run_id
    enriched["agent_id"] = frame.agent_id
    enriched["sector_team_id"] = frame.sector_team_id
    enriched["node_name"] = frame.node_name
    enriched["run_type"] = frame.run_type
    enriched["prompt_id"] = frame.prompt.name if frame.prompt else None
    enriched["prompt_version"] = frame.prompt.version if frame.prompt else None
    enriched["prompt_version_hash"] = frame.prompt.hash if frame.prompt else None

    cost: Optional[float] = None
    model_name = enriched.get("model_name")
    if table is not None and model_name and model_name != "unknown":
        try:
            row_dt = datetime.fromisoformat(enriched["timestamp"].replace("Z", "+00:00")) \
                if isinstance(enriched.get("timestamp"), str) else frame.enter_time
            card = table.get(_normalize_model_for_pricing(model_name), row_dt)
            # Per-row tool-fee pricing: build {tool_name: count} from the
            # row's request fields; resolve fees from the table only when
            # the row has non-zero requests (compute_cost raises if a
            # count has no matching fee). Per-row granularity matches the
            # JSONL contract — each row's cost_usd reflects its own tokens
            # AND its own tool requests, not a frame-level rollup.
            tool_requests: dict[str, int] = {}
            ws = int(enriched.get("web_search_requests", 0) or 0)
            wf = int(enriched.get("web_fetch_requests", 0) or 0)
            if ws > 0:
                tool_requests["web_search"] = ws
            if wf > 0:
                tool_requests["web_fetch"] = wf
            tool_fees = None
            if tool_requests and tool_fee_table is not None:
                tool_fees = {
                    name: tool_fee_table.get(name, row_dt)
                    for name in tool_requests
                }
            cost = compute_cost(
                input_tokens=enriched["input_tokens"],
                output_tokens=enriched["output_tokens"],
                cache_read_tokens=enriched["cache_read_tokens"],
                cache_create_tokens=enriched["cache_create_tokens"],
                card=card,
                tool_requests=tool_requests or None,
                tool_fees=tool_fees,
            )
        except PriceCardLookupError as exc:
            # Unknown model in the price table — log and leave cost None.
            # This is not a hard-fail because the JSONL is best-effort
            # for analytics; the aggregator can re-derive later if the
            # price table is updated.
            logger.warning(
                "[cost_tracker] no price card for model=%s at %s — "
                "row cost_usd left None (token counts preserved): %s",
                model_name, enriched.get("timestamp"), exc,
            )
    enriched["cost_usd"] = cost
    return enriched


def _flush_cost_rows_to_s3(
    *,
    frame: _Frame,
    table: Optional[PriceTable],
    tool_fee_table: Optional[ToolFeeTable] = None,
    s3_client: Any | None = None,
) -> Optional[str]:
    """Serialize ``frame.per_call_rows`` to JSONL + write to S3.

    Returns the S3 key written, or None if the buffer is empty (frames
    that opened but had no LLM calls don't emit a JSONL — keeps the
    prefix clean of empty objects).

    Raises :exc:`CostRawWriteError` on any S3 failure per
    ``feedback_no_silent_fails``. Mirrors the hard-fail posture of
    ``capture_decision``.
    """
    if not frame.per_call_rows:
        return None
    if not frame.run_id:
        # Without run_id we can't compute the S3 key. Log + skip — the
        # frame's metadata stash still works for the in-process
        # ``_capture_if_enabled`` consumer; only the JSONL is dropped.
        logger.warning(
            "[cost_tracker] frame for agent_id=%s closed with no run_id — "
            "JSONL flush skipped. Pass run_id= to track_llm_cost to enable.",
            frame.agent_id,
        )
        return None

    enriched_rows = [
        _enrich_row_with_frame_dimensions(
            row, frame, table=table, tool_fee_table=tool_fee_table,
        )
        for row in frame.per_call_rows
    ]
    body = "\n".join(json.dumps(row, default=str) for row in enriched_rows).encode("utf-8")

    s3_key = _build_cost_raw_s3_key(
        capture_dt=frame.enter_time,
        run_id=frame.run_id,
        agent_id=frame.agent_id,
    )
    client = s3_client if s3_client is not None else boto3.client("s3")
    try:
        client.put_object(
            Bucket=_COST_RAW_BUCKET,
            Key=s3_key,
            Body=body,
            ContentType="application/x-ndjson",
        )
    except (BotoCoreError, ClientError) as exc:
        raise CostRawWriteError(
            f"Failed to write cost-raw JSONL to "
            f"s3://{_COST_RAW_BUCKET}/{s3_key}: {exc}"
        ) from exc
    return s3_key


def get_cost_telemetry_callback() -> CostTelemetryCallback:
    """Return the process-singleton callback handler.

    Attached to every ``ChatAnthropic(callbacks=[...])`` constructor so a
    single instance accumulates across all agents. The frame stack does
    the per-decision scoping, not the callback identity.
    """
    global _callback_singleton
    if _callback_singleton is None:
        _callback_singleton = CostTelemetryCallback()
    return _callback_singleton


# ── Context manager: one frame per agent decision ────────────────────────


@contextmanager
def track_llm_cost(
    agent_id: str,
    *,
    sector_team_id: Optional[str] = None,
    node_name: Optional[str] = None,
    run_type: Optional[RunType] = "weekly_research",
    prompt: Optional[LoadedPrompt] = None,
    model_name_fallback: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Iterator[_Frame]:
    """Scope LLM token accumulation to one agent decision.

    Pushes a fresh ``_Frame`` onto the stack on enter; on exit, sums the
    accumulated tokens, recomputes USD cost via the cached price table,
    and stashes a populated ``ModelMetadata`` + ``FullPromptContext``
    keyed by ``agent_id`` for ``_capture_if_enabled`` to read.

    Parameters
    ----------
    agent_id
        Identifier matching the ``DecisionArtifact.agent_id`` field —
        e.g. ``"sector_team:technology"``, ``"macro_economist"``,
        ``"ic_cio"``.
    sector_team_id
        For sector-team agents only. Stamped onto ``ModelMetadata`` for
        cost-by-team drilldowns.
    node_name
        LangGraph node name for cost-by-node drilldowns.
    run_type
        ``weekly_research`` (default — the only research-side value) /
        ``morning`` (predictor) / ``EOD`` (executor). Future PRs in the
        cost-telemetry workstream will populate the latter two.
    prompt
        ``LoadedPrompt`` for the user-facing prompt of this decision.
        Stamps ``prompt_id`` + ``prompt_version`` on ``ModelMetadata``;
        also flows into ``FullPromptContext.user_prompt`` and
        ``prompt_version_hash``.
    model_name_fallback
        If the callback can't extract model_name from the response shape,
        use this. Should match what was passed to ``ChatAnthropic(model=...)``.
        Required for ``recompute_cost`` to look up the right card.
    run_id
        Pipeline-invocation identifier (typically ``state["run_date"]`` or
        Lambda's ``aws_request_id``). Stamped onto per-call JSONL rows
        and used as the partition key in the S3 path. Without it, the
        per-call JSONL flush is skipped (in-process metadata stash for
        ``_capture_if_enabled`` still works).

    Yields
    ------
    _Frame
        The active frame — useful for tests + diagnostic logging. Most
        callers ignore the yielded value.

    Raises
    ------
    RuntimeError
        If the frame stack underflows on exit (enter/exit imbalance —
        usually a bug in the calling code, e.g. swallowed exception).
    RunBudgetExceededError
        At frame exit when cumulative run cost exceeds
        ``ALPHA_ENGINE_RUN_BUDGET_USD`` (raised AFTER the JSONL flush
        so per-call detail is preserved on S3).
    CostRawWriteError
        If the per-call JSONL flush to S3 fails when capture is enabled.
    """
    frame = _Frame(
        agent_id=agent_id,
        sector_team_id=sector_team_id,
        node_name=node_name,
        run_type=run_type,
        prompt=prompt,
        run_id=run_id,
    )
    stack = _frame_stack.get()
    new_stack = stack + [frame]
    token = _frame_stack.set(new_stack)
    exception_raised = False
    try:
        yield frame
    except BaseException:
        # Track failure for the agent-runtime telemetry stream, then
        # re-raise. Distinct from the cost stream's "tokens captured but
        # cost_usd=0" partial-frame case — Failures=1 fires whenever the
        # body raises, regardless of whether any LLM call landed.
        exception_raised = True
        raise
    finally:
        # Best-effort per-agent CW telemetry emission. Runs in the
        # finally block so failure paths still emit (the post-finally
        # cost code only runs on the success path). Independent of the
        # decision-capture flag — Phase 2 observability is always-on.
        try:
            from graph.agent_telemetry import emit_agent_completion

            emit_agent_completion(
                agent_id=agent_id,
                enter_time=frame.enter_time,
                exception_raised=exception_raised,
                llm_call_count=frame.call_count,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "[cost_tracker] agent telemetry emission failed for "
                "agent_id=%s: %s", agent_id, exc,
            )

        # Pop the frame regardless of whether the body raised; we still
        # want to surface accumulated cost for partial decisions.
        current = _frame_stack.get()
        if not current or current[-1] is not frame:
            _frame_stack.reset(token)
            raise RuntimeError(
                f"[cost_tracker] frame stack imbalance for agent_id={agent_id!r}; "
                f"track_llm_cost enter/exit not properly nested",
            )
        _frame_stack.reset(token)

    # Build populated metadata + prompt context.
    model_name = frame.model_name or model_name_fallback
    if model_name is None:
        # No model_name resolved — log loudly and skip recompute. The
        # captured ModelMetadata still records token counts; cost_usd
        # stays 0 and PR 3's aggregator can recompute later if the
        # model is ever pinned to the row out-of-band. Keeping this as
        # a warning rather than a raise so a Lambda log shape change
        # doesn't take down a Sat SF run.
        logger.warning(
            "[cost_tracker] could not resolve model_name for agent_id=%s "
            "(call_count=%d); cost_usd left at 0",
            agent_id, frame.call_count,
        )
        model_name = "unknown"

    metadata = ModelMetadata(
        model_name=model_name,
        input_tokens=frame.input_tokens,
        output_tokens=frame.output_tokens,
        cache_read_tokens=frame.cache_read_tokens,
        cache_create_tokens=frame.cache_create_tokens,
        web_search_requests=frame.web_search_requests,
        web_fetch_requests=frame.web_fetch_requests,
        run_type=run_type,
        node_name=node_name,
        sector_team_id=sector_team_id,
        prompt_id=prompt.name if prompt else None,
        prompt_version=prompt.version if prompt else None,
    )

    # Compute cost if we have a real model and the price table is loadable.
    # The same loaded table is reused for the JSONL flush below — one load
    # covers the aggregate ModelMetadata + every per-call row.
    table_for_flush: Optional[PriceTable] = None
    tool_fee_table_for_flush: Optional[ToolFeeTable] = None
    if model_name != "unknown":
        try:
            table_for_flush = _load_price_table()
            # Tool-fee table is loaded best-effort — when the yaml has no
            # ``tool_fees:`` section, the loader returns None and the
            # recompute_cost path below only raises if the frame actually
            # accumulated non-zero server-tool requests. Dormant wiring on
            # agents that don't bind Anthropic server tools.
            tool_fee_table_for_flush = _load_tool_fee_table()
            # Normalize the snapshot suffix (e.g. -20251001) before lookup —
            # see _normalize_model_for_pricing docstring for the 2026-05-02
            # incident this fixes. We pass a copy with the family-only name
            # so the captured ModelMetadata keeps the snapshot ID
            # (analytics may want it) while pricing is keyed by family.
            metadata_for_pricing = metadata.model_copy(update={
                "model_name": _normalize_model_for_pricing(metadata.model_name),
            })
            recompute_cost(
                metadata_for_pricing, table_for_flush,
                tool_fee_table=tool_fee_table_for_flush,
                at=frame.enter_time,
            )
            metadata.cost_usd = metadata_for_pricing.cost_usd
        except PriceCardLookupError as exc:
            # No card for the (normalized) family at this date. Mirror the
            # per-row path's tolerance: log loudly, leave cost_usd at 0.
            # Token counts are still captured; the aggregator can re-derive
            # later if a card is added retroactively. Hard-failing here
            # would halt the SF on cost-telemetry — disproportionate to a
            # missing yaml entry.
            logger.warning(
                "[cost_tracker] no price card for model=%s (normalized=%s) at %s "
                "— frame cost_usd left at 0: %s",
                model_name,
                _normalize_model_for_pricing(model_name),
                frame.enter_time, exc,
            )
        except (PriceTableLoadError, FileNotFoundError) as exc:
            # Pricing yaml absent or malformed — record tokens, leave
            # cost_usd at 0, log loudly. Cost recompute is a downstream
            # concern and the aggregator can re-derive from tokens.
            logger.warning(
                "[cost_tracker] price-table load failed for agent_id=%s: %s "
                "(token counts captured; cost_usd left at 0)",
                agent_id, exc,
            )

    # FullPromptContext: capture rendered user_prompt text where we have
    # it (via the LoadedPrompt). System prompts are agent-specific and
    # not always loaded by name; downstream PR can plumb them.
    prompt_context = FullPromptContext(
        system_prompt=(
            f"<system prompt for agent_id={agent_id!r} not yet plumbed; "
            f"see agents/ for the gitignored prompt template files>"
        ),
        user_prompt=prompt.text if prompt else f"<user prompt for {agent_id} not provided>",
        prompt_version_hash=prompt.hash if prompt else None,
    )

    # Stash for the next ``_capture_if_enabled`` call to read.
    completed = dict(_completed_metadata.get())
    completed[agent_id] = (metadata, prompt_context)
    _completed_metadata.set(completed)

    # PR 5: per-run cumulative-cost accumulator. Accumulates regardless
    # of the capture flag (tests + smoke runs without S3 access still
    # benefit from the runaway-loop guard). The ceiling check runs
    # AFTER the JSONL flush below so operators can inspect the per-call
    # rows on S3 to diagnose what broke the budget — raising before the
    # flush would lose the calls that contributed to the breach.
    cumulative_after_frame: Optional[float] = None
    if run_id:
        cumulative_after_frame = _accumulate_run_cost(run_id, metadata.cost_usd)

    # PR 3: flush the per-call buffer to S3 as JSONL — gated on the
    # same ALPHA_ENGINE_DECISION_CAPTURE_ENABLED flag as the existing
    # decision-artifact capture. Hard-fail on S3 error per
    # feedback_no_silent_fails, mirroring DecisionCaptureWriteError.
    if _is_capture_enabled():
        try:
            written_key = _flush_cost_rows_to_s3(
                frame=frame, table=table_for_flush,
                tool_fee_table=tool_fee_table_for_flush,
            )
            if written_key:
                logger.debug(
                    "[cost_tracker] flushed %d per-call rows for agent_id=%s "
                    "to s3://%s/%s",
                    len(frame.per_call_rows), agent_id,
                    _COST_RAW_BUCKET, written_key,
                )
        except CostRawWriteError:
            # Hard-fail per design — flush failures must be loud so the
            # cost stream doesn't silently rot. Operators see the SF step
            # fail and investigate (typically IAM or bucket-existence).
            logger.error(
                "[cost_tracker] cost-raw JSONL write failed for "
                "agent_id=%s run_id=%s — raising to fail the run loud "
                "(per feedback_no_silent_fails). Disable via "
                "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED=false if S3/IAM is "
                "broken and you need to recover the run.",
                agent_id, frame.run_id,
            )
            raise

    logger.debug(
        "[cost_tracker] frame closed: agent_id=%s model=%s tokens=%d/%d/%d/%d "
        "cost=$%.6f calls=%d prompt=%s/%s",
        agent_id, model_name,
        frame.input_tokens, frame.output_tokens,
        frame.cache_read_tokens, frame.cache_create_tokens,
        metadata.cost_usd, frame.call_count,
        prompt.name if prompt else "<none>",
        prompt.version if prompt else "<none>",
    )

    # PR 5: hard ceiling check fires LAST — capture metadata + JSONL
    # flush both ran above, so even when the budget is breached the
    # captured artifacts on S3 + the in-process metadata stash for
    # ``_capture_if_enabled`` are intact. Operators can diagnose what
    # broke the budget by inspecting the JSONLs without re-running.
    if cumulative_after_frame is not None:
        ceiling = _resolve_run_budget_ceiling()
        if ceiling > 0 and cumulative_after_frame > ceiling:
            logger.error(
                "[cost_tracker] run budget exceeded for run_id=%s after "
                "frame agent_id=%s: cumulative=$%.4f > ceiling=$%.4f "
                "(ALPHA_ENGINE_RUN_BUDGET_USD). Raising RunBudgetExceededError "
                "to fail the run loud. JSONL + decision-artifact flushes "
                "completed before raise so per-call detail is preserved on S3.",
                run_id, agent_id, cumulative_after_frame, ceiling,
            )
            raise RunBudgetExceededError(
                run_id=run_id,  # type: ignore[arg-type]
                cumulative_cost_usd=cumulative_after_frame,
                ceiling_usd=ceiling,
            )


def pop_metadata_for(agent_id: str) -> Optional[tuple[ModelMetadata, FullPromptContext]]:
    """Retrieve and clear the populated ``(ModelMetadata, FullPromptContext)``
    for ``agent_id`` if a recent ``track_llm_cost`` exit has stashed one.

    Returns ``None`` if no metadata is staged — the caller should fall
    back to a placeholder so capture can still write something.
    Pops the entry to keep the dict bounded under heavy fan-out.
    """
    completed = dict(_completed_metadata.get())
    pair = completed.pop(agent_id, None)
    _completed_metadata.set(completed)
    return pair
