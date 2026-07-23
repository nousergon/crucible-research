"""
Per-agent CloudWatch runtime telemetry — invocations, failures, latency.

Phase 2 observability. Emits a CloudWatch metric stream per LangGraph
agent decision under the ``AlphaEngine/Agents`` namespace, dimensioned
by ``agent_id`` (e.g. ``sector_team:technology``, ``macro_economist``,
``ic_cio``). One emission per ``track_llm_cost`` frame exit — whether
the body succeeded or raised.

**Metrics:**

- ``Invocations`` — 1.0 per frame (Sum over a window = runs in window)
- ``Failures`` — 1.0 if the frame body raised, 0.0 otherwise
  (Sum over window = failed runs; pair with Invocations for failure rate)
- ``DurationMs`` — wall-clock time the frame body held (Average / p50 /
  p95 / p99 derived by CW from individual datapoint samples)
- ``LLMCallCount`` — number of Anthropic API calls inside the frame
  (Sum over window = total LLM calls per agent; useful for "is this
  agent doing the same number of LLM calls as last week?")

**Why this lives next to llm_cost_tracker rather than inside it:**

The cost tracker already wraps every agent decision in a
``track_llm_cost`` frame; moving runtime telemetry into the same frame
boundary avoids a second instrumentation layer that drifts. But the
*emission* logic is independent of cost computation — a frame whose
body raised an exception still has a meaningful ``DurationMs`` and
``Failures=1``, even if no LLM call ever fired and ``cost_usd=0``.
Splitting the emission into its own module keeps the cost tracker's
post-finally code unchanged (which depends on the success path),
while letting the runtime emission run unconditionally inside the
finally block.

**Hard-fail surface:**

- The emission swallows CloudWatch errors and logs at warning. Per
  ``feedback_no_silent_fails``: telemetry is observability, not
  correctness — a CW outage must not take down a Sat SF run. This
  mirrors how the substrate health check tolerates SNS publish
  failures (alpha_engine_lib/transparency.py).

**Disable for tests / smoke runs:** set
``ALPHA_ENGINE_AGENT_TELEMETRY_ENABLED=false``. Default is enabled
(matches cost-telemetry runtime behavior — production wants signal,
tests can opt out).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

NAMESPACE = "AlphaEngine/Agents"
_TELEMETRY_ENV_VAR = "ALPHA_ENGINE_AGENT_TELEMETRY_ENABLED"


def _env_label() -> str:
    """``"prod"`` when running in a deployed runtime, else ``"test"``.

    config#1154 fix: the ``AlphaEngine/Agents`` stream was dimensioned only by
    ``agent_id``, so test/CI runs with real creds (telemetry default-ON) wrote to
    the SAME metric a prod consumer reads — the namespace is test-polluted and a
    report-card consumer can't tell prod datapoints from test ones. Stamping an
    ``env`` dimension lets the consumer query ``{env="prod"}`` and read clean prod
    data regardless of test pollution. ``ALPHA_ENGINE_DEPLOYED`` is the fleet's
    deployed-runtime marker (set on the SF/Lambda runtimes); anything else (local
    dev, CI, smoke) is ``"test"``."""
    return "prod" if os.environ.get("ALPHA_ENGINE_DEPLOYED", "").strip().lower() in (
        "1", "true", "yes",
    ) else "test"


def _dims(agent_id: str) -> list[dict]:
    """Per-agent dimension set: ``agent_id`` + ``env`` (config#1154). Adding
    ``env`` changes the metric identity vs the legacy agent_id-only stream —
    intentional: prod consumers query ``{agent_id, env}`` and are immune to the
    historical test pollution on the agent_id-only series."""
    return [{"Name": "agent_id", "Value": agent_id}, {"Name": "env", "Value": _env_label()}]


def _emit_telemetry_dropped(*, agent_id: str, cw: Any) -> None:
    """Emit a loud TelemetryDropped counter when put_metric_data fails.

    This is the fail-loud recording surface for dropped telemetry — a silent
    drop class can never run for months again (per config#2251 motivation).
    Uses a best-effort fallback path: if even this emission fails, we log
    at ERROR but do not recurse (avoiding cascade failure).
    """
    metric_data = [
        {"MetricName": "TelemetryDropped", "Dimensions": _dims(agent_id),
         "Value": 1.0, "Unit": "Count"},
    ]
    if cw is None:
        try:
            import boto3
            cw = boto3.client("cloudwatch", region_name="us-east-1")
        except Exception as exc:
            logger.error(
                "[agent_telemetry] could not create boto3 client for drop metric: %s",
                exc,
            )
            return

    try:
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
    except Exception as exc:
        # Best-effort: if this fallback also fails, log at ERROR and stop
        # (do not recurse). The primary telemetry already failed; a cascade
        # here is not actionable.
        logger.error(
            "[agent_telemetry] could not emit TelemetryDropped for agent_id=%s: %s",
            agent_id, exc,
        )


def _emit_tripwire_dropped(*, cw: Any) -> None:
    """Emit a loud TelemetryDropped counter for the new-entrant tripwire.

    Tripwire is dimensioned differently (no agent_id); uses a separate
    namespace-level TelemetryDropped metric.
    """
    metric_data = [
        {"MetricName": "TelemetryDropped", "Value": 1.0, "Unit": "Count"},
    ]
    if cw is None:
        try:
            import boto3
            cw = boto3.client("cloudwatch", region_name="us-east-1")
        except Exception as exc:
            logger.error(
                "[agent_telemetry] could not create boto3 client for tripwire drop metric: %s",
                exc,
            )
            return

    try:
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
    except Exception as exc:
        logger.error(
            "[agent_telemetry] could not emit tripwire TelemetryDropped: %s", exc,
        )


def _is_telemetry_enabled() -> bool:
    """Default ON — production emits, tests can disable via env.

    Matches the cost-tracker convention where the production-default
    is to emit. Tests that need a clean CW surface set
    ``ALPHA_ENGINE_AGENT_TELEMETRY_ENABLED=false``.
    """
    raw = os.environ.get(_TELEMETRY_ENV_VAR, "true").lower()
    return raw in ("true", "1", "yes")


def emit_agent_completion(
    *,
    agent_id: str,
    enter_time: datetime,
    exception_raised: bool,
    llm_call_count: int,
    cloudwatch_client: Any = None,
) -> None:
    """Emit one frame's worth of per-agent telemetry to CloudWatch.

    Called from ``track_llm_cost``'s finally block on every frame exit,
    regardless of success or failure. ``exception_raised`` is True when
    the frame body raised — measured by the cost tracker via a
    try/except around the yield.

    Parameters
    ----------
    agent_id
        ``DecisionArtifact.agent_id`` form: ``sector_team:technology``,
        ``macro_economist``, ``ic_cio``, ``thesis_update:{sector}:{ticker}``.
    enter_time
        Frame enter timestamp (UTC). Duration computed as ``now - enter_time``.
    exception_raised
        True if the frame body raised an exception, False if it
        completed normally.
    llm_call_count
        Number of Anthropic API calls observed by the cost tracker
        callback inside this frame.
    cloudwatch_client
        Override for tests. Production passes None and a fresh boto3
        client is created (cheap; alpha-engine-research Lambda lifecycle
        is per-SF-run so we don't reuse).
    """
    if not _is_telemetry_enabled():
        return

    duration_ms = max(
        0.0, (datetime.now(UTC) - enter_time).total_seconds() * 1000.0,
    )
    failures = 1.0 if exception_raised else 0.0
    dims = _dims(agent_id)
    metric_data = [
        {"MetricName": "Invocations", "Dimensions": dims, "Value": 1.0, "Unit": "Count"},
        {"MetricName": "Failures", "Dimensions": dims, "Value": failures, "Unit": "Count"},
        {"MetricName": "DurationMs", "Dimensions": dims, "Value": duration_ms, "Unit": "Milliseconds"},
        {"MetricName": "LLMCallCount", "Dimensions": dims, "Value": float(llm_call_count), "Unit": "Count"},
    ]

    cw = cloudwatch_client
    if cw is None:
        try:
            import boto3

            cw = boto3.client("cloudwatch", region_name="us-east-1")
        except Exception as exc:
            logger.warning(
                "[agent_telemetry] could not create boto3 cloudwatch client: %s",
                exc,
            )
            return

    try:
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
    except Exception as exc:
        logger.error(
            "[agent_telemetry] put_metric_data failed for agent_id=%s: %s",
            agent_id, exc,
        )
        # Emit a loud TelemetryDropped counter via best-effort fallback path
        _emit_telemetry_dropped(agent_id=agent_id, cw=cw)


def emit_agent_retry(
    *,
    agent_id: str,
    attempted: bool,
    succeeded: bool,
    cloudwatch_client: Any = None,
) -> None:
    """Emit one retry-event datapoint per ``run_*_with_retry`` invocation.

    Called from the retry wrappers in ``agents/sector_teams/sector_team.py``
    (added by research#106 for empty-output detection). Even when
    ``attempted=False`` (no retry needed), we emit so the metric stream
    is dense enough to compute "retry rate per agent" without missing
    datapoints.

    Parameters
    ----------
    attempted
        True iff the retry actually fired (initial run produced empty
        output). False if the initial run produced non-empty output and
        no retry was needed.
    succeeded
        True iff a fired retry produced non-empty output. Meaningful only
        when ``attempted=True``; pass False otherwise.
    """
    if not _is_telemetry_enabled():
        return

    dims = _dims(agent_id)
    metric_data = [
        {"MetricName": "RetryAttempts", "Dimensions": dims,
         "Value": 1.0 if attempted else 0.0, "Unit": "Count"},
        {"MetricName": "RetrySuccesses", "Dimensions": dims,
         "Value": 1.0 if (attempted and succeeded) else 0.0, "Unit": "Count"},
    ]

    cw = cloudwatch_client
    if cw is None:
        try:
            import boto3

            cw = boto3.client("cloudwatch", region_name="us-east-1")
        except Exception as exc:
            logger.warning(
                "[agent_telemetry] could not create boto3 cloudwatch client: %s",
                exc,
            )
            return

    try:
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
    except Exception as exc:
        logger.error(
            "[agent_telemetry] retry emission failed for agent_id=%s: %s",
            agent_id, exc,
        )
        # Emit a loud TelemetryDropped counter via best-effort fallback path
        _emit_telemetry_dropped(agent_id=agent_id, cw=cw)


def emit_new_entrant_tripwire(
    net_new_entrants: int,
    alert_floor: int,
    fresh_slate_max_conviction: float | None,
    cloudwatch_client: Any = None,
) -> None:
    """Emit the weekly net-new-entrant count + a breach flag.

    A 0-add (or below-floor) week is a *defensible* outcome (the CIO correctly
    rejecting a weak/saturated fresh slate) — but it must be VISIBLE, not
    silently inferred. Always emits ``NewEntrants`` (the count) so the trend is
    charted; emits ``NewEntrantsBelowFloor=1`` when ``net_new_entrants <
    alert_floor`` so an alarm can page on a saturation streak. The caller logs
    the loud WARN with the human-readable why (fresh-slate max conviction vs
    the entrant bar). Best-effort per ``feedback_no_silent_fails`` — telemetry
    is observability, never blocks the run.
    """
    if not _is_telemetry_enabled():
        return

    breach = 1.0 if net_new_entrants < alert_floor else 0.0
    metric_data = [
        {"MetricName": "NewEntrants", "Value": float(net_new_entrants), "Unit": "Count"},
        {"MetricName": "NewEntrantsBelowFloor", "Value": breach, "Unit": "Count"},
    ]
    if fresh_slate_max_conviction is not None:
        metric_data.append({
            "MetricName": "FreshSlateMaxConviction",
            "Value": float(fresh_slate_max_conviction),
            "Unit": "None",
        })

    cw = cloudwatch_client
    if cw is None:
        try:
            import boto3

            cw = boto3.client("cloudwatch", region_name="us-east-1")
        except Exception as exc:
            logger.warning(
                "[agent_telemetry] could not create boto3 cloudwatch client "
                "for new-entrant tripwire: %s", exc,
            )
            return

    try:
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
    except Exception as exc:
        logger.error(
            "[agent_telemetry] new-entrant tripwire emission failed: %s", exc,
        )
        # Emit a loud TelemetryDropped counter with special tripwire marker
        _emit_tripwire_dropped(cw=cw)
