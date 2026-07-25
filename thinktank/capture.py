"""DecisionArtifact emission — plugs the think tank into the LLM-as-judge.

Every thesis and theme write also emits a ``DecisionArtifact`` into the
shared capture corpus (``decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json``)
via the lib chokepoint ``nousergon_lib.decision_capture.capture_decision``.
That is ALL the wiring the judge needs: the Saturday Submit/Poll/Process
batch chain enumerates the capture corpus, maps ``agent_id`` → rubric, and
the scores flow to the rolling mean / alarm / dashboard with no per-consumer
registration.

agent_ids are deliberately COARSE (``thinktank_thesis``, ``thinktank_theme``
— NOT per-ticker): the rolling-mean alarm floor requires ≥3 samples per
(agent_id × criterion × judge) combo in 28 days, and per-ticker ids starve
that gate (the thesis_update:{team}:{ticker} lesson). Ticker/theme identity
rides in ``run_id`` and the snapshot instead.

Captures are PARTITIONED BY TRADING DAY (Brian, 2026-07-02), per the fleet
date convention: a Saturday/Sunday run analyzes Friday's data
(``now_dual().trading_day``), so its captures belong in Friday's partition —
the last trading day of the week simply accrues more outputs (Fri+Sat+Sun
runs, expected). The judge's trading-day lookback + already-judged dedup
(evals/orchestrator.py) rely on this partitioning.

Gated on ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` (the fleet-wide capture
switch). Write failures raise ``DecisionCaptureWriteError`` — never swallowed.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC

from nousergon_lib.decision_capture import (
    FullPromptContext,
    ModelMetadata,
    capture_decision,
)

from thinktank.client import LLMCallResult

logger = logging.getLogger(__name__)

THESIS_AGENT_ID = "thinktank_thesis"
THEME_AGENT_ID = "thinktank_theme"


def _enabled() -> bool:
    return os.environ.get("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "").lower() in (
        "1",
        "true",
    )


def _partition_timestamp(trading_day: str):
    """Wall-clock time-of-day on the TRADING day's date — controls the
    capture key's date partition while preserving intra-day ordering."""
    from datetime import date, datetime

    y, m, d = (int(x) for x in trading_day.split("-"))
    now = datetime.now(UTC)
    return datetime.combine(date(y, m, d), now.timetz())


def _emit(
    *,
    agent_id: str,
    run_id: str,
    trading_day: str,
    result: LLMCallResult,
    system: str,
    user: str,
    prompt_version_hash: str | None,
    input_data_snapshot: dict,
    agent_output: dict,
    bucket: str,
    s3_client,
) -> str | None:
    if not _enabled():
        return None
    key = capture_decision(
        run_id=run_id,
        agent_id=agent_id,
        model_metadata=ModelMetadata(
            model_name=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        ),
        full_prompt_context=FullPromptContext(
            system_prompt=system,
            user_prompt=user,
            prompt_version_hash=prompt_version_hash,
        ),
        input_data_snapshot=input_data_snapshot,
        agent_output=agent_output,
        s3_bucket=bucket,
        s3_client=s3_client,
        timestamp=_partition_timestamp(trading_day),
    )
    logger.info("decision capture written: %s", key)
    return key


def emit_thesis_capture(
    *,
    base_run_id: str,
    ticker: str,
    version: int,
    trading_day: str,
    result: LLMCallResult,
    system: str,
    user: str,
    prompt_version_hash: str | None,
    input_data_snapshot: dict,
    agent_output: dict,
    bucket: str,
    s3_client,
) -> str | None:
    return _emit(
        agent_id=THESIS_AGENT_ID,
        run_id=f"{base_run_id}-{ticker}-v{version}",
        trading_day=trading_day,
        result=result,
        system=system,
        user=user,
        prompt_version_hash=prompt_version_hash,
        input_data_snapshot=input_data_snapshot,
        agent_output=agent_output,
        bucket=bucket,
        s3_client=s3_client,
    )


def emit_theme_capture(
    *,
    base_run_id: str,
    kind: str,
    key_slug: str,
    version: int,
    trading_day: str,
    result: LLMCallResult,
    system: str,
    user: str,
    prompt_version_hash: str | None,
    input_data_snapshot: dict,
    agent_output: dict,
    bucket: str,
    s3_client,
) -> str | None:
    return _emit(
        agent_id=THEME_AGENT_ID,
        run_id=f"{base_run_id}-{kind}-{key_slug}-v{version}",
        trading_day=trading_day,
        result=result,
        system=system,
        user=user,
        prompt_version_hash=prompt_version_hash,
        input_data_snapshot=input_data_snapshot,
        agent_output=agent_output,
        bucket=bucket,
        s3_client=s3_client,
    )
