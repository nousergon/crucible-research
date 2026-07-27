"""Lambda entry point — OpenRouter shadow-judge scheduled run
(config#2575 items 4-5 persistence runner, wired to a trigger by
alpha-engine-config#2934).

Shares the main runner's ECR image with a CMD override to
``openrouter_shadow_handler.handler`` (the established image-share
pattern — eval_judge / scanner / rationale_clustering / thinktank).
Thin wrapper around ``evals.openrouter_shadow.run_shadow_judge_over_date``
(crucible-research#470) — no new logic, no new judge-tier surface; this
module's only job is translating an EventBridge event into that
function's keyword args and satisfying the Lambda handler contract.

Invocation source: EventBridge rule
``alpha-research-openrouter-shadow-weekly`` (infrastructure/setup-
openrouter-shadow-schedule.sh) — Sunday 10:00 UTC, after the Saturday
weekly SF's own eval-judge batch chain (Submit/Poll/Process) has had a
full day to land the primary Haiku/Sonnet verdicts for the week's
captures, so the shadow tier scores the same corpus the agreement
metric (``evals.openrouter_shadow.compute_shadow_agreement``) will
later pair against. Plain async invoke — no ``mode``/SF coupling.

**Shadow-only, no decision authority** (config#2575 binding constraint,
carried through unchanged by this wiring) — this handler does not read
the shadow judge's scores back into any routing decision; it only
triggers the runner and returns its summary.

Failure contract — RAISE, never return an ERROR dict. Mirrors
``thinktank_handler.py``'s documented rationale: this Lambda is invoked
async by EventBridge with no SF Catch above it, so an ERROR-dict return
is a *successful* invocation as far as the AWS/Lambda Errors metric is
concerned — no retry fires and the failure is silent. Raising instead
drives the Errors metric that ``setup-openrouter-shadow-schedule.sh``'s
alarm watches and engages EventBridge's two built-in async retries.
``run_shadow_judge_over_date`` itself already treats per-artifact
failures as non-fatal (accumulated in the returned summary's ``failed``
list, run continues) — a raise out of this handler means the RUN ITSELF
blew up (e.g. the capture listing call failed), not merely that some
artifacts failed to score.

Event shape (all fields optional):

    {
      "date": "2026-07-19",   # YYYY-MM-DD; default = yesterday UTC
                               # (the schedule fires Sunday for the
                               # Saturday capture partition)
      "bucket": "alpha-engine-research",
      "dry_run_llm": true      # shell-run smoke: boot + imports only,
                                # no S3/CloudWatch, no LLM calls
    }

Returns the ``run_shadow_judge_over_date`` summary dict on success (or
the dry-path sentinel); raises on any run-level failure.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys

# Repo root on sys.path so ``from evals.openrouter_shadow import ...``
# resolves under Lambda's task layout. Mirrors the existing shared-image
# handlers (scanner, rationale_clustering, eval_rolling_mean, thinktank).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.logging import monitor_handler, setup_logging

_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging("openrouter_shadow", flow_doctor_yaml=_FLOW_DOCTOR_YAML)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """One-time cold-start hydration. ``config.OPENROUTER_API_KEY`` is
    already resolved SSM-first at import time (``config.py`` module
    level, same chokepoint the primary ``eval_judge_handler.py`` relies
    on) — this handler needs no bespoke secrets fetch of its own.
    """
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


def _default_date() -> str:
    """Yesterday UTC — the schedule fires the morning after the
    Saturday weekly SF's capture partition closes, so the default event
    (no explicit ``date``) scores THAT partition rather than an
    (empty-so-far) same-day one."""
    return str(datetime.date.today() - datetime.timedelta(days=1))


@monitor_handler
def handler(event, context):
    """Run the OpenRouter shadow-judge tier over ``event["date"]``'s
    capture partition and persist verdicts. Raises on run-level failure
    (see module doc); per-artifact failures are non-fatal and reported
    in the returned summary's ``failed`` list."""
    from evals.lambda_dry import is_dry

    if is_dry(event):
        logger.info(
            "[openrouter_shadow_handler] dry_run_llm=True: shell-run "
            "no-op (no S3/CloudWatch access, no LLM calls)",
        )
        return {"status": "OK", "dry_run": True}

    _ensure_init()

    from evals.openrouter_shadow import run_shadow_judge_over_date

    date = (event.get("date") if isinstance(event, dict) else None) or _default_date()
    bucket = (
        (event.get("bucket") if isinstance(event, dict) else None)
        or os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
    )

    logger.info(
        "[openrouter_shadow_handler] start date=%s bucket=%s "
        "(SHADOW — no decision authority, config#2575)",
        date, bucket,
    )

    summary = run_shadow_judge_over_date(date=date, bucket=bucket)

    logger.info(
        "[openrouter_shadow_handler] done date=%s evaluated=%d failed=%d "
        "skipped_unmapped=%d skipped_empty_or_degenerate=%d "
        "metric_emission_failures=%d",
        date,
        summary["evaluated"],
        len(summary["failed"]),
        summary["skipped_unmapped"],
        summary["skipped_empty_or_degenerate"],
        summary["metric_emission_failures"],
    )
    return {"status": "OK", "summary": summary}
