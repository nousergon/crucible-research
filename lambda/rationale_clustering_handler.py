"""Lambda entry point — cross-week rationale clustering.

Triggered weekly after the Saturday SF eval pipeline completes. Reads
captured decision artifacts from the trailing 8 weeks, clusters
rationales per agent_id, persists per-agent analysis JSON, and emits
the ``agent_rationale_template_concentration`` CloudWatch metric.

Per ROADMAP P0 "Cross-week rationale clustering for agent-justification".

Event shape (all fields optional):

    {
      "end_time_iso": "2026-05-09T00:00:00Z",  # default = now UTC
      "window_days": 56,                        # default 8 weeks
      "dry_run": false                          # if true, skip emit + persist
    }

Returns:

    {
      "status": "OK" | "PARTIAL" | "ERROR",
      "summary": <rationale_clustering.compute_and_emit result>
    }
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime

# Repo root on sys.path so ``from evals.rationale_clustering import ...``
# resolves under Lambda's task layout.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.langsmith_pandas_patch import install as _install_ls_patch

_install_ls_patch()

# Imported after the sys.path.insert above — this Lambda entrypoint isn't
# on sys.path until that line runs (mirrors lambda/handler.py's pattern).
from nousergon_lib.logging import monitor_handler, setup_logging  # noqa: E402

_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "rationale_clustering",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors the eval-judge
    + rolling-mean handlers — Lambda init phase 10s ceiling."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    _init_done = True


@monitor_handler
def handler(event, context):
    """Compute + emit per-agent rationale-template concentration."""
    _ensure_init()

    from evals.lambda_dry import dry_clustering_result, is_dry
    from evals.rationale_clustering import (
        DEFAULT_WINDOW_DAYS,
        compute_and_emit,
    )

    # ── Shell-run dry path ───────────────────────────────────────────
    # Boot + the evals.rationale_clustering import (above) ran for real
    # — that's the keystone's bootstrap smoke. Return BEFORE
    # compute_and_emit, which reads decision_artifacts/, clusters, and
    # (the documented gap) S3-persists _analysis/ JSON via
    # _persist_analysis regardless of the existing `dry_run` flag — that
    # flag only suppresses the CW metric. dry_run_llm short-circuits the
    # entire read+cluster+persist, no Anthropic call.
    if is_dry(event):
        logger.info(
            "[rationale_clustering_handler] dry_run_llm=True: shell-run "
            "no-op (no S3 read/persist, no CW emit)",
        )
        return dry_clustering_result()

    end_time_iso = event.get("end_time_iso")
    end_time = (
        datetime.fromisoformat(end_time_iso.replace("Z", "+00:00"))
        if end_time_iso else None
    )
    window_days = int(event.get("window_days", DEFAULT_WINDOW_DAYS))
    dry_run = bool(event.get("dry_run", False))

    logger.info(
        "[rationale_clustering_handler] start end_time_iso=%s "
        "window_days=%d dry_run=%s",
        end_time_iso or "(now UTC)", window_days, dry_run,
    )

    try:
        summary = compute_and_emit(
            end_time=end_time,
            window_days=window_days,
            emit_metrics=not dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[rationale_clustering_handler] computation failed hard")
        return {"status": "ERROR", "error": str(exc)}

    has_failures = bool(summary["load_failures"]) or bool(summary["cluster_failures"])
    status = "PARTIAL" if has_failures else "OK"

    logger.info(
        "[rationale_clustering_handler] done status=%s agents=%d "
        "skipped_thin=%d load_failures=%d cluster_failures=%d",
        status,
        summary["agents_analyzed"],
        len(summary["agents_skipped_thin_sample"]),
        len(summary["load_failures"]),
        len(summary["cluster_failures"]),
    )
    return {"status": status, "summary": summary}
