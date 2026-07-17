#!/usr/bin/env python3
"""Box entrypoint for the weekly Research run on spot EC2 (config#1687).

Runs the **same production orchestration** the Lambda handler drives today —
by invoking ``lambda/handler.py::handler`` directly with the weekly event —
so the spot-EC2 submit+poll path preserves the PRIOR-population snapshot and
the FAIL-HARD challenger post-step (config#1683) *by construction*, rather
than forking a second orchestration that could drift.

Why this exists
---------------
The weekly SF ``Research`` state is a synchronous ``lambda:invoke`` of
``alpha-engine-research-runner:live`` capped at ``TimeoutSeconds: 900`` — the
AWS Lambda hard maximum. The 2026-07-03 weekly completed primary work at
~874s (97% of the ceiling) and a 26s tail overrun tripped ``States.Timeout``
(config#1650). There is no headroom lever left on Lambda, so the heavy weekly
pass moves to spot EC2 (submit+poll, mirroring PredictorTraining / DataPhase1)
where there is no wall-clock ceiling. The runner Lambda STAYS for intraday
alerts + operator modes (``challengers_only``, ``dry_run_llm``, manual
invokes) — only the weekly heavy pass moves here.

Faithfulness
------------
The Lambda ``handler(event, context)`` never reads ``context`` (verified
against ``lambda/handler.py`` on 2026-07-06 — the only occurrence is the
signature). Calling ``handler(event, None)`` is therefore behavior-identical
to the EventBridge invoke: same idempotency gate, same ``ResearchPreflight``,
same ``most_recent_trading_day`` stamping, same ``build_graph`` → ``invoke``
→ ``archive_writer`` → FAIL-HARD ``run_challengers`` → trajectory → health →
manifest → cost-aggregation path. The weekly event mirrors the SF Payload
``{weekly_run, force, skip_dry_run_gate, dry_run_llm}``.

Exit contract (fail-loud, for the SSM poll + SF Catch)
------------------------------------------------------
* handler returns ``status == "OK"``  -> exit 0
* handler returns ``status == "SKIPPED"`` (e.g. ``already_run``) -> exit 0
* handler returns ``status == "ERROR"`` OR raises -> exit 1 (non-zero so the
  ``krepis.ssm_dispatcher`` poller + the SF ``ExtractResearchError`` state
  surface the failure — never a silent green).

``--preflight-only`` is the Friday shell-run dry path (config#1629): it does
the import / lib-pin + read-only ``ResearchPreflight`` and exits 0 WITHOUT
invoking the graph — no LLM calls, no S3 writes, no email.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("weekly_box_runner")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "handler.py"


def _import_handler():
    """Load ``lambda/handler.py`` via importlib.

    ``lambda/`` collides with the Python keyword, so it cannot be a normal
    package import — this mirrors the loader used by the handler test suite
    (``tests/test_challengers_only_mode.py``).
    """
    # The handler's deferred imports (``from preflight import ...``,
    # ``from archive.manager import ...``, ``from graph.research_graph import
    # ...``) resolve against the repo root, so it must be importable.
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("research_handler_box", _HANDLER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load handler spec from {_HANDLER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_event(
    *,
    force: bool = False,
    skip_dry_run_gate: bool = True,
    dry_run_llm: bool = False,
) -> dict:
    """Build the weekly event, mirroring the SF ``Research`` Payload.

    ``skip_dry_run_gate`` defaults ``True``: the weekly production run skips
    the auto-gate stub-pass (which costs ~8 min) exactly as the Lambda path's
    ``skip_dry_run_gate=true`` production optimization does (config#1687
    gotcha — this MUST carry over or the box run pays the stub cost twice).
    """
    return {
        "weekly_run": True,
        "force": force,
        "skip_dry_run_gate": skip_dry_run_gate,
        "dry_run_llm": dry_run_llm,
    }


def _run_preflight_only() -> int:
    """Friday shell-run dry path: prove the stack imports + env/S3 reachable,
    WITHOUT invoking the graph. No LLM calls, no writes. Exit 0 on PASS."""
    logging.getLogger().info("[1/2] Importing handler stack + asserting lib pin...")
    import nousergon_lib  # noqa: F401 — lib-pin presence (version asserted by requirements pin)

    _import_handler()  # exec_module validates the full deferred-import surface
    logging.getLogger().info("      OK — nousergon_lib + lambda/handler import clean")

    logging.getLogger().info("[2/2] ResearchPreflight (env vars + S3 reachability)...")
    from preflight import ResearchPreflight

    ResearchPreflight(
        bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
        mode="weekly",
    ).run()
    logging.getLogger().info("      OK — env present, S3 bucket reachable")

    print()
    print("=" * 60)
    print("  PREFLIGHT-ONLY RESULT: PASS")
    print("=" * 60)
    print("  Imports:          nousergon_lib + lambda/handler clean")
    print("  ResearchPreflight: PASS (env + S3 reachable)")
    print("  Graph invoke:     SKIPPED (no build_graph/invoke)")
    print("  LLM / S3 / email: NONE")
    print("=" * 60)
    return 0


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weekly_box_runner",
        description="Run the weekly Research pipeline on spot EC2 (config#1687).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Friday shell-run dry path: import + lib-pin + ResearchPreflight, exit 0. "
        "No graph invoke, no LLM, no writes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the time / trading-day / idempotency gates (manual re-run).",
    )
    parser.add_argument(
        "--no-skip-dry-run-gate",
        dest="skip_dry_run_gate",
        action="store_false",
        help="Run the auto-gate stub-pass before the real pass (default: skipped, "
        "matching the Lambda weekly production optimization).",
    )
    parser.add_argument(
        "--dry-run-llm",
        action="store_true",
        help="Stub-only mode: no real LLM calls, no S3 writes, no email.",
    )
    parser.set_defaults(skip_dry_run_gate=True)
    args = parser.parse_args(argv)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
        )

    if args.preflight_only:
        return _run_preflight_only()

    handler_mod = _import_handler()
    event = build_event(
        force=args.force,
        skip_dry_run_gate=args.skip_dry_run_gate,
        dry_run_llm=args.dry_run_llm,
    )

    t0 = time.time()
    logging.getLogger().info("Invoking research handler with weekly event: %s", event)
    result = handler_mod.handler(event, None)
    wall_clock_s = round(time.time() - t0, 1)

    status = (result or {}).get("status", "ERROR")
    # Runtime telemetry (config#1687 deliverable 5): the weekly wall-clock is
    # the motivating metric (874s baseline on the 900s Lambda ceiling). Emit it
    # in a machine-readable summary line the SF ResultSelector / run summary can
    # pick up so the post-migration trend stays visible.
    summary = {
        "weekly_research_box_run": True,
        "status": status,
        "wall_clock_s": wall_clock_s,
        "lambda_ceiling_s": 900,
        "date": (result or {}).get("date"),
        "reason": (result or {}).get("reason"),
    }
    print("WEEKLY_RESEARCH_RUN_SUMMARY " + json.dumps(summary, default=str))
    logging.getLogger().info(
        "Weekly research run finished: status=%s wall_clock=%ss (Lambda ceiling was 900s)",
        status,
        wall_clock_s,
    )

    if status == "ERROR":
        logging.getLogger().error("Research handler returned ERROR: %s", result)
        return 1
    return 0


def main() -> None:  # pragma: no cover - thin CLI shim
    raise SystemExit(run())


if __name__ == "__main__":
    main()
