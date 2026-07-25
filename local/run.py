"""
Local test runner — bypasses Lambda for development and testing.

Usage:
  python local/run.py                    # run today's pipeline (full: APIs + S3 + email)
  python local/run.py --local            # same as above (explicit)
  python local/run.py --no-s3            # real APIs but write signals to local file, skip email
  python local/run.py --date 2026-03-05  # run for a specific date
  python local/run.py --offline          # full offline: no API/LLM/S3 calls (synthetic data)

Requires environment variables (unless --offline):
  ANTHROPIC_API_KEY
  FMP_API_KEY
  FRED_API_KEY
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (for S3)
  S3_BUCKET (default: alpha-engine-research)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(override=True)

# Disable LangSmith tracing for local runs. The tracer tries to serialize
# the LangGraph state (which contains pandas DataFrames in price_data) and
# fails on Timestamp-keyed dicts, spamming warnings on every node transition.
# Prod runs through the Lambda handler, not this script, so this is
# local-only and does not affect any production tracing setup.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)
os.environ.pop("LANGSMITH_API_KEY", None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Run alpha-engine-research pipeline locally")
    parser.add_argument("--date", type=str, default=None,
                        help="Run date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--local", action="store_true",
                        help="Full pipeline from laptop: real APIs, writes signals to S3, sends email.")
    parser.add_argument("--no-s3", action="store_true",
                        help="Real APIs but write signals.json to local file instead of S3, skip email. "
                             "Safe preprod check — does not affect live executor.")
    parser.add_argument("--offline", action="store_true",
                        help="Full offline mode: stub all API/LLM/S3/email calls with synthetic data.")
    parser.add_argument("--stub-llm", action="store_true",
                        help="Real data + real archive, but stub all Anthropic LLM agent calls. "
                             "Costs $0 in tokens. For debugging data-shape bugs, score_aggregator "
                             "regressions, archive_writer issues — anything below the LLM layer.")
    args = parser.parse_args()

    # Install offline stubs BEFORE any graph/agent imports
    if args.offline:
        from local.offline_stubs import install_offline_stubs
        install_offline_stubs()
    elif args.stub_llm:
        from local.offline_stubs import install_llm_only_stubs
        install_llm_only_stubs()

    run_date = args.date or str(datetime.date.today())

    print(f"alpha-engine-research local run — {run_date}")
    if args.offline:
        print("OFFLINE MODE: all external calls stubbed with synthetic data")
    elif args.stub_llm:
        print("STUB-LLM MODE: real data + real archive, agent LLM calls stubbed ($0 tokens)")
    elif args.no_s3:
        print("NO-S3 MODE: real APIs, signals written to local file, email skipped")

    # Check trading day (use importlib: 'lambda' is a reserved keyword)
    import importlib
    _handler = importlib.import_module("lambda.handler")
    is_trading_day = _handler.is_trading_day
    is_early_close = _handler.is_early_close
    d = datetime.datetime.strptime(run_date, "%Y-%m-%d").date()

    if not is_trading_day(d):
        print(f"NOTE: {run_date} is not an NYSE trading day. Running anyway for testing.")

    early_close = is_early_close(d)

    # Set up archive (use project-root research.db so sync_db push can find it)
    from archive.manager import ArchiveManager
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_db = os.path.join(project_root, "research.db")
    archive = ArchiveManager(local_db_path=local_db)
    print("Downloading research.db from S3...")
    try:
        archive.download_db()
    except Exception as e:
        print(f"S3 download failed ({e}). Starting with empty DB.")
        import sqlite3
        archive.db_conn = sqlite3.connect(archive.local_db_path)
        archive.db_conn.row_factory = sqlite3.Row
        archive._ensure_schema()

    # Build and run graph
    from graph.research_graph import build_graph, create_initial_state

    # Patch graph module local name bindings after import
    if args.offline:
        from local.offline_stubs import patch_graph_modules
        patch_graph_modules()
    elif args.stub_llm:
        from local.offline_stubs import patch_graph_modules_llm_only
        patch_graph_modules_llm_only()

    graph = build_graph()
    state = create_initial_state(
        run_date=run_date,
        archive_manager=archive,
        is_early_close=early_close,
    )

    # --no-s3 or --stub-llm: intercept S3 writes → local files, skip email
    # (--stub-llm implies --no-s3 — never overwrite prod signals.json from a
    # debug run with synthetic LLM output.)
    if (args.no_s3 or args.stub_llm) and not args.offline:
        import json as _json
        _out_dir = os.path.join(project_root, "local", "output")
        os.makedirs(_out_dir, exist_ok=True)

        import graph.research_graph as _gm

        _orig_archive_writer = _gm.archive_writer
        def _local_archive_writer(state):
            """Run archive writer but redirect signals.json to local file."""
            from graph.research_graph import _build_signals_payload
            try:
                payload = _build_signals_payload(state)
                out_path = os.path.join(_out_dir, f"signals-{run_date}.json")
                with open(out_path, "w") as f:
                    _json.dump(payload, f, indent=2)
                print(f"\n=== SIGNALS WRITTEN TO: {out_path} ===")
            except Exception as e:
                print(f"WARNING: could not write local signals: {e}")
            # Still run archive writer for SQLite but skip S3 uploads
            am = state.get("archive_manager")
            if am:
                am.upload_db = lambda *a, **k: None
                am.write_signals_json = lambda *a, **k: None
            return _orig_archive_writer(state)
        _gm.archive_writer = _local_archive_writer

        # Skip email
        _gm.email_sender = lambda state: {"email_sent": False}

    print("Running pipeline...")
    final_state = graph.invoke(state)

    # Trajectory validation (only when LangSmith tracing is active)
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and not args.offline:
        try:
            from evals.trajectory import validate_trajectory
            traj_result = validate_trajectory(
                project_name=os.environ.get("LANGCHAIN_PROJECT", "alpha-research"),
            )
            if traj_result:
                status = "PASS" if traj_result["passed"] else "FAIL"
                print(f"\n=== TRAJECTORY VALIDATION: {status} ===")
                if not traj_result["passed"]:
                    for f in traj_result["failures"]:
                        print(f"  FAIL: {f}")
                print(f"  Nodes: {sum(traj_result['node_counts'].values())} | "
                      f"Duration: {traj_result['duration_ms']}ms")
        except Exception as e:
            print(f"Trajectory validation skipped: {e}")

    print("\n=== RUN COMPLETE ===")
    print(f"Email sent: {final_state.get('email_sent', False)}")
    print(f"Tickers processed: {len(final_state.get('investment_theses', {}))}")

    print("\n=== CONSOLIDATED REPORT ===")
    print(final_state.get("consolidated_report", ""))

    archive.close()


if __name__ == "__main__":
    main()
