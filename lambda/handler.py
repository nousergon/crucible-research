"""
Lambda entry point — main research pipeline.

Weekly (primary): triggered by the Saturday Step Function via EventBridge
Saturday 06:00 UTC (Friday ~10-11pm PT). EventBridge passes {"weekly_run": true}
— bypasses the 5:45am PT time gate.

Weekday (disabled, available for rollback): EventBridge at 12:45+13:45 UTC
(5:45am PT after DST time gate). Checks for market holidays.

Pass {"force": true} to bypass all gates (manual testing).
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time

# Ensure the project root is on sys.path so sibling modules
# (graph.langsmith_pandas_patch) can be imported below.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Install the LangSmith pandas DataFrame serializer patch BEFORE any
# langchain / langgraph import that could trigger a tracer callback.
#
# Background: the research graph state holds `price_data: dict[str,
# pd.DataFrame]`. LangSmith's `_serialize_json` iterates a hardcoded
# list of methods (including `to_dict`) on unknown objects and calls
# `df.to_dict()` — which returns `{col: {pd.Timestamp: value}}`.
# orjson's C-level dict-key handler does a strict type check
# (`PyDateTime_DateTimeType`) and doesn't recognize pd.Timestamp even
# though it subclasses datetime.datetime in Python, so it raises
# TypeError. LangSmith then falls back to stdlib `json.dumps` which
# rejects all non-primitive dict keys, and every agent callback
# crashes with the flood we saw on 2026-04-11.
#
# Fix: graph/langsmith_pandas_patch.py monkey-patches
# langsmith._internal._serde._serialize_json to intercept DataFrames
# and Series before the `to_dict` path fires, returning a safe
# summary string. Idempotent — safe to call once here and again if
# anything else re-imports it. Supersedes the temporary
# `LANGCHAIN_TRACING_V2=false` disable from earlier in this session.
from graph.langsmith_pandas_patch import install as _install_ls_patch
_install_ls_patch()

# Structured logging + flow-doctor singleton from alpha-engine-lib. When
# FLOW_DOCTOR_ENABLED=1, attaches a FlowDoctorHandler at ERROR so every
# log.error() call routes through flow-doctor's dispatch (email +
# optional GitHub issue) without explicit fd.report() plumbing.
# flow-doctor.yaml ships in the Lambda task root (Dockerfile COPY).
# exclude_patterns starts empty by deliberate convention: add patterns
# only after observing real ERROR-level noise from the Saturday SF — the
# canonical lib pattern (mirrors executor/main.py:65-67) forces every
# entrypoint to think about it explicitly rather than inherit defaults.
from alpha_engine_lib.logging import monitor_handler, setup_logging
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flow-doctor.yaml")
setup_logging(
    "research",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)

# Expensive init is deferred to the first handler invocation to keep
# Lambda's cold-start init phase under the 10-second hard timeout.
# `pytz`, `exchange_calendars` (~3-5s — materializes the full NYSE
# schedule on import), and the SSM secrets fetch all used to run at
# module-top, and on 2026-04-11 a cold-start container timed out with
# `INIT_REPORT Init Duration: 9999.47 ms — Status: timeout`. Moving
# them to the handler body pays the same cost on the first invocation
# but in the configurable 15-minute handler budget instead of the
# rigid 10s init wall. Idempotent via the `_init_done` flag.
_init_done = False


def _ensure_init() -> None:
    """Run expensive init once, on the first handler invocation."""
    global _init_done
    if _init_done:
        return
    import exchange_calendars  # noqa: F401 — heavy; cached in sys.modules
    import pytz  # noqa: F401
    _init_done = True


def _scorecard_enabled() -> bool:
    """Read the `RESEARCH_SCORECARD_ENABLED` flag.

    Default-off until Phase 2 (CIO/Macro prompt wiring) lands and the
    operator flips this on to start the ≥4-cycle shadow soak.
    """
    return os.environ.get("RESEARCH_SCORECARD_ENABLED", "").lower() in ("1", "true", "yes")


def _team_accuracy_enabled() -> bool:
    """Read the `TEAM_ACCURACY_PRODUCER_ENABLED` flag.

    Default-ON (unlike the scorecard flag): config#1422 exists specifically
    to start live history accruing NOW, ahead of `ADAPTIVE_SLOT_ALLOCATION_ENABLED`
    being flipped — the consumer (`compute_team_slots` in
    `agents/sector_teams/team_config.py`) already gates its own read behind
    that separate flag and degrades gracefully when the artifact is absent,
    so there's no risk in writing it early. Kept as an env-overridable flag
    (not hardcoded True) purely for operator kill-switch safety, mirroring
    every other producer's pattern in this handler.
    """
    return os.environ.get("TEAM_ACCURACY_PRODUCER_ENABLED", "true").lower() in ("1", "true", "yes")


def _maybe_emit_team_accuracy(archive, trading_date: datetime.date) -> None:
    """Build + write the per-team historical-accuracy artifact. Shadow-safe.

    Producer half of config#926's adaptive slot allocation (config#1422):
    the consumer (`archive/manager.py::load_team_accuracy` +
    `ADAPTIVE_SLOT_ALLOCATION_ENABLED` + `compute_team_slots`'s
    `team_accuracy` nudge) shipped in config#926 but nothing wrote
    `config/team_accuracy.json`, so flipping that flag was a no-op.

    WARN-and-continue on any failure — same shadow-mode posture as
    `_maybe_emit_scorecard` — so the morning briefing is never blocked by
    this secondary-observability producer, and the adaptive-slot consumer
    already treats a missing/stale artifact as "degrade to static
    allocation" by design.
    """
    if not _team_accuracy_enabled():
        logger.debug("team_accuracy producer disabled (flag off) — skipping emission")
        return
    try:
        import boto3  # local import — only paid when flag is on
        from evals.team_accuracy import analyze_team_performance, save_team_accuracy
        team_accuracy = analyze_team_performance(archive.db_conn, as_of_date=trading_date)
        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        save_team_accuracy(
            team_accuracy,
            s3_client=boto3.client("s3"),
            bucket=bucket,
        )
        logger.info(
            "team_accuracy emitted: %d teams with resolved observations (%s)",
            len(team_accuracy),
            {tid: v["n_obs"] for tid, v in team_accuracy.items()},
        )
    except Exception as tae:
        # Shadow-mode WARN-not-fatal — see docstring.
        logger.warning(
            "team_accuracy emission failed (shadow mode — non-fatal): %s",
            tae,
            exc_info=True,
        )


def _maybe_emit_scorecard(archive, trading_date: datetime.date) -> None:
    """Build + write the prior-cycle scorecard. Shadow-safe.

    Pulled out of the handler body so the wiring + failure posture can
    be unit-tested with a stub archive + stub S3 client. The handler
    body sees only the helper call site.

    Skips when the flag is off; logs WARN-and-continues on any failure
    so the morning briefing isn't blocked by a producer artifact that
    no consumer is reading yet (the shadow-mode carve-out
    `[[feedback_no_silent_fails]]` permits).
    """
    if not _scorecard_enabled():
        logger.debug("scorecard disabled (flag off) — skipping emission")
        return
    try:
        import boto3  # local import — only paid when flag is on
        from evals.last_week_scorecard import build_scorecard, emit_scorecard_to_s3
        sc = build_scorecard(archive.db_conn, as_of_date=trading_date)
        bucket = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
        result = emit_scorecard_to_s3(
            sc,
            s3_client=boto3.client("s3"),
            bucket=bucket,
        )
        logger.info(
            "scorecard emitted run_id=%s dated_key=%s n_resolved_predictions=%d "
            "n_resolved_signals_21d=%d",
            result["run_id"],
            result["dated_key"],
            sc.n_resolved_predictions,
            sc.n_resolved_signals_21d,
        )
    except Exception as sce:
        # Shadow-mode WARN-not-fatal. Promote to ERROR + raise in Phase 2
        # when CIO/Macro prompts depend on the scorecard artifact.
        logger.warning(
            "Scorecard emission failed (shadow mode — non-fatal): %s",
            sce,
            exc_info=True,
        )


def is_trading_day(date: datetime.date | None = None) -> bool:
    """Return True if date (default: today) is an NYSE trading day.

    Delegates to the alpha_engine_lib.trading_calendar chokepoint
    (L4466/config#886) — the scanner Lambda already resolves through the
    lib, and two calendar sources in one repo is the drift class that
    produced the 2026-05-30 calendar-vs-trading-day recovery failure.
    """
    from alpha_engine_lib import trading_calendar as _tc
    d = date or datetime.date.today()
    return _tc.is_trading_day(d)


def most_recent_trading_day(date: datetime.date | None = None) -> datetime.date:
    """Return the most recent NYSE trading day on or before the given date.

    If the given date is itself a trading day, returns it unchanged.
    Otherwise rewinds one day at a time until a trading day is found.

    Used to stamp signals, scanner_evaluations, team_candidates, and
    cio_evaluations with the data-close date research actually saw —
    the prior trading day's close is the anchor for 5d-forward-return
    evaluation and aligns with the standard quant convention (measure
    signals against the close that fed them).

    Replaces the earlier `next_trading_day` stamping (2026-04-13,
    commit 9a94e34), which stamped with the Monday the signals would
    be traded on. That fixed the executor's staleness check at the
    cost of shifting the evaluator's 5d forward window by one trading
    day, producing Mon->Mon returns instead of the cleaner Fri->Fri
    window. Executor's staleness check uses a 7 calendar-day threshold,
    so Friday-stamped signals read Monday morning (age=3d) stay well
    inside tolerance.

    Delegates to the alpha_engine_lib.trading_calendar chokepoint
    (L4466/config#886) — formerly a repo-local exchange_calendars
    resolver, a second calendar source of truth that could silently
    drift from the lib the scanner resolves through.
    """
    from alpha_engine_lib import trading_calendar as _tc
    d = date or datetime.date.today()
    return d if _tc.is_trading_day(d) else _tc.previous_trading_day(d)


def is_early_close(date: datetime.date | None = None) -> bool:
    """
    Return True if the NYSE has an early close today (partial session).
    Early closes: day before July 4th, Black Friday, Christmas Eve.
    These still run — the morning report executes normally.
    """
    from exchange_calendars import get_calendar
    nyse = get_calendar("XNYS")
    d = date or datetime.date.today()
    try:
        # exchange_calendars exposes early close dates
        session = nyse.schedule.loc[str(d)] if str(d) in nyse.schedule.index else None
        if session is not None:
            close_time = session["market_close"]
            # NYSE standard close is 4pm ET = 21:00 UTC
            standard_close_utc_hour = 21
            if close_time.hour < standard_close_utc_hour:
                return True
    except (KeyError, AttributeError, TypeError):
        pass  # expected: schedule format edge cases
    except Exception as e:
        logger.warning("Early close detection failed: %s — assuming normal close", e)
    return False


def _is_scheduled_run_time() -> bool:
    """
    Return True if current PT time is within the 5:40–5:55am run window.
    Used by the weekday EventBridge rule (12:45+13:45 UTC).
    Only the invocation that lands in 5:45am PT proceeds.
    """
    import pytz
    pt = datetime.datetime.now(pytz.timezone("America/Los_Angeles"))
    return pt.hour == 5 and 40 <= pt.minute <= 55


def _run_challengers_only(event: dict) -> dict:
    """Operator recovery mode (config#1683): re-emit the challenger producers'
    shadow cohort for the MOST RECENT weekly run without re-running the
    champion graph.

    Event: ``{"mode": "challengers_only", "date": "YYYY-MM-DD"}`` where
    ``date`` is the run_date of the latest completed weekly run.

    The Saturday path snapshots the PRIOR population before the champion
    mutates it; after the fact that snapshot is gone, so this mode
    reconstructs it MEMBERSHIP-EXACTLY from the live population table by
    dropping the rows the run itself entered (``entry_date == run_date``).
    Membership is all the producers consume for selection (the exclusion set
    + carry-forward), so the selection comparison is unaffected; carried
    metadata (scores/tenure) reflects the post-run refresh. This is only
    valid for the LATEST run — the fail-loud guard against
    ``signals/latest.json`` refuses anything else.
    """
    import json as _json

    run_date = event.get("date")
    if not run_date:
        raise ValueError("challengers_only requires event['date'] (YYYY-MM-DD)")

    from archive.manager import ArchiveManager
    archive = ArchiveManager()
    archive.download_db()
    try:
        latest = _json.loads(
            archive.s3.get_object(
                Bucket=archive.bucket, Key="signals/latest.json"
            )["Body"].read()
        )
        latest_date = latest.get("date")
        if latest_date != run_date:
            raise ValueError(
                f"challengers_only is only valid for the latest run "
                f"(latest={latest_date!r}, requested={run_date!r}) — the prior-"
                f"population reconstruction is membership-exact only against "
                f"the most recent population commit (config#1683)."
            )

        population = archive.load_population()
        prior_population = [
            p for p in population if p.get("entry_date") != run_date
        ]
        logger.info(
            "[challengers_only] run_date=%s prior_population=%d "
            "(current %d minus %d entered on run_date)",
            run_date, len(prior_population), len(population),
            len(population) - len(prior_population),
        )

        from producers.runner import run_challengers
        shadow = run_challengers(
            archive, run_date,
            run_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            population=prior_population,
        )
        return {
            "status": "OK",
            "mode": "challengers_only",
            "date": run_date,
            "written": shadow["written"],
        }
    finally:
        archive.close()


@monitor_handler
def handler(event, context):
    """
    AWS Lambda handler for the research pipeline.

    Gate logic:
      - force=True  → bypass all gates (manual testing)
      - weekly_run=True → bypass time gate (Saturday 06:00 UTC weekly schedule)
      - Otherwise → require 5:40-5:55am PT time window AND NYSE trading day

    Operator modes:
      - mode="challengers_only" + date=YYYY-MM-DD → re-emit the challenger
        shadow cohort for the latest weekly run (config#1683 recovery path);
        bypasses the time/trading-day gates, runs nothing else.

    Returns:
        dict with status: "OK" | "SKIPPED" | "ERROR"
    """
    # Run one-time expensive imports + SSM secrets fetch on the first
    # invocation. Warm-container calls are a no-op via the _init_done flag.
    _ensure_init()
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    if event.get("mode") == "challengers_only":
        return _run_challengers_only(event)

    force = event.get("force", False)
    weekly = event.get("weekly_run", False)
    # Dry-run controls (added 2026-04-30):
    #   skip_dry_run_gate — bypass the auto-gate stub-pass, run real only.
    #   dry_run_llm — exclusive stub-only mode (no real LLM calls). Implies
    #                 no S3 writes, no email, no DB upload.
    # Default behavior (both false): run a stub-pass first, halt on failure,
    # only proceed to real pass if stub-pass succeeds.
    skip_dry_run_gate = event.get("skip_dry_run_gate", False)
    dry_run_llm = event.get("dry_run_llm", False)
    fd = None

    # Time gate: weekly runs and force bypass; weekday runs require 5:40-5:55am PT
    if not force and not weekly and not _is_scheduled_run_time():
        return {"status": "SKIPPED", "reason": "wrong_time"}

    today = datetime.date.today()

    # Trading day gate: force bypasses; weekly runs Saturday (never a trading day,
    # so weekly always proceeds — signals are stamped with the most recent trading
    # day below). Weekday runs require an actual NYSE session.
    if not force and not is_trading_day(today):
        if weekly:
            logger.info("Non-trading day %s — running anyway (weekly population refresh).", today)
        else:
            logger.info("Market holiday on %s — skipping run.", today)
            return {"status": "SKIPPED", "reason": "market_holiday", "date": str(today)}

    # Preflight runs AFTER the skip gates — no point paying head_bucket +
    # ANTHROPIC_API_KEY validation on invocations we're about to skip.
    # ANTHROPIC_API_KEY is resolved on-demand via
    # alpha_engine_lib.secrets.get_secret() at consumer sites.
    from preflight import ResearchPreflight
    ResearchPreflight(
        bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
        mode="weekly",
    ).run()

    early_close = is_early_close(today) if not weekly else False
    # Stamp signals with the *most recent* trading day whose close fed
    # this run — never today (weekend) and never a future trading day.
    # System-wide rule: every eval_date (signal folder, latest.json,
    # scanner_evaluations, team_candidates, cio_evaluations, universe_returns)
    # anchors to "most recent trading day with data available at run time."
    #
    # Supersedes commit 9a94e34 (2026-04-13) which stamped with
    # next_trading_day. That fix addressed executor staleness but shifted
    # the evaluator's 5d forward window by one trading day (Mon->Mon vs
    # Fri->Fri), mis-aligning the measurement anchor with the close data
    # research actually saw. The executor's staleness check uses a 7
    # calendar-day threshold (signal_reader._warn_if_stale), so
    # Friday-stamped signals read Monday morning show age=3 days, well
    # inside tolerance.
    trading_date = most_recent_trading_day(today)
    run_date = str(trading_date)

    # Idempotency gate: skip if signals already written for this date
    if not force:
        try:
            import boto3
            from botocore.exceptions import ClientError
            s3 = boto3.client("s3")
            s3.head_object(Bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
                           Key=f"signals/{run_date}/signals.json")
            logger.info("Signals already exist for %s — skipping (use force=True to override)", run_date)
            return {"status": "SKIPPED", "reason": "already_run", "date": run_date}
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                logger.warning("S3 idempotency check failed: %s — proceeding with run", e)
        except Exception as e:
            logger.warning("S3 idempotency check failed: %s — proceeding with run", e)

    run_type = "weekly population refresh" if weekly else "weekday"
    logger.info(
        "Starting alpha-engine-research run for %s (%s)%s",
        run_date, run_type, " [early close]" if early_close else "",
    )

    _health_start = time.time()

    # Import pipeline (deferred to reduce cold-start time)
    try:
        from archive.manager import ArchiveManager
        from graph.research_graph import build_graph, create_initial_state

        # ── Validate required env vars (fail fast, not 30 min in) ─────
        from config import ANTHROPIC_API_KEY, FMP_API_KEY, FRED_API_KEY
        _missing = []
        if not ANTHROPIC_API_KEY:
            _missing.append("ANTHROPIC_API_KEY")
        if not FMP_API_KEY:
            _missing.append("FMP_API_KEY")
        if not FRED_API_KEY:
            _missing.append("FRED_API_KEY")
        if _missing:
            msg = f"Missing required env vars: {', '.join(_missing)}"
            # ERROR — pipeline can't proceed; flow-doctor should escalate
            # so the operator notices the missing-secret class fast
            # (vs surfacing only via Step Function failure email).
            logger.error("FATAL: %s", msg)
            return {"statusCode": 500, "body": msg}

        archive = ArchiveManager()
        archive.download_db()

        # Phase 1.B.2 — research-feedback scorecard.
        # Emits the prior-cycle realized-outcomes scorecard artifact under
        # `research/last_week_scorecard/{run_id}.json` for Phase 2 prompt
        # injection. Gated default-OFF behind `RESEARCH_SCORECARD_ENABLED`
        # until Phase 2 (CIO/Macro prompt wiring) is ready to consume it.
        # Shadow-mode failure posture: WARN log + continue. Per
        # [[feedback_no_silent_fails]] secondary-observability carve-out —
        # (a) swallowing scorecard build/write failures, (b) primary
        # deliverable (Saturday morning briefing) is independent of the
        # scorecard during shadow, (c) WARN log is the recording surface
        # (promotable to ERROR + CloudWatch alarm in Phase 2 when the
        # consumer depends on the artifact).
        _maybe_emit_scorecard(archive, trading_date)

        # config#1422 — per-team historical-accuracy producer for adaptive
        # slot allocation (config#926). Writes `config/team_accuracy.json`
        # from CIO-ADVANCED picks' realized 21d beat-SPY outcomes, joined by
        # team_id. Default-ON (unlike the scorecard) so live history starts
        # accruing now, ahead of `ADAPTIVE_SLOT_ALLOCATION_ENABLED` being
        # flipped for the soak. Same shadow-mode WARN-and-continue posture.
        _maybe_emit_team_accuracy(archive, trading_date)

        # Build and run the LangGraph pipeline
        graph = build_graph()
        initial_state = create_initial_state(
            run_date=run_date,
            archive_manager=archive,
            is_early_close=early_close,
        )

        # Extract episodic memories from newly completed signal outcomes
        try:
            from memory.episodic import extract_memories
            n_memories = extract_memories(archive.db_conn)
            if n_memories:
                logger.info("Extracted %d new episodic memories from outcomes", n_memories)
        except Exception as _me:
            logger.warning("memory extraction skipped: %s", _me)

        # ── Auto-gate: stub-LLM dry-run before real pass ─────────────
        # Catches bugs below the LLM layer (graph orchestration, schema
        # parse, reducer behavior, archive writes) without paying for
        # Anthropic tokens. Real pass only fires if stub-pass succeeds.
        # Skipped when skip_dry_run_gate=True or when dry_run_llm=True
        # (which is itself the stub-only mode, no real pass to gate).
        if not skip_dry_run_gate and not dry_run_llm:
            from dry_run import install_dry_run_stubs
            logger.info("Stub-LLM dry-run gate: starting...")
            _restore = install_dry_run_stubs(archive)
            try:
                _stub_graph = build_graph()
                _stub_state = create_initial_state(
                    run_date=run_date,
                    archive_manager=archive,
                    is_early_close=early_close,
                )
                _stub_graph.invoke(_stub_state)
                logger.info("Stub-LLM dry-run gate: OK (proceeding to real pass)")
            except Exception as _se:
                # ERROR — the stub-pass is the cheap-tokens gate that
                # catches sub-LLM-layer bugs before we burn Anthropic
                # budget on the real pass. flow-doctor must escalate.
                logger.error(
                    "Stub-LLM dry-run gate: FAILED — halting before real LLM calls. "
                    "Stub-pass error: %s",
                    _se,
                    exc_info=True,
                )
                return {
                    "status": "ERROR",
                    "phase": "stub_pass",
                    "date": run_date,
                    "error": str(_se),
                }
            finally:
                _restore()

            # Stub-pass mutated state + may have left archive in odd shape.
            # Rebuild archive + supporting state for the real pass so it
            # starts from a clean slate.
            archive.close()
            archive = ArchiveManager()
            archive.download_db()
            graph = build_graph()
            initial_state = create_initial_state(
                run_date=run_date,
                archive_manager=archive,
                is_early_close=early_close,
            )

        if dry_run_llm:
            # Preflight-equivalent boot validation — the Saturday SF shell-run
            # wires this via research_dry=true, and it boots EVERY stage in
            # preflight-only mode (the spots run --preflight-only), so NO
            # upstream per-date artifacts are produced. The research graph
            # hard-depends on upstream data — compute_factor_profiles_node
            # reads features/{run_date}/*.parquet (written by DataPhase1) and
            # CORRECTLY hard-fails when it is absent (feedback_no_silent_fails).
            # Invoking the full graph in a dry rehearsal therefore fails on
            # upstream data the dry run can never have produced (the 2026-06-18
            # shell-run failed exactly here). So in dry mode we validate the
            # bootstrap/import/wiring surface — the keystone's actual purpose,
            # symmetric with the spots' --preflight-only — and return
            # ok_dry_run WITHOUT executing the data-dependent graph. The
            # full-graph integration smoke (orchestration / schema / reducers)
            # is covered by CI (fixtured) and the real Saturday run (real
            # upstream data); it never belonged on a data-less dry pass.
            #
            # build_graph() under stubs validates node wiring + imports (the
            # 2026-05-06 RAG-import class of bug); create_initial_state()
            # validates state construction. Neither needs upstream artifacts.
            # Stubs are still installed so the direct-bound archive_writer /
            # email_sender nodes captured at build_graph() time stay inert.
            from dry_run import install_dry_run_stubs
            logger.info(
                "dry_run_llm=True: boot/import/wiring validation only "
                "(preflight-equivalent; data-dependent graph not executed)"
            )
            _restore = install_dry_run_stubs(archive)
            try:
                _dry_graph = build_graph()
                _dry_state = create_initial_state(
                    run_date=run_date,
                    archive_manager=archive,
                    is_early_close=early_close,
                )
            finally:
                _restore()
            logger.info("dry_run_llm boot validation OK for %s", run_date)
            return {
                "status": "OK",
                "dry_run_llm": True,
                "phase": "boot_validation",
                "date": run_date,
            }
        else:
            # Snapshot the PRIOR population BEFORE the champion's graph mutates
            # + commits it (archive_writer node) so the challenger producers
            # (post-step below) start from the SAME held book — a clean
            # selection-only comparison (config#1223 B3).
            _prior_population = archive.load_population()
            final_state = graph.invoke(initial_state)

            # ── Challenger producers (config#1223 research observe substrate) ──
            # FAIL-HARD (Brian ruling 2026-07-03, config#1683): the champion's
            # signals.json is ALREADY persisted by the graph's archive_writer
            # node before this step, so a raise here loses no live deliverable —
            # it turns an experiment gap into a red run instead of weeks of
            # silently-empty signals_shadow/ (the 6/27 audit, then the 7/3
            # observe_alerts packaging miss — both invisible behind the previous
            # WARN swallow). Experiment producers are producers; producers fail
            # loud (feedback_no_silent_fails). run_challengers itself raises
            # ChallengerShadowGapError on any producer gap.
            from producers.runner import run_challengers
            _shadow = run_challengers(
                archive, run_date,
                run_time=final_state.get("run_time", "") or run_date,
                population=_prior_population,
            )
            logger.info("[handler] challenger shadows: %s", _shadow.get("written"))

        # ── Trajectory validation (Phase 2 eval) ──────────────────
        _trajectory_result = None
        try:
            from evals.trajectory import validate_trajectory
            _trajectory_result = validate_trajectory(
                project_name=os.environ.get("LANGCHAIN_PROJECT", "alpha-research"),
            )
            if _trajectory_result and not _trajectory_result["passed"]:
                import logging as _logging
                _logging.getLogger("evals.trajectory").error(
                    "Trajectory validation failed: %s", _trajectory_result["failures"]
                )
        except Exception as _te:
            logger.warning("trajectory validation skipped: %s", _te)

        archive.close()

        # Write health status on success
        try:
            from nousergon_lib.health import Deliverable, write_health
            _population = final_state.get("new_population", [])
            _rotations = final_state.get("population_rotation_events", [])
            _email_sent = final_state.get("email_sent", False)
            write_health(
                module_name="research",
                deliverables=[
                    Deliverable(name="signals", required=True, produced=True),
                    Deliverable(
                        name="research_email",
                        required=False,
                        produced=_email_sent,
                    ),
                ],
                run_date=run_date,
                duration_seconds=time.time() - _health_start,
                summary={
                    "n_population": len(_population) if isinstance(_population, list) else 0,
                    "n_rotations": len(_rotations) if isinstance(_rotations, list) else 0,
                    "market_regime": final_state.get("market_regime", "unknown"),
                },
                bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
            )
        except Exception as he:
            logger.warning("health status write failed: %s", he)

        # Write data manifest
        try:
            from data_manifest import write_data_manifest
            write_data_manifest(
                bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
                module_name="research",
                run_date=run_date,
                manifest={
                    "n_population": len(_population) if isinstance(_population, list) else 0,
                    "n_rotations": len(_rotations) if isinstance(_rotations, list) else 0,
                    "market_regime": final_state.get("market_regime", "unknown"),
                    "n_buy_candidates": len(final_state.get("buy_candidates", [])),
                    "n_universe": len(final_state.get("universe_scores", [])),
                    "weekly_run": weekly,
                    "email_sent": final_state.get("email_sent", False),
                },
            )
        except Exception as _me:
            logger.warning("data manifest write failed: %s", _me)

        # ── Cost-telemetry aggregation ────────────────────────────────
        # Aggregate today's per-call JSONLs into a single parquet that
        # the Backtester evaluator email reads to render the
        # ``## LLM cost report`` section. Previously a manual CLI step
        # (``scripts/aggregate_costs.py``); now invoked inline at the
        # end of every Research Lambda run so no manual action is
        # required between Research and Backtester.
        #
        # Failure is non-fatal — Research already succeeded by this
        # point and the Backtester gracefully renders an empty cost
        # section if the parquet is absent. Logged WARN so a recurring
        # failure surfaces without blocking trading.
        if final_state.get("email_sent"):
            try:
                import boto3 as _boto3_agg
                from scripts.aggregate_costs import aggregate_day
                _agg_summary = aggregate_day(
                    s3_client=_boto3_agg.client("s3"),
                    bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
                    # L4466/config#886: key the parquet by the TRADING day the
                    # run's cost rows were written under (llm_cost_tracker keys
                    # by run_date) — date.today() on a Saturday run pointed the
                    # aggregator at an empty calendar-date partition.
                    target_date=trading_date,
                )
                if _agg_summary is not None:
                    logger.info(
                        "[cost_aggregator] wrote parquet: rows=%d cost=$%.4f → %s",
                        _agg_summary.get("rows_in", 0),
                        _agg_summary.get("total_cost_usd", 0.0),
                        _agg_summary.get("output_key", "<unknown>"),
                    )
                else:
                    logger.warning(
                        "[cost_aggregator] no JSONL files found for today — "
                        "Backtester email will render empty cost section"
                    )
            except Exception as _agg_exc:
                logger.warning(
                    "[cost_aggregator] aggregation failed (non-fatal — "
                    "Backtester gracefully renders empty cost section): %s",
                    _agg_exc,
                )

        # ── Distillation SFT-corpus stats ─────────────────────────────
        # Read the cumulative _sft_raw corpus and (re)write the compact
        # ``distillation/corpus_stats/latest.json`` artifact the console
        # Distillation-Corpus panel + the config#1542 kill-gate trigger
        # consume (deduped, teacher-segregated, per-task counts + growth).
        # Reads the WHOLE corpus each run, so weekly cadence keeps the
        # cumulative number correct. Non-fatal — pure observability hung
        # off a primary path that already succeeded; a recurring failure
        # surfaces via WARN without blocking trading.
        if final_state.get("email_sent"):
            try:
                import boto3 as _boto3_cs
                from scripts.corpus_stats import compute_corpus_stats
                _cs = compute_corpus_stats(
                    s3_client=_boto3_cs.client("s3"),
                    bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
                    target_date=trading_date,
                )
                logger.info(
                    "[corpus_stats] deduped=%d quant_calibrator=%d/%d → %s",
                    _cs["totals"]["deduped_pairs"],
                    _cs["trigger"]["deduped_single_teacher"],
                    _cs["trigger"]["target_pairs"],
                    _cs.get("output_key", "<unknown>"),
                )
            except Exception as _cs_exc:
                logger.warning(
                    "[corpus_stats] stats refresh failed (non-fatal — "
                    "console panel renders last artifact): %s",
                    _cs_exc,
                )

        logger.info("Run complete. Email sent: %s", final_state.get("email_sent", False))
        return {
            "status": "OK",
            "date": run_date,
            "email_sent": final_state.get("email_sent", False),
            "early_close": early_close,
            "weekly_run": weekly,
            "trajectory_passed": _trajectory_result["passed"] if _trajectory_result else None,
        }

    except Exception as e:
        # ERROR — top-level pipeline crash; flow-doctor must escalate
        # so the operator gets paged before the next Step Function tick
        # (vs only finding out via the SF failure email).
        logger.error("Pipeline error: %s", e, exc_info=True)

        # Write health status on failure
        try:
            from nousergon_lib.health import Deliverable, write_health
            write_health(
                module_name="research",
                deliverables=[
                    Deliverable(name="signals", required=True, produced=False),
                ],
                run_date=run_date,
                duration_seconds=time.time() - _health_start,
                error=str(e),
                bucket=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
            )
        except Exception as he:
            logger.warning("health status write failed: %s", he)

        return {"status": "ERROR", "date": run_date, "error": str(e)}
