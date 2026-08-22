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
import tempfile

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
from nousergon_lib.logging import monitor_handler, setup_logging  # noqa: E402

_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flow-doctor.yaml"
)
setup_logging(
    "research",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-graph",
)

logger = logging.getLogger(__name__)


class RetiredResearchPathError(RuntimeError):
    """This function was invoked for a stage that no longer exists.

    The champion research graph was retired 2026-07-12 and its code deleted
    (alpha-engine-config-I7827, champion-challenger-policy.md §6). Raised
    rather than returning ``SKIPPED`` so a reactivated trigger is LOUD: a
    silent skip is indistinguishable from a run that had nothing to do, and
    that ambiguity is how this component was read as live three times.
    """

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


def _maybe_emit_self_test(trading_date: datetime.date) -> None:
    """Run the numeric self-test and publish its artifact + console row.

    alpha-engine-config-I7262. It used to be described against the handler's
    two OBSERVABILITY producers (`_maybe_emit_scorecard` /
    `_maybe_emit_team_accuracy`), both of which were deleted with the champion
    pass they hung off (alpha-engine-config-I7827). The distinction they drew
    still governs this function: an observability producer's absence degrades a
    dashboard, while this one is a §2.3a CORRECTNESS VERDICT whose absence must
    never read as a pass. So the failure paths are deliberately asymmetric:

    * the battery itself never raises (its own contract), and a FAIL or UNKNOWN
      verdict is logged at ERROR and carried into the artifact — it does NOT
      fail the run, because a verdict stage that dies must not kill the stages
      that do not depend on it;
    * a failed ARTIFACT write is logged at ERROR and continues, because the
      verdict is already in the logs and the briefing is the primary
      deliverable;
    * a failed CONSOLE publish is swallowed by the emitter, and the console
      renders the missing row as `unreadable`/stale rather than green — so the
      gap stays visible rather than becoming a false all-clear.

    The swallowed failure mode here is "the self-test itself could not run";
    the primary deliverable (the weekly signal set) survives untouched; the
    recording surface is the ERROR log plus the ABSENT artifact and console row.
    """
    run_date = trading_date.isoformat()
    try:
        from scoring.self_test import publish_console_row, run_self_test, write_self_test

        result = run_self_test(run_date=run_date)
        log_at = logger.info if result.get("verdict") == "PASS" else logger.error
        log_at(
            "Research self-test: verdict=%s cases=%s failed=%s errored=%s known_gaps=%s libs=%s",
            result.get("verdict"),
            result.get("n_cases"),
            result.get("n_failed"),
            result.get("n_errored"),
            result.get("n_known_gaps"),
            result.get("libraries"),
        )
        try:
            bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")
            write_self_test(bucket, run_date, result)
        except Exception:  # noqa: BLE001 — evidence emission never blocks the run
            logger.error(
                "self-test artifact emission failed for %s (verdict=%s is still in the logs)",
                run_date,
                result.get("verdict"),
                exc_info=True,
            )
        # `principles.md` §2.7 — a check that reports nowhere is unobserved.
        publish_console_row(result)
    except Exception:  # noqa: BLE001 — the battery must never fail the briefing
        logger.error(
            "Research self-test could not run at all — NO correctness guarantee is granted for this run's numbers.",
            exc_info=True,
        )


def _resolve_self_test_date(event: dict) -> str:
    """Trading day for a `mode: self_test` invocation.

    Normalised through the same fleet chokepoint `_run_challengers_only` uses
    (`nousergon_lib.dates.resolve_trading_day`) and for the identical reason
    recorded there under alpha-engine-config-I7419: the weekly SF passes the
    EXECUTION's calendar date, and the weekly run is a Saturday — never itself a
    session. Keying the artifact on Saturday would write it to a date no
    consumer looks up, which is a different way of never producing it.
    """
    from nousergon_lib.dates import resolve_trading_day

    raw = event.get("date") or datetime.date.today().isoformat()
    return str(resolve_trading_day(raw))


def _run_challengers_only(event: dict) -> dict:
    """Operator recovery mode (config#1683): re-emit the challenger producers'
    shadow cohort for the MOST RECENT weekly run without re-running the
    champion graph.

    Event: ``{"mode": "challengers_only", "date": "YYYY-MM-DD"}``. ``date`` may
    be either the CALENDAR date of the run (what the weekly SF passes as
    ``$.run_date``) or a trading day; it is normalised to the **trading day**
    before anything reads it, because every artifact this function touches --
    ``signals/latest.json``, the population table's ``entry_date`` -- is keyed
    on the knowledge axis (``DATE_CONVENTIONS``: ``trading_day`` is always
    backward-looking, and is *"the right answer for ~99% of consumers"*).

    **Why the normalisation is load-bearing (alpha-engine-config-I7419).**
    The weekly SF passes the execution's calendar date, and the weekly run is
    a SATURDAY -- never itself a session. So the guard below compared Saturday
    against Friday's population commit and raised on **every scheduled weekly
    run**, measured on both 2026-08-15 executions::

        ValueError: challengers_only is only valid for the latest run
        (latest='2026-08-14', requested='2026-08-15')

    The stage is non-blocking, so it degraded rather than failing -- but
    ``$.research_degraded_local`` propagates to ``$.degraded_summary``, and
    since ``config-I6891`` a degraded summary terminates the run in the
    ``DegradedRun`` Fail state. This one argument therefore made an honest
    weekly terminal impossible on any Saturday for as long as it was wired.

    ``nousergon_lib.dates.resolve_trading_day`` is the fleet chokepoint the
    backtester, evaluator, Scanner and ``signals_envelope_handler`` all resolve
    through -- the last of which carries a comment about this exact class. This
    function was the holdout.

    The freshness guard is KEPT, not replaced by the resolution: resolving the
    trading day answers *which* run this is, and the comparison against
    ``signals/latest.json`` answers whether that run's population commit is
    actually the most recent one. Dropping the comparison -- e.g. by reading
    the date out of ``latest.json`` instead -- would silently reconstruct
    against a stale cohort on any cycle where ``SignalsEnvelope`` did not run.

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

    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    # This function backs the SF's ChallengerShadow state exclusively (the
    # only caller is handler()'s mode="challengers_only" dispatch, which is
    # exactly the SF's ChallengerShadow invocation shape) — no ambiguity
    # about which stage this attribution belongs to.
    _started = datetime.datetime.now(datetime.UTC)

    from nousergon_lib.dates import resolve_trading_day

    calendar_date = event.get("date")
    if not calendar_date:
        raise ValueError("challengers_only requires event['date'] (YYYY-MM-DD)")

    # alpha-engine-config-I8155: `calendar_date` is the SF execution's own
    # run_date (delivered as `$.date` per the SF Payload for ChallengerShadow),
    # un-normalized, and is never reassigned below — that is what
    # stage_coverage groups a run's verdicts by (the ONLY key one execution's
    # verdicts share).
    execution_run_date = calendar_date

    run_date = resolve_trading_day(calendar_date[:10])
    if run_date != calendar_date[:10]:
        logger.info(
            "[challengers_only] normalized run_date %s (calendar) -> %s "
            "(trading day) - signals/ and the population table are keyed on "
            "the knowledge axis (config-I7419)",
            calendar_date,
            run_date,
        )

    from archive.manager import ArchiveManager

    archive = ArchiveManager()
    archive.download_db()
    try:
        latest = _json.loads(archive.s3.get_object(Bucket=archive.bucket, Key="signals/latest.json")["Body"].read())
        latest_date = latest.get("date")
        if latest_date != run_date:
            raise ValueError(
                f"challengers_only is only valid for the latest run "
                f"(latest={latest_date!r}, requested={run_date!r}, from "
                f"calendar date {calendar_date!r}) — the prior-population "
                f"reconstruction is membership-exact only against the most "
                f"recent population commit (config#1683). Both dates are "
                f"trading days: this is a genuinely stale cohort, not the "
                f"calendar-vs-trading-day mismatch of config-I7419."
            )

        population = archive.load_population()
        prior_population = [p for p in population if p.get("entry_date") != run_date]
        logger.info(
            "[challengers_only] run_date=%s prior_population=%d (current %d minus %d entered on run_date)",
            run_date,
            len(prior_population),
            len(population),
            len(population) - len(prior_population),
        )

        from producers.runner import run_challengers

        shadow = run_challengers(
            archive,
            run_date,
            run_time=datetime.datetime.now(datetime.UTC).isoformat(),
            population=prior_population,
        )
        result = {
            "status": "OK",
            "mode": "challengers_only",
            "date": run_date,
            "written": shadow["written"],
        }

        # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
        # §2.3a rescope): the assertion lives in the stage's own handler,
        # immediately before it returns, rather than a separate end-of-run
        # SF state. OBSERVE MODE ONLY — never enables enforcement, never
        # raises.
        if not execution_run_date:
            # alpha-engine-config-I8155: never fabricate a date to satisfy
            # the (now-required) run_date argument. The event['date']
            # required-field guard above makes this unreachable today, but
            # the check stays as the contract boundary rather than relying
            # on that upstream guard.
            logger.error(
                "[challengers_only] stage-coverage assertion SKIPPED for "
                "ChallengerShadow: no execution run_date on this event "
                "(execution identity absent)"
            )
            result["stage_coverage"] = {
                "stage": "ChallengerShadow",
                "status": "UNMEASURED",
                "reason": "execution run_date absent from event",
            }
        else:
            try:
                from krepis.stage_coverage import assert_stage_coverage

                result["stage_coverage"] = assert_stage_coverage(
                    "ChallengerShadow",
                    run_date=execution_run_date,
                    window_start=_started,
                )
            except ImportError as exc:
                # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe
                # mode — the handler's own outcome is unchanged (config-I7214).
                logger.error("stage-coverage assertion unavailable: %s", exc)
                result["stage_coverage"] = {
                    "stage": "ChallengerShadow",
                    "status": "UNMEASURED",
                    "reason": f"assertion unavailable: {exc}",
                }
            except Exception as exc:  # noqa: BLE001 — never let the observer kill the stage it observes
                # alpha-engine-config-I8155: the krepis landing this arc
                # makes run_date a required, contract-enforced kwarg
                # (TypeError on omission, StageCoverageContractError on
                # blank/None). Both are library-internal contract failures,
                # not stage failures — log loudly and degrade to UNMEASURED
                # rather than raising out of the handler.
                logger.error(
                    "[challengers_only] stage-coverage assertion raised "
                    "for ChallengerShadow: %s: %s",
                    type(exc).__name__, exc,
                )
                result["stage_coverage"] = {
                    "stage": "ChallengerShadow",
                    "status": "UNMEASURED",
                    "reason": f"assertion raised: {type(exc).__name__}: {exc}",
                }

        return result
    finally:
        archive.close()


@monitor_handler
def handler(event, context):
    """Entry point. Runs the handler, then flushes cost telemetry.

    The `finally` is the whole point (alpha-engine-config-I7423).
    `krepis.cost_sink.S3JsonlCostSink` buffers to 200 records per
    `(date, callsite_id)` group and otherwise relies on an `atexit` hook —
    and **an AWS Lambda container is FROZEN between invocations, not exited,
    so `atexit` never runs.** A handler finishing below the threshold writes
    nothing at all, and the container may be reclaimed hours later without
    ever reaching interpreter shutdown.

    Measured 2026-08-15 on weekly-SF execution `watch-rerun-2026-08-15-2`:
    `AggregateCosts` reported `single-agent-quant` among `2 stage(s) ran and
    emitted no cost record ... Observed producers: (none)`. The env wiring was
    correct, the sink was constructed, the records were priced and accepted,
    and every one of them died in memory.

    Applied to EVERY handler in this directory rather than to the ones known
    to call an LLM today: `flush_default_sink` returns 0 when no sink is
    configured and never raises, so the uniform rule costs nothing and leaves
    no per-handler judgment call for the next producer to get wrong.
    """
    try:
        return _run(event, context)
    finally:
        try:
            from krepis.cost_sink import flush_default_sink
            _n = flush_default_sink()
            if _n:
                logger.info("cost sink flushed: %d object(s)", _n)
        except ImportError as exc:
            # Loud, not silent: the image's krepis pin predates the function
            # (floor is >=0.59.8). Cost records for this invocation are lost,
            # and AggregateCosts' fan-in coverage check will name this stage.
            logger.error("cost-sink flush unavailable — records lost: %s", exc)


def _run(event, context):
    """
    AWS Lambda handler for the research runner.

    Three modes reach real work; there is no fourth, and no default run.
    The champion LangGraph pass this handler used to carry was retired on
    2026-07-12 and DELETED (alpha-engine-config-I7827) — see the closing
    block of this function for what was removed and what proved it dead.

      - ``mode="challengers_only"`` + ``date=YYYY-MM-DD`` → the weekly SF's
        ChallengerShadow stage (and the config#1683 operator recovery path):
        builds every registered challenger's shadow cohort. THE live path.
      - ``mode="self_test"`` → the §2.3a numeric correctness verdict
        (alpha-engine-config-I7726). ``dry_run: true`` exercises the battery
        and writes nothing.
      - ``{"dry_run_llm": true}`` → the deploy canary
        (``infrastructure/deploy.sh``): a deterministic, side-effect-free boot
        validation of the LIVE producer path.

    Anything else RAISES ``RetiredResearchPathError``.

    Returns:
        dict with status: "OK" | "SKIPPED" | "ERROR"
    """
    # Run one-time expensive imports + SSM secrets fetch on the first
    # invocation. Warm-container calls are a no-op via the _init_done flag.
    _ensure_init()
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

    if event.get("mode") == "challengers_only":
        return _run_challengers_only(event)

    # alpha-engine-config-I7726 — the self-test needs a reachable entry point.
    #
    # `_maybe_emit_self_test` lives at the end of the weekly_run branch below,
    # but MEASURED 2026-08-19: the weekly SF's ONLY invocation of this Lambda is
    # the ChallengerShadow state, payload {"mode": "challengers_only"}, which
    # returns immediately above — and the documented fallback trigger, the
    # EventBridge rule `alpha-research-weekly` cron(0 6 ? * SAT *), is DISABLED.
    # So the call site was structurally unreachable and
    # `research/{date}/self_test.json` had ZERO instances in the bucket, ever,
    # since the row was registered on 2026-08-13. A §2.3a CORRECTNESS VERDICT
    # whose absence must never read as a pass had never once been emitted.
    #
    # Its own mode rather than a graft onto ChallengerShadow: `run_self_test` is
    # a known-answer battery over the DEPLOYED instrument (scoring/self_test.py
    # -> nousergon_lib.quant.selftest), so it does not depend on the champion
    # graph having run and must not inherit a shadow-cohort stage's contract or
    # its failure modes. One stage, one contract, one registry row, one honest
    # `produced_by` — which is what the registry could not be given while this
    # was reachable only as a side effect of something else.
    if event.get("mode") == "self_test":
        run_date = _resolve_self_test_date(event)
        # `dry_run` mirrors the RunScope convention already used in this graph
        # (config-I7620): the Friday-PM shell run exercises the full path and
        # WRITES NOTHING. That is load-bearing here, not cosmetic — a shell run
        # that published research/{friday}/self_test.json would leave a fresh,
        # genuine verdict sitting in the bucket, and the freshness monitor would
        # read it as the week's artifact. Saturday's real run could then fail to
        # produce one and nothing would say so. A rehearsal must not be able to
        # satisfy the check that watches the performance.
        if event.get("dry_run"):
            logger.info(
                "self-test dry run for %s — battery exercised, no artifact and "
                "no console row written (config-I7726)", run_date,
            )
            from scoring.self_test import run_self_test

            result = run_self_test(run_date=run_date)
            return {
                "status": "OK", "mode": "self_test", "date": run_date,
                "dry_run": True, "verdict": result.get("verdict"),
            }
        _maybe_emit_self_test(datetime.date.fromisoformat(run_date))
        # The battery never raises and never fails the caller: its verdict is
        # carried in the artifact, the console row and the logs. The SF state is
        # non-blocking for the same reason — a verdict stage that dies must not
        # kill the stages that do not depend on it.
        return {"status": "OK", "mode": "self_test", "date": run_date}

    force = event.get("force", False)
    weekly = event.get("weekly_run", False)
    # `dry_run_llm` — the deploy canary's stub-only mode: no real LLM call, no
    # S3 write, no email, no DB upload. `skip_dry_run_gate` went with the
    # champion pass (alpha-engine-config-I7827): it only ever chose whether to
    # run the stub-LLM auto-gate before the real graph pass, and there is no
    # longer a graph pass to gate.
    dry_run_llm = event.get("dry_run_llm", False)

    # ── dry_run_llm — deploy-canary boot/import/wiring smoke ──────────
    # dry_run_llm is EXCLUSIVELY the deploy canary's smoke-test mode
    # (infrastructure/deploy.sh sends {"dry_run_llm": true}). It MUST be a
    # deterministic, side-effect-free boot validation of the LIVE path:
    # `producers.boot_check.run_live_producer_boot_check()` imports every
    # producer the ChallengerShadow stage will build (derived from
    # producers/registry.py, so a newly-registered arm is covered the day it
    # lands) and drives the shared signals-payload assembly on a synthetic
    # in-memory state. Neither needs upstream data, S3, the DB, or the wall
    # clock, so the canary behaves identically no matter WHEN a deploy lands.
    # (The time gate, the trading-day gate, preflight and the DB download that
    # used to sit between the invoke and this return went with the champion
    # pass.) Previously the dry return was buried past real I/O, so a
    # deploy landing inside the 5:40-5:55am PT weekday gate window executed
    # real S3/DB work in the canary; a transient failure there returned
    # status=ERROR and tripped a spurious auto-rollback (2026-07-21 incident).
    # This matches the fleet convention already used by the thinktank /
    # aggregate_costs / rationale_clustering / eval_judge dry paths (return
    # before any S3 / secrets).
    #
    # Until alpha-engine-config-I7827 this block called
    # `graph.research_graph.build_graph()` — a producer RETIRED 2026-07-12 and
    # unreachable in production since. The deploy's own safety check was
    # smoke-testing dead code and NOT smoke-testing the live producers, so a
    # deploy that broke `producers/` passed the canary green. The stub
    # installation is KEPT unchanged: that is what the 2026-05-04 misfire (a
    # canary that produced a real signals.json + research email outside the
    # Saturday cadence) is about, and it is orthogonal to which path runs.
    # tests/test_handler_dry_run_canary_noop.py holds the no-op contract.
    if dry_run_llm:
        from nousergon_lib.dates import resolve_trading_day

        from archive.manager import ArchiveManager
        from dry_run import install_dry_run_stubs
        from producers.boot_check import run_live_producer_boot_check

        _dry_run_date = resolve_trading_day(datetime.date.today().isoformat())
        logger.info(
            "dry_run_llm=True: LIVE producer boot/import/wiring validation only "
            "(preflight-equivalent; no LLM, no S3 write, no email, no DB)"
        )
        # ArchiveManager() is constructed but never downloads the DB — boot
        # validation needs no upstream artifacts. Stubs are installed so any
        # write path reachable from the producer modules stays inert even
        # though the check itself performs no I/O.
        _dry_archive = ArchiveManager()
        _restore = install_dry_run_stubs(_dry_archive)
        try:
            _boot = run_live_producer_boot_check()
        finally:
            _restore()
        logger.info(
            "dry_run_llm boot validation OK for %s — producers=%s",
            _dry_run_date,
            _boot["producers"],
        )
        return {
            "status": "OK",
            "dry_run_llm": True,
            "phase": "boot_validation",
            "date": _dry_run_date,
            "producers": _boot["producers"],
        }

    # ── The champion graph pass is GONE (alpha-engine-config-I7827) ──────
    #
    # Everything below this point used to be the weekday/weekly champion run:
    # the time gate, the trading-day gate, preflight, the DB download, the
    # scorecard / team-accuracy emitters, `build_graph()`, the stub-LLM
    # auto-gate, `graph.invoke()`, the challenger post-step and the health /
    # manifest / cost-aggregation tail. Every one of those was reachable ONLY
    # from `force`, `weekly_run`, or the 5:40-5:55am PT weekday window, and
    # measured 2026-08-20 none of the three has an invoker:
    #
    #   * the weekly SF (`ne-weekly-freshness-pipeline`) has had no `Research`
    #     state since 2026-07-14 (nousergon-data#814); its only invocation of
    #     `alpha-engine-research-runner:live` is ChallengerShadow, handled at
    #     the top of this function;
    #   * EventBridge `alpha-research-weekly` and `alpha-research-daily` are
    #     both DISABLED;
    #   * `infrastructure/spot_research_weekly.sh` → `weekly_box_runner.py`
    #     → `handler(weekly_run=True)` had no invoker in any fleet repo, and
    #     both files were DELETED 2026-08-20 (alpha-engine-config-I7856).
    #
    # The producer it ran (`agentic_sector_teams`) is `kind="retired"`,
    # `retired_date="2026-07-12"`, and champion-challenger-policy.md §6 is
    # explicit that a retired arm's CODE is deleted, not left dormant: "a
    # disabled-but-present arm ... reads as capability while doing nothing, and
    # a future change can silently reactivate it."
    #
    # So an invocation reaching here is a DEFECT, not a run: something is
    # sending this function a payload whose stage no longer exists. It raises
    # rather than returning SKIPPED, because a silent skip is exactly how the
    # previous three misreadings of this component happened.
    raise RetiredResearchPathError(
        "the champion research graph was retired 2026-07-12 and its code was "
        "deleted (alpha-engine-config-I7827). This function serves "
        'mode="challengers_only", mode="self_test" and the deploy canary '
        '({"dry_run_llm": true}) only. Received an event with '
        f"force={force!r} weekly_run={weekly!r} mode={event.get('mode')!r} — "
        "whatever sent it is pointing at a stage that no longer exists."
    )
