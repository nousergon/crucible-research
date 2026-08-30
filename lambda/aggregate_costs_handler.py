"""Lambda entry point — daily cost aggregation.

Triggered by Saturday SF after Research completes. Reads per-call JSONL
files from ``decision_artifacts/_cost_raw/{date}/``, writes the daily
parquet at ``decision_artifacts/_cost/{date}/cost.parquet``, and emits
per-agent_id CloudWatch metrics.

Per ROADMAP L1146 (SF-wire ``aggregate_costs.py`` CLI). The script was
manual-trigger-only since PR #74 shipped 2026-05-01; this handler is the
institutional path that closes the manual surface. Shared image with
``handler.py`` + the eval-judge / rationale-clustering Lambdas — CMD
override sets entry point at deploy time.

Event shape (all fields optional except ``date``):

    {
      "date": "2026-05-25",            # ISO YYYY-MM-DD (required) — the END
                                       #   of the aggregation window
      "lookback_days": 8,              # optional; default DEFAULT_LOOKBACK_DAYS
      "bucket": "alpha-engine-research", # default RESEARCH_BUCKET env / fallback
      "dry_run_llm": true,              # shell-run dry path — early return
    }

``date`` names the window END, not the only date aggregated (config-I7407).
Capture is daily and this Lambda is invoked weekly with the SF's ``run_date``,
so a single-date aggregation left every intervening day's rows unaggregated —
measured 2026-08-15, two days of real cost rows and a DEGRADED weekly run.

Returns one of:

    {"status": "OK", "summary": {...}}                — aggregated + parquet written
    {"status": "SKIPPED", "reason": "no_cost_raw_in_window", "date": "..."}
                                                       — no JSONL partitions anywhere in the window
    {"status": "ERROR", "error": "<msg>"}             — exception caught hard

The ``SKIPPED`` status mirrors data #295's pattern (deploy.sh canary
accepts both ``OK`` and ``SKIPPED``) and the L3277 audit's contract —
legitimate upstream no-op MUST NOT trigger rollback.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import UTC, datetime
from datetime import date as date_type

# Repo root on sys.path so ``from scripts.aggregate_costs import ...``
# resolves under Lambda's task layout (mirrors rationale_clustering /
# eval_rolling_mean handlers).
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
    "aggregate_costs",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-aggregate-costs",
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors the other
    shared-image handlers — Lambda init phase 10s ceiling."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    _init_done = True


def _attach_stage_coverage(result: dict, *, run_date: str, window_start) -> None:
    """Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    §2.3a rescope): the assertion lives in the stage's own handler,
    immediately before it returns, rather than a separate end-of-run SF
    state. OBSERVE MODE ONLY — never enables enforcement, never raises.
    Shared by both this handler's terminal returns (OK and the legitimate
    SKIPPED no-op) since both are real completions, not failures.

    ``run_date`` here is ``date_str`` == ``event["date"]`` == the SF
    Payload's ``$.run_date`` verbatim (alpha-engine-config-I8155 — verified
    against nousergon-data/infrastructure/step_function.json's
    ``AggregateCosts`` state) — the execution's own un-normalized run_date,
    never reassigned by this handler. Both call sites already guard against
    a missing/blank ``date_str`` before reaching here (the handler returns
    ERROR early), so this never fabricates a substitute.
    """
    try:
        from krepis.stage_coverage import assert_stage_coverage

        result["stage_coverage"] = assert_stage_coverage(
            "AggregateCosts", run_date=run_date, window_start=window_start,
        )
    except ImportError as exc:
        # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
        # the handler's own outcome is unchanged (config-I7214).
        logger.error("stage-coverage assertion unavailable: %s", exc)
        result["stage_coverage"] = {
            "stage": "AggregateCosts",
            "status": "UNMEASURED",
            "reason": f"assertion unavailable: {exc}",
        }
    except Exception as exc:  # noqa: BLE001 — never let the observer kill the stage it observes
        # alpha-engine-config-I8155: the krepis landing this arc makes
        # run_date a required, contract-enforced kwarg (TypeError on
        # omission, StageCoverageContractError on blank/None). Log loudly
        # and degrade to UNMEASURED rather than raising out of the handler.
        logger.error(
            "[aggregate_costs_handler] stage-coverage assertion raised for "
            "AggregateCosts: %s: %s", type(exc).__name__, exc,
        )
        result["stage_coverage"] = {
            "stage": "AggregateCosts",
            "status": "UNMEASURED",
            "reason": f"assertion raised: {type(exc).__name__}: {exc}",
        }


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
    """Aggregate per-call JSONL cost files into the daily parquet."""
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.now(UTC)
    _ensure_init()

    import boto3

    from evals.lambda_dry import is_dry
    from scripts.aggregate_costs import (
        DEFAULT_LOOKBACK_DAYS,
        aggregate_window,
    )

    # Shell-run dry path — boot + imports above already exercised the
    # bootstrap smoke. Return BEFORE aggregate_day (which reads S3 +
    # writes parquet + emits CW). dry_run_llm short-circuits everything
    # for the Friday-Preflight shell run that doesn't actually need to
    # produce a parquet.
    if is_dry(event):
        logger.info(
            "[aggregate_costs_handler] dry_run_llm=True: shell-run "
            "no-op (no S3 read/write, no CW emit)",
        )
        return {"status": "OK", "dry_run": True}

    date_str = event.get("date")
    if not date_str:
        logger.error(
            "[aggregate_costs_handler] event missing required 'date' field"
        )
        return {
            "status": "ERROR",
            "error": "event missing required 'date' field (ISO YYYY-MM-DD)",
        }

    try:
        target_date = date_type.fromisoformat(date_str)
    except ValueError as exc:
        logger.error(
            "[aggregate_costs_handler] invalid date %r: %s", date_str, exc,
        )
        return {
            "status": "ERROR",
            "error": f"invalid date {date_str!r}: {exc}",
        }

    bucket = event.get("bucket", _DEFAULT_BUCKET)

    logger.info(
        "[aggregate_costs_handler] start date=%s bucket=%s",
        date_str, bucket,
    )

    s3_client = boto3.client("s3")
    cw_client = boto3.client("cloudwatch")

    # config-I7407: aggregate the WINDOW ending at `date`, not that one date.
    # Capture is daily (krepis.cost_sink writes under the wall-clock date of
    # the call, so the Think Tank's daily runs land on dates no SF names)
    # while this Lambda was invoked with a single `$.run_date`. Measured
    # 2026-08-15: raw existed for 08-10/08-11/08-12, parquet stopped at 08-10,
    # and the Saturday run asked for 08-14 -- which had no raw at all, so the
    # stage produced nothing and reported SKIPPED. The `cost_telemetry`
    # transparency row then failed its 8-day freshness check, which is the
    # single [FAIL] behind that run's DEGRADED terminal.
    #
    # The window default matches that row's `max_age_days`, so the producer
    # covers exactly what the consumer asserts. Overridable per invocation for
    # a wider backfill.
    lookback = int(event.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
    try:
        result = aggregate_window(
            s3_client=s3_client,
            bucket=bucket,
            end_date=target_date,
            lookback_days=lookback,
            cw_client=cw_client,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[aggregate_costs_handler] aggregation failed hard"
        )
        return {"status": "ERROR", "error": str(exc)}

    # A window with nothing to do is the legitimate quiet case; a window that
    # aggregated at least one date is a success naming what it covered.
    summary = None if not result["aggregated"] else {
        "dates_aggregated": result["aggregated"],
        "dates_quiet": result["skipped"],
        "lookback_days": lookback,
    }

    if summary is None:
        # Legitimate upstream no-op — no JSONL partitions emitted ANYWHERE
        # in the window (e.g. cost-telemetry kill switch on, or a genuinely
        # quiet stretch). config-I7407 narrowed what this means: it used to
        # fire whenever the ONE named date was empty, which is the common
        # case rather than the exceptional one. Per
        # [[feedback_no_silent_fails]] the no-op is loudly visible
        # (WARN-log + named SKIPPED status), but does NOT raise — the
        # consumer / canary must accept SKIPPED as pass.
        #
        # It is NOT, however, a pass for coverage, and this is the branch
        # that mattered most: an EMPTY partition on a day the pipeline's
        # LLM stages ran is the I7179 defect in its purest form, and until
        # this line existed it returned SKIPPED and the state succeeded.
        # "Legitimately quiet" and "every producer went silent" are the
        # same shape here, so the coverage check runs FIRST and the
        # declaration is what tells them apart.
        _check_fan_in_coverage(event, bucket, date_str)
        logger.info(
            "[aggregate_costs_handler] no _cost_raw partitions in the %d-day "
            "window ending %s — skipping parquet write (no error)",
            lookback, date_str,
        )
        skipped_result = {
            "status": "SKIPPED",
            "reason": "no_cost_raw_in_window",
            "date": date_str,
            "lookback_days": lookback,
        }
        # Deliberately on the SKIPPED path too, and before the return.
        # "Nothing to aggregate" is exactly the shape a total capture
        # outage takes, so the path that reports it is the path that must
        # carry the capture verdict (config-I7407 deliverable 4). Raising
        # here reaches the SF's States.ALL Catch.
        _publish_capture_freshness(s3_client, bucket, target_date, skipped_result)
        _attach_stage_coverage(skipped_result, run_date=date_str, window_start=_started)
        return skipped_result

    logger.info(
        "[aggregate_costs_handler] done dates_aggregated=%s quiet=%d "
        "lookback_days=%d",
        ",".join(summary["dates_aggregated"]),
        len(summary["dates_quiet"]),
        lookback,
    )

    # ── fan-in coverage (alpha-engine-config-I7179) ────────────────────
    #
    # Deliberately AFTER the parquet write and deliberately NOT inside the
    # try/except above.
    #
    # After, because the parquet is the primary deliverable and a coverage
    # verdict must not cost the run its artifact. Outside the except,
    # because that block converts a failure into `{"status": "ERROR"}` —
    # and nothing downstream reads this state's status. The Step Function
    # has no Choice on it; its only failure path is the `States.ALL` Catch,
    # which fires on a RAISED error and routes to
    # MarkAggregateCostsDegraded. A returned ERROR here would be silent,
    # which is the exact shape of defect this check exists to end.
    coverage_verdict = _check_fan_in_coverage(event, bucket, date_str)
    if coverage_verdict is not None:
        summary["coverage"] = coverage_verdict

    _refresh_corpus_stats(s3_client, bucket, date_str, summary)

    result = {"status": "OK", "summary": summary, "date": date_str}
    _publish_capture_freshness(s3_client, bucket, target_date, result)
    _attach_stage_coverage(result, run_date=date_str, window_start=_started)
    return result


def _refresh_corpus_stats(s3_client, bucket: str, date_str: str, summary: dict) -> None:
    """Rewrite the distillation SFT-corpus stats artifact.

    REPOINTED PRODUCER (alpha-engine-config-I7856).
    ``decision_artifacts/distillation/corpus_stats/latest.json`` had exactly one
    invoker — an inline post-step in ``lambda/handler.py``'s champion pass,
    deleted with the research graph in ``crucible-research-PR685``. Measured
    2026-08-20: the artifact is frozen at 2026-07-01 and
    ``compute_corpus_stats`` had no invoker anywhere in the fleet.

    Two consumers, both real and both GRACEFUL, which is precisely why the
    absence went unnoticed for seven weeks: the console Distillation-Corpus
    panel renders an explainer on ``None``, and
    ``alpha-engine-config/scripts/check_corpus_trigger.py`` prints "no stats
    artifact — nothing to do" and returns 0. The second one matters: the
    config#1542 distill-or-shelve kill-gate clock reads its trigger metric from
    this artifact, so a frozen corpus_stats means the 90-day clock can never
    auto-start — a decision gate silently disarmed, not a panel gone blank.
    Meanwhile the corpus itself keeps growing: Think Tank writes
    ``decision_artifacts/_sft_raw/`` daily through ``nousergon_lib.sft``.

    THIS stage is the right host. ``compute_corpus_stats`` reads the WHOLE
    cumulative corpus on every call, so the cadence only has to be regular, not
    frequent — and AggregateCosts is the weekly SF stage that survived the same
    deletion (``scripts/aggregate_costs.py`` lost its inline call site too and
    this Lambda is its live second invoker). Same repo, same weekly cadence,
    same bucket, no new schedule, no new IAM.

    Fail-soft with an alert, mirroring the two sibling post-steps in
    ``signals_envelope_handler``: the parquet is already written and is this
    stage's primary deliverable, so sinking the run on an observability refresh
    would trade the deliverable for the stat. Never silent — the alert is the
    recording surface, and the artifact's own ``generated_date`` is what a
    freshness row reads.
    """
    try:
        from scripts.corpus_stats import compute_corpus_stats

        stats = compute_corpus_stats(s3_client, bucket, target_date=date_str)
        summary["corpus_stats"] = {
            "deduped_pairs": stats.get("totals", {}).get("deduped_pairs"),
            "output_key": stats.get("output_key"),
        }
        logger.info(
            "[corpus_stats] refreshed for %s: deduped_pairs=%s -> %s",
            date_str,
            summary["corpus_stats"]["deduped_pairs"],
            summary["corpus_stats"]["output_key"],
        )
    except Exception as exc:  # noqa: BLE001 — secondary observability, never fatal
        logger.warning(
            "[corpus_stats] refresh FAILED for %s (non-fatal — the cost "
            "parquet is already written): %s", date_str, exc,
        )
        try:
            from observe_alerts import publish_observe_alert

            publish_observe_alert(
                f"distillation corpus_stats refresh FAILED for {date_str} "
                f"(non-fatal, cost parquet already written). The config#1542 "
                f"kill-gate trigger reads a stale artifact until this "
                f"succeeds: {exc}",
                source="aggregate_costs_handler:corpus_stats",
                dedup_key=f"corpus_stats_refresh_fail:{date_str}",
            )
        except Exception:  # noqa: BLE001 — the alerter itself is best-effort
            logger.exception("[corpus_stats] alert publication also failed")


def _publish_capture_freshness(s3_client, bucket: str, target_date, result: dict):
    """Publish the capture-stream sentinel and assert the stream is alive.

    config-I7407 deliverable 4. ``llm_cost_parquet`` watches the PRODUCT of
    capture; until this ran, nothing watched capture. The two absences are
    not the same: an aggregator that dies stops writing a parquet and ages
    into STALE, while producers that all stop emitting leave the aggregator
    correctly reporting ``SKIPPED`` on a green Step Function — the exact
    shape of I7179, where 812s of LLM calls produced zero rows and five
    detectors described it as five other things.

    Raises :exc:`CostCaptureStaleError` past the ceiling, which reaches the
    Step Function's ``States.ALL`` Catch and its DEGRADED terminal. That is
    on purpose and is the whole deliverable: ``principles.md`` §2.7 —
    *no data* is never rendered as green. Deliberately NOT swallowed into
    the returned status, because nothing downstream reads this state's
    status (see ``_check_fan_in_coverage`` above for the same reasoning).

    Runs after the parquet write on the OK path and after the coverage check
    on both, so a raise here never costs the run its primary artifact. It
    also runs on the SKIPPED path, because a window with nothing in it is
    precisely the case this is here to grade rather than accept.
    """
    from scripts.cost_capture_freshness import evaluate_and_publish

    state = evaluate_and_publish(s3_client, bucket, as_of=target_date)
    result["capture_stream"] = {
        "last_capture_date": state["last_capture_date"],
        "days_since_last_capture": state["days_since_last_capture"],
        "producers": state["producers_on_last_capture_date"],
    }
    return state


def _check_fan_in_coverage(event, bucket: str, date_str: str):
    """Assert every stage that ran also emitted a cost record.

    Returns the verdict dict, or ``None`` when the Step Function supplied
    no ``coverage`` block (a manual or legacy invocation — this handler is
    also reachable from the deploy canary and from
    ``scripts/aggregate_costs.py``). Raises on a breach or on being unable
    to measure; both reach the SF's Catch.
    """
    declaration = event.get("coverage")
    if not declaration:
        logger.info(
            "[aggregate_costs_handler] no coverage declaration in the event — "
            "fan-in coverage NOT checked for %s. This is expected for a manual "
            "or canary invocation and never for a weekly-SF run.",
            date_str,
        )
        return None

    import boto3

    from scripts.aggregate_costs import _INPUT_PREFIX
    from scripts.cost_coverage import (
        evaluate_coverage,
        execution_started_at,
        observed_producers_for_execution,
        stages_entered,
    )

    # alpha-engine-config-I9261. This read used to be a single prefix,
    # f"{_INPUT_PREFIX}/{date_str}/". `date_str` is the SF's run_date; the
    # cost sink partitions by the record's own UTC ts. The weekly pipeline
    # starts 09:00 UTC on the day AFTER its run_date, so every one of its
    # producers wrote into a partition the check never opened, while the
    # partition it DID open held the previous day's daily-pipeline objects.
    # `entered` is a fact about this execution; `observed` now is too.
    execution_arn = declaration.get("execution_arn", "")
    sfn_client = boto3.client("stepfunctions")
    started_at = execution_started_at(execution_arn, sfn_client=sfn_client)
    observed = observed_producers_for_execution(
        boto3.client("s3"),
        bucket,
        _INPUT_PREFIX,
        started_at=started_at,
    )
    entered = stages_entered(execution_arn, sfn_client=sfn_client)
    verdict = evaluate_coverage(
        observed=observed, declaration=declaration, entered=entered
    )
    logger.info(
        "[aggregate_costs_handler] fan-in coverage OK for %s — covered=%s "
        "stages_not_entered=%s",
        date_str,
        ",".join(verdict["covered"]) or "(none)",
        ",".join(verdict["stages_not_entered"]) or "(none)",
    )
    return verdict
