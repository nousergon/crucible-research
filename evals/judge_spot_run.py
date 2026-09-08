"""The eval judge, run to completion on a spot instance.

**Ruling.** Brian, 2026-08-29, verbatim: *"perhaps if the lambda times out then
we need to put the judge on a spot instance."* And, confirming the coverage
target the same day: *"ok lets keep 100% coverage and move to spot."* Tracked
as ``alpha-engine-config-I9309``.

**What was wrong.** ``EvalJudgeProcess`` ran as a Lambda under a 960s ceiling.
At the measured 45-105s per synchronous judge call it covered roughly 8-15 of a
~85-artifact corpus, reported ``complete=False`` honestly, and returned
success. Honest reporting on a successful stage is the anti-pattern Brian named
on 2026-08-14 about the Director — *"director should NEVER time out, if it
times out it FAILS"* — recorded with the rationale that retries and bigger
ceilings are not fixes **because they make a latency regression survivable
instead of visible**.

**Topology, and why the three-stage chain collapsed to one run.** Submit → Poll
→ Process existed to drive an ASYNCHRONOUS provider batch API: submit, wait,
collect. That API is retired (``alpha-engine-config-I9263``, Brian: *"at this
point we shouldn't be using the anthropic api at all"*), and with no batch rung
there is nothing to poll — ``poll_batch`` returns terminal immediately for both
synthetic id prefixes. On a box with no execution ceiling the natural shape is
one synchronous run to completion, so:

* **Submit stays a Lambda.** Its work is an S3 listing and a plan manifest —
  seconds, no LLM call, no ceiling risk — and it is where the transport rung is
  resolved and the degradation record written. Moving it would buy nothing and
  cost the stage identity that ``ARTIFACT_REGISTRY`` and the cost fan-in key
  off.
* **Poll is deleted**, states and Lambda both. It is pure async-API residue,
  and a state that can only ever fall straight through is a reader's trap.
* **Process keeps its NAME and changes its SUBSTRATE.** This module is what
  ``EvalJudgeProcess`` now runs, over SSM on a spot instance. The name is
  load-bearing beyond aesthetics: ``eval_artifact_latest.produced_by`` names
  ``EvalJudgeProcess``, ``AggregateCosts.required_producers`` keys on it, and
  the stage-coverage registry has a row for it. All three statements remain
  true, because the stage still produces exactly what they say it does.

**Coverage is a verdict, not a field.** ``evals.judge_coverage.enforce_coverage``
compares artifacts planned against artifacts graded and raises on any gap.
A shortfall exits non-zero, which fails the SSM command, which fails the SF
stage. The transport rung stays honestly *reported* (``degraded_transport``) —
that is a legitimate, priced, deliberate ladder step. Coverage is the different
question, and it is now pass/fail.

**No deadline.** ``process_batch_results(remaining_s=None)`` — the budget
machinery's own docstring already reserved that path for "the CLI, spot runs
and tests". Nothing here re-imposes a ceiling: a ceiling is the defect.

**The router is still the only way out.** Nothing in this module names a model
id, a base URL, a provider or an SDK client. The launcher exports
``KREPIS_EXEC_CONTEXT=ec2`` and ``evals.judge.judge_exec_context`` reads it, so
the same image asks the router the right question from either substrate.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

#: Exit code for a coverage shortfall, distinct from a crash.
#:
#: A crash and an under-covered sweep need different first moves — one is a
#: defect in the judge, the other is capacity or an upstream producer — and the
#: SSM ``ResponseCode`` is the only signal that survives to the Step Function
#: when a log has not been read yet.
EXIT_COVERAGE_SHORTFALL = 3

#: S3 key for the run record this module leaves behind.
RUN_RECORD_PREFIX = "decision_artifacts/_eval_batch_plans"


def run_record_key(*, date: str) -> str:
    """Where the spot run's own summary lands.

    Beside the plan manifest and the transport degradation record, so one
    listing answers what was planned, what transport served it, and what it
    covered. A reader chasing "why does this week have fewer evals" should not
    need to know three prefixes.
    """
    return f"{RUN_RECORD_PREFIX}/{date}/spot_run.json"


def _persist_run_record(record: dict, *, bucket: str, s3_client=None) -> str:
    """Write the run record to S3 and return its key.

    Unguarded, and called BEFORE the coverage verdict is enforced. Both
    choices matter: this record is the durable evidence of what the run
    covered, so on the failing path it is the only artifact that explains the
    failure — writing it after the raise would mean the runs that most need
    evidence are exactly the runs that leave none.
    """
    import boto3

    s3 = s3_client or boto3.client("s3")
    key = run_record_key(date=record["date"])
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(record, indent=2, default=str).encode("utf-8"),
    )
    logger.info("[judge_spot_run] run record -> s3://%s/%s", bucket, key)
    return key


def _stage_coverage(*, run_date: str, window_start) -> dict:
    """EvalJudgeProcess's stage-coverage self-assertion, OBSERVE MODE ONLY.

    Carried over from ``lambda/eval_judge_process_handler.py`` when that
    handler was retired (alpha-engine-config-I9329). The SF stage did not go
    away — only its substrate did — so the assertion must not go away either:
    a stage that stops emitting a coverage verdict is indistinguishable, on
    the console, from a stage that found nothing wrong (config-I7214,
    sf-pipeline-policy.md §2.3a).

    One thing IMPROVES on the substrate move. The Lambda had no ``run_date``
    on its event and had to recover the execution identity from
    ``plan_s3_key``'s ``{date}`` path segment, recording UNMEASURED whenever
    that failed (alpha-engine-config-I8155). Here the SF passes ``--date``
    directly, so the identity is first-class and that whole recovery path is
    gone.

    Never raises: an observer that can kill the stage it observes is worse
    than no observer. Every failure degrades to UNMEASURED **with a reason**,
    which is a value the console can render — not silence.
    """
    from stage_coverage_run_date import resolve_stage_run_date

    # alpha-engine-config-I10171: the SF builds this argv as
    # `--date {$.run_date}` (nousergon-data/infrastructure/step_function.json,
    # `EvalJudgeProcess`), so `--date` already carries the cycle's TRADING
    # day and this stage was never a partition-split writer. Presented to the
    # shared resolver as `{"run_date": ...}` so this substrate is held to the
    # same STATED preference as every Lambda-backed stage — an operator run
    # given a calendar date is recorded, not silently filed one partition
    # over.
    resolved, provenance = resolve_stage_run_date(
        {"run_date": run_date}, stage="EvalJudgeProcess", logger=logger,
    )
    if not resolved:
        # alpha-engine-config-I8155: never fabricate a date.
        logger.error(
            "[judge_spot_run] stage-coverage assertion SKIPPED for "
            "EvalJudgeProcess: no run_date on this invocation",
        )
        return {
            "stage": "EvalJudgeProcess",
            "status": "UNMEASURED",
            "reason": "run_date absent from this invocation",
            **provenance,
        }
    try:
        from krepis.stage_coverage import assert_stage_coverage

        return {
            **assert_stage_coverage(
                "EvalJudgeProcess", run_date=resolved,
                window_start=window_start,
            ),
            # alpha-engine-config-I10171: which field the partition key came
            # from travels WITH the verdict, into the run record.
            **provenance,
        }
    except ImportError as exc:
        logger.error("[judge_spot_run] stage-coverage assertion unavailable: %s", exc)
        return {
            "stage": "EvalJudgeProcess",
            "status": "UNMEASURED",
            **provenance,
            "reason": f"assertion unavailable: {exc}",
        }
    except Exception as exc:  # noqa: BLE001 — never let the observer kill the stage it observes
        logger.error(
            "[judge_spot_run] stage-coverage assertion raised for "
            "EvalJudgeProcess: %s: %s", type(exc).__name__, exc,
        )
        return {
            "stage": "EvalJudgeProcess",
            "status": "UNMEASURED",
            **provenance,
            "reason": f"assertion raised: {type(exc).__name__}: {exc}",
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m evals.judge_spot_run",
        description=(
            "Run the eval judge to completion on a spot instance "
            "(alpha-engine-config-I9309). Grades every planned artifact or "
            "exits non-zero naming the shortfall."
        ),
    )
    p.add_argument(
        "--date", required=True,
        help="Cycle date (YYYY-MM-DD). The SF passes $.run_date, already "
             "normalized to the NYSE trading day.",
    )
    p.add_argument(
        "--plan-s3-key", default=None,
        help="Plan manifest written by EvalJudgeSubmit. When omitted the plan "
             "is built here — used by --preflight-only and ad-hoc replay.",
    )
    p.add_argument(
        "--batch-id", default=None,
        help="Batch id from EvalJudgeSubmit. Required with --plan-s3-key.",
    )
    p.add_argument(
        "--bucket", default=os.environ.get("RESEARCH_BUCKET", "alpha-engine-research"),
    )
    p.add_argument(
        "--preflight-only", action="store_true",
        help="Boot + import + a read-only S3 probe, then exit 0. The Friday "
             "shell-run dry path. Grades nothing and writes nothing.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for a shortfall.

    The shortfall is converted to an exit code here rather than allowed to
    propagate as a traceback, because the caller is an SSM command whose
    ``ResponseCode`` the Step Function reads. A traceback and a
    ``sys.exit(3)`` are both failures, but only one of them says which kind.
    """
    try:
        return _run(argv)
    finally:
        # config-I7423: spot runs call LLMs via the same default sink as the
        # retired Lambda Process handler, but unlike a Lambda container this
        # process exits for real — so atexit would eventually fire. The buffer
        # threshold is 200 records per group; a full judge corpus is ~114, so
        # without an explicit flush ``evaljudge-sync`` emits nothing and
        # ``AggregateCosts`` names EvalJudgeProcess (measured watch-rerun-8,
        # 2026-08-30).
        try:
            from krepis.cost_sink import flush_default_sink

            _n = flush_default_sink()
            if _n:
                logger.info("[judge_spot_run] cost sink flushed: %d object(s)", _n)
        except ImportError as exc:
            logger.error(
                "[judge_spot_run] cost-sink flush unavailable — records lost: %s",
                exc,
            )


def _run(argv: list[str] | None = None) -> int:
    """Judge spot run body — separated so ``main`` can flush in ``finally``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    # Imported after logging is configured and inside main() so
    # --preflight-only exercises the import for real — that is the whole
    # point of the dry path — while a plain --help costs nothing.
    from evals.judge_coverage import JudgeCoverageShortfall, enforce_coverage
    from evals.orchestrator import process_batch_results

    started = datetime.datetime.now(datetime.UTC)
    logger.info(
        "[judge_spot_run] start date=%s bucket=%s plan=%s exec_context=%s",
        args.date, args.bucket, args.plan_s3_key,
        os.environ.get("KREPIS_EXEC_CONTEXT", "(unset — defaults to lambda)"),
    )

    if args.preflight_only:
        # Deliberately mirrors `_spot_common.sh::maybe_run_preflight_only_and_exit`
        # in intent: boot and imports ran for real above; nothing below spends
        # money or writes an artifact. config#4497 is the incident that made
        # this non-optional — two of three split launchers treated
        # --preflight-only as unrecognized and fell through into real work.
        import boto3

        boto3.client("s3").list_objects_v2(
            Bucket=args.bucket, Prefix=f"{RUN_RECORD_PREFIX}/", MaxKeys=1,
        )
        logger.info(
            "[judge_spot_run] preflight OK — imports resolved and S3 is "
            "reachable; NO artifacts graded, NO writes performed"
        )
        return 0

    if not args.plan_s3_key or not args.batch_id:
        logger.error(
            "[judge_spot_run] --plan-s3-key and --batch-id are both required "
            "for a real run (EvalJudgeSubmit returns them). Refusing to "
            "rebuild a plan here: a second plan would mint a second "
            "judge_run_id and split one cycle's evals across two prefixes."
        )
        return 2

    summary = process_batch_results(
        batch_id=args.batch_id,
        plan_s3_key=args.plan_s3_key,
        bucket=args.bucket,
        # No deadline. This is the fix (alpha-engine-config-I9309): the
        # Lambda passed `context.get_remaining_time_in_millis`, and every
        # phase stopped against a 960s wall. `_next_item_affordable` treats
        # None as always-affordable, so the loops run the corpus out.
        remaining_s=None,
    )

    # ── The dedup index, written where the grading happened ───────────
    # `_eval_by_capture/{capture_date}/manifest.json` is the ONLY index
    # `orchestrator.load_already_judged_keys` reads to decide what has
    # already been scored. Nothing else writes it.
    #
    # `lambda/eval_judge_process_handler.py` used to build it here, with a
    # comment naming config#4776 as the incident that put it at this exact
    # point in the flow. That Lambda was DELETED on 2026-08-29 (5e13fe95,
    # "retire the Poll and Process Lambdas", alpha-engine-config-I9329) and
    # this module replaced it — without the manifest build. Measured
    # 2026-09-08 (alpha-engine-config-I10169):
    #
    #   * the last manifest ever written indexes capture date 2026-08-11,
    #     generated 2026-08-13T16:59Z — the last process-Lambda run;
    #   * the first judge runs after the migration, 2026-08-30 19:18Z
    #     through 2026-08-31 01:21Z, graded 876 artifacts covering only
    #     142 distinct (agent, run_id, judge_model) triples — **6.17x
    #     re-judging of one corpus**, because the index each run needed
    #     was no longer being written by the run before it;
    #   * every one of those duplicate gradings emitted a full set of
    #     `agent_quality_score` datapoints, which is what took the
    #     2026-08-26 CloudWatch week to 769 reviews for a combo whose
    #     neighbouring weeks carry 12.
    #
    # So this is not bookkeeping. It is the thing that makes weekly judged
    # volume a function of how much work happened rather than of how many
    # times the pipeline was retried.
    #
    # It runs BEFORE `enforce_coverage` deliberately: a run that graded
    # only part of the corpus still persisted what it graded, and those
    # artifacts must be indexed or the next run re-judges them. The
    # exception is recorded on the run record and logged at ERROR — never
    # swallowed silently — but it does not replace the coverage verdict as
    # the stage's outcome.
    manifest_capture_dates: list[str] = []
    manifest_error: str | None = None
    try:
        import boto3

        from evals.eval_manifest import build_manifests

        written = build_manifests(
            s3_client=boto3.client("s3"),
            bucket=args.bucket,
        )
        manifest_capture_dates = sorted(written)
        logger.info(
            "[judge_spot_run] _eval_by_capture manifests rebuilt for %d "
            "capture date(s): %s",
            len(manifest_capture_dates), manifest_capture_dates,
        )
    except Exception as exc:  # noqa: BLE001
        # (a) failure mode swallowed: the dedup index is not refreshed, so
        #     the NEXT judge run re-judges this run's corpus and inflates
        #     the week's review count. (b) the primary deliverable — the
        #     graded eval artifacts — is already persisted and unaffected.
        #     (c) recorded on the run record below and at ERROR here.
        manifest_error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[judge_spot_run] _eval_by_capture manifest build FAILED — the "
            "next judge run will re-judge this corpus and inflate its "
            "weekly review count (alpha-engine-config-I10169): %s",
            manifest_error,
        )

    from evals.judge_coverage import assess_coverage

    verdict = assess_coverage(summary)
    record = {
        "schema_version": 1,
        "kind": "eval_judge_spot_run",
        "date": args.date,
        "batch_id": args.batch_id,
        "plan_s3_key": args.plan_s3_key,
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.datetime.now(datetime.UTC)
        .isoformat().replace("+00:00", "Z"),
        "substrate": "ec2-spot",
        "exec_context": os.environ.get("KREPIS_EXEC_CONTEXT") or "(unset)",
        "coverage": verdict,
        # config-I7214: the stage's own coverage verdict, observe mode. It
        # rides in the run record because that is this substrate's durable
        # surface — the Lambda returned it in its result payload, and a spot
        # run has no result payload to return.
        "stage_coverage": _stage_coverage(
            run_date=args.date, window_start=started,
        ),
        # The transport rung is reported here too, and separately from the
        # coverage verdict, so the two are never read as one number.
        "degraded_transport": summary.get("degraded_transport"),
        # The dedup index this run refreshed. Absent or short here means the
        # next run re-judges this corpus (alpha-engine-config-I10169).
        "manifest_capture_dates": manifest_capture_dates,
        "manifest_error": manifest_error,
        "summary": {
            k: v for k, v in summary.items()
            # `persisted_keys` and `ungraded_entries` can both be corpus-sized;
            # the counts are on `coverage` and the keys are discoverable by
            # listing the eval prefix. A run record nobody can open is not a
            # record.
            if k not in {"persisted_keys", "ungraded_entries"}
        },
    }
    _persist_run_record(record, bucket=args.bucket)

    try:
        enforce_coverage(summary)
    except JudgeCoverageShortfall as exc:
        logger.error(
            "[judge_spot_run] FAILING the stage on coverage "
            "(alpha-engine-config-I9309):\n%s", exc,
        )
        return EXIT_COVERAGE_SHORTFALL

    logger.info(
        "[judge_spot_run] done — graded %d of %d planned artifacts, "
        "degraded_transport=%s, elapsed=%.0fs",
        verdict["graded"], verdict["planned"],
        summary.get("degraded_transport"),
        (datetime.datetime.now(datetime.UTC) - started).total_seconds(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — process entry point
    sys.exit(main())
