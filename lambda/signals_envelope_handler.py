"""Lambda entry point — no-agent signals-envelope producer (alpha-engine-config
epic #2515 Phase B).

Shares the main runner's ECR image with a CMD override to
``signals_envelope_handler.handler`` (the established image-share pattern —
thinktank / scanner / rationale_clustering). Invoked SYNCHRONOUSLY by the
weekly SF's ``SignalsEnvelope`` state (``arn:aws:states:::lambda:invoke``),
placed immediately after ``RegimeSubstrate`` so the regime read this producer
takes is same-day fresh (config#1580's no-week-old-data invariant).

Thin wrapper around ``scoring/signals_envelope.py``'s library API
(``read_universe_board`` / ``read_regime_substrate`` / ``build_signals_
envelope`` / ``write_envelope``) — no CLI shell-out (the module's ``main()``
is a standalone operator entry point, not this Lambda's call path), no LLM
calls, no LangGraph. See that module's docstring for the full field-policy
rationale (why every research-authored per-name judgment is a documented
neutral default) and the fail-soft/raise split cited below.

Failure contract — RAISE, never return an ERROR dict, ever. Mirrors
``thinktank_handler.py``'s documented rationale exactly (cited here because
this Lambda shares the identical invocation shape): this Lambda is invoked
by an ``arn:aws:states:::lambda:invoke`` SF Task, and the Catch only
triggers on an actual RAISED Lambda error — a normal return value (even an
error-shaped ``{"status": "ERROR", ...}`` dict) is a *successful* Task
completion and would never route through the non-blocking Catch, exactly
the no-silent-fails failure mode. A missing universe board is a hard
precondition failure — ``read_universe_board`` already raises
``RuntimeError`` for it (no-silent-fails: an empty universe board means no
trading day can be constructed from pure-quant sources) — and that
exception propagates UNCAUGHT through this handler. A missing/unreadable
regime substrate is the module's ONE documented fail-soft exception
(``read_regime_substrate`` returns ``None`` -> ``market_regime`` defaults to
``"neutral"`` + a WARN log, mirroring the executor's own
``signal_reader.read_regime_substrate`` posture) and must NOT be re-wrapped
into an error path here either — it is not a failure of this producer's
primary deliverable.

Event shape:

    {
      "run_date": "2026-07-14",   # ISO YYYY-MM-DD (required)
      "dry_run_llm": true,        # shell-run smoke: boot + imports only,
                                   # return BEFORE any S3 access (no LLM
                                   # calls exist in this producer at all —
                                   # the flag name is kept identical to the
                                   # other shared-image handlers' shell-run
                                   # dry contract, see evals/lambda_dry.py)
      "target": "shadow"          # "shadow" (default) or "production" —
                                   # forwarded to write_envelope() verbatim
                                   # (production intent is always explicit,
                                   # never inferred)
      "preflight": true           # config-I2916: Friday-PM shell-run signal
                                   # (SF threads preflight.$: $.research_dry).
                                   # DISTINCT from dry_run_llm — keeps the full
                                   # read/build/write path live (transport
                                   # smoke) and only downgrades the I2880
                                   # universe-board fallback-staleness guard to
                                   # a WARN (the dry Scanner leaves the dated
                                   # board absent, so the stale fallback is
                                   # expected on Fridays). Default false.
      "bucket": "alpha-engine-research"   # optional, default RESEARCH_BUCKET
    }

Returns ``{"status": "OK", "dated_key": ..., "latest_key": ..., "universe_
count": ..., "market_regime": ..., "target": ...}`` on success (or the dry-
path variant ``{"status": "OK", "dry_run": True}``); raises on any failure.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from datetime import UTC, datetime

# Repo root on sys.path so ``from scoring.signals_envelope import ...``
# resolves under Lambda's task layout. Mirrors the existing shared-image
# handlers (thinktank, scanner, rationale_clustering, eval_rolling_mean).
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nousergon_lib.logging import monitor_handler, setup_logging

_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "signals_envelope",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    flow_name="research-signals-envelope",
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")
_DEFAULT_TARGET = "shadow"
_VALID_TARGETS = ("shadow", "production")

_init_done = False


def _ensure_init() -> None:
    """One-time cold-start hydration.

    Unlike ``thinktank_handler``, this producer reads NO secrets — it is
    pure-quant (no LLM/RAG calls, per ``scoring/signals_envelope.py``'s
    module doc), so there is nothing to hydrate from SSM. Kept for
    structural parity with the other shared-image handlers (init-phase
    10s ceiling discipline) and for the ``XDG_CACHE_HOME`` fix some
    downstream pandas/Arctic paths rely on.
    """
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    _init_done = True


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
    """Build + write the no-agent signals envelope. Raises on failure (see
    module doc's RAISE contract)."""
    # Wall-clock start for the research health stamp's duration_seconds
    # (config-I6053). Taken before the dry-path return so the field measures
    # the same span the retired runner's stamp measured — the whole handler.
    _health_start = time.time()
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.now(UTC)

    from evals.lambda_dry import is_dry

    # Shell-run dry path — boot + imports above already exercised the
    # bootstrap smoke. Return BEFORE any S3 access. This producer makes no
    # LLM calls at all; the ``dry_run_llm`` flag name is kept identical to
    # the other shared-image handlers' shell-run dry contract so the
    # Friday-PM SF keystone can treat every state uniformly.
    if is_dry(event):
        logger.info(
            "[signals_envelope_handler] dry_run_llm=True: shell-run no-op "
            "(no S3 read/write)",
        )
        return {"status": "OK", "dry_run": True}

    if not isinstance(event, dict) or not event.get("run_date"):
        raise ValueError(
            "signals_envelope_handler: event missing required 'run_date' "
            "field (ISO YYYY-MM-DD). RAISES rather than returning an "
            "ERROR dict — see module doc's RAISE-on-failure contract."
        )
    run_date = event["run_date"]
    if not isinstance(run_date, str) or len(run_date) < 10:
        raise ValueError(
            f"signals_envelope_handler: invalid run_date {run_date!r} — "
            "expected ISO YYYY-MM-DD."
        )

    # ── Trading-day normalization (DATE_CONVENTIONS, config-I6653) ───────────
    # Every artifact this handler writes keys by the TRADING DAY, not the
    # calendar date. The weekly SF's InitializeInput derives run_date from
    # date($$.Execution.StartTime) — a CALENDAR date — so on a Saturday cycle
    # it threads e.g. 2026-08-08 while the trading day is Friday 2026-08-07.
    #
    # Until this normalization the handler used that value verbatim, and the
    # Scanner in the SAME execution normalized: on 2026-08-08 one run wrote
    # candidates/2026-08-07/ and universe_membership/2026-08-07/ at 03:11-03:13
    # while this handler wrote signals/2026-08-08/ and
    # scanner/universe/trajectory/2026-08-08/ at 03:13-03:15. research_signals
    # is a severity:critical freshness row whose key template is
    # {trading_day}, so the probe read the stale Friday file and PASSED.
    #
    # resolve_trading_day is the shared chokepoint (nousergon_lib.dates) — the
    # same normalizer the backtester and evaluator adopted, and whose docstring
    # names this exact class. Adopted here rather than copying the Scanner's
    # inline block, per shared-code-policy: this is the fourth call site.
    # Idempotent, so a weekday run (calendar == trading day) is a no-op.
    from nousergon_lib.dates import resolve_trading_day

    calendar_date = run_date[:10]
    # alpha-engine-config-I8155: `calendar_date` is the SF execution's own
    # run_date, un-normalized, and is never reassigned below — that is what
    # stage_coverage groups a run's verdicts by (the ONLY key one execution's
    # verdicts share). Alias it explicitly so the intent survives a future
    # edit near this normalization block.
    execution_run_date = calendar_date
    run_date = resolve_trading_day(calendar_date)
    if run_date != calendar_date:
        logger.info(
            "[signals_envelope_handler] normalized run_date %s (calendar) → %s "
            "(trading day) per DATE_CONVENTIONS — every key this handler writes "
            "must share the Scanner's trading-day axis (config-I6653)",
            calendar_date,
            run_date,
        )

    target = event.get("target", _DEFAULT_TARGET)
    if target not in _VALID_TARGETS:
        raise ValueError(
            f"signals_envelope_handler: invalid target {target!r} — must "
            f"be one of {_VALID_TARGETS!r}."
        )

    bucket = event.get("bucket", _DEFAULT_BUCKET)

    # config-I2916: the weekly SF threads ``preflight.$: $.research_dry`` (true
    # ONLY on the Friday-PM shell run). It is DISTINCT from ``dry_run_llm``
    # above: dry_run_llm short-circuits before any S3 access, whereas preflight
    # keeps the full read/build/write path LIVE (bootstrap/transport smoke is
    # the preflight's whole point) and only downgrades the universe-board
    # fallback-staleness guard from a hard raise to a WARN — because the dry
    # Scanner leaves this cycle's dated board intentionally absent, so the
    # ~5-trading-day-stale prior-Saturday fallback is EXPECTED on Fridays, not
    # a real scanner miss. On the real Saturday run research_dry=false, so
    # preflight=false and the I2880 guard stays fully in force.
    preflight = bool(event.get("preflight", False))

    _ensure_init()

    from scoring.signals_envelope import (
        build_signals_envelope,
        read_regime_substrate,
        read_universe_board,
        write_envelope,
    )

    logger.info(
        "[signals_envelope_handler] start run_date=%s target=%s bucket=%s",
        run_date, target, bucket,
    )

    import boto3

    s3 = boto3.client("s3")

    # Board missing = hard precondition failure. read_universe_board
    # already raises RuntimeError (no-silent-fails) — propagates uncaught,
    # never converted to an ERROR dict here.
    board = read_universe_board(
        bucket, run_date=run_date, s3_client=s3, preflight=preflight,
    )

    # Substrate missing/unreadable = the ONE documented fail-soft exception
    # in this module: returns None + WARN, market_regime defaults to
    # "neutral" downstream in build_signals_envelope(). Never raises, never
    # re-wrapped as an error here.
    substrate = read_regime_substrate(bucket, s3_client=s3)

    # build_signals_envelope raises ValueError on an empty universe board
    # (no-silent-fails) and ContractViolation if the assembled envelope
    # violates the shared signals v1 JSON Schema (nousergon_lib.contracts)
    # — both propagate uncaught, per the RAISE contract above.
    envelope = build_signals_envelope(run_date, board, substrate)

    # Dual-track per DATE_CONVENTIONS: the artifact keys by trading day, and
    # ALSO records the calendar date of the cycle that produced it. Without
    # this, normalization silently discards which run wrote the file — two
    # cycles a day apart become indistinguishable inside the artifact
    # (config-I6653). Only stamped when the two differ, so weekday artifacts
    # are byte-identical to before this change.
    if calendar_date != run_date:
        envelope["calendar_date"] = calendar_date

    # target is forwarded verbatim — production intent is always explicit,
    # this handler never infers or defaults it silently past the event's
    # own "target" field (defaulted to "shadow" above if absent).
    dated_key, latest_key = write_envelope(
        envelope, run_date, target=target, bucket=bucket, s3_client=s3,
    )

    # ── Ported secondary artifacts (config-I3290) ─────────────────────────
    # research_consolidated_morning + scanner_universe_trajectory used to be
    # written from inside the old multi-agent archive_writer node
    # (the retired research graph), which config#2515 removed from the weekly
    # SF entirely. Both artifacts still have live consumers (dashboard
    # Research Briefing Archive / Attractiveness Trends views, the
    # backtester's attractiveness_eval IC grading) so they are ported here
    # as post-steps, gated to a real production run (never on a shadow/test
    # cycle) and fail-soft — a secondary-artifact failure must never sink
    # the primary signals.json deliverable just written above.
    if target == "production":
        from scoring.morning_brief import build_morning_brief_markdown, write_morning_brief

        try:
            brief_md = build_morning_brief_markdown(envelope)
            write_morning_brief(run_date, brief_md, bucket=bucket, s3_client=s3)
        except Exception as e:  # noqa: BLE001 — secondary observability, never fatal
            logger.warning(
                "[signals_envelope_handler] morning brief write FAILED "
                "(non-fatal — signals.json unaffected): %s", e,
            )
            from observe_alerts import publish_observe_alert
            publish_observe_alert(
                f"morning brief write FAILED for {run_date} (non-fatal, "
                f"signals.json already persisted): {e}",
                source="signals_envelope_handler:morning_brief",
                dedup_key=f"morning_brief_write_fail:{run_date}",
            )

        from scoring.attractiveness_trajectory import compute_and_write_trajectory

        try:
            compute_and_write_trajectory(run_date, bucket=bucket, s3_client=s3)
        except Exception as e:  # noqa: BLE001 — secondary observability, never fatal
            logger.warning(
                "[signals_envelope_handler] attractiveness trajectory write "
                "FAILED (non-fatal — signals.json unaffected): %s", e,
            )
            from observe_alerts import publish_observe_alert
            publish_observe_alert(
                f"attractiveness trajectory write FAILED for {run_date} "
                f"(non-fatal, signals.json already persisted): {e}",
                source="signals_envelope_handler:trajectory",
                dedup_key=f"trajectory_write_fail:{run_date}",
            )

        # ── Research module health stamp (config-I6053 / config-I6344) ────
        # REPOINTED WRITER, not a new artifact. `health/research.json` had
        # exactly one writer — nousergon_lib.health.write_health(
        # module_name="research") inside crucible-research
        # `lambda/handler.py::handler()`, downstream of the multi-agent
        # Research graph that nousergon-data#814 (config-I2515) removed from
        # the weekly SF on 2026-07-14. No SF state invokes that path any more
        # (`alpha-engine-research-runner:live` is invoked only with
        # `mode="challengers_only"` by ChallengerShadow, and by the deploy
        # canary with `dry_run_llm=true`), so the stamp froze at
        # 2026-07-21T12:49Z carrying `{"status": "failed", "error": "[Errno 28]
        # No space left on device"}` — the last real full_run, which failed.
        #
        # That is ARCHITECTURE §128's exact shape one layer up: retiring a
        # producer leaves every consumer's read path working, so the dashboard
        # health checker kept succeeding on a file that had stopped changing.
        # It is ALSO now load-bearing on the weekly SF's terminal status:
        # config-I6891 (shipped 2026-08-12) routes any degraded run through
        # CheckDegradedOutcome -> WriteCompletionMarkerDegraded -> DegradedRun,
        # a Fail state, and `SaturdayHealthCheck` exits non-zero on any stale
        # entry — so a dead health stamp now FAILS the whole weekly run.
        #
        # The fix is the repoint the ARTIFACT_REGISTRY row's own comment names
        # as the correct option (config-I6344): the champion scanner ->
        # signals-envelope -> predictor path owns research-module health now.
        # This handler is the right writer of the three because the stamp's
        # declared required deliverable IS `signals`, and this handler is the
        # sole producer of `signals/{date}/signals.json` + `signals/latest.json`
        # since the cutover. Deleting the row instead was rejected: the four
        # health_* rows are a lockstep contract (nousergon_lib.health.
        # REGISTRY_HEALTH_ARTIFACTS + validate_artifact_registry.py +
        # the dashboard alignment tests), and research is DECOMPOSED, not
        # retired — a live capability whose freshness detector we would be
        # deleting rather than repairing (principles.md §2.7).
        #
        # Gated to a real production cycle: `target == "production"` (the
        # enclosing branch) AND `not preflight` — the Friday-PM shell run
        # threads preflight=true and must never stamp research health fresh
        # off a transport smoke, which is the false-green this stamp exists
        # to make impossible.
        #
        # Fail-soft with an alert, mirroring the two secondary artifacts
        # above: signals.json is already persisted, so sinking the run on an
        # observability write would trade the primary deliverable for the
        # stamp. Never silent — the alert is the recording surface.
        if not preflight:
            try:
                from nousergon_lib.health import Deliverable, write_health

                write_health(
                    module_name="research",
                    deliverables=[
                        Deliverable(name="signals", required=True, produced=True),
                    ],
                    run_date=run_date,
                    duration_seconds=time.time() - _health_start,
                    summary={
                        "producer": "signals_envelope_handler",
                        "universe_count": len(envelope["universe"]),
                        "market_regime": envelope["market_regime"],
                        "dated_key": dated_key,
                    },
                    bucket=bucket,
                    s3_client=s3,
                )
            except Exception as e:  # noqa: BLE001 — secondary observability, never fatal
                logger.warning(
                    "[signals_envelope_handler] research health stamp write "
                    "FAILED (non-fatal — signals.json unaffected): %s", e,
                )
                from observe_alerts import publish_observe_alert
                publish_observe_alert(
                    f"research health stamp write FAILED for {run_date} "
                    f"(non-fatal, signals.json already persisted). "
                    f"health/research.json will read stale to "
                    f"SaturdayHealthCheck: {e}",
                    source="signals_envelope_handler:health",
                    dedup_key=f"research_health_write_fail:{run_date}",
                )

            # ── Flow-doctor end-of-run heartbeat (config#646) ─────────────
            # REPOINTED WRITER, exactly like the health stamp above.
            # `_flow_doctor/heartbeat/research/{date}.json` had one producer —
            # `lambda/handler.py::_emit_flow_doctor_heartbeat` — deleted with the
            # research graph in crucible-research-PR685. Measured 2026-08-20:
            # `_flow_doctor/heartbeat/` holds backtester, data-collector, executor
            # and predictor-inference, and NO research row has ever existed there.
            #
            # The console Flow-Doctor pane discovers flows from what is present in
            # S3 (crucible-dashboard `list_flow_doctor_heartbeat_flows`), so the
            # research producer does not render as broken — it does not render at
            # all. That is `principles.md` §2.7's UNREPORTED state: four of the
            # five producers report and the fifth is silently absent from the list
            # nobody counts.
            #
            # Gated identically to the health stamp — production target, not
            # preflight — because a heartbeat off a Friday transport smoke is the
            # same false-green the stamp's gate exists to prevent. `emit_heartbeat`
            # soft-fails internally and the `hasattr` guard keeps a version-skewed
            # lib pin from AttributeError-ing at end-of-run (the lib deploys
            # independently of this image), mirroring the predictor's call site.
            try:
                from nousergon_lib.logging import get_flow_doctor

                fd = get_flow_doctor()
                if fd and hasattr(fd, "emit_heartbeat"):
                    fd.emit_heartbeat(bucket=bucket)
            except Exception as e:  # noqa: BLE001 — secondary observability, never fatal
                logger.warning(
                    "[signals_envelope_handler] flow-doctor heartbeat write "
                    "FAILED (non-fatal — signals.json unaffected): %s", e,
                )
                from observe_alerts import publish_observe_alert
                publish_observe_alert(
                    f"flow-doctor heartbeat write FAILED for {run_date} "
                    f"(non-fatal, signals.json already persisted). The console "
                    f"Flow-Doctor pane will not list the research producer: {e}",
                    source="signals_envelope_handler:flow_doctor_heartbeat",
                    dedup_key=f"flow_doctor_heartbeat_write_fail:{run_date}",
                )

    logger.info(
        "[signals_envelope_handler] done run_date=%s target=%s dated_key=%s "
        "universe=%d market_regime=%s",
        run_date, target, dated_key,
        len(envelope["universe"]), envelope["market_regime"],
    )
    result = {
        "status": "OK",
        "dated_key": dated_key,
        "latest_key": latest_key,
        "universe_count": len(envelope["universe"]),
        "market_regime": envelope["market_regime"],
        "target": target,
    }

    # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    # §2.3a rescope): the assertion lives in the stage's own handler,
    # immediately before it returns, rather than a separate end-of-run SF
    # state. OBSERVE MODE ONLY — never enables enforcement, never raises.
    if not execution_run_date:
        # alpha-engine-config-I8155: never fabricate a date to satisfy the
        # (now-required) run_date argument. The required-field guard above
        # makes this unreachable today, but the check stays as the contract
        # boundary rather than relying on that upstream guard.
        logger.error(
            "[signals_envelope_handler] stage-coverage assertion SKIPPED "
            "for SignalsEnvelope: no execution run_date on this event "
            "(execution identity absent)"
        )
        result["stage_coverage"] = {
            "stage": "SignalsEnvelope",
            "status": "UNMEASURED",
            "reason": "execution run_date absent from event",
        }
    else:
        try:
            from krepis.stage_coverage import assert_stage_coverage

            result["stage_coverage"] = assert_stage_coverage(
                "SignalsEnvelope",
                run_date=execution_run_date,
                window_start=_started,
            )
        except ImportError as exc:
            # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
            # the handler's own outcome is unchanged (config-I7214).
            logger.error("stage-coverage assertion unavailable: %s", exc)
            result["stage_coverage"] = {
                "stage": "SignalsEnvelope",
                "status": "UNMEASURED",
                "reason": f"assertion unavailable: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 — never let the observer kill the stage it observes
            # alpha-engine-config-I8155: the krepis landing this arc makes
            # run_date a required, contract-enforced kwarg (TypeError on
            # omission, StageCoverageContractError on blank/None). Both are
            # library-internal contract failures, not stage failures — log
            # loudly and degrade to UNMEASURED rather than raising out of
            # the handler.
            logger.error(
                "[signals_envelope_handler] stage-coverage assertion "
                "raised for SignalsEnvelope: %s: %s",
                type(exc).__name__, exc,
            )
            result["stage_coverage"] = {
                "stage": "SignalsEnvelope",
                "status": "UNMEASURED",
                "reason": f"assertion raised: {type(exc).__name__}: {exc}",
            }

    return result
