"""Lambda entry point — standalone scanner producing ``candidates.json``.

ROADMAP L1995 Phase 1 (plan doc:
``~/Development/alpha-engine-docs/private/scanner-rag-resequence-260524.md``).

Splits the quant scanner out of the Research Lambda into a dedicated
Saturday-SF state, writing ``s3://alpha-engine-research/candidates/
{run_date}/candidates.json`` so RAGIngestion (Phase 4) can ingest fresh
context for *every* stock the agents will evaluate — including the
~10-20 new picks each week the prior-signals.json-keyed RAG ingest
misses today.

Phase 1 posture: this Lambda exists + can be invoked (Phase 2 inserts
the SF state, gated default-off; operator flips the flag for Phase 3
soak). The Research Lambda still runs its own internal scanner — the
new artifact is written in parallel-observe mode, no consumer reads
it yet. Phase 5 will cut Research over to read this artifact + retire
the internal scanner.

Event shape (all fields optional except ``run_date``):

    {
      "run_date": "2026-05-30",          # ISO YYYY-MM-DD (required)
      "bucket": "alpha-engine-research", # default RESEARCH_BUCKET env
      "market_regime": "neutral",        # default "neutral"
      "dry_run_llm": false,              # shell-run dry path
      "mode": "scan",                    # "scan" (default) | "scanner_leaderboard"
    }

``mode`` (alpha-engine-config-I7813) selects WHICH work this invocation
does, so an observe-only board can be its own Step Functions leaf state
instead of riding inline on the scan:

``scan`` (default)
    The live path: ``candidates.json``, the challenger shadows, the
    universe-membership artifact, the funnel-cut leaderboard and the
    weekly cut-promotion decision. It does NOT build
    ``scanner/leaderboard/{date}.json`` any more — that board moved to
    the leaf below. The cut leaderboard and the promotion decision stay
    here because the promotion engine BRANCHES on the cut board and
    writes the live champion pointer: a control, not a report.

``scanner_leaderboard``
    Builds ``scanner/leaderboard/{date}.json`` and returns. Nothing else
    reads that board — the dashboard renders it and gate predicates poll
    it — so it is a report, and it runs as a post-Report-Card leaf state
    on the weekly SF (``ne-weekly-freshness-pipeline``) whose failure
    cannot reach any stage that does not consume it (sf-pipeline-policy
    §2.1). Unlike ``scan``, a build failure here RAISES: the board is
    this invocation's only deliverable, so a swallowed error would leave
    the SF task green with nothing written.

``date_str`` is only the OUTPUT key on the board producer — the cohort
dates come from the S3 walk over ``candidates_shadow/`` — so a Saturday
``run_date`` scores correctly.

Returns one of:

    {"status": "OK", "summary": {...}}                — artifact written
    {"status": "ERROR", "error": "<msg>"}             — hard failure caught

Cost contract mirrors data #295 + L3277 audit: the scanner Lambda
never returns SKIPPED today (constituents.json + feature store are
hard preconditions; their absence raises). The status allowlist
nonetheless includes SKIPPED for forward-compat if Phase 5 ever adds
legitimate no-op paths (e.g. holiday short cycles).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import UTC, datetime

# Repo root on sys.path so ``from data.scanner_orchestrator import ...``
# resolves under Lambda's task layout. Mirrors the existing handlers
# (rationale_clustering, eval_rolling_mean, aggregate_costs).
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
    "scanner",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
    flow_name="research-scanner",
)

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = os.environ.get("RESEARCH_BUCKET", "alpha-engine-research")

# alpha-engine-config-I7813. Closed set, and an unknown value is an ERROR
# rather than a silent fall-through to the live scan: a typo in a Step
# Functions Payload would otherwise run the FULL scan under a name that
# asked for a board, and the operator would read a green task.
_MODE_SCAN = "scan"
_MODE_SCANNER_LEADERBOARD = "scanner_leaderboard"
_MODES = (_MODE_SCAN, _MODE_SCANNER_LEADERBOARD)

_init_done = False


def _ensure_init() -> None:
    """Defer expensive init to first invocation. Mirrors the other
    shared-image handlers — Lambda init phase 10s ceiling."""
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


class ScannerLeaderboardBuildError(RuntimeError):
    """The ``scanner_leaderboard`` leaf's board did not get written.

    Raised so the Step Functions Task FAILS and its own ``Catch`` fires. In
    ``scan`` mode a board failure is swallowed to a status field because the
    live ``candidates.json`` is that invocation's deliverable and must never be
    downgraded by an observe-only scorer. In the leaf there is no other
    deliverable: swallowing it would leave a green SF task with nothing written
    to S3, which is the one shape sf-pipeline-policy.md §2.3 forbids outright.
    """


def _run_scanner_leaderboard(s3_client, bucket: str, run_date: str) -> dict:
    """``mode="scanner_leaderboard"`` — build ONLY ``scanner/leaderboard/{date}.json``.

    alpha-engine-config-I7813. The board is observe-only: the dashboard's
    Experiments view renders it and gate predicates poll its ``n_dates``, and no
    pipeline stage branches on any number in it. So it is a REPORT, and it runs
    as a leaf state of the weekly SF placed after the Report Card and Director,
    where its failure cannot reach a stage that does not consume its output
    (sf-pipeline-policy.md §2.1's blast-radius test).

    ``build_scanner_leaderboard`` never raises — it returns a status dict and
    publishes its own observe alert on the way. That contract is right for the
    inline caller and WRONG here, so this wrapper converts a non-terminal status
    into a raise. Statuses:

    ``ok``            — written, returns OK.
    ``unmeasurable``  — a decision, not a failure: the producer wrote the
                        artifact carrying its own reason (immature cohorts, an
                        empty closes panel). Returns OK; the reason is in the
                        summary and in the artifact.
    ``error``         — nothing was written. Raises.
    """
    logger.info(
        "[scanner_handler] mode=scanner_leaderboard run_date=%s bucket=%s",
        run_date,
        bucket,
    )
    from scoring.leaderboard_producers import build_scanner_leaderboard

    status = build_scanner_leaderboard(s3_client, bucket, run_date)
    state = (status or {}).get("status")
    logger.info(
        "[scanner_handler] scanner leaderboard status=%s key=%s",
        state,
        (status or {}).get("key"),
    )
    if state not in ("ok", "unmeasurable"):
        raise ScannerLeaderboardBuildError(
            f"scanner leaderboard build did not write scanner/leaderboard/{run_date}.json: "
            f"status={state!r} error={(status or {}).get('error')!r}"
        )
    return {
        "status": "OK",
        "mode": _MODE_SCANNER_LEADERBOARD,
        "summary": {
            "leaderboard": {
                "status": state,
                "key": (status or {}).get("key"),
                "reason": (status or {}).get("reason"),
            },
        },
    }


def _run(event, context):
    """Produce the candidates.json artifact for ``event['run_date']``."""
    # Captured at entry, before any work — an artifact older than this is a
    # leftover from a previous cycle, not this run's output (config-I7214).
    _started = datetime.now(UTC)
    _ensure_init()

    import boto3

    from data.scanner_orchestrator import (
        ScannerOrchestratorError,
        build_candidates_artifact,
        build_shadow_candidate_artifacts,
        write_candidates_artifact,
        write_shadow_candidates_artifact,
        write_shadow_status_record,
        write_universe_board_for_scanner_run,
    )
    from data.scanner_specs import build_shadow_status_record, challenger_specs
    from evals.lambda_dry import is_dry

    # Shell-run dry path — boot + imports above already exercised the
    # bootstrap smoke. Return BEFORE the orchestrator (which reads
    # constituents + feature store + writes S3). dry_run_llm short-
    # circuits everything for Friday-Preflight shell runs.
    if is_dry(event):
        logger.info(
            "[scanner_handler] dry_run_llm=True: shell-run no-op (no S3 read/write, no scanner pass)",
        )
        return {"status": "OK", "dry_run": True}

    run_date = event.get("run_date")
    if not run_date:
        logger.error("[scanner_handler] event missing required 'run_date' field")
        return {
            "status": "ERROR",
            "error": "event missing required 'run_date' field (ISO YYYY-MM-DD)",
        }

    # Lenient parse — accept anything the orchestrator's S3 key
    # template accepts (the orchestrator doesn't itself parse run_date
    # as a date, just slots it into the partition prefix).
    if not isinstance(run_date, str) or len(run_date) < 10:
        logger.error(
            "[scanner_handler] invalid run_date %r — expected ISO YYYY-MM-DD",
            run_date,
        )
        return {
            "status": "ERROR",
            "error": f"invalid run_date {run_date!r}: expected ISO YYYY-MM-DD",
        }

    # ── Trading-day normalization (DATE_CONVENTIONS) ─────────────────────────
    # Every trade artifact in the system keys by the TRADING DAY, not the
    # calendar date: signals.json, sector_team_runs, scanner_evaluations, and
    # the Research run itself all key off most_recent_trading_day(today). The
    # Saturday SF passes a CALENDAR run_date (date(Execution.StartTime)) — e.g.
    # 2026-05-30 (Sat) — while Research keys off Friday 2026-05-29. candidates.json
    # MUST land on the same trading-day key or the Phase-5 consumer (Research
    # fetch_data → load_candidates_json) can't find it. The 2026-05-30 L4464
    # recovery failed exactly here: Scanner wrote candidates/2026-05-30/, Research
    # read candidates/2026-05-29/. Normalize at the producer to the canonical
    # trading-day axis (lib chokepoint), preserving on-or-before semantics so an
    # explicit operator backfill date is normalized too.
    #
    # resolve_trading_day is the shared chokepoint (nousergon_lib.dates) — the
    # same normalizer the backtester, evaluator and signals_envelope_handler
    # resolve through (config-I6667). It is defensive where the inline block it
    # replaces raised out of date.fromisoformat on a malformed value; the
    # `len(run_date) < 10` guard above still precedes this call, so a malformed
    # run_date is rejected with an ERROR return before normalization is reached
    # and the softened contract changes nothing reachable here.
    from nousergon_lib.dates import resolve_trading_day

    _trading_day = resolve_trading_day(run_date[:10])
    if _trading_day != run_date[:10]:
        logger.info(
            "[scanner_handler] normalized run_date %s (calendar) → %s (trading "
            "day) per DATE_CONVENTIONS — candidates.json keys by trading day to "
            "match Research + signals.json",
            run_date,
            _trading_day,
        )
    run_date = _trading_day

    bucket = event.get("bucket", _DEFAULT_BUCKET)
    market_regime = event.get("market_regime", "neutral")

    logger.info(
        "[scanner_handler] start run_date=%s (trading day) bucket=%s market_regime=%s",
        run_date,
        bucket,
        market_regime,
    )

    s3_client = boto3.client("s3")

    # ── Mode dispatch (alpha-engine-config-I7813) ────────────────────────────
    # Placed AFTER run_date normalization + bucket resolution so the leaf mode
    # keys its output on exactly the same trading day the scan would have.
    mode = event.get("mode", _MODE_SCAN)
    if mode not in _MODES:
        logger.error("[scanner_handler] unknown mode %r — expected one of %s", mode, _MODES)
        return {
            "status": "ERROR",
            "error": f"unknown mode {mode!r}: expected one of {_MODES}",
        }
    if mode == _MODE_SCANNER_LEADERBOARD:
        return _run_scanner_leaderboard(s3_client, bucket, run_date)

    try:
        artifact = build_candidates_artifact(
            run_date=run_date,
            s3_client=s3_client,
            bucket=bucket,
            market_regime=market_regime,
        )
    except ScannerOrchestratorError as exc:
        # Hard precondition failure (constituents missing, feature store
        # empty). The orchestrator already logged the cause; surface as
        # ERROR so the SF Catch handles it.
        logger.error("[scanner_handler] orchestrator precondition failed: %s", exc)
        return {"status": "ERROR", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("[scanner_handler] orchestrator failed hard")
        return {"status": "ERROR", "error": str(exc)}

    try:
        s3_key = write_candidates_artifact(
            artifact,
            s3_client=s3_client,
            bucket=bucket,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[scanner_handler] S3 write failed hard")
        return {"status": "ERROR", "error": f"S3 write failed: {exc}"}

    # ── Champion/challenger OBSERVE shadows (config#1221) ────────────────────
    # Best-effort secondary path: emit challenger candidate-gen specs to the
    # isolated candidates_shadow/ prefix for forward scoring. WHOLLY fail-soft —
    # the live candidates.json above is the primary deliverable and is already
    # written; a shadow failure is recorded (WARN + ``shadows`` summary field)
    # but NEVER downgrades the OK status (no-silent-fails: the recording surface
    # is the WARN log + the response field).
    shadows: dict[str, str] = {}
    shadow_error: str | None = None
    shadow_build_errors: dict[str, str] = {}
    try:
        shadow_artifacts, shadow_build_errors = build_shadow_candidate_artifacts(artifact)
        for spec_name, shadow_artifact in shadow_artifacts.items():
            try:
                shadows[spec_name] = write_shadow_candidates_artifact(
                    shadow_artifact,
                    spec_name,
                    s3_client=s3_client,
                    bucket=bucket,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[scanner_handler] shadow write failed for spec %s (non-fatal, live unaffected): %s",
                    spec_name,
                    exc,
                )
                shadow_error = f"{spec_name}: {exc}"
                # The BUILD succeeded but the persist failed — the spec still
                # produced no durable candidates_shadow/ artifact this cycle,
                # so the status record below must record it as a miss too
                # (never assert an action that never happened).
                shadow_build_errors[spec_name] = str(exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[scanner_handler] shadow candidate-gen build failed (non-fatal, live unaffected): %s",
            exc,
        )
        shadow_error = str(exc)
        shadow_build_errors = {spec.name: str(exc) for spec in challenger_specs()}

    # ── Explicit per-spec, per-cycle MISS record (config#6428) ───────────────
    # champion-challenger-policy.md §3: "A cycle where an arm produces no
    # output is recorded as a miss, not omitted. Silent absence and a genuine
    # zero must never render identically." Every registered challenger spec
    # gets a scanner_shadow_status.v1 record EVERY cycle — mirroring
    # producers/experiment_record.py's experiment_record.v1 vocabulary — built
    # AFTER the write attempt above so `status` reflects the TRUE final
    # outcome. ADDITIVE to the WARN + observe-alert path already inside
    # build_shadow_artifacts (never a replacement) and wholly fail-soft: a
    # status-record write failure here is logged and never downgrades the OK
    # response.
    for spec in challenger_specs():
        record = build_shadow_status_record(
            spec,
            run_date,
            shadow_candidates_key=shadows.get(spec.name),
            error=shadow_build_errors.get(spec.name),
        )
        try:
            write_shadow_status_record(
                record,
                spec.name,
                run_date,
                s3_client=s3_client,
                bucket=bucket,
            )
        except Exception as exc:  # noqa: BLE001 — status recording is best-effort
            logger.warning(
                "[scanner_handler] shadow status-record write failed for spec %s (non-fatal): %s",
                spec.name,
                exc,
            )

    # ── Universe scoreboard (alpha-engine-config-I2515) ──────────────────────
    # Standalone Scanner path becomes a universe-board producer, completing
    # ROADMAP L1995 Phase 5's producer side (the Research graph's
    # archive_writer has been the SOLE producer of scanner/universe/ until
    # now). DUAL-WRITE TRANSITION STATE: the Research graph KEEPS writing
    # this board until the SF cutover retires its internal scanner (S3
    # contract safety — both producers coexist); a same-day overwrite by
    # whichever producer runs last is expected + idempotent-ish, though the
    # two producers' rows can differ on agent-audit fields (focus_*/
    # agent_override) since only the Research graph has a real agent run
    # backing those. See alpha-engine-config-I2515 + write_universe_board_
    # for_scanner_run's docstring for the factor-profiles ordering
    # resolution. WHOLLY fail-soft — the live candidates.json above is the
    # primary deliverable and is already written; a board failure is
    # recorded (WARN + response field) but NEVER downgrades the OK status
    # (no-silent-fails: the recording surface is the WARN log + this field).
    universe_board_key: str | None = None
    universe_board_error: str | None = None
    try:
        universe_board_key = write_universe_board_for_scanner_run(
            artifact,
            market_regime=market_regime,
            s3_client=s3_client,
            bucket=bucket,
        )
        logger.info(
            "[scanner_handler] universe scoreboard written → %s",
            universe_board_key,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[scanner_handler] universe scoreboard write failed (non-fatal, "
            "dashboard visibility only — candidates.json unaffected): %s",
            exc,
        )
        universe_board_error = str(exc)

    # ── Universe membership artifact (alpha-engine-config-I4818) ─────────────
    # LOAD-BEARING, and therefore deliberately NOT wrapped in a fail-soft
    # except like every other post-candidates write in this handler: the
    # predictor resolves its DAILY scoring universe from this artifact. A
    # missing membership file does not degrade a dashboard panel — it leaves
    # the predictor scoring whatever stale membership it last saw, which is
    # exactly the defect this artifact was built to end (the predictor spent
    # three weekly cycles on a frozen 2026-07-10 population before anyone
    # noticed, because its producer's disappearance was silent).
    #
    # Placement: AFTER candidates.json is written (it reads scanner_tickers
    # from the in-memory artifact) and after the board block (which produces
    # the factor profiles this read needs — see write_universe_board_for_
    # scanner_run's factor-profiles ordering note). A board failure upstream
    # therefore surfaces HERE as a loud membership failure rather than as a
    # silent stale universe tomorrow morning, which is the correct severity.
    # The failure is surfaced as an ERROR *response* rather than a raised
    # exception purely for consistency with this handler's other hard-failure
    # paths (orchestrator precondition, candidates S3 write) — the SF treats
    # both identically, and one convention per handler is worth more than the
    # marginal difference.
    from scoring.universe_membership import compute_and_write_universe_membership

    try:
        membership_key = compute_and_write_universe_membership(
            run_date,
            artifact["scanner_tickers"],
            # Carries the incumbent's own ``tech_score`` ranking (I4983) so the
            # membership artifact records the champion and incumbent arms at
            # equal width. Already in the in-memory artifact — no extra read.
            scanner_eval_log=artifact.get("scanner_eval_log"),
            bucket=bucket,
            s3_client=s3_client,
        )
    except Exception as exc:  # noqa: BLE001 — re-surfaced as ERROR, never swallowed
        logger.exception("[scanner_handler] universe membership write failed hard")
        return {"status": "ERROR", "error": f"universe membership failed: {exc}"}
    logger.info(
        "[scanner_handler] universe membership written → %s",
        membership_key,
    )

    # ── Funnel-cut leaderboard (alpha-engine-config-I7584) ───────────────────
    # Scores attractiveness_top_60, attractiveness_top_20 and the gate baseline
    # against the population each narrowed. Stays in THIS invocation — while the
    # sibling scanner leaderboard left for its own SF leaf (I7813) — because the
    # cut-promotion block below BRANCHES on this board and writes the live
    # champion pointer from it. That makes it a control input, not a report, and
    # the issue's own discriminator keeps controls where their consumer is.
    #
    # (alpha-engine-config-I7813) The panel-cache rationale this comment used to
    # carry — "the scanner leaderboard read the ~904-symbol closes panel first,
    # so this build reuses it" — was MEASURED FALSE on 2026-08-20 and is not
    # what keeps this block here. `_PANEL_CACHE` keys on the cohort entry-date
    # set and holds exactly ONE entry (it `.clear()`s before inserting). The two
    # builds walk different prefixes: this one `universe_membership/` (23 dates
    # live), the scanner board `candidates_shadow/` (21 dates, a different set).
    # The keys never matched, so the second build has always paid its own full
    # panel read — splitting the boards across invocations adds no read at all.
    # (alpha-engine-config-I7841 D1/D3) Sentinel + start-log discipline: this is
    # the block a 450s Lambda timeout landed inside on 2026-08-20 —
    # this line never printed, and no `except` ran. The freshness detector on
    # `research/cuts_leaderboard/{trading_day}.json` (I7841 D1, coordinated via
    # alpha-engine-config-PR7820 / I7833 — see PR body) is what catches an
    # invocation that dies before even this log line; this block's own
    # attempted/completed reporting covers the narrower case where the
    # invocation returns but this call raised or degraded.
    cuts_leaderboard_status: dict = {"status": "not_attempted"}
    logger.info("[scanner_handler] attempting cuts leaderboard run_date=%s", run_date)
    try:
        from scoring.leaderboard_producers import build_cuts_leaderboard

        cuts_leaderboard_status = build_cuts_leaderboard(s3_client, bucket, run_date)
        logger.info(
            "[scanner_handler] cuts leaderboard status=%s key=%s",
            cuts_leaderboard_status.get("status"),
            cuts_leaderboard_status.get("key"),
        )
    except Exception as exc:  # noqa: BLE001 — observe-only, live unaffected
        logger.warning(
            "[scanner_handler] cuts leaderboard build failed (non-fatal, live unaffected): %s",
            exc,
        )
        cuts_leaderboard_status = {"status": "error", "error": str(exc)}

    # ── Weekly cut-promotion decision (alpha-engine-config-I7826) ───────────
    # Decides which of the two count-matched 60s holds the sector-team feed and
    # WRITES that decision, promote or hold, to config/scanner_cut_champion.json
    # plus an immutable dated audit record. Runs here because this is the moment
    # the board it reads exists — and the freshly built leaderboard is handed in
    # directly rather than re-fetched, so the decision reads the exact artifact
    # this run produced.
    #
    # CADENCE-AGNOSTIC on purpose. Its hysteresis is measured in CALENDAR days
    # (cooldown_days), not in invocations, so it behaves identically whether the
    # scanner runs every weekday (today) or weekly (after nousergon-data#1464) —
    # nothing here waits on that track. A re-evaluation that reaches the same
    # conclusion rewrites the same hold and raises the liveness resolution of the
    # audit record; it can never flip the feed twice inside one cooldown.
    #
    # FAIL-SOFT + LOUD, matching the two blocks above: the live candidates.json
    # and the membership artifact are already written and must never be
    # downgraded by a decision step, and a promotion failure is SAFE — the
    # pointer keeps naming the standing champion. It is never SILENT: the engine
    # raises on a defective board, and that raise lands here as an ERROR log, an
    # ops alert and an explicit status in the summary.
    promotion_status: dict = {"status": "not_attempted"}
    logger.info("[scanner_handler] attempting cut promotion run_date=%s", run_date)
    try:
        from scoring.cut_promotion import run_cut_promotion

        promotion_status = run_cut_promotion(
            run_date,
            bucket=bucket,
            s3_client=s3_client,
            leaderboard=(cuts_leaderboard_status or {}).get("leaderboard"),
        )
        logger.info(
            "[scanner_handler] cut promotion decision=%s champion=%s reason_code=%s",
            promotion_status.get("decision"),
            promotion_status.get("champion"),
            promotion_status.get("reason_code"),
        )
    except Exception as exc:  # noqa: BLE001 — live path already delivered; never silent
        logger.exception(
            "[scanner_handler] cut promotion FAILED on %s — the champion pointer "
            "keeps naming the standing champion", run_date,
        )
        promotion_status = {"status": "error", "error": str(exc)}
        try:
            from observe_alerts import publish_observe_alert

            publish_observe_alert(
                message=(
                    f"[cut_promotion] the scanner-cut promotion engine FAILED on "
                    f"{run_date}: {exc}. The feed still resolves from the standing "
                    "champion, so this is not a live outage — but no decision was "
                    "recorded for this cycle (alpha-engine-config-I7826)."
                ),
                source="research:cut_promotion",
                dedup_key=f"cut_promotion_error:{run_date}",
                severity="ERROR",
            )
        except Exception:  # noqa: BLE001 — alerting is secondary, never fatal
            logger.warning("[scanner_handler] cut promotion alert publish failed")

    summary = {
        "s3_key": s3_key,
        "scanner_tickers": len(artifact["scanner_tickers"]),
        "population_tickers": len(artifact["population_tickers"]),
        "agent_input_set": len(artifact["agent_input_set"]),
        "new_vs_prior_cycle": len(artifact["stats"]["new_vs_prior_cycle"]),
        "dropped_vs_prior_cycle": len(artifact["stats"]["dropped_vs_prior_cycle"]),
        "baseline_missing": artifact["stats"]["baseline_missing"],
        "universe_membership": membership_key,
        "shadows": shadows,
        "cuts_leaderboard": {
            "status": cuts_leaderboard_status.get("status"),
            "key": cuts_leaderboard_status.get("key"),
        },
        "cut_promotion": {
            "status": promotion_status.get("status", "ok"),
            "decision": promotion_status.get("decision"),
            "champion": promotion_status.get("champion"),
            "reason_code": promotion_status.get("reason_code"),
            "error": promotion_status.get("error"),
        },
        "universe_board": {
            "status": "OK" if universe_board_key else "error",
            "key": universe_board_key,
        },
    }
    if shadow_error:
        summary["shadow_error"] = shadow_error
    if universe_board_error:
        summary["universe_board_error"] = universe_board_error

    # ── Post-membership board legibility rollup (alpha-engine-config-I7841 D3) ──
    # The two post-membership stages above (cuts leaderboard, cut promotion —
    # the scanner leaderboard left for its own SF leaf state, I7813) run
    # sequentially in one invocation and each
    # already carries its own nested status. This collapses them into one
    # explicit attempted/completed view so a PARTIAL invocation — one stage's
    # try/except caught a real error rather than the invocation dying outright
    # — is legible straight from the Step Functions execution history's output
    # field, without opening CloudWatch. "attempted" is driven by the
    # `not_attempted` sentinel each block is seeded with before its own try:
    # today every block always runs once membership succeeds, so this is
    # false only if a future refactor makes one of them conditional — exactly
    # the case this rollup exists to keep visible. It does NOT cover a Lambda
    # TIMEOUT: that freezes the invocation before it ever returns, so no
    # summary — this or any other shape — reaches the SF execution history.
    # That failure mode is covered externally by the
    # `research/cuts_leaderboard/{trading_day}.json` freshness detector
    # (I7841 D1), keyed on the scanner having run rather than the calendar.
    _board_stages = {
        "cuts_leaderboard": cuts_leaderboard_status.get("status"),
        "cut_promotion": promotion_status.get("status", "ok"),
    }
    summary["boards"] = {
        "attempted": [name for name, status in _board_stages.items() if status != "not_attempted"],
        "completed": [name for name, status in _board_stages.items() if status in ("ok", "unmeasurable")],
        "not_attempted": [name for name, status in _board_stages.items() if status == "not_attempted"],
        "errored": [name for name, status in _board_stages.items() if status == "error"],
    }

    logger.info(
        "[scanner_handler] done run_date=%s scanner_tickers=%d population=%d new=%d dropped=%d",
        run_date,
        summary["scanner_tickers"],
        summary["population_tickers"],
        summary["new_vs_prior_cycle"],
        summary["dropped_vs_prior_cycle"],
    )
    # Dedicated metric marker line (config#785). The human-readable summary
    # line above embeds the count inside a ``scanner_tickers=<n>`` token,
    # which a CloudWatch Logs metric filter cannot split on ``=`` to extract
    # the integer. This line emits the count as a standalone whitespace-
    # delimited numeric token so the space-delimited filter pattern in
    # infrastructure/setup_scanner_alarm.sh can bind it as the metricValue.
    # Format is locked by tests/test_scanner_metric_marker.py.
    logger.info(
        "[scanner_handler] METRIC scanner_tickers_count %d",
        summary["scanner_tickers"],
    )

    # Stage-coverage self-assertion (config-I7214, sf-pipeline-policy.md
    # §2.3a rescope): the assertion lives in the stage's own handler,
    # immediately before it returns, rather than a separate end-of-run SF
    # state. OBSERVE MODE ONLY — never enables enforcement, never raises.
    result = {"status": "OK", "summary": summary, "date": run_date}
    try:
        from krepis.stage_coverage import assert_stage_coverage

        result["stage_coverage"] = assert_stage_coverage(
            "Scanner", run_date=run_date, window_start=_started,
        )
    except ImportError as exc:
        # Loud, not silent: the krepis pin predates the module (krepis-PR148 not yet merged). Observe mode —
        # the handler's own outcome is unchanged (config-I7214).
        logger.error("stage-coverage assertion unavailable: %s", exc)

    return result
