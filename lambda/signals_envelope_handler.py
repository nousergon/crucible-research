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
setup_logging("signals_envelope", flow_doctor_yaml=_FLOW_DOCTOR_YAML)

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
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


@monitor_handler
def handler(event, context):
    """Build + write the no-agent signals envelope. Raises on failure (see
    module doc's RAISE contract)."""
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

    # target is forwarded verbatim — production intent is always explicit,
    # this handler never infers or defaults it silently past the event's
    # own "target" field (defaulted to "shadow" above if absent).
    dated_key, latest_key = write_envelope(
        envelope, run_date, target=target, bucket=bucket, s3_client=s3,
    )

    # ── Ported secondary artifacts (config-I3290) ─────────────────────────
    # research_consolidated_morning + scanner_universe_trajectory used to be
    # written from inside the old multi-agent archive_writer node
    # (graph/research_graph.py), which config#2515 removed from the weekly
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

    logger.info(
        "[signals_envelope_handler] done run_date=%s target=%s dated_key=%s "
        "universe=%d market_regime=%s",
        run_date, target, dated_key,
        len(envelope["universe"]), envelope["market_regime"],
    )
    return {
        "status": "OK",
        "dated_key": dated_key,
        "latest_key": latest_key,
        "universe_count": len(envelope["universe"]),
        "market_regime": envelope["market_regime"],
        "target": target,
    }
