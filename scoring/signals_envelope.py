"""Thin, NO-AGENT producer for the executor-facing ``signals.json`` envelope
(alpha-engine-config epic #2515).

Context (epic scope-clarification, 2026-07-14). The multi-agent research
GRAPH RUNNER (six sector teams + CIO) is being pulled out of the weekly Step
Function; the research MODULE — scanner, quant factor scoring, universe
board, RAG, Think Tank — all remain. A traced consumer audit (see epic
comment) found the executor's real dependency on agent-authored content is
much smaller than assumed:

  * Entry selection/ordering/veto is already predictor-driven
    (``alpha-engine/executor/deciders.py`` orders by ``prediction_confidence``
    + ``gbm_veto``; the champion arm, ``executor/champion.py``, already
    synthesizes ENTER candidates with neutral per-name fields).
  * Regime-conditional sizing already reads the quant regime substrate
    directly (``position_sizer.py::regime_conditional_size_multiplier`` ←
    ``regime/composite.py``), not ``signals.json``'s ``market_regime`` label.
  * Exits are strategy/technical/risk-driven (Slot S plugin exits +
    drawdown-forced), not research-stance-driven.

What genuinely remains research-authored: per-name ``conviction`` /
``price_target_upside`` sizing tilts (already neutralized on the champion
path), the ``sector_ratings`` sizing multiplier, the ``market_regime`` label
(residual non-sizing uses — see ``derive_market_regime``), universe
membership for held-name context, and the pre-derived per-ticker ``signal``
string the Slot S exit-rule contract reads as ``research_action``
(``executor/strategies/contract.py`` / ``exit_manager.py`` — derived at READ
time as ``research_signal.get("signal", "HOLD")``, not a separate JSON key).

This module is a bridge producer: it builds the SAME ``signals/{date}/
signals.json`` S3 artifact from PURE QUANT SOURCES ONLY (no LLM calls, no
LangGraph) — the scanner's universe board + the predictor's regime substrate
— with every research-authored per-name judgment replaced by a documented
neutral default. Same S3 key (S3 Contract Safety); the executor and
predictor need no changes to read it. Wiring this into the scanner Lambda
happens in a follow-on change; this module + CLI are standalone.

Consumed-field inventory (traced 2026-07-14 against ``alpha-engine``
executor + ``alpha-engine-predictor`` + ``alpha-engine-data`` — see the
mapping comment in ``tests/test_signals_envelope_contract.py``) and the
existing ``nousergon_lib.contracts`` ``signals`` v1 JSON Schema (the SAME
Slot-R contract this producer targets — see ``build_signals_envelope``)
together define every field below.

Field policy v1 (this module IS the schema_version 1 producer):

* ``market_regime`` — mapped from the regime substrate's composite
  ``intensity_z``: ``>= +0.5`` -> ``"bull"``, ``<= -0.5`` -> ``"bear"``,
  else ``"neutral"``. NOTE: the executor's real 3-class taxonomy is
  ``bull``/``neutral``/``bear`` (v0.42.0 3-class Ang-Bekaert regime
  retirement of the legacy 4-class ``caution`` label — see
  ``nousergon_lib.contracts`` ``signals.schema.json``'s ``market_regime``
  enum and ``alpha-engine/executor/main.py``'s ``_macro_rank = {"bull": 0,
  "neutral": 1, "bear": 2}`` / ``market_regime == "bear"`` gates in
  ``risk_guard.py``). An earlier informal "risk_on/risk_off" framing for
  this field does not exist ANYWHERE in the executor or research
  codebases (grepped clean) and would silently defeat the bear-market
  protective gates (string-equality checks, not a lookup with a safe
  default) — corrected here to the verified real enum.
* Substrate missing/unreadable -> ``"neutral"`` + a WARN log. This is the
  ONE fail-soft exception in this module (mirrors
  ``alpha-engine/executor/signal_reader.py::read_regime_substrate``'s own
  contract): the regime substrate Lambda is a non-blocking weekly SF
  producer with its own freshness monitoring: a missing/stale substrate
  must not block this producer's primary deliverable.
* ``sector_ratings`` — every sector present on the scanner universe board
  gets ``{"rating": "market_weight", "modifier": 1.0, "rationale": ...}``
  (neutral sizing multiplier; ``sector_adj_map.get(rating, 1.00)`` in
  ``position_sizer.py`` maps ``market_weight`` -> 1.00).
* ``sector_modifiers`` — every board sector -> ``1.0`` (predictor's
  ``sector_macro_modifier`` feature is ``sector_modifiers.get(sector, 1.0)
  - 1`` -> 0, neutral either way; emitted explicitly for schema parity).
* ``universe[]`` — one row per board name: ``conviction: "stable"``
  (position_sizer's decline-derate only fires on ``"declining"``),
  ``price_target_upside: None`` (upside-derate only fires when not None
  and below the configured floor), ``rating: "HOLD"``, ``sector_rating:
  "market_weight"``, ``signal: "HOLD"`` (the Slot S contract's
  ``research_action`` is DERIVED from this field at read time, defaulting
  to ``"HOLD"`` itself — see ``exit_manager.py:826``, so this is doubly
  neutral). ``score`` is the board's real ``attractiveness_score`` (quant
  fact, not fabricated) so the drawdown-forced-exit conviction ranking
  (``main.py``'s ``_conviction_rank``) still has real signal instead of
  uniformly defaulting to 50 for every held name.
* ``buy_candidates`` — always ``[]``. VERIFIED against
  ``alpha-engine/executor/champion.py::apply_champion_selection`` (lines
  ~280-288): ``n = n_buy_candidates if n_buy_candidates > 0 else
  config.get("champion_top_n_default", 10)`` — an empty ``buy_candidates``
  list is the INTENDED trigger for the champion's count-fallback when the
  ``scanner_predictor_direct`` champion arm is active (it synthesizes its
  own ENTER candidates in-memory from the research-free predictor cohort,
  subject to the SAME universe/coverage gates). CAVEAT (operational, not a
  defect of this producer): while the champion pointer resolves to the
  default ``"agentic"`` arm, ``apply_champion_selection`` is a no-op
  passthrough and an empty ``buy_candidates`` here means NO new entries
  ever get proposed — this producer's shadow/production rollout must be
  sequenced with (or after) the champion promotion to
  ``scanner_predictor_direct`` (tracked by the epic; out of scope for this
  module, called out again in this producer's CLI help text).
* ``date`` / ``run_date`` — the run date (YYYY-MM-DD). ``producer:
  "signals_envelope"`` and ``schema_version: 1`` are stamped so a future
  consumer can distinguish this producer from the multi-agent one if ever
  needed (additive fields per S3 Contract Safety).

Schema / M0 contract discipline: ``signals.json`` is Slot R's existing
product contract — ``nousergon_lib.contracts`` ALREADY ships a versioned
JSON Schema for it (``signals.schema.json`` / ``SLOT_SCHEMAS["signals"]``,
shipped with the ``nousergon-lib[contracts]`` extra this repo already
pins). This producer is a SECOND implementation of that SAME slot, not a
new artifact — per the M0 "build as if a second implementation of each slot
will exist" discipline, the correct move is to validate against the
EXISTING schema (avoiding a parallel, driftable schema file) rather than
authoring a new one. ``build_signals_envelope`` calls
``nousergon_lib.contracts.validate("signals", envelope)`` before returning
— a producer that emits an envelope violating its own declared contract
fails LOUD, not silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from nousergon_lib.contracts import validate as validate_contract

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PRODUCER_NAME = "signals_envelope"
CONTRACT_NAME = "signals"

DEFAULT_BUCKET = "alpha-engine-research"

UNIVERSE_BOARD_DATED_TPL = "scanner/universe/{date}/universe.json"
UNIVERSE_BOARD_LATEST_KEY = "scanner/universe/latest.json"

# Regime substrate — same canonical sidecar prefix the executor's own
# ``signal_reader.read_regime_substrate`` resolves via
# ``nousergon_lib.eval_artifacts.load_latest_eval_artifact``.
REGIME_SUBSTRATE_PREFIX = "regime"

# intensity_z thresholds mapping the regime composite onto the verified
# 3-class taxonomy (see module docstring for why this is bull/neutral/bear,
# not risk_on/risk_off).
_INTENSITY_Z_BULL_FLOOR = 0.5
_INTENSITY_Z_BEAR_CEIL = -0.5

_NEUTRAL_SECTOR_RATIONALE = (
    "signals_envelope v1 (no-agent producer): neutral sizing — no "
    "research-authored sector view available from pure-quant sources."
)


# ── S3 read helpers ──────────────────────────────────────────────────────────


def _client(s3_client: Any = None):
    return s3_client or boto3.client("s3")


def read_universe_board(
    bucket: str, run_date: str | None = None, s3_client: Any = None,
) -> dict:
    """Read the scanner universe board (schema_version 3, ``scoring/
    universe_board.py``). RAISES loud if unavailable at either the
    dated key or the ``latest.json`` sidecar.

    The board is the SOLE source of universe membership for this
    producer — an empty/absent board means no trading day can be
    constructed from pure-quant sources, which is a real fault, not a
    degrade-gracefully case (unlike the regime substrate — see
    ``read_regime_substrate``).
    """
    s3 = _client(s3_client)
    keys = []
    if run_date:
        keys.append(UNIVERSE_BOARD_DATED_TPL.format(date=run_date))
    keys.append(UNIVERSE_BOARD_LATEST_KEY)

    last_exc: Exception | None = None
    for key in keys:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return json.loads(obj["Body"].read())
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                last_exc = e
                continue
            raise

    raise RuntimeError(
        f"signals_envelope: no scanner universe board found at any of "
        f"{keys!r} in bucket {bucket!r}. The board "
        "(scoring/universe_board.py) is the sole universe-membership "
        "source for this no-agent producer — refusing to emit an empty "
        "envelope (no-silent-fails). Ensure the scanner has run for this "
        "cycle before invoking signals_envelope."
    ) from last_exc


def read_regime_substrate(bucket: str, s3_client: Any = None) -> dict | None:
    """Fail-SOFT read of the regime substrate artifact (``regime/latest.json``
    -> dated artifact). Returns ``None`` on ANY failure mode (missing
    sidecar, malformed pointer, missing artifact body, parse error, S3
    error) — mirrors ``alpha-engine/executor/signal_reader
    .read_regime_substrate`` exactly, since consumers of this producer's
    ``market_regime`` field already tolerate ``"neutral"`` as the
    no-signal default.

    This is the ONE deliberate fail-soft exception in this module (see
    module docstring): the substrate Lambda has its own non-blocking
    weekly SF slot and freshness monitoring; a substrate read failure
    must never block this producer's primary deliverable.
    """
    from nousergon_lib.eval_artifacts import load_latest_eval_artifact

    s3 = _client(s3_client)
    try:
        return load_latest_eval_artifact(s3, bucket=bucket, prefix=REGIME_SUBSTRATE_PREFIX)
    except Exception as exc:  # noqa: BLE001 — documented fail-soft exception (see docstring)
        # recording surface: this WARN log (mirrors signal_reader's own
        # graceful-degrade posture for the identical artifact).
        logger.warning(
            "signals_envelope: regime substrate unreadable (%s) — "
            "market_regime will default to 'neutral'.", exc,
        )
        return None


def _extract_intensity_z(substrate: dict | None) -> float | None:
    """Pull ``composite.intensity_z`` out of a regime substrate payload.

    Mirrors ``alpha-engine/executor/signal_reader.extract_intensity_z``.
    """
    if not isinstance(substrate, dict):
        return None
    composite = substrate.get("composite")
    if not isinstance(composite, dict):
        return None
    val = composite.get("intensity_z")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def derive_market_regime(substrate: dict | None) -> str:
    """Map the regime substrate's ``composite.intensity_z`` onto the
    verified 3-class taxonomy (``bull``/``neutral``/``bear`` — see module
    docstring for why NOT ``risk_on``/``risk_off``).

    ``intensity_z >= +0.5`` -> ``"bull"``; ``<= -0.5`` -> ``"bear"``; else
    ``"neutral"``. Substrate unavailable / no numeric ``intensity_z`` ->
    ``"neutral"`` with a WARN (the one fail-soft exception — see
    ``read_regime_substrate``).
    """
    z = _extract_intensity_z(substrate)
    if z is None:
        logger.warning(
            "signals_envelope: no usable intensity_z from regime substrate "
            "— market_regime='neutral' (fail-soft; substrate has its own "
            "non-blocking SF producer + freshness monitor)."
        )
        return "neutral"
    if z >= _INTENSITY_Z_BULL_FLOOR:
        return "bull"
    if z <= _INTENSITY_Z_BEAR_CEIL:
        return "bear"
    return "neutral"


# ── Envelope construction (pure — no I/O) ───────────────────────────────────


def _board_stocks(board: dict) -> list[dict]:
    stocks = board.get("stocks")
    return stocks if isinstance(stocks, list) else []


def _board_sectors(stocks: list[dict]) -> list[str]:
    """Unique, sorted sector names present on the board (falsy/None dropped)."""
    sectors = {s.get("sector") for s in stocks if isinstance(s, dict) and s.get("sector")}
    return sorted(sectors)


def build_sector_ratings(sectors: list[str]) -> dict[str, dict]:
    return {
        sector: {
            "rating": "market_weight",
            "modifier": 1.0,
            "rationale": _NEUTRAL_SECTOR_RATIONALE,
        }
        for sector in sectors
    }


def build_sector_modifiers(sectors: list[str]) -> dict[str, float]:
    return {sector: 1.0 for sector in sectors}


def _build_universe_entry(stock: dict) -> dict[str, Any] | None:
    ticker = stock.get("ticker")
    if not ticker:
        return None
    sector = stock.get("sector") or "Unknown"
    score = stock.get("attractiveness_score")
    quality_pillar = (stock.get("pillars") or {}).get("quality")
    return {
        "ticker": ticker,
        "signal": "HOLD",
        "score": score,
        "rating": "HOLD",
        "conviction": "stable",
        "sector": sector,
        "sector_rating": "market_weight",
        "price_target_upside": None,
        "thesis_summary": None,
        # Distinct provenance value (not agentic vocabulary — cio_entrant /
        # carryover / reaffirmed_hold / exit) so the evaluator's
        # stance_source_provenance grader (config#859) can tell agentic
        # picks apart from this producer's quant-only rows.
        "stance_source": "quant_envelope_producer",
        "quant_score": score,
        "qual_score": None,
        "factor_quality_score": quality_pillar,
        "sub_scores": {"quant": score, "qual": None},
    }


def build_universe_entries(stocks: list[dict]) -> list[dict]:
    entries = []
    for stock in stocks:
        if not isinstance(stock, dict):
            continue
        entry = _build_universe_entry(stock)
        if entry is not None:
            entries.append(entry)
    return entries


def build_signals_envelope(
    run_date: str,
    board: dict,
    substrate: dict | None,
) -> dict:
    """Assemble the full envelope (pure function — no I/O).

    Validates the built payload against the existing ``nousergon_lib
    .contracts`` ``signals`` v1 JSON Schema before returning — a producer
    emitting a non-conformant envelope is a build-time bug, not a
    runtime-tolerated shape (raises ``ContractViolation`` on failure).
    """
    stocks = _board_stocks(board)
    if not stocks:
        raise ValueError(
            "signals_envelope: universe board carries an empty/missing "
            "'stocks' list — refusing to build an envelope with zero "
            "universe membership (no-silent-fails)."
        )
    sectors = _board_sectors(stocks)
    universe = build_universe_entries(stocks)
    market_regime = derive_market_regime(substrate)
    now_iso = datetime.now(timezone.utc).strftime("%H:%M:%S")

    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER_NAME,
        "date": run_date,
        "run_date": run_date,
        "time": now_iso,
        "run_time": now_iso,
        "market_regime": market_regime,
        "sector_ratings": build_sector_ratings(sectors),
        "sector_modifiers": build_sector_modifiers(sectors),
        "universe": universe,
        "buy_candidates": [],
        # Legacy list-of-ticker-strings shape (matches the multi-agent
        # producer's own "population" field byte-for-byte — see
        # graph/research_graph.py's `"population": [p["ticker"] for p in pop]`).
        "population": [e["ticker"] for e in universe],
        # Legacy v2 ticker-keyed dict (schema-tolerated free-form object;
        # matches the multi-agent producer's own `signals[ticker] = {...}`
        # shape for byte-for-byte parity with today's consumers).
        "signals": {e["ticker"]: e for e in universe},
    }

    validate_contract(CONTRACT_NAME, envelope)
    return envelope


# ── S3 write ─────────────────────────────────────────────────────────────────


def _s3_keys_for_target(target: str, run_date: str) -> tuple[str, str]:
    if target == "shadow":
        return (f"signals_envelope/{run_date}/signals.json", "signals_envelope/latest.json")
    if target == "production":
        return (f"signals/{run_date}/signals.json", "signals/latest.json")
    raise ValueError(f"signals_envelope: unknown target {target!r} — must be 'shadow' or 'production'")


def write_envelope(
    envelope: dict,
    run_date: str,
    *,
    target: str,
    bucket: str,
    s3_client: Any = None,
) -> tuple[str, str]:
    """Write the envelope to the dated key + ``latest.json`` sidecar for
    ``target``. Returns ``(dated_key, latest_key)``."""
    dated_key, latest_key = _s3_keys_for_target(target, run_date)
    s3 = _client(s3_client)
    body = json.dumps(envelope, separators=(",", ":"), default=str).encode("utf-8")
    for key in (dated_key, latest_key):
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info(
        "signals_envelope: wrote %s (+%s) | universe=%d | market_regime=%s | target=%s",
        dated_key, latest_key, len(envelope.get("universe", [])),
        envelope.get("market_regime"), target,
    )
    return dated_key, latest_key


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signals_envelope",
        description=(
            "No-agent producer for signals/{date}/signals.json — builds the "
            "envelope from the scanner universe board + regime substrate "
            "only (no LLM calls). Defaults to --target shadow (writes "
            "signals_envelope/{date}/signals.json, never the live key). "
            "CAVEAT: with --target production, an empty buy_candidates list "
            "only produces trading entries if the champion pointer "
            "(config/producer_champion.json) is already 'scanner_predictor_"
            "direct' — under the default 'agentic' champion this envelope "
            "alone yields a no-entries trading day (see module docstring)."
        ),
    )
    parser.add_argument("--date", default=None, help="Run date YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket (default: {DEFAULT_BUCKET})")
    parser.add_argument(
        "--target", choices=("shadow", "production"), default="shadow",
        help="shadow (default): signals_envelope/{date}/signals.json. "
             "production: the live signals/{date}/signals.json key.",
    )
    parser.add_argument(
        "--i-know-this-is-production", action="store_true", dest="ack_production",
        help="Required alongside --target production — refuses to run "
             "without it, so a stray manual invocation cannot clobber the "
             "live trading key.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.target == "production" and not args.ack_production:
        parser.error(
            "--target production requires --i-know-this-is-production "
            "(refusing to risk clobbering the live signals.json key on a "
            "stray manual run)."
        )

    run_date = args.date or str(date.today())
    s3 = boto3.client("s3")

    board = read_universe_board(args.bucket, run_date=run_date, s3_client=s3)
    substrate = read_regime_substrate(args.bucket, s3_client=s3)
    envelope = build_signals_envelope(run_date, board, substrate)
    dated_key, latest_key = write_envelope(
        envelope, run_date, target=args.target, bucket=args.bucket, s3_client=s3,
    )

    print(json.dumps({
        "dated_key": dated_key,
        "latest_key": latest_key,
        "universe_count": len(envelope["universe"]),
        "market_regime": envelope["market_regime"],
        "target": args.target,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
