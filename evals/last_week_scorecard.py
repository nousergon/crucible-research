"""Prior-cycle realized-outcomes scorecard for the research multi-agent loop.

Closes the only auto-config loop in the system that today has no outcome
feedback: the LLM-driven research agents themselves. Backtester already
auto-tunes scoring weights / executor risk params / predictor veto /
research signal-boost configs weekly. The agents writing the BUY/HOLD/
EXIT theses have no idea whether their prior calls worked.

This module joins the substrate that DOES carry realized outcomes
(score_performance for BUY-signal hit rates at the canonical 21d horizon,
predictor_outcomes for 21d log-alpha at the per-ticker level) into a
compact snapshot — the "last week's scorecard" — that downstream phases
will inject into the CIO + Macro Economist prompts under a labeled section.

Substrate joins:
  - score_performance — per-BUY-signal identity (symbol/score_date/score),
    joined against the long-format `score_performance_outcomes` store (via
    `evals.outcome_store`) for the canonical-primary-horizon (21d) beat-SPY
    flag + realized log-alpha (config#1483/config#1530 cutover — this
    replaces the retired wide horizon-suffixed score_performance column
    reads). Sector is taken from a JOIN against `population` since neither
    table is self-describing on sector.
  - predictor_outcomes — per-prediction realized 21d log-alpha + correctness
    (canonical `correct` / `actual_log_alpha` with legacy `correct_5d` /
    `actual_5d_return` fallback for pre-2026-05-09 rows).
  - macro_snapshots — last-cycle market regime label, surfaced verbatim
    on the scorecard so agents reason in the regime that actually held.

Phase 1 (this module): pure JOIN + render. Read SQLite, compute the
Scorecard dataclass, format as JSON or text. No S3 emission, no prompt
wiring. Composes with the Phase 1.B follow-on (S3 + SF wiring) and the
Phase 2 prompt-injection PR.

Failure modes designed against:
  - **Goodhart on rubric outputs:** only REALIZED outcome data here;
    Sonnet rubric scores from `evals/judge.py` are intentionally NOT in
    the scorecard. Feeding rubric back creates a Goodhart loop.
  - **Recency overfit:** caps the lookback window at 4 weeks by default.
    Older outcomes are weakly informative for next week's regime and
    bias the agents toward chasing whatever just happened.
  - **Survivorship bias on surprises:** "top-3 surprises / confirmations"
    sample the FULL predicted universe, not just held positions.
  - **Token bloat:** caps each list at K=3 entries and the per-section
    body length so the rendered text lands well under the ~1500-token
    budget claimed in the ROADMAP entry.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from nousergon_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)

from evals import outcome_store

logger = logging.getLogger(__name__)

# Canonical S3 prefix for the scorecard pipeline. Same partition shape
# as `predictor/variant_gates/triple_barrier` and the eval-judge pipeline
# — flat `{prefix}/{run_id}.json` + `{prefix}/latest.json` sidecar per
# the institutional layout codified in `nousergon_lib.eval_artifacts`.
DEFAULT_SCORECARD_PREFIX = "research/last_week_scorecard"

# Top-K surprise / confirmation lists. Bounded so prompt-injection cost
# is bounded; 3 is the smallest k that still surfaces a pattern (a single
# outlier in a 50-prediction window is plausibly noise; three coordinated
# outliers across sectors is a signal worth feeding back).
_TOP_K = 3

# Default lookback window for the scorecard. 4 weeks matches the
# backtester's own rolling-mean horizon for predictor / signal-quality
# metrics; agents see the same window the auto-config loop optimizes
# against.
_DEFAULT_LOOKBACK_WEEKS = 4

# Minimum sample size before we surface a per-sector hit rate. Sectors
# with < this many resolved signals get omitted rather than reporting a
# noisy 1-of-2 = 50% hit rate as if it were meaningful.
_MIN_SECTOR_N = 3


@dataclass
class SectorRow:
    """Per-sector realized hit-rate roll-up over the lookback window."""

    sector: str
    n_signals: int
    hit_rate_21d: Optional[float]
    mean_log_alpha_21d: Optional[float]


@dataclass
class TickerOutcome:
    """A single predicted ticker + its realized 21d log-alpha.

    `surprise_sigma` is the realized log-alpha standardized against the
    cross-sectional std of all resolved log-alphas in the window. Sign
    is direction-agnostic — positive means realized BETTER than mean,
    negative means realized WORSE than mean. Combined with
    `predicted_direction` it tells the agents which way the call missed.
    """

    symbol: str
    prediction_date: str
    predicted_direction: str
    prediction_confidence: float
    realized_log_alpha: float
    surprise_sigma: float


@dataclass
class Scorecard:
    """Compact realized-outcomes snapshot fed into next cycle's prompts."""

    as_of_date: str
    lookback_weeks: int
    n_resolved_predictions: int
    n_resolved_signals_21d: int
    overall_predictor_hit_rate: Optional[float]
    overall_signal_hit_rate_21d: Optional[float]
    market_regime: Optional[str]
    per_sector: list[SectorRow] = field(default_factory=list)
    top_surprises: list[TickerOutcome] = field(default_factory=list)
    top_confirmations: list[TickerOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scorecard":
        """Hydrate from a JSON-deserialized dict (round-trips ``to_dict()``).

        Used by the S3 loader to reconstruct a typed Scorecard from the
        latest.json artifact so `format_scorecard_text` works against it.
        """
        return cls(
            as_of_date=d["as_of_date"],
            lookback_weeks=d["lookback_weeks"],
            n_resolved_predictions=d["n_resolved_predictions"],
            n_resolved_signals_21d=d["n_resolved_signals_21d"],
            overall_predictor_hit_rate=d.get("overall_predictor_hit_rate"),
            overall_signal_hit_rate_21d=d.get("overall_signal_hit_rate_21d"),
            market_regime=d.get("market_regime"),
            per_sector=[SectorRow(**row) for row in (d.get("per_sector") or [])],
            top_surprises=[TickerOutcome(**row) for row in (d.get("top_surprises") or [])],
            top_confirmations=[TickerOutcome(**row) for row in (d.get("top_confirmations") or [])],
        )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_scorecard(
    conn: sqlite3.Connection,
    as_of_date: date,
    lookback_weeks: int = _DEFAULT_LOOKBACK_WEEKS,
) -> Scorecard:
    """Compute the scorecard from research.db.

    `as_of_date` is the Saturday the scorecard is being built FOR — i.e.,
    next cycle's run date. The lookback window ends one day before
    `as_of_date` so the current cycle's own predictions don't leak in.
    """
    window_end = as_of_date - timedelta(days=1)
    window_start = window_end - timedelta(weeks=lookback_weeks)
    window_start_s = window_start.isoformat()
    window_end_s = window_end.isoformat()

    predictor_rows = _fetch_predictor_outcomes(conn, window_start_s, window_end_s)
    signal_rows = _fetch_signal_outcomes(conn, window_start_s, window_end_s)

    n_predictor_resolved = sum(1 for r in predictor_rows if r["resolved"] is not None)
    overall_pred_hit = (
        sum(r["resolved"] for r in predictor_rows if r["resolved"] is not None) / n_predictor_resolved
        if n_predictor_resolved
        else None
    )

    sig_21d = [r for r in signal_rows if r["beat_spy"] is not None]
    overall_sig_21d = sum(r["beat_spy"] for r in sig_21d) / len(sig_21d) if sig_21d else None

    per_sector = _build_sector_rows(signal_rows)
    surprises, confirmations = _build_surprise_lists(predictor_rows)
    regime = _fetch_market_regime(conn, window_end_s)

    return Scorecard(
        as_of_date=as_of_date.isoformat(),
        lookback_weeks=lookback_weeks,
        n_resolved_predictions=n_predictor_resolved,
        n_resolved_signals_21d=len(sig_21d),
        overall_predictor_hit_rate=overall_pred_hit,
        overall_signal_hit_rate_21d=overall_sig_21d,
        market_regime=regime,
        per_sector=per_sector,
        top_surprises=surprises,
        top_confirmations=confirmations,
    )


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _fetch_predictor_outcomes(
    conn: sqlite3.Connection, start: str, end: str
) -> list[dict]:
    """Pull per-prediction outcomes in window with canonical/legacy COALESCE.

    Post-2026-05-09 rows populate `correct` / `actual_log_alpha`; pre-cutover
    rows populate `correct_5d` / `actual_5d_return`. `alpha-engine-data` no
    longer dual-writes, so reading only the legacy columns silently drops
    every live-system row. COALESCE collapses both into a single resolved
    surface at the SQL boundary.
    """
    sql = """
        SELECT
            symbol,
            prediction_date,
            predicted_direction,
            prediction_confidence,
            COALESCE(correct, correct_5d)               AS resolved,
            COALESCE(actual_log_alpha, actual_5d_return) AS realized_alpha
        FROM predictor_outcomes
        WHERE prediction_date BETWEEN ? AND ?
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    return [
        {
            "symbol": r[0],
            "prediction_date": r[1],
            "predicted_direction": r[2],
            "prediction_confidence": r[3],
            "resolved": r[4],
            "realized_alpha": r[5],
        }
        for r in rows
    ]


def _fetch_signal_outcomes(
    conn: sqlite3.Connection, start: str, end: str
) -> list[dict]:
    """Pull per-BUY-signal hit-rate rows with sector via population JOIN.

    `score_performance` doesn't carry sector or the canonical 21d outcome
    (config#1483/config#1530 cutover: the outcome now lives in the
    long-format `score_performance_outcomes` store, read via
    `evals.outcome_store` and joined here by `(symbol, score_date)` — NOT the
    retired wide horizon-suffixed score_performance columns). Population's
    current sector is a reasonable proxy — sector reassignments inside a
    4-week window are rare enough to not bias the per-sector roll-up
    materially. When a symbol is missing from `population` (was scanned but
    never entered the tracked population), the row gets `sector="(unknown)"`
    and still contributes to the overall hit rate.
    """
    sql = """
        SELECT
            sp.symbol,
            sp.score_date,
            sp.score,
            COALESCE(p.sector, '(unknown)') AS sector
        FROM score_performance sp
        LEFT JOIN population p ON p.symbol = sp.symbol
        WHERE sp.score_date BETWEEN ? AND ?
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    outcomes = outcome_store.load_primary_outcomes(conn, start, end)
    result = []
    for r in rows:
        symbol, score_date = r[0], r[1]
        outcome = outcomes.get((symbol, score_date))
        result.append(
            {
                "symbol": symbol,
                "score_date": score_date,
                "score": r[2],
                "beat_spy": outcome.beat_spy if outcome else None,
                "log_alpha": outcome.log_alpha if outcome else None,
                "sector": r[3],
            }
        )
    return result


def _fetch_market_regime(conn: sqlite3.Connection, on_or_before: str) -> Optional[str]:
    """Latest market regime label at or before window end."""
    try:
        row = conn.execute(
            "SELECT regime FROM macro_snapshots "
            "WHERE date <= ? ORDER BY date DESC LIMIT 1",
            (on_or_before,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _build_sector_rows(signal_rows: list[dict]) -> list[SectorRow]:
    """Per-sector hit rate + mean realized log-alpha at the canonical 21d horizon."""
    by_sector: dict[str, list[dict]] = {}
    for r in signal_rows:
        by_sector.setdefault(r["sector"], []).append(r)

    out: list[SectorRow] = []
    for sector, rows in sorted(by_sector.items()):
        if len(rows) < _MIN_SECTOR_N:
            continue
        h21 = [r["beat_spy"] for r in rows if r["beat_spy"] is not None]
        sector_log_alphas = [
            r["log_alpha"] for r in rows if r["log_alpha"] is not None
        ]
        out.append(
            SectorRow(
                sector=sector,
                n_signals=len(rows),
                hit_rate_21d=sum(h21) / len(h21) if h21 else None,
                mean_log_alpha_21d=(
                    sum(sector_log_alphas) / len(sector_log_alphas)
                    if sector_log_alphas else None
                ),
            )
        )
    return out


def _build_surprise_lists(
    predictor_rows: list[dict],
) -> tuple[list[TickerOutcome], list[TickerOutcome]]:
    """Top-K predicted-UP surprises (realized far below mean) + confirmations.

    Standardizes realized log-alpha against the cross-sectional std of
    all resolved alphas in the window. For predicted-UP calls:
      - large NEGATIVE surprise_sigma = "we said UP, it went hard the
        other way" → surprise
      - large POSITIVE surprise_sigma = "we said UP, it went hard our
        way" → confirmation

    Returns ([], []) when there's no realized-alpha cross-section to
    standardize against (need ≥2 resolved alphas for a std).
    """
    resolved = [
        r for r in predictor_rows
        if r["realized_alpha"] is not None and r["predicted_direction"]
    ]
    if len(resolved) < 2:
        return [], []

    alphas = [r["realized_alpha"] for r in resolved]
    mean = sum(alphas) / len(alphas)
    var = sum((a - mean) ** 2 for a in alphas) / len(alphas)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0.0:
        return [], []

    up_rows = [r for r in resolved if r["predicted_direction"] == "UP"]

    def _to_outcome(r: dict) -> TickerOutcome:
        return TickerOutcome(
            symbol=r["symbol"],
            prediction_date=r["prediction_date"],
            predicted_direction=r["predicted_direction"],
            prediction_confidence=r["prediction_confidence"] or 0.0,
            realized_log_alpha=r["realized_alpha"],
            surprise_sigma=(r["realized_alpha"] - mean) / std,
        )

    outcomes = [_to_outcome(r) for r in up_rows]
    # Surprises: predicted UP, realized worst (most negative sigma).
    surprises = sorted(outcomes, key=lambda o: o.surprise_sigma)[:_TOP_K]
    # Confirmations: predicted UP, realized best (most positive sigma).
    confirmations = sorted(outcomes, key=lambda o: -o.surprise_sigma)[:_TOP_K]
    return surprises, confirmations


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%"


def _fmt_signed(v: Optional[float], decimals: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:+.{decimals}f}"


def format_scorecard_text(sc: Scorecard) -> str:
    """Markdown-ish plain-text rendering for prompt injection.

    Compact on purpose — total budget ~1500 tokens including the section
    framing the consuming prompt will wrap around it.
    """
    lines: list[str] = []
    lines.append(f"## Prior cycle's realized outcomes ({sc.lookback_weeks}-week lookback through {sc.as_of_date})")
    if sc.market_regime:
        lines.append(f"Market regime over the window: **{sc.market_regime}**.")
    lines.append("")
    lines.append("### Overall")
    lines.append(
        f"- Predictor hit rate: {_fmt_pct(sc.overall_predictor_hit_rate)} "
        f"({sc.n_resolved_predictions} resolved)"
    )
    lines.append(
        f"- Research signal hit rate (21d vs SPY): {_fmt_pct(sc.overall_signal_hit_rate_21d)} "
        f"({sc.n_resolved_signals_21d} resolved)"
    )

    if sc.per_sector:
        lines.append("")
        lines.append("### Per-sector hit rate (≥3 resolved signals)")
        for s in sc.per_sector:
            lines.append(
                f"- {s.sector}: 21d {_fmt_pct(s.hit_rate_21d)} "
                f"(n={s.n_signals}, mean 21d log-α vs SPY {_fmt_signed(s.mean_log_alpha_21d, 3)})"
            )

    if sc.top_surprises:
        lines.append("")
        lines.append("### Surprises — predicted UP, realized worst")
        for o in sc.top_surprises:
            lines.append(
                f"- {o.symbol} ({o.prediction_date}, conf {_fmt_pct(o.prediction_confidence)}): "
                f"realized log-α {_fmt_signed(o.realized_log_alpha)} "
                f"({o.surprise_sigma:+.2f}σ vs cross-section)"
            )

    if sc.top_confirmations:
        lines.append("")
        lines.append("### Confirmations — predicted UP, realized best")
        for o in sc.top_confirmations:
            lines.append(
                f"- {o.symbol} ({o.prediction_date}, conf {_fmt_pct(o.prediction_confidence)}): "
                f"realized log-α {_fmt_signed(o.realized_log_alpha)} "
                f"({o.surprise_sigma:+.2f}σ vs cross-section)"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# S3 emission
# ---------------------------------------------------------------------------


def emit_scorecard_to_s3(
    sc: Scorecard,
    *,
    s3_client: Any,
    bucket: str,
    prefix: str = DEFAULT_SCORECARD_PREFIX,
    run_id: Optional[str] = None,
) -> dict:
    """Write `sc` to S3 under the canonical eval-artifacts partition.

    Two keys land per call:
      - `{prefix}/{run_id}.json` — forensic, never overwritten (run_id
        is `YYMMDDHHMM` so concurrent same-minute writes are the only
        clobber risk and aren't relevant on a weekly cadence).
      - `{prefix}/latest.json` — operator-UX sidecar pointing to the
        most recent payload by exact-mirror copy.

    `s3_client` is injected so tests can pass a stub. Caller is
    responsible for constructing a real boto3 client when wired into
    the Research Lambda (Phase 1.B.2).

    Returns the `{dated_key, latest_key, run_id}` dict so callers can
    log + emit observability without re-deriving the keys.

    Per [[feedback_no_silent_fails]] this raises on any S3 failure —
    the scorecard is a PRODUCER artifact (Phase 2 consumers read it).
    Best-effort posture would mask Phase-2 prompt-injection on real
    failures, which is the worst possible outcome.
    """
    if not bucket:
        raise ValueError("emit_scorecard_to_s3 requires a non-empty bucket")
    run_id = run_id or new_eval_run_id()
    dated_key = eval_artifact_key(prefix, run_id)
    latest_key = eval_latest_key(prefix)
    payload = json.dumps(sc.to_dict(), indent=2).encode("utf-8")

    s3_client.put_object(
        Bucket=bucket,
        Key=dated_key,
        Body=payload,
        ContentType="application/json",
    )
    s3_client.put_object(
        Bucket=bucket,
        Key=latest_key,
        Body=payload,
        ContentType="application/json",
    )
    logger.info(
        "scorecard emitted bucket=%s dated=%s latest=%s",
        bucket,
        dated_key,
        latest_key,
    )
    return {"dated_key": dated_key, "latest_key": latest_key, "run_id": run_id}


# ---------------------------------------------------------------------------
# S3 read (consumer-side, Phase 2.A)
# ---------------------------------------------------------------------------


def load_latest_scorecard(
    *,
    s3_client: Any,
    bucket: str,
    prefix: str = DEFAULT_SCORECARD_PREFIX,
) -> Optional[Scorecard]:
    """Fetch `{prefix}/latest.json` and hydrate to a Scorecard.

    Returns None on any failure (404 = Phase 1.B.2 not yet flag-on or
    first cycle of soak; network / parse / hydrate errors = transient
    or corrupt artifact). The consumer's job is to render and inject
    the result; the no-data case is "agents reason without the
    scorecard," which is their pre-Phase-2 behavior.

    Failure mode posture: graceful — return None, log WARN. Consumers
    that want hard-fail-on-miss can check `result is None` and decide
    for themselves. The intended LLM-prompt-injection consumer should
    treat missing data as "no prior cycle to learn from" rather than
    fail the whole research cycle on a missing observability artifact.
    """
    try:
        key = eval_latest_key(prefix)
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        data = json.loads(body)
        return Scorecard.from_dict(data)
    except Exception as e:
        logger.warning(
            "scorecard load failed s3://%s/%s/latest.json: %s — returning None",
            bucket, prefix.strip("/"), e,
        )
        return None


def load_latest_scorecard_text(
    *,
    s3_client: Any,
    bucket: str,
    prefix: str = DEFAULT_SCORECARD_PREFIX,
) -> str:
    """Convenience: load latest scorecard, render to prompt-ready text.

    Returns empty string when the artifact is missing or invalid. The
    intended consumer wires this string into a prompt template's
    `{prior_cycle_scorecard}` placeholder; empty string means the
    placeholder renders as nothing and the agents fall back to their
    pre-Phase-2 behavior. Mirrors the established pattern (see
    `agents/macro_agent.py::regime_substrate_block`).
    """
    sc = load_latest_scorecard(s3_client=s3_client, bucket=bucket, prefix=prefix)
    if sc is None:
        return ""
    return format_scorecard_text(sc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the prior-cycle realized-outcomes scorecard from "
            "research.db. Phase 1 — pure JOIN + render; S3 emission "
            "and SF wiring follow."
        )
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to research.db on the local filesystem.",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="ISO date the scorecard is being built for (typically the next Saturday).",
    )
    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=_DEFAULT_LOOKBACK_WEEKS,
        help=f"Lookback window in weeks (default {_DEFAULT_LOOKBACK_WEEKS}).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format. 'json' is the machine-consumed shape; 'text' is the prompt-ready rendering.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help=(
            "Optional S3 bucket. When provided, emits the scorecard "
            "JSON to `{prefix}/{run_id}.json` + `{prefix}/latest.json` "
            "alongside the stdout render. Operator-driven invocations "
            "typically leave this blank; the Saturday SF Research Lambda "
            "passes the production bucket."
        ),
    )
    parser.add_argument(
        "--s3-prefix",
        default=DEFAULT_SCORECARD_PREFIX,
        help=f"S3 prefix root (default: {DEFAULT_SCORECARD_PREFIX}).",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 2

    as_of = date.fromisoformat(args.as_of)
    with sqlite3.connect(str(db_path)) as conn:
        sc = build_scorecard(conn, as_of_date=as_of, lookback_weeks=args.lookback_weeks)

    if args.format == "json":
        print(json.dumps(sc.to_dict(), indent=2))
    else:
        print(format_scorecard_text(sc), end="")

    if args.s3_bucket:
        # Import boto3 lazily — CLI invocations without --s3-bucket
        # should not require boto3 to be importable.
        import boto3
        client = boto3.client("s3")
        result = emit_scorecard_to_s3(
            sc,
            s3_client=client,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
        )
        print(
            f"emitted s3://{args.s3_bucket}/{result['dated_key']} "
            f"+ s3://{args.s3_bucket}/{result['latest_key']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
