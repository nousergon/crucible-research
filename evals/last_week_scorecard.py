"""Prior-cycle realized-outcomes scorecard for the research multi-agent loop.

Closes the only auto-config loop in the system that today has no outcome
feedback: the LLM-driven research agents themselves. Backtester already
auto-tunes scoring weights / executor risk params / predictor veto /
research signal-boost configs weekly. The agents writing the BUY/HOLD/
EXIT theses have no idea whether their prior calls worked.

This module joins the substrate that DOES carry realized outcomes
(score_performance for BUY-signal hit rates at 10d/30d, predictor_outcomes
for 21d log-alpha at the per-ticker level) into a compact snapshot —
the "last week's scorecard" — that downstream phases will inject into
the CIO + Macro Economist prompts under a labeled section.

Substrate joins:
  - score_performance — per-BUY-signal beat-SPY at 10d / 30d (sector taken
    from a JOIN against `population` since score_performance is not
    self-describing on sector).
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
from typing import Optional

logger = logging.getLogger(__name__)

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
    hit_rate_10d: Optional[float]
    hit_rate_30d: Optional[float]
    mean_excess_10d: Optional[float]


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
    n_resolved_signals_10d: int
    n_resolved_signals_30d: int
    overall_predictor_hit_rate: Optional[float]
    overall_signal_hit_rate_10d: Optional[float]
    overall_signal_hit_rate_30d: Optional[float]
    market_regime: Optional[str]
    per_sector: list[SectorRow] = field(default_factory=list)
    top_surprises: list[TickerOutcome] = field(default_factory=list)
    top_confirmations: list[TickerOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


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

    sig_10d = [r for r in signal_rows if r["beat_spy_10d"] is not None]
    sig_30d = [r for r in signal_rows if r["beat_spy_30d"] is not None]
    overall_sig_10d = sum(r["beat_spy_10d"] for r in sig_10d) / len(sig_10d) if sig_10d else None
    overall_sig_30d = sum(r["beat_spy_30d"] for r in sig_30d) / len(sig_30d) if sig_30d else None

    per_sector = _build_sector_rows(signal_rows)
    surprises, confirmations = _build_surprise_lists(predictor_rows)
    regime = _fetch_market_regime(conn, window_end_s)

    return Scorecard(
        as_of_date=as_of_date.isoformat(),
        lookback_weeks=lookback_weeks,
        n_resolved_predictions=n_predictor_resolved,
        n_resolved_signals_10d=len(sig_10d),
        n_resolved_signals_30d=len(sig_30d),
        overall_predictor_hit_rate=overall_pred_hit,
        overall_signal_hit_rate_10d=overall_sig_10d,
        overall_signal_hit_rate_30d=overall_sig_30d,
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

    `score_performance` doesn't carry sector. Population's current sector
    is a reasonable proxy — sector reassignments inside a 4-week window
    are rare enough to not bias the per-sector roll-up materially. When
    a symbol is missing from `population` (was scanned but never entered
    the tracked population), the row gets `sector="(unknown)"` and still
    contributes to the overall hit rate.
    """
    sql = """
        SELECT
            sp.symbol,
            sp.score_date,
            sp.score,
            sp.beat_spy_10d,
            sp.beat_spy_30d,
            sp.return_10d,
            sp.spy_10d_return,
            COALESCE(p.sector, '(unknown)') AS sector
        FROM score_performance sp
        LEFT JOIN population p ON p.symbol = sp.symbol
        WHERE sp.score_date BETWEEN ? AND ?
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    return [
        {
            "symbol": r[0],
            "score_date": r[1],
            "score": r[2],
            "beat_spy_10d": r[3],
            "beat_spy_30d": r[4],
            "return_10d": r[5],
            "spy_10d_return": r[6],
            "sector": r[7],
        }
        for r in rows
    ]


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
    """Per-sector hit rate + mean excess return at 10d horizon."""
    by_sector: dict[str, list[dict]] = {}
    for r in signal_rows:
        by_sector.setdefault(r["sector"], []).append(r)

    out: list[SectorRow] = []
    for sector, rows in sorted(by_sector.items()):
        if len(rows) < _MIN_SECTOR_N:
            continue
        h10 = [r["beat_spy_10d"] for r in rows if r["beat_spy_10d"] is not None]
        h30 = [r["beat_spy_30d"] for r in rows if r["beat_spy_30d"] is not None]
        excess_10d = [
            r["return_10d"] - r["spy_10d_return"]
            for r in rows
            if r["return_10d"] is not None and r["spy_10d_return"] is not None
        ]
        out.append(
            SectorRow(
                sector=sector,
                n_signals=len(rows),
                hit_rate_10d=sum(h10) / len(h10) if h10 else None,
                hit_rate_30d=sum(h30) / len(h30) if h30 else None,
                mean_excess_10d=sum(excess_10d) / len(excess_10d) if excess_10d else None,
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
        f"- Research signal hit rate (10d vs SPY): {_fmt_pct(sc.overall_signal_hit_rate_10d)} "
        f"({sc.n_resolved_signals_10d} resolved)"
    )
    lines.append(
        f"- Research signal hit rate (30d vs SPY): {_fmt_pct(sc.overall_signal_hit_rate_30d)} "
        f"({sc.n_resolved_signals_30d} resolved)"
    )

    if sc.per_sector:
        lines.append("")
        lines.append("### Per-sector hit rate (≥3 resolved signals)")
        for s in sc.per_sector:
            lines.append(
                f"- {s.sector}: 10d {_fmt_pct(s.hit_rate_10d)} / 30d {_fmt_pct(s.hit_rate_30d)} "
                f"(n={s.n_signals}, mean 10d excess vs SPY {_fmt_signed(s.mean_excess_10d, 3)})"
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
