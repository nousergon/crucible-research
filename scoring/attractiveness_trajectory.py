"""Attractiveness trajectory signal — factor-momentum + lead-lag (under-reaction).

The SOTA "becoming more attractive AND outpacing related price gains" signal,
computed weekly from the attractiveness history (``scoring/attractiveness_history``)
+ ArcticDB prices. Per ticker:

  1. attr_slope  = Theil-Sen slope of ``attractiveness_raw`` over the last W
     weekly points (robust to a single noisy week; raw not percentile).
  2. sector_rel_price_ret = W-week log return − its SECTOR-ETF W-week return
     (sector-neutral price momentum; the 11 GICS ETFs).
  3. Cross-sectionally z-score both; OLS-regress slope_z ~ price_mom_z and take
     the RESIDUAL = ``pre_repricing_score`` — the attractiveness improvement NOT
     explained by price (rising attractiveness the market hasn't repriced).

Flags: ``rising`` (slope>0, Theil-Sen CI excludes 0); ``pre_repricing``
(rising AND residual in the top decile). Output:
``scanner/universe/trajectory/{date}.json`` (+ latest), schema_version 1.

The signal is OBSERVE-MODE — measured, not trusted; its forward IC is tracked
before it influences anything (OBSERVATION_REGISTRY). Failure posture: SECONDARY
observability — the wiring fail-SOFTs (a signal failure must not fail the run).
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

TRAJECTORY_SCHEMA_VERSION = 1
DEFAULT_WINDOW_WEEKS = 8
DEFAULT_MIN_POINTS = 4
PRE_REPRICING_PCT = 90.0

# GICS sector → SPDR sector ETF (the sector-neutral price benchmark). Both GICS
# canonical names and the research-internal short names are mapped so either
# sector source resolves.
_SECTOR_ETF = {
    "Information Technology": "XLK", "Technology": "XLK",
    "Health Care": "XLV", "Healthcare": "XLV",
    "Financials": "XLF", "Financial": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC", "Communications": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP", "Defensives": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}
SECTOR_ETFS = sorted(set(_SECTOR_ETF.values()))

CONSOLE_BASE_URL = os.environ.get("CONSOLE_BASE_URL", "https://console.nousergon.ai")


# ── pure numeric helpers ─────────────────────────────────────────────────────

def _zmap(d: dict[str, Optional[float]]) -> dict[str, float]:
    """Cross-sectional z-score of the non-null values (std==0 / <2 → all 0)."""
    vals = [v for v in d.values() if v is not None]
    if len(vals) < 2:
        return {t: 0.0 for t in d}
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    if sd == 0:
        return {t: 0.0 for t in d}
    return {t: (0.0 if v is None else (v - m) / sd) for t, v in d.items()}


def _percentile(sorted_vals: list[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _theil_sen(y) -> tuple[float, float, float]:
    """Theil-Sen slope = median of pairwise slopes over the weekly index
    (x = 0..n-1). Returns ``(median_slope, ci_lo, ci_hi)`` where the CI is the
    2.5/97.5 percentile band of the pairwise slopes (the slope is "significant"
    when that band excludes 0). Pure-numpy — robust to a single noisy week, and
    avoids a scipy dependency on the Lambda image."""
    import numpy as np

    n = len(y)
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    arr = np.asarray(slopes, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ── the signal ───────────────────────────────────────────────────────────────

def build_trajectory(
    history_df: "Any",
    price_ret: dict[str, float],
    sector_etf_ret: dict[str, float],
    *,
    run_date: str,
    window_weeks: int = DEFAULT_WINDOW_WEEKS,
    min_points: int = DEFAULT_MIN_POINTS,
    pre_repricing_pct: float = PRE_REPRICING_PCT,
) -> dict:
    """Build the trajectory signal artifact.

    Args:
        history_df: attractiveness history (cols ``as_of, ticker,
            attractiveness_raw, sector``). The last ``window_weeks`` points per
            ticker drive the Theil-Sen slope.
        price_ret: ``{ticker: W-week log return}``.
        sector_etf_ret: ``{sector_name: W-week sector-ETF log return}``.
    """
    import numpy as np

    hist = history_df.dropna(subset=["attractiveness_raw"]).sort_values("as_of")
    sector_by_ticker = (
        hist.groupby("ticker")["sector"].last().to_dict() if "sector" in hist.columns else {}
    )

    rows: dict[str, dict] = {}
    for ticker, g in hist.groupby("ticker"):
        y = g["attractiveness_raw"].tail(window_weeks).to_numpy(dtype=float)
        if len(y) < min_points:
            continue
        slope, lo, hi = _theil_sen(y)
        rows[ticker] = {
            "attr_slope": round(slope, 5),
            "n_points": int(len(y)),
            "slope_significant": bool(lo > 0 or hi < 0),
        }

    # sector-neutral price momentum
    for t, r in rows.items():
        pr = price_ret.get(t)
        sec = sector_by_ticker.get(t)
        er = sector_etf_ret.get(sec) if sec is not None else None
        r["sector"] = sec
        r["price_ret"] = None if pr is None else round(float(pr), 5)
        r["sector_rel_price_ret"] = None if (pr is None or er is None) else round(float(pr - er), 5)

    slope_z = _zmap({t: r["attr_slope"] for t, r in rows.items()})
    relpr_z = _zmap({t: r.get("sector_rel_price_ret") for t, r in rows.items()})
    for t, r in rows.items():
        r["attr_slope_z"] = round(slope_z[t], 4)
        r["price_mom_z"] = round(relpr_z[t], 4)

    # orthogonalize: residual of slope_z ~ price_mom_z (only over names with price)
    priced = [t for t in rows if rows[t].get("sector_rel_price_ret") is not None]
    if len(priced) >= 3:
        X = np.array([relpr_z[t] for t in priced])
        Y = np.array([slope_z[t] for t in priced])
        beta, alpha = np.polyfit(X, Y, 1)
        resid = {t: float(slope_z[t] - (beta * relpr_z[t] + alpha)) for t in priced}
        beta_used = round(float(beta), 4)
    else:
        # no usable price cross-section → residual degrades to the raw slope_z
        resid = {t: slope_z[t] for t in rows}
        beta_used = None
    for t, r in rows.items():
        r["pre_repricing_score"] = round(resid.get(t, slope_z[t]), 4)

    cut = _percentile(sorted(r["pre_repricing_score"] for r in rows.values()), pre_repricing_pct)
    for r in rows.values():
        r["rising"] = bool(r["attr_slope"] > 0 and r["slope_significant"])
        r["pre_repricing"] = bool(r["rising"] and cut is not None and r["pre_repricing_score"] >= cut)

    for i, t in enumerate(sorted([t for t in rows if rows[t]["rising"]],
                                 key=lambda t: -rows[t]["attr_slope_z"])):
        rows[t]["rising_rank"] = i + 1
    for i, t in enumerate(sorted(rows, key=lambda t: -rows[t]["pre_repricing_score"])):
        rows[t]["pre_repricing_rank"] = i + 1

    stocks = [{"ticker": t, "rising_rank": None, "pre_repricing_rank": None, **r} for t, r in rows.items()]
    stocks.sort(key=lambda s: -s["pre_repricing_score"])

    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "as_of": run_date,
        "window_weeks": window_weeks,
        "method": "theilsen_slope_orthogonalized_residual",
        "orthogonalization_beta": beta_used,
        "n_universe": len(stocks),
        "n_rising": sum(1 for s in stocks if s["rising"]),
        "n_pre_repricing": sum(1 for s in stocks if s["pre_repricing"]),
        "provisional_ic": None,  # forward IC tracked in observe-mode (Phase-5 eval)
        "stocks": stocks,
    }


# ── wiring: read prices + history, write artifact, email digest ──────────────

def _window_log_return(df, window_days: int) -> Optional[float]:
    """Log return over the trailing ``window_days`` of a price frame's Close
    (slices to the window — ``fetch_price_data`` only returns coarse periods)."""
    import pandas as pd

    if df is None or "Close" not in df.columns:
        return None
    s = df["Close"].dropna()
    if len(s) < 2:
        return None
    cutoff = s.index.max() - pd.Timedelta(days=window_days)
    win = s[s.index >= cutoff]
    if len(win) < 2:
        win = s  # whole frame is shorter than the window → use what we have
    a, b = float(win.iloc[0]), float(win.iloc[-1])
    if a <= 0 or b <= 0:
        return None
    return math.log(b / a)


def _fetch_period_for(window_weeks: int) -> str:
    """Smallest supported fetch_price_data period that covers the window."""
    need = window_weeks * 7 + 10
    return "3mo" if need <= 90 else ("6mo" if need <= 180 else "1y")


def _read_price_returns(tickers: list[str], window_weeks: int) -> tuple[dict, dict]:
    """W-week log returns for ``tickers`` + the sector ETFs, from ArcticDB.
    Returns ``(price_ret_by_ticker, sector_etf_ret_by_sector)``. Fail-soft: a
    price read failure yields empty maps (the signal degrades to rising-only)."""
    window_days = window_weeks * 7
    try:
        from data.fetchers.price_fetcher import fetch_price_data
        frames = fetch_price_data(sorted(set(tickers)) + SECTOR_ETFS,
                                  period=_fetch_period_for(window_weeks))
    except Exception as e:
        # §61 pre-persistence carve-out (config#1684): this runs BEFORE the
        # trajectory artifact is written, so it cannot raise without losing the
        # whole signal — degrades to rising-only. Non-fatal, but the failure now
        # lands on an ALARMED surface with a consumer, not a bare WARN.
        logger.warning("[trajectory] price read failed — signal degrades to rising-only: %s", e)
        from observe_alerts import publish_observe_alert
        publish_observe_alert(
            f"attractiveness_trajectory price read FAILED (non-fatal, signal "
            f"degrades to rising-only): {e}",
            source="research-runner:trajectory_price_read",
            dedup_key="trajectory_price_read_fail",
        )
        return {}, {}
    price_ret = {}
    for t in tickers:
        r = _window_log_return(frames.get(t), window_days)
        if r is not None:
            price_ret[t] = r
    etf_ret = {}
    for etf in SECTOR_ETFS:
        r = _window_log_return(frames.get(etf), window_days)
        if r is not None:
            etf_ret[etf] = r
    # remap ETF returns onto sector names so build_trajectory can look up by sector
    sector_etf_ret = {sec: etf_ret[etf] for sec, etf in _SECTOR_ETF.items() if etf in etf_ret}
    return price_ret, sector_etf_ret


def compute_and_write_trajectory(
    run_date: str,
    *,
    window_weeks: int = DEFAULT_WINDOW_WEEKS,
    bucket: str | None = None,
    s3_client: Any = None,
    send_email: bool = True,
) -> Optional[str]:
    """archive_writer entry point — read history + prices, build + write the
    trajectory artifact, send the weekly digest. Returns the dated S3 key (or
    None when there's not enough history yet). Caller wraps fail-soft."""
    from scoring.attractiveness_history import read_history
    from scoring.universe_board import _bucket, _client

    history = read_history(bucket=bucket, s3_client=s3_client)
    if history is None or history.empty or history["as_of"].nunique() < DEFAULT_MIN_POINTS:
        n = 0 if history is None or history.empty else int(history["as_of"].nunique())
        logger.info("[trajectory] only %d history cycle(s) — need %d; skipping (warm-up).",
                    n, DEFAULT_MIN_POINTS)
        return None

    tickers = sorted(history["ticker"].dropna().unique().tolist())
    price_ret, sector_etf_ret = _read_price_returns(tickers, window_weeks)
    artifact = build_trajectory(history, price_ret, sector_etf_ret,
                                run_date=run_date, window_weeks=window_weeks)

    s3 = _client(s3_client)
    b = _bucket(bucket)
    body = json.dumps(artifact, separators=(",", ":"), default=str).encode("utf-8")
    dated_key = f"scanner/universe/trajectory/{run_date}/trajectory.json"
    for key in (dated_key, "scanner/universe/trajectory/latest.json"):
        s3.put_object(Bucket=b, Key=key, Body=body, ContentType="application/json")
    logger.info("[trajectory] wrote %d names (%d rising, %d pre-repricing) → s3://%s/%s",
                artifact["n_universe"], artifact["n_rising"], artifact["n_pre_repricing"], b, dated_key)

    if send_email:
        try:
            _send_digest(artifact, run_date)
        except Exception as e:
            # §61 (config#1684): the trajectory artifact is already persisted
            # above, so a digest-email failure is a secondary-notification
            # miss — non-fatal, but loud on an alarmed surface, not a silent
            # WARN.
            logger.warning("[trajectory] digest email failed (non-fatal): %s", e)
            from observe_alerts import publish_observe_alert
            publish_observe_alert(
                f"attractiveness_trajectory digest email FAILED for {run_date} "
                f"(non-fatal, trajectory artifact already persisted): {e}",
                source="research-runner:trajectory_digest_email",
                dedup_key=f"trajectory_digest_email_fail:{run_date}",
            )
    return dated_key


def format_digest_markdown(artifact: dict, *, top_n: int = 10) -> str:
    """Thin digest markdown: top pre-repricing + rising, deep-linking to the
    console page (the established operator thin-digest pattern)."""
    stocks = artifact.get("stocks", [])
    pre = sorted([s for s in stocks if s.get("pre_repricing")],
                 key=lambda s: s.get("pre_repricing_rank") or 1e9)[:top_n]
    rising = sorted([s for s in stocks if s.get("rising")],
                    key=lambda s: s.get("rising_rank") or 1e9)[:top_n]
    link = f"{CONSOLE_BASE_URL}/Attractiveness_Trends"
    lines = [f"# Attractiveness trends — {artifact.get('as_of')}", ""]
    lines.append(f"_{artifact.get('n_rising', 0)} rising · {artifact.get('n_pre_repricing', 0)} "
                 f"pre-repricing · {artifact.get('window_weeks')}-week window._")
    lines.append("")
    lines.append("## 📈 Pre-repricing (rising attractiveness, price lagging sector)")
    if pre:
        for s in pre:
            rel = s.get("sector_rel_price_ret")
            rel_txt = "n/a" if rel is None else f"{rel * 100:+.1f}% vs sector"
            lines.append(f"- **{s['ticker']}** — residual {s['pre_repricing_score']:+.2f} · "
                         f"attr-slope {s['attr_slope']:+.3f}/wk · price {rel_txt}")
    else:
        lines.append("- _none this cycle_")
    lines.append("")
    lines.append("## ⬆️ Rising attractiveness")
    if rising:
        for s in rising:
            lines.append(f"- **{s['ticker']}** — slope {s['attr_slope']:+.3f}/wk "
                         f"(z {s['attr_slope_z']:+.2f})")
    else:
        lines.append("- _none this cycle_")
    lines.append("")
    lines.append(f"→ Full leaderboard + per-stock charts: {link}")
    return "\n".join(lines)


def _send_digest(artifact: dict, run_date: str) -> bool:
    from config import EMAIL_RECIPIENTS, EMAIL_SENDER
    from emailer.formatter import format_email
    from emailer.sender import send_email as _send

    md = format_digest_markdown(artifact)
    html, plain = format_email(md, run_date)
    subject = (f"📈 Attractiveness trends — {run_date} "
               f"({artifact.get('n_pre_repricing', 0)} pre-repricing)")
    return _send(subject, html, plain, EMAIL_RECIPIENTS, EMAIL_SENDER)
