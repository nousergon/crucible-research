"""Tests for scoring/attractiveness_trajectory.py — the orthogonalized
factor-momentum signal.

Locks:
  1. Orthogonality — same attractiveness slope, but the name whose price has NOT
     run (price flat vs sector) gets a HIGHER pre_repricing residual than the one
     whose price already ran (residual isolates improvement-not-in-price).
  2. Sector-neutralization — sector_rel_price_ret = stock return − sector-ETF.
  3. Theil-Sen robustness to a single noisy week.
  4. Flags (rising / pre_repricing) + min-points quality gate + schema.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.attractiveness_trajectory import build_trajectory, format_digest_markdown  # noqa: E402

_DATES = ["2026-05-08", "2026-05-15", "2026-05-22", "2026-05-29", "2026-06-05"]


def _history(series_by_ticker: dict[str, list], sector="Information Technology"):
    rows = []
    for ticker, ys in series_by_ticker.items():
        for d, y in zip(_DATES, ys):
            rows.append({"as_of": d, "ticker": ticker, "attractiveness_raw": y, "sector": sector})
    return pd.DataFrame(rows)


def _build(**kw):
    # UP / UP2 share the SAME rising slope; UP's price is flat, UP2's price ran.
    hist = _history({
        "UP":   [-1.0, -0.5, 0.0, 0.5, 1.0],   # slope +0.5/wk
        "UP2":  [-1.0, -0.5, 0.0, 0.5, 1.0],   # slope +0.5/wk
        "FLAT": [0.0, 0.0, 0.0, 0.0, 0.0],     # slope 0
        "DOWN": [1.0, 0.5, 0.0, -0.5, -1.0],   # slope -0.5
    })
    price_ret = {"UP": 0.0, "UP2": 0.20, "FLAT": 0.0, "DOWN": -0.05}
    sector_etf_ret = {"Information Technology": 0.0}  # sector flat → sector_rel == price_ret
    return build_trajectory(hist, price_ret, sector_etf_ret, run_date="2026-06-05", **kw)


def _by(art):
    return {s["ticker"]: s for s in art["stocks"]}


def test_orthogonality_residual_isolates_unpriced_improvement():
    b = _by(_build())
    # Same slope, but UP's price hasn't moved while UP2's ran → UP residual higher.
    assert b["UP"]["attr_slope"] == b["UP2"]["attr_slope"]
    assert b["UP"]["pre_repricing_score"] > b["UP2"]["pre_repricing_score"]
    # UP is the pre-repricing pick (rising + top residual); UP2 is not.
    assert b["UP"]["pre_repricing"] is True
    assert b["UP2"]["pre_repricing"] is False


def test_sector_neutralization():
    b = _by(_build())
    # sector flat (0) → sector_rel == raw price return.
    assert b["UP2"]["sector_rel_price_ret"] == 0.20
    assert b["DOWN"]["sector_rel_price_ret"] == -0.05


def test_flags_rising_and_quality_gate():
    b = _by(_build())
    assert b["UP"]["rising"] is True and b["UP"]["slope_significant"] is True
    assert b["FLAT"]["rising"] is False    # zero slope
    assert b["DOWN"]["rising"] is False    # negative slope
    # min-points gate: a ticker with < min_points is excluded entirely.
    short = pd.DataFrame([{"as_of": d, "ticker": "SHORT", "attractiveness_raw": v,
                           "sector": "Information Technology"}
                          for d, v in zip(_DATES[:3], [0.0, 0.5, 1.0])])
    hist = pd.concat([_history({"UP": [-1.0, -0.5, 0.0, 0.5, 1.0]}), short], ignore_index=True)
    art = build_trajectory(hist, {"UP": 0.0}, {"Information Technology": 0.0},
                           run_date="2026-06-05", min_points=4)
    assert "SHORT" not in _by(art)


def test_theilsen_robust_to_noise():
    # One whipsaw week must not flip the slope sign.
    hist = _history({"UP": [-1.0, -0.5, 5.0, 0.5, 1.0]})  # spike at midpoint
    b = _by(build_trajectory(hist, {"UP": 0.0}, {"Information Technology": 0.0},
                             run_date="2026-06-05"))
    assert b["UP"]["attr_slope"] > 0  # Theil-Sen ignores the outlier


def test_schema_and_counts():
    art = _build()
    assert art["schema_version"] == 1
    assert art["method"] == "theilsen_slope_orthogonalized_residual"
    assert art["window_weeks"] == 8 and art["n_universe"] == 4
    assert art["n_rising"] >= 1 and art["n_pre_repricing"] >= 1
    # sorted by pre_repricing_score desc.
    scores = [s["pre_repricing_score"] for s in art["stocks"]]
    assert scores == sorted(scores, reverse=True)


def test_digest_markdown_renders():
    md = format_digest_markdown(_build())
    assert "Pre-repricing" in md and "Rising attractiveness" in md
    assert "UP" in md and "/Attractiveness_Trends" in md


def test_price_read_failure_degrades_gracefully_and_fires_observe_alert():
    """§61 pre-persistence carve-out (config#1684): a price-fetch failure must
    not raise (the signal degrades to rising-only) AND must fire the ALARMED
    observe surface, not just a WARN log."""
    from unittest.mock import patch

    from scoring.attractiveness_trajectory import _read_price_returns

    with patch(
        "data.fetchers.price_fetcher.fetch_price_data",
        side_effect=RuntimeError("arcticdb unavailable"),
    ), patch("observe_alerts.publish_observe_alert") as mock_alert:
        price_ret, sector_etf_ret = _read_price_returns(["UP", "FLAT"], window_weeks=8)

    assert price_ret == {} and sector_etf_ret == {}
    assert mock_alert.called, "price-read failure must fire an observe alert (§61)"
    kwargs = mock_alert.call_args.kwargs
    assert "trajectory_price_read" in kwargs.get("source", "")
    assert "arcticdb unavailable" in mock_alert.call_args.args[0]
