"""Guard for the ``scanner_evaluations`` row assembly (archive_writer).

Regression covered: the archive_writer used to RECONSTRUCT scanner-eval rows
from ``technical_scores`` + agent team-picks, which (a) left every dropped
name's ``filter_fail_reason`` NULL — the dashboard Scanner page bucketed all
~850 failures as "(unspecified)" — and (b) set ``quant_filter_pass`` from agent
team-picks rather than the scanner's own gate verdict, so the backtester's
e2e_lift graded scanner recall/lift against the wrong column.

``_build_scanner_eval_rows`` now joins ``run_quant_filter._last_eval_log`` (the
authoritative per-ticker scanner verdict). These tests pin that join so the
reason/scan_path/gate-flag data and the scanner-survival semantics cannot
silently regress again.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.research_graph import _build_scanner_eval_rows  # noqa: E402


def _eval_log():
    """A realistic scanner eval log: one survivor + one of each fail reason."""
    return [
        {"ticker": "PASS", "sector": "Tech", "quant_filter_pass": 1,
         "liquidity_pass": 1, "volatility_pass": 1, "scan_path": "momentum",
         "tech_score": 72.0, "atr_pct": 3.1, "current_price": 100.0,
         "avg_volume_20d": 5_000_000.0},
        {"ticker": "LIQ", "sector": "Tech", "quant_filter_pass": 0,
         "liquidity_pass": 0, "filter_fail_reason": "liquidity"},
        {"ticker": "VOL", "sector": "Health", "quant_filter_pass": 0,
         "liquidity_pass": 1, "volatility_pass": 0,
         "filter_fail_reason": "volatility_momentum", "tech_score": 65.0,
         "atr_pct": 12.0, "current_price": 50.0, "avg_volume_20d": 9_000_000.0},
        {"ticker": "BELOW", "sector": "Health", "quant_filter_pass": 0,
         "liquidity_pass": 1, "volatility_pass": 1,
         "filter_fail_reason": "below_thresholds", "tech_score": 40.0},
        {"ticker": "RANK", "sector": "Energy", "quant_filter_pass": 0,
         "liquidity_pass": 1, "volatility_pass": 1,
         "filter_fail_reason": "rank_cutoff", "scan_path": "momentum",
         "tech_score": 61.0},
        {"ticker": "NODATA", "sector": "Energy", "quant_filter_pass": 0,
         "liquidity_pass": 0, "filter_fail_reason": "no_data"},
    ]


def _rows(**overrides):
    kwargs = dict(
        scanner_universe=["PASS", "LIQ", "VOL", "BELOW", "RANK", "NODATA"],
        extra_override_tickers=[],
        technical_scores={},
        sector_map={"PASS": "Tech", "LIQ": "Tech", "VOL": "Health",
                    "BELOW": "Health", "RANK": "Energy", "NODATA": "Energy"},
        scanner_eval_log=_eval_log(),
        focus_lookup={},
        run_date="2026-06-26",
    )
    kwargs.update(overrides)
    rows = _build_scanner_eval_rows(**kwargs)
    return {r["ticker"]: r for r in rows}


def test_every_failed_name_carries_a_fail_reason():
    """The 857-"(unspecified)" regression: a failed name MUST carry a non-null
    filter_fail_reason whenever the scanner recorded one."""
    rows = _rows()
    failed = [r for r in rows.values() if r["quant_filter_pass"] == 0]
    assert failed, "fixture must contain failures"
    assert all(r["filter_fail_reason"] for r in failed), {
        r["ticker"]: r["filter_fail_reason"] for r in failed
    }


def test_quant_filter_pass_reflects_scanner_verdict_not_team_picks():
    """quant_filter_pass mirrors the scanner eval log (survival), independent of
    any agent selection — the column the backtester e2e_lift grades."""
    rows = _rows()
    assert rows["PASS"]["quant_filter_pass"] == 1
    assert sum(r["quant_filter_pass"] for r in rows.values()) == 1
    # A focus_lookup (agent selection) must NOT flip quant_filter_pass.
    rows2 = _rows(focus_lookup={"LIQ": {"focus_list_passed": 1, "agent_override": 0}})
    assert rows2["LIQ"]["quant_filter_pass"] == 0
    assert rows2["LIQ"]["focus_list_passed"] == 1


def test_scan_path_and_gate_flags_propagate():
    rows = _rows()
    assert rows["PASS"]["scan_path"] == "momentum"
    assert rows["RANK"]["scan_path"] == "momentum"
    assert rows["LIQ"]["liquidity_pass"] == 0
    assert rows["VOL"]["volatility_pass"] == 0
    # Absent flags are omitted (fall to the schema NOT NULL DEFAULT 1), never
    # fabricated as 0.
    assert "volatility_pass" not in rows["LIQ"]


def test_metrics_prefer_eval_log_then_state_technical_scores():
    rows = _rows(technical_scores={"BELOW": {"technical_score": 40.0, "rsi_14": 55.0}})
    # eval-log tech_score wins where present…
    assert rows["PASS"]["tech_score"] == 72.0
    # …and the state technical_scores slice fills coverage gaps in the log.
    assert rows["BELOW"]["rsi_14"] == 55.0


def test_override_ticker_outside_scan_universe_degrades_honestly():
    """An agent-override name not in the scanner universe (absent from the eval
    log) is recorded as not-scanner-evaluated, never a fabricated pass."""
    rows = _rows(
        extra_override_tickers=["OVR"],
        focus_lookup={"OVR": {"agent_override": 1, "focus_list_passed": 0}},
    )
    assert rows["OVR"]["quant_filter_pass"] == 0
    assert rows["OVR"]["filter_fail_reason"] is None
    assert rows["OVR"]["agent_override"] == 1


def test_empty_eval_log_does_not_crash():
    """A cycle where the stash is unavailable degrades to pass=0 / null reason
    rather than raising (fail-soft, with the archive_writer WARN as the surface)."""
    rows = _rows(scanner_eval_log=[])
    assert all(r["quant_filter_pass"] == 0 for r in rows.values())
    assert all(r["filter_fail_reason"] is None for r in rows.values())
