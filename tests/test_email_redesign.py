"""Tests for the 2026-05-16 consolidated-brief email redesign.

Pins the four decided-with-user changes:

  1. "NOTABLE DEVELOPMENTS" section is gone; a per-ticker development note
     (e.g. an exit reason) is folded into that ticker's Universe-Ratings
     Rationale cell instead.
  2. Universe Ratings table has two clean axes:
       - Status ∈ {New, Existing}        (lifecycle only)
       - Recommendation ∈ {Buy, Hold, Sell}  (action)
     An exit renders as Existing + Sell.
  3. Bench BUY-recs (rated BUY, no slot) appear in a separate
     "BUY CANDIDATES (NO SLOT)" subsection and NOT in the main table.
  4. The static one-shot regime snapshot is replaced by a REGIME TREND
     block built from the last N weekly substrate artifacts; it degrades
     gracefully with zero / one artifact and never crashes the brief.

Convention: ``monkeypatch`` fixture only — NEVER ``unittest.mock.patch``
(documented full-suite bleed via ``sys.modules`` reassignment in
``tests/test_dry_run.py``; see MEMORY feedback note).
"""
from __future__ import annotations

import re

import pytest

from graph.research_graph import (
    _build_notable_developments,
    _build_regime_trend,
    consolidator,
)


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeArchiveManager:
    """Stands in for ArchiveManager.list_regime_substrates only."""

    def __init__(self, artifacts: list[dict] | Exception):
        self._artifacts = artifacts

    def list_regime_substrates(self, n_recent: int = 8) -> list[dict]:
        if isinstance(self._artifacts, Exception):
            raise self._artifacts
        return self._artifacts[-n_recent:]


def _artifact(
    *,
    run_id: str,
    trading_day: str,
    argmax: str,
    iz: float,
    change: bool = False,
    weeks_in_state: int = 1,
    guardrails: dict | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "trading_day": trading_day,
        "calendar_date": trading_day,
        "hmm": {
            "argmax": argmax,
            "weeks_in_current_state": weeks_in_state,
            "probs": {"bear": 0.0, "neutral": 1.0, "bull": 0.0},
        },
        "composite": {"intensity_z": iz, "implied_severity": "risk_on_tilted"},
        "bocpd": {"change_signal": change},
        "guardrails": guardrails or {
            "vix_caution_breached": False,
            "active_severity_floor": None,
        },
    }


def _base_state(**overrides) -> dict:
    state = {
        "run_date": "2026-05-16",
        "market_regime": "neutral",
        "macro_report": "Macro narrative stays unchanged.",
        "archive_manager": None,
        "current_population": [],
        "new_population": [],
        "exits": [],
        "investment_theses": {},
        "prior_theses": {},
        "entry_theses": {},
        "sector_team_outputs": {},
        "ic_decisions": [],
        "technical_scores": {},
        "sector_ratings": {},
    }
    state.update(overrides)
    return state


# ── Change 2: Status / Recommendation axes ──────────────────────────────────


class TestUniverseRatingsAxes:
    def test_status_and_recommendation_value_domains(self):
        state = _base_state(
            current_population=[{"ticker": "HELD"}],
            new_population=[{"ticker": "HELD"}, {"ticker": "FRESH"}],
            investment_theses={
                "HELD": {"rating": "HOLD", "final_score": 71, "bull_case": "steady"},
                "FRESH": {"rating": "BUY", "final_score": 88, "bull_case": "momentum"},
            },
            entry_theses={"FRESH": {"bull_case": "new momentum entry"}},
        )
        report = consolidator(state)["consolidated_report"]

        # Parse the Universe Ratings table rows.
        statuses, recs = set(), set()
        for line in report.splitlines():
            if line.startswith("| ") and " | " in line and "Ticker" not in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) == 5 and cells[1] in ("New", "Existing"):
                    statuses.add(cells[1])
                    recs.add(cells[2])

        assert statuses, "no parsed Universe Ratings rows"
        assert statuses <= {"New", "Existing"}, statuses
        assert recs <= {"Buy", "Hold", "Sell"}, recs
        # No screaming-caps lifecycle labels leak through.
        assert "| NEW |" not in report
        assert "| UPDATED |" not in report
        assert "| BUY REC |" not in report

    def test_score_column_labeled_and_legend_present(self):
        state = _base_state(
            new_population=[{"ticker": "AAA"}],
            investment_theses={"AAA": {"rating": "HOLD", "final_score": 60}},
        )
        report = consolidator(state)["consolidated_report"]
        assert "Score (0–100)" in report
        assert "composite of quant + qual sub-scores" in report
        assert "drives population ranking" in report

    def test_exit_renders_existing_and_sell(self):
        state = _base_state(
            current_population=[{"ticker": "GONE"}],
            new_population=[],
            exits=[{"ticker_out": "GONE", "score_out": 42,
                    "reason": "min_rotation_floor breached"}],
        )
        report = consolidator(state)["consolidated_report"]
        exit_row = [
            ln for ln in report.splitlines()
            if ln.startswith("| GONE |")
        ]
        assert exit_row, "GONE exit row missing"
        cells = [c.strip() for c in exit_row[0].strip("|").split("|")]
        assert cells[1] == "Existing"
        assert cells[2] == "Sell"


# ── Change 3: Bench buy-recs in their own subsection ────────────────────────


class TestBenchBuyCandidates:
    def test_bench_buy_recs_in_subsection_not_main_table(self):
        state = _base_state(
            current_population=[{"ticker": "HELD"}],
            new_population=[{"ticker": "HELD"}],
            investment_theses={
                "HELD": {"rating": "HOLD", "final_score": 70},
                "BENCH": {"rating": "BUY", "final_score": 91,
                          "bull_case": "great but no slot"},
            },
        )
        report = consolidator(state)["consolidated_report"]

        assert "## c.1. BUY CANDIDATES (NO SLOT)" in report
        assert "Rated Buy but not currently held" in report

        # BENCH must appear AFTER the subsection header, not in the main
        # Universe Ratings table.
        idx_universe = report.index("## c. UNIVERSE RATINGS")
        idx_subsection = report.index("## c.1. BUY CANDIDATES (NO SLOT)")
        main_table = report[idx_universe:idx_subsection]
        subsection = report[idx_subsection:]
        assert "BENCH" not in main_table
        assert "BENCH" in subsection

    def test_subsection_omitted_when_no_bench_recs(self):
        state = _base_state(
            new_population=[{"ticker": "HELD"}],
            investment_theses={"HELD": {"rating": "HOLD", "final_score": 70}},
        )
        report = consolidator(state)["consolidated_report"]
        assert "BUY CANDIDATES (NO SLOT)" not in report


# ── Change 1: Notable Developments dropped, folded into rationale ───────────


class TestNotableDevelopmentsFolded:
    def test_no_notable_developments_header(self):
        state = _base_state(
            new_population=[{"ticker": "AAA"}],
            investment_theses={"AAA": {"rating": "HOLD", "final_score": 50}},
            sector_team_outputs={
                "tech": {"recommendations": [
                    {"ticker": "AAA", "bull_case": "huge upside",
                     "conviction": 85},
                ]},
            },
        )
        report = consolidator(state)["consolidated_report"]
        assert "NOTABLE DEVELOPMENTS" not in report

    def test_exit_reason_folded_into_ticker_rationale(self):
        # An exit whose folded note adds info beyond the base rationale:
        # a high-conviction sector-team note on the same ticker stays
        # attached to ZZZ's own Rationale cell (not a separate section).
        state = _base_state(
            current_population=[{"ticker": "ZZZ"}],
            new_population=[],
            exits=[{"ticker_out": "ZZZ", "score_out": 30,
                    "reason": "rotated for higher-score peer"}],
            sector_team_outputs={
                "tech": {"recommendations": [
                    {"ticker": "ZZZ", "bull_case": "valuation reset complete",
                     "conviction": 88},
                ]},
            },
        )
        report = consolidator(state)["consolidated_report"]
        zzz_row = [ln for ln in report.splitlines() if ln.startswith("| ZZZ |")]
        assert zzz_row
        # Base exit reason present, plus the distinct development note
        # folded into the SAME ticker's rationale cell.
        assert "rotated for higher-score peer" in zzz_row[0]
        assert "High conviction" in zzz_row[0]
        assert "valuation reset complete" in zzz_row[0]
        # And no standalone section.
        assert "NOTABLE DEVELOPMENTS" not in report

    def test_redundant_exit_note_not_echoed(self):
        # When the only note restates the exit reason verbatim, the cell
        # is not doubled up.
        state = _base_state(
            current_population=[{"ticker": "DUP"}],
            new_population=[],
            exits=[{"ticker_out": "DUP", "score_out": 30,
                    "reason": "min_rotation_floor breached"}],
        )
        report = consolidator(state)["consolidated_report"]
        dup_row = [ln for ln in report.splitlines() if ln.startswith("| DUP |")][0]
        assert dup_row.count("min_rotation_floor breached") == 1

    def test_builder_returns_per_ticker_mapping(self):
        state = _base_state(
            exits=[{"ticker_out": "EXT", "reason": ">2 ATR move on news spike"}],
            ic_decisions=[{"decision": "ADVANCE", "ticker": "ADV",
                           "rationale": "catalyst confirmed"}],
            sector_team_outputs={
                "tech": {"recommendations": [
                    {"ticker": "HC", "bull_case": "strong", "conviction": 90},
                ]},
            },
        )
        notes = _build_notable_developments(state)
        assert isinstance(notes, dict)
        assert "EXT" in notes and any("ATR" in n for n in notes["EXT"])
        assert "ADV" in notes and any("CIO advance" in n for n in notes["ADV"])
        assert "HC" in notes and any("High conviction" in n for n in notes["HC"])


# ── Change 4: Regime trend block ────────────────────────────────────────────


class TestRegimeTrendBlock:
    def test_renders_n_weeks_from_artifacts(self):
        artifacts = [
            _artifact(run_id="2604010000", trading_day="2026-04-01",
                      argmax="bull", iz=-0.4, weeks_in_state=3),
            _artifact(run_id="2604080000", trading_day="2026-04-08",
                      argmax="neutral", iz=0.1, weeks_in_state=1),
            _artifact(run_id="2604150000", trading_day="2026-04-15",
                      argmax="neutral", iz=0.6, change=True, weeks_in_state=2),
        ]
        am = _FakeArchiveManager(artifacts)
        lines = _build_regime_trend(am, n_weeks=8)
        joined = "\n".join(lines)
        assert "2026-04-01" in joined
        assert "2026-04-08" in joined
        assert "2026-04-15" in joined
        # Continuous dial values rendered.
        assert "-0.40" in joined
        assert "+0.60" in joined
        # BOCPD change surfaced.
        assert "yes" in joined
        # Summary line: rising over the window (-0.4 → +0.6).
        assert "**Summary:**" in joined
        assert "rising" in joined

    def test_consolidator_includes_regime_trend_section(self):
        am = _FakeArchiveManager([
            _artifact(run_id="2605010000", trading_day="2026-05-01",
                      argmax="neutral", iz=0.2),
            _artifact(run_id="2605080000", trading_day="2026-05-08",
                      argmax="neutral", iz=0.3),
        ])
        state = _base_state(archive_manager=am)
        report = consolidator(state)["consolidated_report"]
        assert "## a.0. REGIME TREND" in report
        # Static one-shot "P(neutral)=1.00" style snapshot is gone.
        assert not re.search(r"P\(neutral\)=", report)

    def test_zero_artifacts_degrades_gracefully(self):
        am = _FakeArchiveManager([])
        lines = _build_regime_trend(am, n_weeks=8)
        assert lines == ["_Regime substrate unavailable — no weekly artifacts found._"]
        # Brief still generates.
        state = _base_state(archive_manager=am)
        report = consolidator(state)["consolidated_report"]
        assert "Regime substrate unavailable" in report
        assert "## c. UNIVERSE RATINGS" in report

    def test_one_artifact_degrades_to_no_trend_message(self):
        am = _FakeArchiveManager([
            _artifact(run_id="2605160000", trading_day="2026-05-16",
                      argmax="neutral", iz=0.52),
        ])
        lines = _build_regime_trend(am, n_weeks=8)
        joined = "\n".join(lines)
        assert "Single artifact only — no trend yet" in joined
        assert "+0.52" in joined
        assert "**Summary:**" not in joined  # no trend summary with 1 point

    def test_no_archive_manager_returns_empty(self):
        assert _build_regime_trend(None, n_weeks=8) == []

    def test_list_failure_degrades_not_crash(self):
        am = _FakeArchiveManager(RuntimeError("S3 down"))
        # _build_regime_trend swallows the exception → empty list.
        assert _build_regime_trend(am, n_weeks=8) == []
        # And the brief still generates with the section skipped.
        state = _base_state(archive_manager=am)
        report = consolidator(state)["consolidated_report"]
        assert "## c. UNIVERSE RATINGS" in report
        assert "## a.0. REGIME TREND" not in report

    def test_guardrail_breach_surfaced_in_summary(self):
        am = _FakeArchiveManager([
            _artifact(run_id="2605010000", trading_day="2026-05-01",
                      argmax="neutral", iz=0.1),
            _artifact(run_id="2605080000", trading_day="2026-05-08",
                      argmax="bear", iz=1.4,
                      guardrails={"vix_bear_breached": True,
                                  "active_severity_floor": "bear"}),
        ])
        lines = _build_regime_trend(am, n_weeks=8)
        joined = "\n".join(lines)
        assert "guardrail breached: vix_bear_breached" in joined
