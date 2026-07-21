"""Per-ticker quarantine contract (config#2247).

Brian's 2026-07-11 ruling amends all-agents-strict at the SCOPE level only: a
held ticker whose thesis update fails DETERMINISTICALLY is QUARANTINED —
omitted from signals.json with an explicit ``quarantined`` record, no
stale-thesis carry-forward — and the run completes for the rest, UNLESS the
run-level floor (> MAX_QUARANTINED_TICKERS, OR a whole failed/missing/partial
team) is breached, in which case the run still hard-fails.

Layers covered here:
  * score_aggregator — aggregation across teams + the per-ticker floor, and
    that a WHOLE-team failure still hard-fails (the other floor arm).
  * _build_signals_payload — the explicit-absence field + suppression of the
    stale-thesis carry-forward for a quarantined held ticker.

The raise contract of ``_update_thesis_for_held_stock`` (it raises a
``QuarantinableThesisError`` on deterministic failure, caught upstream) is
exercised in test_held_thesis_strict.py.
"""
from __future__ import annotations

import pytest

from agents.sector_teams.team_config import ALL_TEAM_IDS
from graph.research_graph import score_aggregator, _build_signals_payload
from config import MAX_QUARANTINED_TICKERS


def _state(team_outputs: dict) -> dict:
    return {
        "sector_team_outputs": team_outputs,
        "sector_modifiers": {},
        "sector_map": {},
    }


def _clean(team_id: str, quarantined=None) -> dict:
    return {
        "team_id": team_id,
        "recommendations": [],
        "thesis_updates": {},
        "error": None,
        "partial": False,
        "partial_reasons": [],
        "quarantined": quarantined or [],
    }


def _all_clean(extra=None) -> dict:
    out = {tid: _clean(tid) for tid in ALL_TEAM_IDS}
    if extra:
        out.update(extra)
    return out


def _q(ticker, team_id):
    return {
        "ticker": ticker,
        "team_id": team_id,
        "stage": "held_thesis_update",
        "reason": f"held_thesis_update for {ticker} failed deterministically",
    }


# ── score_aggregator: aggregation + floor ────────────────────────────────


class TestQuarantineFloor:
    def test_within_floor_completes_and_returns_quarantined(self):
        team = next(iter(ALL_TEAM_IDS))
        outs = _all_clean({team: _clean(team, quarantined=[_q("CRUS", team)])})
        result = score_aggregator(_state(outs))
        assert result["investment_theses"] == {}
        assert [q["ticker"] for q in result["quarantined"]] == ["CRUS"]
        # team_id is preserved for downstream attribution.
        assert result["quarantined"][0]["team_id"] == team

    def test_exactly_at_floor_still_completes(self):
        # MAX_QUARANTINED_TICKERS quarantined == at floor (not OVER) → completes.
        tickers = [f"T{i}" for i in range(MAX_QUARANTINED_TICKERS)]
        team = next(iter(ALL_TEAM_IDS))
        outs = _all_clean({
            team: _clean(team, quarantined=[_q(t, team) for t in tickers]),
        })
        result = score_aggregator(_state(outs))
        assert sorted(q["ticker"] for q in result["quarantined"]) == sorted(tickers)

    def test_over_floor_hard_fails(self):
        # One MORE than the floor → hard-fail the whole run.
        n = MAX_QUARANTINED_TICKERS + 1
        team = next(iter(ALL_TEAM_IDS))
        outs = _all_clean({
            team: _clean(team, quarantined=[_q(f"T{i}", team) for i in range(n)]),
        })
        with pytest.raises(RuntimeError, match="QUARANTINE-FLOOR"):
            score_aggregator(_state(outs))

    def test_quarantine_aggregates_across_teams(self):
        team_ids = list(ALL_TEAM_IDS)[:2]
        outs = _all_clean({
            team_ids[0]: _clean(team_ids[0], quarantined=[_q("AAA", team_ids[0])]),
            team_ids[1]: _clean(team_ids[1], quarantined=[_q("BBB", team_ids[1])]),
        })
        result = score_aggregator(_state(outs))
        assert sorted(q["ticker"] for q in result["quarantined"]) == ["AAA", "BBB"]

    def test_whole_team_failure_still_hard_fails_over_quarantine(self):
        # The OTHER floor arm: a failed team hard-fails regardless of quarantine
        # (the all-agents-strict team gate fires before the per-ticker floor).
        team_ids = list(ALL_TEAM_IDS)
        outs = _all_clean({
            team_ids[0]: {
                "team_id": team_ids[0],
                "recommendations": [],
                "thesis_updates": {},
                "error": "RecursionError: exhausted",
                "quarantined": [],
            },
        })
        with pytest.raises(RuntimeError, match="ALL-AGENTS-STRICT"):
            score_aggregator(_state(outs))

    def test_clean_run_has_empty_quarantine(self):
        result = score_aggregator(_state(_all_clean()))
        assert result["quarantined"] == []


# ── _build_signals_payload: explicit absence + no carry-forward ──────────


class TestSignalsPayloadQuarantine:
    def _payload_state(self, quarantined):
        # CRUS is a held/population ticker with a prior thesis. Absent a
        # quarantine it would be CARRIED FORWARD in the population pass.
        return {
            "run_date": "2026-07-11",
            "run_time": "2026-07-11T17:00:00Z",
            "investment_theses": {},
            "prior_theses": {
                "CRUS": {
                    "rating": "BUY",
                    "score": 71.0,
                    "conviction": "stable",
                    "thesis_summary": "stale bull case",
                    "team_id": "technology",
                },
            },
            "new_population": [{"ticker": "CRUS", "sector": "Technology"}],
            "sector_map": {"CRUS": "Technology"},
            "sector_ratings": {},
            "entry_theses": {},
            "exits": [],
            "advanced_tickers": [],
            "quarantined": quarantined,
        }

    def test_quarantined_held_ticker_is_not_carried_forward(self):
        state = self._payload_state([_q("CRUS", "technology")])
        payload = _build_signals_payload(state)
        # The stale prior thesis must NOT reappear as a signal / universe row.
        assert "CRUS" not in payload["signals"]
        assert all(u["ticker"] != "CRUS" for u in payload["universe"])
        assert all(b["ticker"] != "CRUS" for b in payload["buy_candidates"])

    def test_quarantined_field_records_the_absence(self):
        state = self._payload_state([_q("CRUS", "technology")])
        payload = _build_signals_payload(state)
        assert [q["ticker"] for q in payload["quarantined"]] == ["CRUS"]
        rec = payload["quarantined"][0]
        assert rec["team_id"] == "technology"
        assert rec["stage"] == "held_thesis_update"
        assert "deterministically" in rec["reason"]

    def test_without_quarantine_the_held_ticker_carries_forward(self):
        # Control: with no quarantine, the prior-thesis carry-forward still
        # happens (proves the exclusion above is what suppresses it).
        state = self._payload_state([])
        payload = _build_signals_payload(state)
        assert "CRUS" in payload["signals"]
        assert payload["quarantined"] == []
