"""Per-ticker quarantine contract (config#2247).

Brian's 2026-07-11 ruling amends all-agents-strict at the SCOPE level only: a
held ticker whose thesis update fails DETERMINISTICALLY is QUARANTINED —
omitted from signals.json with an explicit ``quarantined`` record, no
stale-thesis carry-forward — and the run completes for the rest, UNLESS the
run-level floor (> MAX_QUARANTINED_TICKERS, OR a whole failed/missing/partial
team) is breached, in which case the run still hard-fails.

Layers covered here:
  * _build_signals_payload — the explicit-absence field + suppression of the
    stale-thesis carry-forward for a quarantined held ticker.

The raise contract of ``_update_thesis_for_held_stock`` (it raises a
``QuarantinableThesisError`` on deterministic failure, caught upstream) is
exercised in test_held_thesis_strict.py.
"""
from __future__ import annotations

from agents.sector_teams.team_config import ALL_TEAM_IDS
from scoring.signals_payload import _build_signals_payload


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


# ── score_aggregator tests DELETED with the champion graph ──────────────
# (alpha-engine-config-I7827). `TestQuarantineFloor` exercised
# `graph/research_graph.py::score_aggregator` — the run-level quarantine floor
# and the whole-team hard-fail. That function was part of the retired arm
# (`agentic_sector_teams`, retired 2026-07-12) and was deleted with it, so
# the floor it enforced no longer exists to be tested. The half of this
# contract that IS live — the explicit-absence field and the suppression of
# stale-thesis carry-forward, both in `_build_signals_payload` — is below and
# is unchanged.

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
