"""Stub-quarantine regression — the dangerous 2026-05-15 bug.

s3://alpha-engine-research/signals/2026-05-15/signals.json (written
2026-05-16T17:08:46Z by the recovery-postfix194 run) shipped synthetic
``dry_run.py`` stub output PROMOTED AS REAL: every new buy_candidate
(GOOG / AFL / AXP / ABT / APD / ADBE / AMD) had a ``thesis_summary``
starting ``"[DRY-RUN] Strong fundamentals…"`` and the email rendered
them as real picks — on a run that was NOT ``dry_run_llm``.

These tests pin BOTH halves of the structural fix:

  1. ``install_dry_run_stubs`` now also no-ops ``save_sector_team_run``
     / ``save_agent_run`` so the stub-pass CANNOT write the resume
     persistence keys that the real pass would otherwise load and
     promote (THE precise leak mechanism).
  2. ``graph.stub_quarantine.assert_no_stub_output`` refuses to write
     signals.json / send email / upload DB if the ``[DRY-RUN`` marker
     appears anywhere in a promotable surface, or a sector team is
     missing — using the EXACT 2026-05-15 failure shape.

Uses the ``monkeypatch`` fixture (NOT ``unittest.mock.patch``).
"""

from __future__ import annotations

import pytest

from agents.sector_teams.team_config import ALL_TEAM_IDS
from dry_run import DRY_RUN_MARKER, install_dry_run_stubs
from graph.stub_quarantine import (
    StubQuarantineError,
    assert_no_stub_output,
)


def _full_clean_team_outputs():
    return {
        tid: {
            "team_id": tid,
            "recommendations": [],
            "thesis_updates": {},
            "error": None,
            "partial": False,
            "partial_reasons": [],
        }
        for tid in ALL_TEAM_IDS
    }


# ── Part 1: stub-pass cannot persist resume keys ──────────────────────────


class _FakeArchive:
    """Stands in for ArchiveManager — records whether the persistence
    methods were patched to no-ops by install_dry_run_stubs."""

    def __init__(self):
        self.persisted: list[tuple] = []

    def upload_db(self, *a, **k):
        self.persisted.append(("upload_db",))

    def write_signals_json(self, *a, **k):
        self.persisted.append(("write_signals_json",))

    def save_sector_team_run(self, run_date, team_id, output):
        self.persisted.append(("save_sector_team_run", team_id))

    def save_agent_run(self, run_date, agent_id, output):
        self.persisted.append(("save_agent_run", agent_id))


def test_install_dry_run_stubs_noops_resume_persistence():
    """The precise leak fix: the stub-pass MUST NOT be able to write
    save_sector_team_run / save_agent_run (those keys are what the real
    pass's resume short-circuit loads — that is how the 2026-05-15 stub
    thesis got promoted)."""
    arch = _FakeArchive()
    restore = install_dry_run_stubs(arch)
    try:
        # Post-install these are no-ops — calling them records nothing.
        arch.save_sector_team_run("2026-05-15", "technology", {"x": 1})
        arch.save_agent_run("2026-05-15", "cio", {"x": 1})
        arch.write_signals_json("2026-05-15", "", {})
        arch.upload_db("2026-05-15")
    finally:
        restore()

    assert arch.persisted == [], (
        "stub-pass persisted state — this is the exact leak path that "
        "promoted synthetic [DRY-RUN] theses on 2026-05-15"
    )

    # After restore the real methods work again (record calls).
    arch.save_sector_team_run("2026-05-15", "technology", {"x": 1})
    assert ("save_sector_team_run", "technology") in arch.persisted


# ── Part 2: write-site quarantine guard ───────────────────────────────────


def test_clean_full_run_passes():
    """The only shape that promotes: all 6 teams present, none
    degraded, no [DRY-RUN] marker anywhere."""
    assert_no_stub_output(
        signals_payload={"signals": {}, "buy_candidates": []},
        consolidated_report="Real research narrative for the week.",
        state={"sector_team_outputs": _full_clean_team_outputs()},
    )  # must not raise


def test_2026_05_15_exact_failure_shape_blocks_signals_json():
    """The EXACT 2026-05-15 promoted-stub shape: buy_candidates with
    thesis_summary starting '[DRY-RUN] Strong fundamentals'. The guard
    must refuse to write signals.json."""
    signals_payload = {
        "signals": {},
        "buy_candidates": [
            {"ticker": t, "thesis_summary":
             "[DRY-RUN] Strong fundamentals, attractive valuation"}
            for t in ("GOOG", "AFL", "AXP", "ABT", "APD", "ADBE", "AMD")
        ],
    }
    with pytest.raises(StubQuarantineError) as exc:
        assert_no_stub_output(
            signals_payload=signals_payload,
            consolidated_report="",
            state={"sector_team_outputs": _full_clean_team_outputs()},
        )
    assert DRY_RUN_MARKER in str(exc.value)
    assert "signals_payload" in str(exc.value)


def test_marker_in_email_body_blocks():
    with pytest.raises(StubQuarantineError, match="consolidated_report"):
        assert_no_stub_output(
            signals_payload={"signals": {}},
            consolidated_report="Top pick: NVDA — [DRY-RUN] Synthetic",
            state={"sector_team_outputs": _full_clean_team_outputs()},
        )


def test_marker_in_sector_team_output_blocks():
    """The leak's in-memory shape: a resumed team carries synthetic
    bull_case text from the stub-persisted output."""
    outs = _full_clean_team_outputs()
    outs["technology"]["recommendations"] = [
        {"ticker": "NVDA", "bull_case": "[DRY-RUN] Strong fundamentals"}
    ]
    with pytest.raises(StubQuarantineError, match="sector_team_outputs"):
        assert_no_stub_output(
            signals_payload={"signals": {}},
            consolidated_report="",
            state={"sector_team_outputs": outs},
        )


def test_marker_in_investment_theses_blocks():
    with pytest.raises(StubQuarantineError, match="investment_theses"):
        assert_no_stub_output(
            signals_payload={"signals": {}},
            consolidated_report="",
            state={
                "sector_team_outputs": _full_clean_team_outputs(),
                "investment_theses": {
                    "GOOG": {"bull_case": "[DRY-RUN] Strong fundamentals"}
                },
            },
        )


def test_missing_sector_team_blocks_promotion():
    """A promoted artifact requires every agent to have run — a missing
    team is fatal at the write site too (independent of the upstream
    score_aggregator gate)."""
    partial = {"technology": _full_clean_team_outputs()["technology"]}
    with pytest.raises(StubQuarantineError, match="missing"):
        assert_no_stub_output(
            signals_payload={"signals": {}},
            consolidated_report="",
            state={"sector_team_outputs": partial},
        )


def test_failed_team_at_write_site_blocks():
    outs = _full_clean_team_outputs()
    outs["financials"]["error"] = "RateLimitError 429"
    with pytest.raises(StubQuarantineError, match="failed/partial"):
        assert_no_stub_output(
            signals_payload={"signals": {}},
            consolidated_report="",
            state={"sector_team_outputs": outs},
        )


def test_empty_team_outputs_does_not_falsely_trip():
    """Unit/exit-only states have no teams — the team-coverage assertion
    only fires once teams exist (so non-pipeline callers don't break)."""
    assert_no_stub_output(
        signals_payload={"signals": {}},
        consolidated_report="clean",
        state={"sector_team_outputs": {}},
    )  # must not raise
