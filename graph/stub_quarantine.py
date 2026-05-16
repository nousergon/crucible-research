"""Stub-quarantine guard for promoted Research artifacts.

THE DANGEROUS BUG THIS CLOSES
─────────────────────────────
``s3://alpha-engine-research/signals/2026-05-15/signals.json`` (written
2026-05-16T17:08:46Z by the recovery-postfix194 run) shipped with
synthetic ``dry_run.py`` stub output PROMOTED AS REAL: every new
buy_candidate (GOOG / AFL / AXP / ABT / APD / ADBE / AMD) had a
``thesis_summary`` starting ``"[DRY-RUN] Strong fundamentals…"`` and
the morning email rendered them as real picks — on a run that was NOT
``dry_run_llm``.

Root cause (precise leak path):

  1. The default dry-run gate (``lambda/handler.py``) runs a synthetic
     stub-pass first via ``install_dry_run_stubs(archive)``.
  2. The stub-pass runs the FULL graph. ``sector_team_node`` calls
     ``ArchiveManager.save_sector_team_run`` the moment a team
     "succeeds" — and ``_stub_run_sector_team`` returns a
     normal-looking dict with ``error=None``. ``install_dry_run_stubs``
     only suppressed ``write_signals_json`` + ``upload_db``, NOT the
     per-team resume-persistence path, so the stub-pass PERSISTED
     synthetic ``[DRY-RUN]`` sector-team output to
     ``archive/sector_team_runs/{run_date}/{team_id}.json``.
  3. The subsequent REAL pass's ``sector_team_node`` resume
     short-circuit (#194) LOADED that stub-persisted output and
     promoted the synthetic theses straight into signals.json + the
     email — with ZERO real Haiku calls. (A real-pass 429 degrade
     under #194 isolation would also have promoted retained stub
     state; #194 isolation no longer exists post-rework, but the
     persistence leak above is the actual mechanism for 2026-05-15.)

DEFENSE IN DEPTH (both shipped in this rework)
──────────────────────────────────────────────
  * Structural: ``install_dry_run_stubs`` now also no-ops
    ``save_sector_team_run`` / ``save_agent_run`` so the stub-pass
    CANNOT write the resume keys at all.
  * This guard: a last-line assertion at the signals.json / email /
    DB write site. A promoted artifact may ONLY be produced by a
    fully-real, all-agents-complete pass. If ANY agent output is
    stub/synthetic (carries the ``DRY_RUN_MARKER``) or missing, the
    guard raises ``StubQuarantineError`` BEFORE any write — the run
    hard-fails (status:ERROR), nothing is promoted. No ``[DRY-RUN]``
    string can ever appear in a promoted artifact.
"""

from __future__ import annotations

import logging

from dry_run import DRY_RUN_MARKER

logger = logging.getLogger(__name__)


class StubQuarantineError(RuntimeError):
    """A promoted artifact contained synthetic stub output, or an
    expected real-agent output was missing. The Research run MUST
    hard-fail (no signals.json / email / DB write) rather than promote
    synthetic data as real."""


def _contains_marker(value: object) -> str | None:
    """Recursively scan ``value`` for the ``DRY_RUN_MARKER`` substring.

    Returns a short location/excerpt string on the FIRST hit (so the
    error names where the synthetic text was), or None when clean.
    Walks dicts / lists / tuples / sets; only str leaves are matched.
    """
    if isinstance(value, str):
        if DRY_RUN_MARKER in value:
            return value[:160]
        return None
    if isinstance(value, dict):
        for k, v in value.items():
            hit = _contains_marker(v)
            if hit is not None:
                return f"{k!r}: {hit}"
        return None
    if isinstance(value, (list, tuple, set)):
        for i, item in enumerate(value):
            hit = _contains_marker(item)
            if hit is not None:
                return f"[{i}] {hit}"
        return None
    return None


def assert_no_stub_output(
    *,
    signals_payload: dict,
    consolidated_report: str,
    state: dict,
) -> None:
    """Refuse to promote if any agent output is stub/synthetic or missing.

    Called at the signals.json / email / DB write boundary (the
    ``archive_writer`` node). Raises ``StubQuarantineError`` on:

      * the ``DRY_RUN_MARKER`` substring appearing ANYWHERE in the
        signals.json payload (every stub string embeds it), the
        consolidated email/report body, the per-ticker investment
        theses, or any persisted/in-memory sector-team output;
      * a missing sector team (any of ``ALL_TEAM_IDS`` absent from
        ``sector_team_outputs``) — a promoted artifact may only come
        from an all-agents-complete pass.

    A clean return is the ONLY way a write proceeds. This composes with
    the all-agents-strict score_aggregator gate (which already
    hard-fails on missing/failed/partial teams upstream): this guard is
    the structural backstop directly at the write site, independent of
    whichever upstream node first detects the gap.
    """
    # 1. Synthetic-marker scan across every promotable surface.
    surfaces: list[tuple[str, object]] = [
        ("signals_payload", signals_payload),
        ("consolidated_report", consolidated_report),
        ("investment_theses", state.get("investment_theses", {})),
        ("sector_team_outputs", state.get("sector_team_outputs", {})),
        ("entry_theses", state.get("entry_theses", {})),
        ("ic_decisions", state.get("ic_decisions", [])),
        ("macro_report", state.get("macro_report", "")),
    ]
    for name, surface in surfaces:
        hit = _contains_marker(surface)
        if hit is not None:
            msg = (
                f"STUB-QUARANTINE: synthetic dry-run output detected in "
                f"{name} ({DRY_RUN_MARKER!r} marker) — REFUSING to write "
                f"signals.json / send email / upload DB. A promoted "
                f"artifact may ONLY be produced by a fully-real, "
                f"all-agents-complete pass. This is the exact "
                f"2026-05-15 failure shape (stub thesis promoted as "
                f"real). First hit: {hit!r}"
            )
            logger.error("[stub_quarantine] %s", msg)
            raise StubQuarantineError(msg)

    # 2. All sector teams must have produced output. (score_aggregator's
    #    all-agents-strict gate already enforces this upstream; we
    #    re-assert at the write site so the structural guarantee holds
    #    even if a future refactor reorders nodes or a team is dropped
    #    by the reducer after aggregation.)
    from agents.sector_teams.team_config import ALL_TEAM_IDS

    team_outputs = state.get("sector_team_outputs", {}) or {}
    present = set(team_outputs)
    # Only assert once teams exist (unit/exit-only states have none).
    if present:
        missing = sorted(set(ALL_TEAM_IDS) - present)
        if missing:
            msg = (
                f"STUB-QUARANTINE: {len(missing)} sector team(s) missing "
                f"from sector_team_outputs at the write site "
                f"({missing}) — REFUSING to promote an incomplete run. "
                f"A promoted artifact requires every agent to have run "
                f"for real."
            )
            logger.error("[stub_quarantine] %s", msg)
            raise StubQuarantineError(msg)
        # A team carrying an error / partial flag must never reach the
        # write site (score_aggregator should have hard-failed first);
        # assert defensively.
        degraded = [
            tid for tid, out in team_outputs.items()
            if isinstance(out, dict)
            and (out.get("error") or out.get("partial"))
        ]
        if degraded:
            msg = (
                f"STUB-QUARANTINE: sector team(s) {sorted(degraded)} "
                f"reached the write site still flagged failed/partial "
                f"— REFUSING to promote. all-agents-strict requires "
                f"every agent to have produced real output."
            )
            logger.error("[stub_quarantine] %s", msg)
            raise StubQuarantineError(msg)

    logger.info(
        "[stub_quarantine] clean — all agents produced real output, "
        "no %s marker in any promotable surface; write may proceed",
        DRY_RUN_MARKER,
    )
