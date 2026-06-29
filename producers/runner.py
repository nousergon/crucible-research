"""Run the challenger research producers in shadow (config#1223 B3).

Invoked as a best-effort post-step in the Saturday research Lambda, AFTER the
champion's signals.json is written, with the PRIOR population snapshotted before
the champion mutated it. Each challenger's signals.json goes to the isolated
``signals_shadow/{producer}/{date}/`` prefix (never read by live trading).

Fully fail-soft PER producer (no-silent-fails: the failure is recorded — WARN +
the returned ``errors`` map): a challenger raising never affects the champion's
already-persisted deliverable or the other challengers.
"""

from __future__ import annotations

import logging

from observe_alerts import publish_observe_alert
from producers.registry import challenger_producers

logger = logging.getLogger(__name__)


def run_challengers(
    archive_manager,
    run_date: str,
    *,
    run_time: str = "",
    population: list[dict] | None = None,
) -> dict:
    """Build + write every challenger producer's shadow signals. Returns
    ``{"written": {producer: s3_key}, "errors": {producer: reason}}``."""
    generated_at = run_time or run_date
    written: dict[str, str] = {}
    errors: dict[str, str] = {}
    for spec in challenger_producers():
        try:
            payload = spec.build(
                run_date, archive_manager, run_time=run_time, population=population,
            )
            key = archive_manager.write_shadow_signals_json(
                spec.name, run_date, generated_at, payload,
            )
            written[spec.name] = key
        except Exception as exc:  # noqa: BLE001 — shadow is best-effort
            logger.warning(
                "[producers] challenger %s failed (shadow mode, non-fatal, "
                "champion + other challengers unaffected): %s", spec.name, exc,
            )
            errors[spec.name] = str(exc)
    logger.info(
        "[producers] challenger shadows: wrote %d (%s), %d failed (%s)",
        len(written), list(written), len(errors), list(errors),
    )

    # Fail-LOUD on an always-on producer that emitted nothing (config#1403).
    # The challengers are OBSERVATION_REGISTRY ``always-on`` — they are expected
    # to write one shadow cohort EVERY run. A producer that builds nothing (raise
    # → ``errors``) or is silently absent leaves ``signals_shadow/`` incomplete,
    # which is exactly why ``research/producer_leaderboard/`` never accrued cohorts
    # (the 2026-06-27 audit). Until now that was only a WARN log; surface it loudly
    # so the gap is seen within minutes of SF completion, not after weeks of no
    # data. Best-effort + deduped per run-date; NEVER raises into the live path.
    expected = [spec.name for spec in challenger_producers()]
    missing = [name for name in expected if name not in written]
    if missing or errors:
        publish_observe_alert(
            message=(
                f"[producers] challenger shadow gap on {run_date}: only "
                f"{len(written)}/{len(expected)} always-on producers emitted "
                f"(wrote={list(written)}, missing/failed={missing}, "
                f"errors={errors}). signals_shadow/ is incomplete → "
                f"research/producer_leaderboard/ cohorts will not accrue. "
                f"Investigate the failing producer(s) (config#1403)."
            ),
            source="research:challenger_producers",
            dedup_key=f"challenger_shadow_gap:{run_date}",
        )

    return {"written": written, "errors": errors}
