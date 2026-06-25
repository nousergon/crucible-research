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
    return {"written": written, "errors": errors}
