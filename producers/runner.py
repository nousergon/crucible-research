"""Run the challenger research producers in shadow (config#1223 B3).

Invoked as a post-step in the Saturday research Lambda, AFTER the champion's
signals.json is written, with the PRIOR population snapshotted before the
champion mutated it. Each challenger's signals.json goes to the isolated
``signals_shadow/{producer}/{date}/`` prefix (never read by live trading).

FAIL-HARD (Brian ruling 2026-07-03, config#1683): per-producer isolation is
kept only so every producer gets its ATTEMPT (one crash doesn't starve the
other's artifact), but any gap — a producer that raised or emitted nothing —
raises ``ChallengerShadowGapError`` after the observe alert fires. An
experiment producer is still a PRODUCER under the no-silent-fails doctrine;
the 2026-06→07 weeks of empty ``signals_shadow/`` behind a WARN swallow are
exactly the breakage fail-soft invites. The champion's signals.json is already
persisted before this step runs, so failing the run loses no live deliverable.
"""

from __future__ import annotations

import logging

from observe_alerts import publish_observe_alert
from producers.registry import challenger_producers

logger = logging.getLogger(__name__)


class ChallengerShadowGapError(RuntimeError):
    """An always-on challenger producer failed to emit its shadow cohort."""


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
        except Exception as exc:  # noqa: BLE001 — isolation only; gap RAISES below
            logger.error(
                "[producers] challenger %s failed (other challengers still get "
                "their attempt; the gap raises after the loop — config#1683): %s",
                spec.name, exc, exc_info=True,
            )
            errors[spec.name] = str(exc)
    logger.info(
        "[producers] challenger shadows: wrote %d (%s), %d failed (%s)",
        len(written), list(written), len(errors), list(errors),
    )

    # FAIL-HARD on an always-on producer that emitted nothing (config#1403 +
    # config#1683). The challengers are OBSERVATION_REGISTRY ``always-on`` —
    # they must write one shadow cohort EVERY run. A producer that builds
    # nothing (raise → ``errors``) or is silently absent leaves
    # ``signals_shadow/`` incomplete, which is exactly why
    # ``research/producer_leaderboard/`` never accrued cohorts (the 2026-06-27
    # audit, then AGAIN on 2026-07-03 behind a WARN swallow). The observe alert
    # fires first (deduped page even if a caller catches), then the gap RAISES
    # so the run goes red instead of silently thin.
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
        raise ChallengerShadowGapError(
            f"challenger shadow gap on {run_date}: "
            f"missing/failed={missing or list(errors)} errors={errors} "
            f"(wrote={list(written)}) — experiment producers fail hard "
            f"(config#1683)"
        )

    return {"written": written, "errors": errors}
