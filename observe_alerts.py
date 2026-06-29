"""Loud-but-non-fatal alerts for always-on OBSERVE artifacts (config#1403).

The champion/challenger shadow producers + the scanner/producer leaderboards are
OBSERVE-ONLY and must NEVER raise into the live signal/eval path. But per the
no-silent-fails posture ([[feedback_no_silent_fails]] / ARCHITECTURE §37) a
failure — or a silent NON-emission — of an artifact the OBSERVATION_REGISTRY
marks ``always-on`` must be **LOUD**, not a WARN log nobody monitors. The
2026-06-27 Saturday audit (config#1403) found ``signals_shadow/`` empty and
neither ``scanner/leaderboard/`` nor ``research/producer_leaderboard/`` written
to S3 — yet every one of those failures only ever produced a swallowed WARN, so
the gap went unseen for weeks (its earn-its-keep cohort gate has ZERO data).

This wraps the canonical ``krepis.alerts.publish`` in a best-effort call so the
operator is paged (SNS + Telegram) within minutes of SF completion. It mirrors
the existing inline pattern in ``graph/research_graph.py`` (pillar-coverage /
pillar-distribution gates). Alerting is SECONDARY observability: if the publish
itself fails we log + return False — the WARN log + CW Logs alarm remain the
backstop — and we NEVER raise (the caller is on an observe-only, fail-soft path).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def publish_observe_alert(
    message: str,
    *,
    source: str,
    dedup_key: str | None = None,
    severity: str = "WARN",
) -> bool:
    """Publish a LOUD alert for an always-on observe-artifact failure/gap.

    Best-effort (SNS + Telegram via the canonical ``krepis.alerts.publish``).
    Returns ``True`` iff the alert was published; NEVER raises — alerting is
    secondary observability and the caller is on a fail-soft path.
    """
    try:
        from krepis.alerts import publish as alerts_publish

        kwargs: dict = {
            "message": message,
            "severity": severity,
            "source": source,
            "sns": True,
            "telegram": True,
        }
        if dedup_key is not None:
            kwargs["dedup_key"] = dedup_key
        alerts_publish(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001 — secondary observability, never fatal
        logger.warning(
            "[observe_alerts] loud alert publish failed (source=%s): %s — the "
            "WARN log + CW Logs alarm remain the failure surface",
            source,
            exc,
        )
        return False
