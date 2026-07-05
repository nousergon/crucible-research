"""Dual-channel ops alerts for research (SNS + flow-doctor forum topics).

Migration arc: config#1740 T3 / config#1749 — retire raw ``telegram=True`` and
``send_message`` bypasses to Telegram General.
"""

from __future__ import annotations

import logging
import os

from alpha_engine_lib.logging import get_flow_doctor

logger = logging.getLogger(__name__)


def _normalize_flow_doctor_severity(severity: str) -> str:
    normalized = severity.lower()
    if normalized == "warn":
        return "warning"
    return normalized


def _format_rollup(findings: list[str], *, header: str | None = None) -> str:
    lines: list[str] = []
    if header:
        lines.append(f"*{header}*")
    lines.extend(f"- {item}" for item in findings)
    return "\n".join(lines)


# Matches flow-doctor.yaml + fleet_telegram_forum_topics_ops.md — no lib
# flow_doctor_fleet import until research lib-pin catches up (v0.82.0+).
_OPS_HEALTH_THREAD_ENV = "FLOW_DOCTOR_TELEGRAM_THREAD_OPS_HEALTH"


def _telegram_notifier_for_thread_env(fd, thread_env: str) -> object | None:
    from flow_doctor.notify.telegram import TelegramNotifier

    want = os.environ.get(thread_env)
    if not want:
        return None
    for notifier in fd._notifiers:
        if not isinstance(notifier, TelegramNotifier):
            continue
        thread_id = getattr(notifier, "message_thread_id", None)
        if thread_id is not None and str(thread_id) == str(want):
            return notifier
    return None


def publish_ops_alert(
    message: str,
    *,
    severity: str,
    source: str,
    dedup_key: str | None = None,
) -> None:
    """SNS via ``krepis.alerts.publish(telegram=False)`` + Telegram via flow-doctor."""
    from krepis.alerts import publish as alerts_publish

    alerts_publish(
        message=message,
        severity=severity,
        source=source,
        sns=True,
        telegram=False,
        dedup_key=dedup_key,
    )
    fd = get_flow_doctor()
    if fd is None:
        return
    try:
        subject = message.split("\n", 1)[0].replace("*", "").strip()
        if not subject:
            subject = f"Research alert [{severity.upper()}]"
        fd.notify_event(
            subject,
            body=message,
            severity=_normalize_flow_doctor_severity(severity),
            dedup_key=dedup_key or subject,
            context={"source": source},
        )
    except Exception as exc:
        logger.warning(
            "flow-doctor notify_event failed for ops alert (%s): %s — SNS already sent",
            source,
            exc,
        )


def publish_ops_digest(
    findings: list[str],
    *,
    header: str | None = None,
    source: str,
    dedup_key: str | None = None,
) -> bool:
    """Silent surveillance digest — flow-doctor forum topic only (no SNS).

    Mirrors legacy ``send_rollup(..., disable_notification=True)``: in-channel
    visibility without phone buzz. Routes via the ops-health forum notifier
    until ``#research`` SSM is seeded (config#1748).
    """
    if not findings:
        return True

    message = _format_rollup(findings, header=header)
    fd = get_flow_doctor()
    if fd is None:
        logger.warning(
            "flow-doctor inactive — surveillance digest not sent (source=%s)",
            source,
        )
        return False

    notifier = _telegram_notifier_for_thread_env(fd, _OPS_HEALTH_THREAD_ENV)
    if notifier is None:
        logger.warning(
            "ops-health Telegram notifier unavailable — digest not sent (source=%s)",
            source,
        )
        return False

    try:
        target = notifier.send_raw(message, disable_notification=True)
        return target is not None
    except Exception as exc:
        logger.warning(
            "flow-doctor digest send failed (source=%s): %s",
            source,
            exc,
        )
        return False
