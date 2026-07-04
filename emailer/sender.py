"""
Email sender — delegates to ``nousergon_lib.email_sender.send_email``
(the L4356 chokepoint that consolidates Gmail SMTP + AWS SES fallback
across alpha-engine modules).

Pre-consolidation note: the local Gmail/SES dispatch with App Password
documentation lived here. The lib chokepoint preserves the same Gmail
SMTP primary + SES fallback semantics and the same secret resolution
(``GMAIL_APP_PASSWORD``, ``EMAIL_SENDER``, ``EMAIL_RECIPIENTS``,
``AWS_REGION``). See ``nousergon_lib/email_sender.py`` for the
canonical docstring.
"""

from __future__ import annotations

import logging

from nousergon_lib.email_sender import send_email as _lib_send_email
from config import AWS_REGION

logger = logging.getLogger(__name__)


def send_email(
    subject: str,
    html_body: str,
    plain_body: str,
    recipients: list[str],
    sender: str,
    region: str = AWS_REGION,
) -> bool:
    """Backwards-compatible wrapper over ``nousergon_lib.email_sender.send_email``.

    Existing callers in this repo pass ``(subject, html_body, plain_body,
    recipients, sender, region)`` — the lib API takes
    ``(subject, body, *, recipients, html, sender, region)``. This wrapper
    bridges the shape so callers don't churn.

    Returns True on success, False on failure (lib never raises).
    """
    return _lib_send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=region,
    )
