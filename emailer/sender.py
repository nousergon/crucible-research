"""
Email sender — dual-backend: Gmail SMTP (primary) or AWS SES (fallback).

Gmail SMTP is the default when GMAIL_APP_PASSWORD is set in the environment.
This avoids the SPF/DKIM failure that occurs when AWS SES sends on behalf of
a @gmail.com sender address (SES is not authorized by Gmail's SPF policy, so
emails land in the recipient's spam folder).

Setup (one-time):
  1. Enable 2-Factor Authentication on the Gmail account.
  2. Go to https://myaccount.google.com/apppasswords
  3. Create an App Password (name: "Alpha Engine").
  4. Set the environment variable:
       GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   (16-char code, spaces OK)
  5. Also set in Lambda environment variables (via console or IaC).

When GMAIL_APP_PASSWORD is not set, falls back to AWS SES (requires a
custom domain sender to avoid spam; @gmail.com senders will land in spam).
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import boto3
from botocore.exceptions import ClientError

from alpha_engine_lib.secrets import get_secret
from config import AWS_REGION

logger = logging.getLogger(__name__)

_GMAIL_SMTP_HOST = "smtp.gmail.com"
_GMAIL_SMTP_PORT = 587


def _send_via_gmail_smtp(
    subject: str,
    html_body: str,
    plain_body: str,
    recipients: list[str],
    sender: str,
    app_password: str,
) -> bool:
    """
    Send email through Gmail's SMTP relay using an App Password.

    The email originates from Gmail's own servers, so SPF/DKIM/DMARC all
    pass for @gmail.com senders — no spam-folder issues.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    # Strip spaces from App Password (Google sometimes displays with spaces)
    password = app_password.replace(" ", "")

    try:
        with smtplib.SMTP(_GMAIL_SMTP_HOST, _GMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Email sent via Gmail SMTP: '%s' to %s", subject, recipients)
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(
            f"Gmail SMTP auth failed: {e}. "
            "Check GMAIL_APP_PASSWORD and that 2FA is enabled on the account."
        )
        return False
    except Exception as e:
        logger.error("Gmail SMTP send error: %s", e)
        return False


def _send_via_ses(
    subject: str,
    html_body: str,
    plain_body: str,
    recipients: list[str],
    sender: str,
    region: str,
) -> bool:
    """
    Send email via AWS SES.

    NOTE: If sender is a @gmail.com address, SES delivery will succeed but
    Gmail will route the message to spam (SPF alignment failure). Use a
    custom domain sender for reliable inbox delivery via SES.
    """
    ses = boto3.client("ses", region_name=region)

    try:
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )
        print(f"Email sent via SES: '{subject}' to {recipients}")
        return True
    except ClientError as e:
        print(f"SES send failed: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"SES send error: {e}")
        return False


def send_email(
    subject: str,
    html_body: str,
    plain_body: str,
    recipients: list[str],
    sender: str,
    region: str = AWS_REGION,
) -> bool:
    """
    Send a multipart (HTML + plain text) email.

    Routing logic:
    - If GMAIL_APP_PASSWORD env var is set → use Gmail SMTP (recommended for
      @gmail.com senders; passes SPF/DKIM, lands in inbox).
    - Otherwise → fall back to AWS SES (works correctly only with a custom
      domain sender that has SES DKIM/SPF configured).

    Returns True on success, False on failure.
    """
    app_password = (get_secret("GMAIL_APP_PASSWORD", required=False, default="") or "").strip()

    if app_password:
        return _send_via_gmail_smtp(
            subject=subject,
            html_body=html_body,
            plain_body=plain_body,
            recipients=recipients,
            sender=sender,
            app_password=app_password,
        )
    else:
        print(
            "GMAIL_APP_PASSWORD not set — falling back to SES. "
            "If sender is @gmail.com, email may land in spam."
        )
        return _send_via_ses(
            subject=subject,
            html_body=html_body,
            plain_body=plain_body,
            recipients=recipients,
            sender=sender,
            region=region,
        )
