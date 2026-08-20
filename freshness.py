"""Upstream-artifact freshness assertion — one primitive, every consumer.

The bug class this exists to end: **a running job consuming a FROZEN
upstream artifact with no staleness detection, emitting fresh-dated
output.** Absence of an upstream is usually handled (a ``None`` return, a
WARN, a recorded ``sources_present`` false). *Staleness* is not: the read
succeeds, the bytes parse, the job completes, and it stamps today's date
on a conclusion drawn from an input that stopped moving months ago. The
output is indistinguishable from a healthy one — which is the defect,
not the stale file itself (``principles.md`` §2.7).

Two live instances motivated this module:

- ``archive/macro/macro_report.md`` — LastModified 2026-03-16, its producer
  (``ArchiveManager.save_macro_report``) call-site-less since the multi-agent
  graph retired. Think Tank read it daily and reconciled its themes against a
  five-month-old macro backdrop, silently, while being scored as the shadow
  arm of the producer leaderboard.
- ``decision_artifacts/{Y}/{M}/{D}/`` captures — frozen at 2026-07-11 for the
  retired graph's agents, yet the weekly rationale-clustering rollup kept
  writing freshly-dated ``_analysis/{agent_id}/{YYYY-WW}.json`` and emitting
  a CloudWatch datapoint at today's timestamp.

Contract
--------

``assert_upstream_fresh`` takes the artifact's **identity**, its **as-of
timestamp**, its expected **cadence**, and an optional explicit
**tolerance**, and decides loudly:

- **fresh** — age within tolerance. Returns a verdict; the caller records it.
- **stale** / **undated** — RAISES ``UpstreamStaleError`` by default. An
  undated artifact is never "fresh": a missing timestamp is *unobserved*, not
  healthy (``observability-policy.md`` §8.3 — no data is never rendered green).

Fail-loud is the default (``~/Development/CLAUDE.md``). A consumer whose
contract forbids raising passes ``on_stale="degrade"`` **and must name the
failure mode in ``degraded_reason``** — the swallow declares itself or the
call is a ``ValueError``. A degraded call still:

1. logs at ERROR,
2. publishes an ops alert (durable + machine-readable channel, deduped on the
   artifact identity — a causal key, never a time window), and
3. returns a verdict the caller is obliged to persist onto its OWN output, so
   a downstream reader of that output can see the anchor was stale.

(3) is the part that cannot be skipped. An alert nobody joins to the artifact
leaves the artifact still looking healthy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "CADENCE_TOLERANCE_DAYS",
    "FreshnessVerdict",
    "UpstreamStaleError",
    "assert_upstream_fresh",
    "s3_last_modified",
]


class UpstreamStaleError(RuntimeError):
    """Raised when an upstream artifact is older than its tolerance (or
    carries no usable timestamp at all) and the consumer did not opt into
    degradation."""

    def __init__(self, verdict: FreshnessVerdict) -> None:
        self.verdict = verdict
        super().__init__(verdict.banner())


#: Default staleness tolerance per declared cadence, in days. Deliberately
#: generous relative to the nominal period: a weekly artifact written every
#: Saturday is at most 7 days old in normal operation, so 10 days tolerates
#: one skipped run before it screams, and a genuinely frozen artifact trips it
#: within a fortnight rather than never. A consumer with a tighter contract
#: passes ``tolerance_days`` explicitly.
CADENCE_TOLERANCE_DAYS: dict[str, float] = {
    "daily": 3.0,
    "weekday": 4.0,
    "weekly": 10.0,
    "biweekly": 21.0,
    "monthly": 40.0,
    "quarterly": 100.0,
}

Status = Literal["fresh", "stale", "undated"]
OnStale = Literal["raise", "degrade"]


@dataclass(frozen=True)
class FreshnessVerdict:
    """The decision, in a shape that serializes onto a consumer's artifact."""

    artifact: str
    cadence: str
    tolerance_days: float
    checked_at: datetime
    status: Status
    as_of: datetime | None = None
    age_days: float | None = None
    degraded_reason: str | None = None

    @property
    def is_fresh(self) -> bool:
        return self.status == "fresh"

    def banner(self) -> str:
        """One line, safe to embed in a prompt, a log, an alert, or a report."""
        if self.status == "undated":
            return (
                f"STALE-INPUT[{self.artifact}]: no as-of timestamp available — "
                f"age UNKNOWN against a {self.cadence} cadence "
                f"(tolerance {self.tolerance_days:g}d). Treated as NOT fresh."
            )
        if self.status == "stale":
            return (
                f"STALE-INPUT[{self.artifact}]: {self.age_days:.1f} days old "
                f"(as-of {self.as_of.isoformat() if self.as_of else '?'}) against a "
                f"{self.cadence} cadence, tolerance {self.tolerance_days:g}d. "
                f"Conclusions drawn from it are anchored to that date, NOT to "
                f"{self.checked_at.date().isoformat()}."
            )
        return (
            f"fresh[{self.artifact}]: {self.age_days:.1f}d old, "
            f"tolerance {self.tolerance_days:g}d ({self.cadence})"
        )

    def as_record(self) -> dict[str, Any]:
        """JSON-safe record for persisting onto the consumer's own output."""
        return {
            "artifact": self.artifact,
            "cadence": self.cadence,
            "tolerance_days": self.tolerance_days,
            "checked_at": self.checked_at.isoformat(),
            "as_of": self.as_of.isoformat() if self.as_of is not None else None,
            "age_days": (
                round(self.age_days, 3) if self.age_days is not None else None
            ),
            "status": self.status,
            "degraded_reason": self.degraded_reason,
        }


def _coerce(value: Any) -> datetime | None:
    """Accept datetime / date / ISO-ish string / None. Unparseable → None,
    which is ``undated`` — loud, never silently fresh."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        for parse in (datetime.fromisoformat,):
            try:
                parsed = parse(text)
            except ValueError:
                continue
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        logger.error("freshness: unparseable as_of %r for an upstream artifact", value)
        return None
    logger.error("freshness: unsupported as_of type %s", type(value).__name__)
    return None


def _publish(verdict: FreshnessVerdict, *, source: str, severity: str) -> None:
    """Emit the staleness onto the ops alert channel, deduped on the artifact
    identity (a causal key — one artifact, one notification, per
    ``observability-policy.md`` §7.2a)."""
    try:
        from ops_alerts import publish_ops_alert

        publish_ops_alert(
            verdict.banner(),
            severity=severity,
            source=source,
            dedup_key=f"stale-upstream:{verdict.artifact}",
        )
    except Exception as exc:  # noqa: BLE001
        # Swallowed failure mode: the ALERT TRANSPORT is unavailable (SNS /
        # flow-doctor / krepis import missing in a slim image). Swallowed
        # because the staleness verdict is already recorded on two other
        # surfaces that do not depend on this one — the ERROR log line below,
        # and the verdict the caller persists onto its own artifact. Losing
        # the notification must not also lose the detection.
        logger.error(
            "freshness: ops alert publish failed for %s: %s", verdict.artifact, exc
        )


def assert_upstream_fresh(
    artifact: str,
    *,
    as_of: Any,
    cadence: str,
    tolerance_days: float | None = None,
    checked_at: datetime | None = None,
    on_stale: OnStale = "raise",
    degraded_reason: str | None = None,
    source: str = "crucible-research.freshness",
    alert: bool = True,
) -> FreshnessVerdict:
    """Decide whether ``artifact`` is fresh enough to be consumed, loudly.

    Args:
        artifact: stable identity of the upstream — an S3 key, a prefix, a
            table name. Becomes the alert dedup key, so it must be stable
            across runs.
        as_of: the artifact's own timestamp (``datetime``/``date``/ISO string).
            ``None`` or unparseable ⇒ ``undated``, which is NOT fresh.
        cadence: expected production cadence; keys of
            ``CADENCE_TOLERANCE_DAYS``, or any label when ``tolerance_days``
            is given explicitly.
        tolerance_days: overrides the cadence default.
        checked_at: now, injectable for tests.
        on_stale: ``"raise"`` (default, fail-loud) or ``"degrade"``.
        degraded_reason: REQUIRED with ``on_stale="degrade"`` — names the
            failure mode being accepted and why the run may continue.
        alert: set False only where the caller publishes its own grouped
            notification covering this verdict.

    Returns:
        The verdict. The caller MUST persist ``verdict.as_record()`` onto its
        own output when the status is not ``fresh``.

    Raises:
        ValueError: ``on_stale="degrade"`` without a ``degraded_reason``, or an
            unknown cadence with no explicit tolerance.
        UpstreamStaleError: stale/undated under the default ``on_stale="raise"``.
    """
    if on_stale == "degrade" and not (degraded_reason or "").strip():
        raise ValueError(
            f"assert_upstream_fresh({artifact!r}, on_stale='degrade') requires "
            "degraded_reason — a deliberate swallow names the failure mode it "
            "accepts and the surface that records it."
        )

    if tolerance_days is None:
        if cadence not in CADENCE_TOLERANCE_DAYS:
            raise ValueError(
                f"unknown cadence {cadence!r} for {artifact!r} — pass "
                f"tolerance_days explicitly or use one of "
                f"{sorted(CADENCE_TOLERANCE_DAYS)}"
            )
        tolerance_days = CADENCE_TOLERANCE_DAYS[cadence]

    now = checked_at or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    stamp = _coerce(as_of)
    if stamp is None:
        status: Status = "undated"
        age: float | None = None
    else:
        age = (now - stamp).total_seconds() / 86400.0
        status = "fresh" if age <= tolerance_days else "stale"

    verdict = FreshnessVerdict(
        artifact=artifact,
        cadence=cadence,
        tolerance_days=float(tolerance_days),
        checked_at=now,
        status=status,
        as_of=stamp,
        age_days=age,
        degraded_reason=degraded_reason if status != "fresh" else None,
    )

    if verdict.is_fresh:
        logger.debug("freshness: %s", verdict.banner())
        return verdict

    logger.error(
        "%s%s",
        verdict.banner(),
        f" DEGRADED-CONTINUE: {degraded_reason}" if on_stale == "degrade" else "",
    )
    if alert:
        _publish(verdict, source=source, severity="error")

    if on_stale == "raise":
        raise UpstreamStaleError(verdict)
    return verdict


def s3_last_modified(s3: Any, *, bucket: str, key: str) -> datetime | None:
    """``LastModified`` of one S3 object, or ``None`` when absent/unreadable.

    ``None`` feeds ``as_of`` as ``undated`` — never as fresh. Used for
    upstreams that carry no in-band date (a Markdown report, a blob).
    """
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        # Swallowed failure mode: the object is missing, or HeadObject was
        # denied/errored. Recorded by returning None, which the caller turns
        # into an `undated` verdict — loud, alerted, and stamped on its own
        # output. Never silently treated as fresh.
        logger.warning("freshness: head_object failed for s3://%s/%s: %s", bucket, key, exc)
        return None
    return _coerce(head.get("LastModified"))
