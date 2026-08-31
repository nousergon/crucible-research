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
  arm of the producer leaderboard. RESOLVED 2026-08-21 (alpha-engine-config
  -I2638, Brian ruling): both the writer and the read are retired, and Think
  Tank self-anchors on the regime substrate + news aggregates instead. This
  primitive is what made the 158-day freeze visible in the first place.
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

Fan-out (alpha-engine-config-I8173/PR726, and this module's own history):
a caller that checks N upstreams sharing one producer must NOT let each
``assert_upstream_fresh`` call page independently — one dead producer must
not become N pages. Pass ``alert=False`` to every call in the batch, collect
the verdicts, and call ``publish_grouped_alerts(verdicts, source=...)`` once.
Grouping is on the DATA — ``(driver, upstream_prefix, as_of)`` — never on a
time window or "everything this evaluation happened to check": findings that
do not share a driver, prefix, and as-of stay in separate groups and page
separately. See ``group_stale_verdicts`` / ``StaleGroup`` / ``FreshnessVerdict.driver``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "CADENCE_TOLERANCE_DAYS",
    "FreshnessVerdict",
    "StaleGroup",
    "UpstreamStaleError",
    "assert_upstream_fresh",
    "group_stale_verdicts",
    "publish_grouped_alerts",
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

#: The CLOSED set of reasons an input is not fresh — exhaustive, with a
#: terminal ``unattributed`` branch rather than a silent fallthrough. Computed
#: on EVERY verdict (fresh or not), never only on the alerting path, so a
#: consumer that persists ``as_record()`` always carries the answer, not just
#: the assignment:
#:
#: - ``producer-halted`` — the upstream HAS written before (a parseable
#:   ``as_of`` exists) and has simply not written recently enough.
#: - ``never-written`` — no ``as_of`` was supplied at all (``as_of=None``):
#:   the artifact was never produced, or the caller found nothing to date.
#: - ``timestamp-unreadable`` — an ``as_of`` value was supplied but could not
#:   be parsed into a date (unsupported type, or a string that doesn't parse).
#: - ``unattributed`` — defensive terminal case: reached only if ``status``
#:   and the coerced timestamp disagree in a way the three branches above
#:   don't cover. Never silently absorbed into one of the other three.
Driver = Literal["producer-halted", "never-written", "timestamp-unreadable", "unattributed"]


def _attribute_driver(*, status: Status, as_of_input: Any, stamp: datetime | None) -> Driver | None:
    """Determine WHY an input is not fresh, from the closed set above.

    ``None`` only for ``status == "fresh"`` — attribution answers "why is
    this stale", which has no meaning for a fresh verdict.
    """
    if status == "fresh":
        return None
    if status == "stale" and stamp is not None:
        return "producer-halted"
    if status == "undated" and as_of_input is None:
        return "never-written"
    if status == "undated" and stamp is None:
        return "timestamp-unreadable"
    # Defensive: e.g. status == "stale" with stamp is None, which _coerce's
    # contract should make unreachable (a stale status only follows a
    # successful coercion). Loud rather than folded into a plausible-looking
    # neighbor.
    return "unattributed"


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
    driver: Driver | None = None
    remediation: str | None = None

    @property
    def is_fresh(self) -> bool:
        return self.status == "fresh"

    def _driver_detail(self) -> str:
        """One clause naming the driver in prose, for banners/group messages."""
        if self.driver == "producer-halted":
            when = self.as_of.isoformat() if self.as_of else "?"
            return f"the upstream producer has not written since {when}"
        if self.driver == "never-written":
            return "the upstream artifact was never written (no as-of found)"
        if self.driver == "timestamp-unreadable":
            return "the upstream carries a timestamp that could not be parsed"
        return "driver could not be attributed from the closed set (unattributed)"

    def banner(self) -> str:
        """One line, safe to embed in a prompt, a log, an alert, or a report."""
        if self.status == "undated":
            return (
                f"STALE-INPUT[{self.artifact}]: no as-of timestamp available — "
                f"age UNKNOWN against a {self.cadence} cadence "
                f"(tolerance {self.tolerance_days:g}d). Treated as NOT fresh. "
                f"Driver: {self._driver_detail()}."
                + (f" Clears when: {self.remediation}." if self.remediation else "")
            )
        if self.status == "stale":
            return (
                f"STALE-INPUT[{self.artifact}]: {self.age_days:.1f} days old "
                f"(as-of {self.as_of.isoformat() if self.as_of else '?'}) against a "
                f"{self.cadence} cadence, tolerance {self.tolerance_days:g}d. "
                f"Conclusions drawn from it are anchored to that date, NOT to "
                f"{self.checked_at.date().isoformat()}. Driver: {self._driver_detail()}."
                + (f" Clears when: {self.remediation}." if self.remediation else "")
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
            "driver": self.driver,
            "remediation": self.remediation,
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


def _publish_raw(message: str, *, source: str, severity: str, dedup_key: str) -> None:
    """The one seam that actually reaches the ops alert transport. Both a
    single-verdict publish and a grouped publish funnel through here, so a
    test (or an outage) only has to reason about one boundary."""
    try:
        from ops_alerts import publish_ops_alert

        publish_ops_alert(message, severity=severity, source=source, dedup_key=dedup_key)
    except Exception as exc:  # noqa: BLE001
        # Swallowed failure mode: the ALERT TRANSPORT is unavailable (SNS /
        # flow-doctor / krepis import missing in a slim image). Swallowed
        # because the staleness verdict is already recorded on two other
        # surfaces that do not depend on this one — the ERROR log line below,
        # and the verdict the caller persists onto its own artifact. Losing
        # the notification must not also lose the detection.
        logger.error("freshness: ops alert publish failed for %s: %s", dedup_key, exc)


def _publish(verdict: FreshnessVerdict, *, source: str, severity: str) -> None:
    """Emit ONE verdict's staleness onto the ops alert channel, deduped on the
    artifact identity (a causal key — one artifact, one notification, per
    ``observability-policy.md`` §7.2a). Only correct when the caller checks a
    single upstream per evaluation; a caller checking several upstreams that
    can share one root cause must pass ``alert=False`` here and publish once
    via ``publish_grouped_alerts`` instead — see its docstring."""
    _publish_raw(
        verdict.banner(),
        source=source,
        severity=severity,
        dedup_key=f"stale-upstream:{verdict.artifact}",
    )


def _upstream_prefix(artifact: str) -> str:
    """The shared-cause prefix an artifact identity is derived from — the
    part that does NOT vary across sibling findings produced by the same
    upstream failure. ``"decision_artifacts/*/sector_quant:consumer"`` and
    ``"decision_artifacts/*/thesis_update:healthcare:LNTH"`` both collapse to
    ``"decision_artifacts"``; unrelated identities do not collapse together.
    """
    head = artifact.split("/*")[0] if "/*" in artifact else artifact.rsplit("/", 1)[0]
    return head or artifact


@dataclass(frozen=True)
class StaleGroup:
    """One causal failure, evidenced by N stale-input findings that share a
    driver, an upstream prefix, AND an as-of timestamp — never grouped on a
    time window or on "everything this run happened to check"
    (``observability-policy.md`` §7.2a)."""

    driver: Driver
    upstream_prefix: str
    as_of: datetime | None
    members: tuple[FreshnessVerdict, ...]

    def episode_key(self) -> str:
        """Stable while the episode stands; moves the moment the upstream
        producer writes again (a new ``as_of``) — no new persisted state
        needed, this key IS the episode identity that the alert transport's
        own dedup marker keys off of."""
        as_of_part = self.as_of.isoformat() if self.as_of is not None else "no-as-of"
        return f"stale-upstream-episode:{self.upstream_prefix}:{self.driver}:{as_of_part}"

    def banner(self, *, max_members: int = 10) -> str:
        """Names the group's cause AND its members — never a bare count. A
        long list is truncated, but the truncation is marked in-band and the
        total count is exact (never a sample presented as a census,
        ``observability-policy.md`` §5.1)."""
        sample = self.members
        driver_detail = sample[0]._driver_detail() if sample else self.driver
        names = [v.artifact for v in sample]
        shown = names[:max_members]
        truncated = len(names) > max_members
        member_text = ", ".join(shown)
        if truncated:
            member_text += f", … (+{len(names) - max_members} more — {len(names)} total)"
        impacts = sorted({v.degraded_reason for v in sample if v.degraded_reason})
        remediations = sorted({v.remediation for v in sample if v.remediation})
        parts = [
            f"STALE-INPUT-GROUP[{self.upstream_prefix}]: {len(sample)} finding"
            f"{'s' if len(sample) != 1 else ''} share one cause — {driver_detail}.",
            f"Members: {member_text}.",
        ]
        if impacts:
            parts.append(f"If nobody acts: {'; '.join(impacts)}.")
        if remediations:
            parts.append(f"Clears when: {'; '.join(remediations)}.")
        return " ".join(parts)


def group_stale_verdicts(verdicts: Any) -> list[StaleGroup]:
    """Partition non-fresh verdicts into causal groups.

    The causal key is derived from the DATA — ``(driver, upstream_prefix,
    as_of)`` — never from a time window and never "every finding this
    evaluation produced". Two findings collapse into one group only when
    they share all three: the same reason for being stale, the same
    upstream family, AND the same as-of timestamp (the evidence that they
    trace to one failed write). Fresh verdicts are dropped — there is
    nothing to page.
    """
    buckets: dict[tuple[Driver, str, str], list[FreshnessVerdict]] = defaultdict(list)
    for verdict in verdicts:
        if verdict.is_fresh:
            continue
        driver = verdict.driver or "unattributed"
        prefix = _upstream_prefix(verdict.artifact)
        as_of_part = verdict.as_of.isoformat() if verdict.as_of is not None else "no-as-of"
        buckets[(driver, prefix, as_of_part)].append(verdict)

    groups: list[StaleGroup] = []
    for (driver, prefix, _as_of_part), members in buckets.items():
        groups.append(
            StaleGroup(
                driver=driver,
                upstream_prefix=prefix,
                as_of=members[0].as_of,
                members=tuple(members),
            )
        )
    return groups


def publish_grouped_alerts(
    verdicts: Any,
    *,
    source: str,
    severity: str = "error",
    max_members: int = 10,
) -> list[StaleGroup]:
    """Publish ONE alert per causal group across every stale-input finding
    from one evaluation, instead of one alert per finding.

    Callers checking more than one upstream in a single evaluation MUST pass
    ``alert=False`` to each ``assert_upstream_fresh`` call and call this
    ONCE at the end with every verdict collected (fresh ones included — they
    are simply dropped here). This is what turns a 47-artifact fan-out from
    one dead producer into one page, per ``observability-policy.md`` §7.2a
    ("one group failure, one notification for the group; the group names its
    members").

    Returns the groups actually published, for logging/testing.
    """
    groups = group_stale_verdicts(verdicts)
    for group in groups:
        logger.error("%s", group.banner(max_members=max_members))
        _publish_raw(
            group.banner(max_members=max_members),
            source=source,
            severity=severity,
            dedup_key=group.episode_key(),
        )
    return groups


def assert_upstream_fresh(
    artifact: str,
    *,
    as_of: Any,
    cadence: str,
    tolerance_days: float | None = None,
    checked_at: datetime | None = None,
    on_stale: OnStale = "raise",
    degraded_reason: str | None = None,
    remediation: str | None = None,
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
        remediation: what would clear this verdict — the concrete artifact
            that must be written. Optional but recommended: it is what
            ``banner()``/a group alert tells the reader to go do.
        alert: set False where the caller checks MULTIPLE upstreams in one
            evaluation and will publish its own grouped notification for all
            of them via ``publish_grouped_alerts`` — never for a caller
            checking a single upstream, which still wants the immediate,
            individually-dedup'd alert this function publishes by default.

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

    # Attribution is computed on EVERY evaluation, not only when about to
    # alert — a consumer that persists ``as_record()`` always carries the
    # answer (observability-policy.md: no data is never rendered green).
    driver = _attribute_driver(status=status, as_of_input=as_of, stamp=stamp)

    verdict = FreshnessVerdict(
        artifact=artifact,
        cadence=cadence,
        tolerance_days=float(tolerance_days),
        checked_at=now,
        status=status,
        as_of=stamp,
        age_days=age,
        degraded_reason=degraded_reason if status != "fresh" else None,
        driver=driver,
        remediation=remediation if status != "fresh" else None,
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
