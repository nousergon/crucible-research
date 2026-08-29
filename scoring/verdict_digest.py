"""verdict_digest.py — deliver a champion/challenger verdict to Brian, every cycle.

THE DEFECT THIS RETIRES (alpha-engine-config-I9278). The scanner-cut slot has
written a decision to three S3 keys on every evaluation since 2026-08-21 and
**nothing renders any of them**. A full-fleet search on 2026-08-29 found zero
consumers of ``config/scanner_cut_champion.json``, of
``config/apply_audit/scanner_cut_champion/``, of
``research/cuts_weekly_ledger/ledger.parquet`` and of
``research/cuts_leaderboard/{date}.json`` — no dashboard, no emailer, no report
card, no notifier. The only path that reaches Brian at all is
``nousergon-data/infrastructure/overseer/playbooks.yaml::
research_cut_promotion_alerts``, which fires **only when the engine RAISES**.

So the loop's dominant outcome — a hold, which is what 52 of 52 cycles a year
will legitimately be for a slot whose evidence accrues one week at a time — is
**by design indistinguishable from the loop being dead**. ``principles.md`` §2.7:
a component emitting nothing is not healthy, it is unobserved, and *no data* is
never rendered as green. ``champion-challenger-policy.md`` §7.2 says the same
thing in its own words: an unmeasurable result must fail LOUD, not render as an
empty success.

WHY THIS SHAPE, AND WHY IT IS SLOT-PARAMETERISED
------------------------------------------------
``crucible-backtester/optimizer/champion_digest.py`` (merged today as
``crucible-backtester-PR753``) already proves this pattern in production for the
RESEARCH selection-producer slot: build a subject and a markdown body from the
audit record, push it through the fleet's single SES/SMTP chokepoint
``krepis.email_sender.send_email``, deep-link to a console page for the detail,
and fire on **every** outcome. That module's own docstring names the trigger
this file crosses:

    "A third slot adopting it is the trigger to lift a generic ``verdict_digest``
    builder into ``nousergon-lib``."

The scanner has TWO further slots — the CUT slot (this module's first caller)
and the SPEC slot (``scoring/spec_promotion.py``, landing alongside) — so this
is the third AND fourth adoption, and the lift is due
(``shared-code-policy.md``). It is deliberately NOT taken in this PR, and the
delta is recorded here rather than left implied:

* **SOTA option skipped:** a generic ``verdict_digest`` in ``nousergon-lib``,
  with the backtester's copy retired onto it in the same change.
* **Why the alternative is correct for this instance:** ``optimizer/`` in
  ``crucible-backtester`` is another agent's live lane this session; editing
  ``champion_digest.py`` to retire it onto a library would be a second PR
  against work already open, which ``engagement-protocol-policy.md`` §4.0
  forbids. A lift that leaves the original copy in place is not a lift — it is a
  third copy wearing a library's name.
* **Cost the SOTA path imposes:** a cross-repo change (``nousergon-lib`` +
  ``crucible-backtester`` + ``crucible-research``) with a merge order through a
  branch that is not mine, to move code that is already correct.
* **Tracked follow-up:** ``alpha-engine-config-I9285`` — lift this module into
  ``nousergon-lib`` and retire ``champion_digest.py`` onto it. Until then this
  file is the SECOND implementation of the shape and it says so.

Within crucible-research it is generic across slots from the first line: the
CUT and SPEC records are the same document family, so a per-slot copy of the
body builder would be the duplication the policy is about.

WHAT IS DELIVERED, AND WHEN
---------------------------
**On EVERY outcome — promote and hold alike.** A hold is the news. A digest that
fires only on a promotion re-creates the exact condition being fixed: silence
would again mean either "nothing to do" or "the engine died", and the reader
could not tell. The subject line therefore always states an outcome, in the same
``promoted: none`` phrasing the model-zoo rotation digest and the research
champion digest use, so the fleet's weekly champion emails read as one family in
an inbox.

**Fail-loud on silence.** ``send_email`` is fire-and-forget by contract: it
returns ``False`` and never raises. A ``False`` here means the verdict reached
nobody, which is the defect this module exists to retire, so it is escalated as
an ops alert rather than swallowed (``AGENTS.md``: no silent degrade on a
producer). The send itself is non-fatal to the promotion run — a notification
must never red the pipeline it reports on, and the durable record is already
written by the time this runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Cross-repo contract with crucible-dashboard's console page slug. Kept
# caller-side for the same reason champion_digest.EXPERIMENTS_SLUG is: the slug
# is this email's contract with that page, not a krepis concern.
EXPERIMENTS_SLUG = "experiments"

DEDUP_WINDOW_MIN = 1440
"""24h — covers a same-day rerun (the weekly SF's scanner state is re-invocable
by hand and by the watch-rerun path), never the next cycle."""

UNDELIVERED_ALERT_SEVERITY = "error"


@dataclass(frozen=True)
class VerdictSlot:
    """Everything about a slot that the digest cannot infer from its record.

    Deliberately small. Every number in the email comes from the audit document
    itself — champion-challenger-policy.md §7.5, provenance true by
    construction: a digest that restated a slot's metric from a constant here
    would go stale the week the slot's basis changed, which is exactly what the
    ``schema_version`` 1→2 cutover was for.
    """

    slot_id: str
    # Human-readable name, used in the subject line.
    title: str
    # The three S3 keys the reader is pointed at, for the "if you want the raw
    # record" footer. Formatted with ``date``.
    pointer_key: str
    audit_dated_key: str
    dedup_key_prefix: str


def _fmt(value: Any, spec: str = "+.6f") -> str:
    """A numeric cell.

    ``None`` renders as an explicit em dash, never as ``0``. "The arm produced
    no comparable evidence" and "the arm scored zero" are different facts and
    must never render identically (champion-challenger-policy.md §3) — this is
    the single most repeated defect in this family and it is one line to
    prevent.
    """
    if value is None:
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def build_subject(doc: dict, slot: VerdictSlot) -> str:
    """``Alpha Engine | Scanner cut champion/challenger 2026-08-28 | 5 arms | promoted: none (insufficient_weeks)``."""
    decision = doc.get("decision") or "unknown"
    arms = doc.get("arms") or {}
    n = len(arms)
    if decision == "promote":
        label = f"PROMOTED: {doc.get('champion')}"
    else:
        label = f"promoted: none ({doc.get('reason_code') or decision})"
    return (
        f"Alpha Engine | {slot.title} {doc.get('decided_on')} "
        f"| {n} arm{'' if n == 1 else 's'} | {label}"
    )


def build_body_md(doc: dict, slot: VerdictSlot, *, console_base_url: str | None = None) -> str:
    """The verdict as markdown: what was decided, on what evidence, what is
    blocking it, and when it could next change — enough that a reader never has
    to open S3.

    **Every arm the record names gets a row, including arms that scored
    nothing.** An arm silently missing from the table is the "well-formed
    artifact containing nothing" class §7.2 names, and an arm that is measured
    but ineligible is shown as measured-and-ineligible with its reason, never
    dropped (alpha-engine-config-I9272).
    """
    decided_on = str(doc.get("decided_on") or "")
    decision = doc.get("decision") or "unknown"
    before = doc.get("champion_before")
    after = doc.get("champion")
    moved = bool(after) and after != before
    arms = doc.get("arms") or {}
    earliest = doc.get("decision_earliest_on") or {}

    lines = [
        f"## {slot.title} — {decided_on}",
        "",
        f"**Decision:** `{decision}` · **reason_code:** `{doc.get('reason_code')}`",
        (
            f"**Live pointer:** `{before}` → `{after}` (MOVED)"
            if moved
            else f"**Live pointer:** `{before}` (unchanged)"
        ),
        f"**Decision metric:** `{doc.get('decision_metric')}` "
        f"({doc.get('decision_cadence')}, from `{doc.get('decision_source')}`)",
    ]
    if doc.get("last_promoted_on"):
        lines.append(f"**Last promotion:** {doc['last_promoted_on']}")

    # The number that says how far off a real decision is — and whether that
    # number is itself provisional (alpha-engine-config-I9284). A bare date here
    # reads as a commitment; the whole point of the field is to let a reader
    # tell a working loop from a stuck one, so the caveat travels with it.
    if isinstance(earliest, dict) and earliest.get("date"):
        provisional = " (PROVISIONAL)" if earliest.get("provisional") else ""
        lines.append(f"**Earliest possible decision:** {earliest['date']}{provisional}")
        if earliest.get("basis"):
            lines.append(f"> {earliest['basis']}")
    elif earliest:
        # A v1 record carried a bare string here. Rendered as what it is, never
        # relabelled as though it meant what the v2 field means.
        lines.append(f"**Earliest possible decision (schema v1 field):** {earliest}")

    if doc.get("defect"):
        lines.append(f"**DEFECT:** {doc['defect']} — the run raised after writing this record.")

    lines += [
        "",
        "### Arms",
        "",
        "| arm | role | paired weeks | mean/week | t | confidence | eligible |",
        "|---|---|---|---|---|---|---|",
    ]
    if not arms:
        lines.append("| _the record names no arms_ | — | — | — | — | — | — |")
    for name, ev in arms.items():
        if name == before:
            role = "champion"
        elif name == after and moved:
            role = "**PROMOTED**"
        else:
            role = "challenger"
        if ev.get("eligible_for_promotion", True):
            eligible = "yes"
        else:
            eligible = f"no — {ev.get('ineligibility_reason') or 'no reason recorded'}"
        lines.append(
            f"| `{name}` | {role} | {ev.get('n_weeks_paired', '—')} "
            f"| {_fmt(ev.get('mean_paired_log_return'))} "
            f"| {_fmt(ev.get('t_stat'), '+.2f')} "
            f"| {ev.get('confidence', '—')} | {eligible} |"
        )

    # Arms excluded from promotion by the REGISTER, as opposed to by this
    # cycle's evidence. Stated even when empty: "no arm is excluded" and "this
    # email does not say" are different messages (alpha-engine-config-I9272).
    excluded = doc.get("excluded_arms")
    if excluded is not None:
        lines += ["", "### Excluded from promotion by the register", ""]
        if excluded:
            lines += [f"- `{a}` — {e.get('reason')}" for a, e in sorted(excluded.items())]
        else:
            lines.append("- _none — every scored arm of this slot is promotion-eligible._")

    lines += ["", "### Reason", "", str(doc.get("reason") or "_no reason recorded_")]
    lines += [
        "",
        "---",
        f"Record: `s3://alpha-engine-research/{slot.audit_dated_key.format(date=decided_on)}`",
        f"Pointer: `s3://alpha-engine-research/{slot.pointer_key}`",
        "",
        f"[Champion/challenger ledgers on the console]"
        f"({experiments_url(decided_on, console_base_url)})",
    ]
    return "\n".join(lines)


def experiments_url(run_date: str, console_base_url: str | None = None) -> str:
    """Deep-link to the console page for ``run_date``. Thin wrapper over the
    ``krepis.console.console_url`` chokepoint, mirroring
    ``champion_digest.experiments_url``."""
    from krepis.console import console_url  # noqa: PLC0415

    return console_url(EXPERIMENTS_SLUG, date=run_date, base=console_base_url)


def send_verdict_digest(
    doc: dict,
    slot: VerdictSlot,
    *,
    console_base_url: str | None = None,
    send_email_fn: Any = None,
    alert_fn: Any = None,
) -> bool:
    """Deliver ``doc`` as this cycle's verdict email. Returns whether it landed.

    Fires on EVERY decision — see the module docstring. Returns ``False`` when
    the send did not land, having first escalated an ops alert, because an
    undelivered verdict must not become a second silence layered on the first.

    ``send_email_fn`` / ``alert_fn`` are seams for tests only; production
    resolves ``krepis.email_sender.send_email`` and
    ``ops_alerts.publish_ops_alert`` lazily, so importing this module never
    requires either.
    """
    decided_on = doc.get("decided_on")
    subject = build_subject(doc, slot)
    body = build_body_md(doc, slot, console_base_url=console_base_url)

    if send_email_fn is None:
        from krepis.email_sender import send_email as send_email_fn  # noqa: PLC0415

    sent = False
    failure = None
    try:
        sent = bool(
            send_email_fn(
                subject,
                body,
                dedup_key=f"{slot.dedup_key_prefix}:{decided_on}",
                dedup_window_min=DEDUP_WINDOW_MIN,
            )
        )
    except Exception as exc:  # noqa: BLE001
        # send_email's contract is "never raises", but a contract is not a
        # guarantee and a raise here would take down a promotion run over a
        # notification. Recorded, escalated below, never swallowed silently.
        logger.exception("[verdict_digest] %s verdict email raised", slot.slot_id)
        failure = repr(exc)

    if sent:
        logger.info(
            "[verdict_digest] metric verdict_digest_sent slot=%s decided_on=%s "
            "decision=%s reason_code=%s",
            slot.slot_id, decided_on, doc.get("decision"), doc.get("reason_code"),
        )
        return True

    reason = failure or "send_email returned False (missing config, auth, or network)"
    logger.error(
        "[verdict_digest] %s verdict for %s did NOT land (%s) — the decision "
        "reached no operator surface",
        slot.slot_id, decided_on, reason,
    )
    _escalate_undelivered(doc, slot, reason, alert_fn=alert_fn)
    return False


def _escalate_undelivered(
    doc: dict, slot: VerdictSlot, reason: str, *, alert_fn: Any = None
) -> None:
    """An undelivered verdict is an ops alert, not a log line.

    Best-effort by necessity — this is already the failure path of the
    notification layer — but it never passes silently: an alerting failure is
    logged with its traceback.
    """
    if alert_fn is None:
        try:
            from ops_alerts import publish_ops_alert as alert_fn  # noqa: PLC0415
        except ImportError as exc:
            logger.error(
                "[verdict_digest] undelivered-verdict alert skipped — "
                "ops_alerts unavailable: %s", exc,
            )
            return
    decided_on = doc.get("decided_on")
    try:
        alert_fn(
            f"{slot.title} verdict for {decided_on} was computed "
            f"(decision={doc.get('decision')}, reason_code={doc.get('reason_code')}, "
            f"pointer={doc.get('champion')}) but the digest email did not land: "
            f"{reason}. The verdict exists only in "
            f"s3://alpha-engine-research/{slot.audit_dated_key.format(date=decided_on)}.",
            severity=UNDELIVERED_ALERT_SEVERITY,
            source=f"crucible-research/scoring/verdict_digest.py::send_verdict_digest[{slot.slot_id}]",
            dedup_key=f"{slot.slot_id}_verdict_undelivered_{decided_on}",
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the run
        logger.exception(
            "[verdict_digest] undelivered-verdict alert publish ALSO failed for %s",
            slot.slot_id,
        )
