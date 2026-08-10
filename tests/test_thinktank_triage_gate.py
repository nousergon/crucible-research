"""The expensive write tier fires only behind a recorded triage decision.

alpha-engine-config-I6649. ``products/thinktank.md`` §2.4 requires a
three-stage ladder — a wide cheap sweep detects candidate events, a **triage
tier decides whether the event changes the belief**, and the expensive write
tier fires only on a triage yes. The implementation was detect->write: the
sweep flagged a name and ``build_thesis`` ran immediately.

Sweep precision is low BY DESIGN — it is wide, cheap, and reads news
aggregates alone — so without the gate every false positive it produced was
paid for at the ``med`` group, which is the exact cost shape §2.4 forbids:
"the cost of a wide sweep lands on the most expensive stage".

Four properties, each pinned against a specific way this can silently regress:

1. **The gate precedes the write.** Structural, because the alternative is a
   full run harness with a live LLM client.
2. **Both verdicts are recorded.** §2.3: "the no-update decisions are the
   denominator; without them the gate's precision is unmeasurable and a gate
   that has silently stopped firing looks identical to a quiet week."
3. **A triage failure escalates rather than suppressing.** The gate exists to
   save cost; the cost of wrongly skipping a real re-underwrite is a stale
   belief no later run revisits, because the sweep only flags NEW events.
4. **A ticker with no standing thesis escalates without an LLM call.** There is
   no claim for the event to leave unchanged, and inventing a negative would
   silently suppress first coverage.
"""

from __future__ import annotations

import inspect
import os
import re
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinktank import analyst as tt_analyst  # noqa: E402
from thinktank import run as tt_run  # noqa: E402
from thinktank.schemas import (  # noqa: E402
    EventRecord,
    RunManifest,
    TickerEventAssessment,
    TriageDecisionLLM,
)

_SRC = inspect.getsource(tt_run._build_and_sweep)


def _pos(pattern: str) -> int:
    m = re.search(pattern, _SRC)
    assert m, f"anchor not found in _build_and_sweep: {pattern!r}"
    return m.start()


# ── 1. The gate precedes the write ───────────────────────────────────────────


def test_triage_is_called_before_the_event_driven_build_thesis():
    """If this inverts, the ladder is detect->write again and the gate is a
    decoration that costs a call and saves nothing."""
    triage_at = _pos(r"decision = triage\(")
    write_at = _pos(r'if a\.action == "update_thesis" and escalated:')
    assert triage_at < write_at, (
        "the triage call does not precede the gated build_thesis — §2.4's "
        "expensive tier must fire only behind an escalation decision"
    )


def test_the_event_driven_build_thesis_is_gated_on_the_escalation_flag():
    """The write must be guarded by the DECISION, not merely by the sweep's
    own action field — that guard is what the sweep already provides."""
    assert re.search(r'if a\.action == "update_thesis" and escalated:', _SRC), (
        "the event-driven build_thesis is not gated on `escalated` — a gate "
        "whose verdict does not guard the spend is not a gate"
    )


# ── 2. Both verdicts are recorded ────────────────────────────────────────────


def test_the_manifest_counts_holds_as_well_as_escalations():
    """events_flagged alone cannot show the gate's precision. A gate that has
    silently stopped firing (always-yes) is invisible without the denominator."""
    m = RunManifest(
        run_id="x", mode="daily", trading_day="2026-08-10",
        calendar_date="2026-08-10", started_at="2026-08-10T00:00:00Z",
    )
    for field in ("triage_yes", "triage_no", "triage_errors"):
        assert hasattr(m, field), f"RunManifest is missing {field}"
    assert re.search(r"manifest\.triage_no \+= 1", _SRC), (
        "the HOLD branch does not increment triage_no — without the "
        "denominator the gate's precision is unmeasurable (§2.3)"
    )
    assert re.search(r"manifest\.triage_yes \+= 1", _SRC)


def test_the_event_row_carries_the_triage_verdict_and_its_reason():
    """§2.3 wants every sweep decision recorded WITH ITS REASON, escalations
    and holds alike, at thinktank/events/{trading_day}.jsonl."""
    held = EventRecord(
        ticker="AAPL", trading_day="2026-08-10", action="update_thesis",
        severity=55, rationale="guidance chatter",
        triage_escalated=False, triage_reason="the thesis already anticipates it",
    )
    assert held.triage_escalated is False
    assert held.triage_reason
    assert held.thesis_version_written is None, (
        "a held event must not carry a written thesis version"
    )
    assert re.search(r'row\["triage_escalated"\] = escalated', _SRC)
    assert re.search(r'row\["triage_reason"\] = triage_reason', _SRC)


def test_a_row_the_gate_never_saw_is_distinguishable_from_a_hold():
    """action="none" never reaches the gate. `None` must not read as a hold —
    otherwise the denominator silently absorbs everything the sweep rejected
    and the gate's measured precision becomes meaningless."""
    unseen = EventRecord(
        ticker="MSFT", trading_day="2026-08-10", action="none",
        severity=5, rationale="routine drift",
    )
    assert unseen.triage_escalated is None
    assert unseen.triage_escalated is not False


def test_every_assessment_is_recorded_before_any_expensive_write_can_abort():
    """alpha-engine-config-I6817 D3, measured on run b150c317eeef (2026-08-10).

    The first run after the I6650 ordering fix: the sweep completed over 178
    tickers across 8 paid calls, the write tier then aborted on the first
    flagged name, and thinktank/events/2026-08-10.jsonl landed with SIX rows.
    Detection ran and was billed; its RECORD did not survive, because rows were
    appended interleaved with the writes. §2.3 grades the gate on the record.

    The triage tier makes this strictly worse if left alone — it adds a second
    thing that can raise inside the same loop.
    """
    record_at = _pos(r"rows_by_ticker\[a\.ticker\] = row")
    triage_at = _pos(r"decision = triage\(")
    write_at = _pos(r"thesis = build_thesis\(")
    assert record_at < triage_at < write_at, (
        "event rows are not all recorded before the first call that can raise "
        "— an abort mid-loop loses the record of detection that was paid for"
    )


def test_the_gate_verdict_is_stamped_before_the_write_it_gates():
    """A write that aborts must still leave the DECISION on the record: it was
    already made, and §2.4 requires the decision be recorded, not the write."""
    stamp_at = _pos(r'row\["triage_escalated"\] = escalated')
    write_at = _pos(r"thesis = build_thesis\(")
    assert stamp_at < write_at


# ── 3. A triage failure escalates rather than suppressing ────────────────────


def test_a_triage_error_escalates_and_is_counted_never_silently_held():
    """The one deliberate fail-open in this module. A swallowed triage error
    that defaulted to `hold` would drop a real re-underwrite permanently: the
    sweep flags NEW events, so no later run revisits the one it missed."""
    assert re.search(r"manifest\.triage_errors \+= 1", _SRC), (
        "a triage failure is not counted — an unbounded silent failure rate "
        "would make the gate's own health unobservable"
    )
    err_at = _pos(r"manifest\.triage_errors \+= 1")
    tail = _SRC[err_at : err_at + 400]
    assert re.search(r"escalated = True", tail), (
        "a triage failure does not escalate — the gate would silently suppress "
        "belief updates, which is strictly worse than the cost it saves"
    )


# ── 4. No standing thesis escalates without an LLM call ──────────────────────


def test_no_standing_thesis_escalates_without_calling_the_model(monkeypatch):
    """A flagged name with no belief to contradict is exactly the case where
    writing one is correct. It must not consume a triage call to learn that."""
    monkeypatch.setattr(tt_analyst, "load_latest_thesis", lambda store, ticker: None)

    def _boom(*a, **k):  # pragma: no cover - asserted not to run
        raise AssertionError("triage called the model with no standing thesis")

    monkeypatch.setattr(tt_analyst, "load_prompt", _boom)

    class _Client:
        def complete(self, *a, **k):  # pragma: no cover - asserted not to run
            raise AssertionError("triage called the model with no standing thesis")

    decision = tt_analyst.triage(
        object(),
        _Client(),
        object(),
        assessment=TickerEventAssessment(
            ticker="NEWCO", action="update_thesis", severity=70, rationale="IPO lockup"
        ),
        trading_day="2026-08-10",
    )
    assert isinstance(decision, TriageDecisionLLM)
    assert decision.escalate is True
    assert "no standing thesis" in decision.reason.lower()


# ── The tier itself ──────────────────────────────────────────────────────────


def test_the_triage_tier_is_named_and_distinct_from_the_write_tier():
    """A triage tier that resolves to the same tier as the write defeats the
    purpose: the ladder would be two rungs wearing three names."""
    assert tt_analyst.TRIAGE_TIER == "triage"
    assert tt_analyst.TRIAGE_TIER != tt_analyst.THESIS_TIER
    assert tt_analyst.TRIAGE_TIER != tt_analyst.SWEEP_TIER


@pytest.mark.parametrize("field", ["escalate", "reason"])
def test_the_decision_schema_forces_a_stated_reason(field):
    """An escalation with no reason cannot be graded, and §2.4 requires the
    decision be recorded WITH its reason."""
    assert field in TriageDecisionLLM.model_fields
    with pytest.raises(ValidationError):
        TriageDecisionLLM(**{field: None})
