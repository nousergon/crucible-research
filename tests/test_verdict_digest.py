"""The verdict digest — alpha-engine-config-I9278.

THE DEFECT. The scanner-cut engine has written a decision to three S3 keys on
every evaluation since 2026-08-21 and nothing rendered any of them. A full-fleet
search on 2026-08-29 found zero consumers of ``config/scanner_cut_champion.json``,
of ``config/apply_audit/scanner_cut_champion/``, of the weekly ledger and of
``research/cuts_leaderboard/{date}.json``. The only path reaching Brian fires
when the engine RAISES — so 52 legitimate holds a year and 52 silent failures
are, by design, indistinguishable.

RED VERIFICATION (champion-challenger-policy.md §7.4). Against ``origin/main``
every test here fails at COLLECTION: ``scoring/verdict_digest.py`` does not
exist, and ``run_cut_promotion`` has no call to it. That is the honest red for a
missing surface. The behavioural guards below were additionally verified against
MUTANTS of the post-fix module, each applied alone:

* a ``send_verdict_digest`` that returns early on ``decision == "hold"`` →
  ``test_a_hold_is_delivered_as_loudly_as_a_promotion`` goes red, and nothing
  else does. This is the mutation that matters: it is the shape the defect
  would take if someone "tidied up the noise" later.
* ``_fmt(None)`` returning ``"0"`` instead of an em dash →
  ``test_an_unmeasured_arm_renders_as_a_dash_never_as_zero`` goes red.
* the undelivered path returning ``True`` instead of escalating →
  ``test_an_undelivered_verdict_escalates_rather_than_passing_silently``.
* the digest call moved AFTER the defect raise in ``run_cut_promotion`` →
  ``test_a_defective_board_still_delivers_before_the_raise``.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.cut_promotion import (  # noqa: E402
    VERDICT_SLOT,
    CutPromotionError,
    decide_cut_champion,
    run_cut_promotion,
)
from scoring.verdict_digest import (  # noqa: E402
    build_body_md,
    build_subject,
    send_verdict_digest,
)

CHAMP = "attractiveness_top_60"
DATE = "2026-08-28"


class _Sink:
    """A ``send_email`` seam that records rather than sends."""

    def __init__(self, result: bool = True, raises: bool = False):
        self.calls: list[tuple] = []
        self.result = result
        self.raises = raises

    def __call__(self, subject, body, **kw):
        self.calls.append((subject, body, kw))
        if self.raises:
            raise RuntimeError("SES exploded")
        return self.result


class _Alerts:
    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, message, **kw):
        self.calls.append((message, kw))


def _doc(**over) -> dict:
    base = {
        "decided_on": DATE,
        "decision": "hold",
        "reason_code": "insufficient_weeks",
        "reason": "no eligible challenger has 5 paired weeks",
        "champion": CHAMP,
        "champion_before": CHAMP,
        "decision_metric": "paired_weekly_net_log_return_vs_champion",
        "decision_cadence": "weekly",
        "decision_source": "research/cuts_weekly_ledger/ledger.parquet",
        "excluded_arms": {},
        "decision_earliest_on": {
            "date": "2026-09-25",
            "provisional": True,
            "counted_from": "2026-08-20",
            "basis": "PROVISIONAL — no ledger week prices every promotable arm",
        },
        "arms": {
            CHAMP: {
                "n_weeks_paired": 1, "mean_paired_log_return": 0.0,
                "t_stat": None, "confidence": "thin",
                "eligible_for_promotion": False,
                "ineligibility_reason": "insufficient_weeks: n_weeks_paired=1",
            },
            "tech_score_top_60": {
                "n_weeks_paired": 0, "mean_paired_log_return": None,
                "t_stat": None, "confidence": "insufficient",
                "eligible_for_promotion": False,
                "ineligibility_reason": "insufficient_weeks: n_weeks_paired=0",
            },
        },
    }
    return base | over


def test_a_hold_is_delivered_as_loudly_as_a_promotion():
    """THE test. A digest that fires only on a promotion re-creates the exact
    condition being fixed: silence would again mean either "nothing to do" or
    "the engine died", and no reader could tell.

    52 holds a year is the loop's DOMINANT outcome, not its failure mode.
    """
    sink = _Sink()
    assert send_verdict_digest(_doc(), VERDICT_SLOT, send_email_fn=sink) is True
    assert len(sink.calls) == 1
    subject, body, _ = sink.calls[0]
    assert "promoted: none" in subject
    assert "insufficient_weeks" in subject
    assert "insufficient_weeks" in body


def test_a_promotion_names_the_arm_that_won_in_the_subject():
    sink = _Sink()
    send_verdict_digest(
        _doc(decision="promote", reason_code="promoted", champion="tech_score_top_60"),
        VERDICT_SLOT, send_email_fn=sink,
    )
    subject, body, _ = sink.calls[0]
    assert "PROMOTED: tech_score_top_60" in subject
    assert "MOVED" in body


def test_every_arm_the_record_names_gets_a_row_including_the_unscored():
    """An arm silently missing from the table is the "well-formed artifact
    containing nothing" class §7.2 names."""
    body = build_body_md(_doc(), VERDICT_SLOT, console_base_url="https://c")
    for arm in (CHAMP, "tech_score_top_60"):
        assert f"`{arm}`" in body


def test_an_unmeasured_arm_renders_as_a_dash_never_as_zero():
    """"Produced no comparable evidence" and "scored zero" are different facts
    and must never render identically (champion-challenger-policy.md §3)."""
    body = build_body_md(_doc(), VERDICT_SLOT)
    line = next(ln for ln in body.splitlines() if "`tech_score_top_60`" in ln)
    # Assert on the NUMERIC cells only. The ineligibility cell legitimately
    # contains an em dash of its own ("no — <reason>"), so a whole-line search
    # would pass against a `_fmt` that renders None as 0 — the mutant this test
    # exists to kill, which survived the first version of this assertion.
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    mean_cell, t_cell = cells[3], cells[4]
    assert mean_cell == "—", cells
    assert t_cell == "—", cells
    # And the champion's genuine 0.0 IS a number, rendered as one.
    champ_line = next(ln for ln in body.splitlines() if f"`{CHAMP}` |" in ln)
    assert "+0.000000" in champ_line


def test_a_provisional_earliest_date_is_never_rendered_as_a_commitment():
    """alpha-engine-config-I9284: `decision_earliest_on` is the field a reader
    uses to tell a working loop from a stuck one."""
    body = build_body_md(_doc(), VERDICT_SLOT)
    assert "2026-09-25 (PROVISIONAL)" in body
    assert "no ledger week prices every promotable arm" in body


def test_a_v1_bare_string_earliest_date_is_labelled_as_v1_not_relabelled():
    """A v1 record carried a bare string here. Rendering it under the v3 field's
    meaning would be a fabricated fact — the whole reason the version bumped."""
    body = build_body_md(_doc(decision_earliest_on="2027-02-22"), VERDICT_SLOT)
    assert "schema v1 field" in body
    assert "PROVISIONAL" not in body


def test_excluded_arms_is_stated_even_when_empty():
    """"No arm is excluded" and "this email does not say" are different
    messages (alpha-engine-config-I9272)."""
    body = build_body_md(_doc(), VERDICT_SLOT)
    assert "every scored arm of this slot is promotion-eligible" in body

    with_exclusion = build_body_md(
        _doc(excluded_arms={"x_top_60": {"arm": "x_top_60", "reason": "a stated reason"}}),
        VERDICT_SLOT,
    )
    assert "a stated reason" in with_exclusion


def test_an_undelivered_verdict_escalates_rather_than_passing_silently():
    """``send_email`` is fire-and-forget by contract — it returns False and never
    raises. A False here means the verdict reached nobody, which IS the defect
    this module exists to retire, so it must not become a second silence layered
    on the first (AGENTS.md: no silent degrade on a producer)."""
    alerts = _Alerts()
    ok = send_verdict_digest(
        _doc(), VERDICT_SLOT, send_email_fn=_Sink(result=False), alert_fn=alerts
    )
    assert ok is False
    assert len(alerts.calls) == 1
    message, kw = alerts.calls[0]
    assert kw["severity"] == "error"
    assert DATE in message
    assert "insufficient_weeks" in message


def test_a_raising_send_is_recorded_and_escalated_never_swallowed_or_propagated():
    """A raise here would take down a weekly promotion run over a notification;
    swallowing it would hide the undelivered verdict. Neither is acceptable."""
    alerts = _Alerts()
    ok = send_verdict_digest(
        _doc(), VERDICT_SLOT, send_email_fn=_Sink(raises=True), alert_fn=alerts
    )
    assert ok is False
    assert len(alerts.calls) == 1
    assert "SES exploded" in alerts.calls[0][0]


def test_the_send_is_deduped_per_cycle_not_per_run():
    """``evaluate`` is re-invocable by hand and by the watch-rerun path; three
    near-identical digests is how the backtester digest got its own dedup key."""
    sink = _Sink()
    send_verdict_digest(_doc(), VERDICT_SLOT, send_email_fn=sink)
    _, _, kw = sink.calls[0]
    assert kw["dedup_key"] == f"scanner-cut-verdict:{DATE}"
    assert kw["dedup_window_min"] == 1440


# ── The engine actually calls it ─────────────────────────────────────────────


class _Body:
    def __init__(self, b): self._b = b
    def read(self): return self._b


class _S3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.written: dict[str, dict] = {}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key in self.written:
            return {"Body": _Body(json.dumps(self.written[Key]).encode())}
        if Key in self.objects:
            return {"Body": _Body(json.dumps(self.objects[Key]).encode())}
        raise RuntimeError(f"NoSuchKey: {Key}")

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803
        self.written[Key] = json.loads(Body)


def test_run_cut_promotion_delivers_the_verdict_on_a_plain_hold(monkeypatch):
    """The wiring. RED before the fix: nothing consumed the record at all."""
    sent: list = []
    monkeypatch.setattr(
        "scoring.cut_promotion.send_verdict_digest",
        lambda doc, slot, **kw: sent.append((doc, slot)) or True,
    )
    s3 = _S3()
    doc = run_cut_promotion(DATE, bucket="b", s3_client=s3, ledger_rows=None)
    assert doc["decision"] == "hold"
    assert len(sent) == 1
    assert sent[0][0]["reason_code"] == doc["reason_code"]
    assert sent[0][1].slot_id == "scanner_cut"


def test_a_defective_board_still_delivers_before_the_raise(monkeypatch):
    """A cycle that held on a DEFECTIVE board is precisely the cycle Brian most
    needs delivered — and a raise ordered before the send would deliver nothing.

    RED against the mutant that moves the digest call below the defect raise.
    """
    sent: list = []
    monkeypatch.setattr(
        "scoring.cut_promotion.send_verdict_digest",
        lambda doc, slot, **kw: sent.append(doc) or True,
    )
    board = {
        "horizons": [
            {"horizon_days": 21, "specs": [{"name": CHAMP}, {"name": CHAMP}]},
        ]
    }
    s3 = _S3({f"research/cuts_leaderboard/{DATE}.json": board})
    with pytest.raises(CutPromotionError):
        run_cut_promotion(DATE, bucket="b", s3_client=s3, ledger_rows=None)
    assert len(sent) == 1
    assert sent[0]["reason_code"] == "board_defective"
    assert sent[0]["defect"]


def test_the_subject_states_the_arm_count_so_a_shrinking_slot_is_visible():
    """A slot quietly losing arms is the I9272 defect in its next form; the
    subject line is the cheapest place it becomes visible."""
    d = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE
    )
    subject = build_subject(d.to_document(), VERDICT_SLOT)
    assert "5 arms" in subject
