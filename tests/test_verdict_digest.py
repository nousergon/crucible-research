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
        "slot_id": "scanner_cut",
        "decided_on": DATE,
        "decision": "hold",
        "reason_code": "arena_unmeasurable",
        "reason": "no challenger shares a usable window with the incumbent",
        "champion": CHAMP,
        "champion_before": CHAMP,
        "decision_metric": "weekly_population_relative_net_log_return",
        "decision_cadence": "weekly",
        "decision_source": "research/cuts_weekly_ledger/ledger.parquet",
        "excluded_arms": {},
        "arena": {
            "cycle_key": f"arena/universe_cut/{DATE}.json",
            "status": "unmeasurable",
        },
        "arms": {
            CHAMP: {
                "n_weeks_paired": 1,
                "mean_paired_log_return": 0.0,
                "confseq_supported": False,
                "comparison_status": "incumbent",
                "eligible_for_promotion": True,
            },
            "tech_score_top_60": {
                "n_weeks_paired": 0,
                "mean_paired_log_return": None,
                "confseq_supported": False,
                "comparison_status": "unmeasurable",
                "eligible_for_promotion": True,
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
    assert "arena_unmeasurable" in subject
    assert "arena_unmeasurable" in body


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
    mean_cell, confseq_cell = cells[3], cells[4]
    assert mean_cell == "—", cells
    assert confseq_cell == "no", cells
    # And the champion's genuine 0.0 IS a number, rendered as one.
    champ_line = next(ln for ln in body.splitlines() if f"`{CHAMP}` |" in ln)
    assert "+0.000000" in champ_line


def test_v4_records_without_earliest_date_still_render():
    body = build_body_md(_doc(), VERDICT_SLOT)
    assert "Earliest possible decision" not in body
    assert "arena/universe_cut" in body


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
    assert "arena_unmeasurable" in message


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
    assert sent[0]["defect"]


def test_the_subject_states_the_arm_count_so_a_shrinking_slot_is_visible():
    """A slot quietly losing arms is the I9272 defect in its next form; the
    subject line is the cheapest place it becomes visible."""
    d = decide_cut_champion(
        ledger_rows=None, board=None, champion_before=CHAMP, decided_on=DATE
    )
    subject = build_subject(d.to_document(), VERDICT_SLOT)
    assert "5 arms" in subject


# ── The cross-module contract (alpha-engine-config-I9278) ────────────────────


def test_every_guarded_import_of_this_module_names_a_symbol_that_exists():
    """THE guard for this arc's near-miss.

    ``scoring/spec_promotion.py`` imports from this module behind a
    ``try/except ImportError``, because the two modules landed on sibling
    branches. A missing MODULE and a missing SYMBOL raise the SAME exception, so
    that guard would have gone on swallowing a symbol mismatch forever — logging
    "verdict_digest.py not present yet" while the file sat right there, and the
    spec slot's verdict would have silently never delivered.

    That is exactly the defect alpha-engine-config-I9278 exists to retire,
    re-created by the merge order of its own fix. A guarded import is a place
    where a typo becomes permanent silence, so every symbol any module imports
    from here is asserted to exist.

    RED before the fix: `deliver_slot_verdict` did not exist on this module and
    `spec_promotion` imported it.
    """
    import ast
    import pathlib

    import scoring.verdict_digest as vd

    root = pathlib.Path(__file__).resolve().parents[1]
    found_any = False
    for path in (root / "scoring").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in ("scoring.verdict_digest", "verdict_digest"):
                continue
            for alias in node.names:
                found_any = True
                assert hasattr(vd, alias.name), (
                    f"{path.name} imports {alias.name!r} from verdict_digest, "
                    f"which does not export it. If that import is guarded by "
                    f"except ImportError, the mismatch is INVISIBLE at runtime "
                    f"and the verdict silently never delivers "
                    f"(alpha-engine-config-I9278)."
                )
    assert found_any, "no module imports from verdict_digest — has it been orphaned?"


def test_the_registry_holds_a_row_for_every_slot_that_writes_a_decision():
    """A promotion engine whose records cannot be delivered is a registration
    gap, not a formatting question. Both scanner slots must be renderable."""
    from scoring.verdict_digest import VERDICT_SLOTS

    assert set(VERDICT_SLOTS) == {"scanner_cut", "scanner_spec"}
    # Per-slot dedup prefixes: a shared key would let one slot's send suppress
    # another's inside the same 24h window.
    prefixes = [s.dedup_key_prefix for s in VERDICT_SLOTS.values()]
    assert len(set(prefixes)) == len(prefixes)


def test_the_registry_row_names_the_keys_this_module_writes():
    """The property that used to come from declaring the row beside its keys.

    `cut_promotion.VERDICT_SLOT` now resolves from the registry, so a key rename
    in one place and not the other is caught here rather than by an email
    pointing at an S3 object that does not exist.
    """
    from scoring import cut_promotion as cp

    assert cp.VERDICT_SLOT.audit_dated_key == cp.AUDIT_DATED_KEY
    assert cp.VERDICT_SLOT.pointer_key == cp.CUT_CHAMPION_POINTER_KEY


def test_a_spec_slot_record_resolves_its_own_slot_and_delivers():
    """The merged `spec_promotion._deliver_verdict` hands over ONLY the doc, so
    the slot must be resolvable from `slot_id` alone."""
    from scoring.verdict_digest import deliver_slot_verdict

    sink = _Sink()
    doc = _doc(slot_id="scanner_spec", champion="momentum_sleeve",
               champion_before="momentum_sleeve", reason_code="no_eligible_challenger")
    assert deliver_slot_verdict(doc, send_email_fn=sink) is True
    subject, body, kw = sink.calls[0]
    assert "Scanner spec champion/challenger" in subject
    assert kw["dedup_key"].startswith("scanner-spec-verdict:")
    assert "scanner_spec_champion" in body


def test_an_unknown_slot_id_raises_rather_than_guessing():
    """Guessing would send an email whose subject and S3 footer describe a
    different decision than the one in the body — a record asserting something
    false about its own origin (§7.5)."""
    from scoring.verdict_digest import UnknownVerdictSlot, deliver_slot_verdict

    with pytest.raises(UnknownVerdictSlot) as exc:
        deliver_slot_verdict(_doc(slot_id="predictor_model"), send_email_fn=_Sink())
    assert "predictor_model" in str(exc.value)


def test_the_cut_engine_still_delivers_through_the_registry_row():
    """`cut_promotion` moved from its own VerdictSlot literal to the registry;
    the behaviour must be identical."""
    sink = _Sink()
    assert send_verdict_digest(_doc(), VERDICT_SLOT, send_email_fn=sink) is True
    subject, _, kw = sink.calls[0]
    assert "Scanner cut champion/challenger" in subject
    assert kw["dedup_key"] == f"scanner-cut-verdict:{DATE}"
