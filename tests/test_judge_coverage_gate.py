"""Coverage is a pass/fail verdict, not a field (alpha-engine-config-I9309).

Brian, 2026-08-29: *"perhaps if the lambda times out then we need to put the
judge on a spot instance"*, and *"ok lets keep 100% coverage and move to spot"*.

Every test here fails against pre-I9309 code: `evals/judge_coverage.py` did not
exist, `process_batch_results` emitted no coverage ledger, and grading 13 of 85
artifacts was a SUCCESSFUL stage carrying `complete=False`.
"""

from __future__ import annotations

import pytest


def _entry(cid: str, agent: str = "thinktank_thesis", judge: str = "claude-haiku-4-5"):
    return {
        "custom_id": cid,
        "capture_s3_key": f"captures/{cid}.json",
        "agent_id": agent,
        "run_id": f"run-{cid}",
        "judge_model": judge,
        "rubric_id": "thesis",
    }


def _summary(*, planned: int, graded: int, **over):
    ungraded = [_entry(f"c{i}") for i in range(graded, planned)]
    base = {
        "date": "2026-08-29",
        "batch_id": "sync-2026-08-29",
        "plan_entry_count": planned,
        "plan_entries_graded": graded,
        "plan_entries_ungraded": planned - graded,
        "coverage_complete": graded == planned,
        "ungraded_entries": ungraded,
        "budget_stopped_phases": [],
        "n_skipped_for_budget": {},
        "degraded_transport": True,
        "failed": [],
    }
    base.update(over)
    return base


class TestVerdict:
    def test_full_coverage_passes(self):
        from evals.judge_coverage import enforce_coverage

        v = enforce_coverage(_summary(planned=83, graded=83))
        assert v["status"] == "PASS"
        assert v["graded"] == 83

    def test_the_measured_lambda_shortfall_now_raises(self):
        """13 of 85 — the shape the Lambda produced every week.

        It reported `complete=False` and returned success. That is the exact
        combination Brian's 2026-08-14 Director ruling forbids: a capacity
        regression made survivable instead of visible.
        """
        from evals.judge_coverage import JudgeCoverageShortfall, enforce_coverage

        with pytest.raises(JudgeCoverageShortfall) as exc:
            enforce_coverage(_summary(
                planned=85, graded=13,
                budget_stopped_phases=["sync_fallback"],
                n_skipped_for_budget={"sync_fallback": 72},
            ))
        assert exc.value.planned == 85
        assert exc.value.graded == 13
        assert "13 of 85" in str(exc.value)

    def test_a_degraded_transport_alone_is_not_a_failure(self):
        """The two questions stay apart.

        `degraded_transport` is a legitimate, priced ladder step and the run's
        output is complete on either rung. Collapsing it into the coverage
        verdict would make a correct transport choice fatal — and today EVERY
        run takes the sync rung, so that mistake would fail every week.
        """
        from evals.judge_coverage import enforce_coverage

        v = enforce_coverage(_summary(planned=83, graded=83, degraded_transport=True))
        assert v["status"] == "PASS"
        assert v["degraded_transport"] is True

    def test_a_summary_with_no_ledger_is_unmeasured_never_pass(self):
        """`principles.md` §2.7 — no data is never rendered as green.

        A producer that cannot say what it covered has not shown it covered
        anything. Defaulting a missing ledger to PASS would re-open the exact
        hole this module closes, on the first image whose pin lags.
        """
        from evals.judge_coverage import (
            JudgeCoverageShortfall,
            assess_coverage,
            enforce_coverage,
        )

        stale = {"date": "2026-08-29", "haiku_evaluated": 13, "failed": []}
        assert assess_coverage(stale)["status"] == "UNMEASURED"
        with pytest.raises(JudgeCoverageShortfall):
            enforce_coverage(stale)


class TestShortfallMessage:
    def test_names_the_cause_split_and_both_halves_appear_when_zero(self):
        """A count alone sends a reader to the wrong place.

        A deadline stop and a judge error are different defects with different
        first moves. Both statements are emitted even when the count is zero,
        so their absence is legible rather than ambiguous.
        """
        from evals.judge_coverage import describe_shortfall

        msg = describe_shortfall(
            planned=85, graded=80,
            ungraded_entries=[_entry(f"c{i}") for i in range(5)],
            budget_stopped_phases=[],
            n_skipped_for_budget={},
            failed=[{"stage": "sync_fallback_judge", "agent_id": "thinktank_theme"}],
        )
        assert "no phase stopped on a deadline" in msg
        assert "failed in the judge itself" in msg
        assert "thinktank_theme" in msg

    def test_truncates_a_corpus_sized_shortfall(self):
        """The message lands in an SF `cause` field with a size limit."""
        from evals.judge_coverage import describe_shortfall

        msg = describe_shortfall(
            planned=85, graded=0,
            ungraded_entries=[_entry(f"c{i}") for i in range(85)],
            budget_stopped_phases=["sync_fallback"],
            n_skipped_for_budget={"sync_fallback": 85},
            failed=[],
        )
        assert "+75 more" in msg


class TestLedgerFromProcessBatchResults:
    """The ledger counts PLAN ENTRIES, once each, on whichever rung graded them."""

    @staticmethod
    def _run(monkeypatch, *, n, fail_indices=(), escalate=False):
        import json as _json
        from unittest.mock import MagicMock

        from evals import orchestrator as orch
        from tests.test_eval_judge_batch_transport import _minimal_plan

        plan = _minimal_plan(n=n)
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: _json.dumps(plan).encode())
        }
        monkeypatch.setattr(orch, "_load_capture_artifact", lambda *a, **k: object())

        seen = {"i": -1}

        def _fake_eval(artifact, **kwargs):
            seen["i"] += 1
            if seen["i"] in fail_indices:
                raise RuntimeError("judge blew up on this artifact")
            ev = MagicMock()
            ev.judge_model = kwargs["judge_model"]
            return ev

        monkeypatch.setattr(orch, "evaluate_artifact", _fake_eval)
        monkeypatch.setattr(orch, "persist_eval_artifact", lambda *a, **k: "key")
        monkeypatch.setattr(
            orch, "should_escalate_to_sonnet", lambda *a, **k: escalate,
        )
        return orch.process_batch_results(
            batch_id="sync-2026-08-29",
            plan_s3_key="plan.json",
            bucket="alpha-engine-research",
            s3_client=s3,
            emit_metrics=False,
        )

    def test_a_clean_sync_run_reports_full_coverage(self, monkeypatch):
        from evals.judge_coverage import enforce_coverage

        summary = self._run(monkeypatch, n=5)
        assert summary["plan_entry_count"] == 5
        assert summary["plan_entries_graded"] == 5
        assert enforce_coverage(summary)["status"] == "PASS"

    def test_one_failed_judge_call_is_a_shortfall(self, monkeypatch):
        """A per-artifact judge failure used to land in `failed` and leave the
        stage green. It is now a coverage gap with a name."""
        from evals.judge_coverage import JudgeCoverageShortfall, enforce_coverage

        summary = self._run(monkeypatch, n=5, fail_indices=(2,))
        assert summary["plan_entries_graded"] == 4
        assert summary["coverage_complete"] is False
        assert len(summary["ungraded_entries"]) == 1
        with pytest.raises(JudgeCoverageShortfall):
            enforce_coverage(summary)

    def test_escalations_do_not_inflate_coverage(self, monkeypatch):
        """A Sonnet escalation is a SECOND eval of an already-graded entry.

        Counting it would let escalations paper over an ungraded entry
        elsewhere: 4 graded + 1 escalation would read as 5 of 5 while one
        agent silently lost its eval. Here every entry escalates, so a
        double-count would report 10 graded against 5 planned.
        """
        summary = self._run(monkeypatch, n=5, escalate=True)
        assert summary["sonnet_evaluated"] == 5, "escalation tail must have run"
        assert summary["plan_entries_graded"] == 5, "coverage counts entries, not evals"
        assert summary["plan_entry_count"] == 5


class TestExecContextIsResolvedPerCall:
    """The same image runs in two contexts (alpha-engine-config-I9309).

    `JUDGE_EXEC_CONTEXT` was a hard `"lambda"` constant whose own docstring
    said *"a future non-Lambda caller should pass its own declared context
    rather than assume this one"*. The spot box is that caller.

    This matters beyond tidiness: `_entry_reachable_from` gates which registry
    rows a context may use, so a spot box claiming `lambda` could be handed a
    route reachable only from Lambda — failing at connect time, or silently
    resolving a member the registry does not intend for EC2.
    """

    def test_defaults_to_lambda_when_the_environment_states_nothing(
        self, monkeypatch,
    ):
        from evals.judge import JUDGE_EXEC_CONTEXT, judge_exec_context

        monkeypatch.delenv("KREPIS_EXEC_CONTEXT", raising=False)
        assert judge_exec_context() == "lambda" == JUDGE_EXEC_CONTEXT

    def test_reads_the_krepis_env_var_the_spot_bootstrap_exports(
        self, monkeypatch,
    ):
        """`KREPIS_EXEC_CONTEXT`, not a judge-specific name.

        The fact being stated ("this process runs on EC2") belongs to the
        process, and `thinktank_spot_dispatcher` already exports exactly this
        var for the fleet's other EC2-resident routed call site.
        """
        from evals.judge import judge_exec_context

        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ec2")
        assert judge_exec_context() == "ec2"

    def test_is_not_frozen_at_import_time(self, monkeypatch):
        """A value computed once at import is a constant wearing a function's
        clothes — and untestable without reloading the module."""
        from evals.judge import judge_exec_context

        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ec2")
        assert judge_exec_context() == "ec2"
        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")
        assert judge_exec_context() == "lambda"

    def test_the_batch_transport_asks_from_the_same_context(self, monkeypatch):
        """Both judge call sites must agree on where they are.

        A transport resolving as `lambda` while the judge itself resolves as
        `ec2` would make the degradation record name a context the grading
        never ran in — the record exists to be trusted after the fact.
        """
        import evals.judge_batch_transport as jbt

        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "ec2")
        seen = {}

        def _fake_resolve(group, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here — only the exec_context matters")

        import sys
        from types import SimpleNamespace
        monkeypatch.setitem(
            sys.modules, "krepis.router",
            SimpleNamespace(resolve_group_spec=_fake_resolve),
        )
        monkeypatch.setitem(
            sys.modules, "krepis.model_registry",
            SimpleNamespace(CapabilityUnavailableError=type(
                "CapabilityUnavailableError", (Exception,), {},
            )),
        )
        with pytest.raises(RuntimeError):
            jbt.resolve_batch_transport()
        assert seen["exec_context"] == "ec2"
