"""Regression tests for the eval-judge migration off the direct Anthropic API.

alpha-engine-config-I9263. Brian's ruling 2026-08-29, verbatim: *"I will not
fund the anthropic account, at this point we shouldn't be using the anthropic
api at all."*

Every test here fails against the pre-migration code, where
``eval_judge_submit_handler``, the retired ``eval_judge_poll_handler`` and
``eval_judge_process_handler`` each built ``anthropic.Anthropic(
api_key=ANTHROPIC_API_KEY)`` at the call site and one vendor's billing state
took down a whole Step Function branch (three times: alpha-engine-config-I7448,
-I9049, and the 2026-08-29 weekly run).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from evals import judge_batch_transport as jbt


class TestNoProviderSDKAtTheCallSite:
    """The class-level property, asserted structurally rather than per-file.

    A test that only checked the three handlers would pass the day a fourth
    one was written. This reads the shipped source of every module in the
    eval-judge chain.
    """

    @staticmethod
    def _executable_source(path):
        """*path*'s source with every comment and string literal removed.

        The prose in these modules necessarily QUOTES the forbidden
        constructions — that is what makes the migration legible to the next
        reader. Scanning raw text would therefore make documenting the defect
        indistinguishable from committing it, and the cure would be to stop
        explaining. Tokenizing and dropping COMMENT/STRING leaves exactly the
        executable code, which is the only place a client can actually be
        built.
        """
        import io
        import tokenize

        with open(path, "rb") as fh:
            toks = list(tokenize.tokenize(io.BytesIO(fh.read()).readline))
        return " ".join(
            t.string for t in toks
            if t.type not in (tokenize.COMMENT, tokenize.STRING)
        )

    @pytest.mark.parametrize("relpath", [
        "lambda/eval_judge_submit_handler.py",
        # Poll and Process are no longer Lambdas (alpha-engine-config-I9329);
        # the Process leg's substrate-independent entrypoint carries the same
        # no-direct-SDK obligation and takes their place here.
        "evals/judge_spot_run.py",
        "evals/orchestrator.py",
        "evals/judge_batch_transport.py",
    ])
    def test_no_anthropic_sdk_construction(self, relpath):
        from pathlib import Path

        src = self._executable_source(
            Path(__file__).resolve().parent.parent / relpath
        )
        for forbidden in (
            "anthropic . Anthropic (",
            "anthropic . AsyncAnthropic (",
            "from anthropic import Anthropic",
            "ANTHROPIC_API_KEY",
        ):
            assert forbidden not in src, (
                f"{relpath} constructs or keys a provider SDK client directly "
                f"({forbidden!r}). alpha-engine-config-I9263: the judge "
                f"addresses a capability class through the router; a provider "
                f"client belongs in the krepis adapter, never here."
            )


class TestBatchCapabilityResolution:

    def test_unroutable_capability_is_a_ladder_signal_not_a_crash(
        self, monkeypatch,
    ):
        """krepis cannot express `batches` today, so resolution must degrade.

        This is the branch that fires in production as of 2026-08-29:
        ``ROUTABLE_CAPABILITIES == ("tool_choice", "streaming")``.
        """
        import krepis.router as kr

        def _reject(group, **kwargs):
            raise ValueError(
                "'batches' is not a routable capability; declared "
                "capabilities are ['tool_choice', 'streaming']."
            )

        monkeypatch.setattr(kr, "resolve_group_spec", _reject)
        with pytest.raises(jbt.BatchCapabilityUnavailable) as excinfo:
            jbt.resolve_batch_transport()
        assert excinfo.value.capability == "batches"
        assert "cannot route on it" in excinfo.value.reason

    def test_unknown_group_valueerror_is_NOT_laundered_into_a_degradation(
        self, monkeypatch,
    ):
        """A misconfiguration must fail loud, not spend 2x quietly."""
        import krepis.router as kr

        def _reject(group, **kwargs):
            raise ValueError("Model group 'nope' not found in registry.")

        monkeypatch.setattr(kr, "resolve_group_spec", _reject)
        with pytest.raises(ValueError, match="not found in registry"):
            jbt.resolve_batch_transport()

    def test_no_declaring_member_is_a_ladder_signal(self, monkeypatch):
        import krepis.router as kr
        from krepis.model_registry import CapabilityUnavailableError

        def _reject(group, **kwargs):
            raise CapabilityUnavailableError("low", "batches", [("m1", "does not declare it")])

        monkeypatch.setattr(kr, "resolve_group_spec", _reject)
        with pytest.raises(jbt.BatchCapabilityUnavailable) as excinfo:
            jbt.resolve_batch_transport()
        assert "no live group member declares it" in excinfo.value.reason

    def test_a_resolved_batch_route_never_silently_degrades(self):
        """Resolvable-but-undrivable is a defect, not a degradation."""
        with pytest.raises(NotImplementedError, match="alpha-engine-config-I9263"):
            jbt.batch_client_for_route(MagicMock(model="m"), {"route": "litellm_proxy"})


class TestDegradationIsRecordedDurably:

    def test_record_carries_what_a_reader_needs(self):
        rec = jbt.build_degradation_record(
            date="2026-08-29", reason="because", group="low",
            capability="batches", exec_context="lambda", request_count=42,
            plan_s3_key="k",
        )
        assert rec["requested_rung"] == "batch"
        assert rec["served_rung"] == "sync"
        assert rec["request_count"] == 42
        assert rec["batch_discount_forfeited_multiplier"] == 2.0
        # The migration is a net cost REDUCTION against the route it replaced —
        # asserted so a future edit cannot quietly restate it as an increase.
        assert rec["cost_ratio_vs_retired_batch_route"]["output"] < 1.0
        assert "shouldn't be using the anthropic api" in rec["ruling"]

    def test_persist_raises_rather_than_degrading_unobserved(self):
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("s3 down")
        rec = jbt.build_degradation_record(
            date="2026-08-29", reason="r", group="low", capability="batches",
            exec_context="lambda", request_count=1,
        )
        with pytest.raises(RuntimeError, match="s3 down"):
            jbt.persist_degradation_record(rec, bucket="b", s3_client=s3)

    def test_persist_writes_the_record_beside_the_plan(self):
        s3 = MagicMock()
        rec = jbt.build_degradation_record(
            date="2026-08-29", reason="r", group="low", capability="batches",
            exec_context="lambda", request_count=3,
        )
        key = jbt.persist_degradation_record(rec, bucket="b", s3_client=s3)
        assert key == (
            "decision_artifacts/_eval_batch_plans/2026-08-29/degradation.json"
        )
        body = s3.put_object.call_args.kwargs["Body"]
        assert json.loads(body)["served_rung"] == "sync"


class TestSyncRungSentinel:

    def test_sync_batch_ids_are_recognised(self):
        assert jbt.is_sync_batch_id(jbt.sync_batch_id("2026-08-29"))
        assert not jbt.is_sync_batch_id("msgbatch_abc")
        assert not jbt.is_sync_batch_id("empty-2026-08-29")

    def test_poll_treats_a_sync_batch_as_terminal_without_a_client(self):
        from evals.orchestrator import poll_batch

        # No client passed at all — the pre-migration signature REQUIRED one.
        result = poll_batch(batch_id=jbt.sync_batch_id("2026-08-29"))
        assert result["processing_status"] == "ended"


def _minimal_plan(*, date="2026-08-29", n=2):
    entries = [
        {
            "custom_id": f"cid{i}",
            "capture_s3_key": f"captures/{date}/a{i}.json",
            "agent_id": f"agent_{i}",
            "run_id": f"run_{i}",
            "rubric_id": "rubric_x",
            "judge_model": "claude-haiku-4-5",
        }
        for i in range(n)
    ]
    return {
        "bucket": "alpha-engine-research",
        "date": date,
        "requests": [{"custom_id": e["custom_id"]} for e in entries],
        "plan_entries": entries,
        "eval_prefix": "evals/",
        "cw_namespace": "ns",
        "force_sonnet_pass": False,
        "haiku_model": "claude-haiku-4-5",
        "sonnet_model": "claude-sonnet-4-6",
        "judge_run_id": "jr-1",
        "client_side_skips": [],
        "capture_keys_total": n,
        "skipped_unmapped": 0,
    }


class TestSubmitTakesTheSyncRung:
    """End-to-end: the exact production shape as of 2026-08-29.

    Pre-migration this call raised ``400 ... 'Your credit balance is too low to
    access the Anthropic API'`` and set ``branch_a_degraded`` for the whole
    weekly run. It must now land on the sync rung, loudly and durably.
    """

    def test_submit_degrades_and_records(self, monkeypatch):
        import evals.orchestrator as orch

        def _unavailable(**kwargs):
            raise jbt.BatchCapabilityUnavailable(
                group="low", capability="batches", exec_context="lambda",
                reason="no batch route exists",
            )

        monkeypatch.setattr(orch, "resolve_batch_transport", _unavailable)
        s3 = MagicMock()
        # No client is passed — the pre-migration signature required one.
        result = orch.submit_batch(_minimal_plan(), s3_client=s3)

        assert result["batch_id"] == "sync-2026-08-29"
        assert result["processing_status"] == "ended_sync"
        assert result["degraded"] is True
        assert result["degradation"]["served_rung"] == "sync"
        assert result["degradation"]["request_count"] == 2
        written = {c.kwargs["Key"] for c in s3.put_object.call_args_list}
        assert any("degradation.json" in k for k in written), (
            "the degradation must reach a durable artifact, not only a log line"
        )

    def test_a_real_transport_fault_is_not_a_degradation(self, monkeypatch):
        """The router being down must fail loud, not spend 2x quietly."""
        import evals.orchestrator as orch

        def _boom(**kwargs):
            raise ConnectionError("router unreachable")

        monkeypatch.setattr(orch, "resolve_batch_transport", _boom)
        with pytest.raises(ConnectionError):
            orch.submit_batch(_minimal_plan(), s3_client=MagicMock())


class TestProcessRunsTheSyncRung:

    def test_every_plan_entry_is_judged_through_the_router_path(
        self, monkeypatch,
    ):
        import evals.orchestrator as orch

        plan = _minimal_plan(n=3)
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(plan).encode())
        }
        monkeypatch.setattr(orch, "_load_capture_artifact", lambda *a, **k: object())
        judged = []

        def _fake_eval(artifact, **kwargs):
            judged.append(kwargs["judge_model"])
            ev = MagicMock()
            ev.judge_model = kwargs["judge_model"]
            return ev

        monkeypatch.setattr(orch, "evaluate_artifact", _fake_eval)
        monkeypatch.setattr(orch, "persist_eval_artifact", lambda *a, **k: "key")
        monkeypatch.setattr(orch, "should_escalate_to_sonnet", lambda *a, **k: False)

        summary = orch.process_batch_results(
            batch_id="sync-2026-08-29",
            plan_s3_key="plan.json",
            bucket="alpha-engine-research",
            s3_client=s3,
            emit_metrics=False,
        )
        assert len(judged) == 3, "every plan entry must be judged on the sync rung"
        assert summary["sync_fallback_evaluated"] == 3
        assert summary["degraded_transport"] is True
        assert summary["haiku_evaluated"] == 3
        assert summary["complete"] is True

    def test_the_batch_rung_reports_its_transport_too(self, monkeypatch):
        """`degraded_transport` is emitted on BOTH rungs.

        A field that appears only when something is wrong is
        indistinguishable from a field nobody emitted.
        """
        import evals.orchestrator as orch

        plan = _minimal_plan(n=0)
        plan["requests"] = []
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(plan).encode())
        }
        summary = orch.process_batch_results(
            batch_id="empty-2026-08-29",
            plan_s3_key="plan.json",
            bucket="alpha-engine-research",
            s3_client=s3,
            emit_metrics=False,
        )
        assert summary["degraded_transport"] is False
        assert summary["sync_fallback_evaluated"] == 0
