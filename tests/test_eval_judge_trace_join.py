"""Every EvalJudge verdict names its grader and joins to the judge call's own
``llm_trace.v1`` row (alpha-engine-config-I12312 programme item 4, carrying
alpha-engine-config-I11484's grader identity).

The verdict already recorded the served model (``judge_resolved_model``), the
prompt version (``rubric_version`` + ``rubric_hash``) and the JUDGED
artifact's trace (``judged_trace_id``). These tests pin the remaining facts on
``judge_call``: the router group, and the judge call's own ``trace_id`` /
``span_id`` plus the run id of the trace sink its row went to — and that the
``latest.json`` sidecar and the ``_eval_by_capture`` manifest row carry them
flat, so an eval can join a verdict to its trace without opening the verdict.
"""

from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from evals import eval_manifest
from evals import judge as judge_mod
from evals.eval_manifest import JUDGE_CALL_JOIN_KEYS, _make_manifest_entry
from graph.state_schemas import JudgeCallRef, RubricEvalArtifact
from tests.test_eval_judge import (  # noqa: F401 — live_router_resolution is an autouse fixture
    _patch_llm_client,
    live_router_resolution,
)
from tests.test_eval_judge_citations import (
    _cited_prompt,
    _judge_args,
    _response,
    _thesis_artifact,
)

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN = "00f067aa0ba902b7"
_KREPIS_HAS_LLM_TRACE = importlib.util.find_spec("krepis.llm_trace") is not None


class _CostSink:
    run_id = "judge-run-cost-01"

    def __call__(self, record):
        pass


class _TraceSink:
    """Stands in for krepis's ``S3JsonlTraceSink``: ``sink(row, blobs)``."""

    run_id = "judge-run-trace-01"

    def __init__(self):
        self.rows = []

    def __call__(self, row, blobs=None):
        self.rows.append(row)


# ── _judge_call_ref: the read from krepis's LLMResult ─────────────────────


def _call_ref(result, **kw):
    kw.setdefault("callsite_id", "evaljudge-sync")
    kw.setdefault("cost_sink", _CostSink())
    kw.setdefault("attempts", 1)
    kw.setdefault("max_tokens", 4096)
    return judge_mod._judge_call_ref(result, **kw)


class TestJudgeCallRef:
    def test_records_trace_ids_sink_run_and_group(self):
        result = SimpleNamespace(trace_id=TRACE, span_id=SPAN, trace_emission_error=None)
        ref = _call_ref(result, trace_sink=_TraceSink(), model_group="low")
        assert (ref.trace_id, ref.span_id) == (TRACE, SPAN)
        assert ref.trace_run_id == _TraceSink.run_id
        assert ref.model_group == "low"
        assert ref.trace_emission_error is None

    def test_no_trace_sink_says_no_row_was_written(self):
        # krepis stamps trace ids whether or not a sink exists; trace_run_id
        # None is what tells a reader the ids name a row that is not there.
        ref = _call_ref(SimpleNamespace(trace_id=TRACE, span_id=SPAN), trace_sink=None, model_group="low")
        assert ref.trace_id == TRACE
        assert ref.trace_run_id is None

    def test_records_a_failed_trace_emission(self):
        result = SimpleNamespace(trace_id=TRACE, span_id=SPAN, trace_emission_error="AccessDenied")
        ref = _call_ref(result, trace_sink=_TraceSink(), model_group="low")
        assert ref.trace_emission_error == "AccessDenied"

    def test_krepis_without_llm_trace_records_none(self):
        # krepis < 0.59.74 has no trace fields on LLMResult.
        ref = _call_ref(SimpleNamespace(), model_group="low")
        assert (ref.trace_id, ref.span_id, ref.trace_run_id, ref.trace_emission_error) == (None,) * 4
        assert ref.model_group == "low"

    def test_old_verdict_without_the_fields_still_validates(self):
        ref = JudgeCallRef.model_validate({"callsite_id": "evaljudge-sync", "attempts": 1})
        assert (ref.model_group, ref.trace_id, ref.span_id, ref.trace_run_id) == (None,) * 4


# ── End to end through the real krepis client over a stubbed transport ────


def _run_sync_judge(monkeypatch, *, trace_sink):
    monkeypatch.setattr(judge_mod, "load_prompt", lambda name: _cited_prompt())
    monkeypatch.setattr(judge_mod, "_judge_cost_sink", lambda: _CostSink())
    art = _thesis_artifact()
    fake = MagicMock()
    fake.chat.completions.create.return_value = _response(_judge_args(art))
    with _patch_llm_client(judge_mod, fake) as llm_cls:
        if trace_sink is not None:
            build = llm_cls.side_effect

            def _with_trace_sink(spec, **kwargs):
                client = build(spec, **kwargs)
                client._trace_sink = trace_sink
                return client

            llm_cls.side_effect = _with_trace_sink
        verdict = judge_mod.evaluate_artifact(
            art,
            judge_model="claude-haiku-4-5",
            api_key="sk-test",
            judged_artifact_s3_key="decision_artifacts/2026/10/12/thinktank_thesis/tt-run-001.json",
        )
    return verdict


class TestSyncRungEndToEnd:
    def test_verdict_names_its_grader(self, monkeypatch):
        verdict = _run_sync_judge(monkeypatch, trace_sink=None)
        call = verdict.judge_call
        assert call.model_group == judge_mod.JUDGE_MODEL_GROUP == "low"
        assert verdict.judge_resolved_model == "deepseek-v4-flash"
        assert verdict.rubric_version == "1.0.0" and verdict.rubric_hash == "f" * 64

    def test_verdict_joins_to_the_judge_calls_trace_row(self, monkeypatch):
        sink = _TraceSink()
        verdict = _run_sync_judge(monkeypatch, trace_sink=sink)
        call = verdict.judge_call
        assert call.trace_run_id == _TraceSink.run_id
        if _KREPIS_HAS_LLM_TRACE:
            # krepis wrote exactly one row for the judge call, and the verdict
            # carries that row's keys.
            assert len(sink.rows) == 1
            row = sink.rows[0]
            assert (call.trace_id, call.span_id) == (row["trace_id"], row["span_id"])
            assert len(call.trace_id) == 32 and len(call.span_id) == 16
            from krepis.llm_trace import row_extension

            ext = row_extension(row)
            assert ext["callsite_id"] == "evaljudge-sync"
            # The row is keyed on the run of the sink the verdict names.
            assert ext["run_id"] == call.trace_run_id
        else:
            # krepis < 0.59.74 stamps no trace ids and writes no row.
            assert sink.rows == []
            assert (call.trace_id, call.span_id) == (None, None)

    def test_without_a_trace_sink_no_run_is_claimed(self, monkeypatch):
        verdict = _run_sync_judge(monkeypatch, trace_sink=None)
        assert verdict.judge_call.trace_run_id is None


# ── Flat projection onto the sidecar and the manifest row ─────────────────


def _verdict(**call) -> RubricEvalArtifact:
    return RubricEvalArtifact(
        run_id="tt-run-001",
        judge_run_id="jr-1",
        timestamp="2026-10-16T00:00:00Z",
        judged_agent_id="thinktank_thesis",
        judged_artifact_s3_key="decision_artifacts/2026/10/12/thinktank_thesis/tt-run-001.json",
        rubric_id="eval_rubric_thinktank_thesis",
        rubric_version="1.0.0",
        judge_model="claude-haiku-4-5",
        judge_resolved_model="deepseek-v4-flash",
        overall_reasoning="o",
        judge_call=JudgeCallRef(callsite_id="evaljudge-sync", **call) if call else None,
    )


class TestFlatProjection:
    def test_sidecar_and_manifest_carry_grader_and_trace_keys(self):
        verdict = _verdict(model_group="low", trace_id=TRACE, span_id=SPAN, trace_run_id="judge-run-trace-01")
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="eval-test-bucket")
            key = judge_mod.persist_eval_artifact(verdict, s3_client=s3, bucket="eval-test-bucket")
            body = json.loads(s3.get_object(Bucket="eval-test-bucket", Key=key)["Body"].read())
            sidecar = json.loads(
                s3.get_object(Bucket="eval-test-bucket", Key="decision_artifacts/_eval/latest.json")["Body"].read()
            )
        for row in (sidecar, _make_manifest_entry(body, key)):
            assert row["judge_model_group"] == "low"
            assert (row["judge_trace_id"], row["judge_span_id"]) == (TRACE, SPAN)
            assert row["judge_trace_run_id"] == "judge-run-trace-01"

    def test_keys_are_absent_not_none_when_unknown(self):
        # Batch rung: no router group, no krepis trace.
        verdict = _verdict(custom_id="cid-1")
        entry = _make_manifest_entry(json.loads(verdict.model_dump_json()), "k")
        for _, flat in JUDGE_CALL_JOIN_KEYS:
            assert flat not in entry
        assert all(flat not in judge_mod._sidecar_citation_fields(verdict) for _, flat in JUDGE_CALL_JOIN_KEYS)

    def test_one_key_map_serves_both_projections(self):
        assert judge_mod.JUDGE_CALL_JOIN_KEYS is eval_manifest.JUDGE_CALL_JOIN_KEYS
        assert {src for src, _ in JUDGE_CALL_JOIN_KEYS} <= set(JudgeCallRef.model_fields)

    @pytest.mark.parametrize("call", [None, {}])
    def test_verdict_without_judge_call_projects_nothing_extra(self, call):
        verdict = _verdict()
        body = json.loads(verdict.model_dump_json())
        if call is not None:
            body["judge_call"] = call
        entry = _make_manifest_entry(body, "k")
        for _, flat in JUDGE_CALL_JOIN_KEYS:
            assert flat not in entry
