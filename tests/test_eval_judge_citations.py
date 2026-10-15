"""Judge evidence contract ``judge_evidence.v1`` (alpha-engine-config-I12312).

Every EvalJudge finding must be tied to the text it rests on: a verbatim quote
of the JUDGED output at a field path + character span, and the id of the judge
input element it was checked against. These tests pin:

* the schema (LLM-facing lenient shape, persisted validated shape, the tool
  spec the judge is shown);
* the bundle the judge is shown (stable ids, raw output fields);
* the deterministic validator — exact, relocated, unlocated, unknown /
  missing source_ref, empty evidence;
* backward compatibility for verdicts written before the contract;
* an end-to-end sync judge call over a stubbed transport, through to the
  persisted verdict, its ``latest.json`` sidecar and its manifest row.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws
from nousergon_lib.decision_capture import DecisionArtifact, FullPromptContext, ModelMetadata

from agents.prompt_loader import LoadedPrompt
from evals import judge as judge_mod
from evals.eval_manifest import _make_manifest_entry
from evals.judge_citations import (
    CITATION_CONTRACT,
    build_bundle,
    build_judge_inputs,
    declared_citation_contract,
    flatten_output,
    render_cited_rubric,
    validate_dimension_scores,
    validate_evidence,
)
from graph.state_schemas import (
    CitedRubricDimensionLLM,
    CitedRubricEvalLLMOutput,
    ClaimSpan,
    JudgeEvidence,
    RubricDimensionScore,
    RubricEvalArtifact,
)
from tests.test_eval_judge import (  # noqa: F401 — live_router_resolution is an autouse fixture
    _openai_tool_call,
    _patch_llm_client,
    live_router_resolution,
)

# ── Fixtures ────────────────────────────────────────────────────────────

RISK_0 = "Customer concentration: the top three buyers are 41% of revenue."
MOAT = "Switching costs are high; migrations take quarters. Switching costs compound."
SUMMARY = "Attractive at 18x earnings with a durable moat."


def _thesis_artifact(**extra_prompt) -> DecisionArtifact:
    return DecisionArtifact(
        run_id="tt-run-001",
        timestamp="2026-10-12T22:30:00.000Z",
        agent_id="thinktank_thesis",
        model_metadata=ModelMetadata(model_name="deepseek-v4-flash"),
        full_prompt_context=FullPromptContext(
            system_prompt="sys",
            user_prompt=extra_prompt.get("user_prompt", "<rendered at run time>"),
        ),
        input_data_snapshot={
            "ticker": "QLYS",
            "update_reason": "initial",
            "event_context": None,
            "board_row": {"pe": 18.2, "rev_growth": 0.11},
            "news_aggregate": {},
            "filings_excerpts": [
                "[10-K | 2026-02-14]\nTop three customers accounted for 41% of revenue.",
                "[8-K | 2026-09-04]\nAnnounced a $200M buyback.",
            ],
            "prior_thesis": None,
        },
        agent_output={
            "ticker": "QLYS",
            "version": 1,
            "thesis": {
                "moat": MOAT,
                "risks": [RISK_0, "Pricing pressure from bundled suites."],
                "summary": SUMMARY,
                "conviction": 64,
            },
        },
    )


def _ev(quote, field="", start=None, end=None, ref="in:board_row") -> JudgeEvidence:
    return JudgeEvidence(
        claim_quote=quote,
        claim_span=ClaimSpan(field=field, start=start, end=end),
        source_ref=ref,
    )


CITED_TEMPLATE = (
    "# rubric\n\n═══ DIMENSION 1: input_groundedness ═══\nGrounded?\n\n"
    "═══ DIMENSION 2: risk_specificity ═══\nSpecific?\n\n"
    f"═══ EVIDENCE CONTRACT: {CITATION_CONTRACT} ═══\nCite.\n\n"
    "Agent input:\n{agent_input}\n\nAgent output:\n{agent_output}\n"
)


def _cited_prompt() -> LoadedPrompt:
    from pathlib import Path

    return LoadedPrompt(
        name="eval_rubric_thinktank_thesis",
        text=CITED_TEMPLATE,
        version="1.0.0",
        hash="f" * 64,
        source_path=Path("/dev/null"),
    )


# ── Activation + rendering ──────────────────────────────────────────────


class TestActivationAndRendering:
    def test_contract_is_declared_in_the_rubric_text(self):
        assert declared_citation_contract(CITED_TEMPLATE) == "judge_evidence.v1"
        assert declared_citation_contract("═══ DIMENSION 1: x ═══\n{agent_input}") is None
        assert declared_citation_contract("") is None

    def test_uncited_rubric_renders_byte_identically_to_the_legacy_renderer(self):
        prompt = SimpleNamespace(text="IN:\n{agent_input}\nOUT:\n{agent_output}")
        art = _thesis_artifact()
        rendered = judge_mod._render_for_judge(art, prompt)
        assert rendered.contract is None and rendered.bundle is None
        assert rendered.text == judge_mod._render_rubric(art, prompt)

    def test_input_ids_are_stable_snapshot_ordered_and_skip_empties(self):
        elements, absent = build_judge_inputs(_thesis_artifact().input_data_snapshot)
        assert [e.id for e in elements] == [
            "in:ticker",
            "in:update_reason",
            "in:board_row",
            "in:filings_excerpts[0]",
            "in:filings_excerpts[1]",
        ]
        assert absent == ("event_context", "news_aggregate", "prior_thesis")
        # Strings render verbatim; containers as JSON.
        assert elements[0].text == "QLYS"
        assert json.loads(elements[2].text) == {"pe": 18.2, "rev_growth": 0.11}

    def test_cid_source_blocks_in_the_producer_prompt_become_inputs(self):
        prompt = (
            'Board:\n<source cid="board">\n{"pe": 18.2}\n</source>\n'
            'Filings:\n<source cid="f1" kind="rag_chunk">\nTop three customers.\n</source>\n'
            '<source cid="f1">\nduplicate cid is ignored\n</source>'
        )
        elements, _ = build_judge_inputs({}, producer_prompt=prompt)
        assert [(e.id, e.text) for e in elements] == [
            ("cid:board", '{"pe": 18.2}'),
            ("cid:f1", "Top three customers."),
        ]

    def test_output_fields_are_raw_strings_by_path(self):
        out = flatten_output(_thesis_artifact().agent_output)
        assert out["thesis.risks[0]"] == RISK_0
        assert out["thesis.conviction"] == "64"
        assert out["version"] == "1"

    def test_cited_render_labels_every_element(self):
        art = _thesis_artifact()
        text = render_cited_rubric(CITED_TEMPLATE, build_bundle(art))
        assert '<input id="in:filings_excerpts[0]">\n[10-K | 2026-02-14]' in text
        assert f'<output field="thesis.risks[0]">\n{RISK_0}\n</output>' in text
        assert "Absent inputs (empty — no id, cannot be cited): event_context, news_aggregate, prior_thesis" in text
        assert "{agent_input}" not in text and "{agent_output}" not in text


# ── Deterministic validator ─────────────────────────────────────────────


class TestValidator:
    def setup_method(self):
        self.bundle = build_bundle(_thesis_artifact())
        self.ids = self.bundle.input_ids
        self.out = self.bundle.outputs

    def _v(self, ev):
        return validate_evidence(ev, outputs=self.out, input_ids=self.ids)

    def test_exact(self):
        q = "top three buyers are 41% of revenue"
        start = RISK_0.index(q)
        v = self._v(_ev(q, "thesis.risks[0]", start, start + len(q), "in:filings_excerpts[0]"))
        assert (v.location, v.source_status, v.valid, v.reason) == ("exact", "known", True, None)
        assert v.located_span == ClaimSpan(field="thesis.risks[0]", start=start, end=start + len(q))

    def test_relocated_wrong_offsets_unique_in_field(self):
        q = "top three buyers"
        v = self._v(_ev(q, "thesis.risks[0]", 0, 5, "in:filings_excerpts[0]"))
        assert v.location == "relocated" and v.valid and v.reason == "span_mismatch"
        assert v.located_span.start == RISK_0.index(q)
        # The judge's own span is kept, never rewritten.
        assert (v.claim_span.start, v.claim_span.end) == (0, 5)

    def test_relocated_wrong_field_unique_elsewhere(self):
        v = self._v(_ev("18x earnings", "thesis.moat", None, None, "in:board_row"))
        assert v.location == "relocated" and v.reason == "field_mismatch" and v.valid
        assert v.located_span.field == "thesis.summary"

    def test_relocated_unknown_field_path(self):
        v = self._v(_ev("18x earnings", "thesis.valuation", 1, 2, "in:board_row"))
        assert v.location == "relocated" and v.reason == "unknown_field"

    def test_unlocated_paraphrase(self):
        v = self._v(_ev("buyers make up 41 percent", "thesis.risks[0]", 0, 10))
        assert v.location == "unlocated" and not v.valid and v.reason == "quote_not_found"
        assert v.located_span is None

    def test_unlocated_ambiguous_in_field(self):
        v = self._v(_ev("Switching costs", "thesis.moat"))
        assert v.location == "unlocated" and v.reason == "quote_ambiguous" and not v.valid

    def test_exact_span_wins_over_ambiguity(self):
        q = "Switching costs"
        second = MOAT.index(q, 1)
        v = self._v(_ev(q, "thesis.moat", second, second + len(q)))
        assert v.location == "exact" and v.valid

    def test_unlocated_ambiguous_across_fields(self):
        # Not in the named field; twice elsewhere — no single place to point.
        v = self._v(_ev("Switching costs", "thesis.summary"))
        assert v.location == "unlocated" and v.reason == "quote_ambiguous"

    def test_relocated_unique_elsewhere_from_a_wrong_field(self):
        v = self._v(_ev("QLYS", "thesis.summary", ref="in:ticker"))
        assert v.location == "relocated" and v.located_span.field == "ticker" and v.valid

    def test_unknown_source_ref(self):
        v = self._v(_ev(SUMMARY, "thesis.summary", 0, len(SUMMARY), "in:news_aggregate"))
        # news_aggregate was EMPTY, so it was never offered under an id.
        assert v.location == "exact" and v.source_status == "unknown"
        assert not v.valid and v.reason == "unknown_source_ref"

    def test_missing_source_ref(self):
        v = self._v(_ev(SUMMARY, "thesis.summary", 0, len(SUMMARY), ""))
        assert v.source_status == "missing" and not v.valid

    def test_empty_quote(self):
        v = self._v(_ev("", "thesis.summary", 0, 0, "in:board_row"))
        assert v.location == "unlocated" and v.reason == "empty_quote" and not v.valid

    def test_unlocated_and_unknown_reasons_compose(self):
        v = self._v(_ev("not there", "thesis.summary", ref="in:made_up"))
        assert v.reason == "quote_not_found+unknown_source_ref"

    def test_dimension_scores_counts_and_unsupported_flags(self):
        q = "top three buyers are 41% of revenue"
        start = RISK_0.index(q)
        scores = [
            CitedRubricDimensionLLM(
                dimension="risk_specificity", score=4, reasoning="r",
                evidence=[
                    _ev(q, "thesis.risks[0]", start, start + len(q), "in:filings_excerpts[0]"),
                    _ev("paraphrase", "thesis.risks[0]", ref="in:filings_excerpts[0]"),
                ],
            ),
            CitedRubricDimensionLLM(dimension="input_groundedness", score=2, reasoning="r", evidence=[]),
            CitedRubricDimensionLLM(
                dimension="valuation_linkage", score=3, reasoning="r",
                evidence=[_ev("18x earnings", "thesis.summary", ref="in:nothing")],
            ),
        ]
        scored, check = validate_dimension_scores(scores, self.bundle)
        assert [s.valid_evidence_count for s in scored] == [1, 0, 0]
        assert [s.unsupported for s in scored] == [False, True, True]
        assert check.citations_total == 3
        assert check.citations_valid == 1
        assert check.citations_invalid == 2
        assert (check.located_exact, check.located_relocated, check.unlocated) == (1, 1, 1)
        assert (check.unknown_source_ref, check.missing_source_ref) == (1, 0)
        assert (check.findings_total, check.findings_unsupported) == (3, 2)
        assert check.unsupported_dimensions == ["input_groundedness", "valuation_linkage"]
        assert check.inputs_offered == 5
        assert check.output_fields_offered == len(self.out)


# ── Schema ──────────────────────────────────────────────────────────────


class TestSchema:
    def test_tool_schema_requires_evidence_but_parser_is_lenient(self):
        spec = judge_mod._build_rubric_tool_spec(cited=True)
        defs = spec["input_schema"]["$defs"]
        assert "evidence" in defs["CitedRubricDimensionLLM"]["required"]
        assert set(defs["JudgeEvidence"]["required"]) == {"claim_quote", "claim_span", "source_ref"}
        assert spec["name"] == "RubricEvalLLMOutput"
        # A judge that omits evidence still parses — and is then flagged.
        out = CitedRubricEvalLLMOutput.model_validate(
            {"dimension_scores": [{"dimension": "a", "score": 3, "reasoning": "r"}], "overall_reasoning": "o"}
        )
        assert out.dimension_scores[0].evidence == []

    def test_uncited_tool_spec_is_unchanged(self):
        from graph.state_schemas import RubricEvalLLMOutput

        assert judge_mod._build_rubric_tool_spec()["input_schema"] == RubricEvalLLMOutput.model_json_schema()

    def test_shape_coercions_never_repair_content(self):
        e = JudgeEvidence.model_validate({"claim_quote": None, "claim_span": [3, 9], "source_ref": None})
        assert (e.claim_quote, e.source_ref) == ("", "")
        assert e.claim_span == ClaimSpan(field="", start=3, end=9)
        d = CitedRubricDimensionLLM.model_validate(
            {"dimension": "a", "score": 2, "reasoning": "r",
             "evidence": json.dumps([{"claim_quote": "x", "claim_span": "thesis.moat", "source_ref": "in:a"}])}
        )
        assert d.evidence[0].claim_span.field == "thesis.moat"

    def test_parse_batch_message_cited(self):
        payload = {"content": [{"type": "tool_use", "name": "RubricEvalLLMOutput", "input": {
            "dimension_scores": [{"dimension": "a", "score": 4, "reasoning": "r", "evidence": [
                {"claim_quote": "q", "claim_span": {"field": "f", "start": 0, "end": 1}, "source_ref": "in:x"}]}],
            "overall_reasoning": "o"}}]}
        out = judge_mod.parse_batch_message(payload, cited=True)
        assert isinstance(out, CitedRubricEvalLLMOutput)
        assert out.dimension_scores[0].evidence[0].source_ref == "in:x"


# ── Backward compatibility ─────────────────────────────────────────────


_V2_VERDICT = {
    "schema_version": 2,
    "run_id": "r1",
    "judge_run_id": "2610121530",
    "timestamp": "2026-10-12T15:30:00Z",
    "judged_agent_id": "thinktank_thesis",
    "judged_artifact_s3_key": "decision_artifacts/2026/10/12/thinktank_thesis/r1.json",
    "rubric_id": "eval_rubric_thinktank_thesis",
    "rubric_version": "0.2.0",
    "judge_model": "claude-haiku-4-5",
    "judge_request_model": "low",
    "judge_resolved_model": "deepseek-v4-flash",
    "dimension_scores": [{"dimension": "input_groundedness", "score": 4, "reasoning": "ok"}],
    "overall_reasoning": "fine",
    "judge_skip_reason": None,
}


class TestBackwardCompat:
    def test_v2_verdict_reads_unchanged(self):
        a = RubricEvalArtifact.model_validate_json(json.dumps(_V2_VERDICT))
        assert a.schema_version == 2
        assert a.citation_check is None and a.judge_call is None and a.judge_inputs == []
        assert a.judged_trace_id is None
        d = a.dimension_scores[0]
        assert (d.dimension, d.score, d.evidence, d.unsupported, d.valid_evidence_count) == (
            "input_groundedness", 4, [], None, None,
        )

    def test_v2_manifest_row_and_sidecar_are_unchanged(self):
        entry = _make_manifest_entry(_V2_VERDICT, "decision_artifacts/_eval/x.json")
        assert "citations_total" not in entry and "judged_trace_id" not in entry
        a = RubricEvalArtifact.model_validate(_V2_VERDICT)
        assert judge_mod._sidecar_citation_fields(a) == {}

    def test_new_verdicts_are_written_as_v3(self):
        a = RubricEvalArtifact.model_validate({**_V2_VERDICT, "schema_version": 3})
        assert a.schema_version == 3
        assert RubricEvalArtifact.model_fields["schema_version"].default == 3

    def test_unvalidated_evidence_on_an_uncited_score_is_not_persisted(self):
        stray = RubricDimensionScore(dimension="d", score=3, reasoning="r", evidence=[{"claim_quote": "x"}])
        a = RubricEvalArtifact.model_validate({**_V2_VERDICT, "dimension_scores": [stray]})
        assert a.dimension_scores[0].evidence == []
        assert a.dimension_scores[0].unsupported is None


# ── End to end over a stubbed transport ─────────────────────────────────


class _Sink:
    run_id = "judge-run-cost-01"

    def __init__(self):
        self.records = []

    def __call__(self, record):
        self.records.append(record)


def _response(args: dict, *, resp_id="gen-abc123"):
    message = SimpleNamespace(content=None, tool_calls=[_openai_tool_call("RubricEvalLLMOutput", args)])
    choice = SimpleNamespace(finish_reason="tool_calls", message=message)
    return SimpleNamespace(id=resp_id, choices=[choice], model="deepseek-v4-flash", usage=SimpleNamespace(cost=0.0002))


def _judge_args(art: DecisionArtifact) -> dict:
    q = "top three buyers are 41% of revenue"
    start = RISK_0.index(q)
    return {
        "dimension_scores": [
            {"dimension": "input_groundedness", "score": 4, "reasoning": "Risk ties to the 10-K.",
             "evidence": [
                 {"claim_quote": q, "claim_span": {"field": "thesis.risks[0]", "start": start, "end": start + len(q)},
                  "source_ref": "in:filings_excerpts[0]"},
                 {"claim_quote": "18x earnings", "claim_span": {"field": "thesis.summary", "start": 0, "end": 3},
                  "source_ref": "in:board_row"},
             ]},
            {"dimension": "risk_specificity", "score": 2, "reasoning": "Second risk is generic.",
             "evidence": [
                 {"claim_quote": "pricing pressure", "claim_span": {"field": "thesis.risks[1]", "start": 0, "end": 16},
                  "source_ref": "in:analyst_revisions"},
             ]},
            {"dimension": "made_up_dimension", "score": 5, "reasoning": "phantom", "evidence": []},
        ],
        "overall_reasoning": "Grounded risks; weak specificity.",
    }


class TestEndToEnd:
    @pytest.fixture
    def judged(self, monkeypatch):
        sink = _Sink()
        monkeypatch.setattr(judge_mod, "load_prompt", lambda name: _cited_prompt())
        monkeypatch.setattr(judge_mod, "_judge_cost_sink", lambda: sink)
        base = _thesis_artifact()
        # DecisionArtifact carries trace_id/span_id natively since nousergon-lib
        # v0.124.173 (lib-PR472); crucible-research#849 stamps them.
        art = base.model_copy(update={"trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "span_id": "00f067aa0ba902b7"})
        fake = MagicMock()
        fake.chat.completions.create.return_value = _response(_judge_args(art))
        with _patch_llm_client(judge_mod, fake):
            verdict = judge_mod.evaluate_artifact(
                art, judge_model="claude-haiku-4-5", api_key="sk-test",
                judged_artifact_s3_key="decision_artifacts/2026/10/12/thinktank_thesis/tt-run-001.json",
            )
        return verdict, fake, sink

    def test_verdict_carries_validated_evidence(self, judged):
        verdict, _, _ = judged
        assert verdict.schema_version == 3
        # Undeclared dimension dropped BEFORE validation, as before I12312.
        assert [d.dimension for d in verdict.dimension_scores] == ["input_groundedness", "risk_specificity"]
        g, r = verdict.dimension_scores
        assert [e.location for e in g.evidence] == ["exact", "relocated"]
        assert g.evidence[1].located_span.field == "thesis.summary"
        assert (g.valid_evidence_count, g.unsupported) == (2, False)
        # Case differs ("Pricing" in the output) and the ref names an input
        # that was never given: both failures recorded, nothing repaired.
        assert r.evidence[0].reason == "quote_not_found+unknown_source_ref"
        assert (r.valid_evidence_count, r.unsupported) == (0, True)
        c = verdict.citation_check
        assert (c.citations_total, c.citations_valid, c.citations_invalid) == (3, 2, 1)
        assert (c.findings_total, c.findings_unsupported, c.unsupported_dimensions) == (2, 1, ["risk_specificity"])

    def test_verdict_records_replay_bundle_and_version_tuple(self, judged):
        verdict, fake, _ = judged
        assert [i.id for i in verdict.judge_inputs] == [
            "in:ticker", "in:update_reason", "in:board_row", "in:filings_excerpts[0]", "in:filings_excerpts[1]",
        ]
        assert all(len(i.sha256) == 64 and i.chars > 0 for i in verdict.judge_inputs)
        assert verdict.rubric_version == "1.0.0" and verdict.rubric_hash == "f" * 64
        assert verdict.judge_parser_version == judge_mod.JUDGE_PARSER_VERSION
        sent = fake.chat.completions.create.call_args.kwargs["messages"]
        from evals.judge_citations import judge_prompt_sha256

        assert verdict.judge_prompt_sha256 == judge_prompt_sha256(sent[0]["content"], sent[1]["content"])
        assert '<input id="in:filings_excerpts[0]">' in sent[1]["content"]

    def test_verdict_joins_to_the_judged_trace_and_the_judge_call(self, judged):
        verdict, _, sink = judged
        assert verdict.judged_trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert verdict.judged_span_id == "00f067aa0ba902b7"
        call = verdict.judge_call
        assert call.callsite_id == "evaljudge-sync"
        assert call.response_id == "gen-abc123"
        assert call.cost_run_id == _Sink.run_id
        assert call.attempts == 1 and call.max_tokens == judge_mod.DEFAULT_MAX_TOKENS
        assert call.finish_reason == "tool_calls"

    def test_request_shape_is_unchanged_but_for_the_cited_tool(self, judged):
        _, fake, _ = judged
        kw = fake.chat.completions.create.call_args.kwargs
        extra_body = dict(kw["extra_body"])
        # krepis's router-edge attribution (alpha-engine-config-I12312): the
        # request carries call-site / run / trace metadata the LiteLLM spend
        # log persists. Consumed by the proxy, never forwarded upstream.
        metadata = extra_body.pop("metadata", None)
        if metadata is not None:
            assert metadata["tags"][0].startswith("callsite:")
            assert "trace_id" in metadata["spend_logs_metadata"]
        assert extra_body == {"reasoning": {"exclude": True}}
        assert kw["tools"][0]["function"]["parameters"] == CitedRubricEvalLLMOutput.model_json_schema()
        assert kw["tool_choice"] == {"type": "function", "function": {"name": "RubricEvalLLMOutput"}}

    def test_persisted_verdict_sidecar_and_manifest_carry_the_counts(self, judged):
        verdict, _, _ = judged
        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="eval-test-bucket")
            key = judge_mod.persist_eval_artifact(verdict, s3_client=s3, bucket="eval-test-bucket")
            body = json.loads(s3.get_object(Bucket="eval-test-bucket", Key=key)["Body"].read())
            sidecar = json.loads(
                s3.get_object(Bucket="eval-test-bucket", Key="decision_artifacts/_eval/latest.json")["Body"].read()
            )
        assert RubricEvalArtifact.model_validate(body) == verdict
        for row in (sidecar, _make_manifest_entry(body, key)):
            assert row["citation_contract"] == "judge_evidence.v1"
            assert (row["citations_total"], row["citations_valid"], row["citations_invalid"]) == (3, 2, 1)
            assert row["findings_unsupported"] == 1
            assert row["judged_trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_uncited_rubric_end_to_end_records_no_citation_check(self, monkeypatch):
        from pathlib import Path

        legacy = LoadedPrompt(
            name="eval_rubric_thinktank_thesis",
            text="═══ DIMENSION 1: input_groundedness ═══\n{agent_input}\n{agent_output}",
            version="0.2.0", hash="a" * 64, source_path=Path("/dev/null"),
        )
        monkeypatch.setattr(judge_mod, "load_prompt", lambda name: legacy)
        fake = MagicMock()
        fake.chat.completions.create.return_value = _response(
            {"dimension_scores": [{"dimension": "input_groundedness", "score": 3, "reasoning": "r"}],
             "overall_reasoning": "o"}
        )
        with _patch_llm_client(judge_mod, fake):
            verdict = judge_mod.evaluate_artifact(_thesis_artifact(), api_key="sk-test")
        assert verdict.citation_check is None and verdict.judge_inputs == []
        assert verdict.dimension_scores[0].unsupported is None
        assert verdict.judge_call.response_id == "gen-abc123"
        assert verdict.rubric_hash == "a" * 64
        from graph.state_schemas import RubricEvalLLMOutput

        kw = fake.chat.completions.create.call_args.kwargs
        assert kw["tools"][0]["function"]["parameters"] == RubricEvalLLMOutput.model_json_schema()


# ── The live rubric (whichever version alpha-engine-config serves) ──────


class TestLiveRubric:
    @pytest.mark.parametrize("rubric_id", ["eval_rubric_thinktank_thesis", "eval_rubric_thinktank_theme"])
    def test_live_rubric_renders_in_its_declared_mode(self, rubric_id):
        from agents.prompt_loader import load_prompt
        from evals.rubric_dimensions import declared_dimensions

        prompt = load_prompt(rubric_id)
        rendered = judge_mod._render_for_judge(_thesis_artifact(), prompt)
        assert declared_dimensions(prompt.text)
        if rendered.contract is not None:
            assert rendered.contract == CITATION_CONTRACT
            assert '<input id="in:board_row">' in rendered.text
            assert '<output field="thesis.summary">' in rendered.text
            # The template must stay renderable by pre-contract code too.
            assert judge_mod._render_rubric(_thesis_artifact(), prompt)
