"""The EvalJudge call's ``llm_trace.v1`` row carries typed source citations
(alpha-engine-config-I12312, one more door after the Think Tank and the
Director).

Every input element and output field the judge is shown under
``judge_evidence.v1`` becomes a ``krepis.llm_trace`` source whose text is the
element's own characters and whose offsets index the user content actually
sent, so an eval can join a verdict's ``evidence.source_ref`` to the trace's
``citations[]`` without re-rendering anything.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import pytest

from evals import judge as judge_mod
from evals.judge_citations import (
    CITATION_CONTRACT,
    TRACE_SOURCE_KIND,
    build_bundle,
    judge_trace_sources,
    judge_trace_sources_or_none,
    render_cited_rubric,
)
from tests.test_eval_judge import (  # noqa: F401 — live_router_resolution is an autouse fixture
    _patch_llm_client,
    live_router_resolution,
)
from tests.test_eval_judge_citations import (
    CITED_TEMPLATE,
    _cited_prompt,
    _judge_args,
    _response,
    _thesis_artifact,
)

KEY = "decision_artifacts/2026/10/12/thinktank_thesis/tt-run-001.json"
_KREPIS_HAS_LLM_TRACE = importlib.util.find_spec("krepis.llm_trace") is not None

pytestmark = pytest.mark.skipif(not _KREPIS_HAS_LLM_TRACE, reason="krepis.llm_trace needs krepis>=0.59.74")


def _user_content():
    bundle = build_bundle(_thesis_artifact())
    rendered = render_cited_rubric(CITED_TEMPLATE, bundle)
    _system, user = judge_mod._split_rubric_for_caching(rendered)
    return user, bundle


class TestJudgeTraceSources:
    def test_one_source_per_element_with_exact_offsets(self):
        user, bundle = _user_content()
        rs = judge_trace_sources(user, bundle, judged_artifact_s3_key=KEY)
        assert rs is not None and rs.text == user
        refs = [s.block.meta["judge_ref"] for s in rs.sources]
        assert refs == [e.id for e in bundle.inputs] + [f"out:{p}" for p in bundle.outputs]
        from krepis.llm_trace import sha256_hex

        for s in rs.sources:
            # The recorded offsets slice exactly the text that was hashed.
            assert user[s.char_start:s.char_end] == s.block.text
            assert s.content_sha256 == sha256_hex(s.block.text)
            assert s.block.kind == TRACE_SOURCE_KIND
            assert s.block.document_id == KEY
            assert s.block.source_id == f"{KEY}#{s.block.meta['judge_ref']}"
            assert s.block.meta["contract"] == CITATION_CONTRACT
        assert [s.block.cid for s in rs.sources] == [f"j{i}" for i in range(1, len(rs.sources) + 1)]

    def test_roles_name_inputs_and_outputs(self):
        user, bundle = _user_content()
        rs = judge_trace_sources(user, bundle)
        roles = {s.block.meta["judge_ref"]: s.block.meta["role"] for s in rs.sources}
        assert roles["in:filings_excerpts[0]"] == "input"
        assert roles["out:thesis.risks[0]"] == "output"
        # Without the judged key the ref alone is the identity.
        assert {s.block.source_id for s in rs.sources} == set(roles)

    def test_an_element_not_in_the_user_content_is_left_out(self):
        user, bundle = _user_content()
        # Drop the outputs section: those elements were not sent, so they get
        # no offset rather than an offset into the wrong text.
        head = user[: user.index("Agent output:")]
        rs = judge_trace_sources(head, bundle)
        assert all(s.block.meta["role"] == "input" for s in rs.sources)

    def test_a_repeated_block_is_left_out(self):
        user, bundle = _user_content()
        block = f'<input id="in:board_row">\n{bundle.inputs[2].text}\n</input>'
        assert bundle.inputs[2].id == "in:board_row"
        rs = judge_trace_sources(user + "\n" + block, bundle)
        assert "in:board_row" not in {s.block.meta["judge_ref"] for s in rs.sources}

    def test_nothing_to_cite_is_none(self):
        _, bundle = _user_content()
        assert judge_trace_sources("", bundle) is None
        assert judge_trace_sources("no tags here", bundle) is None
        assert judge_trace_sources("anything", None) is None

    def test_or_none_never_raises(self, monkeypatch):
        import evals.judge_citations as jc

        def _boom(*a, **k):
            raise RuntimeError("renderer fault")

        monkeypatch.setattr(jc, "judge_trace_sources", _boom)
        assert jc.judge_trace_sources_or_none("x", None) is None

    def test_krepis_accepts_the_rendered_sources(self):
        from krepis.llm_trace import bind_sources

        user, bundle = _user_content()
        rs = judge_trace_sources_or_none(user, bundle, judged_artifact_s3_key=KEY)
        content, bound = bind_sources(user, rs)
        assert content == user
        assert [(b.char_start, b.char_end) for b in bound] == [(s.char_start, s.char_end) for s in rs.sources]


class _TraceSink:
    run_id = "judge-run-trace-02"

    def __init__(self):
        self.rows = []

    def __call__(self, row, blobs=None):
        self.rows.append(row)


class _CostSink:
    run_id = "judge-run-cost-02"

    def __call__(self, record):
        pass


def _judge_once(monkeypatch, entry, sink):
    monkeypatch.setattr(judge_mod, "load_prompt", lambda name: _cited_prompt())
    monkeypatch.setattr(judge_mod, "_judge_cost_sink", lambda: _CostSink())
    art = _thesis_artifact()
    fake = MagicMock()
    fake.chat.completions.create.return_value = _response(_judge_args(art))
    with _patch_llm_client(judge_mod, fake) as llm_cls:
        build = llm_cls.side_effect

        def _with_trace_sink(spec, **kwargs):
            client = build(spec, **kwargs)
            client._trace_sink = sink
            return client

        llm_cls.side_effect = _with_trace_sink
        verdict = getattr(judge_mod, entry)(
            art, judge_model="claude-haiku-4-5", api_key="sk-test", judged_artifact_s3_key=KEY,
        )
    return verdict, fake


@pytest.mark.parametrize("entry,callsite", [
    ("evaluate_artifact", "evaljudge-sync"),
    ("evaluate_artifact_openrouter", "evaljudge-shadow"),
])
def test_judge_trace_row_carries_typed_citations(monkeypatch, entry, callsite):
    from krepis.llm_trace import row_extension

    sink = _TraceSink()
    verdict, fake = _judge_once(monkeypatch, entry, sink)
    assert len(sink.rows) == 1
    row = sink.rows[0]
    ext = row_extension(row)
    assert ext["callsite_id"] == callsite
    citations = ext.get("citations") or row.get("citations") or []
    by_ref = {c["meta"]["judge_ref"]: c for c in citations if c.get("meta")}
    # Every input the verdict says the judge saw is a typed citation on the row,
    # under the id the verdict's evidence names.
    assert {i.id for i in verdict.judge_inputs} <= set(by_ref)
    cite = by_ref["in:filings_excerpts[0]"]
    assert cite["kind"] == "artifact"
    assert cite["source_id"] == f"{KEY}#in:filings_excerpts[0]"
    assert cite["document_id"] == KEY
    # The offsets index the user message the model was actually sent.
    sent = fake.chat.completions.create.call_args.kwargs["messages"]
    user_text = next(m["content"] for m in sent if m["role"] == "user")
    if isinstance(user_text, list):  # content blocks
        user_text = "".join(part.get("text", "") for part in user_text)
    span = cite["span"]
    assert user_text[span["char_start"]:span["char_end"]] == next(
        i for i in _thesis_artifact().input_data_snapshot["filings_excerpts"][:1]
    )


def test_uncited_rubric_sends_no_sources(monkeypatch):
    """A rubric without ``judge_evidence.v1`` has no bundle, so the call is
    exactly as before: no sources, no citations."""
    from pathlib import Path

    from krepis.llm_trace import row_extension

    from agents.prompt_loader import LoadedPrompt

    legacy = LoadedPrompt(
        name="eval_rubric_thinktank_thesis",
        text="# rubric\n\nAgent input:\n{agent_input}\n\nAgent output:\n{agent_output}\n",
        version="1.0.0", hash="e" * 64, source_path=Path("/dev/null"),
    )
    sink = _TraceSink()
    monkeypatch.setattr(judge_mod, "load_prompt", lambda name: legacy)
    monkeypatch.setattr(judge_mod, "_judge_cost_sink", lambda: _CostSink())
    art = _thesis_artifact()
    args = _judge_args(art)
    for d in args["dimension_scores"]:
        d.pop("evidence", None)
    fake = MagicMock()
    fake.chat.completions.create.return_value = _response(args)
    with _patch_llm_client(judge_mod, fake) as llm_cls:
        build = llm_cls.side_effect

        def _with_trace_sink(spec, **kwargs):
            client = build(spec, **kwargs)
            client._trace_sink = sink
            return client

        llm_cls.side_effect = _with_trace_sink
        judge_mod.evaluate_artifact(art, judge_model="claude-haiku-4-5", api_key="sk-test",
                                    judged_artifact_s3_key=KEY)
    ext = row_extension(sink.rows[0])
    assert not (ext.get("citations") or sink.rows[0].get("citations"))
