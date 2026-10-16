"""Think Tank prompts cite their inputs as typed sources on the call's
``llm_trace.v1`` record (alpha-engine-config-I12312, audit-2026-10-14 §6.3).

Before this, ``filings_excerpts`` reduced every retrieved chunk to a
``[doc_type | filed_date]`` header and every S3 input was an anonymous JSON
dump, so no record could say which filing or which object version a thesis
was built from.

Needs krepis with ``krepis.llm_trace`` (krepis PR 247) and nousergon-lib with
``RetrievalResult.as_source()`` (nousergon-lib PR 472).
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest
from krepis.llm_trace import CITATION_INSTRUCTION  # noqa: F401 — pins the krepis floor
from krepis.trace_sink import InMemoryTraceSink
from nousergon_lib.rag.retrieval import RetrievalResult
from pydantic import BaseModel

from thinktank.client import ThinktankClient
from thinktank.settings import ThinktankSettings, TierSpec
from thinktank.sources import Slot, format_with_sources, s3_object_source

# ── format_with_sources ──────────────────────────────────────────────────────


class _Prompt:
    name = "unit_prompt"

    def __init__(self, text: str) -> None:
        self.text = text

    def format(self, **kwargs) -> str:
        return self.text.format(**kwargs)


def _src(cid: str, text: str) -> dict:
    return s3_object_source(cid, bucket="b", key=f"k/{cid}.json", text=text, version="v1")


def test_offsets_point_at_exactly_the_injected_text() -> None:
    prompt = _Prompt("Ticker {ticker}\nBoard: {board}\nFilings:\n{filings}\nEnd")
    user, rendered = format_with_sources(
        prompt,
        ticker="QLYS",
        board=Slot(sources=[_src("board", '{"ticker": "QLYS"}')], empty="{}"),
        filings=Slot(
            sources=[_src("f1", "[8-K | 2026-09-04]\nalpha"), _src("f2", "[10-Q | x]\nbeta")],
            empty="(none)",
            separator="\n---\n",
        ),
    )
    assert rendered.text == user
    assert [s.block.cid for s in rendered.sources] == ["board", "f1", "f2"]
    for s in rendered.sources:
        assert user[s.char_start:s.char_end] == s.block.text
    assert '<source cid="board">' in user
    assert "\n---\n" in user


def test_an_absent_input_renders_its_old_placeholder_and_cites_nothing() -> None:
    user, rendered = format_with_sources(
        _Prompt("{filings}"), filings=Slot(sources=[], empty="(no filings context available)")
    )
    assert user == "(no filings context available)"
    assert rendered.sources == ()


def test_a_slot_the_template_does_not_use_is_not_traced() -> None:
    user, rendered = format_with_sources(
        _Prompt("only {a}"), a="x", unused=Slot(sources=[_src("u", "t")])
    )
    assert user == "only x"
    assert rendered.sources == ()


def test_a_slot_used_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        format_with_sources(_Prompt("{a} {a}"), a=Slot(sources=[_src("a", "t")]))


def test_s3_object_source_shape() -> None:
    src = s3_object_source(
        "board", bucket="alpha-engine-research", key="scanner/universe/latest.json",
        text="{}", version="3HL4kqtJlcpXroDTDmJ", meta={"ticker": "QLYS"},
    )
    assert src["kind"] == "s3_object"
    assert src["source_id"] == "s3://alpha-engine-research/scanner/universe/latest.json"
    assert src["version"] == "3HL4kqtJlcpXroDTDmJ"


# ── filings_sources keeps the ids ────────────────────────────────────────────


def test_filings_sources_keep_chunk_and_document_ids(monkeypatch) -> None:
    import nousergon_lib.rag as rag

    from thinktank.context import FILINGS_EXCERPT_CHARS, filings_sources

    hit = RetrievalResult(
        content="x" * 3000,
        ticker="QLYS",
        doc_type="8-K",
        filed_date=date(2026, 9, 4),
        section_label="Item 2.02",
        similarity=0.81,
        chunk_id="chunk-1",
        retrieval_method="hybrid",
        combined_score=0.81,
        document_id="doc-1",
        url="https://www.sec.gov/x.htm",
    )
    monkeypatch.setattr(rag, "retrieve", lambda **kw: [hit])
    out = filings_sources("QLYS")
    assert len(out) == 1
    src = out[0]
    assert src["cid"] == "f1"
    assert src["source_id"] == "rag.chunks/chunk-1"
    assert src["document_id"] == "rag.documents/doc-1"
    assert src["uri"] == "https://www.sec.gov/x.htm"
    assert src["retrieval"]["rank"] == 1
    # The injected text is exactly what the prompt used to carry.
    assert src["text"] == "[8-K | 2026-09-04]\n" + "x" * FILINGS_EXCERPT_CHARS


# ── the call writes a trace that carries them ────────────────────────────────


class _Out(BaseModel):
    model_config = {"extra": "forbid"}
    answer: str


_FAKE_ROUTE = {
    "schema_version": 2,
    "group": "med",
    "route": "litellm_proxy",
    "provider": "litellm",
    "deployment_id": "med",
    "api_base_url": "https://router.example:8443",
    "auth_token_type": "litellm_master_key",
    "registry_id": "litellm:group:med",
    "primary_registry_id": "deepseek-v4-flash-max",
    "params": {},
}


def _client(monkeypatch, sink, completions):
    from datetime import date as _date

    import krepis.router as _kr
    from krepis.cost import PriceCard, PriceTable

    monkeypatch.setenv("LITELLM_MASTER_KEY", "consumer-test")
    monkeypatch.setattr(_kr, "resolve_group_structured", lambda *a, **k: dict(_FAKE_ROUTE))
    table = PriceTable(cards=[PriceCard(
        model_name="deepseek-v4-flash-max", effective_from=_date(2026, 1, 1),
        input_per_1m=1.0, output_per_1m=2.0,
        cache_read_per_1m=0.0, cache_create_per_1m=0.0,
    )])
    monkeypatch.setattr("krepis.cost.load_default_pricing", lambda: table)
    settings = ThinktankSettings(
        bucket="alpha-engine-research",
        daily_new_names=5,
        rank_ceiling=150,
        sweep_chunk_size=25,
        stale_after_days=30,
        monthly_budget_usd_default=25.0,
        budget_ssm_param="/thinktank/monthly_budget_usd",
        tiers={"thesis": TierSpec(
            name="thesis", group="med", max_tokens=100, structured_outputs=True)},
    )
    holder = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return ThinktankClient(settings=settings, run_id="testrun",
                           client_factory=lambda spec, key: holder, trace_sink=sink)


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp-1",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps({"answer": "per [f1]"})),
                finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=500),
            model="deepseek-v4-flash-max",
        )


def test_the_trace_row_cites_every_source_and_the_result_carries_its_ids(monkeypatch):
    sink = InMemoryTraceSink()
    completions = _Completions()
    client = _client(monkeypatch, sink, completions)
    user, rendered = format_with_sources(
        _Prompt("Board {board}\nFilings {filings}"),
        board=Slot(sources=[_src("board", '{"ticker": "QLYS"}')], empty="{}"),
        filings=Slot(sources=[{
            "cid": "f1", "kind": "rag_chunk", "source_id": "rag.chunks/c1",
            "document_id": "rag.documents/d1", "text": "[8-K | 2026-09-04]\nalpha",
        }]),
    )
    result = client.complete(
        "thesis", agent_id="analyst_thesis", system="s", user=user, response_model=_Out,
        prompt_id="thinktank_thesis", prompt_version="0.3.0",
        sources=rendered, subject={"kind": "ticker", "id": "QLYS"},
    )

    # The prompt is sent byte for byte as rendered — nothing prepended.
    assert completions.calls[0]["messages"][-1]["content"] == user
    assert len(sink.rows) == 1
    row = sink.rows[0]
    ext = row["krepis"]
    assert result.trace_id == row["trace_id"]
    assert result.span_id == row["span_id"]
    assert ext["subject"] == {"kind": "ticker", "id": "QLYS"}
    assert ext["prompt"]["template_id"] == "thinktank_thesis"
    assert ext["prompt"]["template_version"] == "0.3.0"
    by_cid = {c["cid"]: c for c in ext["citations"]}
    assert by_cid["f1"]["source_id"] == "rag.chunks/c1"
    assert by_cid["f1"]["document_id"] == "rag.documents/d1"
    assert by_cid["f1"]["cited_in_output"] is True
    assert by_cid["board"]["kind"] == "s3_object"
    assert by_cid["board"]["source_id"] == "s3://b/k/board.json"
    assert by_cid["board"]["cited_in_output"] is False
    for c in ext["citations"]:
        span = c["span"]
        source = next(s for s in rendered.sources if s.block.cid == c["cid"])
        assert user[span["char_start"]:span["char_end"]] == source.block.text
