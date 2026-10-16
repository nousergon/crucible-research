"""Think Tank thesis and pillar outputs cite the sources they rely on
(alpha-engine-config-I12312).

#849 rendered every thesis/pillar input as a ``<source cid="…">`` block on
the call's ``llm_trace.v1`` record, but no prompt asked the model to cite, so
every source read ``cited_in_output: false``. The cited prompt versions carry
``krepis.llm_trace.CITATION_INSTRUCTION`` and ask for a structured ``claims``
list; ``thinktank.citations.check_citations`` validates that list against the
cids actually rendered into the prompt.

Needs krepis with ``krepis.llm_trace`` (krepis PR 247) and nousergon-lib with
``RetrievalResult.as_source()`` (nousergon-lib PR 472), like
tests/test_thinktank_trace_sources.py.
"""

from __future__ import annotations

import json
import os

import pytest
from krepis import llm_trace
from krepis.llm_trace import CITATION_INSTRUCTION, render_sources
from krepis.trace_sink import InMemoryTraceSink
from pydantic import ValidationError

from agents import prompt_loader
from thinktank.analyst import (
    PILLAR_PROMPT,
    THESIS_PROMPT,
    _prior_thesis_text,
    _thesis_input_slots,
)
from thinktank.citations import (
    CITE_MARKER_RE,
    check_citations,
    marker_tokens,
    require_citation_instruction,
    strip_citation_markers,
)
from thinktank.context import ContextBundle
from thinktank.schemas import (
    CitedClaim,
    CitedPillarAssessmentLLM,
    CompanyThesis,
    CompanyThesisCitedLLM,
    CompanyThesisLLM,
)
from thinktank.sources import Slot, format_with_sources, s3_object_source

_ON_CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}


# ── fixtures ─────────────────────────────────────────────────────────────────


def _thesis_body(**over) -> dict:
    body = {
        "business_summary": "Sells endpoint security [f1].",
        "moat": "Switching costs [f1, news].",
        "filings_review": "Revenue up 12% [f1].",
        "news_sentiment": "Quiet [news].",
        "valuation": "Fair.",
        "market_dynamics": "Neutral.",
        "risks": ["Pricing pressure [f2]"],
        "catalysts": ["Q3 print"],
        "stance": "attractive",
        "conviction": 70,
        "summary": "Compounder at a fair price [f1].",
        "rating": 72,
        "rating_rationale": "Growth [f1] at a fair multiple.",
        "claims": [
            {"section": "filings_review", "claim": "Revenue up 12% [f1].", "citations": ["f1"]},
            {"section": "risks", "claim": "Pricing pressure [f2].", "citations": ["f2", "f9"]},
            {"section": "catalysts", "claim": "Q3 print.", "citations": []},
        ],
    }
    body.update(over)
    return body


def _pillar_body() -> dict:
    def _sub(pillar: str) -> dict:
        return {"pillar": pillar, "score": 60, "confidence": "medium",
                "evidence": [f"{pillar} evidence [f1]"]}

    return {
        "quality": _sub("quality"),
        "quality_moat": {"primary_type": "none", "secondary_types": [], "width": "none",
                         "durability_years": 0, "trend": "stable", "evidence": []},
        "value": _sub("value"),
        "momentum": _sub("momentum"),
        "growth": _sub("growth"),
        "stewardship": _sub("stewardship"),
        "defensiveness": _sub("defensiveness"),
        "catalyst_horizon_modulation": 0,
        "claims": [{"section": "quality", "claim": "Margins expand [f1].", "citations": ["f1"]}],
    }


def _rendered(*cids: str):
    return render_sources(
        [s3_object_source(c, bucket="b", key=f"k/{c}", text=f"text of {c}") for c in cids]
    )


# ── the marker grammar is krepis's ───────────────────────────────────────────


def test_marker_grammar_is_the_trace_parsers_grammar() -> None:
    # The check here and the trace row's cited_in_output must read the same
    # markers, or the two would disagree about what was cited.
    assert CITE_MARKER_RE.pattern == llm_trace._CITE_MARKER_RE.pattern
    text = "a [f1] b [f2, news] c [1] d [ board ]"
    assert marker_tokens(text) == ["f1", "f2", "news", "1", "board"]
    assert llm_trace.cited_cids(text, ["f1", "f2", "news", "board"]) == [
        "f1", "f2", "news", "board",
    ]


# ── the cited prompts ────────────────────────────────────────────────────────


def _load(name: str):
    prompt_loader.clear_cache()
    try:
        return prompt_loader.load_prompt(name)
    except FileNotFoundError:
        if _ON_CI:
            raise
        pytest.skip(f"alpha-engine-config prompts not staged ({name})")


@pytest.mark.parametrize("name,floor", [(THESIS_PROMPT, (0, 4, 0)), (PILLAR_PROMPT, (0, 2, 0))])
def test_cited_prompts_carry_the_krepis_instruction_verbatim(name, floor) -> None:
    prompt = _load(name)
    assert CITATION_INSTRUCTION in prompt.text
    assert "claims" in prompt.text
    assert tuple(int(p) for p in prompt.version.split(".")) >= floor
    require_citation_instruction(prompt)  # does not raise


def test_a_prompt_without_the_instruction_is_refused_before_spend() -> None:
    class _P:
        name, version, text = "thinktank_thesis", "0.3.0", "Write a thesis for {ticker}."

    with pytest.raises(ValueError, match="CITATION_INSTRUCTION"):
        require_citation_instruction(_P())


def _slots(filings: list[dict]) -> dict[str, Slot]:
    return _thesis_input_slots(
        ContextBundle(), bucket="alpha-engine-research", ticker="QLYS",
        board_row={"ticker": "QLYS", "sector": "Tech", "metrics": {"pe": 30}},
        signals_entry=None, news={"aggregate_date": "2026-10-14", "n": 3},
        insider=None, analyst=None, inst_own=None, filings=filings,
    )


_FILINGS = [
    {"cid": "f1", "kind": "rag_chunk", "source_id": "rag.chunks/c1",
     "text": "[10-Q | 2026-08-01]\nrevenue grew"},
    {"cid": "f2", "kind": "rag_chunk", "source_id": "rag.chunks/c2",
     "text": "[8-K | 2026-09-04]\npricing"},
]


@pytest.mark.parametrize("name", [THESIS_PROMPT, PILLAR_PROMPT])
def test_cited_prompt_renders_every_input_as_a_cited_source(name) -> None:
    prompt = _load(name)
    fields = dict(ticker="QLYS", sector="Tech", macro_theme="m", sector_theme="s",
                  **_slots(_FILINGS))
    if name == THESIS_PROMPT:
        fields.update(update_reason="initial", prior_thesis="(none — initial coverage)",
                      event_context="(none)")
    user, rendered = format_with_sources(prompt, **fields)
    assert [s.block.cid for s in rendered.sources] == ["board", "news", "f1", "f2"]
    for cid in ("board", "news", "f1", "f2"):
        assert f'<source cid="{cid}">' in user
    # The instruction sits AFTER the sources it refers to, in the same turn.
    assert user.index(CITATION_INSTRUCTION) > user.index('<source cid="f2">')
    for s in rendered.sources:
        assert user[s.char_start:s.char_end] == s.block.text


# ── the output schema ────────────────────────────────────────────────────────


def test_thesis_response_requires_the_claims_key_but_allows_it_empty() -> None:
    body = _thesis_body()
    del body["claims"]
    with pytest.raises(ValidationError, match="claims"):
        CompanyThesisCitedLLM.model_validate(body)
    assert CompanyThesisCitedLLM.model_validate(_thesis_body(claims=[])).claims == []


def test_a_claim_with_an_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CitedClaim.model_validate({"section": "x", "claim": "y", "citations": [], "cid": "f1"})


def test_stored_theses_from_before_citations_still_parse() -> None:
    old = _thesis_body()
    del old["claims"]
    assert CompanyThesisLLM.model_validate(old).claims == []
    artifact = CompanyThesis.model_validate({
        "schema_version": 2, "ticker": "QLYS", "version": 3, "trading_day": "2026-10-01",
        "calendar_date": "2026-10-01", "update_reason": "event", "thesis": old,
    })
    assert artifact.thesis_citations is None and artifact.pillar_citations is None


def test_the_cited_thesis_round_trips_through_the_stored_artifact() -> None:
    parsed = CompanyThesisCitedLLM.model_validate(_thesis_body())
    artifact = CompanyThesis(
        ticker="QLYS", version=1, trading_day="d", calendar_date="d",
        update_reason="initial", thesis=parsed,
    )
    again = CompanyThesis.model_validate(json.loads(artifact.model_dump_json()))
    assert [c.citations for c in again.thesis.claims] == [["f1"], ["f2", "f9"], []]


def test_pillar_claims_are_split_off_the_library_shaped_assessment() -> None:
    from nousergon_lib.pillars import QualitativePillarAssessment

    cited = CitedPillarAssessmentLLM.model_validate(_pillar_body())
    plain = cited.assessment()
    assert type(plain) is QualitativePillarAssessment
    assert "claims" not in plain.model_dump()
    assert plain.quality.evidence == ["quality evidence [f1]"]
    body = _pillar_body()
    del body["claims"]
    with pytest.raises(ValidationError, match="claims"):
        CitedPillarAssessmentLLM.model_validate(body)


# ── the validator ────────────────────────────────────────────────────────────


def test_check_records_invalid_cids_and_flags_uncited_claims() -> None:
    parsed = CompanyThesisCitedLLM.model_validate(_thesis_body())
    check = check_citations(parsed.model_dump(), parsed.claims,
                            _rendered("board", "news", "f1", "f2"),
                            prompt_id=THESIS_PROMPT, prompt_version="0.4.0")
    assert check.prompt_id == THESIS_PROMPT and check.prompt_version == "0.4.0"
    assert check.rendered_cids == ["board", "news", "f1", "f2"]
    assert check.cited_cids == ["news", "f1", "f2"]
    assert check.uncited_sources == ["board"]
    # f9 was never rendered: recorded as invalid, kept on its claim.
    assert check.invalid_cids == ["f9"]
    risks = check.claims[1]
    assert risks.citations == ["f2"] and risks.invalid_citations == ["f9"]
    assert risks.uncited is False
    # The catalyst claim cites nothing: flagged, not fatal.
    assert check.claims[2].uncited is True
    assert check.claim_count == 3 and check.uncited_claim_count == 1


def test_a_claim_citing_only_unknown_cids_is_uncited_and_nothing_is_dropped() -> None:
    claims = [CitedClaim(section="moat", claim="Network effects.", citations=["c1", "f7"])]
    check = check_citations({}, claims, _rendered("f1"))
    assert check.claims[0].uncited is True
    assert check.claims[0].invalid_citations == ["c1", "f7"]
    assert check.invalid_cids == ["c1", "f7"]
    assert check.cited_cids == []


def test_an_inline_marker_in_the_claim_counts_when_the_list_omits_it() -> None:
    claims = [CitedClaim(section="valuation", claim="Cheap vs peers [f1, board].", citations=[])]
    check = check_citations({}, claims, _rendered("board", "f1"))
    assert check.claims[0].citations == ["f1", "board"]
    assert check.claims[0].uncited is False
    assert check.cited_cids == ["board", "f1"]  # prompt order


def test_unresolved_inline_markers_are_recorded_not_counted_invalid() -> None:
    check = check_citations({"summary": "Step [1] then [f1] and [f3]."}, [], _rendered("f1"))
    assert check.cited_cids == ["f1"]
    assert check.unresolved_markers == ["1", "f3"]
    assert check.invalid_cids == []
    assert check.claim_count == 0


def test_a_prompt_with_no_sources_makes_every_citation_invalid() -> None:
    claims = [CitedClaim(section="moat", claim="x [f1]", citations=["f1"])]
    check = check_citations({}, claims, None)
    assert check.rendered_cids == [] and check.cited_cids == []
    assert check.invalid_cids == ["f1"]
    assert check.claims[0].uncited is True


# ── the prior thesis never carries yesterday's cids into today's prompt ─────


def test_prior_thesis_text_is_unchanged_for_a_pre_citation_thesis() -> None:
    old = _thesis_body()
    del old["claims"]
    for key, value in list(old.items()):
        if isinstance(value, str):
            old[key] = value.split(" [")[0]
        elif isinstance(value, list):
            old[key] = [v.split(" [")[0] for v in value]
    prior = CompanyThesis(ticker="QLYS", version=1, trading_day="d", calendar_date="d",
                          update_reason="initial", thesis=CompanyThesisLLM.model_validate(old))
    # What the prompt carried before citations: model_dump_json() of a thesis
    # that had no claims field.
    expected = prior.thesis.model_dump_json(exclude={"claims"})
    assert _prior_thesis_text(prior) == expected


def test_prior_thesis_text_strips_markers_and_claims() -> None:
    prior = CompanyThesis(ticker="QLYS", version=1, trading_day="d", calendar_date="d",
                          update_reason="initial",
                          thesis=CompanyThesisCitedLLM.model_validate(_thesis_body()))
    text = _prior_thesis_text(prior)
    assert "[f1" not in text and "[news]" not in text and "[f2]" not in text
    assert '"claims"' not in text
    assert "Revenue up 12%." in text
    assert _prior_thesis_text(None) == "(none — initial coverage)"


def test_strip_citation_markers_walks_nested_values() -> None:
    assert strip_citation_markers({"a": ["x [f1]", {"b": "y [f1, f2] z"}], "n": 3}) == {
        "a": ["x", {"b": "y z"}], "n": 3,
    }


# ── end to end: a stubbed call writes cited_in_output: true ─────────────────


class _Completions:
    def __init__(self, body: dict) -> None:
        self.body = body
        self.calls: list[dict] = []

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp-1",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(self.body)), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=500),
            model="deepseek-v4-flash-max",
        )


def test_a_cited_thesis_call_marks_its_sources_cited_in_the_trace_row(monkeypatch) -> None:
    from tests.test_thinktank_trace_sources import _client

    prompt = _load(THESIS_PROMPT)
    require_citation_instruction(prompt)
    user, rendered = format_with_sources(
        prompt, ticker="QLYS", sector="Tech", update_reason="initial",
        prior_thesis="(none — initial coverage)", event_context="(none)",
        macro_theme="m", sector_theme="s", **_slots(_FILINGS),
    )
    sink = InMemoryTraceSink()
    completions = _Completions(_thesis_body())
    client = _client(monkeypatch, sink, completions)
    result = client.complete(
        "thesis", agent_id="analyst_thesis", system="s", user=user,
        response_model=CompanyThesisCitedLLM, prompt_id=prompt.name,
        prompt_version=prompt.version, sources=rendered,
        subject={"kind": "ticker", "id": "QLYS"},
    )

    # The prompt the model saw carries the instruction, byte for byte.
    assert CITATION_INSTRUCTION in completions.calls[0]["messages"][-1]["content"]
    row = sink.rows[0]
    ext = row["krepis"]
    assert ext["prompt"]["template_id"] == THESIS_PROMPT
    assert ext["prompt"]["template_version"] == prompt.version
    cited = {c["cid"]: c["cited_in_output"] for c in ext["citations"]}
    assert cited == {"board": False, "news": True, "f1": True, "f2": True}

    check = check_citations(result.parsed.model_dump(), result.parsed.claims, rendered,
                            prompt_id=prompt.name, prompt_version=prompt.version)
    # The deterministic check agrees with the trace row about every source.
    assert set(check.cited_cids) == {cid for cid, v in cited.items() if v}
    assert check.invalid_cids == ["f9"]
    assert check.uncited_claim_count == 1


def test_structured_citations_alone_do_not_set_cited_in_output(monkeypatch) -> None:
    """Why the prompt asks for inline markers too: the trace parser reads
    ``[cid]`` markers, and a JSON list ``["f1"]`` is not one."""
    from tests.test_thinktank_trace_sources import _client

    user, rendered = format_with_sources(
        type("P", (), {"name": "p", "format": lambda self, **k: "S {f}".format(**k)})(),
        f=Slot(sources=[_FILINGS[0]]),
    )
    body = _thesis_body(
        business_summary="b", moat="m", filings_review="f", news_sentiment="n",
        risks=["r"], summary="s", rating_rationale="r",
        claims=[{"section": "moat", "claim": "no marker", "citations": ["f1"]}],
    )
    sink = InMemoryTraceSink()
    client = _client(monkeypatch, sink, _Completions(body))
    result = client.complete(
        "thesis", agent_id="analyst_thesis", system="s", user=user,
        response_model=CompanyThesisCitedLLM, sources=rendered,
    )
    assert sink.rows[0]["krepis"]["citations"][0]["cited_in_output"] is False
    check = check_citations(result.parsed.model_dump(), result.parsed.claims, rendered)
    assert check.cited_cids == ["f1"]
