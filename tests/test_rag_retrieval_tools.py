"""Tests for the Wave 1 PR E RAG retrieval tools.

Covers:
  - search_news_impl: happy path, empty results, transient failure
  - search_filings_impl: default forms list, narrowing via forms arg,
    transient failure
  - search_transcripts_impl: quarter-based lookback
  - Stats tracking
  - _format_results compose
  - build_rag_retrieval_tools returns 3 @tool callables
  - thesis_update _augment_news_summary_with_rag composition
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agents.sector_teams.rag_retrieval_tools import (
    DEFAULT_FILINGS_DOC_TYPES,
    build_rag_retrieval_tools,
    get_rag_retrieval_stats,
    reset_rag_retrieval_stats,
    search_filings_impl,
    search_news_impl,
    search_transcripts_impl,
)


def _fake_result(
    *,
    doc_type="news",
    filed_date=date(2026, 5, 10),
    section_label="headline",
    content="story body",
    source="news_polygon",
    chunk_id="c1",
    vector_score=0.9,
    keyword_score=0.5,
    combined_score=0.78,
):
    return SimpleNamespace(
        doc_type=doc_type,
        filed_date=filed_date,
        section_label=section_label,
        content=content,
        source=source,
        chunk_id=chunk_id,
        vector_score=vector_score,
        keyword_score=keyword_score,
        combined_score=combined_score,
    )


@pytest.fixture(autouse=True)
def _reset_stats():
    reset_rag_retrieval_stats()
    yield


# ── search_news ────────────────────────────────────────────────────────


class TestSearchNews:
    def test_happy_path_returns_formatted_text(self):
        fake_retrieve = MagicMock(return_value=[
            _fake_result(content="article 1 body"),
            _fake_result(content="article 2 body", chunk_id="c2"),
        ])
        out = search_news_impl(
            "AAPL", "fda approval",
            retrieve_fn=fake_retrieve,
        )
        assert "article 1 body" in out
        assert "article 2 body" in out
        # Source included in news result header
        assert "news_polygon" in out

        # Verify retrieve was called with the right shape
        kwargs = fake_retrieve.call_args.kwargs
        assert kwargs["tickers"] == ["AAPL"]
        assert kwargs["doc_types"] == ["news"]
        assert kwargs["method"] == "hybrid"
        assert kwargs["top_k"] == 8

    def test_empty_results_returns_friendly_message(self):
        fake_retrieve = MagicMock(return_value=[])
        out = search_news_impl(
            "AAPL", "x",
            retrieve_fn=fake_retrieve,
        )
        assert "No recent news found for AAPL" in out

    def test_transient_failure_returns_unavailable_message(self):
        fake_retrieve = MagicMock(side_effect=RuntimeError("pgvector down"))
        out = search_news_impl(
            "AAPL", "x",
            retrieve_fn=fake_retrieve,
        )
        assert "temporarily unavailable" in out

    def test_days_back_controls_min_date(self):
        fake_retrieve = MagicMock(return_value=[])
        search_news_impl(
            "AAPL", "x", days_back=7, retrieve_fn=fake_retrieve,
        )
        min_date = fake_retrieve.call_args.kwargs["min_date"]
        # min_date should be ~7 days ago
        assert (date.today() - min_date).days == 7

    def test_stats_incremented_on_attempt_and_success(self):
        fake_retrieve = MagicMock(return_value=[_fake_result()])
        search_news_impl("AAPL", "x", retrieve_fn=fake_retrieve)
        stats = get_rag_retrieval_stats()["search_news"]
        assert stats["attempted"] == 1
        assert stats["succeeded"] == 1
        assert stats["failed"] == 0

    def test_stats_track_failures(self):
        fake_retrieve = MagicMock(side_effect=RuntimeError("x"))
        search_news_impl("AAPL", "x", retrieve_fn=fake_retrieve)
        stats = get_rag_retrieval_stats()["search_news"]
        assert stats["attempted"] == 1
        assert stats["succeeded"] == 0
        assert stats["failed"] == 1


# ── search_filings ─────────────────────────────────────────────────────


class TestSearchFilings:
    def test_default_doc_types_cover_expanded_set(self):
        fake_retrieve = MagicMock(return_value=[])
        search_filings_impl("AAPL", "x", retrieve_fn=fake_retrieve)
        doc_types = fake_retrieve.call_args.kwargs["doc_types"]
        # Pin the canonical set (13F removed 2026-07-13 — structured data,
        # not text; no RAG producer; the get_institutional_activity tool
        # reads from the inst_ownership derived table instead)
        for form in ("10-K", "10-Q", "8-K", "14A", "S-1"):
            assert form in doc_types

    def test_forms_arg_narrows_query(self):
        fake_retrieve = MagicMock(return_value=[])
        search_filings_impl(
            "AAPL", "x", forms="8-K,10-Q", retrieve_fn=fake_retrieve,
        )
        doc_types = fake_retrieve.call_args.kwargs["doc_types"]
        assert doc_types == ["8-K", "10-Q"]

    def test_empty_forms_falls_back_to_defaults(self):
        fake_retrieve = MagicMock(return_value=[])
        search_filings_impl(
            "AAPL", "x", forms="", retrieve_fn=fake_retrieve,
        )
        doc_types = fake_retrieve.call_args.kwargs["doc_types"]
        assert doc_types == list(DEFAULT_FILINGS_DOC_TYPES)

    def test_happy_path_formatted(self):
        fake_retrieve = MagicMock(return_value=[
            _fake_result(doc_type="10-K", filed_date=date(2025, 11, 1),
                         content="risk factor body", source="sec_edgar"),
        ])
        out = search_filings_impl("AAPL", "risks", retrieve_fn=fake_retrieve)
        assert "10-K" in out
        assert "risk factor body" in out
        # Filings results don't include source column
        assert "sec_edgar" not in out

    def test_transient_failure(self):
        fake_retrieve = MagicMock(side_effect=RuntimeError("x"))
        out = search_filings_impl(
            "AAPL", "x", retrieve_fn=fake_retrieve,
        )
        assert "temporarily unavailable" in out


# ── search_transcripts ─────────────────────────────────────────────────


class TestSearchTranscripts:
    def test_doc_types_pinned_to_earnings_transcript(self):
        fake_retrieve = MagicMock(return_value=[])
        search_transcripts_impl(
            "AAPL", "guidance", retrieve_fn=fake_retrieve,
        )
        assert (
            fake_retrieve.call_args.kwargs["doc_types"]
            == ["earnings_transcript"]
        )

    def test_quarters_back_translates_to_days(self):
        fake_retrieve = MagicMock(return_value=[])
        search_transcripts_impl(
            "AAPL", "x", quarters_back=2, retrieve_fn=fake_retrieve,
        )
        min_date = fake_retrieve.call_args.kwargs["min_date"]
        # 2 quarters ≈ 190 days
        assert 180 <= (date.today() - min_date).days <= 200

    def test_empty_results_with_quarters(self):
        fake_retrieve = MagicMock(return_value=[])
        out = search_transcripts_impl(
            "AAPL", "x", quarters_back=4, retrieve_fn=fake_retrieve,
        )
        assert "last 4 quarters" in out


# ── build_rag_retrieval_tools ──────────────────────────────────────────


class TestBuildRagTools:
    def test_returns_three_callables(self):
        tools = build_rag_retrieval_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"search_news", "search_filings", "search_transcripts"}

    def test_tools_have_docstrings_for_react_agent(self):
        tools = build_rag_retrieval_tools()
        for t in tools:
            assert t.description, f"tool {t.name} missing description"


# ── thesis_update RAG augment ──────────────────────────────────────────


class TestThesisUpdateRagAugment:
    def test_augment_combines_base_and_rag(self):
        from agents.sector_teams.sector_team import _augment_news_summary_with_rag

        with patch(
            "agents.sector_teams.rag_retrieval_tools.search_news_impl",
            return_value="[news | 2026-05-10 | news_polygon | headline]\n"
                         "fda approval narrative",
        ), patch(
            "agents.sector_teams.rag_retrieval_tools.search_filings_impl",
            return_value="[8-K | 2026-05-09 | Item 8.01]\nmaterial event detail",
        ):
            out = _augment_news_summary_with_rag(
                ticker="AAPL",
                triggers=["price_move_gt_2atr"],
                base_news_summary="- Headline 1\n- Headline 2",
            )

        # All three pieces preserved
        assert "Headline 1" in out
        assert "fda approval narrative" in out
        assert "material event detail" in out
        # Section labels
        assert "Recent news context" in out
        assert "Recent filings context" in out

    def test_augment_returns_base_when_rag_unavailable(self):
        """When both RAG calls return 'unavailable', the helper falls
        back to the base summary."""
        from agents.sector_teams.sector_team import _augment_news_summary_with_rag

        with patch(
            "agents.sector_teams.rag_retrieval_tools.search_news_impl",
            return_value="News search temporarily unavailable for AAPL.",
        ), patch(
            "agents.sector_teams.rag_retrieval_tools.search_filings_impl",
            return_value="Filing search temporarily unavailable for AAPL.",
        ):
            out = _augment_news_summary_with_rag(
                ticker="AAPL",
                triggers=["x"],
                base_news_summary="- Base headline",
            )
        # The "temporarily unavailable" messages get appended (not
        # filtered) because they're non-empty real strings. Acceptable
        # — they're an honest signal to the LLM that the retrievers
        # didn't work on this run, and the base summary is still
        # surfaced. Pin the contract:
        assert "Base headline" in out

    def test_augment_skips_empty_results(self):
        from agents.sector_teams.sector_team import _augment_news_summary_with_rag

        with patch(
            "agents.sector_teams.rag_retrieval_tools.search_news_impl",
            return_value="No recent news found for AAPL matching 'x'.",
        ), patch(
            "agents.sector_teams.rag_retrieval_tools.search_filings_impl",
            return_value="No filings found for AAPL matching 'x'.",
        ):
            out = _augment_news_summary_with_rag(
                ticker="AAPL",
                triggers=["x"],
                base_news_summary="- Headline",
            )
        # "No recent news" / "No filings" prefixes get filtered out by
        # the helper — only the base summary remains
        assert out == "- Headline"

    def test_augment_handles_exceptions_gracefully(self):
        from agents.sector_teams.sector_team import _augment_news_summary_with_rag

        with patch(
            "agents.sector_teams.rag_retrieval_tools.search_news_impl",
            side_effect=RuntimeError("boom"),
        ), patch(
            "agents.sector_teams.rag_retrieval_tools.search_filings_impl",
            side_effect=RuntimeError("boom"),
        ):
            out = _augment_news_summary_with_rag(
                ticker="AAPL",
                triggers=["x"],
                base_news_summary="- Headline",
            )
        # Falls back to base summary; never crashes
        assert "- Headline" in out

    def test_augment_uses_triggers_in_query(self):
        """The RAG query should incorporate trigger context — not just
        the ticker."""
        from agents.sector_teams.sector_team import _augment_news_summary_with_rag

        captured_queries = []

        def capture_news(ticker, query, **kw):
            captured_queries.append(query)
            return ""

        def capture_filings(ticker, query, **kw):
            captured_queries.append(query)
            return ""

        with patch(
            "agents.sector_teams.rag_retrieval_tools.search_news_impl",
            side_effect=capture_news,
        ), patch(
            "agents.sector_teams.rag_retrieval_tools.search_filings_impl",
            side_effect=capture_filings,
        ):
            _augment_news_summary_with_rag(
                ticker="AAPL",
                triggers=["price_move_gt_2atr", "earnings_beat"],
                base_news_summary="",
            )
        # Trigger words converted from snake_case to spaces in the query
        for q in captured_queries:
            assert "AAPL" in q
            assert "price move gt 2atr" in q
            assert "earnings beat" in q


# ── Tool wiring into qual/quant tools ──────────────────────────────────


class TestToolsetWiring:
    def test_qual_tools_includes_rag_retrieval_tools(self):
        from agents.sector_teams.qual_tools import create_qual_tools

        tools = create_qual_tools(context={})
        names = {t.name for t in tools}
        assert "search_news" in names
        assert "search_filings" in names
        assert "search_transcripts" in names

    def test_quant_tools_includes_rag_retrieval_tools(self):
        from agents.sector_teams.quant_tools import create_quant_tools

        tools = create_quant_tools(context={})
        names = {t.name for t in tools}
        assert "search_news" in names
        assert "search_filings" in names
        assert "search_transcripts" in names
