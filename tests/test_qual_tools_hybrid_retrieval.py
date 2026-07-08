"""Caller-side test for the hybrid-retrieval wiring (PR 3 of 5 in the
BM25 + vector arc).

The qual analyst's ``query_filings`` tool was previously calling
``alpha_engine_lib.rag.retrieve()`` with default arguments (vector
mode). After PR 3, it explicitly opts into hybrid mode with
``vector_weight=0.7`` and emits a structured INFO log carrying the
per-result component scores for downstream observability (decision
artifacts + LangSmith traces + the eval harness in PR 4).

Live retrieval is out of scope here; we mock ``retrieve`` and assert
the caller's call shape + log payload.
"""

from __future__ import annotations

import logging
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from agents.sector_teams.qual_tools import create_qual_tools


def _mock_retrieval_result(
    chunk_id: str,
    *,
    vec: float | None = None,
    kw: float | None = None,
    combined: float | None = None,
    rerank_score: float | None = None,
    rerank_method: str | None = None,
) -> MagicMock:
    """Stand-in for `alpha_engine_lib.rag.retrieval.RetrievalResult`. We
    don't import the real dataclass here because the test runs against
    the lib version pinned in requirements.txt, and a mocked result with
    the right attributes is sufficient.
    """
    r = MagicMock()
    r.chunk_id = chunk_id
    r.content = f"chunk content {chunk_id}"
    r.ticker = "TEST"
    r.doc_type = "10-K"
    r.filed_date = date(2025, 6, 30)
    r.section_label = "MD&A"
    r.similarity = combined or vec or kw or 0.5
    r.vector_score = vec
    r.keyword_score = kw
    r.combined_score = combined
    r.retrieval_method = "hybrid"
    # Rerank fields default to None so existing tests that don't pass
    # them still serialize cleanly into the structured log line
    # (otherwise MagicMock's auto-attribute child mocks would leak
    # ``<MagicMock id=...>`` into the log payload).
    r.rerank_score = rerank_score
    r.rerank_method = rerank_method
    return r


def _get_query_filings_tool():
    """Pull `query_filings` out of the create_qual_tools factory output."""
    tools = create_qual_tools(context={})
    for t in tools:
        if t.name == "query_filings":
            return t
    raise AssertionError("query_filings not found in create_qual_tools output")


class TestQualToolsHybridWiring:
    def test_query_filings_passes_method_hybrid_and_vector_weight(self) -> None:
        captured: dict = {}

        def fake_retrieve(**kwargs):
            captured.update(kwargs)
            return [_mock_retrieval_result("c1", vec=0.7, combined=0.49)]

        # `from alpha_engine_lib.rag import retrieve` runs INSIDE
        # query_filings (deferred import for cold-start cost), so the
        # patch target is the lib symbol — the inline import resolves
        # against the patched module attribute.
        with patch("nousergon_lib.rag.retrieve", side_effect=fake_retrieve):
            tool_fn = _get_query_filings_tool()
            tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})

        assert captured.get("method") == "hybrid", (
            f"query_filings must opt into hybrid retrieval explicitly; got "
            f"method={captured.get('method')!r}"
        )
        assert captured.get("vector_weight") == 0.7, (
            f"vector_weight should be 0.7 in PR 3; PR 5 may move it to "
            f"config after PR 4 calibration. Got {captured.get('vector_weight')!r}"
        )
        # Ensure existing call shape is preserved.
        assert captured.get("tickers") == ["AAPL"]
        assert captured.get("top_k") == 8

    def test_query_filings_emits_structured_log_with_component_scores(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Per-result vector_score / keyword_score / combined_score must
        end up in the structured log so the PR 4 eval harness can read
        calibration data straight from prod logs.
        """
        results = [
            _mock_retrieval_result("c1", vec=0.91, kw=None, combined=0.637),
            _mock_retrieval_result("c2", vec=None, kw=0.42, combined=0.126),
        ]

        with patch("nousergon_lib.rag.retrieve", return_value=results):
            with caplog.at_level(logging.INFO, logger="agents.sector_teams.qual_tools"):
                tool_fn = _get_query_filings_tool()
                tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})

        rag_logs = [r for r in caplog.records if "RAG_RETRIEVE" in r.getMessage()]
        assert len(rag_logs) == 1, "expected exactly one RAG_RETRIEVE log line"
        msg = rag_logs[0].getMessage()
        assert "method=hybrid" in msg
        assert "vector_weight=0.7" in msg
        assert "n_results=2" in msg
        # Per-chunk component scores serialize into the log payload.
        assert "'chunk_id': 'c1'" in msg
        assert "'vector_score': 0.91" in msg
        assert "'keyword_score': 0.42" in msg
        assert "'combined_score': 0.637" in msg or "'combined_score': 0.126" in msg

    def test_query_filings_returns_empty_message_when_no_results(self) -> None:
        with patch("nousergon_lib.rag.retrieve", return_value=[]):
            tool_fn = _get_query_filings_tool()
            out = tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})
        assert "No filing data found" in out

    def test_query_filings_swallows_retrieve_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Failure mode contract preserved: a RAG outage must not crash
        the qual agent — return a fallback message and log a WARNING.
        """
        with patch("nousergon_lib.rag.retrieve", side_effect=RuntimeError("neon down")):
            with caplog.at_level(logging.WARNING, logger="agents.sector_teams.qual_tools"):
                tool_fn = _get_query_filings_tool()
                out = tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})
        assert "temporarily unavailable" in out
        assert any("RAG_UNAVAILABLE" in r.getMessage() for r in caplog.records)


class TestQualToolsRerankFlag:
    """L1303 PR 2 — `RAG_RERANK` env-var toggle wiring.

    The flag defaults unset (==> no rerank kwargs passed, hybrid-only
    path preserved). When set, `query_filings` widens the candidate
    fetch and passes the rerank knobs through to
    ``alpha_engine_lib.rag.retrieve()``.

    These tests patch the module-level ``_RAG_RERANK`` /
    ``_RAG_RERANK_INPUT_N`` constants directly because they're resolved
    once at import time — env-var changes via ``monkeypatch.setenv``
    after import are invisible.
    """

    def test_rerank_unset_omits_kwargs(self) -> None:
        """Default config — no rerank kwargs leak into the retrieve() call."""
        captured: dict = {}

        def fake_retrieve(**kwargs):
            captured.update(kwargs)
            return [_mock_retrieval_result("c1", vec=0.7, combined=0.49)]

        with patch("agents.sector_teams.qual_tools._RAG_RERANK", None), \
             patch("nousergon_lib.rag.retrieve", side_effect=fake_retrieve):
            tool_fn = _get_query_filings_tool()
            tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})

        assert "rerank" not in captured
        assert "rerank_input_n" not in captured

    def test_rerank_cross_encoder_passes_through(self) -> None:
        """RAG_RERANK=cross_encoder → kwargs land on retrieve() with the
        configured input_n.
        """
        captured: dict = {}

        def fake_retrieve(**kwargs):
            captured.update(kwargs)
            return [_mock_retrieval_result(
                "c1", vec=0.7, combined=0.49,
                rerank_score=0.92, rerank_method="cross_encoder",
            )]

        with patch("agents.sector_teams.qual_tools._RAG_RERANK", "cross_encoder"), \
             patch("agents.sector_teams.qual_tools._RAG_RERANK_INPUT_N", 30), \
             patch("nousergon_lib.rag.retrieve", side_effect=fake_retrieve):
            tool_fn = _get_query_filings_tool()
            tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})

        assert captured.get("rerank") == "cross_encoder"
        assert captured.get("rerank_input_n") == 30
        # Other kwargs still pass through unchanged
        assert captured.get("method") == "hybrid"
        assert captured.get("top_k") == 8

    def test_log_includes_rerank_fields_when_set(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RAG_RETRIEVE log line surfaces rerank_method + per-result
        rerank_score so the PR 3 eval harness can read both pre- and
        post-rerank calibration data from prod logs.
        """
        results = [
            _mock_retrieval_result(
                "c1", vec=0.91, combined=0.637,
                rerank_score=0.95, rerank_method="cross_encoder",
            ),
        ]

        with patch("agents.sector_teams.qual_tools._RAG_RERANK", "cross_encoder"), \
             patch("agents.sector_teams.qual_tools._RAG_RERANK_INPUT_N", 30), \
             patch("nousergon_lib.rag.retrieve", return_value=results):
            with caplog.at_level(logging.INFO, logger="agents.sector_teams.qual_tools"):
                tool_fn = _get_query_filings_tool()
                tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})

        rag_logs = [r for r in caplog.records if "RAG_RETRIEVE" in r.getMessage()]
        assert len(rag_logs) == 1
        msg = rag_logs[0].getMessage()
        assert "rerank=cross_encoder" in msg
        assert "rerank_input_n=30" in msg
        assert "'rerank_score': 0.95" in msg
        assert "'rerank_method': 'cross_encoder'" in msg

    def test_log_rerank_none_when_unset(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unset flag still produces a parseable log line — fields show
        rerank=none / rerank_input_n=0 / per-result rerank_score=None so
        eval-harness consumers see consistent shape across both modes.
        """
        results = [_mock_retrieval_result("c1", vec=0.91, combined=0.637)]

        with patch("agents.sector_teams.qual_tools._RAG_RERANK", None), \
             patch("nousergon_lib.rag.retrieve", return_value=results):
            with caplog.at_level(logging.INFO, logger="agents.sector_teams.qual_tools"):
                tool_fn = _get_query_filings_tool()
                tool_fn.invoke({"ticker": "AAPL", "query": "competitive moat"})

        msg = [r for r in caplog.records if "RAG_RETRIEVE" in r.getMessage()][0].getMessage()
        assert "rerank=none" in msg
        assert "rerank_input_n=0" in msg
        assert "'rerank_score': None" in msg
