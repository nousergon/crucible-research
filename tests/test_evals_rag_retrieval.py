"""Tests for the RAG hybrid-retrieval eval harness (PR 4 of 5 in the
BM25 + vector arc).

Three layers exercised:

1. Pure functions — `recall_at_k`, `aggregate_recall`,
   `aggregate_by_category`, `render_markdown_report`. No live DB.
2. Harness orchestrator — `run_eval` with a fake `retrieve_fn`.
3. CLI loader — `scripts/run_rag_retrieval_eval.py::load_queries`
   YAML schema validation.

Live retrieval against Neon is NOT exercised here — that's the
operator runs the CLI against, after curating the YAML test set.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.rag_retrieval import (
    CATEGORIES,
    DEFAULT_CONDITIONS,
    DEFAULT_K_VALUES,
    AggregateRow,
    Condition,
    EvalQuery,
    QueryResult,
    aggregate_by_category,
    aggregate_recall,
    recall_at_k,
    render_markdown_report,
    run_eval,
)


# ── recall_at_k ─────────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_full_hit_single_relevant(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0

    def test_miss_single_relevant(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["d"], k=3) == 0.0

    def test_partial_hit_multi_relevant(self) -> None:
        # 2 of 4 expected appear in top-k → recall = 0.5
        assert recall_at_k(["a", "b", "c", "x"], ["a", "c", "y", "z"], k=4) == 0.5

    def test_top_k_truncation_excludes_relevant(self) -> None:
        # Expected chunk is at position 3 but k=2 → miss.
        assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0

    def test_top_k_truncation_includes_relevant(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a"], k=2) == 1.0

    def test_empty_expected_returns_zero_not_crash(self) -> None:
        # Defensive — curation error shouldn't crash the harness.
        assert recall_at_k(["a"], [], k=5) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        assert recall_at_k([], ["a"], k=5) == 0.0


# ── aggregate_recall + aggregate_by_category ────────────────────────────────


def _q(query: str, *expected: str, category: str = "abstract_thesis") -> EvalQuery:
    return EvalQuery(
        query=query, expected_chunk_ids=tuple(expected), category=category
    )


def _qr(query: EvalQuery, condition: str, recalls: dict[int, float]) -> QueryResult:
    return QueryResult(
        query=query, condition_name=condition,
        recalls=recalls, retrieved_chunk_ids=(),
    )


class TestAggregateRecall:
    def test_groups_by_condition_and_means(self) -> None:
        q1 = _q("query 1", "c1")
        q2 = _q("query 2", "c2")
        results = [
            _qr(q1, "vector", {5: 1.0, 10: 1.0}),
            _qr(q2, "vector", {5: 0.0, 10: 1.0}),
            _qr(q1, "hybrid w=0.7", {5: 1.0, 10: 1.0}),
            _qr(q2, "hybrid w=0.7", {5: 1.0, 10: 1.0}),
        ]
        agg = aggregate_recall(results, k_values=[5, 10])
        assert agg["vector"].n_queries == 2
        assert agg["vector"].mean_recall[5] == 0.5
        assert agg["vector"].mean_recall[10] == 1.0
        assert agg["hybrid w=0.7"].mean_recall[5] == 1.0

    def test_empty_results_returns_empty_dict(self) -> None:
        assert aggregate_recall([], k_values=[5, 10]) == {}

    def test_missing_k_in_recalls_yields_zero(self) -> None:
        # Defensive: a partial result missing some k entries.
        q = _q("q", "c")
        results = [_qr(q, "vector", {5: 1.0})]  # no k=10 entry
        agg = aggregate_recall(results, k_values=[5, 10])
        assert agg["vector"].mean_recall[5] == 1.0
        assert agg["vector"].mean_recall[10] == 0.0


class TestAggregateByCategory:
    def test_separates_categories(self) -> None:
        q_cat1 = _q("q1", "c1", category="ticker_named_entity")
        q_cat2 = _q("q2", "c2", category="quantitative_line_item")
        results = [
            _qr(q_cat1, "vector", {5: 1.0}),
            _qr(q_cat2, "vector", {5: 0.0}),
            _qr(q_cat1, "hybrid w=0.7", {5: 1.0}),
            _qr(q_cat2, "hybrid w=0.7", {5: 1.0}),
        ]
        per_cat = aggregate_by_category(results, k_values=[5])
        assert set(per_cat.keys()) == {"ticker_named_entity", "quantitative_line_item"}
        assert per_cat["ticker_named_entity"]["vector"].mean_recall[5] == 1.0
        assert per_cat["quantitative_line_item"]["vector"].mean_recall[5] == 0.0
        assert per_cat["quantitative_line_item"]["hybrid w=0.7"].mean_recall[5] == 1.0


# ── run_eval orchestrator ───────────────────────────────────────────────────


class TestRunEval:
    def test_calls_retrieve_for_every_query_condition_combo(self) -> None:
        queries = [_q(f"q{i}", f"chunk{i}") for i in range(3)]
        # 6 default conditions × 3 queries = 18 calls.
        call_count = 0

        def fake_retrieve(**kwargs):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.chunk_id = "miss"  # always miss → recall=0 across the board
            return [r]

        results = run_eval(queries=queries, retrieve_fn=fake_retrieve)
        assert call_count == 3 * len(DEFAULT_CONDITIONS)
        assert len(results) == 3 * len(DEFAULT_CONDITIONS)

    def test_passes_method_and_vector_weight_kwargs(self) -> None:
        captured: list[dict] = []

        def fake_retrieve(**kwargs):
            captured.append(kwargs)
            return []

        run_eval(
            queries=[_q("q", "c")],
            retrieve_fn=fake_retrieve,
            conditions=[
                Condition("vector-only", "vector", None),
                Condition("hybrid-mid", "hybrid", 0.5),
            ],
            k_values=[5],
        )
        assert captured[0]["method"] == "vector"
        assert "vector_weight" not in captured[0]
        assert captured[1]["method"] == "hybrid"
        assert captured[1]["vector_weight"] == 0.5

    def test_records_recall_at_each_k(self) -> None:
        # Retrieved set: ["c1", "miss1", "c2", "miss2", "miss3"]
        # Expected: ["c1", "c2"] — recall@1=0.5, recall@3=1.0, recall@5=1.0
        retrieved = []
        for cid in ["c1", "miss1", "c2", "miss2", "miss3"]:
            r = MagicMock()
            r.chunk_id = cid
            retrieved.append(r)

        def fake_retrieve(**kwargs):
            return retrieved

        results = run_eval(
            queries=[_q("q", "c1", "c2")],
            retrieve_fn=fake_retrieve,
            conditions=[Condition("test", "vector", None)],
            k_values=[1, 3, 5],
        )
        assert len(results) == 1
        assert results[0].recalls[1] == 0.5
        assert results[0].recalls[3] == 1.0
        assert results[0].recalls[5] == 1.0

    def test_top_k_defaults_to_max_k_value(self) -> None:
        captured: list[dict] = []

        def fake_retrieve(**kwargs):
            captured.append(kwargs)
            return []

        run_eval(
            queries=[_q("q", "c")],
            retrieve_fn=fake_retrieve,
            conditions=[Condition("v", "vector", None)],
            k_values=[5, 10, 20],
        )
        assert captured[0]["top_k"] == 20  # max(k_values)


# ── render_markdown_report ──────────────────────────────────────────────────


class TestRenderMarkdownReport:
    def test_empty_queries_renders_curation_prompt(self) -> None:
        md = render_markdown_report(
            run_date=date(2026, 5, 9), queries=[], results=[],
        )
        assert "No queries curated yet" in md
        assert "rag_retrieval_queries.yaml" in md

    def test_empty_results_with_queries_says_runner_failed(self) -> None:
        md = render_markdown_report(
            run_date=date(2026, 5, 9),
            queries=[_q("q", "c")],
            results=[],
        )
        assert "looks like" in md and "runner failed" in md

    def test_renders_overall_table_and_per_category(self) -> None:
        q_ticker = _q("ABBV moat", "c1", category="ticker_named_entity")
        q_quant = _q("PFE R&D $", "c2", category="quantitative_line_item")
        results = [
            _qr(q_ticker, "vector", {5: 1.0, 10: 1.0, 20: 1.0}),
            _qr(q_quant, "vector", {5: 0.0, 10: 0.0, 20: 1.0}),
            _qr(q_ticker, "hybrid w=0.7", {5: 1.0, 10: 1.0, 20: 1.0}),
            _qr(q_quant, "hybrid w=0.7", {5: 1.0, 10: 1.0, 20: 1.0}),
        ]
        md = render_markdown_report(
            run_date=date(2026, 5, 9),
            queries=[q_ticker, q_quant],
            results=results,
            conditions=[
                Condition("vector", "vector", None),
                Condition("hybrid w=0.7", "hybrid", 0.7),
            ],
            k_values=[5, 10, 20],
        )
        assert "# RAG retrieval eval — 2026-05-09" in md
        assert "## Overall recall@k" in md
        assert "## By category" in md
        assert "ticker_named_entity" in md
        assert "quantitative_line_item" in md
        assert "## Recommendation" in md
        # Hybrid should win at recall@10 → recommended
        assert "hybrid w=0.7" in md and "Best condition" in md

    def test_recommendation_calls_out_negative_lift(self) -> None:
        # Pure vector wins → recommendation should still show, with
        # the lift over vector being 0.000 since vector is the best.
        q = _q("q", "c1")
        results = [
            _qr(q, "vector", {5: 1.0, 10: 1.0, 20: 1.0}),
            _qr(q, "hybrid w=0.7", {5: 0.0, 10: 0.0, 20: 0.0}),
        ]
        md = render_markdown_report(
            run_date=date(2026, 5, 9),
            queries=[q],
            results=results,
            conditions=[
                Condition("vector", "vector", None),
                Condition("hybrid w=0.7", "hybrid", 0.7),
            ],
            k_values=[5, 10, 20],
        )
        # The "best" should be vector here.
        assert "Best condition" in md
        assert "vector" in md


# ── CLI loader ──────────────────────────────────────────────────────────────


class TestLoadQueries:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        from scripts.run_rag_retrieval_eval import load_queries

        yaml_text = """
queries:
  - query: "ABBV moat"
    expected_chunk_ids:
      - "uuid-1"
    category: "abstract_thesis"
    note: "Tests cosine paraphrase"
  - query: "PFE R&D"
    expected_chunk_ids:
      - "uuid-2"
      - "uuid-3"
    category: "quantitative_line_item"
"""
        path = tmp_path / "q.yaml"
        path.write_text(yaml_text)
        out = load_queries(path)
        assert len(out) == 2
        assert out[0].query == "ABBV moat"
        assert out[0].expected_chunk_ids == ("uuid-1",)
        assert out[0].category == "abstract_thesis"
        assert out[0].note == "Tests cosine paraphrase"
        assert out[1].expected_chunk_ids == ("uuid-2", "uuid-3")

    def test_empty_queries_list_is_valid(self, tmp_path: Path) -> None:
        from scripts.run_rag_retrieval_eval import load_queries

        path = tmp_path / "q.yaml"
        path.write_text("queries: []\n")
        assert load_queries(path) == []

    def test_unknown_category_raises(self, tmp_path: Path) -> None:
        from scripts.run_rag_retrieval_eval import load_queries

        yaml_text = """
queries:
  - query: "x"
    expected_chunk_ids: ["c"]
    category: "made_up_category"
"""
        path = tmp_path / "q.yaml"
        path.write_text(yaml_text)
        with pytest.raises(ValueError, match="not in"):
            load_queries(path)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        from scripts.run_rag_retrieval_eval import load_queries

        yaml_text = """
queries:
  - query: "x"
    category: "abstract_thesis"
"""
        path = tmp_path / "q.yaml"
        path.write_text(yaml_text)
        with pytest.raises(ValueError, match="missing required field"):
            load_queries(path)

    def test_empty_expected_chunks_raises(self, tmp_path: Path) -> None:
        # A query with no expected chunks is a curation bug — fail loud.
        from scripts.run_rag_retrieval_eval import load_queries

        yaml_text = """
queries:
  - query: "x"
    expected_chunk_ids: []
    category: "abstract_thesis"
"""
        path = tmp_path / "q.yaml"
        path.write_text(yaml_text)
        with pytest.raises(ValueError, match="non-empty"):
            load_queries(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from scripts.run_rag_retrieval_eval import load_queries

        with pytest.raises(FileNotFoundError):
            load_queries(tmp_path / "nope.yaml")

    def test_todo_sentinel_silently_skipped(self, tmp_path: Path) -> None:
        """Curate workflow uses ['TODO'] to mean 'no candidate in top-10
        was relevant — skip this query'. Eval drops them silently.
        """
        from scripts.run_rag_retrieval_eval import load_queries

        yaml_text = """
queries:
  - query: "real query"
    expected_chunk_ids: ["uuid-1"]
    category: "abstract_thesis"
  - query: "skipped query"
    expected_chunk_ids: ["TODO"]
    category: "filing_type"
  - query: "another real"
    expected_chunk_ids: ["uuid-2"]
    category: "date_range"
"""
        path = tmp_path / "q.yaml"
        path.write_text(yaml_text)
        out = load_queries(path)
        assert len(out) == 2  # TODO entry dropped
        assert [q.query for q in out] == ["real query", "another real"]
