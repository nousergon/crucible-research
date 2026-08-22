"""Unit tests for cross-week rationale clustering.

Covers:
- Per-agent rationale extraction across the 6 RETIRED agent_id families
  (sector_quant, sector_qual, sector_peer_review, macro_economist,
  ic_cio, thesis_update) plus the unknown-agent silent-skip case. Kept
  live and unit-tested per alpha-engine-config-I8173 deliverable 4: if
  one of these families is ever revived under a new path, this parsing
  stays correct.
- TF-IDF char n-gram vectorization correctness — same template
  produces high cosine; different templates produce low cosine.
- Greedy single-linkage clustering — template rationales merge,
  distinct rationales stay separate.
- Top-3 concentration math (edge cases: single cluster, fewer than 3
  clusters, empty corpus).
- ``compute_and_emit`` RETIRED (alpha-engine-config-I8173): no S3 read,
  no S3 write, no CloudWatch emission — an explicit retirement summary
  instead. Previously this file also covered the live S3-read pipeline
  (thin-sample skip, load-failure handling, scope caps, key-level caps,
  dry_run path) — all deleted here, their subject removed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

# ── Per-agent extraction ──────────────────────────────────────────────────


class TestExtractRationales:
    def test_sector_quant_ranked_picks(self):
        # Real capture shape: ranked_picks[*].rationale
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "sector_quant:technology",
            {
                "ranked_picks": [
                    {"ticker": "NVDA", "rationale": "P/E of 32 attractive"},
                    {"ticker": "AAPL", "rationale": "FCF yield strong"},
                ]
            },
        )
        assert out == ["P/E of 32 attractive", "FCF yield strong"]

    def test_sector_qual_assessments(self):
        # Real capture shape: assessments[*].bull_case
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "sector_qual:healthcare",
            {
                "assessments": [
                    {"ticker": "PFE", "bull_case": "Pipeline strong"},
                    {"ticker": "MRK", "bull_case": "Oncology lead"},
                ]
            },
        )
        assert out == ["Pipeline strong", "Oncology lead"]

    def test_sector_peer_review_recommendations_plus_team(self):
        # Real capture shape: recommendations[*].peer_review_rationale +
        # top-level peer_review_rationale.
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "sector_peer_review:financials",
            {
                "recommendations": [
                    {"ticker": "JPM", "peer_review_rationale": "Strong NIM tailwind"},
                ],
                "peer_review_rationale": "Sector concentration controlled",
            },
        )
        assert "Strong NIM tailwind" in out
        assert "Sector concentration controlled" in out

    def test_macro_economist_picks_macro_report(self):
        # Real capture shape: macro_report (~2KB narrative); other keys
        # don't appear on real captures but stay as fallback.
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "macro_economist",
            {
                "macro_report": "Full regime narrative",
                "market_regime": "BULL",
            },
        )
        assert out == ["Full regime narrative"]

    def test_macro_economist_falls_back_to_alt_keys(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "macro_economist",
            {"regime_rationale": "fallback narrative"},
        )
        assert out == ["fallback narrative"]

    def test_ic_cio_decisions(self):
        # Real capture shape: ic_decisions[*].rationale
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "ic_cio",
            {
                "ic_decisions": [
                    {"ticker": "META", "rationale": "Composite 78, R/R 2.1"},
                    {"ticker": "GOOG", "rationale": "Composite 81, R/R 2.4"},
                ]
            },
        )
        assert out == ["Composite 78, R/R 2.1", "Composite 81, R/R 2.4"]

    def test_thesis_update_pulls_all_four_narrative_fields(self):
        # Real capture shape: bull_case + conviction_rationale +
        # thesis_summary + triggers_response.
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "thesis_update:AAPL",
            {
                "bull_case": "Services growth accelerating",
                "conviction_rationale": "Confirmed by Q3 print",
                "thesis_summary": "Long-term compounder; current setup favorable",
                "triggers_response": "Triggers within tolerance — hold",
            },
        )
        assert len(out) == 4
        assert "Services growth accelerating" in out
        assert "Confirmed by Q3 print" in out
        assert "Long-term compounder; current setup favorable" in out
        assert "Triggers within tolerance — hold" in out

    def test_unknown_agent_returns_empty(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales("brand_new_agent", {"some_field": "data"})
        assert out == []

    def test_empty_agent_output_returns_empty(self):
        from evals.rationale_clustering import extract_rationales

        assert extract_rationales("sector_quant:tech", {}) == []
        assert extract_rationales("sector_quant:tech", None) == []  # type: ignore[arg-type]

    def test_drops_blank_strings(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "sector_quant:technology",
            {
                "ranked_picks": [
                    {"ticker": "X", "rationale": "valid"},
                    {"ticker": "Y", "rationale": ""},
                    {"ticker": "Z"},  # missing key entirely
                ]
            },
        )
        assert out == ["valid"]


# ── TF-IDF + cosine similarity ────────────────────────────────────────────


class TestTfidfCosine:
    def test_identical_strings_have_cosine_one(self):
        from evals.rationale_clustering import _build_tfidf_matrix, _cosine_sim

        vecs, _ = _build_tfidf_matrix(["the same text", "the same text"])
        assert _cosine_sim(vecs[0], vecs[1]) == pytest.approx(1.0, abs=1e-6)

    def test_template_differs_only_in_numbers_high_cosine(self):
        from evals.rationale_clustering import _build_tfidf_matrix, _cosine_sim

        # Same skeleton, different numerics — exactly what we want to
        # detect as "template-generation."
        vecs, _ = _build_tfidf_matrix(
            [
                "P/E of 12 attractive vs sector median of 18",
                "P/E of 25 attractive vs sector median of 30",
            ]
        )
        sim = _cosine_sim(vecs[0], vecs[1])
        assert sim > 0.65, f"expected >0.65 for template match, got {sim}"

    def test_distinct_rationales_have_low_cosine(self):
        from evals.rationale_clustering import _build_tfidf_matrix, _cosine_sim

        vecs, _ = _build_tfidf_matrix(
            [
                "Cyclical recovery driven by capex",
                "Pipeline approval expected Q3 catalyst",
            ]
        )
        sim = _cosine_sim(vecs[0], vecs[1])
        assert sim < 0.5, f"expected <0.5 for distinct rationales, got {sim}"


# ── Clustering ────────────────────────────────────────────────────────────


class TestClusterRationales:
    def test_empty_input_returns_empty(self):
        from evals.rationale_clustering import cluster_rationales

        assert cluster_rationales([]) == []

    def test_template_rationales_merge_into_one_cluster(self):
        from evals.rationale_clustering import cluster_rationales

        rationales = [
            "P/E of 12 attractive vs sector median of 18",
            "P/E of 25 attractive vs sector median of 30",
            "P/E of 8 attractive vs sector median of 14",
        ]
        clusters = cluster_rationales(rationales)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_distinct_rationales_stay_separate(self):
        from evals.rationale_clustering import cluster_rationales

        rationales = [
            "Cyclical recovery driven by capex",
            "Pipeline approval expected Q3 catalyst",
            "Margin expansion from cost-cutting program",
        ]
        clusters = cluster_rationales(rationales)
        assert len(clusters) == 3

    def test_mixed_corpus_partitions_correctly(self):
        from evals.rationale_clustering import cluster_rationales

        # 3 templated + 2 distinct → expect 3 clusters total.
        rationales = [
            "P/E of 12 attractive vs sector median of 18",
            "P/E of 25 attractive vs sector median of 30",
            "P/E of 8 attractive vs sector median of 14",
            "Cyclical recovery driven by capex",
            "Pipeline approval expected Q3 catalyst",
        ]
        clusters = cluster_rationales(rationales)
        assert len(clusters) == 3
        sizes = sorted((len(c) for c in clusters), reverse=True)
        assert sizes == [3, 1, 1]


class TestComputeConcentration:
    def test_empty_clusters_returns_zero(self):
        from evals.rationale_clustering import compute_concentration

        assert compute_concentration([]) == 0.0

    def test_single_cluster_is_one(self):
        from evals.rationale_clustering import compute_concentration

        assert compute_concentration([[0, 1, 2, 3]]) == 1.0

    def test_top3_of_5_clusters(self):
        from evals.rationale_clustering import compute_concentration

        # Sizes 5, 4, 3, 2, 1 → total 15, top3 = 12 → 0.8
        clusters = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11],
            [12, 13],
            [14],
        ]
        assert compute_concentration(clusters, top_k=3) == pytest.approx(12 / 15)

    def test_fewer_than_k_clusters_uses_all(self):
        from evals.rationale_clustering import compute_concentration

        clusters = [[0, 1], [2]]
        # top_k=3 but only 2 clusters → use both → 1.0
        assert compute_concentration(clusters, top_k=3) == 1.0


# ── End-to-end pipeline ───────────────────────────────────────────────────




class TestComputeAndEmit:
    """RETIRED (alpha-engine-config-I8173): no S3 listing, read, cluster,
    persist, or CloudWatch emission may happen for any of the six retired
    agent families. Any client passed in must see ZERO calls."""

    def test_no_s3_or_cloudwatch_calls_are_made(self):
        from evals.rationale_clustering import compute_and_emit

        s3 = MagicMock()
        cw = MagicMock()
        compute_and_emit(
            end_time=datetime(2026, 8, 22, tzinfo=UTC),
            s3_client=s3,
            cloudwatch_client=cw,
        )
        s3.get_paginator.assert_not_called()
        s3.get_object.assert_not_called()
        s3.put_object.assert_not_called()
        cw.put_metric_data.assert_not_called()

    def test_summary_names_retirement_and_the_six_families(self):
        from evals.rationale_clustering import (
            RETIRED_AGENT_FAMILIES,
            compute_and_emit,
        )

        summary = compute_and_emit(
            end_time=datetime(2026, 8, 22, tzinfo=UTC),
            s3_client=MagicMock(),
            cloudwatch_client=MagicMock(),
        )
        assert summary["status"] == "retired"
        assert set(summary["retired_agent_families"]) == RETIRED_AGENT_FAMILIES
        assert "alpha-engine-config-I7827" in summary["retired_reason"]
        assert "alpha-engine-config-I7817" in summary["retired_reason"]
        # Nothing was read, clustered, or persisted.
        assert summary["agents_analyzed"] == 0
        assert summary["artifacts_discovered"] == 0
        assert summary["per_agent"] == []
        assert summary["agents_stale_corpus"] == []
        # The handler's `has_failures` check still works unmodified.
        assert summary["load_failures"] == []
        assert summary["cluster_failures"] == []

    def test_retired_agent_families_matches_the_extractor_families(self):
        """The retirement set must be exactly the families
        ``extract_rationales`` ever had a branch for — no more, no fewer.
        Drift here is exactly how a live family could go unmonitored."""
        from evals.rationale_clustering import (
            RETIRED_AGENT_FAMILIES,
            extract_rationales,
        )

        assert RETIRED_AGENT_FAMILIES == {
            "ic_cio",
            "macro_economist",
            "sector_peer_review",
            "sector_qual",
            "sector_quant",
            "thesis_update",
        }
        for family in RETIRED_AGENT_FAMILIES:
            # Each still parses correctly if ever revived (deliverable 4) —
            # a non-empty output proves the branch logic, not retirement.
            assert extract_rationales(family, {}) == []

    def test_default_cap_value(self):
        """Pin the default to the published value (mirrors Counterfactual
        #228's DEFAULT_MAX_ARTIFACTS_PER_AGENT=500). Unused by the retired
        pipeline today; kept because ``max_rationales_per_agent`` stays a
        stable kwarg on ``compute_and_emit`` for the handler."""
        from evals.rationale_clustering import DEFAULT_MAX_RATIONALES_PER_AGENT

        assert DEFAULT_MAX_RATIONALES_PER_AGENT == 500
