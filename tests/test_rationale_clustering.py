"""Unit tests for cross-week rationale clustering.

Covers:
- Per-agent rationale extraction across the 6 supported agent_id
  families (sector_quant, sector_qual, sector_peer_review,
  macro_economist, ic_cio, thesis_update) plus the unknown-agent
  silent-skip case.
- TF-IDF char n-gram vectorization correctness — same template
  produces high cosine; different templates produce low cosine.
- Greedy single-linkage clustering — template rationales merge,
  distinct rationales stay separate.
- Top-3 concentration math (edge cases: single cluster, fewer than 3
  clusters, empty corpus).
- ``compute_and_emit`` end-to-end with stubbed S3 + CloudWatch:
  happy path, thin-sample skip, load-failure handling, dry_run path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


# ── Per-agent extraction ──────────────────────────────────────────────────


class TestExtractRationales:
    def test_sector_quant_top5_picks(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "sector_quant:tech",
            {
                "top5_picks": [
                    {"ticker": "NVDA", "quant_rationale": "P/E of 32 attractive"},
                    {"ticker": "AAPL", "quant_rationale": "FCF yield strong"},
                ]
            },
        )
        assert out == ["P/E of 32 attractive", "FCF yield strong"]

    def test_sector_qual_assessments(self):
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

    def test_sector_peer_review_decisions_plus_team(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "sector_peer_review:financials",
            {
                "selected_decisions": [
                    {"ticker": "JPM", "rationale": "Strong NIM tailwind"},
                ],
                "team_rationale": "Sector concentration controlled",
            },
        )
        assert "Strong NIM tailwind" in out
        assert "Sector concentration controlled" in out

    def test_macro_economist_picks_longest_field(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "macro_economist",
            {
                "regime_rationale": "short",
                "macro_report": "much longer narrative about regime classification",
                "summary": "tiny",
            },
        )
        # Picks longest of the candidate fields.
        assert out == ["much longer narrative about regime classification"]

    def test_ic_cio_decisions(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "ic_cio",
            {
                "decisions": [
                    {"ticker": "META", "rationale": "Composite 78, R/R 2.1"},
                    {"ticker": "GOOG", "rationale": "Composite 81, R/R 2.4"},
                ]
            },
        )
        assert out == ["Composite 78, R/R 2.1", "Composite 81, R/R 2.4"]

    def test_thesis_update_pulls_bull_case_and_conviction(self):
        from evals.rationale_clustering import extract_rationales

        out = extract_rationales(
            "thesis_update:AAPL",
            {
                "bull_case": "Services growth accelerating",
                "conviction_rationale": "Confirmed by Q3 print",
            },
        )
        assert "Services growth accelerating" in out
        assert "Confirmed by Q3 print" in out

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
            "sector_quant:tech",
            {
                "top5_picks": [
                    {"ticker": "X", "quant_rationale": "valid"},
                    {"ticker": "Y", "quant_rationale": ""},
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


def _build_s3_stub_with_artifacts(artifacts_by_key: dict[str, dict]) -> MagicMock:
    """Build a MagicMock S3 client backed by a synthetic per-day key
    listing + per-key get_object payload."""
    s3 = MagicMock()

    # Group keys by date partition so per-day listing returns the right
    # subset. The compute_and_emit pipeline lists per-day; we mirror that.
    by_prefix: dict[str, list[str]] = {}
    for key in artifacts_by_key:
        # decision_artifacts/YYYY/MM/DD/agent/run.json → prefix is everything up to and including DD/.
        parts = key.split("/")
        prefix = "/".join(parts[:4]) + "/"
        by_prefix.setdefault(prefix, []).append(key)

    paginator = MagicMock()

    def paginate(*, Bucket, Prefix):
        keys = by_prefix.get(Prefix, [])
        return [{"Contents": [{"Key": k} for k in keys]}]

    paginator.paginate.side_effect = paginate
    s3.get_paginator.return_value = paginator

    def get_object(*, Bucket, Key):
        body = MagicMock()
        body.read.return_value = json.dumps(artifacts_by_key[Key]).encode("utf-8")
        return {"Body": body}

    s3.get_object.side_effect = get_object
    s3.put_object = MagicMock()
    return s3


class TestComputeAndEmit:
    def test_thin_sample_skipped(self):
        from evals.rationale_clustering import compute_and_emit

        # Single sector_quant artifact with 2 picks → below MIN floor.
        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        key = "decision_artifacts/2026/05/09/sector_quant/run-1.json"
        artifacts = {
            key: {
                "agent_id": "sector_quant",
                "agent_output": {
                    "top5_picks": [
                        {"ticker": "X", "quant_rationale": "rationale a"},
                        {"ticker": "Y", "quant_rationale": "rationale b"},
                    ]
                },
            }
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        summary = compute_and_emit(
            end_time=end,
            window_days=1,
            s3_client=s3,
            cloudwatch_client=cw,
        )
        assert summary["agents_analyzed"] == 0
        assert summary["agents_skipped_thin_sample"][0]["agent_id"] == "sector_quant"
        # No metric emitted for skipped agent.
        cw.put_metric_data.assert_not_called()

    def test_high_concentration_template_corpus(self):
        from evals.rationale_clustering import compute_and_emit

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        # 8 templated rationales spread across 2 artifacts → above floor.
        templates = [
            f"P/E of {n} attractive vs sector median of {n + 5}"
            for n in (10, 12, 14, 16, 18, 20, 22, 24)
        ]
        key1 = "decision_artifacts/2026/05/09/sector_quant/run-1.json"
        key2 = "decision_artifacts/2026/05/09/sector_quant/run-2.json"
        artifacts = {
            key1: {
                "agent_id": "sector_quant",
                "agent_output": {
                    "top5_picks": [
                        {"ticker": f"T{i}", "quant_rationale": templates[i]}
                        for i in range(4)
                    ]
                },
            },
            key2: {
                "agent_id": "sector_quant",
                "agent_output": {
                    "top5_picks": [
                        {"ticker": f"T{i}", "quant_rationale": templates[i]}
                        for i in range(4, 8)
                    ]
                },
            },
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        summary = compute_and_emit(
            end_time=end,
            window_days=1,
            s3_client=s3,
            cloudwatch_client=cw,
        )

        assert summary["agents_analyzed"] == 1
        per_agent = summary["per_agent"][0]
        assert per_agent["agent_id"] == "sector_quant"
        assert per_agent["n_rationales"] == 8
        # All 8 follow the same template → top-3 concentration = 1.0.
        assert per_agent["top3_concentration"] == pytest.approx(1.0)
        # Metric emitted with concentration + n_rationales (2 datapoints).
        cw.put_metric_data.assert_called_once()
        call = cw.put_metric_data.call_args
        assert call.kwargs["Namespace"] == "AlphaEngine/Eval"
        names = [d["MetricName"] for d in call.kwargs["MetricData"]]
        assert "agent_rationale_template_concentration" in names
        assert "agent_rationale_template_concentration_n_rationales" in names
        # Per-agent analysis was persisted.
        assert s3.put_object.called
        put_call = s3.put_object.call_args
        assert "_analysis/sector_quant/2026-W19" in put_call.kwargs["Key"]

    def test_dry_run_skips_metric_emission(self):
        from evals.rationale_clustering import compute_and_emit

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        templates = [
            f"P/E of {n} attractive vs sector median of {n + 5}"
            for n in range(10, 18)
        ]
        key = "decision_artifacts/2026/05/09/sector_quant/run-1.json"
        artifacts = {
            key: {
                "agent_id": "sector_quant",
                "agent_output": {
                    "top5_picks": [
                        {"ticker": f"T{i}", "quant_rationale": templates[i]}
                        for i in range(8)
                    ]
                },
            },
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)

        summary = compute_and_emit(
            end_time=end,
            window_days=1,
            s3_client=s3,
            emit_metrics=False,
        )
        assert summary["agents_analyzed"] == 1
        # put_object still fires (analysis persists even on dry-run); no
        # CloudWatch path because cloudwatch_client is None.
        assert s3.put_object.called

    def test_load_failure_continues_to_next_artifact(self):
        from evals.rationale_clustering import compute_and_emit

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        good_key = "decision_artifacts/2026/05/09/sector_quant/good.json"
        bad_key = "decision_artifacts/2026/05/09/sector_quant/bad.json"
        # Only register the good key in the stub; the bad key triggers KeyError.
        artifacts = {
            good_key: {
                "agent_id": "sector_quant",
                "agent_output": {
                    "top5_picks": [
                        {"ticker": f"T{i}", "quant_rationale": f"rationale {i}"}
                        for i in range(8)
                    ]
                },
            }
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        # Inject the bad key into the listing.
        s3.get_paginator.return_value.paginate.side_effect = lambda *, Bucket, Prefix: [
            {"Contents": [
                {"Key": good_key},
                {"Key": bad_key},
            ]}
        ] if Prefix == "decision_artifacts/2026/05/09/" else [{"Contents": []}]

        summary = compute_and_emit(
            end_time=end,
            window_days=1,
            s3_client=s3,
            cloudwatch_client=MagicMock(),
        )
        assert len(summary["load_failures"]) == 1
        assert summary["load_failures"][0]["key"] == bad_key
        # Good artifact still processed → 1 agent analyzed.
        assert summary["agents_analyzed"] == 1
