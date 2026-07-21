"""Tests for `graph.research_graph.load_scorecard_node` (Phase 2.A.3).

The node sits between `load_regime_substrate_node` and `macro_economist_node`
and produces the `prior_cycle_scorecard_text` state field. Graceful-
degrade contract: any failure path returns empty string so the cycle
proceeds without scorecard data (agents fall back to pre-Phase-2
behavior).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.research_graph import load_scorecard_node


class TestLoadScorecardNode:
    def test_returns_text_when_artifact_present(self, monkeypatch):
        # load_latest_scorecard_text returns the rendered string;
        # the node passes it into the state under the canonical key.
        monkeypatch.setenv("RESEARCH_BUCKET", "alpha-engine-research")
        with patch("evals.last_week_scorecard.load_latest_scorecard_text",
                   return_value="SCORECARD RENDERED TEXT"):
            result = load_scorecard_node(state={})
        assert result == {"prior_cycle_scorecard_text": "SCORECARD RENDERED TEXT"}

    def test_returns_empty_string_when_artifact_missing(self, monkeypatch):
        # Producer flag-off / first cycle of soak / artifact deleted —
        # load_latest_scorecard_text returns "" (its own graceful path).
        monkeypatch.setenv("RESEARCH_BUCKET", "alpha-engine-research")
        with patch("evals.last_week_scorecard.load_latest_scorecard_text",
                   return_value=""):
            result = load_scorecard_node(state={})
        assert result == {"prior_cycle_scorecard_text": ""}

    def test_boto3_construction_failure_is_graceful(self, monkeypatch):
        # If boto3.client itself raises (no credentials in env, etc),
        # the node should NOT propagate — empty string is the contract.
        monkeypatch.setenv("RESEARCH_BUCKET", "alpha-engine-research")
        with patch("boto3.client", side_effect=RuntimeError("no creds")):
            result = load_scorecard_node(state={})
        assert result == {"prior_cycle_scorecard_text": ""}

    def test_load_function_exception_is_graceful(self, monkeypatch):
        # If load_latest_scorecard_text itself raises (corrupt artifact
        # that escapes its own try/except, schema drift exposed at the
        # consumer boundary, etc.), node must still return ""
        monkeypatch.setenv("RESEARCH_BUCKET", "alpha-engine-research")
        with patch("evals.last_week_scorecard.load_latest_scorecard_text",
                   side_effect=Exception("schema drift")):
            result = load_scorecard_node(state={})
        assert result == {"prior_cycle_scorecard_text": ""}

    def test_uses_research_bucket_env_var(self, monkeypatch):
        # The node resolves the bucket via the RESEARCH_BUCKET env var
        # so it works in both Lambda (env carries the production bucket)
        # and local-dev (defaults to alpha-engine-research).
        monkeypatch.setenv("RESEARCH_BUCKET", "custom-bucket")
        captured: dict = {}

        def _capturing_load(*, s3_client, bucket, prefix=None):
            captured["bucket"] = bucket
            return "ok"

        with patch("evals.last_week_scorecard.load_latest_scorecard_text",
                   side_effect=_capturing_load):
            load_scorecard_node(state={})
        assert captured["bucket"] == "custom-bucket"

    def test_default_bucket_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_BUCKET", raising=False)
        captured: dict = {}

        def _capturing_load(*, s3_client, bucket, prefix=None):
            captured["bucket"] = bucket
            return "ok"

        with patch("evals.last_week_scorecard.load_latest_scorecard_text",
                   side_effect=_capturing_load):
            load_scorecard_node(state={})
        assert captured["bucket"] == "alpha-engine-research"


class TestGraphWiring:
    """Verifies the node + state field + edge wiring all land together."""

    def test_state_carries_prior_cycle_scorecard_text_field(self):
        from graph.research_graph import ResearchState
        # TypedDict annotations expose the field set via __annotations__.
        assert "prior_cycle_scorecard_text" in ResearchState.__annotations__

    def test_build_graph_registers_load_scorecard_node(self):
        # build_graph wires the node and the edge — verify the node
        # name is in the compiled graph's node set.
        from graph.research_graph import build_graph
        graph = build_graph()
        # LangGraph compiled graphs expose nodes via .nodes attribute.
        assert "load_scorecard_node" in graph.nodes
