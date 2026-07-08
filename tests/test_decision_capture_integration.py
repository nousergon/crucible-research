"""
Integration tests for decision-artifact capture in the research graph.

Verifies:
- Feature flag default-off: capture functions are no-ops without env var.
- Feature flag on + mocked S3: each producer node writes one artifact at
  the canonical S3 key.
- Hard-fail behavior: S3 unavailability raises through the node, not
  swallowed silently (per ``feedback_no_silent_fails``).
- Per-node payload helpers produce JSON-serializable dicts that survive
  round-trip through ``DecisionArtifact.model_validate``.

Capture is gated on ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED``; default-off
preserves existing behavior. Production turns it on once IAM grant for
``s3:PutObject`` on ``decision_artifacts/*`` is in place on the
research-runner Lambda role.

Workstream design: ``alpha-engine-docs/private/alpha-engine-research-typed-
state-capture-260429.md``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from nousergon_lib.decision_capture import (
    DecisionArtifact,
    DecisionCaptureWriteError,
    capture_decision,
    FullPromptContext,
    ModelMetadata,
)


# ── Feature-flag gate ─────────────────────────────────────────────────────


class TestFeatureFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is False

    def test_true_value_enables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is True

    def test_one_value_enables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "1")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is True

    def test_false_value_disables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "false")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is False

    def test_arbitrary_value_disables(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "yolo")
        from graph.decision_capture_helpers import is_decision_capture_enabled
        assert is_decision_capture_enabled() is False


# ── Per-node payload builders ─────────────────────────────────────────────


@pytest.fixture
def fake_ctx():
    from agents.sector_teams.sector_team import SectorTeamContext
    return SectorTeamContext(
        scanner_universe=["AAPL", "MSFT", "JPM", "JNJ", "XOM"],
        agent_input_set=["AAPL", "MSFT", "JPM", "JNJ", "XOM"],
        sector_map={
            "AAPL": "Technology", "MSFT": "Technology",
            "JPM": "Financial", "JNJ": "Healthcare", "XOM": "Energy",
        },
        price_data={},
        technical_scores={
            "AAPL": {"rsi_14": 55, "technical_score": 70},
            "MSFT": {"rsi_14": 50, "technical_score": 65},
        },
        market_regime="neutral",
        prior_theses={"AAPL": {"final_score": 65, "rating": "BUY"}},
        held_tickers=["AAPL"],
        news_data_by_ticker={
            "AAPL": {"articles": [{"headline": "x"}]},
            "MSFT": {"articles": []},
        },
        analyst_data_by_ticker={"AAPL": {"consensus_rating": "Buy"}},
        insider_data_by_ticker={},
        prior_sector_ratings={},
        current_sector_ratings={"Technology": {"rating": "overweight"}},
        run_date="2026-04-29",
        episodic_memories={},
        semantic_memories={},
    )


class TestSectorQuantPayloadBuilder:
    def test_payload_is_json_serializable(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_quant_capture_payload
        snapshot, summary = build_sector_quant_capture_payload(
            "technology", fake_ctx, team_tickers=["AAPL", "MSFT"],
        )
        json.dumps(snapshot)
        assert isinstance(summary, str)
        assert "team_id=technology" in summary

    def test_payload_includes_required_fields(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_quant_capture_payload
        snapshot, _ = build_sector_quant_capture_payload(
            "technology", fake_ctx, team_tickers=["AAPL", "MSFT"],
        )
        for key in (
            "team_id", "run_date", "market_regime",
            "scanner_universe_size", "sector_tickers",
            "sector_tickers_count", "technical_scores_team",
        ):
            assert key in snapshot, f"missing field: {key}"

    def test_excludes_qual_tier_inputs(self, fake_ctx):
        # Quant doesn't see news/analyst/insider data — those are qual-tier.
        # Capturing them would over-state quant's inputs and pollute eval.
        from graph.decision_capture_helpers import build_sector_quant_capture_payload
        snapshot, _ = build_sector_quant_capture_payload(
            "technology", fake_ctx, team_tickers=["AAPL", "MSFT"],
        )
        assert "news_data_by_ticker" not in snapshot
        assert "analyst_data_by_ticker" not in snapshot
        assert "insider_data_by_ticker" not in snapshot
        assert "prior_theses_in_team" not in snapshot

    def test_technical_scores_filtered_to_team(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_quant_capture_payload
        snapshot, _ = build_sector_quant_capture_payload(
            "technology", fake_ctx, team_tickers=["AAPL", "MSFT"],
        )
        assert set(snapshot["technical_scores_team"].keys()) == {"AAPL", "MSFT"}


class TestSectorQualPayloadBuilder:
    def test_payload_is_json_serializable(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_qual_capture_payload
        snapshot, summary = build_sector_qual_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL", "score": 70}, {"ticker": "MSFT", "score": 65}],
        )
        json.dumps(snapshot)
        assert isinstance(summary, str)
        assert "team_id=technology" in summary
        assert "top5=2" in summary

    def test_payload_includes_required_fields(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_qual_capture_payload
        snapshot, _ = build_sector_qual_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL", "score": 70}, {"ticker": "MSFT", "score": 65}],
        )
        for key in (
            "team_id", "run_date", "market_regime",
            "quant_top5", "quant_top5_tickers", "held_in_top5",
            "prior_theses_for_top5", "news_data_for_top5",
            "analyst_data_for_top5", "insider_data_for_top5",
            "prior_sector_ratings", "current_sector_ratings",
            "memories_summary",
        ):
            assert key in snapshot, f"missing field: {key}"

    def test_inputs_scoped_to_top5(self, fake_ctx):
        # Qual only reviews top5 — capturing news/analyst data outside that
        # set would over-state inputs.
        from graph.decision_capture_helpers import build_sector_qual_capture_payload
        snapshot, _ = build_sector_qual_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL", "score": 70}],
        )
        assert snapshot["quant_top5_tickers"] == ["AAPL"]
        assert set(snapshot["news_data_for_top5"].keys()) == {"AAPL"}
        assert set(snapshot["analyst_data_for_top5"].keys()) == {"AAPL"}
        assert snapshot["held_in_top5"] == ["AAPL"]
        assert "AAPL" in snapshot["prior_theses_for_top5"]

    def test_skips_picks_without_ticker(self, fake_ctx):
        # quant LLM output parsing can drop the ticker key — match
        # run_sector_team's filter at sector_team.py:96-109.
        from graph.decision_capture_helpers import build_sector_qual_capture_payload
        snapshot, _ = build_sector_qual_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL"}, {"score": 50}, {"ticker": "MSFT"}],
        )
        assert snapshot["quant_top5_tickers"] == ["AAPL", "MSFT"]


class TestSectorPeerReviewPayloadBuilder:
    def test_payload_is_json_serializable(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_peer_review_capture_payload
        snapshot, summary = build_sector_peer_review_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL", "score": 70}, {"ticker": "MSFT", "score": 65}],
            qual_assessments=[{"ticker": "AAPL", "qual_score": 72}],
            qual_additional_candidate={"ticker": "NVDA", "score": 68},
        )
        json.dumps(snapshot)
        assert isinstance(summary, str)
        assert "team_id=technology" in summary
        assert "addition=yes" in summary

    def test_no_addition_summary(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_peer_review_capture_payload
        snapshot, summary = build_sector_peer_review_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL", "score": 70}],
            qual_assessments=[{"ticker": "AAPL", "qual_score": 72}],
            qual_additional_candidate=None,
        )
        assert snapshot["qual_additional_candidate"] is None
        assert "addition=no" in summary

    def test_review_set_includes_addition(self, fake_ctx):
        # When qual adds a candidate, technical_scores_review_set must
        # include the addition's ticker (peer review reviews the
        # addition's quant case).
        from graph.decision_capture_helpers import build_sector_peer_review_capture_payload
        # Add MSFT to technical_scores; AAPL already there from fixture
        fake_ctx.technical_scores["NVDA"] = {"technical_score": 75}
        snapshot, _ = build_sector_peer_review_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL"}],
            qual_assessments=[],
            qual_additional_candidate={"ticker": "NVDA"},
        )
        assert "AAPL" in snapshot["technical_scores_review_set"]
        assert "NVDA" in snapshot["technical_scores_review_set"]

    def test_payload_includes_required_fields(self, fake_ctx):
        from graph.decision_capture_helpers import build_sector_peer_review_capture_payload
        snapshot, _ = build_sector_peer_review_capture_payload(
            "technology", fake_ctx,
            quant_top5=[{"ticker": "AAPL"}],
            qual_assessments=[],
            qual_additional_candidate=None,
        )
        for key in (
            "team_id", "run_date", "market_regime",
            "quant_top5", "qual_assessments", "qual_additional_candidate",
            "technical_scores_review_set",
        ):
            assert key in snapshot, f"missing field: {key}"


class TestThesisUpdatePayloadBuilder:
    def test_payload_is_json_serializable(self, fake_ctx):
        from graph.decision_capture_helpers import build_thesis_update_capture_payload
        snapshot, summary = build_thesis_update_capture_payload(
            "technology", "AAPL", fake_ctx,
            triggers=["earnings_beat", "analyst_upgrade"],
        )
        json.dumps(snapshot)
        assert isinstance(summary, str)
        assert "ticker=AAPL" in summary
        assert "triggers=2" in summary

    def test_pulls_per_ticker_inputs(self, fake_ctx):
        # Thesis-update prompt sees prior_thesis + news + analyst data
        # for the held ticker; verify the snapshot mirrors that.
        from graph.decision_capture_helpers import build_thesis_update_capture_payload
        snapshot, _ = build_thesis_update_capture_payload(
            "technology", "AAPL", fake_ctx,
            triggers=["news_event"],
        )
        assert snapshot["ticker"] == "AAPL"
        assert snapshot["prior_thesis"] == {"final_score": 65, "rating": "BUY"}
        assert snapshot["news_data"] == {"articles": [{"headline": "x"}]}
        assert snapshot["analyst_data"] == {"consensus_rating": "Buy"}

    def test_missing_per_ticker_inputs(self, fake_ctx):
        # Held ticker may have no news/analyst data — capture should
        # carry None rather than crashing.
        from graph.decision_capture_helpers import build_thesis_update_capture_payload
        snapshot, summary = build_thesis_update_capture_payload(
            "technology", "ZZZZ", fake_ctx,
            triggers=["sector_regime_change"],
        )
        assert snapshot["prior_thesis"] is None
        assert snapshot["news_data"] is None
        assert snapshot["analyst_data"] is None
        assert "news_articles=0" in summary


class TestMacroEconomistPayloadBuilder:
    def test_minimal_state(self):
        from graph.decision_capture_helpers import build_macro_economist_capture_payload
        state = {"run_date": "2026-04-29", "macro_data": {"vix": 14.2}}
        snapshot, summary = build_macro_economist_capture_payload(state)
        json.dumps(snapshot)
        assert snapshot["macro_data"] == {"vix": 14.2}
        assert "run_date=2026-04-29" in summary

    def test_with_prior_report(self):
        from graph.decision_capture_helpers import build_macro_economist_capture_payload
        state = {
            "run_date": "2026-04-29",
            "macro_data": {"vix": 14.2, "tnx": 4.31},
            "prior_macro_report": "x" * 500,
            "prior_macro_snapshots": [{"date": "2026-04-22"}],
        }
        snapshot, summary = build_macro_economist_capture_payload(state)
        assert snapshot["prior_date"] == "2026-04-22"
        assert snapshot["prior_snapshots_count"] == 1
        assert "prior_report_chars=500" in summary

    def test_includes_regime_substrate_when_present(self):
        """Stage C.2 T3 pin — the judge's regime_decision_process
        rubric dimension scores the macro agent's regime call against
        the substrate. The substrate must land in agent_input so the
        judge can see it; if it's dropped here, the rubric becomes
        unscoreable for that dimension."""
        from graph.decision_capture_helpers import build_macro_economist_capture_payload
        substrate = {
            "run_id": "2605170230",
            "hmm": {"argmax": "bear", "probs": {"bear": 0.7, "neutral": 0.2, "bull": 0.1}},
            "composite": {"intensity_z": -1.8},
            "bocpd": {"change_signal": True},
        }
        state = {
            "run_date": "2026-05-17",
            "macro_data": {"vix": 28.0},
            "regime_substrate": substrate,
        }
        snapshot, summary = build_macro_economist_capture_payload(state)
        assert snapshot["regime_substrate"] == substrate
        # Summary surfaces a compact substrate marker for log + operator UX
        assert "regime_substrate=present" in summary
        assert "argmax=bear" in summary
        assert "intensity_z=-1.80" in summary

    def test_regime_substrate_none_marked_absent_in_summary(self):
        """When the upstream substrate Lambda hasn't published yet
        (pre-deploy state) or the non-blocking SF Catch tripped, state
        has regime_substrate=None. The capture snapshot still includes
        the field (judge sees None and skips the rubric dimension);
        summary marks the absence for operator clarity."""
        from graph.decision_capture_helpers import build_macro_economist_capture_payload
        state = {
            "run_date": "2026-05-17",
            "macro_data": {"vix": 14.2},
            "regime_substrate": None,
        }
        snapshot, summary = build_macro_economist_capture_payload(state)
        assert snapshot["regime_substrate"] is None
        assert "regime_substrate=absent" in summary

    def test_regime_substrate_missing_key_treated_as_absent(self):
        """Defensive — state with no regime_substrate key at all (older
        graph code paths, tests) gets the same absent treatment as
        explicit None."""
        from graph.decision_capture_helpers import build_macro_economist_capture_payload
        state = {"run_date": "2026-05-17", "macro_data": {"vix": 14.2}}
        snapshot, summary = build_macro_economist_capture_payload(state)
        assert snapshot["regime_substrate"] is None
        assert "regime_substrate=absent" in summary


class TestCIOPayloadBuilder:
    def test_minimal(self):
        from graph.decision_capture_helpers import build_cio_capture_payload
        state = {
            "run_date": "2026-04-29",
            "market_regime": "neutral",
            "macro_report": "...",
            "sector_ratings": {},
            "remaining_population": [],
            "open_slots": 5,
            "exits": [],
        }
        candidates = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        prior_ic = []
        snapshot, summary = build_cio_capture_payload(
            state, candidates=candidates, prior_ic=prior_ic,
        )
        json.dumps(snapshot)
        assert snapshot["candidates_count"] == 2
        assert snapshot["open_slots"] == 5
        assert "candidates=2" in summary


# ── End-to-end: feature flag + capture writes artifact ────────────────────


@pytest.fixture
def mocked_s3(monkeypatch):
    """moto-mocked S3 + ``alpha-engine-research`` bucket pre-created.

    Patches ``boto3.client`` so any code path that calls
    ``boto3.client("s3")`` (including capture_decision under the hood)
    gets the mocked client.
    """
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="alpha-engine-research")

        # Also patch boto3.client globally for any caller using the
        # default-client path. capture_decision accepts s3_client= so
        # we typically test via that injection path.
        yield client


class TestCaptureFiresWhenEnabled:
    def test_capture_writes_artifact_via_helper(self, mocked_s3, monkeypatch):
        # Direct test of capture_decision with the helper-built payload.
        # Verifies the integration produces a valid DecisionArtifact end-to-end.
        from graph.decision_capture_helpers import build_macro_economist_capture_payload

        state = {
            "run_date": "2026-04-29",
            "macro_data": {"vix": 14.2},
            "prior_macro_report": "",
            "prior_macro_snapshots": [],
        }
        snapshot, summary = build_macro_economist_capture_payload(state)

        s3_key = capture_decision(
            run_id="test-run-001",
            agent_id="macro_economist",
            model_metadata=ModelMetadata(model_name="claude-sonnet-4-6"),
            full_prompt_context=FullPromptContext(
                system_prompt="<placeholder>", user_prompt="<placeholder>",
            ),
            input_data_snapshot=snapshot,
            input_data_summary=summary,
            agent_output={"macro_report": "test", "market_regime": "neutral"},
            s3_client=mocked_s3,
            timestamp=datetime(2026, 4, 29, 22, 30, tzinfo=timezone.utc),
        )

        assert s3_key == "decision_artifacts/2026/04/29/macro_economist/test-run-001.json"
        obj = mocked_s3.get_object(Bucket="alpha-engine-research", Key=s3_key)
        artifact = DecisionArtifact.model_validate(json.loads(obj["Body"].read()))
        assert artifact.agent_id == "macro_economist"
        assert artifact.model_metadata.model_name == "claude-sonnet-4-6"
        assert artifact.input_data_summary == summary


class TestCaptureNoOpWhenDisabled:
    def test_capture_if_enabled_skips_when_flag_off(self, monkeypatch, mocked_s3):
        """When the flag is off, ``_capture_if_enabled`` short-circuits
        before any S3 call. Test by invoking it with an invalid bucket
        — it must NOT raise (no S3 attempt at all)."""
        monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)

        from graph.research_graph import _capture_if_enabled

        # No exception even though the bucket doesn't exist — capture
        # function must short-circuit on the env-var check.
        _capture_if_enabled(
            state={"run_date": "2026-04-29"},
            agent_id="sector_quant:technology",
            model_name_key="sector_team",
            input_data_snapshot={"x": 1},
            input_data_summary="test",
            agent_output={"ranked_picks": []},
        )

        # Verify nothing was written to S3
        objects = mocked_s3.list_objects_v2(Bucket="alpha-engine-research")
        assert "Contents" not in objects or not objects["Contents"]


class TestCaptureHardFailsOnS3Error:
    def test_capture_if_enabled_raises_on_s3_error(self, monkeypatch):
        """When the flag is on AND S3 unreachable, ``_capture_if_enabled``
        re-raises ``DecisionCaptureWriteError`` per ``feedback_no_silent_fails``.

        Mocks ``boto3.client`` to return a stub whose ``put_object`` raises
        a ``ClientError`` shaped like NoSuchBucket. Avoids moto-vs-botocore
        version brittleness — moto 5.x's S3 response serialization fails
        on newer botocore versions (``Unsupported protocol [rest-json] for
        service s3``), which surfaced when this test ran in CI on
        botocore 1.43+.
        """
        from unittest.mock import patch
        from botocore.exceptions import ClientError

        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")

        fake_s3 = MagicMock()
        fake_s3.put_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"}},
            "PutObject",
        )

        # capture_decision (in alpha_engine_lib.decision_capture) calls
        # boto3.client("s3") at write time. Patch the lib-side import so
        # the stub takes effect across this whole call.
        with patch("nousergon_lib.decision_capture.boto3.client", return_value=fake_s3):
            from graph.research_graph import _capture_if_enabled

            with pytest.raises(DecisionCaptureWriteError):
                _capture_if_enabled(
                    state={"run_date": "2026-04-29"},
                    agent_id="sector_quant:technology",
                    model_name_key="sector_team",
                    input_data_snapshot={"x": 1},
                    input_data_summary="test",
                    agent_output={"ranked_picks": []},
                )


# ── Provenance stamps: data_snapshot_id + code_sha threading (1b / #781) ──


class TestProvenanceStampThreading:
    """``_capture_if_enabled`` must thread the run-level ``data_snapshot_id``
    (from state, surfaced by the price fetcher) and ``code_sha`` (from the
    ``ALPHA_ENGINE_CODE_SHA`` deploy env var) into ``capture_decision`` so
    every captured DecisionArtifact records the immutable data snapshot +
    code revision it was computed on (L4567 1b)."""

    def _invoke_capturing_kwargs(self, monkeypatch, state, env=None):
        """Run ``_capture_if_enabled`` with capture enabled and
        ``capture_decision`` stubbed; return the kwargs it was called with."""
        from unittest.mock import patch

        monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
        monkeypatch.delenv("ALPHA_ENGINE_CODE_SHA", raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)

        import graph.research_graph as rg

        captured = {}

        def _fake_capture(**kwargs):
            captured.update(kwargs)
            return "s3://fake/key"

        with patch.object(rg, "capture_decision", _fake_capture):
            rg._capture_if_enabled(
                state=state,
                agent_id="sector_quant:technology",
                model_name_key="sector_team",
                input_data_snapshot={"x": 1},
                input_data_summary="test",
                agent_output={"ranked_picks": []},
            )
        return captured

    def test_data_snapshot_id_threaded_from_state(self, monkeypatch):
        captured = self._invoke_capturing_kwargs(
            monkeypatch,
            {"run_date": "2026-04-29", "data_snapshot_id": "7"},
        )
        assert captured["data_snapshot_id"] == "7"

    def test_code_sha_threaded_from_env(self, monkeypatch):
        captured = self._invoke_capturing_kwargs(
            monkeypatch,
            {"run_date": "2026-04-29", "data_snapshot_id": "7"},
            env={"ALPHA_ENGINE_CODE_SHA": "abc123def"},
        )
        assert captured["code_sha"] == "abc123def"

    def test_missing_data_snapshot_id_records_unknown(self, monkeypatch):
        # State without the stamp (e.g. resume path skipping fetch_data) →
        # "unknown" sentinel, no crash, artifact still written.
        captured = self._invoke_capturing_kwargs(
            monkeypatch, {"run_date": "2026-04-29"},
        )
        assert captured["data_snapshot_id"] == "unknown"

    def test_missing_code_sha_is_none(self, monkeypatch):
        # No deploy env var (local/dev) → code_sha None, artifact still lands.
        captured = self._invoke_capturing_kwargs(
            monkeypatch, {"run_date": "2026-04-29", "data_snapshot_id": "7"},
        )
        assert captured["code_sha"] is None

    def test_unknown_string_in_state_passed_through(self, monkeypatch):
        captured = self._invoke_capturing_kwargs(
            monkeypatch,
            {"run_date": "2026-04-29", "data_snapshot_id": "unknown"},
        )
        assert captured["data_snapshot_id"] == "unknown"


# ── Run-id derivation ─────────────────────────────────────────────────────


class TestDeriveRunId:
    def test_explicit_run_id_used(self):
        from graph.decision_capture_helpers import derive_run_id
        assert derive_run_id({"run_id": "explicit-123", "run_date": "2026-04-29"}) == "explicit-123"

    def test_falls_back_to_run_date(self):
        from graph.decision_capture_helpers import derive_run_id
        assert derive_run_id({"run_date": "2026-04-29"}) == "2026-04-29"

    def test_unknown_when_neither(self):
        from graph.decision_capture_helpers import derive_run_id
        assert derive_run_id({}) == "unknown"
