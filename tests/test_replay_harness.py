"""
Tests for the scenario-replay harness (``scripts/replay_harness.py``,
L4567 sub-item 2b / alpha-engine-config#781).

Exercises the harness's OWN mechanics (artifact fetch, diffing,
perturbation application, distributional summarization, dispatch) against
a FAKE deterministic node function — independent of the real agent's real
behavior (which needs a live Anthropic call and is out of scope for a unit
test). ``tests/test_price_fetcher_snapshot_id.py`` is the sibling pattern
for ArcticDB mocking; ``tests/test_decision_capture_integration.py`` is the
sibling pattern for moto-mocked S3.
"""

from __future__ import annotations

import json
from datetime import timedelta

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from scripts.replay_harness import (
    ArtifactNotFoundError,
    FieldDiff,
    PerturbationSpec,
    ReplayNotSupportedError,
    coin_loss_floor_perturbation,
    coin_risk_override_perturbation,
    counterfactual_replay,
    diff_outputs,
    faithful_replay,
    fetch_artifact,
    rehydrate_price_data,
    run_coin_scenario,
)

BUCKET = "alpha-engine-research"


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mocked_s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _thesis_update_artifact(
    *,
    agent_id: str = "thesis_update:financials:COIN",
    run_id: str = "2026-06-05",
    ticker: str = "COIN",
    team_id: str = "financials",
    triggers: list[str] | None = None,
    prior_thesis: dict | None = None,
    news_data: dict | None = None,
    analyst_data: dict | None = None,
    agent_output: dict | None = None,
    data_snapshot_id: str | None = "42",
    code_sha: str | None = "abc123",
) -> dict:
    """Build a DecisionArtifact-shaped dict matching
    ``build_thesis_update_capture_payload``'s snapshot shape."""
    return {
        "schema_version": 2,
        "run_id": run_id,
        "timestamp": "2026-06-06T12:00:00+00:00",
        "agent_id": agent_id,
        "model_metadata": {"model_name": "claude-haiku-4-5"},
        "full_prompt_context": {"system_prompt": "sys", "user_prompt": "usr"},
        "input_data_snapshot": {
            "team_id": team_id,
            "ticker": ticker,
            "run_date": "2026-06-05",
            "triggers": triggers if triggers is not None else ["price_move_gt_2atr"],
            "prior_thesis": prior_thesis if prior_thesis is not None else {
                "bull_case": "Exchange volume recovering.",
                "conviction": 55,
            },
            "news_data": news_data,
            "analyst_data": analyst_data,
        },
        "input_data_summary": "team_id=financials, ticker=COIN",
        "input_data_truncated_at": None,
        "agent_output": agent_output if agent_output is not None else {
            "bull_case": "Exchange volume recovering.",
            "bear_case": "Regulatory overhang.",
            "catalysts": ["Q2 earnings"],
            "risks": ["SEC action"],
            "conviction": 55,
            "conviction_rationale": "Stable volumes.",
            "thesis_summary": "Hold through volatility.",
            "triggers_response": "Price move within thesis tolerance.",
            "last_updated": "2026-06-05",
            "stale_days": 0,
        },
        "code_sha": code_sha,
        "data_snapshot_id": data_snapshot_id,
    }


def _put_artifact(client, artifact: dict, *, date: str = "2026/06/06"):
    agent_id = artifact["agent_id"]
    run_id = artifact["run_id"]
    key = f"decision_artifacts/{date}/{agent_id}/{run_id}.json"
    client.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(artifact).encode())
    return key


# ── fetch_artifact ──────────────────────────────────────────────────────


class TestFetchArtifact:
    def test_fetch_by_explicit_run_id(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)

        fetched = fetch_artifact(
            "thesis_update:financials:COIN", "2026-06-06",
            run_id="2026-06-05", s3_client=mocked_s3,
        )
        assert fetched["agent_id"] == "thesis_update:financials:COIN"
        assert fetched["data_snapshot_id"] == "42"

    def test_fetch_lists_and_takes_newest_when_no_run_id(self, mocked_s3):
        older = _thesis_update_artifact(run_id="2026-05-01")
        newer = _thesis_update_artifact(run_id="2026-06-05")
        _put_artifact(mocked_s3, older)
        _put_artifact(mocked_s3, newer)

        # moto's put_object LastModified has ~1s granularity, so two puts in
        # the same test tie in practice — real S3 doesn't have this problem
        # (distinct sub-second timestamps), so fake the LastModified spread
        # here rather than relying on wall-clock ordering to prove the
        # "take newest" selection logic specifically (list_objects_v2 itself
        # returns keys in lexicographic, not chronological, order).
        real_list = mocked_s3.list_objects_v2

        def _list_with_distinct_timestamps(**kwargs):
            resp = real_list(**kwargs)
            for obj in resp.get("Contents", []):
                bump = timedelta(seconds=1) if "2026-06-05" in obj["Key"] else timedelta(0)
                obj["LastModified"] = obj["LastModified"] + bump
            return resp

        mocked_s3.list_objects_v2 = _list_with_distinct_timestamps

        fetched = fetch_artifact(
            "thesis_update:financials:COIN", "2026-06-06", s3_client=mocked_s3,
        )
        assert fetched["run_id"] == "2026-06-05"

    def test_missing_artifact_raises_not_found(self, mocked_s3):
        with pytest.raises(ArtifactNotFoundError):
            fetch_artifact(
                "thesis_update:financials:COIN", "2026-06-06", s3_client=mocked_s3,
            )

    def test_missing_explicit_run_id_raises_not_found(self, mocked_s3):
        _put_artifact(mocked_s3, _thesis_update_artifact())
        with pytest.raises(ArtifactNotFoundError):
            fetch_artifact(
                "thesis_update:financials:COIN", "2026-06-06",
                run_id="nonexistent", s3_client=mocked_s3,
            )


# ── diff_outputs ────────────────────────────────────────────────────────


class TestDiffOutputs:
    def test_identical_outputs_produce_no_diffs(self):
        a = {"conviction": 55, "bull_case": "x"}
        b = {"conviction": 55, "bull_case": "x"}
        assert diff_outputs(a, b) == []

    def test_changed_field_is_reported(self):
        recorded = {"conviction": 55, "bull_case": "x"}
        replayed = {"conviction": 70, "bull_case": "x"}
        diffs = diff_outputs(recorded, replayed)
        assert len(diffs) == 1
        assert diffs[0] == FieldDiff(field="conviction", recorded=55, replayed=70)

    def test_added_and_removed_fields_are_reported(self):
        recorded = {"a": 1, "b": 2}
        replayed = {"a": 1, "c": 3}
        diffs = {d.field: d for d in diff_outputs(recorded, replayed)}
        assert set(diffs) == {"b", "c"}
        assert diffs["b"].recorded == 2 and diffs["b"].replayed is None
        assert diffs["c"].recorded is None and diffs["c"].replayed == 3

    def test_diff_surfaces_what_changed_not_just_bool(self):
        recorded = {"conviction": 55}
        replayed = {"conviction": 90}
        diffs = diff_outputs(recorded, replayed)
        d = diffs[0].to_dict()
        assert d == {"field": "conviction", "recorded": 55, "replayed": 90}


# ── faithful_replay (fake deterministic node) ────────────────────────────


def _echo_node_factory(output: dict):
    """A fake node_fn that ignores the snapshot and always returns
    ``output`` — the "faithful, nothing changed" case."""
    def _node(snapshot, recorded_output, *, temperature=None):
        return dict(output)
    return _node


def _snapshot_sensitive_node(snapshot, recorded_output, *, temperature=None):
    """A fake node_fn whose output DEPENDS on the snapshot's triggers list —
    used to prove the harness's diffing/perturbation mechanics are wired
    correctly (not testing the real LLM agent)."""
    triggers = snapshot.get("triggers") or []
    base_conviction = 55
    if "loss_floor_breach" in triggers:
        conviction = 10  # a fix-aware agent would sharply cut conviction
    else:
        conviction = base_conviction
    return {
        "bull_case": "Exchange volume recovering.",
        "bear_case": "Regulatory overhang.",
        "catalysts": ["Q2 earnings"],
        "risks": ["SEC action"],
        "conviction": conviction,
        "conviction_rationale": "Stable volumes.",
        "thesis_summary": "Hold through volatility.",
        "triggers_response": ",".join(triggers),
        "last_updated": "2026-06-05",
        "stale_days": 0,
    }


class TestFaithfulReplay:
    def test_identical_replay_has_no_diffs_and_matches(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)
        node = _echo_node_factory(artifact["agent_output"])

        result = faithful_replay(
            "thesis_update:financials:COIN", "2026-06-06",
            run_id="2026-06-05", s3_client=mocked_s3, node_fn=node,
        )
        assert result.matches
        assert result.diffs == []
        assert result.data_snapshot_id == "42"
        assert result.pinned is True
        assert result.code_sha == "abc123"

    def test_diverged_replay_reports_field_diffs(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)

        def _node(snapshot, recorded_output, *, temperature=None):
            out = dict(artifact["agent_output"])
            out["conviction"] = 20
            return out

        result = faithful_replay(
            "thesis_update:financials:COIN", "2026-06-06",
            run_id="2026-06-05", s3_client=mocked_s3, node_fn=_node,
        )
        assert not result.matches
        assert len(result.diffs) == 1
        assert result.diffs[0].field == "conviction"
        assert result.diffs[0].recorded == 55
        assert result.diffs[0].replayed == 20

    def test_unpinned_when_data_snapshot_id_unknown(self, mocked_s3):
        artifact = _thesis_update_artifact(data_snapshot_id="unknown")
        _put_artifact(mocked_s3, artifact)
        node = _echo_node_factory(artifact["agent_output"])

        result = faithful_replay(
            "thesis_update:financials:COIN", "2026-06-06",
            run_id="2026-06-05", s3_client=mocked_s3, node_fn=node,
        )
        assert result.pinned is False

    def test_unpinned_when_data_snapshot_id_absent(self, mocked_s3):
        artifact = _thesis_update_artifact(data_snapshot_id=None)
        _put_artifact(mocked_s3, artifact)
        node = _echo_node_factory(artifact["agent_output"])

        result = faithful_replay(
            "thesis_update:financials:COIN", "2026-06-06",
            run_id="2026-06-05", s3_client=mocked_s3, node_fn=node,
        )
        assert result.pinned is False

    def test_unsupported_agent_id_raises(self, mocked_s3):
        artifact = _thesis_update_artifact(agent_id="sector_quant:financials")
        _put_artifact(mocked_s3, artifact)

        with pytest.raises(ReplayNotSupportedError):
            faithful_replay(
                "sector_quant:financials", "2026-06-06",
                run_id="2026-06-05", s3_client=mocked_s3,
            )

    def test_missing_artifact_raises(self, mocked_s3):
        with pytest.raises(ArtifactNotFoundError):
            faithful_replay(
                "thesis_update:financials:COIN", "2026-06-06",
                run_id="nope", s3_client=mocked_s3,
            )


# ── PerturbationSpec ────────────────────────────────────────────────────


class TestPerturbationSpec:
    def test_apply_does_not_mutate_original_snapshot(self):
        snapshot = {"triggers": ["price_move_gt_2atr"]}
        spec = PerturbationSpec(
            name="add_trigger",
            description="append a trigger",
            field_path="triggers",
            mutate=lambda cur: (cur or []) + ["news_volume_spike"],
        )
        mutated = spec.apply(snapshot)
        assert snapshot["triggers"] == ["price_move_gt_2atr"]  # unchanged
        assert mutated["triggers"] == ["price_move_gt_2atr", "news_volume_spike"]

    def test_apply_creates_missing_intermediate_dict(self):
        snapshot = {}
        spec = PerturbationSpec(
            name="set_flag",
            description="set a nested flag",
            field_path="prior_thesis.risk_override_active",
            mutate=lambda _cur: True,
        )
        mutated = spec.apply(snapshot)
        assert mutated["prior_thesis"]["risk_override_active"] is True

    def test_apply_on_existing_nested_dict_preserves_siblings(self):
        snapshot = {"prior_thesis": {"conviction": 55, "bull_case": "x"}}
        spec = PerturbationSpec(
            name="set_flag",
            description="set a nested flag",
            field_path="prior_thesis.risk_override_active",
            mutate=lambda _cur: True,
        )
        mutated = spec.apply(snapshot)
        assert mutated["prior_thesis"]["conviction"] == 55
        assert mutated["prior_thesis"]["bull_case"] == "x"
        assert mutated["prior_thesis"]["risk_override_active"] is True


# ── counterfactual_replay (distributional) ───────────────────────────────


class TestCounterfactualReplay:
    def test_n_replay_1_temp_0_is_a_single_point(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)
        spec = PerturbationSpec(
            name="noop", description="no-op mutation",
            field_path="triggers", mutate=lambda cur: cur or [],
        )
        dist = counterfactual_replay(
            "thesis_update:financials:COIN", "2026-06-06", spec,
            n_replay=1, temperature=0.0,
            s3_client=mocked_s3, node_fn=_snapshot_sensitive_node,
            run_id="2026-06-05",
        )
        assert dist.n_replay == 1
        assert len(dist.conviction_values) == 1

    def test_distributional_n_replay_nonzero_temp(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)
        spec = coin_loss_floor_perturbation()

        dist = counterfactual_replay(
            "thesis_update:financials:COIN", "2026-06-06", spec,
            n_replay=10, temperature=0.7,
            s3_client=mocked_s3, node_fn=_snapshot_sensitive_node,
            run_id="2026-06-05",
        )
        assert dist.n_replay == 10
        assert len(dist.conviction_values) == 10
        # The fake node deterministically drops conviction to 10 once the
        # loss_floor_breach trigger is present — proves the perturbation
        # actually reached the node (mechanics test, not an LLM test).
        assert all(c == 10 for c in dist.conviction_values)
        assert dist.conviction_mean == 10
        assert dist.conviction_delta_from_recorded == 10 - 55

    def test_perturbation_does_not_mutate_recorded_artifact(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)
        spec = coin_loss_floor_perturbation()

        dist = counterfactual_replay(
            "thesis_update:financials:COIN", "2026-06-06", spec,
            n_replay=3, temperature=0.5,
            s3_client=mocked_s3, node_fn=_snapshot_sensitive_node,
            run_id="2026-06-05",
        )
        # actual_recorded must be the UNMODIFIED artifact output.
        assert dist.actual_recorded["conviction"] == 55

    def test_field_value_counts_track_distribution_shape(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)

        calls = {"n": 0}

        def _flaky_node(snapshot, recorded_output, *, temperature=None):
            calls["n"] += 1
            conviction = 10 if calls["n"] % 2 == 0 else 20
            return {**recorded_output, "conviction": conviction}

        spec = PerturbationSpec(
            name="noop", description="", field_path="triggers",
            mutate=lambda cur: cur or [],
        )
        dist = counterfactual_replay(
            "thesis_update:financials:COIN", "2026-06-06", spec,
            n_replay=4, temperature=0.7,
            s3_client=mocked_s3, node_fn=_flaky_node,
            run_id="2026-06-05",
        )
        conviction_counts = dist.field_value_counts["conviction"]
        assert conviction_counts[json.dumps(10)] == 2
        assert conviction_counts[json.dumps(20)] == 2

    def test_invalid_n_replay_raises(self, mocked_s3):
        artifact = _thesis_update_artifact()
        _put_artifact(mocked_s3, artifact)
        spec = coin_loss_floor_perturbation()
        with pytest.raises(ValueError):
            counterfactual_replay(
                "thesis_update:financials:COIN", "2026-06-06", spec,
                n_replay=0, s3_client=mocked_s3, node_fn=_snapshot_sensitive_node,
                run_id="2026-06-05",
            )


# ── COIN acceptance scenario ─────────────────────────────────────────────


class TestCoinScenario:
    def test_coin_perturbation_specs_target_expected_fields(self):
        trigger_spec = coin_loss_floor_perturbation()
        override_spec = coin_risk_override_perturbation()
        assert trigger_spec.field_path == "triggers"
        assert override_spec.field_path == "prior_thesis.risk_override_active"
        assert "loss_floor_breach" in trigger_spec.mutate([])

    def test_run_coin_scenario_end_to_end_against_fake_node(self, mocked_s3):
        artifact = _thesis_update_artifact(
            triggers=["price_move_gt_2atr"],
            prior_thesis={"bull_case": "Exchange volume recovering.", "conviction": 55},
            agent_output={
                "bull_case": "Exchange volume recovering.",
                "conviction": 55,
                "thesis_summary": "Hold through volatility.",
                "triggers_response": "price_move_gt_2atr",
            },
        )
        _put_artifact(mocked_s3, artifact)

        dist = run_coin_scenario(
            "2026-06-06",
            n_replay=15, temperature=0.7,
            s3_client=mocked_s3, node_fn=_snapshot_sensitive_node,
            run_id="2026-06-05",
        )
        assert dist.n_replay == 15
        # snapshot-sensitive fake node cuts conviction once the injected
        # loss_floor_breach trigger is present — proves BOTH perturbations
        # (trigger append + risk_override flag) were applied to the same
        # rehydrated snapshot before the N replays fired.
        assert all(c == 10 for c in dist.conviction_values)
        assert dist.actual_recorded["conviction"] == 55
        assert dist.conviction_delta_from_recorded == 10 - 55

    def test_run_coin_scenario_missing_artifact_raises(self, mocked_s3):
        with pytest.raises(ArtifactNotFoundError):
            run_coin_scenario(
                "2026-06-06", s3_client=mocked_s3, node_fn=_snapshot_sensitive_node,
            )


# ── rehydrate_price_data (as-of pinning) ─────────────────────────────────


class _FakeVersionedItem:
    def __init__(self, data: pd.DataFrame, version):
        self.data = data
        self.version = version


class _FakeArcticLib:
    """Records the ``as_of`` kwarg it was called with, per ticker, so tests
    can assert the harness actually pins reads to the artifact's
    ``data_snapshot_id`` — mirrors the fake-library pattern in
    ``tests/test_price_fetcher_snapshot_id.py``."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame
        self.calls: list[dict] = []

    def read(self, ticker, date_range=None, columns=None, as_of=None):
        self.calls.append({"ticker": ticker, "as_of": as_of})
        return _FakeVersionedItem(self._frame, as_of or 0)


def _ohlcv_frame(rows: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0},
        index=idx,
    )


class TestRehydratePriceData:
    def test_pins_read_to_data_snapshot_id_version(self):
        lib = _FakeArcticLib(_ohlcv_frame())
        result = rehydrate_price_data(
            ["COIN"], "42", arctic_lib=lib,
        )
        assert "COIN" in result
        assert lib.calls == [{"ticker": "COIN", "as_of": 42}]

    def test_unknown_snapshot_id_reads_unpinned(self):
        lib = _FakeArcticLib(_ohlcv_frame())
        rehydrate_price_data(["COIN"], "unknown", arctic_lib=lib)
        assert lib.calls == [{"ticker": "COIN", "as_of": None}]

    def test_none_snapshot_id_reads_unpinned(self):
        lib = _FakeArcticLib(_ohlcv_frame())
        rehydrate_price_data(["COIN"], None, arctic_lib=lib)
        assert lib.calls == [{"ticker": "COIN", "as_of": None}]

    def test_non_numeric_snapshot_id_reads_unpinned(self):
        lib = _FakeArcticLib(_ohlcv_frame())
        rehydrate_price_data(["COIN"], "not-a-version", arctic_lib=lib)
        assert lib.calls == [{"ticker": "COIN", "as_of": None}]

    def test_missing_ticker_read_failure_is_dropped_not_raised(self):
        class _RaisingLib(_FakeArcticLib):
            def read(self, ticker, **kwargs):
                raise RuntimeError("no such symbol")

        lib = _RaisingLib(_ohlcv_frame())
        result = rehydrate_price_data(["MISSING"], "42", arctic_lib=lib)
        assert result == {}
