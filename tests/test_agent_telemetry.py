"""Tests for graph.agent_telemetry — per-agent CW runtime metrics.

Three contract layers:

1. ``emit_agent_completion`` emits exactly 4 datapoints (Invocations,
   Failures, DurationMs, LLMCallCount) per call with correct dimensions
   + values.
2. ``emit_agent_retry`` emits exactly 2 datapoints (RetryAttempts,
   RetrySuccesses) and pairs ``succeeded=False`` whenever
   ``attempted=False`` (no false-positive retry counts).
3. ``track_llm_cost`` finally block fires emission on both success and
   failure paths — Failures=1 only on the exception path.

CloudWatch is stubbed via a tiny fake whose ``put_metric_data`` records
calls. We never make real boto3 calls in tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from graph.agent_telemetry import (
    NAMESPACE,
    emit_agent_completion,
    emit_agent_retry,
)


class StubCloudWatch:
    """Records put_metric_data calls for assertions."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_metric_data(self, *, Namespace: str, MetricData: list) -> None:
        self.calls.append({"Namespace": Namespace, "MetricData": list(MetricData)})


@pytest.fixture
def cw() -> StubCloudWatch:
    return StubCloudWatch()


@pytest.fixture(autouse=True)
def _enable_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-enable telemetry so the gate never accidentally swallows
    test emissions. Tests that need it disabled monkeypatch in-test."""
    monkeypatch.setenv("ALPHA_ENGINE_AGENT_TELEMETRY_ENABLED", "true")


# ---------------------------------------------------------------------------
# emit_agent_completion
# ---------------------------------------------------------------------------


def _enter_time(seconds_ago: float) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds_ago)


class TestEmitAgentCompletion:
    def test_emits_four_metrics_with_agent_id_and_env_dimension(self, cw: StubCloudWatch) -> None:
        emit_agent_completion(
            agent_id="sector_team:technology",
            enter_time=_enter_time(2.5),
            exception_raised=False,
            llm_call_count=4,
            cloudwatch_client=cw,
        )
        assert len(cw.calls) == 1
        assert cw.calls[0]["Namespace"] == NAMESPACE
        names = [m["MetricName"] for m in cw.calls[0]["MetricData"]]
        assert names == ["Invocations", "Failures", "DurationMs", "LLMCallCount"]
        # config#1154: agent_id + env dimensions (env="test" by default since
        # ALPHA_ENGINE_DEPLOYED is unset under test).
        for m in cw.calls[0]["MetricData"]:
            assert m["Dimensions"] == [
                {"Name": "agent_id", "Value": "sector_team:technology"},
                {"Name": "env", "Value": "test"},
            ]

    def test_env_dimension_is_prod_when_deployed(self, cw: StubCloudWatch, monkeypatch) -> None:
        monkeypatch.setenv("ALPHA_ENGINE_DEPLOYED", "1")
        emit_agent_completion(
            agent_id="ic_cio", enter_time=_enter_time(1.0),
            exception_raised=False, llm_call_count=1, cloudwatch_client=cw,
        )
        for m in cw.calls[0]["MetricData"]:
            assert {"Name": "env", "Value": "prod"} in m["Dimensions"]

    def test_failures_value_is_one_on_exception(self, cw: StubCloudWatch) -> None:
        emit_agent_completion(
            agent_id="ic_cio",
            enter_time=_enter_time(0.5),
            exception_raised=True,
            llm_call_count=0,
            cloudwatch_client=cw,
        )
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        assert metrics["Failures"]["Value"] == 1.0
        assert metrics["Invocations"]["Value"] == 1.0  # invocation still counts

    def test_failures_value_is_zero_on_success(self, cw: StubCloudWatch) -> None:
        emit_agent_completion(
            agent_id="macro_economist",
            enter_time=_enter_time(0.5),
            exception_raised=False,
            llm_call_count=1,
            cloudwatch_client=cw,
        )
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        assert metrics["Failures"]["Value"] == 0.0

    def test_duration_ms_is_positive_and_in_milliseconds(self, cw: StubCloudWatch) -> None:
        emit_agent_completion(
            agent_id="ic_cio",
            enter_time=_enter_time(2.0),
            exception_raised=False,
            llm_call_count=1,
            cloudwatch_client=cw,
        )
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        # ~2000ms with timing slop
        assert metrics["DurationMs"]["Value"] >= 1500.0
        assert metrics["DurationMs"]["Value"] < 5000.0
        assert metrics["DurationMs"]["Unit"] == "Milliseconds"

    def test_llm_call_count_propagates(self, cw: StubCloudWatch) -> None:
        emit_agent_completion(
            agent_id="sector_team:financials",
            enter_time=_enter_time(0.1),
            exception_raised=False,
            llm_call_count=7,
            cloudwatch_client=cw,
        )
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        assert metrics["LLMCallCount"]["Value"] == 7.0

    def test_disabled_via_env_var_skips_emission(
        self, cw: StubCloudWatch, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ALPHA_ENGINE_AGENT_TELEMETRY_ENABLED", "false")
        emit_agent_completion(
            agent_id="ic_cio",
            enter_time=_enter_time(0.1),
            exception_raised=False,
            llm_call_count=1,
            cloudwatch_client=cw,
        )
        assert cw.calls == []

    def test_swallows_cw_errors_without_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CW outage must NOT take down a Sat SF run."""

        class BrokenCW:
            def put_metric_data(self, **kwargs):
                raise RuntimeError("cw is down")

        # Should not raise.
        emit_agent_completion(
            agent_id="ic_cio",
            enter_time=_enter_time(0.1),
            exception_raised=False,
            llm_call_count=1,
            cloudwatch_client=BrokenCW(),
        )


# ---------------------------------------------------------------------------
# emit_agent_retry
# ---------------------------------------------------------------------------


class TestEmitAgentRetry:
    def test_emits_two_metrics_with_agent_id_dimension(self, cw: StubCloudWatch) -> None:
        emit_agent_retry(
            agent_id="sector_quant:technology",
            attempted=True,
            succeeded=True,
            cloudwatch_client=cw,
        )
        assert len(cw.calls) == 1
        names = [m["MetricName"] for m in cw.calls[0]["MetricData"]]
        assert names == ["RetryAttempts", "RetrySuccesses"]
        for m in cw.calls[0]["MetricData"]:
            assert m["Dimensions"] == [
                {"Name": "agent_id", "Value": "sector_quant:technology"},
                {"Name": "env", "Value": "test"},
            ]

    def test_no_retry_emits_zeros_for_density(self, cw: StubCloudWatch) -> None:
        """Even no-retry path emits so the metric stream is dense (CW
        success-rate calculations need same denominator)."""
        emit_agent_retry(
            agent_id="sector_quant:financials",
            attempted=False,
            succeeded=False,
            cloudwatch_client=cw,
        )
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        assert metrics["RetryAttempts"]["Value"] == 0.0
        assert metrics["RetrySuccesses"]["Value"] == 0.0

    def test_attempted_but_failed_records_attempt_only(self, cw: StubCloudWatch) -> None:
        emit_agent_retry(
            agent_id="sector_quant:technology",
            attempted=True,
            succeeded=False,
            cloudwatch_client=cw,
        )
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        assert metrics["RetryAttempts"]["Value"] == 1.0
        assert metrics["RetrySuccesses"]["Value"] == 0.0

    def test_invariant_succeeded_implies_attempted(self, cw: StubCloudWatch) -> None:
        """Defensive: passing succeeded=True with attempted=False must
        not emit RetrySuccesses=1 — otherwise success-rate is corrupted
        when callers misuse the API."""
        emit_agent_retry(
            agent_id="sector_quant:technology",
            attempted=False,
            succeeded=True,  # nonsensical input
            cloudwatch_client=cw,
        )
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        # The implementation guards via `attempted and succeeded`.
        assert metrics["RetrySuccesses"]["Value"] == 0.0


# ---------------------------------------------------------------------------
# track_llm_cost finally-block emission (integration with existing tracker)
# ---------------------------------------------------------------------------


class TestTrackLLMCostFinallyEmission:
    def test_success_path_emits_failures_zero(self, cw: StubCloudWatch) -> None:
        from graph.llm_cost_tracker import track_llm_cost

        with patch("boto3.client", return_value=cw):
            with track_llm_cost(
                agent_id="ic_cio", model_name_fallback="claude-haiku-4-5",
            ):
                pass

        # First call is the agent_telemetry emission (4 metrics)
        assert len(cw.calls) >= 1
        names = [m["MetricName"] for m in cw.calls[0]["MetricData"]]
        assert names == ["Invocations", "Failures", "DurationMs", "LLMCallCount"]
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        assert metrics["Failures"]["Value"] == 0.0

    def test_failure_path_emits_failures_one(self, cw: StubCloudWatch) -> None:
        from graph.llm_cost_tracker import track_llm_cost

        with patch("boto3.client", return_value=cw):
            with pytest.raises(ValueError, match="boom"):
                with track_llm_cost(
                    agent_id="sector_team:technology",
                    model_name_fallback="claude-haiku-4-5",
                ):
                    raise ValueError("boom")

        # Telemetry must have been emitted in the finally block
        # before the exception propagated.
        assert len(cw.calls) >= 1
        metrics = {m["MetricName"]: m for m in cw.calls[0]["MetricData"]}
        assert metrics["Failures"]["Value"] == 1.0
        assert metrics["Invocations"]["Value"] == 1.0
        # agent_id + env dimensions correctly threaded (config#1154).
        assert metrics["Failures"]["Dimensions"] == [
            {"Name": "agent_id", "Value": "sector_team:technology"},
            {"Name": "env", "Value": "test"},
        ]

    def test_telemetry_failure_does_not_break_tracker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CW outage in agent telemetry must not halt the SF — tracker
        continues to its post-finally cost-computation path."""
        from graph.llm_cost_tracker import track_llm_cost

        def broken(*args, **kwargs):
            raise RuntimeError("cw is down")

        monkeypatch.setattr(
            "graph.agent_telemetry.emit_agent_completion", broken,
        )

        # The block should still run + complete cleanly.
        with track_llm_cost(
            agent_id="ic_cio", model_name_fallback="claude-haiku-4-5",
        ):
            pass
