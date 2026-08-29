"""Unit tests for the aggregate_costs Lambda handler (ROADMAP L1146)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date as date_type
from pathlib import Path
import datetime as _dt
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "aggregate_costs_handler.py"


def _load_handler_module():
    """Import lambda/aggregate_costs_handler.py without using ``lambda``
    as a package name (Python keyword)."""
    module_name = "lambda_aggregate_costs_handler"
    spec = importlib.util.spec_from_file_location(module_name, _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def handler_mod():
    mod = _load_handler_module()
    mod._init_done = False
    yield mod


def _fresh_state(as_of: str = "2026-05-25") -> dict:
    return {
        "schema_version": 1,
        "as_of": as_of,
        "dates_present": [as_of],
        "last_capture_date": as_of,
        "days_since_last_capture": 0,
        "producers_on_last_capture_date": ["replay-concordance"],
        "max_age_days": 8,
    }


@pytest.fixture(autouse=True)
def capture_stream_alive():
    """Default the capture-stream verdict to FRESH for the aggregation tests.

    These tests are about what the handler does with the aggregation window;
    the capture-stream detector (config-I7407 D4) is a separate assertion with
    its own tests below and in ``test_cost_capture_freshness.py``. Without
    this, every one of them would fail on a stale-capture raise about a
    stream the test never set up — which would test the fixture, not the
    handler.
    """
    with patch(
        "scripts.cost_capture_freshness.evaluate_and_publish",
        side_effect=lambda s3, bucket, **kw: _fresh_state(
            kw["as_of"].isoformat()
        ),
    ) as p:
        yield p


def _ok_summary() -> dict:
    return {
        "rows_in": 1234,
        "files_read": 87,
        "output_key": "decision_artifacts/_cost/2026-05-25/cost.parquet",
        "total_cost_usd": 12.3456,
        "total_input_tokens": 5_000_000,
        "total_output_tokens": 300_000,
        "total_cache_read_tokens": 200_000,
        "total_cache_create_tokens": 50_000,
        "total_web_search_requests": 0,
        "total_web_fetch_requests": 0,
        "by_sector_team": {"tech": 4.0, "financials": 3.5},
        "by_model": {"claude-sonnet-4-6": 8.0, "claude-haiku-4-5": 4.0},
        "by_run_type": {"saturday_research": 12.0},
        "by_agent_id": {},
    }


class TestHandler:
    def test_ok_when_the_window_aggregated_at_least_one_date(self, handler_mod):
        # Real boto3 clients are constructed inside the handler before
        # aggregation is called; patch the boto3 module wholesale to
        # avoid network credential errors in CI.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", side_effect=lambda s3, b, d: d.isoformat() == "2026-05-24"),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"date": "2026-05-25"},
                context=None,
            )
        assert result["status"] == "OK"
        assert result["summary"]["dates_aggregated"] == ["2026-05-24"]
        assert result["date"] == "2026-05-25"

    def test_skipped_only_when_the_WHOLE_window_is_quiet(self, handler_mod):
        # config-I7407: this used to fire whenever the ONE named date was
        # empty — the common case, not the exceptional one, since capture is
        # daily and the SF names a single run_date.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=False),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"date": "2026-05-25"},
                context=None,
            )
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "no_cost_raw_in_window"
        assert result["date"] == "2026-05-25"

    def test_error_when_aggregate_raises(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=True),
            patch("scripts.aggregate_costs.aggregate_day", side_effect=RuntimeError("S3 unreachable")),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"date": "2026-05-25"},
                context=None,
            )
        assert result["status"] == "ERROR"
        assert "S3 unreachable" in result["error"]

    def test_error_when_date_missing(self, handler_mod):
        # Hard contract: SF state MUST thread `state.run_date` into the
        # event. Empty event triggers an explicit ERROR with a clear
        # message rather than a silent default to "today" (which would
        # silently aggregate the wrong partition on a recovery SF that
        # re-runs an older date).
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "date" in result["error"]

    def test_error_when_date_invalid(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler(
                {"date": "not-a-date"},
                context=None,
            )
        assert result["status"] == "ERROR"
        assert "not-a-date" in result["error"]

    def test_dry_run_short_circuits_before_s3(self, handler_mod):
        # dry_run_llm shell-run path must NOT touch S3 or call
        # aggregate_day. Mirrors the rationale_clustering / eval_judge
        # dry path used by Friday-Preflight shell runs.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day") as mock_agg,
            patch("boto3.client") as mock_boto,
        ):
            result = handler_mod.handler(
                {"dry_run_llm": True, "date": "2026-05-25"},
                context=None,
            )
        assert result["status"] == "OK"
        assert result["dry_run"] is True
        mock_agg.assert_not_called()
        mock_boto.assert_not_called()

    def test_target_date_threaded_through(self, handler_mod):
        # The handler must pass the parsed date_type instance through —
        # not the raw string. aggregate_day's signature requires
        # date_type so a string would TypeError at the call site.
        captured = {}

        def fake_aggregate(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day", side_effect=fake_aggregate),
            patch("boto3.client", return_value=MagicMock()),
        ):
            handler_mod.handler({"date": "2026-05-25"}, context=None)

        assert captured["target_date"] == date_type(2026, 5, 25)
        # Bucket defaults to the configured RESEARCH_BUCKET env var
        # (or the fallback constant) — confirms the kwarg is wired.
        assert captured["bucket"] in (
            "alpha-engine-research",
            handler_mod._DEFAULT_BUCKET,
        )

    def test_bucket_override(self, handler_mod):
        captured = {}

        def fake_aggregate(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day", side_effect=fake_aggregate),
            patch("boto3.client", return_value=MagicMock()),
        ):
            handler_mod.handler(
                {"date": "2026-05-25", "bucket": "test-bucket"},
                context=None,
            )

        assert captured["bucket"] == "test-bucket"


_COVERAGE_DECL = {
    "execution_arn": "arn:aws:states:us-east-1:1:execution:ne-weekly-freshness-pipeline:e1",
    "required_producers": {"ReplayConcordance": ["replay-concordance"]},
    "conditional_producers": {},
    "allowed_producers": ["thinktank-*"],
}


#: alpha-engine-config-I9261 — the coverage read is anchored on the
#: execution's own start time, not on a calendar date, so these stubs
#: supply one.
_STARTED_AT = _dt.datetime(2026, 5, 25, 9, 0, tzinfo=_dt.timezone.utc)


class TestFanInCoverage:
    """alpha-engine-config-I7179 — the check must reach the Step Function's
    Catch, which fires on a RAISED error. Nothing downstream reads this
    state's returned status, so a returned ERROR here is silence."""

    def test_a_silent_stage_raises_rather_than_returning_error(self, handler_mod):
        """It must RAISE. The SF has no Choice on this state's status; its
        only failure path is the States.ALL Catch that routes to
        MarkAggregateCostsDegraded."""
        from scripts.cost_coverage import CostCoverageError

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("scripts.cost_coverage.execution_started_at", return_value=_STARTED_AT),
            patch(
                "scripts.cost_coverage.observed_producers_for_execution",
                return_value={"thinktank-sweep"},
            ),
            patch("scripts.cost_coverage.stages_entered", return_value={"ReplayConcordance"}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            with pytest.raises(CostCoverageError, match="replay-concordance"):
                handler_mod.handler(
                    {"date": "2026-05-25", "coverage": _COVERAGE_DECL},
                    context=None,
                )

    def test_an_empty_partition_is_not_a_free_pass(self, handler_mod):
        """The branch that mattered most. An EMPTY partition on a day the
        pipeline's LLM stages ran is the I7179 defect in its purest form,
        and it used to return SKIPPED with the state succeeding."""
        from scripts.cost_coverage import CostCoverageError

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch(
                "scripts.aggregate_costs.aggregate_window", return_value={"aggregated": [], "skipped": ["2026-05-25"]}
            ),
            patch("scripts.cost_coverage.execution_started_at", return_value=_STARTED_AT),
            patch(
                "scripts.cost_coverage.observed_producers_for_execution",
                return_value=set(),
            ),
            patch("scripts.cost_coverage.stages_entered", return_value={"ReplayConcordance"}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            with pytest.raises(CostCoverageError):
                handler_mod.handler(
                    {"date": "2026-05-25", "coverage": _COVERAGE_DECL},
                    context=None,
                )

    def test_an_empty_partition_on_a_day_nothing_ran_is_still_skipped(self, handler_mod):
        """The other half of the same branch: a quiet week is legitimate,
        and the declaration plus the execution history is what tells the
        two apart."""
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch(
                "scripts.aggregate_costs.aggregate_window", return_value={"aggregated": [], "skipped": ["2026-05-25"]}
            ),
            patch("scripts.cost_coverage.execution_started_at", return_value=_STARTED_AT),
            patch(
                "scripts.cost_coverage.observed_producers_for_execution",
                return_value=set(),
            ),
            patch("scripts.cost_coverage.stages_entered", return_value={"CheckSkipReplayConcordance"}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"date": "2026-05-25", "coverage": _COVERAGE_DECL},
                context=None,
            )
        assert result["status"] == "SKIPPED"

    def test_full_coverage_lands_the_verdict_on_the_summary(self, handler_mod):
        """The verdict is data, not just an exception message — an
        exception message is not a surface and cannot be trended."""
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("scripts.cost_coverage.execution_started_at", return_value=_STARTED_AT),
            patch(
                "scripts.cost_coverage.observed_producers_for_execution",
                return_value={"replay-concordance", "thinktank-sweep"},
            ),
            patch("scripts.cost_coverage.stages_entered", return_value={"ReplayConcordance"}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"date": "2026-05-25", "coverage": _COVERAGE_DECL},
                context=None,
            )
        assert result["status"] == "OK"
        assert result["summary"]["coverage"]["covered"] == ["replay-concordance"]
        assert result["summary"]["coverage"]["missing"] == []

    def test_being_unable_to_measure_is_not_a_pass(self, handler_mod):
        """principles.md §2.7 — no data is never rendered as green. And it
        is a DIFFERENT exception from a breach, so a harness fault is never
        reported as a domain finding."""
        from scripts.cost_coverage import CostCoverageUnmeasured

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("scripts.cost_coverage.execution_started_at", return_value=_STARTED_AT),
            patch(
                "scripts.cost_coverage.observed_producers_for_execution",
                return_value=set(),
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            with pytest.raises(CostCoverageUnmeasured):
                handler_mod.handler(
                    {"date": "2026-05-25", "coverage": {**_COVERAGE_DECL, "execution_arn": ""}},
                    context=None,
                )

    def test_no_declaration_means_no_check_not_a_failure(self, handler_mod):
        """The handler is also reachable from the deploy canary and from
        the CLI, neither of which is a pipeline run."""
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"date": "2026-05-25"}, context=None)
        assert result["status"] == "OK"
        assert "coverage" not in result["summary"]


class TestCaptureStreamFreshness:
    """config-I7407 deliverable 4 — a freshness detector on CAPTURE itself.

    ``llm_cost_parquet`` watches the product of capture. These assert the
    handler now also grades the stream that feeds it, on BOTH terminal
    paths, and that a dead stream reaches the Step Function's Catch rather
    than a green ``SKIPPED``.
    """

    def test_verdict_lands_on_the_ok_result(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=True),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"date": "2026-05-25"}, context=None)
        assert result["status"] == "OK"
        assert result["capture_stream"]["last_capture_date"] == "2026-05-25"
        assert result["capture_stream"]["producers"] == ["replay-concordance"]

    def test_the_quiet_window_is_graded_not_accepted(
        self, handler_mod, capture_stream_alive
    ):
        """A window with nothing in it is the exact shape of a total capture
        outage, so the SKIPPED path must assert the stream, not return."""
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=False),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"date": "2026-05-25"}, context=None)
        assert result["status"] == "SKIPPED"
        assert capture_stream_alive.called
        assert result["capture_stream"]["days_since_last_capture"] == 0

    def test_a_dead_stream_raises_out_of_the_skipped_path(
        self, handler_mod, capture_stream_alive
    ):
        """The regression this exists to prevent: capture stops fleet-wide,
        the aggregator honestly finds nothing, and the stage succeeds."""
        from scripts.cost_capture_freshness import CostCaptureStaleError

        capture_stream_alive.side_effect = CostCaptureStaleError("stream empty")
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=False),
            patch("boto3.client", return_value=MagicMock()),
        ):
            with pytest.raises(CostCaptureStaleError):
                handler_mod.handler({"date": "2026-05-25"}, context=None)

    def test_a_dead_stream_raises_out_of_the_ok_path_too(
        self, handler_mod, capture_stream_alive
    ):
        """A parquet built from an old partition is still a live parquet.
        The window fix (#634) means an OK status can sit on top of a stream
        that stopped days ago, so OK is not an exemption."""
        from scripts.cost_capture_freshness import CostCaptureStaleError

        capture_stream_alive.side_effect = CostCaptureStaleError("stream stale")
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=True),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("boto3.client", return_value=MagicMock()),
        ):
            with pytest.raises(CostCaptureStaleError):
                handler_mod.handler({"date": "2026-05-25"}, context=None)

    def test_the_dry_shell_run_does_not_grade_the_stream(self, handler_mod):
        """``dry_run_llm`` is the Friday-Preflight bootstrap smoke — it reads
        and writes no S3, so it has no verdict to offer and must not raise
        one."""
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"date": "2026-05-25", "dry_run_llm": True}, context=None
            )
        assert result == {"status": "OK", "dry_run": True}


class TestFanInCoverageReadsTheExecutionNotTheRunDate:
    """alpha-engine-config-I9261 — end to end, through the real listing.

    Reproduces the 2026-08-29 weekly run against a fake S3 holding exactly
    what production held. FAILS on main with the live CostCoverageError:
    `1 stage(s) ran and emitted no cost record: replay-concordance`.
    """

    _RUN_DATE = "2026-08-28"
    _STARTED = _dt.datetime(2026, 8, 29, 9, 0, 49, tzinfo=_dt.timezone.utc)

    class _ProdS3:
        """`_cost_raw/` as measured on 2026-08-29."""

        _OBJECTS = {
            # Friday's DAILY pipelines, in the run_date partition.
            "decision_artifacts/_cost_raw/2026-08-28/8e9fff668cc8/thinktank-sweep.0.jsonl":
                _dt.datetime(2026, 8, 28, 15, 16, 30, tzinfo=_dt.timezone.utc),
            "decision_artifacts/_cost_raw/2026-08-28/krepis-7faeb93de042/single-agent-quant.0.jsonl":
                _dt.datetime(2026, 8, 28, 22, 20, 7, tzinfo=_dt.timezone.utc),
            # THIS execution, in the UTC partition it actually ran on.
            "decision_artifacts/_cost_raw/2026-08-29/krepis-31e1cf2e27b5/single-agent-quant.0.jsonl":
                _dt.datetime(2026, 8, 29, 11, 14, 32, tzinfo=_dt.timezone.utc),
            "decision_artifacts/_cost_raw/2026-08-29/krepis-b1444bfa3454/replay-concordance.0.jsonl":
                _dt.datetime(2026, 8, 29, 12, 27, 43, tzinfo=_dt.timezone.utc),
        }

        def get_paginator(self, name):
            objects = self._OBJECTS

            class _P:
                def paginate(self, **kwargs):
                    prefix = kwargs["Prefix"]
                    contents = [
                        {"Key": k, "LastModified": ts}
                        for k, ts in sorted(objects.items())
                        if k.startswith(prefix)
                    ]
                    return [{"Contents": contents} if contents else {}]

            return _P()

        def list_objects_v2(self, **kwargs):
            prefix = kwargs["Prefix"]
            contents = [
                {"Key": k, "LastModified": ts}
                for k, ts in sorted(self._OBJECTS.items())
                if k.startswith(prefix)
            ]
            return {"Contents": contents} if contents else {}

    def _client(self, service, *a, **k):
        if service == "s3":
            return self._ProdS3()
        return MagicMock()

    def test_the_weekly_runs_own_producers_are_covered(self, handler_mod):
        decl = {
            "execution_arn": "arn:aws:states:us-east-1:1:execution:ne-weekly-freshness-pipeline:e1",
            "required_producers": {
                "ChallengerShadow": ["single-agent-quant"],
                "ReplayConcordance": ["replay-concordance"],
            },
            "conditional_producers": {},
            "allowed_producers": ["thinktank-*"],
        }
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_ok_summary()),
            patch("scripts.aggregate_costs.aggregate_window", return_value={
                "aggregated": [self._RUN_DATE], "skipped": [], "summaries": {}
            }),
            patch("scripts.cost_coverage.execution_started_at", return_value=self._STARTED),
            patch(
                "scripts.cost_coverage.stages_entered",
                return_value={"ChallengerShadow", "ReplayConcordance"},
            ),
            patch(
                "scripts.cost_coverage.datetime",
                wraps=_dt.datetime,
            ) as fake_dt,
            patch("boto3.client", side_effect=self._client),
        ):
            fake_dt.now.return_value = _dt.datetime(
                2026, 8, 29, 14, 2, 40, tzinfo=_dt.timezone.utc
            )
            result = handler_mod.handler(
                {"date": self._RUN_DATE, "coverage": decl}, context=None
            )

        verdict = result["summary"]["coverage"]
        assert verdict["missing"] == []
        assert verdict["undeclared"] == []
        assert sorted(verdict["covered"]) == [
            "replay-concordance",
            "single-agent-quant",
        ]
        # Friday's thinktank objects belong to a run this execution never
        # made — they must not appear as this execution's producers.
        assert "thinktank-sweep" not in verdict["observed"]
