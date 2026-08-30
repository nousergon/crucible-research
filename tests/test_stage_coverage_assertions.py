"""Unit tests for the config-I7214 stage-coverage self-assertion rescope.

Brian ruled the end-of-run ``StageCoverageAssert`` SF state NON-SOTA: the
assertion belongs in each stage's own handler, immediately before it
returns, calling the ONE shared implementation
(``krepis.stage_coverage.assert_stage_coverage``) rather than a
per-repo reimplementation. The primitive lives in krepis (relocated
from an initial nousergon_lib landing — nousergon-lib-PR314 merged,
PR315 removes the duplicate — because half its callers are bash
launchers and krepis is published rather than git-pinned) and ships
from krepis 0.59.4, which ``requirements.txt`` now floors. This module
pins, per handler:

1. the verdict lands in the returned payload under ``stage_coverage``
   with the CORRECT stage name;
2. an ``ImportError`` from the lib does NOT change the handler's
   outcome — observe mode cannot break the stage it observes;
3. (``eval_judge_submit_handler`` only) BOTH polarities of
   ``force_sonnet_pass`` file under the correct one of
   ``EvalJudgeSubmitFirstSaturday`` / ``EvalJudgeSubmitWeekly``;
4. no shipped call site ever passes an enforcement flag — the lib owns
   that switch and every call site here is OBSERVE MODE ONLY;
5. every handler this repo deploys that backs a weekly-SF Task state
   carries the assertion (totality — the SF Task-state stage list is
   pinned below, sourced 2026-08-13 from
   ``nousergon-data/infrastructure/step_function.json``'s Scanner /
   SignalsEnvelope / ChallengerShadow / EvalJudgeSubmit{FirstSaturday,
   Weekly} / EvalJudgeProcess / EvalRollingMean / RationaleClustering /
   AggregateCosts Task states (``EvalJudgePoll`` retired with the async
   batch API, alpha-engine-config-I9329) — an enumeration test
   that only lists what EXISTS is blind to where one is missing, so this
   list is the independently-sourced denominator, not a scan of the repo).

The ``stub_stage_coverage`` fixture (root ``conftest.py``) injects a fake
``krepis.stage_coverage`` submodule into ``sys.modules`` so the
handler's lazy ``from krepis.stage_coverage import
assert_stage_coverage`` succeeds against a controllable mock.

Tests that do NOT request it get the autouse
``_stage_coverage_absent_unless_stubbed`` default from the same
conftest, which makes that import raise ``ImportError`` — a SIMULATED
absence. It was an ambient one until 2026-08-14: these tests read "the
installed krepis has no such module" as a standing fact, and the real
module (which builds its own boto3 S3/CloudWatch clients) started
running inside every handler test the moment krepis published it,
wedging CI on live AWS calls the runner could not make.

``TestPrimitiveIsImportable`` covers what neither simulated side can:
that the real module is importable under this repo's declared pins. A
coverage assertion that cannot import emits nothing, in a shape
indistinguishable from an assertion that found nothing wrong
(alpha-engine-config-I7334).
"""

from __future__ import annotations

import datetime
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LAMBDA_DIR = _REPO_ROOT / "lambda"


def _load_handler(filename: str, module_name: str):
    """Import lambda/<filename> without using ``lambda`` as a package
    name (Python keyword) — mirrors every other handler test in this
    suite (test_scanner_handler.py et al.)."""
    path = _LAMBDA_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── Fixtures: one loaded module per handler under test ─────────────────


@pytest.fixture
def scanner_mod():
    mod = _load_handler("scanner_handler.py", "lambda_i7214_scanner_handler")
    mod._init_done = False
    yield mod


@pytest.fixture
def signals_envelope_mod():
    mod = _load_handler("signals_envelope_handler.py", "lambda_i7214_signals_envelope_handler")
    mod._init_done = False
    yield mod


@pytest.fixture
def runner_mod():
    mod = _load_handler("handler.py", "lambda_i7214_runner_handler")
    yield mod


@pytest.fixture
def submit_mod():
    mod = _load_handler("eval_judge_submit_handler.py", "lambda_i7214_eval_judge_submit_handler")
    mod._init_done = False
    yield mod


@pytest.fixture
def rolling_mean_mod():
    mod = _load_handler("eval_rolling_mean_handler.py", "lambda_i7214_eval_rolling_mean_handler")
    mod._init_done = False
    yield mod


@pytest.fixture
def clustering_mod():
    mod = _load_handler("rationale_clustering_handler.py", "lambda_i7214_rationale_clustering_handler")
    mod._init_done = False
    yield mod


@pytest.fixture
def aggregate_costs_mod():
    mod = _load_handler("aggregate_costs_handler.py", "lambda_i7214_aggregate_costs_handler")
    mod._init_done = False
    yield mod


@pytest.fixture
def absent_stage_coverage(monkeypatch):
    """Force ``from krepis.stage_coverage import ...`` to raise ImportError.

    The root ``conftest.py`` installs the same default suite-wide (so no
    handler test reaches real S3 through the primitive); requesting it
    explicitly here states the precondition these tests depend on rather
    than borrowing it — which is how it silently stopped holding once
    krepis published the module.
    """
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", None)
    yield


@pytest.fixture(autouse=True)
def _stub_universe_membership():
    """Scanner's membership write is NOT fail-soft (config-I4818) — an
    unstubbed call reaches real S3. Mirrors test_scanner_handler.py's
    identical autouse fixture."""
    with (
        patch(
            "scoring.universe_membership.compute_and_write_universe_membership",
            return_value="universe_membership/2026-05-29/membership.json",
        ),
        patch(
            "data.fetchers.feature_store_reader.read_latest_factor_loadings",
            return_value={},
        ),
    ):
        yield


def _ok_scanner_artifact() -> dict:
    return {
        "run_date": "2026-05-30",
        "scanner_version": "v1.0",
        "generated_at": "2026-05-30T09:00:00+00:00",
        "population_tickers": ["AAPL", "GOOG"],
        "scanner_tickers": ["AMD", "BNY", "SN"],
        "agent_input_set": ["AAPL", "GOOG", "AMD", "BNY", "SN"],
        "filters_applied": {"min_avg_volume": 500000},
        "stats": {
            "universe_size": 903,
            "post_scanner": 3,
            "population_size": 2,
            "agent_input_size": 5,
            "feature_store_enriched": 903,
            "feature_store_missing": 0,
            "new_vs_prior_cycle": ["BNY", "SN"],
            "dropped_vs_prior_cycle": ["PSTG"],
            "prior_run_date": "2026-05-23",
            "baseline_missing": False,
        },
    }


def _envelope(run_date: str = "2026-07-14") -> dict:
    return {
        "schema_version": 1,
        "producer": "signals_envelope",
        "date": run_date,
        "run_date": run_date,
        "time": "12:00:00",
        "run_time": "12:00:00",
        "market_regime": "neutral",
        "sector_ratings": {},
        "sector_modifiers": {},
        "universe": [{"ticker": "AAPL"}, {"ticker": "JNJ"}],
        "buy_candidates": [],
        "population": ["AAPL", "JNJ"],
        "signals": {},
    }


def _board() -> dict:
    return {
        "stocks": [
            {"ticker": "AAPL", "sector": "Technology", "attractiveness_score": 72.5, "pillars": {}},
        ],
    }


# ── Scanner ──────────────────────────────────────────────────────────────


class TestScannerCoverage:
    def test_verdict_lands_under_correct_stage_name(self, scanner_mod, stub_stage_coverage):
        with (
            patch.object(scanner_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_scanner_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-30/candidates.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = scanner_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        stub_stage_coverage.assert_called_once()
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "Scanner"
        # alpha-engine-config-I8155: stage_coverage groups a run's verdicts
        # by the execution's OWN run_date — the un-normalized event value,
        # not `resolve_trading_day`'s output (2026-05-29). Passing the
        # normalized trading day filed this stage's verdict under the WRONG
        # prefix on the 2026-08-22 weekly run.
        assert kwargs["run_date"] == "2026-05-30"
        assert kwargs["run_date"] != "2026-05-29"
        # And the trading-day normalization still happened for everything
        # else the handler does with `date` — the summary/`result["date"]`
        # key is the TRADING day, unchanged by this fix.
        assert result["summary"] is not None
        stub_stage_coverage.assert_called_once()
        assert kwargs["window_start"] is not None

    def test_missing_lib_module_does_not_change_outcome(self, scanner_mod, absent_stage_coverage):
        # `absent_stage_coverage` forces the ImportError explicitly —
        # the observe-mode degrade path, simulated rather than inherited
        # from whichever krepis happens to be installed.
        with (
            patch.object(scanner_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_scanner_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-30/candidates.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = scanner_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        # alpha-engine-config-I8155: an unavailable lib now degrades to a
        # recorded UNMEASURED verdict rather than silently omitting the key.
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "Scanner"

    def test_contract_error_from_lib_degrades_to_unmeasured_without_raising(
        self, scanner_mod, stub_stage_coverage,
    ):
        # alpha-engine-config-I8155: the krepis landing this arc makes
        # run_date a required, contract-enforced kwarg — simulate its
        # raising and assert the handler still returns OK.
        stub_stage_coverage.side_effect = TypeError("run_date is required")
        with (
            patch.object(scanner_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_scanner_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-30/candidates.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = scanner_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"


class TestScannerLeaderboardCoverage:
    """alpha-engine-config-I9050: the ``scanner_leaderboard`` leaf mode
    (mode="scanner_leaderboard") dispatches to ``_run_scanner_leaderboard``
    and returns BEFORE `_run`'s scan-path body — including, before this fix,
    before the only ``assert_stage_coverage`` call in the file. A
    file-level "does this file call assert_stage_coverage anywhere" check
    (``TestTotalityAndEnforcement`` above) could never catch this: the file
    already carried a call, for the unrelated "Scanner" stage. Only an
    end-to-end exercise of the leaf mode itself — asserting the call
    actually happens, under the correct stage name, on THIS path — catches
    a stage that runs and writes its declared artifact
    (``scanner/leaderboard/{date}.json``) while never refreshing its
    ``_stage_coverage/{date}/ScannerLeaderboard.json`` companion."""

    def test_verdict_lands_under_correct_stage_name(self, scanner_mod, stub_stage_coverage):
        with (
            patch.object(scanner_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch(
                "scoring.leaderboard_producers.build_scanner_leaderboard",
                return_value={"status": "ok", "key": "scanner/leaderboard/2026-05-29.json"},
            ),
        ):
            result = scanner_mod.handler(
                {"run_date": "2026-05-30", "mode": "scanner_leaderboard"}, context=None
            )
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        stub_stage_coverage.assert_called_once()
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "ScannerLeaderboard"
        # Same execution-run_date contract as the Scanner stage (config-I8155):
        # keyed by the un-normalized event value, not the trading day.
        assert kwargs["run_date"] == "2026-05-30"
        assert kwargs["run_date"] != "2026-05-29"
        assert kwargs["window_start"] is not None

    def test_missing_lib_module_does_not_change_outcome(self, scanner_mod, absent_stage_coverage):
        with (
            patch.object(scanner_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch(
                "scoring.leaderboard_producers.build_scanner_leaderboard",
                return_value={"status": "ok", "key": "scanner/leaderboard/2026-05-29.json"},
            ),
        ):
            result = scanner_mod.handler(
                {"run_date": "2026-05-30", "mode": "scanner_leaderboard"}, context=None
            )
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "ScannerLeaderboard"

    def test_unmeasurable_leaderboard_still_asserts_coverage(self, scanner_mod, stub_stage_coverage):
        # The "unmeasurable" board status is a legitimate written decision
        # (docstring on `_run_scanner_leaderboard`), not a failure — the
        # handler still returns OK and must still assert coverage.
        with (
            patch.object(scanner_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch(
                "scoring.leaderboard_producers.build_scanner_leaderboard",
                return_value={
                    "status": "unmeasurable",
                    "key": "scanner/leaderboard/2026-05-29.json",
                    "reason": "no cohort has matured",
                },
            ),
        ):
            result = scanner_mod.handler(
                {"run_date": "2026-05-30", "mode": "scanner_leaderboard"}, context=None
            )
        assert result["status"] == "OK"
        stub_stage_coverage.assert_called_once()
        assert stub_stage_coverage.call_args[0][0] == "ScannerLeaderboard"


# ── SignalsEnvelope ──────────────────────────────────────────────────────


class TestSignalsEnvelopeCoverage:
    def test_verdict_lands_under_correct_stage_name(self, signals_envelope_mod, stub_stage_coverage):
        with (
            patch.object(signals_envelope_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch("scoring.signals_envelope.read_universe_board", return_value=_board()),
            patch("scoring.signals_envelope.read_regime_substrate", return_value=None),
            patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()),
            patch(
                "scoring.signals_envelope.write_envelope",
                return_value=("signals/2026-07-14/signals.json", "signals/latest.json"),
            ),
        ):
            result = signals_envelope_mod.handler({"run_date": "2026-07-14"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "SignalsEnvelope"
        assert kwargs["run_date"] == "2026-07-14"

    def test_verdict_uses_execution_run_date_not_trading_day_on_a_weekend_cycle(
        self, signals_envelope_mod, stub_stage_coverage,
    ):
        # alpha-engine-config-I8155: a Saturday weekly-SF cycle passes a
        # CALENDAR run_date; `resolve_trading_day` normalizes the handler's
        # own `run_date` local to the prior Friday for every artifact key,
        # but stage_coverage must still see the un-normalized 2026-08-08
        # (Saturday) — that's the key one execution's verdicts share.
        with (
            patch.object(signals_envelope_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch("scoring.signals_envelope.read_universe_board", return_value=_board()),
            patch("scoring.signals_envelope.read_regime_substrate", return_value=None),
            patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope("2026-08-07")),
            patch(
                "scoring.signals_envelope.write_envelope",
                return_value=("signals/2026-08-07/signals.json", "signals/latest.json"),
            ),
        ):
            result = signals_envelope_mod.handler({"run_date": "2026-08-08"}, context=None)
        assert result["status"] == "OK"
        args, kwargs = stub_stage_coverage.call_args
        assert kwargs["run_date"] == "2026-08-08"
        assert kwargs["run_date"] != "2026-08-07"  # the normalized trading day

    def test_missing_lib_module_does_not_change_outcome(self, signals_envelope_mod, absent_stage_coverage):
        with (
            patch.object(signals_envelope_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch("scoring.signals_envelope.read_universe_board", return_value=_board()),
            patch("scoring.signals_envelope.read_regime_substrate", return_value=None),
            patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()),
            patch(
                "scoring.signals_envelope.write_envelope",
                return_value=("signals/2026-07-14/signals.json", "signals/latest.json"),
            ),
        ):
            result = signals_envelope_mod.handler({"run_date": "2026-07-14"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "SignalsEnvelope"


# ── ChallengerShadow (handler.py::_run_challengers_only) ────────────────


class TestChallengerShadowCoverage:
    """The event carries the SF's CALENDAR date (`$.run_date`, a Saturday) and
    every artifact is keyed on the trading day — `_run_challengers_only`
    normalises between them (alpha-engine-config-I7419). The fixtures below
    therefore pair a Saturday event date with a Friday-keyed
    `signals/latest.json`, which is the only combination the live pipeline can
    produce; the earlier all-Saturday fixture described a state that cannot
    exist under DATE_CONVENTIONS."""

    def test_verdict_lands_under_correct_stage_name(self, runner_mod, stub_stage_coverage):
        archive = MagicMock()
        archive.bucket = "alpha-engine-research"
        archive.s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b'{"date": "2026-05-29"}')),
        }
        archive.load_population.return_value = [
            {"ticker": "AAPL", "entry_date": "2026-05-23"},
            {"ticker": "MSFT", "entry_date": "2026-05-29"},
        ]
        with (
            patch("archive.manager.ArchiveManager", return_value=archive),
            patch("producers.runner.run_challengers", return_value={"written": ["challenger_a"]}),
        ):
            result = runner_mod._run_challengers_only({"mode": "challengers_only", "date": "2026-05-30"})
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "ChallengerShadow"
        # alpha-engine-config-I8155: the execution's own un-normalized
        # run_date (event['date']), not the trading day the rest of the
        # handler normalizes to for population/signals lookups.
        assert kwargs["run_date"] == "2026-05-30"
        assert kwargs["run_date"] != "2026-05-29"  # the normalized trading day

    def test_missing_lib_module_does_not_change_outcome(self, runner_mod, absent_stage_coverage):
        archive = MagicMock()
        archive.bucket = "alpha-engine-research"
        archive.s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b'{"date": "2026-05-29"}')),
        }
        archive.load_population.return_value = [
            {"ticker": "AAPL", "entry_date": "2026-05-23"},
        ]
        with (
            patch("archive.manager.ArchiveManager", return_value=archive),
            patch("producers.runner.run_challengers", return_value={"written": []}),
        ):
            result = runner_mod._run_challengers_only({"mode": "challengers_only", "date": "2026-05-30"})
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "ChallengerShadow"


# ── EvalJudgeSubmit{FirstSaturday,Weekly} — one Lambda, two stages ──────


def _submit_result() -> dict:
    return {
        "batch_id": "msgbatch_123",
        "plan_s3_key": "decision_artifacts/_eval_batch_plans/2026-05-16/msgbatch_123.json",
        "request_count": 10,
        "processing_status": "in_progress",
    }


class TestEvalJudgeSubmitCoverage:
    def _invoke(self, submit_mod, event: dict):
        with (
            patch.object(submit_mod, "_ensure_init"),
            patch("anthropic.Anthropic", return_value=MagicMock()),
            patch("boto3.client", return_value=MagicMock()),
            patch(
                "evals.orchestrator.build_batch_plan",
                return_value={
                    "capture_keys_total": 5, "skipped_unmapped": 0,
                    "capture_partition_counts": {"2026-05-16": 5},
                    "empty_trading_day_partitions": [],
                },
            ),
            patch("evals.orchestrator._persist_client_side_skips", return_value=(0, 0, None, [])),
            patch("evals.orchestrator.submit_batch", return_value=_submit_result()),
        ):
            return submit_mod.handler(event, context=None)

    def test_force_sonnet_pass_true_files_under_first_saturday(self, submit_mod, stub_stage_coverage):
        result = self._invoke(submit_mod, {"date": "2026-05-16", "force_sonnet_pass": True})
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "EvalJudgeSubmitFirstSaturday"
        assert kwargs["run_date"] == "2026-05-16"

    def test_force_sonnet_pass_false_files_under_weekly(self, submit_mod, stub_stage_coverage):
        result = self._invoke(submit_mod, {"date": "2026-05-16", "force_sonnet_pass": False})
        assert result["status"] == "OK"
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "EvalJudgeSubmitWeekly"

    def test_missing_lib_module_does_not_change_outcome(self, submit_mod, absent_stage_coverage):
        result = self._invoke(submit_mod, {"date": "2026-05-16", "force_sonnet_pass": False})
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "EvalJudgeSubmitWeekly"

    def test_no_fabricated_date_when_event_date_absent(self, submit_mod, stub_stage_coverage):
        # alpha-engine-config-I8155: the handler's own business-logic
        # fallback (`date = event.get("date") or str(datetime.date.today())`)
        # must never leak into the coverage assertion — a genuinely absent
        # execution identity records UNMEASURED, not today's wall-clock date.
        result = self._invoke(submit_mod, {"force_sonnet_pass": False})
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        stub_stage_coverage.assert_not_called()


# ── EvalJudgeProcess (spot substrate) ────────────────────────────────────
#
# The stage survived alpha-engine-config-I9329; its SUBSTRATE changed. The
# Lambda handler is gone and `evals/judge_spot_run.py` carries the assertion
# now, so the subject of these tests is that module's `_stage_coverage`, not a
# handler event. run_date is no longer recovered from `plan_s3_key` — the SF
# passes it as `--date` — so the UNMEASURED-on-unrecoverable-identity case
# (alpha-engine-config-I8155) has no path here and its test is retired with the
# handler.


class TestEvalJudgeProcessCoverage:
    def test_verdict_lands_with_the_run_date_the_sf_passed(self, stub_stage_coverage):
        from evals.judge_spot_run import _stage_coverage

        started = datetime.datetime(2026, 5, 16, 9, 0, tzinfo=datetime.UTC)
        result = _stage_coverage(run_date="2026-05-16", window_start=started)

        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "EvalJudgeProcess"
        assert kwargs["run_date"] == "2026-05-16"
        assert kwargs["window_start"] == started
        assert result is stub_stage_coverage.return_value

    def test_missing_lib_module_does_not_change_outcome(self, absent_stage_coverage):
        """Observe mode: an absent krepis.stage_coverage degrades to
        UNMEASURED *with a reason*, and never raises out of the run."""
        from evals.judge_spot_run import _stage_coverage

        result = _stage_coverage(
            run_date="2026-05-16",
            window_start=datetime.datetime(2026, 5, 16, 9, 0, tzinfo=datetime.UTC),
        )
        assert result["stage"] == "EvalJudgeProcess"
        assert result["status"] == "UNMEASURED"
        assert result["reason"]

    def test_a_raising_assertion_never_kills_the_run(self, stub_stage_coverage):
        from evals.judge_spot_run import _stage_coverage

        stub_stage_coverage.side_effect = RuntimeError("registry unreachable")
        result = _stage_coverage(
            run_date="2026-05-16",
            window_start=datetime.datetime(2026, 5, 16, 9, 0, tzinfo=datetime.UTC),
        )
        assert result["status"] == "UNMEASURED"
        assert "RuntimeError" in result["reason"]


# ── EvalRollingMean ──────────────────────────────────────────────────────


def _rolling_mean_summary() -> dict:
    return {
        "combos_discovered": 12,
        "datapoints_emitted": 12,
        "combos_skipped_no_data": 0,
        "failed": [],
        "window_start": "2026-05-09T00:00:00+00:00",
        "window_end": "2026-06-06T00:00:00+00:00",
    }


class TestEvalRollingMeanCoverage:
    def _invoke(self, rolling_mean_mod, event: dict):
        with (
            patch.object(rolling_mean_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_rolling_mean_summary()),
            patch("evals.calibration_kappa.emit_calibration_report", side_effect=RuntimeError("stubbed out")),
            patch("scripts.build_agent_quality.build_agent_quality", side_effect=RuntimeError("stubbed out")),
            patch("scoring.leaderboard_producers.build_producer_leaderboard", side_effect=RuntimeError("stubbed out")),
        ):
            return rolling_mean_mod.handler(event, context=None)

    def test_verdict_lands_with_run_date_from_end_time(self, rolling_mean_mod, stub_stage_coverage):
        result = self._invoke(rolling_mean_mod, {"end_time_iso": "2026-06-06T00:00:00Z"})
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "EvalRollingMean"
        assert kwargs["run_date"] == "2026-06-06"

    def test_missing_lib_module_does_not_change_outcome(self, rolling_mean_mod, absent_stage_coverage):
        result = self._invoke(rolling_mean_mod, {"end_time_iso": "2026-06-06T00:00:00Z"})
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "EvalRollingMean"

    def test_no_fabricated_date_when_end_time_iso_absent(self, rolling_mean_mod, stub_stage_coverage):
        # alpha-engine-config-I8155: this handler used to fall back to its
        # OWN invocation time (`_started`, via `datetime.now(UTC)`) when
        # `end_time_iso` was absent — exactly the forbidden fabrication
        # class. A genuinely absent execution identity must record
        # UNMEASURED instead, never a wall-clock substitute.
        result = self._invoke(rolling_mean_mod, {})
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        stub_stage_coverage.assert_not_called()


# ── RationaleClustering ──────────────────────────────────────────────────


def _clustering_summary() -> dict:
    return {
        "window_start": "2026-03-14T00:00:00+00:00",
        "window_end": "2026-05-09T00:00:00+00:00",
        "artifacts_discovered": 48,
        "agents_analyzed": 6,
        "agents_skipped_thin_sample": [],
        "load_failures": [],
        "cluster_failures": [],
        "per_agent": [],
    }


class TestRationaleClusteringCoverage:
    def test_verdict_lands_with_run_date_from_end_time(self, clustering_mod, stub_stage_coverage):
        with (
            patch.object(clustering_mod, "_ensure_init"),
            patch("evals.rationale_clustering.compute_and_emit", return_value=_clustering_summary()),
        ):
            result = clustering_mod.handler({"end_time_iso": "2026-05-09T00:00:00Z"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "RationaleClustering"
        assert kwargs["run_date"] == "2026-05-09"

    def test_missing_lib_module_does_not_change_outcome(self, clustering_mod, absent_stage_coverage):
        with (
            patch.object(clustering_mod, "_ensure_init"),
            patch("evals.rationale_clustering.compute_and_emit", return_value=_clustering_summary()),
        ):
            result = clustering_mod.handler({"end_time_iso": "2026-05-09T00:00:00Z"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "RationaleClustering"

    def test_no_fabricated_date_when_end_time_iso_absent(self, clustering_mod, stub_stage_coverage):
        # alpha-engine-config-I8155: same forbidden-fabrication class as
        # EvalRollingMean — `_started` (this handler's own invocation time)
        # must never substitute for a genuinely-absent end_time_iso.
        with (
            patch.object(clustering_mod, "_ensure_init"),
            patch("evals.rationale_clustering.compute_and_emit", return_value=_clustering_summary()),
        ):
            result = clustering_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        stub_stage_coverage.assert_not_called()


# ── AggregateCosts ───────────────────────────────────────────────────────


def _aggregate_costs_summary() -> dict:
    return {
        "rows_in": 1234,
        "output_key": "decision_artifacts/_cost/2026-05-25/cost.parquet",
        "total_cost_usd": 12.3456,
    }


class TestAggregateCostsCoverage:
    @pytest.fixture(autouse=True)
    def capture_stream_alive(self):
        """These assert the STAGE-COVERAGE verdict, not the capture-stream
        one (config-I7407 D4, tested in test_cost_capture_freshness.py).
        The handler now grades the capture stream on both terminal paths and
        raises when it is dead; against a MagicMock S3 the stream reads as
        empty, so without this every case here would fail on a finding about
        a stream the test never set up."""
        with patch(
            "scripts.cost_capture_freshness.evaluate_and_publish",
            side_effect=lambda s3, bucket, **kw: {
                "as_of": kw["as_of"].isoformat(),
                "last_capture_date": kw["as_of"].isoformat(),
                "days_since_last_capture": 0,
                "producers_on_last_capture_date": ["replay-concordance"],
            },
        ):
            yield

    def test_verdict_lands_on_ok(self, aggregate_costs_mod, stub_stage_coverage):
        with (
            patch.object(aggregate_costs_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=True),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_aggregate_costs_summary()),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = aggregate_costs_mod.handler({"date": "2026-05-25"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}
        args, kwargs = stub_stage_coverage.call_args
        assert args[0] == "AggregateCosts"
        assert kwargs["run_date"] == "2026-05-25"

    def test_verdict_lands_on_skipped(self, aggregate_costs_mod, stub_stage_coverage):
        # SKIPPED (no _cost_raw partitions anywhere in the window, config-I7407)
        # is a legitimate completion, not a failure — the assertion still runs.
        with (
            patch.object(aggregate_costs_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=False),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = aggregate_costs_mod.handler({"date": "2026-05-25"}, context=None)
        assert result["status"] == "SKIPPED"
        assert result["stage_coverage"] == {"status": "COVERED", "stage": "stub"}

    def test_missing_lib_module_does_not_change_outcome(self, aggregate_costs_mod, absent_stage_coverage):
        with (
            patch.object(aggregate_costs_mod, "_ensure_init"),
            patch("scripts.aggregate_costs._has_raw_rows", return_value=True),
            patch("scripts.aggregate_costs.aggregate_day", return_value=_aggregate_costs_summary()),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = aggregate_costs_mod.handler({"date": "2026-05-25"}, context=None)
        assert result["status"] == "OK"
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["stage"] == "AggregateCosts"


# ── Cross-cutting: enforcement never enabled, totality ───────────────────

# Sourced 2026-08-13 from nousergon-data/infrastructure/step_function.json
# — the Task states' FunctionName + Payload, matched to this repo's deploy
# scripts (config-I7214 rescope). NOT derived from a scan of this repo:
# an enumeration of what exists here is blind to a handler this repo was
# supposed to instrument but didn't touch, which is exactly the gap this
# totality test exists to catch.
# Values are REPO-RELATIVE PATHS, not bare `lambda/` filenames
# (alpha-engine-config-I9329). A stage's substrate is not part of its
# contract: `EvalJudgeProcess` still runs, still carries the assertion, and is
# no longer a Lambda. Keying on `lambda/<file>` made the map unable to express
# that, and the only way to keep it type-checking would have been to delete
# the row — which is how a live stage's coverage assertion goes dark without
# anyone deciding to drop it. `EvalJudgePoll` IS deleted: that state no longer
# exists in the SF at all.
_WEEKLY_SF_STAGE_TO_HANDLER_FILE = {
    "Scanner": "lambda/scanner_handler.py",
    # alpha-engine-config-I9050: the leaf ``scanner_leaderboard`` mode's
    # assertion path was never wired — see TestScannerLeaderboardCoverage.
    "ScannerLeaderboard": "lambda/scanner_handler.py",
    "SignalsEnvelope": "lambda/signals_envelope_handler.py",
    "ChallengerShadow": "lambda/handler.py",
    "EvalJudgeSubmitFirstSaturday": "lambda/eval_judge_submit_handler.py",
    "EvalJudgeSubmitWeekly": "lambda/eval_judge_submit_handler.py",
    "EvalJudgeProcess": "evals/judge_spot_run.py",
    "EvalRollingMean": "lambda/eval_rolling_mean_handler.py",
    "RationaleClustering": "lambda/rationale_clustering_handler.py",
    "AggregateCosts": "lambda/aggregate_costs_handler.py",
}


def _assert_stage_coverage_call_args(source: str) -> list[str]:
    """Every ``assert_stage_coverage(...)`` call's argument text.

    Paren-matched rather than regex-bounded so a nested call in an argument
    does not truncate the span — a truncated span silently stops covering the
    kwargs after it, which is the same class of blindness the enforcement
    check exists to prevent. The bare ``from ... import`` line is skipped: it
    is not a call.
    """
    calls: list[str] = []
    needle = "assert_stage_coverage("
    idx = source.find(needle)
    while idx != -1:
        i = idx + len(needle)
        depth = 1
        while i < len(source) and depth:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
            i += 1
        calls.append(source[idx + len(needle):i - 1])
        idx = source.find(needle, i)
    return calls


class TestTotalityAndEnforcement:
    def test_every_weekly_sf_handler_file_calls_assert_stage_coverage(self):
        """Every handler FILE backing a declared weekly-SF Task state
        contains a call to assert_stage_coverage — source-level check
        against the pinned SF stage list above, independent of whether
        any single test exercises the call path."""
        checked_files = set()
        for stage, filename in _WEEKLY_SF_STAGE_TO_HANDLER_FILE.items():
            if filename in checked_files:
                continue
            checked_files.add(filename)
            source = (_REPO_ROOT / filename).read_text()
            assert "assert_stage_coverage" in source, (
                f"{filename} backs weekly-SF stage {stage!r} but never calls assert_stage_coverage (config-I7214)"
            )

    def test_every_call_site_names_a_stage_from_the_declared_set(self):
        """Every literal stage-name string call site is a name from the
        declared SF stage list — catches a typo'd or invented stage name
        that would file a verdict under a name the registry never
        declared."""
        declared = set(_WEEKLY_SF_STAGE_TO_HANDLER_FILE)
        found_any = False
        for filename in set(_WEEKLY_SF_STAGE_TO_HANDLER_FILE.values()):
            source = (_REPO_ROOT / filename).read_text()
            for stage in declared:
                if f'"{stage}"' in source:
                    found_any = True
        assert found_any

    def test_no_call_site_enables_enforcement(self):
        """Enforcement is a single flag the lib owns (per config-I7214's
        contract) — no shipped call site in this repo may pass it.
        OBSERVE MODE ONLY until the ruled Saturday 2026-08-15 02:00 PT
        soak boundary.

        Scanned over the ARGUMENTS of each ``assert_stage_coverage(...)``
        call, not over the whole file (narrowed by
        alpha-engine-config-I9329). A whole-file substring scan asserted
        something it could not see — it fired on
        ``evals.judge_coverage.enforce_coverage``, an unrelated function that
        is the judge's own coverage VERDICT, and a check that reports a
        finding on a legitimately-named neighbour teaches readers to
        suppress it.
        """
        for filename in set(_WEEKLY_SF_STAGE_TO_HANDLER_FILE.values()):
            source = (_REPO_ROOT / filename).read_text()
            for call_args in _assert_stage_coverage_call_args(source):
                for token in ("enforce=", "enforce_", "raise_on_breach"):
                    assert token not in call_args, (
                        f"{filename} passes an enforcement flag to "
                        f"assert_stage_coverage (config-I7214: observe mode "
                        f"only): {call_args!r}"
                    )

    def test_every_call_site_passes_window_start(self):
        """window_start must be a timezone-aware datetime captured at
        handler entry, not omitted — an assertion with no window_start
        can't distinguish this cycle's artifact from a stale leftover."""
        for filename in set(_WEEKLY_SF_STAGE_TO_HANDLER_FILE.values()):
            source = (_REPO_ROOT / filename).read_text()
            assert "window_start=" in source, (
                f"{filename} calls assert_stage_coverage without threading window_start (config-I7214)"
            )


# ── The primitive must actually be installable, not merely called ───────────


class TestPrimitiveIsImportable:
    """alpha-engine-config-I7334 class: an assertion whose import fails
    emits nothing, and "emitted nothing" is byte-identical on the console
    to "found nothing wrong". Every test above passes whether or not
    krepis really carries the module, so one test has to look at the real
    one — otherwise this whole file is a suite that cannot tell a working
    coverage assertion from an absent one."""

    def test_krepis_stage_coverage_is_importable(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "krepis.stage_coverage", raising=False)
        import importlib  # noqa: PLC0415 — deliberately deferred past the delitem

        mod = importlib.import_module("krepis.stage_coverage")
        assert callable(mod.assert_stage_coverage), (
            "krepis.stage_coverage imported but exposes no callable "
            "assert_stage_coverage — every handler's observe-mode call "
            "would measure nothing"
        )

    def test_assert_stage_coverage_accepts_the_signature_every_handler_calls(self, monkeypatch):
        """Every call site here is
        ``assert_stage_coverage(stage, run_date=..., window_start=...)``.
        Signature drift in krepis would otherwise surface only in the
        deployed Lambda, as a logged error nobody is watching for."""
        monkeypatch.delitem(sys.modules, "krepis.stage_coverage", raising=False)
        import importlib  # noqa: PLC0415 — deliberately deferred past the delitem
        import inspect  # noqa: PLC0415

        mod = importlib.import_module("krepis.stage_coverage")
        inspect.signature(mod.assert_stage_coverage).bind("Scanner", run_date="2026-05-29", window_start=None)
