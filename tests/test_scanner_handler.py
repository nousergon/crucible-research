"""Unit tests for the scanner Lambda handler (ROADMAP L1995 Phase 1)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "scanner_handler.py"


def _load_handler_module():
    """Import lambda/scanner_handler.py without using ``lambda`` as a
    package name (Python keyword)."""
    module_name = "lambda_scanner_handler"
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


def _ok_artifact() -> dict:
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


@pytest.fixture(autouse=True)
def stub_universe_membership():
    """Stub the load-bearing membership producer for every handler test.

    Unlike the other post-candidates writes, the membership producer is NOT
    fail-soft (alpha-engine-config-I4818) — an unstubbed call would reach real
    S3 and raise, failing every test in this module for a reason unrelated to
    what each one asserts. Tests that care about membership behavior patch the
    same target again inside their own `with`, which takes precedence.

    The factor-loading read is stubbed for the same reason, one layer along:
    ``build_shadow_candidate_artifacts`` now reads the DAILY ArcticDB source,
    and arcticdb ABORTS the interpreter (not a catchable exception) when it
    cannot reach a store, so the builder's own fail-soft ``except`` cannot
    contain it. Returning ``{}`` exercises the real degrade path — the
    builder logs and returns ``{}`` without touching the live artifact.

    This only surfaced in a FULL-suite run: ``run_quant_filter`` stashes
    ``_last_eval_log`` as a module attribute, so these tests reach the
    shadow path only once an earlier file has populated it. Alone, the
    empty stash short-circuits before the read.

    The weekly cut ledger (alpha-engine-config-I8264) is stubbed for the same
    reason one layer further along: it reads the membership artifact this run
    just wrote and then prices a week out of ArcticDB, neither of which exists
    under a MagicMock S3. Its own contract tests live in
    ``tests/test_weekly_ledger_wiring.py``; here it returns the commonest real
    outcome — a run where no week closed — so the handler tests exercise the
    wiring without asserting anything about the ledger's own behaviour. Its
    membership READ is stubbed alongside it: the handler re-reads the artifact
    it just wrote to hand the ledger the cut it must measure against, and under
    a MagicMock S3 that read raises before the stub above is ever reached.
    """
    with (
        patch(
            "scoring.universe_membership.compute_and_write_universe_membership",
            return_value="universe_membership/2026-05-29/membership.json",
        ),
        patch(
            "data.fetchers.feature_store_reader.read_latest_factor_loadings",
            return_value={},
        ),
        patch(
            "scoring.universe_membership.read_latest_membership",
            return_value={"cut_effective_date": "2026-05-29", "turnover": None},
        ),
        patch(
            "scoring.weekly_ledger.record_completed_week",
            return_value={"status": "skipped", "reason": "no_week_closed"},
        ),
    ):
        yield


class TestHandler:
    def test_ok_when_orchestrator_succeeds(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-30/candidates.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"},
                context=None,
            )
        assert result["status"] == "OK"
        # 2026-05-30 (Sat) normalizes to the 2026-05-29 (Fri) trading day —
        # candidates.json keys by trading day to match Research (DATE_CONVENTIONS).
        assert result["date"] == "2026-05-29"
        # Summary surfaces the operationally interesting counts.
        assert result["summary"]["scanner_tickers"] == 3
        assert result["summary"]["population_tickers"] == 2
        assert result["summary"]["new_vs_prior_cycle"] == 2
        assert result["summary"]["dropped_vs_prior_cycle"] == 1
        assert result["summary"]["s3_key"] == "candidates/2026-05-30/candidates.json"

    def test_error_when_orchestrator_precondition_fails(self, handler_mod):
        from data.scanner_orchestrator import ScannerOrchestratorError

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch(
                "data.scanner_orchestrator.build_candidates_artifact",
                side_effect=ScannerOrchestratorError("feature store empty"),
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"},
                context=None,
            )
        assert result["status"] == "ERROR"
        assert "feature store" in result["error"]

    def test_error_when_orchestrator_raises_unexpected(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", side_effect=RuntimeError("S3 unreachable")),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"},
                context=None,
            )
        assert result["status"] == "ERROR"
        assert "S3 unreachable" in result["error"]

    def test_error_when_s3_write_fails(self, handler_mod):
        # Build succeeded but the S3 write itself blew up — must surface
        # as ERROR (the artifact was lost; SF Catch handles).
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch("data.scanner_orchestrator.write_candidates_artifact", side_effect=RuntimeError("PutObject denied")),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler(
                {"run_date": "2026-05-30"},
                context=None,
            )
        assert result["status"] == "ERROR"
        assert "S3 write failed" in result["error"]
        assert "PutObject denied" in result["error"]

    def test_error_when_run_date_missing(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "run_date" in result["error"]

    def test_error_when_run_date_invalid(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"):
            result = handler_mod.handler({"run_date": "bad"}, context=None)
        assert result["status"] == "ERROR"
        assert "run_date" in result["error"] or "bad" in result["error"]

    def test_dry_run_short_circuits_before_s3(self, handler_mod):
        # dry_run_llm shell-run path must NOT touch S3 or call the
        # orchestrator. Mirrors the rationale_clustering / eval_judge
        # dry path used by Friday-Preflight shell runs.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact") as mock_build,
            patch("boto3.client") as mock_boto,
        ):
            result = handler_mod.handler(
                {"dry_run_llm": True, "run_date": "2026-05-30"},
                context=None,
            )
        assert result["status"] == "OK"
        assert result["dry_run"] is True
        mock_build.assert_not_called()
        mock_boto.assert_not_called()

    def test_run_date_threaded_through_to_orchestrator(self, handler_mod):
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return _ok_artifact()

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", side_effect=fake_build),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-30/candidates.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            handler_mod.handler(
                {"run_date": "2026-05-30"},
                context=None,
            )
        # Normalized Sat→Fri trading day before reaching the orchestrator.
        assert captured["run_date"] == "2026-05-29"

    def test_bucket_and_market_regime_overrides(self, handler_mod):
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            return _ok_artifact()

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", side_effect=fake_build),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-30/candidates.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            handler_mod.handler(
                {
                    "run_date": "2026-05-30",
                    "bucket": "test-bucket",
                    "market_regime": "bull",
                },
                context=None,
            )
        assert captured["bucket"] == "test-bucket"
        assert captured["market_regime"] == "bull"

    def test_run_date_normalized_to_trading_day(self, handler_mod):
        """A weekend/holiday calendar run_date is normalized to the most
        recent trading day so candidates.json lands on the SAME key Research
        reads (DATE_CONVENTIONS). The 2026-05-30 recovery failed because the
        Scanner keyed by calendar date (Sat) while Research read trading day
        (Fri). Saturday 2026-05-30 → Friday 2026-05-29; a trading-day input
        passes through unchanged."""
        for given, expected in [
            ("2026-05-30", "2026-05-29"),  # Sat → Fri
            ("2026-05-31", "2026-05-29"),  # Sun → Fri
            ("2026-05-29", "2026-05-29"),
        ]:  # Fri → Fri
            captured = {}

            def fake_build(*, _captured=captured, **kwargs):
                _captured.update(kwargs)
                return _ok_artifact()

            with (
                patch.object(handler_mod, "_ensure_init"),
                patch("data.scanner_orchestrator.build_candidates_artifact", side_effect=fake_build),
                patch(
                    "data.scanner_orchestrator.write_candidates_artifact",
                    return_value=f"candidates/{expected}/candidates.json",
                ),
                patch("boto3.client", return_value=MagicMock()),
            ):
                result = handler_mod.handler({"run_date": given}, context=None)
            assert captured["run_date"] == expected, (
                f"run_date {given} must normalize to trading day {expected}, got {captured.get('run_date')}"
            )
            assert result["date"] == expected

    def test_shadow_specs_written_and_summarized(self, handler_mod):
        # Champion/challenger OBSERVE shadows (config#1221): the handler writes
        # each challenger artifact to the isolated shadow prefix and records the
        # keys in summary.shadows — without disturbing the live OK path.
        shadow_art = {"momentum_sleeve": {"run_date": "2026-05-29", "scanner_tickers": ["A"]}}
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.build_shadow_candidate_artifacts",
                return_value=(shadow_art, {}),
            ),
            patch(
                "data.scanner_orchestrator.write_shadow_candidates_artifact",
                return_value="candidates_shadow/momentum_sleeve/2026-05-29/candidates.json",
            ),
            # config#6428: the per-spec status-record write is a SEPARATE
            # best-effort S3 write from the candidates artifact above — stub
            # it too so this test only asserts the (unchanged) `shadows`
            # summary contract.
            patch("data.scanner_orchestrator.write_shadow_status_record", return_value={}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["shadows"] == {
            "momentum_sleeve": "candidates_shadow/momentum_sleeve/2026-05-29/candidates.json"
        }
        assert "shadow_error" not in result["summary"]

    def test_shadow_failure_does_not_downgrade_live_ok(self, handler_mod):
        # A shadow build/write failure is WHOLLY fail-soft: live stays OK, the
        # failure is recorded in summary.shadow_error (no-silent-fails).
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.build_shadow_candidate_artifacts",
                side_effect=RuntimeError("loadings exploded"),
            ),
            patch("data.scanner_orchestrator.write_shadow_status_record", return_value={}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["shadows"] == {}
        assert "loadings exploded" in result["summary"]["shadow_error"]

    def test_shadow_status_record_written_for_every_challenger_spec(self, handler_mod):
        # config#6428, champion-challenger-policy.md §3: a failed shadow spec
        # gets an explicit MISS status record — not just the in-memory
        # `shadow_error` summary field (which never survives past the
        # response), and not just the WARN + observe-alert already inside
        # build_shadow_artifacts. This is the durable record.
        from data.scanner_specs import challenger_specs

        written_records = {}

        def fake_write_status(record, spec_name, run_date, **kwargs):
            written_records[spec_name] = record
            return {"dated_key": f"candidates_shadow_status/{spec_name}/{run_date}.json"}

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.build_shadow_candidate_artifacts",
                return_value=({}, {"tech_score_gate": "synthetic forced failure"}),
            ),
            patch("data.scanner_orchestrator.write_shadow_status_record", side_effect=fake_write_status),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)

        assert result["status"] == "OK"
        expected_specs = {spec.name for spec in challenger_specs()}
        assert set(written_records) == expected_specs
        record = written_records["tech_score_gate"]
        assert record["status"] == "failed"
        assert record["artifacts"][0]["status"] == "absent"
        assert "synthetic forced failure" in record["artifacts"][0]["reason"]

    def test_universe_board_written_and_summarized(self, handler_mod):
        # alpha-engine-config-I2515: the standalone Scanner path becomes a
        # universe-board producer. The handler records the written key in
        # summary.universe_board without disturbing the live OK path.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                return_value="scanner/universe/2026-05-29/universe.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["universe_board"] == {
            "status": "OK",
            "key": "scanner/universe/2026-05-29/universe.json",
        }
        assert "universe_board_error" not in result["summary"]

    def test_universe_board_failure_does_not_downgrade_live_ok(self, handler_mod):
        # A board build/write failure is WHOLLY fail-soft: live stays OK, the
        # failure is recorded in summary.universe_board_error (no-silent-fails)
        # — mirrors the shadow-artifact fail-soft contract above.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                side_effect=RuntimeError("factor profiles unreadable"),
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["universe_board"] == {"status": "error", "key": None}
        assert "factor profiles unreadable" in result["summary"]["universe_board_error"]

    def test_universe_board_receives_market_regime_and_artifact(self, handler_mod):
        # market_regime must thread through so build_pure_quant_focus_lookup
        # blends on the SAME regime the scanner used, not a stale default.
        captured = {}
        ok_artifact = _ok_artifact()

        def fake_write(artifact, **kwargs):
            captured.update(kwargs)
            captured["artifact"] = artifact
            return "scanner/universe/2026-05-29/universe.json"

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=ok_artifact),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch("data.scanner_orchestrator.write_universe_board_for_scanner_run", side_effect=fake_write),
            patch("boto3.client", return_value=MagicMock()),
        ):
            handler_mod.handler(
                {"run_date": "2026-05-30", "market_regime": "bull"},
                context=None,
            )
        assert captured["market_regime"] == "bull"
        assert captured["artifact"] is ok_artifact

    # ── Universe membership (alpha-engine-config-I4818) ──────────────────────
    # The membership artifact is LOAD-BEARING (the predictor resolves its daily
    # scoring universe from it), so its contract is the OPPOSITE of the board's
    # fail-soft one above: a write failure must fail the Scanner run.

    def test_universe_membership_written_and_summarized(self, handler_mod):
        captured = {}

        def fake_write(run_date, scanner_tickers, **kwargs):
            captured["run_date"] = run_date
            captured["scanner_tickers"] = scanner_tickers
            captured.update(kwargs)
            return "universe_membership/2026-05-29/membership.json"

        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                return_value="scanner/universe/2026-05-29/universe.json",
            ),
            patch("scoring.universe_membership.compute_and_write_universe_membership", side_effect=fake_write),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["universe_membership"] == ("universe_membership/2026-05-29/membership.json")
        # Keyed by TRADING day (same normalization as candidates.json), and fed
        # the run's own scanner cut — not a re-read of some other artifact.
        assert captured["run_date"] == "2026-05-29"
        assert captured["scanner_tickers"] == _ok_artifact()["scanner_tickers"]

    def test_universe_membership_failure_fails_the_run(self, handler_mod):
        # The regression this pins: a silently-absent membership artifact leaves
        # the predictor resolving a STALE universe (it scored a frozen
        # 2026-07-10 population for three weekly cycles). A membership failure
        # must therefore surface as a red Scanner run, NOT an OK-with-error-field
        # like the fail-soft board write.
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                return_value="scanner/universe/2026-05-29/universe.json",
            ),
            patch(
                "scoring.universe_membership.compute_and_write_universe_membership",
                side_effect=RuntimeError("no factor profiles readable"),
            ),
            patch("boto3.client", return_value=MagicMock()),
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "ERROR"
        assert "universe membership failed" in result["error"]
        assert "no factor profiles readable" in result["error"]


class TestBoardsLegibilityRollup:
    """alpha-engine-config-I7841 D3: the returned summary's ``boards`` field
    must make an in-invocation partial (one post-membership stage errored,
    or was never reached) legible without reading CloudWatch. Two stages since
    alpha-engine-config-I7813 moved ``scanner_leaderboard`` out of this
    invocation into its own weekly-SF leaf state — see
    ``TestScannerLeaderboardLeafMode``. These pin the
    ``attempted``/``completed``/``errored``/``not_attempted`` classification
    directly against each stage's real status vocabulary (``ok`` /
    ``unmeasurable`` / ``error``), not a fixture that assumes it.

    A Lambda TIMEOUT (the 2026-08-20 incident this issue exists for) still
    produces no summary at all — that failure mode is covered by the
    external freshness detector on ``research/cuts_leaderboard/{trading_day}
    .json`` (I7841 D1), not by anything testable here.
    """

    def _patched(self, **overrides):
        base = {
            "cuts": patch(
                "scoring.leaderboard_producers.build_cuts_leaderboard",
                return_value={"status": "ok", "key": "research/cuts_leaderboard/2026-05-29.json"},
            ),
            "promotion": patch(
                "scoring.cut_promotion.run_cut_promotion",
                return_value={"decision": "hold", "champion": "tech_score_top_60", "reason_code": "cooldown"},
            ),
        }
        base.update(overrides)
        return base

    def test_both_boards_complete_clean(self, handler_mod):
        patches = self._patched()
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                return_value="scanner/universe/2026-05-29/universe.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
            patches["cuts"],
            patches["promotion"],
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        boards = result["summary"]["boards"]
        assert sorted(boards["completed"]) == ["cut_promotion", "cuts_leaderboard", "weekly_ledger"]
        assert boards["attempted"] == boards["completed"]  # nothing errored or was skipped
        assert boards["errored"] == []
        assert boards["not_attempted"] == []

    def test_cuts_leaderboard_error_is_attempted_but_not_completed(self, handler_mod):
        # This is the shape of a NON-timeout partial: build_cuts_leaderboard
        # raised (caught by the handler's own fail-soft try/except) rather
        # than the invocation being killed outright. The rollup must show it
        # attempted (the block ran) and errored (it did not complete), while
        # its sibling stays clean — never let one failed board's status
        # bleed into another's.
        patches = self._patched(
            cuts=patch(
                "scoring.leaderboard_producers.build_cuts_leaderboard",
                side_effect=RuntimeError("panel read timed out"),
            ),
        )
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                return_value="scanner/universe/2026-05-29/universe.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
            patches["cuts"],
            patches["promotion"],
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        # Live path never downgrades — this stays the primary deliverable's contract.
        assert result["status"] == "OK"
        boards = result["summary"]["boards"]
        assert boards["errored"] == ["cuts_leaderboard"]
        assert "cuts_leaderboard" in boards["attempted"]
        assert "cuts_leaderboard" not in boards["completed"]
        assert "cut_promotion" in boards["completed"]
        assert result["summary"]["cuts_leaderboard"]["status"] == "error"
        # I7813: the scanner board is no longer this invocation's business at
        # all — not "completed", not "errored", ABSENT. A stage that moved out
        # must stop appearing in the rollup, or the next reader counts a board
        # this Lambda did not build.
        assert "scanner_leaderboard" not in result["summary"]
        for bucket in boards.values():
            assert "scanner_leaderboard" not in bucket

    def test_unmeasurable_cohort_counts_as_completed_not_errored(self, handler_mod):
        # A fresh date with no matured 21d outcome ships n_dates=0 /
        # status=unmeasurable — a correct, non-error outcome
        # (leaderboard_producers.py's documented cohort-gate contract). The
        # rollup must not conflate "nothing has matured yet" with a failure.
        patches = self._patched(
            cuts=patch(
                "scoring.leaderboard_producers.build_cuts_leaderboard",
                return_value={"status": "unmeasurable", "reason": "no cut had a usable width"},
            ),
        )
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                return_value="scanner/universe/2026-05-29/universe.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
            patches["cuts"],
            patches["promotion"],
        ):
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        boards = result["summary"]["boards"]
        assert "cuts_leaderboard" in boards["completed"]
        # The ledger stub reports a week that did not close — a correct outcome,
        # so it completes rather than erroring (I8264).
        assert "weekly_ledger" in boards["completed"]
        assert boards["errored"] == []


class TestScannerLeaderboardLeafMode:
    """alpha-engine-config-I7813 — ``mode`` splits the observe-only scanner
    board out of the live scan so it can be its own weekly-SF leaf state.

    The three properties that matter, and each is a real failure this pins:

    1. The default (``scan``) path must NOT build the scanner board any more.
       If it still does, the leaf is pure duplicated cost and the "moved"
       claim in the issue is false while every test still passes.
    2. The leaf must NOT run the scan. If it does, a state placed after the
       Report Card re-runs the universe scan and overwrites the day's
       ``candidates.json`` from a stage nobody thinks writes one.
    3. A board that did not get written must FAIL the task. ``build_scanner_
       leaderboard`` never raises — it returns a status dict — so without the
       wrapper's raise the SF task ends green having written nothing, which is
       the exact shape sf-pipeline-policy.md §2.3 forbids.
    """

    def _scan_patches(self, handler_mod):
        return (
            patch.object(handler_mod, "_ensure_init"),
            patch("data.scanner_orchestrator.build_candidates_artifact", return_value=_ok_artifact()),
            patch(
                "data.scanner_orchestrator.write_candidates_artifact",
                return_value="candidates/2026-05-29/candidates.json",
            ),
            patch(
                "data.scanner_orchestrator.write_universe_board_for_scanner_run",
                return_value="scanner/universe/2026-05-29/universe.json",
            ),
            patch("boto3.client", return_value=MagicMock()),
            patch(
                "scoring.leaderboard_producers.build_cuts_leaderboard",
                return_value={"status": "ok", "key": "research/cuts_leaderboard/2026-05-29.json"},
            ),
            patch(
                "scoring.cut_promotion.run_cut_promotion",
                return_value={"decision": "hold", "champion": "tech_score_top_60", "reason_code": "cooldown"},
            ),
        )

    def test_default_scan_mode_does_not_build_the_scanner_board(self, handler_mod):
        import contextlib

        with contextlib.ExitStack() as stack:
            for cm in self._scan_patches(handler_mod):
                stack.enter_context(cm)
            build = stack.enter_context(
                patch("scoring.leaderboard_producers.build_scanner_leaderboard")
            )
            result = handler_mod.handler({"run_date": "2026-05-30"}, context=None)
        assert result["status"] == "OK"
        build.assert_not_called()

    def test_leaf_mode_builds_only_the_board(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch("data.scanner_orchestrator.build_candidates_artifact") as scan,
            patch(
                "scoring.leaderboard_producers.build_scanner_leaderboard",
                return_value={"status": "ok", "key": "scanner/leaderboard/2026-05-29.json"},
            ) as build,
        ):
            result = handler_mod.handler(
                {"run_date": "2026-05-30", "mode": "scanner_leaderboard"}, context=None
            )
        scan.assert_not_called()
        build.assert_called_once()
        # The board keys on the TRADING day, not the Saturday calendar date the
        # SF passes — same normalization the scan path applies.
        assert build.call_args[0][2] == "2026-05-29"
        assert result["status"] == "OK"
        assert result["mode"] == "scanner_leaderboard"
        assert result["summary"]["leaderboard"]["key"] == "scanner/leaderboard/2026-05-29.json"

    def test_leaf_mode_raises_when_the_board_was_not_written(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch(
                "scoring.leaderboard_producers.build_scanner_leaderboard",
                return_value={"status": "error", "error": "closes panel empty"},
            ),
            pytest.raises(Exception) as exc,
        ):
            handler_mod.handler({"run_date": "2026-05-30", "mode": "scanner_leaderboard"}, context=None)
        assert "closes panel empty" in str(exc.value)
        assert type(exc.value).__name__ == "ScannerLeaderboardBuildError"

    def test_leaf_mode_treats_unmeasurable_as_a_written_outcome(self, handler_mod):
        # An immature cohort is a DECISION the producer records in the artifact
        # it writes, not a failure. Raising here would page every week until the
        # first cohort matures.
        with (
            patch.object(handler_mod, "_ensure_init"),
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
            result = handler_mod.handler(
                {"run_date": "2026-05-30", "mode": "scanner_leaderboard"}, context=None
            )
        assert result["status"] == "OK"
        assert result["summary"]["leaderboard"]["status"] == "unmeasurable"
        assert result["summary"]["leaderboard"]["reason"] == "no cohort has matured"

    def test_unknown_mode_is_an_error_not_a_silent_scan(self, handler_mod):
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("boto3.client", return_value=MagicMock()),
            patch("data.scanner_orchestrator.build_candidates_artifact") as scan,
        ):
            result = handler_mod.handler(
                {"run_date": "2026-05-30", "mode": "leaderboards"}, context=None
            )
        assert result["status"] == "ERROR"
        assert "unknown mode" in result["error"]
        scan.assert_not_called()
