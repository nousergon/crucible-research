"""Unit tests for ``data/scanner_orchestrator.py`` (ROADMAP L1995 Phase 1).

Covers the artifact contract end-to-end with mocked S3 + feature store
+ scanner primitives. The Phase 3 soak will compare the orchestrator's
output against Research Lambda's internal scanner; these tests pin the
ARTIFACT SHAPE, not numerical scanner behavior (which lives in
``test_scanner.py``).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_s3_get(payload: bytes | dict | None):
    """Build a mock S3 client whose get_object returns the given body."""
    if isinstance(payload, dict):
        body_bytes = json.dumps(payload).encode("utf-8")
    elif payload is None:
        # Mock that raises on get_object — simulates "key absent".
        client = MagicMock()
        client.get_object.side_effect = Exception("NoSuchKey")
        return client
    else:
        body_bytes = payload
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = body_bytes
    client.get_object.return_value = {"Body": body}
    return client


class TestReadPriorSignalsUniverseTickers:
    def test_empty_when_pointer_missing(self):
        from data.scanner_orchestrator import (
            _read_prior_signals_universe_tickers,
        )
        s3 = _make_s3_get(None)  # raises on get_object
        pop, picks, date = _read_prior_signals_universe_tickers(
            s3, "test-bucket",
        )
        assert pop == []
        assert picks == []
        assert date is None

    def test_returns_population_and_scanner_picks(self):
        from data.scanner_orchestrator import (
            _read_prior_signals_universe_tickers,
        )
        # Mock two get_object calls: pointer + the signals.json itself.
        pointer = {"s3_key": "signals/2026-05-23/signals.json", "date": "2026-05-23"}
        signals = {
            "population": ["AAPL", "GOOG"],
            "universe": [
                {"ticker": "AAPL"}, {"ticker": "GOOG"},
                {"ticker": "AMD"}, {"ticker": "BNY"},
            ],
        }
        s3 = MagicMock()
        s3.get_object.side_effect = [
            {"Body": MagicMock(read=lambda: json.dumps(pointer).encode())},
            {"Body": MagicMock(read=lambda: json.dumps(signals).encode())},
        ]
        pop, picks, date = _read_prior_signals_universe_tickers(
            s3, "test-bucket",
        )
        assert pop == ["AAPL", "GOOG"]
        # universe - population = scanner picks
        assert set(picks) == {"AMD", "BNY"}
        assert date == "2026-05-23"

    def test_universe_as_flat_list_of_strings(self):
        # Some historical signals.json have universe as list of strings,
        # not dicts. Orchestrator must handle both.
        from data.scanner_orchestrator import (
            _read_prior_signals_universe_tickers,
        )
        pointer = {"s3_key": "signals/old/signals.json", "date": "2026-04-01"}
        signals = {
            "population": ["AAPL"],
            "universe": ["AAPL", "MSFT", "NVDA"],
        }
        s3 = MagicMock()
        s3.get_object.side_effect = [
            {"Body": MagicMock(read=lambda: json.dumps(pointer).encode())},
            {"Body": MagicMock(read=lambda: json.dumps(signals).encode())},
        ]
        pop, picks, date = _read_prior_signals_universe_tickers(
            s3, "test-bucket",
        )
        assert pop == ["AAPL"]
        assert set(picks) == {"MSFT", "NVDA"}


class TestBuildCandidatesArtifact:
    def _setup_patches(
        self,
        *,
        constituents: list[str],
        sector_map: dict[str, str],
        fs_features: dict[str, dict],
        daily_closes: dict[str, float],
        quant_result: list[dict],
        prior_pop: list[str] | None = None,
        prior_picks: list[str] | None = None,
        prior_date: str | None = None,
    ) -> dict[str, Any]:
        """Patch every external dependency the orchestrator touches.
        Returns the patch dict so the test can wrap them via ExitStack.
        """
        return {
            "fetch_sp500_sp400_with_sectors": patch(
                "data.fetchers.price_fetcher.fetch_sp500_sp400_with_sectors",
                return_value=(constituents, sector_map),
            ),
            "read_latest_features": patch(
                "data.fetchers.feature_store_reader.read_latest_features",
                return_value=fs_features,
            ),
            "read_latest_daily_closes": patch(
                "data.fetchers.feature_store_reader.read_latest_daily_closes",
                return_value=daily_closes,
            ),
            "compute_technical_score": patch(
                "scoring.technical.compute_technical_score",
                # Return a stable score; the real scanner is exercised in
                # the run_quant_filter mock below.
                return_value=70.0,
            ),
            "run_quant_filter": patch(
                "data.scanner.run_quant_filter",
                return_value=quant_result,
            ),
            "_read_prior_signals_universe_tickers": patch(
                "data.scanner_orchestrator._read_prior_signals_universe_tickers",
                return_value=(
                    prior_pop or [], prior_picks or [], prior_date,
                ),
            ),
        }

    def _apply_patches(self, patches: dict[str, Any]):
        """Helper: apply all patches as context managers."""
        from contextlib import ExitStack
        stack = ExitStack()
        for name, p in patches.items():
            stack.enter_context(p)
        return stack

    def test_artifact_shape_matches_plan_doc_contract(self):
        from data.scanner_orchestrator import build_candidates_artifact

        constituents = [f"T{i}" for i in range(900)]  # ≥800 floor
        sector_map = {t: "Technology" for t in constituents}
        # All tickers in feature store — single-source-of-truth happy path.
        fs_features = {t: {"rsi_14": 55.0, "atr_14_pct": 0.02} for t in constituents}
        daily_closes = {t: 100.0 for t in constituents}
        quant_result = [{"ticker": f"T{i}"} for i in range(60)]

        patches = self._setup_patches(
            constituents=constituents, sector_map=sector_map,
            fs_features=fs_features, daily_closes=daily_closes,
            quant_result=quant_result,
            prior_pop=["T0", "T1"],
            prior_picks=["T2", "T3"],
            prior_date="2026-05-23",
        )
        with self._apply_patches(patches):
            artifact = build_candidates_artifact(
                run_date="2026-05-30",
                s3_client=MagicMock(),
                bucket="test-bucket",
            )

        # Required top-level fields per plan-doc §3 artifact contract.
        for key in (
            "run_date", "scanner_version", "generated_at",
            "population_tickers", "scanner_tickers", "agent_input_set",
            "scanner_eval_log", "filters_applied", "stats",
        ):
            assert key in artifact, f"artifact missing field: {key}"

        assert artifact["run_date"] == "2026-05-30"
        assert artifact["scanner_version"]  # non-empty
        assert artifact["population_tickers"] == ["T0", "T1"]
        assert artifact["scanner_tickers"] == [f"T{i}" for i in range(60)]
        # agent_input_set = UNION of population + top-50 scanner picks
        # (matches research_graph.py:734's `population + scanner[:50]`).
        # In this fixture pop ⊂ scanner_top_50, so the union dedups to
        # exactly the top-50 scanner set — 50 elements, not 52.
        assert len(artifact["agent_input_set"]) == 50
        assert "T0" in artifact["agent_input_set"]
        assert "T49" in artifact["agent_input_set"]
        # T50..T59 (scanner positions 50-59) are NOT in agent_input_set —
        # only the top-50 scanner picks are passed to agents per the
        # research_graph convention. Pin this invariant.
        assert "T50" not in artifact["agent_input_set"]

    def test_new_vs_prior_cycle_diff_correctness(self):
        from data.scanner_orchestrator import build_candidates_artifact

        constituents = [f"T{i}" for i in range(900)]
        sector_map = {t: "Technology" for t in constituents}
        fs_features = {t: {"rsi_14": 55.0} for t in constituents}
        daily_closes = {t: 100.0 for t in constituents}
        # This cycle's scanner: T0..T9
        quant_result = [{"ticker": f"T{i}"} for i in range(10)]

        # Prior cycle's scanner: T0, T1, X1, X2
        # → new this cycle: T2..T9 (8 tickers)
        # → dropped this cycle: X1, X2 (2 tickers)
        patches = self._setup_patches(
            constituents=constituents, sector_map=sector_map,
            fs_features=fs_features, daily_closes=daily_closes,
            quant_result=quant_result,
            prior_picks=["T0", "T1", "X1", "X2"],
            prior_date="2026-05-23",
        )
        with self._apply_patches(patches):
            artifact = build_candidates_artifact(
                run_date="2026-05-30",
                s3_client=MagicMock(),
                bucket="test-bucket",
            )

        new = artifact["stats"]["new_vs_prior_cycle"]
        dropped = artifact["stats"]["dropped_vs_prior_cycle"]
        assert set(new) == {f"T{i}" for i in range(2, 10)}
        assert set(dropped) == {"X1", "X2"}

    def test_baseline_missing_flag_on_cold_start(self):
        from data.scanner_orchestrator import build_candidates_artifact

        constituents = [f"T{i}" for i in range(900)]
        sector_map = {t: "Technology" for t in constituents}
        fs_features = {t: {"rsi_14": 55.0} for t in constituents}

        # No prior signals.json — diff fields should be empty + flag set.
        patches = self._setup_patches(
            constituents=constituents, sector_map=sector_map,
            fs_features=fs_features, daily_closes={},
            quant_result=[{"ticker": "T0"}],
            prior_date=None,  # None → baseline_missing
        )
        with self._apply_patches(patches):
            artifact = build_candidates_artifact(
                run_date="2026-05-30",
                s3_client=MagicMock(),
                bucket="test-bucket",
            )

        assert artifact["stats"]["baseline_missing"] is True
        assert artifact["stats"]["new_vs_prior_cycle"] == []
        assert artifact["stats"]["dropped_vs_prior_cycle"] == []

    def test_raises_when_constituents_below_floor(self):
        from data.scanner_orchestrator import (
            build_candidates_artifact, ScannerOrchestratorError,
        )

        # Too few constituents — orchestrator must refuse rather than
        # silently produce a malformed artifact (no-silent-fails).
        constituents = ["AAPL", "MSFT"]
        sector_map = {"AAPL": "Tech", "MSFT": "Tech"}
        patches = self._setup_patches(
            constituents=constituents, sector_map=sector_map,
            fs_features={}, daily_closes={}, quant_result=[],
        )
        with self._apply_patches(patches):
            with pytest.raises(ScannerOrchestratorError, match="constituents.json"):
                build_candidates_artifact(
                    run_date="2026-05-30",
                    s3_client=MagicMock(),
                    bucket="test-bucket",
                )

    def test_raises_when_feature_store_empty(self):
        from data.scanner_orchestrator import (
            build_candidates_artifact, ScannerOrchestratorError,
        )

        constituents = [f"T{i}" for i in range(900)]
        sector_map = {t: "Tech" for t in constituents}
        # Empty feature store — upstream DataPhase1 didn't run.
        patches = self._setup_patches(
            constituents=constituents, sector_map=sector_map,
            fs_features={}, daily_closes={}, quant_result=[],
        )
        with self._apply_patches(patches):
            with pytest.raises(ScannerOrchestratorError, match="feature store"):
                build_candidates_artifact(
                    run_date="2026-05-30",
                    s3_client=MagicMock(),
                    bucket="test-bucket",
                )

    def test_filters_applied_records_resolved_params(self):
        from data.scanner_orchestrator import build_candidates_artifact

        constituents = [f"T{i}" for i in range(900)]
        sector_map = {t: "Tech" for t in constituents}
        fs_features = {t: {"rsi_14": 55.0} for t in constituents}
        patches = self._setup_patches(
            constituents=constituents, sector_map=sector_map,
            fs_features=fs_features, daily_closes={},
            quant_result=[],
        )
        with self._apply_patches(patches):
            artifact = build_candidates_artifact(
                run_date="2026-05-30",
                s3_client=MagicMock(),
                bucket="test-bucket",
            )

        fa = artifact["filters_applied"]
        # Pin the schema — these keys are the operationally interesting
        # snapshot of THIS cycle's S3-configured params.
        for key in (
            "min_avg_volume", "min_price", "max_atr_pct", "tech_score_min",
        ):
            assert key in fa


class TestScannerEvalLogPassthrough:
    """config#1458: candidates.json must carry run_quant_filter's per-ticker
    eval log so the Research Lambda (a SEPARATE process from the one that
    calls run_quant_filter here) can join it in without relying on the
    process-local ``run_quant_filter._last_eval_log`` module-attribute stash.
    """

    def _setup_patches(self, **kwargs):
        return TestBuildCandidatesArtifact._setup_patches(self, **kwargs)

    def _apply_patches(self, patches):
        return TestBuildCandidatesArtifact._apply_patches(self, patches)

    def _build_with_eval_log(self, eval_log, *, quant_result=None):
        """Build the artifact with ``run_quant_filter`` mocked to mimic its
        real side effect: stashing ``_last_eval_log`` on the (module-level)
        callable itself. Patching ``data.scanner.run_quant_filter`` replaces
        that name with a MagicMock, so the attribute must be set on the mock
        instance — exactly what ``data.scanner_orchestrator``'s
        ``getattr(run_quant_filter, "_last_eval_log", ...)`` reads."""
        from data.scanner_orchestrator import build_candidates_artifact

        constituents = [f"T{i}" for i in range(900)]
        sector_map = {t: "Technology" for t in constituents}
        fs_features = {t: {"rsi_14": 55.0} for t in constituents}

        patches = self._setup_patches(
            constituents=constituents, sector_map=sector_map,
            fs_features=fs_features, daily_closes={},
            quant_result=quant_result if quant_result is not None else [],
        )
        with self._apply_patches(patches):
            import data.scanner as scanner_mod
            if eval_log is not None:
                scanner_mod.run_quant_filter._last_eval_log = eval_log
            else:
                # Simulate the stash never having been set (fresh mock).
                if hasattr(scanner_mod.run_quant_filter, "_last_eval_log"):
                    del scanner_mod.run_quant_filter._last_eval_log
            artifact = build_candidates_artifact(
                run_date="2026-05-30",
                s3_client=MagicMock(),
                bucket="test-bucket",
            )
        return artifact

    def test_artifact_captures_eval_log_stashed_by_run_quant_filter(self):
        """build_candidates_artifact must copy run_quant_filter's stashed
        eval log into the artifact's ``scanner_eval_log`` field."""
        eval_log = [
            {"ticker": "T0", "quant_filter_pass": 1, "scan_path": "momentum"},
            {"ticker": "T1", "quant_filter_pass": 0,
             "filter_fail_reason": "liquidity"},
        ]
        artifact = self._build_with_eval_log(
            eval_log, quant_result=[{"ticker": "T0"}],
        )
        assert artifact["scanner_eval_log"] == eval_log

    def test_artifact_eval_log_empty_when_stash_unavailable(self):
        """No _last_eval_log stashed (e.g. run_quant_filter mocked out
        entirely in a test, or a future contract break) — must degrade to
        [] rather than raise."""
        artifact = self._build_with_eval_log(None, quant_result=[])
        assert artifact["scanner_eval_log"] == []

    def test_eval_log_numpy_scalars_are_cast_json_safe(self):
        """Defensive belt-and-suspenders: if a future change to the
        eval-log inputs leaks a numpy scalar in, build_candidates_artifact
        must still produce a JSON-serializable artifact (write_candidates_artifact
        calls json.dumps on it)."""
        import numpy as np
        from data.scanner_orchestrator import write_candidates_artifact

        eval_log = [
            {"ticker": "T0", "quant_filter_pass": np.int64(1),
             "tech_score": np.float64(72.5), "avg_volume_20d": np.float32(1e6)},
        ]
        artifact = self._build_with_eval_log(
            eval_log, quant_result=[{"ticker": "T0"}],
        )

        # Values still numerically correct, but now plain python scalars.
        rec = artifact["scanner_eval_log"][0]
        assert rec["quant_filter_pass"] == 1
        assert isinstance(rec["quant_filter_pass"], int)
        assert rec["tech_score"] == pytest.approx(72.5)
        assert isinstance(rec["tech_score"], float)
        assert isinstance(rec["avg_volume_20d"], float)

        # And json.dumps (what write_candidates_artifact actually calls)
        # must not raise.
        s3 = MagicMock()
        write_candidates_artifact(artifact, s3_client=s3, bucket="test-bucket")
        call_kwargs = s3.put_object.call_args.kwargs
        body = json.loads(call_kwargs["Body"])
        assert body["scanner_eval_log"][0]["quant_filter_pass"] == 1

    def test_scanner_eval_log_round_trips_through_write_and_load(self):
        """End-to-end: build -> write -> read back via
        ArchiveManager.load_candidates_json must preserve scanner_eval_log
        byte-for-byte (the eval-log entries are plain JSON scalars)."""
        from data.scanner_orchestrator import write_candidates_artifact
        from archive.manager import ArchiveManager

        eval_log = [
            {"ticker": "T0", "quant_filter_pass": 1, "scan_path": "momentum",
             "tech_score": 81.3, "sector": "Technology"},
            {"ticker": "T1", "quant_filter_pass": 0,
             "filter_fail_reason": "rank_cutoff"},
        ]
        artifact = self._build_with_eval_log(
            eval_log, quant_result=[{"ticker": "T0"}],
        )

        # Fake S3 store: capture the put_object body, serve it back on get_object.
        store: dict[str, bytes] = {}

        def _put_object(Bucket, Key, Body, ContentType):
            store[Key] = Body if isinstance(Body, bytes) else bytes(Body)

        s3 = MagicMock()
        s3.put_object.side_effect = _put_object
        write_candidates_artifact(artifact, s3_client=s3, bucket="test-bucket")

        class _AM:
            def _s3_get(self, key):
                data = store.get(key)
                return data.decode("utf-8") if data is not None else None

        loaded = ArchiveManager.load_candidates_json(_AM(), "2026-05-30")
        assert loaded["scanner_eval_log"] == eval_log


class TestWriteCandidatesArtifact:
    def test_writes_to_canonical_key(self):
        from data.scanner_orchestrator import write_candidates_artifact

        s3 = MagicMock()
        artifact = {
            "run_date": "2026-05-30",
            "scanner_version": "v1.0",
            "generated_at": "2026-05-30T09:00:00+00:00",
            "population_tickers": ["AAPL"],
            "scanner_tickers": ["AMD"],
            "agent_input_set": ["AAPL", "AMD"],
            "filters_applied": {},
            "stats": {
                "universe_size": 900,
                "post_scanner": 1,
                "new_vs_prior_cycle": [],
                "dropped_vs_prior_cycle": [],
            },
        }

        key = write_candidates_artifact(artifact, s3_client=s3, bucket="b")

        assert key == "candidates/2026-05-30/candidates.json"
        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "b"
        assert call_kwargs["Key"] == "candidates/2026-05-30/candidates.json"
        assert call_kwargs["ContentType"] == "application/json"
        # Round-trip the body and confirm it matches the artifact.
        body = json.loads(call_kwargs["Body"])
        assert body == artifact
