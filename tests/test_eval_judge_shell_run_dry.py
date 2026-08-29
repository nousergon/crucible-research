"""Shell-run dry-path tests for the eval-judge chain + rationale
clustering Lambda handlers.

Closes the two SF keystone skip-exceptions (alpha-engine-data #260):
``skip_eval_judge`` + ``skip_rationale_clustering`` were hard-skipped
under the Friday ``shell_run`` because their handlers persisted + called
Anthropic even nominally. These tests pin the new ``dry_run_llm`` event
flag short-circuit:

- the dry flag → ZERO Anthropic client instantiation/calls
- ZERO S3 put_object / get_object
- returns a status the SF treats as OK
- import / boot (the keystone's whole point) still ran

Mirrors ``tests/test_eval_judge_handler.py`` /
``tests/test_rationale_clustering_handler.py`` handler-load style.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from evals.lambda_dry import (
    DRY_FLAG,
    DRY_SENTINEL_BATCH_ID,
    DRY_SENTINEL_PLAN_S3_KEY,
    dry_submit_result,
    is_dry,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_handler(filename: str, module_name: str):
    """Import lambda/<filename> without using ``lambda`` as a package
    name (it's a Python keyword) — mirrors the existing handler tests.
    The exec_module call IS the import/boot smoke this PR protects."""
    path = _REPO_ROOT / "lambda" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def submit_mod():
    mod = _load_handler(
        "eval_judge_submit_handler.py", "lambda_eval_judge_submit_handler"
    )
    mod._init_done = False
    yield mod


@pytest.fixture
def poll_mod():
    mod = _load_handler(
        "eval_judge_poll_handler.py", "lambda_eval_judge_poll_handler"
    )
    mod._init_done = False
    yield mod


@pytest.fixture
def process_mod():
    mod = _load_handler(
        "eval_judge_process_handler.py", "lambda_eval_judge_process_handler"
    )
    mod._init_done = False
    yield mod


@pytest.fixture
def clustering_mod():
    mod = _load_handler(
        "rationale_clustering_handler.py", "lambda_rationale_clustering_handler"
    )
    mod._init_done = False
    yield mod


# ── shared dry helper ────────────────────────────────────────────────


class TestLambdaDryHelper:
    def test_flag_key_is_dry_run_llm_verbatim(self):
        # The SF keystone rewire passes this EXACT key — pin it so a
        # rename can't silently desync the SF Payload.
        assert DRY_FLAG == "dry_run_llm"

    def test_is_dry_on_flag(self):
        assert is_dry({"dry_run_llm": True}) is True
        assert is_dry({"dry_run_llm": False}) is False
        assert is_dry({}) is False
        assert is_dry(None) is False

    def test_is_dry_on_threaded_sentinel_batch_id(self):
        # Poll/Process treat the sentinel batch_id threaded from Submit
        # as dry even if the SF only set the flag on the Submit state.
        assert is_dry({"batch_id": DRY_SENTINEL_BATCH_ID}) is True
        assert is_dry({"batch_id": "msgbatch_real"}) is False


# ── eval_judge_submit ────────────────────────────────────────────────


class TestSubmitDry:
    def test_dry_no_anthropic_no_s3_returns_empty_sentinel(self, submit_mod):
        with patch("anthropic.Anthropic") as anthropic_cls, \
             patch("boto3.client") as boto3_client, \
             patch("evals.orchestrator.build_batch_plan") as build_plan, \
             patch("evals.orchestrator.submit_batch") as submit_batch, \
             patch("evals.orchestrator._persist_client_side_skips") as skips:
            result = submit_mod.handler(
                {"date": "2026-05-16", DRY_FLAG: True}, context=None
            )

        # No external calls of any kind.
        anthropic_cls.assert_not_called()
        boto3_client.assert_not_called()
        build_plan.assert_not_called()
        submit_batch.assert_not_called()
        skips.assert_not_called()

        # Sentinel threads to Process (status EMPTY skips the poll loop).
        assert result["status"] == "EMPTY"
        assert result["batch_id"] == DRY_SENTINEL_BATCH_ID
        assert result["plan_s3_key"] == DRY_SENTINEL_PLAN_S3_KEY
        assert result["request_count"] == 0
        assert result["processing_status"] == "ended_empty"
        assert result["dry_run"] is True

    def test_no_null_anywhere_in_the_dry_submit_return(self):
        """DERIVED, not enumerated (alpha-engine-config-I9329).

        Every value the Submit dry return carries is reachable by a Step
        Functions intrinsic on the Friday shell run, and ``States.Format``
        raises ``States.Runtime`` on a null argument — a definition that works
        on Saturday and dies on Friday. ``plan_s3_key`` is the value the new
        ``ssm:sendCommand`` stage formats into its judge command, and it was
        the null. Walking the whole structure rather than restating the keys
        is what makes this cover the NEXT key someone adds as ``None``.
        """
        def _nulls(value, path="dry_submit_result"):
            if value is None:
                return [path]
            if isinstance(value, dict):
                return [p for k, v in value.items()
                        for p in _nulls(v, f"{path}.{k}")]
            if isinstance(value, (list, tuple)):
                return [p for i, v in enumerate(value)
                        for p in _nulls(v, f"{path}[{i}]")]
            return []

        offenders = _nulls(dry_submit_result("2026-05-16"))
        assert offenders == [], (
            f"null value(s) in the Submit dry return: {offenders} — a null "
            "reaching a States.Format argument raises States.Runtime on the "
            "Friday shell run only. Use a non-null sentinel."
        )

    def test_non_dry_still_takes_real_path(self, submit_mod):
        """Production Saturday SF passes no dry_run_llm — must NOT
        short-circuit."""
        with patch("anthropic.Anthropic"), \
             patch("boto3.client"), \
             patch("evals.orchestrator.build_batch_plan",
                   return_value={"capture_keys_total": 0,
                                 "skipped_unmapped": 0,
                                 "capture_partition_counts": {},
                                 "empty_trading_day_partitions": []}), \
             patch("evals.orchestrator._persist_client_side_skips",
                   return_value=(0, 0, [], [])), \
             patch("evals.orchestrator.submit_batch",
                   return_value={"batch_id": "msgbatch_x",
                                 "plan_s3_key": "k",
                                 "request_count": 1,
                                 "processing_status": "in_progress"}):
            result = submit_mod.handler({"date": "2026-05-16"}, context=None)
        # Real path ran (status OK, real batch id) — not the sentinel.
        assert result["batch_id"] == "msgbatch_x"
        assert result.get("dry_run") is not True


# ── eval_judge_poll ──────────────────────────────────────────────────


class TestPollDry:
    def test_dry_flag_no_anthropic_terminal_ended(self, poll_mod):
        with patch("anthropic.Anthropic") as anthropic_cls, \
             patch("evals.orchestrator.poll_batch") as poll_batch:
            result = poll_mod.handler(
                {"batch_id": DRY_SENTINEL_BATCH_ID, DRY_FLAG: True},
                context=None,
            )
        anthropic_cls.assert_not_called()
        poll_batch.assert_not_called()
        assert result["processing_status"] == "ended"
        assert result["exceeded_max_wait"] is False
        assert result["dry_run"] is True

    def test_dry_sentinel_batch_id_alone_short_circuits(self, poll_mod):
        # Even without the flag, the threaded sentinel batch_id is dry.
        with patch("anthropic.Anthropic") as anthropic_cls, \
             patch("evals.orchestrator.poll_batch") as poll_batch:
            result = poll_mod.handler(
                {"batch_id": DRY_SENTINEL_BATCH_ID}, context=None
            )
        anthropic_cls.assert_not_called()
        poll_batch.assert_not_called()
        assert result["processing_status"] == "ended"


# ── eval_judge_process ───────────────────────────────────────────────


class TestProcessDry:
    def test_dry_no_anthropic_no_s3_read_returns_ok(self, process_mod):
        with patch("anthropic.Anthropic") as anthropic_cls, \
             patch("evals.orchestrator.process_batch_results") as proc:
            result = process_mod.handler(
                {"batch_id": DRY_SENTINEL_BATCH_ID,
                 "plan_s3_key": None, DRY_FLAG: True},
                context=None,
            )
        anthropic_cls.assert_not_called()
        # process_batch_results does the S3 plan get_object + stream +
        # per-artifact persist — must never be called on the dry path.
        proc.assert_not_called()
        assert result["status"] == "OK"
        assert result["dry_run"] is True
        assert result["summary"]["persisted_keys"] == []

    def test_dry_via_threaded_sentinel_without_flag(self, process_mod):
        with patch("anthropic.Anthropic") as anthropic_cls, \
             patch("evals.orchestrator.process_batch_results") as proc:
            result = process_mod.handler(
                {"batch_id": DRY_SENTINEL_BATCH_ID, "plan_s3_key": None},
                context=None,
            )
        anthropic_cls.assert_not_called()
        proc.assert_not_called()
        assert result["status"] == "OK"


# ── rationale_clustering ─────────────────────────────────────────────


class TestRationaleClusteringDry:
    def test_dry_no_compute_no_persist_returns_ok(self, clustering_mod):
        with patch("evals.rationale_clustering.compute_and_emit") as cae:
            result = clustering_mod.handler(
                {"end_time_iso": "2026-05-16T00:00:00Z", DRY_FLAG: True},
                context=None,
            )
        # compute_and_emit is a RETIRED no-op stub (alpha-engine-config-
        # I8173) but must still never be CALLED on the dry path — the
        # shell-run guarantee is "no compute_and_emit call at all."
        cae.assert_not_called()
        assert result["status"] == "OK"
        assert result["dry_run"] is True
        assert result["summary"]["agents_analyzed"] == 0

    def test_existing_dry_run_flag_still_takes_real_path(self, clustering_mod):
        """The pre-existing ``dry_run`` flag only suppresses the CW
        metric — it must STILL run compute_and_emit (the documented
        gap). Only the new ``dry_run_llm`` is the full short-circuit."""
        fake_summary = {
            "agents_analyzed": 3,
            "agents_skipped_thin_sample": [],
            "load_failures": [],
            "cluster_failures": [],
            "per_agent": [],
        }
        with patch("evals.rationale_clustering.compute_and_emit",
                   return_value=fake_summary) as cae:
            result = clustering_mod.handler(
                {"end_time_iso": "2026-05-16T00:00:00Z", "dry_run": True},
                context=None,
            )
        cae.assert_called_once()
        # emit_metrics gated off by the legacy dry_run, but compute ran.
        assert cae.call_args.kwargs["emit_metrics"] is False
        assert result["status"] == "OK"
        assert result.get("dry_run") is not True

    def test_non_dry_runs_compute(self, clustering_mod):
        fake_summary = {
            "agents_analyzed": 6,
            "agents_skipped_thin_sample": [],
            "load_failures": [],
            "cluster_failures": [],
            "per_agent": [],
        }
        with patch("evals.rationale_clustering.compute_and_emit",
                   return_value=fake_summary) as cae:
            result = clustering_mod.handler(
                {"end_time_iso": "2026-05-16T00:00:00Z"}, context=None
            )
        cae.assert_called_once()
        assert cae.call_args.kwargs["emit_metrics"] is True
        assert result["status"] == "OK"


# ── import/boot smoke (the keystone's whole point) ───────────────────


class TestBootStillRuns:
    def test_all_four_handlers_import_clean(
        self, submit_mod, poll_mod, process_mod, clustering_mod
    ):
        # Each fixture exec_module'd the handler file (running its
        # module-level imports + setup_logging + flow-doctor wiring).
        # If any import/bootstrap broke, the fixture would have raised
        # before reaching here — that IS the shell-run smoke.
        for mod in (submit_mod, poll_mod, process_mod, clustering_mod):
            assert hasattr(mod, "handler")
            assert callable(mod.handler)

    def test_dry_path_does_not_skip_module_import(self, monkeypatch):
        """A broken module-level import must still fail the dry pass —
        the dry short-circuit is INSIDE handler(), after imports."""
        # Loading the module runs imports; a syntax/import error here
        # would raise. Reaching the assertion proves boot ran.
        mod = _load_handler(
            "eval_judge_submit_handler.py",
            "lambda_eval_judge_submit_handler_bootcheck",
        )
        assert callable(mod.handler)
