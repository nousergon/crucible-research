"""A stage that finished its work must not be able to report DEGRADED (I9102).

Regression cover for the 2026-08-28 weekly failure: the rolling-mean Lambda
emitted its primary deliverable in 1.6s, then sat 298s inside a SECONDARY
aggregation until the 300s function ceiling killed it. Step Functions raised
``States.Timeout``, the research/predictor branch fail-opened, and the weekly
run terminated ``FAILED`` — for a stage that had succeeded.

Every test here fails without the ``invocation_budget`` wiring:
``test_a_hanging_secondary_block_does_not_hang_the_stage`` blocks for the full
stall and trips the elapsed-time assertion, and
``test_every_secondary_block_is_bounded`` is the detection half — a FIFTH
aggregation bolted onto this handler without a bound fails here rather than
in production nine months later.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "eval_rolling_mean_handler.py"

# How long the fake stalled block blocks for. Long enough that an UNBOUNDED
# handler visibly blows the elapsed assertion; short enough that a broken
# build fails in seconds rather than hanging a CI runner.
_STALL_S = 30.0


def _load_handler_module():
    module_name = "lambda_eval_rolling_mean_handler_budget"
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


class _FakeContext:
    """Stand-in for the Lambda context object.

    ``get_remaining_time_in_millis`` counts down from ``seconds`` in real time,
    exactly as the runtime's does — a frozen value would let a budget bug pass.
    """

    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds

    def get_remaining_time_in_millis(self) -> float:
        return max(0.0, (self._deadline - time.monotonic()) * 1000.0)


def _ok_summary() -> dict:
    return {
        "combos_discovered": 12,
        "datapoints_emitted": 12,
        "combos_skipped_no_data": 0,
        "failed": [],
        "window_start": "2026-08-01T00:00:00+00:00",
        "window_end": "2026-08-28T00:00:00+00:00",
    }


def _calib_report() -> dict:
    return {"status": "empty", "n_cells": 0, "n_cells_sufficient": 0, "n_paired_reviews": 0}


class TestInvocationBudget:
    def test_unbounded_without_a_lambda_context(self):
        from invocation_budget import InvocationBudget

        b = InvocationBudget.from_context(None)
        assert not b.bounded
        assert b.remaining() == float("inf")
        assert b.quote(30.0) == 30.0

    def test_quote_is_capped_by_what_the_invocation_can_still_afford(self):
        from invocation_budget import InvocationBudget

        b = InvocationBudget(lambda: 100_000, reserve_s=20.0)
        assert b.bounded
        assert b.quote(240.0) == pytest.approx(80.0)
        assert b.quote(30.0) == pytest.approx(30.0)

    def test_a_block_is_declined_rather_than_started_when_it_cannot_finish(self):
        from invocation_budget import MIN_VIABLE_BLOCK_S, InvocationBudget

        b = InvocationBudget(lambda: 21_000, reserve_s=20.0)  # 1s of usable budget
        assert b.remaining() == pytest.approx(1.0)
        assert b.remaining() < MIN_VIABLE_BLOCK_S
        assert b.quote(240.0) == 0.0

    def test_run_bounded_returns_the_value_and_re_raises_real_failures(self):
        from invocation_budget import InvocationBudget, run_bounded

        b = InvocationBudget.from_context(None)
        assert run_bounded(lambda: 7, name="x", ceiling_s=10.0, budget=b) == 7

        def _boom():
            raise RuntimeError("S3 down")

        # A real failure must reach the caller's own except branch unchanged —
        # the bound must not convert a genuine error into a timeout.
        with pytest.raises(RuntimeError, match="S3 down"):
            run_bounded(_boom, name="x", ceiling_s=10.0, budget=b)

    def test_run_bounded_abandons_a_block_that_overruns(self):
        from invocation_budget import BlockTimeout, InvocationBudget, run_bounded

        released = threading.Event()
        b = InvocationBudget(lambda: 600_000, reserve_s=0.0)
        started = time.monotonic()
        try:
            with pytest.raises(BlockTimeout):
                run_bounded(
                    lambda: released.wait(_STALL_S),
                    name="stalled",
                    ceiling_s=0.5,
                    budget=b,
                )
            assert time.monotonic() - started < 10.0
        finally:
            released.set()

    def test_the_watchdog_thread_is_a_daemon(self):
        """Load-bearing: a non-daemon thread would delay the invocation
        response and keep the runtime alive past the handler's return — the
        very failure mode the issue hypothesised and this module must not
        introduce."""
        from invocation_budget import InvocationBudget, run_bounded

        seen: dict[str, bool] = {}
        b = InvocationBudget.from_context(None)
        run_bounded(
            lambda: seen.update(daemon=threading.current_thread().daemon),
            name="probe", ceiling_s=10.0, budget=b,
        )
        assert seen["daemon"] is True


class TestStageSurvivesASecondaryStall:
    @pytest.fixture(autouse=True)
    def _stub_side_paths(self):
        with (
            patch("evals.calibration_kappa.emit_calibration_report", return_value=_calib_report()),
            # Stubbed for the same reason the leaderboard is: these are
            # HANDLER-WIRING tests, and an unstubbed control-bands pass reaches
            # real CloudWatch/S3 (it does so from a developer laptop today and
            # merely gets AccessDenied, which is luck, not isolation).
            patch(
                "evals.control_bands.compute_and_emit_control_bands",
                return_value={
                    "failed": [], "combos_discovered": 0,
                    "combos_insufficient_history": 0, "breach_count": 0, "breach_emits": [],
                },
            ),
            patch(
                "scoring.leaderboard_producers.build_producer_leaderboard",
                return_value={"status": "ok", "key": None, "leaderboard": {"n_dates": 0}},
            ),
        ):
            yield

    def test_a_hanging_secondary_block_does_not_hang_the_stage(self, handler_mod):
        """THE regression. ``agent_quality`` stalls; the stage still returns OK.

        Without the bound the handler blocks for the full ``_STALL_S`` — which
        in production was 298 seconds against a 300s ceiling — and the elapsed
        assertion fails. With it, the stage returns its rolling mean in about
        the ceiling and records the stalled block as TIMEOUT.
        """
        released = threading.Event()

        def _stall(*_a, **_kw):
            released.wait(_STALL_S)
            raise AssertionError("the stalled block must never be waited out")

        alerts: list[dict] = []
        started = time.monotonic()
        try:
            with (
                patch.object(handler_mod, "_ensure_init"),
                patch.object(handler_mod, "_CEILING_AGENT_QUALITY_S", 0.5),
                patch.object(
                    handler_mod, "_emit_producer_failure_alert",
                    side_effect=lambda **kw: alerts.append(kw),
                ),
                patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
                patch("scripts.build_agent_quality.build_agent_quality", side_effect=_stall),
            ):
                result = handler_mod.handler({}, _FakeContext(600.0))
            elapsed = time.monotonic() - started
        finally:
            released.set()

        # The primary deliverable is intact and the stage is NOT degraded.
        assert result["status"] == "OK"
        assert result["summary"]["datapoints_emitted"] == 12
        # The stalled block is recorded, not silently skipped.
        assert result["agent_quality"]["status"] == "TIMEOUT"
        assert [a["producer"] for a in alerts] == ["agent_quality"]
        # And the stage returned on its OWN budget, not the stall's.
        assert elapsed < _STALL_S / 2, (
            f"handler took {elapsed:.1f}s — the secondary block was not bounded"
        )

    def test_a_block_is_skipped_when_the_invocation_is_nearly_out_of_time(self, handler_mod):
        """No budget left ⇒ declined with a reason, never started and killed."""
        alerts: list[dict] = []
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch.object(
                handler_mod, "_emit_producer_failure_alert",
                side_effect=lambda **kw: alerts.append(kw),
            ),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch(
                "scripts.build_agent_quality.build_agent_quality",
                side_effect=AssertionError("must not be started with no budget"),
            ),
        ):
            result = handler_mod.handler({}, _FakeContext(21.0))  # 1s usable

        assert result["status"] == "OK"
        assert result["agent_quality"]["status"] == "SKIPPED_NO_BUDGET"
        assert result["producer_leaderboard"]["status"] == "SKIPPED_NO_BUDGET"
        assert {a["producer"] for a in alerts} >= {"agent_quality", "producer_leaderboard"}

    def test_no_lambda_context_leaves_every_block_unbounded(self, handler_mod):
        """Local runs and the existing test suite must behave exactly as before."""
        with (
            patch.object(handler_mod, "_ensure_init"),
            patch("evals.rolling_mean.compute_and_emit_4w_mean", return_value=_ok_summary()),
            patch("scripts.build_agent_quality.build_agent_quality", return_value={"date": "2026-08-28"}),
            patch("scripts.build_agent_quality.write_agent_quality", return_value="k"),
        ):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["agent_quality"]["status"] == "OK"


def test_every_secondary_block_is_bounded():
    """Detection, not just repair: a new secondary aggregation must carry a bound.

    The defect class is "optional work bolted onto a stage can spend the
    stage's whole invocation". Fixing the four blocks that exist today does
    nothing about the fifth, and the fifth is exactly how this recurs. Every
    ``_CEILING_*_S`` constant must be consumed by a ``run_bounded`` call, and
    every ``run_bounded`` call must name one — so adding a block without a
    ceiling, or a ceiling nothing enforces, fails here.
    """
    src = _HANDLER_PATH.read_text(encoding="utf-8")
    declared = set(re.findall(r"^(_CEILING_\w+_S) = ", src, flags=re.M))
    used = set(re.findall(r"ceiling_s=(_CEILING_\w+_S)", src))
    assert declared, "no per-block ceilings declared in the handler"
    assert declared == used, (
        f"ceiling constants and run_bounded call sites disagree — "
        f"declared-but-unused={sorted(declared - used)}, "
        f"used-but-undeclared={sorted(used - declared)}"
    )
    # One bounded call per secondary aggregation named in the return value.
    assert len(used) == 4, f"expected 4 bounded secondary blocks, found {sorted(used)}"
    for producer in ("calibration_kappa", "control_bands", "agent_quality", "producer_leaderboard"):
        assert f'name="{producer}"' in src, f"{producer} is not run under a bound"
