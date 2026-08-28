"""A secondary aggregation cannot spend the primary deliverable's budget.

alpha-engine-config#9102.

``eval_rolling_mean_handler`` produces the rolling mean in ~1.5 s and then runs
four best-effort aggregations. Two of them — ``agent_quality`` and
``producer_leaderboard`` — scan an S3 corpus that grows every cycle, and both did
so through a bare ``boto3.client("s3")`` carrying botocore's defaults
(``connect_timeout=60``, ``read_timeout=60``, legacy retries up to 5). Neither
logged anything between entering its ``try`` and finishing, so a slow scan was
indistinguishable from a hang while it was happening.

Measured 2026-08-28: this state's ``REPORT`` durations ran 22-37 s across seven
invocations, then 120 s, then 65 s, then hit the 300 s ceiling. That is a scan
outgrowing its budget, not a network blip. At the ceiling the Step Function
recorded ``States.Timeout``, ``MarkEvalRollingMeanDegraded`` fired, and the whole
research/predictor branch fail-opened — so a run that HAD produced its primary
deliverable terminated DEGRADED.

The carve-out letting these fail without sinking the rolling mean is legitimate.
What was missing: it only ever covered EXCEPTIONS. A block that never returns is
not an exception, so it consumed the budget of the thing it was forbidden to
sink. These tests pin the two properties that close that gap — a bound on the
call, and a bound on the block — plus the rule that blowing the bound is
reported rather than swallowed.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "eval_rolling_mean_handler.py"


def _load_handler_module():
    module_name = "lambda_eval_rolling_mean_handler_bounds"
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


class TestNoUnboundedClientSurvives:
    """The source-level guard: a bare client is the defect, so ban the shape."""

    def test_the_handler_constructs_no_bare_boto3_client(self) -> None:
        """Parsed, not grepped.

        A line-based scan reports `_bounded_client`'s own multi-line
        ``return boto3.client(`` — the one construction that IS bounded, with the
        ``config=`` on a later line — and it would equally miss a bare client
        written across two lines. Walking the AST asks the actual question: does
        every ``boto3.client(...)`` call carry a ``config`` keyword.
        """
        import ast

        tree = ast.parse(_HANDLER_PATH.read_text(encoding="utf-8"))
        offending = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "client"
                and isinstance(func.value, ast.Name)
                and func.value.id == "boto3"
            ):
                continue
            if not any(kw.arg == "config" for kw in node.keywords):
                service = (
                    node.args[0].value
                    if node.args and isinstance(node.args[0], ast.Constant)
                    else "?"
                )
                offending.append(f"line {node.lineno}: boto3.client({service!r})")
        assert not offending, (
            "every AWS client in this handler must carry an explicit timeout "
            "Config — use _bounded_client(). Unbounded: " + "; ".join(offending)
        )

    def test_bounded_client_sets_real_timeouts(self, handler_mod) -> None:
        pytest.importorskip("botocore.config")
        fake_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": fake_boto3}):
            handler_mod._bounded_client("s3")
        config = fake_boto3.client.call_args.kwargs["config"]
        assert config.connect_timeout == 5
        assert config.read_timeout == 15
        assert config.retries["max_attempts"] == 2


class TestSecondaryDeadline:
    def test_a_slow_aggregation_is_abandoned_at_its_deadline(self, handler_mod) -> None:
        """The regression: without this, the block runs until the SF times out."""
        started = threading.Event()

        def _never_returns() -> dict:
            started.set()
            time.sleep(30)
            return {"status": "OK"}

        with patch.object(handler_mod, "_emit_producer_failure_alert") as alert:
            t0 = time.monotonic()
            result = handler_mod._run_bounded(
                "agent_quality", "backtest/{date}/agent_quality.json",
                _never_returns, deadline_s=0.5,
            )
            elapsed = time.monotonic() - t0

        assert started.is_set(), "the worker must actually have run"
        assert elapsed < 5, f"the deadline did not bind — took {elapsed:.1f}s"
        assert result["status"] == "TIMEOUT"
        assert result["deadline_s"] == 0.5
        assert alert.call_count == 1, "a blown deadline must reach the alert bus"
        assert alert.call_args.kwargs["producer"] == "agent_quality"

    def test_the_abandoned_worker_cannot_hold_the_invocation_open(self, handler_mod) -> None:
        """A non-daemon worker would move the hang one layer down, not remove it."""
        names: list[threading.Thread] = []
        real_thread = threading.Thread

        def _capture(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            names.append(t)
            return t

        with patch.object(handler_mod, "_emit_producer_failure_alert"), \
                patch("threading.Thread", _capture):
            handler_mod._run_bounded(
                "producer_leaderboard", "the leaderboard",
                lambda: time.sleep(30), deadline_s=0.3,
            )

        assert names, "expected a worker thread"
        assert all(t.daemon for t in names), (
            "the worker must be a daemon — a non-daemon thread keeps the Lambda "
            "runtime alive after the handler returns, which is the same defect "
            "one layer down"
        )

    def test_a_fast_aggregation_returns_its_own_result_untouched(self, handler_mod) -> None:
        with patch.object(handler_mod, "_emit_producer_failure_alert") as alert:
            result = handler_mod._run_bounded(
                "agent_quality", "artifact", lambda: {"status": "OK", "key": "k"},
            )
        assert result == {"status": "OK", "key": "k"}
        assert alert.call_count == 0

    def test_an_exception_still_propagates_to_the_existing_handler(self, handler_mod) -> None:
        """The pre-existing except-branch stays in charge of real exceptions.

        The deadline covers the case an exception cannot: a block that neither
        returns nor raises. It must not quietly absorb the case that already had
        an owner, or the alert text loses the actual error.
        """
        def _boom() -> dict:
            raise AttributeError("'str' object has no attribute 'isoformat'")

        with patch.object(handler_mod, "_emit_producer_failure_alert"):
            with pytest.raises(AttributeError, match="isoformat"):
                handler_mod._run_bounded("agent_quality", "artifact", _boom)


class TestDeadlineFitsTheStageBudget:
    def test_the_deadlines_cannot_add_up_past_the_sf_budget(self, handler_mod) -> None:
        """Two bounded blocks plus the primary must fit inside the SF's 240s.

        nousergon-data/infrastructure/step_function.json gives EvalRollingMean a
        240 s budget, itself deliberately below the Lambda's 300 s ceiling. The
        primary deliverable measures ~1.5 s, so the secondaries have room — but
        only while their deadlines stay small enough that the sum cannot reach it.
        """
        deadline = handler_mod._SECONDARY_DEADLINE_S
        bounded_blocks = 2
        primary_observed_s = 2.0
        assert deadline * bounded_blocks + primary_observed_s < 240, (
            f"{bounded_blocks} blocks at {deadline}s each no longer fit inside the "
            "240s SF budget — raise the budget deliberately or lower the deadline"
        )
