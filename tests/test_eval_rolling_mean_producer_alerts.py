"""A secondary-producer failure in this handler must reach the alert bus.

alpha-engine-config-I8177. ``eval_rolling_mean_handler`` hangs four
best-effort aggregations off the weekly rolling-mean Lambda
(``calibration_kappa``, ``control_bands``, ``agent_quality``,
``producer_leaderboard``). Each is wrapped in ``except Exception`` so a
failure cannot sink the primary deliverable — a legitimate carve-out.

What was NOT legitimate: recording the failure only in a ``logger.warning``
and in a ``{"status": "ERROR"}`` field of the handler's return value that
nothing consumes and nothing pages on. ``observability-policy.md`` §7.3 — a
human-only notification is invisible, because the response plane cannot see
it and nothing owns follow-through.

The cost of that was measured: the ``agent_quality`` producer raised
``'str' object has no attribute 'isoformat'`` on every weekly run from
2026-06-23 to 2026-08-22 — two months, eight report-card components stuck at
``N/A-MISSING-INPUT`` — and the only trace was a WARN line nobody read.

These tests pin the class across all four blocks, not just the one that
broke: every swallow path emits onto the machine-readable bus, the alert
names the producer, and the alerting itself can never sink the handler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "eval_rolling_mean_handler.py"

# Every best-effort producer in this handler. Adding a fifth aggregation
# without adding it here is caught by
# ``test_every_swallow_path_alerts`` below.
EXPECTED_PRODUCERS = {
    "calibration_kappa",
    "control_bands",
    "agent_quality",
    "producer_leaderboard",
}


def _load_handler_module():
    module_name = "lambda_eval_rolling_mean_handler_alerts"
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


def test_alert_helper_publishes_to_the_bus(handler_mod) -> None:
    published = MagicMock()
    with patch.dict(sys.modules, {}, clear=False), \
            patch("krepis.alerts.publish", published):
        handler_mod._emit_producer_failure_alert(
            producer="agent_quality",
            artifact="backtest/{date}/agent_quality.json",
            exc=AttributeError("'str' object has no attribute 'isoformat'"),
        )
    assert published.call_count == 1
    body = published.call_args.args[0]
    kwargs = published.call_args.kwargs
    assert "agent_quality" in body
    assert "backtest/{date}/agent_quality.json" in body
    assert "isoformat" in body, "the alert must carry the actual exception text"
    assert kwargs["severity"] == "error"
    assert "eval_rolling_mean" in kwargs["source"]


def test_alert_is_deduped_per_producer_per_week(handler_mod) -> None:
    """One alert per weekly cycle, and it re-fires next week — never latches."""
    published = MagicMock()
    with patch("krepis.alerts.publish", published):
        handler_mod._emit_producer_failure_alert(
            producer="agent_quality", artifact="a", exc=RuntimeError("x"),
        )
        handler_mod._emit_producer_failure_alert(
            producer="control_bands", artifact="b", exc=RuntimeError("y"),
        )
    keys = [c.kwargs["dedup_key"] for c in published.call_args_list]
    assert len(set(keys)) == 2, "different producers must not collapse into one alert"
    for key in keys:
        assert "-W" in key, f"dedup key must be week-scoped so it re-fires: {key}"
    # Window None = the week token alone bounds the dedup; a fixed minute
    # window would let a second alert through mid-cycle.
    assert all(c.kwargs["dedup_window_min"] is None for c in published.call_args_list)


def test_alerting_failure_never_sinks_the_handler(handler_mod, caplog) -> None:
    """The one correct swallow: the ERROR log is already the surviving record."""
    with patch("krepis.alerts.publish", side_effect=RuntimeError("SNS down")):
        handler_mod._emit_producer_failure_alert(
            producer="agent_quality", artifact="a", exc=RuntimeError("original"),
        )  # must not raise
    assert any("could not publish" in r.message for r in caplog.records)


def test_every_swallow_path_alerts() -> None:
    """Class, not instance — every ``status: ERROR`` block calls the emitter.

    Read statically so a newly-added fifth aggregation that forgets to alert
    fails this test rather than shipping a silent producer.
    """
    source = _HANDLER_PATH.read_text()
    error_blocks = source.count('= {"status": "ERROR", "error": str(exc)}')
    # alpha-engine-config#9102 added a SECOND kind of swallow: a secondary
    # aggregation that neither returns nor raises, abandoned at its deadline by
    # `_run_bounded`. An exception was never the only way one of these can stop
    # producing — a growing S3 scan simply ran until the Step Function timed out
    # — so the deadline path is counted here as the alerting site it is, rather
    # than left outside the class this test exists to pin.
    deadline_paths = source.count('"status": "TIMEOUT"')
    emitter_calls = source.count("_emit_producer_failure_alert(")
    # one definition + one call per swallow path, of either kind
    assert emitter_calls == error_blocks + deadline_paths + 1, (
        f"{error_blocks} exception paths + {deadline_paths} deadline paths but "
        f"{emitter_calls - 1} alert calls — a swallow path is silent"
    )
    assert deadline_paths >= 1, (
        "the deadline swallow path disappeared — a secondary aggregation can "
        "once again run until the SF budget is gone (alpha-engine-config#9102)"
    )
    for producer in EXPECTED_PRODUCERS:
        assert f'producer="{producer}"' in source, (
            f"{producer} failure path does not name itself on the alert bus"
        )


def test_no_swallow_path_logs_only_at_warning() -> None:
    """A producer failure is an ERROR — WARNING is what made this invisible."""
    source = _HANDLER_PATH.read_text()
    assert "failed (non-fatal)" not in source, (
        "'(non-fatal)' at WARNING was the wording that let a two-month "
        "producer outage read as routine noise"
    )
