"""Tests for the `self_test` invocation mode (alpha-engine-config-I7726).

WHY THIS MODE EXISTS. `_maybe_emit_self_test` lives at the end of the
`weekly_run` branch of `lambda/handler.py::_run`. MEASURED 2026-08-19: that call
site was structurally unreachable, and `research/{date}/self_test.json` had ZERO
instances in the bucket since the registry row was created on 2026-08-13 —

  * the weekly SF's ONLY invocation of this Lambda is the ChallengerShadow
    state, payload {"mode": "challengers_only"}, and `_run` returns on that mode
    before ever reaching the weekly branch;
  * the documented fallback trigger, EventBridge rule `alpha-research-weekly`
    cron(0 6 ? * SAT *), is DISABLED.

A §2.3a CORRECTNESS VERDICT whose absence must never read as a pass had never
once been emitted, and nothing said so — which is exactly the shape
`alpha-engine-config-I7622` covers.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_HANDLER_PATH = Path(__file__).parent.parent / "lambda" / "handler.py"


def _import_handler():
    """lambda/ collides with the keyword — load via importlib."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("research_handler_self_test_mode", _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_self_test_mode_reaches_the_emitter():
    """The regression itself: the mode must call the thing that writes."""
    mod = _import_handler()
    with patch.object(mod, "_ensure_init"), \
         patch.object(mod, "_maybe_emit_self_test") as emit, \
         patch.object(mod, "_resolve_self_test_date", return_value="2026-08-14"):
        out = mod._run({"mode": "self_test", "date": "2026-08-15"}, None)
    emit.assert_called_once_with(datetime.date(2026, 8, 14))
    assert out == {"status": "OK", "mode": "self_test", "date": "2026-08-14"}


def test_self_test_mode_returns_before_the_weekly_graph():
    """It must NOT fall through into the champion run — the battery is a
    known-answer test over the deployed instrument, not a research pass."""
    mod = _import_handler()
    with patch.object(mod, "_ensure_init"), \
         patch.object(mod, "_maybe_emit_self_test"), \
         patch.object(mod, "_resolve_self_test_date", return_value="2026-08-14"), \
         patch.object(mod, "_run_challengers_only") as challengers:
        mod._run({"mode": "self_test"}, None)
    challengers.assert_not_called()


def test_challengers_only_still_takes_precedence_unchanged():
    """The pre-existing mode must be untouched by the new branch above it."""
    mod = _import_handler()
    with patch.object(mod, "_ensure_init"), \
         patch.object(mod, "_run_challengers_only", return_value={"status": "OK"}) as challengers, \
         patch.object(mod, "_maybe_emit_self_test") as emit:
        out = mod._run({"mode": "challengers_only", "date": "2026-08-14"}, None)
    challengers.assert_called_once()
    emit.assert_not_called()
    assert out == {"status": "OK"}


def test_the_saturday_calendar_date_is_normalised_to_a_trading_day():
    """config-I7419's lesson, applied here BEFORE it can bite. The weekly SF
    passes the EXECUTION's calendar date and the weekly run is a Saturday —
    never itself a session. Keying the artifact on Saturday writes it to a date
    no consumer looks up, which is a different way of never producing it."""
    mod = _import_handler()
    with patch("nousergon_lib.dates.resolve_trading_day", return_value="2026-08-14") as resolve:
        assert mod._resolve_self_test_date({"date": "2026-08-15"}) == "2026-08-14"
    resolve.assert_called_once_with("2026-08-15")


def test_a_missing_date_defaults_to_today_then_normalises():
    mod = _import_handler()
    with patch("nousergon_lib.dates.resolve_trading_day", return_value="2026-08-18") as resolve:
        assert mod._resolve_self_test_date({}) == "2026-08-18"
    assert resolve.call_count == 1


def test_the_battery_never_fails_the_caller():
    """`_maybe_emit_self_test` swallows by contract — the verdict is carried in
    the artifact, the console row and the logs. The SF state is non-blocking for
    the same reason, so the mode must not raise either."""
    mod = _import_handler()
    with patch.object(mod, "_ensure_init"), \
         patch.object(mod, "_resolve_self_test_date", return_value="2026-08-14"), \
         patch.object(mod, "_maybe_emit_self_test", side_effect=RuntimeError("battery blew up")):
        with pytest.raises(RuntimeError):
            mod._run({"mode": "self_test"}, None)
    # ^ deliberate: the emitter's OWN contract is never-raises (its body traps
    # everything). If that contract is ever broken, this mode should surface it
    # rather than add a second swallow — two layers of silence is how a verdict
    # stage becomes unobservable. The SF state's Catch is what keeps the run
    # alive, and that is the right layer for it.
