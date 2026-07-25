"""Unit tests for the signals-envelope Lambda handler (alpha-engine-config
epic #2515 Phase B).

Pins the handler's load-bearing contracts:

1. **Dry path** — ``dry_run_llm`` (shell-run smoke) returns before
   ``_ensure_init`` / any S3 access.
2. **Raise-on-failure** — the handler must PROPAGATE exceptions, never
   convert them to an ERROR-dict return. This Lambda is invoked
   synchronously by an ``arn:aws:states:::lambda:invoke`` SF Task: the
   Catch only triggers on an actual raised Lambda error, so an ERROR-dict
   return would be a *successful* Task completion that never routes
   through the Catch — mirrors ``thinktank_handler.py``'s identical
   contract (see test_thinktank_handler.py::TestRaiseOnFailure).
3. **Target routing** — ``target`` (default ``"shadow"``) reaches
   ``write_envelope`` verbatim; an invalid target raises before any S3
   call.
4. **Required-field validation** — a missing/invalid ``run_date`` raises
   (never an ERROR dict).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda" / "signals_envelope_handler.py"


def _load_handler_module():
    """Import lambda/signals_envelope_handler.py without using ``lambda``
    as a package name (Python keyword). Mirrors test_thinktank_handler.py /
    test_scanner_handler.py."""
    module_name = "lambda_signals_envelope_handler"
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
    mod._init_done = False


def _board() -> dict:
    return {
        "stocks": [
            {
                "ticker": "AAPL",
                "sector": "Technology",
                "attractiveness_score": 72.5,
                "pillars": {"quality": 80.0},
            },
            {
                "ticker": "JNJ",
                "sector": "Healthcare",
                "attractiveness_score": 55.0,
                "pillars": {"quality": 60.0},
            },
        ],
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


class TestDryPath:
    def test_shell_run_dry_short_circuits_before_init(self, handler_mod):
        """dry_run_llm must return before _ensure_init — no S3 access.
        The Friday shell-run keystone contract, same as every other
        shared-image handler's dry-path."""
        with patch.object(handler_mod, "_ensure_init") as init, \
             patch("boto3.client") as boto_client:
            result = handler_mod.handler({"dry_run_llm": True}, None)
        assert result == {"status": "OK", "dry_run": True}
        init.assert_not_called()
        boto_client.assert_not_called()

    def test_dry_short_circuits_even_without_run_date(self, handler_mod):
        """The dry check must happen BEFORE required-field validation —
        a shell-run smoke event carries no run_date at all."""
        result = handler_mod.handler({"dry_run_llm": True}, None)
        assert result == {"status": "OK", "dry_run": True}


class TestRequiredFieldValidation:
    def test_missing_run_date_raises(self, handler_mod):
        with pytest.raises(ValueError, match="run_date"):
            handler_mod.handler({}, None)

    def test_non_dict_event_raises(self, handler_mod):
        with pytest.raises(ValueError, match="run_date"):
            handler_mod.handler(None, None)

    def test_invalid_run_date_type_raises(self, handler_mod):
        with pytest.raises(ValueError, match="run_date"):
            handler_mod.handler({"run_date": 20260714}, None)

    def test_short_run_date_string_raises(self, handler_mod):
        with pytest.raises(ValueError, match="run_date"):
            handler_mod.handler({"run_date": "2026-07"}, None)

    def test_invalid_target_raises_before_any_s3_call(self, handler_mod):
        """Target validation must fail before the envelope-building S3
        reads (board / substrate) run. Pinned via the library functions
        rather than a blanket ``boto3.client`` assertion — the shared
        ``monitor_handler`` flow-doctor wrapper may independently touch S3
        for its OWN incident-reporting side channel when active in-process
        (an orthogonal concern, not this handler's envelope-building path)."""
        with patch("scoring.signals_envelope.read_universe_board") as read_board, \
             patch("scoring.signals_envelope.read_regime_substrate") as read_substrate:
            with pytest.raises(ValueError, match="target"):
                handler_mod.handler(
                    {"run_date": "2026-07-14", "target": "production_typo"}, None,
                )
        read_board.assert_not_called()
        read_substrate.assert_not_called()


class TestSuccessPath:
    def test_default_target_is_shadow(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()) as read_board, \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None) as read_substrate, \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()) as build, \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 return_value=("signals_envelope/2026-07-14/signals.json", "signals_envelope/latest.json"),
             ) as write:
            result = handler_mod.handler({"run_date": "2026-07-14"}, None)

        read_board.assert_called_once()
        assert read_board.call_args.kwargs.get("run_date") == "2026-07-14"
        read_substrate.assert_called_once()
        build.assert_called_once_with("2026-07-14", _board(), None)
        assert write.call_args.kwargs["target"] == "shadow"
        assert result == {
            "status": "OK",
            "dated_key": "signals_envelope/2026-07-14/signals.json",
            "latest_key": "signals_envelope/latest.json",
            "universe_count": 2,
            "market_regime": "neutral",
            "target": "shadow",
        }

    def test_target_production_reaches_write_envelope_explicitly(self, handler_mod):
        """The explicit-production-intent contract: 'target': 'production'
        on the event must reach write_envelope verbatim, never inferred or
        silently downgraded to shadow."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 return_value=("signals/2026-07-14/signals.json", "signals/latest.json"),
             ) as write:
            result = handler_mod.handler(
                {"run_date": "2026-07-14", "target": "production"}, None,
            )

        assert write.call_args.kwargs["target"] == "production"
        assert result["target"] == "production"
        assert result["dated_key"] == "signals/2026-07-14/signals.json"

    def test_bucket_override_forwarded(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()) as read_board, \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 return_value=("k1", "k2"),
             ) as write:
            handler_mod.handler(
                {"run_date": "2026-07-14", "bucket": "custom-bucket"}, None,
            )
        assert read_board.call_args.args[0] == "custom-bucket"
        assert write.call_args.kwargs["bucket"] == "custom-bucket"

    def test_preflight_flag_forwarded_to_read_universe_board(self, handler_mod):
        """config-I2916: the weekly SF threads ``preflight.$: $.research_dry``
        (true only on the Friday-PM shell run). The handler must forward it to
        read_universe_board so the fallback-staleness guard is downgraded to a
        WARN for the dry-Scanner preflight — and it is DISTINCT from
        dry_run_llm (the full read/build/write path still runs)."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()) as read_board, \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 return_value=("k1", "k2"),
             ):
            handler_mod.handler(
                {"run_date": "2026-07-14", "preflight": True}, None,
            )
        # Not short-circuited: the read path actually ran (distinct from dry).
        read_board.assert_called_once()
        assert read_board.call_args.kwargs.get("preflight") is True

    def test_preflight_defaults_false_on_real_run(self, handler_mod):
        """Absent ``preflight`` (every real Saturday invoke) must forward
        preflight=False so the I2880 staleness guard stays fully in force."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()) as read_board, \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 return_value=("k1", "k2"),
             ):
            handler_mod.handler({"run_date": "2026-07-14"}, None)
        assert read_board.call_args.kwargs.get("preflight") is False

    def test_substrate_none_is_not_an_error(self, handler_mod):
        """The module's ONE documented fail-soft exception: a None
        substrate (missing/unreadable) must reach build_signals_envelope
        unchanged and must NOT raise or downgrade the result."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()) as build, \
             patch("scoring.signals_envelope.write_envelope", return_value=("k1", "k2")):
            result = handler_mod.handler({"run_date": "2026-07-14"}, None)
        build.assert_called_once_with("2026-07-14", _board(), None)
        assert result["status"] == "OK"


class TestRaiseOnFailure:
    def test_missing_universe_board_propagates(self, handler_mod):
        """read_universe_board's own RuntimeError (board missing at both
        the dated key and latest.json) must propagate uncaught — the
        module's hard precondition failure, never softened here."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch(
                 "scoring.signals_envelope.read_universe_board",
                 side_effect=RuntimeError("no scanner universe board found"),
             ):
            with pytest.raises(RuntimeError, match="no scanner universe board found"):
                handler_mod.handler({"run_date": "2026-07-14"}, None)

    def test_build_envelope_failure_propagates(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch(
                 "scoring.signals_envelope.build_signals_envelope",
                 side_effect=ValueError("empty universe"),
             ):
            with pytest.raises(ValueError, match="empty universe"):
                handler_mod.handler({"run_date": "2026-07-14"}, None)

    def test_write_envelope_failure_propagates(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 side_effect=RuntimeError("S3 put failed"),
             ):
            with pytest.raises(RuntimeError, match="S3 put failed"):
                handler_mod.handler({"run_date": "2026-07-14"}, None)

    def test_handler_source_never_returns_error_status(self):
        """Belt-and-suspenders source pin: the SF-handler idiom
        ``return {"status": "ERROR", ...}`` must never appear here — a
        sync-invoked SF Task Lambda that returns ERROR is a silent
        failure (see module doc)."""
        import re

        text = _HANDLER_PATH.read_text(encoding="utf-8")
        assert not re.search(r'return\s*\{\s*"status":\s*"ERROR"', text)


class TestSecondaryArtifacts:
    """config-I3290 port: research_consolidated_morning + scanner_universe_
    trajectory, written as post-steps gated to a real production run,
    fail-soft (must never sink the primary signals.json deliverable)."""

    def test_production_target_writes_morning_brief_and_trajectory(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch("scoring.signals_envelope.write_envelope", return_value=("k1", "k2")), \
             patch("scoring.morning_brief.build_morning_brief_markdown", return_value="# brief") as build_brief, \
             patch("scoring.morning_brief.write_morning_brief") as write_brief, \
             patch("scoring.attractiveness_trajectory.compute_and_write_trajectory") as write_traj:
            handler_mod.handler({"run_date": "2026-07-14", "target": "production"}, None)

        build_brief.assert_called_once_with(_envelope())
        write_brief.assert_called_once()
        assert write_brief.call_args.args[0] == "2026-07-14"
        write_traj.assert_called_once()
        assert write_traj.call_args.args[0] == "2026-07-14"

    def test_shadow_target_skips_both(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch("scoring.signals_envelope.write_envelope", return_value=("k1", "k2")), \
             patch("scoring.morning_brief.build_morning_brief_markdown") as build_brief, \
             patch("scoring.morning_brief.write_morning_brief") as write_brief, \
             patch("scoring.attractiveness_trajectory.compute_and_write_trajectory") as write_traj:
            handler_mod.handler({"run_date": "2026-07-14", "target": "shadow"}, None)

        build_brief.assert_not_called()
        write_brief.assert_not_called()
        write_traj.assert_not_called()

    def test_morning_brief_failure_is_fail_soft(self, handler_mod):
        """A brief-write failure must not propagate — signals.json is
        already persisted by this point."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 return_value=("signals/2026-07-14/signals.json", "signals/latest.json"),
             ), \
             patch(
                 "scoring.morning_brief.build_morning_brief_markdown",
                 side_effect=RuntimeError("boom"),
             ), \
             patch("scoring.attractiveness_trajectory.compute_and_write_trajectory"), \
             patch("observe_alerts.publish_observe_alert") as alert:
            result = handler_mod.handler(
                {"run_date": "2026-07-14", "target": "production"}, None,
            )

        assert result["status"] == "OK"
        assert result["dated_key"] == "signals/2026-07-14/signals.json"
        alert.assert_called_once()

    def test_trajectory_failure_is_fail_soft(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("boto3.client", return_value=MagicMock()), \
             patch("scoring.signals_envelope.read_universe_board", return_value=_board()), \
             patch("scoring.signals_envelope.read_regime_substrate", return_value=None), \
             patch("scoring.signals_envelope.build_signals_envelope", return_value=_envelope()), \
             patch(
                 "scoring.signals_envelope.write_envelope",
                 return_value=("signals/2026-07-14/signals.json", "signals/latest.json"),
             ), \
             patch("scoring.morning_brief.build_morning_brief_markdown", return_value="# brief"), \
             patch("scoring.morning_brief.write_morning_brief"), \
             patch(
                 "scoring.attractiveness_trajectory.compute_and_write_trajectory",
                 side_effect=RuntimeError("boom"),
             ), \
             patch("observe_alerts.publish_observe_alert") as alert:
            result = handler_mod.handler(
                {"run_date": "2026-07-14", "target": "production"}, None,
            )

        assert result["status"] == "OK"
        assert result["dated_key"] == "signals/2026-07-14/signals.json"
        alert.assert_called_once()
