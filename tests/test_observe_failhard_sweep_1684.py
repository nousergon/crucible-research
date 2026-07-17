"""§61 fail-hard/alarmed-surface sweep — config#1684.

Brian ruling 2026-07-03: experiment/observe producers fail hard, or (when they
run before the primary deliverable is persisted and so cannot raise) route their
failure through an ALARMED surface with a consumer — never a bare WARN log
(ARCHITECTURE.md §61). This pins the scanner shadow-spec surface: a challenger
spec that raises must (a) NOT take out the live artifact or sibling shadows, and
(b) fire a loud observe alert, not just a WARN.

The handler.py pre-persistence surfaces (scorecard / team_accuracy / memory /
trajectory) get the same `publish_observe_alert` treatment; they are validated by
the full research suite (they import the heavy graph stack) — see the PR body.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _min_live_artifact() -> dict:
    return {"tickers": ["AAA"], "run_date": "2026-07-04", "params": {}}


def test_shadow_spec_failure_fires_observe_alert_and_is_non_fatal():
    """A challenger spec that raises → omitted (non-fatal) AND an observe alert
    is published to the alarmed surface (not just a WARN)."""
    import data.scanner_specs as ss

    eval_log = [{"liquidity_pass": 1, "ticker": "AAA"}]

    # Force the single challenger's rank() to raise.
    with patch.object(
        ss, "challenger_specs",
        return_value=[type("S", (), {
            "name": "boom_spec",
            "rank": staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom"))),
        })()],
    ), patch("observe_alerts.publish_observe_alert") as mock_alert:
        out = ss.build_shadow_artifacts(
            live_artifact=_min_live_artifact(),
            eval_log=eval_log,
            factor_loadings=None,
            params={},
        )

    # (a) non-fatal: no exception propagated, failing spec omitted from output.
    assert "boom_spec" not in out
    # (b) loud: the alarmed surface was invoked, tagged to the failing spec.
    assert mock_alert.called, "shadow-spec failure must fire an observe alert (§61)"
    kwargs = mock_alert.call_args.kwargs
    assert "boom_spec" in kwargs.get("source", "")
    assert "kaboom" in mock_alert.call_args.args[0]


def test_healthy_shadow_spec_does_not_alert():
    """A spec that ranks fine emits its artifact and fires NO alert."""
    import data.scanner_specs as ss

    eval_log = [{"liquidity_pass": 1, "ticker": "AAA"}]
    good = type("S", (), {
        "name": "ok_spec",
        "rank": staticmethod(lambda *a, **k: ["AAA"]),
    })()

    with patch.object(ss, "challenger_specs", return_value=[good]), \
            patch.object(ss, "_shadow_artifact", return_value={"tickers": ["AAA"]}), \
            patch("observe_alerts.publish_observe_alert") as mock_alert:
        out = ss.build_shadow_artifacts(
            live_artifact=_min_live_artifact(),
            eval_log=eval_log,
            factor_loadings=None,
            params={},
        )

    assert out.get("ok_spec") == {"tickers": ["AAA"]}
    assert not mock_alert.called
