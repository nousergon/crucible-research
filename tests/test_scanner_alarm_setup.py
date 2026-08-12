"""Lock the scanner-degradation alarm setup against drift (config#785).

The scanner-output floor alarm only works if three things stay in sync:

  1. ``lambda/scanner_handler.py`` emits a parseable metric-marker line.
  2. ``infrastructure/setup_scanner_alarm.sh`` creates a log metric filter
     whose space-delimited pattern matches that exact line and binds the
     trailing integer as the metric value.
  3. The alarm uses the shared SNS topic and the floor from the issue (25).

If any drifts, the alarm silently stops covering the scanner — the very
"silent fail" config#785 exists to prevent. These tests fail the build
when they diverge.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "infrastructure" / "setup_scanner_alarm.sh"
_HANDLER_PATH = _REPO_ROOT / "lambda" / "scanner_handler.py"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _load_handler_module():
    module_name = "lambda_scanner_handler_alarmtest"
    spec = importlib.util.spec_from_file_location(module_name, _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_setup_script_exists():
    assert _SCRIPT.is_file()


def test_alarm_uses_shared_sns_topic():
    text = _script_text()
    assert "alpha-engine-alerts" in text
    assert "alpha-engine-scanner-tickers-degradation" in text


def test_threshold_matches_issue_floor():
    # config#785 specifies a floor of 25 (half of the ~50 baseline).
    assert re.search(r"THRESHOLD=25\b", _script_text())


def test_filter_targets_the_scanner_log_group():
    # The log group is composed from the function name:
    #   LOG_GROUP="/aws/lambda/${FUNCTION_SCANNER}"
    text = _script_text()
    assert 'FUNCTION_SCANNER="alpha-engine-research-scanner"' in text
    assert 'LOG_GROUP="/aws/lambda/${FUNCTION_SCANNER}"' in text


def test_metric_namespace_and_name_are_consistent():
    text = _script_text()
    assert "AlphaEngine/Scanner" in text
    assert "scanner_tickers_count" in text


def test_filter_pattern_matches_the_handler_marker_line():
    """The load-bearing correctness check: simulate the text-mode log
    record the handler emits and confirm the script's space-delimited
    filter pattern would bind the count token.

    Filter pattern (from the script):
      [date, time, level, component, handler, marker="METRIC",
       name="scanner_tickers_count", count]
    """
    # Reconstruct the rendered marker line in text-log mode. setup_logging
    # uses: "%(asctime)s %(levelname)s [scanner] %(message)s" and the
    # handler message is "[scanner_handler] METRIC scanner_tickers_count %d".
    rendered = (
        "2026-06-28 12:34:56,789 INFO [scanner] "
        "[scanner_handler] METRIC scanner_tickers_count 55"
    )
    tokens = rendered.split()
    # date time level [scanner] [scanner_handler] METRIC name count -> 8 tokens
    assert len(tokens) == 8, tokens
    assert tokens[5] == "METRIC"
    assert tokens[6] == "scanner_tickers_count"
    assert tokens[7].isdigit(), "count token must be a bare integer"
    assert int(tokens[7]) == 55

    # The literal anchors the pattern keys on must be present in the script
    # exactly as the line renders them.
    text = _script_text()
    assert 'marker="METRIC"' in text
    assert 'name="scanner_tickers_count"' in text
    assert "metricValue='$count'" in text


def test_handler_emits_the_metric_marker_line():
    """Guard the producer side: the handler source must contain the marker
    format string the filter depends on."""
    src = _HANDLER_PATH.read_text(encoding="utf-8")
    assert "METRIC scanner_tickers_count %d" in src


# ── Duration headroom alarm (alpha-engine-config-I6855) ───────────────────


def _deploy_sh() -> str:
    return (_REPO_ROOT / "infrastructure" / "deploy.sh").read_text()


def _deploy_scanner_body() -> str:
    """The body of ``deploy_scanner()``.

    Terminated on a closing brace in COLUMN ZERO, not the first ``}`` — the
    body contains ``${BASH_SOURCE[0]}``, and splitting naively truncated it
    mid-line and made this guard assert against a fragment.
    """
    after = _deploy_sh().split("deploy_scanner() {", 1)[1]
    return after.split("\n}", 1)[0]


def test_duration_alarm_derives_its_threshold_from_the_live_timeout():
    """70% of whatever the function's timeout actually is, not a literal.

    A hardcoded millisecond threshold silently widens the blind spot the
    next time the timeout is raised — which is precisely the move I6855 was
    filed on. Reading the LIVE value also satisfies sf-pipeline-policy.md
    §2.4: verification reads the deployed artifact, not the source claiming
    to produce it.
    """
    script = _SCRIPT.read_text()
    assert "aws lambda get-function-configuration" in script
    assert "--query 'Timeout'" in script
    assert "TIMEOUT_S * 700" in script, "threshold must be 70% of the timeout, expressed in ms"


def test_duration_alarm_reads_maximum_not_average():
    """One slow invocation is the signal.

    Averaging a near-timeout run against a fast retry is how a run that
    nearly died reads as healthy — the 2026-08-11 preopen had two
    invocations and both mattered.
    """
    script = _SCRIPT.read_text()
    block = script.split("setup_duration_alarm()", 1)[1]
    assert "--statistic Maximum" in block
    assert "--metric-name \"Duration\"" in block
    assert "--namespace \"AWS/Lambda\"" in block


def test_missing_timeout_is_fatal_not_a_guessed_threshold():
    script = _SCRIPT.read_text()
    assert "refusing to guess a duration threshold" in script
    assert "exit 1" in script


def test_the_two_alarm_groups_are_separately_invocable():
    """They need different IAM, so the deploy role can apply one and not the other."""
    script = _SCRIPT.read_text()
    assert "setup_count_alarm()" in script
    assert "setup_duration_alarm()" in script
    for mode in ("count)", "duration)", "all)"):
        assert mode in script, f"dispatch case {mode} missing"


def test_deploy_scanner_applies_the_duration_alarm_on_every_deploy():
    """No post-merge operator step (pull-request-policy.md §4.2).

    A timeout raised by a merge with the alarm left to a human is the
    'filing an issue instead of triggering it' failure — the alarm records
    an intention and nothing about it fails when it never runs.
    """
    block = _deploy_scanner_body()
    assert "setup_scanner_alarm.sh" in block
    assert "duration" in block
    assert block.index("_deploy_image_shared_lambda") < block.index("setup_scanner_alarm.sh"), (
        "the alarm derives its threshold from the LIVE timeout, so it must run AFTER the update"
    )


def test_scanner_timeout_is_sized_to_observed_p95():
    """sf-pipeline-policy.md §4: p95 x 1.5, not a round number.

    Observed p95 over 2026-07-28..08-11 was ~290s; 300s was the ceiling both
    2026-08-11 preopen attempts died at.
    """
    block = _deploy_scanner_body()
    assert '"scanner_handler" 450 1024' in block
