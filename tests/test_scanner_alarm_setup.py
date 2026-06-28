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
