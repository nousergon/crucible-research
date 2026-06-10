"""Lock the eval-alarm setup script against producer-metric drift (L4578e).

If a producer renames its metric constant, the alarm in
infrastructure/setup_eval_alarms.sh silently stops covering it. This
test fails the build when the script's metric names diverge from the
constants the producers actually emit.
"""

from __future__ import annotations

from pathlib import Path

from evals.control_bands import BREACH_COUNT_METRIC_NAME
from evals.rolling_mean import DERIVED_FLOOR_METRIC_NAME

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "infrastructure"
    / "setup_eval_alarms.sh"
)


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_setup_script_exists():
    assert _SCRIPT.is_file()


def test_control_breach_metric_name_matches_producer():
    assert BREACH_COUNT_METRIC_NAME in _script_text(), (
        f"setup_eval_alarms.sh does not reference the control-breach "
        f"metric {BREACH_COUNT_METRIC_NAME!r} — the alarm is orphaned "
        f"from its producer (evals/control_bands.py). Update the script."
    )


def test_floor_metric_name_matches_producer():
    assert DERIVED_FLOOR_METRIC_NAME in _script_text(), (
        f"setup_eval_alarms.sh does not reference the quality-floor "
        f"metric {DERIVED_FLOOR_METRIC_NAME!r} — the alarm is orphaned "
        f"from its producer (evals/rolling_mean.py). Update the script."
    )


def test_alarms_use_the_shared_sns_topic():
    text = _script_text()
    assert "alpha-engine-alerts" in text
    assert "alpha-engine-eval-control-breach" in text
    assert "alpha-engine-eval-quality-regression" in text
