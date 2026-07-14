"""Lock the think-tank daily schedule + failure alarm against drift
(config#1579 P1).

The daily cadence only works if four things stay in sync:

  1. ``infrastructure/setup-thinktank-schedule.sh`` schedules the rule at
     14:30 UTC daily — AFTER the weekday SF's RunDailyNews (~13:15-13:30
     UTC) so the events sweep sees same-day news, and after the Saturday
     SF's fresh weekly artifacts so themes reconcile the same day.
  2. The rule targets the ``live`` alias (deploy.sh publishes a version +
     moves ``live`` on every merge), so an alias revert governs the
     schedule immediately.
  3. ``lambda/thinktank_handler.py`` raises on failure (pinned in
     test_thinktank_handler.py), producing exactly 3 Errors datapoints
     for a definitively-failed run (initial + 2 EventBridge async
     retries).
  4. The alarm threshold is 3/day — it pages on "the run failed after
     all retries", not on self-healed transient blips.

If any drifts, the daily run silently stops or silently fails — the
bug classes these pins exist to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "infrastructure" / "setup-thinktank-schedule.sh"
_DEPLOY_SH = _REPO_ROOT / "infrastructure" / "deploy.sh"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_setup_script_exists():
    assert _SCRIPT.is_file()


def test_schedule_is_1430_utc_daily_seven_days():
    """cron(30 14 * * ? *) — 14:30 UTC every day, no day-of-week filter.
    Weekend runs are by-design (trading-day partitioning absorbs them)."""
    assert "cron(30 14 * * ? *)" in _script_text()


def test_rule_targets_live_alias():
    text = _script_text()
    assert 'FUNCTION_THINKTANK="alpha-engine-research-thinktank"' in text
    assert ":function:${FUNCTION_THINKTANK}:live" in text
    assert "--qualifier live" in text


def test_alarm_threshold_is_all_retries_exhausted():
    """Threshold 3 = initial invoke + both EventBridge async retries all
    raised. Lower would page on self-healed blips; higher can never fire
    (a day has at most 3 error datapoints from one scheduled run)."""
    text = _script_text()
    assert re.search(r"--threshold 3\b", text)
    assert "--treat-missing-data notBreaching" in text
    assert re.search(r"--period 86400\b", text)


def test_alarm_uses_shared_sns_topic():
    text = _script_text()
    assert "alpha-engine-alerts" in text
    assert "alpha-engine-thinktank-daily-run-failed" in text


def test_deploy_sh_ships_the_thinktank_target():
    """The schedule is only as fresh as the function behind it: deploy.sh
    must define the thinktank target (image-share, 900s timeout for the
    theme re-seed worst case) and stage the private thinktank.yaml into
    the image alongside scoring.yaml/universe.yaml."""
    text = _DEPLOY_SH.read_text(encoding="utf-8")
    assert 'FUNCTION_THINKTANK="alpha-engine-research-thinktank"' in text
    assert re.search(
        r'_deploy_image_shared_lambda "\$FUNCTION_THINKTANK" "thinktank_handler" 900 1024',
        text,
    )
    assert "for yaml in scoring.yaml universe.yaml thinktank.yaml; do" in text
