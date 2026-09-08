#!/usr/bin/env bash
# setup_eval_alarms.sh — RETIRED as an alarm-creation path.
#
# alpha-engine-config-I10182, completing the config-I7339 ownership ruling: an
# alarm is an account resource (infrastructure-ownership-policy.md §2), and
# this repo is PUBLIC while the alarm definitions and their applier live in
# the PRIVATE nous-ergon-ops repo. This script's `put-metric-alarm` calls are
# gone — tests/test_eval_alarm_setup.py fails the build if they come back.
#
# WHY IT HAD TO STOP, not merely why it moved. `nous-ergon-ops/infrastructure/
# cloudwatch/alarms/` already codified all three eval alarms and applied them
# on merge via cloudwatch-alarm-apply-on-merge.yml — this script duplicated
# that authorship on every deploy.yml push to this repo. Two appliers, one
# resource: the live alarm was whichever ran last, and nothing anywhere would
# have failed if the two declarations disagreed. Found while fixing
# alpha-engine-config-I10166: the eval-control-breach alarm's period and
# missing-data policy had to be edited in lockstep by hand, across two repos,
# in two PRs.
#
# Every alarm this script used to create is codified as a JSON file:
#
#   nous-ergon-ops/infrastructure/cloudwatch/alarms/
#     alpha-engine-eval-quality-regression.json         (MetricName agent_quality_score_4w_mean_min)
#     alpha-engine-eval-control-breach.json              (MetricName agent_quality_score_control_breach_count)
#     alpha-engine-eval-control-bands-no-datapoint.json  (MetricName agent_quality_score_control_breach_count)
#
# To change one, edit that file in nous-ergon-ops and open a PR there — never
# edit this script and run it. To apply immediately rather than waiting for
# the next merge touching that file, an operator with nous-ergon-ops checked
# out runs:
#
#   infrastructure/cloudwatch/apply.py --prefix alpha-engine-eval-
#
# THE METRIC-NAME CONTRACT IS NOT LOST. `tests/test_eval_alarm_setup.py`
# still fails the build if either producer constant below stops matching what
# the alarms above watch — the exact reason this script existed in the first
# place, without needing a live command line to check it against:
#
#   evals/control_bands.py::BREACH_COUNT_METRIC_NAME  = "agent_quality_score_control_breach_count"
#   evals/rolling_mean.py::DERIVED_FLOOR_METRIC_NAME   = "agent_quality_score_4w_mean_min"
#
# This file is a pointer, not a stub with hidden behavior: it does nothing and
# exits 0 so a stale muscle-memory invocation is a no-op, not a failure.

set -euo pipefail
echo "setup_eval_alarms.sh no longer creates alarms (alpha-engine-config-I10182)."
echo "Edit nous-ergon-ops/infrastructure/cloudwatch/alarms/alpha-engine-eval-*.json instead."
echo "To apply immediately: nous-ergon-ops/infrastructure/cloudwatch/apply.py --prefix alpha-engine-eval-"
exit 0
