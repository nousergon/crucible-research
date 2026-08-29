#!/usr/bin/env bash
#
# Idempotent CloudWatch alarm setup for the eval-quality observability
# surface (AlphaEngine/Eval namespace). Re-runnable: put-metric-alarm is
# an upsert, so running this repeatedly converges to the declared state
# and never duplicates.
#
# Why this exists (L4578e-alarm):
#   * The control-bands metric (agent_quality_score_control_breach_count,
#     emitted by evals/control_bands.py, L4578(e)) had NO alarm — the
#     drift detection was dark, emitting a metric nothing paged on.
#   * Auditing that gap surfaced a second one: the existing rolling-mean
#     floor alarm (alpha-engine-eval-quality-regression) was created
#     out-of-band and never codified — deploy.sh deferred it ("lands in
#     PR 4c") and it lived only as live AWS state. This script brings
#     BOTH under infra-as-code so neither drifts or gets lost.
#
# The metric names below MUST match the producer constants
# (evals/control_bands.py BREACH_COUNT_METRIC_NAME, evals/rolling_mean.py
# DERIVED_FLOOR_METRIC_NAME). tests/test_eval_alarm_setup.py locks that.
#
# Usage:
#   bash infrastructure/setup_eval_alarms.sh
#   SNS_TOPIC_ARN=arn:aws:sns:...:my-topic bash infrastructure/setup_eval_alarms.sh

set -euo pipefail

SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
NAMESPACE="AlphaEngine/Eval"

FLOOR_METRIC="agent_quality_score_4w_mean_min"
BREACH_METRIC="agent_quality_score_control_breach_count"

echo "[setup_eval_alarms] SNS=${SNS_TOPIC_ARN} namespace=${NAMESPACE}"

# ── Rolling-mean quality-floor alarm (ROADMAP §1634) ──────────────────────
# Fires when the MIN across all (agent,criterion,judge) 4-week means drops
# below 3.0 — an absolute-quality floor. Mirrors the live alarm exactly so
# this re-put is a no-op against existing state.
echo "[setup_eval_alarms] put alpha-engine-eval-quality-regression (${FLOOR_METRIC})"
# alpha-engine-config-I9321 — two corrections, both measured 2026-08-29.
#
# 1. `--period 86400` on a metric emitted ONCE A WEEK. Six of every seven
#    evaluation windows were empty by construction, which is why the period
#    and the missing-data policy have to change together: `breaching` on a
#    daily period would flap the alarm every week on a metric behaving
#    perfectly. 604800 matches what `EvalRollingMean` actually publishes.
#
# 2. `--treat-missing-data ignore` retains the LAST state when data stops.
#    The floor last published 2026-08-20; `AlphaEngine/Eval/agent_quality_score`
#    has zero live streams, so the floor is not being computed at all — and
#    `ignore` made a blind alarm indistinguishable from a breaching one on
#    every surface. `breaching` renders "we did not measure quality this week"
#    as a problem, which is what it is (`principles.md` §2.7: no data is never
#    rendered as green). The producer-side half of the same fix makes
#    `EvalRollingMean` FAIL rather than publish nothing quietly.
#
# Note on why Brian was never notified, which is a THIRD thing and not fixed
# by either line above: the floor has been below 3.0 in every datapoint since
# 2026-04-30 (measured: 1.0 -> 2.0 -> 2.006, never once >= 3.0). CloudWatch
# notifies on a TRANSITION, so this alarm paged once, on 2026-05-07, and
# structurally could not page again. A threshold alarm on a permanently
# breached level is a red light, not a pager. The change-detector that DOES
# transition is `alpha-engine-eval-control-breach` below, which moved 0->2 on
# 2026-08-27 and is the live channel for "this agent got worse".
#
# Re-putting with changed configuration RESETS alarm state to
# INSUFFICIENT_DATA, so this deploy also un-latches the 114-day-old ALARM and
# lets the next evaluation produce a real transition.
aws cloudwatch put-metric-alarm --alarm-name "alpha-engine-eval-quality-regression" --alarm-description "Eval quality floor: min 4-week-mean agent_quality_score < 3.0 (rolling_mean.py). Missing data is BREACHING (alpha-engine-config-I9321): an unpublished floor means quality went unmeasured, which is never reported as healthy." --namespace "${NAMESPACE}" --metric-name "${FLOOR_METRIC}" --statistic Minimum --period 604800 --evaluation-periods 1 --threshold 3.0 --comparison-operator LessThanThreshold --treat-missing-data breaching --alarm-actions "${SNS_TOPIC_ARN}" --ok-actions "${SNS_TOPIC_ARN}"

# ── Control-band breach alarm (L4578e) ────────────────────────────────────
# Fires when >= 1 combo is OUT_OF_CONTROL (a downward Shewhart or CUSUM
# breach) on the weekly control-band run. Catches drift/steps the flat
# floor misses. The metric is emitted every run (incl. 0), so the stream
# stays alive and the alarm sits OK rather than INSUFFICIENT_DATA between
# breaches. Maximum over the day == the weekly datapoint; missing days
# (no run) are ignored, matching the floor alarm's cadence.
echo "[setup_eval_alarms] put alpha-engine-eval-control-breach (${BREACH_METRIC})"
aws cloudwatch put-metric-alarm --alarm-name "alpha-engine-eval-control-breach" --alarm-description "Eval control bands (L4578e): >=1 (agent,criterion,judge) combo OUT_OF_CONTROL (downward Shewhart/CUSUM breach) in evals/control_bands.py." --namespace "${NAMESPACE}" --metric-name "${BREACH_METRIC}" --statistic Maximum --period 86400 --evaluation-periods 1 --threshold 1 --comparison-operator GreaterThanOrEqualToThreshold --treat-missing-data ignore --alarm-actions "${SNS_TOPIC_ARN}" --ok-actions "${SNS_TOPIC_ARN}"

echo "[setup_eval_alarms] done — both eval alarms converged."
