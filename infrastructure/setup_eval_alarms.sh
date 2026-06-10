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
aws cloudwatch put-metric-alarm --alarm-name "alpha-engine-eval-quality-regression" --alarm-description "Eval quality floor: min 4-week-mean agent_quality_score < 3.0 (rolling_mean.py)." --namespace "${NAMESPACE}" --metric-name "${FLOOR_METRIC}" --statistic Minimum --period 86400 --evaluation-periods 1 --threshold 3.0 --comparison-operator LessThanThreshold --treat-missing-data ignore --alarm-actions "${SNS_TOPIC_ARN}" --ok-actions "${SNS_TOPIC_ARN}"

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
