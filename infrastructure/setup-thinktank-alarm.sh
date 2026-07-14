#!/usr/bin/env bash
#
# Idempotent CloudWatch failure alarm for the think-tank Lambda
# (config#1579 P1). Re-runnable: put-metric-alarm is an upsert.
#
# 2026-07-14 cadence consolidation (alpha-engine-config-I2487 incident +
# SOTA follow-up): the standalone EventBridge schedule is RETIRED. The
# Lambda is now invoked ONLY by the Saturday weekly SF's
# ThinkTankCoverage state (mode=sf_cover) — no separate daily/weekday
# rule exists. Rationale: sf_cover mode's run_daily() already performs
# theme reconciliation + full events sweep + intake unconditionally
# (mode only overrides intake sizing), so a separate weekday invocation
# was pure duplication of work the Saturday run already does; and the
# universe board this Lambda's intake ranks against is itself only
# refreshed on Saturday (Scanner runs immediately before this state in
# the same SF branch), so a weekday pass never had new ranking data to
# act on. This file used to ALSO manage that rule; it no longer does —
# `setup-thinktank-schedule.sh` (deleted this change) is gone.
#
# This script now manages ONLY the failure alarm. The SF's
# ThinkTankCoverage state carries a Retry block mirroring the sibling
# `Research` state's bridge pattern (States.Timeout/Lambda.Unknown, 1
# retry, 60s interval) — worst case for a fully-exhausted-retries
# failure is 2 Errors datapoints (initial + 1 retry), not the 3 the old
# EventBridge-daily alarm assumed (1 initial + 2 EventBridge async
# retries). Threshold is 2 accordingly.
#
# Failure surface (no-silent-fails): thinktank_handler.py RAISES on any
# failure (no ERROR-dict returns), which drives the AWS/Lambda Errors
# metric AND is what the SF Task's Catch depends on to route to its
# non-blocking continuation (CheckSkipRAGIngestion) — a raise is
# required for both the SF Catch routing and this alarm to work.
#
# Usage:
#   bash infrastructure/setup-thinktank-alarm.sh
#   SNS_TOPIC_ARN=arn:aws:sns:...:my-topic bash infrastructure/setup-thinktank-alarm.sh

set -euo pipefail

FUNCTION_THINKTANK="alpha-engine-research-thinktank"
ALARM_NAME="alpha-engine-thinktank-run-failed"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
REGION="${AWS_REGION:-us-east-1}"

echo "[setup-thinktank-alarm] alarm ${ALARM_NAME} (Errors >= 2 / day → SNS)"

aws cloudwatch put-metric-alarm \
  --alarm-name "$ALARM_NAME" \
  --alarm-description "The think-tank Saturday SF ThinkTankCoverage run (config#1579, cadence consolidation 2026-07-14) definitively failed: >= 2 Lambda Errors in a day = the initial invoke + the SF state's one Retry (States.Timeout/Lambda.Unknown) both raised (thinktank_handler.py raises on failure by contract). Non-blocking Catch means the SF itself won't fail — this alarm is the only loud signal. Check /aws/lambda/${FUNCTION_THINKTANK} logs, then re-invoke manually with mode=sf_cover or wait for next Saturday's SF run." \
  --namespace "AWS/Lambda" \
  --metric-name Errors \
  --dimensions "Name=FunctionName,Value=${FUNCTION_THINKTANK}" \
  --statistic Sum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 2 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --region "$REGION"

echo ""
echo "Done. Alarm ${ALARM_NAME} armed (Errors >= 2/day)."
echo ""
echo "NOTE: the prior daily EventBridge rule (alpha-research-thinktank-daily)"
echo "and its alarm (alpha-engine-thinktank-daily-run-failed) are NOT deleted"
echo "by this script — decommission them separately (aws events"
echo "remove-targets + delete-rule, aws cloudwatch delete-alarms) once this"
echo "alarm is confirmed healthy."
echo ""
