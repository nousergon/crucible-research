#!/usr/bin/env bash
#
# Idempotent EventBridge schedule + CloudWatch alarm for the daily
# think-tank Lambda (config#1579 P1). Re-runnable: put-rule, put-targets,
# add-permission (tolerated conflict) and put-metric-alarm are all
# upserts, so repeated runs converge and never duplicate.
#
# Schedule: 14:30 UTC (7:30 AM PT), 7 days/week.
#   * Weekdays: the weekday SF starts 12:45 UTC and its RunDailyNews
#     tail state lands data/news_aggregates_daily/ by ~13:15-13:30 UTC —
#     14:30 gives the events sweep a comfortable buffer to see SAME-DAY
#     news.
#   * Saturday: the weekly SF starts 09:00 UTC; by 14:30 the fresh
#     signals.json / archive/macro artifacts exist, so the themes layer
#     reconciles against the new weekly anchor the same day.
#   * Weekend/holiday runs are by-design: thinktank captures + events
#     partition to the last TRADING day (thinktank/capture.py), so they
#     accrue into Friday's partition; themes are churn-gated no-ops on
#     quiet days. The SSM monthly budget cap bounds spend regardless.
#
# Failure surface (no-silent-fails): thinktank_handler.py RAISES on any
# failure (no ERROR-dict returns — EventBridge async treats those as
# success), which drives the AWS/Lambda Errors metric. EventBridge async
# retries twice, so one logical failed run = exactly 3 Errors datapoints;
# the alarm threshold of 3/day therefore means "today's run definitively
# failed after all retries", while a transient provider blip that
# self-heals on retry (1-2 errors) does not page.
#
# Usage:
#   bash infrastructure/setup-thinktank-schedule.sh
#   SNS_TOPIC_ARN=arn:aws:sns:...:my-topic bash infrastructure/setup-thinktank-schedule.sh

set -euo pipefail

FUNCTION_THINKTANK="alpha-engine-research-thinktank"
RULE_THINKTANK="alpha-research-thinktank-daily"
ALARM_NAME="alpha-engine-thinktank-daily-run-failed"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# The live alias — deploy.sh's _deploy_image_shared_lambda publishes a
# version + moves 'live' on every deploy, so targeting the alias keeps
# the schedule on the blessed version and makes an alias revert
# (rollback.sh pattern) immediately govern what the schedule runs.
TARGET_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_THINKTANK}:live"

echo "[setup-thinktank-schedule] rule=${RULE_THINKTANK} → ${FUNCTION_THINKTANK}:live (14:30 UTC daily)"

aws events put-rule \
  --name "$RULE_THINKTANK" \
  --schedule-expression "cron(30 14 * * ? *)" \
  --state ENABLED \
  --description "Daily 14:30 UTC (7:30 AM PT, 7d/wk) — research think-tank run (config#1579): thesis intake + events sweep + theme updates" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_THINKTANK" \
  --targets '[{"Id":"1","Arn":"'"${TARGET_ARN}"'"}]' \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_THINKTANK" \
  --qualifier live \
  --statement-id "alpha-research-thinktank-daily" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_THINKTANK}" \
  --region "$REGION" 2>/dev/null || true

echo "[setup-thinktank-schedule] alarm ${ALARM_NAME} (Errors >= 3 / day → SNS)"

aws cloudwatch put-metric-alarm \
  --alarm-name "$ALARM_NAME" \
  --alarm-description "The daily think-tank run (config#1579) definitively failed: >= 3 Lambda Errors in a day = the initial invoke + both EventBridge async retries all raised (thinktank_handler.py raises on failure by contract). 1-2 errors = transient blip that self-healed on retry; no page. Check /aws/lambda/${FUNCTION_THINKTANK} logs, then re-invoke manually or wait for tomorrow's 14:30 UTC fire." \
  --namespace "AWS/Lambda" \
  --metric-name Errors \
  --dimensions "Name=FunctionName,Value=${FUNCTION_THINKTANK}" \
  --statistic Sum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 3 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --region "$REGION"

echo ""
echo "Done. Rule ${RULE_THINKTANK} ENABLED (cron(30 14 * * ? *)); alarm ${ALARM_NAME} armed."
echo ""
