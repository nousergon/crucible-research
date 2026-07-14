#!/usr/bin/env bash
#
# Idempotent EventBridge schedule + CloudWatch alarm for the think-tank
# MAINTENANCE Lambda run (config#1579 P1; cadence split 2026-07-14).
# Re-runnable: put-rule, put-targets, add-permission (tolerated conflict)
# and put-metric-alarm are all upserts, so repeated runs converge and
# never duplicate.
#
# 2026-07-14 cadence split (alpha-engine-config-I2487 incident + SOTA
# follow-up): this rule now covers THEME RECONCILIATION + EVENTS SWEEP
# on already-covered names ONLY — coverage growth/staleness refresh
# moved entirely to the Saturday SF's ThinkTankCoverage sf_cover step
# (nousergon-data/infrastructure/step_function.json, target/ceiling
# bumped to rank_ceiling=150 the same day). Rationale: the universe
# board (scanner/universe/latest.json) that intake ranks against is
# itself only refreshed on Saturday (Scanner runs inside the weekly SF
# immediately before ThinkTankCoverage), so a weekday intake pass had
# zero new ranking information to act on — pure redundant spend.
# `research/thinktank.yaml`'s base `coverage.daily_new_names: 0` is
# what makes this rule's runs skip intake (see that file's comment).
#
# Schedule: 14:30 UTC (7:30 AM PT), Mon/Wed/Fri — down from 7 days/week.
#   * Each firing day: the weekday SF starts 12:45 UTC and its
#     RunDailyNews tail state lands data/news_aggregates_daily/ by
#     ~13:15-13:30 UTC — 14:30 gives the events sweep a comfortable
#     buffer to see SAME-DAY news. Mon/Wed/Fri bounds macro/theme
#     staleness to <=2 trading days at any point (vs. up to 6 on a
#     weekly-only cadence) while cutting LLM sweep-call volume ~57%
#     vs. the old daily cadence.
#   * Saturday's reconciliation now happens via the SF's ThinkTankCoverage
#     step itself (mode=sf_cover runs the same themes.ensure_current()
#     code path) — this rule does not need to also fire Saturday.
#   * Weekend/holiday non-fire days are covered by the SF's own weekly
#     reconciliation; thinktank captures + events partition to the last
#     TRADING day (thinktank/capture.py) regardless of which day a run
#     lands on. The SSM monthly budget cap bounds spend regardless.
#
# Failure surface (no-silent-fails): thinktank_handler.py RAISES on any
# failure (no ERROR-dict returns — EventBridge async treats those as
# success), which drives the AWS/Lambda Errors metric. EventBridge async
# retries twice, so one logical failed run = exactly 3 Errors datapoints;
# the alarm threshold of 3/day therefore means "today's run definitively
# failed after all retries", while a transient provider blip that
# self-heals on retry (1-2 errors) does not page. Note this Errors metric
# is shared with the Saturday SF's sf_cover invocations of the same
# Lambda (no separate dimension) — a same-day overlap of a maintenance
# failure and an sf_cover failure could jointly cross the threshold;
# check the CloudWatch Logs timestamp to attribute which invocation.
#
# Usage:
#   bash infrastructure/setup-thinktank-schedule.sh
#   SNS_TOPIC_ARN=arn:aws:sns:...:my-topic bash infrastructure/setup-thinktank-schedule.sh

set -euo pipefail

FUNCTION_THINKTANK="alpha-engine-research-thinktank"
RULE_THINKTANK="alpha-research-thinktank-maintenance"
ALARM_NAME="alpha-engine-thinktank-maintenance-run-failed"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# The live alias — deploy.sh's _deploy_image_shared_lambda publishes a
# version + moves 'live' on every deploy, so targeting the alias keeps
# the schedule on the blessed version and makes an alias revert
# (rollback.sh pattern) immediately govern what the schedule runs.
TARGET_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_THINKTANK}:live"

echo "[setup-thinktank-schedule] rule=${RULE_THINKTANK} → ${FUNCTION_THINKTANK}:live (14:30 UTC Mon/Wed/Fri)"

aws events put-rule \
  --name "$RULE_THINKTANK" \
  --schedule-expression "cron(30 14 ? * MON,WED,FRI *)" \
  --state ENABLED \
  --description "Mon/Wed/Fri 14:30 UTC (7:30 AM PT) — research think-tank MAINTENANCE run (config#1579, cadence split 2026-07-14): theme reconciliation + events sweep only (daily_new_names=0 — coverage growth lives in the Saturday SF's sf_cover step)" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_THINKTANK" \
  --targets '[{"Id":"1","Arn":"'"${TARGET_ARN}"'"}]' \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_THINKTANK" \
  --qualifier live \
  --statement-id "alpha-research-thinktank-maintenance" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_THINKTANK}" \
  --region "$REGION" 2>/dev/null || true

echo "[setup-thinktank-schedule] alarm ${ALARM_NAME} (Errors >= 3 / day → SNS)"

aws cloudwatch put-metric-alarm \
  --alarm-name "$ALARM_NAME" \
  --alarm-description "The think-tank maintenance run (config#1579, cadence split 2026-07-14) definitively failed: >= 3 Lambda Errors in a day = the initial invoke + both EventBridge async retries all raised (thinktank_handler.py raises on failure by contract). 1-2 errors = transient blip that self-healed on retry; no page. Shared Errors metric with the Saturday SF's sf_cover invocations of the same Lambda — check /aws/lambda/${FUNCTION_THINKTANK} logs to attribute. Then re-invoke manually or wait for the next Mon/Wed/Fri 14:30 UTC fire." \
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
echo "Done. Rule ${RULE_THINKTANK} ENABLED (cron(30 14 ? * MON,WED,FRI *)); alarm ${ALARM_NAME} armed."
echo ""
echo "NOTE: the prior daily rule/alarm (alpha-research-thinktank-daily /"
echo "alpha-engine-thinktank-daily-run-failed) are NOT deleted by this script"
echo "— decommission them separately (aws events remove-targets + delete-rule,"
echo "aws cloudwatch delete-alarms) once this rule is confirmed healthy."
echo ""
