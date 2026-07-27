#!/usr/bin/env bash
#
# Idempotent EventBridge schedule + CloudWatch alarm for the OpenRouter
# shadow-judge Lambda (alpha-engine-config#2934). Re-runnable: put-rule,
# put-targets, add-permission (tolerated conflict) and put-metric-alarm
# are all upserts, so repeated runs converge and never duplicate.
#
# Standalone EventBridge cron + Lambda — the lower-blast-radius path
# config#2934 asked for, explicitly NOT a new state on the production
# Saturday Batches-API Submit/Poll/Process Step Functions chain (that
# stays untouched: no new SF states, no new IAM grants beyond the
# existing alpha-engine-research-role this Lambda shares with every
# other image-share handler, e.g. thinktank/scanner/eval_judge).
#
# Schedule: Sunday 10:00 UTC, weekly.
#   * The Saturday weekly SF (09:00 UTC start) runs Research, then its
#     own EvalJudge Submit/Poll/Process Batches-API chain scores that
#     day's captures with the primary Haiku/Sonnet judges. By Sunday
#     10:00 UTC that chain has had a full day (well past its own
#     multi-hour Batches-API poll ceiling) to land verdicts, so the
#     shadow tier scores the SAME capture partition the primary judges
#     already covered — exactly what compute_shadow_agreement (item 5,
#     evals/openrouter_shadow.py) needs to pair against.
#   * ``openrouter_shadow_handler.py`` defaults ``date`` to yesterday
#     UTC when the event omits it (Sunday run -> Saturday's partition),
#     so this plain, argument-free EventBridge target needs no
#     date-templating.
#
# Failure surface (no-silent-fails): openrouter_shadow_handler.py RAISES
# on any run-level failure (no ERROR-dict returns — EventBridge async
# treats those as success), which drives the AWS/Lambda Errors metric.
# EventBridge async retries twice, so one logical failed run = exactly 3
# Errors datapoints; the alarm threshold of 3/week mirrors
# setup-thinktank-schedule.sh's daily 3-per-day convention, scaled to
# this rule's weekly cadence. Per-artifact eval failures inside a
# successful run do NOT raise (evals.openrouter_shadow.
# run_shadow_judge_over_date accumulates those in its own ``failed``
# list) — this alarm only pages when the run itself blew up after both
# retries.
#
# Usage:
#   bash infrastructure/setup-openrouter-shadow-schedule.sh
#   SNS_TOPIC_ARN=arn:aws:sns:...:my-topic bash infrastructure/setup-openrouter-shadow-schedule.sh

set -euo pipefail

FUNCTION_OPENROUTER_SHADOW="alpha-engine-research-openrouter-shadow"
RULE_OPENROUTER_SHADOW="alpha-research-openrouter-shadow-weekly"
ALARM_NAME="alpha-engine-openrouter-shadow-weekly-run-failed"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# The live alias — deploy.sh's _deploy_image_shared_lambda publishes a
# version + moves 'live' on every deploy, so targeting the alias keeps
# the schedule on the blessed version and makes an alias revert
# (rollback.sh pattern) immediately govern what the schedule runs.
TARGET_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_OPENROUTER_SHADOW}:live"

echo "[setup-openrouter-shadow-schedule] rule=${RULE_OPENROUTER_SHADOW} -> ${FUNCTION_OPENROUTER_SHADOW}:live (Sunday 10:00 UTC)"

aws events put-rule \
  --name "$RULE_OPENROUTER_SHADOW" \
  --schedule-expression "cron(0 10 ? * SUN *)" \
  --state ENABLED \
  --description "Weekly Sunday 10:00 UTC — OpenRouter shadow-judge run over the prior day's (Saturday's) capture partition (alpha-engine-config#2934, crucible-research#470)" \
  --region "$REGION"

aws events put-targets \
  --rule "$RULE_OPENROUTER_SHADOW" \
  --targets '[{"Id":"1","Arn":"'"${TARGET_ARN}"'"}]' \
  --region "$REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_OPENROUTER_SHADOW" \
  --qualifier live \
  --statement-id "alpha-research-openrouter-shadow-weekly" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_OPENROUTER_SHADOW}" \
  --region "$REGION" 2>/dev/null || true

echo "[setup-openrouter-shadow-schedule] alarm ${ALARM_NAME} (Errors >= 3 / week -> SNS)"

aws cloudwatch put-metric-alarm \
  --alarm-name "$ALARM_NAME" \
  --alarm-description "The weekly OpenRouter shadow-judge run (alpha-engine-config#2934) definitively failed: >= 3 Lambda Errors in the week's evaluation period = the initial invoke + both EventBridge async retries all raised (openrouter_shadow_handler.py raises on run-level failure by contract). 1-2 errors = transient blip that self-healed on retry; no page. Check /aws/lambda/${FUNCTION_OPENROUTER_SHADOW} logs, then re-invoke manually (optionally with an explicit {\"date\": \"YYYY-MM-DD\"} event) or wait for next Sunday's 10:00 UTC fire." \
  --namespace "AWS/Lambda" \
  --metric-name Errors \
  --dimensions "Name=FunctionName,Value=${FUNCTION_OPENROUTER_SHADOW}" \
  --statistic Sum \
  --period 604800 \
  --evaluation-periods 1 \
  --threshold 3 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --region "$REGION"

echo ""
echo "Done. Rule ${RULE_OPENROUTER_SHADOW} ENABLED (cron(0 10 ? * SUN *)); alarm ${ALARM_NAME} armed."
echo ""
