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

# ── SUPERSEDED — this script must not run (alpha-engine-config-I5720) ────────
#
# The §47 spot cutover (alpha-engine-config-I5208) moved the daily Think Tank
# off this Lambda onto a self-terminating EC2 spot box behind
# `alpha-engine-thinktank-spot-dispatcher` (nousergon-data). That cutover
# repointed the SAME EventBridge rule this script targets, and deliberately
# DELETED the two alarms this script creates.
#
# So re-running this would silently revert the migration, twice over:
#
#   1. `put-targets` on `alpha-research-thinktank-daily` is an upsert, and two
#      scripts in two repos now aim it at different functions. Last writer
#      wins. This one wins it back to the 900s Lambda that died mid-loop every
#      day for 11 days — the exact failure I5208 exists to close.
#   2. It re-creates `...-daily-run-failed` and `...-daily-run-failed-timeout`
#      on a function that no longer runs the work. Both carry
#      `--treat-missing-data notBreaching`, so zero invocations evaluates to
#      OK: they would sit GREEN forever while nothing ran. The dispatcher's
#      deploy.sh deletes them for precisely that reason, in the same action
#      that blinds them.
#
# The header below still describes the pre-cutover world and is kept verbatim
# as the record of what this script used to own. Nothing here is live.
#
# Current ownership:
#   rule + dispatch alarm  ->  nousergon-data/infrastructure/lambdas/
#                              thinktank-spot-dispatcher/deploy.sh --cutover
#   end-to-end run health  ->  ARTIFACT_REGISTRY row
#                              thinktank_challenger_selection (720min SLA)
#
# Refusing rather than deleting the file: the deploy checklist and two runbooks
# still name this path, and a missing script fails with "No such file" while a
# refusing one says where to go.
cat >&2 <<'SUPERSEDED'
[setup-thinktank-schedule] REFUSING TO RUN — superseded by the §47 spot cutover.

  This script points alpha-research-thinktank-daily at
  alpha-engine-research-thinktank:live and arms two alarms on that function.
  The daily Think Tank no longer runs there. Running this would repoint the
  live rule back at the retired Lambda and re-create two alarms that read
  GREEN on zero invocations.

  To (re)provision the schedule:
    nousergon-data/infrastructure/lambdas/thinktank-spot-dispatcher/deploy.sh --cutover

  Tracked: alpha-engine-config-I5720 · alpha-engine-config-I5208
SUPERSEDED
exit 1


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

# alpha-engine-config-I5208 (2026-07-28) — the Think Tank ran silently dead for
# 11 days because every run hit the 900s Lambda ceiling mid-loop. The deadline
# guard (crucible-research-PR516) now truncates gracefully inside a 120s reserve,
# so a timeout means the guard itself failed or the terminal writes exceeded the
# reserve. Either way, it is a capacity signal: the run needs the §47 spot
# migration. The Error alarm above catches hard failures (handler raises); this
# alarm catches the capacity ceiling. A timeout with Duration=900000ms fires only
# when the Lambda actually hits the wall — a successful truncation inside the
# reserve exits cleanly (Duration < 900000) and does NOT fire.
TIMEOUT_ALARM="${ALARM_NAME}-timeout"

echo "[setup-thinktank-schedule] alarm ${TIMEOUT_ALARM} (Duration >= 899000 ms in a day → SNS)"

aws cloudwatch put-metric-alarm \
  --alarm-name "$TIMEOUT_ALARM" \
  --alarm-description "The daily think-tank run (config#1579) hit the 900s Lambda ceiling — the deadline guard (PR516) failed to truncate inside its 120s reserve, or the terminal writes exceeded the reserve. Capacity signal: the run needs the §47 spot migration (alpha-engine-config-I5208). Check /aws/lambda/${FUNCTION_THINKTANK} logs for deadline_truncated manifest fields." \
  --namespace "AWS/Lambda" \
  --metric-name Duration \
  --dimensions "Name=FunctionName,Value=${FUNCTION_THINKTANK}" \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --threshold 899000 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --region "$REGION"

echo ""
echo "Done. Rule ${RULE_THINKTANK} ENABLED (cron(30 14 * * ? *)); alarms ${ALARM_NAME} (Errors) + ${TIMEOUT_ALARM} (Timeout) armed."
echo ""
