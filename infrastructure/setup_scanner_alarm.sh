#!/usr/bin/env bash
#
# Idempotent CloudWatch setup for the scanner-output observability gap
# (config#785). Re-runnable: both put-metric-filter and put-metric-alarm
# are upserts, so running this repeatedly converges to the declared state
# and never duplicates.
#
# Why this exists (config#785, ROADMAP follow-up to #238):
#   * The standalone scanner Lambda (alpha-engine-research-scanner) logs
#     its candidate count every run, but NOTHING paged on it. A silent
#     6-month regression hid in plain sight precisely because there was
#     no operator-visible counter on scanner output — a count collapsing
#     from a ~50-60 baseline to near-zero (gate-tuning regression or an
#     upstream feature-store coverage drop) went unnoticed.
#   * This brings the gap under infra-as-code, mirroring the eval-alarm
#     pattern in infrastructure/setup_eval_alarms.sh.
#
# How the count reaches CloudWatch:
#   scanner_handler.py emits a dedicated marker line in text-log mode:
#       <ts> <lvl> [scanner] [scanner_handler] METRIC scanner_tickers_count <n>
#   The space-delimited metric filter below binds the trailing integer
#   <n> as the metric value. (The human-readable "done ... scanner_tickers=N"
#   line cannot be used directly: a CloudWatch filter cannot split the
#   "scanner_tickers=N" token on "=" to extract the integer.)
#   tests/test_scanner_metric_marker.py locks the marker format against
#   drift; tests/test_scanner_alarm_setup.py locks this script.
#
# Two alarm groups, separately invocable because they need DIFFERENT IAM:
#
#   count     the log metric filter + candidate-count floor. Needs
#             logs:PutMetricFilter on /aws/lambda/*, which
#             github-actions-lambda-deploy deliberately does NOT hold (its
#             logs grants are scoped to the Step Functions log groups), so
#             this group stays an operator command run under an admin
#             identity.
#   duration  the runtime-headroom alarm (alpha-engine-config-I6855). Needs
#             only cloudwatch:PutMetricAlarm on alarm:alpha-engine-* and
#             lambda:GetFunctionConfiguration, both of which the deploy role
#             already holds — so deploy.sh's deploy_scanner() calls it on
#             every merge and the alarm can never drift from the timeout it
#             is derived from.
#
# Splitting on the IAM boundary rather than widening the deploy role keeps
# the applier from gaining write access to Lambda log groups for one alarm.
#
# Usage:
#   bash infrastructure/setup_scanner_alarm.sh            # both groups
#   bash infrastructure/setup_scanner_alarm.sh count
#   bash infrastructure/setup_scanner_alarm.sh duration
#   SNS_TOPIC_ARN=arn:aws:sns:...:my-topic bash infrastructure/setup_scanner_alarm.sh

set -euo pipefail

SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
REGION="${AWS_REGION:-us-east-1}"
NAMESPACE="AlphaEngine/Scanner"

FUNCTION_SCANNER="alpha-engine-research-scanner"
LOG_GROUP="/aws/lambda/${FUNCTION_SCANNER}"
METRIC_NAME="scanner_tickers_count"
METRIC_FILTER_NAME="scanner-tickers-count"
ALARM_NAME="alpha-engine-scanner-tickers-degradation"

# Floor = 25 (config#785: half of the ~50 baseline). A scanner run that
# returns fewer candidates than this is treated as a degradation.
THRESHOLD=25

echo "[setup_scanner_alarm] SNS=${SNS_TOPIC_ARN} region=${REGION} namespace=${NAMESPACE}"

setup_count_alarm() {
# ── Metric filter: extract the candidate count from the scanner log ───────
# Pattern is space-delimited. Token layout of the marker line (text mode):
#   date time level [scanner] [scanner_handler] METRIC scanner_tickers_count <n>
# We anchor on the literal "METRIC" + metric-name tokens and bind the
# trailing integer as the metric value ($count). metricValue=$count emits
# the actual count (not "1"); defaultValue is intentionally omitted so the
# alarm's treat-missing-data governs no-run days rather than a synthetic 0.
echo "[setup_scanner_alarm] put metric filter ${METRIC_FILTER_NAME} on ${LOG_GROUP}"
# shellcheck disable=SC2016  # metricValue='$count' is a CloudWatch filter
# token reference (the [...] pattern's $count binding), not a shell var; it
# must reach AWS literally, so single quotes are intentional.
aws logs put-metric-filter \
  --log-group-name "${LOG_GROUP}" \
  --filter-name "${METRIC_FILTER_NAME}" \
  --filter-pattern '[date, time, level, component, handler, marker="METRIC", name="scanner_tickers_count", count]' \
  --metric-transformations \
      metricName="${METRIC_NAME}",metricNamespace="${NAMESPACE}",metricValue='$count' \
  --region "${REGION}"

# ── Alarm: fire when the candidate count drops below the floor ────────────
# 1 datapoint (1 evaluation period). The scanner runs weekly, so the
# metric is sparse; treat-missing-data=notBreaching keeps the alarm OK
# (not INSUFFICIENT_DATA) between runs and avoids paging on no-run days —
# only a real low-count datapoint trips it. period 86400 / Minimum mirrors
# the eval alarms' daily cadence: any day with a datapoint below the floor
# breaches.
echo "[setup_scanner_alarm] put ${ALARM_NAME} (${METRIC_NAME} < ${THRESHOLD})"
aws cloudwatch put-metric-alarm \
  --alarm-name "${ALARM_NAME}" \
  --alarm-description "Scanner output floor (config#785): a scanner run produced < ${THRESHOLD} candidates (scanner_tickers_count), half the ~50 baseline — likely a gate-tuning regression or upstream feature-store coverage drop. Emitted by lambda/scanner_handler.py." \
  --namespace "${NAMESPACE}" \
  --metric-name "${METRIC_NAME}" \
  --statistic Minimum \
  --period 86400 \
  --evaluation-periods 1 \
  --datapoints-to-alarm 1 \
  --threshold "${THRESHOLD}" \
  --comparison-operator LessThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "${SNS_TOPIC_ARN}" \
  --ok-actions "${SNS_TOPIC_ARN}" \
  --region "${REGION}"
}

setup_duration_alarm() {
# ── Alarm: fire when the run's DURATION approaches the function timeout ───
# alpha-engine-config-I6855. The count alarm above answers "did the scanner
# produce enough names?" and says nothing about "is the scanner about to stop
# producing anything at all". On 2026-08-11 both preopen attempts died at
# exactly 300.00s and the pipeline terminated DEGRADED — and the two weeks of
# creep that led there (p95 ~290s, 7/31 within 3% of the ceiling, 8/6 already
# timing out once) paged nobody, because nothing watched duration.
#
# Threshold is 70% of the function timeout, not a fixed number of seconds:
# the point is headroom remaining, so the two move together and a future
# timeout raise cannot silently widen the blind spot. Read the LIVE timeout
# rather than hardcoding deploy.sh's literal — the alarm must describe the
# function that exists, not the one the repo intends (sf-pipeline-policy.md
# §2.4: verification reads the deployed artifact).
#
# Maximum (not Average) over a 1-day period: one slow invocation is the
# signal, and averaging it against a fast retry is how a run that nearly
# died reads as healthy. treat-missing-data=notBreaching for the same
# no-run-day reason as the count alarm.
TIMEOUT_S="$(aws lambda get-function-configuration \
  --function-name "${FUNCTION_SCANNER}" \
  --query 'Timeout' --output text --region "${REGION}")"

if [[ -z "${TIMEOUT_S}" || "${TIMEOUT_S}" == "None" ]]; then
  echo "[setup_scanner_alarm] FATAL: could not read ${FUNCTION_SCANNER} Timeout — refusing to guess a duration threshold." >&2
  exit 1
fi

DURATION_ALARM_NAME="alpha-engine-scanner-duration-headroom"
DURATION_THRESHOLD_MS=$(( TIMEOUT_S * 700 ))  # 70% of timeout, in ms

echo "[setup_scanner_alarm] put ${DURATION_ALARM_NAME} (Duration > ${DURATION_THRESHOLD_MS}ms = 70% of ${TIMEOUT_S}s)"
aws cloudwatch put-metric-alarm \
  --alarm-name "${DURATION_ALARM_NAME}" \
  --alarm-description "Scanner runtime headroom (alpha-engine-config-I6855): a scanner invocation used more than 70% of its ${TIMEOUT_S}s timeout. This is the creep warning, not the failure — when it breaches, find what grew before raising the timeout (sf-pipeline-policy.md §4: timeouts are budgets, not accommodations). Both 2026-08-11 preopen attempts died at the ceiling and cost that day's universe board, membership, leaderboard and trajectory." \
  --namespace "AWS/Lambda" \
  --metric-name "Duration" \
  --dimensions Name=FunctionName,Value="${FUNCTION_SCANNER}" \
  --statistic Maximum \
  --period 86400 \
  --evaluation-periods 1 \
  --datapoints-to-alarm 1 \
  --threshold "${DURATION_THRESHOLD_MS}" \
  --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions "${SNS_TOPIC_ARN}" \
  --ok-actions "${SNS_TOPIC_ARN}" \
  --region "${REGION}"
}

case "${1:-all}" in
  count)    setup_count_alarm ;;
  duration) setup_duration_alarm ;;
  all)      setup_count_alarm; setup_duration_alarm ;;
  *)        echo "Usage: $0 [all|count|duration]" >&2; exit 1 ;;
esac

echo "[setup_scanner_alarm] done — ${1:-all} converged."
