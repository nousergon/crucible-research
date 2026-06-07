#!/usr/bin/env bash
# infrastructure/spot_research_weekly.sh — Run the weekly Research pipeline on a spot EC2.
#
# Migrates Research OFF the 15-min Lambda ceiling (ROADMAP L4509). Research is
# the only heavy Saturday-SF step still on a Lambda; DataPhase1 / RAGIngestion /
# PredictorTraining / Backtester all run on EC2 spot precisely to escape the
# unextendable 15-min wall. The L4508 deadline guard reduces but cannot
# eliminate the failure probability under extreme sector-team tail latency
# (15 min is the absolute Lambda max). This launcher removes the wall entirely.
#
# It launches a capacity-resilient spot, clones alpha-engine-research, stages
# the private research config (universe.yaml + scoring.yaml) from the
# dispatcher's alpha-engine-config clone via S3, runs `python main.py --date
# <trading_day>` (the SAME LangGraph pipeline the Lambda handler drives — writes
# signals.json to S3 + sends the morning email), emits a heartbeat, and
# self-terminates. Mirrors infrastructure/spot_data_weekly.sh (the gold-standard
# spot launcher) so no new IAM / security-group / subnet resources are
# introduced.
#
# Transport is `aws ssm send-command` wrapped at the lib chokepoint
# `python -m alpha_engine_lib.ssm_dispatcher run` (IAM-authenticated,
# CloudTrail-audited, no port-22 inbound). Secrets (ANTHROPIC_API_KEY,
# FMP_API_KEY, FRED_API_KEY) resolve on the spot from SSM via
# alpha_engine_lib.secrets.get_secret() at Python startup — the spot's IAM
# profile (alpha-engine-executor-profile) grants ssm:GetParameter on
# /alpha-engine/*; no .env, no manual export.
#
# Usage:
#   ./infrastructure/spot_research_weekly.sh                      # full run, date = most-recent trading day
#   ./infrastructure/spot_research_weekly.sh --date 2026-06-05    # full run for an explicit trading day
#   ./infrastructure/spot_research_weekly.sh --smoke-only         # imports + --stub-llm dry run, then terminate
#   ./infrastructure/spot_research_weekly.sh --preflight-only     # boot + ResearchPreflight, exit 0 (NO LLM spend, NO S3 write)
#   ./infrastructure/spot_research_weekly.sh --instance-type c5.xlarge
#   ./infrastructure/spot_research_weekly.sh --branch my-branch
#
# Prerequisites on the launching host (ae-dashboard when invoked by the
# Saturday Step Function):
#   - AWS CLI with RunInstances / TerminateInstances / DescribeInstances /
#     ssm:SendCommand / GetCommandInvocation perms
#   - alpha-engine-research checked out at the script's parent dir
#   - alpha-engine-config cloned (provides research/universe.yaml + scoring.yaml)
#   - alpha-engine-lib installed in the dispatcher's .venv (LIB_PYTHON) —
#     provides both `ec2_spot` and `ssm_dispatcher` CLIs

set -euo pipefail

# SSM RunCommand does not set HOME; default it for the config-file lookup below.
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Spot configuration ──────────────────────────────────────────────────────
# Values mirror infrastructure/spot_data_weekly.sh so no new IAM / SG / subnet
# resources are introduced. If any change there, change them here in lockstep.
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
BRANCH="${BRANCH:-main}"
# Capacity-resilient instance-type fallback set (all 2 vCPU / 4-8 GB RAM).
# Research is LLM-orchestration-bound (network/IO on Anthropic calls), not
# memory-heavy, so c5.large is ample; the rotation only matters for capacity.
INSTANCE_TYPES="${INSTANCE_TYPES:-c5.large,m5.large,c6i.large,c5a.large}"
INSTANCE_TYPE=""
AMI_ID="ami-0c421724a94bba6d6"      # Amazon Linux 2023 x86_64
# Research historically runs 5-9 min, tail-spiked to ~13 min on 2026-06-06.
# 45 min covers boot + pip install + a 3x tail with headroom — no Lambda wall.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-2700}"
# ── Spot-interruption resilience (mirrors spot_data_weekly.sh) ───────────────
# A confirmed spot reclamation (no-capacity / by-price / oversubscribed)
# relaunches a fresh spot up to MAX_SPOT_ATTEMPTS; a genuine workload error
# (the inner script raised) is NOT retried — fail loud per the fail-fast
# posture. SPOT_ATTEMPT is threaded across re-execs via the env.
MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-2}"
SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
SPOT_RETRY_BACKOFF_SECONDS="${SPOT_RETRY_BACKOFF_SECONDS:-20}"
# Key-pair kept ONLY for ec2_spot's --key-name flag — nothing SSH's in.
KEY_NAME="alpha-engine-key"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
IAM_PROFILE="alpha-engine-executor-profile"
LIB_PYTHON="${LIB_PYTHON:-/home/ec2-user/alpha-engine-dashboard/.venv/bin/python}"

# ── Parse flags ──────────────────────────────────────────────────────────────
# RUN_MODE:
#   full           — python main.py --date <trading_day> (writes signals.json + email)
#   smoke-only     — import main + --stub-llm dry run ($0 tokens), then terminate
# PREFLIGHT_ONLY modifier: boot + ResearchPreflight + exit 0 (NO LLM, NO write).
RUN_MODE="full"
PREFLIGHT_ONLY=0
RUN_DATE=""
ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-only) RUN_MODE="smoke-only"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --date) RUN_DATE="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════"
echo "  Weekly Research Spot Run — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"
if [ -n "$INSTANCE_TYPE" ]; then
    INSTANCE_TYPES="$INSTANCE_TYPE"
fi
echo "  Instance types: $INSTANCE_TYPES"
echo "  Subnets       : $SUBNETS"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Run mode      : $RUN_MODE"
echo "  Run date      : ${RUN_DATE:-<most-recent trading day, computed on spot>}"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS  (relaunch on confirmed spot interruption)"
echo "  Preflight-only: $PREFLIGHT_ONLY  (1 = boot + preflight + exit 0, NO LLM/write)"
echo "  S3 bucket     : $S3_BUCKET"
echo "  Transport     : SSM via lib chokepoint (python -m alpha_engine_lib.ssm_dispatcher)"
echo ""

# ── Locate the private research config on the dispatcher ─────────────────────
# config.py::_find_config searches ~/alpha-engine-config/research/<file> first;
# the dispatcher (ae-dashboard) clones the private config repo daily via
# boot-pull.sh. Stage both universe.yaml + scoring.yaml to S3 for the spot.
CONFIG_DIR="/home/ec2-user/alpha-engine-config/research"
if [ ! -d "$CONFIG_DIR" ]; then
    CONFIG_DIR="$HOME/Development/alpha-engine-config/research"
fi
for f in universe.yaml scoring.yaml; do
    if [ ! -f "$CONFIG_DIR/$f" ]; then
        echo "ERROR: research config not found at $CONFIG_DIR/$f — is alpha-engine-config cloned + pulled?"
        exit 1
    fi
done

# ── Cleanup + spot-interruption retry trap (mirrors spot_data_weekly.sh) ─────
INSTANCE_ID=""
S3_STAGING=""

cleanup() {
    if [ -n "$INSTANCE_ID" ]; then
        echo ""
        echo "==> Terminating spot instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
    fi
    [ -n "$S3_STAGING" ] && aws s3 rm "$S3_STAGING" --recursive --quiet 2>/dev/null || true
    [ -n "$INSTANCE_ID" ] && echo "  Instance terminated; S3 staging cleaned."
    return 0
}

# Echoes a non-empty reason + returns 0 when the just-failed run was a CONFIRMED
# spot interruption (launch exhaustion rc 64, or AWS reclaim). A genuine inner
# workload failure returns 1 → NOT retryable (blind retry would mask a real bug).
_spot_failure_reason() {
    local rc="$1"
    if [ "$rc" -eq 64 ]; then echo "launch-capacity-exhausted"; return 0; fi
    [ -z "$INSTANCE_ID" ] && return 1
    local sir_code
    sir_code=$(aws ec2 describe-spot-instance-requests \
        --filters "Name=instance-id,Values=$INSTANCE_ID" \
        --query 'SpotInstanceRequests[0].Status.Code' \
        --output text --region "$AWS_REGION" 2>/dev/null || echo "")
    case "$sir_code" in
        instance-terminated-no-capacity|instance-terminated-by-price|instance-terminated-capacity-oversubscribed|instance-stopped-no-capacity|instance-stopped-by-price|instance-stopped-capacity-oversubscribed|marked-for-termination)
            echo "$sir_code"; return 0 ;;
    esac
    local state_reason
    state_reason=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[].Instances[].StateReason.Code' \
        --output text --region "$AWS_REGION" 2>/dev/null || echo "")
    case "$state_reason" in
        Server.SpotInstanceTermination|Server.InsufficientInstanceCapacity)
            echo "$state_reason"; return 0 ;;
    esac
    return 1
}

on_exit() {
    local rc=$?
    local reason=""
    if [ "$rc" -ne 0 ]; then
        reason="$(_spot_failure_reason "$rc")" || reason=""
    fi
    cleanup
    if [ "$rc" -ne 0 ] && [ -n "$reason" ] && [ "$SPOT_ATTEMPT" -lt "$MAX_SPOT_ATTEMPTS" ]; then
        aws cloudwatch put-metric-data \
            --namespace "AlphaEngine" \
            --metric-name "SpotInterruptionRetry" \
            --dimensions "Process=research" \
            --value 1 --unit "Count" \
            --region "$AWS_REGION" 2>/dev/null || true
        echo "" >&2
        echo "==> Spot interruption (reason=$reason) on attempt $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS — relaunching a fresh spot in ${SPOT_RETRY_BACKOFF_SECONDS}s..." >&2
        sleep "$SPOT_RETRY_BACKOFF_SECONDS"
        trap - EXIT
        SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
    fi
    if [ "$rc" -ne 0 ] && [ -n "$reason" ]; then
        echo "ERROR: spot interruption (reason=$reason) persisted across all $MAX_SPOT_ATTEMPTS attempt(s) — giving up. The weekly research run fails loud; redrive once spot capacity returns." >&2
    fi
    exit "$rc"
}
trap on_exit EXIT

echo "==> Requesting spot instance (lib CLI rotation: types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."
INSTANCE_ID=$("$LIB_PYTHON" -m alpha_engine_lib.ec2_spot launch \
    --types "$INSTANCE_TYPES" \
    --subnets "$SUBNETS" \
    --image-id "$AMI_ID" \
    --key-name "$KEY_NAME" \
    --security-group "$SECURITY_GROUP" \
    --iam-profile "$IAM_PROFILE" \
    --name "alpha-engine-research-weekly-$(date +%Y%m%d)" \
    --region "$AWS_REGION")
ec2_spot_rc=$?
if [ "$ec2_spot_rc" -ne 0 ] || [ -z "$INSTANCE_ID" ]; then
    if [ "$ec2_spot_rc" -eq 64 ]; then
        echo "ERROR: capacity exhausted across all instance_type × subnet combinations. Wait + retry, or expand the lists." >&2
    fi
    exit "${ec2_spot_rc:-1}"
fi

echo "  Instance ID: $INSTANCE_ID"

RUN_ID="$(date +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
S3_STAGING_PREFIX="tmp/spot_research_weekly/${RUN_ID}"
S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

echo "==> Staging research config → ${S3_STAGING}/config/"
aws s3 cp "$CONFIG_DIR/universe.yaml" "${S3_STAGING}/config/universe.yaml" --region "$AWS_REGION" --quiet
aws s3 cp "$CONFIG_DIR/scoring.yaml" "${S3_STAGING}/config/scoring.yaml" --region "$AWS_REGION" --quiet

# ── Wait for the SSM agent to register ────────────────────────────────────────
echo "==> Waiting for SSM agent to come Online..."
for i in $(seq 1 36); do  # 36 × 5s = 180s budget
    ping=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' \
        --output text --region "$AWS_REGION" 2>/dev/null || true)
    if [ "$ping" = "Online" ]; then
        echo "  SSM agent Online."
        break
    fi
    if [ "$i" -eq 36 ]; then
        echo "ERROR: SSM agent not Online after 180s (instance $INSTANCE_ID)"
        exit 1
    fi
    sleep 5
done

# ── SSM dispatch primitive (lib chokepoint) ──────────────────────────────────
# run_ssm "<description>" [timeout_seconds] <<HEREDOC ... HEREDOC
# Stdin-fed so the script body (with apostrophes / $(...) ) survives verbatim.
run_ssm() {
    local description="$1" timeout_s="${2:-2700}"
    "$LIB_PYTHON" -m alpha_engine_lib.ssm_dispatcher run \
        --instance-id "$INSTANCE_ID" \
        --description "research-weekly: $description" \
        --timeout "$timeout_s" \
        --output-bucket "$S3_BUCKET" \
        --output-key-prefix "${S3_STAGING_PREFIX}/ssm-output" \
        --region "$AWS_REGION" \
        --diagnostics-bucket "$S3_BUCKET" \
        --diagnostics-prefix "_spot_diagnostics/ae-research" \
        --script-stdin
}

# Each run_ssm step is a fresh SSM shell with a minimal env. AL2023 spots
# install python3.12 but have no bare `python` symlink; main.py is invoked via
# $PYTHON_BIN. AWS_REGION/AWS_DEFAULT_REGION are required by boto3 +
# get_secret (single-region us-east-1).
read -r -d '' ENV_SOURCE <<'ENV_EOF' || true
export HOME=/home/ec2-user
export XDG_CACHE_HOME=/tmp
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
command -v python3.12 >/dev/null && PYTHON_BIN=python3.12 || PYTHON_BIN=python3
export PYTHON_BIN
ENV_EOF

# ── Bootstrap: watchdog + python + git + clone + config ──────────────────────
echo "==> Bootstrapping spot (watchdog, python, clone, config)..."
run_ssm "bootstrap" 600 <<BOOTSTRAP
set -eo pipefail
${ENV_SOURCE}

# Spot-side hard-timeout watchdog: shuts the box down after MAX_RUNTIME_SECONDS
# regardless of dispatcher state (orphan-reaper backstop). AL2023 spots default
# InstanceInitiatedShutdownBehavior=terminate, so shutdown = instance gone.
systemd-run --on-active=${MAX_RUNTIME_SECONDS} --unit=alpha-engine-watchdog \
    --description='alpha-engine research spot hard-timeout' /sbin/shutdown -h now

dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
    dnf install -y -q python3 python3-pip python3-devel git gcc
echo "Using: \$(\$PYTHON_BIN --version)"

git clone --depth 1 --branch ${BRANCH} https://github.com/cipher813/alpha-engine-research.git /home/ec2-user/alpha-engine-research

mkdir -p /home/ec2-user/alpha-engine-config/research
aws s3 cp ${S3_STAGING}/config/universe.yaml /home/ec2-user/alpha-engine-config/research/universe.yaml --region ${AWS_REGION} --quiet
aws s3 cp ${S3_STAGING}/config/scoring.yaml /home/ec2-user/alpha-engine-config/research/scoring.yaml --region ${AWS_REGION} --quiet
echo "Bootstrap complete: repo cloned, research config staged."
BOOTSTRAP

# ── Install python deps ─────────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
run_ssm "deps" 900 <<DEPS
set -eo pipefail
${ENV_SOURCE}
cd /home/ec2-user/alpha-engine-research

PIP="\$PYTHON_BIN -m pip"
\$PIP install --upgrade pip -q
\$PIP install -q -r requirements.txt
# numpy<2 pin to match other spot workloads (pyarrow compiled against 1.x).
\$PIP install -q 'numpy<2'
echo "Dependencies installed."
DEPS

# ── Smoke-only: import + --stub-llm dry run ──────────────────────────────────
if [ "$RUN_MODE" = "smoke-only" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SMOKE TEST (imports + --stub-llm dry run, \$0 tokens)"
    echo "═══════════════════════════════════════════════════════════════"
    run_ssm "smoke" 900 <<SMOKE
set -eo pipefail
${ENV_SOURCE}
cd /home/ec2-user/alpha-engine-research

echo "==> Smoke: import main + graph"
\$PYTHON_BIN -c "import main; from graph.research_graph import build_graph; print('import OK')"

echo ""
echo "==> Smoke: python main.py --stub-llm --no-s3 (real graph wiring, stubbed LLM, no writes)"
\$PYTHON_BIN main.py --stub-llm --no-s3 2>&1 | tail -40
SMOKE

    echo "==> Smoke complete — instance will be terminated."
    exit 0
fi

# ── Preflight-only: boot + ResearchPreflight + exit 0 (NO LLM, NO write) ──────
if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  PREFLIGHT-ONLY (boot + ResearchPreflight + exit 0, NO LLM/write)"
    echo "═══════════════════════════════════════════════════════════════"
    run_ssm "preflight" 600 <<PREFLIGHT
set -eo pipefail
${ENV_SOURCE}
cd /home/ec2-user/alpha-engine-research

echo "==> ResearchPreflight (env/secret resolution + S3 reachability, read-only)"
\$PYTHON_BIN -c "
from preflight import ResearchPreflight
ResearchPreflight(bucket='${S3_BUCKET}', mode='weekly').run()
print('ResearchPreflight OK — NO LLM spend, NO write')
"
PREFLIGHT

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Preflight-only complete (NO LLM/write). Instance will be terminated."
    echo "═══════════════════════════════════════════════════════════════"
    exit 0
fi

# ── Full run: python main.py --date <trading_day> ────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL RESEARCH RUN"
echo "═══════════════════════════════════════════════════════════════"

run_ssm "workload" "$MAX_RUNTIME_SECONDS" <<WORKLOAD
set -eo pipefail
${ENV_SOURCE}
cd /home/ec2-user/alpha-engine-research

# ── Spot-side log capture ────────────────────────────────────────────
# SSM get-command-invocation caps StandardOutputContent at 24KB and the spot
# terminates before the dispatcher can fetch logs another way; tee into a log
# file + upload to S3 on ANY exit path for post-mortem.
LOG_FILE=/tmp/research.log
exec > >(tee -a "\$LOG_FILE") 2>&1
upload_log() {
    local exit_code=\$?
    local s3_key="health/research_log/\$(date +%Y-%m-%d)/\$(date +%Y%m%dT%H%M%SZ -u)-exit\${exit_code}.log"
    aws s3 cp "\$LOG_FILE" "s3://${S3_BUCKET}/\$s3_key" --region "\${AWS_REGION:-us-east-1}" 2>/dev/null \\
        && echo "[log-upload] s3://${S3_BUCKET}/\$s3_key" \\
        || echo "[log-upload] WARNING: failed to upload \$LOG_FILE to S3"
}
trap upload_log EXIT

# Trading-day stamping: prefer the explicit --date passed by the SF/operator;
# else compute the most-recent trading day on the spot (matches the Lambda
# handler's most_recent_trading_day stamping). Never date.today() blindly —
# Saturday's run targets Friday's trading_day.
RUN_DATE="${RUN_DATE}"
if [ -z "\$RUN_DATE" ]; then
    RUN_DATE=\$(\$PYTHON_BIN -c "import importlib, datetime; h=importlib.import_module('lambda.handler'); print(h.most_recent_trading_day(datetime.date.today()).isoformat())")
fi
echo "──────────────────────────────────────────────────────────────"
echo "Starting research main.py --date \$RUN_DATE at \$(date)"
echo "──────────────────────────────────────────────────────────────"
if ! \$PYTHON_BIN main.py --date "\$RUN_DATE" 2>&1; then
    echo "ERROR: research main.py --date \$RUN_DATE failed." >&2
    exit 1
fi
echo "Research run complete at \$(date)"
WORKLOAD

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Weekly research run complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

aws cloudwatch put-metric-data \
    --namespace "AlphaEngine" \
    --metric-name "Heartbeat" \
    --dimensions "Process=research" \
    --value 1 --unit "Count" \
    --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
    && echo "Heartbeat emitted: research" \
    || echo "WARNING: Failed to emit heartbeat for research (non-fatal)"
