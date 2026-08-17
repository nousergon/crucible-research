#!/usr/bin/env bash
#
# spot_research_weekly.sh — run the weekly Research heavy pass on a fresh,
# self-terminating spot-EC2 instance via SSM submit+poll (config#1687).
#
# WHY: the weekly SF `Research` state is a synchronous `lambda:invoke` capped
# at the 900s AWS Lambda ceiling. The 2026-07-03 weekly used ~874s (97%);
# there is no headroom lever left on Lambda. Every other heavy weekly workload
# (MorningEnrich, DataPhase1, RAG, PredictorTraining, Backtester) already runs
# on this spot-EC2 SSM submit+poll pattern — this file moves the weekly
# Research pass onto it. The runner Lambda STAYS for intraday alerts +
# operator modes (`challengers_only`, `dry_run_llm`, manual invokes).
#
# The on-box work is `infrastructure/weekly_box_runner.py`, which invokes the
# SAME `lambda/handler.py::handler` orchestration the Lambda drives today
# (graph build → invoke → archive_writer → FAIL-HARD challenger post-step
# (config#1683) → trajectory → health → manifest → cost aggregation), so the
# PRIOR-population snapshot + fail-loud challenger contract are preserved by
# construction. See that file's module docstring.
#
# Transport is SSM (submit+poll), NOT SSH/SCP. Config is repo-local from the
# clone (package-first fallback); secrets are read on the spot via
# krepis.secrets (SSM Parameter Store). Communication + capacity-resilient
# launch + mid-run spot-interruption relaunch all go through the shared lib
# chokepoints (`python -m krepis.ec2_spot`, `krepis.ssm_dispatcher`,
# `krepis.ssm_log_capture`) exactly as the sibling Saturday launchers do.
#
# USAGE:
#   ./infrastructure/spot_research_weekly.sh                 # full weekly run, then terminate
#   ./infrastructure/spot_research_weekly.sh --preflight-only # boot + import/lib-pin + ResearchPreflight, exit 0 (NO graph/LLM/writes)
#   ./infrastructure/spot_research_weekly.sh --force          # bypass time/trading-day/idempotency gates
#
# ── TWO ITEMS TO VERIFY IN THE FRIDAY SHELL-RUN REHEARSAL (config#1629) ───────
#  (1) IAM_PROFILE — the iam:PassRole gap class (config#1290/#1308): confirm
#      the instance profile below can (a) be passed by the dashboard-box role
#      and (b) read the Research SSM secrets + read/write s3://alpha-engine-
#      research. Starts from the established Saturday-launcher profile.
#  (2) CONFIG RESOLUTION — RESOLVED 2026-07-06 (pre-rehearsal review): a
#      fresh public clone has NO prompts and NO real YAMLs (both gitignored;
#      prompt_loader HARD-FAILS with no .example fallback — the 2026-04-11
#      silent-sample-fallback incident is why). This launcher now stages the
#      dispatcher's private alpha-engine-config `research/` subtree to S3 as
#      a single tarball and the box extracts it to
#      /home/ec2-user/alpha-engine-config/research/ — prompt_loader/config.py
#      search path #1 (HOME-sibling), the same files deploy.sh stages into
#      the Lambda image. Hard-fails at dispatch if prompts are absent
#      (deploy.sh parity). Single-key cp avoids needing s3:ListBucket on the
#      spot profile. Rehearsal still validates the resolution end-to-end.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Spot configuration ──────────────────────────────────────────────────────
# Values mirror nousergon-data/infrastructure/spot_data_weekly.sh so no new
# IAM / security-group / subnet resources are introduced. If those change in
# the sibling launchers, this file should change in lockstep.
AWS_REGION="${AWS_REGION:-us-east-1}"
RESEARCH_BUCKET="${RESEARCH_BUCKET:-alpha-engine-research}"
S3_BUCKET="${S3_BUCKET:-$RESEARCH_BUCKET}"
BRANCH="${BRANCH:-main}"
ALPHA_ENGINE_EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
# Capacity-resilient instance-type fallback set (all 2 vCPU / 4-8 GB RAM).
# Order = preference; the lib CLI tries each until one launches.
INSTANCE_TYPES="${INSTANCE_TYPES:-c5.large,m5.large,c6i.large,c5a.large}"
INSTANCE_TYPE=""
# Stage slug for krepis.spot_evidence (evidence prefix + sweep) and for the
# ssm_log_capture --slug already used below — kept as one name so a failure's
# S3 paths and its evidence copy live under the same key.
SPOT_STAGE_NAME="research-weekly"
AMI_ID="ami-0c421724a94bba6d6"      # Amazon Linux 2023 x86_64
# Weekly Research runs ~15 min primary (874s on 2026-07-03) + RAG/held-name
# tail; 90 min with headroom covers that plus pip install + preflight.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-5400}"

# ── Spot-interruption resilience (mirror spot_data_weekly.sh) ────────────────
# on_exit classifies a failure via the lib chokepoint (krepis.ec2_spot
# relaunch-decision): a CONFIRMED spot reclamation relaunches a fresh spot up
# to MAX_SPOT_ATTEMPTS; a genuine inner-workload error is NOT retried (fail
# loud — blind retry masks real bugs). SPOT_ATTEMPT threads across re-execs.
# Raising MAX_SPOT_ATTEMPTS REQUIRES raising the matching SF state
# executionTimeout in lockstep (config#883 coupling).
MAX_SPOT_ATTEMPTS="${MAX_SPOT_ATTEMPTS:-2}"
SPOT_ATTEMPT="${SPOT_ATTEMPT:-1}"
SF_EXECUTION_TIMEOUT="${SF_EXECUTION_TIMEOUT:-}"
SPOT_RETRY_BACKOFF_SECONDS="${SPOT_RETRY_BACKOFF_SECONDS:-20}"

KEY_NAME="alpha-engine-key"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNETS="${SUBNETS:-subnet-a61ec0fb,subnet-1e58307a,subnet-789d3857,subnet-c670118d,subnet-7cff7c43,subnet-e07166ec}"
# VERIFY-IN-REHEARSAL (1): iam:PassRole gap (config#1290/#1308). Established
# Saturday-launcher profile; confirm it grants Research SSM-secret reads +
# s3://alpha-engine-research read/write, and is passable by the dashboard role.
IAM_PROFILE="${IAM_PROFILE:-alpha-engine-executor-profile}"
# Lib CLI path: every spot launcher on the dispatcher box resolves its
# interpreter through the ops-owned guard /opt/nousergon/bin/lib-python
# (nous-ergon-ops: alpha-engine-dashboard/live/infrastructure/bin/lib-python).
# That guard execs the box's DECLARED krepis venv and aborts with EX_CONFIG
# (78), naming the version it found, when the venv is absent or below the
# launcher floor. It never falls back to a co-tenant checkout — the silent
# fallback is exactly the defect alpha-engine-config-I6931/I7343 removes.
# Do NOT add a guard block here: the contract lives ONCE, in the repo that
# owns this box's provisioning (nine copies across five repos is I6922).
LIB_PYTHON="${LIB_PYTHON:-/opt/nousergon/bin/lib-python}"

# ── Parse flags ──────────────────────────────────────────────────────────────
ORIG_ARGS=("$@")
PREFLIGHT_ONLY=0
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --force) FORCE=1; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --max-runtime-seconds) MAX_RUNTIME_SECONDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
# --instance-type X collapses the resilient list to a single type.
[ -n "$INSTANCE_TYPE" ] && INSTANCE_TYPES="$INSTANCE_TYPE"

echo "==> spot_research_weekly.sh"
echo "  Branch        : $BRANCH"
echo "  Bucket        : $RESEARCH_BUCKET"
echo "  Experiment    : $ALPHA_ENGINE_EXPERIMENT_ID (package-first; repo-local fallback)"
echo "  Preflight-only: $PREFLIGHT_ONLY  (1 = boot + preflight + exit 0, NO graph/LLM/writes)"
echo "  Force         : $FORCE"
echo "  Transport     : SSM via lib chokepoint (python -m krepis.ssm_dispatcher)"

# ── Cleanup + spot-interruption retry trap (installed BEFORE launch) ─────────
# Locate the private alpha-engine-config research/ subtree on the dispatcher
# (ae-dashboard clones + boot-pulls the private repo; laptop fallback for
# manual runs). Package-first, legacy top-level fallback — mirrors deploy.sh's
# resolution (config#1042) and the on-box `resolve_experiment_config` call the
# staged tarball ultimately feeds (line 115's ALPHA_ENGINE_EXPERIMENT_ID export
# below only affects the box-side reader; the STAGING step here was picking
# the config up locally before shipping it, and had drifted out of package-first
# lockstep with deploy.sh — config#3066). Hard-fail if prompts are missing —
# deploy.sh parity: an image/box without the real prompts must never ship/run
# (2026-04-11).
for _config_repo_root in "/home/ec2-user/alpha-engine-config" "$HOME/Development/alpha-engine-config"; do
    if [ -d "$_config_repo_root/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/research" ]; then
        RESEARCH_CONFIG_SRC="$_config_repo_root/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/research"
        break
    elif [ -d "$_config_repo_root/research" ]; then
        RESEARCH_CONFIG_SRC="$_config_repo_root/research"
        break
    fi
done
unset _config_repo_root
if ! ls "$RESEARCH_CONFIG_SRC/prompts/"*.txt >/dev/null 2>&1; then
    echo "ERROR: research prompts not found — tried (package-first, experiment=${ALPHA_ENGINE_EXPERIMENT_ID}):" >&2
    echo "  /home/ec2-user/alpha-engine-config/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/research/prompts/" >&2
    echo "  /home/ec2-user/alpha-engine-config/research/prompts/ (legacy)" >&2
    echo "  \$HOME/Development/alpha-engine-config/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/research/prompts/" >&2
    echo "  \$HOME/Development/alpha-engine-config/research/prompts/ (legacy)" >&2
    echo "is alpha-engine-config cloned + pulled on this host? (deploy.sh parity check)" >&2
    exit 1
fi

INSTANCE_ID=""
S3_STAGING=""

# ── Failure-evidence teardown (alpha-engine-config-I7442) ───────────────────
# WHY: run_ssm's --output-key-prefix points OutputS3KeyPrefix (below) at
# ${S3_STAGING_PREFIX}/ssm-output -- the SSM agent's own upload of the FULL
# remote stdout/stderr, the only copy not subject to GetCommandInvocation's
# 24KB inline cap. The unguarded `aws s3 rm "$S3_STAGING" --recursive` this
# replaced destroyed that upload on every failure exit, in the same breath
# that printed a path pointing at it -- the 2026-08-15 PredictorBacktest
# failure (a sibling launcher) is now permanently unrecoverable because of
# it. Route through the shared chokepoint instead of reimplementing
# retain-then-sweep in bash here: `krepis.spot_evidence teardown` (krepis
# branch fix/i7442-resource-kill-classifier) copies the prefix to
# `_spot_evidence/<slug>/<date>/<run-id>/` FIRST and deletes staging only if
# that copy succeeded; on a successful run it deletes with nothing retained.
# Reference: crucible-backtester#675's `_spot_common.sh::
# spot_common_teardown_staging` -- mirrored here inline because this repo has
# no shared `_spot_common.sh` to source (policy-shared-code: a second,
# divergent bash reimplementation of the same 55-line retain/sweep logic is
# exactly what the chokepoint exists to prevent).
_teardown_staging() {
    local _exit_code="$1"
    if [ -z "$S3_STAGING" ]; then
        return 0
    fi
    if "$LIB_PYTHON" -m krepis.spot_evidence teardown \
            --staging "$S3_STAGING" \
            --slug "$SPOT_STAGE_NAME" \
            --exit-code "$_exit_code"; then
        return 0
    fi
    # FAIL-SAFE DIRECTION IS LOAD-BEARING: an unreachable chokepoint (a box
    # whose krepis pin predates the release carrying spot_evidence) must
    # RETAIN the staging prefix, never fall back to `aws s3 rm` -- that
    # fallback is the exact defect alpha-engine-config-I7442 removes, and
    # this is what makes the merge order safe before every box's krepis pin
    # is bumped. Must not change the trap's exit status: always return 0.
    echo "  spot_evidence: chokepoint unavailable via $LIB_PYTHON — S3 staging RETAINED at $S3_STAGING/ (not deleted)" >&2
    echo "    Full stage stdout/stderr: $S3_STAGING/ssm-output/" >&2
    return 0
}

cleanup() {
    local exit_code="${1:-0}"
    if [ -n "$INSTANCE_ID" ]; then
        echo ""
        echo "==> Terminating spot instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
    fi
    _teardown_staging "$exit_code"
    [ -n "$INSTANCE_ID" ] && echo "  Instance terminated; S3 staging routed through krepis.spot_evidence teardown."
    return 0
}

# Echoes a reason + returns 0 when the just-failed run was a CONFIRMED spot
# interruption (launch-capacity exhaustion rc 64, or a mid-run reclaim per the
# lib's relaunch-decision). A genuine inner-workload failure returns 1 → NOT
# retried (fail loud). Mirrors spot_data_weekly.sh (the config#883 reference).
_spot_failure_reason() {
    local rc="$1"
    if [ "$rc" -eq 64 ]; then echo "launch-capacity-exhausted"; return 0; fi
    [ -z "$INSTANCE_ID" ] && return 1
    # See alpha-engine-config-I7009 — migrated off the exit-code contract to --json.
    local _decide_json="" _decide_rc=0
    _decide_json="$("$LIB_PYTHON" -m krepis.ec2_spot relaunch-decision \
        --instance-id "$INSTANCE_ID" \
        --region "$AWS_REGION" \
        --attempt "$SPOT_ATTEMPT" \
        --max-attempts "$MAX_SPOT_ATTEMPTS" \
        ${SF_EXECUTION_TIMEOUT:+--sf-execution-timeout "$SF_EXECUTION_TIMEOUT" --per-attempt-seconds "$MAX_RUNTIME_SECONDS"} \
        --json \
        2>/dev/null)" || _decide_rc=$?
    echo "  spot relaunch-decision (attempt $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS): rc=$_decide_rc ${_decide_json:+[$_decide_json]}" >&2
    # A non-zero exit here means the CLI could not answer — not a verdict.
    # Treat as hold (not a confirmed reclaim): return 1, same as the CLI
    # answering "hold" via the JSON payload below.
    [ "$_decide_rc" -eq 0 ] || return 1
    local _relaunch=""
    _relaunch="$(printf '%s' "$_decide_json" | "$LIB_PYTHON" -c 'import json,sys; print("1" if json.load(sys.stdin).get("relaunch") else "0")')"
    [ "$_relaunch" = "1" ] || return 1
    echo "confirmed-reclaim${_decide_json:+ ($_decide_json)}"
}

on_exit() {
    local rc=$?
    # Classify BEFORE cleanup() terminates the instance (request status is
    # only queryable while the instance still exists).
    local reason=""
    if [ "$rc" -ne 0 ]; then
        reason="$(_spot_failure_reason "$rc")" || reason=""
    fi
    cleanup "$rc"
    if [ "$rc" -ne 0 ] && [ -n "$reason" ] && [ "$SPOT_ATTEMPT" -lt "$MAX_SPOT_ATTEMPTS" ]; then
        aws cloudwatch put-metric-data \
            --namespace "AlphaEngine" \
            --metric-name "SpotInterruptionRetry" \
            --dimensions "Process=research-weekly" \
            --value 1 --unit "Count" \
            --region "$AWS_REGION" 2>/dev/null || true
        echo "" >&2
        echo "==> Spot interruption (reason=$reason) on attempt $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS — relaunching a fresh spot in ${SPOT_RETRY_BACKOFF_SECONDS}s..." >&2
        sleep "$SPOT_RETRY_BACKOFF_SECONDS"
        trap - EXIT
        SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
    fi
    if [ "$rc" -ne 0 ] && [ -n "$reason" ]; then
        echo "ERROR: spot interruption (reason=$reason) persisted across all $MAX_SPOT_ATTEMPTS attempt(s) — giving up. The weekly pipeline fails loud; redrive once spot capacity returns." >&2
    fi
    exit "$rc"
}
trap on_exit EXIT

# ── Launch spot (capacity-resilient; fail-loud on empty id) ──────────────────
echo "==> Requesting spot instance (types=[$INSTANCE_TYPES], subnets=[$SUBNETS])..."
INSTANCE_ID=$("$LIB_PYTHON" -m krepis.ec2_spot launch \
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
    if [ "$ec2_spot_rc" -eq 0 ]; then
        # rc=0 with an EMPTY id = the launch layer produced nothing. Must fail
        # loud (config#1646 — the guard-less shim no-op that recorded a silent
        # success on 2026-07-03; closed at the transport by config#1649).
        echo "ERROR: ec2_spot launch exited 0 without an instance id — failing loud (config#1646)" >&2
        ec2_spot_rc=1
    fi
    exit "$ec2_spot_rc"
fi
echo "  Instance ID: $INSTANCE_ID"

RUN_ID="$(date +%Y%m%dT%H%M%SZ)-${INSTANCE_ID}"
S3_STAGING_PREFIX="tmp/spot_research_weekly/${RUN_ID}"
S3_STAGING="s3://${S3_BUCKET}/${S3_STAGING_PREFIX}"

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Stage the private research config surface (prompts + scoring/universe/
# thinktank YAMLs + any experiment-package subdirs) as ONE tarball — the
# spot pulls it with a single GetObject (no ListBucket needed on the spot
# profile, matching the proven spot_data_weekly config.yaml pattern).
echo "==> Staging alpha-engine-config/research/ → ${S3_STAGING}/research-config.tgz"
tar -C "$RESEARCH_CONFIG_SRC" -czf /tmp/research-config-${RUN_ID}.tgz .
aws s3 cp "/tmp/research-config-${RUN_ID}.tgz" "${S3_STAGING}/research-config.tgz" --region "$AWS_REGION" --only-show-errors
rm -f "/tmp/research-config-${RUN_ID}.tgz"

# ── SSM dispatch primitive (lib chokepoint) ──────────────────────────────────
# Thin wrapper around `python -m krepis.ssm_dispatcher run` (invoked directly
# via krepis per config#1649); failure-only substrate, preserves inner exit.
run_ssm() {
    local description="$1" timeout_s="${2:-3600}"
    "$LIB_PYTHON" -m krepis.ssm_dispatcher run \
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

# Each run_ssm step is a fresh SSM shell with a minimal env — export the region
# + resolve the interpreter per block (AL2023 installs python3.12 with no bare
# `python` symlink). STRICT resolution, no fallback: requirements.txt is
# resolved against 3.12, and a fallback to the AMI's bare python3 silently
# resolves different wheels (alpha-engine-config-I7372, ported from
# crucible-predictor#462 / nousergon-data#1296 — removing the fallback from
# the bootstrap alone was NOT the fix there either; every downstream step
# reading $PYTHON_BIN had to resolve strictly too).
read -r -d '' ENV_SOURCE <<'ENV_EOF' || true
export HOME=/home/ec2-user
export XDG_CACHE_HOME=/tmp
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export ALPHA_ENGINE_DEPLOYED=1
command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found on spot — requirements.txt is resolved against 3.12; refusing to silently fall back to the AMI python3 (alpha-engine-config-I7372)" >&2; exit 1; }
PYTHON_BIN=python3.12
export PYTHON_BIN
ENV_EOF

# ── Bootstrap: watchdog + python + git + clone ───────────────────────────────
# Rendered by krepis.spot_bootstrap (alpha-engine-config-I7372, mirrors
# nousergon-data#1388 / crucible-predictor#504) rather than hand-carried as an
# inline heredoc — the fleet's single canonical source for the watchdog unit
# (SSM-liveness supervision; NOT carried by this file before), the strict
# python3.12 interpreter assertion, and the clone shape. Repo/branch are baked
# in as launcher-side literals via --repo-url/--branch rather than left for
# the remote shell to expand, which removes the spot-expanded-but-never-
# exported class of bug (crucible-predictor#463: ${REPO_URL}/${BRANCH}
# interpolated into a heredoc but never exported, so the remote expansion
# resolved to an empty string and the clone died on `fatal: repository ''`).
# --max-runtime-seconds arms the SAME systemd-run --on-active hard-timeout
# timer this heredoc used to hand-carry (now `ec2-spot-hard-timeout`, was
# `alpha-engine-watchdog`) — a SEPARATE guarantee from the renderer's own
# `ec2-spot-watchdog` SSM-liveness unit, which this launcher gains for the
# first time by this cutover.
#
# The tarball extract + `ls` prompts-present assertion below is repo-specific
# staging the renderer cannot express (it only knows `aws s3 cp` config
# copies, not tar extraction) — kept verbatim, appended after the rendered
# script into the SAME run_ssm "bootstrap" call rather than as a second inline
# bootstrap.
echo "==> Bootstrapping spot (watchdog, python, clone)..."
_BOOTSTRAP_SCRIPT="$("$LIB_PYTHON" -m krepis.spot_bootstrap render \
    --repo-url https://github.com/nousergon/crucible-research.git \
    --checkout /home/ec2-user/research \
    --branch "${BRANCH:-main}" \
    --region "$AWS_REGION" \
    --max-runtime-seconds "$MAX_RUNTIME_SECONDS" \
    --export "S3_STAGING=${S3_STAGING}")"
_BOOTSTRAP_TAIL="$(cat <<BOOTSTRAP_TAIL
# Private research config surface (prompts + YAMLs): extract to the
# prompt_loader/config.py HOME-sibling search path. Without this the box
# HARD-FAILS at load_prompt (no .example fallback, by design).
mkdir -p /home/ec2-user/alpha-engine-config/research
aws s3 cp ${S3_STAGING}/research-config.tgz /tmp/research-config.tgz --region ${AWS_REGION} --only-show-errors
tar -xzf /tmp/research-config.tgz -C /home/ec2-user/alpha-engine-config/research
rm -f /tmp/research-config.tgz
ls /home/ec2-user/alpha-engine-config/research/prompts/*.txt >/dev/null || { echo "ERROR: staged prompts missing after extract"; exit 1; }
echo "Bootstrap complete: crucible-research cloned at ${BRANCH}; research config staged."
BOOTSTRAP_TAIL
)"
printf '%s\n%s\n' "$_BOOTSTRAP_SCRIPT" "$_BOOTSTRAP_TAIL" | run_ssm "bootstrap" 600

# ── Install python deps ──────────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
run_ssm "deps" 900 <<DEPS
set -eo pipefail
${ENV_SOURCE}
cd /home/ec2-user/research
PIP="\$PYTHON_BIN -m pip"
\$PIP install --upgrade pip -q
\$PIP install -q -r requirements.txt
echo "Dependencies installed."
DEPS

# ── Preflight-only (Friday shell-run dry path, config#1629) ──────────────────
if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  PREFLIGHT-ONLY (import + lib-pin + ResearchPreflight, NO run)"
    echo "═══════════════════════════════════════════════════════════════"
    run_ssm "preflight-only" 600 <<PREFLIGHT
set -eo pipefail
${ENV_SOURCE}
export ALPHA_ENGINE_EXPERIMENT_ID=${ALPHA_ENGINE_EXPERIMENT_ID} RESEARCH_BUCKET=${RESEARCH_BUCKET}
cd /home/ec2-user/research
\$PYTHON_BIN -m krepis.ssm_log_capture run --slug research-preflight --log /var/log/research-preflight.log --bucket "${S3_BUCKET}" -- \\
    \$PYTHON_BIN infrastructure/weekly_box_runner.py --preflight-only
PREFLIGHT
    echo ""
    echo "==> Preflight-only mode — PASS. No graph, no LLM, no writes. Exiting 0."
    exit 0
fi

# ── Weekly Research run (full) ───────────────────────────────────────────────
# Routes the workload through krepis.ssm_log_capture so stdout+stderr are teed
# to a spot-local logfile AND shipped to S3 on EXIT (incl. OOM-kill) BEFORE the
# dispatcher tears the box down, with the workload exit code propagated verbatim
# so a real failure still trips set -eo pipefail + the SF ExtractResearchError.
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  WEEKLY RESEARCH RUN (reuses lambda/handler orchestration)"
echo "═══════════════════════════════════════════════════════════════"
FORCE_FLAG=""
[ "$FORCE" = "1" ] && FORCE_FLAG="--force"
run_ssm "weekly" "$MAX_RUNTIME_SECONDS" <<WEEKLY
set -eo pipefail
${ENV_SOURCE}
export ALPHA_ENGINE_EXPERIMENT_ID=${ALPHA_ENGINE_EXPERIMENT_ID} RESEARCH_BUCKET=${RESEARCH_BUCKET} S3_BUCKET=${RESEARCH_BUCKET}
cd /home/ec2-user/research
\$PYTHON_BIN -m krepis.ssm_log_capture run --slug research-weekly --log /var/log/research-weekly.log --bucket "${S3_BUCKET}" -- \\
    \$PYTHON_BIN infrastructure/weekly_box_runner.py ${FORCE_FLAG}
WEEKLY

echo "==> Weekly Research run complete; spot self-terminates via the EXIT trap."
