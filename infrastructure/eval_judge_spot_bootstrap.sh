#!/usr/bin/env bash
#
# eval_judge_spot_bootstrap.sh — on-box SETUP for the eval-judge spot instance
# (alpha-engine-config-I9329; the judge itself is evals/judge_spot_run.py,
# added by crucible-research-PR766).
#
# Invoked by the alpha-engine-eval-judge-spot dispatcher's SSM prelude, which
# has already run `krepis.spot_bootstrap.render_bootstrap()`: the interpreter
# is installed, the watchdog/deadman timers are ARMED, this repo is
# shallow-cloned to /home/ec2-user/crucible-research, and the environment file
# at /home/ec2-user/eval-judge.env carries the router / registry /
# exec-context variables. Everything version-controlled lives here rather than
# in the dispatcher's command string, so a change to the box's shape is a PR in
# this repo, not a Lambda redeploy.
#
# ── THREE DELIBERATE DIFFERENCES FROM infrastructure/thinktank_spot_bootstrap.sh
#
# 1. SETUP-ONLY. This script does NOT run the judge. The Step Function drives
#    the run with its OWN, LATER `ssm:sendCommand`, which sources that
#    environment file and invokes `python -m evals.judge_spot_run`. So this
#    script's whole contract is: leave the box ALIVE and READY.
#
# 2. NO SELF-TERMINATING `on_exit` TRAP. The thinktank script's contract is to
#    `shutdown -h now` unconditionally; here that would pull the box out from
#    under the SF stage that is about to use it. Orphan prevention is NOT
#    abandoned — it is relocated: the dispatcher's `krepis.spot_bootstrap`
#    deadman timer and `max_runtime_seconds` cap are armed BEFORE this script
#    starts and terminate the box on their own schedule.
#    STATED EXPLICITLY, because it is a deviation from the sibling script's
#    contract: a bootstrap FAILURE here also relies on those timers rather
#    than on a self-shutdown. A box that fails to bootstrap stays up until the
#    deadman fires; it is reported (below) but it is not reaped by this file.
#
# 3. REPORTING IS KEPT. On every exit path the log ships to
#    s3://alpha-engine-research/_ssm_logs/eval-judge-spot/bootstrap/<date>/,
#    and on failure an alert is published to the alpha-engine-alerts SNS topic
#    — same reason the thinktank script does it: stderr from a bootstrap that
#    fails BEFORE the Python run has no other channel, and the SSM command
#    status is watched by nothing (alpha-engine-config-I5752).
#
# Everything load-bearing FAILS LOUD: `set -euo pipefail` plus an explicit
# `|| fail` on each step, so the exit code propagates to the SSM command status
# and the SF stage that polls it.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
REPO_DIR="${REPO_DIR:-/home/ec2-user/crucible-research}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
CONFIG_DIR="${CONFIG_DIR:-/home/ec2-user/alpha-engine-config}"
CONFIG_REPO="${CONFIG_REPO:-nousergon/alpha-engine-config}"
CONFIG_PAT_SSM="${CONFIG_PAT_SSM:-/alpha-engine/saturday_sf_watch/github_pat}"
ALPHA_ENGINE_EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"

SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
RUN_TOKEN="${EVAL_JUDGE_SPOT_RUN_TOKEN:-unknown}"
BOOTSTRAP_DAY="$(date -u +%Y-%m-%d)"
BOOTSTRAP_LOG="${BOOTSTRAP_LOG:-/var/log/eval-judge-spot-bootstrap-${RUN_TOKEN}.log}"
S3_LOG="s3://alpha-engine-research/_ssm_logs/eval-judge-spot/bootstrap/${BOOTSTRAP_DAY}/bootstrap-$(hostname)-${RUN_TOKEN}.log"

log() { echo "[eval-judge-bootstrap] $*"; }
fail() { echo "[eval-judge-bootstrap] FATAL: $*" >&2; exit 1; }

# Report the bootstrap's outcome off-box on EVERY exit path — and do NOT
# terminate (difference 2 above).
#
# Every step in here is best-effort and `|| true`. No-silent-fails carve-out
# (AGENTS.md "Fail loud and fast"):
#   (a) what is swallowed — a failed S3 log upload or a failed SNS publish,
#       i.e. a failure to REPORT the outcome;
#   (b) why the primary deliverable survives — the outcome is already decided
#       by `rc`, which this function preserves and re-exits with, so the SSM
#       command status (and therefore the SF stage) is unaffected by whether
#       the report landed;
#   (c) recording surface — $S3_LOG, the SNS topic, and the SSM command
#       status/stderr, which this function never overwrites.
on_exit() {
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        log "bootstrap finished rc=0 — box is READY and INTENTIONALLY LEFT RUNNING for the SF's judge command"
    else
        log "bootstrap FAILED rc=${rc} — box left running; the dispatcher's deadman/max_runtime timers own its teardown"
    fi

    # Ship on success too: a successful bootstrap's log is the baseline the
    # next failure gets diffed against.
    aws s3 cp "$BOOTSTRAP_LOG" "$S3_LOG" --region "$REGION" --quiet 2>/dev/null || true

    if [ "$rc" -ne 0 ]; then
        aws sns publish --topic-arn "$SNS_TOPIC_ARN" --region "$REGION" \
            --subject "Eval judge spot box bootstrap FAILED (rc=${rc})" \
            --message "The eval-judge spot box exited ${rc} during BOOTSTRAP, before the Step Function's judge command could run.
day=${BOOTSTRAP_DAY} run_token=${RUN_TOKEN} instance=$(hostname)
Log: ${S3_LOG}
Last 40 lines:
$(tail -n 40 "$BOOTSTRAP_LOG" 2>/dev/null || echo '(log unavailable)')

This alert covers the window nothing else can see: a failure BEFORE
evals.judge_spot_run starts (private rubric prompts absent, gitleaks, venv,
deps), where stderr dies with the box. A failure INSIDE the judge reports
itself through the SF stage's own non-zero SSM ResponseCode.
The box is NOT self-terminated here (setup-only script); the dispatcher's
deadman + max_runtime_seconds timers reap it. See alpha-engine-config-I9329." \
            >/dev/null 2>&1 || true
    fi

    exit "$rc"
}
trap on_exit EXIT

cd "$REPO_DIR" || fail "repo not found at $REPO_DIR (dispatcher prelude clone failed?)"

# ── Private config (rubric prompts) ─────────────────────────────────────────
# evals/judge.py loads its rubrics via agents.prompt_loader.load_prompt, whose
# candidate #1 is the HOME-sibling checkout — exactly where we clone. The
# loader HARD-FAILS on a miss (no sample fallback, per feedback_no_example_
# fallback.md), so an absent config repo would surface as a mid-run
# FileNotFoundError after the SF had already declared the box ready. We assert
# the rubrics are present here instead, mirroring the thinktank script's
# refuse-to-run-on-sample-prompts check.
log "cloning private config for judge rubrics (experiment=${ALPHA_ENGINE_EXPERIMENT_ID})"
PAT=$(aws ssm get-parameter --name "$CONFIG_PAT_SSM" --with-decryption \
    --query Parameter.Value --output text --region "$REGION") \
    || fail "could not read config PAT from ${CONFIG_PAT_SSM}"
[ -n "$PAT" ] || fail "config PAT empty"
git config --global --add safe.directory '*' || true
rm -rf "$CONFIG_DIR"
git clone --depth 1 "https://x-access-token:${PAT}@github.com/${CONFIG_REPO}.git" \
    "$CONFIG_DIR" >/dev/null 2>&1 || fail "config clone failed"
unset PAT

# evals/judge.py::resolve_rubric_for_agent returns eval_rubric_* names only, so
# eval_rubric_*.txt is the judge's REAL prompt requirement — not the
# thinktank_*.txt the sibling script checks for. Same directory resolution
# (experiment-scoped first, then the legacy bare path) as
# agents/prompt_loader.py::_resolve_prompt_path.
PROMPT_DIR=""
for candidate in \
    "$CONFIG_DIR/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/research/prompts" \
    "$CONFIG_DIR/research/prompts"; do
    if ls "$candidate"/eval_rubric_*.txt >/dev/null 2>&1; then PROMPT_DIR="$candidate"; break; fi
done
[ -n "$PROMPT_DIR" ] || fail "eval rubric prompts (eval_rubric_*.txt) not found under $CONFIG_DIR — refusing to declare the box ready against sample prompts"
log "rubrics resolved: $PROMPT_DIR"

# ── DLP scan prerequisite (krepis session_dlp.py) ───────────────────────────
# The krepis DLP hook shells out to gitleaks on every routed LLM call and fails
# CLOSED when the binary or config dir is absent — without this, every judge
# call the box makes dies. Mirrors infrastructure/thinktank_spot_bootstrap.sh's
# install exactly (same version, same checksum, same /opt/llm-routing config,
# which is session_dlp.py's first standard-path fallback, so no
# KREPIS_GITLEAKS_DIR override is needed).
if ! command -v gitleaks >/dev/null 2>&1; then
    GITLEAKS_VERSION=8.30.1
    GITLEAKS_SHA256=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
    case "$(uname -m)" in
        aarch64|arm64) GITLEAKS_ARCH=arm64 ;;
        *) GITLEAKS_ARCH=x64 ;;
    esac
    log "installing gitleaks ${GITLEAKS_VERSION} (${GITLEAKS_ARCH})"
    curl -fsSL -o /tmp/gitleaks.tar.gz \
        "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_${GITLEAKS_ARCH}.tar.gz" \
        || fail "gitleaks download failed"
    if [ "$GITLEAKS_ARCH" = "x64" ]; then
        echo "${GITLEAKS_SHA256}  /tmp/gitleaks.tar.gz" | sha256sum -c - || fail "gitleaks checksum mismatch"
    fi
    tar -xzf /tmp/gitleaks.tar.gz -C /tmp gitleaks || fail "gitleaks extract failed"
    install /tmp/gitleaks /usr/local/bin/gitleaks || fail "gitleaks install failed"
    rm -f /tmp/gitleaks.tar.gz
fi
command -v gitleaks >/dev/null 2>&1 || fail "gitleaks binary unavailable after install (fail-closed)"
mkdir -p /opt/llm-routing
printf '[extend]\nuseDefault = true\n' > /opt/llm-routing/gitleaks-egress.toml

# ── Runtime ─────────────────────────────────────────────────────────────────
# Strict interpreter: python3.12 or nothing. requirements.txt is resolved
# against 3.12 and the wheels differ; a fallback onto the AMI's system python3
# is the silent-degradation shape tests/test_spot_bootstrap_invariants.py
# forbids repo-wide.
log "building venv at ${VENV_DIR}"
python3.12 -m venv "$VENV_DIR" || fail "venv creation failed"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip || fail "pip upgrade failed"
"$VENV_DIR/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt" || fail "requirements install failed"

# The SF's judge command runs as ec2-user, not root.
chown -R ec2-user:ec2-user "$REPO_DIR" || fail "chown of ${REPO_DIR} failed"
chown -R ec2-user:ec2-user "$CONFIG_DIR" || fail "chown of ${CONFIG_DIR} failed"

# ── Readiness assertions ────────────────────────────────────────────────────
# A bootstrap that "succeeds" and leaves a box the SF's command cannot run on
# is the one failure mode the SF poll loop cannot distinguish from a judge bug:
# it would see a non-zero ResponseCode from the RUN stage and blame the judge.
# So the ready verdict is PROVED here, in the same interpreter the run uses.
log "asserting readiness"
"$VENV_DIR/bin/python" -c "import krepis.ssm_log_capture" \
    || fail "venv python cannot import krepis.ssm_log_capture — the SF's judge command wraps its run in it"
( cd "$REPO_DIR" && "$VENV_DIR/bin/python" -c "import evals.judge_spot_run" ) \
    || fail "venv python cannot import evals.judge_spot_run"
( cd "$REPO_DIR" && "$VENV_DIR/bin/python" -m evals.judge_spot_run --help >/dev/null ) \
    || fail "python -m evals.judge_spot_run --help exited non-zero"

log "READY"
